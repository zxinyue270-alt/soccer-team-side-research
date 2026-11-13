"""Compare detected highlights with labels of the same clip.

A label file lists marked moments of the clip, each with a type, a window and a peak frame,
optionally a tolerance:

    {"id": ..., "moments": [{"id": "M1", "type": "duel_tackle", "start": 100,
      "peak": 150, "end": 160, "peak_tolerance_frames": 6}, ...], "cuts": [...]}

It may say where its labels come from (:data:`SOURCE_FIELDS`): who or what wrote them, how
and when (``labelled_by``, ``method``, ``labelled_on``), what had been looked at before
(``blinding``) and what the comparison is for (``status``). :func:`source_zh` quotes these
fields as written and says so when there are none: nothing is claimed about the labels
(a person, frame by frame, before any output) that the file does not say.

:func:`evaluate` matches every labelled moment to the detected highlight whose window
overlaps it most (each highlight is used at most once, in label order), and reports the
peak error in frames, whether it is within the tolerance, the type agreement and the
window overlap (intersection over union). Detected highlights that match no label are
listed separately: they are not necessarily wrong (the labeller may have missed a moment),
so they are counted, not scored. Labels from the clip a method was tuned on give a
regression check, never a validation, and the report says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: tolerance when a label gives none: half a second at 25 fps
DEFAULT_TOLERANCE_FRAMES = 12

SOURCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("labelled_by", "标注者"),
    ("method", "方法"),
    ("labelled_on", "标注日期"),
    ("blinding", "盲法"),
    ("status", "说明"),
)
"""Fields of a label file that say where its labels come from, with the Chinese name
:func:`source_zh` gives each."""


class LabelEvalError(ValueError):
    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("label evaluation: " + "; ".join(self.problems))


def _check(labels: Mapping[str, Any], highlights: Sequence[Mapping[str, Any]]) -> None:
    problems: list[str] = []
    moments = labels.get("moments")
    if not isinstance(moments, list):
        problems.append("labels need a 'moments' list")
        moments = []
    for i, m in enumerate(moments):
        for key in ("id", "type", "start", "peak", "end"):
            if key not in m:
                problems.append(f"moment {i}: missing {key!r}")
        if all(k in m for k in ("start", "peak", "end")) and not (
            m["start"] <= m["peak"] <= m["end"]
        ):
            problems.append(f"moment {m.get('id', i)}: needs start <= peak <= end")
    for i, h in enumerate(highlights):
        for key in ("id", "type", "start_frame", "peak_frame", "end_frame"):
            if key not in h:
                problems.append(f"highlight {i}: missing {key!r}")
    if problems:
        raise LabelEvalError(problems)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, float]:
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter, (inter / union if union else 0.0)


def evaluate(labels: Mapping[str, Any], highlights_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Match labelled moments to highlights; see the module docstring."""
    highlights = list(highlights_doc.get("highlights", []))
    _check(labels, highlights)
    used: set[str] = set()
    rows = []
    for m in labels["moments"]:
        best, best_inter, best_iou = None, 0, 0.0
        for h in highlights:
            if h["id"] in used:
                continue
            inter, iou = _overlap(m["start"], m["end"], h["start_frame"], h["end_frame"])
            if inter > best_inter or (inter == best_inter and inter and iou > best_iou):
                best, best_inter, best_iou = h, inter, iou
        tolerance = int(m.get("peak_tolerance_frames", DEFAULT_TOLERANCE_FRAMES))
        row: dict[str, Any] = {
            "label": m["id"],
            "label_type": m["type"],
            "label_peak": m["peak"],
            "tolerance_frames": tolerance,
        }
        if best is None:
            covering = [
                h["id"] for h in highlights if h["start_frame"] <= m["peak"] <= h["end_frame"]
            ]
            row.update({"matched": False, "covered_by": covering[0] if covering else None})
        else:
            used.add(best["id"])
            error = int(best["peak_frame"]) - int(m["peak"])
            row.update(
                {
                    "matched": True,
                    "highlight": best["id"],
                    "type": best["type"],
                    "type_agrees": best["type"] == m["type"],
                    "peak": int(best["peak_frame"]),
                    "peak_error_frames": error,
                    "peak_within_tolerance": abs(error) <= tolerance,
                    "window_iou": round(best_iou, 3),
                    "label_peak_inside_window": best["start_frame"]
                    <= m["peak"]
                    <= best["end_frame"],
                }
            )
        rows.append(row)
    unmatched = [
        {
            "highlight": h["id"],
            "type": h["type"],
            "peak": h["peak_frame"],
            "start": h["start_frame"],
            "end": h["end_frame"],
            "score": h.get("score"),
        }
        for h in highlights
        if h["id"] not in used
    ]
    matched = [r for r in rows if r["matched"]]
    return {
        "labels_id": labels.get("id"),
        "labels_status": labels.get("status"),
        "labels_source": labels_source(labels),
        "n_labels": len(rows),
        "n_matched": len(matched),
        "n_peak_within_tolerance": sum(r["peak_within_tolerance"] for r in matched),
        "n_type_agrees": sum(r["type_agrees"] for r in matched),
        "n_highlights": len(highlights),
        "rows": rows,
        "unmatched_highlights": unmatched,
    }


def labels_source(labels: Mapping[str, Any]) -> dict[str, str]:
    """The fields of :data:`SOURCE_FIELDS` that ``labels`` gives as non-empty text."""
    found = {}
    for key, _name in SOURCE_FIELDS:
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            found[key] = value.strip()
    return found


def source_zh(labels: Mapping[str, Any]) -> str:
    """One Chinese sentence on where the labels come from: the label file's own
    :data:`SOURCE_FIELDS`, quoted word for word (they are not translated, and not checked);
    without any of them, that the file does not say."""
    found = labels_source(labels)
    if not found:
        return "标注文件没有写标注者、方法、盲法等来源字段，标注的来源不明。"
    quoted = [f"{name}“{found[key]}”" for key, name in SOURCE_FIELDS if key in found]
    return "标注来源（照录标注文件）：" + "；".join(quoted) + "。"


def summary_zh(result: Mapping[str, Any], fps: float = 25.0) -> list[str]:
    """Chinese sentences for the report, one per labelled moment plus a total. They call the
    labels 标注 (labels), never 人工标注: who wrote them is the label file's to say
    (:func:`source_zh`)."""
    lines = []
    for r in result["rows"]:
        if not r["matched"]:
            where = (
                f"没有单独成段，但在 {r['covered_by']} 的窗口里"
                if r.get("covered_by")
                else "未检出"
            )
            lines.append(
                f"标注 {r['label']}（{r['label_type']}，第 {r['label_peak']} 帧）：{where}。"
            )
            continue
        error_ms = r["peak_error_frames"] / fps * 1000.0
        verdict = "在容差内" if r["peak_within_tolerance"] else "超出容差"
        kind = "一致" if r["type_agrees"] else f"不一致（{r['type']}）"
        lines.append(
            f"标注 {r['label']}（第 {r['label_peak']} 帧）→ 检出 {r['highlight']}"
            f"（第 {r['peak']} 帧）：关键帧差 {r['peak_error_frames']:+d} 帧"
            f"（{error_ms:+.0f} 毫秒，容差 ±{r['tolerance_frames']} 帧，{verdict}）；类型{kind}；"
            f"窗口重合 IoU {r['window_iou']:.2f}。"
        )
    lines.append(
        f"合计：{result['n_labels']} 个标注中检出 {result['n_matched']} 个，关键帧在容差内 "
        f"{result['n_peak_within_tolerance']} 个，类型一致 {result['n_type_agrees']} 个；"
        f"另有 {len(result['unmatched_highlights'])} 个检出片段不在标注里（不一定是错检）。"
    )
    return lines
