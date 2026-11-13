"""What the local web page shows of a run, and which of its files may be served.

A run is a subfolder of the runs folder that holds a ``run.json`` (gbf.pipeline). This
module reads a run's documents (run.json, highlights_final.json, label_eval.json,
measurements.jsonl and the render manifest of the full clip) and shapes them for the page
(``GET /api/runs`` and ``GET /api/runs/<run>``). It never writes to a run.

Display rules
    Every number is shown the way the report shows it (gbf.report.metric_row over
    gbf.pipeline.display_record): value text, unit, tier and status badges, among them 未验证
    while its error is unmeasured and 仅画面平面 for an image-plane value. A value computed
    from the 3D lift (gbf.lift3d: :data:`LIFT3D_METRICS`, the registry's A_candidate tier)
    also gets the badge 3D 参考: the analysis is done on the original frames and 3D is only a
    reference whose results are drawn back onto them. C-tier rows are never shown, only
    counted. Highlights are named H1, H2, ... in peak order, as in the report, and the label
    comparison is shown only when the run itself used labels (as in the report).

Times
    Frames are canonical frame indices of the source video; clocks are
    gbf.render.format_clock (minutes, seconds and hundredths of a second, not frames).
    clips/full.mp4 runs 1:1 with the source, so source frame f starts at f / fps;
    :func:`full_clip_seeker` reads the mapping from the clip's render manifest when there is
    one and seeks to the middle of the frame, (f + 0.5) / fps, so that rounding in a player
    never shows the frame before.

Which files may be served (:func:`run_folder`, :func:`resolve_artifact`)
    Every decoded URL segment is checked before the file system is touched: no empty, ``.``
    or ``..`` segment, no separator (``/``, ``\\``), drive or stream colon, wildcard or
    control character, no leading dot, no trailing dot or space and no Windows device name
    (:func:`segment_problem`). A run is a folder that resolves to a direct child of the
    resolved runs folder (a link to a folder elsewhere is refused) and holds a run.json. A
    file must resolve, links included, to a regular file inside the resolved run folder (a
    symbolic link or junction that leaves it is refused), and both the requested name and
    the resolved file must have a suffix of :data:`SERVED_SUFFIXES`. The names a path
    resolves to must pass the segment rules too: a link, or a Windows 8.3 short name such as
    ``HIDDEN~1``, can stand for a hidden name. Directories are never listed. Documents the
    page reads (run.json and the others) follow the same rules. A hard link is not detected:
    it is a second name of the same file, so one inside a run folder is served even when the
    file's other name lies elsewhere (making one takes write access to the runs folder).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from gbf.ingest.rates import format_rate, parse_rate
from gbf.measure import MEASUREMENTS_NAME
from gbf.moments import BALL_MOMENT_KINDS, TOUCHES_NOTE_ZH
from gbf.pipeline import FINAL, FULL_CLIP_ID, WATERMARK, WATERMARK_TAGS, display_record
from gbf.render import HIGHLIGHT_TYPES_ZH, format_clock
from gbf.report import (
    METRIC_SIGNS_ZH,
    MOMENT_KINDS_ZH,
    REFERENCE_3D_METRICS,
    REFERENCE_3D_ZH,
    REFINED_BY_ZH,
    REPORT_NAME,
    TIER_ZH,
    TIMELINE_DARK_NAME,
    TIMELINE_NAME,
    ReportInputError,
    format_number,
    metric_row,
)

RUN_JSON = "run.json"
HIGHLIGHTS_NAME = "highlights.json"
LABEL_EVAL_NAME = "label_eval.json"
FULL_CLIP = f"clips/{FULL_CLIP_ID}.mp4"
FULL_MANIFEST = f"clips/{FULL_CLIP_ID}.render.json"

SERVED_SUFFIXES: Mapping[str, str] = MappingProxyType(
    {
        ".html": "text/html; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".mp4": "video/mp4",
        ".json": "application/json; charset=utf-8",
        # JSON lines has no registered type; as plain text the browser shows it instead of
        # saving a copy of the run's data into the downloads folder.
        ".jsonl": "text/plain; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".svg": "image/svg+xml",
    }
)
"""The only suffixes served from a run folder (lower case), with their content types."""

LIFT3D_METRICS: frozenset[str] = REFERENCE_3D_METRICS
"""Metrics gbf.measure computes from the gbf.lift3d height-prior skeleton (the report's
:data:`gbf.report.REFERENCE_3D_METRICS`). They are the registry's A_candidate tier (3D;
unverified until P4 measures them and they beat a 2D baseline), which :func:`is_lift3d`
also accepts for metrics added later."""

LIFT3D_BADGE = REFERENCE_3D_ZH

MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
"""Largest JSON or JSON lines document the page reads (the full clip's manifest is a few MB)."""

MAX_LISTED_FILES = 2000
"""Most files listed for one run."""

_FORBIDDEN_CHARACTERS = frozenset('/\\:*?"<>|')
_WINDOWS_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{n}" for port in ("COM", "LPT") for n in range(1, 10)}
)
_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


class RunPathError(Exception):
    """A URL path that names nothing servable. ``status`` is the HTTP status (403 when a
    rule refuses it, 404 when nothing is there) and ``reason`` says why, for the server's
    console only."""

    def __init__(self, status: int, reason: str) -> None:
        self.status = status
        self.reason = reason
        super().__init__(f"{status}: {reason}")


# --------------------------------------------------------------------------- paths


def segment_problem(segment: str) -> str | None:
    """Why one decoded URL path segment may not name a run or a file in one (None if it may).

    Refused: an empty, ``.`` or ``..`` segment, a separator, colon, wildcard or control
    character, a leading dot (hidden files, the server's own ``.gbf-web`` folder), a
    trailing dot or space (Windows drops them, so ``a.json.`` would open ``a.json``) and a
    Windows device name such as ``NUL`` or ``com1.json``.
    """
    if segment in ("", ".", ".."):
        return "empty or dot segment"
    if any(ch in _FORBIDDEN_CHARACTERS or ord(ch) < 32 or ord(ch) == 127 for ch in segment):
        return "separator, colon, wildcard or control character"
    if segment.startswith("."):
        return "hidden name"
    if segment[-1] in ". ":
        return "trailing dot or space"
    if segment.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_DEVICES:
        return "Windows device name"
    return None


def _inside(path: Path, base: Path) -> bool:
    return path == base or path.is_relative_to(base)


def _resolved_problem(path: Path, base: Path) -> str | None:
    """segment_problem of the first name of ``path`` (resolved, inside ``base``) below
    ``base`` that the segment rules refuse, else None."""
    for part in path.relative_to(base).parts:
        problem = segment_problem(part)
        if problem is not None:
            return problem
    return None


def _is_plain_dir(path: Path) -> bool:
    """True for a directory that is not a link: it resolves to itself (``path`` lies in a
    resolved folder, so only a link at its last component could resolve elsewhere)."""
    try:
        return path.is_dir() and path.resolve(strict=True) == path
    except (OSError, RuntimeError):
        return False


def run_file(base: Path, rel: str) -> Path | None:
    """The regular file ``rel`` (run-relative, POSIX) of the resolved run folder ``base``
    when it resolves inside ``base`` to names the segment rules allow, else None."""
    try:
        path = base.joinpath(*rel.split("/")).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not _inside(path, base) or _resolved_problem(path, base) is not None:
        return None
    if not path.is_file():
        return None
    return path


def run_folder(root: Path, name: str) -> Path:
    """The resolved folder of run ``name`` in the resolved runs folder ``root``.

    Raises RunPathError: 403 for a name a rule refuses, also once resolved (a link or an
    8.3 short name standing for a hidden folder), or a folder that resolves outside ``root``
    (a link), 404 when there is no such folder or it holds no run.json.
    """
    problem = segment_problem(name)
    if problem is not None:
        raise RunPathError(403, f"run name: {problem}")
    try:
        real = (root / name).resolve(strict=True)
    except (OSError, RuntimeError):
        raise RunPathError(404, "no such run") from None
    if real.parent != root:
        raise RunPathError(403, "the run folder resolves outside the runs folder")
    problem = segment_problem(real.name)
    if problem is not None:
        raise RunPathError(403, f"the run folder resolves to a refused name: {problem}")
    if not real.is_dir() or run_file(real, RUN_JSON) is None:
        raise RunPathError(404, "not a run (no run.json)")
    return real


def resolve_artifact(root: Path, name: str, parts: Sequence[str]) -> Path:
    """The file a request for ``/runs/<name>/<parts...>`` may be answered with.

    ``parts`` are the decoded segments after the run name. Raises RunPathError (403 for a
    refused segment, an unknown suffix, a path that leaves the run or one that resolves to
    a refused name; 404 when nothing is there or it is a directory).
    """
    base = run_folder(root, name)
    if not parts:
        raise RunPathError(404, "no file named")
    for part in parts:
        problem = segment_problem(part)
        if problem is not None:
            raise RunPathError(403, f"path segment: {problem}")
    if PurePosixPath(parts[-1]).suffix.lower() not in SERVED_SUFFIXES:
        raise RunPathError(403, "suffix not served")
    try:
        target = base.joinpath(*parts).resolve(strict=True)
    except (OSError, RuntimeError):
        raise RunPathError(404, "no such file") from None
    if not _inside(target, base):
        raise RunPathError(403, "the path resolves outside the run folder")
    problem = _resolved_problem(target, base)
    if problem is not None:
        raise RunPathError(403, f"the path resolves to a refused name: {problem}")
    if target.suffix.lower() not in SERVED_SUFFIXES:
        raise RunPathError(403, "the resolved file has a suffix that is not served")
    if not target.is_file():
        raise RunPathError(404, "not a file")
    return target


def served_files(base: Path) -> list[dict[str, Any]]:
    """The files of the resolved run folder ``base`` that may be served: run-relative POSIX
    path and size, sorted by path; hidden entries and links that leave the run are left out,
    and at most :data:`MAX_LISTED_FILES` are listed."""
    found: list[dict[str, Any]] = []
    for folder, dirs, files in os.walk(base):
        # Never descend through a link (a symbolic link or a junction resolves elsewhere):
        # its files may lie outside the run, and a link to a parent would never end.
        dirs[:] = sorted(
            d for d in dirs if segment_problem(d) is None and _is_plain_dir(Path(folder, d))
        )
        for file_name in sorted(files):
            if segment_problem(file_name) is not None:
                continue
            if PurePosixPath(file_name).suffix.lower() not in SERVED_SUFFIXES:
                continue
            rel = Path(folder, file_name).relative_to(base).as_posix()
            path = run_file(base, rel)
            if path is None or path.suffix.lower() not in SERVED_SUFFIXES:
                continue
            found.append({"path": rel, "bytes": path.stat().st_size})
            if len(found) >= MAX_LISTED_FILES:
                return sorted(found, key=lambda item: item["path"])
    return sorted(found, key=lambda item: item["path"])


# --------------------------------------------------------------------------- documents


def _no_constant(name: str) -> None:
    return None  # NaN and Infinity in a document are shown as missing values


def load_document(base: Path, rel: str, problems: list[str]) -> Any:
    """The parsed JSON document ``rel`` of the run folder ``base``, or None when it is
    missing; a document that cannot be read adds a Chinese note to ``problems``."""
    path = run_file(base, rel)
    if path is None:
        return None
    try:
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            problems.append(f"{rel} 太大，没有读取。")
            return None
        return json.loads(path.read_bytes().decode("utf-8"), parse_constant=_no_constant)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        problems.append(f"{rel} 无法读取（{type(exc).__name__}）。")
        return None


def load_records(base: Path, rel: str, problems: list[str]) -> list[dict[str, Any]]:
    """The records of the JSON lines file ``rel``; a line that does not parse as a JSON object
    is skipped and noted in ``problems``."""
    path = run_file(base, rel)
    if path is None:
        return []
    try:
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            problems.append(f"{rel} 太大，没有读取。")
            return []
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        problems.append(f"{rel} 无法读取（{type(exc).__name__}）。")
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, parse_constant=_no_constant)
        except ValueError:
            record = None
        if isinstance(record, dict):
            records.append(record)
        else:
            problems.append(f"{rel} 第 {number} 行不是有效的记录，已跳过。")
    return records


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def tags_of(value: Any) -> list[str]:
    """The restriction tags of a document's ``restrictions`` value (strings only), sorted."""
    if not isinstance(value, list):
        return []
    return sorted({tag for tag in value if isinstance(tag, str) and tag})


def banner_for(tags: Sequence[str]) -> str | None:
    """The private-material banner the renders carry (gbf.pipeline.WATERMARK) when the tags
    call for it, else None."""
    return WATERMARK if WATERMARK_TAGS & set(tags) else None


def _rate(*candidates: Any) -> Fraction | None:
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                return parse_rate(candidate)
            except ValueError:
                continue
    return None


def clock(frame: int, fps: Fraction | None) -> str | None:
    """gbf.render.format_clock of a canonical frame, or None without a usable rate."""
    if fps is None or frame < 0:
        return None
    return format_clock(frame, fps)


def _used_labels(run: Mapping[str, Any]) -> bool:
    inputs = run.get("inputs")
    return isinstance(inputs, list) and any(
        isinstance(item, Mapping) and item.get("name") == "labels" for item in inputs
    )


# --------------------------------------------------------------------------- listing


def _highlight_items(document: Any) -> list[Mapping[str, Any]] | None:
    items = _mapping(document).get("highlights")
    if not isinstance(items, list):
        return None
    return [item for item in items if isinstance(item, Mapping)]


def run_summary(base: Path, name: str) -> dict[str, Any]:
    """One row of the run list: name, status, creation time, video, highlight count, the
    restriction tags and the banner they call for."""
    problems: list[str] = []
    run = load_document(base, RUN_JSON, problems)
    document = _mapping(run)
    video = _mapping(_mapping(document.get("result")).get("video"))
    highlights = load_document(base, FINAL, problems)
    if highlights is None:
        highlights = load_document(base, HIGHLIGHTS_NAME, problems)
    items = _highlight_items(highlights)
    tags = tags_of(document.get("restrictions"))
    return {
        "name": name,
        "status": _text(document.get("status")) or ("unreadable" if run is None else "unknown"),
        "created_at": _text(document.get("created_at")),
        "video": _text(video.get("file_name")),
        "frames": _int(video.get("frames")),
        "fps": _text(video.get("fps")),
        "highlight_count": None if items is None else len(items),
        "restrictions": tags,
        "banner_zh": banner_for(tags),
        "problems": problems,
    }


def list_runs(root: Path) -> list[dict[str, Any]]:
    """Every run in the resolved runs folder ``root``, newest first (creation time, then
    name). Hidden folders, names that resolve to one, folders without run.json and links
    that leave ``root`` are not runs."""
    runs = []
    try:
        entries = sorted(root.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return []
    for entry in entries:
        if segment_problem(entry.name) is not None:
            continue
        try:
            base = run_folder(root, entry.name)
        except RunPathError:
            continue
        runs.append(run_summary(base, entry.name))
    runs.sort(key=lambda item: item["name"], reverse=True)
    runs.sort(key=lambda item: item["created_at"] or "", reverse=True)
    return runs


# --------------------------------------------------------------------------- the full clip


def full_clip_seeker(
    manifest: Any, fps: Fraction | None
) -> tuple[str, Callable[[int], float | None]]:
    """How to find a source frame in clips/full.mp4: (mapping, seek) where ``seek(frame)``
    is the player time in seconds (the middle of the frame) or None when the clip does not
    show that frame.

    With a render manifest (gbf.render, ``segments`` with ``src_start``, ``src_end`` and
    ``out_start``), frames are mapped through its normal-speed segments at its output rate
    (mapping "manifest"); without one, frame f is at f / fps of the source (mapping
    "frame_over_fps", the 1:1 rule of the full clip). Without a usable rate nothing can be
    sought (mapping "none").
    """
    document = _mapping(manifest)
    segments = document.get("segments")
    out_rate = _rate(_mapping(document.get("output")).get("fps")) or fps
    if isinstance(segments, list) and out_rate is not None:
        spans = []
        for segment in segments:
            item = _mapping(segment)
            if item.get("kind") != "normal" or _int(item.get("repeat")) not in (None, 1):
                continue
            start, end, out = (_int(item.get(key)) for key in ("src_start", "src_end", "out_start"))
            if start is not None and end is not None and out is not None and start <= end:
                spans.append((start, end, out))

        def seek_manifest(frame: int) -> float | None:
            for start, end, out in spans:
                if start <= frame <= end:
                    return float((out + frame - start + Fraction(1, 2)) / out_rate)
            return None

        return "manifest", seek_manifest
    if fps is None:
        return "none", lambda frame: None
    rate = fps

    def seek_one_to_one(frame: int) -> float | None:
        return float((frame + Fraction(1, 2)) / rate) if frame >= 0 else None

    return "frame_over_fps", seek_one_to_one


# --------------------------------------------------------------------------- measurements


def is_lift3d(record: Mapping[str, Any]) -> bool:
    """True for a value computed from the 3D lift: a metric of :data:`LIFT3D_METRICS` or a
    record of the A_candidate tier."""
    metric = record.get("metric_id") or record.get("metricId") or record.get("metric")
    return metric in LIFT3D_METRICS or record.get("tier") == "A_candidate"


def measurement_row(
    record: Mapping[str, Any], seek: Callable[[int], float | None]
) -> dict[str, Any] | None:
    """One evidence record as the page shows it, or None for a C-tier record (never shown).

    The fields come from gbf.report.metric_row of gbf.pipeline.display_record, as in the
    report, with the report's 3D 参考 badge as the page's ``lift3d`` kind (added here only for
    a value :func:`is_lift3d` accepts that the report does not badge), the tier's Chinese
    name, the sign convention of a signed angle and the full clip's time of the record's
    frame. Raises gbf.report.ReportInputError for a record that is not a mapping or names no
    metric.
    """
    row = metric_row(display_record(record))
    if row.blocked:
        return None
    lift = is_lift3d(record)
    badges = [
        {"kind": "lift3d" if kind == "reference_3d" else kind, "text": text}
        for kind, text in row.badges
    ]
    if lift and not any(badge["kind"] == "lift3d" for badge in badges):
        badges.append({"kind": "lift3d", "text": LIFT3D_BADGE})
    sign = METRIC_SIGNS_ZH.get(row.metric) if row.value_text != "—" else None
    return {
        "metric": row.metric,
        "label": row.label,
        "value_text": row.value_text,
        "unit": row.unit,
        "tier": row.tier,
        "tier_zh": TIER_ZH.get(row.tier, row.tier) if row.tier else "未分档",
        "badges": badges,
        "reason": row.reason,
        "sign_note": sign[1] if sign else None,
        "frame": row.frame,
        "track_id": row.track_id,
        "evidence_id": row.evidence_id,
        "lift3d": lift,
        "seek_s": seek(row.frame) if row.frame is not None else None,
    }


# --------------------------------------------------------------------------- highlights


def _touch_sentence(touch: Mapping[str, Any]) -> str | None:
    count = _int(touch.get("count"))
    if count is None:
        return None
    gap = _number(touch.get("mean_interval_s"))
    gap_text = f"，平均间隔 {format_number(gap, 2)} 秒" if gap else ""
    who = _text(touch.get("label_zh")) or "球员"
    track = _int(touch.get("track_id"))
    if track is not None and f"轨迹 {track}" not in who:
        who += f"（轨迹 {track}）"
    return f"检测到{who}触球 {count} 次{gap_text}（{TOUCHES_NOTE_ZH}；画面平面估计，未验证）。"


def _moments(
    item: Mapping[str, Any], fps: Fraction | None, seek: Callable[[int], float | None]
) -> list[dict[str, Any]]:
    rows = []
    for moment in item.get("key_moments") or []:
        moment = _mapping(moment)
        frame = _int(moment.get("frame"))
        if frame is None or frame < 0:
            continue
        kind = str(moment.get("kind") or "")
        tracks = moment.get("track_ids")
        rows.append(
            {
                "frame": frame,
                "clock": clock(frame, fps),
                "kind": kind,
                "kind_zh": MOMENT_KINDS_ZH.get(kind, kind),
                "text_zh": _text(moment.get("text_zh")) or "",
                "track_ids": [t for t in tracks if _int(t) is not None]
                if isinstance(tracks, list)
                else [],
                "seek_s": seek(frame),
            }
        )
    return rows


def _media(base: Path, rel: str) -> str | None:
    return rel if run_file(base, rel) is not None else None


def highlight_card(
    base: Path,
    item: Mapping[str, Any],
    name: str,
    fps: Fraction | None,
    seek: Callable[[int], float | None],
    rows: Sequence[dict[str, Any]],
    blocked: int,
) -> dict[str, Any]:
    """One highlight as the page shows it: name (H1, ...), type, title, summary, window and
    key frame with clocks and full-clip times, its media files, key moments, touches and
    measurement rows (already shaped by :func:`measurement_row`)."""
    ident = str(item["id"])
    start, peak, end = (int(item[key]) for key in ("start_frame", "peak_frame", "end_frame"))
    kind = str(item.get("type") or "")
    moments = _moments(item, fps, seek)
    touches = [_mapping(t) for t in item.get("touches") or []]
    with_ball = bool(touches) or any(m["kind"] in BALL_MOMENT_KINDS for m in moments)
    tier = item.get("tier")
    keyframe = _media(base, f"keyframes/{ident}.png")
    return {
        "id": ident,
        "name": name,
        "type": kind,
        "type_zh": HIGHLIGHT_TYPES_ZH.get(kind, kind),
        "tier": tier if isinstance(tier, str) else None,
        "tier_zh": {"main": "主要高光", "candidate": "候选"}.get(str(tier)),
        "rank": _int(item.get("rank")),
        "score": _number(item.get("score")),
        "score_text": format_number(float(item["score"]), 2)
        if _number(item.get("score")) is not None
        else None,
        "title_zh": _text(item.get("title_zh")) or f"{HIGHLIGHT_TYPES_ZH.get(kind, kind)} {name}",
        "summary_zh": _text(item.get("summary_zh")),
        "key_frame_rule_zh": REFINED_BY_ZH.get(str(item.get("refined_by"))),
        "start_frame": start,
        "end_frame": end,
        "peak_frame": peak,
        "start_clock": clock(start, fps),
        "end_clock": clock(end, fps),
        "peak_clock": clock(peak, fps),
        "peak_seek_s": seek(peak),
        "keyframe": keyframe,
        "card_image": _media(base, f"keyframes/{ident}-card.png") if keyframe else None,
        "clip": _media(base, f"clips/{ident}.mp4"),
        "manifest": _media(base, f"clips/{ident}.render.json"),
        "moments": moments,
        "moments_source_zh": "2D 姿态和球" if with_ball else "2D 姿态",
        "touches_zh": [s for s in (_touch_sentence(t) for t in touches) if s],
        "measurements": list(rows),
        "blocked_count": blocked,
    }


def _valid_highlight(item: Mapping[str, Any]) -> bool:
    if not isinstance(item.get("id"), str | int) or isinstance(item.get("id"), bool):
        return False
    frames = [_int(item.get(key)) for key in ("start_frame", "peak_frame", "end_frame")]
    return all(frame is not None and frame >= 0 for frame in frames)


def display_codes(text: str, names: Mapping[str, str]) -> str:
    """``text`` with highlight ids replaced by the page's names (H1, ...) and highlight type
    codes by their Chinese names, as the report does."""

    def name(match: re.Match[str]) -> str:
        code = match.group(0)
        return names.get(code) or HIGHLIGHT_TYPES_ZH.get(code) or code

    return _CODE.sub(name, text)


def label_comparison(document: Any, names: Mapping[str, str]) -> dict[str, Any] | None:
    """The comparison with labels (gbf.label_eval) for the page: counts, the Chinese
    sentences with the page's names, one row per label and the highlights no label
    matched, and where the labels come from in the label file's own words (``source_zh``
    and ``labels_source`` of label_eval.json, quoted as written, None when the run recorded
    none); None without a usable document."""
    data = _mapping(document)
    rows_in = data.get("rows")
    if not isinstance(rows_in, list):
        return None
    rows = []
    for row in rows_in:
        row = _mapping(row)
        highlight = row.get("highlight")
        covered = row.get("covered_by")
        label_type = str(row.get("label_type") or "")
        rows.append(
            {
                "label": str(row.get("label") or ""),
                "label_type_zh": HIGHLIGHT_TYPES_ZH.get(label_type, label_type),
                "label_peak": _int(row.get("label_peak")),
                "matched": bool(row.get("matched")),
                "highlight": names.get(str(highlight), str(highlight)) if highlight else None,
                "covered_by": names.get(str(covered), str(covered)) if covered else None,
                "peak": _int(row.get("peak")),
                "peak_error_frames": _int(row.get("peak_error_frames")),
                "tolerance_frames": _int(row.get("tolerance_frames")),
                "peak_within_tolerance": row.get("peak_within_tolerance")
                if isinstance(row.get("peak_within_tolerance"), bool)
                else None,
                "type_agrees": row.get("type_agrees")
                if isinstance(row.get("type_agrees"), bool)
                else None,
                "window_iou": _number(row.get("window_iou")),
            }
        )
    sentences = data.get("summary_zh")
    source = data.get("source_zh")
    fields = _mapping(data.get("labels_source"))
    unmatched = [
        names.get(str(_mapping(u).get("highlight")), str(_mapping(u).get("highlight")))
        for u in data.get("unmatched_highlights") or []
        if _mapping(u).get("highlight") is not None
    ]
    return {
        "counts": {
            key: _int(data.get(key))
            for key in (
                "n_labels",
                "n_highlights",
                "n_matched",
                "n_peak_within_tolerance",
                "n_type_agrees",
            )
        },
        "sentences_zh": [display_codes(s, names) for s in sentences if isinstance(s, str)]
        if isinstance(sentences, list)
        else [],
        "rows": rows,
        "unmatched": unmatched,
        "source_zh": source if isinstance(source, str) and source.strip() else None,
        "labels_source": {
            str(key): value for key, value in fields.items() if isinstance(value, str)
        },
    }


def run_detail(root: Path, name: str) -> dict[str, Any]:
    """Everything the run page shows (``GET /api/runs/<name>``); see the module docstring.

    The documents themselves are included as read (``highlights_final``, ``label_eval``),
    next to the shapes the page draws (``highlights``, ``labels``). Raises RunPathError
    when ``name`` is not a run of ``root``.
    """
    base = run_folder(root, name)
    problems: list[str] = []
    run = _mapping(load_document(base, RUN_JSON, problems))
    result = _mapping(run.get("result"))
    video = _mapping(result.get("video"))
    final = load_document(base, FINAL, problems)
    final_doc = _mapping(final)
    fps = _rate(video.get("fps"), final_doc.get("fps"))
    frame_count = _int(video.get("frames")) or _int(final_doc.get("frame_count"))
    tags = tags_of(run.get("restrictions"))

    full = _media(base, FULL_CLIP)
    manifest = load_document(base, FULL_MANIFEST, problems) if full else None
    mapping, seek = full_clip_seeker(manifest, fps) if full else ("none", lambda f: None)

    by_highlight: dict[str, list[dict[str, Any]]] = {}
    blocked: dict[str, int] = {}
    for record in load_records(base, MEASUREMENTS_NAME, problems):
        key = str(record.get("highlight_id") or record.get("highlightId") or "")
        try:
            row = measurement_row(record, seek)
        except ReportInputError:
            problems.append(f"{MEASUREMENTS_NAME} 里有一条记录没有指标名，已跳过。")
            continue
        if row is None:
            blocked[key] = blocked.get(key, 0) + 1
        else:
            by_highlight.setdefault(key, []).append(row)

    items = _highlight_items(final) or []
    valid = [item for item in items if _valid_highlight(item)]
    if len(valid) < len(items):
        problems.append(f"{FINAL} 里有 {len(items) - len(valid)} 个高光缺少编号或帧号，没有显示。")
    ordered = sorted(valid, key=lambda item: (int(item["peak_frame"]), str(item["id"])))
    names = {str(item["id"]): f"H{rank}" for rank, item in enumerate(ordered, start=1)}
    cards = [
        highlight_card(
            base,
            item,
            names[str(item["id"])],
            fps,
            seek,
            by_highlight.get(str(item["id"]), []),
            blocked.get(str(item["id"]), 0),
        )
        for item in ordered
    ]

    label_eval = load_document(base, LABEL_EVAL_NAME, problems) if _used_labels(run) else None
    report = _media(base, REPORT_NAME)
    timeline = _media(base, TIMELINE_NAME)
    parameters = _mapping(result.get("parameters"))
    team_names = _mapping(parameters.get("team_names"))
    automatic = (
        _mapping(load_document(base, "automatic.json", problems))
        if _media(base, "automatic.json")
        else {}
    )
    return {
        "name": name,
        "run": {
            "status": _text(run.get("status")) or "unknown",
            "created_at": _text(run.get("created_at")),
            "video": {
                "file_name": _text(video.get("file_name")),
                "frames": _int(video.get("frames")),
                "fps": _text(video.get("fps")),
                "width": _int(video.get("width")),
                "height": _int(video.get("height")),
            },
            "audio_source": _text(result.get("audio_source")),
            "team_names": {k: v for k, v in team_names.items() if isinstance(v, str)},
            "notes_zh": [n for n in result.get("notes_zh") or [] if isinstance(n, str)],
            "error": [e for e in result.get("error") or [] if isinstance(e, str)],
            "stages": [
                {
                    "name": str(_mapping(s).get("name")),
                    "status": str(_mapping(s).get("status")),
                    "wall_s": _number(_mapping(s).get("wall_s")),
                }
                for s in result.get("stages") or []
                if _mapping(s).get("name")
            ],
            "wall_s": _number(result.get("wall_s")),
        },
        "restrictions": tags,
        "automatic": {
            "wall_s": _number(automatic.get("wall_s")),
            "converted": automatic.get("conversion") is not None,
        }
        if automatic
        else None,
        "banner_zh": banner_for(tags),
        "fps": format_rate(fps) if fps is not None else None,
        "frame_count": frame_count,
        "files": served_files(base),
        "report": report,
        "timeline": {"light": timeline, "dark": _media(base, TIMELINE_DARK_NAME)}
        if timeline
        else None,
        "full_clip": {"path": full, "mapping": mapping} if full else None,
        "highlights": cards,
        "labels": label_comparison(label_eval, names) if label_eval is not None else None,
        "highlights_final": final,
        "label_eval": label_eval,
        "problems": problems,
    }


__all__ = [
    "LIFT3D_BADGE",
    "LIFT3D_METRICS",
    "SERVED_SUFFIXES",
    "RunPathError",
    "banner_for",
    "full_clip_seeker",
    "is_lift3d",
    "label_comparison",
    "list_runs",
    "measurement_row",
    "resolve_artifact",
    "run_detail",
    "run_file",
    "run_folder",
    "run_summary",
    "segment_problem",
    "served_files",
]
