"""The Chinese HTML report of one highlight run (M1).

:func:`write_report` writes ``report.html`` into ``out_dir``: one self-contained page (all
CSS inline, no script, no external font or library) whose only links are relative links to
PNG and MP4 files inside ``out_dir``. It also draws the highlight-score timeline with
Pillow (``timeline.png`` and ``timeline-dark.png``, selected by the reader's colour scheme).

Sections: overview (clip facts, the method in one paragraph, what is measured and what is
not), the timeline (with a table of the highlights as its text twin), one card per
highlight (title, times and frame range, type, key-frame image, the cues that selected it,
a metrics table with value, unit, tier badge and status badges, links to the clips), with
labels the comparison with them (where the labels come from in the label file's own words,
gbf.label_eval.source_zh), honest notes, and an appendix with the sha256 of every artifact.

Conventions and honesty rules (ADR 0004, docs/design/m1-highlights.md):

- Times are canonical frame indices, shown as minutes, seconds and hundredths of a second
  (frame / fps, floored, :func:`gbf.render.format_clock`); the page says that the two
  digits after the point are hundredths, not frames (``mm:ss.ff`` usually means frames).
- Highlights are called H1, H2, ... in peak order everywhere on the page; highlight ids
  and type codes in the evaluation sentences are replaced by these names and by the
  Chinese type names.
- Speeds are stored in m/s and shown in km/h with one decimal, rounded once from the stored
  value by one function for the cue list and the metrics table alike, so the two never
  disagree about the same stored speed.
- A value whose error is not measured carries the "未验证" badge; a value computed only in
  the image plane carries "仅画面平面"; a value of the 3D lift (facing and trunk lean, the
  A_candidate tier) carries "3D 参考": it is a reference, and every conclusion rests on the
  original frames; a missing value is shown as "不可用" with its reason, never as zero.
  C-tier values are never shown (only counted).
- The page says which numbers come from the original frames (2D pose, ball, player boxes
  and positions on calibrated frames) and which from the 3D lift (facing and trunk lean,
  the facing drawn back on the frames as an arrow), in the overview and in the notes.
- Every text is HTML-escaped. No absolute path, user name or drive letter is written.

Measurement records (``measurements.jsonl``, schema ``urn:gbf:schema:measurement``) are
read through :func:`metric_row`, which accepts camelCase or snake_case keys; the fields it
uses are listed there.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw

import gbf
from gbf.fonts import FontInfo, find_cjk_font, load_font
from gbf.highlights import REFINE_METHODS_ZH
from gbf.ingest.rates import format_rate, parse_rate
from gbf.kinematics import chord_half_frames
from gbf.measure import DEFAULT_PARAMS as MEASURE_PARAMS
from gbf.measure import metric_label
from gbf.moments import BALL_MOMENT_KINDS, TOUCHES_NOTE_ZH
from gbf.render import (
    FACING_ARROW_LENGTH_M,
    FACING_LABEL_ZH,
    HIGHLIGHT_TYPES_ZH,
    Badge,
    CardRow,
    format_clock,
    save_png,
    stamp_watermark,
)
from gbf.util.hashing import sha256_file
from gbf.util.io import write_text

REPORT_NAME = "report.html"
TIMELINE_NAME = "timeline.png"
TIMELINE_DARK_NAME = "timeline-dark.png"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
CLIP_SUFFIXES = frozenset({".mp4"})

TIER_ZH: dict[str, str] = {
    "official": "官方",
    "A": "A 档",
    "A_candidate": "A 候选（3D）",
    "B": "B 档（实验）",
    "C": "C 档（屏蔽）",
    "image_plane": "画面平面量",
}
"""Chinese names of the record tiers (gbf.measure RECORD_TIERS and C); ``image_plane``
marks an image measurement outside the A/B/C tiers."""

CUE_ZH: dict[str, str] = {
    "cut_emphasis": "导播切镜强调",
    "grammar": "转播语法",
    "broadcast_grammar": "转播语法",
    "audio": "现场声音响度变化",
    "audio_z": "现场声音响度变化",
    "excitement_z": "现场声音响度变化",
    "speed": "冲刺速度",
    "peak_speed": "冲刺速度",
    "box_pressure": "禁区内进攻人数",
    "attackers_in_box": "禁区内进攻人数",
    "ball_goal": "球距球门",
    "duel": "球附近双方最近距离",
    "nearest_opponent": "最近对手距离",
    "strike": "触球事件",
    "ball_strike": "触球事件",
    "score": "综合分数",
}
"""Chinese labels of highlight cues (gbf.highlights CUE_COLUMNS and older spellings);
unknown cue names are shown as they are. The audio cue measures loudness and spectral flux
only, the duel cue the distance between the nearest pair of opposing players near the ball
(close or not: it is no claim of body contact), and a ball event is any abrupt change of
the ball's image velocity (often a touch)."""

REFINED_BY_ZH: dict[str, str] = {
    **REFINE_METHODS_ZH,
    "pose_key_moment": "以原始画面上逐帧定出的关键瞬间为准",
}
"""How the key frame of a highlight was chosen (``refined_by`` of highlights.json): the
wording of gbf.highlights, plus the key moments of gbf.moments (pose alone, or pose and
ball)."""

REASON_ZH: dict[str, str] = {
    "no_calibration": "本帧没有可用标定",
    "not_pitch_usable": "本帧标定不满足球场可用条件",
    "calibration_unusable": "本帧标定不满足球场可用条件",
    "low_confidence": "置信度不足",
    "low_keypoint_confidence": "关键点置信度不足",
    "occluded": "被遮挡",
    "out_of_frame": "不在画面内",
    "too_small": "目标框太小",
    "box_too_small": "目标框太小",
    "no_ball": "没有检测到球",
    "no_pose": "没有可用姿态",
    "no_track": "没有可用轨迹",
    "segment_too_short": "连续可用帧太少",
    "tier_c_blocked": "C 档指标按规定屏蔽",
    "not_lifted": "未做三维提升",
}
"""Chinese text of common ``reasonCode`` values; unknown codes are shown as they are."""

METRIC_SIGNS_ZH: dict[str, tuple[str, str]] = {
    "image-trunk-lean": (
        "画面右倾为正",
        "髋部中点到两肩中点的连线与画面竖直方向的夹角：0° 为竖直，正值是上身向画面右侧倾斜，"
        "负值向左；画面里的角度，不是身体的真实前倾",
    ),
    "trunk-lean-at-contact": ("前倾为正", "正值是躯干前倾，负值是后仰"),
}
"""Sign conventions of signed angle metrics, by metric id: (short form for the data card,
sentence for the report's metrics table). The image trunk angle is the line from the hip
midpoint to the shoulder midpoint (gbf.lift3d.image_trunk_lean_deg)."""

REFERENCE_3D_METRICS: frozenset[str] = frozenset({"body-orientation", "trunk-lean-at-contact"})
"""gbf.measure metrics computed from the 3D lift (gbf.lift3d's height-prior skeleton: facing
and trunk lean). :func:`metric_row` badges them, and every A_candidate value (ADR 0004:
A_candidate is 3D), "3D 参考": a reference beside what the original frames show."""

REFERENCE_3D_ZH = "3D 参考"
"""Badge text of a value of the 3D lift (badge kind ``reference_3d``)."""

UNIT_ZH: dict[str, str] = {
    "deg": "°",
    "degree": "°",
    "degrees": "°",
    "m": "m",
    "px": "px",
    "s": "s",
    "frame": "帧",
    "frames": "帧",
    "km/h": "km/h",
    "count": "个",
}

_DRIVE = re.compile(r"^[A-Za-z]:")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


class ReportInputError(ValueError):
    """The report inputs are invalid. ``problems`` lists every problem found."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"write_report: {len(self.problems)} problem(s)\n  - {joined}")


@dataclass(frozen=True)
class ClipLink:
    """A clip of one highlight: a Chinese label and a path relative to ``out_dir``."""

    label: str
    path: str | Path


@dataclass(frozen=True)
class ImageLink:
    """An image of one highlight: a path relative to ``out_dir`` and an optional caption
    (default: "关键帧 ..." for the first image of a highlight, "附图 n" for the others)."""

    path: str | Path
    caption: str | None = None


@dataclass(frozen=True)
class ArtifactRef:
    """An artifact listed in the appendix: display name, sha256 and size in bytes."""

    name: str
    sha256: str
    bytes: int | None = None


@dataclass(frozen=True)
class ReportMeta:
    """Facts about the clip and the run that the report shows.

    Attributes:
        clip_name: display name of the clip (a file name, never a path).
        fps: exact frame rate, e.g. "25/1".
        frame_count: canonical frames of the clip.
        width, height: frame size in pixels.
        source_sha256: sha256 of the source video bytes.
        title: page title.
        cuts: first frame of every new shot (hard cuts).
        shots: (start_frame, end_frame inclusive, shot type) per shot.
        score: per-frame highlight score (``signals.parquet`` column ``score``), one value
            per canonical frame; NaN allowed; ``None`` when not available.
        calibration: summary numbers, any of: frames; fresh, rep_err_median_px and
            rep_err_p90_px (the plugin's per-frame calibration); pitch_usable (frames that
            give pitch coordinates, after the refit when there was one); threshold_px (the
            largest per-frame reprojection error, of the plugin's frames and of the refit);
            refit: None when the calibration was not refitted, else a mapping with any of
            pitch_usable_before, refined_frames, rejected_frames, holdout_median_px,
            holdout_p90_px and holdout_p90_max_px (gbf.calib_refine: usable frames before
            the refit, refitted frames, frames of camera groups rejected for their
            held-out error, the held-out keypoint error and the rejection limit). With
            refined_frames 0 the report says that no frame was refitted; without held-out
            numbers or rejected frames it says nothing about a held-out check.
        audio_source: file name of the file the audio came from.
        restrictions: restriction tags (ADR 0002) of the material in this report.
        watermark: optional visible badge stamped on both exported timeline images.
        artifacts: extra artifacts for the appendix (files linked from the report are
            hashed automatically).
        tools: (name, version) pairs shown in the appendix.
        font: font for the timeline chart (default: a system CJK font).
        method_zh: replaces the default method paragraph.
        notes_zh: extra honest notes.
        evaluation_zh: sentences comparing the highlights with labels of the clip
            (gbf.label_eval.summary_zh); shown in their own section when given.
        evaluation_source_zh: where those labels come from, in the label file's own words
            (gbf.label_eval.source_zh); without it the section says that no source was
            given, and claims none.
    """

    clip_name: str
    fps: str
    frame_count: int
    width: int
    height: int
    source_sha256: str
    title: str = "高光时刻分析报告"
    cuts: tuple[int, ...] = ()
    shots: tuple[tuple[int, int, str], ...] = ()
    score: tuple[float, ...] | None = None
    calibration: Mapping[str, Any] | None = None
    audio_source: str | None = None
    restrictions: tuple[str, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    tools: tuple[tuple[str, str], ...] = ()
    font: FontInfo | None = None
    method_zh: str | None = None
    notes_zh: tuple[str, ...] = ()
    evaluation_zh: tuple[str, ...] = ()
    evaluation_source_zh: str | None = None
    watermark: str | None = None


@dataclass(frozen=True)
class MetricRow:
    """One measurement record as the report shows it (see :func:`metric_row`)."""

    evidence_id: str | None
    highlight_id: str | None
    metric: str
    label: str
    value_text: str
    unit: str
    tier: str | None
    badges: tuple[tuple[str, str], ...]
    reason: str | None
    frame: int | None
    track_id: int | None
    blocked: bool


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _e(value: Any) -> str:
    """HTML-escaped text (quotes included)."""
    return html.escape(str(value), quote=True)


def _get(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def format_number(value: float, decimals: int | None = None) -> str:
    """Display text of a number: ``decimals`` if given, else 0 decimals from 100 up,
    1 from 1 up and 2 below 1 (so 27.36 -> "27.4", 152.3 -> "152", 0.456 -> "0.46")."""
    if not math.isfinite(value):
        return "—"
    if decimals is None:
        magnitude = abs(value)
        decimals = 0 if magnitude >= 100 else 1 if magnitude >= 1 else 2
    text = f"{value:.{decimals}f}"
    return "0" if text in ("-0", "-0.0", "-0.00") and decimals == 0 else text


def _kmh(mps: float) -> str:
    """A speed stored in m/s as the page shows it: km/h with one decimal (as
    gbf.measure.display_text), rounded once from the stored value. The cue list and the
    metrics table both use it, so the same stored speed never reads two ways."""
    return format_number(float(mps) * 3.6, 1)


def metric_row(record: Mapping[str, Any]) -> MetricRow:
    """Normalise one measurement record for display.

    Fields read (camelCase or snake_case): ``evidenceId``; ``highlightId``;
    ``metric``/``metricId``; ``label_zh``/``name_zh``/``label``; ``value``; ``unit``;
    ``decimals``; ``tier``; ``error.kind`` (or ``errorKind``) and ``error.receiptId``;
    ``status`` ("unavailable" marks a missing value); ``reasonCode`` and ``reason_zh``;
    ``plane``/``space`` ("image" or "image_plane" marks an image-plane quantity) and
    ``method`` ("image_plane_only" too); ``claimType``; ``frame``; ``trackId``.

    Speeds stored in m/s are shown in km/h with one decimal (whatever ``decimals`` says, as
    on the record's own display text). A value of the 3D lift (tier A_candidate, or a
    metric of :data:`REFERENCE_3D_METRICS`) carries the "3D 参考" badge, available or not,
    after its other badges. Raises :class:`ReportInputError` when the record is not a
    mapping or names no metric.
    """
    if not isinstance(record, Mapping):
        raise ReportInputError([f"a measurement record must be a mapping, got {record!r}"])
    metric = _get(record, "metric", "metricId", "metric_id")
    if not isinstance(metric, str) or not metric:
        raise ReportInputError([f"measurement record without a metric id: {dict(record)!r}"])
    label = _get(record, "label_zh", "labelZh", "name_zh", "nameZh", "label", "name")
    label = label if isinstance(label, str) and label else metric
    tier = _get(record, "tier")
    tier = tier if isinstance(tier, str) else None
    unit = _get(record, "unit")
    unit = unit if isinstance(unit, str) else ""
    value = record.get("value")
    decimals = _int_or_none(record.get("decimals"))
    status = _get(record, "status")
    error = record.get("error") if isinstance(record.get("error"), Mapping) else {}
    error_kind = _get(error, "kind") or _get(record, "errorKind", "error_kind")
    receipt = _get(error, "receiptId", "receipt_id")
    plane = _get(record, "plane", "space", "frame_of_reference")
    method = _get(record, "method")
    claim = _get(record, "claimType", "claim_type")
    reason_code = _get(record, "reasonCode", "reason_code")
    reason = _get(record, "reason_zh", "reasonZh")
    if not isinstance(reason, str) and isinstance(reason_code, str):
        reason = REASON_ZH.get(reason_code, reason_code)
    blocked = tier == "C"
    unavailable = status in ("unavailable", "missing") or value is None

    if unavailable or blocked:
        value_text = "—"
    elif isinstance(value, bool):
        value_text = "是" if value else "否"
    elif isinstance(value, int | np.integer) and unit != "m/s" and decimals is None:
        value_text = str(int(value))  # counts stay integers
    elif isinstance(value, int | float | np.integer | np.floating) and unit == "m/s":
        value_text, unit = _kmh(float(value)), "km/h"
    elif isinstance(value, int | float | np.integer | np.floating):
        value_text = format_number(float(value), decimals)
    else:
        value_text = str(value)
    unit_text = UNIT_ZH.get(unit, unit)

    badges: list[tuple[str, str]] = []
    if blocked:
        badges.append(("unavailable", "已屏蔽"))
        reason = reason or REASON_ZH["tier_c_blocked"]
    elif unavailable:
        badges.append(("unavailable", "不可用"))
    else:
        if tier == "official" or claim in ("provider_model", "official_fact"):
            badges.append(("official", "官方数据"))
        elif error_kind == "measured" and receipt:
            badges.append(("measured", "已测量"))
        elif error_kind == "literature":
            badges.append(("unverified", "未验证（仅文献参考）"))
        else:
            badges.append(("unverified", "未验证"))
        if tier == "B":
            badges.append(("experimental", "实验"))
    image_plane = (isinstance(plane, str) and plane.lower() in ("image", "image_plane")) or (
        method == "image_plane_only"
    )
    if image_plane and not blocked:
        badges.append(("image_plane", "仅画面平面"))
    if (tier == "A_candidate" or metric in REFERENCE_3D_METRICS) and not blocked:
        badges.append(("reference_3d", REFERENCE_3D_ZH))
    evidence = _get(record, "evidenceId", "evidence_id", "id")
    highlight = _get(record, "highlightId", "highlight_id")
    return MetricRow(
        evidence_id=str(evidence) if evidence is not None else None,
        highlight_id=str(highlight) if highlight is not None else None,
        metric=metric,
        label=label,
        value_text=value_text,
        unit=unit_text if value_text != "—" else "",
        tier=tier,
        badges=tuple(badges),
        reason=reason if isinstance(reason, str) else None,
        frame=_int_or_none(_get(record, "frame", "peak_frame", "peakFrame")),
        track_id=_int_or_none(_get(record, "trackId", "track_id")),
        blocked=blocked,
    )


def card_rows(
    rows: Sequence[MetricRow],
    limit: int = 5,
    key_frame: int | None = None,
    *,
    name_tracks: bool = False,
    focus_track: int | None = None,
) -> tuple[CardRow, ...]:
    """Rows of a freeze-frame data card (gbf.render.InfoCard) for the first ``limit``
    displayable measurement rows: label, value, unit, the status badges and the evidence id.

    A row measured on another frame than ``key_frame`` names its frame and, with
    ``name_tracks``, a row whose value belongs to another player than ``focus_track`` (to
    any player when it is None) names the player, so a number on the frozen frame is never
    mistaken for a measurement of that frame or of the player in focus. Both go before the
    label, without spaces ("轨迹42·第2391帧·<label>"): gbf.render cuts a label that does not
    fit at its end, so the player and the frame stay on the card and leave the label as much
    room as they can. A signed angle says which sign is which after its unit
    (:data:`METRIC_SIGNS_ZH`). C-tier rows are left out, as everywhere else."""
    out: list[CardRow] = []
    for row in rows:
        if row.blocked:
            continue
        badges = tuple(Badge(text, kind) for kind, text in row.badges)  # type: ignore[arg-type]
        evidence = (row.evidence_id,) if row.evidence_id else ()
        qualifiers = []
        if name_tracks and row.track_id is not None and row.track_id != focus_track:
            qualifiers.append(f"轨迹{row.track_id}")
        if key_frame is not None and row.frame is not None and row.frame != key_frame:
            qualifiers.append(f"第{row.frame}帧")
        label = "·".join((*qualifiers, row.label))
        unit = row.unit
        sign = METRIC_SIGNS_ZH.get(row.metric)
        if sign is not None and row.value_text != "—":
            unit = f"{unit}（{sign[0]}）"
        out.append(CardRow(label, row.value_text, unit, badges, evidence))
        if len(out) == limit:
            break
    return tuple(out)


def _relative_file(
    out_dir: Path, value: Any, suffixes: frozenset[str], what: str, problems: list[str]
) -> str | None:
    """``value`` as a POSIX path relative to ``out_dir`` that names an existing file."""
    if isinstance(value, Path):
        candidate = value
        text = value.as_posix()
    elif isinstance(value, str) and value:
        text = value
        candidate = Path(value)
    else:
        problems.append(f"{what}: expected a relative file path, got {value!r}")
        return None
    root = out_dir.resolve()
    if candidate.is_absolute() or _DRIVE.match(text) or text.startswith(("/", "//")):
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            problems.append(f"{what}: {candidate.name} lies outside the report directory")
            return None
        relative = resolved.relative_to(root).as_posix()
    else:
        if "\\" in text:
            problems.append(f"{what}: use forward slashes in {text!r}")
            return None
        if _SCHEME.match(text):
            problems.append(f"{what}: {text!r} is not a relative file path")
            return None
        pure = PurePosixPath(text)
        if any(part == ".." for part in pure.parts):
            problems.append(f"{what}: {text!r} must not contain '..'")
            return None
        relative = pure.as_posix()
        resolved = (root / pure).resolve()
        if not resolved.is_relative_to(root):
            problems.append(f"{what}: {text!r} lies outside the report directory")
            return None
    if resolved.suffix.lower() not in suffixes:
        problems.append(f"{what}: {relative!r} must end in one of {sorted(suffixes)}")
        return None
    if not resolved.is_file():
        problems.append(f"{what}: {relative!r} does not exist in the report directory")
        return None
    return relative


def _href(relative: str) -> str:
    return _e(quote(relative, safe="/"))


# --------------------------------------------------------------------------------------
# input checks
# --------------------------------------------------------------------------------------
_REQUIRED_HIGHLIGHT_KEYS = ("id", "type", "start_frame", "peak_frame", "end_frame", "score")


def _check_highlights(
    highlights: Any, meta: ReportMeta, problems: list[str]
) -> list[Mapping[str, Any]]:
    items: Any = highlights
    if isinstance(highlights, Mapping):
        fps = highlights.get("fps")
        if fps is not None:
            try:
                if parse_rate(str(fps)) != parse_rate(meta.fps):
                    problems.append(f"highlights fps {fps} differs from meta.fps {meta.fps}")
            except ValueError as exc:
                problems.append(f"highlights fps: {exc}")
        count = highlights.get("frame_count")
        if count is not None and count != meta.frame_count:
            problems.append(
                f"highlights frame_count {count} differs from meta.frame_count {meta.frame_count}"
            )
        items = highlights.get("highlights")
    if not isinstance(items, Sequence) or isinstance(items, str | bytes):
        problems.append("highlights must be a list of highlight objects (or highlights.json)")
        return []
    checked: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        where = f"highlights[{index}]"
        if not isinstance(item, Mapping):
            problems.append(f"{where} must be an object, got {type(item).__name__}")
            continue
        missing = [key for key in _REQUIRED_HIGHLIGHT_KEYS if key not in item]
        if missing:
            problems.append(f"{where} misses {', '.join(missing)}")
            continue
        ident = item["id"]
        if not isinstance(ident, str) or not ident:
            problems.append(f"{where}.id must be a non-empty string, got {ident!r}")
            continue
        if ident in seen:
            problems.append(f"{where}.id {ident!r} is used twice")
        seen.add(ident)
        frames = [_int_or_none(item[key]) for key in ("start_frame", "peak_frame", "end_frame")]
        if None in frames:
            problems.append(f"{where} ({ident}): start/peak/end frames must be integers")
            continue
        start, peak, end = (int(f) for f in frames)  # type: ignore[arg-type]
        if not 0 <= start <= peak <= end < meta.frame_count:
            problems.append(
                f"{where} ({ident}): need 0 <= start <= peak <= end < {meta.frame_count}, "
                f"got {start}, {peak}, {end}"
            )
        score = item["score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            problems.append(f"{where} ({ident}): score must be a finite number, got {score!r}")
        cues = item.get("cues", {})
        if cues is not None and not isinstance(cues, Mapping):
            problems.append(f"{where} ({ident}): cues must be an object")
        for key in ("title_zh", "summary_zh"):
            if item.get(key) is not None and not isinstance(item.get(key), str):
                problems.append(f"{where} ({ident}): {key} must be a string")
        checked.append(item)
    return checked


def _check_meta(meta: Any, problems: list[str]) -> bool:
    if not isinstance(meta, ReportMeta):
        problems.append(f"meta must be a ReportMeta, got {type(meta).__name__}")
        return False
    try:
        parse_rate(meta.fps)
    except ValueError as exc:
        problems.append(f"meta.fps: {exc}")
    for name in ("frame_count", "width", "height"):
        value = getattr(meta, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            problems.append(f"meta.{name} must be a positive int, got {value!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(meta.source_sha256)):
        problems.append("meta.source_sha256 must be 64 lowercase hex digits")
    if "/" in meta.clip_name or "\\" in meta.clip_name or _DRIVE.match(meta.clip_name):
        problems.append("meta.clip_name must be a file name, not a path")
    if meta.score is not None and len(meta.score) != meta.frame_count:
        problems.append(
            f"meta.score has {len(meta.score)} values, expected one per frame ({meta.frame_count})"
        )
    bad_cuts = [c for c in meta.cuts if not isinstance(c, int) or not 0 < c < meta.frame_count]
    if bad_cuts:
        problems.append(f"meta.cuts must be frames in 1..frame_count-1, got {bad_cuts}")
    problems.extend(_calibration_problems(meta.calibration))
    for name in ("notes_zh", "evaluation_zh"):
        problems.extend(_sentences_problems(name, getattr(meta, name)))
    source = meta.evaluation_source_zh
    if source is not None and not isinstance(source, str):
        problems.append(
            f"meta.evaluation_source_zh must be a string or None, got {type(source).__name__}"
        )
    if meta.watermark is not None and (
        not isinstance(meta.watermark, str) or not meta.watermark.strip()
    ):
        problems.append("meta.watermark must be a non-empty string or None")
    return True


def _sentences_problems(name: str, value: Any) -> list[str]:
    """``meta.<name>`` must be a sequence of strings (one sentence each)."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        return [f"meta.{name} must be a sequence of strings, got {type(value).__name__}"]
    return [
        f"meta.{name}[{index}] must be a string, got {type(item).__name__} {item!r}"
        for index, item in enumerate(value)
        if not isinstance(item, str)
    ]


_CALIBRATION_NUMBERS = (
    "frames",
    "fresh",
    "pitch_usable",
    "rep_err_median_px",
    "rep_err_p90_px",
    "threshold_px",
)
_REFIT_NUMBERS = (
    "pitch_usable_before",
    "refined_frames",
    "rejected_frames",
    "holdout_median_px",
    "holdout_p90_px",
    "holdout_p90_max_px",
)


def _calibration_problems(calibration: Any) -> list[str]:
    """Every number the overview prints must be a finite number or None."""
    if calibration is None:
        return []
    if not isinstance(calibration, Mapping):
        return [f"meta.calibration must be a mapping or None, got {type(calibration).__name__}"]
    refit = calibration.get("refit")
    if refit is not None and not isinstance(refit, Mapping):
        return [f"meta.calibration['refit'] must be a mapping or None, got {refit!r}"]
    found = []
    for where, values, keys in (
        ("meta.calibration", calibration, _CALIBRATION_NUMBERS),
        ("meta.calibration['refit']", refit or {}, _REFIT_NUMBERS),
    ):
        for key in keys:
            value = values.get(key)
            if value is None:
                continue
            number = isinstance(value, int | float | np.integer | np.floating)
            if isinstance(value, bool) or not number or not math.isfinite(float(value)):
                found.append(f"{where}[{key!r}] must be a finite number or None, got {value!r}")
    return found


# --------------------------------------------------------------------------------------
# timeline chart (Pillow)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _ChartTheme:
    surface: tuple[int, int, int]
    ink: tuple[int, int, int]
    ink_secondary: tuple[int, int, int]
    muted: tuple[int, int, int]
    grid: tuple[int, int, int]
    baseline: tuple[int, int, int]
    score: tuple[int, int, int]
    window: tuple[int, int, int]


# Colours from the project's validated chart palette (both pairs pass the CVD and contrast
# checks on their own surface): blue for the score, orange for highlight windows.
LIGHT_THEME = _ChartTheme(
    surface=(252, 252, 251),
    ink=(11, 11, 11),
    ink_secondary=(82, 81, 78),
    muted=(137, 135, 129),
    grid=(225, 224, 217),
    baseline=(195, 194, 183),
    score=(42, 120, 214),
    window=(235, 104, 52),
)
DARK_THEME = _ChartTheme(
    surface=(26, 26, 25),
    ink=(240, 239, 236),
    ink_secondary=(195, 194, 183),
    muted=(137, 135, 129),
    grid=(44, 44, 42),
    baseline=(56, 56, 53),
    score=(57, 135, 229),
    window=(217, 89, 38),
)


def _nice_step(span: float, target: int) -> float:
    raw = span / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10 * magnitude


#: Steps of the timeline's time axis in hundredths of a second: round fractions of a
#: second, round numbers of seconds and of minutes.
_TIME_STEPS_CS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 3000, 6000, 12000, 30000)


def _time_ticks(frame_count: int, rate: Fraction, target: int = 8) -> list[tuple[Fraction, str]]:
    """Ticks of the timeline's time axis as (frame position, label): the smallest step of
    :data:`_TIME_STEPS_CS` that makes at most ``target`` steps of the clip, from 0 to the
    last frame. Whole seconds are labelled mm:ss and a step under a second mm:ss.cc
    (hundredths, as on the rest of the page), so no two ticks share a label."""
    span = Fraction(frame_count) * 100 / rate / max(target, 1)
    step = next((s for s in _TIME_STEPS_CS if s >= span), math.ceil(span / 6000) * 6000)
    last = math.floor(Fraction(frame_count - 1) * 100 / rate)
    ticks = []
    for centis in range(0, last + 1, step):
        minutes, rest = divmod(centis, 6000)
        seconds, hundredths = divmod(rest, 100)
        label = f"{minutes:02d}:{seconds:02d}"
        if step % 100:
            label += f".{hundredths:02d}"
        ticks.append((Fraction(centis, 100) * rate, label))
    return ticks


def draw_timeline(
    meta: ReportMeta,
    highlights: Sequence[Mapping[str, Any]],
    *,
    theme: _ChartTheme = LIGHT_THEME,
    size: tuple[int, int] = (1600, 420),
    font: FontInfo | None = None,
) -> Image.Image:
    """The highlight-score timeline as an RGB image of ``size`` pixels.

    x is time (canonical frame / fps, ticks from :func:`_time_ticks`); y the per-frame score
    (2 px line over a light wash) when ``meta.score`` is given; highlight windows are orange
    bands with a marker on the per-frame score of the key frame (``peak_frame``; not the
    highlight's smoothed ``score``, which the legend and the report's table keep apart) and
    a direct label (H1, H2, ...); hard cuts are hairlines. Drawn at twice the size and
    reduced, so lines are anti-aliased.
    """
    scale = 2
    width, height = size[0] * scale, size[1] * scale
    font_info = font or meta.font or find_cjk_font("regular")
    small = load_font(font_info, 13 * scale)
    label_font = load_font(font_info, 14 * scale)
    image = Image.new("RGB", (width, height), theme.surface)
    draw = ImageDraw.Draw(image, "RGBA")
    left, right, top, bottom = 64 * scale, 20 * scale, 52 * scale, 44 * scale
    plot_w, plot_h = width - left - right, height - top - bottom
    rate = parse_rate(meta.fps)

    def x_of(frame: float) -> float:
        return left + plot_w * (frame / max(meta.frame_count - 1, 1))

    score = np.asarray(meta.score, dtype=np.float64) if meta.score is not None else None
    finite = score[np.isfinite(score)] if score is not None else np.empty(0)
    if finite.size:
        y_min = min(0.0, float(finite.min()))
        y_max = float(finite.max())
    else:
        y_min, y_max = 0.0, 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    step = _nice_step(y_max - y_min, 4)
    y_min = math.floor(y_min / step) * step
    y_max = math.ceil(y_max * 1.08 / step) * step

    def y_of(value: float) -> float:
        return top + plot_h * (1.0 - (value - y_min) / (y_max - y_min))

    # gridlines and y ticks (score)
    tick = y_min
    while tick <= y_max + step * 1e-6:
        y = y_of(tick)
        draw.line([(left, y), (left + plot_w, y)], fill=theme.grid, width=scale)
        if finite.size:
            draw.text((left - 8 * scale, y), f"{tick:g}", fill=theme.muted, font=small, anchor="rm")
        tick += step
    # x ticks (time); a label at the edge is moved inside the image, its tick stays
    for position, label in _time_ticks(meta.frame_count, rate):
        x = x_of(float(position))
        draw.line(
            [(x, top + plot_h), (x, top + plot_h + 5 * scale)], fill=theme.baseline, width=scale
        )
        half = draw.textlength(label, font=small) / 2
        draw.text(
            (min(max(x, half), width - half), top + plot_h + 9 * scale),
            label,
            fill=theme.muted,
            font=small,
            anchor="ma",
        )
    # highlight windows (under the score line)
    for item in highlights:
        x0, x1 = x_of(item["start_frame"]), x_of(item["end_frame"])
        draw.rectangle([x0, top, max(x1, x0 + scale), top + plot_h], fill=(*theme.window, 34))
    # cuts
    for cut in meta.cuts:
        x = x_of(cut - 0.5)
        draw.line([(x, top), (x, top + plot_h)], fill=theme.muted, width=scale)
    # score
    if score is not None and finite.size:
        xs = [x_of(i) for i in range(meta.frame_count)]
        runs: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for i, value in enumerate(score):
            if math.isfinite(value):
                current.append((xs[i], y_of(float(value))))
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        base_y = y_of(max(y_min, 0.0))
        for run in runs:
            if len(run) >= 2:
                polygon = [(run[0][0], base_y), *run, (run[-1][0], base_y)]
                draw.polygon(polygon, fill=(*theme.score, 26))
        for run in runs:
            if len(run) >= 2:
                draw.line(run, fill=theme.score, width=2 * scale, joint="curve")
    else:
        draw.text(
            (left + plot_w / 2, top + plot_h / 2),
            "没有逐帧分数（只标出高光窗口）",
            fill=theme.muted,
            font=label_font,
            anchor="mm",
        )
    # baseline axis
    draw.line(
        [(left, top + plot_h), (left + plot_w, top + plot_h)], fill=theme.baseline, width=scale
    )
    # peak markers and direct labels
    for rank, item in enumerate(highlights, start=1):
        peak = int(item["peak_frame"])
        x = x_of(peak)
        if score is not None and math.isfinite(float(score[peak])):
            y = y_of(float(score[peak]))
        else:
            y = top + 10 * scale
        r = 5 * scale
        draw.ellipse(
            [x - r - 2 * scale, y - r - 2 * scale, x + r + 2 * scale, y + r + 2 * scale],
            fill=theme.surface,
        )
        draw.ellipse([x - r, y - r, x + r, y + r], fill=theme.window)
        draw.text((x, y - r - 6 * scale), f"H{rank}", fill=theme.ink, font=label_font, anchor="mb")
    # legend (identity is never colour-alone: every key carries a label)
    lx, ly = left, 18 * scale
    entries: list[tuple[str, str]] = []
    if score is not None and finite.size:
        entries.append(("line", "逐帧高光分数"))
    entries.append(("band", "高光窗口（H 编号见下表）"))
    entries.append(("dot", "关键帧的逐帧分数" if score is not None and finite.size else "关键帧"))
    if meta.cuts:
        entries.append(("cut", "硬切镜"))
    for kind, text in entries:
        if kind == "line":
            draw.line([(lx, ly), (lx + 22 * scale, ly)], fill=theme.score, width=2 * scale)
        elif kind == "band":
            draw.rectangle(
                [lx, ly - 7 * scale, lx + 22 * scale, ly + 7 * scale], fill=(*theme.window, 60)
            )
        elif kind == "dot":
            draw.ellipse(
                [lx + 6 * scale, ly - 5 * scale, lx + 16 * scale, ly + 5 * scale], fill=theme.window
            )
        else:
            draw.line(
                [(lx + 11 * scale, ly - 8 * scale), (lx + 11 * scale, ly + 8 * scale)],
                fill=theme.muted,
                width=scale,
            )
        lx += 28 * scale
        draw.text((lx, ly), text, fill=theme.ink_secondary, font=label_font, anchor="lm")
        lx += draw.textlength(text, font=label_font) + 26 * scale
    return image.resize(size, Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------
_NAV_ITEMS = (
    ("overview", "概览"),
    ("timeline", "时间轴"),
    ("highlights", "高光"),
    ("evaluation", "标注对比"),
    ("notes", "诚实说明"),
    ("appendix", "附录"),
)


def _nav(meta: ReportMeta) -> str:
    return "".join(
        f'<a href="#{anchor}">{label}</a>'
        for anchor, label in _NAV_ITEMS
        if anchor != "evaluation" or meta.evaluation_zh
    )


#: Chinese names of the key-moment kinds of gbf.moments. ``contact_start`` and
#: ``contact_end`` bound the time two players' keypoints are close in the image (身体贴近),
#: which is not a measured touch; ``on_ground`` marks the frame the trunk passes 45° in the
#: image (the body reaches the ground a few frames later); ``ball_away`` is a ball leaving
#: a player's feet without the two coming close.
MOMENT_KINDS_ZH = {
    "contact_start": "身体贴近",
    "touch": "触球",
    "ball_release": "球脱离争抢",
    "ball_away": "球离开脚下",
    "fall_onset": "躯干开始倾倒",
    "on_ground": "躯干过 45°",
    "contact_end": "身体贴近结束",
    "hand_up": "抬手过肩",
    "arm_top": "手举到最高（伸直）",
    "hand_top": "手举到最高",
}

_CSS = """
:root {
  color-scheme: light dark;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --line: #e1e0d9; --accent: #eb6834; --blue: #2a78d6;
  --warn: #fab219; --serious: #ec835a; --critical: #d03b3b; --good: #0ca30c;
  --plane: #6250d6; --grey: #898781;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --line: #2c2c2a; --accent: #d95926; --blue: #3987e5; --plane: #9085e9;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 16px/1.65 "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Noto Sans SC",
    system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 28px; line-height: 1.3; margin: 8px 0 4px; }
h2 {
  font-size: 21px; margin: 40px 0 12px; padding-bottom: 6px;
  border-bottom: 1px solid var(--line);
}
h3 { font-size: 18px; margin: 0; }
p { margin: 8px 0; }
a { color: var(--blue); }
.sub { color: var(--ink-2); margin: 0; }
.private {
  border: 1px solid var(--critical); border-left-width: 6px; border-radius: 8px;
  padding: 10px 14px; margin: 16px 0;
  background: color-mix(in srgb, var(--critical) 8%, var(--surface));
}
nav { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 12px 0 0; font-size: 14px; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }
.fact {
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 10px 12px;
}
.fact .k { color: var(--muted); font-size: 13px; }
.fact .v { font-size: 18px; font-weight: 600; overflow-wrap: anywhere; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.panel {
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 12px 16px;
}
.panel h3 { font-size: 16px; margin-bottom: 4px; }
ul { margin: 6px 0; padding-left: 20px; }
figure { margin: 12px 0; }
figure img {
  width: 100%; height: auto; display: block; border-radius: 10px;
  border: 1px solid var(--line);
}
figcaption { color: var(--muted); font-size: 13px; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td {
  text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line);
  vertical-align: top;
}
th { color: var(--muted); font-weight: 500; font-size: 13px; }
td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.tablewrap { overflow-x: auto; }
.card {
  background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
  padding: 16px; margin: 18px 0;
}
.card header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 12px; }
.rank { color: var(--accent); font-weight: 700; }
.when { color: var(--ink-2); font-size: 14px; font-variant-numeric: tabular-nums; }
.chip {
  display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line);
  border-radius: 999px; padding: 1px 9px; font-size: 12.5px; color: var(--ink-2);
  white-space: nowrap; margin: 2px 4px 2px 0;
}
.chip::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--grey); }
.chip.unverified::before { background: var(--warn); }
.chip.image_plane::before { background: var(--plane); }
.chip.reference_3d::before { background: var(--blue); }
.chip.experimental::before { background: var(--serious); }
.chip.unavailable::before { background: var(--grey); }
.chip.official::before, .chip.measured::before { background: var(--good); }
.chip.private::before { background: var(--critical); }
.chip.tier { font-weight: 600; color: var(--ink); }
.chip.tier::before { display: none; }
.cue { font-variant-numeric: tabular-nums; }
.reason { color: var(--muted); font-size: 13px; }
video { width: 100%; max-height: 70vh; background: #000; border-radius: 10px; margin-top: 8px; }
.clips a { margin-right: 12px; font-size: 14px; }
.empty { color: var(--muted); }
code {
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 12.5px;
  overflow-wrap: anywhere;
}
footer { color: var(--muted); font-size: 13px; margin-top: 48px; }
"""

_DEFAULT_METHOD = (
    "用 ffmpeg 按恒定帧率逐帧解码原始转播片段（规范帧号 = 解码序号），检测硬切镜并给镜头分类。"
    "plugin-sngs（GPL 插件，独立进程运行）给出逐帧的球员框、跟踪编号、分队和单帧相机标定；"
    "只有通过标定检验的帧才换算成球场米制坐标（规则见“诚实说明”）。"
    "另外检测球，并计算原声的响度和频谱通量。逐帧高光分数由三类线索组成："
    "转播语法（导播从全景切到更近镜头的前后）、现场声音响度变化、比赛线索（冲刺速度、"
    "球在禁区内或紧挨禁区时禁区内的进攻人数、看得到球时球到球门的距离、球附近双方球员的最近"
    "距离、球的像面速度突变即触球事件）；"
    "按最小间隔挑出峰值，再把窗口对齐到镜头边界。每个高光窗口都在原始画面上分析："
    "相关球员、2D 姿态（RTMPose Halpe-26）、球和触球帧；有标定的帧给出球场位置、速度和"
    "到最近对手的距离，无标定的中景和特写只给画面平面里的角度。"
    "身体朝向和躯干前倾来自 3D 提升（标定 + 身高先验的骨架），只作参考，标“3D 参考”；"
    "朝向用箭头画回原始画面。所有结论以原始画面为准。"
)

_SOURCES_DEFAULT = (
    "原始画面：2D 姿态、球、球员框和跟踪，以及由它们在原始画面上算出的数：有标定的帧上的"
    "球场位置、速度和距离，触球帧和关键时刻，无标定镜头里画面平面的角度和距离",
    "3D 提升（只作参考）：身体朝向和躯干前倾。骨架由 2D 关键点、本帧标定和身高先验"
    "推出，是近似，不是测出来的三维姿态；这两个数标“3D 参考”，朝向还用箭头画回原始画面"
    f"（焦点球员脚下标着“{FACING_LABEL_ZH}”的箭头）",
)
"""The overview's 数据来源 panel: which numbers come from the original frames and which
from the 3D lift."""


def _measured_default(fps: str) -> tuple[str, ...]:
    """The overview's 测了什么 items. The peak speed is named for this clip's rate as the
    data card names it (gbf.measure.metric_label with the rate): the chord actually measured
    spans whole frames, 0.42 s for 0.4 s at 24 fps."""
    rate = parse_rate(fps)
    half = chord_half_frames(rate, MEASURE_PARAMS.peak_speed_window_s)
    seconds = f"{float(Fraction(2 * half) / rate):.2f}".rstrip("0").rstrip(".")
    label = metric_label("peak-speed", MEASURE_PARAMS, rate)
    return (
        f"有标定的帧：球员在球场上的位置（米）、高光窗口内最高的 {seconds} 秒平均速度"
        f"（数据卡上的“{label}”）、到最近对手的距离（A 档；误差未测量，一律“未验证”）",
        "身体朝向、躯干前倾（A 候选，来自 3D 提升，只作参考，标“3D 参考”；取球员本人检测到触球的"
        "那一帧前后，没有就取该球员的参考帧，通常是关键帧；在误差测完并赢过 2D 基线之前一律"
        "“未验证”）",
        "无标定的中景和特写：画面平面里的身体角度（只在图像平面，不换算成米或三维）",
    )


_NOT_MEASURED_DEFAULT = (
    "任何数值的真实误差：都还没有对照真值测量（标“未验证”）",
    "无标定镜头里的位置、速度和距离",
    "C 档指标：关节角速度、球和脚的真实速度（km/h）、触球点与触球时长、膝外翻、离地时间、"
    "起跳高度、全场跑动距离、加速度等（25 fps 测不了，按规定屏蔽）",
    "球的高度：球的位置是“贴地”近似，空中球的球场坐标不可信",
)


def _chip(kind: str, text: str) -> str:
    return f'<span class="chip {_e(kind)}">{_e(text)}</span>'


def _tier_chip(tier: str | None) -> str:
    if tier is None:
        return '<span class="chip tier">未分档</span>'
    return f'<span class="chip tier">{_e(TIER_ZH.get(tier, tier))}</span>'


def _clock(frame: int, fps: str) -> str:
    return format_clock(int(frame), fps)


_CLOCK_ZH = "分:秒.百分之一秒"
"""How the page names its time format (the digits after the point are not frames)."""


def _frame_ms(fps: str) -> str:
    """Duration of one frame, e.g. "40 ms" at 25 fps."""
    return f"{format_number(float(1000 / parse_rate(fps)), 0)} ms"


def _contact_frame_text(fps: str) -> str:
    return f"以帧为单位（未验证；本片一帧 {_frame_ms(fps)}），不到亚帧"


def _fact(key: str, value: str) -> str:
    return f'<div class="fact"><div class="k">{_e(key)}</div><div class="v">{_e(value)}</div></div>'


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return format_number(float(value), 2)
    return str(value)


def _cue_detail(name: str, cue: Mapping[str, Any]) -> str:
    """The evidence a cue object carries, in words (gbf.highlights cue objects). The frame a
    cue was read on is named once: not again after the touch or cut frame when it is the
    same frame. A speed names its player's track after the frame, as the metrics table's
    位置 column does: another player can have a speed row on the same frame."""

    def number(key: str) -> float | None:
        value = cue.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float | np.number):
            return None
        return float(value) if math.isfinite(float(value)) else None

    parts: list[str] = []
    named: int | None = None
    if name == "audio" and number("excitement_z") is not None:
        parts.append(f"z = {format_number(number('excitement_z') or 0.0, 2)}")
    elif name == "speed" and number("speed_mps") is not None:
        parts.append(f"{_kmh(number('speed_mps') or 0.0)} km/h")
    elif name == "box_pressure" and number("attackers") is not None:
        parts.append(f"{int(number('attackers') or 0)} 人")
    elif name == "ball_goal" and number("distance_m") is not None:
        parts.append(f"{format_number(number('distance_m') or 0.0, 1)} m")
    elif name == "duel" and number("distance") is not None:
        unit = cue.get("unit")
        distance = format_number(number("distance") or 0.0, 2)
        parts.append(
            f"{distance} 个身高" if unit == "box_h" else f"{distance} {unit or ''}".strip()
        )
    elif name == "cut_emphasis" and number("cut_frame") is not None:
        named = int(number("cut_frame") or 0)
        parts.append(f"切镜帧 {named}")
    elif name == "strike" and number("event_frame") is not None:
        named = int(number("event_frame") or 0)
        parts.append(f"触球帧 {named}")
    frame = number("frame")
    if frame is not None and int(frame) != named:
        parts.append(f"帧 {int(frame)}")
    track = number("track_id")
    if name == "speed" and parts and track is not None and track >= 0:
        parts.append(f"轨迹 {int(track)}")
    return "，".join(parts)


def _cue_chips(cues: Mapping[str, Any]) -> tuple[str, str]:
    """(chips of the cues that contributed, names of those that did not).

    A cue is a number, or an object with ``contribution`` (its share of the score) and its
    evidence; objects are sorted by contribution, largest first, and contributions that
    show as zero at the three decimals printed (+0.000) are listed separately.
    """
    scored: list[tuple[float, str]] = []
    idle: list[str] = []
    for key, value in cues.items():
        label = CUE_ZH.get(str(key), str(key))
        if isinstance(value, Mapping):
            contribution = value.get("contribution")
            if isinstance(contribution, int | float) and not isinstance(contribution, bool):
                shown = format_number(float(contribution), 3)
                if not math.isfinite(float(contribution)) or float(shown) <= 0.0:
                    idle.append(label)
                    continue
                detail = _cue_detail(str(key), value)
                text = f"{label} +{shown}"
                text += f"（{detail}）" if detail else ""
                scored.append((float(contribution), text))
                continue
            scalar = value.get("value")
            scored.append(
                (0.0, f"{label} {_format_scalar(scalar)}" if scalar is not None else label)
            )
            continue
        scored.append((0.0, f"{label} {_format_scalar(value)}"))
    scored.sort(key=lambda item: -item[0])
    chips = "".join(f'<span class="chip cue">{_e(text)}</span>' for _, text in scored)
    return chips, "、".join(idle)


def _px(value: Any) -> str:
    return f"{format_number(float(value), 1)} px"


def _errors(name: str, median: Any, p90: Any) -> list[str]:
    """``name`` with its median and P90 in pixels as fact parts; a P90 without a median
    keeps the name, so no bare "P90" is left without saying of what."""
    if median is not None:
        return [f"{name}中位 {_px(median)}"] + ([f"P90 {_px(p90)}"] if p90 is not None else [])
    return [f"{name} P90 {_px(p90)}"] if p90 is not None else []


def _refitted(refit: Any) -> bool:
    """``refit`` describes a refit that changed the calibration: a mapping that does not
    say ``refined_frames`` 0 (with no refitted frame, the plugin's calibration stayed)."""
    return isinstance(refit, Mapping) and refit.get("refined_frames") != 0


def _held_out(refit: Mapping[str, Any]) -> bool:
    """The refit was checked on held-out keypoints (it gives their error or rejected a
    camera group for it)."""
    numbers = ("holdout_median_px", "holdout_p90_px")
    return any(refit.get(key) is not None for key in numbers) or bool(refit.get("rejected_frames"))


def _calibration_facts(calibration: Mapping[str, Any]) -> list[str]:
    """Overview facts of the calibration: the one the numbers in metres use (after the
    refit, when there was one) and, labelled as such, the plugin's per-frame statistics."""
    frames = calibration.get("frames")
    usable = calibration.get("pitch_usable")
    fresh = calibration.get("fresh")
    threshold = calibration.get("threshold_px")
    refit = calibration.get("refit")
    plugin = []
    if fresh is not None and frames:
        plugin.append(f"新算 {fresh}/{frames} 帧")
    plugin += _errors(
        "自洽误差", calibration.get("rep_err_median_px"), calibration.get("rep_err_p90_px")
    )
    if not _refitted(refit):
        used = [f"球场可用 {usable}/{frames} 帧"] if usable is not None and frames else []
        if threshold is not None:
            plugin.append(f"阈值 {_px(threshold)}")
        text = " · ".join(used + plugin)
        title = "标定（插件逐帧）" if refit is None else "标定（插件逐帧，没有一帧重拟合）"
        return [_fact(title, text or "见附录")]
    used = []
    before = refit.get("pitch_usable_before")
    if usable is not None and frames:
        span = f"{before} → {usable}" if before is not None else f"{usable}"
        used.append(f"球场可用 {span}/{frames} 帧")
    if refit.get("refined_frames") is not None:
        used.append(f"重拟合 {refit['refined_frames']} 帧")
    used += _errors("留出关键点误差", refit.get("holdout_median_px"), refit.get("holdout_p90_px"))
    rejected = refit.get("rejected_frames")
    if rejected is not None:
        limit = refit.get("holdout_p90_max_px")
        why = (
            f"（所在机位留出误差 P90 超过 {_px(limit)}）" if limit is not None and rejected else ""
        )
        used.append(f"拒用 {rejected} 帧{why}")
    if threshold is not None:
        used.append(f"单帧阈值 {_px(threshold)}")
    facts = [_fact("标定（按机位重拟合后）", " · ".join(used) or "见附录")]
    if plugin:
        facts.append(_fact("插件逐帧标定（重拟合前）", " · ".join(plugin)))
    return facts


def _overview(meta: ReportMeta, rows: Sequence[MetricRow], n_highlights: int) -> str:
    rate = parse_rate(meta.fps)
    duration = float(Fraction(meta.frame_count) / rate)
    facts = [
        _fact("片段", meta.clip_name),
        _fact("时长", f"{duration:.2f} 秒（{_clock(meta.frame_count - 1, meta.fps)} 为最后一帧）"),
        _fact("帧数 / 帧率", f"{meta.frame_count} 帧 · {format_rate(rate)} fps"),
        _fact("时间写法", f"{_CLOCK_ZH}：01:02.40 即 1 分 2.40 秒，小数点后不是帧号"),
        _fact("分辨率", f"{meta.width} × {meta.height}"),
        _fact("镜头", f"{len(meta.cuts) + 1} 个镜头 · {len(meta.cuts)} 次硬切"),
        _fact("高光", f"{n_highlights} 个"),
    ]
    if meta.calibration:
        facts.extend(_calibration_facts(meta.calibration))
    if meta.audio_source:
        facts.append(_fact("原声来源", meta.audio_source))
    facts.append(_fact("源文件 sha256", meta.source_sha256[:16] + "…"))
    present: dict[str, set[str]] = {}
    for row in rows:
        if not row.blocked:
            present.setdefault(row.label, set()).add(
                TIER_ZH.get(row.tier or "", row.tier or "未分档")
            )
    present_items = "".join(
        f"<li>{_e(label)}：{_e('、'.join(sorted(tiers)))}</li>"
        for label, tiers in sorted(present.items())
    )
    present_html = (
        f"<ul>{present_items}</ul>" if present_items else '<p class="empty">本次没有测量记录。</p>'
    )
    measured = "".join(
        f"<li>{_e(text)}</li>"
        for text in (
            *_measured_default(meta.fps),
            "触球帧：按球的像面速度突变加脚部关键点判断，" + _contact_frame_text(meta.fps),
        )
    )
    not_measured = "".join(f"<li>{_e(text)}</li>" for text in _NOT_MEASURED_DEFAULT)
    sources = "".join(f"<li>{_e(text)}</li>" for text in _SOURCES_DEFAULT)
    return f"""
<section id="overview">
<h2>概览</h2>
<div class="facts">{"".join(facts)}</div>
<h3 style="margin-top:20px">方法</h3>
<p>{_e(meta.method_zh or _DEFAULT_METHOD)}</p>
<div class="cols">
<div class="panel"><h3>数据来源</h3><ul>{sources}</ul></div>
<div class="panel"><h3>测了什么</h3><ul>{measured}</ul>
<p class="reason">本报告实际出现的指标：</p>{present_html}</div>
<div class="panel"><h3>没测什么</h3><ul>{not_measured}</ul></div>
</div>
</section>"""


def _timeline_section(meta: ReportMeta, highlights: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for rank, item in enumerate(highlights, start=1):
        start, peak, end = int(item["start_frame"]), int(item["peak_frame"]), int(item["end_frame"])
        title = item.get("title_zh") or HIGHLIGHT_TYPES_ZH.get(item["type"], item["type"])
        rows.append(
            "<tr>"
            f'<td><a href="#hl-{_e(item["id"])}">H{rank}</a></td>'
            f"<td>{_e(title)}</td>"
            f"<td>{_e(HIGHLIGHT_TYPES_ZH.get(item['type'], item['type']))}</td>"
            f'<td class="num">{_e(_clock(start, meta.fps))}–{_e(_clock(end, meta.fps))}</td>'
            f'<td class="num">{start}–{end}</td>'
            f'<td class="num">{peak}（{_e(_clock(peak, meta.fps))}）</td>'
            f'<td class="num">{_e(format_number(float(item["score"]), 2))}</td>'
            "</tr>"
        )
    table = (
        '<div class="tablewrap"><table><thead><tr><th>编号</th><th>标题</th><th>类型</th>'
        f"<th>时间（{_CLOCK_ZH}）</th><th>帧范围</th><th>关键帧</th>"
        "<th>评分（平滑后的峰值）</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        if rows
        else '<p class="empty">没有找到高光。</p>'
    )
    if meta.score is not None:
        dots = (
            "圆点画在关键帧的逐帧分数上；表中“评分”是平滑后的峰值分数（只用于挑选片段），"
            "两者不是同一个数。"
        )
    else:
        dots = "没有逐帧分数，圆点只标出关键帧的位置。"
    ticks = _time_ticks(meta.frame_count, parse_rate(meta.fps))
    axis = _CLOCK_ZH if any("." in label for _, label in ticks) else "分:秒"
    return f"""
<section id="timeline">
<h2>时间轴</h2>
<figure>
<picture>
<source media="(prefers-color-scheme: dark)" srcset="{_href(TIMELINE_DARK_NAME)}">
<img src="{_href(TIMELINE_NAME)}" alt="逐帧高光分数与高光窗口的时间轴，各高光的数据见下表">
</picture>
<figcaption>横轴是时间（{axis}），纵轴是逐帧高光分数；橙色带是高光窗口，竖线是硬切镜。
{_e(dots)}表中时间写作{_CLOCK_ZH}，小数点后两位不是帧号。</figcaption>
</figure>
{table}
</section>"""


def _metrics_table(rows: Sequence[MetricRow]) -> str:
    shown = [row for row in rows if not row.blocked]
    if not shown:
        return '<p class="empty">这个高光没有测量记录。</p>'
    body = []
    for row in shown:
        badges = "".join(_chip(kind, text) for kind, text in row.badges)
        reason = f'<div class="reason">{_e(row.reason)}</div>' if row.reason else ""
        sign = METRIC_SIGNS_ZH.get(row.metric) if row.value_text != "—" else None
        sign_html = f'<div class="reason">{_e(sign[1])}</div>' if sign else ""
        evidence = (
            f'<div class="reason"><code>{_e(row.evidence_id)}</code></div>'
            if row.evidence_id
            else ""
        )
        where = []
        if row.frame is not None:
            where.append(f"帧 {row.frame}")
        if row.track_id is not None:
            where.append(f"轨迹 {row.track_id}")
        body.append(
            "<tr>"
            f"<td>{_e(row.label)}{sign_html}{evidence}</td>"
            f'<td class="num">{_e(row.value_text)}</td>'
            f"<td>{_e(row.unit)}</td>"
            f"<td>{_tier_chip(row.tier)}</td>"
            f"<td>{badges}{reason}</td>"
            f'<td class="num">{_e(" · ".join(where))}</td>'
            "</tr>"
        )
    return (
        '<div class="tablewrap"><table><thead><tr><th>指标</th><th>数值</th><th>单位</th>'
        "<th>档位</th><th>状态</th><th>位置</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _manifest_of(out_dir: Path, clip: str) -> str | None:
    """The ``<clip stem>.render.json`` written beside a clip by gbf.render, if present."""
    candidate = str(PurePosixPath(clip).with_suffix("")) + ".render.json"
    return candidate if (out_dir / candidate).is_file() else None


def _card(
    rank: int,
    item: Mapping[str, Any],
    meta: ReportMeta,
    rows: Sequence[MetricRow],
    images: Sequence[tuple[str, str | None]],
    clips: Sequence[tuple[str, str]],
    manifests: Sequence[tuple[str, str]] = (),
) -> str:
    """One highlight card; ``manifests`` are (label of the clip, render manifest) pairs."""
    start, peak, end = int(item["start_frame"]), int(item["peak_frame"]), int(item["end_frame"])
    type_zh = HIGHLIGHT_TYPES_ZH.get(item["type"], item["type"])
    title = item.get("title_zh") or f"{type_zh} H{rank}"
    summary = item.get("summary_zh")
    cues = item.get("cues") or {}
    cue_html, idle_cues = _cue_chips(cues if isinstance(cues, Mapping) else {})
    idle_html = f'<span class="reason">（未贡献：{_e(idle_cues)}）</span>' if idle_cues else ""
    cue_block = (
        "入选线索：" + cue_html + idle_html
        if cue_html
        else '<span class="empty">没有记录入选线索。</span>'
    )
    refined = REFINED_BY_ZH.get(str(item.get("refined_by")), "")
    refined_html = f"，{_e(refined)}" if refined else ""
    track_ids = item.get("track_ids") or []
    tracks = " · 相关轨迹 " + _e("、".join(str(t) for t in track_ids)) if track_ids else ""
    figures = []
    for index, (image, given) in enumerate(images):
        caption = given or (
            f"关键帧 {peak}（{_clock(peak, meta.fps)}）· 原始画面上的标注"
            if index == 0
            else f"附图 {index}"
        )
        figures.append(
            f'<figure><img src="{_href(image)}" alt="{_e(title)} 的{_e(caption)}" loading="lazy">'
            f"<figcaption>{_e(caption)}</figcaption></figure>"
        )
    clip_html = ""
    if clips:
        first_label, first_path = clips[0]
        links = "".join(f'<a href="{_href(path)}">{_e(label)}</a>' for label, path in clips)
        # Every manifest link is named after its clip (two clips may share a label).
        names = [f"渲染清单 · {label}" for label, _ in manifests]
        if len(set(names)) < len(names):
            names = [
                f"{name}（{PurePosixPath(path).name}）"
                for name, (_, path) in zip(names, manifests, strict=True)
            ]
        links += "".join(
            f'<a href="{_href(path)}" title="逐帧来源与证据">{_e(name)}</a>'
            for name, (_, path) in zip(names, manifests, strict=True)
        )
        clip_html = (
            f'<div class="clips"><video controls preload="metadata" src="{_href(first_path)}" '
            f'aria-label="{_e(first_label)}"></video><p>{links}</p></div>'
        )
    tier = item.get("tier")
    tier_html = (
        f'<span class="chip">{"主要高光" if tier == "main" else "候选"}</span>'
        if tier in ("main", "candidate")
        else ""
    )
    blocked = sum(1 for row in rows if row.blocked)
    blocked_html = (
        f'<p class="reason">另有 {blocked} 条 C 档记录按规定不显示数值。</p>' if blocked else ""
    )
    return f"""
<article class="card" id="hl-{_e(item["id"])}">
<header><span class="rank">H{rank}</span><h3>{_e(title)}</h3>
<span class="chip tier">{_e(type_zh)}</span>
<span class="chip">评分 {_e(format_number(float(item["score"]), 2))}</span>{tier_html}</header>
<p class="when">{_e(_clock(start, meta.fps))}–{_e(_clock(end, meta.fps))} · 帧 {start}–{end} ·
关键帧 {peak}（{_e(_clock(peak, meta.fps))}{refined_html}）{tracks}</p>
{f"<p>{_e(summary)}</p>" if summary else ""}
{_key_moments(item, meta.fps)}
{"".join(figures) if figures else '<p class="empty">没有关键帧图片。</p>'}
<p>{cue_block}</p>
{_metrics_table(rows)}
{blocked_html}
{clip_html if clip_html else '<p class="empty">没有片段。</p>'}
</article>"""


_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _display_codes(text: str, names: Mapping[str, str]) -> str:
    """``text`` with highlight ids (``names``: id -> "H1", ...) and highlight type codes
    replaced by the names the page uses; other words stay as they are."""

    def name(match: re.Match[str]) -> str:
        code = match.group(0)
        return names.get(code) or HIGHLIGHT_TYPES_ZH.get(code) or code

    return _CODE.sub(name, text)


def _evaluation(meta: ReportMeta, names: Mapping[str, str] | None = None) -> str:
    """The comparison with labels. Where the labels come from is the label file's to say
    (``meta.evaluation_source_zh``, quoted as given and kept apart from the renaming of
    highlight codes, which could change words of a quotation); the page adds only what
    holds for every label file."""
    if not meta.evaluation_zh:
        return ""
    items = "".join(
        f"<li>{_e(_display_codes(text, names or {}))}</li>" for text in meta.evaluation_zh
    )
    source = meta.evaluation_source_zh or "没有给出标注的来源。"
    return (
        '<section id="evaluation"><h2>与标注对比</h2>'
        f"<p>{_e(source)}检测器的阈值只在一段开发片段上调过；下面只是检出结果和这份标注的对照，"
        "不是验证。</p>"
        f"<ul>{items}</ul></section>"
    )


def _key_moments(item: Mapping[str, Any], fps: str) -> str:
    """The key moments of a highlight (gbf.moments) as a table, with what they are read from:
    the ball only when one of them, or a touch count, comes from it."""
    moments = item.get("key_moments") or []
    if not moments:
        return ""
    with_ball = bool(item.get("touches")) or any(
        m.get("kind") in BALL_MOMENT_KINDS for m in moments
    )
    source = "2D 姿态和球" if with_ball else "2D 姿态"
    body = "".join(
        f"<tr><td>{_e(_clock(int(m['frame']), fps))}</td><td>{int(m['frame'])}</td>"
        f"<td>{_e(MOMENT_KINDS_ZH.get(str(m.get('kind')), str(m.get('kind'))))}</td>"
        f"<td>{_e(m.get('text_zh') or '')}</td></tr>"
        for m in moments
    )
    touches = "".join(f"<p>{_e(_touch_sentence(t))}</p>" for t in item.get("touches") or [])
    return (
        f"<h4>关键时刻（逐帧，原始画面上的 {source}）</h4>"
        '<div class="tablewrap"><table><thead><tr><th>时间</th><th>帧</th><th>事件</th>'
        f"<th>说明</th></tr></thead><tbody>{body}</tbody></table></div>{touches}"
    )


def _touch_sentence(touch: Mapping[str, Any]) -> str:
    """One player's detected touches (gbf.moments ``touches``) as a sentence that says who
    was detected touching the ball, with the player's track (as the data card and the
    metrics table name it) and one parenthesis for how the touches are found."""
    who = str(touch.get("label_zh") or "球员")
    track = _int_or_none(touch.get("track_id"))
    if track is not None and f"轨迹 {track}" not in who:
        who += f"（轨迹 {track}）"
    interval = touch.get("mean_interval_s")
    gap = f"，平均间隔 {format_number(float(interval), 2)} 秒" if interval else ""
    return (
        f"检测到{who}触球 {int(touch['count'])} 次{gap}"
        f"（{TOUCHES_NOTE_ZH}；画面平面估计，未验证）。"
    )


def _calibration_rule(calibration: Mapping[str, Any]) -> str:
    """The rule that decides which frames give pitch coordinates, in Chinese."""
    threshold = calibration.get("threshold_px")
    limit = _px(threshold) if threshold is not None else "阈值"
    plugin = (
        "插件的单帧标定只做了自洽检验：rep_err_px 是本帧检测到的球场关键点和投影模型点之间的"
        "中位像素距离，不是对真值的误差。"
    )
    refit = calibration.get("refit")
    plugin_rule = f"只有新算（fresh）且 rep_err_px ≤ {limit} 的帧才用来算球场坐标、速度和距离。"
    if not isinstance(refit, Mapping):
        return plugin + plugin_rule
    if not _refitted(refit):
        stayed = "插件导出了球场关键点，但没有一帧按机位重拟合，全部沿用插件的单帧标定："
        return plugin + stayed + plugin_rule
    check = group = ""
    if _held_out(refit):
        tail = refit.get("holdout_p90_max_px")
        tail_text = f"超过 {_px(tail)}" if tail is not None else "过大"
        check = (
            "留出关键点误差的量法：每帧留出一部分关键点，另用其余关键点再拟合一次，"
            "量留出点的误差；实际使用的标定用全部关键点拟合，这仍是自洽检验。"
        )
        group = f"，且所在机位留出关键点误差的 P90 没有{tail_text}（否则整个机位不用）"
    return (
        f"{plugin}全景镜头按机位重拟合了标定（同一机位固定相机中心，每帧旋转和焦距平滑）。"
        f"{check}用来算球场坐标、速度和距离的帧：重拟合过的机位里，重拟合误差不超过 "
        f"{limit} 的帧{group}；"
        f"不在重拟合机位里的帧沿用插件的单帧标定，要求新算（fresh）且 rep_err_px ≤ {limit}。"
    )


def _notes(meta: ReportMeta) -> str:
    notes = [
        _calibration_rule(meta.calibration or {}),
        "所有数值的误差都还没有对照真值测量，所以一律标“未验证”。"
        "ADR 0004 规定公开版本要自动隐藏未验证的卡片，但这条规则还没有实现："
        "本报告照常显示所有未验证的数值。",
        "无标定的镜头（中景、特写、低机位）只给画面平面里的量，不换算成米、速度或三维量。",
        "球的球场坐标是“贴地”近似：球在空中时不可信。",
        f"触球帧{_contact_frame_text(meta.fps)}；C 档指标（角速度、球和脚的真实速度、触球时长等）"
        "不输出。",
        "姿态模型（RTMPose，经 rtmlib）的权重训练数据许可尚未核实：姿态结果只用于实验和私下查看。",
        "3D 只作参考，所有结论以原始画面为准。身体朝向和躯干前倾来自 3D 提升：骨架由 2D "
        "关键点、本帧标定和身高先验推出，是近似，不是测出来的三维姿态；这两个数标“3D 参考”，"
        "在误差测完并赢过 2D 基线之前只作参考。朝向画回原始画面：有标定的帧上，焦点球员这一帧"
        "的朝向通过数据卡用的单帧门槛（朝向置信度、不确定度和目标框高度）时，脚下画一支 "
        f"{FACING_ARROW_LENGTH_M:g} 米长的箭头，标“{FACING_LABEL_ZH}”；箭头只表示这一帧的估计，"
        "会随门槛时有时无；箭头的标签没有空位（箭头头部在数据卡等面板下面，或标签放在哪里都会"
        "压住别的标签）时，箭头也不画。其余内容（2D 姿态、球、球员位置、速度、距离、触球帧和"
        "关键时刻）都来自原始画面。",
    ]
    if meta.restrictions:
        notes.append(
            "私有素材：本报告和其中的图片、视频含转播画面及其派生数据（限制标签："
            + "、".join(meta.restrictions)
            + "），只能在本机私下查看，禁止分享、上传、发布或提交到仓库。"
        )
    else:
        notes.append("输入没有标注限制标签；发布前仍须按 ADR 0002 核对画面和数据的来源权利。")
    notes.extend(meta.notes_zh)
    items = "".join(f"<li>{_e(text)}</li>" for text in notes)
    return f'<section id="notes"><h2>诚实说明</h2><ul>{items}</ul></section>'


def _appendix(
    meta: ReportMeta,
    out_dir: Path,
    linked: Sequence[str],
    unattached: Sequence[MetricRow],
    blocked_total: int,
) -> str:
    rows = []
    rows.append(
        f'<tr><td>{_e(meta.clip_name)}（源视频）</td><td class="num">—</td>'
        f"<td><code>{_e(meta.source_sha256)}</code></td></tr>"
    )
    for artifact in meta.artifacts:
        size = str(artifact.bytes) if artifact.bytes is not None else "—"
        rows.append(
            f'<tr><td>{_e(artifact.name)}</td><td class="num">{_e(size)}</td>'
            f"<td><code>{_e(artifact.sha256)}</code></td></tr>"
        )
    for relative in linked:
        path = out_dir / relative
        rows.append(
            f'<tr><td>{_e(relative)}</td><td class="num">{path.stat().st_size}</td>'
            f"<td><code>{_e(sha256_file(path))}</code></td></tr>"
        )
    tools = "".join(f"<li>{_e(name)} {_e(version)}</li>" for name, version in meta.tools)
    font = meta.font
    font_html = (
        f"<li>图表字体：{_e(font.family)} {_e(font.style)}（{_e(font.file_name)}，sha256 "
        f"<code>{_e(font.sha256)}</code>）</li>"
        if font is not None
        else ""
    )
    unattached_html = ""
    if unattached:
        unattached_html = "<h3>没有归到任何高光的测量记录</h3>" + _metrics_table(unattached)
    blocked_html = (
        f"<p>按 ADR 0004 屏蔽的 C 档记录共 {blocked_total} 条（不显示数值）。</p>"
        if blocked_total
        else ""
    )
    return f"""
<section id="appendix">
<h2>附录</h2>
<h3>产物哈希</h3>
<div class="tablewrap"><table><thead><tr><th>文件</th><th>字节</th><th>sha256</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<h3 style="margin-top:16px">工具与字体</h3>
<ul>{tools}{font_html}<li>报告生成：gbf {_e(gbf.__version__)}</li></ul>
{blocked_html}
{unattached_html}
</section>"""


def _attach(
    rows: Sequence[MetricRow], highlights: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, list[MetricRow]], list[MetricRow]]:
    """Rows per highlight id (by explicit id, else by frame inside the window)."""
    by_id: dict[str, list[MetricRow]] = {str(item["id"]): [] for item in highlights}
    unattached: list[MetricRow] = []
    for row in rows:
        if row.highlight_id is not None and row.highlight_id in by_id:
            by_id[row.highlight_id].append(row)
            continue
        if row.highlight_id is None and row.frame is not None:
            hits = [
                str(item["id"])
                for item in highlights
                if int(item["start_frame"]) <= row.frame <= int(item["end_frame"])
            ]
            if hits:
                for ident in hits:
                    by_id[ident].append(row)
                continue
        unattached.append(row)
    return by_id, unattached


def _links(
    out_dir: Path,
    ids: set[str],
    images: Mapping[str, Any],
    clips: Mapping[str, Any],
    problems: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, str]]]]:
    image_links: dict[str, list[tuple[str, str | None]]] = {}
    clip_links: dict[str, list[tuple[str, str]]] = {}
    if not isinstance(images, Mapping):
        problems.append("images must map highlight ids to image paths")
        images = {}
    if not isinstance(clips, Mapping):
        problems.append("clips must map highlight ids to clip links")
        clips = {}
    for ident, value in images.items():
        if ident not in ids:
            problems.append(f"images: unknown highlight id {ident!r}")
            continue
        values = [value] if isinstance(value, str | Path | ImageLink) else list(value)
        for index, item in enumerate(values):
            where = f"images[{ident}][{index}]"
            caption = None
            if isinstance(item, ImageLink):
                item, caption = item.path, item.caption
                if caption is not None and not isinstance(caption, str):
                    problems.append(f"{where}: caption must be a string")
                    continue
            rel = _relative_file(out_dir, item, IMAGE_SUFFIXES, where, problems)
            if rel is not None:
                image_links.setdefault(ident, []).append((rel, caption))
    for ident, value in clips.items():
        if ident not in ids:
            problems.append(f"clips: unknown highlight id {ident!r}")
            continue
        values = [value] if isinstance(value, ClipLink | str | Path) else list(value)
        for index, item in enumerate(values):
            where = f"clips[{ident}][{index}]"
            if isinstance(item, ClipLink):
                label, path = item.label, item.path
            elif isinstance(item, str | Path):
                label, path = "片段", item
            elif isinstance(item, Mapping):
                label, path = item.get("label", "片段"), item.get("path")
            elif isinstance(item, Sequence) and len(item) == 2:
                label, path = item[0], item[1]
            else:
                problems.append(f"{where}: expected ClipLink, (label, path) or a path")
                continue
            if not isinstance(label, str) or not label:
                problems.append(f"{where}: label must be a non-empty string")
                continue
            rel = _relative_file(out_dir, path, CLIP_SUFFIXES, where, problems)
            if rel is not None:
                clip_links.setdefault(ident, []).append((label, rel))
    return image_links, clip_links


def write_report(
    out_dir: str | Path,
    highlights: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    images: Mapping[str, str | Path | ImageLink | Sequence[str | Path | ImageLink]],
    clips: Mapping[str, Any],
    meta: ReportMeta,
) -> Path:
    """Write ``out_dir/report.html`` (and the timeline PNGs) and return the report's path.

    Args:
        out_dir: report directory; every image and clip must be a file inside it.
        highlights: the ``highlights`` list of ``highlights.json`` or the whole document.
        measurements: evidence records (``measurements.jsonl`` lines), see
            :func:`metric_row`; attached to a highlight by ``highlightId`` or, failing that,
            by ``frame`` inside its window.
        images: highlight id -> key-frame image path (or a list, the key frame first),
            relative to ``out_dir`` (absolute paths inside ``out_dir`` are converted); an
            :class:`ImageLink` adds a caption.
        clips: highlight id -> list of :class:`ClipLink`, ``(label, path)`` pairs,
            ``{"label", "path"}`` objects or bare paths (MP4 inside ``out_dir``).
        meta: :class:`ReportMeta`.

    Raises:
        ReportInputError: listing every problem with the inputs (bad highlights, frames
            outside the clip, unknown ids, missing files, absolute or escaping links, a
            malformed measurement record, a score series of the wrong length, notes or
            evaluation sentences that are not strings, ...).
        gbf.fonts.FontNotFoundError: no CJK font for the timeline chart.
    """
    out = Path(out_dir)
    problems: list[str] = []
    meta_ok = _check_meta(meta, problems)
    checked = _check_highlights(highlights, meta, problems) if meta_ok else []
    rows: list[MetricRow] = []
    if not isinstance(measurements, Sequence) or isinstance(measurements, str | bytes):
        problems.append("measurements must be a list of records")
        measurements = []
    for index, record in enumerate(measurements):
        try:
            rows.append(metric_row(record))
        except ReportInputError as exc:
            problems.extend(f"measurements[{index}]: {p}" for p in exc.problems)
    if not out.is_dir():
        problems.append(f"out_dir is not a directory: {out.name}")
    ids = {str(item["id"]) for item in checked}
    image_links, clip_links = (
        _links(out, ids, images, clips, problems) if out.is_dir() else ({}, {})
    )
    if problems:
        raise ReportInputError(problems)

    ordered = sorted(checked, key=lambda item: (int(item["peak_frame"]), str(item["id"])))
    font = meta.font or find_cjk_font("regular")
    meta_with_font = meta if meta.font is not None else _with_font(meta, font)
    light = draw_timeline(meta_with_font, ordered, theme=LIGHT_THEME)
    dark = draw_timeline(meta_with_font, ordered, theme=DARK_THEME)
    for image, name in ((light, TIMELINE_NAME), (dark, TIMELINE_DARK_NAME)):
        chart = np.asarray(image.convert("RGB"))
        if meta.watermark:
            chart = stamp_watermark(chart, meta.watermark)
        save_png(out / name, chart, restrictions=meta.restrictions)

    by_id, unattached = _attach(rows, ordered)
    cards = "".join(
        _card(
            rank,
            item,
            meta_with_font,
            by_id[str(item["id"])],
            image_links.get(str(item["id"]), []),
            clip_links.get(str(item["id"]), []),
            [
                (label, manifest)
                for label, path in clip_links.get(str(item["id"]), [])
                if (manifest := _manifest_of(out, path)) is not None
            ],
        )
        for rank, item in enumerate(ordered, start=1)
    )
    linked: list[str] = [TIMELINE_NAME, TIMELINE_DARK_NAME]
    for item in ordered:
        linked.extend(path for path, _ in image_links.get(str(item["id"]), []))
        for _, path in clip_links.get(str(item["id"]), []):
            linked.append(path)
            manifest = _manifest_of(out, path)
            if manifest is not None:
                linked.append(manifest)
    linked = list(dict.fromkeys(linked))
    names = {str(item["id"]): f"H{rank}" for rank, item in enumerate(ordered, start=1)}
    blocked_total = sum(1 for row in rows if row.blocked)
    private = ""
    if meta.restrictions:
        private = (
            '<div class="private"><strong>私有素材 · 禁止分享</strong>：'
            "本报告含转播画面及其派生数据"
            f"（限制标签：{_e('、'.join(meta.restrictions))}），只能在本机私下查看。</div>"
        )
    subtitle = _e(
        f"{meta.clip_name} · {meta.frame_count} 帧 · {format_rate(parse_rate(meta.fps))} fps"
        f" · {len(ordered)} 个高光"
    )
    cards_html = cards if cards else '<p class="empty">没有找到高光。</p>'
    footer = _e(
        f"gbf {gbf.__version__} 生成。本页没有脚本和外部资源，所有链接都是本目录内的相对链接。"
    )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>{_e(meta.title)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header>
<h1>{_e(meta.title)}</h1>
<p class="sub">{subtitle}</p>
{private}
<nav>{_nav(meta_with_font)}</nav>
</header>
{_overview(meta_with_font, rows, len(ordered))}
{_timeline_section(meta_with_font, ordered)}
<section id="highlights"><h2>高光</h2>{cards_html}</section>
{_evaluation(meta_with_font, names)}
{_notes(meta_with_font)}
{_appendix(meta_with_font, out, linked, unattached, blocked_total)}
<footer>{footer}</footer>
</main>
</body>
</html>
"""
    path = out / REPORT_NAME
    write_text(path, page)
    return path


def _with_font(meta: ReportMeta, font: FontInfo) -> ReportMeta:
    return replace(meta, font=font)


__all__ = [
    "CLIP_SUFFIXES",
    "CUE_ZH",
    "DARK_THEME",
    "HIGHLIGHT_TYPES_ZH",
    "IMAGE_SUFFIXES",
    "REFINED_BY_ZH",
    "LIGHT_THEME",
    "METRIC_SIGNS_ZH",
    "MOMENT_KINDS_ZH",
    "REASON_ZH",
    "REFERENCE_3D_METRICS",
    "REFERENCE_3D_ZH",
    "REPORT_NAME",
    "TIER_ZH",
    "TIMELINE_DARK_NAME",
    "TIMELINE_NAME",
    "ArtifactRef",
    "ClipLink",
    "ImageLink",
    "MetricRow",
    "ReportInputError",
    "ReportMeta",
    "card_rows",
    "draw_timeline",
    "format_number",
    "metric_row",
    "write_report",
]
