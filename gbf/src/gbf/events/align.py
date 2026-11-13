"""``python -m gbf align``: one constant offset per half between StatsBomb and the video.

Clocks and units
    StatsBomb ``timestamp`` restarts at 0 in every period (milliseconds). SoccerNet
    Labels-v2 ``position`` is milliseconds from the first frame of each half video.
    Label half h is StatsBomb period h (h = 1, 2).

Method ``nearest-shot-median-v1`` (fixed; a change needs a new method name)
    1. Keep the Labels-v2 labels "Shots on target" and "Shots off target". "Goal"
       labels are excluded: they mark the ball entering the net, not the shot.
    2. For each kept label, take the StatsBomb Shot in the same period with the
       smallest ``|timestamp - position|`` (ties: earlier timestamp, then smaller
       event id). Matching is not one-to-one; shared matches are flagged.
    3. ``diff_s = statsbomb_timestamp_s - video_position_s`` for every pair.
    4. ``offset_s`` = median of the half's diffs (mean of the two middle values for
       an even count); ``residual_s = diff_s - offset_s``.
    5. ``status = needs_review`` if ``|residual_s| > tolerance_s`` (0.15 s), else ``ok``.

    A video position follows as ``statsbomb_timestamp_s - offset_s``. All arithmetic
    is exact (integer milliseconds and fractions), so the 0.15 s boundary is exact.

Frame convention
    Canonical frame = (sha256 of the half's source video, CFR frame index within the
    half). Frame n spans ``[n/fps, (n+1)/fps)`` of video time, video time 0 being the
    first frame of the half video, so ``frame_for_time = floor(video_time * fps)``.

The receipt (``align.json``) follows ``urn:gbf:schema:align``; it carries the
``no_share``, ``non_commercial`` and ``statsbomb_logo`` restrictions and is never
committed (ADR 0002).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from bisect import bisect_right
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from gbf import contracts
from gbf.events import providers
from gbf.events._meta import code_version, environment, relpath_under, utc_timestamp
from gbf.events.pins import Pin, PinSet, load_pins
from gbf.util import hashing
from gbf.util.io import write_json

PRODUCER = "gbf.events.align"
DATA_ROOT_ENV = "DATA_ROOT"
HALVES = (1, 2)
SHOT_LABELS = ("Shots on target", "Shots off target")
EXCLUDED_LABELS = ("Goal",)
KICKOFF_LABEL = "Kick-off"
DEFAULT_TOLERANCE_S = Fraction(3, 20)
DEFAULT_KICKOFF_TOLERANCE_S = Fraction(3)
DEFAULT_EXPECT_OFFSET_TOLERANCE_S = Fraction(1, 25)
MAIN_CAMERA_LABELS = ("Main camera center", "Main camera left", "Main camera right")

# Input names in the receipt, in order, with the restrictions assumed for a file
# that matches no pin (the pinned restrictions are used otherwise).
INPUT_ROLES = ("statsbomb_events", "statsbomb_lineups", "soccernet_labels_v2")
UNPINNED_RESTRICTIONS = {
    "statsbomb_events": ("no_share", "non_commercial", "statsbomb_logo"),
    "statsbomb_lineups": ("no_share", "non_commercial", "statsbomb_logo"),
    "soccernet_labels_v2": ("no_share",),
}

# These strings are constants of the schema (urn:gbf:schema:align); keep them equal.
METHOD_NAME = "nearest-shot-median-v1"
METHOD: Mapping[str, str] = {
    "name": METHOD_NAME,
    "statsbomb_event_type": "Shot",
    "matching": (
        "For each shot label, the StatsBomb Shot in the same period (label half h = "
        "StatsBomb period h) with the smallest |statsbomb_timestamp_s - video_position_s|; "
        "ties go to the earlier timestamp, then the smaller event id. Matching is not "
        "one-to-one."
    ),
    "offset": (
        "offset_s is the median of diff_s over the half's pairs (the mean of the two "
        "middle values for an even count)."
    ),
    "status_rule": (
        "residual_s = diff_s - offset_s; status is needs_review if |residual_s| > "
        "tolerance_s, else ok."
    ),
}
OFFSET_CONVENTION = (
    "diff_s = statsbomb_timestamp_s - video_position_s, where statsbomb_timestamp_s is the "
    "StatsBomb timestamp (restarts at 0 in every period) and video_position_s is the "
    "Labels-v2 position / 1000 (milliseconds from the first frame of the half video). A "
    "positive offset means the StatsBomb clock reads later than the video for the same "
    "moment: video_position_s = statsbomb_timestamp_s - offset_s."
)
FRAME_CONVENTION = (
    "Canonical frame = (sha256 of the half's source video, CFR frame index within the "
    "half). Frame n spans [n/fps, (n+1)/fps) of video time, 0 being the first frame of "
    "the half video: frame = floor((statsbomb_timestamp_s - offset_s) * fps)."
)
KICKOFF_RULE = (
    "Per half: earliest Labels-v2 'Kick-off' label and earliest StatsBomb kick-off pass; "
    "deviation_s = (statsbomb_timestamp_s - offset_s) - video_position_s; ok if "
    "|deviation_s| <= kickoff_tolerance_s. A coarse sanity bound (wrong half, wrong cut, "
    "gross clock error), not an accuracy measure: the half video may start just after the "
    "kick-off."
)
SCOPE = (
    "One constant offset per half between StatsBomb event timestamps and SoccerNet "
    "half-video positions, estimated from Labels-v2 shot labels matched to the nearest "
    "StatsBomb Shot of the same period; every pair with its residual and status; a "
    "kick-off sanity check per half; input hashes and pinned source revisions."
)
LIMITATIONS = (
    "Offsets measure agreement between two human annotations (StatsBomb and SoccerNet "
    "Labels-v2), not error against the true ball-contact frame.",
    "One constant offset per half is assumed; drift within a half shows only in the residuals.",
    "Labels-v2 has no pass or reception classes; residuals for passes and receptions are "
    "not measured here.",
    "Goal labels are excluded: they mark the ball entering the net, not the shot.",
    "The kick-off check is a coarse sanity bound, not an accuracy measurement.",
    "No video is read: frame indices follow from these offsets and the half video's frame "
    "rate; attack direction and camera coverage are not checked here.",
)


class FrameRangeError(ValueError):
    """A time maps outside the half video. ``problems`` lists every reason."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


# ----------------------------------------------------------------------- exact numbers


def to_fraction(value: Fraction | int | float | Decimal | str) -> Fraction:
    """Exact rational value of a time or tolerance.

    Floats are read through their shortest decimal representation (``repr``), so
    ``0.15`` means exactly 15/100. Strings may be decimals (``"1.91"``) or ratios
    (``"3/20"``). Booleans, NaN and infinities are rejected.
    """
    if isinstance(value, bool):
        raise TypeError("a boolean is not a number here")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"not a finite number: {value!r}")
        return Fraction(repr(value))
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"not a finite number: {value!r}")
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value.strip())
    raise TypeError(f"not a number: {value!r}")


def parse_fps(fps: Fraction | int | str) -> Fraction:
    """Frame rate as an exact positive fraction; accepts ``"25/1"``, ``"30000/1001"``, 25.

    Floats are refused: 29.97 is not 30000/1001, and frame indices must be exact.
    """
    if isinstance(fps, float):
        raise TypeError("pass the frame rate as a rational string such as '30000/1001'")
    rate = to_fraction(fps)
    if rate <= 0:
        raise ValueError(f"frame rate must be positive: {fps!r}")
    return rate


def _seconds(ms: Fraction | int) -> float:
    """Milliseconds to seconds for the receipt (exact input, correctly rounded float)."""
    return float(Fraction(ms) / 1000)


# --------------------------------------------------------------------------- alignment


@dataclass(frozen=True)
class Pair:
    """A shot label and its nearest StatsBomb shot. Times in integer/fractional ms."""

    label: providers.Label
    shot: providers.StatsBombEvent
    diff_ms: int
    residual_ms: Fraction
    status: str
    shared_match: bool


@dataclass(frozen=True)
class HalfOffset:
    """Offset of one half; ``offset_ms`` is ``None`` when the half has no pair."""

    half: int
    offset_ms: Fraction | None
    pairs: tuple[Pair, ...]
    unmatched_labels: tuple[providers.Label, ...]

    @property
    def aligned(self) -> bool:
        return self.offset_ms is not None

    @property
    def offset_s(self) -> Fraction | None:
        return None if self.offset_ms is None else self.offset_ms / 1000


@dataclass(frozen=True)
class Alignment:
    halves: tuple[HalfOffset, ...]
    tolerance_ms: Fraction
    excluded_label_counts: Mapping[str, int]
    unmatched_shots: tuple[providers.StatsBombEvent, ...]
    ignored_shots: tuple[providers.StatsBombEvent, ...]

    def half(self, half: int) -> HalfOffset:
        for item in self.halves:
            if item.half == half:
                return item
        raise KeyError(f"no half {half}")

    @property
    def pairs(self) -> tuple[Pair, ...]:
        return tuple(pair for half in self.halves for pair in half.pairs)


def median_ms(values: Sequence[int]) -> Fraction:
    """Exact median; the mean of the two middle values for an even count."""
    if not values:
        raise ValueError("median of an empty sequence")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return Fraction(ordered[middle])
    return Fraction(ordered[middle - 1] + ordered[middle], 2)


def nearest_shot(
    position_ms: int, shots: Sequence[providers.StatsBombEvent]
) -> providers.StatsBombEvent:
    """Shot with the smallest ``|timestamp - position|``; ties: earlier, then smaller id."""
    if not shots:
        raise ValueError("no shots to match")
    return min(
        shots,
        key=lambda shot: (abs(shot.timestamp_ms - position_ms), shot.timestamp_ms, shot.event_id),
    )


def align_shots(
    shots: Sequence[providers.StatsBombEvent],
    labels: Sequence[providers.Label],
    *,
    tolerance_s: Fraction | int | float | str = DEFAULT_TOLERANCE_S,
) -> Alignment:
    """Per-half offsets by method ``nearest-shot-median-v1`` (see the module docstring).

    ``shots`` are StatsBomb Shot events (period-relative ms); ``labels`` all Labels-v2
    annotations. Shots outside periods 1 and 2 are ignored and reported.
    """
    tolerance_ms = to_fraction(tolerance_s) * 1000
    if tolerance_ms <= 0:
        raise ValueError(f"tolerance must be positive: {tolerance_s!r}")
    excluded = Counter(label.label for label in labels if label.label in EXCLUDED_LABELS)
    shot_labels = sorted(
        (label for label in labels if label.label in SHOT_LABELS),
        key=lambda label: (label.half, label.position_ms, label.index),
    )
    halves = []
    matched_ids: set[str] = set()
    for half in HALVES:
        period_shots = [shot for shot in shots if shot.period == half]
        half_labels = [label for label in shot_labels if label.half == half]
        if not period_shots or not half_labels:
            halves.append(HalfOffset(half, None, (), tuple(half_labels)))
            continue
        matches = [(label, nearest_shot(label.position_ms, period_shots)) for label in half_labels]
        diffs = [shot.timestamp_ms - label.position_ms for label, shot in matches]
        offset = median_ms(diffs)
        uses = Counter(shot.event_id for _, shot in matches)
        pairs = []
        for (label, shot), diff in zip(matches, diffs, strict=True):
            residual = Fraction(diff) - offset
            pairs.append(
                Pair(
                    label=label,
                    shot=shot,
                    diff_ms=diff,
                    residual_ms=residual,
                    status="needs_review" if abs(residual) > tolerance_ms else "ok",
                    shared_match=uses[shot.event_id] > 1,
                )
            )
        matched_ids.update(uses)
        halves.append(HalfOffset(half, offset, tuple(pairs), ()))
    ordered_shots = sorted(shots, key=lambda shot: (shot.period, shot.timestamp_ms, shot.event_id))
    return Alignment(
        halves=tuple(halves),
        tolerance_ms=tolerance_ms,
        excluded_label_counts=dict(sorted(excluded.items())),
        unmatched_shots=tuple(
            shot
            for shot in ordered_shots
            if shot.period in HALVES and shot.event_id not in matched_ids
        ),
        ignored_shots=tuple(shot for shot in ordered_shots if shot.period not in HALVES),
    )


@dataclass(frozen=True)
class KickoffCheck:
    """Kick-off sanity check of one half (see ``KICKOFF_RULE``). Times in ms."""

    half: int
    status: str
    label: providers.Label | None
    kickoff: providers.StatsBombEvent | None
    predicted_video_ms: Fraction | None
    deviation_ms: Fraction | None
    tolerance_ms: Fraction
    missing: tuple[str, ...]


def check_kickoffs(
    kickoffs: Sequence[providers.StatsBombEvent],
    labels: Sequence[providers.Label],
    alignment: Alignment,
    *,
    tolerance_s: Fraction | int | float | str = DEFAULT_KICKOFF_TOLERANCE_S,
) -> tuple[KickoffCheck, ...]:
    """Compare each half's first kick-off in both sources through the half's offset.

    Status ``missing`` names what is absent (``label``, ``statsbomb_kickoff``,
    ``offset``); otherwise ``ok`` or ``needs_review`` against ``tolerance_s``.
    """
    tolerance_ms = to_fraction(tolerance_s) * 1000
    if tolerance_ms <= 0:
        raise ValueError(f"kick-off tolerance must be positive: {tolerance_s!r}")
    checks = []
    for half in HALVES:
        half_labels = [
            label for label in labels if label.half == half and label.label == KICKOFF_LABEL
        ]
        label = min(half_labels, key=lambda item: (item.position_ms, item.index), default=None)
        period_kickoffs = [event for event in kickoffs if event.period == half]
        kickoff = min(
            period_kickoffs,
            key=lambda event: (event.timestamp_ms, event.event_id),
            default=None,
        )
        offset = alignment.half(half).offset_ms
        missing = tuple(
            name
            for name, value in (
                ("label", label),
                ("statsbomb_kickoff", kickoff),
                ("offset", offset),
            )
            if value is None
        )
        predicted = None
        if kickoff is not None and offset is not None:
            predicted = Fraction(kickoff.timestamp_ms) - offset
        deviation = None
        if predicted is not None and label is not None:
            deviation = predicted - label.position_ms
        if missing:
            status = "missing"
        elif deviation is not None and abs(deviation) <= tolerance_ms:
            status = "ok"
        else:
            status = "needs_review"
        checks.append(
            KickoffCheck(half, status, label, kickoff, predicted, deviation, tolerance_ms, missing)
        )
    return tuple(checks)


# ----------------------------------------------------------------------- frame mapping


def video_time_s(
    provider_time_s: Fraction | int | float | Decimal | str,
    *,
    offset_s: Fraction | int | float | Decimal | str,
) -> Fraction:
    """Video time (s from the first frame of the half) of a provider time in that half.

    ``provider_time_s`` is the StatsBomb period timestamp in seconds; the result is
    ``provider_time_s - offset_s`` (see ``OFFSET_CONVENTION``).
    """
    return to_fraction(provider_time_s) - to_fraction(offset_s)


def frame_for_time(
    provider_time_s: Fraction | int | float | Decimal | str,
    *,
    offset_s: Fraction | int | float | Decimal | str,
    fps: Fraction | int | str,
    n_frames: int | None = None,
) -> int:
    """Canonical CFR frame index within the half that shows ``provider_time_s``.

    Frame n spans ``[n/fps, (n+1)/fps)`` of video time, so the index is
    ``floor((provider_time_s - offset_s) * fps)``: a time exactly on a boundary
    belongs to the later frame. ``fps`` is exact (``"25/1"``, ``"30000/1001"``).
    Raises :class:`FrameRangeError` when the time falls before the first frame or,
    with ``n_frames`` given, at or after the end of the half video.
    """
    rate = parse_fps(fps)
    video_time = video_time_s(provider_time_s, offset_s=offset_s)
    problems = []
    if video_time < 0:
        problems.append(f"video time {float(video_time)} s is before the first frame")
    frame = math.floor(video_time * rate)
    if n_frames is not None:
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 0:
            raise TypeError(f"n_frames must be a non-negative integer: {n_frames!r}")
        if frame >= n_frames:
            problems.append(f"frame {frame} is past the last frame ({n_frames - 1})")
    if problems:
        raise FrameRangeError(problems)
    return frame


def frame_start_s(frame: int, fps: Fraction | int | str) -> Fraction:
    """Video time at which frame ``frame`` starts: ``frame / fps`` (exact)."""
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError(f"frame must be a non-negative integer: {frame!r}")
    return Fraction(frame) / parse_fps(fps)


def statsbomb_video_times(
    events: Iterable[providers.StatsBombEvent],
    offsets_s: Mapping[int, Fraction | int | float | Decimal | str],
) -> list[tuple[int, Fraction]]:
    """``(half, video_time_s)`` of StatsBomb events, through the per-half offsets.

    ``video_time_s = timestamp_s - offset_s`` (see ``OFFSET_CONVENTION``), exact. These
    are the event times ``main_camera_share`` expects. Raises ``ValueError`` listing
    every event whose period has no offset.
    """
    exact = {half: to_fraction(offset) for half, offset in offsets_s.items()}
    times = []
    problems = []
    for event in events:
        if event.period not in exact:
            problems.append(f"event {event.event_id}: no offset for period {event.period}")
            continue
        times.append((event.period, Fraction(event.timestamp_ms, 1000) - exact[event.period]))
    if problems:
        raise ValueError("; ".join(problems))
    return times


def offsets_from_receipt(receipt: Mapping[str, Any]) -> dict[int, Fraction]:
    """Exact per-half offsets (s) of the aligned halves in an ``align.json`` receipt."""
    contracts.validate(receipt, "align")
    return {
        int(item["half"]): to_fraction(item["offset_s"])
        for item in receipt["result"]["halves"]
        if item["aligned"]
    }


# ------------------------------------------------------------------- main-camera share


@dataclass(frozen=True)
class CameraAtTime:
    """Camera on air at one event time; ``camera`` is ``None`` before the first change."""

    half: int
    time_s: Fraction
    camera: str | None
    replay: bool | None
    is_main: bool


@dataclass(frozen=True)
class CameraShare:
    """How many event times fall in a main-camera, live segment (integer counts)."""

    n_events: int
    n_main: int
    n_unknown: int
    main_labels: tuple[str, ...]
    include_replays: bool
    events: tuple[CameraAtTime, ...]

    @property
    def share(self) -> Fraction | None:
        return None if self.n_events == 0 else Fraction(self.n_main, self.n_events)


def main_camera_share(
    labels_cameras: Mapping[str, Any] | bytes | Sequence[providers.CameraChange],
    event_times: Iterable[tuple[int, Fraction | int | float | Decimal | str]],
    *,
    main_labels: Collection[str] = MAIN_CAMERA_LABELS,
    include_replays: bool = False,
) -> CameraShare:
    """Share of event times at which a main camera is on air (pure function).

    ``labels_cameras`` is a parsed ``Labels-cameras.json`` (or its bytes, or parsed
    changes). Each annotation starts a segment ``[position, next position)`` of its
    half; the last one runs to the end of the half. ``event_times`` are
    ``(half, video_time_s)`` pairs, video time counted from the first frame of that
    half video (Labels-v2 position / 1000, or StatsBomb time minus the half offset).

    An event counts as main when its segment's camera is in ``main_labels`` and the
    segment is live (``replay`` is ``False``); replay segments, and segments whose
    replay flag is missing, count only with ``include_replays``. Times before the
    half's first annotation have no camera (``n_unknown``) and do not count.

    The class set and the replay rule are part of any pre-registered decision that
    uses this share; freeze them with its thresholds.
    """
    if isinstance(labels_cameras, (Mapping, bytes, bytearray)):
        changes = providers.parse_labels_cameras(labels_cameras)
    else:
        changes = tuple(labels_cameras)
    main_set = frozenset(main_labels)
    by_half: dict[int, list[providers.CameraChange]] = {half: [] for half in HALVES}
    for change in sorted(changes, key=lambda c: (c.half, c.position_ms, c.index)):
        by_half.setdefault(change.half, []).append(change)
    starts = {half: [c.position_ms for c in items] for half, items in by_half.items()}
    problems = []
    results = []
    for number, (half, time) in enumerate(event_times):
        if isinstance(half, bool) or half not in by_half:
            problems.append(f"event {number}: half must be one of {list(HALVES)}, got {half!r}")
            continue
        time_s = to_fraction(time)
        if time_s < 0:
            problems.append(f"event {number}: negative video time {float(time_s)} s")
            continue
        index = bisect_right(starts[half], time_s * 1000) - 1
        if index < 0:
            results.append(CameraAtTime(half, time_s, None, None, False))
            continue
        change = by_half[half][index]
        live = change.replay is False
        is_main = change.label in main_set and (include_replays or live)
        results.append(CameraAtTime(half, time_s, change.label, change.replay, is_main))
    if problems:
        raise ValueError("; ".join(problems))
    return CameraShare(
        n_events=len(results),
        n_main=sum(item.is_main for item in results),
        n_unknown=sum(item.camera is None for item in results),
        main_labels=tuple(sorted(main_set)),
        include_replays=include_replays,
        events=tuple(results),
    )


# ---------------------------------------------------------------------------- receipt


@dataclass(frozen=True)
class Expectations:
    """Optional reproduction targets, e.g. the P0 acceptance (offsets within one frame)."""

    offsets_s: Mapping[int, Fraction]
    offset_tolerance_s: Fraction = DEFAULT_EXPECT_OFFSET_TOLERANCE_S
    min_ok: int | None = None


def evaluate_expectations(alignment: Alignment, expectations: Expectations) -> dict[str, Any]:
    """Compare the alignment with the expected offsets and ``ok`` count. Changes nothing."""
    tolerance = to_fraction(expectations.offset_tolerance_s)
    offsets = []
    for half, expected in sorted(expectations.offsets_s.items()):
        if half not in HALVES:
            raise ValueError(f"expected offset for unknown half {half!r}")
        actual = alignment.half(half).offset_s
        expected_f = to_fraction(expected)
        offsets.append(
            {
                "half": half,
                "expected_s": float(expected_f),
                "actual_s": None if actual is None else float(actual),
                "tolerance_s": float(tolerance),
                "ok": actual is not None and abs(actual - expected_f) <= tolerance,
            }
        )
    min_ok = None
    if expectations.min_ok is not None:
        actual_ok = sum(1 for pair in alignment.pairs if pair.status == "ok")
        min_ok = {
            "expected": expectations.min_ok,
            "actual": actual_ok,
            "ok": actual_ok >= expectations.min_ok,
        }
    ok = all(item["ok"] for item in offsets) and (min_ok is None or min_ok["ok"])
    return {"ok": ok, "offsets": offsets, "min_ok": min_ok}


def _label_ref(label: providers.Label) -> dict[str, Any]:
    return {
        "half": label.half,
        "label": label.label,
        "label_index": label.index,
        "label_game_time": label.game_time,
        "label_position_ms": label.position_ms,
    }


def _shot_ref(shot: providers.StatsBombEvent) -> dict[str, Any]:
    return {
        "statsbomb_event_id": shot.event_id,
        "statsbomb_period": shot.period,
        "statsbomb_timestamp": shot.timestamp,
        "statsbomb_timestamp_s": _seconds(shot.timestamp_ms),
        "statsbomb_team": shot.team,
    }


def _pair_json(pair: Pair) -> dict[str, Any]:
    label, shot = pair.label, pair.shot
    team_match = label.team == shot.team_side if label.team in ("home", "away") else None
    return {
        **_label_ref(label),
        "video_position_s": _seconds(label.position_ms),
        "label_team": label.team,
        "label_visibility": label.visibility,
        **_shot_ref(shot),
        "statsbomb_team_side": shot.team_side,
        "statsbomb_shot_outcome": shot.outcome,
        "diff_s": _seconds(pair.diff_ms),
        "residual_s": _seconds(pair.residual_ms),
        "status": pair.status,
        "team_match": team_match,
        "shared_match": pair.shared_match,
    }


def _half_json(half: HalfOffset) -> dict[str, Any]:
    residuals = [abs(pair.residual_ms) for pair in half.pairs]
    return {
        "half": half.half,
        "statsbomb_period": half.half,
        "aligned": half.aligned,
        "offset_s": None if half.offset_ms is None else _seconds(half.offset_ms),
        "n_pairs": len(half.pairs),
        "n_ok": sum(pair.status == "ok" for pair in half.pairs),
        "n_needs_review": sum(pair.status == "needs_review" for pair in half.pairs),
        "n_unmatched_labels": len(half.unmatched_labels),
        "max_abs_residual_s": _seconds(max(residuals)) if residuals else None,
    }


def _kickoff_json(check: KickoffCheck) -> dict[str, Any]:
    def maybe_seconds(value: Fraction | int | None) -> float | None:
        return None if value is None else _seconds(value)

    return {
        "half": check.half,
        "status": check.status,
        "missing": list(check.missing),
        "label_index": None if check.label is None else check.label.index,
        "label_position_ms": None if check.label is None else check.label.position_ms,
        "statsbomb_event_id": None if check.kickoff is None else check.kickoff.event_id,
        "statsbomb_timestamp": None if check.kickoff is None else check.kickoff.timestamp,
        "statsbomb_timestamp_s": None
        if check.kickoff is None
        else _seconds(check.kickoff.timestamp_ms),
        "predicted_video_position_s": maybe_seconds(check.predicted_video_ms),
        "deviation_s": maybe_seconds(check.deviation_ms),
        "tolerance_s": _seconds(check.tolerance_ms),
    }


@dataclass(frozen=True)
class InputFile:
    """One input as read: its role, bytes, and (for relpaths) where it was read from."""

    role: str
    data: bytes
    path: Path | None = None


def _match_pin(pin_set: PinSet, item: InputFile) -> tuple[Pin | None, list[str]]:
    blob = hashing.git_blob_sha1(item.data)
    candidates = pin_set.by_blob(blob)
    for pin in candidates:
        if pin.role == item.role:
            return pin, []
    if candidates:
        other = candidates[0]
        return None, [
            f"{item.role}: content is pin {other.name} (role {other.role}), not a {item.role} file"
        ]
    return None, [f"{item.role}: content matches no pin (git blob {blob}); provenance unverified"]


def build_receipt(
    *,
    inputs: Sequence[InputFile],
    match: providers.StatsBombMatch,
    labels: Sequence[providers.Label],
    alignment: Alignment,
    kickoffs: Sequence[KickoffCheck],
    pin_set: PinSet,
    data_root: str | Path | None,
    expectations: Expectations | None = None,
    code: Mapping[str, Any] | None = None,
    created_at: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the ``align.json`` receipt and validate it against ``urn:gbf:schema:align``.

    Raises :class:`gbf.contracts.ContractError` listing every schema violation.
    """
    roles = tuple(item.role for item in inputs)
    if roles != INPUT_ROLES:
        raise ValueError(f"inputs must be {list(INPUT_ROLES)} in this order, got {list(roles)}")
    check_halves = [check.half for check in kickoffs]
    if check_halves != list(HALVES):
        raise ValueError(f"kick-off checks must cover halves {list(HALVES)}, got {check_halves}")
    if len({check.tolerance_ms for check in kickoffs}) != 1:
        raise ValueError("kick-off checks must share one tolerance")
    warnings: list[str] = []
    entries = []
    sources = []
    restrictions: set[str] = set()
    unpinned = False
    for item in inputs:
        pin, pin_warnings = _match_pin(pin_set, item)
        warnings.extend(pin_warnings)
        unpinned = unpinned or pin is None
        entry: dict[str, Any] = {
            "name": item.role,
            "sha256": hashing.sha256_bytes(item.data),
            "git_blob_sha1": hashing.git_blob_sha1(item.data),
            "bytes": len(item.data),
            "source_revision": pin.revision if pin is not None else "unpinned",
        }
        relpath = relpath_under(item.path, data_root) if item.path is not None else None
        if relpath is not None:
            entry["relpath"] = relpath
        entries.append(entry)
        sources.append(
            {
                "name": item.role,
                "pin": None if pin is None else pin.name,
                "host": None if pin is None else pin.host,
                "repository": None if pin is None else pin.repository,
                "revision": None if pin is None else pin.revision,
                "path": None if pin is None else pin.path,
            }
        )
        restrictions.update(
            pin.restrictions if pin is not None else UNPINNED_RESTRICTIONS[item.role]
        )
    if data_root is None:
        warnings.append(f"{DATA_ROOT_ENV} is not set: input relpaths are omitted")
    elif any("relpath" not in entry for entry in entries):
        warnings.append(f"some inputs are not under {DATA_ROOT_ENV}: their relpaths are omitted")
    warnings.extend(providers.game_time_disagreements(labels))
    for pair in alignment.pairs:
        if pair.shared_match:
            warnings.append(
                f"label {pair.label.index} shares its nearest StatsBomb shot "
                f"{pair.shot.event_id} with another label"
            )
    if alignment.ignored_shots:
        warnings.append(
            f"{len(alignment.ignored_shots)} StatsBomb shot(s) outside periods 1-2 were ignored"
        )

    expectation_result = (
        evaluate_expectations(alignment, expectations) if expectations is not None else None
    )
    aligned = [half for half in alignment.halves if half.aligned]
    if not aligned or (expectation_result is not None and not expectation_result["ok"]):
        status = "failed"
    elif len(aligned) < len(HALVES) or unpinned or any(check.status != "ok" for check in kickoffs):
        status = "partial"
    else:
        status = "passed"

    pairs = alignment.pairs
    result: dict[str, Any] = {
        "method": {
            **METHOD,
            "shot_labels": list(SHOT_LABELS),
            "excluded_labels": list(EXCLUDED_LABELS),
        },
        "offset_convention": OFFSET_CONVENTION,
        "frame_convention": FRAME_CONVENTION,
        "kickoff_rule": KICKOFF_RULE,
        "tolerance_s": _seconds(alignment.tolerance_ms),
        "kickoff_tolerance_s": _seconds(kickoffs[0].tolerance_ms),
        "halves": [_half_json(half) for half in alignment.halves],
        "pairs": [_pair_json(pair) for pair in pairs],
        "unmatched_labels": [
            _label_ref(label) for half in alignment.halves for label in half.unmatched_labels
        ],
        "unmatched_statsbomb_shots": [_shot_ref(shot) for shot in alignment.unmatched_shots],
        "excluded_label_counts": dict(alignment.excluded_label_counts),
        "kickoff_checks": [_kickoff_json(check) for check in kickoffs],
        "sources": sources,
        "statsbomb_loader": match.loader,
        "summary": {
            "n_pairs": len(pairs),
            "n_ok": sum(pair.status == "ok" for pair in pairs),
            "n_needs_review": sum(pair.status == "needs_review" for pair in pairs),
            "n_unmatched_labels": sum(len(half.unmatched_labels) for half in alignment.halves),
            "n_unmatched_statsbomb_shots": len(alignment.unmatched_shots),
            "n_excluded_labels": sum(alignment.excluded_label_counts.values()),
        },
        "warnings": warnings,
    }
    if expectation_result is not None:
        result["expectations"] = expectation_result
    receipt = {
        "schema_version": 1,
        "kind": "align",
        "status": status,
        "created_at": created_at if created_at is not None else utc_timestamp(),
        "producer": PRODUCER,
        "code": dict(code) if code is not None else code_version(),
        "inputs": entries,
        "scope": SCOPE,
        "limitations": list(LIMITATIONS),
        "restrictions": sorted(restrictions),
        "reviewer_type": "automated",
        "environment": dict(env) if env is not None else environment(),
        "result": result,
    }
    contracts.validate(receipt, "align")
    return receipt


def run(
    *,
    events: str | Path,
    lineups: str | Path,
    labels: str | Path,
    out: str | Path,
    data_root: str | Path | None = None,
    tolerance_s: Fraction | int | float | str = DEFAULT_TOLERANCE_S,
    kickoff_tolerance_s: Fraction | int | float | str = DEFAULT_KICKOFF_TOLERANCE_S,
    expectations: Expectations | None = None,
    force: bool = False,
    pin_set: PinSet | None = None,
    code: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Read the three local files, align, validate and write ``out``; return the receipt.

    ``out`` is never overwritten unless ``force`` (artifacts are immutable).
    """
    out_path = Path(out)
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path.name} already exists; pass --force to replace it")
    inputs = [
        InputFile("statsbomb_events", Path(events).read_bytes(), Path(events)),
        InputFile("statsbomb_lineups", Path(lineups).read_bytes(), Path(lineups)),
        InputFile("soccernet_labels_v2", Path(labels).read_bytes(), Path(labels)),
    ]
    match = providers.load_statsbomb(inputs[0].data, inputs[1].data)
    parsed_labels = providers.parse_labels_v2(inputs[2].data)
    alignment = align_shots(match.shots, parsed_labels, tolerance_s=tolerance_s)
    kickoffs = check_kickoffs(
        match.kickoffs, parsed_labels, alignment, tolerance_s=kickoff_tolerance_s
    )
    receipt = build_receipt(
        inputs=inputs,
        match=match,
        labels=parsed_labels,
        alignment=alignment,
        kickoffs=kickoffs,
        pin_set=pin_set if pin_set is not None else load_pins(),
        data_root=data_root,
        expectations=expectations,
        code=code,
        created_at=created_at,
    )
    write_json(out_path, receipt)
    return receipt


# -------------------------------------------------------------------------------- CLI


def _expected_offset(text: str) -> tuple[int, Fraction]:
    half, sep, seconds = text.partition("=")
    if not sep or half.strip() not in {"1", "2"}:
        raise argparse.ArgumentTypeError(f"use HALF=SECONDS with HALF 1 or 2, got {text!r}")
    try:
        return int(half), to_fraction(seconds)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(f"bad seconds in {text!r}: {error}") from error


def _positive_fraction(text: str) -> Fraction:
    try:
        value = to_fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {text!r}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gbf align",
        description="Align StatsBomb events to SoccerNet half videos (writes align.json).",
    )
    parser.add_argument("--events", required=True, help="StatsBomb events JSON (local file)")
    parser.add_argument("--lineups", required=True, help="StatsBomb lineups JSON (local file)")
    parser.add_argument("--labels", required=True, help="SoccerNet Labels-v2.json (local file)")
    parser.add_argument("--out", required=True, help="align.json to write")
    parser.add_argument(
        "--data-root", help=f"root for input relpaths (default: ${DATA_ROOT_ENV}, if set)"
    )
    parser.add_argument(
        "--tolerance-s",
        type=_positive_fraction,
        default=DEFAULT_TOLERANCE_S,
        help="needs_review above this |residual| (default 0.15)",
    )
    parser.add_argument(
        "--kickoff-tolerance-s",
        type=_positive_fraction,
        default=DEFAULT_KICKOFF_TOLERANCE_S,
        help="kick-off sanity bound (default 3)",
    )
    parser.add_argument(
        "--expect-offset",
        type=_expected_offset,
        action="append",
        default=[],
        metavar="HALF=SECONDS",
        help="expected offset of a half, e.g. 1=1.91 (repeatable)",
    )
    parser.add_argument(
        "--expect-offset-tolerance-s",
        type=_positive_fraction,
        default=DEFAULT_EXPECT_OFFSET_TOLERANCE_S,
        help="allowed |offset - expected| (default 0.04 = one frame at 25 fps)",
    )
    parser.add_argument(
        "--expect-min-ok", type=int, default=None, help="minimum number of ok pairs"
    )
    parser.add_argument("--force", action="store_true", help="replace an existing --out")
    return parser


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    expectations = None
    if args.expect_offset or args.expect_min_ok is not None:
        offsets = dict(args.expect_offset)
        if len(offsets) != len(args.expect_offset):
            print("error: --expect-offset given twice for the same half", file=sys.stderr)
            return 2
        expectations = Expectations(
            offsets_s=offsets,
            offset_tolerance_s=args.expect_offset_tolerance_s,
            min_ok=args.expect_min_ok,
        )
    data_root = args.data_root or os.environ.get(DATA_ROOT_ENV) or None
    try:
        receipt = run(
            events=args.events,
            lineups=args.lineups,
            labels=args.labels,
            out=args.out,
            data_root=data_root,
            tolerance_s=args.tolerance_s,
            kickoff_tolerance_s=args.kickoff_tolerance_s,
            expectations=expectations,
            force=args.force,
        )
    except (
        OSError,
        ValueError,
        providers.ProviderDataError,
        contracts.ContractError,
    ) as error:
        print(f"align failed: {error}", file=sys.stderr)
        return 2
    result = receipt["result"]
    for half in result["halves"]:
        offset = "n/a" if half["offset_s"] is None else f"{half['offset_s']:+.4f} s"
        print(
            f"half {half['half']}: offset {offset}, {half['n_ok']}/{half['n_pairs']} pairs "
            f"within {result['tolerance_s']} s"
        )
    for check in result["kickoff_checks"]:
        print(f"kick-off half {check['half']}: {check['status']}")
    if "expectations" in result:
        print(f"expectations: {'met' if result['expectations']['ok'] else 'NOT met'}")
    print(
        f"status: {receipt['status']}; restrictions: {', '.join(receipt['restrictions'])} "
        "(never commit or share this file)"
    )
    return 0 if receipt["status"] == "passed" else 1
