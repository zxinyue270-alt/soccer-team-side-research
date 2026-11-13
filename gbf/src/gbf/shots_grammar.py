"""Broadcast shot grammar: shots between hard cuts and a coarse type for each shot.

A shot is a run of canonical frames between two hard cuts. ``cut_frames`` holds the first
frame of every shot except the first (what :func:`gbf.ingest.detect_cuts` reports), so with
cuts c_1 < ... < c_k and ``n_frames`` frames the shots are [0, c_1 - 1], [c_1, c_2 - 1],
..., [c_k, n_frames - 1], inclusive at both ends.

Each shot gets two cues and a type (``shots.parquet``, docs/design/m1-highlights.md):

- ``median_box_h``: how large people appear, in pixels of a 1080-line frame. For every
  frame of the shot with at least one person box (roles :data:`PERSON_ROLES`), take the
  median box height (y2 - y1) scaled by 1080 / frame height; ``median_box_h`` is the median
  of those per-frame medians, so every frame counts once whether it shows 2 people or 20.
  NaN when no frame of the shot has a person box.
- ``fresh_share``: the share of the shot's frames whose calibration is pitch-usable
  (calib.parquet ``pitch_usable``: fresh and rep_err <= the import's maximum). Frames
  without a calib row count as not usable. The column keeps its contract name.
- ``shot_type`` (:func:`classify_shot`), empirical starting values, not validated:
  "close" when median_box_h >= 220 px, or when fresh_share < 0.2 and median_box_h >= 150 px,
  or when no person was detected at all and fresh_share < 0.2 (a crowd, bench or replay
  graphic); "wide" when median_box_h < 130 px and fresh_share >= 0.5; "medium" otherwise.

The grammar cue built on this (a cut from a wide shot to a close shot marks an action the
director chose to emphasise) lives in gbf.highlights; this module only describes shots.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gbf.ingest import DEFAULT_CUT_THRESHOLD, detect_cuts, iter_frames

SHOT_TYPES: tuple[str, ...] = ("wide", "medium", "close")
PERSON_ROLES: tuple[str, ...] = ("player", "goalkeeper", "referee")
"""Roles whose box height measures the zoom: people on the pitch, whose height is similar."""

REFERENCE_HEIGHT_PX = 1080
"""Box heights are normalised to a frame of this many lines."""

CLOSE_MIN_BOX_H_PX = 220.0
"""A shot whose median person box is at least this tall (1080-line pixels) is close."""
CLOSE_UNCALIBRATED_MIN_BOX_H_PX = 150.0
"""... or at least this tall when fewer than CLOSE_MAX_FRESH_SHARE of its frames are usable."""
CLOSE_MAX_FRESH_SHARE = 0.2
WIDE_MAX_BOX_H_PX = 130.0
"""A shot is wide when its median person box is below this height ..."""
WIDE_MIN_FRESH_SHARE = 0.5
"""... and at least this share of its frames is pitch-usable."""

SHOTS_COLUMNS: tuple[str, ...] = (
    "shot_id",
    "start_frame",
    "end_frame",
    "shot_type",
    "fresh_share",
    "median_box_h",
)


class ShotsInputError(ValueError):
    """Invalid input to the shot grammar; ``problems`` lists every problem found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        return None
    return int(value)


def shot_bounds(cut_frames: Iterable[int], n_frames: int) -> list[tuple[int, int]]:
    """Inclusive (start_frame, end_frame) of every shot, in order.

    ``cut_frames`` are the first frames of the shots after the first one; each must be an
    integer in 1 .. n_frames - 1 and appear once (any order). Raises ShotsInputError
    listing every problem.
    """
    problems: list[str] = []
    count = _as_int(n_frames)
    if count is None or count < 1:
        problems.append(f"n_frames must be a positive integer, got {n_frames!r}")
    cuts: list[int] = []
    try:
        values = list(cut_frames)
    except TypeError:
        values = []
        problems.append(f"cut_frames must be an iterable of frame indices, got {cut_frames!r}")
    for value in values:
        cut = _as_int(value)
        if cut is None:
            problems.append(f"cut frame {value!r} is not an integer")
        elif count is not None and not 1 <= cut <= count - 1:
            problems.append(
                f"cut frame {cut} must lie in 1..{count - 1} (it is the first frame of a shot"
                " that follows another one)"
            )
        elif cut in cuts:
            problems.append(f"cut frame {cut} is listed more than once")
        else:
            cuts.append(cut)
    if problems or count is None:
        raise ShotsInputError("shot_bounds", problems)
    starts = [0, *sorted(cuts)]
    ends = [start - 1 for start in starts[1:]] + [count - 1]
    return list(zip(starts, ends, strict=True))


def classify_shot(median_box_h: float, fresh_share: float) -> str:
    """Shot type from the median person box height (1080-line pixels, NaN when nobody was
    detected) and the pitch-usable share (0..1); see the module docstring for the rules."""
    problems = []
    if not isinstance(median_box_h, int | float | np.floating) or isinstance(median_box_h, bool):
        problems.append(f"median_box_h must be a number (NaN allowed), got {median_box_h!r}")
    if (
        not isinstance(fresh_share, int | float | np.floating)
        or isinstance(fresh_share, bool)
        or not 0.0 <= float(fresh_share) <= 1.0
    ):
        problems.append(f"fresh_share must be a number in [0, 1], got {fresh_share!r}")
    if problems:
        raise ShotsInputError("classify_shot", problems)
    height = float(median_box_h)
    share = float(fresh_share)
    if math.isnan(height):
        return "close" if share < CLOSE_MAX_FRESH_SHARE else "medium"
    if height >= CLOSE_MIN_BOX_H_PX:
        return "close"
    if share < CLOSE_MAX_FRESH_SHARE and height >= CLOSE_UNCALIBRATED_MIN_BOX_H_PX:
        return "close"
    if height < WIDE_MAX_BOX_H_PX and share >= WIDE_MIN_FRESH_SHARE:
        return "wide"
    return "medium"


def _frame_height(calib_df: pd.DataFrame, frame_height: int | None, problems: list[str]) -> int:
    if frame_height is not None:
        height = _as_int(frame_height)
        if height is None or height < 1:
            problems.append(f"frame_height must be a positive integer, got {frame_height!r}")
            return REFERENCE_HEIGHT_PX
        return height
    if "height" not in calib_df.columns or calib_df.empty:
        problems.append("calib_df has no 'height' values; pass frame_height")
        return REFERENCE_HEIGHT_PX
    heights = np.unique(calib_df["height"].to_numpy())
    if heights.size != 1:
        problems.append(
            f"calib_df lists several frame heights {heights.tolist()}; pass frame_height"
        )
        return REFERENCE_HEIGHT_PX
    return int(heights[0])


def build_shots(
    cut_frames: Iterable[int],
    n_frames: int,
    tracks_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    *,
    frame_height: int | None = None,
) -> pd.DataFrame:
    """The shots table (``shots.parquet`` columns, :data:`SHOTS_COLUMNS`).

    Args:
        cut_frames: first frames of every shot but the first (e.g. from
            :func:`detect_cut_frames`).
        n_frames: canonical frame count of the video.
        tracks_df: tracks.parquet rows (needs frame, role, y1, y2), pixels of the export.
        calib_df: calib.parquet rows (needs frame_index, pitch_usable, height).
        frame_height: pixel height of the frames the boxes refer to; default: the single
            height listed in calib_df.

    Returns one row per shot: shot_id (0, 1, ... in time order), start_frame and end_frame
    (inclusive), shot_type ("wide" | "medium" | "close"), fresh_share (0..1) and
    median_box_h (1080-line pixels, NaN without person boxes). Raises ShotsInputError
    listing every problem (bad cuts, missing columns, frames outside 0..n_frames - 1,
    duplicate calib rows).
    """
    problems: list[str] = []
    bounds: list[tuple[int, int]] = []
    try:
        bounds = shot_bounds(cut_frames, n_frames)
    except ShotsInputError as exc:
        problems.extend(exc.problems)
    for name, frame, needed in (
        ("tracks_df", tracks_df, ("frame", "role", "y1", "y2")),
        ("calib_df", calib_df, ("frame_index", "pitch_usable")),
    ):
        if not isinstance(frame, pd.DataFrame):
            problems.append(f"{name} must be a pandas DataFrame, got {type(frame).__name__}")
            continue
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            problems.append(f"{name} lacks column(s) {', '.join(missing)}")
    if problems:
        raise ShotsInputError("build_shots", problems)
    height = _frame_height(calib_df, frame_height, problems)
    count = int(n_frames)

    track_frames = tracks_df["frame"].to_numpy(dtype=np.int64)
    outside = np.count_nonzero((track_frames < 0) | (track_frames >= count))
    if outside:
        problems.append(f"tracks_df has {outside} row(s) with frame outside 0..{count - 1}")
    calib_frames = calib_df["frame_index"].to_numpy(dtype=np.int64)
    if np.unique(calib_frames).size != calib_frames.size:
        problems.append("calib_df has duplicate frame_index values")
    calib_outside = np.count_nonzero((calib_frames < 0) | (calib_frames >= count))
    if calib_outside:
        problems.append(f"calib_df has {calib_outside} row(s) with frame_index outside the video")
    if problems:
        raise ShotsInputError("build_shots", problems)

    usable = np.zeros(count, dtype=bool)
    usable[calib_frames] = calib_df["pitch_usable"].to_numpy(dtype=bool)

    people = tracks_df.loc[tracks_df["role"].isin(PERSON_ROLES).to_numpy(dtype=bool)]
    box_h = (people["y2"].to_numpy(np.float64) - people["y1"].to_numpy(np.float64)) * (
        REFERENCE_HEIGHT_PX / height
    )
    per_frame = (
        pd.Series(box_h, index=people["frame"].to_numpy(dtype=np.int64)).groupby(level=0).median()
    )
    frame_median = np.full(count, np.nan)
    frame_median[per_frame.index.to_numpy(dtype=np.int64)] = per_frame.to_numpy()

    rows = []
    for shot_id, (start, end) in enumerate(bounds):
        span = slice(start, end + 1)
        share = float(np.count_nonzero(usable[span])) / float(end - start + 1)
        medians = frame_median[span]
        medians = medians[np.isfinite(medians)]
        median_h = float(np.median(medians)) if medians.size else float("nan")
        rows.append(
            {
                "shot_id": shot_id,
                "start_frame": start,
                "end_frame": end,
                "shot_type": classify_shot(median_h, share),
                "fresh_share": share,
                "median_box_h": median_h,
            }
        )
    shots = pd.DataFrame(rows, columns=list(SHOTS_COLUMNS))
    return shots.astype(
        {
            "shot_id": np.int64,
            "start_frame": np.int64,
            "end_frame": np.int64,
            "fresh_share": np.float64,
            "median_box_h": np.float64,
        }
    )


def shot_ids(shots_df: pd.DataFrame, frames: Sequence[int] | np.ndarray) -> np.ndarray:
    """shot_id (int64) of each frame index; -1 for frames that no shot covers.

    ``shots_df`` must hold non-overlapping inclusive [start_frame, end_frame] ranges.
    Raises ShotsInputError listing every problem with shots_df.
    """
    problems = []
    if not isinstance(shots_df, pd.DataFrame):
        raise ShotsInputError("shot_ids", [f"shots_df must be a DataFrame, got {shots_df!r}"])
    missing = [c for c in ("shot_id", "start_frame", "end_frame") if c not in shots_df.columns]
    if missing:
        raise ShotsInputError("shot_ids", [f"shots_df lacks column(s) {', '.join(missing)}"])
    ordered = shots_df.sort_values("start_frame", kind="stable")
    starts = ordered["start_frame"].to_numpy(dtype=np.int64)
    ends = ordered["end_frame"].to_numpy(dtype=np.int64)
    ids = ordered["shot_id"].to_numpy(dtype=np.int64)
    if np.any(ends < starts):
        problems.append("shots_df has a shot whose end_frame precedes its start_frame")
    if starts.size > 1 and np.any(starts[1:] <= ends[:-1]):
        problems.append("shots_df has overlapping shots")
    if problems:
        raise ShotsInputError("shot_ids", problems)
    values = np.asarray(frames, dtype=np.int64)
    result = np.full(values.shape, -1, dtype=np.int64)
    if ends.size == 0:
        return result
    position = np.searchsorted(starts, values, side="right") - 1
    safe = np.clip(position, 0, None)
    inside = (position >= 0) & (values <= ends[safe])
    result[inside] = ids[safe[inside]]
    return result


def detect_cut_frames(
    video_path: str | Path,
    *,
    threshold: float = DEFAULT_CUT_THRESHOLD,
    fps: str | Fraction | None = None,
) -> list[int]:
    """Hard cuts of a video: the first frame of every shot after the first.

    Streams the frames once through gbf.ingest.iter_frames (the project's single frame
    reader; memory stays at two histograms) into gbf.ingest.detect_cuts with its documented
    default threshold (an empirical starting value, see gbf.ingest.cuts). ``fps`` maps a
    non-CFR file onto the canonical grid first; leave it None for canonical files.
    """
    return detect_cuts(iter_frames(video_path, fps=fps), threshold)
