"""Relink, inputs: the rows of one single-shot clip in the form every relink module reads.

The relink step (gbf.relink) is a port of a method of the re-identification lab (reid_lab,
outside the repository). This module replaces that lab's data loaders: it turns the refine
stage's track table, the calibration and the plugin's appearance embeddings into one
:class:`ClipInputs` per shot, with the lab's derived quantities:

- unit embeddings: the stored float32 rows, converted to float64 and divided by their length;
  a row with a value that is not finite (or of length zero) has no embedding (all NaN);
- clean rows: boxes fit to describe a person by. A box is clean when it has an embedding, does
  not lie at a border of the frame (within ``CLEAN_EDGE_PX_720``), is at least
  ``MIN_BOX_H_PX_720`` tall and intersects no other box of its frame, tracked or not;
- cameras: the projection matrix P = K R [I | -C] of every frame with a calibration.

Frames are rebased to the shot (0 = its first frame): the method was built for clips of one
continuous camera, and gbf pieces never span a camera cut. The two pixel constants are given
at 720 lines, the frame height of the clips the method was developed on, and scale with the
frame height (``px_scale``); at 1280x720 every comparison is the lab's.

Nothing here reads an annotation or a file, and nothing is cached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gbf.relink import shirts

REFERENCE_HEIGHT_PX = 720
"""Frame height, in lines, at which the pixel constants of the relink step are given."""

CLEAN_EDGE_PX_720 = 1.0
"""Pixels at 720 lines: a box this close to a border of the frame is not clean."""

MIN_BOX_H_PX_720 = 40.0
"""Pixels at 720 lines: a clean box is at least this tall."""

TRACK_COLUMNS: tuple[str, ...] = (
    "frame",
    "track_id",
    "source_track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "role",
    "team",
    "pitch_x",
    "pitch_y",
    "pitch_usable",
    "in_pitch",
)
"""Columns of the refine stage's track table that the relink step reads."""

CALIB_COLUMNS: tuple[str, ...] = ("frame_index", "K", "R", "C")
"""Columns of the calibration table that the relink step reads."""


@dataclass(frozen=True, eq=False)
class ClipInputs:
    """One single-shot clip: n rows of the track table, D embedding dimensions.

    Attributes:
        frame: (n,) int64 frame inside the clip, 0 .. ``n_frames`` - 1.
        piece: (n,) int64 gbf piece id (the track id before relink); negative = no track.
        source: (n,) int64 the plugin's track id.
        box: (n, 4) float64 x1, y1, x2, y2 in pixels.
        role, team: (n,) object, a string or None.
        pitch_xy: (n, 2) float64 metres, NaN where there is none.
        pitch_usable, in_pitch: (n,) bool.
        emb: (n, D) float64 unit rows; an all-NaN row = no embedding.
        clean: (n,) bool, see the module docstring.
        cam_P: (``n_frames``, 3, 4) float64 projection matrices, NaN where the frame has no
            calibration.
        n_frames: frames of the clip.
        fps: frames per second.
        width, height: frame size in pixels.
        px_scale: ``height / REFERENCE_HEIGHT_PX``.
        shirt_lab: optional (n, 3) upper-central CIELAB measurements; missing rows are NaN.
            Without this independent RGB evidence no row is isolated.

    Equality is identity; compare the arrays explicitly.
    """

    frame: np.ndarray
    piece: np.ndarray
    source: np.ndarray
    box: np.ndarray
    role: np.ndarray
    team: np.ndarray
    pitch_xy: np.ndarray
    pitch_usable: np.ndarray
    in_pitch: np.ndarray
    emb: np.ndarray
    clean: np.ndarray
    cam_P: np.ndarray
    n_frames: int
    fps: float
    width: float
    height: float
    px_scale: float
    shirt_lab: np.ndarray | None = None


def unit_embeddings(raw: np.ndarray) -> np.ndarray:
    """The stored embeddings as float64 unit rows; a row that is not finite, or whose length
    is zero, becomes all NaN (no embedding). ``raw`` is (n, D), as stored (float32)."""
    unit = np.asarray(raw).astype(np.float64)
    if unit.ndim != 2:
        raise ValueError(f"embeddings must be an (n, D) array, got shape {unit.shape}")
    with np.errstate(invalid="ignore", divide="ignore"):
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    unit[~np.isfinite(unit).all(axis=1)] = np.nan
    return unit


_HAS_BLOCK_ROWS = 4096
"""Rows converted at a time by :func:`has_embedding` (a memory block, not a parameter)."""


def has_embedding(raw: np.ndarray) -> np.ndarray:
    """(n,) bool: the rows of the stored embeddings that have a unit embedding, by the rule
    of :func:`unit_embeddings` (every value finite and a length above zero). Works through
    the rows in blocks, so a whole track table costs no float64 copy of its embeddings."""
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f"embeddings must be an (n, D) array, got shape {raw.shape}")
    has = np.zeros(len(raw), dtype=bool)
    for start in range(0, len(raw), _HAS_BLOCK_ROWS):
        block = unit_embeddings(raw[start : start + _HAS_BLOCK_ROWS])
        has[start : start + len(block)] = np.isfinite(block).all(axis=1)
    return has


def clean_rows(
    frame: np.ndarray, box: np.ndarray, has: np.ndarray, width: float, height: float
) -> np.ndarray:
    """(n,) bool: the rows whose box is fit to describe a person by.

    ``frame`` (n,), ``box`` (n, 4) pixels and ``has`` (n,) bool (the row has an embedding)
    describe every row of the clip, rows without a track included: a box that intersects
    any other box of its frame is not clean. ``width`` and ``height`` are the frame size.
    """
    frame = np.asarray(frame)
    box = np.asarray(box, dtype=np.float64)
    has = np.asarray(has, dtype=bool)
    if box.shape != (len(frame), 4) or has.shape != frame.shape:
        raise ValueError(
            f"clean_rows: expected box ({len(frame)}, 4) and has ({len(frame)},), got "
            f"{box.shape} and {has.shape}"
        )
    scale = height / REFERENCE_HEIGHT_PX
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    e = CLEAN_EDGE_PX_720 * scale
    edge = (x1 <= e) | (y1 <= e) | (x2 >= width - 1 - e) | (y2 >= height - 1 - e)
    overlap = np.zeros(len(frame), dtype=bool)
    if not len(frame):
        return overlap
    order = np.argsort(frame, kind="stable")
    ordered = frame[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    stops = np.r_[starts[1:], len(ordered)]
    for start, stop in zip(starts, stops, strict=True):
        rows = order[start:stop]
        if len(rows) < 2:
            continue
        ix = np.minimum(x2[rows][:, None], x2[rows][None]) - np.maximum(
            x1[rows][:, None], x1[rows][None]
        )
        iy = np.minimum(y2[rows][:, None], y2[rows][None]) - np.maximum(
            y1[rows][:, None], y1[rows][None]
        )
        hit = (ix > 0) & (iy > 0)
        np.fill_diagonal(hit, False)
        overlap[rows] = hit.any(axis=1)
    return (~edge) & (~overlap) & ((y2 - y1) >= MIN_BOX_H_PX_720 * scale) & has


def cameras(calib: pd.DataFrame, start_frame: int, n_frames: int) -> np.ndarray:
    """(``n_frames``, 3, 4) projection matrices K R [I | -C] of the frames ``start_frame`` ..
    ``start_frame + n_frames - 1`` of the calibration table (columns ``frame_index``, ``K``
    and ``R`` as 9 row-major numbers, ``C`` as 3); NaN where a frame has no row, or a row
    without a finite K, R and C."""
    P = np.full((int(n_frames), 3, 4), np.nan)
    frames = calib["frame_index"].to_numpy(dtype=np.int64)
    inside = np.flatnonzero((frames >= start_frame) & (frames < start_frame + n_frames))
    inside = inside[np.argsort(frames[inside], kind="stable")]
    Ks, Rs, Cs = (calib[name].to_numpy(dtype=object) for name in ("K", "R", "C"))
    for i in inside:
        if Ks[i] is None or Rs[i] is None or Cs[i] is None:
            continue
        K = np.asarray(Ks[i], dtype=np.float64)
        R = np.asarray(Rs[i], dtype=np.float64)
        C = np.asarray(Cs[i], dtype=np.float64)
        if K.shape != (9,) or R.shape != (9,) or C.shape != (3,):
            continue  # a missing value read as NaN, or not a calibration
        if not (np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(C).all()):
            continue
        K, R = K.reshape(3, 3), R.reshape(3, 3)
        P[int(frames[i]) - start_frame] = K @ R @ np.hstack([np.eye(3), -C[:, None]])
    return P


def _labels(column: pd.Series) -> np.ndarray:
    """A string column as an object array of str or None (pandas reads a missing string as
    NaN or NA)."""
    values = column.to_numpy(dtype=object)
    return np.array([v if isinstance(v, str) else None for v in values], dtype=object)


def build_clip(
    tracks: pd.DataFrame,
    calib: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
    frame_size: tuple[int, int],
    shirt_lab: np.ndarray | None = None,
) -> tuple[ClipInputs, np.ndarray]:
    """The relink inputs of the shot ``start_frame`` .. ``end_frame`` (inclusive).

    Args:
        tracks: the refine stage's track table (:data:`TRACK_COLUMNS`), any row order.
        calib: the calibration table (:data:`CALIB_COLUMNS`).
        embeddings: (rows of ``tracks``, D) the plugin's embeddings as stored, row i for
            row i of ``tracks``; a NaN row = no embedding.
        start_frame, end_frame: the shot, canonical frames.
        fps: frames per second.
        frame_size: (width, height) of the video in pixels.

    Returns the inputs and the positions (in ``tracks``) of the rows they hold, in the
    table's order. Frames are rebased: ``frame - start_frame``.

    Raises ValueError naming every problem: a missing column, embeddings of another length,
    a shot without frames, or a piece (id >= 0) with two rows in one frame.
    """
    problems = [f"tracks: column {name} is missing" for name in TRACK_COLUMNS if name not in tracks]
    problems += [f"calib: column {name} is missing" for name in CALIB_COLUMNS if name not in calib]
    embeddings = np.asarray(embeddings)
    shirt_lab = shirts.checked_colours(shirt_lab, len(tracks))
    if embeddings.ndim != 2 or len(embeddings) != len(tracks) or embeddings.shape[1] < 1:
        problems.append(
            f"embeddings: expected one row per track row ({len(tracks)}, D >= 1), got shape "
            f"{embeddings.shape}"
        )
    if end_frame < start_frame:
        problems.append(f"the shot {start_frame}..{end_frame} has no frame")
    width, height = (float(v) for v in frame_size)
    if not (width > 0 and height > 0):
        problems.append(f"frame_size must be two positive numbers, got {frame_size!r}")
    if not float(fps) > 0:
        problems.append(f"fps must be positive, got {fps!r}")
    if problems:
        raise ValueError("relink inputs: " + "; ".join(problems))

    all_frames = tracks["frame"].to_numpy(dtype=np.int64)
    rows = np.flatnonzero((all_frames >= start_frame) & (all_frames <= end_frame))
    part = tracks.iloc[rows]
    frame = all_frames[rows] - int(start_frame)
    piece = part["track_id"].to_numpy(dtype=np.int64)
    tracked = piece >= 0
    keys = np.stack([piece[tracked], frame[tracked]], axis=1)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    twice = unique[counts > 1]
    if len(twice):
        raise ValueError(
            f"relink inputs: {len(twice)} (piece, frame) pair(s) have more than one row, "
            f"first: piece {int(twice[0, 0])} in frame {int(twice[0, 1]) + int(start_frame)}"
        )
    box = part[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
    emb = unit_embeddings(embeddings[rows])
    n_frames = int(end_frame) - int(start_frame) + 1
    inp = ClipInputs(
        frame=frame,
        piece=piece,
        source=part["source_track_id"].to_numpy(dtype=np.int64),
        box=box,
        role=_labels(part["role"]),
        team=_labels(part["team"]),
        pitch_xy=part[["pitch_x", "pitch_y"]].to_numpy(dtype=np.float64),
        pitch_usable=part["pitch_usable"].to_numpy(dtype=bool),
        in_pitch=part["in_pitch"].to_numpy(dtype=bool),
        emb=emb,
        clean=clean_rows(frame, box, ~np.isnan(emb[:, 0]), width, height),
        cam_P=cameras(calib, int(start_frame), n_frames),
        n_frames=n_frames,
        fps=float(fps),
        width=width,
        height=height,
        px_scale=height / REFERENCE_HEIGHT_PX,
        shirt_lab=None if shirt_lab is None else shirt_lab[rows],
    )
    return inp, rows
