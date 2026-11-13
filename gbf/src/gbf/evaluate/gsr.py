"""Deviation of a gbf run from SoccerNet Game State Reconstruction (GSR) annotations.

Implements the measures of ``prereg/eval-sngs021-v1.md`` that gbf computes itself: the sanity
checks and stop rules, the matching M, the pitch-position deviation P and the speed deviation
S, with their cluster bootstrap intervals. Tracking (T) is measured by TrackEval out of process
(``tools/run_trackeval.py``, run by :func:`run_trackeval`); :func:`write_mot` writes the files
it reads. Results are the
deviation of the run from the annotation (the reference), never an error: the annotation is a
reference, not the truth.

Reference
    A GSR ``Labels-GameState.json``. Source image ``N.jpg`` is canonical frame ``N - 1``.
    People are the object annotations whose role is player, goalkeeper or referee; their box
    is ``bbox_image`` (x, y, w, h) and their pitch position the ``bbox_pitch`` bottom middle
    (``bbox_pitch_raw`` is read for context only), in the SoccerNet world frame (metres,
    origin at the centre spot, far touchline at y = -34), the frame gbf's ``pitch_x`` and
    ``pitch_y`` use too. Rows with role ``other`` are distractors; ignore regions are refused.

Prediction
    A run folder: ``refine/tracks.parquet`` and ``import/tracks.parquet``,
    ``refine/calib.parquet`` and ``kinematics.parquet``.

Matching (M)
    Per frame: IoU on continuous coordinates between reference boxes in ascending track id
    order and predicted boxes in ascending ``track_id`` order; IoU below :data:`IOU_MIN` is
    set to 0; ``linear_sum_assignment(score, maximize=True)``; the assigned pairs with
    IoU >= IOU_MIN are kept. Predictions left unmatched are matched the same way to distractor
    boxes; those are dropped from the precision count.

Speeds
    :func:`reference_kinematics` applies gbf's own smoother and chord to the annotated
    positions (``gbf.kinematics``: Savitzky-Golay window 11, order 2, the 0.4 s chord, gaps of
    up to 3 missing frames filled linearly, segments of at least 25 frames, filled ones
    included), without gbf's plausibility cap and outlier tests: implausible reference speeds
    are counted as the noise floor instead.

Quantiles are numpy's default (linear interpolation between order statistics). A statistic
over an empty set is null with n = 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from gbf.kinematics import chord_half_frames

IOU_MIN = 0.5
"""Smallest box IoU of a matched pair (``gates.v1.json`` gsr_evaluation.matching)."""

PERSON_ROLES: frozenset[str] = frozenset({"player", "goalkeeper", "referee"})
"""Reference roles that are people to track."""

DISTRACTOR_ROLES: frozenset[str] = frozenset({"other"})
"""Reference roles whose boxes are distractors: a prediction matched to one is not counted."""

FPS = 25.0
"""Frame rate of the GSR sequences (all are 25 fps CFR)."""

SG_WINDOW = 11
SG_POLY = 2
CHORD_S = 0.4
MAX_GAP_FRAMES = 3
MIN_SEGMENT_FRAMES = 25
"""The run's smoother (``gbf.pipeline.KINEMATICS_PARAMS``; a test pins these to it)."""

MIN_COMPARABLE_FRAMES = 25
"""Shortest common frame set F that makes a reference segment comparable (S1)."""

NOISE_SPEED_MPS = 11.0
NOISE_ACCELERATION_MPS2 = 10.0
"""Noise-floor thresholds of the reference speeds (``gsr_evaluation.noise_floor``)."""

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

ALIGN_FRAMES: tuple[int, ...] = (2, 200, 400, 600, 747)
"""Decoded frames whose source image check 0 looks up."""
ALIGN_OFFSETS: tuple[int, ...] = (-2, -1, 0, 1, 2)

STOP_MIN_MATCHED_SHARE = 0.30
STOP_MIN_MEDIAN_IOU = 0.6
STOP_MAX_MEDIAN_DEVIATION_M = 10.0
"""Stop rules 2 and 3 of the protocol's section 8."""

RESTRICTIONS: tuple[str, ...] = ("no_share",)
"""Tags of every result: they derive from no_share annotations and frames."""

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACKEVAL_RUNNER = REPO_ROOT / "tools" / "run_trackeval.py"
"""Measure T's runner, run out of process by ``--trackeval-python``."""
TRACKERS = {"refine": "gbf-refine", "import": "gbf-import"}
"""TrackEval tracker names of T-primary (refine) and T-secondary (import)."""


class GsrEvalError(ValueError):
    """Unusable input; ``problems`` lists every problem found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        super().__init__(f"{where}: " + "; ".join(self.problems))


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Reference:
    """People and distractor boxes of a GSR annotation file, by canonical frame."""

    people: pd.DataFrame
    """One row per person box: frame, ref_id, role, x1, y1, x2, y2, pitch_x, pitch_y,
    pitch_x_raw, pitch_y_raw (the raw pair is NaN where the file has none)."""
    distractors: pd.DataFrame
    """One row per distractor box: frame, x1, y1, x2, y2."""
    image_numbers: tuple[int, ...]
    """The numbers N of the images ``N.jpg``, sorted."""
    frames_without_camera: frozenset[int]
    """Frames whose image has ``has_labeled_camera`` false."""
    width: int
    height: int

    @property
    def n_frames(self) -> int:
        return len(self.image_numbers)


def _image_number(file_name: str) -> int:
    stem = Path(file_name).stem
    if not stem.isdigit() or int(stem) < 1:
        raise GsrEvalError("labels", [f"image name {file_name!r} is not a positive number"])
    return int(stem)


def _bottom_middle(pitch: Mapping[str, Any] | None) -> tuple[float, float]:
    pitch = pitch or {}
    x, y = pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")
    return (math.nan if x is None else float(x), math.nan if y is None else float(y))


def load_reference(path: Path) -> Reference:
    """Read a GSR ``Labels-GameState.json`` (see the module docstring)."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    frame_of: dict[str, int] = {}
    numbers: list[int] = []
    no_camera: set[int] = set()
    sizes: set[tuple[int, int]] = set()
    for image in document.get("images") or []:
        name = str(image.get("file_name"))
        if image.get("ignore_regions_x") or image.get("ignore_regions_y"):
            problems.append(f"image {name} has ignore regions (not in v1)")
        if not image.get("is_labeled", False):
            problems.append(f"image {name} is not labelled")
        number = _image_number(name)
        numbers.append(number)
        frame_of[str(image["image_id"])] = number - 1
        if not image.get("has_labeled_camera", True):
            no_camera.add(number - 1)
        sizes.add((int(image["width"]), int(image["height"])))
    if len(sizes) != 1:
        problems.append(f"images have {len(sizes)} sizes, expected one")
    people: list[dict[str, Any]] = []
    distractors: list[dict[str, Any]] = []
    for row in document.get("annotations") or []:
        if row.get("supercategory") != "object":
            continue
        role = (row.get("attributes") or {}).get("role")
        box = row.get("bbox_image") or {}
        image_id = str(row.get("image_id"))
        if image_id not in frame_of:
            problems.append(f"annotation {row.get('id')} names unknown image {image_id}")
            continue
        corners = {
            "frame": frame_of[image_id],
            "x1": float(box["x"]),
            "y1": float(box["y"]),
            "x2": float(box["x"]) + float(box["w"]),
            "y2": float(box["y"]) + float(box["h"]),
        }
        if role in PERSON_ROLES:
            px, py = _bottom_middle(row.get("bbox_pitch"))
            rx, ry = _bottom_middle(row.get("bbox_pitch_raw"))
            people.append(
                {
                    **corners,
                    "ref_id": int(row["track_id"]),
                    "role": str(role),
                    "pitch_x": px,
                    "pitch_y": py,
                    "pitch_x_raw": rx,
                    "pitch_y_raw": ry,
                }
            )
        elif role in DISTRACTOR_ROLES:
            distractors.append(corners)
    if problems:
        raise GsrEvalError("labels", problems)
    columns = ["frame", "ref_id", "role", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y"]
    columns += ["pitch_x_raw", "pitch_y_raw"]
    people_df = pd.DataFrame(people, columns=columns).sort_values(["frame", "ref_id"])
    distractor_df = pd.DataFrame(distractors, columns=["frame", "x1", "y1", "x2", "y2"])
    width, height = sizes.pop() if sizes else (0, 0)
    return Reference(
        people=people_df.reset_index(drop=True),
        distractors=distractor_df.reset_index(drop=True),
        image_numbers=tuple(sorted(numbers)),
        frames_without_camera=frozenset(no_camera),
        width=width,
        height=height,
    )


def load_tracks(path: Path) -> pd.DataFrame:
    """The columns of a gbf tracks table the evaluation reads, sorted by frame and track."""
    table = pd.read_parquet(path)
    needed = ["frame", "track_id", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y", "pitch_usable"]
    missing = [name for name in needed if name not in table.columns]
    if missing:
        raise GsrEvalError("tracks", [f"{path} lacks column(s) {', '.join(missing)}"])
    out = table[needed].copy()
    out["pitch_usable"] = out["pitch_usable"].astype(bool)
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- check 0


def grey(frame: np.ndarray) -> np.ndarray:
    """Mean of the colour channels as float (channel order does not matter)."""
    array = np.asarray(frame, dtype=float)
    return array.mean(axis=2) if array.ndim == 3 else array


def frame_alignment(
    decoded: Mapping[int, np.ndarray],
    images: Mapping[int, np.ndarray],
    frames: Iterable[int] = ALIGN_FRAMES,
    offsets: Sequence[int] = ALIGN_OFFSETS,
) -> dict[str, Any]:
    """Check 0: for each decoded frame k, the mean absolute grey-level difference to source
    image k + 1 + o for every offset o; it passes when every k is closest at o = 0."""
    rows: dict[str, list[float]] = {}
    passed = True
    for k in frames:
        diffs = [
            float(np.mean(np.abs(grey(decoded[k]) - grey(images[k + 1 + o])))) for o in offsets
        ]
        rows[str(k)] = diffs
        passed &= int(np.argmin(diffs)) == list(offsets).index(0)
    return {"offsets": list(offsets), "mean_abs_grey_difference": rows, "passed": bool(passed)}


def load_alignment_frames(
    video: Path, images_dir: Path, frames: Iterable[int] = ALIGN_FRAMES
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """The decoded frames and the source images check 0 compares (Pillow reads the JPEGs)."""
    from PIL import Image

    from gbf.ingest import read_frames

    frames = list(frames)
    decoded = read_frames(video, frames)
    wanted = sorted({k + 1 + o for k in frames for o in ALIGN_OFFSETS})
    images = {}
    for number in wanted:
        with Image.open(images_dir / f"{number:06d}.jpg") as image:
            images[number] = np.asarray(image.convert("RGB"))
    return decoded, images


# --------------------------------------------------------------------------- matching (M)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box of ``a`` (n, 4: x1, y1, x2, y2) with every box of ``b`` (m, 4)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _assign(iou: np.ndarray, iou_min: float) -> list[tuple[int, int, float]]:
    """Hungarian pairs (row, column, IoU) maximising the summed IoU, sub-threshold IoU set to
    0, keeping IoU >= iou_min."""
    if iou.size == 0:
        return []
    score = np.where(iou >= iou_min, iou, 0.0)
    rows, cols = linear_sum_assignment(score, maximize=True)
    return [
        (int(r), int(c), float(iou[r, c]))
        for r, c in zip(rows, cols, strict=True)
        if iou[r, c] >= iou_min
    ]


def match(
    reference: Reference, predictions: pd.DataFrame, iou_min: float = IOU_MIN
) -> tuple[pd.DataFrame, np.ndarray]:
    """Matching M. Returns the pairs (frame, ref_id, track_id, iou, ref_row, pred_row; rows
    index ``reference.people`` and ``predictions``) and a boolean mask of the predictions
    matched to a distractor (dropped from the precision count)."""
    people = reference.people
    boxes = ["x1", "y1", "x2", "y2"]
    distractor = np.zeros(len(predictions), dtype=bool)
    pairs: list[dict[str, Any]] = []
    ref_by_frame = people.groupby("frame").indices
    pred_by_frame = predictions.groupby("frame").indices
    dis_by_frame = reference.distractors.groupby("frame").indices
    for frame in sorted(set(ref_by_frame) | set(pred_by_frame)):
        r_rows = np.asarray(ref_by_frame.get(frame, []), dtype=np.int64)
        p_rows = np.asarray(pred_by_frame.get(frame, []), dtype=np.int64)
        taken: set[int] = set()
        if r_rows.size and p_rows.size:
            iou = box_iou(
                people.loc[r_rows, boxes].to_numpy(), predictions.loc[p_rows, boxes].to_numpy()
            )
            for r, c, value in _assign(iou, iou_min):
                taken.add(int(p_rows[c]))
                pairs.append(
                    {
                        "frame": int(frame),
                        "ref_id": int(people.at[r_rows[r], "ref_id"]),
                        "track_id": int(predictions.at[p_rows[c], "track_id"]),
                        "iou": value,
                        "ref_row": int(r_rows[r]),
                        "pred_row": int(p_rows[c]),
                    }
                )
        left = np.asarray([row for row in p_rows if int(row) not in taken], dtype=np.int64)
        d_rows = np.asarray(dis_by_frame.get(frame, []), dtype=np.int64)
        if left.size and d_rows.size:
            iou = box_iou(
                reference.distractors.loc[d_rows, boxes].to_numpy(),
                predictions.loc[left, boxes].to_numpy(),
            )
            for _, c, _ in _assign(iou, iou_min):
                distractor[left[c]] = True
    columns = ["frame", "ref_id", "track_id", "iou", "ref_row", "pred_row"]
    return pd.DataFrame(pairs, columns=columns), distractor


def detection_summary(
    reference: Reference, predictions: pd.DataFrame, pairs: pd.DataFrame, distractor: np.ndarray
) -> dict[str, Any]:
    """Recall and precision of M (the cross-check of TrackEval's Rcll and Prcn)."""
    n_ref = len(reference.people)
    n_pred = int(len(predictions) - int(distractor.sum()))
    n_pairs = len(pairs)
    return {
        "reference_boxes": n_ref,
        "predicted_boxes": n_pred,
        "predicted_boxes_on_distractors": int(distractor.sum()),
        "matched_pairs": n_pairs,
        "recall": n_pairs / n_ref if n_ref else math.nan,
        "precision": n_pairs / n_pred if n_pred else math.nan,
        "median_iou": float(np.median(pairs["iou"])) if n_pairs else math.nan,
    }


# --------------------------------------------------------------------------- position (P)


def camera_ground_points(calib: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Ground point under each frame's camera centre (C's x, y in the SoccerNet frame)."""
    points: dict[int, tuple[float, float]] = {}
    for frame, centre in zip(calib["frame_index"], calib["C"], strict=True):
        c = np.asarray(centre, dtype=float).reshape(-1)
        if c.size == 3 and np.all(np.isfinite(c)):
            points[int(frame)] = (float(c[0]), float(c[1]))
    return points


def position_rows(
    reference: Reference,
    predictions: pd.DataFrame,
    pairs: pd.DataFrame,
    cameras: Mapping[int, tuple[float, float]],
) -> pd.DataFrame:
    """Measure P: one row per pair whose prediction has a pitch position (``pitch_usable`` and
    finite) and whose reference position is finite, with the deviation d = predicted -
    reference (dx, dy), its length, its radial (along the camera-to-player ground direction,
    positive away from the camera) and lateral (d · (-u_y, u_x)) components, the horizontal
    distance from the camera's ground point to the reference position (NaN without a camera)
    and the reference's distance to its ``bbox_pitch_raw`` position (context)."""
    people = reference.people
    pred = predictions.loc[pairs["pred_row"].to_numpy()]
    ref = people.loc[pairs["ref_row"].to_numpy()]
    usable = (
        pred["pitch_usable"].to_numpy(dtype=bool)
        & np.isfinite(pred["pitch_x"].to_numpy(dtype=float))
        & np.isfinite(pred["pitch_y"].to_numpy(dtype=float))
        & np.isfinite(ref["pitch_x"].to_numpy(dtype=float))
        & np.isfinite(ref["pitch_y"].to_numpy(dtype=float))
    )
    frames = pairs["frame"].to_numpy(dtype=np.int64)[usable]
    px = pred["pitch_x"].to_numpy(dtype=float)[usable]
    py = pred["pitch_y"].to_numpy(dtype=float)[usable]
    rx = ref["pitch_x"].to_numpy(dtype=float)[usable]
    ry = ref["pitch_y"].to_numpy(dtype=float)[usable]
    raw_x = ref["pitch_x_raw"].to_numpy(dtype=float)[usable]
    raw_y = ref["pitch_y_raw"].to_numpy(dtype=float)[usable]
    dx, dy = px - rx, py - ry
    cam = np.array([cameras.get(int(f), (math.nan, math.nan)) for f in frames], dtype=float)
    cam = cam.reshape(-1, 2)
    vx, vy = rx - cam[:, 0], ry - cam[:, 1]
    distance = np.hypot(vx, vy)
    with np.errstate(divide="ignore", invalid="ignore"):
        ux, uy = vx / distance, vy / distance
    return pd.DataFrame(
        {
            "frame": frames,
            "ref_id": pairs["ref_id"].to_numpy(dtype=np.int64)[usable],
            "track_id": pairs["track_id"].to_numpy(dtype=np.int64)[usable],
            "role": ref["role"].to_numpy(dtype=object)[usable],
            "dx": dx,
            "dy": dy,
            "deviation": np.hypot(dx, dy),
            "radial": dx * ux + dy * uy,
            "lateral": -dx * uy + dy * ux,
            "camera_distance": distance,
            "raw_gap": np.hypot(raw_x - rx, raw_y - ry),
        }
    )


def terciles(values: np.ndarray) -> tuple[float, float]:
    """Tercile cut points (numpy quantiles 1/3 and 2/3); a value <= the first is near, <= the
    second is middle, else far."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (math.nan, math.nan)
    return float(np.quantile(finite, 1 / 3)), float(np.quantile(finite, 2 / 3))


# --------------------------------------------------------------------------- speed (S)


def reference_kinematics(people: pd.DataFrame, fps: float = FPS) -> pd.DataFrame:
    """Smoothed positions, chord velocities, speeds and accelerations of every reference track
    (see the module docstring). One row per frame of every segment, filled frames included:
    ref_id, role, segment, frame, filled, x, y, vx, vy, speed, defined (speed usable),
    acceleration (NaN where it is not defined)."""
    k = chord_half_frames(fps, CHORD_S)
    h = (SG_WINDOW - 1) // 2
    seconds = 2 * k / fps
    parts: list[pd.DataFrame] = []
    segment_id = 0
    roles = people.groupby("ref_id")["role"].first() if "role" in people else None
    for ref_id, track in people.groupby("ref_id", sort=True):
        track = track.sort_values("frame")
        frames = track["frame"].to_numpy(dtype=np.int64)
        xs = track["pitch_x"].to_numpy(dtype=float)
        ys = track["pitch_y"].to_numpy(dtype=float)
        keep = np.isfinite(xs) & np.isfinite(ys)
        frames, xs, ys = frames[keep], xs[keep], ys[keep]
        if frames.size == 0:
            continue
        starts = np.flatnonzero(np.diff(frames) > MAX_GAP_FRAMES + 1) + 1
        for lo, hi in zip(np.r_[0, starts], np.r_[starts, frames.size], strict=True):
            seg_f, seg_x, seg_y = frames[lo:hi], xs[lo:hi], ys[lo:hi]
            full = np.arange(seg_f[0], seg_f[-1] + 1, dtype=np.int64)
            filled = ~np.isin(full, seg_f)
            x = np.interp(full, seg_f, seg_x)
            y = np.interp(full, seg_f, seg_y)
            n = full.size
            vx = np.full(n, math.nan)
            vy = np.full(n, math.nan)
            defined = np.zeros(n, dtype=bool)
            if n >= MIN_SEGMENT_FRAMES:
                x = savgol_filter(x, SG_WINDOW, SG_POLY, mode="interp")
                y = savgol_filter(y, SG_WINDOW, SG_POLY, mode="interp")
                position = np.arange(n)
                anchor = ~filled & (position >= h) & (position <= n - 1 - h)
                if n > 2 * k:
                    a = np.arange(n - 2 * k)
                    b = a + 2 * k
                    vx[a + k] = (x[b] - x[a]) / seconds
                    vy[a + k] = (y[b] - y[a]) / seconds
                    defined[a + k] = anchor[a] & anchor[b]
            else:
                x = np.full(n, math.nan)
                y = np.full(n, math.nan)
            vx = np.where(defined, vx, math.nan)
            vy = np.where(defined, vy, math.nan)
            acceleration = np.full(n, math.nan)
            if n > 2 * k:
                ax = (vx[2 * k :] - vx[: -2 * k]) / seconds
                ay = (vy[2 * k :] - vy[: -2 * k]) / seconds
                acceleration[k : n - k] = np.hypot(ax, ay)
            parts.append(
                pd.DataFrame(
                    {
                        "ref_id": int(ref_id),
                        "role": None if roles is None else roles.loc[ref_id],
                        "segment": segment_id,
                        "frame": full,
                        "filled": filled,
                        "x": x,
                        "y": y,
                        "vx": vx,
                        "vy": vy,
                        "speed": np.hypot(vx, vy),
                        "defined": defined,
                        "acceleration": acceleration,
                    }
                )
            )
            segment_id += 1
    columns = ["ref_id", "role", "segment", "frame", "filled", "x", "y", "vx", "vy", "speed"]
    columns += ["defined", "acceleration"]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)[columns]


def speed_frames(
    ref_kin: pd.DataFrame,
    pairs: pd.DataFrame,
    pred_kin: pd.DataFrame,
    usable_frames: set[int],
) -> pd.DataFrame:
    """Per reference-segment frame: the reference speed and acceleration, the predicted track M
    pairs with the reference track there, that track's speed and whether it is valid, and
    whether the frame is in F (reference defined, paired, predicted valid, calibration
    usable)."""
    paired = pairs.set_index(["ref_id", "frame"])["track_id"]
    pred = pred_kin.set_index(["track_id", "frame"])
    rows = ref_kin.copy()
    keys = list(
        zip(
            rows["ref_id"].to_numpy(dtype=np.int64),
            rows["frame"].to_numpy(np.int64),
            strict=True,
        )
    )
    track = np.array([paired.get(key, -1) for key in keys], dtype=np.int64)
    pred_speed = np.full(len(rows), math.nan)
    pred_valid = np.zeros(len(rows), dtype=bool)
    for i, (key, t) in enumerate(zip(keys, track, strict=True)):
        if t < 0:
            continue
        at = (int(t), int(key[1]))
        if at in pred.index:
            pred_speed[i] = float(pred.at[at, "speed"])
            pred_valid[i] = bool(pred.at[at, "valid"])
    rows["track_id"] = track
    rows["pred_speed"] = pred_speed
    rows["pred_valid"] = pred_valid
    rows["usable"] = np.isin(rows["frame"].to_numpy(dtype=np.int64), sorted(usable_frames))
    rows["in_f"] = rows["defined"] & (track >= 0) & pred_valid & rows["usable"]
    return rows


SEGMENT_COLUMNS = [
    "segment",
    "ref_id",
    "role",
    "n_frames",
    "n_defined",
    "n_f",
    "comparable",
    "deviation",
    "peak_in_f",
    "quiet_f",
    "deviation_all",
    "identity_exposure",
]


def segment_peaks(frames: pd.DataFrame) -> pd.DataFrame:
    """S1, S1-all, S4 and the S1 sensitivity flags per reference segment of at least
    MIN_SEGMENT_FRAMES frames. ``quiet_f``: no reference speed above NOISE_SPEED_MPS and no
    reference acceleration above NOISE_ACCELERATION_MPS2 on F."""
    out: list[dict[str, Any]] = []
    for segment, seg in frames.groupby("segment", sort=True):
        if len(seg) < MIN_SEGMENT_FRAMES:
            continue
        f = seg[seg["in_f"]]
        defined = seg[seg["defined"]]
        row: dict[str, Any] = {
            "segment": int(segment),
            "ref_id": int(seg["ref_id"].iloc[0]),
            "role": seg["role"].iloc[0] if "role" in seg else None,
            "n_frames": len(seg),
            "n_defined": len(defined),
            "n_f": len(f),
            "comparable": len(f) >= MIN_COMPARABLE_FRAMES,
            "deviation": math.nan,
            "peak_in_f": None,
            "quiet_f": None,
            "deviation_all": math.nan,
            "identity_exposure": math.nan,
        }
        if row["comparable"]:
            row["deviation"] = float(f["pred_speed"].max() - f["speed"].max())
            peak_frame = int(defined.loc[defined["speed"].idxmax(), "frame"])
            row["peak_in_f"] = bool(peak_frame in set(f["frame"].tolist()))
            accel = f["acceleration"].to_numpy(dtype=float)
            row["quiet_f"] = bool(
                not (f["speed"] > NOISE_SPEED_MPS).any()
                and not (np.nan_to_num(accel, nan=0.0) > NOISE_ACCELERATION_MPS2).any()
            )
            counts = f["track_id"].value_counts()
            top = int(counts.max())
            main = min(int(t) for t, c in counts.items() if int(c) == top)
            row["identity_exposure"] = float((f["track_id"] != main).mean())
        valid = seg[(seg["track_id"] >= 0) & seg["pred_valid"]]
        if len(valid) and len(defined):
            row["deviation_all"] = float(valid["pred_speed"].max() - defined["speed"].max())
        out.append(row)
    return pd.DataFrame(out, columns=SEGMENT_COLUMNS)


def smoothed_position_rows(
    ref_kin: pd.DataFrame, pairs: pd.DataFrame, pred_kin: pd.DataFrame
) -> pd.DataFrame:
    """P-secondary-2: the predicted kinematics position of the paired track against the
    reference's smoothed position, on annotated reference frames of segments of at least
    MIN_SEGMENT_FRAMES frames, where the paired track has a kinematics row that is not
    interpolated."""
    ran = ref_kin.groupby("segment")["frame"].transform("size") >= MIN_SEGMENT_FRAMES
    ref = ref_kin[ran & ~ref_kin["filled"]].merge(pairs[["ref_id", "frame", "track_id"]])
    pred = pred_kin[~pred_kin["interpolated"]][["track_id", "frame", "x", "y"]]
    both = ref.merge(pred, on=["track_id", "frame"], suffixes=("_ref", "_pred"))
    deviation = np.hypot(both["x_pred"] - both["x_ref"], both["y_pred"] - both["y_ref"])
    return pd.DataFrame(
        {"ref_id": both["ref_id"].to_numpy(dtype=np.int64), "deviation": deviation.to_numpy()}
    )


# --------------------------------------------------------------------------- statistics


def _quantiles(values: Any) -> dict[str, Any]:
    """n, median, P90 and mean of the finite values, and how many were not finite."""
    v = np.asarray(values, dtype=float).reshape(-1)
    finite = v[np.isfinite(v)]
    out: dict[str, Any] = {"n": int(finite.size), "n_excluded": int(v.size - finite.size)}
    if finite.size == 0:
        return {**out, "median": None, "p90": None, "mean": None}
    return {
        **out,
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
        "mean": float(finite.mean()),
    }


def _signed_median(values: Any) -> float | None:
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


def cluster_bootstrap(
    groups: Mapping[int, np.ndarray],
    statistic: Callable[[np.ndarray], float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """95 % percentile interval of ``statistic`` over resamples of the reference tracks: the
    track ids with at least one finite value, sorted (n of them), drawn by a fresh
    ``default_rng(seed)`` as ``integers(0, n, size=n)`` per resample; every value of a drawn
    track counts once per draw. None when no track has a value."""
    keys = []
    arrays = []
    for key in sorted(int(k) for k in groups):
        values = np.asarray(groups[key], dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            keys.append(key)
            arrays.append(values)
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    stats = np.empty(resamples)
    for i in range(resamples):
        drawn = rng.integers(0, len(keys), size=len(keys))
        stats[i] = statistic(np.concatenate([arrays[j] for j in drawn]))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def _p90(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.9))


def _with_intervals(values: Any, groups: Mapping[int, np.ndarray]) -> dict[str, Any]:
    """_quantiles plus the cluster-bootstrap intervals of the median and the P90."""
    out = _quantiles(values)
    out["median_ci95"] = cluster_bootstrap(groups, np.median) if out["n"] else None
    out["p90_ci95"] = cluster_bootstrap(groups, _p90) if out["n"] else None
    return out


def _groups(frame: pd.DataFrame, column: str, transform: Callable = np.asarray) -> dict:
    return {
        int(k): transform(g[column].to_numpy(dtype=float))
        for k, g in frame.groupby("ref_id", sort=True)
    }


# --------------------------------------------------------------------------- TrackEval files


def write_mot(
    reference: Reference, predictions: pd.DataFrame, root: Path, tracker: str, seq: str
) -> None:
    """Write the MOTChallenge files TrackEval's MotChallenge2DBox reads (benchmark ``GSR``,
    split ``eval``): ``root/gt/GSR-eval/<seq>/gt/gt.txt`` and ``seqinfo.ini`` with every person
    box as class 1 (distractors as class 8, a MOT distractor class), confidence 1 and
    visibility 1, and ``root/trackers/GSR-eval/<tracker>/data/<seq>.txt`` with every
    prediction; frames are 1-based, boxes left, top, width, height."""
    gt_dir = root / "gt" / "GSR-eval" / seq
    (gt_dir / "gt").mkdir(parents=True, exist_ok=True)
    people = reference.people
    lines = [
        f"{int(r.frame) + 1},{int(r.ref_id)},{r.x1:.3f},{r.y1:.3f},{r.x2 - r.x1:.3f},"
        f"{r.y2 - r.y1:.3f},1,1,1"
        for r in people.itertuples(index=False)
    ]
    next_id = int(people["ref_id"].max()) + 1 if len(people) else 1
    for i, r in enumerate(reference.distractors.itertuples(index=False)):
        lines.append(
            f"{int(r.frame) + 1},{next_id + i},{r.x1:.3f},{r.y1:.3f},{r.x2 - r.x1:.3f},"
            f"{r.y2 - r.y1:.3f},1,8,1"
        )
    (gt_dir / "gt" / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (gt_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={seq}\nimDir=img1\nframeRate={int(FPS)}\nseqLength={reference.n_frames}\n"
        f"imWidth={reference.width}\nimHeight={reference.height}\nimExt=.jpg\n",
        encoding="utf-8",
    )
    data_dir = root / "trackers" / "GSR-eval" / tracker / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pred_lines = [
        f"{int(r.frame) + 1},{int(r.track_id)},{r.x1:.3f},{r.y1:.3f},{r.x2 - r.x1:.3f},"
        f"{r.y2 - r.y1:.3f},1,-1,-1,-1"
        for r in predictions.itertuples(index=False)
    ]
    (data_dir / f"{seq}.txt").write_text("\n".join(pred_lines) + "\n", encoding="utf-8")


def run_trackeval(
    python: Path, runner: Path, root: Path, seq: str, n_frames: int
) -> dict[str, Any]:
    """Run the TrackEval runner with interpreter ``python`` (isolated mode) on the files under
    ``root``; returns its summary. Raises CalledProcessError when the runner fails."""
    summary = root / "summary.json"
    summary.unlink(missing_ok=True)
    subprocess.run(
        [str(python), "-I", str(runner), str(root), seq, str(n_frames), str(summary)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return json.loads(summary.read_text(encoding="utf-8"))


def tracking_block(summary: Mapping[str, Any], runner: Path) -> dict[str, Any]:
    """Measure T from the runner's summary: T-primary, T-secondary, whether every number of
    T-secondary equals T-primary's (None when one is missing), the tool and its runner."""
    trackers = summary.get("trackers") or {}
    primary = trackers.get(TRACKERS["refine"])
    secondary = trackers.get(TRACKERS["import"])
    equal = None
    if primary is not None and secondary is not None:
        keys = sorted((set(primary) | set(secondary)) - {"message"})
        equal = all(primary.get(key) == secondary.get(key) for key in keys)
    return {
        "tool": summary.get("tool"),
        "version": summary.get("version"),
        "python": summary.get("python"),
        "runner_sha256": _sha256(runner),
        "t_primary_refine": primary,
        "t_secondary_import": secondary,
        "t_secondary_equals_t_primary": equal,
    }


# --------------------------------------------------------------------------- evaluation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def _run_paths(labels: Path, run: Path) -> dict[str, Path]:
    return {
        "labels": labels,
        "refine_tracks": run / "refine" / "tracks.parquet",
        "import_tracks": run / "import" / "tracks.parquet",
        "refine_calib": run / "refine" / "calib.parquet",
        "kinematics": run / "kinematics.parquet",
        "run": run / "run.json",
    }


def checks(
    reference: Reference,
    refine: pd.DataFrame,
    calib: pd.DataFrame,
    alignment: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """The protocol's section 8: check 0 (given), 1, 2 and 3. Returns the check numbers, the
    failed checks, and M's pairs, P-primary rows and detection summary for the measures to
    reuse."""
    failed: list[str] = []
    if alignment is None or not alignment.get("passed"):
        failed.append("check 0: frame alignment did not pass (or was not run)")
    expected = tuple(range(1, reference.n_frames + 1))
    pred_frames = refine["frame"].to_numpy(dtype=np.int64)
    sizes = {(int(w), int(h)) for w, h in zip(calib["width"], calib["height"], strict=True)}
    if reference.image_numbers != expected:
        failed.append("check 1: reference images are not 1..n")
    if pred_frames.size and (pred_frames.min() < 0 or pred_frames.max() > reference.n_frames - 1):
        failed.append("check 1: a predicted frame lies outside 0..n-1")
    if sizes != {(reference.width, reference.height)}:
        failed.append(f"check 1: image sizes differ ({sorted(sizes)} against the reference)")
    pairs, distractor = match(reference, refine)
    detection = detection_summary(reference, refine, pairs, distractor)
    cameras = camera_ground_points(calib)
    positions = position_rows(reference, refine, pairs, cameras)
    median_dev = float(np.median(positions["deviation"])) if len(positions) else math.nan
    if not detection["recall"] >= STOP_MIN_MATCHED_SHARE:
        failed.append(f"check 2: matched share {detection['recall']:.4f} below 0.30")
    if not detection["median_iou"] >= STOP_MIN_MEDIAN_IOU:
        failed.append(f"check 2: median IoU {detection['median_iou']:.4f} below 0.6")
    if not median_dev <= STOP_MAX_MEDIAN_DEVIATION_M:
        failed.append(f"check 3: P-primary median deviation {median_dev:.4f} m above 10 m")
    numbers = {
        "check0_frame_alignment": dict(alignment) if alignment is not None else None,
        "reference_frames": reference.n_frames,
        "image_sizes": {
            "reference": [reference.width, reference.height],
            "calibration": [list(size) for size in sorted(sizes)],
        },
        "matched_share": detection["recall"],
        "median_iou": detection["median_iou"],
        "p_primary_median_deviation_m": median_dev,
    }
    return numbers, failed, pairs, positions, detection


def _position_block(
    positions: pd.DataFrame, pairs: pd.DataFrame, extra: bool, frames_without_camera: frozenset
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "pairs": len(pairs),
        "coverage": len(positions) / len(pairs) if len(pairs) else None,
        "deviation": _with_intervals(positions["deviation"], _groups(positions, "deviation")),
    }
    if not extra:
        return block
    kept = positions[~positions["frame"].isin(sorted(frames_without_camera))]
    block["deviation_without_frames_lacking_a_labelled_camera"] = {
        "frames_left_out": sorted(int(f) for f in frames_without_camera),
        **_quantiles(kept["deviation"]),
    }
    block["by_role"] = {
        str(role): _quantiles(g["deviation"]) for role, g in positions.groupby("role", sort=True)
    }
    near, far = terciles(positions["camera_distance"].to_numpy())
    distance = positions["camera_distance"].to_numpy(dtype=float)
    bins = {
        "near": positions[distance <= near],
        "middle": positions[(distance > near) & (distance <= far)],
        "far": positions[distance > far],
    }
    block["by_camera_distance"] = {
        "cut_points_m": [near, far],
        **{name: _quantiles(rows["deviation"]) for name, rows in bins.items()},
    }
    block["depth_direction"] = {
        part: {
            "signed_median": _signed_median(positions[part]),
            "abs": _quantiles(np.abs(positions[part].to_numpy(dtype=float))),
        }
        for part in ("radial", "lateral")
    }
    block["context_bbox_pitch_vs_raw"] = _quantiles(positions["raw_gap"])
    return block


def _speed_block(peaks: pd.DataFrame, per_frame: pd.DataFrame, ref_kin: pd.DataFrame) -> dict:
    comparable = peaks[peaks["comparable"]]
    in_f = per_frame[per_frame["in_f"] & per_frame["segment"].isin(comparable["segment"])]
    frame_dev = in_f.assign(dev=in_f["pred_speed"] - in_f["speed"])
    quiet = comparable[comparable["quiet_f"].astype(bool)] if len(comparable) else comparable
    single = comparable[comparable["identity_exposure"] == 0]
    defined = ref_kin["speed"][ref_kin["defined"]].to_numpy(dtype=float)
    acceleration = ref_kin["acceleration"].to_numpy(dtype=float)
    acceleration = acceleration[np.isfinite(acceleration)]
    with_valid = peaks[peaks["deviation_all"].notna()]
    return {
        "segments": len(peaks),
        "comparable_segments": len(comparable),
        "s1_peak": {
            "signed_median": _signed_median(comparable["deviation"]),
            "abs": _with_intervals(
                np.abs(comparable["deviation"].to_numpy(dtype=float)),
                _groups(comparable, "deviation", np.abs),
            ),
            "by_role": {
                str(role): {
                    "segments": len(g),
                    "signed_median": _signed_median(g["deviation"]),
                    "abs": _quantiles(np.abs(g["deviation"].to_numpy(dtype=float))),
                }
                for role, g in comparable.groupby("role", sort=True)
            },
            "f_share_per_segment": [
                {
                    "segment": int(r.segment),
                    "ref_id": int(r.ref_id),
                    "n_f": int(r.n_f),
                    "n_defined": int(r.n_defined),
                }
                for r in peaks.itertuples(index=False)
            ],
            "share_with_reference_peak_in_f": float(comparable["peak_in_f"].astype(bool).mean())
            if len(comparable)
            else None,
            "sensitivity_quiet_segments": {
                "segments": len(quiet),
                "signed_median": _signed_median(quiet["deviation"]),
                "abs": _quantiles(np.abs(quiet["deviation"].to_numpy(dtype=float))),
            },
        },
        "s1_all": {
            "segments_with_a_valid_paired_speed": len(with_valid),
            "segments_without": int(len(peaks) - len(with_valid)),
            "signed_median": _signed_median(with_valid["deviation_all"]),
            "abs": _quantiles(np.abs(with_valid["deviation_all"].to_numpy(dtype=float))),
        },
        "s1_single_track": {
            "segments": len(single),
            "abs": _quantiles(np.abs(single["deviation"].to_numpy(dtype=float))),
        },
        "s2_per_frame": {
            "signed_median": _signed_median(frame_dev["dev"]),
            "abs": _with_intervals(
                np.abs(frame_dev["dev"].to_numpy(dtype=float)),
                _groups(frame_dev, "dev", np.abs),
            ),
        },
        "s3_noise_floor": {
            "reference_speeds": int(defined.size),
            "share_above_11_mps": float(np.mean(defined > NOISE_SPEED_MPS))
            if defined.size
            else None,
            "reference_accelerations": int(acceleration.size),
            "share_above_10_mps2": float(np.mean(acceleration > NOISE_ACCELERATION_MPS2))
            if acceleration.size
            else None,
        },
        "s4_identity_exposure": _quantiles(comparable["identity_exposure"]),
    }


def evaluate(labels: Path, run: Path, alignment: Mapping[str, Any] | None) -> dict[str, Any]:
    """The protocol for one run: section 8 first; when a check fails, only its numbers."""
    reference = load_reference(labels)
    paths = _run_paths(labels, run)
    calib = pd.read_parquet(paths["refine_calib"])
    refine = load_tracks(paths["refine_tracks"])
    numbers, failed, pairs, positions, detection = checks(reference, refine, calib, alignment)
    result: dict[str, Any] = {
        "protocol": "prereg/eval-sngs021-v1.md",
        "restrictions": list(RESTRICTIONS),
        "wording": "deviation from the GSR annotation (reference), never error",
        "inputs": {name: {"sha256": _sha256(p)} for name, p in paths.items()},
        "code": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checks": numbers,
        "stopped": failed,
        "deviations": [],
    }
    if failed:
        return result
    cameras = camera_ground_points(calib)
    usable = {
        int(f) for f, u in zip(calib["frame_index"], calib["pitch_usable"], strict=True) if bool(u)
    }
    result["reference"] = {
        "frames": reference.n_frames,
        "people_boxes": len(reference.people),
        "people_tracks": int(reference.people["ref_id"].nunique()),
        "distractor_boxes": len(reference.distractors),
    }
    refine_block: dict[str, Any] = {
        "matching": detection,
        "predicted_tracks": int(refine["track_id"].nunique()),
        "position": _position_block(positions, pairs, True, reference.frames_without_camera),
    }
    imported = load_tracks(paths["import_tracks"])
    import_pairs, import_distractor = match(reference, imported)
    import_positions = position_rows(reference, imported, import_pairs, cameras)
    keys = ["frame", "ref_id", "track_id"]
    common = import_positions.merge(positions[keys], on=keys)
    refine_common = positions.merge(import_positions[keys], on=keys)
    import_block: dict[str, Any] = {
        "matching": detection_summary(reference, imported, import_pairs, import_distractor),
        "predicted_tracks": int(imported["track_id"].nunique()),
        "boxes_and_ids_equal_refine": bool(
            imported[["frame", "track_id", "x1", "y1", "x2", "y2"]]
            .reset_index(drop=True)
            .equals(refine[["frame", "track_id", "x1", "y1", "x2", "y2"]].reset_index(drop=True))
        ),
        "position": _position_block(
            import_positions, import_pairs, False, reference.frames_without_camera
        ),
        "position_on_pairs_both_have": {
            "import": _with_intervals(common["deviation"], _groups(common, "deviation")),
            "refine": _quantiles(refine_common["deviation"]),
        },
    }
    ref_kin = reference_kinematics(reference.people)
    pred_kin = pd.read_parquet(paths["kinematics"])
    smoothed = smoothed_position_rows(ref_kin, pairs, pred_kin)
    refine_block["position_smoothed"] = _with_intervals(
        smoothed["deviation"], _groups(smoothed, "deviation")
    )
    per_frame = speed_frames(ref_kin, pairs, pred_kin, usable)
    peaks = segment_peaks(per_frame)
    refine_block["speed"] = _speed_block(peaks, per_frame, ref_kin)
    result["refine"] = refine_block
    result["import"] = import_block
    return result


def _jsonable(value: Any) -> Any:
    """NaN and infinities become null; numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str]) -> int:
    """``python -m gbf eval-gsr --labels L --run R --video V --images I --out O
    [--trackeval-python P]``: write ``results.json`` and the TrackEval files under ``O`` and,
    with ``--trackeval-python``, run measure T and add it to ``results.json``. Returns 0; 2 for
    unusable input; 3 when a check of the protocol's section 8 stops the evaluation (then only
    the check numbers are written); 4 when the TrackEval runner fails."""
    parser = argparse.ArgumentParser(
        prog="python -m gbf eval-gsr",
        description="Deviation of a gbf run from GSR annotations (prereg/eval-sngs021-v1.md).",
    )
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path, help="the run's input video")
    parser.add_argument("--images", required=True, type=Path, help="folder of N.jpg images")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seq", default="SNGS-021")
    parser.add_argument(
        "--align-frames",
        default=",".join(str(k) for k in ALIGN_FRAMES),
        help="decoded frames that check 0 looks up",
    )
    parser.add_argument(
        "--trackeval-python",
        type=Path,
        help="interpreter with sn-trackeval installed; without it measure T is not run",
    )
    parser.add_argument("--trackeval-runner", type=Path, default=TRACKEVAL_RUNNER)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        frames = [int(part) for part in args.align_frames.split(",") if part.strip()]
        decoded, images = load_alignment_frames(args.video, args.images, frames)
        alignment = frame_alignment(decoded, images, frames)
        result = evaluate(args.labels, args.run, alignment)
        result["inputs"]["video"] = {"sha256": _sha256(args.video)}
    except (GsrEvalError, OSError, KeyError, ValueError) as exc:
        print(f"eval-gsr: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    code = 3 if result["stopped"] else 0
    if not result["stopped"]:
        reference = load_reference(args.labels)
        root = args.out / "trackeval"
        for name, tracker in TRACKERS.items():
            tracks = load_tracks(args.run / name / "tracks.parquet")
            write_mot(reference, tracks, root, tracker, args.seq)
        result["tracking"] = None
        if args.trackeval_python is not None:
            try:
                summary = run_trackeval(
                    args.trackeval_python, args.trackeval_runner, root, args.seq, reference.n_frames
                )
                result["tracking"] = tracking_block(summary, args.trackeval_runner)
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                stderr = getattr(exc, "stderr", None) or ""
                result["tracking"] = {
                    "error": str(exc),
                    "stderr_tail": stderr.strip().splitlines()[-5:],
                }
                print(f"eval-gsr: TrackEval failed: {exc}", file=sys.stderr)
                code = 4
    text = json.dumps(_jsonable(result), ensure_ascii=False, indent=2, allow_nan=False)
    (args.out / "results.json").write_text(text + "\n", encoding="utf-8")
    for stop in result["stopped"]:
        print(f"eval-gsr: STOP {stop}", file=sys.stderr)
    return code
