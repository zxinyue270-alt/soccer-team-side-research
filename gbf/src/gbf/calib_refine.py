"""Temporally consistent calibration of broadcast wide shots from pitch keypoints.

A broadcast camera turns on a fixed head: within one camera the centre ``C`` stays put while
rotation and focal length change smoothly. Calibrations fitted frame by frame (as the
sn-gamestate plugin does) trade focal length against camera distance from one frame to the
next. Their reprojection error stays small, but the ground positions of players jump by
tenths of a metre per frame, and those jumps become fake speed peaks.

:func:`refine_calibration` fits, for each group of wide shots that share a camera
(:func:`camera_groups`), one centre ``C`` for the group plus a rotation and a focal length
per frame. The cost is the robust (soft-L1) reprojection error of the frame's pitch
keypoints plus a penalty on the second difference of rotation and log focal length between
consecutive frames of the same shot; nothing is smoothed across a cut. Principal point and
square pixels are taken from the plugin's intrinsics; lens distortion is not modelled.

The report entry of a fitted group gives the solver's status and evaluations and whether it
converged: a solver stopped at ``max_nfev`` has not, and where it stopped depends on
floating-point detail, so it is not the fit this method defines. A group whose refit (the one
on all keypoints, which is returned) did not converge falls back to the plugin's calibration
and usability, as described below, unless it is rejected.

A frame is refined when it has at least ``min_points`` keypoints or lies in a run of weaker
frames of at most ``max_gap_frames`` between two such frames of the same shot. Refined
frames get ``calib_status`` ``'refined'``; within a group only refined frames whose
reprojection error is at most ``rep_err_max_px`` are pitch-usable. Frames outside every
group keep the plugin's calibration and usability, and so do the frames of a group that
cannot be fitted (fewer than three frames to fit, or none with a fresh plugin calibration
to start from): the report's group entry gives the reason under ``not_built``.

Evaluation (``holdout_fraction > 0``): a seeded share of every frame's keypoints is held
out, the refinement and a per-frame fit with a free centre (the plugin's model) are both
fitted on the remaining points, and the report gives the error of each on the held-out
points. The calibration that is returned is fitted on all points. A group whose held-out
90th-percentile error is more than ``holdout_fallback_ratio`` times that of the per-frame
fits falls back: the refit made the calibration worse, so its frames keep the plugin's
calibration and usability, as those of a group that cannot be fitted do (the report's group
entry says why under ``fallback``, and the group stays out of the report's held-out
summary). Otherwise a group whose held-out 90th-percentile error exceeds
``holdout_p90_max_px`` is rejected: its frames are still refined but none is pitch-usable
(the report's group entry says why); when both fits are that far off, neither is trusted.
The group entry's ``holdout`` also says whether the refit on the remaining points converged,
as the entry does for the refit on all points.

Keypoints come from a plugin as pixel-to-pitch correspondences ``[u, v, X, Y, Z, score]``
(SoccerNet world frame, metres); see :func:`load_keypoints`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from gbf.geometry import Camera, pixel_to_ground

KEYPOINT_FIELDS = ("u", "v", "X", "Y", "Z", "score")
REFINED_COLUMNS = ("refined", "refit_group", "refit_n_points", "refit_rep_err")
JITTER_KEYS = ("frames", "p50_m", "p90_m", "p99_m")

_PITCH_HALF_LENGTH_M = 52.5
_LOGF_MIN = math.log(100.0)  # focal length limits of the fit, pixels
_LOGF_MAX = math.log(100000.0)
_PITCH_HALF_WIDTH_M = 34.0


class CalibRefineError(ValueError):
    """Bad input to the calibration refinement; ``problems`` lists everything found."""

    def __init__(self, where: str, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s): " + "; ".join(self.problems))


@dataclass(frozen=True)
class RefineSettings:
    """Parameters of :func:`refine_calibration`; all are recorded in the report."""

    #: keypoints below this detector score are ignored
    min_score: float = 0.4
    #: a frame takes part with at least this many keypoints ...
    min_points: int = 4
    #: ... or inside a run of weaker frames no longer than this, within one shot
    max_gap_frames: int = 12
    #: pixel noise of a keypoint (heatmap detectors quantise to their output stride)
    sigma_px: float = 2.5
    #: prior on the change of angular velocity, radians per frame squared
    sigma_rot_acc: float = 0.0005
    #: prior on the change of zoom rate, log focal length per frame squared
    sigma_logf_acc: float = 0.001
    #: wide shots whose median plugin centres are closer than this share one camera
    same_camera_m: float = 8.0
    #: a wide shot needs this many good plugin frames to estimate its centre
    min_group_frames: int = 10
    #: plugin frames with a larger reprojection error are not used to start the fit;
    #: refined frames with a larger error are not pitch-usable
    rep_err_max_px: float = 8.0
    #: share of each frame's keypoints held out for the evaluation (0 disables it)
    holdout_fraction: float = 0.2
    #: seed of the held-out split
    seed: int = 0
    #: evaluation cap of the solver per group, set from convergence counts alone (the dev clips
    #: needed up to about 850); a group that reaches it has not converged: its frames keep
    #: the plugin's calibration, and its report entry says so
    max_nfev: int = 2000
    #: a group whose held-out keypoints have a larger 90th-percentile error is not trusted:
    #: its frames are not pitch-usable (the median can look fine while a tenth of the pitch
    #: is tens of pixels off)
    holdout_p90_max_px: float = 25.0
    #: a group whose held-out 90th-percentile error is more than this many times that of
    #: per-frame fits with a free centre (the plugin's model) falls back: the refit made the
    #: calibration worse, so its frames keep the plugin's calibration and usability
    holdout_fallback_ratio: float = 2.0

    def problems(self) -> list[str]:
        found: list[str] = []
        positive = (
            "sigma_px",
            "sigma_rot_acc",
            "sigma_logf_acc",
            "same_camera_m",
            "rep_err_max_px",
            "holdout_p90_max_px",
        )
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                found.append(f"{name} must be a positive number, got {value!r}")
        if not 0.0 <= float(self.min_score) <= 1.0:
            found.append(f"min_score must be in [0, 1], got {self.min_score!r}")
        if not 0.0 <= float(self.holdout_fraction) < 0.5:
            found.append(f"holdout_fraction must be in [0, 0.5), got {self.holdout_fraction!r}")
        ratio = self.holdout_fallback_ratio
        if not isinstance(ratio, int | float) or not math.isfinite(ratio) or ratio < 1:
            found.append(f"holdout_fallback_ratio must be a number >= 1, got {ratio!r}")
        for name, low in (
            ("min_points", 3),
            ("max_gap_frames", 0),
            ("min_group_frames", 1),
            ("max_nfev", 1),
            ("seed", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                found.append(f"{name} must be an integer >= {low}, got {value!r}")
        return found


@dataclass
class RefineResult:
    """``calib``: the refined calibration table; ``report``: settings, groups, evaluation."""

    calib: pd.DataFrame
    report: dict[str, Any]


def _point_problem(point: Any) -> str | None:
    """Why ``point`` is not a keypoint ``[u, v, X, Y, Z, score, ...]``, or None."""
    if not isinstance(point, list):
        return f"is a {type(point).__name__}, not a list"
    if len(point) < len(KEYPOINT_FIELDS):
        return f"has {len(point)} value(s)"
    head = point[: len(KEYPOINT_FIELDS)]
    if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in head):
        return "has a value that is not a number"
    return None


def load_keypoints(path: str | Path) -> dict[int, np.ndarray]:
    """Read a plugin ``keypoints.jsonl``: one ``{"frame": n, "points": [[u, v, X, Y, Z,
    score, ...], ...]}`` per line. Returns ``{frame: (n, 6) float array}``; extra trailing
    values in a point (such as a detector id) are ignored. Every point is checked on its
    own (a flat reshape would turn six points of five numbers into five points of six).
    Every problem is reported."""
    problems: list[str] = []
    result: dict[int, np.ndarray] = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                problems.append(f"line {number}: not JSON ({error.msg})")
                continue
            frame = row.get("frame") if isinstance(row, dict) else None
            points = row.get("points") if isinstance(row, dict) else None
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                problems.append(f"line {number}: frame must be an integer >= 0, got {frame!r}")
                continue
            if frame in result:
                problems.append(f"line {number}: frame {frame} repeated")
                continue
            if not isinstance(points, list):
                problems.append(f"line {number}: points must be a list")
                continue
            bad = [(i, why) for i, p in enumerate(points) if (why := _point_problem(p))]
            if bad:
                problems.append(
                    f"line {number}: {len(bad)} of {len(points)} point(s) malformed (point "
                    f"{bad[0][0]} {bad[0][1]}); every point needs 6 numbers {KEYPOINT_FIELDS}"
                )
                continue
            try:
                array = np.asarray([p[:6] for p in points], dtype=float).reshape(-1, 6)
            except OverflowError:
                problems.append(f"line {number}: a point holds a number too large for a float")
                continue
            if not np.isfinite(array).all():
                problems.append(f"line {number}: non-finite values in points")
                continue
            if ((array[:, 5] < 0) | (array[:, 5] > 1)).any():
                problems.append(f"line {number}: scores must be in [0, 1]")
                continue
            result[frame] = array
    if problems:
        raise CalibRefineError("load_keypoints", problems)
    return result


def _matrix(value: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size != math.prod(shape) or not np.isfinite(array).all():
        return None
    return array.reshape(shape)


def _project(
    C: np.ndarray, R: np.ndarray, f: np.ndarray, cx: float, cy: float, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (n, 2) and depths (n,) of world points ``X`` (n, 3), per-point ``R``, ``f``."""
    xc = np.einsum("nij,nj->ni", R, X - C)
    z = xc[:, 2]
    safe = np.where(z > 1e-6, z, 1e-6)
    uv = np.stack([f * xc[:, 0] / safe + cx, f * xc[:, 1] / safe + cy], axis=1)
    return uv, z


def _check_inputs(calib_df: Any, shots_df: Any, keypoints: Any) -> list[str]:
    problems: list[str] = []
    needed = {
        "calib_df": (
            calib_df,
            (
                "frame_index",
                "width",
                "height",
                "K",
                "R",
                "C",
                "calib_status",
                "rep_err",
                "pitch_usable",
            ),
        ),
        "shots_df": (shots_df, ("shot_id", "start_frame", "end_frame", "shot_type")),
    }
    for name, (frame, columns) in needed.items():
        if not isinstance(frame, pd.DataFrame):
            problems.append(f"{name} must be a pandas DataFrame")
            continue
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            problems.append(f"{name} misses column(s) {missing}")
    if isinstance(calib_df, pd.DataFrame) and "frame_index" in calib_df.columns:
        if calib_df["frame_index"].duplicated().any():
            problems.append("calib_df has repeated frame_index values")
        if "width" in calib_df.columns and "height" in calib_df.columns and len(calib_df):
            sizes = set(zip(calib_df["width"].tolist(), calib_df["height"].tolist(), strict=True))
            if len(sizes) != 1:
                problems.append(f"calib_df mixes frame sizes {sorted(sizes)}")
    if not isinstance(keypoints, dict):
        problems.append("keypoints must be a dict {frame: (n, 6) array}")
    else:
        for frame, points in keypoints.items():
            array = np.asarray(points)
            if array.ndim != 2 or array.shape[1] < 6:
                problems.append(f"keypoints[{frame!r}] must have shape (n, 6)")
                break
    return problems


def camera_groups(
    shots_df: pd.DataFrame, calib_df: pd.DataFrame, settings: RefineSettings | None = None
) -> list[list[int]]:
    """Shot ids of the wide shots that share a camera, one list per camera.

    A wide shot's centre is the median plugin centre over its fresh frames with
    ``rep_err <= rep_err_max_px``; shots with fewer than ``min_group_frames`` such frames
    are left out. Shots join the first group whose first centre is within
    ``same_camera_m``.
    """
    settings = settings or RefineSettings()
    by_frame = calib_df.set_index("frame_index")
    centres: list[tuple[int, np.ndarray]] = []
    for shot in shots_df.sort_values("start_frame").itertuples():
        if shot.shot_type != "wide":
            continue
        rows = by_frame.loc[
            (by_frame.index >= shot.start_frame) & (by_frame.index <= shot.end_frame)
        ]
        good = rows[
            (rows["calib_status"] == "fresh") & (rows["rep_err"] <= settings.rep_err_max_px)
        ]
        found = [c for c in (_matrix(v, (3,)) for v in good["C"]) if c is not None]
        if len(found) >= settings.min_group_frames:
            centres.append((int(shot.shot_id), np.median(np.stack(found), axis=0)))
    groups: list[list[int]] = []
    anchors: list[np.ndarray] = []
    for shot_id, centre in centres:
        for index, anchor in enumerate(anchors):
            if np.linalg.norm(centre - anchor) < settings.same_camera_m:
                groups[index].append(shot_id)
                break
        else:
            groups.append([shot_id])
            anchors.append(centre)
    return groups


def _frames_to_fit(frames: np.ndarray, counts: np.ndarray, settings: RefineSettings) -> np.ndarray:
    strong = counts >= settings.min_points
    keep = strong.copy()
    index = np.flatnonzero(strong)
    for a, b in zip(index[:-1], index[1:], strict=True):
        if 1 < b - a <= settings.max_gap_frames + 1:
            keep[a + 1 : b] = True
    return frames[keep]


@dataclass
class _Group:
    frames: np.ndarray  # canonical frame numbers, sorted
    shot_of: np.ndarray  # shot id per frame
    obs_frame: np.ndarray  # index into frames, per observation
    obs_uv: np.ndarray
    obs_X: np.ndarray
    holdout: np.ndarray  # bool per observation
    R0: np.ndarray
    f0: np.ndarray
    C0: np.ndarray
    cx: float
    cy: float


def _build_group(
    shot_ids: list[int],
    shots_df: pd.DataFrame,
    by_frame: pd.DataFrame,
    keypoints: dict[int, np.ndarray],
    settings: RefineSettings,
    rng: np.random.Generator,
) -> _Group | str:
    """The fitting problem of one camera group, or the reason why there is none."""
    chosen, owners = [], []
    for shot in (
        shots_df[shots_df["shot_id"].isin(shot_ids)].sort_values("start_frame").itertuples()
    ):
        frames = np.arange(int(shot.start_frame), int(shot.end_frame) + 1)
        counts = np.array(
            [
                int((keypoints.get(int(f), np.zeros((0, 6)))[:, 5] >= settings.min_score).sum())
                for f in frames
            ]
        )
        picked = _frames_to_fit(frames, counts, settings)
        chosen.append(picked)
        owners.append(np.full(len(picked), int(shot.shot_id)))
    frames = np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)
    if len(frames) < 3:
        return (
            f"{len(frames)} frame(s) to fit, at least 3 needed (frames with at least "
            f"{settings.min_points} keypoints of score >= {settings.min_score:g}, and gaps of "
            f"at most {settings.max_gap_frames} frames between them)"
        )
    shot_of = np.concatenate(owners)
    obs_frame, obs_uv, obs_X, holdout = [], [], [], []
    for i, frame in enumerate(frames):
        points = keypoints.get(int(frame))
        if points is None:
            continue
        points = np.asarray(points, dtype=float)[:, :6]
        points = points[points[:, 5] >= settings.min_score]
        if not len(points):
            continue
        held = np.zeros(len(points), dtype=bool)
        k = int(math.floor(settings.holdout_fraction * len(points)))
        if k and len(points) - k >= settings.min_points:
            held[rng.choice(len(points), size=k, replace=False)] = True
        obs_frame.append(np.full(len(points), i))
        obs_uv.append(points[:, :2])
        obs_X.append(points[:, 2:5])
        holdout.append(held)
    if not obs_frame:
        return f"no keypoint of score >= {settings.min_score:g} on the frames to fit"
    R0 = np.zeros((len(frames), 3, 3))
    f0 = np.zeros(len(frames))
    have = np.zeros(len(frames), dtype=bool)
    centres, principal = [], None
    for i, frame in enumerate(frames):
        row = by_frame.loc[int(frame)]
        K, R, C = _matrix(row["K"], (3, 3)), _matrix(row["R"], (3, 3)), _matrix(row["C"], (3,))
        if (
            K is None
            or R is None
            or C is None
            or K[0, 0] <= 0
            or row["calib_status"] not in ("fresh", "held")
        ):
            continue
        R0[i], f0[i], have[i] = R, K[0, 0], True
        rep = row["rep_err"]
        if (
            row["calib_status"] == "fresh"
            and rep is not None
            and np.isfinite(rep)
            and rep <= settings.rep_err_max_px
        ):
            centres.append(C)
            principal = (float(K[0, 2]), float(K[1, 2]))
    if not centres or principal is None:
        return (
            f"none of the {len(frames)} frames to fit has a fresh plugin calibration with "
            f"rep_err <= {settings.rep_err_max_px:g} px to start from"
        )
    for i in np.flatnonzero(~have):
        same = np.flatnonzero(have & (shot_of == shot_of[i]))
        if not len(same):
            same = np.flatnonzero(have)
        j = same[np.argmin(np.abs(same - i))]
        R0[i], f0[i] = R0[j], f0[j]
    return _Group(
        frames=frames,
        shot_of=shot_of,
        obs_frame=np.concatenate(obs_frame),
        obs_uv=np.concatenate(obs_uv),
        obs_X=np.concatenate(obs_X),
        holdout=np.concatenate(holdout),
        R0=R0,
        f0=f0,
        C0=np.median(np.stack(centres), axis=0),
        cx=principal[0],
        cy=principal[1],
    )


def _fit_group(
    group: _Group, use: np.ndarray, settings: RefineSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit C, per-frame rotations and focal lengths on the observations ``use``."""
    n_frames = len(group.frames)
    reference = np.zeros((n_frames, 3, 3))
    omega0 = np.zeros((n_frames, 3))
    for shot in np.unique(group.shot_of):
        members = np.flatnonzero(group.shot_of == shot)
        anchor = group.R0[members[len(members) // 2]]
        reference[members] = anchor
        relative = np.einsum("nij,kj->nik", group.R0[members], anchor)
        omega0[members] = Rotation.from_matrix(relative).as_rotvec()
    same_shot = (group.shot_of[:-2] == group.shot_of[1:-1]) & (
        group.shot_of[1:-1] == group.shot_of[2:]
    )
    consecutive = (np.diff(group.frames)[:-1] == 1) & (np.diff(group.frames)[1:] == 1)
    triples = np.flatnonzero(same_shot & consecutive)
    a, b, c = triples, triples + 1, triples + 2
    obs = np.flatnonzero(use)
    j = group.obs_frame[obs]
    X = group.obs_X[obs]
    uv_obs = group.obs_uv[obs]

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            theta[:3],
            theta[3 : 3 + 3 * n_frames].reshape(n_frames, 3),
            theta[3 + 3 * n_frames :],
        )

    def residuals(theta: np.ndarray) -> np.ndarray:
        C, omega, logf = unpack(theta)
        R = np.einsum("nij,njk->nik", Rotation.from_rotvec(omega).as_matrix(), reference)
        # Clipped instead of bounded: least_squares with bounds took tiny steps and stopped
        # at max_nfev on the development clip; the clip only guards trial steps.
        focal = np.exp(np.clip(logf[j], _LOGF_MIN, _LOGF_MAX))
        uv, z = _project(C, R[j], focal, group.cx, group.cy, X)
        pixel = ((uv - uv_obs) / settings.sigma_px).ravel()
        behind = np.where(z <= 1e-6, 100.0, 0.0)
        rot = ((omega[c] - 2.0 * omega[b] + omega[a]) / settings.sigma_rot_acc).ravel()
        zoom = (logf[c] - 2.0 * logf[b] + logf[a]) / settings.sigma_logf_acc
        return np.concatenate([pixel, behind, rot, zoom])

    n_obs, n_trip = len(obs), len(triples)
    n_params = 3 + 4 * n_frames
    sparsity = lil_matrix((3 * n_obs + 4 * n_trip, n_params), dtype=np.int8)
    rows = np.arange(n_obs)
    for k in range(3):
        sparsity[: 3 * n_obs, k] = 1
    for row_of in (2 * rows, 2 * rows + 1, 2 * n_obs + rows):
        for comp in range(3):
            sparsity[row_of, 3 + 3 * j + comp] = 1
        sparsity[row_of, 3 + 3 * n_frames + j] = 1
    base = 3 * n_obs
    for t, frames3 in enumerate(zip(a, b, c, strict=True)):
        for frame in frames3:
            for comp in range(3):
                sparsity[base + 3 * t + comp, 3 + 3 * frame + comp] = 1
            sparsity[base + 3 * n_trip + t, 3 + 3 * n_frames + frame] = 1
    theta0 = np.concatenate(
        [group.C0, omega0.ravel(), np.clip(np.log(group.f0), _LOGF_MIN, _LOGF_MAX)]
    )
    solution = least_squares(
        residuals,
        theta0,
        jac_sparsity=sparsity.tocsr(),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        tr_solver="lsmr",
        max_nfev=settings.max_nfev,
    )
    C, omega, logf = unpack(solution.x)
    R = np.einsum("nij,njk->nik", Rotation.from_rotvec(omega).as_matrix(), reference)
    return C, R, np.exp(np.clip(logf, _LOGF_MIN, _LOGF_MAX)), solution


def _free_fit_errors(group: _Group, settings: RefineSettings) -> np.ndarray:
    """Held-out errors of per-frame fits with a free centre (the plugin's model)."""
    errors: list[np.ndarray] = []
    for i in range(len(group.frames)):
        mine = group.obs_frame == i
        fit = mine & ~group.holdout
        held = mine & group.holdout
        if not held.any() or fit.sum() < settings.min_points:
            continue
        X, uv = group.obs_X[fit], group.obs_uv[fit]
        anchor = group.R0[i]

        def residuals(
            theta: np.ndarray, X: np.ndarray = X, uv: np.ndarray = uv, anchor: np.ndarray = anchor
        ) -> np.ndarray:
            R = Rotation.from_rotvec(theta[3:6]).as_matrix() @ anchor
            pred, _ = _project(
                theta[:3],
                np.repeat(R[None], len(X), 0),
                np.full(len(X), math.exp(theta[6])),
                group.cx,
                group.cy,
                X,
            )
            return ((pred - uv) / settings.sigma_px).ravel()

        theta0 = np.concatenate([group.C0, np.zeros(3), [math.log(group.f0[i])]])
        solution = least_squares(residuals, theta0, loss="soft_l1", f_scale=1.0, max_nfev=200)
        R = Rotation.from_rotvec(solution.x[3:6]).as_matrix() @ anchor
        pred, _ = _project(
            solution.x[:3],
            np.repeat(R[None], int(held.sum()), 0),
            np.full(int(held.sum()), math.exp(solution.x[6])),
            group.cx,
            group.cy,
            group.obs_X[held],
        )
        errors.append(np.hypot(*(pred - group.obs_uv[held]).T))
    return np.concatenate(errors) if errors else np.zeros(0)


def _summary(errors: np.ndarray) -> dict[str, Any]:
    if not len(errors):
        return {"n": 0, "median_px": None, "p90_px": None}
    return {
        "n": int(len(errors)),
        "median_px": round(float(np.median(errors)), 3),
        "p90_px": round(float(np.percentile(errors, 90)), 3),
    }


def refine_calibration(
    calib_df: pd.DataFrame,
    keypoints: dict[int, np.ndarray],
    shots_df: pd.DataFrame,
    settings: RefineSettings | None = None,
) -> RefineResult:
    """Refit every wide-shot camera group; see the module docstring.

    Returns a copy of ``calib_df`` with the columns :data:`REFINED_COLUMNS` added: refined
    frames carry the new ``K``, ``R`` and ``C`` and ``calib_status`` ``'refined'``;
    ``refit_rep_err`` is the median reprojection error over the frame's keypoints.
    """
    settings = settings or RefineSettings()
    problems = settings.problems() + _check_inputs(calib_df, shots_df, keypoints)
    if problems:
        raise CalibRefineError("refine_calibration", problems)
    out = calib_df.copy()
    out["refined"] = False
    out["refit_group"] = np.int64(-1)
    out["refit_n_points"] = np.int64(0)
    out["refit_rep_err"] = np.nan
    by_frame = calib_df.set_index("frame_index")
    position = {int(f): i for i, f in enumerate(calib_df["frame_index"].tolist())}
    report: dict[str, Any] = {"settings": asdict(settings), "groups": []}
    held_refined, held_free = [], []
    in_group = np.zeros(len(out), dtype=bool)
    rejected = np.zeros(len(out), dtype=bool)
    fallback = np.zeros(len(out), dtype=bool)
    for gid, shot_ids in enumerate(camera_groups(shots_df, calib_df, settings)):
        rng = np.random.default_rng([settings.seed, gid])
        group = _build_group(shot_ids, shots_df, by_frame, keypoints, settings, rng)
        if isinstance(group, str):
            report["groups"].append(
                {
                    "group": gid,
                    "shots": [int(s) for s in shot_ids],
                    "frames": 0,
                    "not_built": f"{group}: frames keep the plugin calibration",
                }
            )
            continue
        span = np.zeros(len(out), dtype=bool)
        for shot in shots_df[shots_df["shot_id"].isin(shot_ids)].itertuples():
            span |= (out["frame_index"].to_numpy() >= shot.start_frame) & (
                out["frame_index"].to_numpy() <= shot.end_frame
            )
        entry: dict[str, Any] = {
            "group": gid,
            "shots": [int(s) for s in shot_ids],
            "frames": int(len(group.frames)),
            "centre_start": [round(float(v), 3) for v in group.C0],
        }
        held: tuple[np.ndarray, np.ndarray] | None = None
        if settings.holdout_fraction > 0 and group.holdout.any():
            C_h, R_h, f_h, held_solution = _fit_group(group, ~group.holdout, settings)
            held = np.flatnonzero(group.holdout)
            jh = group.obs_frame[held]
            pred, _ = _project(C_h, R_h[jh], f_h[jh], group.cx, group.cy, group.obs_X[held])
            refined_err = np.hypot(*(pred - group.obs_uv[held]).T)
            free_err = _free_fit_errors(group, settings)
            entry["holdout"] = {
                "refined": _summary(refined_err),
                "free_centre": _summary(free_err),
                # this fit decides fallback and rejection; status 0 as for the full refit
                "solver_status": int(held_solution.status),
                "solver_nfev": int(held_solution.nfev),
                "converged": bool(held_solution.status > 0),
            }
            p90 = float(np.percentile(refined_err, 90))
            free_p90 = float(np.percentile(free_err, 90)) if len(free_err) else math.nan
            if p90 > settings.holdout_fallback_ratio * free_p90:
                entry["fallback"] = (
                    f"held-out P90 {p90:.1f} px > {settings.holdout_fallback_ratio:g} x "
                    f"{free_p90:.1f} px of per-frame fits with a free centre: "
                    "frames keep the plugin calibration"
                )
                fallback |= span
                report["groups"].append(entry)
                continue
            held = (refined_err, free_err)
            if p90 > settings.holdout_p90_max_px:
                entry["rejected"] = (
                    f"held-out P90 {p90:.1f} px > {settings.holdout_p90_max_px:g} px: "
                    "frames not pitch-usable"
                )
        C, R, f, solution = _fit_group(group, np.ones(len(group.obs_frame), dtype=bool), settings)
        pred, _ = _project(
            C, R[group.obs_frame], f[group.obs_frame], group.cx, group.cy, group.obs_X
        )
        errors = np.hypot(*(pred - group.obs_uv).T)
        entry.update(
            {
                "centre": [round(float(v), 3) for v in C],
                "observations": int(len(errors)),
                "rep_err_median_px": round(float(np.median(errors)), 3),
                "rep_err_p90_px": round(float(np.percentile(errors, 90)), 3),
                "solver_status": int(solution.status),
                "solver_nfev": int(solution.nfev),
                # Status 0: stopped at max_nfev, wherever the solver happened to be.
                "converged": bool(solution.status > 0),
            }
        )
        if not entry["converged"] and "rejected" not in entry:
            entry["fallback"] = (
                f"the refit stopped at {int(solution.nfev)} evaluations without converging, "
                "wherever the solver happened to be: frames keep the plugin calibration"
            )
            fallback |= span
            report["groups"].append(entry)
            continue
        if held is not None:
            held_refined.append(held[0])
            held_free.append(held[1])
        # Only a fitted group decides the usability of its frames.
        in_group |= span
        for i, frame in enumerate(group.frames):
            mine = group.obs_frame == i
            row = position[int(frame)]
            K = np.array([[f[i], 0.0, group.cx], [0.0, f[i], group.cy], [0.0, 0.0, 1.0]])
            out.at[out.index[row], "K"] = K.ravel()
            out.at[out.index[row], "R"] = R[i].ravel()
            out.at[out.index[row], "C"] = C.copy()
            out.at[out.index[row], "calib_status"] = "refined"
            out.at[out.index[row], "refined"] = True
            out.at[out.index[row], "refit_group"] = gid
            out.at[out.index[row], "refit_n_points"] = int(mine.sum())
            out.at[out.index[row], "refit_rep_err"] = (
                float(np.median(errors[mine])) if mine.any() else np.nan
            )
            rejected[row] = "rejected" in entry
        report["groups"].append(entry)
    rep = out["refit_rep_err"].to_numpy(dtype=float)
    refined = out["refined"].to_numpy(dtype=bool)
    usable_refined = refined & np.isfinite(rep) & (rep <= settings.rep_err_max_px) & ~rejected
    out["pitch_usable"] = np.where(
        in_group, usable_refined, out["pitch_usable"].to_numpy(dtype=bool)
    )
    if held_refined:
        report["holdout"] = {
            "refined": _summary(np.concatenate(held_refined)),
            "free_centre": _summary(np.concatenate(held_free)),
        }
    report["counts"] = {
        "refined_frames": int(refined.sum()),
        "pitch_usable_before": int(calib_df["pitch_usable"].sum()),
        "pitch_usable_after": int(out["pitch_usable"].sum()),
        "rejected_group_frames": int(rejected.sum()),
        "fallback_group_frames": int(fallback.sum()),
    }
    return RefineResult(calib=out, report=report)


def reproject_tracks(
    tracks_df: pd.DataFrame, calib_df: pd.DataFrame, *, in_pitch_margin_m: float = 3.0
) -> pd.DataFrame:
    """Recompute ``pitch_x``, ``pitch_y``, ``pitch_usable``, ``calib_status`` and
    ``in_pitch`` of every track row from its foot pixel and the frame's calibration."""
    problems = []
    for column in ("frame", "foot_u", "foot_v"):
        if column not in tracks_df.columns:
            problems.append(f"tracks_df misses column {column!r}")
    if problems:
        raise CalibRefineError("reproject_tracks", problems)
    out = tracks_df.copy()
    n = len(out)
    pitch_x = np.full(n, np.nan)
    pitch_y = np.full(n, np.nan)
    usable = np.zeros(n, dtype=bool)
    status = np.full(n, "none", dtype=object)
    by_frame = calib_df.set_index("frame_index")
    feet = out[["foot_u", "foot_v"]].to_numpy(dtype=float)
    for frame, rows in out.groupby("frame").indices.items():
        if int(frame) not in by_frame.index:
            continue
        calib = by_frame.loc[int(frame)]
        status[rows] = calib["calib_status"]
        if not bool(calib["pitch_usable"]):
            continue
        camera = Camera(
            K=_matrix(calib["K"], (3, 3)),
            R=_matrix(calib["R"], (3, 3)),
            C=_matrix(calib["C"], (3,)),
            width=int(calib["width"]),
            height=int(calib["height"]),
        )
        points, valid = pixel_to_ground(camera, feet[rows])
        pitch_x[rows] = np.where(valid, points[:, 0], np.nan)
        pitch_y[rows] = np.where(valid, points[:, 1], np.nan)
        usable[rows] = True
    out["pitch_x"] = pitch_x
    out["pitch_y"] = pitch_y
    out["pitch_usable"] = usable
    out["calib_status"] = (
        pd.array(status.tolist(), dtype=out["calib_status"].dtype)
        if "calib_status" in out.columns
        else status
    )
    out["in_pitch"] = (
        usable
        & np.isfinite(pitch_x)
        & (np.abs(pitch_x) <= _PITCH_HALF_LENGTH_M + in_pitch_margin_m)
        & (np.abs(pitch_y) <= _PITCH_HALF_WIDTH_M + in_pitch_margin_m)
    )
    return out


def ground_jitter(tracks_df: pd.DataFrame) -> dict[str, Any]:
    """Per frame, the median over players of |p(t+1) - 2 p(t) + p(t-1)| (metres).

    Real players change velocity by a few millimetres per frame squared, so this is
    dominated by calibration and box noise. Returns frame count and P50/P90/P99."""
    measured = tracks_df[
        tracks_df["pitch_usable"]
        & tracks_df["in_pitch"]
        & (tracks_df["track_id"] >= 0)
        & tracks_df["pitch_x"].notna()
        & tracks_df["pitch_y"].notna()
    ].sort_values(["track_id", "frame"])
    frames = measured["frame"].to_numpy()
    tracks = measured["track_id"].to_numpy()
    x = measured["pitch_x"].to_numpy(dtype=float)
    y = measured["pitch_y"].to_numpy(dtype=float)
    ok = (
        (tracks[2:] == tracks[:-2])
        & (tracks[1:-1] == tracks[:-2])
        & (frames[2:] == frames[:-2] + 2)
    )
    second = np.hypot(x[2:] - 2 * x[1:-1] + x[:-2], y[2:] - 2 * y[1:-1] + y[:-2])[ok]
    per_frame = pd.Series(second).groupby(frames[1:-1][ok]).median()
    if per_frame.empty:
        return {"frames": 0, "p50_m": None, "p90_m": None, "p99_m": None}
    return {
        "frames": int(len(per_frame)),
        "p50_m": round(float(per_frame.quantile(0.5)), 4),
        "p90_m": round(float(per_frame.quantile(0.9)), 4),
        "p99_m": round(float(per_frame.quantile(0.99)), 4),
    }
