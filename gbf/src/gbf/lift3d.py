"""The "3D base": a calibrated height-prior skeleton, facing, trunk lean, image-plane angles.

What this is, plainly
    The height-prior skeleton is an APPROXIMATION for display and coarse orientation (which
    way a player faces, roughly how far the trunk leans). It is NOT a measured 3D pose.
    Every joint is placed on its own pixel ray at the height a standing player of the fitted
    stature would have it (only the hips and knees follow a fitted crouch), so any posture
    that changes joint heights (a raised knee, a kick, a jump, arms up) puts that joint at
    the wrong depth along its ray. Its error has not been measured (ADR 0004:
    "unverified"); use it for overlays and for facing/lean bins, never for centimetre
    claims, and check ``facing_sigma_deg`` / ``trunk_lean_sigma_deg`` before showing an
    angle.

Frames and units
    Input keypoints are Halpe-26 ``[u, v, score]`` in the calibrated frame's pixel grid
    (:mod:`gbf.pose`). Cameras are :class:`gbf.geometry.Camera` (SoccerNet world, metres,
    z down). Output joints are in pitch_zup metres (x along the touchlines, z UP, far
    touchline at y = +34; ``gbf.geometry.soccernet_to_pitch_zup``). Angles are degrees.
    ``facing_deg`` is measured in pitch_zup from +x, counter-clockwise (+90 = +y, towards
    the far touchline), in (-180, 180].

(a) Height-prior lifting with a fitted stature (:func:`lift_height_prior`)
    Heights above the grass for a 1.80 m player (:data:`JOINT_HEIGHT_PRIOR_M`): toes 0.03,
    heel 0.05, ankle 0.09, knee 0.50, hips 0.95, wrist 0.85, elbow 1.10, shoulder 1.45,
    neck 1.55, nose/eyes/ears 1.65, head top 1.80; all scale with stature / 1.80.

    1. Anchor: the confident ankles (else heels and toes) are intersected with their height
       planes; their score-weighted mean is the ground point under the hips. If the two
       feet land more than :data:`MAX_FEET_SPREAD_M` apart, one is taken to be off
       the ground and only the foot lower in the image is used.
    2. Stature, hips and trunk tilt: the body is a hip centre above the anchor and a
       straight trunk-and-head axis from it that may tilt in any direction. Unknowns: the
       stature (bounded by :data:`STATURE_RANGE_M`), the hip height (prior 0.95 * stature
       / 1.80, spread :data:`HIP_HEIGHT_SD_M` for bent knees), the hip's horizontal offset
       from the anchor (prior 0, spread :data:`HIP_OFFSET_SD_M` for strides and leans) and
       the tilt. They are the regularised least-squares fit (soft-L1, pixels) of the
       projected hip centre, shoulder midpoint, neck, ear midpoint and head top. For an
       upright body this is "the stature that best reprojects the head keypoints with the
       feet anchored on the ground"; the tilt keeps a leaning player from being read as a
       shorter one, and the soft hip keeps a crouch or a lunge from being read as a lean.
       Because a lean away from the camera is ambiguous (see c), the fit is run from
       upright and from the mirrored solution and the more upright fit is kept.
    3. Lifting: each keypoint with score >= the threshold is intersected with the
       horizontal plane z = -h at its height h: feet at their scaled priors, hips at the
       fitted hip height, knees lowered in proportion to the hips, trunk and head points
       at the hip height plus their scaled prior offset times cos(tilt), elbows and wrists
       hanging straight down from the tilted shoulders.
    ``reproj_err_px`` is the RMS pixel distance between those axial keypoints and the
    fitted model's projections. Every lifted joint lies on its own ray and so reprojects
    exactly; the residual measures how well the body model explains the image.

    On broadcast wide shots (players around 100 px tall) the fitted statures tend to come
    out low, often at the lower bound, while the detector boxes of the same players measure
    plausible heights with the same calibration: running postures and the keypoints'
    placement make players look shorter than the standing priors. Read ``stature_m`` as a
    scale of the fit, not as a body measurement.

    Not lifted (method "image_plane_only", with a reason): no usable calibration, a frame
    the caller marks airborne, no confident foot keypoint, no confident hip or no upper
    axial keypoint, or a foot ray that misses the grass.

(b) Facing (:func:`estimate_facing`): the body axis is perpendicular to the shoulder and
    hip lines (score-weighted, horizontal). Its sign starts from the left/right labels
    (right-to-left across the body, turned clockwise, is forward) and is then voted on by
    independent cues: toes ahead of heels, the nose ahead of the ears (weighted towards
    profile views), and face visibility (a visible face faces the camera, a hidden one
    faces away; weighted towards views along the camera direction). A mirrored
    left/right labelling is outvoted. ``facing_conf`` in [0, 1] multiplies the axis
    agreement by the sign agreement.

(c) Trunk lean (:func:`trunk_lean`): the hip-centre -> neck vector of the lifted skeleton
    against the vertical. ``trunk_lean_deg`` is that angle measured in the sagittal plane
    (the vertical plane through the facing direction), positive = leaning forward;
    ``trunk_tilt_deg`` is the full, unsigned angle to the vertical (it includes sideways
    lean). ``trunk_lean_sigma_deg`` (:func:`trunk_lean_sigma_deg`) is its first-order
    error for 1 px of keypoint noise and 5 cm of height error on the neck and the hip: the
    component of lean along the camera ray depends on heights through 1 / tan(depression),
    so a player facing towards or away from a main camera gets tens of degrees, a side-on
    player a few. Do not show a lean whose sigma is large.

    Monocular ambiguity, stated plainly: from one camera a trunk tilted theta away from
    the camera (in the vertical plane through the camera ray) projects almost exactly
    like one tilted 2 * alpha - theta, alpha being the ray's depression angle (about 12
    to 15 degrees for a main camera). The fit keeps the more upright of the two (a2);
    ``trunk_lean_alt_deg`` (:func:`mirrored_trunk_lean`) is the other one. They agree for
    a player side-on to the camera. For a player facing away and leaning forward by more
    than alpha, trunk_lean_deg UNDER-states the lean and trunk_lean_alt_deg is the true
    value.

(d) Image-plane angles (:func:`image_plane_angles`), for frames without usable calibration
    and reported for every row: the hip-midpoint -> shoulder-midpoint vector against the
    IMAGE vertical (positive = towards the image right; :func:`image_trunk_lean_deg`, the
    one image-plane trunk angle, which gbf.moments reads falls from) and both knee angles in
    the image (180 = straight). They are image measurements, labelled as such; they are not
    body angles.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import least_squares

from gbf.geometry import Camera, pixel_rays, pixel_to_ground, soccernet_to_pitch_zup
from gbf.pose import HALPE26_INDEX, HALPE26_KEYPOINTS, N_KEYPOINTS, keypoints_array

LiftMethod = Literal["height_prior_calibrated", "image_plane_only"]
LIFT_METHODS: tuple[LiftMethod, ...] = ("height_prior_calibrated", "image_plane_only")

LiftReason = Literal[
    "lifted",
    "no_calibration",
    "airborne",
    "feet_low_confidence",
    "torso_low_confidence",
    "foot_ray_miss",
]
LIFT_REASONS: tuple[LiftReason, ...] = (
    "lifted",
    "no_calibration",
    "airborne",
    "feet_low_confidence",
    "torso_low_confidence",
    "foot_ray_miss",
)

PRIOR_STATURE_M = 1.80
"""Stature the joint-height priors are written for, metres."""

STATURE_RANGE_M: tuple[float, float] = (1.55, 2.05)
"""Bounds of the fitted stature, metres."""

DEFAULT_SCORE_THRESHOLD = 0.3
"""Keypoints scoring below this (RTMPose SimCC score) are ignored."""

MAX_TRUNK_TILT_COMPONENT = 0.7
"""Bound of each horizontal component of the unit trunk axis (about 44 degrees per axis)."""

MAX_FEET_SPREAD_M = 1.4
"""Feet whose ground points are farther apart than this (for a 1.80 m player) are not both
taken as standing: only the foot lower in the image anchors the body."""

HIP_HEIGHT_SD_M = 0.05
"""Prior spread of the hip-centre height around 0.95 * stature / 1.80: room for bent knees."""

HIP_OFFSET_SD_M = 0.25
"""Prior spread of the hip centre's horizontal offset from the feet anchor (strides, leans)."""

MIRROR_COST_TOLERANCE = 0.05
"""The fit is run from upright and from the mirror of that solution; the more upright of
the two wins unless its cost is more than this fraction (plus 0.05) above the other's."""

ROBUST_SCALE = 3.0
"""Residuals beyond this many standard deviations count linearly (soft-L1), not squared."""

POSTURE_HEIGHT_SD_M = 0.05
"""Height uncertainty of a trunk joint used by trunk_lean_sigma_deg, metres."""

SHOULDER_WIDTH_M = 0.38
"""Typical shoulder keypoint spacing of a 1.80 m player (plausibility weight only)."""

HIP_WIDTH_M = 0.22
"""Typical hip keypoint spacing of a 1.80 m player (plausibility weight only)."""

FACE_VISIBLE_SCORE = 0.6
"""Nose and both eyes at or above this: the face is visible (faces the camera)."""

FACE_HIDDEN_SCORE = 0.25
"""Nose and both eyes below this while the head is seen: the face is hidden (faces away)."""

LABEL_CUE_WEIGHT = 1.0
"""Weight of the left/right-label vote on the facing sign. Each foot's toe vote and the
nose vote weigh at most 1, scaled by scores and by how informative the view is."""

FACE_VISIBLE_WEIGHT = 1.5
"""Weight of a visible face (times |cos| of the angle between facing and camera direction):
a face seen front-on outweighs a mirrored left/right labelling."""

FACE_HIDDEN_WEIGHT = 0.5
"""Weight of a hidden face (low scores can also mean occlusion or blur, so it counts less)."""

_PRIOR_BY_NAME: dict[str, float] = {
    "nose": 1.65,
    "left_eye": 1.65,
    "right_eye": 1.65,
    "left_ear": 1.65,
    "right_ear": 1.65,
    "left_shoulder": 1.45,
    "right_shoulder": 1.45,
    "left_elbow": 1.10,
    "right_elbow": 1.10,
    "left_wrist": 0.85,
    "right_wrist": 0.85,
    "left_hip": 0.95,
    "right_hip": 0.95,
    "left_knee": 0.50,
    "right_knee": 0.50,
    "left_ankle": 0.09,
    "right_ankle": 0.09,
    "head": 1.80,
    "neck": 1.55,
    "hip": 0.95,
    "left_big_toe": 0.03,
    "right_big_toe": 0.03,
    "left_small_toe": 0.03,
    "right_small_toe": 0.03,
    "left_heel": 0.05,
    "right_heel": 0.05,
}

JOINT_HEIGHT_PRIOR_M: tuple[float, ...] = tuple(_PRIOR_BY_NAME[name] for name in HALPE26_KEYPOINTS)
"""Height above the grass of each Halpe-26 joint for a 1.80 m standing player, metres."""

_PRIOR = np.array(JOINT_HEIGHT_PRIOR_M)
_HIP_PRIOR = _PRIOR_BY_NAME["hip"]
_SHOULDER_PRIOR = _PRIOR_BY_NAME["left_shoulder"]
_I = HALPE26_INDEX
_TRUNK_JOINTS = tuple(
    _I[n]
    for n in (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "head",
        "neck",
    )
)
_ARM_JOINTS = tuple(_I[n] for n in ("left_elbow", "right_elbow", "left_wrist", "right_wrist"))
_KNEES = (_I["left_knee"], _I["right_knee"])
_HIPS = (_I["left_hip"], _I["right_hip"], _I["hip"])
_ANKLES = (_I["left_ankle"], _I["right_ankle"])
_LEFT_FOOT = (_I["left_ankle"], _I["left_heel"], _I["left_big_toe"], _I["left_small_toe"])
_RIGHT_FOOT = (_I["right_ankle"], _I["right_heel"], _I["right_big_toe"], _I["right_small_toe"])
_FOOT_FALLBACK = (
    _I["left_heel"],
    _I["right_heel"],
    _I["left_big_toe"],
    _I["right_big_toe"],
    _I["left_small_toe"],
    _I["right_small_toe"],
)
# Axial points of the stature/tilt fit: (keypoint indices averaged, height above the hip
# centre for a 1.80 m player). Midpoints of left/right pairs lie on the body axis.
_AXIAL: tuple[tuple[tuple[int, ...], float], ...] = (
    ((_I["left_shoulder"], _I["right_shoulder"]), _SHOULDER_PRIOR - _HIP_PRIOR),
    ((_I["neck"],), _PRIOR_BY_NAME["neck"] - _HIP_PRIOR),
    ((_I["left_ear"], _I["right_ear"]), _PRIOR_BY_NAME["left_ear"] - _HIP_PRIOR),
    ((_I["head"],), _PRIOR_BY_NAME["head"] - _HIP_PRIOR),
)


class LiftInputError(ValueError):
    """Malformed input to the lift. ``problems`` lists every problem found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


@dataclass(frozen=True)
class ImagePlaneAngles:
    """Angles measured in the image, not on the body (labelled "image" everywhere).

    Attributes:
        trunk_lean_deg: angle of the hip-midpoint -> shoulder-midpoint image vector from
            image up, positive when the shoulders are to the image right of the hips; NaN
            unless both hips and both shoulders are seen (:func:`image_trunk_lean_deg`).
        knee_left_deg, knee_right_deg: image angle at the knee between thigh and shank,
            180 = straight; NaN if hip, knee or ankle is below the threshold.
        plane: always "image".
    """

    trunk_lean_deg: float
    knee_left_deg: float
    knee_right_deg: float
    plane: str = "image"


@dataclass(frozen=True)
class FacingEstimate:
    """Facing on the ground in pitch_zup: degrees from +x, counter-clockwise, (-180, 180].

    facing_conf = axis_conf * sign_conf, all in [0, 1]. ``votes`` lists the sign cues as
    (name, signed weight) relative to the reported facing (positive = agrees).
    """

    facing_deg: float
    facing_conf: float
    axis_conf: float
    sign_conf: float
    votes: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class LiftResult:
    """One detection lifted (or not) to the height-prior skeleton.

    Attributes:
        method: "height_prior_calibrated" or "image_plane_only".
        reason: "lifted" or why it was not (see LIFT_REASONS).
        joints3d: (26, 3) pitch_zup metres in Halpe-26 order; NaN rows for keypoints below
            the threshold, and all NaN when not lifted.
        stature_m: fitted stature, metres (NaN when not lifted).
        stature_at_bound: the fit stopped at a bound of STATURE_RANGE_M (suspect).
        facing_deg, facing_conf: see FacingEstimate (NaN when not lifted; facing_conf 0
            when lifted but no shoulder or hip line is seen).
        facing_sigma_deg: first-order 1-sigma error of facing_deg for 1 px of independent
            noise on each shoulder and hip keypoint coordinate, from the lift geometry
            (see :func:`facing_sigma_deg`); NaN without a facing.
        trunk_lean_deg: sagittal-plane lean, positive forward (NaN without a facing).
        trunk_tilt_deg: unsigned angle of the hip-centre -> neck vector to the vertical.
        trunk_lean_alt_deg: sagittal lean of the other monocular solution
            (:func:`mirrored_trunk_lean`); NaN when not lifted.
        trunk_lean_sigma_deg: first-order 1-sigma error of trunk_lean_deg for 1 px keypoint
            noise and 5 cm height error (:func:`trunk_lean_sigma_deg`); NaN when not lifted.
        reproj_err_px: RMS pixel residual of the stature/tilt fit (NaN when not lifted).
        n_fit_points: axial keypoints the fit used (hip centre included).
        image: image-plane angles (always computed from the 2D keypoints).
    """

    method: LiftMethod
    reason: LiftReason
    joints3d: np.ndarray
    stature_m: float
    stature_at_bound: bool
    facing_deg: float
    facing_conf: float
    facing_sigma_deg: float
    trunk_lean_deg: float
    trunk_tilt_deg: float
    trunk_lean_alt_deg: float
    trunk_lean_sigma_deg: float
    reproj_err_px: float
    n_fit_points: int
    image: ImagePlaneAngles


# ----------------------------------------------------------------------------- checks


def _checked_keypoints(value: Any, problems: list[str]) -> np.ndarray | None:
    try:
        array = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        problems.append(f"keypoints: cannot be read as numbers ({error})")
        return None
    if array.shape != (N_KEYPOINTS, 3):
        problems.append(f"keypoints: expected shape ({N_KEYPOINTS}, 3) [u, v, score]")
        return None
    if not np.isfinite(array[:, 2]).all():
        problems.append("keypoints: scores must be finite (u and v may be NaN)")
        return None
    return array


def _checked_threshold(value: Any, problems: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        problems.append(f"score_threshold: expected a number, got {value!r}")
        return None
    if not math.isfinite(float(value)) or float(value) < 0.0:
        problems.append(f"score_threshold: must be finite and >= 0, got {value!r}")
        return None
    return float(value)


def _usable(keypoints: np.ndarray, threshold: float) -> np.ndarray:
    return np.isfinite(keypoints[:, :2]).all(axis=1) & (keypoints[:, 2] >= threshold)


# ------------------------------------------------------------------------ image plane


def _pair_midpoint(
    points: np.ndarray, usable: np.ndarray, pair: tuple[int, int]
) -> np.ndarray | None:
    """Midpoint of a left/right pair when both are usable, else None."""
    if usable[pair[0]] and usable[pair[1]]:
        return 0.5 * (points[pair[0]] + points[pair[1]])
    return None


def _midpoint(
    points: np.ndarray, usable: np.ndarray, single: int, pair: tuple[int, int]
) -> np.ndarray | None:
    """``points[single]`` when usable, else the midpoint of ``pair``, else None."""
    if usable[single]:
        return points[single]
    return _pair_midpoint(points, usable, pair)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 0.0:
        return math.nan
    cosine = float(np.clip(np.dot(a, b) / norm, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def image_trunk_lean_deg(
    keypoints: np.ndarray, score_threshold: float = DEFAULT_SCORE_THRESHOLD
) -> np.ndarray:
    """Trunk lean in the image, degrees, of each row of ``keypoints`` (n, 26, 3): the angle of
    the vector from the hip midpoint to the shoulder midpoint from image up (-v), positive
    when the shoulders are to the image right of the hips, in (-180, 180]. NaN unless both
    hips and both shoulders reach ``score_threshold`` (and the two midpoints differ).

    The one image-plane trunk angle: gbf.moments reads a fall from its absolute value (the
    trunk passing 45°) and gbf.measure's image-trunk-lean (躯干倾斜（画面）) shows it, so
    the two never disagree about a frame. The shoulder line follows the trunk the eye sees
    in a fall better than the neck keypoint does.
    """
    k = np.asarray(keypoints, dtype=np.float64).reshape(-1, N_KEYPOINTS, 3)
    seen = np.isfinite(k[:, :, :2]).all(axis=2) & (k[:, :, 2] >= score_threshold)

    def midpoint(a: str, b: str) -> np.ndarray:
        mid = (k[:, _I[a], :2] + k[:, _I[b], :2]) / 2.0
        mid[~(seen[:, _I[a]] & seen[:, _I[b]])] = np.nan
        return mid

    du, dv = (midpoint("left_shoulder", "right_shoulder") - midpoint("left_hip", "right_hip")).T
    lean = np.degrees(np.arctan2(du, -dv))
    lean[(du == 0.0) & (dv == 0.0)] = np.nan
    return lean


def image_plane_angles(
    keypoints: Any, *, score_threshold: float = DEFAULT_SCORE_THRESHOLD
) -> ImagePlaneAngles:
    """Trunk lean against the image vertical and knee angles in the image, degrees.

    The trunk is the hip midpoint -> shoulder midpoint vector (:func:`image_trunk_lean_deg`).
    Image up is -v. Raises LiftInputError for malformed input.
    """
    problems: list[str] = []
    array = _checked_keypoints(keypoints, problems)
    threshold = _checked_threshold(score_threshold, problems)
    if problems or array is None or threshold is None:
        raise LiftInputError("image_plane_angles", problems)
    usable = _usable(array, threshold)
    uv = array[:, :2]
    lean = float(image_trunk_lean_deg(array[None], threshold)[0])
    knees = []
    for side in ("left", "right"):
        hip_i, knee_i, ankle_i = (_I[f"{side}_{part}"] for part in ("hip", "knee", "ankle"))
        if usable[hip_i] and usable[knee_i] and usable[ankle_i]:
            knees.append(_angle_deg(uv[hip_i] - uv[knee_i], uv[ankle_i] - uv[knee_i]))
        else:
            knees.append(math.nan)
    return ImagePlaneAngles(trunk_lean_deg=lean, knee_left_deg=knees[0], knee_right_deg=knees[1])


# ---------------------------------------------------------------------------- facing


def _wrap_deg(angle: float) -> float:
    """Angle in (-180, 180]."""
    wrapped = math.fmod(angle, 360.0)
    if wrapped <= -180.0:
        wrapped += 360.0
    elif wrapped > 180.0:
        wrapped -= 360.0
    return wrapped


def estimate_facing(
    joints3d_zup: Any,
    scores: Any,
    *,
    camera_xy: Sequence[float] | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    stature_m: float = PRIOR_STATURE_M,
) -> FacingEstimate:
    """Facing direction on the ground from a pitch_zup skeleton (see module docstring, b).

    Args:
        joints3d_zup: (26, 3) pitch_zup metres; NaN rows are unseen joints.
        scores: (26,) keypoint scores.
        camera_xy: camera centre (x, y) in pitch_zup metres, for the face-visibility vote
            and to weight the nose vote; None skips both weightings.
        score_threshold: joints scoring below it are ignored.
        stature_m: scales the plausibility widths and the minimum cue lengths.

    Returns facing_deg NaN with all confidences 0 when neither the shoulder nor the hip
    line is seen.
    """
    problems: list[str] = []
    joints = np.asarray(joints3d_zup, dtype=np.float64)
    marks = np.asarray(scores, dtype=np.float64)
    if joints.shape != (N_KEYPOINTS, 3):
        problems.append(f"joints3d_zup: expected shape ({N_KEYPOINTS}, 3), got {joints.shape}")
    if marks.shape != (N_KEYPOINTS,):
        problems.append(f"scores: expected shape ({N_KEYPOINTS},), got {marks.shape}")
    threshold = _checked_threshold(score_threshold, problems)
    if problems or threshold is None:
        raise LiftInputError("estimate_facing", problems)
    scale = stature_m / PRIOR_STATURE_M
    seen = np.isfinite(joints).all(axis=1) & (marks >= threshold)
    weight_of = np.minimum(marks, 1.0)

    lines: list[tuple[np.ndarray, float]] = []
    for left, right, width in (
        (_I["left_shoulder"], _I["right_shoulder"], SHOULDER_WIDTH_M),
        (_I["left_hip"], _I["right_hip"], HIP_WIDTH_M),
    ):
        if not (seen[left] and seen[right]):
            continue
        across = (joints[left] - joints[right])[:2]
        length = float(np.linalg.norm(across))
        if length < 1e-6:
            continue
        expected = width * scale
        plausibility = min(length / expected, expected / length)
        lines.append((across / length, min(weight_of[left], weight_of[right]) * plausibility))
    if not lines:
        return FacingEstimate(math.nan, 0.0, 0.0, 0.0)
    labels_consistent = True
    if len(lines) == 2 and float(np.dot(lines[0][0], lines[1][0])) < 0.0:
        labels_consistent = False  # shoulders and hips disagree: one pair is mirrored
        weaker = 0 if lines[0][1] < lines[1][1] else 1
        lines[weaker] = (-lines[weaker][0], lines[weaker][1])
    summed = sum(weight * unit for unit, weight in lines)
    total = sum(weight for _, weight in lines)
    length = float(np.linalg.norm(summed))
    if total <= 0.0 or length <= 0.0:
        return FacingEstimate(math.nan, 0.0, 0.0, 0.0)
    axis = summed / length
    axis_conf = (length / total) * min(1.0, total / 1.5)
    forward = np.array([axis[1], -axis[0]])  # right-to-left across the body, turned clockwise

    to_camera = None
    if camera_xy is not None:
        body = _midpoint(joints, seen, _I["hip"], (_I["left_hip"], _I["right_hip"]))
        if body is None:
            body = np.nanmean(joints[seen], axis=0)
        offset = np.asarray(camera_xy, dtype=np.float64)[:2] - body[:2]
        if float(np.linalg.norm(offset)) > 0.0:
            to_camera = offset / np.linalg.norm(offset)
    along_view = abs(float(np.dot(forward, to_camera))) if to_camera is not None else 0.0

    votes: list[tuple[str, float]] = []
    if labels_consistent:
        votes.append(("labels", LABEL_CUE_WEIGHT))
    for side, tips, backs in (
        ("left", (_I["left_big_toe"], _I["left_small_toe"]), (_I["left_heel"], _I["left_ankle"])),
        (
            "right",
            (_I["right_big_toe"], _I["right_small_toe"]),
            (_I["right_heel"], _I["right_ankle"]),
        ),
    ):
        tip_ids = [i for i in tips if seen[i]]
        back = next((i for i in backs if seen[i]), None)
        if not tip_ids or back is None:
            continue
        vector = (joints[tip_ids].mean(axis=0) - joints[back])[:2]
        length = float(np.linalg.norm(vector))
        if length < 0.05 * scale:
            continue
        strength = min([*(weight_of[i] for i in tip_ids), weight_of[back]])
        weight = strength * min(1.0, length / (0.15 * scale))
        votes.append((f"toes_{side}", math.copysign(weight, float(np.dot(vector, forward)))))
    nose = _I["nose"]
    if seen[nose]:
        reference = _pair_midpoint(joints, seen, (_I["left_ear"], _I["right_ear"]))
        if reference is None and seen[_I["neck"]]:
            reference = joints[_I["neck"]]
        if reference is not None:
            vector = (joints[nose] - reference)[:2]
            length = float(np.linalg.norm(vector))
            if length >= 0.03 * scale:
                weight = weight_of[nose] * min(1.0, length / (0.08 * scale)) * (1.0 - along_view)
                votes.append(("nose", math.copysign(weight, float(np.dot(vector, forward)))))
    if to_camera is not None:
        toward = float(np.dot(forward, to_camera))
        face = marks[[nose, _I["left_eye"], _I["right_eye"]]]
        head_seen = seen[_I["head"]] or seen[_I["left_ear"]] or seen[_I["right_ear"]]
        if float(face.min()) >= FACE_VISIBLE_SCORE:
            votes.append(("face_visible", FACE_VISIBLE_WEIGHT * toward))  # looks at the camera
        elif float(face.max()) < FACE_HIDDEN_SCORE and head_seen:
            votes.append(("face_hidden", -FACE_HIDDEN_WEIGHT * toward))  # looks away
    balance = sum(weight for _, weight in votes)
    magnitude = sum(abs(weight) for _, weight in votes)
    if balance < 0.0:
        forward = -forward
        votes = [(name, -weight) for name, weight in votes]
    sign_conf = abs(balance) / magnitude if magnitude > 0.0 else 0.0
    facing = _wrap_deg(math.degrees(math.atan2(forward[1], forward[0])))
    return FacingEstimate(
        facing_deg=facing,
        facing_conf=float(axis_conf * sign_conf),
        axis_conf=float(axis_conf),
        sign_conf=float(sign_conf),
        votes=tuple(votes),
    )


def trunk_lean(joints3d_zup: Any, facing_deg: float) -> tuple[float, float]:
    """(trunk_lean_deg, trunk_tilt_deg) of a pitch_zup skeleton, degrees.

    The vector runs from the hip centre (keypoint 19, else the hip midpoint) to the neck
    (18, else the shoulder midpoint). trunk_lean_deg is its angle to the vertical inside
    the sagittal plane, positive forward along ``facing_deg``, NaN when the facing is NaN;
    trunk_tilt_deg is the unsigned angle to the vertical. Both are NaN when a point is
    missing or the neck is not above the hip.
    """
    trunk = _trunk_vector(joints3d_zup, "trunk_lean")
    if trunk is None:
        return math.nan, math.nan
    vector, _ = trunk
    tilt = math.degrees(math.atan2(math.hypot(vector[0], vector[1]), vector[2]))
    return _sagittal_lean_deg(vector, facing_deg), tilt


def _trunk_vector(joints3d_zup: Any, where: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(neck - hip centre, their midpoint) in pitch_zup, or None (missing, neck not above)."""
    joints = np.asarray(joints3d_zup, dtype=np.float64)
    if joints.shape != (N_KEYPOINTS, 3):
        raise LiftInputError(where, [f"joints3d_zup: expected ({N_KEYPOINTS}, 3)"])
    seen = np.isfinite(joints).all(axis=1)
    neck = _midpoint(joints, seen, _I["neck"], (_I["left_shoulder"], _I["right_shoulder"]))
    hip = _midpoint(joints, seen, _I["hip"], (_I["left_hip"], _I["right_hip"]))
    if neck is None or hip is None or neck[2] <= hip[2]:
        return None
    return neck - hip, 0.5 * (neck + hip)


def _sagittal_lean_deg(vector: np.ndarray, facing_deg: float) -> float:
    """Angle of an upward vector to the vertical inside the sagittal plane, + forward."""
    if not math.isfinite(facing_deg):
        return math.nan
    heading = math.radians(facing_deg)
    forward = vector[0] * math.cos(heading) + vector[1] * math.sin(heading)
    return math.degrees(math.atan2(forward, vector[2]))


def mirrored_trunk_lean(
    joints3d_zup: Any, facing_deg: float, camera_centre_zup: Sequence[float]
) -> float:
    """Sagittal lean (degrees, + forward) of the OTHER monocular solution for the trunk.

    From one camera, a trunk whose tilt away from the camera (in the vertical plane through
    the camera ray) is theta projects almost exactly like one tilted 2 * alpha - theta,
    where alpha is the ray's depression angle at the trunk: both have the same projected
    length (weak perspective). The fit starts upright and so returns the solution tilted
    less far away from the camera; this is the other one, with the sideways component
    kept. It equals trunk_lean_deg for a player side-on to the camera and differs most for
    one facing towards or away from it. NaN when the trunk, the facing or the geometry is
    missing, or when the mirror would be (near) horizontal.
    """
    trunk = _trunk_vector(joints3d_zup, "mirrored_trunk_lean")
    centre = np.asarray(camera_centre_zup, dtype=np.float64)
    if trunk is None or centre.shape != (3,) or not np.isfinite(centre).all():
        return math.nan
    vector, middle = trunk
    away = middle[:2] - centre[:2]
    distance = float(np.linalg.norm(away))
    if distance <= 0.0:
        return math.nan
    along = away / distance
    across = np.array([-along[1], along[0]])
    depression = math.atan2(centre[2] - middle[2], distance)
    tilt_along = math.atan2(float(np.dot(vector[:2], along)), vector[2])
    mirrored = 2.0 * depression - tilt_along
    if abs(mirrored) >= math.radians(89.0):
        return math.nan
    slope_across = float(np.dot(vector[:2], across)) / vector[2]
    horizontal = math.tan(mirrored) * along + slope_across * across
    return _sagittal_lean_deg(np.array([horizontal[0], horizontal[1], 1.0]), facing_deg)


def facing_sigma_deg(
    camera: Camera,
    keypoints: np.ndarray,
    heights_m: np.ndarray,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    pixel_sigma: float = 1.0,
) -> float:
    """First-order 1-sigma error of the shoulder/hip facing axis, degrees.

    Each usable shoulder and hip keypoint is lifted onto its height plane (``heights_m``,
    metres above the grass, per joint) and moved by one pixel in u and in v in turn; the
    resulting changes of the axis-only facing (left/right labels, no cue votes) are
    combined as independent errors and scaled by ``pixel_sigma``. Depth along a camera ray
    is poorly measured from a camera a few degrees above the grass, so a player whose
    shoulder line runs across the image (facing towards or away from the camera) gets a
    large value, and a side-on player a small one. NaN when no line is seen.
    """
    ids = [
        i
        for i in (_I["left_shoulder"], _I["right_shoulder"], _I["left_hip"], _I["right_hip"])
        if np.isfinite(keypoints[i, :2]).all() and keypoints[i, 2] >= score_threshold
    ]

    def axis_facing(pixels: np.ndarray) -> float:
        joints = np.full((N_KEYPOINTS, 3), np.nan)
        points, valid = pixel_to_ground(camera, pixels[ids], -heights_m[ids])
        joints[np.array(ids)[valid]] = soccernet_to_pitch_zup(points[valid])
        estimate = estimate_facing(joints, keypoints[:, 2], score_threshold=score_threshold)
        return estimate.facing_deg

    if len(ids) < 2:
        return math.nan
    pixels = keypoints[:, :2].copy()
    base = axis_facing(pixels)
    if not math.isfinite(base):
        return math.nan
    total = 0.0
    for index in ids:
        for axis in (0, 1):
            moved = pixels.copy()
            moved[index, axis] += 1.0
            changed = axis_facing(moved)
            if not math.isfinite(changed):
                return math.nan
            total += _wrap_deg(changed - base) ** 2
    return float(pixel_sigma * math.sqrt(total))


def _lift_subset(
    camera: Camera, keypoints: np.ndarray, heights_m: np.ndarray, ids: Sequence[int]
) -> np.ndarray:
    """(26, 3) pitch_zup joints with only ``ids`` lifted onto their height planes."""
    joints = np.full((N_KEYPOINTS, 3), np.nan)
    chosen = np.array(list(ids), dtype=np.intp)
    points, valid = pixel_to_ground(camera, keypoints[chosen, :2], -heights_m[chosen])
    joints[chosen[valid]] = soccernet_to_pitch_zup(points[valid])
    return joints


def trunk_lean_sigma_deg(
    camera: Camera,
    keypoints: np.ndarray,
    heights_m: np.ndarray,
    facing_deg: float,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    pixel_sigma: float = 1.0,
    height_sigma_m: float = POSTURE_HEIGHT_SD_M,
) -> float:
    """First-order 1-sigma error of trunk_lean_deg, degrees.

    The neck (or both shoulders) and the hip centre (or both hips) are lifted onto their
    height planes (``heights_m``, metres above the grass, per joint); each keypoint
    coordinate is moved by ``pixel_sigma`` and the neck and hip planes by
    ``height_sigma_m`` in turn, and the changes of the sagittal lean are combined as
    independent errors. NaN without a facing or a trunk.
    """
    usable = _usable(keypoints, score_threshold)
    neck = [_I["neck"]] if usable[_I["neck"]] else []
    if not neck and usable[_I["left_shoulder"]] and usable[_I["right_shoulder"]]:
        neck = [_I["left_shoulder"], _I["right_shoulder"]]
    hip = [_I["hip"]] if usable[_I["hip"]] else []
    if not hip and usable[_I["left_hip"]] and usable[_I["right_hip"]]:
        hip = [_I["left_hip"], _I["right_hip"]]
    if not neck or not hip or not math.isfinite(facing_deg):
        return math.nan
    ids = neck + hip

    def lean(pixels: np.ndarray, heights: np.ndarray) -> float:
        return trunk_lean(_lift_subset(camera, pixels, heights, ids), facing_deg)[0]

    base = lean(keypoints, heights_m)
    if not math.isfinite(base):
        return math.nan
    total = 0.0
    for index in ids:
        for axis in (0, 1):
            moved = keypoints.copy()
            moved[index, axis] += pixel_sigma
            total += (lean(moved, heights_m) - base) ** 2
    for group in (neck, hip):
        raised = heights_m.copy()
        raised[group] += height_sigma_m
        total += (lean(keypoints, raised) - base) ** 2
    return float(math.sqrt(total)) if math.isfinite(total) else math.nan


# ------------------------------------------------------------------------------ lift


def _not_lifted(
    reason: LiftReason, image: ImagePlaneAngles, *, method: LiftMethod = "image_plane_only"
) -> LiftResult:
    return LiftResult(
        method=method,
        reason=reason,
        joints3d=np.full((N_KEYPOINTS, 3), np.nan),
        stature_m=math.nan,
        stature_at_bound=False,
        facing_deg=math.nan,
        facing_conf=math.nan,
        facing_sigma_deg=math.nan,
        trunk_lean_deg=math.nan,
        trunk_tilt_deg=math.nan,
        trunk_lean_alt_deg=math.nan,
        trunk_lean_sigma_deg=math.nan,
        reproj_err_px=math.nan,
        n_fit_points=0,
        image=image,
    )


def _foot_indices(camera: Camera, keypoints: np.ndarray, usable: np.ndarray) -> list[int]:
    """Foot keypoints that anchor the body (see module docstring, a.1)."""
    ankles = [i for i in _ANKLES if usable[i]]
    chosen = ankles if ankles else [i for i in _FOOT_FALLBACK if usable[i]]
    if not chosen:
        return []
    left = [i for i in chosen if i in _LEFT_FOOT]
    right = [i for i in chosen if i in _RIGHT_FOOT]
    if not left or not right:
        return chosen
    heights = -_PRIOR[chosen]
    ground, valid = pixel_to_ground(camera, keypoints[chosen, :2], heights)
    if not valid.all():
        return chosen  # the caller reports the ray miss
    weights = np.minimum(keypoints[chosen, 2], 1.0)
    mask_left = np.array([i in left for i in chosen])
    centre_left = np.average(ground[mask_left, :2], axis=0, weights=weights[mask_left])
    centre_right = np.average(ground[~mask_left, :2], axis=0, weights=weights[~mask_left])
    if float(np.linalg.norm(centre_left - centre_right)) <= MAX_FEET_SPREAD_M:
        return chosen
    lower_left = keypoints[left, 1].mean() >= keypoints[right, 1].mean()
    return left if lower_left else right


def _mirrored_axis_start(
    params: np.ndarray,
    rays: np.ndarray,
    foot_prior: np.ndarray,
    foot_w: np.ndarray,
    centre: np.ndarray,
    above_hip: np.ndarray,
) -> np.ndarray | None:
    """(tx, ty) of the trunk axis mirrored about the camera ray (SoccerNet), or None.

    Same construction as :func:`mirrored_trunk_lean`: the tilt away from the camera theta
    becomes 2 * alpha - theta with alpha the ray's depression angle at mid-trunk; the
    sideways component is kept.
    """
    k, hip_height, ox, oy, tx, ty = (float(v) for v in params)
    t = (-foot_prior * k - centre[2]) / rays[:, 2]
    anchor = foot_w @ (centre[:2] + t[:, None] * rays[:, :2])
    middle = np.array([anchor[0] + ox, anchor[1] + oy])
    mid_height = hip_height + 0.5 * float(above_hip.max(initial=0.0)) * k
    away = middle - centre[:2]
    distance = float(np.linalg.norm(away))
    if distance <= 0.0:
        return None
    along = away / distance
    across = np.array([-along[1], along[0]])
    depression = math.atan2(-centre[2] - mid_height, distance)
    up = math.sqrt(max(1.0 - tx * tx - ty * ty, 1e-12))
    horizontal = np.array([tx, ty])
    tilt_along = math.atan2(float(horizontal @ along), up)
    mirrored = 2.0 * depression - tilt_along
    if abs(mirrored) >= math.radians(80.0):
        return None
    direction = math.tan(mirrored) * along + (float(horizontal @ across) / up) * across
    unit = np.append(direction, 1.0)
    unit /= np.linalg.norm(unit)
    return unit[:2]


@dataclass(frozen=True)
class _BodyFit:
    """Solution of the body fit (SoccerNet frame): scale k = stature / 1.80, hip-centre
    height above the grass (m), its horizontal offset from the feet anchor (m), the unit
    trunk axis (pointing up, so its z is negative), RMS pixel residual, observations used."""

    k: float
    hip_height: float
    offset: np.ndarray
    axis: np.ndarray
    rms_px: float
    n_points: int


def _fit_body(
    camera: Camera,
    keypoints: np.ndarray,
    usable: np.ndarray,
    feet: list[int],
) -> _BodyFit | None:
    """Regularised least squares of the body model (module docstring, a.2), or None.

    Unknowns: k, hip height, hip offset (x, y) and trunk axis (x, y). Data: the hip centre,
    shoulder midpoint, neck, ear midpoint and head top, in pixels divided by a keypoint
    noise of max(1, 1% of the player's image height) px and weighted by their scores.
    Priors: hip height around 0.95 k (HIP_HEIGHT_SD_M) and hip offset 0
    (HIP_OFFSET_SD_M); the stature and the tilt have bounds only, so an upright synthetic
    body is recovered without bias. Soft-L1 loss with scale ROBUST_SCALE. The fit runs
    from upright and again from the mirror of that solution about the camera ray
    (:func:`mirrored_trunk_lean`); the more upright of the two is kept unless it fits
    clearly worse (MIRROR_COST_TOLERANCE).
    """
    observations: list[tuple[np.ndarray, float, float]] = []  # (uv, weight, height over hip)
    hip = _midpoint(keypoints[:, :2], usable, _I["hip"], (_I["left_hip"], _I["right_hip"]))
    if hip is None:
        return None
    hip_score = (
        keypoints[_I["hip"], 2]
        if usable[_I["hip"]]
        else min(keypoints[_I["left_hip"], 2], keypoints[_I["right_hip"], 2])
    )
    observations.append((hip, float(hip_score), 0.0))
    for indices, height in _AXIAL:
        if all(usable[i] for i in indices):
            uv = keypoints[list(indices), :2].mean(axis=0)
            observations.append((uv, float(min(keypoints[list(indices), 2])), height))
    if len(observations) < 2:
        return None
    observed = np.array([uv for uv, _, _ in observations])
    root_w = np.sqrt(np.clip([w for _, w, _ in observations], 1e-6, 1.0))
    above_hip = np.array([h for _, _, h in observations])
    projection = camera.projection_matrix()
    centre = camera.C
    rays = pixel_rays(camera, keypoints[feet, :2])
    foot_prior = _PRIOR[feet]
    foot_w = np.minimum(keypoints[feet, 2], 1.0)
    foot_w = foot_w / foot_w.sum()
    seen_v = keypoints[usable, 1]
    noise_px = max(1.0, 0.01 * float(seen_v.max() - seen_v.min()))

    def model_uv(params: np.ndarray) -> np.ndarray:
        k, hip_height, ox, oy, tx, ty = params
        t = (-foot_prior * k - centre[2]) / rays[:, 2]
        ground = centre[:2] + t[:, None] * rays[:, :2]
        anchor = foot_w @ ground
        dz = -math.sqrt(max(1.0 - tx * tx - ty * ty, 1e-12))
        base = np.array([anchor[0] + ox, anchor[1] + oy, -hip_height])
        points = base + np.outer(above_hip * k, [tx, ty, dz])
        homogeneous = points @ projection[:, :3].T + projection[:, 3]
        return homogeneous[:, :2] / homogeneous[:, 2:3]

    def residuals(params: np.ndarray) -> np.ndarray:
        k, hip_height, ox, oy, tx, ty = params
        pixels = ((model_uv(params) - observed) * (root_w[:, None] / noise_px)).ravel()
        priors = [
            (hip_height - _HIP_PRIOR * k) / HIP_HEIGHT_SD_M,
            ox / HIP_OFFSET_SD_M,
            oy / HIP_OFFSET_SD_M,
        ]
        return np.concatenate([pixels, priors])

    low, high = (bound / PRIOR_STATURE_M for bound in STATURE_RANGE_M)
    limit = MAX_TRUNK_TILT_COMPONENT
    reach = 4.0 * HIP_OFFSET_SD_M
    lower = np.array([low, 0.4, -reach, -reach, -limit, -limit])
    upper = np.array([high, 1.3, reach, reach, limit, limit])

    def solve(start: np.ndarray) -> Any:
        return least_squares(
            residuals,
            x0=np.clip(start, lower + 1e-9, upper - 1e-9),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=ROBUST_SCALE,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=600,
        )

    def tilt_of(params: np.ndarray) -> float:
        return math.hypot(params[4], params[5])

    first = solve(np.array([1.0, _HIP_PRIOR, 0.0, 0.0, 0.0, 0.0]))
    solution = first
    mirror = _mirrored_axis_start(first.x, rays, foot_prior, foot_w, centre, above_hip)
    if mirror is not None:
        second = solve(np.concatenate([first.x[:4], mirror]))
        similar = second.cost <= first.cost * (1.0 + MIRROR_COST_TOLERANCE) + 0.05
        if (similar and tilt_of(second.x) < tilt_of(first.x) - 1e-6) or (
            second.cost < first.cost * (1.0 - MIRROR_COST_TOLERANCE) - 0.05
        ):
            solution = second
    k, hip_height, ox, oy, tx, ty = (float(v) for v in solution.x)
    axis = np.array([tx, ty, -math.sqrt(max(1.0 - tx * tx - ty * ty, 1e-12))])
    rms = float(np.sqrt(np.mean(np.sum((model_uv(solution.x) - observed) ** 2, axis=1))))
    return _BodyFit(k, hip_height, np.array([ox, oy]), axis, rms, len(observations))


def lift_height_prior(
    keypoints: Any,
    camera: Camera | None,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    airborne: bool = False,
) -> LiftResult:
    """Lift one detection's Halpe-26 keypoints to the height-prior skeleton (module doc).

    Args:
        keypoints: (26, 3) [u, v, score] in the pixel grid of ``camera`` (u, v may be NaN).
        camera: the frame's calibration when it is pitch-usable (fresh, rep_err_px <= 8);
            None means no usable calibration and only image-plane angles are computed.
        score_threshold: keypoints scoring below it are ignored.
        airborne: the caller knows no foot touches the grass; nothing is lifted.

    Returns a LiftResult; a detection that cannot be lifted comes back with method
    "image_plane_only", a reason and NaN 3D values. Raises LiftInputError for malformed
    input (shape, types, a non-Camera camera).
    """
    problems: list[str] = []
    array = _checked_keypoints(keypoints, problems)
    threshold = _checked_threshold(score_threshold, problems)
    if camera is not None and not isinstance(camera, Camera):
        problems.append(f"camera: expected gbf.geometry.Camera or None, got {camera!r}")
    if not isinstance(airborne, bool | np.bool_):
        problems.append(f"airborne: expected a bool, got {airborne!r}")
    if problems or array is None or threshold is None:
        raise LiftInputError("lift_height_prior", problems)
    image = image_plane_angles(array, score_threshold=threshold)
    if camera is None:
        return _not_lifted("no_calibration", image)
    if airborne:
        return _not_lifted("airborne", image)
    usable = _usable(array, threshold)
    feet = _foot_indices(camera, array, usable)
    if not feet:
        return _not_lifted("feet_low_confidence", image)
    low, high = STATURE_RANGE_M
    for stature in (low, high):
        heights = -_PRIOR[feet] * stature / PRIOR_STATURE_M
        _, valid = pixel_to_ground(camera, array[feet, :2], heights)
        if not valid.all():
            return _not_lifted("foot_ray_miss", image)
    fit = _fit_body(camera, array, usable, feet)
    if fit is None:
        return _not_lifted("torso_low_confidence", image)
    k, axis, rms, n_points = fit.k, fit.axis, fit.rms_px, fit.n_points
    cos_tilt = -axis[2]
    hip_height = fit.hip_height
    heights = _PRIOR * k
    crouch = hip_height / (_HIP_PRIOR * k)  # bent knees lower hips and knees together
    for index in _KNEES:
        heights[index] *= crouch
    for index in _HIPS:
        heights[index] = hip_height
    for index in _TRUNK_JOINTS:
        heights[index] = hip_height + (_PRIOR[index] - _HIP_PRIOR) * k * cos_tilt
    shoulder_height = hip_height + (_SHOULDER_PRIOR - _HIP_PRIOR) * k * cos_tilt
    for index in _ARM_JOINTS:
        heights[index] = shoulder_height - (_SHOULDER_PRIOR - _PRIOR[index]) * k
    joints_sn = np.full((N_KEYPOINTS, 3), np.nan)
    if usable.any():
        points, valid = pixel_to_ground(camera, array[usable, :2], -heights[usable])
        joints_sn[np.flatnonzero(usable)[valid]] = points[valid]
    joints = soccernet_to_pitch_zup(joints_sn)
    camera_zup = soccernet_to_pitch_zup(camera.C)
    stature = k * PRIOR_STATURE_M
    facing = estimate_facing(
        joints,
        array[:, 2],
        camera_xy=camera_zup[:2],
        score_threshold=threshold,
        stature_m=stature,
    )
    lean, tilt = trunk_lean(joints, facing.facing_deg)
    lean_alt = mirrored_trunk_lean(joints, facing.facing_deg, camera_zup)
    sigma = facing_sigma_deg(camera, array, heights, score_threshold=threshold)
    lean_sigma = trunk_lean_sigma_deg(
        camera, array, heights, facing.facing_deg, score_threshold=threshold
    )
    at_bound = bool(min(abs(stature - low), abs(stature - high)) < 1e-6)
    return LiftResult(
        method="height_prior_calibrated",
        reason="lifted",
        joints3d=joints,
        stature_m=stature,
        stature_at_bound=at_bound,
        facing_deg=facing.facing_deg,
        facing_conf=facing.facing_conf,
        facing_sigma_deg=sigma if math.isfinite(facing.facing_deg) else math.nan,
        trunk_lean_deg=lean,
        trunk_tilt_deg=tilt,
        trunk_lean_alt_deg=lean_alt,
        trunk_lean_sigma_deg=lean_sigma,
        reproj_err_px=rms,
        n_fit_points=n_points,
        image=image,
    )


# ----------------------------------------------------------------------------- table

LIFT3D_COLUMNS: tuple[str, ...] = (
    "frame",
    "track_id",
    "method",
    "joints3d",
    "stature_m",
    "facing_deg",
    "facing_conf",
    "trunk_lean_deg",
    "image_lean_deg",
    "reproj_err_px",
    "reason",
    "facing_sigma_deg",
    "trunk_tilt_deg",
    "trunk_lean_alt_deg",
    "trunk_lean_sigma_deg",
    "image_knee_left_deg",
    "image_knee_right_deg",
)
"""``lift3d.parquet`` columns: the ten of docs/design/m1-highlights.md, then six extras
(``reason``, ``facing_sigma_deg``, ``trunk_tilt_deg``, ``trunk_lean_alt_deg``,
``trunk_lean_sigma_deg`` and the two image-plane knee angles)."""

LIFT3D_SCHEMA = pa.schema(
    [
        pa.field("frame", pa.int64(), nullable=False),
        pa.field("track_id", pa.int64(), nullable=False),
        pa.field("method", pa.string(), nullable=False),
        pa.field("joints3d", pa.list_(pa.float64()), nullable=False),
        pa.field("stature_m", pa.float64()),
        pa.field("facing_deg", pa.float64()),
        pa.field("facing_conf", pa.float64()),
        pa.field("trunk_lean_deg", pa.float64()),
        pa.field("image_lean_deg", pa.float64()),
        pa.field("reproj_err_px", pa.float64()),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("facing_sigma_deg", pa.float64()),
        pa.field("trunk_tilt_deg", pa.float64()),
        pa.field("trunk_lean_alt_deg", pa.float64()),
        pa.field("trunk_lean_sigma_deg", pa.float64()),
        pa.field("image_knee_left_deg", pa.float64()),
        pa.field("image_knee_right_deg", pa.float64()),
    ]
)
"""Arrow schema of ``lift3d.parquet``; ``joints3d`` holds 78 floats (26 x [x, y, z])."""


def lift_row(frame: int, track_id: int, result: LiftResult) -> dict[str, Any]:
    """One ``lift3d.parquet`` row from a LiftResult."""
    return {
        "frame": int(frame),
        "track_id": int(track_id),
        "method": result.method,
        "joints3d": result.joints3d.reshape(-1).tolist(),
        "stature_m": result.stature_m,
        "facing_deg": result.facing_deg,
        "facing_conf": result.facing_conf,
        "trunk_lean_deg": result.trunk_lean_deg,
        "image_lean_deg": result.image.trunk_lean_deg,
        "reproj_err_px": result.reproj_err_px,
        "reason": result.reason,
        "facing_sigma_deg": result.facing_sigma_deg,
        "trunk_tilt_deg": result.trunk_tilt_deg,
        "trunk_lean_alt_deg": result.trunk_lean_alt_deg,
        "trunk_lean_sigma_deg": result.trunk_lean_sigma_deg,
        "image_knee_left_deg": result.image.knee_left_deg,
        "image_knee_right_deg": result.image.knee_right_deg,
    }


def lift_pose_table(
    pose: pd.DataFrame,
    cameras: Mapping[int, Camera | None],
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    airborne: Mapping[tuple[int, int], bool] | None = None,
) -> pd.DataFrame:
    """Lift every row of a ``pose.parquet`` table; returns the ``lift3d.parquet`` table.

    ``cameras`` maps a frame to its pitch-usable Camera; frames that are missing or map to
    None get image-plane values only. ``airborne`` optionally flags (frame, track_id)
    pairs. Rows keep the pose table's order.
    """
    missing = [name for name in ("frame", "track_id", "keypoints") if name not in pose.columns]
    if missing:
        raise LiftInputError("lift_pose_table", [f"missing column(s): {', '.join(missing)}"])
    all_keypoints = keypoints_array(pose)
    flags = airborne or {}
    rows = []
    for position, (frame, track_id) in enumerate(
        zip(pose["frame"].tolist(), pose["track_id"].tolist(), strict=True)
    ):
        result = lift_height_prior(
            all_keypoints[position],
            cameras.get(int(frame)),
            score_threshold=score_threshold,
            airborne=bool(flags.get((int(frame), int(track_id)), False)),
        )
        rows.append(lift_row(frame, track_id, result))
    return lift_table(rows)


def lift_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """``lift3d.parquet`` rows as a DataFrame with :data:`LIFT3D_COLUMNS` and fixed dtypes."""
    table = pd.DataFrame(list(rows), columns=list(LIFT3D_COLUMNS))
    floats = [
        name
        for name in LIFT3D_COLUMNS
        if name not in {"frame", "track_id", "method", "joints3d", "reason"}
    ]
    table = table.astype(
        {"frame": "int64", "track_id": "int64", **dict.fromkeys(floats, "float64")}
    )
    table["method"] = table["method"].astype(str)
    table["reason"] = table["reason"].astype(str)
    return table.reset_index(drop=True)


def write_lift3d_parquet(table: pd.DataFrame, path: str | Path) -> None:
    """Write a lift table with :data:`LIFT3D_SCHEMA` (columns and list lengths checked)."""
    problems = [f"missing column {name}" for name in LIFT3D_COLUMNS if name not in table.columns]
    if not problems:
        bad = [
            row
            for row, item in enumerate(table["joints3d"])
            if len(np.asarray(item).reshape(-1)) != N_KEYPOINTS * 3
        ]
        problems += [f"row {row}: joints3d must hold {N_KEYPOINTS * 3} values" for row in bad]
        unknown = sorted(set(table["method"]) - set(LIFT_METHODS))
        problems += [f"unknown method {value!r}" for value in unknown]
    if problems:
        raise LiftInputError("write_lift3d_parquet", problems)
    arrow = pa.Table.from_pandas(
        table[list(LIFT3D_COLUMNS)], schema=LIFT3D_SCHEMA, preserve_index=False
    ).replace_schema_metadata(None)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(arrow, target)
