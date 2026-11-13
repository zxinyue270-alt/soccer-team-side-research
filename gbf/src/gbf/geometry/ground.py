"""Ray-plane intersection and anchoring of monocular 3D skeletons to the grass.

World frame: SoccerNet, metres, z pointing DOWN, grass at z = 0. A point h metres above the
grass has z = -h, so the physically lowest point of a body is the one with the LARGEST z.

Anchoring (plan section 3.3, step 10): a monocular 3D pose model gets the body shape right
but not its distance from the camera, so the camera-frame skeleton J_c is brought to the
world (J_w = R^T J_c + C) and then translated so that its grounded feet sit where their
pixel rays meet the grass. Only frames with a fresh calibration should be anchored; a held
or missing calibration gives wrong rays, so keep the camera-frame pose there instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from gbf.geometry._checks import (
    GeometryInputError,
    finite_float,
    is_bool,
    real_array,
    trailing_dim,
)
from gbf.geometry.camera import Camera, _check_camera, inside_image, pixel_rays

AnchorMethod = Literal["none_airborne", "single_foot", "feet_lstsq", "none_ray_miss"]
"""How anchor_to_ground placed the skeleton.

- ``none_airborne``: the caller marked the frame airborne; nothing was moved.
- ``single_foot``: one grounded foot; the skeleton is translated so that this foot lands
  exactly on its ground point.
- ``feet_lstsq``: two or more grounded feet; least-squares translation, with residual_m.
- ``none_ray_miss``: a grounded foot's pixel ray never meets its contact plane in front of
  the camera (e.g. a keypoint above the horizon); nothing was moved.
"""

ANCHOR_METHODS: tuple[AnchorMethod, ...] = (
    "none_airborne",
    "single_foot",
    "feet_lstsq",
    "none_ray_miss",
)

DEFAULT_PARALLEL_TOL = 1e-9
"""Rays whose angle to the plane has a sine at or below this are treated as parallel."""


def intersect_ground(
    C: ArrayLike,
    d: ArrayLike,
    z0: ArrayLike = 0.0,
    *,
    parallel_tol: float = DEFAULT_PARALLEL_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Where rays X(t) = C + t d (t > 0) meet the horizontal plane z = z0 (world metres).

    Args:
        C: ray origins, shape (..., 3), typically one camera centre of shape (3,).
        d: ray directions, shape (..., 3); they need not be unit length.
        z0: z value(s) of the plane in the z-down world frame, broadcastable against the
            rays: 0.0 is the grass and -2.44 the crossbar (2.44 m up), not +2.44.
        parallel_tol: a ray is parallel to the plane when |d_z| <= parallel_tol * |d|.

    Returns (points, valid): points (..., 3) with z exactly z0 where valid and NaN
    elsewhere; valid (...) bool. A ray is invalid when it is parallel to the plane, meets
    it behind or at its origin (t <= 0, e.g. a ray pointing up from a camera above the
    grass), or has a non-finite origin or direction. Malformed inputs raise
    GeometryInputError listing every problem.
    """
    problems: list[str] = []
    origins = trailing_dim("C", C, 3, problems)
    directions = trailing_dim("d", d, 3, problems)
    heights = real_array("z0", z0, problems)
    if heights is not None and not np.isfinite(heights).all():
        problems.append("z0: plane z values must be finite")
    tol = finite_float("parallel_tol", parallel_tol, problems)
    if tol is not None and tol < 0.0:
        problems.append(f"parallel_tol: must be >= 0, got {tol}")
    shape: tuple[int, ...] | None = None
    if origins is not None and directions is not None and heights is not None:
        try:
            shape = np.broadcast_shapes(origins.shape[:-1], directions.shape[:-1], heights.shape)
        except ValueError:
            problems.append(
                f"C {origins.shape}, d {directions.shape} and z0 {heights.shape}"
                " do not broadcast together"
            )
    if problems or shape is None or tol is None:
        raise GeometryInputError("intersect_ground", problems)

    origin = np.broadcast_to(origins, (*shape, 3))
    direction = np.broadcast_to(directions, (*shape, 3))
    plane = np.broadcast_to(heights, shape)
    d_z = direction[..., 2]
    usable = (
        np.isfinite(origin).all(axis=-1)
        & np.isfinite(direction).all(axis=-1)
        & (np.abs(d_z) > tol * np.linalg.norm(direction, axis=-1))
    )
    t = np.full(shape, np.nan)
    np.divide(plane - origin[..., 2], d_z, out=t, where=usable)
    valid = np.asarray(usable & (t > 0.0), dtype=bool)
    points = origin + t[..., None] * direction
    points[~valid] = np.nan
    points[..., 2] = np.where(valid, plane, np.nan)
    return points, valid


def pixel_to_ground(
    camera: Camera, uv: ArrayLike, z0: ArrayLike = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """World points (..., 3) where the pixel rays of uv (..., 2) meet the plane z = z0.

    The image-to-pitch mapping: intersect_ground(camera.C, pixel_rays(camera, uv), z0).
    Returns (points, valid) as intersect_ground does.
    """
    return intersect_ground(camera.C, pixel_rays(camera, uv), z0)


@dataclass(frozen=True, eq=False)
class AnchorResult:
    """Outcome of anchor_to_ground. Coordinates are SoccerNet world metres (z down).

    Attributes:
        joints: (J, 3) skeleton after anchoring; an unchanged copy of the input when nothing
            was anchored.
        translation: (3,) vector added to every joint, or None when nothing was anchored
            (never zero-filled, so "not anchored" cannot pass for "no correction needed").
        method: how the skeleton was placed, see AnchorMethod.
        residual_m: for "feet_lstsq", the RMS over grounded feet of the distance between a
            foot's ground point and the anchored foot, metres. None otherwise: a single foot
            fits exactly by construction, so its zero residual would be no evidence of fit.
        ground_points: (F, 3) where each grounded foot's pixel ray meets its contact plane,
            in the order of foot_joint_indices, NaN rows for rays that miss; None when the
            frame is airborne.
    """

    joints: np.ndarray
    translation: np.ndarray | None
    method: AnchorMethod
    residual_m: float | None
    ground_points: np.ndarray | None = None


def _checked_skeleton(name: str, value: Any, problems: list[str]) -> np.ndarray | None:
    joints = real_array(name, value, problems)
    if joints is None:
        return None
    if joints.ndim != 2 or joints.shape[1] != 3 or joints.shape[0] == 0:
        problems.append(f"{name}: expected shape (J, 3) with J >= 1, got {joints.shape}")
        return None
    return joints


def _checked_indices(
    name: str, value: Any, n_joints: int | None, problems: list[str], *, empty_hint: str
) -> np.ndarray | None:
    """1-D array of distinct joint indices in [0, n_joints), or None after noting problems."""
    try:
        indices = np.asarray(value)
    except (TypeError, ValueError) as exc:
        problems.append(f"{name}: cannot be read as integers ({exc})")
        return None
    if indices.size == 0:
        problems.append(f"{name}: empty; {empty_hint}")
        return None
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        problems.append(f"{name}: expected a 1-D sequence of integers, got {indices.tolist()!r}")
        return None
    found = len(problems)
    bound = "" if n_joints is None else f" < {n_joints}"
    for position, index in enumerate(indices.tolist()):
        if index < 0 or (n_joints is not None and index >= n_joints):
            problems.append(f"{name}[{position}] = {index}; must be 0 <= index{bound}")
    values, counts = np.unique(indices, return_counts=True)
    for index in values[counts > 1].tolist():
        problems.append(f"{name}: joint {index} is listed more than once")
    return indices.astype(np.intp) if len(problems) == found else None


def _checked_foot_pixels(
    value: Any,
    n_feet: int | None,
    camera: Camera | None,
    require_inside_image: bool,
    problems: list[str],
) -> np.ndarray | None:
    pixels = real_array("foot_uv", value, problems)
    if pixels is None:
        return None
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        problems.append(f"foot_uv: expected shape (F, 2), got {pixels.shape}")
        return None
    found = len(problems)
    if n_feet is not None and pixels.shape[0] != n_feet:
        problems.append(
            f"foot_uv: {pixels.shape[0]} row(s) for {n_feet} foot joint index(es); "
            "give one (u, v) per grounded foot"
        )
    finite = np.isfinite(pixels).all(axis=1)
    for row in np.flatnonzero(~finite).tolist():
        problems.append(f"foot_uv[{row}]: non-finite pixel coordinates")
    if camera is not None and require_inside_image:
        outside = finite & ~inside_image(camera, pixels)
        for row in np.flatnonzero(outside).tolist():
            u, v = pixels[row]
            problems.append(
                f"foot_uv[{row}] = ({u:.1f}, {v:.1f}) is outside the {camera.width}x"
                f"{camera.height} image; are the keypoints from another resolution?"
            )
    return pixels if len(problems) == found else None


def _checked_foot_heights(value: Any, n_feet: int | None, problems: list[str]) -> np.ndarray | None:
    heights = real_array("foot_height_m", value, problems)
    if heights is None:
        return None
    per_foot = heights.ndim == 1 and (n_feet is None or heights.size == n_feet)
    if heights.ndim != 0 and not per_foot:
        problems.append(
            "foot_height_m: expected a scalar or one value per grounded foot,"
            f" got shape {heights.shape}"
        )
        return None
    if not np.isfinite(heights).all():
        problems.append("foot_height_m: must be finite")
        return None
    if (heights < 0.0).any():
        problems.append(
            "foot_height_m: a height above the grass in metres, positive up even though"
            f" world z points down; must be >= 0, got {heights.tolist()!r}"
        )
        return None
    return heights


def anchor_to_ground(
    J_w: ArrayLike,
    foot_joint_indices: Sequence[int] | np.ndarray,
    foot_uv: ArrayLike,
    camera: Camera,
    airborne: bool,
    *,
    foot_height_m: ArrayLike = 0.0,
    require_inside_image: bool = True,
) -> AnchorResult:
    """Translate a world-frame skeleton so that its grounded feet stand on the grass.

    Args:
        J_w: (J, 3) skeleton in the SoccerNet world frame, metres (J_w = R^T J_c + C).
            Joints other than the grounded feet may be NaN; they stay NaN. The grounded
            foot joints must be finite.
        foot_joint_indices: indices into J_w of the feet touching the grass, preferably each
            foot's lowest point (largest z, see lowest_joint_index). Ignored when airborne.
        foot_uv: (F, 2) pixel coordinates of those joints in this camera's width x height
            grid, one finite row per index (u right, v down, pixel centres at integers).
            Ignored when airborne.
        camera: the frame's calibration, which must be fresh for this frame.
        airborne: True when no foot touches the grass; the skeleton is then returned
            unchanged with method "none_airborne".
        foot_height_m: height above the grass of each foot joint at contact, metres,
            positive up (scalar or one per foot). 0 for a sole or toe point; an ankle joint
            sits roughly 0.07-0.10 m up. Each ray is intersected with z = -foot_height_m.
        require_inside_image: reject foot pixels outside the image, which usually means the
            keypoints come from a different resolution than the camera.

    Each grounded foot's pixel ray meets its contact plane at a ground point g_i. With one
    foot, the translation moves that foot joint f onto g exactly ("single_foot"). With two
    or more, it is the least-squares translation t = mean_i(g_i - f_i), and residual_m is
    sqrt(mean_i |g_i - (f_i + t)|^2) ("feet_lstsq"). If any ray misses (for instance a
    keypoint above the horizon), nothing is moved and the method is "none_ray_miss".

    Reminder: z points down, so the physically lowest joint has the LARGEST z.

    Raises GeometryInputError listing every malformed input, including NaN in a grounded
    foot's joint or pixel (a foot the caller calls grounded must be measured).
    """
    problems: list[str] = []
    joints = _checked_skeleton("J_w", J_w, problems)
    _check_camera(camera, problems)
    if not is_bool(airborne):
        problems.append(f"airborne: expected a bool, got {airborne!r}")
    if not is_bool(require_inside_image):
        problems.append(f"require_inside_image: expected a bool, got {require_inside_image!r}")
    if is_bool(airborne) and bool(airborne):
        if problems or joints is None:
            raise GeometryInputError("anchor_to_ground", problems)
        return AnchorResult(
            joints=joints, translation=None, method="none_airborne", residual_m=None
        )

    n_joints = None if joints is None else joints.shape[0]
    indices = _checked_indices(
        "foot_joint_indices",
        foot_joint_indices,
        n_joints,
        problems,
        empty_hint="pass airborne=True when no foot touches the grass",
    )
    n_feet = None if indices is None else int(indices.size)
    pixels = _checked_foot_pixels(
        foot_uv,
        n_feet,
        camera if isinstance(camera, Camera) else None,
        bool(require_inside_image) if is_bool(require_inside_image) else True,
        problems,
    )
    heights = _checked_foot_heights(foot_height_m, n_feet, problems)
    if joints is not None and indices is not None:
        for position, index in enumerate(indices.tolist()):
            if not np.isfinite(joints[index]).all():
                problems.append(
                    f"J_w[{index}] (foot_joint_indices[{position}]): a grounded foot joint"
                    " must have finite coordinates"
                )
    if problems or joints is None or indices is None or pixels is None or heights is None:
        raise GeometryInputError("anchor_to_ground", problems)

    ground, hit = pixel_to_ground(camera, pixels, -heights)
    if not hit.all():
        return AnchorResult(
            joints=joints,
            translation=None,
            method="none_ray_miss",
            residual_m=None,
            ground_points=ground,
        )
    feet = joints[indices]
    translation = (ground - feet).mean(axis=0)
    anchored = joints + translation
    if indices.size == 1:
        return AnchorResult(
            joints=anchored,
            translation=translation,
            method="single_foot",
            residual_m=None,
            ground_points=ground,
        )
    residuals = ground - (feet + translation)
    rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return AnchorResult(
        joints=anchored,
        translation=translation,
        method="feet_lstsq",
        residual_m=rms,
        ground_points=ground,
    )


def lowest_joint_index(joints: ArrayLike, candidates: Sequence[int] | None = None) -> int:
    """Index of the physically lowest joint, i.e. the one with the LARGEST z (z points down).

    Args:
        joints: (J, 3) skeleton in the SoccerNet world frame, metres.
        candidates: joint indices to choose from (e.g. heels and toes); all joints if None.

    Joints with non-finite coordinates are skipped and ties go to the earliest candidate.
    Raises GeometryInputError for malformed input or when no candidate joint is finite.
    """
    problems: list[str] = []
    skeleton = _checked_skeleton("joints", joints, problems)
    n_joints = None if skeleton is None else skeleton.shape[0]
    pool: np.ndarray | None
    if candidates is None:
        pool = None if n_joints is None else np.arange(n_joints)
    else:
        pool = _checked_indices(
            "candidates", candidates, n_joints, problems, empty_hint="pass None for all joints"
        )
    if problems or skeleton is None or pool is None:
        raise GeometryInputError("lowest_joint_index", problems)
    usable = np.isfinite(skeleton[pool]).all(axis=1)
    if not usable.any():
        raise GeometryInputError(
            "lowest_joint_index", ["no candidate joint has finite coordinates"]
        )
    best = int(np.argmax(np.where(usable, skeleton[pool, 2], -np.inf)))
    return int(pool[best])
