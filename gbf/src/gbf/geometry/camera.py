"""Pinhole camera for broadcast frames: intrinsics, extrinsics, projection and pixel rays.

The conventions are spelled out in the package docstring (``gbf.geometry``) and pinned by
``tests/test_geometry.py``. In short: SoccerNet world frame in metres with z down;
P = K R [I | -C] with R rotating world to camera and C the camera centre in world
coordinates; camera axes x right, y down, z forward; pixel centres at integer (u, v), so
pixel (0, 0) covers [-0.5, 0.5] x [-0.5, 0.5]. Pinhole only: undistort pixel coordinates
before using them here if a calibration carries lens distortion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from gbf.geometry._checks import (
    GeometryInputError,
    fixed_shape,
    image_size,
    positive_int,
    trailing_dim,
)

ROTATION_TOL = 1e-6
"""Largest allowed entry of |R R^T - I|, and half the allowed |det R - 1|, for a rotation.

Loose enough for rotations stored as float32 (about 1e-7 error), tight enough to reject a
scaled, sheared or mirrored matrix.
"""

INTRINSICS_ZERO_TOL = 1e-9
"""Largest allowed magnitude of K[1, 0], K[2, 0], K[2, 1] and of K[2, 2] - 1."""

REFERENCE_SIZE: tuple[int, int] = (1920, 1080)
"""Frame size (width, height) in pixels for which the synthetic main camera is defined."""

REFERENCE_FOCAL_PX = 3347.0
"""Focal length of the synthetic main camera at 1920x1080, pixels (about 32 degrees of
horizontal field of view): the magnitude of a real broadcast main-camera calibration."""

REFERENCE_CENTRE_M: tuple[float, float, float] = (0.25, 70.3, -17.4)
"""Synthetic main-camera centre in the SoccerNet frame, metres: 36.3 m outside the near
touchline (y = +34) and 17.4 m above the grass (z = -17.4 because z points down)."""


def _checked_intrinsics(name: str, value: Any, problems: list[str]) -> np.ndarray | None:
    """Float64 copy of an intrinsics matrix, or None after recording every problem with it."""
    K = fixed_shape(name, value, (3, 3), problems)
    if K is None:
        return None
    found = len(problems)
    for i, j in ((1, 0), (2, 0), (2, 1)):
        if abs(K[i, j]) > INTRINSICS_ZERO_TOL:
            problems.append(f"{name}[{i},{j}] = {K[i, j]:.6g}; K must be upper-triangular")
    if abs(K[2, 2] - 1.0) > INTRINSICS_ZERO_TOL:
        problems.append(f"{name}[2,2] = {K[2, 2]:.6g}; K must be normalised to K[2,2] = 1")
    if not K[0, 0] > 0.0:
        problems.append(f"{name}[0,0] (fx) = {K[0, 0]:.6g}; the focal length must be > 0 px")
    if not K[1, 1] > 0.0:
        problems.append(f"{name}[1,1] (fy) = {K[1, 1]:.6g}; the focal length must be > 0 px")
    if len(problems) > found:
        return None
    K[1, 0] = K[2, 0] = K[2, 1] = 0.0
    K[2, 2] = 1.0
    return K


def _checked_rotation(name: str, value: Any, problems: list[str]) -> np.ndarray | None:
    """Float64 copy of a rotation matrix, or None after recording every problem with it."""
    R = fixed_shape(name, value, (3, 3), problems)
    if R is None:
        return None
    found = len(problems)
    orthogonality_error = float(np.max(np.abs(R @ R.T - np.eye(3))))
    if orthogonality_error > ROTATION_TOL:
        problems.append(
            f"{name}: not orthonormal (max |R R^T - I| = {orthogonality_error:.3g}"
            f" > {ROTATION_TOL:g})"
        )
    determinant = float(np.linalg.det(R))
    if abs(determinant - 1.0) > 2.0 * ROTATION_TOL:
        hint = "; a mirror, not a rotation" if determinant < 0.0 else ""
        problems.append(f"{name}: det = {determinant:.6g}, must be +1{hint}")
    return R if len(problems) == found else None


def _check_camera(camera: Any, problems: list[str]) -> None:
    if not isinstance(camera, Camera):
        problems.append(f"camera: expected gbf.geometry.Camera, got {type(camera).__name__}")


@dataclass(frozen=True, eq=False)
class Camera:
    """A calibrated pinhole camera for frames of one size.

    Attributes:
        K: (3, 3) intrinsics in pixels of the width x height grid: upper-triangular,
            K[2, 2] = 1, focal lengths K[0, 0] and K[1, 1] > 0, skew K[0, 1] allowed.
        R: (3, 3) rotation from the SoccerNet world frame to the camera frame (det +1).
        C: (3,) camera centre in the SoccerNet world frame, metres.
        width, height: size in pixels of the frames that K refers to.

    Construction validates every field, reports all problems in one GeometryInputError, and
    stores read-only float64 copies, so a Camera never changes after it is built. Equality
    is identity; compare the arrays explicitly.
    """

    K: np.ndarray
    R: np.ndarray
    C: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        problems: list[str] = []
        K = _checked_intrinsics("K", self.K, problems)
        R = _checked_rotation("R", self.R, problems)
        C = fixed_shape("C", self.C, (3,), problems)
        width = positive_int("width", self.width, problems)
        height = positive_int("height", self.height, problems)
        if problems or K is None or R is None or C is None or width is None or height is None:
            raise GeometryInputError("Camera", problems)
        for name, array in (("K", K), ("R", R), ("C", C)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return self.width, self.height

    def projection_matrix(self) -> np.ndarray:
        """P = K R [I | -C], shape (3, 4): homogeneous world points (metres) to pixels."""
        return self.K @ self.R @ np.hstack([np.eye(3), -self.C[:, None]])

    def resized(self, width: int, height: int) -> Camera:
        """The same camera for frames resized to width x height (K via scale_intrinsics)."""
        K = scale_intrinsics(self.K, self.size, (width, height))
        return Camera(K=K, R=self.R, C=self.C, width=width, height=height)


def scale_intrinsics(
    K: ArrayLike, src_size: tuple[int, int], dst_size: tuple[int, int]
) -> np.ndarray:
    """Intrinsics for frames resized from src_size to dst_size, both (width, height) in pixels.

    With s_x = dst_width / src_width and s_y = dst_height / src_height, resizing a frame maps
    its pixel edges -0.5 and src - 0.5 onto -0.5 and dst - 0.5, so every coordinate moves as
    u' = (u + 0.5) * s_x - 0.5 and v' = (v + 0.5) * s_y - 0.5. The result is
    K' = A K with A = [[s_x, 0, (s_x - 1) / 2], [0, s_y, (s_y - 1) / 2], [0, 0, 1]], so
    projecting with K' gives exactly those mapped coordinates: fx and the skew scale by s_x,
    fy by s_y, and the principal point moves as c' = (c + 0.5) * s - 0.5 (the image centre
    stays the image centre). Scaling the principal point as c * s instead would shift every
    projection by (1 - s) / 2 pixels.

    s_x and s_y may differ (an anamorphic resize such as 1920x1080 to 1440x1080); fx / fy
    then differs from the source ratio. Raises GeometryInputError listing every problem.
    """
    problems: list[str] = []
    intrinsics = _checked_intrinsics("K", K, problems)
    source = image_size("src_size", src_size, problems)
    target = image_size("dst_size", dst_size, problems)
    if problems or intrinsics is None or source is None or target is None:
        raise GeometryInputError("scale_intrinsics", problems)
    s_x = target[0] / source[0]
    s_y = target[1] / source[1]
    resize = np.array(
        [
            [s_x, 0.0, 0.5 * (s_x - 1.0)],
            [0.0, s_y, 0.5 * (s_y - 1.0)],
            [0.0, 0.0, 1.0],
        ]
    )
    return resize @ intrinsics


def world_to_camera(camera: Camera, points_w: ArrayLike) -> np.ndarray:
    """X_c = R (X_w - C): world points (..., 3) in metres to the camera frame.

    Camera frame: x to the image right, y down the image, z forward along the optical axis,
    metres. Non-finite entries propagate as NaN.
    """
    problems: list[str] = []
    _check_camera(camera, problems)
    points = trailing_dim("points_w", points_w, 3, problems)
    if problems or points is None:
        raise GeometryInputError("world_to_camera", problems)
    return (points - camera.C) @ camera.R.T


def camera_to_world(camera: Camera, points_c: ArrayLike) -> np.ndarray:
    """X_w = R^T X_c + C: camera-frame points (..., 3) in metres to the world frame.

    This is how a camera-frame skeleton J_c from a 3D pose model becomes J_w. It is the
    exact inverse of world_to_camera. Non-finite entries propagate as NaN.
    """
    problems: list[str] = []
    _check_camera(camera, problems)
    points = trailing_dim("points_c", points_c, 3, problems)
    if problems or points is None:
        raise GeometryInputError("camera_to_world", problems)
    return points @ camera.R + camera.C


def project(camera: Camera, points_w: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points (..., 3), metres, to pixels.

    Returns (uv, depth, valid):
        uv: (..., 2) pixel coordinates (u right, v down, pixel centres at integers);
            NaN where valid is False.
        depth: (...) camera-frame z in metres, the distance along the optical axis (not the
            Euclidean distance to C); negative behind the camera, NaN for non-finite input.
        valid: (...) bool, True where the point is finite and strictly in front of the
            camera (depth > 0). Points at or behind the camera are invalid.

    A valid point may still fall outside the image; use inside_image for that.
    """
    points_c = world_to_camera(camera, points_w)
    x, y, z = points_c[..., 0], points_c[..., 1], points_c[..., 2]
    valid = np.asarray(np.isfinite(points_c).all(axis=-1) & (z > 0.0), dtype=bool)
    safe_z = np.where(valid, z, 1.0)
    K = camera.K
    u = (K[0, 0] * x + K[0, 1] * y) / safe_z + K[0, 2]
    v = K[1, 1] * y / safe_z + K[1, 2]
    uv = np.stack([u, v], axis=-1)
    uv[~valid] = np.nan
    return uv, np.array(z, dtype=np.float64), valid


def pixel_rays(camera: Camera, uv: ArrayLike) -> np.ndarray:
    """Unit world directions (..., 3) of the rays from C through pixels uv (..., 2).

    d = normalise(R^T K^-1 [u, v, 1]^T). The ray is X(t) = C + t d with t > 0 in front of
    the camera. Non-finite pixel coordinates give NaN directions.
    """
    problems: list[str] = []
    _check_camera(camera, problems)
    pixels = trailing_dim("uv", uv, 2, problems)
    if problems or pixels is None:
        raise GeometryInputError("pixel_rays", problems)
    K = camera.K
    y_n = (pixels[..., 1] - K[1, 2]) / K[1, 1]
    x_n = (pixels[..., 0] - K[0, 2] - K[0, 1] * y_n) / K[0, 0]
    directions_c = np.stack([x_n, y_n, np.ones_like(x_n)], axis=-1)
    directions_w = directions_c @ camera.R
    return directions_w / np.linalg.norm(directions_w, axis=-1, keepdims=True)


def inside_image(camera: Camera, uv: ArrayLike) -> np.ndarray:
    """True where (u, v) lies on the image: -0.5 <= u < width - 0.5, -0.5 <= v < height - 0.5.

    Pixel (i, j) covers [i - 0.5, i + 0.5) x [j - 0.5, j + 0.5), so these half-open bounds
    give every image point exactly one pixel. Non-finite coordinates are outside. Returns
    a bool array of shape (...) for uv of shape (..., 2).
    """
    problems: list[str] = []
    _check_camera(camera, problems)
    pixels = trailing_dim("uv", uv, 2, problems)
    if problems or pixels is None:
        raise GeometryInputError("inside_image", problems)
    u, v = pixels[..., 0], pixels[..., 1]
    return np.asarray(
        (u >= -0.5) & (u < camera.width - 0.5) & (v >= -0.5) & (v < camera.height - 0.5),
        dtype=bool,
    )


def look_at(C: ArrayLike, target: ArrayLike, world_up: ArrayLike = (0.0, 0.0, -1.0)) -> np.ndarray:
    """Rotation R (world to camera) of a camera at C whose optical axis passes through target.

    Zero roll: the camera x axis (image right) is horizontal, i.e. perpendicular to
    world_up, and world_up points up the image (towards smaller v). The default world_up
    (0, 0, -1) is "up" in the SoccerNet frame, whose z points down. The rows of R are the
    camera axes in world coordinates: z = unit(target - C), x = unit(z cross world_up) and
    y = z cross x, so det R = +1.

    Raises GeometryInputError listing every problem, including target == C, a zero
    world_up, and a viewing direction parallel to world_up (the roll is then undefined, as
    for a camera looking straight down; pass another world_up).
    """
    problems: list[str] = []
    centre = fixed_shape("C", C, (3,), problems)
    aim = fixed_shape("target", target, (3,), problems)
    up = fixed_shape("world_up", world_up, (3,), problems)
    forward = None
    if centre is not None and aim is not None:
        offset = aim - centre
        length = float(np.linalg.norm(offset))
        if length == 0.0:
            problems.append("target: coincides with C, so the viewing direction is undefined")
        else:
            forward = offset / length
    up_unit = None
    if up is not None:
        up_length = float(np.linalg.norm(up))
        if up_length == 0.0:
            problems.append("world_up: must be a non-zero vector")
        else:
            up_unit = up / up_length
    right = None
    if forward is not None and up_unit is not None:
        cross = np.cross(forward, up_unit)
        cross_length = float(np.linalg.norm(cross))
        if cross_length < 1e-9:
            problems.append(
                "target - C is parallel to world_up, so the roll is undefined; "
                "pass a world_up that is not along the viewing direction"
            )
        else:
            right = cross / cross_length
    if problems or forward is None or right is None:
        raise GeometryInputError("look_at", problems)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def synthetic_broadcast_camera(
    width: int, height: int, *, target: ArrayLike = (0.0, 0.0, 0.0)
) -> Camera:
    """A realistic main broadcast camera for synthetic tests and the L0 smoke run.

    At 1920x1080: focal length 3347 px, square pixels, zero skew, principal point at the
    image centre (959.5, 539.5); centre C = (0.25, 70.3, -17.4) m, i.e. 36.3 m outside
    the near touchline and 17.4 m above the grass; zero roll; looking at ``target`` (world
    metres, default the centre spot). At that zoom the far touchline is in view and the
    near touchline is not.

    Any other size is the same camera with K rescaled by scale_intrinsics from 1920x1080,
    so a world point lands at the same relative image position. A size that is not 16:9
    gives non-square pixels, as an anamorphic resize would.
    """
    problems: list[str] = []
    positive_int("width", width, problems)
    positive_int("height", height, problems)
    aim = fixed_shape("target", target, (3,), problems)
    if problems or aim is None:
        raise GeometryInputError("synthetic_broadcast_camera", problems)
    ref_width, ref_height = REFERENCE_SIZE
    reference_K = np.array(
        [
            [REFERENCE_FOCAL_PX, 0.0, (ref_width - 1) / 2.0],
            [0.0, REFERENCE_FOCAL_PX, (ref_height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    centre = np.array(REFERENCE_CENTRE_M)
    return Camera(
        K=scale_intrinsics(reference_K, REFERENCE_SIZE, (width, height)),
        R=look_at(centre, aim),
        C=centre,
        width=width,
        height=height,
    )
