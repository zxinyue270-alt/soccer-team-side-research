"""Camera geometry for broadcast football: frames, projection, pixel rays, ground anchoring.

Every function in this package follows the conventions below, and tests/test_geometry.py
pins each of them.

World frame (SoccerNet)
    Metres. Origin at the centre spot. x runs along the touchlines, x in [-52.5, 52.5].
    y runs across the pitch, y in [-34, 34], with the FAR touchline at y = -34 and the near
    (main camera) side at y = +34. z points DOWN: the pitch is the plane z = 0, a point
    above the grass has z < 0 (the crossbar is at z = -2.44) and the main camera has
    z of about -17. The frame is right-handed (x cross y = z). Because z points down, the
    physically lowest point of a body is the one with the LARGEST z.

pitch_zup (internal analysis frame)
    pitch_zup = (x, -y, -z): a 180 degree rotation about x, matrix diag(1, -1, -1) with
    determinant +1, so nothing is mirrored. z points up and the far touchline is at
    y = +34. The rotation is its own inverse; points and directions convert the same way.

Camera
    P = K R [I | -C]. R (3x3, det +1) rotates world vectors into the camera frame and C is
    the camera centre in world coordinates, so X_c = R (X_w - C) and J_w = R^T J_c + C.
    Camera axes: x to the right of the image, y down the image, z forward along the
    optical axis. "depth" is the camera-frame z in metres (distance along the optical axis,
    not the Euclidean distance); points at or behind the camera (depth <= 0) do not
    project. K is upper-triangular with K[2,2] = 1 and positive focal lengths in pixels;
    skew is allowed. Pinhole only: no lens distortion (undistort pixels first).

Pixels
    (u, v) with u to the right and v down, in the pixel grid of the frame size the camera
    was calibrated for (Camera.width x Camera.height). Pixel centres sit at integer
    coordinates: pixel (0, 0) covers [-0.5, 0.5] x [-0.5, 0.5], the image covers
    [-0.5, width - 0.5] x [-0.5, height - 0.5] and its centre is
    ((width - 1) / 2, (height - 1) / 2). Resizing a frame by s = dst / src (per axis) moves
    every coordinate as u' = (u + 0.5) * s - 0.5; scale_intrinsics changes K so that
    projecting with the new K gives exactly that. Always use the K of the frames' actual
    resolution.

Arrays and errors
    numpy float64 throughout. Points have shape (..., 3) and pixels (..., 2); functions are
    vectorised over the leading dimensions. Malformed input (wrong shape or type, an
    invalid camera) raises GeometryInputError, a ValueError whose ``problems`` lists every
    problem found. Missing data inside a well-formed array (NaN) is not an error: it comes
    back as NaN with valid = False, except where a docstring says otherwise.
"""

from gbf.geometry._checks import GeometryInputError
from gbf.geometry.camera import (
    INTRINSICS_ZERO_TOL,
    REFERENCE_CENTRE_M,
    REFERENCE_FOCAL_PX,
    REFERENCE_SIZE,
    ROTATION_TOL,
    Camera,
    camera_to_world,
    inside_image,
    look_at,
    pixel_rays,
    project,
    scale_intrinsics,
    synthetic_broadcast_camera,
    world_to_camera,
)
from gbf.geometry.frames import (
    PITCH_ZUP_FROM_SOCCERNET,
    pitch_zup_to_soccernet,
    soccernet_to_pitch_zup,
)
from gbf.geometry.ground import (
    ANCHOR_METHODS,
    DEFAULT_PARALLEL_TOL,
    AnchorMethod,
    AnchorResult,
    anchor_to_ground,
    intersect_ground,
    lowest_joint_index,
    pixel_to_ground,
)

__all__ = [
    "ANCHOR_METHODS",
    "DEFAULT_PARALLEL_TOL",
    "INTRINSICS_ZERO_TOL",
    "PITCH_ZUP_FROM_SOCCERNET",
    "REFERENCE_CENTRE_M",
    "REFERENCE_FOCAL_PX",
    "REFERENCE_SIZE",
    "ROTATION_TOL",
    "AnchorMethod",
    "AnchorResult",
    "Camera",
    "GeometryInputError",
    "anchor_to_ground",
    "camera_to_world",
    "inside_image",
    "intersect_ground",
    "look_at",
    "lowest_joint_index",
    "pitch_zup_to_soccernet",
    "pixel_rays",
    "pixel_to_ground",
    "project",
    "scale_intrinsics",
    "soccernet_to_pitch_zup",
    "synthetic_broadcast_camera",
    "world_to_camera",
]
