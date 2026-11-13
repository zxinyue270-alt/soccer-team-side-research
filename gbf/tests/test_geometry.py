"""Tests for gbf.geometry: frames, projection, rays, intrinsics scaling and ground anchoring.

The P0 acceptance criterion is a synthetic round-trip error below 1 mm. Every test builds
its scene from numbers alone (no media), so the whole file runs in well under a second.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from gbf.geometry import (
    PITCH_ZUP_FROM_SOCCERNET,
    Camera,
    GeometryInputError,
    anchor_to_ground,
    camera_to_world,
    inside_image,
    intersect_ground,
    look_at,
    lowest_joint_index,
    pitch_zup_to_soccernet,
    pixel_rays,
    pixel_to_ground,
    project,
    scale_intrinsics,
    soccernet_to_pitch_zup,
    synthetic_broadcast_camera,
    world_to_camera,
)

ACCEPTANCE_M = 1e-3  # P0: geometry round trip below 1 mm

K_1080 = np.array([[3347.0, 0.0, 959.5], [0.0, 3347.0, 539.5], [0.0, 0.0, 1.0]])

PELVIS, HEAD, L_ANKLE, R_ANKLE, L_TOE, R_TOE = range(6)


def _skeleton(x: float, y: float, *, right_foot_lift_m: float = 0.0) -> np.ndarray:
    """Toy standing skeleton at pitch position (x, y): SoccerNet metres, z down."""
    local = np.array(
        [
            [0.0, 0.0, -0.95],  # pelvis
            [0.0, 0.0, -1.75],  # head: the highest joint, so the smallest z
            [-0.10, 0.0, -0.08],  # left ankle, 8 cm above the grass
            [0.10, 0.0, -0.08],  # right ankle
            [-0.12, -0.15, 0.0],  # left toe, on the grass
            [0.12, -0.15, 0.0],  # right toe, on the grass
        ]
    )
    local[[R_ANKLE, R_TOE], 2] -= right_foot_lift_m
    return local + np.array([x, y, 0.0])


def _pitch_grid(step_m: float) -> np.ndarray:
    """Points on the grass (z = 0) covering the whole pitch, shape (N, 3)."""
    xs = np.arange(-52.5, 52.5 + 1e-9, step_m)
    ys = np.arange(-34.0, 34.0 + 1e-9, step_m)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])


def _pinhole(K: np.ndarray, points_c: np.ndarray) -> np.ndarray:
    """Independent oracle: pixels of camera-frame points through a plain matrix product."""
    homogeneous = points_c @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def _hand_camera() -> Camera:
    """10 m above the centre spot looking straight down; R = I keeps the arithmetic by hand."""
    K = [[100.0, 5.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    return Camera(K=K, R=np.eye(3), C=[0.0, 0.0, -10.0], width=100, height=80)


def _skewed_camera() -> Camera:
    """A camera with skew and an off-centre principal point, to exercise all of K."""
    centre = (-30.0, 60.0, -12.0)
    K = [[1500.0, 2.5, 700.25], [0.0, 1480.0, 380.75], [0.0, 0.0, 1.0]]
    return Camera(K=K, R=look_at(centre, (5.0, -3.0, 0.0)), C=centre, width=1280, height=720)


# --- round trips: the P0 acceptance -------------------------------------------------------


@pytest.mark.parametrize(("width", "height"), [(1920, 1080), (1280, 720)])
def test_ground_points_round_trip_below_1mm(width: int, height: int) -> None:
    camera = synthetic_broadcast_camera(width, height)
    points = _pitch_grid(1.0)
    uv, _, valid = project(camera, points)
    seen = valid & inside_image(camera, uv)
    assert seen.sum() > 1000  # the main camera sees a large part of the pitch

    back, hit = intersect_ground(camera.C, pixel_rays(camera, uv[seen]))

    assert hit.all()
    error = np.linalg.norm(back - points[seen], axis=1)
    assert error.max() < ACCEPTANCE_M
    assert error.max() < 1e-8  # float64 throughout; a float32 step would show up here


def test_points_above_the_grass_round_trip_through_their_own_plane() -> None:
    camera = synthetic_broadcast_camera(1920, 1080)
    rng = np.random.default_rng(2)
    points = np.column_stack(
        [rng.uniform(-30, 30, 200), rng.uniform(-30, 20, 200), -rng.uniform(0.1, 2.5, 200)]
    )  # a ball or a head 0.1-2.5 m up: negative z
    uv, _, valid = project(camera, points)
    assert valid.all()

    back, hit = pixel_to_ground(camera, uv, z0=points[:, 2])

    assert hit.all()
    assert np.linalg.norm(back - points, axis=1).max() < 1e-8


def test_720p_round_trip_from_resized_1080p_pixels() -> None:
    hd = synthetic_broadcast_camera(1920, 1080)
    small = synthetic_broadcast_camera(1280, 720)
    points = _pitch_grid(1.0)
    uv_hd, _, valid = project(hd, points)
    seen = valid & inside_image(hd, uv_hd)
    # What resizing the frame from 1080p to 720p does to a pixel coordinate:
    uv_small = (uv_hd[seen] + 0.5) * (1280 / 1920) - 0.5

    back, hit = pixel_to_ground(small, uv_small)

    assert hit.all()
    assert np.linalg.norm(back - points[seen], axis=1).max() < ACCEPTANCE_M
    # The failure this guards against: 720p pixels read with the 1080p intrinsics.
    stale = Camera(K=hd.K, R=hd.R, C=hd.C, width=1280, height=720)
    wrong, wrong_hit = pixel_to_ground(stale, uv_small)
    assert np.linalg.norm(wrong[wrong_hit] - points[seen][wrong_hit], axis=1).max() > 1.0


# --- intrinsics scaling -------------------------------------------------------------------


def test_scale_intrinsics_follows_the_pixel_centre_convention() -> None:
    small = scale_intrinsics(K_1080, (1920, 1080), (1280, 720))
    focal = 3347.0 * 2.0 / 3.0
    expected = [[focal, 0.0, 639.5], [0.0, focal, 359.5], [0.0, 0.0, 1.0]]
    np.testing.assert_allclose(small, expected, rtol=0.0, atol=1e-12)

    K = np.array([[1500.0, 2.5, 700.25], [0.0, 1480.0, 380.75], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(5)
    points_c = rng.uniform([-30.0, -20.0, 5.0], [30.0, 20.0, 120.0], size=(300, 3))
    resizes = [
        ((1920, 1080), (1280, 720)),
        ((1920, 1080), (1440, 1080)),  # anamorphic: only x shrinks
        ((1280, 720), (3840, 2160)),
        ((1920, 1080), (854, 480)),
    ]
    for src, dst in resizes:
        scale = np.array([dst[0] / src[0], dst[1] / src[1]])
        scaled = scale_intrinsics(K, src, dst)
        np.testing.assert_allclose(
            _pinhole(scaled, points_c), (_pinhole(K, points_c) + 0.5) * scale - 0.5, atol=1e-9
        )
        np.testing.assert_allclose(scale_intrinsics(scaled, dst, src), K, atol=1e-9)


def test_camera_resized_matches_the_synthetic_720p_camera() -> None:
    hd = synthetic_broadcast_camera(1920, 1080)
    small = hd.resized(1280, 720)
    assert small.size == (1280, 720)
    np.testing.assert_allclose(small.K, synthetic_broadcast_camera(1280, 720).K, atol=1e-12)
    np.testing.assert_array_equal(small.R, hd.R)
    np.testing.assert_array_equal(small.C, hd.C)


# --- frames and projection ----------------------------------------------------------------


def test_world_and_camera_frames_are_inverse() -> None:
    camera = _skewed_camera()
    rng = np.random.default_rng(11)
    points = rng.uniform(-60.0, 60.0, size=(4, 50, 3))
    np.testing.assert_allclose(
        camera_to_world(camera, world_to_camera(camera, points)), points, atol=1e-10
    )
    # X_c = R (X_w - C): the centre is the camera origin and the look-at target lies on +z.
    np.testing.assert_allclose(world_to_camera(camera, camera.C), 0.0, atol=1e-12)
    target = np.array([5.0, -3.0, 0.0])
    distance = np.linalg.norm(target - camera.C)
    np.testing.assert_allclose(world_to_camera(camera, target), [0.0, 0.0, distance], atol=1e-9)
    # J_w = R^T J_c + C for a joint 8 m in front of the camera.
    joint_c = np.array([0.3, -1.2, 8.0])
    np.testing.assert_allclose(
        camera_to_world(camera, joint_c), camera.R.T @ joint_c + camera.C, atol=1e-12
    )


def test_project_matches_hand_arithmetic_and_the_projection_matrix() -> None:
    # Camera 10 m up looking down, R = I: (1, 2, 0) is (1, 2, 10) in the camera frame, so
    # u = (100 * 1 + 5 * 2) / 10 + 50 = 61 and v = 100 * 2 / 10 + 40 = 60.
    uv, depth, valid = project(_hand_camera(), [1.0, 2.0, 0.0])
    np.testing.assert_allclose(uv, [61.0, 60.0], atol=1e-12)
    assert float(depth) == pytest.approx(10.0)
    assert bool(valid)

    camera = _skewed_camera()
    rng = np.random.default_rng(13)
    points = np.column_stack(
        [rng.uniform(-40, 40, 100), rng.uniform(-30, 30, 100), rng.uniform(-3, 0, 100)]
    )
    uv, _, valid = project(camera, points)
    assert valid.all()
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ camera.projection_matrix().T
    np.testing.assert_allclose(uv, homogeneous[:, :2] / homogeneous[:, 2:], atol=1e-8)


def test_functions_keep_leading_dimensions() -> None:
    camera = synthetic_broadcast_camera(1280, 720)
    points = np.zeros((2, 5, 3))
    points[..., 0] = np.linspace(-20.0, 20.0, 5)
    points[1, :, 1] = -10.0
    uv, depth, valid = project(camera, points)
    assert (uv.shape, depth.shape, valid.shape) == ((2, 5, 2), (2, 5), (2, 5))
    one_uv, one_depth, one_valid = project(camera, points[1, 3])
    assert (one_uv.shape, one_depth.shape, one_valid.shape) == ((2,), (), ())
    np.testing.assert_allclose(one_uv, uv[1, 3])
    rays = pixel_rays(camera, uv)
    assert rays.shape == (2, 5, 3)
    back, hit = intersect_ground(camera.C, rays)
    assert hit.shape == (2, 5)
    np.testing.assert_allclose(back, points, atol=1e-9)


def test_pitch_zup_is_a_rotation_about_x() -> None:
    np.testing.assert_array_equal(soccernet_to_pitch_zup([1.0, 2.0, -3.0]), [1.0, -2.0, 3.0])
    np.testing.assert_array_equal(pitch_zup_to_soccernet([1.0, -2.0, 3.0]), [1.0, 2.0, -3.0])
    rng = np.random.default_rng(7)
    points = rng.uniform(-60.0, 60.0, size=(5, 4, 3))
    np.testing.assert_array_equal(pitch_zup_to_soccernet(soccernet_to_pitch_zup(points)), points)
    np.testing.assert_array_equal(
        soccernet_to_pitch_zup(points), points @ PITCH_ZUP_FROM_SOCCERNET.T
    )
    assert np.linalg.det(PITCH_ZUP_FROM_SOCCERNET) == pytest.approx(1.0)
    # No mirroring: the images of the axes are still a right-handed triad.
    e_x, e_y, e_z = soccernet_to_pitch_zup(np.eye(3))
    np.testing.assert_array_equal(np.cross(e_x, e_y), e_z)
    # Meaning: the camera is above the grass and the far touchline is at y = +34.
    camera = synthetic_broadcast_camera(1920, 1080)
    assert soccernet_to_pitch_zup(camera.C)[2] == pytest.approx(17.4)
    assert soccernet_to_pitch_zup([0.0, -34.0, 0.0])[1] == 34.0


def test_points_at_or_behind_the_camera_do_not_project() -> None:
    camera = _hand_camera()  # at z = -10 looking down +z
    points = [
        [1.0, 2.0, 0.0],  # in front, depth 10
        [1.0, 2.0, -20.0],  # behind, depth -10
        [3.0, 4.0, -10.0],  # in the camera's own plane, depth 0
        [0.0, 0.0, -10.0],  # the camera centre itself
        [np.nan, 0.0, 0.0],  # missing data
    ]
    uv, depth, valid = project(camera, points)
    np.testing.assert_array_equal(valid, [True, False, False, False, False])
    assert np.isfinite(uv[0]).all()
    assert np.isnan(uv[1:]).all()
    np.testing.assert_array_equal(depth[:4], [10.0, -10.0, 0.0, 0.0])
    assert np.isnan(depth[4])

    broadcast = synthetic_broadcast_camera(1920, 1080)
    behind = broadcast.C - 30.0 * broadcast.R[2]
    uv, depth, valid = project(broadcast, behind)
    assert not bool(valid)
    assert float(depth) == pytest.approx(-30.0)
    assert np.isnan(uv).all()


def test_intersect_ground_rejects_parallel_and_backward_rays() -> None:
    centre = np.array([0.0, 0.0, -10.0])  # 10 m above the grass
    directions = [
        [0.0, 0.0, 1.0],  # straight down: hits (0, 0, 0)
        [3.0, 4.0, 10.0],  # slanted down and not unit length: hits (3, 4, 0)
        [1.0, 0.0, 0.0],  # parallel to the grass
        [1.0, 0.0, 1e-12],  # parallel within parallel_tol
        [0.0, 0.0, -1.0],  # pointing up: would meet the plane behind the camera
        [0.0, 0.0, 0.0],  # no direction at all
        [np.nan, 0.0, 1.0],  # missing data
    ]
    points, valid = intersect_ground(centre, directions)
    np.testing.assert_array_equal(valid, [True, True, False, False, False, False, False])
    np.testing.assert_allclose(points[:2], [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], atol=1e-12)
    assert np.isnan(points[2:]).all()

    # Other planes: z0 is in the z-down frame, so the crossbar (2.44 m up) is z0 = -2.44.
    heights = [0.0, -2.44, -9.0]
    points, valid = intersect_ground(centre, [0.0, 0.0, 1.0], z0=heights)
    assert valid.all()
    np.testing.assert_allclose(points[:, 2], heights)
    # A ray that starts on the plane meets it at t = 0, which does not count.
    _, on_plane = intersect_ground([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    assert not bool(on_plane)


def test_pixel_rays_are_unit_vectors_through_their_pixels() -> None:
    camera = _skewed_camera()
    rng = np.random.default_rng(3)
    uv = rng.uniform([-0.5, -0.5], [camera.width - 0.5, camera.height - 0.5], size=(200, 2))
    rays = pixel_rays(camera, uv)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-12)
    homogeneous = np.column_stack([uv, np.ones(len(uv))])
    expected = (camera.R.T @ np.linalg.inv(camera.K) @ homogeneous.T).T
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(rays, expected, atol=1e-12)
    # A point down each ray is in front of the camera and projects back onto its pixel.
    back, depth, valid = project(camera, camera.C + 25.0 * rays)
    assert valid.all()
    assert (depth > 0.0).all()
    np.testing.assert_allclose(back, uv, atol=1e-8)


def test_inside_image_uses_pixel_edges() -> None:
    camera = synthetic_broadcast_camera(1280, 720)
    uv = [
        [-0.5, -0.5],  # top-left corner of pixel (0, 0)
        [1279.49, 719.49],  # inside the last pixel
        [1279.5, 0.0],  # right edge: belongs to a pixel that does not exist
        [0.0, 719.5],  # bottom edge
        [-0.51, 0.0],
        [np.nan, 0.0],
        [639.5, 359.5],  # image centre
    ]
    np.testing.assert_array_equal(
        inside_image(camera, uv), [True, True, False, False, False, False, True]
    )


# --- look_at and the synthetic camera -----------------------------------------------------


@pytest.mark.parametrize(
    ("centre", "target"),
    [
        ((0.25, 70.3, -17.4), (0.0, 0.0, 0.0)),
        ((-60.0, -50.0, -30.0), (20.0, 10.0, -1.0)),
        ((40.0, 45.0, -8.0), (40.0, 0.0, 0.0)),
    ],
)
def test_look_at_points_the_optical_axis_at_the_target(
    centre: tuple[float, float, float], target: tuple[float, float, float]
) -> None:
    R = look_at(centre, target)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
    direction = np.subtract(target, centre)
    np.testing.assert_allclose(R[2], direction / np.linalg.norm(direction), atol=1e-12)
    assert abs(R[0, 2]) < 1e-12  # zero roll: image x is horizontal
    assert R[1, 2] > 0.0  # image "down" leans towards world down (+z)

    camera = Camera(K=K_1080, R=R, C=centre, width=1920, height=1080)
    uv, _, valid = project(camera, target)
    assert bool(valid)
    np.testing.assert_allclose(uv, [959.5, 539.5], atol=1e-9)
    above, _, _ = project(camera, np.add(target, [0.0, 0.0, -1.0]))  # 1 m higher
    assert above[1] < 539.5


def test_look_at_rejects_degenerate_geometry() -> None:
    with pytest.raises(GeometryInputError, match="parallel to world_up"):
        look_at((0.0, 0.0, -10.0), (0.0, 0.0, 0.0))  # straight down: roll undefined
    with pytest.raises(GeometryInputError) as caught:
        look_at((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), world_up=(0.0, 0.0, 0.0))
    assert len(caught.value.problems) == 2  # target == C, and a zero world_up
    # With another world_up the straight-down camera is fine.
    R = look_at((0.0, 0.0, -10.0), (0.0, 0.0, 0.0), world_up=(0.0, -1.0, 0.0))
    np.testing.assert_allclose(R[2], [0.0, 0.0, 1.0], atol=1e-12)


def test_synthetic_broadcast_camera_matches_a_real_main_camera() -> None:
    camera = synthetic_broadcast_camera(1920, 1080)
    assert (camera.K[0, 0], camera.K[1, 1]) == (3347.0, 3347.0)
    np.testing.assert_allclose(camera.K[:2, 2], [959.5, 539.5])
    assert camera.C[1] - 34.0 == pytest.approx(36.3, abs=0.5)  # outside the near touchline
    assert -camera.C[2] == pytest.approx(17.4, abs=0.5)  # height above the grass
    centre_spot, _, _ = project(camera, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(centre_spot, [959.5, 539.5], atol=1e-9)
    far, _, _ = project(camera, [0.0, -34.0, 0.0])
    near, _, _ = project(camera, [0.0, 34.0, 0.0])
    assert bool(inside_image(camera, far))
    assert near[1] > camera.height  # the near touchline is below the frame
    # z points down: a head 1.8 m above the grass shows above the feet; +x shows to the right.
    (feet, head), _, _ = project(camera, [[10.0, 5.0, 0.0], [10.0, 5.0, -1.8]])
    assert head[1] < feet[1] - 50.0
    right, _, _ = project(camera, [20.0, 0.0, 0.0])
    assert right[0] > 959.5 + 500.0

    small = synthetic_broadcast_camera(1280, 720)
    focal = 3347.0 * 2.0 / 3.0
    np.testing.assert_allclose(
        small.K, [[focal, 0.0, 639.5], [0.0, focal, 359.5], [0.0, 0.0, 1.0]], atol=1e-12
    )
    np.testing.assert_array_equal(small.R, camera.R)
    np.testing.assert_array_equal(small.C, camera.C)


# --- anchoring ----------------------------------------------------------------------------


@pytest.mark.parametrize("slide_m", [-6.0, -1.5, 2.5, 9.0])
def test_anchor_recovers_ground_contact_after_a_depth_error(slide_m: float) -> None:
    camera = synthetic_broadcast_camera(1920, 1080)
    truth = _skeleton(12.0, 6.0, right_foot_lift_m=0.3)  # only the left foot is down
    foot_uv, _, _ = project(camera, truth[[L_TOE]])
    ray = truth[PELVIS] - camera.C
    ray /= np.linalg.norm(ray)
    estimate = truth + slide_m * ray  # monocular error: right shape, wrong distance
    assert abs(estimate[L_TOE, 2]) > 0.3  # the error visibly lifts or sinks the foot

    result = anchor_to_ground(estimate, [L_TOE], foot_uv, camera, False)

    assert result.method == "single_foot"
    assert result.residual_m is None
    assert np.linalg.norm(result.joints - truth, axis=1).max() < ACCEPTANCE_M
    np.testing.assert_allclose(result.translation, -slide_m * ray, atol=1e-9)
    assert abs(result.joints[L_TOE, 2]) < 1e-9  # back on the grass
    np.testing.assert_allclose(result.ground_points, truth[[L_TOE]], atol=1e-9)


def test_anchor_two_grounded_feet_without_noise_is_exact() -> None:
    camera = synthetic_broadcast_camera(1280, 720)
    truth = _skeleton(-8.0, 15.0)
    feet = [L_TOE, R_TOE]
    foot_uv, _, _ = project(camera, truth[feet])
    ray = truth[PELVIS] - camera.C
    estimate = truth + 4.0 * ray / np.linalg.norm(ray)

    result = anchor_to_ground(estimate, feet, foot_uv, camera, False)

    assert result.method == "feet_lstsq"
    assert result.residual_m is not None
    assert result.residual_m < 1e-9
    assert np.linalg.norm(result.joints - truth, axis=1).max() < ACCEPTANCE_M


def test_anchor_two_feet_least_squares_has_the_known_residual() -> None:
    camera = synthetic_broadcast_camera(1920, 1080)
    truth = _skeleton(-20.0, -10.0)
    feet = [L_TOE, R_TOE]
    foot_uv, _, _ = project(camera, truth[feet])
    offset = np.array([0.4, -1.2, 0.3])  # the placement error anchoring must undo
    spread = np.array([0.03, -0.04, 0.0])  # the feet are also 5 cm off, in opposite ways
    estimate = truth - offset
    estimate[L_TOE] += spread
    estimate[R_TOE] -= spread

    result = anchor_to_ground(estimate, feet, foot_uv, camera, False)

    # t = mean(g_i - f_i) = offset; residuals are -spread and +spread, so the RMS is |spread|.
    assert result.method == "feet_lstsq"
    np.testing.assert_allclose(result.translation, offset, atol=1e-9)
    assert result.residual_m == pytest.approx(0.05, abs=1e-9)
    body = [PELVIS, HEAD, L_ANKLE, R_ANKLE]
    np.testing.assert_allclose(result.joints[body], truth[body], atol=1e-9)
    np.testing.assert_allclose(result.ground_points, truth[feet], atol=1e-9)


def test_anchor_residual_is_the_rms_over_contacts() -> None:
    # With two feet the residuals are always equal and opposite, so RMS, mean and max of
    # their lengths coincide. Three contacts with unequal residuals tell them apart.
    camera = synthetic_broadcast_camera(1920, 1080)
    truth = _skeleton(3.0, -2.0)
    contacts = [L_TOE, R_TOE, L_ANKLE]
    heights = [0.0, 0.0, 0.08]
    contact_uv, _, _ = project(camera, truth[contacts])
    offset = np.array([-0.7, 2.1, 0.2])
    spread = np.array([[0.06, 0.0, 0.0], [-0.03, 0.03, 0.0], [-0.03, -0.03, 0.0]])  # sums to 0
    estimate = truth - offset
    estimate[contacts] += spread

    result = anchor_to_ground(estimate, contacts, contact_uv, camera, False, foot_height_m=heights)

    np.testing.assert_allclose(result.translation, offset, atol=1e-9)
    expected_rms = np.sqrt((0.06**2 + 2 * 0.03**2 + 2 * 0.03**2) / 3)  # 0.04899 m
    assert result.residual_m == pytest.approx(expected_rms, abs=1e-9)
    assert abs(expected_rms - np.linalg.norm(spread, axis=1).mean()) > 5e-4


def test_anchor_foot_height_sets_the_contact_plane() -> None:
    camera = synthetic_broadcast_camera(1920, 1080)
    truth = _skeleton(5.0, -12.0)
    ankles = [L_ANKLE, R_ANKLE]  # 0.08 m above the grass at contact
    ankle_uv, _, _ = project(camera, truth[ankles])
    estimate = truth + np.array([0.5, 2.0, -0.4])

    right = anchor_to_ground(estimate, ankles, ankle_uv, camera, False, foot_height_m=0.08)
    np.testing.assert_allclose(right.joints, truth, atol=1e-9)
    # Treating the ankles as if they touched the grass misplaces the player along the ray.
    wrong = anchor_to_ground(estimate, ankles, ankle_uv, camera, False)
    assert np.linalg.norm(wrong.joints - truth, axis=1).min() > 0.05

    # Per-foot heights: a toe on the grass and an ankle 8 cm up.
    mixed = [L_TOE, R_ANKLE]
    mixed_uv, _, _ = project(camera, truth[mixed])
    result = anchor_to_ground(estimate, mixed, mixed_uv, camera, False, foot_height_m=[0.0, 0.08])
    np.testing.assert_allclose(result.joints, truth, atol=1e-9)
    assert result.residual_m is not None
    assert result.residual_m < 1e-9


def test_anchor_leaves_an_airborne_frame_unchanged() -> None:
    camera = synthetic_broadcast_camera(1920, 1080)
    jump = _skeleton(0.0, 0.0) + np.array([0.0, 0.0, -0.4])  # 0.4 m in the air
    feet_uv, _, _ = project(camera, jump[[L_TOE, R_TOE]])
    for indices, uv in (([], np.empty((0, 2))), ([L_TOE, R_TOE], feet_uv)):
        result = anchor_to_ground(jump, indices, uv, camera, True)
        assert result.method == "none_airborne"
        np.testing.assert_array_equal(result.joints, jump)
        assert result.joints is not jump
        assert result.translation is None
        assert result.residual_m is None
        assert result.ground_points is None


def test_anchor_reports_a_foot_ray_that_misses_the_grass() -> None:
    centre = (0.0, 70.0, -17.4)
    level = Camera(  # looking level at the horizon: the top half of the image is sky
        K=K_1080, R=look_at(centre, (0.0, -500.0, -17.4)), C=centre, width=1920, height=1080
    )
    skeleton = _skeleton(0.0, 0.0)
    result = anchor_to_ground(skeleton, [L_TOE], [[959.5, 200.0]], level, False)
    assert result.method == "none_ray_miss"
    assert result.translation is None
    assert result.residual_m is None
    np.testing.assert_array_equal(result.joints, skeleton)
    assert result.ground_points is not None
    assert np.isnan(result.ground_points).all()


def test_lowest_joint_index_picks_the_largest_z() -> None:
    skeleton = _skeleton(0.0, 0.0, right_foot_lift_m=0.3)
    assert lowest_joint_index(skeleton) == L_TOE  # z = 0; the head has z = -1.75
    assert lowest_joint_index(skeleton, [HEAD, PELVIS]) == PELVIS
    skeleton[L_TOE] = np.nan
    assert lowest_joint_index(skeleton, [R_TOE, L_TOE, L_ANKLE]) == L_ANKLE


# --- validation ---------------------------------------------------------------------------


def test_camera_validation_lists_every_problem() -> None:
    with pytest.raises(GeometryInputError) as caught:
        Camera(
            K=[[-1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
            R=2.0 * np.eye(3),
            C=[np.nan, 0.0, 0.0],
            width=0,
            height=1.5,
        )
    expected = [
        "K[1,0]",
        "K[2,2]",
        "K[0,0] (fx)",
        "R: not orthonormal",
        "R: det = 8",
        "C: 1 non-finite",
        "width: must be > 0",
        "height: expected a positive integer",
    ]
    problems = caught.value.problems
    for fragment in expected:
        assert any(fragment in problem for problem in problems), (fragment, problems)
    assert len(problems) == len(expected)
    assert f"{len(expected)} problem(s)" in str(caught.value)
    assert isinstance(caught.value, ValueError)


def test_camera_rejects_a_mirror_but_accepts_float32_rounding() -> None:
    camera = synthetic_broadcast_camera(1920, 1080)
    with pytest.raises(GeometryInputError, match="a mirror, not a rotation"):
        Camera(K=K_1080, R=np.diag([1.0, 1.0, -1.0]), C=camera.C, width=1920, height=1080)
    stored = Camera(
        K=K_1080.astype(np.float32),
        R=camera.R.astype(np.float32),
        C=camera.C,
        width=np.int64(1920),
        height=1080,
    )
    assert stored.R.dtype == np.float64
    assert type(stored.width) is int


def test_camera_keeps_read_only_copies() -> None:
    K = K_1080.copy()
    camera = Camera(K=K, R=np.eye(3), C=[0.0, 0.0, -10.0], width=1920, height=1080)
    K[0, 0] = 1.0
    assert camera.K[0, 0] == 3347.0
    with pytest.raises(ValueError, match="read-only"):
        camera.K[0, 0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        camera.width = 3


def test_anchor_validation_lists_every_problem() -> None:
    camera = synthetic_broadcast_camera(1280, 720)
    skeleton = _skeleton(0.0, 0.0)
    with pytest.raises(GeometryInputError) as caught:
        anchor_to_ground(
            skeleton,
            [L_TOE, L_TOE, 9],
            [[100.0, 100.0], [1500.0, 100.0]],  # 1500 is outside a 1280-wide frame
            camera,
            False,
            foot_height_m=-0.1,
        )
    problems = caught.value.problems
    for fragment in ["[2] = 9", "joint 4 is listed more than once", "outside the 1280x720"]:
        assert any(fragment in problem for problem in problems), (fragment, problems)
    assert any("foot_height_m" in problem for problem in problems)
    assert len(problems) == 4

    with pytest.raises(GeometryInputError) as caught:
        anchor_to_ground(np.zeros((6, 2)), [4], [[1.0, 2.0], [3.0, 4.0]], "camera", "no")
    assert len(caught.value.problems) == 4  # J_w shape, camera, airborne, row count

    broken = skeleton.copy()
    broken[L_TOE] = np.nan
    with pytest.raises(GeometryInputError) as caught:
        anchor_to_ground(broken, [L_TOE], [[np.nan, 3.0]], camera, False, foot_height_m=[0, 0])
    assert len(caught.value.problems) == 3  # foot joint NaN, foot pixel NaN, height count

    with pytest.raises(GeometryInputError, match="pass airborne=True"):
        anchor_to_ground(skeleton, [], np.empty((0, 2)), camera, False)


@pytest.mark.parametrize(
    ("call", "count"),
    [
        pytest.param(lambda camera: project("camera", np.zeros((4, 2))), 2, id="project"),
        pytest.param(lambda camera: pixel_rays(camera, np.zeros((4, 3))), 1, id="pixel_rays"),
        pytest.param(
            lambda camera: intersect_ground(np.zeros(3), np.zeros((2, 3)), z0=[0.0, 1.0, 2.0]),
            1,
            id="intersect_broadcast",
        ),
        pytest.param(
            lambda camera: intersect_ground(
                np.zeros(2), [["a", "b", "c"]], z0=np.nan, parallel_tol=-1.0
            ),
            4,
            id="intersect_all",
        ),
        pytest.param(
            lambda camera: scale_intrinsics(np.eye(2), (1920, 0), (1280.0, 720, 1)),
            3,
            id="scale_intrinsics",
        ),
        pytest.param(lambda camera: soccernet_to_pitch_zup(np.zeros((3, 2))), 1, id="zup"),
        pytest.param(lambda camera: synthetic_broadcast_camera(0, True), 2, id="synthetic"),
        pytest.param(lambda camera: lowest_joint_index(np.full((2, 3), np.nan)), 1, id="lowest"),
    ],
)
def test_geometry_functions_report_every_input_problem(
    call: Callable[[Camera], object], count: int
) -> None:
    with pytest.raises(GeometryInputError) as caught:
        call(synthetic_broadcast_camera(1280, 720))
    assert len(caught.value.problems) == count, caught.value.problems
