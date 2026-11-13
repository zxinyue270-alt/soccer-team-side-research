"""Tests for gbf.lift3d: the calibrated height-prior skeleton, facing, trunk lean, image angles.

Synthetic players with a known stature, facing and trunk lean are built from a body template
written for this file (heights from the M1 task statement, horizontal offsets chosen here),
projected through synthetic cameras, and lifted back. No media and no model are involved.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow.parquet as pq
import pytest

from gbf import lift3d
from gbf.geometry import (
    Camera,
    look_at,
    pitch_zup_to_soccernet,
    project,
    synthetic_broadcast_camera,
)
from gbf.lift3d import (
    JOINT_HEIGHT_PRIOR_M,
    LIFT3D_COLUMNS,
    LIFT3D_SCHEMA,
    STATURE_RANGE_M,
    LiftInputError,
    estimate_facing,
    image_plane_angles,
    lift_height_prior,
    lift_pose_table,
    trunk_lean,
    write_lift3d_parquet,
)
from gbf.pose import HALPE26_FLIP_PAIRS, HALPE26_INDEX, HALPE26_KEYPOINTS, pose_table

STATURE_TOL_M = 0.02
FACING_TOL_DEG = 3.0
LEAN_TOL_DEG = 3.0

# (forward, left, height) in metres for a 1.80 m upright player facing "forward".
TEMPLATE: dict[str, tuple[float, float, float]] = {
    "nose": (0.10, 0.0, 1.65),
    "left_eye": (0.08, 0.035, 1.65),
    "right_eye": (0.08, -0.035, 1.65),
    "left_ear": (0.0, 0.075, 1.65),
    "right_ear": (0.0, -0.075, 1.65),
    "left_shoulder": (0.0, 0.19, 1.45),
    "right_shoulder": (0.0, -0.19, 1.45),
    "left_elbow": (0.0, 0.22, 1.10),
    "right_elbow": (0.0, -0.22, 1.10),
    "left_wrist": (0.03, 0.23, 0.85),
    "right_wrist": (0.03, -0.23, 0.85),
    "left_hip": (0.0, 0.10, 0.95),
    "right_hip": (0.0, -0.10, 0.95),
    "left_knee": (0.02, 0.10, 0.50),
    "right_knee": (0.02, -0.10, 0.50),
    "left_ankle": (0.0, 0.10, 0.09),
    "right_ankle": (0.0, -0.10, 0.09),
    "head": (0.0, 0.0, 1.80),
    "neck": (0.0, 0.0, 1.55),
    "hip": (0.0, 0.0, 0.95),
    "left_big_toe": (0.17, 0.08, 0.03),
    "right_big_toe": (0.17, -0.08, 0.03),
    "left_small_toe": (0.15, 0.13, 0.03),
    "right_small_toe": (0.15, -0.13, 0.03),
    "left_heel": (-0.05, 0.10, 0.05),
    "right_heel": (-0.05, -0.10, 0.05),
}
# Joints that rotate with the trunk when the player leans (about the hip centre).
UPPER_BODY = {
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "head",
    "neck",
}
# Joints whose height the lifter models exactly for this template (legs, feet, trunk axis,
# shoulders, ears); the face points and the arms are only approximately placed.
EXACT_JOINTS = [
    HALPE26_INDEX[name]
    for name in HALPE26_KEYPOINTS
    if name not in {"nose", "left_eye", "right_eye"} and "elbow" not in name and "wrist" not in name
]
KP = HALPE26_INDEX


def player_zup(
    stature: float, facing_deg: float, lean_deg: float, x: float, y: float
) -> np.ndarray:
    """(26, 3) pitch_zup joints of a template player standing at (x, y)."""
    scale = stature / 1.80
    lean = math.radians(lean_deg)
    heading = math.radians(facing_deg)
    forward = np.array([math.cos(heading), math.sin(heading), 0.0])
    left = np.array([-math.sin(heading), math.cos(heading), 0.0])
    hip_height = 0.95 * scale
    joints = np.zeros((26, 3))
    for name, (f, l_, h) in TEMPLATE.items():
        f, l_, h = f * scale, l_ * scale, h * scale
        if name in UPPER_BODY:
            above = h - hip_height
            f, h = (
                f * math.cos(lean) + above * math.sin(lean),
                hip_height + above * math.cos(lean) - f * math.sin(lean),
            )
        joints[KP[name]] = np.array([x, y, 0.0]) + f * forward + l_ * left + np.array([0, 0, h])
    return joints


def observe(camera: Camera, joints_zup: np.ndarray, score: float = 0.9) -> np.ndarray:
    """Exact (26, 3) [u, v, score] keypoints of pitch_zup joints."""
    uv, _, valid = project(camera, pitch_zup_to_soccernet(joints_zup))
    assert valid.all()
    return np.column_stack([uv, np.full(26, score)])


def broadcast_camera(x: float, y: float) -> Camera:
    """The synthetic main camera, aimed at a player at pitch_zup (x, y)."""
    return synthetic_broadcast_camera(1920, 1080, target=(x, -y, 0.0))


def custom_camera(centre_zup: tuple[float, float, float], target_zup, focal: float) -> Camera:
    centre = pitch_zup_to_soccernet(np.array(centre_zup))
    target = pitch_zup_to_soccernet(np.array(target_zup))
    K = np.array([[focal, 0.0, 959.5], [0.0, focal, 539.5], [0.0, 0.0, 1.0]])
    return Camera(K=K, R=look_at(centre, target), C=centre, width=1920, height=1080)


def camera_for(name: str, x: float, y: float) -> Camera:
    if name == "broadcast":
        return broadcast_camera(x, y)
    if name == "behind_goal":  # high behind the right goal, looking back up the pitch
        return custom_camera((62.0, 0.0, 12.0), (x, y, 1.0), 2600.0)
    if name == "low_side":  # low on the far side, close to the play
        return custom_camera((x + 6.0, 42.0, 6.0), (x, y, 1.0), 1900.0)
    raise AssertionError(name)


def angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def facing_towards(camera: Camera, x: float, y: float) -> float:
    """Facing (pitch_zup degrees) of a player at (x, y) looking straight at the camera."""
    centre = pitch_zup_to_soccernet(camera.C)
    return math.degrees(math.atan2(centre[1] - y, centre[0] - x))


def with_face_visibility(
    keypoints: np.ndarray, camera: Camera, facing: float, x: float, y: float
) -> np.ndarray:
    """Scores a pose model would give the face: low when the player turns away (> 110 deg)."""
    scored = keypoints.copy()
    if angle_diff(facing, facing_towards(camera, x, y)) > 110.0:
        scored[[KP["nose"], KP["left_eye"], KP["right_eye"]], 2] = 0.1
    return scored


# Every case is one the lift can resolve: the lean is side-on to the camera, towards it,
# or away from it by less than the camera's depression angle (see the ambiguity test).
CASES = [
    # camera, stature m, facing deg, lean deg, x, y
    ("broadcast", 1.80, 0.0, 0.0, 0.0, 0.0),
    ("broadcast", 1.62, 0.0, 12.0, -20.0, 10.0),  # side-on, leaning forward
    ("broadcast", 1.98, -135.0, -8.0, 30.0, -20.0),
    ("broadcast", 1.75, 180.0, 20.0, 10.0, 25.0),
    ("broadcast", 1.80, -90.0, 18.0, 0.0, 0.0),  # faces the camera, leans towards it
    ("broadcast", 1.80, 90.0, -10.0, 0.0, 0.0),  # faces away, leans back towards it
    ("behind_goal", 1.85, 45.0, 15.0, 35.0, -5.0),
    ("behind_goal", 1.68, -170.0, 6.0, 20.0, 12.0),
    ("low_side", 1.70, 30.0, 10.0, -5.0, -15.0),
    ("low_side", 1.92, 120.0, -12.0, 15.0, 20.0),
]


@pytest.mark.parametrize(("camera_name", "stature", "facing", "lean", "x", "y"), CASES)
def test_lift_recovers_stature_facing_and_lean(
    camera_name: str, stature: float, facing: float, lean: float, x: float, y: float
) -> None:
    camera = camera_for(camera_name, x, y)
    truth = player_zup(stature, facing, lean, x, y)
    keypoints = with_face_visibility(observe(camera, truth), camera, facing, x, y)
    result = lift_height_prior(keypoints, camera)

    assert result.method == "height_prior_calibrated"
    assert result.reason == "lifted"
    assert abs(result.stature_m - stature) < STATURE_TOL_M
    assert angle_diff(result.facing_deg, facing) < FACING_TOL_DEG
    assert abs(result.trunk_lean_deg - lean) < LEAN_TOL_DEG
    assert abs(result.trunk_tilt_deg - abs(lean)) < LEAN_TOL_DEG
    assert result.reproj_err_px < 0.05
    assert result.n_fit_points == 5
    assert not result.stature_at_bound
    assert result.facing_conf > 0.5
    # Joints whose heights the model describes come back where they were (centimetre).
    error = np.linalg.norm(result.joints3d[EXACT_JOINTS] - truth[EXACT_JOINTS], axis=1)
    assert float(error.max()) < 0.01
    # Every lifted joint reprojects onto its own keypoint; unseen ones are not lifted.
    lifted = np.isfinite(result.joints3d).all(axis=1)
    np.testing.assert_array_equal(lifted, keypoints[:, 2] >= 0.3)
    uv, _, valid = project(camera, pitch_zup_to_soccernet(result.joints3d[lifted]))
    assert valid.all()
    np.testing.assert_allclose(uv, keypoints[lifted, :2], atol=1e-6)


def test_heights_follow_the_prior_table_for_an_upright_player() -> None:
    camera = broadcast_camera(5.0, 3.0)
    truth = player_zup(1.80, 30.0, 0.0, 5.0, 3.0)
    result = lift_height_prior(observe(camera, truth), camera)
    exact = [i for i in EXACT_JOINTS]
    np.testing.assert_allclose(
        result.joints3d[exact, 2], np.array(JOINT_HEIGHT_PRIOR_M)[exact], atol=1e-3
    )
    assert JOINT_HEIGHT_PRIOR_M[KP["head"]] == 1.80
    assert JOINT_HEIGHT_PRIOR_M[KP["left_ankle"]] == 0.09
    assert JOINT_HEIGHT_PRIOR_M[KP["left_heel"]] == 0.05
    assert JOINT_HEIGHT_PRIOR_M[KP["right_small_toe"]] == 0.03


def test_lean_sign_follows_the_facing_not_the_camera() -> None:
    x, y = 0.0, 0.0
    camera = broadcast_camera(x, y)
    forward = lift_height_prior(observe(camera, player_zup(1.80, 0.0, 15.0, x, y)), camera)
    backward = lift_height_prior(observe(camera, player_zup(1.80, 0.0, -15.0, x, y)), camera)
    turned = lift_height_prior(observe(camera, player_zup(1.80, 180.0, 15.0, x, y)), camera)
    assert forward.trunk_lean_deg == pytest.approx(15.0, abs=LEAN_TOL_DEG)
    assert backward.trunk_lean_deg == pytest.approx(-15.0, abs=LEAN_TOL_DEG)
    assert turned.trunk_lean_deg == pytest.approx(15.0, abs=LEAN_TOL_DEG)
    assert angle_diff(turned.facing_deg, 180.0) < FACING_TOL_DEG


def test_a_leaning_player_is_not_read_as_a_short_one() -> None:
    # Facing the camera and leaning towards it shortens the image of the trunk; the tilt
    # in the model has to absorb that instead of the stature.
    x, y = 0.0, 0.0
    camera = broadcast_camera(x, y)
    facing = facing_towards(camera, x, y)
    result = lift_height_prior(observe(camera, player_zup(1.80, facing, 25.0, x, y)), camera)
    assert abs(result.stature_m - 1.80) < STATURE_TOL_M
    assert result.trunk_lean_deg == pytest.approx(25.0, abs=LEAN_TOL_DEG)


def depression_deg(camera: Camera, x: float, y: float, height: float = 1.25) -> float:
    """Angle below the horizontal of the camera ray to (x, y, height) in pitch_zup."""
    centre = pitch_zup_to_soccernet(camera.C)
    distance = math.hypot(centre[0] - x, centre[1] - y)
    return math.degrees(math.atan2(centre[2] - height, distance))


def test_a_forward_lean_away_from_the_camera_comes_back_as_its_mirror() -> None:
    # The documented monocular ambiguity: theta away from the camera and 2 alpha - theta
    # project alike. The fit keeps the more upright one; trunk_lean_alt_deg holds the other.
    x, y = 0.0, 0.0
    camera = broadcast_camera(x, y)
    facing = facing_towards(camera, x, y) + 180.0
    alpha = depression_deg(camera, x, y)
    keypoints = with_face_visibility(
        observe(camera, player_zup(1.80, facing, 22.0, x, y)), camera, facing, x, y
    )
    result = lift_height_prior(keypoints, camera)
    assert 10.0 < alpha < 16.0
    assert result.trunk_lean_deg == pytest.approx(2.0 * alpha - 22.0, abs=LEAN_TOL_DEG)
    assert result.trunk_lean_alt_deg == pytest.approx(22.0, abs=LEAN_TOL_DEG)
    assert abs(result.stature_m - 1.80) < STATURE_TOL_M  # the hips fix the scale
    assert angle_diff(result.facing_deg, facing) < FACING_TOL_DEG


def test_the_mirror_solution_agrees_for_a_side_on_player() -> None:
    x, y = 6.0, -3.0
    camera = broadcast_camera(x, y)
    facing = facing_towards(camera, x, y) + 90.0
    result = lift_height_prior(observe(camera, player_zup(1.80, facing, 15.0, x, y)), camera)
    assert result.trunk_lean_deg == pytest.approx(15.0, abs=LEAN_TOL_DEG)
    assert result.trunk_lean_alt_deg == pytest.approx(result.trunk_lean_deg, abs=0.5)


def crouched_zup(
    stature: float, facing: float, lean: float, x: float, y: float, drop: float = 0.12
) -> np.ndarray:
    """The template player with bent knees: everything above the knees ``drop`` m lower,
    knees 0.15 m forward and 0.05 m lower; feet unchanged."""
    joints = player_zup(stature, facing, lean, x, y)
    heading = math.radians(facing)
    forward = np.array([math.cos(heading), math.sin(heading), 0.0])
    for name in TEMPLATE:
        if "knee" in name:
            joints[KP[name]] += 0.15 * forward - np.array([0.0, 0.0, 0.05])
        elif not any(part in name for part in ("ankle", "toe", "heel")):
            joints[KP[name]] -= np.array([0.0, 0.0, drop])
    return joints


@pytest.mark.parametrize("lean", [0.0, 15.0])
def test_a_crouch_side_on_keeps_its_lean_and_facing(lean: float) -> None:
    x, y = -15.0, 8.0
    camera = broadcast_camera(x, y)
    facing = facing_towards(camera, x, y) + 90.0
    truth = crouched_zup(1.80, facing, lean, x, y)
    result = lift_height_prior(observe(camera, truth), camera)
    assert result.reason == "lifted"
    # The lowered hips are lifted at the fitted hip height (0.88 m here, truth 0.83 m), not
    # at the standing prior (0.91 m for the fitted stature), so they slide less far along
    # their rays away from the feet.
    hip = result.joints3d[KP["hip"]]
    assert abs(hip[2] - truth[KP["hip"], 2]) < 0.06
    assert np.linalg.norm(hip - truth[KP["hip"]]) < 0.3
    assert result.trunk_lean_deg == pytest.approx(lean, abs=LEAN_TOL_DEG)
    assert angle_diff(result.facing_deg, facing) < FACING_TOL_DEG
    assert result.trunk_lean_sigma_deg < 6.0
    assert 1.62 < result.stature_m < 1.80  # bent knees read as a shorter player (documented)


@pytest.mark.parametrize("offset", [0.0, 180.0])
def test_lean_along_the_camera_ray_is_flagged_by_its_sigma(offset: float) -> None:
    # A crouched, upright player facing the camera or turned away from it: the bent knees
    # lower the trunk, which the lift can read as a lean along the ray. The facing stays
    # right and the lean's sigma says not to trust the lean.
    x, y = 0.0, 0.0
    camera = broadcast_camera(x, y)
    facing = facing_towards(camera, x, y) + offset
    keypoints = with_face_visibility(
        observe(camera, crouched_zup(1.80, facing, 0.0, x, y)), camera, facing, x, y
    )
    result = lift_height_prior(keypoints, camera)
    assert angle_diff(result.facing_deg, facing) < FACING_TOL_DEG
    assert result.trunk_lean_sigma_deg > 15.0
    assert abs(result.trunk_lean_deg) < 2.0 * result.trunk_lean_sigma_deg
    side_on = lift_height_prior(observe(camera, player_zup(1.80, facing + 90.0, 0.0, x, y)), camera)
    assert result.trunk_lean_sigma_deg > 4.0 * side_on.trunk_lean_sigma_deg


def test_low_score_keypoints_are_ignored() -> None:
    x, y = -12.0, 6.0
    camera = broadcast_camera(x, y)
    clean = observe(camera, player_zup(1.84, 70.0, 9.0, x, y))
    reference = lift_height_prior(clean, camera)
    noisy = clean.copy()
    for name, shift in (("head", (90.0, -60.0)), ("left_wrist", (-150.0, 40.0)), ("nose", (0, 70))):
        noisy[KP[name], :2] += shift
        noisy[KP[name], 2] = 0.05
    result = lift_height_prior(noisy, camera)
    assert result.stature_m == pytest.approx(reference.stature_m, abs=1e-3)
    assert result.facing_deg == pytest.approx(reference.facing_deg, abs=0.05)
    assert result.trunk_lean_deg == pytest.approx(reference.trunk_lean_deg, abs=0.05)
    assert result.n_fit_points == reference.n_fit_points - 1  # the head top is not used
    for name in ("head", "left_wrist", "nose"):
        assert np.isnan(result.joints3d[KP[name]]).all()
    others = [i for i in range(26) if i not in {KP["head"], KP["left_wrist"], KP["nose"]}]
    np.testing.assert_allclose(result.joints3d[others], reference.joints3d[others], atol=2e-3)
    # The same corrupted keypoints with a confident score do change the answer.
    trusted = noisy.copy()
    trusted[[KP["head"], KP["left_wrist"], KP["nose"]], 2] = 0.9
    moved = lift_height_prior(trusted, camera)
    assert abs(moved.stature_m - reference.stature_m) > 1e-3


def mirror_labels(keypoints: np.ndarray) -> np.ndarray:
    """Swap every left/right pair, as a pose model does when it mistakes front for back."""
    mirrored = keypoints.copy()
    for left, right in HALPE26_FLIP_PAIRS:
        mirrored[[left, right]] = keypoints[[right, left]]
    return mirrored


def test_mirrored_labels_facing_away_are_outvoted_by_toes_and_the_hidden_face() -> None:
    x, y = 4.0, -6.0
    camera = broadcast_camera(x, y)
    facing = facing_towards(camera, x, y) + 180.0  # back to the camera
    keypoints = observe(camera, player_zup(1.80, facing, 0.0, x, y))
    keypoints[[KP["nose"], KP["left_eye"], KP["right_eye"]], 2] = 0.1  # face not visible
    correct = lift_height_prior(keypoints, camera)
    mirrored = lift_height_prior(mirror_labels(keypoints), camera)
    assert angle_diff(correct.facing_deg, facing) < FACING_TOL_DEG
    assert angle_diff(mirrored.facing_deg, facing) < FACING_TOL_DEG
    assert mirrored.facing_conf < correct.facing_conf  # the labels disagreed


def test_mirrored_labels_in_profile_are_outvoted_by_toes_and_the_nose() -> None:
    x, y = -8.0, 4.0
    camera = broadcast_camera(x, y)
    facing = facing_towards(camera, x, y) + 90.0  # side-on to the camera
    keypoints = observe(camera, player_zup(1.80, facing, 5.0, x, y))
    mirrored = lift_height_prior(mirror_labels(keypoints), camera)
    assert angle_diff(mirrored.facing_deg, facing) < FACING_TOL_DEG


def test_mirrored_labels_facing_the_camera_are_outvoted_by_the_visible_face() -> None:
    x, y = 10.0, 0.0
    camera = broadcast_camera(x, y)
    facing = facing_towards(camera, x, y)
    keypoints = observe(camera, player_zup(1.80, facing, 0.0, x, y))
    keypoints[[KP[n] for n in TEMPLATE if "toe" in n or "heel" in n], 2] = 0.1  # feet unclear
    mirrored = lift_height_prior(mirror_labels(keypoints), camera)
    assert angle_diff(mirrored.facing_deg, facing) < FACING_TOL_DEG


def test_without_any_cue_the_labels_decide() -> None:
    x, y = 0.0, 0.0
    camera = broadcast_camera(x, y)
    facing = facing_towards(camera, x, y) + 180.0
    keypoints = observe(camera, player_zup(1.80, facing, 0.0, x, y))
    no_cues = [KP[n] for n in TEMPLATE if "toe" in n or "heel" in n]
    no_cues += [KP["nose"], KP["left_eye"], KP["right_eye"]]
    keypoints[no_cues, 2] = 0.4  # between the hidden and visible face thresholds
    keypoints[[KP["nose"]], 2] = 0.2
    result = lift_height_prior(keypoints, camera)
    flipped = lift_height_prior(mirror_labels(keypoints), camera)
    assert angle_diff(result.facing_deg, facing) < FACING_TOL_DEG
    assert angle_diff(flipped.facing_deg, facing + 180.0) < FACING_TOL_DEG


def test_not_lifted_cases_report_a_reason() -> None:
    x, y = 3.0, 3.0
    camera = broadcast_camera(x, y)
    keypoints = observe(camera, player_zup(1.80, 20.0, 5.0, x, y))

    for result, reason in (
        (lift_height_prior(keypoints, None), "no_calibration"),
        (lift_height_prior(keypoints, camera, airborne=True), "airborne"),
    ):
        assert result.method == "image_plane_only"
        assert result.reason == reason
        assert np.isnan(result.joints3d).all()
        assert math.isnan(result.stature_m)
        assert math.isnan(result.facing_deg)
        assert math.isfinite(result.image.trunk_lean_deg)

    feetless = keypoints.copy()
    feetless[[KP[n] for n in TEMPLATE if "ankle" in n or "toe" in n or "heel" in n], 2] = 0.1
    assert lift_height_prior(feetless, camera).reason == "feet_low_confidence"

    torsoless = keypoints.copy()
    torsoless[[KP["hip"], KP["left_hip"]], 2] = 0.1
    assert lift_height_prior(torsoless, camera).reason == "torso_low_confidence"
    headless = keypoints.copy()
    upper = ["neck", "head", "left_shoulder", "right_shoulder", "left_ear", "right_ear"]
    headless[[KP[n] for n in upper], 2] = 0.1
    assert lift_height_prior(headless, camera).reason == "torso_low_confidence"


def test_heels_and_toes_anchor_the_body_when_the_ankles_are_unsure() -> None:
    x, y = 3.0, 3.0
    camera = broadcast_camera(x, y)
    keypoints = observe(camera, player_zup(1.80, 20.0, 0.0, x, y))
    keypoints[[KP["left_ankle"], KP["right_ankle"]], 2] = 0.1
    result = lift_height_prior(keypoints, camera)
    assert result.reason == "lifted"
    # Heels and toes sit ahead of the ankles, so the stature drifts a little (documented).
    assert abs(result.stature_m - 1.80) < 0.1


def test_a_foot_ray_above_the_horizon_is_not_lifted() -> None:
    # A camera 2 m up looking level: a "foot" drawn above the image centre looks at the sky.
    camera = custom_camera((0.0, -30.0, 2.0), (0.0, 0.0, 2.0), 1500.0)
    keypoints = np.column_stack([np.full(26, 959.5), np.linspace(300, 500, 26), np.full(26, 0.9)])
    result = lift_height_prior(keypoints, camera)
    assert result.reason == "foot_ray_miss"
    assert result.method == "image_plane_only"


def test_a_raised_kicking_foot_does_not_anchor_the_body() -> None:
    x, y = 0.0, 0.0
    camera = broadcast_camera(x, y)
    truth = player_zup(1.80, 0.0, 0.0, x, y)
    # The right leg swings forward and up: ankle 0.9 m ahead and 0.6 m above the grass.
    for name, (df, dh) in {
        "right_knee": (0.45, 0.25),
        "right_ankle": (0.9, 0.55),
        "right_heel": (0.85, 0.55),
        "right_big_toe": (1.05, 0.6),
        "right_small_toe": (1.03, 0.6),
    }.items():
        truth[KP[name]] += np.array([df, 0.0, dh])
    result = lift_height_prior(observe(camera, truth), camera)
    assert result.reason == "lifted"
    assert abs(result.stature_m - 1.80) < 0.06  # the standing foot alone sits 0.1 m aside


def test_a_very_tall_player_stops_at_the_stature_bound() -> None:
    x, y = 0.0, 0.0
    camera = broadcast_camera(x, y)
    result = lift_height_prior(observe(camera, player_zup(2.25, 0.0, 0.0, x, y)), camera)
    assert result.stature_m == pytest.approx(STATURE_RANGE_M[1])
    assert result.stature_at_bound
    assert result.reproj_err_px > 1.0


@pytest.mark.parametrize("offset_deg", [0.0, 90.0])
def test_facing_sigma_predicts_the_spread_under_one_pixel_of_noise(offset_deg: float) -> None:
    # A ~72 px tall player in the wide shot, facing the camera (0) or side-on (90), with
    # 1 px of Gaussian noise on every keypoint coordinate, 40 fixed seeds.
    x, y = -15.0, 8.0
    camera = broadcast_camera(x, y)
    facing = facing_towards(camera, x, y) + offset_deg
    clean = with_face_visibility(
        observe(camera, player_zup(1.80, facing, 5.0, x, y)), camera, facing, x, y
    )
    predicted = lift_height_prior(clean, camera).facing_sigma_deg
    errors, statures = [], []
    for seed in range(40):
        noisy = clean.copy()
        noisy[:, :2] += np.random.default_rng(seed).normal(0.0, 1.0, size=(26, 2))
        result = lift_height_prior(noisy, camera)
        errors.append((result.facing_deg - facing + 180.0) % 360.0 - 180.0)
        statures.append(result.stature_m)
    rms = math.sqrt(float(np.mean(np.square(errors))))
    assert max(abs(e) for e in errors) < 90.0  # no front/back flip
    assert 0.5 * predicted < rms < 2.0 * predicted
    assert float(np.median(np.abs(np.array(statures) - 1.80))) < 0.05


def test_side_on_facing_is_better_determined_than_frontal() -> None:
    x, y = -15.0, 8.0
    camera = broadcast_camera(x, y)
    sigmas = {}
    for offset in (0.0, 90.0):
        facing = facing_towards(camera, x, y) + offset
        keypoints = observe(camera, player_zup(1.80, facing, 0.0, x, y))
        sigmas[offset] = lift_height_prior(keypoints, camera).facing_sigma_deg
    assert sigmas[0.0] > 3.0 * sigmas[90.0]
    unlifted = lift_height_prior(observe(camera, player_zup(1.80, 0.0, 0.0, x, y)), None)
    assert math.isnan(unlifted.facing_sigma_deg)


def test_image_plane_angles_on_a_planted_pose() -> None:
    keypoints = np.zeros((26, 3))
    keypoints[:, 2] = 0.9
    keypoints[KP["hip"], :2] = (500.0, 600.0)
    keypoints[KP["neck"], :2] = (500.0, 380.0)  # not used: the shoulder line is the trunk
    keypoints[KP["left_shoulder"], :2] = (480.0, 410.0)
    keypoints[KP["right_shoulder"], :2] = (560.0, 390.0)  # shoulder midpoint (520, 400)
    keypoints[KP["left_hip"], :2] = (490.0, 600.0)
    keypoints[KP["left_knee"], :2] = (490.0, 700.0)
    keypoints[KP["left_ankle"], :2] = (590.0, 700.0)
    keypoints[KP["right_hip"], :2] = (510.0, 600.0)  # hip midpoint (500, 600)
    keypoints[KP["right_knee"], :2] = (510.0, 700.0)
    keypoints[KP["right_ankle"], :2] = (510.0, 800.0)
    angles = image_plane_angles(keypoints)
    assert angles.plane == "image"
    assert angles.trunk_lean_deg == pytest.approx(math.degrees(math.atan2(20.0, 200.0)))
    assert angles.knee_left_deg == pytest.approx(90.0)
    assert angles.knee_right_deg == pytest.approx(180.0)
    assert lift3d.image_trunk_lean_deg(keypoints[None])[0] == pytest.approx(angles.trunk_lean_deg)

    leaning_left = keypoints.copy()
    leaning_left[KP["left_shoulder"], :2] = (390.0, 530.0)
    leaning_left[KP["right_shoulder"], :2] = (470.0, 530.0)  # 70 px left, 70 px up
    assert image_plane_angles(leaning_left).trunk_lean_deg == pytest.approx(-45.0)

    one_shoulder = keypoints.copy()
    one_shoulder[KP["right_shoulder"], 2] = 0.1  # no shoulder line: the neck is no stand-in
    assert math.isnan(image_plane_angles(one_shoulder).trunk_lean_deg)

    unsure = keypoints.copy()
    unsure[KP["left_knee"], 2] = 0.1
    assert math.isnan(image_plane_angles(unsure).knee_left_deg)
    assert image_plane_angles(unsure).knee_right_deg == pytest.approx(180.0)


def test_trunk_lean_conventions() -> None:
    joints = np.full((26, 3), np.nan)
    joints[KP["hip"]] = (0.0, 0.0, 1.0)
    joints[KP["neck"]] = (0.0, 0.3, 1.3)  # 45 degrees towards +y
    assert trunk_lean(joints, 90.0) == pytest.approx((45.0, 45.0))
    assert trunk_lean(joints, -90.0) == pytest.approx((-45.0, 45.0))
    lean, tilt = trunk_lean(joints, 0.0)  # sideways for a player facing +x
    assert lean == pytest.approx(0.0, abs=1e-9)
    assert tilt == pytest.approx(45.0)
    lean, tilt = trunk_lean(joints, math.nan)
    assert math.isnan(lean)
    assert tilt == pytest.approx(45.0)
    joints[KP["neck"]] = (0.0, 0.0, 0.5)  # neck below the hip: nonsense, not a number
    assert all(math.isnan(v) for v in trunk_lean(joints, 0.0))


def test_facing_from_the_shoulder_line_alone_and_none_without_lines() -> None:
    joints = np.full((26, 3), np.nan)
    scores = np.full(26, 0.9)
    joints[KP["left_shoulder"]] = (1.0, 0.19, 1.45)
    joints[KP["right_shoulder"]] = (1.0, -0.19, 1.45)
    facing = estimate_facing(joints, scores)
    assert facing.facing_deg == pytest.approx(0.0, abs=1e-9)  # left is +y, so forward is +x
    assert 0.0 < facing.facing_conf < 1.0
    nothing = estimate_facing(np.full((26, 3), np.nan), scores)
    assert math.isnan(nothing.facing_deg)
    assert nothing.facing_conf == 0.0


def test_lift_pose_table_and_parquet_round_trip(unicode_tmp_path) -> None:
    x, y = 2.0, -4.0
    camera = broadcast_camera(x, y)
    keypoints = observe(camera, player_zup(1.80, 10.0, 5.0, x, y))
    rows = [
        {
            "frame": frame,
            "track_id": track,
            "model_id": "fake",
            "x1": 0.0,
            "y1": 0.0,
            "x2": 1.0,
            "y2": 1.0,
            "keypoints": keypoints.reshape(-1).tolist(),
            "mean_score": 0.9,
        }
        for frame, track in ((10, 7), (11, 7), (12, 7))
    ]
    pose = pose_table(rows)
    table = lift_pose_table(pose, {10: camera, 12: camera}, airborne={(12, 7): True})
    assert list(table.columns) == list(LIFT3D_COLUMNS)
    assert table["method"].tolist() == [
        "height_prior_calibrated",
        "image_plane_only",
        "image_plane_only",
    ]
    assert table["reason"].tolist() == ["lifted", "no_calibration", "airborne"]
    assert table["frame"].dtype == np.int64
    assert table["stature_m"].dtype == np.float64
    assert all(len(item) == 78 for item in table["joints3d"])
    assert table["stature_m"].iloc[0] == pytest.approx(1.80, abs=STATURE_TOL_M)
    assert np.isnan(table["stature_m"].iloc[1])
    assert np.isfinite(table["image_lean_deg"]).all()

    path = unicode_tmp_path / "lift3d.parquet"
    write_lift3d_parquet(table, path)
    stored = pq.read_table(path)
    assert stored.schema.equals(LIFT3D_SCHEMA)
    joints = np.array(stored.column("joints3d").to_pylist()[0]).reshape(26, 3)
    np.testing.assert_allclose(joints, np.array(table["joints3d"].iloc[0]).reshape(26, 3))


def test_invalid_input_lists_every_problem() -> None:
    with pytest.raises(LiftInputError) as error:
        lift_height_prior(np.zeros((25, 3)), "camera", score_threshold=-1.0, airborne="no")
    assert len(error.value.problems) == 4
    with pytest.raises(LiftInputError):
        image_plane_angles(np.zeros((26, 2)))
    with pytest.raises(LiftInputError):
        estimate_facing(np.zeros((26, 2)), np.zeros(25))
    bad = np.zeros((26, 3))
    bad[3, 2] = np.nan
    with pytest.raises(LiftInputError, match="scores must be finite"):
        lift_height_prior(bad, None)
