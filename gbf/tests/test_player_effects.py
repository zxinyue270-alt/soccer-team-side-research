"""Player overlays use current, associated poses without filling missing limbs."""

from fractions import Fraction

import numpy as np
import pandas as pd
from PIL import Image

from gbf.player_effects import draw_player_effects, plan_player_effects


def scene():
    tracks = pd.DataFrame(
        [
            dict(frame=f, track_id=tid, x1=x, y1=20, x2=x + 25, y2=80)
            for f in range(3)
            for tid, x in [(9, 60), (10, 100), (11, 450)]
        ]
    )
    joints = np.full((26, 3), [70, 50, 0.9])
    joints[18] = [72, 32, 0.95]
    joints[19] = [72, 58, 0.95]
    poses = tracks[tracks.track_id == 9].copy()
    poses["keypoints"] = [joints.ravel().tolist() for _ in range(3)]
    poses["model_id"] = "example-halpe26"
    h = dict(
        track_ids=[9, 10, 11],
        key_pair=[9, 10],
        key_moments=[dict(kind="ball_away", frame=1, track_ids=[9])],
    )
    return h, tracks, poses


def test_actor_emphasis_uses_event_and_nearby_players_without_remote_runner():
    h, tracks, poses = scene()
    plan = plan_player_effects(h, tracks, poses, 0, 2, 1, (500, 180), Fraction(25))
    assert plan[1]["selected_track_ids"] == [9, 10]
    assert {a["track_id"] for a in plan[1]["actors"]} == {9, 10}
    actor = plan[1]["actors"][0]
    assert actor["pose_reason"] == "associated_pose"
    assert [18, 19] in actor["edges"]
    assert plan[1]["actors"][1]["edges"] == []


def test_missing_duplicate_or_unassociated_pose_never_gets_a_skeleton():
    h, tracks, poses = scene()
    poses.loc[poses.frame == 1, "x1"] = 20
    poses = pd.concat([poses, poses[poses.frame == 2]])
    plan = plan_player_effects(h, tracks, poses, 0, 2, 1, (320, 180), Fraction(25))
    assert plan[1]["actors"][0]["pose_reason"] == "pose_box_mismatch"
    assert plan[1]["actors"][0]["edges"] == []
    assert plan[2]["actors"][0]["pose_reason"] == "duplicate_pose"
    assert plan[2]["actors"][0]["edges"] == []


def test_low_score_and_out_of_box_joints_are_not_connected_or_interpolated():
    h, tracks, poses = scene()
    joints = np.array(poses.iloc[1].keypoints).reshape(26, 3)
    joints[18, 2] = 0.1
    joints[15] = [300, 150, 0.99]
    poses.at[poses.index[1], "keypoints"] = joints.ravel().tolist()
    plan = plan_player_effects(h, tracks, poses, 0, 2, 1, (320, 180), Fraction(25))
    actor = plan[1]["actors"][0]
    assert actor["keypoints"][18] is None
    assert actor["keypoints"][15] is None
    assert not any(18 in e or 15 in e for e in actor["edges"])
    assert plan[0]["actors"][0]["keypoints"][18] is not None
    assert plan[2]["actors"][0]["keypoints"][18] is not None


def test_glow_and_skeleton_are_visible_without_mutating_the_original_image():
    h, tracks, poses = scene()
    plan = plan_player_effects(h, tracks, poses, 0, 2, 1, (320, 180), Fraction(25))
    source = Image.new("RGB", (320, 180), (20, 40, 15))
    before = np.array(source)
    drawn = np.array(draw_player_effects(source, plan[1], skeleton=True))
    halo = np.array(draw_player_effects(source, plan[1], skeleton=False))
    np.testing.assert_array_equal(np.array(source), before)
    assert drawn[45, 72, :].sum() > halo[45, 72, :].sum()
    assert np.any(halo[80:85, 60:90] != before[80:85, 60:90])
