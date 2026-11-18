"""Replay emphasis must follow evidence without manufacturing a ball trajectory."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from gbf.replay_effects import draw_ball_trail, plan_ball_trails, plan_emphasis


def test_release_overrides_running_peak_and_keeps_follow_through():
    highlight = {
        "start_frame": 50,
        "end_frame": 200,
        "peak_frame": 100,
        "key_moments": [{"kind": "ball_away", "frame": 112, "track_ids": [9]}],
    }
    emphasis, edit = plan_emphasis(highlight, 0, 250, Fraction(25), cuts=[])
    assert emphasis["key_frame"] == 112
    assert emphasis["source"] == "key_moments.ball_away"
    assert next(s for s in edit.segments if s.kind == "freeze").source_frames == (112,)
    slow = {f for s in edit.segments if s.kind == "slow" for f in s.source_frames}
    assert {111, 113, 137, 162} <= slow
    assert max(slow) == 162


@pytest.mark.parametrize("event_frame", [True, 49, 201, 300, None])
def test_invalid_or_unrelated_release_keeps_original_peak(event_frame):
    h = dict(
        start_frame=50,
        end_frame=200,
        peak_frame=100,
        key_moments=[dict(kind="ball_away", frame=event_frame)],
    )
    emphasis, _ = plan_emphasis(h, 0, 250, Fraction(25))
    assert emphasis["key_frame"] == 100


def test_release_cannot_pull_replay_across_a_cut():
    h = dict(
        start_frame=50,
        end_frame=200,
        peak_frame=100,
        key_moments=[dict(kind="ball_away", frame=112)],
    )
    emphasis, _ = plan_emphasis(h, 0, 250, Fraction(25), cuts=[105])
    assert emphasis["key_frame"] == 100


def ball_rows():
    return pd.DataFrame(
        [
            dict(
                frame=f,
                u=10 + 2 * f,
                v=30,
                radius_px=2,
                conf=0.7,
                status="detected",
                candidate_count=1,
            )
            for f in range(20)
        ]
    )


def test_trail_uses_only_past_direct_points_and_bounded_history():
    trails = plan_ball_trails(ball_rows(), 0, 19, Fraction(25), (128, 72))
    assert [p["frame"] for p in trails[10]["points"]] == list(range(3, 11))
    assert all(p["frame"] <= f for f, e in trails.items() for p in e["points"])
    changed = ball_rows()
    changed.loc[changed.frame > 10, "u"] = 120
    assert plan_ball_trails(changed, 0, 19, Fraction(25), (128, 72))[10] == trails[10]


@pytest.mark.parametrize("break_kind", ["missing", "interpolated", "cut", "jump", "duplicate"])
def test_trail_never_connects_a_gap_cut_jump_or_duplicate(break_kind):
    ball = ball_rows()
    if break_kind in {"missing", "interpolated"}:
        ball.loc[ball.frame == 8, "status"] = break_kind
    elif break_kind == "jump":
        ball.loc[ball.frame == 8, "u"] = 120
    elif break_kind == "duplicate":
        ball = pd.concat([ball, ball[ball.frame == 8]])
    trails = plan_ball_trails(
        ball, 0, 19, Fraction(25), (128, 72), cuts=[8] if break_kind == "cut" else []
    )
    assert not any(p["frame"] < 8 for p in trails[8]["points"])
    assert not any(p["frame"] < 8 for p in trails[9]["points"])


def test_estimated_path_retains_alternatives_but_rejects_selected_mismatch():
    ball = ball_rows()
    ball["candidate_count"] = 2
    gates = pd.DataFrame(
        [
            dict(frame=f, cand_index=i, u=u, v=30, diameter_px=4, reason="ok")
            for f in range(20)
            for i, u in [(0, 10 + 2 * f), (1, 110)]
        ]
    )
    trails = plan_ball_trails(ball, 0, 19, Fraction(25), (128, 72), candidates=gates)
    assert len(trails[10]["points"]) > 1
    assert trails[10]["points"][-1]["alternative_count"] == 2
    assert trails[10]["identity_verified"] is False
    gates.loc[gates.frame == 10, "u"] = 100
    changed = plan_ball_trails(ball, 0, 19, Fraction(25), (128, 72), candidates=gates)
    assert changed[10]["points"] == []


def test_visible_fading_tail_does_not_modify_source_pixels():
    image = Image.new("RGB", (128, 72), (15, 20, 25))
    before = np.array(image)
    trail = plan_ball_trails(ball_rows(), 0, 19, Fraction(25), image.size)[10]
    rendered = np.array(draw_ball_trail(image, trail))
    np.testing.assert_array_equal(np.array(image), before)
    assert rendered[30, 27, :].sum() > rendered[30, 19, :].sum() > before[30, 19, :].sum()
    assert np.count_nonzero(rendered != before) > 30


def test_camera_pan_places_tail_behind_ground_relative_motion():
    from gbf.camera_motion import BackgroundMotion

    rng = np.random.default_rng(17)
    background = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
    moved = np.roll(background, -8, axis=1)
    camera = BackgroundMotion()
    camera.update(0, background)
    motion = camera.update(1, moved)
    assert motion["valid"] is True
    ball = pd.DataFrame(
        [
            dict(frame=f, u=u, v=100, radius_px=2, conf=0.9, status="detected", candidate_count=1)
            for f, u in [(0, 100), (1, 98)]
        ]
    )
    trail = plan_ball_trails(ball, 0, 1, Fraction(25), (320, 180), camera_motion={1: motion})[1]
    assert trail["points"][0]["u"] == 100
    assert trail["points"][0]["display_u"] == pytest.approx(92, abs=1)
    assert trail["points"][0]["display_u"] < trail["points"][-1]["display_u"]
    assert trail["direction"][0] > 4


def test_uncertain_camera_does_not_draw_an_uncompensated_tail():
    trails = plan_ball_trails(
        ball_rows(),
        0,
        19,
        Fraction(25),
        (128, 72),
        camera_motion={f: {"valid": False} for f in range(20)},
    )
    assert all(len(e["points"]) <= 1 for e in trails.values())


def test_direction_change_never_leaves_old_tail_in_front_of_current_ball():
    ball = ball_rows().iloc[:6].copy()
    ball["u"] = [10, 12, 14, 16, 18, 17]
    trail = plan_ball_trails(ball, 0, 5, Fraction(25), (128, 72))[5]
    assert [p["frame"] for p in trail["points"]] == [4, 5]
    assert trail["points"][0]["display_u"] > trail["points"][-1]["display_u"]
