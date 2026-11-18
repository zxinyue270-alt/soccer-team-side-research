"""Public replay planning contracts: honest missing evidence and bounded context."""

from fractions import Fraction

import pandas as pd
import pytest

from gbf.action_context import associated_ids, context_window, plan_focus


def people():
    return pd.DataFrame(
        [
            dict(frame=f, track_id=t, x1=x, y1=100, x2=x + 30, y2=200)
            for f in range(3)
            for t, x in [(7, 400), (8, 490)]
        ]
    )


def balls():
    return pd.DataFrame(
        [
            dict(frame=f, u=480, v=190, radius_px=3, status="detected", candidate_count=1)
            for f in range(3)
        ]
    )


def test_context_keeps_candidate_adds_two_seconds_and_never_crosses_cut():
    h = dict(start_frame=100, peak_frame=150, end_frame=200)
    shot = dict(start_frame=0, end_frame=400)
    context = context_window(h, shot, 401, Fraction(30000, 1001), cuts=[60, 230])
    assert (context["start_frame"], context["end_frame"]) == (60, 229)
    assert context["action_boundaries_verified"] is False
    assert context["start_limit"] == context["end_limit"] == "shot_or_source"
    limited = context_window(h, shot, 401, Fraction(25))
    assert (limited["start_frame"], limited["end_frame"]) == (50, 250)
    assert limited["end_limit"] == "context_budget"


def test_associations_include_moment_players_without_claiming_first_actor():
    assert associated_ids(
        dict(
            track_ids=[8, 7],
            key_pair=[7, 9],
            key_focus_track=4,
            key_moments=[dict(track_ids=[10, 8])],
        )
    ) == [4, 7, 8, 9, 10]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"status": "interpolated"}, "ball_not_direct"),
        ({"candidate_count": 2}, "ball_candidates_ambiguous"),
        ({"u": float("nan")}, "ball_invalid_geometry"),
        ({"u": 1000}, "ball_invalid_geometry"),
    ],
)
def test_uncertain_ball_uses_full_view_and_preserves_other_observed_frames(changes, reason):
    ball = balls()
    for key, value in changes.items():
        ball.loc[ball.frame == 1, key] = value
    plan = plan_focus(people(), ball, [7], 0, 2, (1000, 600))
    assert plan[1]["crop"] is None
    assert plan[1]["reason"] == reason
    assert plan[0]["crop"] is not None
    assert plan[2]["crop"] is not None
    assert plan[0]["track_ids"] == [7, 8]


def test_duplicate_ball_and_duplicate_associated_box_use_full_view():
    ball = pd.concat([balls(), balls().iloc[[1]]])
    assert plan_focus(people(), ball, [7], 0, 2, (1000, 600))[1]["reason"] == "ball_ambiguous"
    tracks = pd.concat([people(), people().iloc[[2]]])
    plan = plan_focus(tracks, balls(), [7], 0, 2, (1000, 600))
    assert plan[1]["crop"] is None
    assert plan[1]["reason"] == "associated_box_ambiguous_or_invalid"


def test_missing_proposed_identity_is_recorded_without_bridging_or_hiding_other_people():
    tracks = people()
    tracks = tracks[~((tracks.frame == 1) & (tracks.track_id == 7))]
    plan = plan_focus(tracks, balls(), [7], 0, 2, (1000, 600))
    assert plan[1]["associated_missing_ids"] == [7]
    assert plan[1]["track_ids"] == [8]
    assert plan[0]["track_ids"] == plan[2]["track_ids"] == [7, 8]


def test_wide_scene_and_no_ball_source_fall_back_to_full_view():
    tracks = people()
    tracks.loc[tracks.track_id == 7, ["x1", "x2", "y1", "y2"]] = [0, 1000, 0, 600]
    plan = plan_focus(tracks, balls(), [7], 0, 2, (1000, 600))
    assert all(p["crop"] is None and p["reason"] == "region_too_wide" for p in plan.values())
    assert all(
        p["crop"] is None
        for p in plan_focus(people(), pd.DataFrame(), [7], 0, 2, (1000, 600)).values()
    )


def test_multiple_raw_candidates_can_have_one_matching_gated_option():
    ball = balls()
    ball["candidate_count"] = 2
    candidates = pd.DataFrame(
        [
            dict(frame=f, cand_index=i, u=u, v=190, diameter_px=6, reason=reason)
            for f in range(3)
            for i, u, reason in [(0, 480, "ok"), (1, 50, "size")]
        ]
    )
    plan = plan_focus(people(), ball, [7], 0, 2, (1000, 600), candidates=candidates)
    assert all(p["crop"] is not None for p in plan.values())
    assert plan[1]["ball"]["candidate_count"] == 2
    assert plan[1]["ball"]["selection_evidence"] == "unique_matching_gated_candidate"
    assert plan[1]["ball"]["identity_verified"] is False


@pytest.mark.parametrize(
    "problem", ["two_pass", "missing", "duplicate_index", "mismatch", "bad_reason"]
)
def test_incomplete_or_ambiguous_gate_evidence_never_enables_crop(problem):
    ball = balls()
    ball["candidate_count"] = 2
    candidates = pd.DataFrame(
        [
            dict(frame=f, cand_index=i, u=u, v=190, diameter_px=6, reason=reason)
            for f in range(3)
            for i, u, reason in [(0, 480, "ok"), (1, 50, "size")]
        ]
    )
    if problem == "missing":
        candidates = candidates[~((candidates.frame == 1) & (candidates.cand_index == 1))]
    elif problem == "two_pass":
        candidates.loc[candidates.frame == 1, "reason"] = "ok"
    elif problem == "duplicate_index":
        candidates.loc[candidates.frame == 1, "cand_index"] = 0
    elif problem == "mismatch":
        candidates.loc[(candidates.frame == 1) & (candidates.cand_index == 0), "u"] = 481
    else:
        candidates.loc[(candidates.frame == 1) & (candidates.cand_index == 1), "reason"] = "unknown"
    plan = plan_focus(people(), ball, [7], 0, 2, (1000, 600), candidates=candidates)
    assert plan[1]["crop"] is None
    assert plan[0]["crop"] is not None
    assert plan[2]["crop"] is not None


def test_replay_regions_are_stable_and_short_runs_do_not_flash_or_bridge_gaps():
    from gbf.action_context import stabilize_replay

    focus = {
        f: dict(crop=[100 + 10 * f, 50, 300 + 10 * f, 250], reason="ball_and_people")
        for f in range(10)
    }
    focus[4]["crop"] = None
    focus[4]["reason"] = "ball_missing"
    stabilize_replay(focus, set(range(10)), Fraction(10), (1000, 600))
    # At 10 fps a stable half-second needs 5 distinct source frames.
    assert all(focus[f]["crop"] is None for f in range(5))
    assert focus[0]["reason"] == "replay_component_too_short"
    assert all(focus[f]["crop"] == [150, 50, 390, 250] for f in range(5, 10))
    assert focus[4]["reason"] == "ball_missing"
    assert focus[5]["replay_component"] == focus[9]["replay_component"]


def test_scene_people_remain_available_without_ball_or_local_focus():
    plan = plan_focus(people(), pd.DataFrame(), [7], 0, 2, (1000, 600))
    assert all({p["track_id"] for p in row["scene_people"]} == {7, 8} for row in plan.values())
    assert all(row["crop"] is None for row in plan.values())


def test_replay_crop_does_not_bridge_unscheduled_frames_or_expand_to_full_image():
    from gbf.action_context import stabilize_replay

    focus = {f: dict(crop=[0, 0, 500, 400], reason="ball_and_people") for f in range(11)}
    focus[10]["crop"] = [500, 200, 1000, 600]
    stabilize_replay(focus, set(range(5)) | set(range(6, 11)), Fraction(10), (1000, 600))
    assert all(focus[f]["crop"] == [0, 0, 500, 400] for f in range(5))
    assert all(focus[f]["crop"] is None for f in range(6, 11))
    assert focus[10]["reason"] == "replay_region_too_wide"
    assert focus[5]["crop"] == [0, 0, 500, 400]
