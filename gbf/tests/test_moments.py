"""Tests for gbf.moments on synthetic Halpe-26 skeletons."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gbf import lift3d, measure
from gbf import moments as mo
from gbf.pose import HALPE26_INDEX, HALPE26_KEYPOINTS

# An upright player 180 px tall, ankles at the origin, image v pointing down.
UPRIGHT = {
    "nose": (0, -165),
    "left_eye": (-4, -168),
    "right_eye": (4, -168),
    "left_ear": (-8, -166),
    "right_ear": (8, -166),
    "left_shoulder": (-20, -145),
    "right_shoulder": (20, -145),
    "left_elbow": (-25, -112),
    "right_elbow": (25, -112),
    "left_wrist": (-26, -82),
    "right_wrist": (26, -82),
    "left_hip": (-12, -90),
    "right_hip": (12, -90),
    "left_knee": (-12, -45),
    "right_knee": (12, -45),
    "left_ankle": (-12, 0),
    "right_ankle": (12, 0),
    "head": (0, -180),
    "neck": (0, -150),
    "hip": (0, -90),
    "left_big_toe": (-16, 6),
    "right_big_toe": (16, 6),
    "left_small_toe": (-20, 4),
    "right_small_toe": (20, 4),
    "left_heel": (-10, 3),
    "right_heel": (10, 3),
}
BASE = np.array([UPRIGHT[name] for name in HALPE26_KEYPOINTS], dtype=float)


def _skeleton(
    angle_deg: float = 0.0,
    at: tuple[float, float] = (500.0, 600.0),
    upper_only: bool = False,
    score: float = 0.9,
    scale: float = 1.0,
) -> np.ndarray:
    """Body rotated by angle_deg around the ankles (or only above the hips)."""
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    points = BASE.copy() * scale
    if upper_only:
        pivot = BASE[HALPE26_INDEX["hip"]] * scale
        upper = points[:, 1] < pivot[1]
        points[upper] = (points[upper] - pivot) @ rot.T + pivot
    else:
        points = points @ rot.T
    points += np.asarray(at)
    return np.column_stack([points, np.full(26, score)])


def _pose(rows: list[tuple[int, int, np.ndarray]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": f,
                "track_id": t,
                "keypoints": k.ravel().tolist(),
                "mean_score": float(k[:, 2].mean()),
            }
            for f, t, k in rows
        ]
    )


def _boxes(frames: range, track: int, team: str, u: float, v: float = 600.0) -> list[dict]:
    """Track rows (gbf.measure MeasureTables.tracks columns) of a 200 px box of a player
    whose ankles are at (u, v); the foot point is the box's bottom centre."""
    return [
        {
            "frame": f,
            "track_id": track,
            "x1": u - 30.0,
            "y1": v - 190.0,
            "x2": u + 30.0,
            "y2": v + 10.0,
            "role": "player",
            "team": team,
            "foot_u": u,
            "foot_v": v + 10.0,
            "pitch_x": math.nan,
            "pitch_y": math.nan,
            "in_pitch": False,
        }
        for f in frames
    ]


def _ball_events(rows: list[tuple[int, float, float, int]]) -> pd.DataFrame:
    """gbf.ball events (frame, u, v, nearest track); every one within gbf.ball's at_foot
    radius of its nearest player's box."""
    return pd.DataFrame(
        [
            {
                "frame": f,
                "kind": "direction_change",
                "magnitude": 1.5,
                "u": u,
                "v": v,
                "nearest_track_id": track,
                "at_foot": True,
                "speed_before": 1.0,
                "speed_after": 2.0,
            }
            for f, u, v, track in rows
        ]
    )


def _measured_touches(
    h: dict, tracks: pd.DataFrame, events: pd.DataFrame, pose: pd.DataFrame
) -> dict[int, int]:
    """Per player, the foot_keypoint ball-contact-frame records gbf.measure writes for
    highlight h (what measurements.jsonl holds for it) from the same tables."""
    last = int(max(tracks["frame"].max(), pose["frame"].max()))
    tables = measure.MeasureTables(
        tracks=tracks,
        shots=pd.DataFrame({"shot_id": [0], "start_frame": [0], "end_frame": [last]}),
        ball_events=events,
        pose=pose,
        pitch_frames=frozenset(),
    )
    context = measure.MeasureContext(
        fps="25/1",
        restrictions=("no_share",),
        video_sha256="a" * 64,
        inputs=(("tracks.parquet", "b" * 64),),
        code={"gbf_version": "0.0.1", "git_commit": "unknown", "git_dirty": True},
        registry=measure.load_metric_registry(),
    )
    counts: dict[int, int] = {}
    for record in measure.measure_highlight(h, tables, context):
        if record["metric_id"] != "ball-contact-frame" or record["status"] != "ok":
            continue
        if record["quality"]["match"] == "foot_keypoint":
            counts[record["track_id"]] = counts.get(record["track_id"], 0) + 1
    return counts


def _duel_doc(pair: tuple[int, int], end: int, **extra: object) -> dict:
    return {
        "highlights": [
            {
                "id": "h01",
                "type": "attack_in_box",
                "start_frame": 0,
                "peak_frame": 12,
                "end_frame": end,
                "score": 0.3,
                "title_zh": "原标题",
                "summary_zh": "入选依据：x",
                "track_ids": list(pair),
                "refined_by": "strike_event",
                "cues": {"duel": {"track_ids": list(pair)}},
                **extra,
            }
        ]
    }


def test_fall_detected_with_last_upright_frame_as_onset() -> None:
    rows = [(f, 7, _skeleton(max(0.0, (f - 10) * 10.0))) for f in range(30) if f < 19]
    events = mo.fall_events(_pose(rows), 25)
    assert len(events) == 1
    event = events[0]
    assert (event.track_id, event.onset_frame, event.down_frame) == (7, 12, 15)
    assert event.max_trunk_deg == pytest.approx(80.0, abs=0.5)
    assert event.min_hip_ratio < 0.5


def test_trunk_lean_without_hip_drop_is_not_a_fall() -> None:
    rows = [(f, 3, _skeleton(min(60.0, f * 10.0), upper_only=True)) for f in range(12)]
    angles = mo.body_angles(_pose(rows))
    assert angles["trunk_deg"].max() == pytest.approx(60.0, abs=0.5)
    assert mo.fall_events(_pose(rows), 25) == []


def test_slow_lean_is_not_a_fall() -> None:
    rows = [(f, 3, _skeleton(f * 0.5)) for f in range(160)]
    assert mo.fall_events(_pose(rows), 25) == []


def test_low_scores_give_nan_angles() -> None:
    k = _skeleton(30.0)
    k[HALPE26_INDEX["left_hip"], 2] = 0.1
    angles = mo.body_angles(_pose([(0, 1, k), (1, 1, _skeleton(30.0))]))
    assert math.isnan(angles.loc[0, "trunk_deg"])
    assert angles.loc[1, "trunk_deg"] == pytest.approx(30.0, abs=0.01)


def test_contact_intervals_merge_short_gaps() -> None:
    gaps = {f: 300.0 - 25.0 * f for f in range(12)}
    gaps.update({12: 30.0, 13: 2.0, 14: 1.0, 15: 60.0, 16: 0.0, 17: 1.0, 18: 200.0})
    rows = []
    for f, gap in gaps.items():
        rows.append((f, 1, _skeleton(at=(500.0, 600.0))))
        rows.append((f, 2, _skeleton(at=(500.0 + 52.0 + gap, 600.0))))
    series = mo.contact_series(_pose(rows), 1, 2)
    assert mo.contact_intervals(series) == [(13, 17)]
    assert mo.contact_intervals(series, gap_frames=1) == [(13, 14), (16, 17)]


def test_duel_timeline_orders_contact_release_and_fall() -> None:
    rows, ball = [], []
    for f in range(40):
        rows.append((f, 1, _skeleton(at=(500.0, 600.0))))
        angle = 0.0 if f < 28 else min(80.0, (f - 27) * 15.0)
        rows.append((f, 2, _skeleton(angle, at=(500.0 + (60.0 if f >= 5 else 200.0), 600.0))))
        u = 530.0 if f <= 20 else 530.0 - 120.0 * (f - 20)
        ball.append({"frame": f, "u": u, "v": 604.0, "status": "detected"})
    timeline = mo.duel_timeline(
        _pose(rows), (1, 2), 0, 39, 25, ball_track=pd.DataFrame(ball), names={2: "防守球员"}
    )
    kinds = [(m.frame, m.kind) for m in timeline]
    # The falling player rotates away from the other one, which ends the contact.
    assert kinds == [
        (5, "contact_start"),
        (20, "ball_release"),
        (28, "fall_onset"),
        (28, "contact_end"),
        (30, "on_ground"),
    ]
    # What contact_series measures: keypoints close in the image, not a touch.
    assert timeline[0].text_zh == "轨迹 1与防守球员身体贴近（画面平面）"
    assert timeline[3].text_zh == "身体贴近结束"
    # What fall_events measures: the trunk starts to tilt here (no claim about balance).
    assert timeline[2].text_zh == "防守球员躯干开始倾倒"
    assert timeline[2].evidence["down_frame"] == 30
    assert timeline[4].text_zh == "防守球员躯干倾倒超过 45°（画面平面）"


def test_a_ball_seen_again_only_much_later_did_not_leave_fast() -> None:
    rows, ball = [], []
    for f in range(40):
        rows.append((f, 1, _skeleton(at=(500.0, 600.0))))
        rows.append((f, 2, _skeleton(at=(900.0, 600.0))))
    for f in range(40):
        if f <= 12:
            ball.append({"frame": f, "u": 515.0, "v": 604.0, "status": "detected"})
        elif f >= 25:  # lost for half a second, then far from both players
            ball.append({"frame": f, "u": 200.0, "v": 604.0, "status": "detected"})
        else:
            ball.append({"frame": f, "u": math.nan, "v": math.nan, "status": "missing"})
    pose = _pose(rows)
    timeline = mo.duel_timeline(pose, (1, 2), 0, 39, 25, ball_track=pd.DataFrame(ball))
    assert timeline == []
    # Seen again within RELEASE_WITHIN_S, far away: the ball did leave fast.
    soon = 12 + round(mo.RELEASE_WITHIN_S * 25)
    for row in ball:
        if 12 < row["frame"] < 25:
            far = row["frame"] >= soon
            row.update(u=200.0 if far else math.nan, v=604.0 if far else math.nan)
            row["status"] = "detected" if far else "missing"
    timeline = mo.duel_timeline(pose, (1, 2), 0, 39, 25, ball_track=pd.DataFrame(ball))
    assert [(m.frame, m.kind) for m in timeline] == [(12, "ball_away")]
    assert timeline[0].evidence["away_frame"] == soon


#: head to big toe of _skeleton(), the body height _ball_release divides by
BODY_PX = 186.0


#: player 1's right big toe in _release_scene
TOE = (516.0, 606.0)


def _release_scene(
    after: dict[int, tuple[float, float]], rest: tuple[float, float] = (520.0, 604.0)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Player 2 comes into contact with player 1 at frame 5; the ball rests at (u, v) rest,
    by default 4 px from player 1's right big toe, up to frame 20, then is detected at
    after[f] (u, v) only."""
    rows, ball = [], []
    for f in range(40):
        rows.append((f, 1, _skeleton(at=(500.0, 600.0))))
        rows.append((f, 2, _skeleton(at=(500.0 + (60.0 if f >= 5 else 200.0), 600.0))))
        uv = rest if f <= 20 else after.get(f)
        if uv is None:
            ball.append({"frame": f, "u": math.nan, "v": math.nan, "status": "missing"})
        else:
            ball.append({"frame": f, "u": uv[0], "v": uv[1], "status": "detected"})
    return _pose(rows), pd.DataFrame(ball)


def test_a_ball_rolling_off_slowly_after_contact_is_no_release() -> None:
    # One body height per second straight down the image, seen on every frame: it passes
    # RELEASE_AWAY_BODY_H within RELEASE_WITHIN_S of its last frame at the feet, too slowly.
    step = BODY_PX / 25
    pose, ball = _release_scene({f: (520.0, 604.0 + step * (f - 20)) for f in range(21, 40)})
    timeline = mo.duel_timeline(pose, (1, 2), 0, 39, 25, ball_track=ball)
    assert [(m.frame, m.kind) for m in timeline] == [(5, "contact_start")]
    h = mo.add_key_moments(_duel_doc((1, 2), 39), pose, 25, ball_track=ball)["highlights"][0]
    assert (h["type"], h["peak_frame"], h["title_zh"]) == ("attack_in_box", 12, "原标题")


def test_a_ball_leaving_fast_needs_only_one_detection() -> None:
    # Not detected for frames 21-24, then 153 px from the toe at frame 25: 0.8 body heights
    # further in 0.2 s, about 4 body heights per second.
    pose, ball = _release_scene({f: (520.0, 759.0) for f in range(25, 40)})
    timeline = mo.duel_timeline(pose, (1, 2), 0, 39, 25, ball_track=ball)
    assert [(m.frame, m.kind) for m in timeline] == [(5, "contact_start"), (20, "ball_release")]
    evidence = timeline[1].evidence
    assert (evidence["nearest_track_id"], evidence["away_frame"]) == (1, 25)
    assert evidence["away_speed_bh_s"] == pytest.approx(4.0, abs=0.05)
    assert evidence["away_speed_bh_s"] >= mo.RELEASE_MIN_SPEED_BH_S
    h = mo.add_key_moments(_duel_doc((1, 2), 39), pose, 25, ball_track=ball)["highlights"][0]
    assert (h["type"], h["peak_frame"], h["key_focus_track"]) == ("duel_tackle", 20, 1)


@pytest.mark.parametrize(("speed", "released"), [(2.0, False), (3.0, True)])
def test_a_ball_leaves_the_feet_fast_between_two_and_three_body_heights_per_second(
    speed: float, released: bool
) -> None:
    # The ball rolls straight down the image from player 1's toe at speed body heights per
    # second, seen on every frame: its distance from the toe grows evenly, so the speed
    # measured from its last frame at the feet (21) to its first one away is speed. Two is
    # too slow, three is a release (RELEASE_MIN_SPEED_BH_S lies between them).
    step = speed * BODY_PX / 25
    pose, ball = _release_scene(
        {f: (TOE[0], TOE[1] + step * (f - 20)) for f in range(21, 40)}, rest=TOE
    )
    timeline = mo.duel_timeline(pose, (1, 2), 0, 39, 25, ball_track=ball)
    releases = [
        (m.frame, m.evidence["away_frame"], m.evidence["away_speed_bh_s"])
        for m in timeline
        if m.kind == "ball_release"
    ]
    assert releases == ([(21, 23, pytest.approx(speed))] if released else [])
    assert [m.kind for m in timeline if m.kind != "ball_release"] == ["contact_start"]


def test_contact_that_lasts_to_the_cut_has_no_end() -> None:
    rows = []
    for f in range(20):
        rows.append((f, 1, _skeleton(at=(500.0, 600.0))))
        rows.append((f, 2, _skeleton(at=(560.0, 600.0))))
    timeline = mo.duel_timeline(_pose(rows), (1, 2), 0, 19, 25)
    assert [(m.frame, m.kind) for m in timeline] == [(0, "contact_start")]


def test_missing_columns_are_reported() -> None:
    with pytest.raises(mo.MomentsInputError, match="keypoints"):
        mo.body_angles(pd.DataFrame({"frame": [0], "track_id": [1]}))


def test_touches_are_the_ball_contacts_gbf_measure_confirms() -> None:
    rows = [(f, 1, _skeleton(at=(500.0, 600.0))) for f in range(30)]
    rows += [(f, 2, _skeleton(at=(700.0, 600.0))) for f in range(30) if f != 20]
    pose = _pose(rows)
    tracks = pd.DataFrame(
        _boxes(range(30), 1, "left", 500.0) + _boxes(range(30), 2, "right", 700.0)
    )
    events = _ball_events(
        [
            (8, 716.0, 606.0, 2),  # on player 2's right big toe
            (12, 700.0, 560.0, 2),  # 0.21 box heights above player 2's ankles: no touch
            (15, 488.0, 600.0, 1),  # on player 1's left ankle
            (18, 684.0, 606.0, 2),  # on player 2's left big toe
            (20, 700.0, 610.0, 2),  # at player 2's box foot point, but he has no pose here
        ]
    )
    doc = _duel_doc((1, 2), 29)
    h = mo.add_key_moments(doc, pose, 25, tracks=tracks, touches=events)["highlights"][0]
    assert [(m["frame"], m["kind"], m["track_ids"]) for m in h["key_moments"]] == [
        (8, "touch", [2]),
        (15, "touch", [1]),
        (18, "touch", [2]),
    ]
    assert h["key_moments"][0]["evidence"]["keypoint"] == "right_big_toe"
    assert h["key_moments"][0]["evidence"]["distance_box_h"] == pytest.approx(0.0)
    assert [
        (t["track_id"], t["count"], t["frames"], t["mean_interval_s"]) for t in h["touches"]
    ] == [
        (1, 1, [15], None),
        (2, 2, [8, 18], 0.4),
    ]
    # The measurement table's criterion (gbf.measure.match_contacts) on the same tables;
    # its box fallback (a player without a pose) is not a confirmed touch.
    shots = pd.DataFrame({"shot_id": [0], "start_frame": [0], "end_frame": [29]})
    tables = measure.MeasureTables(
        tracks=tracks, shots=shots, ball_events=events, pose=pose, pitch_frames=frozenset()
    )
    matches, examined = measure.match_contacts(tables, 0, 29)
    assert examined == 5
    assert [(m.frame, m.track_id, m.method) for m in matches] == [
        (8, 2, "foot_keypoint"),
        (15, 1, "foot_keypoint"),
        (18, 2, "foot_keypoint"),
        (20, 2, "foot_point_fallback"),
    ]
    assert [(m.frame, m.track_id) for m in mo.confirmed_touches(events, tracks, pose)] == [
        (8, 2),
        (15, 1),
        (18, 2),
    ]


def test_touch_counts_cover_the_window_like_the_measurement_table() -> None:
    # Player 1 dribbles from frame 0; the opponent 2 is seen only from frame 20 on.
    rows = [(f, 1, _skeleton(at=(500.0, 600.0))) for f in range(40)]
    rows += [(f, 2, _skeleton(at=(700.0, 600.0))) for f in range(20, 40)]
    pose = _pose(rows)
    tracks = pd.DataFrame(
        _boxes(range(40), 1, "left", 500.0) + _boxes(range(20, 40), 2, "right", 700.0)
    )
    events = _ball_events([(5, 488.0, 600.0, 1), (10, 488.0, 600.0, 1), (25, 488.0, 600.0, 1)])
    doc = _duel_doc((1, 2), 39, type="duel_tackle")
    h = mo.add_key_moments(doc, pose, 25, tracks=tracks, touches=events)["highlights"][0]
    # Every touch of the window counts, also where only one of the pair has a pose.
    assert [(t["track_id"], t["count"], t["frames"]) for t in h["touches"]] == [(1, 3, [5, 10, 25])]
    assert [m["frame"] for m in h["key_moments"] if m["kind"] == "touch"] == [5, 10, 25]
    assert _measured_touches(h, tracks, events, pose) == {1: 3}


def test_touches_need_the_track_table() -> None:
    rows = [(f, t, _skeleton(at=(400.0 + 200.0 * t, 600.0))) for f in range(10) for t in (1, 2)]
    events = _ball_events([(5, 588.0, 600.0, 1)])
    with pytest.raises(mo.MomentsInputError, match="track table"):
        mo.add_key_moments(_duel_doc((1, 2), 9), _pose(rows), 25, touches=events)


def test_small_players_give_touches_but_no_contact_or_fall() -> None:
    rows = []
    for f in range(40):
        angle = 0.0 if f < 20 else min(80.0, (f - 19) * 15.0)
        rows.append((f, 1, _skeleton(at=(500.0, 600.0), scale=0.4)))
        rows.append((f, 2, _skeleton(angle, at=(505.0, 600.0), scale=0.4)))
    touch = measure.ContactMatch(
        10, 1, "foot_keypoint", 0.05, "left_ankle", 80.0, {"kind": "speed_up"}
    )
    elsewhere = measure.ContactMatch(12, 7, "foot_keypoint", 0.05, "left_ankle", 80.0, {})
    fallback = measure.ContactMatch(14, 2, "foot_point_fallback", 0.3, "", 80.0, {})
    timeline = mo.duel_timeline(
        _pose(rows), (1, 2), 0, 39, 25, touches=[touch, elsewhere, fallback]
    )
    assert [(m.frame, m.kind) for m in timeline] == [(10, "touch")]


def _track_rows(frames: range, track: int, team: str, u: float, h: float = 200.0) -> list[dict]:
    return [
        {
            "frame": f,
            "track_id": track,
            "team": team,
            "role": "player",
            "foot_u": u,
            "foot_v": 800.0,
            "y1": 800.0 - h,
            "y2": 800.0,
        }
        for f in frames
    ]


def test_joined_strike_subject_gets_its_nearest_opponent() -> None:
    tracks = pd.DataFrame(
        _track_rows(range(50, 80), 7, "right", 900.0)
        + _track_rows(range(50, 80), 8, "left", 980.0)  # opponent, 0.4 box heights away
        + _track_rows(range(50, 80), 9, "right", 950.0)  # teammate, closer
        + _track_rows(range(50, 60), 10, "left", 910.0)  # opponent, too briefly seen
        + _track_rows(range(50, 80), 11, "left", 1500.0)  # opponent, too far
    )
    doc = {
        "highlights": [
            {
                "id": "h01",
                "start_frame": 0,
                "peak_frame": 20,
                "end_frame": 79,
                "track_ids": [3, 7],
                "cues": {
                    "cut_emphasis": {
                        "joined_shot_event": {"cue": "strike", "frame": 70, "track_ids": [7]}
                    }
                },
            }
        ]
    }
    out = mo.add_duel_partners(doc, tracks)
    h = out["highlights"][0]
    joined = h["cues"]["cut_emphasis"]["joined_shot_event"]
    assert joined["partner_track_id"] == 8
    assert joined["partner_distance_box_h"] == pytest.approx(0.4)
    assert h["track_ids"] == [3, 7, 8]
    assert mo._duel_pairs(h)[0] == (7, 8)
    assert doc["highlights"][0]["track_ids"] == [3, 7], "the input is not changed"


def test_windows_meeting_at_cuts_are_merged_with_every_part_listed() -> None:
    parts = [
        ("h01", 0, 50, 99, 0.3),
        ("h02", 100, 150, 199, 0.5),
        ("h03", 200, 250, 299, 0.4),
        ("h04", 300, 320, 340, 0.9),  # right after h03, but no cut there
    ]
    doc = {
        "highlights": [
            {
                "id": hid,
                "type": "duel_tackle",
                "start_frame": start,
                "peak_frame": peak,
                "end_frame": end,
                "score": score,
                "title_zh": f"标题 {hid}",
                "track_ids": [number],
            }
            for number, (hid, start, peak, end, score) in enumerate(parts)
        ]
    }
    out = mo.merge_adjacent_highlights(doc, [100, 200], 25)
    merged, alone = out["highlights"]
    assert (merged["id"], merged["start_frame"], merged["peak_frame"], merged["end_frame"]) == (
        "h01",
        0,
        150,
        299,
    )
    # The parts keep the ids they had before the merge, marked as such: the merge renumbers
    # the highlights, and a bare "h02" would now name another highlight (the one at 300).
    assert merged["merged_from"] == ["detect-h01", "detect-h02", "detect-h03"]
    assert [
        (p["id"], p["start_frame"], p["peak_frame"], p["end_frame"], p["score"])
        for p in merged["merged_parts"]
    ] == [
        ("detect-h01", 0, 50, 99, 0.3),
        ("detect-h02", 100, 150, 199, 0.5),
        ("detect-h03", 200, 250, 299, 0.4),
    ]
    assert merged["track_ids"] == [1, 0, 2]
    assert (alone["id"], alone["start_frame"], "merged_from" in alone) == ("h02", 300, False)
    final = {h["id"] for h in out["highlights"]}
    assert final.isdisjoint(merged["merged_from"])
    assert final.isdisjoint(p["id"] for p in merged["merged_parts"])


def test_merged_ids_never_name_a_renumbered_highlight() -> None:
    # h01 and h02 meet at a cut and merge; h03 stays alone and becomes h02.
    doc = {
        "highlights": [
            {
                "id": hid,
                "type": "duel_tackle",
                "start_frame": start,
                "peak_frame": start + 20,
                "end_frame": start + 49,
                "score": 0.3,
                "title_zh": f"t{hid}",
                "track_ids": [1],
            }
            for hid, start in (("h01", 0), ("h02", 50), ("h03", 200))
        ]
    }
    out = mo.merge_adjacent_highlights(doc, [50], 25)
    first, second = out["highlights"]
    assert (first["id"], second["id"], second["title_zh"]) == ("h01", "h02", "th03")
    assert first["merged_from"] == ["detect-h01", "detect-h02"]
    assert "h02" not in first["merged_from"]
    # A second pass keeps the marked ids as they are.
    again = mo.merge_adjacent_highlights(out, [50], 25)
    assert again["highlights"][0]["merged_from"] == ["detect-h01", "detect-h02"]


def test_untracked_boxes_are_nobody_s_partner() -> None:
    tracks = pd.DataFrame(
        _track_rows(range(50, 80), 7, "right", 900.0)
        + _track_rows(range(50, 80), -1, "left", 920.0)  # boxes without a track, closest
        + _track_rows(range(50, 80), 8, "left", 980.0)
    )
    joined = {"cue": "strike", "frame": 70, "track_ids": [7]}
    doc = {
        "highlights": [
            {
                "id": "h01",
                "start_frame": 50,
                "peak_frame": 60,
                "end_frame": 79,
                "track_ids": [7],
                "cues": {"cut_emphasis": {"joined_shot_event": joined}},
            }
        ]
    }
    h = mo.add_duel_partners(doc, tracks)["highlights"][0]
    assert h["cues"]["cut_emphasis"]["joined_shot_event"]["partner_track_id"] == 8
    assert h["track_ids"] == [7, 8]
    stale = {"cues": {"cut_emphasis": {"joined_shot_event": {**joined, "partner_track_id": -1}}}}
    assert mo._duel_pairs(stale) == []


def test_ball_leaving_feet_without_contact_changes_nothing_but_is_listed() -> None:
    rows, ball = [], []
    for f in range(30):
        rows.append((f, 1, _skeleton(at=(500.0, 600.0))))
        rows.append((f, 2, _skeleton(at=(900.0, 600.0))))  # far apart: no contact
        u = 515.0 if f <= 12 else 515.0 - 120.0 * (f - 12)  # away from player 2
        ball.append({"frame": f, "u": u, "v": 604.0, "status": "detected"})
    pose = _pose(rows)
    out = mo.add_key_moments(_duel_doc((1, 2), 29), pose, 25, ball_track=pd.DataFrame(ball))
    h = out["highlights"][0]
    assert [(m["frame"], m["kind"]) for m in h["key_moments"]] == [(12, "ball_away")]
    assert h["key_moments"][0]["text_zh"] == "球从轨迹 1 球员脚下快速离开"
    assert h["key_moments"][0]["evidence"]["nearest_track_id"] == 1
    assert (h["type"], h["title_zh"], h["peak_frame"]) == ("attack_in_box", "原标题", 12)
    assert "previous_peak" not in h
    # Not a duel's key frame: the focus stays with the player gbf.highlights chose.
    assert (h["key_pair"], h["key_focus_track"]) == ([1, 2], None)


def test_a_fall_without_contact_keeps_the_ball_out_of_the_duel() -> None:
    rows, ball = [], []
    for f in range(40):
        rows.append((f, 1, _skeleton(at=(500.0, 600.0))))
        angle = 0.0 if f < 28 else min(80.0, (f - 27) * 15.0)
        rows.append((f, 2, _skeleton(angle, at=(700.0, 600.0))))  # apart: no contact
        u = 520.0 if f <= 20 else 520.0 - 120.0 * (f - 20)
        ball.append({"frame": f, "u": u, "v": 604.0, "status": "detected"})
    out = mo.add_key_moments(_duel_doc((1, 2), 39), _pose(rows), 25, ball_track=pd.DataFrame(ball))
    h = out["highlights"][0]
    assert [(m["frame"], m["kind"]) for m in h["key_moments"]] == [
        (20, "ball_away"),
        (28, "fall_onset"),
        (30, "on_ground"),
    ]
    assert (h["type"], h["peak_frame"], h["key_focus_track"]) == ("duel_tackle", 28, 2)
    assert "球脱离争抢" not in h["title_zh"]
    # The two never came close, and the fall is the trunk starting to tilt on the key
    # frame, then passing 45° two frames later.
    assert h["title_zh"].endswith(
        "轨迹 1 球员与轨迹 2 球员争抢：轨迹 2 球员躯干开始倾倒，随后过 45°（画面平面，第 30 帧）"
    )
    assert "贴身" not in h["title_zh"]
    assert "对抗" not in h["title_zh"], "one word for the duel: 争抢, as in the type name"


def _contact_and_fall(frames: int = 40) -> list[tuple[int, int, np.ndarray]]:
    """Player 1 stands; player 2 comes into contact at frame 5 and falls from frame 28."""
    rows = []
    for f in range(frames):
        rows.append((f, 1, _skeleton(at=(500.0, 600.0))))
        angle = 0.0 if f < 28 else min(80.0, (f - 27) * 15.0)
        rows.append((f, 2, _skeleton(angle, at=(500.0 + (60.0 if f >= 5 else 200.0), 600.0))))
    return rows


def test_a_duel_records_its_pair_the_fallen_player_and_the_detected_touches() -> None:
    tracks = pd.DataFrame(
        _boxes(range(40), 1, "left", 500.0) + _boxes(range(40), 2, "right", 560.0)
    )
    events = _ball_events([(10, 488.0, 600.0, 1), (15, 488.0, 600.0, 1)])  # player 1's ankle
    doc = _duel_doc((1, 2), 39, merged_from=["h01", "h02"])
    h = mo.add_key_moments(doc, _pose(_contact_and_fall()), 25, tracks=tracks, touches=events)[
        "highlights"
    ][0]
    assert [(m["frame"], m["kind"]) for m in h["key_moments"]] == [
        (5, "contact_start"),
        (10, "touch"),
        (15, "touch"),
        (28, "fall_onset"),
        (28, "contact_end"),
        (30, "on_ground"),
    ]
    assert (h["type"], h["peak_frame"], h["refined_by"]) == ("duel_tackle", 28, "pose_key_moment")
    assert (h["key_pair"], h["key_focus_track"]) == ([1, 2], 2)
    assert h["title_zh"].endswith(
        "左侧球队球员与右侧球队球员贴身争抢：右侧球队球员躯干开始倾倒，"
        "随后过 45°（画面平面，第 30 帧）"
    )
    # The texts name the two players with their track ids.
    pair = "左侧球队球员（轨迹 1）与右侧球队球员（轨迹 2）"
    assert f"（第 5 帧）{pair}身体贴近（画面平面）；" in h["summary_zh"]
    assert "（第 28 帧）右侧球队球员（轨迹 2）躯干开始倾倒；" in h["summary_zh"]
    assert "（第 28 帧）身体贴近结束；" in h["summary_zh"]
    assert "接触" not in h["title_zh"] + h["summary_zh"], "closeness in the image is no touch"
    assert h["summary_zh"].startswith("关键时刻（由原始画面上的 2D 姿态和球的位置逐帧确定")
    assert (
        "在第 10–15 帧之间检测到左侧球队球员（轨迹 1）触球 2 次（球速突变且脚在球边；"
        "轻触可能漏检），检测到的触球平均间隔 0.20 秒。"
    ) in h["summary_zh"]
    assert "（第 0–39 帧，由 2 段合并：切镜前后相接的镜头）" in h["summary_zh"]
    # The label of the touch count stays the team's (the data card adds the track).
    assert [(t["track_id"], t["label_zh"]) for t in h["touches"]] == [(1, "左侧球队球员")]


def test_a_duel_won_by_the_ball_leaving_focuses_on_its_nearest_player() -> None:
    rows, ball = _contact_and_fall(), []
    for f in range(40):
        u = 520.0 if f <= 20 else 520.0 - 120.0 * (f - 20)  # at player 1's right toe, then away
        ball.append({"frame": f, "u": u, "v": 604.0, "status": "detected"})
    doc = _duel_doc((2, 1), 39)
    h = mo.add_key_moments(doc, _pose(rows), 25, ball_track=pd.DataFrame(ball))["highlights"][0]
    assert [(m["frame"], m["kind"]) for m in h["key_moments"]][:3] == [
        (5, "contact_start"),
        (20, "ball_release"),
        (28, "fall_onset"),
    ]
    assert h["key_moments"][1]["text_zh"] == "球离开两人脚下"
    assert (h["peak_frame"], h["key_pair"], h["key_focus_track"]) == (20, [2, 1], 1)
    # The title is about the key frame: the trunk passes 45° ten frames later.
    assert h["title_zh"] == (
        "00:00.80（第 20 帧）轨迹 2 球员与轨迹 1 球员贴身争抢："
        "球脱离争抢，轨迹 2 球员躯干随后倾倒过 45°（画面平面，第 30 帧）"
    )


def test_a_duel_without_the_ball_says_it_is_read_from_pose_alone() -> None:
    h = mo.add_key_moments(_duel_doc((1, 2), 39), _pose(_contact_and_fall()), 25)["highlights"][0]
    assert h["title_zh"].endswith(
        "轨迹 1 球员与轨迹 2 球员贴身争抢：轨迹 2 球员躯干开始倾倒，"
        "随后过 45°（画面平面，第 30 帧）"
    )
    assert h["summary_zh"].startswith(
        "关键时刻（由原始画面上的 2D 姿态逐帧确定，画面平面，未验证）"
    )
    assert "（第 28 帧）轨迹 2 球员躯干开始倾倒；" in h["summary_zh"]
    assert "倒地" not in h["title_zh"] + h["summary_zh"]
    assert "失去平衡" not in h["summary_zh"]


def _falling(f: int, start: int) -> float:
    """Trunk angle of a player who leaves upright at frame start (45° two frames later)."""
    return 0.0 if f < start else min(80.0, (f - start + 1) * 15.0)


@pytest.mark.parametrize(
    ("fall", "words"),
    [
        (10, "球脱离争抢，轨迹 2 球员躯干此前倾倒过 45°（画面平面，第 12 帧）"),
        (18, "球脱离争抢，轨迹 2 球员躯干倾倒过 45°（画面平面）"),
    ],
    ids=["earlier", "on_the_key_frame"],
)
def test_a_title_places_a_fall_against_its_key_frame(fall: int, words: str) -> None:
    # Player 2 falls from frame fall (the trunk passes 45° two frames later) and the ball
    # leaves player 1's feet at 20: the key frame is the release, and the title says when
    # the trunk passed 45° unless it was on the key frame.
    rows, ball = [], []
    for f in range(40):
        rows.append((f, 1, _skeleton(at=(500.0, 600.0))))
        rows.append((f, 2, _skeleton(_falling(f, fall), at=(560.0 if f >= 5 else 700.0, 600.0))))
        u = 520.0 if f <= 20 else 520.0 - 120.0 * (f - 20)
        ball.append({"frame": f, "u": u, "v": 604.0, "status": "detected"})
    doc = _duel_doc((1, 2), 39)
    h = mo.add_key_moments(doc, _pose(rows), 25, ball_track=pd.DataFrame(ball))["highlights"][0]
    kinds = ("fall_onset", "on_ground", "ball_release")
    frames = {m["kind"]: m["frame"] for m in h["key_moments"] if m["kind"] in kinds}
    assert frames == {"fall_onset": fall, "on_ground": fall + 2, "ball_release": 20}
    assert h["peak_frame"] == 20
    assert h["title_zh"].endswith(words)


def _two_duels(fall_a: int | None, fall_b: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pose and tracks of two duels in contact from frame 5: players 1 and 2 (2 falls at
    fall_a), players 3 and 4 (4 falls at fall_b); None: no fall."""
    rows, boxes = [], []
    for first, second, u, fall in ((1, 2, 500.0, fall_a), (3, 4, 1500.0, fall_b)):
        for f in range(40):
            angle = 0.0 if fall is None else _falling(f, fall)
            rows.append((f, first, _skeleton(at=(u, 600.0))))
            rows.append((f, second, _skeleton(angle, at=(u + (60.0 if f >= 5 else 200.0), 600.0))))
        boxes += _boxes(range(40), first, "left", u) + _boxes(range(40), second, "right", u + 60.0)
    return _pose(rows), pd.DataFrame(boxes)


def _joined_duel_doc(joined: tuple[int, int], cue: tuple[int, int]) -> dict:
    """A duel_tackle whose joined shot names the duel joined and whose duel cue names cue."""
    event = {"cue": "duel", "frame": 10, "track_ids": list(joined)}
    cues = {"duel": {"track_ids": list(cue)}, "cut_emphasis": {"joined_shot_event": event}}
    return _duel_doc(cue, 39, type="duel_tackle", track_ids=[*joined, *cue], cues=cues)


def test_a_later_duel_pair_is_kept_only_with_more_moments_that_can_move_the_peak() -> None:
    # Both pairs fall once; the duel cue's pair (3, 4) also has three touches. The first
    # candidate, the joined shot's duel, stays: touches and other moments do not rank pairs.
    pose, tracks = _two_duels(28, 32)
    events = _ball_events([(10, 1488.0, 600.0, 3), (14, 1488.0, 600.0, 3), (18, 1488.0, 600.0, 3)])
    doc = _joined_duel_doc((1, 2), (3, 4))
    assert mo._duel_pairs(doc["highlights"][0]) == [(1, 2), (3, 4)]
    h = mo.add_key_moments(doc, pose, 25, tracks=tracks, touches=events)["highlights"][0]
    assert (h["key_pair"], h["peak_frame"], h["key_focus_track"], h["touches"]) == (
        [1, 2],
        28,
        2,
        [],
    )
    # A later pair with more of them (a fall where the first pair has none) is kept.
    pose, tracks = _two_duels(None, 32)
    h = mo.add_key_moments(doc, pose, 25, tracks=tracks, touches=events)["highlights"][0]
    assert (h["key_pair"], h["peak_frame"], h["key_focus_track"]) == ([3, 4], 32, 4)
    assert [t["count"] for t in h["touches"]] == [3]


def test_a_later_duel_pair_with_more_moments_of_other_kinds_ties() -> None:
    # Pair (1, 2): close from frame 5 to the end, the ball leaves player 1's feet fast at 20
    # (two moments). Pair (3, 4): close in frames 5-20, then 4 falls from frame 32 (four
    # moments). One moment that can move the peak each: a tie, which the first candidate
    # wins, however many moments of other kinds the later one has.
    rows, ball = [], []
    for f in range(40):
        rows.append((f, 1, _skeleton(at=(500.0, 600.0))))
        rows.append((f, 2, _skeleton(at=(560.0 if f >= 5 else 700.0, 600.0))))
        rows.append((f, 3, _skeleton(at=(1500.0, 600.0))))
        u4 = 1560.0 if 5 <= f <= 20 else 1700.0  # next to player 3 in frames 5-20 only
        rows.append((f, 4, _skeleton(_falling(f, 32), at=(u4, 600.0))))
        u = 520.0 if f <= 20 else 520.0 - 120.0 * (f - 20)
        ball.append({"frame": f, "u": u, "v": 604.0, "status": "detected"})
    pose, ball_track = _pose(rows), pd.DataFrame(ball)
    later = mo.duel_timeline(pose[pose["track_id"].isin([3, 4])], (3, 4), 0, 39, 25)
    assert [(m.frame, m.kind) for m in later] == [
        (5, "contact_start"),
        (20, "contact_end"),
        (32, "fall_onset"),
        (34, "on_ground"),
    ]
    doc = _joined_duel_doc((1, 2), (3, 4))
    h = mo.add_key_moments(doc, pose, 25, ball_track=ball_track)["highlights"][0]
    assert [(m["frame"], m["kind"]) for m in h["key_moments"]] == [
        (5, "contact_start"),
        (20, "ball_release"),
    ]
    assert (h["key_pair"], h["peak_frame"], h["key_focus_track"]) == ([1, 2], 20, 1)


def test_a_ball_leaving_a_player_s_feet_ranks_a_later_duel_pair() -> None:
    # The first candidate (1, 2) has no key moment; the later one (3, 4) only the ball
    # leaving player 3's feet fast while the two stay apart (ball_away), which ranks it
    # first. It changes nothing else: no duel was found.
    rows, ball = [], []
    for f in range(40):
        for track, u in ((1, 300.0), (2, 700.0), (3, 1100.0), (4, 1500.0)):  # far apart
            rows.append((f, track, _skeleton(at=(u, 600.0))))
        v = 604.0 if f <= 20 else 604.0 + 60.0 * (f - 20)  # down the image, away from all
        ball.append({"frame": f, "u": 1120.0, "v": v, "status": "detected"})
    doc = _joined_duel_doc((1, 2), (3, 4))
    h = mo.add_key_moments(doc, _pose(rows), 25, ball_track=pd.DataFrame(ball))["highlights"][0]
    assert [(m["frame"], m["kind"], m["track_ids"]) for m in h["key_moments"]] == [
        (20, "ball_away", [3])
    ]
    assert (h["key_pair"], h["type"], h["peak_frame"], h["title_zh"]) == (
        [3, 4],
        "duel_tackle",
        12,
        "原标题",
    )
    assert "previous_peak" not in h


def test_pair_and_focus_of_highlights_the_key_moments_did_not_change() -> None:
    rows = [(f, t, _skeleton(at=(300.0 + 200.0 * t, 600.0))) for f in range(20) for t in (1, 2)]
    pose = _pose(rows)
    h = mo.add_key_moments(_duel_doc((1, 2), 19), pose, 25)["highlights"][0]
    assert (h["key_moments"], h["touches"], h["key_pair"], h["key_focus_track"]) == (
        [],
        [],
        None,
        None,
    )
    # A duel found by gbf.highlights whose pair only touched the ball: its key frame is
    # still the duel's, so the first of the pair is the focus.
    tracks = pd.DataFrame(
        _boxes(range(20), 1, "left", 500.0) + _boxes(range(20), 2, "right", 700.0)
    )
    events = _ball_events([(8, 716.0, 606.0, 2)])
    doc = _duel_doc((1, 2), 19, type="duel_tackle")
    h = mo.add_key_moments(doc, pose, 25, tracks=tracks, touches=events)["highlights"][0]
    assert [(m["frame"], m["kind"]) for m in h["key_moments"]] == [(8, "touch")]
    assert (h["type"], h["peak_frame"], h["key_pair"], h["key_focus_track"]) == (
        "duel_tackle",
        12,
        [1, 2],
        1,
    )


def test_the_key_focus_is_always_a_player_on_the_key_frame() -> None:
    rows = []
    for f in range(40):  # the main (wide) shot: the duel cue's pair 1 and 2, small
        rows.append((f, 1, _skeleton(at=(500.0, 600.0), scale=0.4)))
        rows.append((f, 2, _skeleton(at=(540.0, 600.0), scale=0.4)))
    for f in range(40, 80):  # the joined close shot: the duel as tracks 11 and 12
        rows.append((f, 11, _skeleton(scale=3.0)))
        rows.append((f, 12, _skeleton(scale=3.0, at=(680.0 if 50 <= f <= 60 else 800.0, 600.0))))
    pose = _pose(rows)
    tracks = pd.DataFrame(
        _boxes(range(40), 1, "left", 500.0)
        + _boxes(range(40), 2, "right", 540.0)
        + _boxes(range(40, 80), 11, "left", 500.0)
        + _boxes(range(40, 80), 12, "right", 800.0)
    )
    event = {"cue": "duel", "frame": 55, "track_ids": [11, 12]}
    cues = {
        "duel": {"track_ids": [1, 2]},
        "cut_emphasis": {"cut_frame": 40, "joined_shot_event": event},
    }
    doc = _duel_doc((1, 2), 79, type="duel_tackle", peak_frame=20, cues=cues)
    h = mo.add_key_moments(doc, pose, 25, tracks=tracks)["highlights"][0]
    assert [(m["frame"], m["kind"]) for m in h["key_moments"]] == [
        (50, "contact_start"),
        (60, "contact_end"),
    ]
    # The joined shot's pair is analysed, but its track ids do not exist on the key frame
    # gbf.highlights chose in the main shot (ids are split at cuts): no key focus.
    assert (h["type"], h["peak_frame"], h["key_pair"], h["key_focus_track"]) == (
        "duel_tackle",
        20,
        [11, 12],
        None,
    )
    # Without the track table, the pose rows tell who is on the key frame.
    h = mo.add_key_moments(doc, pose, 25)["highlights"][0]
    assert (h["key_pair"], h["key_focus_track"]) == ([11, 12], None)
    # A moved peak is no exception: a subject without a track row there is no focus.
    pose, tracks = _reaction_next_to([800.0] * 60)
    tracks = tracks[(tracks["track_id"] != 4) | (tracks["frame"] != 37)]
    h = mo.add_key_moments(_closeup_doc(), pose, 25, tracks=tracks)["highlights"][0]
    assert (h["peak_frame"], h["key_focus_track"]) == (37, None)


def _raise_arm(k: np.ndarray, side: str, lift_sw: float, straight: bool) -> np.ndarray:
    """A copy of skeleton k with one wrist lift_sw shoulder widths above its shoulder; the
    elbow straight (on the line to the wrist) or bent out to the side (hand on the head)."""
    k = k.copy()
    s = k[HALPE26_INDEX[f"{side}_shoulder"], :2]
    width = float(
        np.linalg.norm(
            k[HALPE26_INDEX["left_shoulder"], :2] - k[HALPE26_INDEX["right_shoulder"], :2]
        )
    )
    out = -1.0 if side == "left" else 1.0  # UPRIGHT has the left arm on the image left
    if straight:
        wrist = s + (0.0, -lift_sw * width)
        elbow = (s + wrist) / 2
    else:
        elbow = s + (out * 0.8 * width, -0.1 * width)
        wrist = s + (-out * 0.1 * width, -lift_sw * width)
    k[HALPE26_INDEX[f"{side}_elbow"], :2] = elbow
    k[HALPE26_INDEX[f"{side}_wrist"], :2] = wrist
    return k


def _closeup_doc(cue: str = "subject") -> dict:
    return {
        "highlights": [
            {
                "id": "h01",
                "type": "closeup_reaction",
                "start_frame": 0,
                "peak_frame": 5,
                "end_frame": 59,
                "score": 0.2,
                "title_zh": "导播切镜强调的瞬间",
                "summary_zh": "特写反应。入选依据：导播切到特写。",
                "track_ids": [4],
                "refined_by": "score_peak",
                "cues": {
                    "cut_emphasis": {
                        "cut_frame": 8,
                        "joined_shot_event": {"cue": cue, "frame": 30, "track_ids": [4]},
                    }
                },
            }
        ]
    }


def _two_raises() -> pd.DataFrame:
    """Close-up subject 4: a hand on the head in frames 10-19, then a straight arm raised in
    frames 30-44, highest at 37."""
    rows = []
    for f in range(60):
        k = _skeleton(scale=3.0)
        if 10 <= f <= 19:
            k = _raise_arm(k, "left", 0.5, straight=False)  # a hand on the head
        if 30 <= f <= 44:
            k = _raise_arm(k, "right", 1.0 - abs(f - 37) * 0.05, straight=True)
        rows.append((f, 4, k))
    return _pose(rows)


def test_straight_arm_raise_becomes_the_reaction_peak() -> None:
    pose = _two_raises()
    gestures = mo.arm_gestures(pose, 4, 25)
    assert [(g.start_frame, g.end_frame, g.straight) for g in gestures] == [
        (10, 19, False),
        (30, 44, True),
    ]
    assert (gestures[1].top_frame, gestures[1].arm) == (37, "right")
    assert gestures[1].max_raise == pytest.approx(1.0)
    out = mo.add_key_moments(_closeup_doc(), pose, 25)
    h = out["highlights"][0]
    assert [(m["frame"], m["kind"]) for m in h["key_moments"]] == [
        (10, "hand_up"),
        (30, "hand_up"),
        (37, "arm_top"),
    ]
    assert (h["peak_frame"], h["refined_by"], h["type"]) == (
        37,
        "pose_key_moment",
        "closeup_reaction",
    )
    assert [m["text_zh"] for m in h["key_moments"][:2]] == [
        "轨迹 4 球员抬手过肩（屈肘，持续 0.4 秒）",
        "轨迹 4 球员抬手过肩（手臂伸直约 0.6 秒）",
    ]
    assert h["previous_peak"]["frame"] == 5
    assert h["key_moments"][2]["text_zh"] == "轨迹 4 球员手臂伸直举过肩，举到最高"
    assert h["key_moments"][2]["evidence"]["straight_at_top"] is True
    # Without the shot table, the shots are not named by size.
    assert h["title_zh"].endswith("（第 37 帧）近镜头中轨迹 4 球员手臂伸直举过肩")
    assert h["summary_zh"].startswith("球员反应。关键时刻（由原始画面上的 2D 姿态逐帧确定")
    assert "导播在第 8 帧切到近镜头" in h["summary_zh"]
    assert h["summary_zh"].endswith("入选依据：导播切到特写。涉及球员：轨迹 4 球员。")
    assert (h["key_pair"], h["key_focus_track"]) == (None, 4)
    # The cut is less than REACTION_LEAD_S after the window's start: the window stays.
    assert (h["start_frame"], h["end_frame"]) == (0, 59)
    assert _closeup_doc()["highlights"][0]["peak_frame"] == 5, "the input is not changed"


def test_an_arm_straightened_after_the_hand_went_up_is_said_so() -> None:
    rows = []
    for f in range(60):
        k = _skeleton(scale=3.0)
        if 30 <= f <= 34:
            k = _raise_arm(k, "right", 0.6, straight=False)  # raised, the elbow still bent
        if 35 <= f <= 44:
            k = _raise_arm(k, "right", 1.0 - abs(f - 39) * 0.05, straight=True)
        rows.append((f, 4, k))
    pose = _pose(rows)
    gestures = mo.arm_gestures(pose, 4, 25)
    assert [(g.start_frame, g.end_frame, g.straight, g.top_frame) for g in gestures] == [
        (30, 44, True, 39)
    ]
    assert gestures[0].evidence["straight_frames"] == 10
    assert gestures[0].evidence["first_straight_frame"] == 35
    h = mo.add_key_moments(_closeup_doc(), pose, 25)["highlights"][0]
    hand_up, top = h["key_moments"]
    assert (hand_up["frame"], hand_up["kind"], top["frame"], top["kind"]) == (
        30,
        "hand_up",
        39,
        "arm_top",
    )
    assert hand_up["text_zh"] == "轨迹 4 球员抬手过肩（随后手臂伸直约 0.4 秒）"
    assert (hand_up["evidence"]["raised_s"], hand_up["evidence"]["straight_s"]) == (0.6, 0.4)


def _shots(shot_type: str, median_box_h: float) -> pd.DataFrame:
    """shots.parquet of _closeup_doc: a wide shot, then from the cut at frame 8 on a shot
    of shot_type whose median box is median_box_h px tall."""
    return pd.DataFrame(
        {
            "shot_id": [0, 1],
            "start_frame": [0, 8],
            "end_frame": [7, 59],
            "shot_type": ["wide", shot_type],
            "fresh_share": [1.0, 0.0],
            "median_box_h": [120.0, median_box_h],
        }
    )


@pytest.mark.parametrize(
    ("shot_type", "median_box_h", "name"),
    [("close", 800.0, "特写"), ("close", 400.0, "中景"), ("medium", 300.0, "中景")],
)
def test_a_reaction_names_its_shot_by_the_size_of_its_players(
    shot_type: str, median_box_h: float, name: str
) -> None:
    shots = _shots(shot_type, median_box_h)
    h = mo.add_key_moments(_closeup_doc(), _two_raises(), 25, shots=shots)["highlights"][0]
    assert h["title_zh"].endswith(f"（第 37 帧）{name}中轨迹 4 球员手臂伸直举过肩")
    assert f"（第 0–59 帧，导播在第 8 帧切到{name}）" in h["summary_zh"]
    assert h["summary_zh"].startswith("球员反应。")


def test_the_cut_and_the_key_frame_name_their_own_shots() -> None:
    # The director cuts to a close shot of upper bodies at frame 8 and to a medium shot at
    # frame 33; the hand is highest at 37. The summary names the shot cut to, the title the
    # shot of the key frame.
    shots = pd.DataFrame(
        {
            "shot_id": [0, 1, 2],
            "start_frame": [0, 8, 33],
            "end_frame": [7, 32, 59],
            "shot_type": ["wide", "close", "medium"],
            "median_box_h": [120.0, 800.0, 300.0],
        }
    )
    h = mo.add_key_moments(_closeup_doc(), _two_raises(), 25, shots=shots)["highlights"][0]
    assert h["peak_frame"] == 37
    assert h["title_zh"].endswith("（第 37 帧）中景中轨迹 4 球员手臂伸直举过肩")
    assert "（第 0–59 帧，导播在第 8 帧切到特写）" in h["summary_zh"]


def test_a_shot_table_without_its_columns_is_reported() -> None:
    shots = pd.DataFrame({"start_frame": [0], "end_frame": [59]})
    with pytest.raises(mo.MomentsInputError, match="shot_type"):
        mo.add_key_moments(_closeup_doc(), _two_raises(), 25, shots=shots)


def _raise_with_top(top_straight: bool) -> pd.DataFrame:
    """Close-up subject 4 raising the right arm in frames 30-44, highest at frame 37; the
    elbow is straight on every frame but 37 (top_straight False), or only on 37."""
    rows = []
    for f in range(60):
        k = _skeleton(scale=3.0)
        if 30 <= f <= 44:
            straight = (f == 37) == top_straight
            k = _raise_arm(k, "right", 1.2 if f == 37 else 0.8, straight=straight)
        rows.append((f, 4, k))
    return _pose(rows)


def test_the_top_of_a_raise_says_what_the_elbow_does_on_that_frame() -> None:
    # A straight-arm gesture whose elbow bends on its highest frame.
    h = mo.add_key_moments(_closeup_doc(), _raise_with_top(False), 25)["highlights"][0]
    hand_up, top = h["key_moments"]
    assert (hand_up["frame"], top["frame"], top["kind"], h["peak_frame"]) == (
        30,
        37,
        "hand_top",
        37,
    )
    assert hand_up["text_zh"] == "轨迹 4 球员抬手过肩（手臂伸直约 0.6 秒）"
    assert top["text_zh"] == "轨迹 4 球员手举到最高（屈肘）"
    assert (top["evidence"]["straight_arm"], top["evidence"]["straight_at_top"]) == (True, False)
    assert top["evidence"]["elbow_deg_at_top"] < mo.ARM_STRAIGHT_DEG
    assert h["title_zh"].endswith("（第 37 帧）近镜头中轨迹 4 球员手抬过肩（屈肘）")
    # A bent-arm gesture straight on its highest frame only: one frame does not make a
    # straight arm, nor is the elbow bent there, so the top says neither and the gesture's
    # start does not call the arm bent throughout.
    h = mo.add_key_moments(_closeup_doc(), _raise_with_top(True), 25)["highlights"][0]
    hand_up, top = h["key_moments"]
    assert (top["frame"], top["kind"]) == (37, "hand_top")
    assert hand_up["text_zh"] == "轨迹 4 球员抬手过肩（持续 0.6 秒，手臂伸直不到 0.3 秒）"
    assert top["text_zh"] == "轨迹 4 球员手举到最高"
    assert "屈肘" not in h["title_zh"] + h["summary_zh"]
    assert (top["evidence"]["straight_arm"], top["evidence"]["straight_at_top"]) == (False, True)
    assert h["title_zh"].endswith("（第 37 帧）近镜头中轨迹 4 球员手抬过肩")


def test_a_reaction_window_starts_shortly_before_the_cut() -> None:
    doc = _closeup_doc()
    h = doc["highlights"][0]
    h.update({"peak_frame": 20, "end_frame": 199})
    h["cues"]["cut_emphasis"]["cut_frame"] = 100
    rows = []
    for f in range(100, 200):  # the subject is seen in the close-up only
        k = _skeleton(scale=3.0)
        if 130 <= f <= 144:
            k = _raise_arm(k, "right", 1.0 - abs(f - 137) * 0.05, straight=True)
        rows.append((f, 4, k))
    h = mo.add_key_moments(doc, _pose(rows), 25)["highlights"][0]
    assert (h["start_frame"], h["peak_frame"], h["end_frame"]) == (50, 137, 199)
    assert h["previous_peak"] == {
        "frame": 20,
        "type": "closeup_reaction",
        "refined_by": "score_peak",
        "title_zh": "导播切镜强调的瞬间",
        "start_frame": 0,
        "end_frame": 199,
    }
    assert "（第 50–199 帧，导播在第 100 帧切到近镜头）" in h["summary_zh"]
    assert round(mo.REACTION_LEAD_S * 25) == 100 - 50


def _reaction_next_to(partner_u: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close-up subject 4 raising a straight arm in frames 30-44 (highest at 37) next to
    opponent 5, whose ankles are at partner_u[f] (the subject's at 500 px; 3x scale)."""
    rows, tracks = [], []
    for f, u in enumerate(partner_u):
        k = _skeleton(scale=3.0)
        if 30 <= f <= 44:
            k = _raise_arm(k, "right", 1.0 - abs(f - 37) * 0.05, straight=True)
        rows.append((f, 4, k))
        rows.append((f, 5, _skeleton(scale=3.0, at=(u, 600.0))))
        tracks.append({"frame": f, "track_id": 4, "team": "left", "role": "player"})
        tracks.append({"frame": f, "track_id": 5, "team": "right", "role": "player"})
        for row, foot in zip(tracks[-2:], (500.0, u), strict=True):
            row.update({"foot_u": foot, "foot_v": 618.0, "y1": 60.0, "y2": 618.0})
    return _pose(rows), pd.DataFrame(tracks)


def test_a_reaction_next_to_its_partner_keeps_its_gestures() -> None:
    pose, tracks = _reaction_next_to([800.0] * 60)  # 0.54 box heights apart, no contact
    doc = mo.add_duel_partners(_closeup_doc(), tracks)
    assert (
        doc["highlights"][0]["cues"]["cut_emphasis"]["joined_shot_event"]["partner_track_id"] == 5
    )
    h = mo.add_key_moments(doc, pose, 25, tracks=tracks)["highlights"][0]
    assert [(m["frame"], m["kind"]) for m in h["key_moments"]] == [
        (30, "hand_up"),
        (37, "arm_top"),
    ]
    assert (h["type"], h["peak_frame"], h["refined_by"]) == (
        "closeup_reaction",
        37,
        "pose_key_moment",
    )
    assert (h["key_pair"], h["key_focus_track"]) == (None, 4)


def test_a_reaction_lists_the_partner_moments_that_change_nothing() -> None:
    partner = [680.0 if 46 <= f <= 53 else 800.0 for f in range(60)]  # contact in 46-53
    pose, tracks = _reaction_next_to(partner)
    doc = mo.add_duel_partners(_closeup_doc(), tracks)
    h = mo.add_key_moments(doc, pose, 25, tracks=tracks)["highlights"][0]
    assert [(m["frame"], m["kind"]) for m in h["key_moments"]] == [
        (30, "hand_up"),
        (37, "arm_top"),
        (46, "contact_start"),
        (53, "contact_end"),
    ]
    assert (h["type"], h["peak_frame"]) == ("closeup_reaction", 37)
    assert (h["key_pair"], h["key_focus_track"]) == ([4, 5], 4)
    pair = "左侧球队球员（轨迹 4）与右侧球队球员（轨迹 5）"
    assert f"（第 46 帧）{pair}身体贴近（画面平面）" in h["summary_zh"]
    assert h["title_zh"].endswith("近镜头中左侧球队球员手臂伸直举过肩"), "titles name the team"
    # The partner, named by the key moments, is listed among the players involved.
    assert h["track_ids"] == [4, 5]
    assert h["summary_zh"].endswith("涉及球员：左侧球队球员（轨迹 4）、右侧球队球员（轨迹 5）。")


def test_a_reaction_counts_the_touches_of_its_final_window() -> None:
    doc = _closeup_doc()
    h = doc["highlights"][0]
    h.update({"peak_frame": 20, "end_frame": 199})
    h["cues"]["cut_emphasis"]["cut_frame"] = 100
    h["cues"]["cut_emphasis"]["joined_shot_event"]["partner_track_id"] = 5
    rows = []
    for f in range(200):
        k = _skeleton()  # 40 px shoulders: gestures are read
        if 130 <= f <= 144:
            k = _raise_arm(k, "right", 1.0 - abs(f - 137) * 0.05, straight=True)
        rows.append((f, 4, k))
        if f >= 140:  # the partner is seen late in the close-up only
            rows.append((f, 5, _skeleton(at=(700.0, 600.0))))
    pose = _pose(rows)
    tracks = pd.DataFrame(
        _boxes(range(200), 4, "left", 500.0) + _boxes(range(140, 200), 5, "right", 700.0)
    )
    events = _ball_events([(30, 488.0, 600.0, 4), (120, 488.0, 600.0, 4)])  # 4's left ankle
    out = mo.add_key_moments(doc, pose, 25, tracks=tracks, touches=events)["highlights"][0]
    assert (out["start_frame"], out["peak_frame"], out["end_frame"]) == (50, 137, 199)
    # Frame 30 lies before the shortened window, frame 120 before the partner is seen.
    assert [(m["frame"], m["kind"]) for m in out["key_moments"]] == [
        (120, "touch"),
        (130, "hand_up"),
        (137, "arm_top"),
    ]
    assert (out["key_pair"], out["key_focus_track"]) == ([4, 5], 4)
    assert [(t["track_id"], t["count"], t["frames"]) for t in out["touches"]] == [(4, 1, [120])]
    assert _measured_touches(out, tracks, events, pose) == {4: 1}
    assert "在第 120 帧检测到左侧球队球员（轨迹 4）触球 1 次" in out["summary_zh"]
    assert out["summary_zh"].startswith("球员反应。关键时刻（由原始画面上的 2D 姿态和球的位置")


def _merged_reaction(duels: list[tuple[str, int, int]], start: int) -> dict:
    """merge_adjacent_highlights of duel parts (id, start, end) and, after them, the
    close-up reaction "hr" of _closeup_doc in frames start-299, whose cut into the close-up
    is at frame 200; every part starts on a cut."""
    reaction = _closeup_doc()["highlights"][0]
    reaction.update({"id": "hr", "start_frame": start, "peak_frame": 250, "end_frame": 299})
    reaction.update({"score": 0.5, "track_ids": [4]})
    reaction["cues"]["cut_emphasis"]["cut_frame"] = 200
    parts = [
        {
            "id": hid,
            "type": "duel_tackle",
            "start_frame": lo,
            "peak_frame": lo + 50,
            "end_frame": hi,
            "score": 0.2,
            "title_zh": "t",
            "track_ids": [1],
        }
        for hid, lo, hi in duels
    ]
    cuts = sorted({lo for _, lo, _ in duels[1:]} | {start, 200})
    return mo.merge_adjacent_highlights({"highlights": [*parts, reaction]}, cuts, 25)


def test_merged_parts_outside_a_shortened_reaction_window_are_dropped() -> None:
    rows = []
    for f in range(200, 300):
        k = _skeleton(scale=3.0)
        if 230 <= f <= 244:
            k = _raise_arm(k, "right", 1.0 - abs(f - 237) * 0.05, straight=True)
        rows.append((f, 4, k))
    pose = _pose(rows)
    doc = _merged_reaction([("ha", 0, 99), ("hb", 100, 199)], 200)
    assert doc["highlights"][0]["merged_from"] == ["detect-ha", "detect-hb", "detect-hr"]
    (h,) = mo.add_key_moments(doc, pose, 25)["highlights"]
    assert (h["start_frame"], h["peak_frame"], h["end_frame"]) == (150, 237, 299)
    assert h["merged_from"] == ["detect-hb", "detect-hr"], "frames 0-99 lie before the window"
    assert [(p["id"], p["start_frame"], p["end_frame"]) for p in h["merged_parts"]] == [
        ("detect-hb", 100, 199),
        ("detect-hr", 200, 299),
    ]
    # Only the reaction's own part is left: the highlight is no longer a merge.
    doc = _merged_reaction([("ha", 0, 99)], 100)
    assert doc["highlights"][0]["merged_from"] == ["detect-ha", "detect-hr"]
    (h,) = mo.add_key_moments(doc, pose, 25)["highlights"]
    assert (h["start_frame"], h["peak_frame"], h["end_frame"]) == (150, 237, 299)
    assert "merged_from" not in h
    assert "merged_parts" not in h
    assert h["previous_peak"]["start_frame"] == 0


def test_brief_or_distant_raises_are_not_gestures() -> None:
    rows = []
    for f in range(40):
        k = _skeleton(scale=3.0)
        if 10 <= f <= 14:  # five frames, shorter than GESTURE_MIN_S at 25 fps
            k = _raise_arm(k, "right", 0.8, straight=True)
        rows.append((f, 4, k))
        far = _skeleton(scale=0.5, at=(1500.0, 300.0))  # a wide-shot player, 20 px shoulders
        rows.append((f, 5, _raise_arm(far, "left", 0.8, straight=True) if f >= 5 else far))
    pose = _pose(rows)
    assert mo.arm_gestures(pose, 4, 25) == []
    assert mo.arm_gestures(pose, 5, 25) == []
    h = mo.add_key_moments(_closeup_doc(), pose, 25)["highlights"][0]
    assert (h["key_moments"], h["peak_frame"]) == ([], 5)
    assert "previous_peak" not in h
    assert (h["key_pair"], h["key_focus_track"]) == (None, None)


def test_a_short_dip_does_not_split_a_gesture_and_a_brief_straight_arm_counts_as_bent() -> None:
    rows = []
    for f in range(40):
        k = _skeleton(scale=3.0)
        if 5 <= f <= 14 or 18 <= f <= 30:  # the hand is down for frames 15-17 only
            k = _raise_arm(k, "left", 0.6, straight=20 <= f <= 25)  # six straight frames
        rows.append((f, 4, k))
    gestures = mo.arm_gestures(_pose(rows), 4, 25)
    assert [(g.start_frame, g.end_frame, g.straight) for g in gestures] == [(5, 30, False)]
    assert gestures[0].evidence["straight_frames"] == 6
    h = mo.add_key_moments(_closeup_doc(), _pose(rows), 25)["highlights"][0]
    assert [m["kind"] for m in h["key_moments"]] == ["hand_up", "hand_top"]
    assert h["title_zh"].endswith("手抬过肩（屈肘）")


def test_reaction_needs_the_closeup_subject() -> None:
    rows = [
        (f, 4, _raise_arm(_skeleton(scale=3.0), "right", 0.8, straight=True)) for f in range(30)
    ]
    h = mo.add_key_moments(_closeup_doc(cue="strike"), _pose(rows), 25)["highlights"][0]
    assert (h["key_moments"], h["peak_frame"]) == ([], 5)


def _subject_doc(subject: int) -> dict:
    """A highlight whose joined shot shows one subject (a ball event of that player)."""
    event = {"cue": "strike", "frame": 70, "track_ids": [subject]}
    return {
        "highlights": [
            {
                "id": "h01",
                "start_frame": 50,
                "peak_frame": 60,
                "end_frame": 79,
                "track_ids": [subject],
                "cues": {"cut_emphasis": {"joined_shot_event": event}},
            }
        ]
    }


def test_a_subject_without_a_team_gets_no_partner() -> None:
    # gbf.team_check clears labels it cannot trust; such a subject next to a teammate was
    # given the teammate as duel partner (the overlay then drew 对抗 between teammates).
    teammate = _track_rows(range(50, 80), 8, "left", 940.0)  # 0.2 box heights away
    opponent = _track_rows(range(50, 80), 9, "right", 1000.0)  # farther, 0.5 box heights
    unlabelled = _track_rows(range(50, 80), 7, None, 900.0)
    tracks = pd.DataFrame(unlabelled + teammate + opponent)
    h = mo.add_duel_partners(_subject_doc(7), tracks)["highlights"][0]
    assert "partner_track_id" not in h["cues"]["cut_emphasis"]["joined_shot_event"]
    assert h["track_ids"] == [7]
    assert mo._duel_pairs(h) == []
    # As many rows of each team: no team either.
    tied = _track_rows(range(50, 65), 7, "left", 900.0)
    tied += _track_rows(range(65, 80), 7, "right", 900.0)
    tracks = pd.DataFrame(tied + teammate + opponent)
    h = mo.add_duel_partners(_subject_doc(7), tracks)["highlights"][0]
    assert "partner_track_id" not in h["cues"]["cut_emphasis"]["joined_shot_event"]
    # Labelled, the subject gets its nearest opponent, never the nearer teammate.
    labelled = _track_rows(range(50, 80), 7, "right", 900.0)
    tracks = pd.DataFrame(labelled + _track_rows(range(50, 80), 8, "right", 940.0) + opponent)
    tracks.loc[tracks["track_id"] == 9, "team"] = "left"
    h = mo.add_duel_partners(_subject_doc(7), tracks)["highlights"][0]
    assert h["cues"]["cut_emphasis"]["joined_shot_event"]["partner_track_id"] == 9


def test_the_players_the_title_names_are_listed_as_involved() -> None:
    # gbf.highlights listed the players involved before the duel pair was known; the title
    # of the duel now names a player the list left out (track 2).
    tracks = pd.DataFrame(
        _boxes(range(40), 1, "left", 500.0) + _boxes(range(40), 2, "right", 560.0)
    )
    summary = "入选依据：第 12 帧检测到触球。涉及球员：左侧球队球员（轨迹 1）。"
    summary += "评分 0.30，仅用于挑选片段。"
    doc = _duel_doc((1, 2), 39, summary_zh=summary, track_ids=[1])
    h = mo.add_key_moments(doc, _pose(_contact_and_fall()), 25, tracks=tracks)["highlights"][0]
    assert h["type"] == "duel_tackle"
    assert "右侧球队球员躯干开始倾倒" in h["title_zh"]
    assert h["track_ids"] == [1, 2]
    assert h["summary_zh"].endswith(
        "涉及球员：左侧球队球员（轨迹 1）、右侧球队球员（轨迹 2）。评分 0.30，仅用于挑选片段。"
    )
    assert h["summary_zh"].count("涉及球员") == 1
    # A highlight the key moments do not change gets the list of its track ids too.
    still = {"id": "h02", "start_frame": 0, "peak_frame": 5, "end_frame": 39, "type": "sprint"}
    still.update({"track_ids": [2, 1], "summary_zh": "入选依据：x。评分 0.20。"})
    (h,) = mo.add_key_moments({"highlights": [still]}, None, 25, tracks=tracks)["highlights"]
    assert h["summary_zh"] == (
        "入选依据：x。涉及球员：右侧球队球员（轨迹 2）、左侧球队球员（轨迹 1）。评分 0.20。"
    )


def test_a_fall_is_read_from_the_trunk_angle_the_report_shows() -> None:
    # One image-plane trunk angle: the hip midpoint to the shoulder midpoint. gbf.lift3d's
    # image-trunk-lean (躯干倾斜（画面）) used the neck keypoint, so the report's angle and
    # the key moment's 45° could disagree about a frame.
    rows = []
    for f in range(30):
        k = _skeleton(max(0.0, (f - 10) * 10.0))
        k[HALPE26_INDEX["neck"], 0] += 12.0  # a neck keypoint off the shoulder line
        rows.append((f, 7, k))
    pose = _pose(rows)
    angles = mo.body_angles(pose)
    shown = [abs(lift3d.image_plane_angles(k).trunk_lean_deg) for _f, _t, k in rows]
    np.testing.assert_allclose(angles["trunk_deg"].to_numpy(), shown, atol=1e-9)
    (event,) = mo.fall_events(pose, 25)
    assert event.down_frame == 15  # 50° there; 40° on frame 14
    # The evidence of a fall onset names the angle per frame, not nested, and the line.
    window = pd.concat([pose, _pose([(f, 8, _skeleton(at=(560.0, 600.0))) for f in range(30)])])
    timeline = mo.duel_timeline(window, (7, 8), 0, 29, 25)
    onset = next(m for m in timeline if m.kind == "fall_onset")
    assert onset.evidence["trunk_deg"][15] == pytest.approx(50.0, abs=0.1)
    assert onset.evidence["trunk_line"] == "hip_midpoint_to_shoulder_midpoint"
    assert onset.evidence["down_frame"] == 15
    assert not isinstance(onset.evidence["trunk_deg"].get("trunk_deg"), dict)
