"""Tests for gbf.highlights: the per-frame cues and the highlight detector.

Every input is synthetic and built here at 25 fps: shots, tracks with pitch positions,
kinematics, ball positions, ball events and audio excitement. The detector tests plant
triangular cue bumps at known frames on top of the cut emphasis that compute_signals
derives from the shots.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gbf import highlights
from gbf.highlights import (
    CUE_COLUMNS,
    CUE_WEIGHTS,
    DUEL_IMAGE_MAX_BOX_H_PX,
    HIGHLIGHT_TYPES,
    REFINE_METHODS_ZH,
    SHOT_UPPER_BODY_MIN_BOX_H_PX,
    SIGNAL_COLUMNS,
    TYPE_NAMES_ZH,
    HighlightInputError,
    compute_signals,
    detect_highlights,
    format_time,
    gaussian_smooth,
    shot_name_zh,
)
from gbf.measure import MeasureParams, display_text

FPS = "25/1"
TOTAL_WEIGHT = sum(CUE_WEIGHTS.values())


# --------------------------------------------------------------------------- builders


def make_shots(*spec: tuple[str, int, float]) -> pd.DataFrame:
    """Shots from ``(shot_type, length in frames, fresh_share)``, tiling the clip from 0."""
    rows, start = [], 0
    for shot_id, (kind, length, fresh) in enumerate(spec):
        rows.append(
            {
                "shot_id": shot_id,
                "start_frame": start,
                "end_frame": start + length - 1,
                "shot_type": kind,
                "fresh_share": fresh,
                "median_box_h": 100.0,
            }
        )
        start += length
    return pd.DataFrame(rows)


TRACK_DEFAULTS: dict[str, Any] = {
    "x1": 0.0,
    "y1": 0.0,
    "x2": 40.0,
    "y2": 100.0,
    "conf": 0.9,
    "role": "player",
    "team": "left",
    "jersey": None,
    "jersey_conf": np.nan,
    "foot_u": 20.0,
    "foot_v": 100.0,
    "calib_status": "fresh",
    "pitch_usable": True,
    "pitch_x": np.nan,
    "pitch_y": np.nan,
    "in_pitch": True,
}


def make_tracks(rows: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame([{**TRACK_DEFAULTS, **row} for row in rows])
    return table.astype({"frame": "int64", "track_id": "int64"})


def make_kinematics(rows: list[tuple[int, int, float, bool]]) -> pd.DataFrame:
    frame, track, speed, valid = zip(*rows, strict=True)
    return pd.DataFrame(
        {
            "frame": np.array(frame, dtype=np.int64),
            "track_id": np.array(track, dtype=np.int64),
            "segment_id": np.zeros(len(rows), dtype=np.int64),
            "x": np.zeros(len(rows)),
            "y": np.zeros(len(rows)),
            "vx": np.zeros(len(rows)),
            "vy": np.zeros(len(rows)),
            "speed": np.array(speed, dtype=np.float64),
            "valid": np.array(valid, dtype=bool),
        }
    )


def make_ball(rows: list[dict[str, Any]]) -> pd.DataFrame:
    defaults = {
        "u": np.nan,
        "v": np.nan,
        "radius_px": 5.0,
        "conf": 0.8,
        "status": "detected",
        "candidate_count": 1,
        "pitch_x": np.nan,
        "pitch_y": np.nan,
        "ground_assumption": True,
    }
    table = pd.DataFrame([{**defaults, **row} for row in rows])
    return table.astype({"frame": "int64"})


def base_signals(shots: pd.DataFrame) -> pd.DataFrame:
    return compute_signals(None, None, None, None, shots, None, FPS)


def plant(signals: pd.DataFrame, bumps: list[tuple[int, dict[str, float], int]]) -> pd.DataFrame:
    """Add triangular cue bumps ``(centre, {cue: height}, half width)`` and rescore."""
    out = signals.copy()
    frames = np.arange(len(out))
    for centre, heights, width in bumps:
        triangle = np.clip(1.0 - np.abs(frames - centre) / width, 0.0, None)
        for cue, height in heights.items():
            out[cue] = np.maximum(out[cue].to_numpy(), height * triangle)
    out["score"] = sum(CUE_WEIGHTS[c] * out[c].to_numpy() for c in CUE_COLUMNS) / TOTAL_WEIGHT
    return out


def by_peak(result: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(result["highlights"], key=lambda h: h["peak_frame"])


def ball_at_box(signals: pd.DataFrame, lo: int, hi: int, metres: float = 0.0) -> pd.DataFrame:
    """Signals whose ball came within ``metres`` of the box the attackers attack on frames
    lo..hi (``ev_ball_box_m``, see "Ball at the box"); unknown elsewhere."""
    out = signals.copy()
    out.loc[lo:hi, "ev_ball_box_m"] = metres
    return out


def box_pressure_at_the_box(signals: pd.DataFrame) -> pd.DataFrame:
    """Planted signals as compute_signals writes them: box_pressure 0 wherever the ball is not
    at the box (``ev_ball_box_m`` beyond the margin or unknown), and the score again."""
    out = signals.copy()
    near = out["ev_ball_box_m"].fillna(np.inf).to_numpy() <= highlights.BALL_AT_BOX_MARGIN_M
    out["box_pressure"] = np.where(near, out["box_pressure"].to_numpy(), 0.0)
    out["score"] = sum(CUE_WEIGHTS[c] * out[c].to_numpy() for c in CUE_COLUMNS) / TOTAL_WEIGHT
    return out


# --------------------------------------------------------------------------- compute_signals


def test_cut_emphasis_marks_the_end_of_a_wide_shot_and_the_close_shot_after_it() -> None:
    shots = make_shots(
        ("wide", 100, 1.0),
        ("close", 50, 0.0),
        ("wide", 100, 1.0),
        ("medium", 50, 0.0),
        ("wide", 50, 1.0),
        ("wide", 30, 1.0),
    )
    signals = base_signals(shots)
    assert list(signals.columns) == list(SIGNAL_COLUMNS)
    assert signals["frame"].tolist() == list(range(380))
    expected = np.zeros(380)
    expected[62:100] = 1.0  # the last 1.5 s (38 frames) of the wide shot
    expected[100:150] = 0.5  # the close shot after it, at a lower weight
    expected[212:250] = 1.0
    expected[250:300] = 0.5
    np.testing.assert_array_equal(signals["cut_emphasis"].to_numpy(), expected)
    cut = signals["ev_cut_frame"].to_numpy()
    assert set(cut[62:150]) == {100}
    assert set(cut[212:300]) == {250}
    assert set(cut[:62]) == {-1}
    assert set(cut[300:]) == {-1}  # a cut between two wide shots is not emphasised
    np.testing.assert_allclose(signals["score"].to_numpy(), expected / TOTAL_WEIGHT)
    assert (
        signals["shot_id"].tolist()
        == [0] * 100 + [1] * 50 + [2] * 100 + [3] * 50 + [4] * 50 + [5] * 30
    )
    for name in ("audio", "speed", "box_pressure", "ball_goal", "duel", "strike"):
        assert not signals[name].any()


def test_audio_cue_is_the_clipped_excitement() -> None:
    shots = make_shots(("wide", 10, 1.0))
    audio = pd.DataFrame(
        {
            "frame": np.arange(10, dtype=np.int64),
            "rms_dbfs": np.zeros(10),
            "flux": np.zeros(10),
            "excitement_z": [-1.0, 0.0, 1.0, 2.0, 4.0, 6.0, np.nan, 3.0, 3.0, 3.0],
            "has_audio": [True] * 7 + [False, True, True],
        }
    )
    signals = compute_signals(None, None, None, None, shots, audio, FPS)
    np.testing.assert_allclose(
        signals["audio"].to_numpy(), [0, 0, 0.25, 0.5, 1, 1, 0, 0, 0.75, 0.75]
    )
    z = signals["ev_audio_z"].to_numpy()
    assert np.isnan(z[6])
    assert np.isnan(z[7])
    assert z[5] == 6.0


def test_speed_cue_maps_4_to_9_mps_on_pitch_frames_of_players_only() -> None:
    shots = make_shots(("wide", 6, 1.0), ("close", 4, 0.0))
    rows = []
    for frame in range(10):
        rows.append({"frame": frame, "track_id": 1, "pitch_x": 0.0, "pitch_y": 0.0})
        rows.append({"frame": frame, "track_id": 2, "team": "right", "pitch_x": 5.0})
        rows.append({"frame": frame, "track_id": 3, "role": "referee", "team": None})
    tracks = make_tracks(rows)
    kinematics = make_kinematics(
        [
            (0, 1, 3.0, True),
            (1, 1, 6.5, True),
            (1, 2, 5.0, True),
            (2, 1, 9.5, True),
            (3, 2, 15.0, True),  # faster than any footballer: a tracking error
            (3, 1, 7.0, True),
            (4, 3, 8.0, True),  # the referee does not count
            (5, 1, 8.0, False),  # not valid
            (7, 1, 8.0, True),  # inside the close shot: not a pitch frame
        ]
    )
    signals = compute_signals(tracks, kinematics, None, None, shots, None, FPS)
    np.testing.assert_allclose(
        signals["speed"].to_numpy(), [0, 0.5, 1, 0.6, 0, 0, 0, 0, 0, 0], atol=1e-12
    )
    assert signals["ev_speed_id"].tolist() == [1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    np.testing.assert_allclose(signals["ev_speed_mps"].to_numpy()[:4], [3.0, 6.5, 9.5, 7.0])
    assert signals["ev_speed_mps"].isna().to_numpy()[4:].all()
    assert signals["ev_pitch"].tolist() == [True] * 6 + [False] * 4


def test_box_pressure_counts_attackers_in_the_box_they_attack() -> None:
    shots = make_shots(("wide", 3, 1.0))
    rows = [
        {"frame": 0, "track_id": 1, "pitch_x": 45.0, "pitch_y": 0.0},
        {"frame": 0, "track_id": 2, "pitch_x": 40.0, "pitch_y": 10.0},
        {"frame": 0, "track_id": 3, "pitch_x": -45.0, "pitch_y": 0.0},  # own box
        {"frame": 0, "track_id": 4, "pitch_x": 30.0, "pitch_y": 0.0},  # outside the box
        {"frame": 0, "track_id": 5, "pitch_x": 45.0, "pitch_y": 25.0},  # wide of the box
        {"frame": 0, "track_id": 6, "pitch_x": 44.0, "pitch_y": 0.0, "in_pitch": False},
        {"frame": 0, "track_id": 7, "pitch_x": 50.0, "pitch_y": 0.0, "role": "goalkeeper"},
        {"frame": 0, "track_id": 8, "pitch_x": 45.0, "pitch_y": 5.0, "team": "right"},
        {"frame": 0, "track_id": 9, "pitch_x": 40.0, "pitch_y": -5.0, "team": "right"},
        {"frame": 1, "track_id": 1, "pitch_x": 45.0, "pitch_y": 0.0, "pitch_usable": False},
        {"frame": 2, "track_id": 1, "pitch_x": 45.0, "pitch_y": 0.0, "team": "nan"},
    ]
    tracks = make_tracks(rows)
    # The ball is in sight, and in the box on frame 0.
    ball = make_ball([{"frame": 0, "u": 900.0, "v": 500.0, "pitch_x": 45.0, "pitch_y": 0.0}])
    signals = compute_signals(tracks, None, ball, None, shots, None, FPS)
    np.testing.assert_allclose(signals["box_pressure"].to_numpy(), [2 / 3, 0, 0])
    assert signals["ev_box_attackers"].tolist() == [2, 0, 0]
    assert signals["ev_box_team"].tolist()[0] == "left"
    # Swap the directions: left now attacks x < 0 (one player there), right x > 0 (two).
    flipped = compute_signals(
        tracks, None, ball, None, shots, None, FPS, attack_direction={"left": -1, "right": 1}
    )
    assert flipped["ev_box_attackers"].tolist() == [2, 0, 0]
    assert flipped["ev_box_team"].tolist()[0] == "right"


def test_ball_goal_and_the_players_nearest_the_ball() -> None:
    shots = make_shots(("wide", 2, 1.0), ("close", 2, 0.0))
    rows = [
        {"frame": 0, "track_id": 1, "pitch_x": 39.0, "pitch_y": 1.0},
        {"frame": 0, "track_id": 2, "team": "right", "pitch_x": 43.0, "pitch_y": 0.0},
        {"frame": 1, "track_id": 1, "pitch_x": 37.0, "pitch_y": 0.0},
        {"frame": 1, "track_id": 2, "team": "right", "pitch_x": 40.5, "pitch_y": 0.0},
        {"frame": 2, "track_id": 1, "foot_u": 520.0, "foot_v": 900.0, "y1": 700.0, "y2": 900.0},
        {
            "frame": 2,
            "track_id": 2,
            "team": "right",
            "foot_u": 700.0,
            "foot_v": 900.0,
            "y1": 700.0,
            "y2": 900.0,
        },
        {"frame": 3, "track_id": 1, "foot_u": 520.0, "foot_v": 900.0, "y1": 700.0, "y2": 900.0},
    ]
    tracks = make_tracks(rows)
    ball = make_ball(
        [
            {"frame": 0, "u": 1.0, "v": 1.0, "pitch_x": 40.0, "pitch_y": 0.0},
            {"frame": 1, "u": 1.0, "v": 1.0, "pitch_x": 40.0, "pitch_y": 0.0},
            {"frame": 2, "u": 500.0, "v": 900.0},
            {"frame": 3, "u": 500.0, "v": 900.0, "status": "missing"},
        ]
    )
    signals = compute_signals(tracks, None, ball, None, shots, None, FPS)
    # Frame 0: left (attacking x > 0) is nearest; the ball is 12.5 m from that goal.
    assert signals["ev_ball_goal_m"].to_numpy()[0] == pytest.approx(12.5)
    assert signals["ball_goal"].to_numpy()[0] == pytest.approx((25 - 12.5) / 14)
    # Frame 1: right is nearest and attacks x < 0, 92.5 m away.
    assert signals["ev_ball_goal_m"].to_numpy()[1] == pytest.approx(92.5)
    assert signals["ball_goal"].to_numpy()[1] == 0.0
    assert signals["ev_ball_team"].tolist()[:2] == ["left", "right"]
    assert signals["ev_ball_near_id"].tolist() == [1, 2, 1, -1]
    assert signals["ev_ball_opp_id"].tolist() == [2, 1, 2, -1]
    # Frame 2 is a close shot: image plane only, no distance to goal.
    assert np.isnan(signals["ev_ball_goal_m"].to_numpy()[2])
    assert signals["ball_goal"].to_numpy()[2] == 0.0


def test_attack_cues_are_off_when_the_attack_direction_is_unknown() -> None:
    # gbf.side_check could not confirm the plugin's left/right names: which goal a team
    # attacks is unknown, so box pressure and ball-to-goal give nothing; the team of the
    # player nearest the ball, the duels and the other cues stay.
    shots = make_shots(("wide", 2, 1.0))
    rows = [
        {"frame": 0, "track_id": 1, "pitch_x": 45.0, "pitch_y": 0.0},
        {"frame": 0, "track_id": 2, "pitch_x": 40.0, "pitch_y": 10.0},
        {"frame": 0, "track_id": 8, "pitch_x": 46.0, "pitch_y": 5.0, "team": "right"},
        {"frame": 1, "track_id": 1, "pitch_x": 39.0, "pitch_y": 1.0},
        {"frame": 1, "track_id": 8, "pitch_x": 43.0, "pitch_y": 0.0, "team": "right"},
    ]
    tracks = make_tracks(rows)
    ball = make_ball(
        [
            {"frame": 0, "u": 900.0, "v": 500.0, "pitch_x": 45.0, "pitch_y": 0.0},
            {"frame": 1, "u": 900.0, "v": 500.0, "pitch_x": 40.0, "pitch_y": 0.0},
        ]
    )
    known = compute_signals(tracks, None, ball, None, shots, None, FPS)
    assert known["box_pressure"].to_numpy()[0] > 0
    assert known["ball_goal"].to_numpy()[1] > 0
    unknown = compute_signals(tracks, None, ball, None, shots, None, FPS, attack_known=False)
    assert unknown["box_pressure"].tolist() == [0.0, 0.0]
    assert unknown["ball_goal"].tolist() == [0.0, 0.0]
    assert unknown["ev_box_attackers"].tolist() == [0, 0]
    assert unknown["ev_box_team"].isna().all()
    assert unknown["ev_ball_goal_m"].isna().all()
    assert unknown["ev_ball_box_m"].isna().all()
    kept = [c for c in known.columns if c not in {"box_pressure", "ball_goal", "score",
            "ev_box_attackers", "ev_box_team", "ev_ball_goal_m", "ev_ball_box_m"}]  # fmt: skip
    pd.testing.assert_frame_equal(unknown[kept], known[kept])
    with pytest.raises(HighlightInputError) as caught:
        compute_signals(tracks, None, ball, None, shots, None, FPS, attack_known="no")
    assert "attack_known" in str(caught.value)


def test_duel_uses_metres_on_pitch_frames_and_box_heights_elsewhere() -> None:
    shots = make_shots(("wide", 2, 1.0), ("close", 2, 0.0))
    rows = [
        {"frame": 0, "track_id": 1, "pitch_x": 10.0, "pitch_y": 0.0},
        {"frame": 0, "track_id": 2, "team": "right", "pitch_x": 11.5, "pitch_y": 0.0},
        {"frame": 0, "track_id": 3, "pitch_x": 10.0, "pitch_y": 3.0},
        {"frame": 0, "track_id": 4, "team": "right", "pitch_x": 20.0, "pitch_y": 0.0},
    ]
    for frame in (2, 3):
        rows += [
            {
                "frame": frame,
                "track_id": 1,
                "foot_u": 500.0,
                "foot_v": 900.0,
                "y1": 700.0,
                "y2": 900.0,
            },
            {
                "frame": frame,
                "track_id": 2,
                "team": "right",
                "foot_u": 600.0,
                "foot_v": 900.0,
                "y1": 700.0,
                "y2": 900.0,
            },
            {
                "frame": frame,
                "track_id": 4,
                "team": "right",
                "foot_u": 1500.0,
                "foot_v": 900.0,
                "y1": 700.0,
                "y2": 900.0,
            },
        ]
    tracks = make_tracks(rows)
    ball = make_ball(
        [
            {"frame": 0, "u": 1.0, "v": 1.0, "pitch_x": 10.5, "pitch_y": 0.5},
            {"frame": 2, "u": 550.0, "v": 890.0},
            {"frame": 3, "status": "missing"},
        ]
    )
    signals = compute_signals(tracks, None, ball, None, shots, None, FPS)
    duel = signals["duel"].to_numpy()
    assert duel[0] == pytest.approx((4.0 - 1.5) / 3.0)
    assert signals["ev_duel_dist"].to_numpy()[0] == pytest.approx(1.5)
    assert signals["ev_duel_unit"].tolist()[0] == "m"
    assert duel[2] == 1.0  # 100 px apart with 200 px boxes: 0.5 box heights
    assert signals["ev_duel_dist"].to_numpy()[2] == pytest.approx(0.5)
    assert signals["ev_duel_unit"].tolist()[2] == "box_h"
    assert signals["ev_duel_id_a"].tolist() == [1, -1, 1, -1]
    assert signals["ev_duel_id_b"].tolist() == [2, -1, 2, -1]
    assert duel[1] == 0.0
    assert duel[3] == 0.0  # no ball, no duel


def test_no_image_plane_duel_in_upper_body_close_ups() -> None:
    shots = make_shots(("close", 2, 0.0), ("close", 2, 0.0))
    shots["median_box_h"] = [800.0, 300.0]  # 1080-line pixels: upper bodies, then full ones
    rows = []
    for frame in range(4):
        rows += [
            {
                "frame": frame,
                "track_id": 1,
                "foot_u": 500.0,
                "foot_v": 900.0,
                "y1": 700.0,
                "y2": 900.0,
            },
            {
                "frame": frame,
                "track_id": 2,
                "team": "right",
                "foot_u": 600.0,
                "foot_v": 900.0,
                "y1": 700.0,
                "y2": 900.0,
            },
        ]
    ball = make_ball([{"frame": f, "u": 550.0, "v": 890.0} for f in range(4)])
    signals = compute_signals(make_tracks(rows), None, ball, None, shots, None, FPS)
    np.testing.assert_array_equal(signals["duel"].to_numpy(), [0.0, 0.0, 1.0, 1.0])
    assert signals["ev_duel_id_a"].tolist() == [-1, -1, 1, 1]
    # The players nearest the ball are still reported in both shots.
    assert signals["ev_ball_near_id"].tolist() == [1, 1, 1, 1]


def test_strike_cue_is_a_triangle_around_each_ball_event() -> None:
    shots = make_shots(("wide", 100, 1.0))
    events = pd.DataFrame({"frame": [50, 60], "conf": [0.8, 0.4], "track_id": [7, 8]})
    signals = compute_signals(None, None, None, events, shots, None, FPS)
    cue = signals["strike"].to_numpy()
    event = signals["ev_strike_frame"].to_numpy()
    assert cue[50] == pytest.approx(0.8)
    assert cue[57] == pytest.approx(0.8 * (1 - 7 / 15))
    assert event[57] == 50
    assert cue[60] == pytest.approx(0.4)
    assert event[60] == 60
    assert cue[65] == pytest.approx(0.4 * (1 - 5 / 15))
    assert cue[36] == pytest.approx(0.8 / 15)
    assert cue[35] == 0.0
    assert event[35] == -1
    unweighted = compute_signals(None, None, None, events[["frame"]], shots, None, FPS)
    assert unweighted["strike"].to_numpy()[60] == 1.0


def test_strike_strength_from_the_ball_module_events() -> None:
    shots = make_shots(("wide", 400, 1.0))
    events = pd.DataFrame(
        {
            "frame": [20, 80, 140, 200, 260, 320],
            "kind": ["speed_up", "direction_change", "slow_down", "speed_up", "speed_up", "x"],
            "magnitude": [12.0, 5.0, 10.0, 10.0, 10.0, 10.0],
            "nearest_track_id": [3, 4, 5, -1, 6, 7],
            "nearest_dist_box_h": [0.2, 0.5, 0.3, np.nan, 1.5, 2.5],
        }
    )
    signals = compute_signals(None, None, None, events, shots, None, FPS)
    cue = signals["strike"].to_numpy()
    assert cue[20] == 1.0  # 12 bh/s is above a full strike, at the foot
    assert cue[80] == pytest.approx(0.5)  # half the full velocity change
    assert cue[140] == pytest.approx(0.5)  # a slow-down: a reception or a block
    assert cue[200] == pytest.approx(0.5)  # nobody near the ball was found
    assert cue[260] == pytest.approx(0.5)  # 1.5 box heights from the nearest foot
    assert cue[320] == 0.0  # no foot within 2 box heights: not a kick
    planted = plant(signals, [(20, {"audio": 1.0}, 15)])
    result = detect_highlights(planted, shots, None, events, FPS)
    first = by_peak(result)[0]
    assert first["type"] == "ball_strike"
    assert first["peak_frame"] == 20
    assert first["track_ids"][0] == 3


def test_pitch_cues_need_a_wide_shot_with_mostly_fresh_calibration() -> None:
    shots = make_shots(("wide", 2, 0.4), ("medium", 2, 1.0), ("wide", 2, 0.9))
    tracks = make_tracks(
        [{"frame": f, "track_id": 1, "pitch_x": 45.0, "pitch_y": 0.0} for f in range(6)]
    )
    kinematics = make_kinematics([(f, 1, 8.0, True) for f in range(6)])
    ball = make_ball(
        [{"frame": f, "u": 900.0, "v": 500.0, "pitch_x": 46.0, "pitch_y": 0.0} for f in range(6)]
    )
    signals = compute_signals(tracks, kinematics, ball, None, shots, None, FPS)
    assert signals["ev_pitch"].tolist() == [False, False, False, False, True, True]
    np.testing.assert_allclose(signals["box_pressure"].to_numpy(), [0, 0, 0, 0, 1 / 3, 1 / 3])
    np.testing.assert_allclose(signals["speed"].to_numpy(), [0, 0, 0, 0, 0.8, 0.8])


def test_attack_cues_need_a_detected_ball_within_a_second_in_the_same_shot() -> None:
    # Two wide shots (cut at frame 150); two left players stand in the box they attack the
    # whole time, as they do while a corner is being prepared.
    shots = make_shots(("wide", 150, 1.0), ("wide", 50, 1.0))
    rows = []
    for frame in range(200):
        rows.append({"frame": frame, "track_id": 1, "pitch_x": 45.0, "pitch_y": 0.0})
        rows.append({"frame": frame, "track_id": 2, "pitch_x": 40.0, "pitch_y": 5.0})
    position = {"u": 900.0, "v": 500.0, "pitch_x": 44.0, "pitch_y": 0.0}
    balls = [{"frame": f, "status": "missing"} for f in range(50)]  # out of the picture
    balls.append({"frame": 60, **position})
    bridged = (*range(61, 65), *range(100, 111))  # 100..110: no detection within 1 s
    balls += [{"frame": f, **position, "status": "interpolated"} for f in bridged]
    balls.append({"frame": 152, **position})  # just after the cut
    signals = compute_signals(make_tracks(rows), None, make_ball(balls), None, shots, None, FPS)
    seen = np.zeros(200, dtype=bool)
    seen[35:86] = True  # within 25 frames (1 s at 25 fps) of the detection at frame 60
    seen[150:178] = True  # the detection at 152 does not reach back across the cut
    np.testing.assert_allclose(signals["box_pressure"].to_numpy(), np.where(seen, 2 / 3, 0.0))
    assert signals["ev_box_attackers"].tolist() == np.where(seen, 2, 0).tolist()
    teams = [t if isinstance(t, str) else None for t in signals["ev_box_team"]]
    assert teams == ["left" if s else None for s in seen]
    goal = signals["ball_goal"].to_numpy()
    assert goal[60] == 1.0  # 8.5 m from the goal left attacks
    np.testing.assert_allclose(signals["ev_ball_goal_m"].to_numpy()[60:65], 8.5)
    assert not goal[100:111].any()  # interpolated positions alone are not a ball in sight
    assert signals["ev_ball_goal_m"].isna().to_numpy()[100:111].all()
    assert not any(isinstance(t, str) for t in signals["ev_ball_team"].tolist()[100:111])
    # The player nearest the ball is still reported there (other cues use it).
    assert signals["ev_ball_near_id"].tolist()[100:111] == [1] * 11
    without_ball = compute_signals(make_tracks(rows), None, None, None, shots, None, FPS)
    assert not without_ball["box_pressure"].any()
    assert not without_ball["ev_box_attackers"].any()


def test_compute_signals_lists_every_input_problem() -> None:
    shots = make_shots(("wide", 10, 1.0), ("close", 5, 0.0))
    shots.loc[1, "start_frame"] = 12
    tracks = make_tracks([{"frame": 0, "track_id": 1}]).drop(columns=["pitch_x"])
    with pytest.raises(HighlightInputError) as caught:
        compute_signals(
            tracks, None, None, None, shots, None, "29.97", attack_direction={"left": 1}
        )
    problems = "\n".join(caught.value.problems)
    assert "29.97" in problems
    assert "attack_direction" in problems
    assert "tile" in problems
    assert "pitch_x" in problems
    good = make_shots(("wide", 10, 1.0))
    ball = make_ball([{"frame": 3}, {"frame": 3}, {"frame": 12}])
    events = pd.DataFrame({"frame": [-1]})
    with pytest.raises(HighlightInputError) as caught:
        compute_signals(None, None, ball, events, good, None, FPS)
    problems = "\n".join(caught.value.problems)
    assert "ball_track has 1 row(s) with frames outside 0..9: 12" in problems
    assert "more than one row for frame(s) 3" in problems
    assert "ball_events has 1 row(s) with frames outside" in problems
    with pytest.raises(HighlightInputError):
        compute_signals(None, None, None, None, good.assign(shot_type="aerial"), None, FPS)


# --------------------------------------------------------------------------- detect_highlights


def test_peaks_are_found_at_the_planted_frames() -> None:
    shots = make_shots(("wide", 700, 1.0))
    signals = plant(
        base_signals(shots),
        [
            (120, {"audio": 1.0, "duel": 1.0}, 20),
            (330, {"audio": 0.8, "duel": 0.8}, 20),
            (560, {"audio": 0.6, "duel": 0.6}, 20),
        ],
    )
    result = detect_highlights(signals, shots, None, None, FPS)
    highlights = by_peak(result)
    assert [h["score_peak_frame"] for h in highlights] == [120, 330, 560]
    assert [h["peak_frame"] for h in highlights] == [120, 330, 560]
    assert [h["type"] for h in highlights] == ["duel_tackle"] * 3
    assert [h["id"] for h in result["highlights"]] == ["h01", "h02", "h03"]
    scores = [h["score"] for h in highlights]
    assert scores[0] > scores[1] > scores[2]


def test_minimum_separation_and_maximum_count_are_respected() -> None:
    shots = make_shots(("wide", 600, 1.0))
    signals = plant(
        base_signals(shots),
        [
            (100, {"audio": 1.0, "duel": 1.0}, 15),
            (150, {"audio": 0.9, "duel": 0.9}, 15),  # 2 s after a stronger peak
            (400, {"audio": 0.8, "duel": 0.8}, 15),
        ],
    )
    result = detect_highlights(signals, shots, None, None, FPS)
    peaks = [h["peak_frame"] for h in by_peak(result)]
    assert peaks == [100, 400]
    loose = detect_highlights(signals, shots, None, None, FPS, min_separation_s=1.0)
    assert [h["peak_frame"] for h in by_peak(loose)] == [100, 150, 400]
    for first in loose["highlights"]:
        for second in loose["highlights"]:
            if first is not second:
                assert abs(first["peak_frame"] - second["peak_frame"]) >= 25
    single = detect_highlights(signals, shots, None, None, FPS, max_highlights=1)
    assert [h["peak_frame"] for h in single["highlights"]] == [100]


def test_windows_are_snapped_to_shot_boundaries() -> None:
    shots = make_shots(("wide", 200, 1.0), ("wide", 400, 1.0))
    signals = plant(
        base_signals(shots),
        [
            (230, {"audio": 1.0, "duel": 1.0}, 10),  # 1.2 s after the cut
            (420, {"audio": 1.0, "duel": 1.0}, 10),
            (590, {"audio": 1.0, "duel": 1.0}, 10),  # 0.4 s before the end
        ],
    )
    windows = [
        (h["start_frame"], h["peak_frame"], h["end_frame"])
        for h in by_peak(detect_highlights(signals, shots, None, None, FPS))
    ]
    assert windows == [(200, 230, 280), (320, 420, 470), (490, 590, 599)]
    # A start 12 frames (< 0.5 s) after the cut snaps back to the cut.
    near = plant(base_signals(shots), [(312, {"audio": 1.0, "duel": 1.0}, 10)])
    (only,) = detect_highlights(near, shots, None, None, FPS)["highlights"]
    assert (only["start_frame"], only["end_frame"]) == (200, 362)


def test_an_emphasised_close_shot_after_the_action_joins_the_window() -> None:
    shots = make_shots(("wide", 300, 1.0), ("close", 150, 0.0), ("wide", 350, 1.0))
    shots.loc[1, "median_box_h"] = 900.0  # upper bodies: a close-up
    signals = plant(
        base_signals(shots),
        [(285, {"audio": 1.0, "strike": 1.0}, 12), (100, {"audio": 1.0, "duel": 1.0}, 12)],
    )
    events = pd.DataFrame({"frame": [287], "conf": [1.0], "track_id": [5]})
    result = detect_highlights(signals, shots, None, events, FPS)
    early, late = by_peak(result)
    assert late["type"] == "ball_strike"
    assert late["peak_frame"] == 287
    assert late["refined_by"] == "strike_event"
    assert (late["start_frame"], late["end_frame"]) == (187, 449)
    assert late["cues"]["cut_emphasis"]["extended_into_shot_id"] == 1
    assert late["cues"]["cut_emphasis"]["joined_shot_event"] is None
    assert late["track_ids"][0] == 5
    assert "片段延伸到导播随后切入的特写（第 300–449 帧）。" in late["summary_zh"]
    assert (early["start_frame"], early["end_frame"]) == (0, 150)
    assert early["cues"]["cut_emphasis"]["extended_into_shot_id"] is None


def test_a_duel_inside_the_joined_close_shot_is_reported() -> None:
    shots = make_shots(("wide", 300, 1.0), ("close", 150, 0.0), ("wide", 350, 1.0))
    signals = plant(
        base_signals(shots),
        [(285, {"audio": 1.0, "box_pressure": 1.0}, 12), (370, {"duel": 0.9}, 30)],
    )
    signals = ball_at_box(signals, 270, 299)
    # The duel in the close shot: players 31 and 42, closest (0.3 box heights) at frame 372.
    frames = np.arange(len(signals))
    inside = (frames >= 340) & (frames <= 400)
    signals.loc[inside, "ev_duel_id_a"] = 31
    signals.loc[inside, "ev_duel_id_b"] = 42
    signals.loc[inside, "ev_duel_unit"] = "box_h"
    signals.loc[inside, "ev_duel_dist"] = 0.3 + 0.02 * np.abs(frames[inside] - 372)
    tracks = make_tracks(
        [
            {"frame": 372, "track_id": 31, "jersey": "9", "jersey_conf": 0.9},
            {"frame": 372, "track_id": 42, "team": "right"},
        ]
    )
    result = detect_highlights(signals, shots, tracks, None, FPS)
    late = by_peak(result)[-1]
    assert late["type"] == "attack_in_box"
    assert late["title_zh"].endswith("在对方禁区内进攻"), "whose box is said explicitly"
    assert late["end_frame"] == 449
    moment = late["cues"]["cut_emphasis"]["joined_shot_event"]
    assert moment == {
        "frame": 372,
        "cue": "duel",
        "distance": 0.3,
        "unit": "box_h",
        "track_ids": [31, 42],
    }
    assert late["track_ids"][-2:] == [31, 42]
    summary = late["summary_zh"]
    assert (
        "左侧球队 9 号（轨迹 31）与右侧球队球员（轨迹 42）平滑后的距离在 00:14.88（第 372 帧）最近"
    ) in summary
    assert "这一帧相距约 0.3 个身高（像面估计，未验证）" in summary


def test_types_follow_the_dominant_cue() -> None:
    shots = make_shots(("close", 200, 0.0), ("wide", 1100, 1.0))
    shots.loc[0, "median_box_h"] = 900.0  # upper bodies: a close-up
    signals = plant(
        base_signals(shots),
        [
            (100, {"audio": 1.0}, 15),
            (350, {"audio": 0.5, "strike": 1.0}, 15),
            (550, {"audio": 0.5, "box_pressure": 1.0, "ball_goal": 0.8}, 15),
            (750, {"audio": 0.5, "duel": 1.0}, 15),
            (950, {"audio": 0.5, "speed": 1.0}, 15),
            (1150, {"audio": 1.0, "speed": 0.1}, 15),
        ],
    )
    signals = ball_at_box(signals, 535, 565)
    events = pd.DataFrame({"frame": [353], "conf": [0.9]})
    result = detect_highlights(signals, shots, None, events, FPS)
    found = [
        (h["score_peak_frame"], h["type"], h["peak_frame"], h["refined_by"])
        for h in by_peak(result)
    ]
    assert found == [
        (100, "closeup_reaction", 100, "score_peak"),
        (350, "ball_strike", 353, "strike_event"),
        (550, "attack_in_box", 550, "attack_peak"),
        (750, "duel_tackle", 750, "duel_min_distance"),
        (950, "sprint", 950, "speed_peak"),
        (1150, "closeup_reaction", 1150, "score_peak"),
    ]
    titles = [h["title_zh"] for h in by_peak(result)]
    assert "特写镜头中的现场反应" in titles[0]
    # A wide shot carried by the sound only: loudness and spectral flux, not "excitement".
    assert titles[5].endswith("现场声音响度变化明显的瞬间")
    # The frozen frame is the 关键帧 everywhere; the speed peak only chose it.
    sprint = by_peak(result)[4]
    assert "关键帧 00:38.00（第 950 帧，以速度最高的一帧为准）" in sprint["summary_zh"]
    assert all("峰值帧" not in text for text in REFINE_METHODS_ZH.values())


def test_type_names_claim_only_what_is_measured() -> None:
    # A ball event is an abrupt change of the ball's image velocity near a player, often a
    # first touch; a reaction is picked by sound and editing, often in a full-body shot.
    assert TYPE_NAMES_ZH["ball_strike"] == "触球（球速突变）"
    assert TYPE_NAMES_ZH["closeup_reaction"] == "球员反应"
    shots = make_shots(("close", 200, 0.0), ("wide", 600, 1.0))
    signals = plant(
        base_signals(shots),
        [(100, {"audio": 1.0}, 15), (350, {"audio": 0.5, "strike": 1.0}, 15)],
    )
    events = pd.DataFrame({"frame": [353], "conf": [0.9], "track_id": [5]})
    tracks = make_tracks([{"frame": 353, "track_id": 5}])
    reaction, touch = by_peak(detect_highlights(signals, shots, tracks, events, FPS))
    assert (reaction["type"], touch["type"]) == ("closeup_reaction", "ball_strike")
    assert touch["title_zh"] == "00:14.12（第 353 帧）左侧球队球员触球（球速突变）"
    assert touch["summary_zh"].startswith("触球（球速突变）。片段 ")
    assert reaction["summary_zh"].startswith("球员反应。片段 ")
    anonymous = by_peak(
        detect_highlights(signals, shots, None, events.drop(columns="track_id"), FPS)
    )
    assert anonymous[1]["title_zh"] == "00:14.12（第 353 帧）触球（球速突变）"
    for highlight in (reaction, touch, *anonymous):
        text = highlight["title_zh"] + highlight["summary_zh"]
        for claim in ("起脚", "击球", "特写反应"):
            assert claim not in text, claim


def test_the_result_is_deterministic_and_order_independent() -> None:
    shots = make_shots(("wide", 600, 1.0))
    signals = plant(
        base_signals(shots),
        [(100, {"audio": 1.0, "duel": 1.0}, 15), (400, {"audio": 1.0, "duel": 1.0}, 15)],
    )
    first = detect_highlights(signals, shots, None, None, FPS)
    second = detect_highlights(signals, shots, None, None, FPS)
    shuffled = signals.sample(frac=1.0, random_state=3)
    third = detect_highlights(shuffled, shots.iloc[::-1], None, None, FPS)
    text = json.dumps(first, sort_keys=True, ensure_ascii=False, allow_nan=False)
    assert text == json.dumps(second, sort_keys=True, ensure_ascii=False, allow_nan=False)
    assert text == json.dumps(third, sort_keys=True, ensure_ascii=False, allow_nan=False)
    # Two equal peaks: both are kept and listed in time order.
    assert [h["peak_frame"] for h in first["highlights"]] == [100, 400]


def test_chinese_titles_carry_the_time_the_frame_and_the_team_names() -> None:
    shots = make_shots(("wide", 1000, 1.0))
    signals = plant(base_signals(shots), [(750, {"audio": 1.0, "duel": 1.0}, 15)])
    signals.loc[740:760, "ev_duel_id_a"] = 11
    signals.loc[740:760, "ev_duel_id_b"] = 22
    signals.loc[740:760, "ev_duel_dist"] = 1.2
    signals.loc[740:760, "ev_duel_unit"] = "m"
    tracks = make_tracks(
        [
            {"frame": 750, "track_id": 11, "jersey": "10", "jersey_conf": 0.9},
            {"frame": 751, "track_id": 11, "jersey": "10", "jersey_conf": 0.8},
            {"frame": 750, "track_id": 22, "team": "right", "jersey": "7", "jersey_conf": 0.1},
        ]
    )
    named = detect_highlights(
        signals, shots, tracks, None, FPS, team_names={"left": "甲队", "right": "乙队"}
    )
    (highlight,) = named["highlights"]
    assert highlight["peak_frame"] == 750
    title = highlight["title_zh"]
    assert format_time(750, FPS) == "00:30.00"
    assert "00:30.00" in title
    assert "第 750 帧" in title
    assert "甲队 10 号" in title
    assert "乙队球员" in title
    assert "7 号" not in title  # a jersey read with low confidence is not printed
    assert highlight["track_ids"][:2] == [11, 22]
    summary = highlight["summary_zh"]
    assert "00:26.00 至 00:32.00" in summary
    pair = "甲队 10 号（轨迹 11）与乙队球员（轨迹 22）"
    assert f"第 750 帧{pair}相距约 1.2 米（A 档，未验证）" in summary
    assert "A 级" not in summary
    assert "未验证" in summary
    fallback = detect_highlights(signals, shots, tracks, None, FPS)
    assert "左侧球队 10 号" in fallback["highlights"][0]["title_zh"]
    assert "右侧球队球员" in fallback["highlights"][0]["title_zh"]


def test_summary_numbers_name_their_frame_and_tier() -> None:
    # A duel closest (smoothed) at frame 450, while the box count peaks at frame 462.
    shots = make_shots(("wide", 900, 1.0))
    signals = plant(
        base_signals(shots),
        [
            (450, {"audio": 1.0, "duel": 1.0}, 15),
            (462, {"box_pressure": 2 / 3, "ball_goal": 0.5}, 15),
        ],
    )
    frames = np.arange(len(signals))
    near = (frames >= 435) & (frames <= 465)
    signals.loc[near, "ev_duel_id_a"] = 11
    signals.loc[near, "ev_duel_id_b"] = 22
    signals.loc[near, "ev_duel_unit"] = "m"
    signals.loc[near, "ev_duel_dist"] = 1.0 + 0.05 * np.abs(frames[near] - 450)
    box = (frames >= 447) & (frames <= 477)
    signals.loc[box, "ev_box_attackers"] = 2
    signals.loc[box, "ev_box_team"] = "left"
    signals.loc[box, "ev_ball_team"] = "left"
    signals.loc[box, "ev_ball_goal_m"] = 18.0
    tracks = make_tracks(
        [{"frame": 450, "track_id": 11}, {"frame": 450, "track_id": 22, "team": "right"}]
    )
    names = {"left": "甲队", "right": "乙队"}
    (highlight,) = detect_highlights(signals, shots, tracks, None, FPS, team_names=names)[
        "highlights"
    ]
    assert (highlight["type"], highlight["peak_frame"]) == ("duel_tackle", 450)
    summary = highlight["summary_zh"]
    assert "关键帧 00:18.00（第 450 帧，球在两人脚下时距离最近的一帧）" in summary
    assert "精确时刻" not in summary
    pair = "甲队球员（轨迹 11）与乙队球员（轨迹 22）"
    assert f"第 450 帧{pair}相距约 1.0 米（A 档，未验证）" in summary
    # The attackers in the box carry the tier of the attackers-in-box measurement (B).
    assert "第 462 帧甲队 2 名球员位于对方禁区内（B 档，实验，未验证）" in summary
    assert "第 462 帧球距甲队进攻的球门约 18 米（地面近似，未验证）" in summary
    assert "A 级" not in summary
    assert highlight["cues"]["box_pressure"]["frame"] == 462


def test_ball_events_are_called_touches_not_strikes() -> None:
    # A ball event at frame 152; a duel at the end of the wide shot, whose window joins the
    # close-up after it, with another ball event at frame 370 inside the close-up.
    shots = make_shots(("wide", 300, 1.0), ("close", 150, 0.0), ("wide", 350, 1.0))
    signals = plant(
        base_signals(shots),
        [
            (150, {"audio": 1.0, "strike": 1.0}, 12),
            (285, {"audio": 1.0, "duel": 0.5}, 12),
            (370, {"strike": 0.8}, 12),
        ],
    )
    signals.loc[138:162, "ev_strike_frame"] = 152
    signals.loc[358:382, "ev_strike_frame"] = 370
    events = pd.DataFrame({"frame": [152, 370], "conf": [1.0, 0.8], "track_id": [5, 6]})
    tracks = make_tracks([{"frame": 370, "track_id": 6, "team": "right"}])
    first, second = by_peak(detect_highlights(signals, shots, tracks, events, FPS))
    assert first["refined_by"] == "strike_event"
    assert "（第 152 帧，以球的像面速度突变的一帧为准）" in first["summary_zh"]
    assert "第 152 帧检测到触球（球的像面速度突变）" in first["summary_zh"]
    # The ball event inside the close-up that joined the second window.
    moment = second["cues"]["cut_emphasis"]["joined_shot_event"]
    assert (moment["cue"], moment["frame"]) == ("strike", 370)
    touch = "00:14.80（第 370 帧）右侧球队球员（轨迹 6）触球（球的像面速度突变）"
    assert touch in second["summary_zh"]
    for highlight in (first, second):
        assert "检测到击球" not in highlight["summary_zh"]
        assert "击球（球的像面速度突变）" not in highlight["summary_zh"]


def test_shots_are_named_by_the_size_of_their_players() -> None:
    shots = make_shots(
        ("wide", 300, 1.0), ("close", 300, 0.0), ("wide", 300, 1.0), ("close", 300, 0.0)
    )
    shots.loc[1, "median_box_h"] = 400.0  # whole players, head to foot: 中景, not 特写
    shots.loc[3, "median_box_h"] = 900.0  # upper bodies cut by the frame edge: 特写
    signals = plant(
        base_signals(shots),
        [
            (285, {"audio": 1.0, "strike": 1.0}, 12),
            (585, {"audio": 1.0}, 15),
            (885, {"audio": 1.0, "strike": 1.0}, 12),
            (1185, {"audio": 1.0}, 15),
        ],
    )
    found = by_peak(detect_highlights(signals, shots, None, None, FPS))
    assert [h["type"] for h in found] == ["ball_strike", "closeup_reaction"] * 2
    first, whole_reaction, second, close_reaction = found
    assert "导播在 00:12.00（第 300 帧）由全景切到中景" in first["summary_zh"]
    assert "片段延伸到导播随后切入的中景（第 300–549 帧）" in first["summary_zh"]
    assert whole_reaction["title_zh"].endswith("中景镜头中的现场反应")
    assert "导播在 00:36.00（第 900 帧）由全景切到特写" in second["summary_zh"]
    assert "片段延伸到导播随后切入的特写（第 900–1149 帧）" in second["summary_zh"]
    assert close_reaction["title_zh"].endswith("特写镜头中的现场反应")
    for highlight in found:
        assert "近景" not in highlight["title_zh"] + highlight["summary_zh"]
    # The size rule on its own: one threshold with the image-plane duel and gbf.measure's
    # feet rule (from half the frame the box bottom is the frame edge, not the feet); a
    # close shot without a person box keeps 特写.
    assert SHOT_UPPER_BODY_MIN_BOX_H_PX == DUEL_IMAGE_MAX_BOX_H_PX == 540.0
    assert MeasureParams().image_feet_max_median_box_px == SHOT_UPPER_BODY_MIN_BOX_H_PX
    cases = [("wide", 100.0), ("medium", 180.0), ("close", 400.0), ("close", 539.9)]
    cases += [("close", 540.0), ("close", 900.0)]
    names = [shot_name_zh(kind, height) for kind, height in cases]
    assert names == ["全景", "中景", "中景", "中景", "特写", "特写"]
    assert shot_name_zh("close", None) == shot_name_zh("close", float("nan")) == "特写"


def test_the_close_up_subject_is_the_player_seen_largest_for_longest() -> None:
    shots = make_shots(("close", 200, 0.0), ("wide", 200, 1.0))
    shots.loc[0, "median_box_h"] = 900.0  # upper bodies: a close-up
    signals = plant(base_signals(shots), [(100, {"audio": 1.0}, 15)])
    rows = [{"frame": f, "track_id": 3, "y1": 100.0, "y2": 1000.0} for f in range(200)]
    # A passer-by fills the frame once; the referee is not a player.
    rows += [{"frame": 100, "track_id": 4, "team": "right", "y1": 0.0, "y2": 1000.0}]
    rows += [
        {"frame": f, "track_id": 5, "role": "referee", "team": None, "y1": 0.0, "y2": 1060.0}
        for f in range(200)
    ]
    result = detect_highlights(signals, shots, make_tracks(rows), None, FPS)
    (highlight,) = result["highlights"]
    assert highlight["type"] == "closeup_reaction"
    assert highlight["track_ids"][0] == 3
    assert highlight["title_zh"].endswith("特写：左侧球队球员的反应")


def test_players_are_named_from_their_rows_inside_the_window() -> None:
    # The tracker reused id 7 after the cut at frame 300 for a player of the other team.
    shots = make_shots(("wide", 300, 1.0), ("wide", 300, 1.0))
    signals = plant(base_signals(shots), [(450, {"audio": 1.0, "duel": 1.0}, 15)])
    signals.loc[440:460, "ev_duel_id_a"] = 8
    signals.loc[440:460, "ev_duel_id_b"] = 7
    signals.loc[440:460, "ev_duel_dist"] = 1.0
    signals.loc[440:460, "ev_duel_unit"] = "m"
    rows = [{"frame": f, "track_id": 7, "team": "left"} for f in range(0, 300, 2)]
    rows += [{"frame": f, "track_id": 7, "team": "right"} for f in range(300, 600, 10)]
    rows += [{"frame": 450, "track_id": 8, "team": "left"}]
    result = detect_highlights(signals, shots, make_tracks(rows), None, FPS)
    (highlight,) = result["highlights"]
    assert "左侧球队球员与右侧球队球员贴身争抢" in highlight["title_zh"]


def test_format_time_is_frame_accurate() -> None:
    assert format_time(0, FPS) == "00:00.00"
    assert format_time(459, FPS) == "00:18.36"
    assert format_time(1680, FPS) == "01:07.20"
    assert format_time(1, "30000/1001") == "00:00.03"
    # The hundredths are floored, as on the banner, the card and the report's tables: frame
    # 2 is 0.0667 s (titles rounded it to 00:00.07, the card floored it to 00:00.06).
    assert format_time(2, "30000/1001") == "00:00.06"
    assert format_time(23, "24000/1001") == "00:00.95"
    with pytest.raises(HighlightInputError):
        format_time(-1, FPS)
    with pytest.raises(HighlightInputError):
        format_time(1, "29.97")


def test_the_result_follows_the_contract_and_is_json() -> None:
    shots = make_shots(("wide", 400, 1.0), ("close", 100, 0.0))
    signals = plant(
        base_signals(shots),
        [(150, {"audio": 1.0, "duel": 0.7}, 15), (390, {"audio": 1.0, "strike": 0.9}, 15)],
    )
    result = detect_highlights(signals, shots, None, None, FPS)
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    assert result["schema_version"] == 1
    assert result["fps"] == "25/1"
    assert result["frame_count"] == 500
    assert len(result["highlights"]) == 2
    keys = {
        "id",
        "type",
        "start_frame",
        "peak_frame",
        "end_frame",
        "score",
        "cues",
        "track_ids",
        "title_zh",
        "summary_zh",
    }
    for highlight in result["highlights"]:
        assert keys <= set(highlight)
        assert highlight["type"] in HIGHLIGHT_TYPES
        assert highlight["start_frame"] <= highlight["peak_frame"] <= highlight["end_frame"]
        assert set(highlight["cues"]) == set(CUE_COLUMNS)
        total = sum(entry["contribution"] for entry in highlight["cues"].values())
        assert total == pytest.approx(highlight["score"], abs=1e-3)


def test_detect_highlights_lists_every_input_problem() -> None:
    shots = make_shots(("wide", 50, 1.0))
    signals = base_signals(shots).iloc[:-1]
    with pytest.raises(HighlightInputError) as caught:
        detect_highlights(
            signals,
            shots,
            None,
            pd.DataFrame({"frame": [70]}),
            FPS,
            min_separation_s=-1.0,
            max_highlights=0,
            team_names={"home": "主队"},
        )
    problems = "\n".join(caught.value.problems)
    for fragment in ("one row per frame", "min_separation_s", "max_highlights", "home", "70"):
        assert fragment in problems
    wrong_shot = base_signals(shots).assign(shot_id=9)
    with pytest.raises(HighlightInputError) as caught:
        detect_highlights(wrong_shot, shots, None, None, FPS)
    assert "shot_id disagrees" in "\n".join(caught.value.problems)
    raw_z = base_signals(shots).assign(audio=3.2)  # a raw z-score is not a cue
    with pytest.raises(HighlightInputError) as caught:
        detect_highlights(raw_z, shots, None, None, FPS)
    assert "signals.audio must lie in [0, 1]" in "\n".join(caught.value.problems)


def test_events_without_a_foot_nearby_do_not_move_the_peak() -> None:
    shots = make_shots(("wide", 300, 1.0))
    signals = plant(base_signals(shots), [(150, {"audio": 1.0, "strike": 1.0}, 15)])
    events = pd.DataFrame(
        {
            "frame": [160],
            "magnitude": [30.0],
            "nearest_track_id": [4],
            "nearest_dist_box_h": [2.5],  # nobody at the ball: not a kick, strength 0
        }
    )
    (highlight,) = detect_highlights(signals, shots, None, events, FPS)["highlights"]
    assert highlight["type"] == "ball_strike"
    assert highlight["peak_frame"] == 150  # the strike cue's own peak, not the empty event
    assert 4 not in highlight["track_ids"]


def test_signals_survive_a_parquet_round_trip(unicode_tmp_path: Path) -> None:
    shots = make_shots(("wide", 300, 1.0), ("close", 100, 0.0))
    tracks = make_tracks(
        [{"frame": f, "track_id": 1, "pitch_x": 40.0, "pitch_y": 0.0} for f in range(300)]
        + [{"frame": f, "track_id": 2, "team": "right", "pitch_x": 41.0} for f in range(300)]
    )
    ball = make_ball(
        [{"frame": f, "u": 1.0, "v": 1.0, "pitch_x": 40.5, "pitch_y": 0.2} for f in range(300)]
    )
    signals = compute_signals(tracks, None, ball, None, shots, None, FPS)
    path = unicode_tmp_path / "signals 信号.parquet"
    signals.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    direct = detect_highlights(signals, shots, tracks, None, FPS)
    again = detect_highlights(back, shots, tracks, None, FPS)
    assert direct == again
    assert direct["highlights"]


def test_gaussian_smoothing_keeps_constants_and_symmetry() -> None:
    np.testing.assert_allclose(gaussian_smooth(np.full(30, 2.0), 3.0), 2.0)
    spike = np.zeros(41)
    spike[20] = 1.0
    smoothed = gaussian_smooth(spike, 2.0)
    assert int(np.argmax(smoothed)) == 20
    np.testing.assert_allclose(smoothed, smoothed[::-1])
    np.testing.assert_array_equal(gaussian_smooth(spike, 0.0), spike)


# --------------------------------------------------------------------------- end to end


def scenario() -> tuple[pd.DataFrame, ...]:
    """16 s wide shot then a 4 s close-up; team left attacks x > 0.

    Player 1 (left) dribbles into the box and shoots at frame 385 (ball event); players 2
    and 3 (left) follow into the box; player 4 (right) defends; the crowd gets loud from
    frame 380. The close-up (frames 400..499) shows player 1 alone.
    """
    shots = make_shots(("wide", 400, 1.0), ("close", 100, 0.0))
    rows, kin, balls = [], [], []
    for f in range(400):
        progress = min(1.0, f / 385)
        x1 = 10.0 + 30.0 * progress
        positions = {
            1: (x1, 2.0, "left"),
            2: (x1 - 2.0, -6.0, "left"),
            3: (x1 - 3.5, 8.0, "left"),
            4: (x1 + 3.0, 1.0, "right"),
        }
        for track, (x, y, team) in positions.items():
            rows.append({"frame": f, "track_id": track, "team": team, "pitch_x": x, "pitch_y": y})
        speed = 4.0 + 4.5 * np.exp(-(((f - 370) / 25.0) ** 2))
        kin.append((f, 1, float(speed), True))
        kin.append((f, 4, 5.0, True))
        ball_x = x1 + 0.6 if f <= 385 else min(52.0, x1 + 0.6 + 0.8 * (f - 385))
        balls.append({"frame": f, "u": 900.0, "v": 500.0, "pitch_x": ball_x, "pitch_y": 2.0})
    for f in range(400, 500):
        rows.append(
            {
                "frame": f,
                "track_id": 1,
                "pitch_usable": False,
                "y1": 100.0,
                "y2": 900.0,
                "foot_u": 900.0,
                "foot_v": 900.0,
            }
        )
    tracks = make_tracks(rows)
    kinematics = make_kinematics(kin)
    ball = make_ball(balls)
    events = pd.DataFrame({"frame": [385], "conf": [0.9], "track_id": [1]})
    z = np.where(np.arange(500) >= 380, 3.5, -0.2)
    z[460:] = 0.5
    audio = pd.DataFrame(
        {
            "frame": np.arange(500, dtype=np.int64),
            "rms_dbfs": np.zeros(500),
            "flux": np.zeros(500),
            "excitement_z": z,
            "has_audio": np.ones(500, dtype=bool),
        }
    )
    return tracks, kinematics, ball, events, shots, audio


def test_end_to_end_a_shot_in_the_box_followed_by_a_close_up() -> None:
    tracks, kinematics, ball, events, shots, audio = scenario()
    signals = compute_signals(tracks, kinematics, ball, events, shots, audio, FPS)
    result = detect_highlights(
        signals, shots, tracks, events, FPS, team_names={"left": "蓝队", "right": "白队"}
    )
    top = max(result["highlights"], key=lambda h: h["score"])
    assert top["type"] in ("ball_strike", "attack_in_box")
    assert top["peak_frame"] == 385
    assert top["refined_by"] == "strike_event"
    assert top["end_frame"] == 499  # the close-up after the shot is part of the moment
    assert top["track_ids"][0] == 1
    assert "00:15.40" in top["title_zh"]
    assert "第 385 帧" in top["title_zh"]
    assert top["cues"]["box_pressure"]["attackers"] == 3
    assert top["cues"]["box_pressure"]["team"] == "left"
    assert top["cues"]["strike"]["event_frame"] == 385
    assert "蓝队" in top["summary_zh"]


def test_one_attacker_in_the_box_without_the_ball_is_not_an_attack() -> None:
    shots = make_shots(("wide", 900, 1.0))
    signals = plant(
        base_signals(shots),
        [
            (150, {"audio": 1.0, "box_pressure": 1 / 3}, 40),  # one attacker, no ball near goal
            (450, {"audio": 1.0, "box_pressure": 2 / 3}, 40),  # two attackers
            (750, {"audio": 1.0, "box_pressure": 1 / 3, "ball_goal": 0.6}, 40),  # ball near goal
        ],
    )
    signals = ball_at_box(signals, 0, 899)  # the ball is at the box all along
    result = detect_highlights(signals, shots, None, None, FPS)
    assert [(h["score_peak_frame"], h["type"]) for h in by_peak(result)] == [
        (150, "closeup_reaction"),
        (450, "attack_in_box"),
        (750, "attack_in_box"),
    ]


# --------------------------------------------------------------------------- ball at the box


def _box_scene(ball_x: float) -> tuple[pd.DataFrame, ...]:
    """A 300-frame wide shot: left players 1 and 2 run into the box they attack (x > 36 m)
    for frames 100-200, left player 3 carries the ball at (ball_x, 0), right player 4 is far
    away; the crowd is loud on frames 140-160."""
    shots = make_shots(("wide", 300, 1.0))
    rows = []
    for f in range(300):
        x = 40.0 if 100 <= f <= 200 else 30.0
        spots = {1: (x, 5.0, "left"), 2: (x, -5.0, "left"), 3: (ball_x - 1.0, 0.0, "left")}
        spots[4] = (20.0, 20.0, "right")
        rows += [
            {"frame": f, "track_id": t, "team": team, "pitch_x": x, "pitch_y": y}
            for t, (x, y, team) in spots.items()
        ]
    ball = make_ball([{"frame": f, "u": 900.0, "v": 500.0, "pitch_x": ball_x} for f in range(300)])
    ball["pitch_y"] = 0.0
    z = np.where((np.arange(300) >= 140) & (np.arange(300) <= 160), 4.0, 0.0)
    audio = pd.DataFrame(
        {
            "frame": np.arange(300, dtype=np.int64),
            "rms_dbfs": np.zeros(300),
            "flux": np.zeros(300),
            "excitement_z": z,
            "has_audio": np.ones(300, dtype=bool),
        }
    )
    return make_tracks(rows), ball, shots, audio


def test_the_distance_of_the_ball_to_the_box_is_read_within_a_second_in_the_shot() -> None:
    # Two left attackers stand in the box they attack all along. The ball is detected 6 m
    # out (x = 30, the box line is at 36) on frames 0-39, interpolated 3 m out on 40-59,
    # detected inside on 60-99; after the cut at 100 it is 1.5 m out; after the cut at 160
    # it is never seen.
    shots = make_shots(("wide", 100, 1.0), ("wide", 60, 1.0), ("wide", 60, 1.0))
    rows = [
        {"frame": f, "track_id": t, "pitch_x": 40.0, "pitch_y": y}
        for f in range(220)
        for t, y in ((1, 5.0), (2, -5.0))
    ]
    balls = [{"frame": f, "u": 900.0, "v": 500.0, "pitch_x": 30.0} for f in range(40)]
    balls += [
        {"frame": f, "u": 900.0, "v": 500.0, "pitch_x": 33.0, "status": "interpolated"}
        for f in range(40, 60)
    ]
    balls += [{"frame": f, "u": 900.0, "v": 500.0, "pitch_x": 37.0} for f in range(60, 100)]
    balls += [{"frame": f, "u": 900.0, "v": 500.0, "pitch_x": 34.5} for f in range(100, 160)]
    balls += [{"frame": f, "status": "missing"} for f in range(160, 220)]
    ball = make_ball(balls)
    ball["pitch_y"] = np.where(ball["status"] == "missing", np.nan, 0.0)
    tracks = make_tracks(rows)
    signals = compute_signals(tracks, None, ball, None, shots, None, FPS)
    near = signals["ev_ball_box_m"].to_numpy()
    assert "ev_ball_box_m" in highlights.EVIDENCE_COLUMNS
    assert near[10] == pytest.approx(6.0)  # frames 0-35: detected 6 m out
    assert near[20] == pytest.approx(3.0)  # the interpolated rows count (frames 40-45)
    assert near[35] == 0.0  # frame 60 lies within a second
    assert near[99] == 0.0
    assert near[100] == pytest.approx(1.5), "the frames before the cut do not count"
    assert np.isnan(near[180]), "no ball in sight: no attacking side, no distance"
    assert signals["ev_box_attackers"].tolist()[10] == 2
    # The distance is to the box the named side attacks: left attacks x < 0 here, and the
    # ball nearest a left player is 66 m from that box.
    flipped = compute_signals(
        tracks, None, ball, None, shots, None, FPS, attack_direction={"left": -1, "right": 1}
    )
    assert flipped["ev_ball_box_m"].to_numpy()[10] == pytest.approx(66.0)


@pytest.mark.parametrize(
    ("where", "attack"),
    [("inside", True), ("margin", True), ("beyond", False), ("far", False), ("unknown", False)],
)
def test_attackers_in_the_box_make_an_attack_only_with_the_ball_at_the_box(
    where: str, attack: bool
) -> None:
    # Two attackers in the box and the ball 18 m from goal: an attack in the box only when
    # the ball is in the box, or within the calibration margin of it; not when it is 7 m
    # out, or when nothing says where it is.
    shots = make_shots(("wide", 900, 1.0))
    signals = plant(
        base_signals(shots), [(450, {"audio": 1.0, "box_pressure": 2 / 3, "ball_goal": 0.5}, 40)]
    )
    signals.loc[400:500, "ev_box_attackers"] = 2
    signals.loc[400:500, "ev_box_team"] = "left"
    margin = highlights.BALL_AT_BOX_MARGIN_M
    distance = {"inside": 0.0, "margin": margin, "beyond": margin + 0.5, "far": 7.0}.get(where)
    if distance is not None:
        signals = ball_at_box(signals, 0, 899, distance)
    signals = box_pressure_at_the_box(signals)
    (found,) = detect_highlights(signals, shots, None, None, FPS)["highlights"]
    if attack:
        assert found["type"] == "attack_in_box"
        assert found["title_zh"].endswith("左侧球队在对方禁区内进攻")
        assert found["cues"]["box_pressure"]["ball_box_m"] == pytest.approx(distance)
    else:
        assert found["type"] == "closeup_reaction"
        assert "禁区内进攻" not in found["title_zh"]
        assert found["title_zh"].endswith("现场声音响度变化明显的瞬间")
        # Without the ball at the box the attackers count in the score no more than in the
        # type, and the summary does not give them as a reason.
        assert found["cues"]["box_pressure"]["contribution"] == 0.0
        assert "名球员位于对方禁区内" not in found["summary_zh"]


@pytest.mark.parametrize("ball_x", [38.0, 34.5, 34.0, 30.0])
def test_attackers_count_in_the_score_only_with_the_ball_at_the_box(ball_x: float) -> None:
    # Two attackers stand in the box on frames 100-200 while the ball is in it (x = 38; the
    # box line is at 36), at the calibration margin (1.5 m out), just beyond it (2 m) or
    # 6 m out. The cue is meant to see an attack in the box near the ball, so the score
    # counts the attackers only in the first two cases, as the type does; the evidence names
    # them always.
    tracks, ball, shots, audio = _box_scene(ball_x)
    signals = compute_signals(tracks, None, ball, None, shots, audio, FPS)
    attackers = 2 + int(ball_x - 1.0 >= 36.0)  # the ball carrier is in the box too at x = 38
    assert (signals["ev_box_attackers"].to_numpy()[100:201] == attackers).all()
    assert set(signals["ev_box_team"].tolist()[100:201]) == {"left"}
    at_box = 36.0 - ball_x <= highlights.BALL_AT_BOX_MARGIN_M
    expected = min(1.0, attackers / highlights.BOX_FULL_ATTACKERS) if at_box else 0.0
    np.testing.assert_allclose(signals["box_pressure"].to_numpy()[100:201], expected)
    weighted = sum(CUE_WEIGHTS[c] * signals[c].to_numpy() for c in CUE_COLUMNS) / TOTAL_WEIGHT
    np.testing.assert_allclose(signals["score"].to_numpy(), weighted)
    # The ball's own distance to the goal still counts: 22.5 m and less is within 25 m.
    if ball_x <= 34.5:
        assert signals["ball_goal"].to_numpy()[150] == pytest.approx((25 - (52.5 - ball_x)) / 14)


@pytest.mark.parametrize(("ball_x", "kind"), [(30.0, "closeup_reaction"), (38.0, "attack_in_box")])
def test_end_to_end_runners_in_the_box_with_the_ball_far_out_are_no_attack(
    ball_x: float, kind: str
) -> None:
    tracks, ball, shots, audio = _box_scene(ball_x)
    signals = compute_signals(tracks, None, ball, None, shots, audio, FPS)
    assert signals["ev_box_attackers"].tolist()[150] >= 2
    expected = 6.0 if ball_x < 36.0 else 0.0
    assert signals["ev_ball_box_m"].to_numpy()[150] == pytest.approx(expected)
    names = {"left": "甲队", "right": "乙队"}
    result = detect_highlights(signals, shots, tracks, None, FPS, team_names=names)
    found = min(result["highlights"], key=lambda h: abs(h["score_peak_frame"] - 150))
    assert abs(found["score_peak_frame"] - 150) <= 10
    assert found["type"] == kind
    if kind == "attack_in_box":
        assert found["title_zh"].endswith("甲队在对方禁区内进攻")
    else:
        assert "禁区内进攻" not in found["title_zh"]


def test_an_attack_keeps_its_key_frame_where_its_condition_holds() -> None:
    # Two attackers in the box on frames 411-489; the ball is 5 m out before frame 445 and
    # in the box from then on. A ball event at frame 432 (the ball still out) does not
    # become the key frame: the title would name a frame without an attack in the box.
    shots = make_shots(("wide", 900, 1.0))
    signals = plant(base_signals(shots), [(450, {"audio": 0.5, "box_pressure": 2 / 3}, 40)])
    signals = ball_at_box(ball_at_box(signals, 400, 444, 5.0), 445, 500, 0.0)
    events = pd.DataFrame({"frame": [432], "conf": [1.0]})
    (found,) = detect_highlights(signals, shots, None, events, FPS)["highlights"]
    assert (found["type"], found["peak_frame"], found["refined_by"]) == (
        "attack_in_box",
        450,
        "attack_peak",
    )
    assert "（第 450 帧）" in found["title_zh"]
    # An event on a frame where the condition holds still sets the key frame.
    events = pd.DataFrame({"frame": [432, 455], "conf": [1.0, 0.5]})
    (found,) = detect_highlights(signals, shots, None, events, FPS)["highlights"]
    assert (found["peak_frame"], found["refined_by"]) == (455, "strike_event")


# --------------------------------------------------------------------------- type and title


@pytest.mark.parametrize(("cue", "kind"), [("duel", "duel_tackle"), ("speed", "sprint")])
def test_a_lone_attacker_does_not_hide_a_duel_or_a_sprint(cue: str, kind: str) -> None:
    # One attacker in the box (box pressure 1/3, below BOX_PRESSURE_ALONE_MIN) and the ball
    # 21 m from goal, at the box: the attack group's larger contribution used to win the
    # contest and then fail the minimum, so a 0.45 duel or run became a closeup_reaction.
    shots = make_shots(("wide", 900, 1.0))
    bump = {cue: 0.45, "box_pressure": 1 / 3, "ball_goal": 0.15}
    signals = ball_at_box(plant(base_signals(shots), [(450, bump, 60)]), 0, 899)
    (found,) = detect_highlights(signals, shots, None, None, FPS)["highlights"]
    assert found["type"] == kind
    # With a second attacker in the box the attack is the stronger evidence.
    two = plant(base_signals(shots), [(450, {**bump, "box_pressure": 2 / 3}, 60)])
    (found,) = detect_highlights(ball_at_box(two, 0, 899), shots, None, None, FPS)["highlights"]
    assert found["type"] == "attack_in_box"


@pytest.mark.parametrize(
    ("audio", "cut", "ending"),
    [
        (0.0, False, "综合评分入选的瞬间"),
        (0.05, False, "综合评分入选的瞬间"),  # sound below BROADCAST_CUE_MIN
        (1.0, False, "现场声音响度变化明显的瞬间"),
        (0.0, True, "导播切镜强调的瞬间"),
    ],
)
def test_a_wide_reaction_title_names_only_the_broadcast_cue_that_carried_it(
    audio: float, cut: bool, ending: str
) -> None:
    # Five play cues at 0.19, none reaching PLAY_EVIDENCE_MIN: a closeup_reaction in a wide
    # shot. The title named the sound even when the clip had no sound cue at all.
    spec = (("wide", 448, 1.0), ("close", 452, 0.0)) if cut else (("wide", 900, 1.0),)
    shots = make_shots(*spec)
    play = {name: 0.19 for name in ("strike", "box_pressure", "ball_goal", "duel", "speed")}
    bump = {**play, "audio": audio} if audio else play
    signals = ball_at_box(plant(base_signals(shots), [(430, bump, 60)]), 0, 899)
    found = detect_highlights(signals, shots, None, None, FPS)["highlights"]
    first = min(found, key=lambda h: abs(h["score_peak_frame"] - 430))
    assert first["type"] == "closeup_reaction"
    assert first["title_zh"].endswith(ending)
    if ending == "综合评分入选的瞬间":
        cues = first["cues"]
        assert cues["audio"]["contribution"] < highlights.BROADCAST_CUE_MIN
        assert cues["cut_emphasis"]["contribution"] < highlights.BROADCAST_CUE_MIN
        assert "声音" not in first["title_zh"]
        assert "导播" not in first["title_zh"]


# --------------------------------------------------------------------------- summary text


@pytest.mark.parametrize("mps", [8.68234567, 4.0147, 4.0658, 8.7061, 6.1132])
def test_a_summary_speed_is_the_speed_of_its_frame_rounded_as_the_table_rounds_it(
    mps: float,
) -> None:
    # 8.6823 m/s was stored as 8.68 and shown as 31.2 km/h next to the table's 31.3; 8.7061
    # would have read 31.4 against 31.3. The cue keeps gbf.measure's four decimals and the
    # text rounds once, as gbf.measure.display_text does; it is the speed of that frame
    # (the chord around it), and the text names the track. Nothing compares it with the
    # player's other frames, so it is neither called a peak nor said not to be one (非峰值
    # was such an unmeasured claim).
    shots = make_shots(("wide", 900, 1.0))
    signals = plant(base_signals(shots), [(450, {"audio": 1.0, "speed": 0.5}, 40)])
    signals.loc[440:460, "ev_speed_mps"] = mps
    signals.loc[440:460, "ev_speed_id"] = 9
    tracks = make_tracks([{"frame": 450, "track_id": 9}])
    names = {"left": "甲队", "right": "乙队"}
    (found,) = detect_highlights(signals, shots, tracks, None, FPS, team_names=names)["highlights"]
    assert found["type"] == "sprint"
    stored = found["cues"]["speed"]["speed_mps"]
    assert stored == round(mps, 4)
    shown = display_text(stored, "m/s", 1, Fraction(25))  # the measurement table's text
    summary = found["summary_zh"]
    assert (
        f"第 450 帧甲队球员（轨迹 9）的速度约 {shown}（该帧前后 0.4 秒平均；A 档，未验证）"
    ) in summary
    assert "峰值" not in summary
    assert "涉及球员：甲队球员（轨迹 9）。" in summary


def test_summaries_name_players_with_their_track_ids() -> None:
    name = highlights.player_name_zh
    assert name(7, "甲队", "9") == "甲队 9 号（轨迹 7）"
    assert name(7, "甲队") == "甲队球员（轨迹 7）"
    assert name(7, None, "9") == "9 号球员（轨迹 7）"
    assert name(7) == "轨迹 7 球员"
    tracks = make_tracks(
        [
            {"frame": 10, "track_id": 5, "jersey": "7", "jersey_conf": 0.9},
            {"frame": 10, "track_id": 6, "team": "right"},
            {"frame": 11, "track_id": 6, "team": "left"},  # a tie: no team
            {"frame": 90, "track_id": 6, "team": "right"},  # outside the frames asked for
        ]
    )
    named = highlights.player_names_zh(tracks, [5, 6, 8], 0, 50, {"left": "甲队"})
    assert named == {5: "甲队 7 号（轨迹 5）", 6: "轨迹 6 球员", 8: "轨迹 8 球员"}
    assert highlights.player_names_zh(None, [3], 0, 9) == {3: "轨迹 3 球员"}
