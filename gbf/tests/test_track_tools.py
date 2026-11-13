"""Tests for gbf.track_tools."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from gbf import track_tools as tt

SHOTS = pd.DataFrame({"shot_id": [0, 1, 2], "start_frame": [0, 10, 20], "end_frame": [9, 19, 29]})


def _rows(spec: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame({"frame": [f for f, _ in spec], "track_id": [t for _, t in spec]})


def test_ids_reused_after_a_cut_get_new_ids() -> None:
    tracks = _rows([(0, 7), (5, 7), (12, 7), (15, 8), (25, 7), (26, -1), (27, 8)])
    out, mapping = tt.split_tracks_at_cuts(tracks, SHOTS)
    assert out["track_id"].tolist() == [7, 7, 100_007, 8, 200_007, -1, 100_008]
    assert out["source_track_id"].tolist() == tracks["track_id"].tolist()
    assert mapping.to_dict("records") == [
        {"source_track_id": 7, "shot_id": 0, "track_id": 7},
        {"source_track_id": 7, "shot_id": 1, "track_id": 100_007},
        {"source_track_id": 7, "shot_id": 2, "track_id": 200_007},
        {"source_track_id": 8, "shot_id": 1, "track_id": 8},
        {"source_track_id": 8, "shot_id": 2, "track_id": 100_008},
    ]


def test_rows_outside_every_shot_are_refused() -> None:
    with pytest.raises(tt.TrackToolsError, match="outside every shot"):
        tt.split_tracks_at_cuts(_rows([(40, 1)]), SHOTS)


def test_only_confident_consistent_jerseys_survive() -> None:
    rows = []
    for i in range(6):
        rows.append({"track_id": 1, "jersey": "20", "jersey_conf": 0.9})  # kept
        rows.append({"track_id": 2, "jersey": "11", "jersey_conf": 0.01})  # low confidence
        rows.append({"track_id": 3, "jersey": "7" if i < 3 else "1", "jersey_conf": 0.9})  # split
    rows.append({"track_id": 4, "jersey": "9", "jersey_conf": 0.95})  # too few reads
    rows.append({"track_id": 1, "jersey": None, "jersey_conf": math.nan})
    out = tt.confident_jerseys(pd.DataFrame(rows))
    by_track = out.groupby("track_id")["jersey"].agg(lambda s: set(s.dropna()))
    assert by_track[1] == {"20"}
    assert by_track[2] == set()
    assert by_track[3] == set()
    assert by_track[4] == set()
    assert out.loc[out["track_id"] == 2, "jersey_conf"].isna().all()


def test_a_box_on_a_players_own_spot_is_no_second_player() -> None:
    # A box around two players that ends at the feet of the front one (the detector merging
    # a player with the one behind) shares the front player's bottom edge: it is that
    # player's spot, whether the boxes overlap much or not.
    front = (100.0, 60.0, 130.0, 140.0)
    merged = (101.0, 10.0, 129.0, 140.0)
    assert tt.box_iou(front, merged) < tt.SAME_SPOT_IOU
    assert tt.same_spot(front, merged)
    assert tt.same_spot(front, merged, (5.0, 3.0), (5.03, 3.0)), "0.03 m on the grass"
    # A player standing right in front of another overlaps it as much, but has feet of its
    # own: a separate player, at any IoU below SAME_SPOT_IOU.
    behind = (101.0, 45.0, 129.0, 125.0)
    assert tt.box_iou(front, behind) > 0.5
    assert not tt.same_spot(front, behind)
    assert not tt.same_spot(front, behind, (5.0, 3.0), (5.0, 6.0))
    # Two players a step apart on the grass are two players; closer than the floor, one.
    beside = (125.0, 61.0, 155.0, 141.0)
    assert not tt.same_spot(front, beside, (5.0, 3.0), (5.0 + tt.MIN_PLAYER_GAP_M + 0.01, 3.0))
    assert tt.same_spot(front, beside, (5.0, 3.0), (5.0 + tt.MIN_PLAYER_GAP_M - 0.01, 3.0))
    # The same person detected twice: nearly the same box.
    assert tt.same_spot(front, (100.5, 60.5, 130.5, 141.0), (5.0, 3.0), (9.0, 3.0))
    assert tt.box_iou(front, (200.0, 0.0, 210.0, 10.0)) == 0.0
    assert tt.box_iou((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)) == 0.0


def test_a_person_off_the_pitch_is_beyond_a_line_by_the_margin() -> None:
    margin = tt.OFF_PITCH_MARGIN_M
    assert not tt.off_pitch(None)
    assert not tt.off_pitch((0.0, 34.0 + margin))
    assert tt.off_pitch((0.0, 34.0 + margin + 0.01)), "a coach in the technical area"
    assert tt.off_pitch((10.0, -34.0 - margin - 2.0)), "a steward behind the far boards"
    assert tt.off_pitch((-52.5 - margin - 0.01, 0.0)), "behind a goal line"
    assert not tt.off_pitch((52.5 + margin, -34.0 - margin)), "on the corner of the margin"


def test_rows_without_a_track_never_get_a_jersey() -> None:
    rows = [{"track_id": -1, "jersey": "5", "jersey_conf": 0.95} for _ in range(8)]
    rows += [{"track_id": 3, "jersey": "5", "jersey_conf": 0.95} for _ in range(8)]
    out = tt.confident_jerseys(pd.DataFrame(rows))
    assert out.loc[out["track_id"] == -1, "jersey"].isna().all()
    assert out.loc[out["track_id"] == -1, "jersey_conf"].isna().all()
    assert (out.loc[out["track_id"] == 3, "jersey"] == "5").all()
