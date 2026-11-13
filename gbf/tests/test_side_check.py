"""Tests for gbf.side_check: whether the plugin's left/right team names are borne out by where
the two teams stand.

Track tables are synthetic: per frame ``k`` players a side, each track at its own x around its
team's mean, with the gbf.sngs_import columns the check reads (pitch positions in metres, NaN
unless pitch-usable).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from gbf import side_check as sc


def _row(frame: int, track: int, role: str, team: str | None, x: float,
         usable: bool = True) -> dict[str, Any]:  # fmt: skip
    return {
        "frame": frame,
        "track_id": track,
        "role": role,
        "team": team,
        "pitch_usable": usable,
        "pitch_x": x if usable else math.nan,
        "pitch_y": 0.0 if usable else math.nan,
    }


def scene(left_x: float, right_x: float, k: int = 10, frames: int = 50) -> pd.DataFrame:
    """``k`` players a side, track i of a team 4 m apart from track i + 1, centred on the
    team's x, each moving by up to 1 m over the frames."""
    rows = []
    for frame in range(frames):
        wiggle = math.sin(frame / 7.0)
        for i in range(k):
            offset = 4.0 * (i - (k - 1) / 2) + wiggle
            rows.append(_row(frame, 1 + i, "player", "left", left_x + offset))
            rows.append(_row(frame, 101 + i, "player", "right", right_x + offset))
    return pd.DataFrame(rows)


def test_teams_standing_where_their_names_say_are_consistent() -> None:
    report = sc.check_sides(scene(-20.0, 20.0))
    assert report["verdict"] == "consistent"
    assert report["attack_direction_known"] is True
    assert report["difference_m"] == pytest.approx(40.0)
    low, high = report["interval_m"]
    assert 0 < low < 40.0 < high
    assert report["tracks"] == {"left": 10, "right": 10}
    assert report["rows"] == {"left": 500, "right": 500}


def test_teams_standing_together_leave_the_sides_uncertain() -> None:
    # the plugin names the teams by the sign of this difference however small it is
    report = sc.check_sides(scene(3.0, 3.4))
    assert report["verdict"] == "uncertain"
    assert report["attack_direction_known"] is False
    low, high = report["interval_m"]
    assert low < 0 < high


def test_a_left_team_standing_further_right_contradicts_the_names() -> None:
    # gbf's own calibration can put the teams the other way round from the plugin's
    report = sc.check_sides(scene(20.0, -20.0))
    assert report["verdict"] == "contradicted"
    assert report["attack_direction_known"] is False
    assert report["interval_m"][1] < 0


def test_too_few_tracks_on_a_team_give_no_verdict() -> None:
    tracks = scene(-20.0, 20.0)
    tracks = tracks[(tracks["team"] == "left") | (tracks["track_id"] == 101)]
    report = sc.check_sides(tracks)
    assert report["verdict"] == "not_judged"
    assert report["attack_direction_known"] is False
    assert report["tracks"] == {"left": 10, "right": 1}


def test_without_team_labels_there_are_no_sides_to_doubt() -> None:
    # gbf.role_check clears every team label when the roles are implausible
    tracks = scene(-20.0, 20.0)
    tracks["team"] = None
    report = sc.check_sides(tracks)
    assert report["verdict"] == "no_teams"
    assert report["attack_direction_known"] is True
    assert report["difference_m"] is None
    assert report["interval_m"] is None


def test_only_outfield_players_with_a_usable_position_count() -> None:
    plain = sc.check_sides(scene(-20.0, 20.0))
    tracks = scene(-20.0, 20.0)
    extra = []
    for frame in range(50):
        extra.append(_row(frame, 200, "goalkeeper", "right", -50.0))  # in the other goal
        extra.append(_row(frame, 201, "referee", "right", -45.0))
        extra.append(_row(frame, 202, "player", "right", -45.0, usable=False))
        extra.append(_row(frame, 203, "player", None, -45.0))
    report = sc.check_sides(pd.concat([tracks, pd.DataFrame(extra)], ignore_index=True))
    assert report == plain


def test_the_check_is_deterministic() -> None:
    tracks = scene(3.0, 4.0)
    assert sc.check_sides(tracks) == sc.check_sides(tracks)


def test_bad_inputs_and_settings_list_every_problem() -> None:
    with pytest.raises(sc.SideCheckError) as caught:
        sc.check_sides(pd.DataFrame({"frame": [0], "team": ["left"]}))
    assert "pitch_x" in str(caught.value)
    assert "role" in str(caught.value)
    with pytest.raises(sc.SideCheckError) as caught:
        sc.SideCheckParams(n_boot=0, level=1.5, min_tracks=1)
    assert len(caught.value.problems) == 3
