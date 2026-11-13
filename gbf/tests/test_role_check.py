"""Tests for gbf.role_check: the plugin's role labels checked against the one referee the laws
allow on the field of play.

Track tables are synthetic: per frame ten players of each team spread over the pitch, people
labelled referee inside it and two assistants labelled referee on the touchlines, with the
gbf.sngs_import columns the check reads (pitch positions in metres, NaN unless pitch-usable).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gbf import role_check as rc


def _row(frame: int, track: int, role: str, team: str | None, x: float, y: float,
         usable: bool = True) -> dict[str, Any]:  # fmt: skip
    return {
        "frame": frame,
        "track_id": track,
        "role": role,
        "team": team,
        "pitch_usable": usable,
        "pitch_x": x if usable else math.nan,
        "pitch_y": y if usable else math.nan,
    }


def scene(
    frames: int = 50,
    inside: int = 1,
    crowded: int | None = None,
    usable: bool = True,
) -> pd.DataFrame:
    """Ten players a side (teams left and right); ``inside`` people labelled referee in
    midfield on the first ``crowded`` frames (all frames when None) and one on the others;
    two assistants labelled referee, one just outside the far touchline and one 0.8 m inside
    the near one."""
    crowded = frames if crowded is None else crowded
    rows = []
    for frame in range(frames):
        for i in range(10):
            rows.append(_row(frame, 1 + i, "player", "left", -40.0 + 4 * i, -20.0 + 2 * i, usable))
            rows.append(_row(frame, 11 + i, "player", "right", -38.0 + 4 * i, 20.0 - 2 * i, usable))
        for k in range(inside if frame < crowded else 1):
            rows.append(_row(frame, 30 + k, "referee", None, -10.0 + 5 * k, 0.0, usable))
        rows.append(_row(frame, 40, "referee", None, -20.0, -34.5, usable))
        rows.append(_row(frame, 41, "referee", None, 20.0, 33.2, usable))
    return pd.DataFrame(rows)


def test_one_referee_inside_and_assistants_on_the_touchlines_are_plausible() -> None:
    tracks = scene()
    out, report = rc.check_roles(tracks)
    assert report["verdict"] == "plausible"
    assert (report["frames"], report["crowded_frames"], report["crowded_share"]) == (50, 0, 0.0)
    assert report["max_referees_inside"] == 1, "the assistants on the touchlines do not count"
    assert report["rows_cleared"] == 0
    pd.testing.assert_frame_equal(out, tracks)
    assert report["params"] == rc.DEFAULT_PARAMS.as_dict()


def test_a_team_labelled_referee_clears_every_team_label_and_keeps_the_roles() -> None:
    tracks = scene(inside=6)
    before = tracks.copy()
    out, report = rc.check_roles(tracks)
    assert report["verdict"] == "implausible"
    assert (report["crowded_frames"], report["crowded_share"]) == (50, 1.0)
    assert report["max_referees_inside"] == 6
    assert report["rows_cleared"] == 50 * 20, "every row with a team label"
    assert out["team"].isna().all()
    assert out["role"].tolist() == tracks["role"].tolist(), "roles are left as they are"
    pd.testing.assert_frame_equal(tracks, before)  # the input is not changed


@pytest.mark.parametrize(("crowded", "verdict"), [(25, "plausible"), (26, "implausible")])
def test_the_crowded_share_must_exceed_the_limit(crowded: int, verdict: str) -> None:
    out, report = rc.check_roles(scene(inside=2, crowded=crowded))
    assert report["crowded_frames"] == crowded
    assert report["verdict"] == verdict
    assert out["team"].isna().all() == (verdict == "implausible")


@pytest.mark.parametrize(
    ("second", "counted"),
    [
        (_row(0, 31, "referee", None, 0.0, 10.0), True),
        (_row(0, 31, "referee", None, 51.0, 0.0), False),  # 1.5 m inside the goal line
        (_row(0, 31, "referee", None, 0.0, -32.5), False),  # 1.5 m inside the far touchline
        (_row(0, 31, "referee", None, 0.0, 10.0, usable=False), False),
        (_row(0, 31, "player", "left", 0.0, 10.0), False),
    ],
)
def test_only_referees_with_a_usable_position_inside_the_margin_count(
    second: dict[str, Any], counted: bool
) -> None:
    tracks = pd.concat([scene(frames=1), pd.DataFrame([second])], ignore_index=True)
    _, report = rc.check_roles(tracks, rc.RoleCheckParams(min_frames=1))
    assert report["max_referees_inside"] == (2 if counted else 1)
    assert report["verdict"] == ("implausible" if counted else "plausible")


def test_too_few_frames_with_a_usable_position_give_no_verdict() -> None:
    tracks = scene(frames=24, inside=6)
    out, report = rc.check_roles(tracks)
    assert (report["verdict"], report["frames"]) == ("not_judged", 24)
    pd.testing.assert_frame_equal(out, tracks)
    # Frames without any pitch-usable row are not counted at all.
    _, report = rc.check_roles(scene(inside=6, usable=False))
    assert (report["verdict"], report["frames"], report["crowded_share"]) == ("not_judged", 0, None)


@pytest.mark.parametrize("dtype", [object, "string"])
def test_cleared_labels_take_the_tables_own_missing_value(dtype: Any) -> None:
    tracks = scene(inside=6)
    tracks["team"] = tracks["team"].astype(dtype)
    out, _ = rc.check_roles(tracks)
    assert out["team"].dtype == tracks["team"].dtype
    missing = tracks["team"][tracks["team"].isna()].iloc[0]
    assert all(value is missing for value in out["team"])


def test_params_are_checked_and_round_trip() -> None:
    assert rc.RoleCheckParams(**rc.DEFAULT_PARAMS.as_dict()) == rc.DEFAULT_PARAMS
    for bad in (
        {"inner_margin_m": -1.0},
        {"inner_margin_m": math.inf},
        {"max_crowded_share": 1.5},
        {"min_frames": 0},
        {"min_frames": True},
    ):
        with pytest.raises(rc.RoleCheckError):
            rc.RoleCheckParams(**bad)


def test_missing_columns_are_reported() -> None:
    with pytest.raises(rc.RoleCheckError, match="pitch_usable"):
        rc.check_roles(scene(frames=1).drop(columns=["pitch_usable"]))
    assert np.isnan(scene(frames=1, usable=False)["pitch_x"]).all()
