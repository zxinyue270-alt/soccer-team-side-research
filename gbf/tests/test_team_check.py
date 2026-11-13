"""Tests for gbf.team_check: the shirt-colour check of the plugin's team labels.

Scenes are synthetic: noisy grass frames with players drawn as a head, a shirt, shorts and
legs inside their boxes, in two team colours (plus other colours for a goalkeeper and a
referee). The track tables are built to match the drawings exactly.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from gbf import team_check as tc

SIZE = (400, 160)  # width, height
GRASS = (60, 140, 50)
SKIN = (220, 170, 140)
SHORTS = (40, 40, 40)
RED = (200, 30, 30)
BLUE = (30, 60, 200)
ORANGE = (240, 130, 20)
PINK = (230, 80, 160)
BOX_W, BOX_H, BOX_TOP = 24, 72, 40


@dataclass
class Player:
    track_id: int
    team: str | None
    shirt: tuple[int, int, int]
    x: int
    frames: range = range(20)
    role: str = "player"
    #: frame -> left edge, where it differs from ``x``
    moves: Mapping[int, int] = field(default_factory=dict)
    #: frame -> shirt colour, where it differs from ``shirt``
    shirts: Mapping[int, tuple[int, int, int]] = field(default_factory=dict)

    def box(self, frame: int) -> tuple[float, float, float, float]:
        x = self.moves.get(frame, self.x)
        return (float(x), float(BOX_TOP), float(x + BOX_W), float(BOX_TOP + BOX_H))


def _tracks(players: list[Player]) -> pd.DataFrame:
    rows = []
    for p in players:
        for frame in p.frames:
            x1, y1, x2, y2 = p.box(frame)
            rows.append(
                {
                    "frame": frame,
                    "track_id": p.track_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "role": p.role,
                    "team": p.team,
                }
            )
    return pd.DataFrame(rows).sort_values(["frame", "track_id"]).reset_index(drop=True)


def _paint(image: np.ndarray, box: tuple[float, ...], shirt: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1, y2 - y1

    def part(fx0: float, fy0: float, fx1: float, fy1: float, colour: tuple[int, ...]) -> None:
        image[y1 + int(fy0 * h) : y1 + int(fy1 * h), x1 + int(fx0 * w) : x1 + int(fx1 * w)] = colour

    part(0.35, 0.0, 0.65, 0.15, SKIN)  # head
    part(0.15, 0.15, 0.85, 0.55, shirt)  # shirt with sleeves
    part(0.2, 0.55, 0.8, 0.7, SHORTS)
    part(0.25, 0.7, 0.45, 1.0, SKIN)  # legs
    part(0.55, 0.7, 0.75, 1.0, SKIN)


def _frame(players: list[Player], index: int) -> np.ndarray:
    rng = np.random.default_rng(index)
    image = np.empty((SIZE[1], SIZE[0], 3), dtype=np.int16)
    image[:] = GRASS
    for p in players:
        if index in p.frames:
            _paint(image, p.box(index), p.shirts.get(index, p.shirt))
    noisy = image + rng.integers(-6, 7, size=image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _frames(players: list[Player], plan: pd.DataFrame) -> dict[int, np.ndarray]:
    return {int(f): _frame(players, int(f)) for f in plan["frame"].unique()}


def _run(players: list[Player], params: tc.TeamCheckParams | None = None):
    tracks = _tracks(players)
    plan = tc.plan_samples(tracks, params, frame_size=SIZE)
    colours, samples = tc.measure_shirts(plan, _frames(players, plan), params)
    checked, report = tc.check_teams(tracks, colours, params, samples=samples)
    return tracks, plan, checked, report


def _teams_by_track(table: pd.DataFrame) -> dict[int, set]:
    labels = table["team"].astype(object).where(table["team"].notna(), None)
    return {int(t): set(labels[table["track_id"] == t]) for t in table["track_id"].unique()}


def _verdicts(report: dict) -> dict[tuple[int, str], str]:
    return {(e["track_id"], e["team"]): e["verdict"] for e in report["tracks"]}


def _base_players() -> list[Player]:
    """Three red players labelled left and three blue players labelled right."""
    reds = [Player(t, "left", RED, 10 + 38 * k) for k, t in enumerate((1, 2, 3))]
    blues = [Player(t, "right", BLUE, 124 + 38 * k) for k, t in enumerate((4, 5, 6))]
    return reds + blues


# --------------------------------------------------------------------------- colour


def test_srgb_to_lab_matches_reference_values() -> None:
    lab = tc.srgb_to_lab(
        np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    )
    expected = [
        (100.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (53.24, 80.09, 67.20),
        (87.73, -86.18, 83.18),
        (32.30, 79.19, -107.86),
    ]
    np.testing.assert_allclose(lab, expected, atol=0.02)
    assert tc.srgb_to_lab(np.zeros((2, 3, 3), dtype=np.uint8)).shape == (2, 3, 3)


def test_shirt_colour_reads_the_torso_and_an_all_grass_crop_gives_none() -> None:
    shirt = Player(1, "left", RED, 100)
    painted = tc.shirt_colour(_frame([shirt], 0), shirt.box(0))
    assert painted is not None
    np.testing.assert_allclose(painted, tc.srgb_to_lab(np.array(RED)), atol=4.0)
    assert tc.shirt_colour(_frame([], 0), shirt.box(0)) is None


def test_boxes_the_torso_crop_does_not_fit_give_no_colour() -> None:
    image = np.full((SIZE[1], SIZE[0], 3), RED, dtype=np.uint8)
    assert tc.shirt_colour(image, (100, 40, 124, 112)) is not None
    assert tc.shirt_colour(image, (100, 40, 124, 80)) is None  # shorter than 50 px
    assert tc.shirt_colour(image, (100, 40, 180, 112)) is None  # wider than tall / 1.5
    assert tc.shirt_colour(image, (0, 40, 24, 112)) is None  # cut by the frame edge
    assert tc.shirt_colour(image, (100, 88, 124, SIZE[1])) is None


# --------------------------------------------------------------------------- check


@pytest.mark.parametrize(
    ("label", "shirt", "other"), [("left", BLUE, "right"), ("right", RED, "left")]
)
def test_a_track_wearing_the_other_teams_colour_loses_its_label(
    label: str, shirt: tuple[int, int, int], other: str
) -> None:
    players = [*_base_players(), Player(7, label, shirt, 352)]
    tracks, _, checked, report = _run(players)
    verdicts = _verdicts(report)
    assert verdicts[(7, label)] == "inconsistent"
    assert all(verdicts[(t, team)] == "consistent" for t, team in verdicts if t != 7)
    entry = next(e for e in report["tracks"] if e["track_id"] == 7)
    assert entry["nearer_team"] == other
    # The entry says that its label was cleared, and why; the others keep theirs.
    assert entry["cleared"] == "inconsistent"
    assert all(e["cleared"] is None for e in report["tracks"] if e["track_id"] != 7)
    assert entry["distance_own"] - entry["distance_other"] > tc.DEFAULT_PARAMS.margin
    by_track = _teams_by_track(checked)
    assert by_track[7] == {None}  # set to missing, never flipped to the other team
    assert checked.loc[checked["track_id"] == 7, "team"].isna().all()
    for t in range(1, 7):
        assert by_track[t] == _teams_by_track(tracks)[t]
    unchanged = [c for c in tracks.columns if c != "team"]
    pd.testing.assert_frame_equal(checked[unchanged], tracks[unchanged])
    assert report["counts"]["inconsistent"] == 1
    assert report["counts"]["rows_cleared"] == 20
    assert report["prototype_distance"] > 50
    assert json.loads(json.dumps(report)) == report  # JSON-ready as documented
    # Within the margin a track is not judged inconsistent.
    margin = report["prototype_distance"] + 1.0
    _, _, kept, lenient = _run(players, tc.TeamCheckParams(margin=margin))
    assert _verdicts(lenient)[(7, label)] == "consistent"
    pd.testing.assert_frame_equal(kept, tracks)


def test_too_few_samples_give_no_verdict_and_keep_the_label() -> None:
    players = [*_base_players(), Player(7, "left", BLUE, 352, frames=range(5, 7))]
    _, _, checked, report = _run(players)
    entry = next(e for e in report["tracks"] if e["track_id"] == 7)
    assert entry["verdict"] == "unchecked"
    assert entry["reason"] == "too_few_samples"
    assert entry["samples"] == 2
    assert entry["distance_own"] is None
    assert entry["cleared"] is None
    assert _teams_by_track(checked)[7] == {"left"}
    assert report["counts"]["rows_cleared"] == 0


def test_goalkeepers_and_referees_are_skipped() -> None:
    players = [
        *_base_players(),
        Player(8, "right", ORANGE, 314, role="goalkeeper"),
        Player(9, None, PINK, 352, role="referee"),
        Player(10, "right", RED, 276, role="goalkeeper"),  # the other team's colour
    ]
    tracks, plan, checked, report = _run(players)
    assert set(plan["track_id"]) == {1, 2, 3, 4, 5, 6}
    assert {e["track_id"] for e in report["tracks"]} == {1, 2, 3, 4, 5, 6}
    pd.testing.assert_frame_equal(checked, tracks)
    assert report["counts"]["skipped_tracks"] == 3
    assert report["counts"]["consistent"] == 6


def test_empty_input() -> None:
    tracks = pd.DataFrame({c: [] for c in tc.TRACK_COLUMNS}).astype(
        {"frame": np.int64, "track_id": np.int64}
    )
    plan = tc.plan_samples(tracks)
    assert list(plan.columns) == list(tc.PLAN_COLUMNS)
    assert plan.empty
    colours, samples = tc.measure_shirts(plan, {})
    assert list(colours.columns) == list(tc.COLOUR_COLUMNS)
    assert colours.empty
    assert list(samples.columns) == list(tc.SAMPLE_COLUMNS)
    assert samples.empty
    checked, report = tc.check_teams(tracks, colours, samples=samples)
    pd.testing.assert_frame_equal(checked, tracks)
    assert report["tracks"] == []
    assert report["prototypes"] == {"left": None, "right": None}
    assert report["counts"]["player_tracks"] == 0
    assert report["counts"]["rows_cleared"] == 0


def test_without_both_prototypes_nothing_is_judged() -> None:
    # Left has two tracks, one of them in the other colour: fewer than min_team_tracks (3).
    players = [_base_players()[0], *_base_players()[3:], Player(7, "left", BLUE, 352)]
    _, _, checked, report = _run(players)
    assert report["prototypes"]["left"] is None
    assert report["prototypes"]["right"]["tracks"] == 3
    assert report["prototype_distance"] is None
    assert {e["verdict"] for e in report["tracks"]} == {"unchecked"}
    assert {e["reason"] for e in report["tracks"]} == {"no_prototypes"}
    assert _teams_by_track(checked)[7] == {"left"}
    assert report["counts"]["rows_cleared"] == 0


def test_a_minority_of_frames_on_another_player_is_counted_not_judged() -> None:
    switched = {f: BLUE for f in range(15, 20)}  # the tracker drifted to an opponent
    players = [*_base_players(), Player(7, "left", RED, 352, shirts=switched)]
    _, plan, checked, report = _run(players)
    entry = next(e for e in report["tracks"] if e["track_id"] == 7)
    assert entry["verdict"] == "consistent"
    later = int((plan.loc[plan["track_id"] == 7, "frame"] >= 15).sum())
    assert later >= 1
    assert entry["samples_against"] == later
    assert report["counts"]["consistent_with_samples_against"] == 1
    assert _teams_by_track(checked)[7] == {"left"}


@pytest.mark.parametrize(
    ("dtype", "missing"), [(object, None), (pd.StringDtype(), pd.NA)], ids=["object", "string"]
)
def test_cleared_labels_use_the_tables_missing_value(dtype: object, missing: object) -> None:
    players = [
        *_base_players(),
        Player(7, "left", BLUE, 352),
        Player(9, None, PINK, 314, role="referee"),
    ]
    tracks = _tracks(players)
    labels = [None if pd.isna(v) else v for v in tracks["team"]]
    tracks["team"] = pd.Series(labels, index=tracks.index, dtype=object).astype(dtype)
    plan = tc.plan_samples(tracks, frame_size=SIZE)
    colours, _ = tc.measure_shirts(plan, _frames(players, plan))
    checked, _ = tc.check_teams(tracks, colours)
    assert checked["team"].dtype == tracks["team"].dtype
    cleared = checked.loc[checked["track_id"] == 7, "team"]
    assert all(value is missing for value in cleared)
    assert tracks.loc[tracks["track_id"] == 7, "team"].eq("left").all()  # input untouched


# --------------------------------------------------------------------------- plan


def _table(lives: list[range]) -> pd.DataFrame:
    """Track ``t`` lives in ``lives[t]``; boxes of different tracks never overlap."""
    rows = []
    for t, life in enumerate(lives):
        x, y = 10.0 + (t % 12) * 32, 20.0 + (t // 12 % 50) * 90
        team = ("left", "right")[t % 2]
        rows.extend((frame, t, x, y, x + 24.0, y + 72.0, "player", team) for frame in life)
    return pd.DataFrame(rows, columns=list(tc.TRACK_COLUMNS))


def test_planned_frames_stay_within_the_limit() -> None:
    # A clip of about a minute with a new four-frame track every four frames: no frame
    # serves two tracks, so a plan without the limit would read every frame.
    tracks = _table([range(s, s + 4) for s in range(0, 1600, 4)])
    unlimited = tc.plan_samples(tracks, tc.TeamCheckParams(max_frames=5000))
    assert unlimited["frame"].nunique() == 1600
    plan = tc.plan_samples(tracks)
    assert plan["frame"].nunique() == tc.DEFAULT_PARAMS.max_frames == 200
    assert tc.plan_samples(tracks, tc.TeamCheckParams(max_frames=7))["frame"].nunique() == 7


def test_frames_alive_for_many_tracks_are_read_first() -> None:
    lives = [range(t * 97 % 1600, min(1600, t * 97 % 1600 + 150 + t * 53 % 600)) for t in range(60)]
    tracks = _table(lives)
    plan = tc.plan_samples(tracks)
    assert len(plan) == 60 * 8  # every part of every life, ...
    assert plan["frame"].nunique() <= 80  # ... from few frames
    small = tc.plan_samples(tracks, tc.TeamCheckParams(max_frames=12))
    assert small["frame"].nunique() == 12
    served = small.groupby("frame").size()
    assert served.min() > tracks.groupby("frame").size().mean()


def test_samples_spread_over_each_life_from_shared_frames() -> None:
    params = tc.TeamCheckParams(samples_per_track=4)
    tracks = _tracks(_base_players())
    plan = tc.plan_samples(tracks, params, frame_size=SIZE)
    assert plan["frame"].nunique() == 4  # every chosen frame serves all six tracks
    late = [*_base_players(), Player(7, "left", RED, 352, frames=range(8, 20))]
    tracks = _tracks(late)
    plan = tc.plan_samples(tracks, params, frame_size=SIZE)
    assert plan["frame"].nunique() == 5  # the late track's parts start elsewhere
    assert plan.groupby("track_id").size().eq(4).all()
    for track, rows in plan.groupby("track_id"):
        first = tracks.loc[tracks["track_id"] == track, "frame"].min()
        span = tracks.loc[tracks["track_id"] == track, "frame"].max() - first + 1
        assert sorted(rows["part"]) == [0, 1, 2, 3]
        assert ((rows["frame"] - first) * 4 // span).tolist() == rows["part"].tolist()


def test_samples_skip_overlapped_boxes_and_the_frame_edge() -> None:
    near = {f: 30 for f in range(0, 10)}  # over track 1's torso in frames 0-9
    players = [
        Player(1, "left", RED, 20),
        Player(2, "right", BLUE, 200, moves=near),
        Player(3, "left", RED, 0, frames=range(0, 10), moves={f: 120 for f in range(5, 10)}),
    ]
    plan = tc.plan_samples(_tracks(players), frame_size=SIZE)
    assert (plan.loc[plan["track_id"].isin([1, 2]), "frame"] >= 10).all()
    assert (plan["overlap"] <= tc.DEFAULT_PARAMS.max_overlap).all()
    assert (plan.loc[plan["track_id"] == 3, "frame"] >= 5).all()  # x1 = 0 touches the edge
    assert len(plan[plan["track_id"] == 1]) == 4  # one per remaining part of its life


# --------------------------------------------------------------------------- inputs


def test_measure_reads_pairs_lazily_and_refuses_missing_frames() -> None:
    players = _base_players()
    tracks = _tracks(players)
    plan = tc.plan_samples(tracks, frame_size=SIZE)
    last = int(plan["frame"].max())
    seen: list[int] = []

    def stream() -> Iterator[tuple[int, np.ndarray]]:
        for index in range(20):
            seen.append(index)
            yield index, _frame(players, index)

    colours, _ = tc.measure_shirts(plan, stream())
    assert seen[-1] == last  # stopped once every planned frame was seen
    assert colours["samples"].sum() == len(plan)
    with pytest.raises(tc.TeamCheckError, match="planned frame"):
        tc.measure_shirts(plan, {last: _frame(players, last)})


def test_bad_inputs_are_refused_with_every_problem() -> None:
    with pytest.raises(tc.TeamCheckError) as error:
        tc.TeamCheckParams(samples_per_track=0, max_overlap=1.5, torso_y=(0.5, 0.2), margin=-1)
    assert len(error.value.problems) == 4
    tracks = _tracks(_base_players())
    with pytest.raises(tc.TeamCheckError, match="misses column"):
        tc.plan_samples(tracks.drop(columns=["role"]))
    odd = tracks.assign(team=tracks["team"].where(tracks["track_id"] != 1, "home"))
    with pytest.raises(tc.TeamCheckError, match="team must be"):
        tc.plan_samples(odd)
    plan = tc.plan_samples(tracks, frame_size=SIZE)
    colours, _ = tc.measure_shirts(plan, _frames(_base_players(), plan))
    with pytest.raises(tc.TeamCheckError, match="without player rows"):
        tc.check_teams(tracks[tracks["track_id"] != 1], colours)


# --------------------------------------------------------------------------- review additions


def test_a_floodlit_white_kit_is_measured_and_still_tells_the_teams_apart() -> None:
    white = (238, 240, 242)
    players = [Player(t, "left", white, 10 + 38 * k) for k, t in enumerate((1, 2, 3))]
    players += [Player(t, "right", BLUE, 124 + 38 * k) for k, t in enumerate((4, 5, 6))]
    players.append(Player(7, "right", white, 352))  # a white shirt labelled right
    _, _, checked, report = _run(players)
    assert report["prototypes"]["left"] is not None, "a white kit still gets a prototype"
    assert report["prototypes"]["left"]["lab"][0] > 85
    assert _verdicts(report)[(7, "right")] == "inconsistent"
    assert _teams_by_track(checked)[7] == {None}
    # White print on a coloured shirt is still left out of that shirt's colour.
    image = _frame([Player(1, "left", RED, 100)], 0)
    image[60:66, 104:120] = white
    np.testing.assert_allclose(
        tc.shirt_colour(image, (100, 40, 124, 112)), tc.srgb_to_lab(np.array(RED)), atol=4.0
    )


def test_parameters_survive_their_json_form() -> None:
    params = tc.TeamCheckParams(torso_x=(0.25, 0.75))
    again = tc.TeamCheckParams(**json.loads(json.dumps(params.as_dict())))
    assert again == params


def test_only_the_rows_of_the_inconsistent_label_are_cleared() -> None:
    # Track 7 is labelled left on frames 0-9 and right on frames 10-19 while wearing blue
    # (the right team's colour) all along: only its left rows contradict the shirt.
    players = [*_base_players(), Player(7, "left", BLUE, 352)]
    tracks = _tracks(players)
    late = (tracks["track_id"] == 7) & (tracks["frame"] >= 10)
    tracks.loc[late, "team"] = "right"
    plan = tc.plan_samples(tracks, frame_size=SIZE)
    colours, samples = tc.measure_shirts(plan, _frames(players, plan))
    checked, report = tc.check_teams(tracks, colours, samples=samples)
    verdicts = _verdicts(report)
    assert verdicts[(7, "left")] == "inconsistent"
    assert verdicts[(7, "right")] == "consistent"
    seven = checked[checked["track_id"] == 7]
    assert seven.loc[seven["frame"] < 10, "team"].isna().all()
    assert (seven.loc[seven["frame"] >= 10, "team"] == "right").all()


def test_a_track_colour_is_the_median_of_its_samples() -> None:
    # One outlying sample (an orange frame) does not move a blue track's colour much.
    shirts = {f: ORANGE for f in range(0, 3)}
    players = [*_base_players(), Player(7, "right", BLUE, 352, shirts=shirts)]
    tracks = _tracks(players)
    plan = tc.plan_samples(tracks, frame_size=SIZE)
    colours, _ = tc.measure_shirts(plan, _frames(players, plan))
    row = colours[colours["track_id"] == 7].iloc[0]
    blue = tc.srgb_to_lab(np.array(BLUE))
    assert np.hypot(*(np.array([row["L"], row["a"], row["b"]]) - blue)[1:]) < 6.0
