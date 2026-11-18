"""Tests for gbf.stitch (the stitch audit).

Every scene is synthetic: players stand or walk on the grass of a synthetic main broadcast
camera (gbf.geometry.synthetic_broadcast_camera at 1920x1080, which sees about 40 m across
the centre circle), their boxes projected from their ground points.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import pytest

from gbf import stitch
from gbf.geometry import Camera, look_at, project, synthetic_broadcast_camera

FPS = 25.0
W, H = 1920, 1080
CAMERA = synthetic_broadcast_camera(W, H)
N_FRAMES = 240
SHOTS = pd.DataFrame(
    {"shot_id": [0], "start_frame": [0], "end_frame": [N_FRAMES - 1], "shot_type": ["wide"]}
)


def _calib(cameras: dict[int, Camera] | None = None) -> pd.DataFrame:
    rows = []
    for frame in range(N_FRAMES):
        camera = (cameras or {}).get(frame, CAMERA)
        rows.append(
            {
                "frame_index": frame,
                "width": W,
                "height": H,
                "K": camera.K.ravel().copy(),
                "R": camera.R.ravel().copy(),
                "C": camera.C.copy(),
                "calib_status": "fresh",
                "calib_age": 0,
                "rep_err": 1.0,
                "rep_err_n_keypoints": 12,
                "pitch_usable": True,
            }
        )
    return pd.DataFrame(rows)


def _box(x: float, y: float, camera: Camera = CAMERA) -> tuple[float, float, float, float]:
    uv, _, _ = project(camera, np.array([[x, y, 0.0], [x, y, -1.8]]))
    (u, v), (_, top) = uv
    height = v - top
    return (u - 0.2 * height, top, u + 0.2 * height, v)


def _person(
    track: int,
    frames: Iterable[int],
    where: Callable[[int], tuple[float, float]],
    *,
    source: int | None = None,
    role: str = "player",
    team: str | None = "left",
    jersey: str | None = None,
    cameras: dict[int, Camera] | None = None,
) -> list[dict]:
    rows = []
    for frame in frames:
        x, y = where(frame)
        x1, y1, x2, y2 = _box(x, y, (cameras or {}).get(frame, CAMERA))
        rows.append(
            {
                "frame": frame,
                "track_id": track,
                "source_track_id": track if source is None else source,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "role": role,
                "team": team,
                "jersey": jersey,
                "pitch_usable": True,
                "pitch_x": x,
                "pitch_y": y,
            }
        )
    return rows


def _tracks(*people: list[dict]) -> pd.DataFrame:
    rows = [row for person in people for row in person]
    return (
        pd.DataFrame(rows).sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)
    )


def _aimed(x: float) -> Camera:
    """The synthetic camera turned to look at (x, 0) on the halfway line."""
    return Camera(K=CAMERA.K, R=look_at(CAMERA.C, (x, 0.0, 0.0)), C=CAMERA.C, width=W, height=H)


def _teams(verified: dict[int, str], against: dict[int, str] | None = None) -> dict:
    """A team-check report: the tracks of ``verified`` consistent with their team, those of
    ``against`` with a colour nearer that team than their own label."""
    entries = [
        {"track_id": t, "team": team, "verdict": "consistent", "nearer_team": team,
         "samples_against": 0, "cleared": None}
        for t, team in verified.items()
    ]  # fmt: skip
    for t, nearer in (against or {}).items():
        own = "right" if nearer == "left" else "left"
        entries.append(
            {"track_id": t, "team": own, "verdict": "inconsistent", "nearer_team": nearer,
             "samples_against": 3, "cleared": "inconsistent"}
        )  # fmt: skip
    return {
        "prototypes": {"left": {"lab": [50, 0, 0]}, "right": {"lab": [50, 40, 40]}},
        "prototype_distance": 56.6,
        "tracks": entries,
    }


ROLES_OK = {"verdict": "plausible"}


def _walk(x0: float, y0: float, vx: float = 0.0, vy: float = 0.0, start: int = 0):
    return lambda f: (x0 + vx * (f - start) / FPS, y0 + vy * (f - start) / FPS)


def _audit(tracks: pd.DataFrame, calib: pd.DataFrame | None = None, **kwargs) -> dict:
    kwargs.setdefault("team_report", _teams({1: "left", 2: "left"}))
    kwargs.setdefault("role_report", ROLES_OK)
    return stitch.audit(tracks, _calib() if calib is None else calib, SHOTS, fps=FPS, **kwargs)


def _pair(report: dict, earlier: int, later: int) -> dict:
    for key in (
        "stitches",
        "conflicts_with_plugin",
        "ambiguous",
        "implied_by_chain",
        "abstentions",
    ):
        for entry in report[key]:
            if (entry["earlier"], entry["later"]) == (earlier, later):
                return entry
    raise AssertionError(f"pair {earlier} -> {later} not in the report")


# A walks from (-5, 0) and is lost at frame 40 in open space; B appears 3 frames later a
# metre on, the only person anywhere near. A third player stands far away on the right.
A_PATH = _walk(-5.0, 0.0, vx=1.5)
B_PATH = _walk(-2.5, 0.0, vx=1.5, start=44)
FAR = _walk(17.0, -25.0)


def _fragment_scene(*extra: list[dict], b_source: int | None = None) -> pd.DataFrame:
    return _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(2, range(44, 160), B_PATH, source=b_source),
        _person(3, range(0, N_FRAMES), FAR, team="right"),
        *extra,
    )


# --------------------------------------------------------------------------- path G


def test_a_fragment_alone_in_a_covered_reach_disc_is_proven_one_person() -> None:
    report = _audit(_fragment_scene())
    assert report["mode"] == "audit"
    assert report["applied"] is False
    pair = _pair(report, 1, 2)
    assert (pair["decision"], pair["geometry"], pair["plugin"]) == ("stitch", "pass", "fresh")
    evidence = pair["geometry_evidence"]
    assert evidence["anchor"]["frame"] == 40
    assert evidence["first_decision"] == 44
    assert evidence["decision_frames"] >= 8
    # The disc grows at 12 m/s: coverage ends long before the 2 s window does.
    assert evidence["last_decision"] < 44 + 50
    assert report["counts"]["decisions"]["stitch"] == 1


def test_the_audit_never_changes_the_track_table() -> None:
    tracks = _fragment_scene()
    before = tracks.copy()
    _audit(tracks)
    pd.testing.assert_frame_equal(tracks, before)


@pytest.mark.parametrize(
    ("track", "role", "team"),
    [(-1, "player", None), (7, "player", "right"), (8, "referee", None), (9, "player", "left")],
)
def test_any_other_box_in_the_image_of_the_disc_is_a_contradiction(
    track: int, role: str, team: str | None
) -> None:
    # Whoever it is: a box without a track, an opponent with a verified colour, the referee or
    # a teammate. The team labels are never used to explain a box away (a tracker may switch
    # ids inside the picture), and a box in front of the disc could hide A.
    other = _person(track, range(40, 120), _walk(-1.0, 3.0), role=role, team=team)
    teams = _teams({1: "left", 2: "left", 7: "right"})
    report = _audit(_fragment_scene(other), team_report=teams)
    pair = _pair(report, 1, 2)
    assert (pair["decision"], pair["geometry"]) == ("abstain", "contradiction")
    assert pair["geometry_evidence"]["contradiction_frames"] > 0
    assert pair["geometry_evidence"]["contradictions"][0]["track_ids"] == [track]


def test_a_player_who_left_at_the_frame_edge_is_not_proven() -> None:
    # A runs out on the right; the camera stays: the reach disc reaches past the edge, so a
    # never-seen teammate coming in there cannot be told from A.
    a = _person(1, range(0, 41), _walk(14.0, 0.0, vx=2.0))
    b = _person(2, range(53, 160), _walk(18.5, 0.0, vx=-2.0, start=53))
    report = _audit(_tracks(a, b))
    pair = _pair(report, 1, 2)
    assert pair["decision"] == "abstain"
    assert pair["geometry"] == "no_decision_frames"
    assert pair["geometry_evidence"]["covered_frames"] == 0
    assert "outside_view" in pair["geometry_evidence"]["uncovered"]


def test_another_piece_lost_just_before_b_is_a_rival() -> None:
    # Track 5 is lost 2 frames before B appears, 2 m from where B appears: B could be him.
    rival = _person(5, range(10, 42), _walk(-1.0, 1.5))
    report = _audit(_fragment_scene(rival))
    pair = _pair(report, 1, 2)
    assert (pair["decision"], pair["geometry"]) == ("abstain", "rivals")
    assert pair["geometry_evidence"]["rivals"] == [5]


def test_a_rival_far_out_of_reach_does_not_block() -> None:
    rival = _person(5, range(10, 42), _walk(15.0, -20.0))
    report = _audit(_fragment_scene(rival))
    assert _pair(report, 1, 2)["decision"] == "stitch"


def test_a_proof_against_the_plugins_own_id_is_listed_apart() -> None:
    # B continues the plugin id of piece 7, seen long before on the far side.
    old = _person(7, range(0, 6), _walk(10.0, -25.0))
    report = _audit(_fragment_scene(old, b_source=7))
    pair = _pair(report, 1, 2)
    assert (pair["decision"], pair["geometry"], pair["plugin"]) == (
        "conflicts_with_plugin",
        "pass",
        "conflict",
    )
    assert report["counts"]["decisions"]["stitch"] == 0


def test_a_re_entry_piece_of_the_same_plugin_id_agrees_with_the_plugin() -> None:
    report = _audit(_fragment_scene(b_source=1))
    pair = _pair(report, 1, 2)
    assert (pair["decision"], pair["plugin"]) == ("stitch", "agrees")


def test_a_camera_jump_between_the_pieces_blocks_the_proof() -> None:
    # A cut the shot detector missed (or a flipped calibration): from frame 42 on the camera
    # looks 10 m to the side.
    turned = Camera(
        K=CAMERA.K, R=look_at(CAMERA.C, (10.0, 0.0, 0.0)), C=CAMERA.C, width=W, height=H
    )
    calib = _calib({f: turned for f in range(42, N_FRAMES)})
    report = _audit(_fragment_scene(), calib=calib)
    pair = _pair(report, 1, 2)
    assert (pair["decision"], pair["geometry"]) == ("abstain", "camera_discontinuity")
    assert report["diagnostics"]["camera_breaks"] == 1


def test_frames_without_a_calibration_break_the_camera_continuity() -> None:
    calib = _calib()
    for column, value in (("K", None), ("R", None), ("C", None)):
        calib[column] = calib[column].astype(object)
        calib.at[41, column] = value
    calib.loc[41, ["calib_status", "pitch_usable"]] = ["none", False]
    pair = _pair(_audit(_fragment_scene(), calib=calib), 1, 2)
    assert pair["geometry"] == "camera_discontinuity"


def test_a_box_too_wide_for_one_person_is_a_contradiction() -> None:
    tracks = _fragment_scene()
    rows = (tracks["track_id"] == 2) & (tracks["frame"] == 50)
    tracks.loc[rows, "x2"] = tracks.loc[rows, "x2"] + 60.0  # two people side by side
    pair = _pair(_audit(tracks), 1, 2)
    assert pair["geometry"] == "contradiction"
    assert pair["geometry_evidence"]["contradictions"][0]["why"] == "box_not_one_person"


def test_b_appearing_beyond_the_reach_of_a_is_a_contradiction() -> None:
    # Track 2 is 12 m from where A was lost one frame earlier: nobody runs that fast.
    tracks = _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(2, range(42, 160), _walk(9.0, 0.0, start=42)),
    )
    pair = _pair(_audit(tracks), 1, 2)
    assert pair["geometry"] == "contradiction"


def test_b_vanishing_from_a_covered_disc_is_a_missing_person() -> None:
    # B is seen for 4 frames, then nobody is in the disc while it is still wholly in view:
    # the person who must stand there has no row, so the detector failed (S3), for A or B.
    tracks = _tracks(_person(1, range(0, 41), A_PATH), _person(2, range(44, 48), B_PATH))
    pair = _pair(_audit(tracks), 1, 2)
    assert pair["geometry"] == "missing_frames"
    assert pair["geometry_evidence"]["missing_frames"] > 0


def test_sparse_rows_of_b_leave_missing_frames() -> None:
    # The reviewers' case: B is seen on 11 scattered frames only.
    frames = [*range(44, 52), 60, 70, 80]
    tracks = _tracks(_person(1, range(0, 41), A_PATH), _person(2, frames, B_PATH))
    assert _pair(_audit(tracks), 1, 2)["geometry"] == "missing_frames"


# A runs to (10, 0), near the right of the picture, where the disc soon outgrows the view.
NEAR_EDGE = _walk(4.0, 0.0, vx=3.75)


def _near_edge_pair(b_start: int, **params: object) -> dict:
    tracks = _tracks(
        _person(1, range(0, 41), NEAR_EDGE),
        _person(2, range(b_start, 160), _walk(10.3, 0.0, start=b_start)),
    )
    return _pair(_audit(tracks, params=stitch.StitchParams(**params)), 1, 2)


def test_too_few_decision_frames_do_not_prove_anything() -> None:
    assert _near_edge_pair(44)["geometry"] == "pass"  # 9 decision frames
    pair = _near_edge_pair(46)
    assert pair["geometry"] == "too_few_decision_frames"
    assert pair["geometry_evidence"]["decision_frames"] == 7


def test_decision_frames_must_span_the_minimum_time() -> None:
    pair = _near_edge_pair(48, min_decision_frames=2)
    assert pair["geometry"] == "decision_span_short"
    assert (
        pair["geometry_evidence"]["last_decision"] - pair["geometry_evidence"]["first_decision"] < 8
    )


def test_a_late_first_decision_leaves_the_start_of_b_unchecked() -> None:
    # The camera looks to the left of the action while B appears and turns to it slowly (0.6
    # degree a frame): B is clean from its start, but the disc comes into view 18 frames later,
    # so nothing checked who B was when it appeared.
    cams = {f: _aimed(-22.0 + 19.0 * min(1.0, max(0.0, (f - 44) / 24))) for f in range(N_FRAMES)}
    tracks = _tracks(
        _person(1, range(0, 41), A_PATH, cameras=cams),
        _person(2, range(44, 160), B_PATH, cameras=cams),
    )
    report = _audit(tracks, calib=_calib(cams))
    pair = _pair(report, 1, 2)
    assert report["diagnostics"]["camera_breaks"] == 0
    assert pair["geometry"] == "late_first_decision"
    assert pair["geometry_evidence"]["decision_frames"] >= 8


def test_the_window_ends_at_a_camera_break_inside_it() -> None:
    # The camera jumps 12 frames after B appears (a missed cut): the frames after it are not
    # used, and the 12 before it still prove the pair.
    turned = _aimed(10.0)
    report = _audit(_fragment_scene(), calib=_calib({f: turned for f in range(56, N_FRAMES)}))
    pair = _pair(report, 1, 2)
    assert pair["geometry"] == "pass"
    evidence = pair["geometry_evidence"]
    assert evidence["last_decision"] == 55
    assert evidence["uncovered"]["camera_discontinuity"] == 1


def _held(change_deg: float, status: str) -> pd.DataFrame:
    """A calibration held over frames 101-104 (or fresh and unchanged), then turned by
    ``change_deg`` at frame 105."""
    turned = _aimed(72.4 * math.tan(math.radians(change_deg)))
    calib = _calib({f: turned for f in range(105, N_FRAMES)})
    calib.loc[101:104, "calib_status"] = status
    return calib


def test_a_held_calibration_lets_the_camera_move_meanwhile() -> None:
    # 6 degrees after four held frames is 1.2 degrees a frame: no break. After four fresh
    # unchanged frames the same change is one frame's: a break.
    held = _audit(_fragment_scene(), calib=_held(6.0, "held"))
    fresh = _audit(_fragment_scene(), calib=_held(6.0, "fresh"))
    assert held["diagnostics"]["camera_breaks"] == 0
    assert fresh["diagnostics"]["camera_breaks"] == 1


def test_the_anchor_is_the_last_clean_row() -> None:
    # A's last 5 rows are cut by the frame edge: their foot points are wrong, and the anchor
    # is the row before them.
    tracks = _fragment_scene()
    cut = (tracks["track_id"] == 1) & (tracks["frame"] >= 36)
    tracks.loc[cut, "x2"] = W - 0.5
    pair = _pair(_audit(tracks), 1, 2)
    assert pair["geometry_evidence"]["anchor"]["frame"] == 35


def test_an_anchor_older_than_the_limit_is_no_anchor() -> None:
    # A teammate runs shoulder to shoulder with A for its last 21 frames and A's earlier rows
    # are not clean either (too small): the last clean row is more than 2 s old.
    mate = _person(9, range(20, 41), _walk(-5.0 + 1.5 * 20 / FPS, 0.4, vx=1.5, start=20))
    tracks = _fragment_scene(mate)
    early = (tracks["track_id"] == 1) & (tracks["frame"] < 20)
    tracks.loc[early, "y1"] = tracks.loc[early, "y2"] - 45.0
    clean_until = tracks[(tracks["track_id"] == 1)]
    assert len(clean_until)
    teams = _teams({1: "left", 2: "left", 9: "left"})
    pair = _pair(_audit(tracks, team_report=teams), 1, 2)
    assert pair["geometry"] == "no_anchor"


def test_two_rows_of_b_on_one_frame_are_a_contradiction() -> None:
    tracks = _fragment_scene()
    extra = tracks[(tracks["track_id"] == 2) & (tracks["frame"] == 50)].copy()
    extra[["x1", "x2"]] += 3.0
    pair = _pair(_audit(pd.concat([tracks, extra], ignore_index=True)), 1, 2)
    assert pair["geometry"] == "contradiction"
    assert pair["geometry_evidence"]["contradictions"][0]["why"] == "two_rows_of_b"


def test_frames_where_b_is_not_clean_decide_nothing() -> None:
    # B's boxes are too small to be clean until the disc has outgrown the picture.
    tracks = _fragment_scene()
    small = (tracks["track_id"] == 2) & (tracks["frame"] < 72)
    tracks.loc[small, "y1"] = tracks.loc[small, "y2"] - 45.0
    pair = _pair(_audit(tracks), 1, 2)
    assert pair["geometry"] == "no_decision_frames"


def test_a_rival_without_a_position_counts() -> None:
    # Piece 5 is lost one second before B appears, far away, but none of its rows is clean
    # (they all overlap a teammate's), so where it went is unknown.
    lost = _person(5, range(0, 20), _walk(15.0, -20.0))
    beside = _person(6, range(0, 120), _walk(15.0, -19.6))
    teams = _teams({1: "left", 2: "left", 5: "left", 6: "left"})
    pair = _pair(_audit(_fragment_scene(lost, beside), team_report=teams), 1, 2)
    assert pair["geometry"] == "rivals"
    assert pair["geometry_evidence"]["rivals"] == [5]


def test_b_running_away_beyond_reach_later_is_a_jump() -> None:
    # The reviewers' case: B holds A's person at first, then (an id switch) somebody 17 m
    # away. One piece cannot be both.
    b = _person(2, range(44, 60), B_PATH) + _person(2, range(60, 120), _walk(12.0, -10.0))
    tracks = _tracks(_person(1, range(0, 41), A_PATH), b)
    pair = _pair(_audit(tracks), 1, 2)
    assert (pair["decision"], pair["veto"]) == ("abstain", "piece_jump")


def test_a_row_of_b_outside_the_image_of_the_disc_is_a_contradiction() -> None:
    # B's ground point says it is in the disc, but its box is drawn far from it: the table is
    # inconsistent, so nothing is proven.
    tracks = _fragment_scene()
    moved = (tracks["track_id"] == 2) & (tracks["frame"] == 50)
    tracks.loc[moved, ["x1", "x2"]] += 700.0
    pair = _pair(_audit(tracks), 1, 2)
    assert pair["geometry"] == "contradiction"
    assert pair["geometry_evidence"]["contradictions"][0]["why"] == "outside_disc"


def test_a_box_far_too_tall_for_one_person_is_a_contradiction() -> None:
    tracks = _fragment_scene()
    tall = (tracks["track_id"] == 2) & (tracks["frame"] == 50)
    tracks.loc[tall, "y1"] = tracks.loc[tall, "y1"] - 120.0
    pair = _pair(_audit(tracks), 1, 2)
    assert pair["geometry_evidence"]["contradictions"][0]["why"] == "box_not_one_person"


# --------------------------------------------------------------------------- vetoes


def test_two_verified_teams_that_differ_veto_the_pair() -> None:
    pair = _pair(_audit(_fragment_scene(), team_report=_teams({1: "left", 2: "right"})), 1, 2)
    assert pair["veto"] == "team_conflict"


def test_opposite_team_labels_veto_the_pair_without_a_colour_check() -> None:
    # Neither piece could be checked; the labels left in the table still disagree.
    tracks = _fragment_scene()
    tracks.loc[tracks["track_id"] == 2, "team"] = "right"
    pair = _pair(_audit(tracks, team_report=_teams({})), 1, 2)
    assert pair["veto"] == "team_conflict"


@pytest.mark.parametrize(
    "change",
    ["close_kits", "roles_implausible", "cleared", "samples_against", "two_entries"],
)
def test_a_team_is_verified_only_on_clean_evidence(change: str) -> None:
    teams = _teams({1: "left", 2: "left"})
    roles = ROLES_OK
    if change == "close_kits":
        teams["prototype_distance"] = 12.0
    elif change == "roles_implausible":
        roles = {"verdict": "implausible"}
    elif change == "cleared":
        for entry in teams["tracks"]:
            entry["cleared"] = "split_piece"
    elif change == "samples_against":
        for entry in teams["tracks"]:
            entry["samples_against"] = 1
    else:
        teams["tracks"] += [dict(entry, team="right") for entry in teams["tracks"]]
    report = _audit(_fragment_scene(), team_report=teams, role_report=roles)
    assert report["diagnostics"]["verified_team_pieces"] == 0


def test_colour_evidence_against_as_team_vetoes_the_pair() -> None:
    teams = _teams({1: "left"}, against={2: "right"})
    pair = _pair(_audit(_fragment_scene(), team_report=teams), 1, 2)
    assert (pair["decision"], pair["veto"]) == ("abstain", "team_conflict")


def test_two_different_confident_numbers_veto_the_pair() -> None:
    tracks = _fragment_scene()
    tracks.loc[tracks["track_id"] == 1, "jersey"] = "9"
    tracks.loc[tracks["track_id"] == 2, "jersey"] = "6"
    pair = _pair(_audit(tracks), 1, 2)
    assert pair["veto"] == "jersey_conflict"


def test_a_goalkeeper_is_never_joined_to_a_player() -> None:
    tracks = _fragment_scene()
    tracks.loc[tracks["track_id"] == 1, "role"] = "goalkeeper"
    pair = _pair(_audit(tracks), 1, 2)
    assert pair["veto"] == "role"


def test_an_unchecked_later_piece_can_still_be_proven() -> None:
    # The proof does not need B's team: B's shirt could not be checked.
    report = _audit(_fragment_scene(), team_report=_teams({1: "left"}))
    assert _pair(report, 1, 2)["decision"] == "stitch"


# --------------------------------------------------------------------------- path J


def _reads(tracks: pd.DataFrame, numbers: dict[int, str], conf: float = 0.95) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    out = tracks[["frame", "track_id", "source_track_id"]].copy()
    out["jersey"] = [numbers.get(int(t)) for t in out["track_id"]]
    out["jersey_conf"] = np.where(
        out["jersey"].notna(), conf + rng.uniform(-0.02, 0.02, len(out)), np.nan
    )
    return out


def _long_gap_scene() -> pd.DataFrame:
    # A is lost at frame 40; B appears 5 s later 20 m away: too long for path G.
    return _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(2, range(166, 230), _walk(8.0, -6.0, start=166)),
    )


def test_a_jersey_key_is_listed_but_does_not_stitch_while_the_path_is_off() -> None:
    tracks = _long_gap_scene()
    report = _audit(tracks, reads=_reads(tracks, {1: "9", 2: "9"}))
    pair = _pair(report, 1, 2)
    assert (pair["decision"], pair["jersey"], pair["path"]) == (
        "stitch_path_disabled",
        "pass",
        "jersey",
    )
    assert pair["geometry"] == "gap_too_long"
    assert report["paths_enabled"] == {"geometry": True, "jersey": False}
    assert report["diagnostics"]["jersey_reads"]["vary"] is True


def test_the_jersey_key_stitches_when_enabled() -> None:
    tracks = _long_gap_scene()
    params = stitch.StitchParams(jersey_enabled=True)
    report = _audit(tracks, reads=_reads(tracks, {1: "9", 2: "9"}), params=params)
    assert _pair(report, 1, 2)["decision"] == "stitch"


def test_a_teammate_reading_a_number_one_edit_away_blocks_the_jersey_key() -> None:
    mate = _person(4, range(60, 140), _walk(10.0, 10.0))
    tracks = _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(2, range(166, 230), _walk(8.0, -6.0, start=166)),
        mate,
    )
    pair = _pair(_audit(tracks, reads=_reads(tracks, {1: "9", 2: "9", 4: "19"})), 1, 2)
    assert pair["jersey"] == "confusable_number"


def test_weak_or_differing_reads_give_no_jersey_key() -> None:
    tracks = _long_gap_scene()
    assert (
        _pair(_audit(tracks, reads=_reads(tracks, {1: "9", 2: "9"}, conf=0.5)), 1, 2)["jersey"]
        == "reads_weak"
    )
    assert (
        _pair(_audit(tracks, reads=_reads(tracks, {1: "9", 2: "8"})), 1, 2)["jersey"]
        == "numbers_differ"
    )


def test_reads_repeated_on_every_row_look_like_votes_for_a_split_pair() -> None:
    # The same number with the same confidence on every row of the plugin track: the reads
    # may be one vote written to every row, so a split piece inherits its parent's number.
    tracks = _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(100001, range(166, 230), _walk(8.0, -6.0, start=166), source=1),
    )
    reads = _reads(tracks, {1: "9", 100001: "9"})
    reads["jersey_conf"] = 0.95
    teams = _teams({1: "left", 100001: "left"})
    report = _audit(tracks, reads=reads, team_report=teams)
    assert report["diagnostics"]["jersey_reads"]["vary"] is False
    assert _pair(report, 1, 100001)["jersey"] == "reads_may_be_votes"


def test_a_player_who_left_twice_is_a_chain_not_an_ambiguity() -> None:
    # 1 -> 6 -> 2, one number throughout: 1 -> 2 is proven too, and implied by the chain.
    tracks = _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(6, range(60, 100), _walk(5.0, -5.0)),
        _person(2, range(166, 230), _walk(8.0, -6.0, start=166)),
    )
    params = stitch.StitchParams(jersey_enabled=True)
    teams = _teams({1: "left", 2: "left", 6: "left"})
    report = _audit(
        tracks, reads=_reads(tracks, {1: "9", 2: "9", 6: "9"}), params=params, team_report=teams
    )
    decisions = {
        (e["earlier"], e["later"]): e["decision"]
        for key in ("stitches", "implied_by_chain", "ambiguous")
        for e in report[key]
    }
    assert decisions == {(1, 6): "stitch", (6, 2): "stitch", (1, 2): "implied_by_chain"}


def test_two_predecessors_that_are_not_one_chain_make_the_pairs_ambiguous() -> None:
    # B is proven to be A by geometry, and to be piece 6 by its number; 6 was seen at the
    # same time as A, so they are two people and B cannot be both.
    tracks = _tracks(
        _person(1, range(0, 101), _walk(-5.0, 0.0, vx=0.5)),
        _person(6, range(0, 41), _walk(5.0, -5.0)),
        _person(2, range(104, 200), _walk(-2.6, 0.0, vx=0.5, start=104)),
        _person(3, range(0, N_FRAMES), FAR, team="right"),
    )
    params = stitch.StitchParams(jersey_enabled=True)
    teams = _teams({1: "left", 2: "left", 6: "left"})
    report = _audit(
        tracks, reads=_reads(tracks, {2: "9", 6: "9"}), params=params, team_report=teams
    )
    assert (_pair(report, 1, 2)["geometry"], _pair(report, 6, 2)["jersey"]) == ("pass", "pass")
    decisions = {(e["earlier"], e["later"]): e["decision"] for e in report["ambiguous"]}
    assert decisions == {(1, 2): "ambiguous", (6, 2): "ambiguous"}
    assert report["counts"]["decisions"]["stitch"] == 0


def test_a_teammate_reading_the_number_while_a_is_seen_blocks_the_jersey_key() -> None:
    # Piece 4 reads 9 at the same time as A: the number is on two people, so it is no key.
    mate = _person(4, range(10, 60), _walk(10.0, 10.0))
    tracks = _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(2, range(166, 230), _walk(8.0, -6.0, start=166)),
        mate,
    )
    pair = _pair(_audit(tracks, reads=_reads(tracks, {1: "9", 2: "9", 4: "9"})), 1, 2)
    assert pair["jersey"] == "number_on_two_pieces"


def test_a_jersey_key_out_of_reach_is_refused() -> None:
    # B reads the same number 0.8 s after A was lost, 33 m away.
    tracks = _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(2, range(60, 120), _walk(14.0, -28.0, start=60)),
    )
    pair = _pair(_audit(tracks, reads=_reads(tracks, {1: "9", 2: "9"})), 1, 2)
    assert pair["jersey"] == "too_far"


def test_one_piece_with_two_successors_that_are_not_one_chain_is_ambiguous() -> None:
    # A -> B1 by geometry, A -> B2 by its number; B1 reads nothing, so B1 -> B2 is unproven
    # and A would hand its person to two pieces.
    tracks = _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(2, range(44, 100), B_PATH),
        _person(5, range(190, 236), _walk(6.0, -8.0, start=190)),
        _person(3, range(0, N_FRAMES), FAR, team="right"),
    )
    params = stitch.StitchParams(jersey_enabled=True)
    teams = _teams({1: "left", 2: "left", 5: "left"})
    report = _audit(
        tracks, reads=_reads(tracks, {1: "9", 5: "9"}), params=params, team_report=teams
    )
    assert (_pair(report, 1, 2)["geometry"], _pair(report, 1, 5)["jersey"]) == ("pass", "pass")
    decisions = {(e["earlier"], e["later"]): e["decision"] for e in report["ambiguous"]}
    assert decisions == {(1, 2): "ambiguous", (1, 5): "ambiguous"}


def test_one_piece_with_two_successors_in_a_chain_keeps_the_nearest() -> None:
    # The same, but B1 reads the number too: A -> B1 -> B2 is one chain.
    tracks = _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(2, range(44, 100), B_PATH),
        _person(5, range(190, 236), _walk(6.0, -8.0, start=190)),
        _person(3, range(0, N_FRAMES), FAR, team="right"),
    )
    params = stitch.StitchParams(jersey_enabled=True)
    teams = _teams({1: "left", 2: "left", 5: "left"})
    reads = _reads(tracks, {1: "9", 2: "9", 5: "9"})
    report = _audit(tracks, reads=reads, params=params, team_report=teams)
    decisions = {
        (e["earlier"], e["later"]): e["decision"]
        for key in ("stitches", "implied_by_chain", "ambiguous")
        for e in report[key]
    }
    assert decisions == {(1, 2): "stitch", (2, 5): "stitch", (1, 5): "implied_by_chain"}


def test_a_disabled_path_never_links_a_chain() -> None:
    # With path J off, B1 -> B2 is only proposed: A -> B2 must not count as implied by it.
    tracks = _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(2, range(44, 100), B_PATH),
        _person(5, range(190, 236), _walk(6.0, -8.0, start=190)),
        _person(3, range(0, N_FRAMES), FAR, team="right"),
    )
    teams = _teams({1: "left", 2: "left", 5: "left"})
    report = _audit(tracks, reads=_reads(tracks, {1: "9", 2: "9", 5: "9"}), team_report=teams)
    assert report["counts"]["decisions"]["implied_by_chain"] == 0
    assert _pair(report, 1, 2)["decision"] == "ambiguous"


def test_pieces_of_different_shots_are_never_paired() -> None:
    shots = pd.DataFrame(
        {"shot_id": [0, 1], "start_frame": [0, 42], "end_frame": [41, N_FRAMES - 1],
         "shot_type": ["wide", "wide"]}
    )  # fmt: skip
    report = stitch.audit(
        _fragment_scene(), _calib(), shots, fps=FPS,
        team_report=_teams({1: "left", 2: "left"}), role_report=ROLES_OK,
    )  # fmt: skip
    pairs = [
        (e["earlier"], e["later"])
        for key in ("stitches", "conflicts_with_plugin", "ambiguous", "implied_by_chain",
                    "abstentions")
        for e in report[key]
    ]  # fmt: skip
    assert (1, 2) not in pairs


def test_the_window_stops_at_the_end_of_the_shot() -> None:
    # After the cut at frame 61 somebody else stands where the disc would be: frames of
    # another shot say nothing about this one.
    shots = pd.DataFrame(
        {"shot_id": [0, 1], "start_frame": [0, 61], "end_frame": [60, N_FRAMES - 1],
         "shot_type": ["wide", "wide"]}
    )  # fmt: skip
    tracks = _tracks(
        _person(1, range(0, 41), A_PATH),
        _person(2, range(44, 61), B_PATH),
        _person(7, range(61, 120), _walk(-1.0, 1.0)),
        _person(3, range(0, N_FRAMES), FAR, team="right"),
    )
    report = stitch.audit(
        tracks, _calib(), shots, fps=FPS,
        team_report=_teams({1: "left", 2: "left"}), role_report=ROLES_OK,
    )  # fmt: skip
    pair = _pair(report, 1, 2)
    assert pair["geometry"] == "pass"
    assert pair["geometry_evidence"]["window"] == [44, 60]


def test_pairs_further_apart_than_the_candidate_gap_are_not_considered() -> None:
    tracks = _tracks(
        _person(1, range(0, 11), A_PATH),
        _person(2, range(11 + 260, 11 + 270), B_PATH),
    )
    shots = pd.DataFrame(
        {"shot_id": [0], "start_frame": [0], "end_frame": [400], "shot_type": ["wide"]}
    )
    calib = pd.concat([_calib(), _calib().assign(frame_index=lambda d: d["frame_index"] + 240)])
    report = stitch.audit(
        tracks, calib.iloc[:401].reset_index(drop=True), shots, fps=FPS,
        team_report=_teams({1: "left", 2: "left"}), role_report=ROLES_OK,
    )  # fmt: skip
    assert report["counts"]["pairs"] == 0


def test_string_columns_with_missing_values_are_read_as_missing() -> None:
    # pandas 3 reads text as a string dtype whose missing value is NaN, not None.
    tracks = _fragment_scene().astype({"team": "string", "jersey": "string", "role": "string"})
    report = _audit(tracks)
    assert _pair(report, 1, 2)["decision"] == "stitch"
    tracks.loc[tracks["track_id"] == 1, "team"] = pd.NA
    assert _pair(_audit(tracks), 1, 2)["decision"] == "stitch"


def test_no_calibration_proves_nothing_and_does_not_fail() -> None:
    calib = _calib()
    for column in ("K", "R", "C"):
        calib[column] = calib[column].astype(object)
        calib[column] = None
    calib["calib_status"] = "none"
    calib["pitch_usable"] = False
    report = _audit(_fragment_scene(), calib=calib)
    assert report["counts"]["decisions"]["stitch"] == 0
    assert report["diagnostics"]["frames_with_camera"] == 0


# --------------------------------------------------------------------------- geometry units


def test_convex_hull_and_box_meeting() -> None:
    square = stitch.convex_hull(
        np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [2, 8]], dtype=float)
    )
    assert len(square) == 4
    boxes = np.array(
        [
            [2, 2, 4, 4],  # inside
            [20, 20, 30, 30],  # outside
            [10, 2, 12, 4],  # touching the edge
            [-5, 4, 15, 6],  # crossing, no corner inside
            [11, 11, 12, 12],  # outside beyond a corner
        ],
        dtype=float,
    )
    assert stitch.boxes_meet_polygon(boxes, square).tolist() == [True, False, True, True, False]
    diamond = stitch.convex_hull(np.array([[5, 0], [10, 5], [5, 10], [0, 5]], dtype=float))
    # The box overlaps the diamond's bounding box but not the diamond.
    assert not stitch.boxes_meet_polygon(np.array([[0, 0, 1.5, 1.5]]), diamond)[0]


def test_reach_image_states() -> None:
    assert stitch.reach_image(CAMERA, (0.0, 0.0), 6.0)[0] == "covered"
    assert stitch.reach_image(CAMERA, (48.0, 0.0), 3.0)[0] == "near_pitch_line"
    assert stitch.reach_image(CAMERA, (15.0, 0.0), 8.0)[0] == "outside_view"
    small = stitch.StitchParams(min_person_px_1080=400.0)
    assert stitch.reach_image(CAMERA, (0.0, 0.0), 6.0, small)[0] == "too_small"
    # A camera above the centre spot looking towards the far touchline: a disc on the near
    # side lies behind it.
    behind = Camera(K=CAMERA.K, R=look_at((0.0, 0.0, -17.0), (0.0, -40.0, 0.0)),
                    C=(0.0, 0.0, -17.0), width=W, height=H)  # fmt: skip
    assert stitch.reach_image(behind, (0.0, 20.0), 4.0)[0] == "behind_camera"
    # The graphics band at the top of the picture: a disc whose upper rim reaches into it.
    banded = stitch.StitchParams(graphics_top_share=0.45)
    assert stitch.reach_image(CAMERA, (0.0, -10.0), 6.0, banded)[0] == "outside_view"


def test_the_hull_contains_a_person_standing_anywhere_in_the_disc() -> None:
    state, hull = stitch.reach_image(CAMERA, (-3.0, 0.0), 8.0)
    assert state == "covered"
    assert hull is not None
    for angle in np.linspace(0, 2 * math.pi, 13):
        for share in (0.0, 0.5, 0.99):
            x = -3.0 + 8.0 * share * math.cos(angle)
            y = 8.0 * share * math.sin(angle)
            assert stitch.boxes_meet_polygon(np.array([_box(x, y)]), hull)[0]


# --------------------------------------------------------------------------- inputs


def test_raw_reads_keep_the_plugins_reads_next_to_the_piece_ids() -> None:
    imported = pd.DataFrame(
        {"frame": [0, 1, 30], "track_id": [5, 5, 5], "jersey": ["9", None, "4"],
         "jersey_conf": [0.9, np.nan, 0.8]}
    )  # fmt: skip
    cleaned = imported.assign(
        track_id=[5, 5, 100005], source_track_id=[5, 5, 5], jersey=[None, None, None]
    )
    reads = stitch.raw_reads(imported, cleaned)
    assert reads["track_id"].tolist() == [5, 5, 100005]
    assert reads["source_track_id"].tolist() == [5, 5, 5]
    assert reads["jersey"].tolist() == ["9", None, "4"]
    with pytest.raises(stitch.StitchError, match="same rows"):
        stitch.raw_reads(imported, cleaned.iloc[::-1].reset_index(drop=True))


def test_params_are_checked() -> None:
    with pytest.raises(stitch.StitchError, match="max_speed_mps"):
        stitch.StitchParams(max_speed_mps=0.0)
    with pytest.raises(stitch.StitchError, match="box_height_ratio"):
        stitch.StitchParams(box_height_ratio=(1.5, 0.6))
    with pytest.raises(stitch.StitchError, match="graphics"):
        stitch.StitchParams(graphics_top_share=0.6, graphics_bottom_share=0.5)
    restored = stitch.StitchParams(**stitch.DEFAULT_PARAMS.as_dict())
    assert restored == stitch.DEFAULT_PARAMS


def test_missing_columns_are_refused() -> None:
    with pytest.raises(stitch.StitchError, match="pitch_x"):
        stitch.audit(_fragment_scene().drop(columns="pitch_x"), _calib(), SHOTS, fps=FPS)
