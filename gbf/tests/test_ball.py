"""Tests for gbf.ball: candidates file, gating, Viterbi tracking, interpolation and events.

Every scene is synthetic. Players, the ball and the distractors are placed on the pitch in
metres (SoccerNet frame, z down) and projected through gbf.geometry's synthetic broadcast
camera, so the expected pixel and pitch positions are known exactly. Ball boxes get the
size the detector convention implies (BallParams.ball_box_scale x the projected 0.22 m).
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbf.ball import (
    BALL_EVENT_COLUMNS,
    BALL_TRACK_COLUMNS,
    CONTACT_PARAMS,
    DEFAULT_PARAMS,
    GATE_COLUMNS,
    GATE_REASONS,
    BallInputError,
    BallParams,
    ball_events,
    gate_candidates,
    load_candidates,
    track_ball,
)
from gbf.geometry import Camera, project, synthetic_broadcast_camera

FPS = 25
W, H = 1920, 1080
BALL_Z = -0.11  # centre of a ball on the grass (z down)
PLAYER_H_M = 1.8


# ----------------------------------------------------------------------------------------
# Scene builders
# ----------------------------------------------------------------------------------------


def camera_at(target_x: float) -> Camera:
    """The synthetic main camera, panned to look at (target_x, 0, 0)."""
    return synthetic_broadcast_camera(W, H, target=(target_x, 0.0, 0.0))


def pixel(camera: Camera, x: float, y: float, z: float) -> tuple[float, float, float]:
    """(u, v, depth) of a world point; fails the test if it does not project."""
    uv, depth, valid = project(camera, np.array([x, y, z]))
    assert bool(valid)
    return float(uv[0]), float(uv[1]), float(depth)


def ball_cand(camera: Camera, x: float, y: float, conf: float) -> list[float]:
    """Detector box [x1, y1, x2, y2, conf] of a grounded ball centred at (x, y)."""
    u, v, depth = pixel(camera, x, y, BALL_Z)
    focal = 0.5 * (camera.K[0, 0] + camera.K[1, 1])
    d = focal * DEFAULT_PARAMS.ball_diameter_m * DEFAULT_PARAMS.ball_box_scale / depth
    return [u - d / 2, v - d / 2, u + d / 2, v + d / 2, conf]


def image_cand(u: float, v: float, d: float, conf: float) -> list[float]:
    return [u - d / 2, v - d / 2, u + d / 2, v + d / 2, conf]


def player_row(camera: Camera, frame: int, track_id: int, x: float, y: float, team: str) -> dict:
    """A tracks.parquet row for a 1.8 m player standing at (x, y)."""
    u_foot, v_foot, _ = pixel(camera, x, y, 0.0)
    _, v_head, _ = pixel(camera, x, y, -PLAYER_H_M)
    h = v_foot - v_head
    return box_row(frame, track_id, u_foot - 0.2 * h, v_head, u_foot + 0.2 * h, v_foot, team)


def box_row(
    frame: int, track_id: int, x1: float, y1: float, x2: float, y2: float, team: str | None
) -> dict:
    return {
        "frame": frame,
        "track_id": track_id,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "conf": 0.9,
        "role": "player",
        "team": team,
        "jersey": None,
        "jersey_conf": math.nan,
        "foot_u": (x1 + x2) / 2,
        "foot_v": y2,
        "calib_status": "fresh",
        "pitch_usable": True,
        "pitch_x": math.nan,
        "pitch_y": math.nan,
        "in_pitch": True,
    }


def calib_row(frame: int, camera: Camera | None) -> dict:
    """A calib.parquet row (calib.v1 plus pitch_usable)."""
    if camera is None:
        return {
            "frame_index": frame,
            "width": W,
            "height": H,
            "K": None,
            "R": None,
            "C": None,
            "calib_status": "none",
            "calib_age": 0,
            "rep_err": math.nan,
            "rep_err_n_keypoints": 0,
            "pitch_usable": False,
        }
    return {
        "frame_index": frame,
        "width": camera.width,
        "height": camera.height,
        "K": camera.K.ravel().tolist(),
        "R": camera.R.ravel().tolist(),
        "C": camera.C.tolist(),
        "calib_status": "fresh",
        "calib_age": 0,
        "rep_err": 2.0,
        "rep_err_n_keypoints": 12,
        "pitch_usable": True,
    }


def shots_table(bounds: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": np.arange(len(bounds), dtype=np.int64),
            "start_frame": [a for a, _ in bounds],
            "end_frame": [b for _, b in bounds],
            "shot_type": ["wide"] * len(bounds),
            "fresh_share": [1.0] * len(bounds),
            "median_box_h": [100.0] * len(bounds),
        }
    )


@dataclasses.dataclass
class Scene:
    candidates: dict[int, list[list[float]]]
    tracks: pd.DataFrame
    calib: pd.DataFrame
    shots: pd.DataFrame
    n_frames: int
    truth: dict[int, tuple[float, float]]  # frame -> true ball (u, v), detected or not
    truth_xy: dict[int, tuple[float, float]]  # frame -> true ball pitch (x, y)
    distractors: dict[str, dict[int, tuple[float, float]]]  # name -> frame -> (u, v)


# Main scene: shot A (frames 0-79) is a calibrated wide shot whose camera pans from x = -4
# to x = +4; shot B (80-119) is another, uncalibrated wide camera whose ball continues
# shot A's image motion and size, so only the cut separates them.
KICK_FRAME = 40
CUT = 80
N_MAIN = 120
GAP_SHORT = (10, 11, 12)  # dropped detections, interpolated
GAP_LONG = tuple(range(50, 60))  # dropped detections, too long to interpolate
BEFORE_CUT = (78, 79)  # dropped detections, never bridged across the cut
GAP_B = (100, 101, 102)  # dropped detections in shot B, interpolated
LOGO_UV = (300.0, 1000.0)
KICKER = (5.5, 8.0)


def ball_xy(frame: int) -> tuple[float, float]:
    """True ball path of shot A: 7 m/s along +x, kicked at KICK_FRAME to (-3, -12) m/s."""
    t = frame / FPS
    t_kick = KICK_FRAME / FPS
    if frame <= KICK_FRAME:
        return -6.0 + 7.0 * t, 8.0
    return -6.0 + 7.0 * t_kick - 3.0 * (t - t_kick), 8.0 - 12.0 * (t - t_kick)


def main_scene(seed: int = 7) -> Scene:
    rng = np.random.default_rng(seed)
    candidates: dict[int, list[list[float]]] = {}
    tracks: list[dict] = []
    calib: list[dict] = []
    truth: dict[int, tuple[float, float]] = {}
    truth_xy: dict[int, tuple[float, float]] = {}
    diameters: dict[int, float] = {}
    distractors: dict[str, dict[int, tuple[float, float]]] = {
        "logo": {},
        "spare_inside_margin": {},
        "spare_off_pitch": {},
        "noise": {},
    }
    players = [(1, *KICKER, "left"), (2, -12.0, 2.0, "right"), (3, 14.0, -6.0, "left")]
    players += [(4, -2.0, -15.0, "right"), (5, 20.0, 12.0, "right"), (6, -18.0, 14.0, "left")]
    for frame in range(CUT):
        camera = camera_at(-4.0 + 8.0 * frame / (CUT - 1))
        calib.append(calib_row(frame, camera))
        tracks.extend(player_row(camera, frame, *p) for p in players)
        x, y = ball_xy(frame)
        cand = ball_cand(camera, x, y, 0.5)
        truth[frame] = ((cand[0] + cand[2]) / 2, (cand[1] + cand[3]) / 2)
        truth_xy[frame] = (x, y)
        diameters[frame] = cand[2] - cand[0]
        row: list[list[float]] = []
        if frame not in GAP_SHORT + GAP_LONG + BEFORE_CUT:
            row.append(cand)
        row.append(image_cand(*LOGO_UV, 14.0, 0.7))
        distractors["logo"][frame] = LOGO_UV
        spares = (("spare_inside_margin", (-2.0, -35.8)), ("spare_off_pitch", (8.0, -38.0)))
        for name, (sx, sy) in spares:
            spare = ball_cand(camera, sx, sy, 0.8)
            row.append(spare)
            distractors[name][frame] = ((spare[0] + spare[2]) / 2, (spare[1] + spare[3]) / 2)
        for _ in range(2):  # clutter, never within 80 px of the ball (that would be the ball)
            u, v = truth[frame]
            while math.hypot(u - truth[frame][0], v - truth[frame][1]) < 80.0:
                u, v = rng.uniform(100, 1800), rng.uniform(450, 1000)
            row.append(image_cand(u, v, rng.uniform(10, 16), rng.uniform(0.05, 0.25)))
            distractors["noise"].setdefault(frame, (u, v))
        candidates[frame] = row
    # Shot B, another wide camera without calibration: its ball continues shot A's image
    # motion and size, so only the cut keeps the tracker from linking and interpolating
    # across it.
    last = np.array(truth[CUT - 3])
    step = last - np.array(truth[CUT - 4])
    box_h = diameters[CUT - 3] / DEFAULT_PARAMS.ball_to_box_height
    for frame in range(CUT, N_MAIN):
        calib.append(calib_row(frame, None))
        u, v = (float(c) for c in last + step * (frame - (CUT - 3)))
        truth[frame] = (u, v)
        for track_id, du in ((11, -150.0), (12, 160.0)):
            x1, x2 = u + du - 0.2 * box_h, u + du + 0.2 * box_h
            tracks.append(box_row(frame, track_id, x1, v - box_h / 2, x2, v + box_h / 2, "left"))
        row = [] if frame in GAP_B else [image_cand(u, v, diameters[CUT - 3], 0.8)]
        row.append(image_cand(*LOGO_UV, 14.0, 0.7))  # the graphic survives the cut
        distractors["logo"][frame] = LOGO_UV
        candidates[frame] = row
    return Scene(
        candidates=candidates,
        tracks=pd.DataFrame(tracks),
        calib=pd.DataFrame(calib),
        shots=shots_table([(0, CUT - 1), (CUT, N_MAIN - 1)]),
        n_frames=N_MAIN,
        truth=truth,
        truth_xy=truth_xy,
        distractors=distractors,
    )


@pytest.fixture(scope="module")
def scene() -> Scene:
    return main_scene()


@pytest.fixture(scope="module")
def tracked(scene: Scene) -> pd.DataFrame:
    return track_ball(scene.candidates, scene.tracks, scene.calib, scene.shots, FPS, scene.n_frames)


@pytest.fixture(scope="module")
def gates(scene: Scene) -> pd.DataFrame:
    return gate_candidates(
        scene.candidates, scene.tracks, scene.calib, scene.shots, FPS, scene.n_frames
    )


# ----------------------------------------------------------------------------------------
# Main scene: tracking
# ----------------------------------------------------------------------------------------


def test_scene_is_in_view(scene: Scene) -> None:
    """Guard for the other tests: everything the scene places lies inside the image."""
    for name, positions in [("ball", scene.truth), *scene.distractors.items()]:
        for frame, (u, v) in positions.items():
            assert 0 <= u < W, (name, frame, u)
            assert 0 <= v < H, (name, frame, v)


def test_track_has_the_contract_columns(scene: Scene, tracked: pd.DataFrame) -> None:
    assert tuple(tracked.columns) == BALL_TRACK_COLUMNS
    assert tracked["frame"].tolist() == list(range(scene.n_frames))
    assert tracked["frame"].dtype == np.int64
    assert tracked["conf"].dtype == np.float32
    assert tracked["candidate_count"].dtype == np.int32
    assert tracked["ground_assumption"].dtype == bool
    for column in ("u", "v", "radius_px", "pitch_x", "pitch_y"):
        assert tracked[column].dtype == np.float64
    assert set(tracked["status"]) <= {"detected", "interpolated", "missing"}
    expected_counts = [len(scene.candidates.get(f, [])) for f in range(scene.n_frames)]
    assert tracked["candidate_count"].tolist() == expected_counts


def test_track_follows_the_true_ball(scene: Scene, tracked: pd.DataFrame) -> None:
    dropped = set(GAP_SHORT + GAP_LONG + BEFORE_CUT + GAP_B)
    for frame, (u, v) in scene.truth.items():
        row = tracked.loc[frame]
        if frame in dropped:
            continue
        assert row["status"] == "detected", frame
        assert math.hypot(row["u"] - u, row["v"] - v) < 1e-6, frame
        assert row["conf"] == pytest.approx(0.8 if frame >= CUT else 0.5)


def test_distractors_are_never_tracked(scene: Scene, tracked: pd.DataFrame) -> None:
    for name, positions in scene.distractors.items():
        for frame, (u, v) in positions.items():
            row = tracked.loc[frame]
            if row["status"] == "missing":
                continue
            assert math.hypot(row["u"] - u, row["v"] - v) > 5.0, (name, frame)


def test_gaps_interpolate_inside_a_shot_but_never_across_a_cut(
    scene: Scene, tracked: pd.DataFrame
) -> None:
    status = tracked["status"]
    for frame in GAP_SHORT + GAP_B:
        assert status[frame] == "interpolated", frame
        u, v = scene.truth[frame]
        assert math.hypot(tracked["u"][frame] - u, tracked["v"][frame] - v) < 1.0, frame
        assert math.isnan(tracked["conf"][frame])
    for frame in GAP_LONG + BEFORE_CUT:
        assert status[frame] == "missing", frame
        assert math.isnan(tracked["u"][frame])
    assert status[CUT - 3] == "detected"
    assert status[CUT] == "detected"


def test_pitch_position_is_the_ground_plane_point(scene: Scene, tracked: pd.DataFrame) -> None:
    shot_a = tracked[(tracked["frame"] < CUT) & (tracked["status"] == "detected")]
    assert len(shot_a) > 50
    for row in shot_a.itertuples():
        x, y = scene.truth_xy[row.frame]
        assert row.pitch_x == pytest.approx(x, abs=1e-6)
        assert row.pitch_y == pytest.approx(y, abs=1e-6)
        assert row.ground_assumption
    shot_b = tracked[tracked["frame"] >= CUT]
    assert shot_b["pitch_x"].isna().all()
    assert shot_b["pitch_y"].isna().all()
    assert not shot_b["ground_assumption"].any()
    missing = tracked[tracked["status"] == "missing"]
    assert missing["pitch_x"].isna().all()
    assert not missing["ground_assumption"].any()
    assert (tracked["ground_assumption"] == tracked["pitch_x"].notna()).all()


# ----------------------------------------------------------------------------------------
# Main scene: gating
# ----------------------------------------------------------------------------------------


def _rows_at(gates: pd.DataFrame, positions: dict[int, tuple[float, float]]) -> pd.DataFrame:
    keys = {(f, round(u, 6), round(v, 6)) for f, (u, v) in positions.items()}
    mask = [
        (f, round(u, 6), round(v, 6)) in keys
        for f, u, v in zip(gates["frame"], gates["u"], gates["v"], strict=True)
    ]
    return gates[mask]


def test_gate_table_columns(scene: Scene, gates: pd.DataFrame) -> None:
    assert tuple(gates.columns) == GATE_COLUMNS
    assert len(gates) == sum(len(c) for c in scene.candidates.values())
    assert set(gates["reason"]) <= set(GATE_REASONS)


def test_screen_fixed_graphic_is_static(scene: Scene, gates: pd.DataFrame) -> None:
    logo = _rows_at(gates, scene.distractors["logo"])
    assert len(logo) == N_MAIN
    assert not logo["static_ok"].any()
    assert (logo["static_run"] == N_MAIN).all()
    in_a = logo[logo["frame"] < CUT]
    assert (in_a["reason"] == "static").all()  # on the pitch, plausible size: only static


def test_spare_balls_beyond_the_touchline_are_rejected(scene: Scene, gates: pd.DataFrame) -> None:
    near = _rows_at(gates, scene.distractors["spare_inside_margin"])
    far = _rows_at(gates, scene.distractors["spare_off_pitch"])
    assert len(near) == CUT
    assert len(far) == CUT
    assert near["pitch_ok"].all()  # y = -35.8 passes the 36.5 m gate ...
    assert (near["reason"] == "off_field_static").all()  # ... but is world-fixed off the field
    assert (far["reason"] == "off_pitch").all()
    assert near["pitch_y"].to_numpy() == pytest.approx(-35.8, abs=1e-6)


def test_true_ball_passes_every_gate(scene: Scene, gates: pd.DataFrame) -> None:
    dropped = set(GAP_SHORT + GAP_LONG + BEFORE_CUT + GAP_B)
    positions = {f: uv for f, uv in scene.truth.items() if f not in dropped}
    ball = _rows_at(gates, positions)
    assert len(ball) == len(positions)
    assert (ball["reason"] == "ok").all()
    calibrated = ball[ball["frame"] < CUT]
    assert calibrated["size_prior"].to_numpy() == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------------------------
# Main scene: events
# ----------------------------------------------------------------------------------------


def test_planted_kick_is_found_at_the_exact_frame(scene: Scene, tracked: pd.DataFrame) -> None:
    events = ball_events(tracked, scene.tracks, FPS, shots=scene.shots)
    assert events["frame"].tolist() == [KICK_FRAME]
    event = events.iloc[0]
    assert event["kind"] == "direction_change"
    assert event["nearest_track_id"] == 1
    assert event["nearest_team"] == "left"
    assert event["at_foot"]
    assert event["nearest_dist_box_h"] < 0.3
    assert event["shot_id"] == 0
    assert (event["u"], event["v"]) == pytest.approx(scene.truth[KICK_FRAME])


# ----------------------------------------------------------------------------------------
# Still camera: a ball at rest, a set piece, parked objects
# ----------------------------------------------------------------------------------------

KICK_STILL = 30
REST = (0.0, 5.0)


def still_scene(*, kicker_at_ball: bool) -> Scene:
    """60 frames from a still, calibrated camera.

    A ball lies at REST until KICK_STILL (0.2 px detector jitter) and is then kicked at
    (15, -5) m/s; player 1 stands at the ball (or 6 m away); a graphic sits in a corner.
    """
    camera = camera_at(0.0)
    rng = np.random.default_rng(3)
    candidates: dict[int, list[list[float]]] = {}
    tracks: list[dict] = []
    calib: list[dict] = []
    truth: dict[int, tuple[float, float]] = {}
    truth_xy: dict[int, tuple[float, float]] = {}
    player = (REST[0] - 0.4, REST[1]) if kicker_at_ball else (REST[0] - 6.0, REST[1] + 3.0)
    for frame in range(60):
        calib.append(calib_row(frame, camera))
        tracks.append(player_row(camera, frame, 1, *player, "left"))
        tracks.append(player_row(camera, frame, 2, 10.0, -5.0, "right"))
        t = max(0, frame - KICK_STILL) / FPS
        x, y = REST[0] + 15.0 * t, REST[1] - 5.0 * t
        cand = ball_cand(camera, x, y, 0.6)
        if frame <= KICK_STILL:
            du, dv = rng.normal(0.0, 0.2, 2)
            cand = [cand[0] + du, cand[1] + dv, cand[2] + du, cand[3] + dv, cand[4]]
        truth[frame] = ((cand[0] + cand[2]) / 2, (cand[1] + cand[3]) / 2)
        truth_xy[frame] = (x, y)
        candidates[frame] = [cand, image_cand(*LOGO_UV, 14.0, 0.7)]
    return Scene(
        candidates=candidates,
        tracks=pd.DataFrame(tracks),
        calib=pd.DataFrame(calib),
        shots=shots_table([(0, 59)]),
        n_frames=60,
        truth=truth,
        truth_xy=truth_xy,
        distractors={"logo": dict.fromkeys(range(60), LOGO_UV)},
    )


def _run(scene: Scene, params: BallParams = DEFAULT_PARAMS) -> tuple[pd.DataFrame, pd.DataFrame]:
    args = (scene.candidates, scene.tracks, scene.calib, scene.shots, FPS, scene.n_frames)
    return track_ball(*args, params=params), gate_candidates(*args, params=params)


def test_resting_ball_with_a_player_is_tracked_and_the_kick_found() -> None:
    scene = still_scene(kicker_at_ball=True)
    tracked, gates = _run(scene)
    assert (tracked["status"] == "detected").all()
    for frame, (u, v) in scene.truth.items():
        assert math.hypot(tracked["u"][frame] - u, tracked["v"][frame] - v) < 1e-6
    logo = _rows_at(gates, scene.distractors["logo"])
    assert (logo["reason"] == "parked").all()  # a still camera: parked, not static
    assert logo["static_ok"].all()
    events = ball_events(tracked, scene.tracks, FPS, shots=scene.shots)
    assert events["frame"].tolist() == [KICK_STILL]
    assert events["kind"].iloc[0] == "speed_up"
    assert events["nearest_track_id"].iloc[0] == 1
    assert bool(events["at_foot"].iloc[0])


def test_resting_ball_without_a_player_is_parked() -> None:
    scene = still_scene(kicker_at_ball=False)
    tracked, gates = _run(scene)
    resting = _rows_at(gates, {f: scene.truth[f] for f in range(KICK_STILL + 1)})
    assert len(resting) == KICK_STILL + 1
    assert (resting["reason"] == "parked").all()
    assert not resting["near_player"].any()
    assert (tracked["status"][: KICK_STILL + 1] == "missing").all()
    assert (tracked["status"][KICK_STILL + 1 :] == "detected").all()


def test_graphic_on_an_uncalibrated_pan_is_static_from_player_motion() -> None:
    """No calibration: the players' median motion shows the pan. Frames where a player is
    at the graphic stay eligible, because players can also move together."""
    candidates: dict[int, list[list[float]]] = {}
    tracks: list[dict] = []
    for frame in range(40):
        for track_id, u0 in enumerate((500.0, 800.0, 1100.0, 1400.0), start=1):
            u = u0 + 6.0 * frame
            tracks.append(box_row(frame, track_id, u - 20, 500.0, u + 20, 600.0, "left"))
        if 20 <= frame <= 24:  # a player walks over the graphic
            tracks.append(box_row(frame, 9, 280.0, 920.0, 320.0, 1020.0, "right"))
        candidates[frame] = [image_cand(*LOGO_UV, 14.0, 0.7)]
    calib = pd.DataFrame([calib_row(f, None) for f in range(40)])
    shots = shots_table([(0, 39)])
    gates = gate_candidates(candidates, pd.DataFrame(tracks), calib, shots, FPS, 40)
    passing = gates["frame"].between(20, 24)
    assert (gates.loc[~passing, "reason"] == "static").all()
    assert gates.loc[passing, "static_ok"].all()
    assert gates.loc[passing, "near_player"].all()


# ----------------------------------------------------------------------------------------
# Uncalibrated image-plane scenes: loss, re-acquisition, interpolation limits
# ----------------------------------------------------------------------------------------


def image_scene(
    path: dict[int, tuple[float, float]],
    n: int,
    *,
    bounds: list[tuple[int, int]] | None = None,
    extra: dict[int, list[list[float]]] | None = None,
    conf: dict[int, float] | None = None,
) -> Scene:
    """Uncalibrated frames with two 100 px players and a 15.5 px ball at ``path``."""
    candidates: dict[int, list[list[float]]] = {}
    tracks: list[dict] = []
    for frame in range(n):
        tracks.append(box_row(frame, 1, 700.0, 450.0, 740.0, 550.0, "left"))
        tracks.append(box_row(frame, 2, 1100.0, 450.0, 1140.0, 550.0, "right"))
        row = []
        if frame in path:
            row.append(image_cand(*path[frame], 15.5, (conf or {}).get(frame, 0.5)))
        row.extend((extra or {}).get(frame, []))
        candidates[frame] = row
    return Scene(
        candidates=candidates,
        tracks=pd.DataFrame(tracks),
        calib=pd.DataFrame([calib_row(f, None) for f in range(n)]),
        shots=shots_table(bounds or [(0, n - 1)]),
        n_frames=n,
        truth=dict(path),
        truth_xy={},
        distractors={},
    )


def test_track_reacquires_after_a_loss_and_ignores_a_brief_distractor() -> None:
    path = {f: (400.0 + 5 * f, 600.0) for f in range(30)}
    path.update({f: (900.0 + 5 * (f - 50), 400.0) for f in range(50, 70)})
    flash = {f: [image_cand(1500.0, 300.0, 15.5, 0.9)] for f in (10, 11)}
    tracked, _ = _run(image_scene(path, 70, extra=flash))
    status = tracked["status"]
    assert (status[list(path)] == "detected").all()
    assert (status[30:50] == "missing").all()
    assert tracked["u"][10] == pytest.approx(450.0)
    assert not (status == "interpolated").any()


def test_slow_gaps_are_interpolated_but_implausibly_fast_ones_stay_missing() -> None:
    """Scale 100 px (15.5 px ball): 30 px per frame is 7.5 bh/s, 95 px is 23.75 bh/s, above
    max_interp_speed_bh_per_s (20) but below max_link_speed_bh_per_s (30)."""
    path = {f: (100.0 + 30 * f, 600.0) for f in range(20) if f != 10}
    path.update({f: (100.0 + 95 * (f - 20), 300.0) for f in range(20, 40) if f != 30})
    tracked, _ = _run(image_scene(path, 40, bounds=[(0, 19), (20, 39)]))
    assert tracked["status"][10] == "interpolated"
    assert tracked["u"][10] == pytest.approx(400.0)
    assert tracked["status"][30] == "missing"
    assert (tracked["status"][[29, 31]] == "detected").all()


@pytest.mark.parametrize(("conf", "kept"), [(0.3, False), (0.7, True)])
def test_isolated_weak_detection_is_dropped(conf: float, kept: bool) -> None:
    path = {f: (300.0 + 5 * f, 600.0) for f in (*range(15), 20, *range(25, 40))}
    tracked, _ = _run(image_scene(path, 40, conf={20: conf}))
    status = tracked["status"]
    assert status[20] == ("detected" if kept else "missing")
    gap = list(range(15, 20)) + list(range(21, 25))
    assert (status[gap] == ("interpolated" if kept else "missing")).all()
    assert (status[25:] == "detected").all()


@pytest.mark.parametrize(
    ("sigma", "stays"), [(DEFAULT_PARAMS.size_change_sigma_log, True), (1e6, False)]
)
def test_track_does_not_switch_to_an_object_of_another_size(sigma: float, stays: bool) -> None:
    """A close shot: an 80 px ball, and for five frames a 30 px object 40 px beside it with
    a higher confidence. Only the size-continuity term keeps the track on the ball (with
    sigma = 1e6 the term is off and the track switches)."""
    path = {f: (600.0 + 10 * f, 700.0) for f in range(30)}
    candidates = {
        f: [image_cand(u, v, 80.0, 0.6)]
        + ([image_cand(u, v - 40.0, 30.0, 0.95)] if 10 <= f < 15 else [])
        for f, (u, v) in path.items()
    }
    tracks = pd.DataFrame({column: [] for column in box_row(0, 0, 0, 0, 1, 1, None)})
    calib = pd.DataFrame([calib_row(f, None) for f in range(30)])
    params = dataclasses.replace(DEFAULT_PARAMS, size_change_sigma_log=sigma)
    tracked = track_ball(candidates, tracks, calib, shots_table([(0, 29)]), FPS, 30, params=params)
    on_ball = tracked["v"][10:15].to_numpy() == pytest.approx(700.0)
    assert on_ball == stays


def test_a_player_is_at_the_ball_beside_and_below_the_box_but_not_above() -> None:
    """Player box x 900-940, y 400-500 (h = 100 px, margin 30 px)."""
    candidates = {
        0: [
            image_cand(920.0, 380.0, 15.0, 0.5),  # above the head: what stands behind him
            image_cand(925.0, 495.0, 15.0, 0.5),  # at the feet
            image_cand(965.0, 470.0, 15.0, 0.5),  # beside the boot, inside the margin
            image_cand(990.0, 470.0, 15.0, 0.5),  # beyond the margin
        ]
    }
    tracks = pd.DataFrame([box_row(0, 1, 900.0, 400.0, 940.0, 500.0, "left")])
    calib = pd.DataFrame([calib_row(0, None)])
    gates = gate_candidates(candidates, tracks, calib, shots_table([(0, 0)]), FPS, 1)
    assert gates["near_player"].tolist() == [False, True, True, False]


def test_calibration_of_another_resolution_is_rescaled() -> None:
    """Calibration at 1920x1080 with calib.jsonl's 'frame' column, video at 960x540."""
    full = camera_at(2.0)
    half = full.resized(960, 540)
    xy = [(-3.0 + 0.2 * f, 6.0) for f in range(30)]
    candidates = {f: [ball_cand(half, x, y, 0.6)] for f, (x, y) in enumerate(xy)}
    calib = pd.DataFrame([calib_row(f, full) for f in range(30)])
    calib = calib.rename(columns={"frame_index": "frame"})
    tracks = pd.DataFrame([player_row(half, f, 1, -10.0, 0.0, "left") for f in range(30)])
    shots = shots_table([(0, 29)])
    tracked = track_ball(candidates, tracks, calib, shots, FPS, 30, frame_size=(960, 540))
    assert (tracked["status"] == "detected").all()
    assert tracked["pitch_x"].to_numpy() == pytest.approx([x for x, _ in xy], abs=1e-6)
    assert tracked["pitch_y"].to_numpy() == pytest.approx([y for _, y in xy], abs=1e-6)


# ----------------------------------------------------------------------------------------
# Candidates file and input checks
# ----------------------------------------------------------------------------------------


def _write_lines(path: Path, lines: list[str], *, newline_at_end: bool = True) -> None:
    text = "\n".join(lines) + ("\n" if newline_at_end else "")
    path.write_bytes(text.encode("utf-8"))


def test_load_candidates_reads_frames_and_empty_lists(unicode_tmp_path: Path) -> None:
    path = unicode_tmp_path / "ball_raw.jsonl"
    rows = [
        {"frame": 3, "cands": [[10, 20, 22.5, 31, 0.5], [1, 1, 5, 5, 0.06]]},
        {"frame": 0, "cands": []},
    ]
    _write_lines(path, [json.dumps(r) for r in rows] + [""])
    loaded = load_candidates(path)
    assert list(loaded) == [0, 3]
    assert loaded[0].shape == (0, 5)
    assert loaded[3].dtype == np.float64
    assert loaded[3].tolist() == [[10, 20, 22.5, 31, 0.5], [1, 1, 5, 5, 0.06]]


def test_load_candidates_lists_every_problem(unicode_tmp_path: Path) -> None:
    path = unicode_tmp_path / "ball_raw.jsonl"
    lines = [
        '{"frame": 0, "cands": [[10, 20, 5, 31, 0.5]]}',
        "not json",
        '{"frame": -1, "cands": []}',
        '{"frame": 0, "cands": []}',
        '{"frame": 3, "cands": [[1, 2, 3]]}',
        '{"frame": 4, "cands": [[1, 2, 3, 4, 1.5]]}',
        '{"frame": 5}',
        '{"frame": 6, "cands": [[1, 2, 3, NaN, 0.5]]}',
    ]
    _write_lines(path, lines)
    with pytest.raises(BallInputError) as info:
        load_candidates(path)
    problems = info.value.problems
    expected = (
        "x2 > x1",
        "invalid JSON",
        ">= 0",
        "more than once",
        "5 numbers",
        "outside [0, 1]",
        '"cands"',
        "non-finite",
    )
    assert len(problems) == len(expected)
    for number, (fragment, problem) in enumerate(zip(expected, problems, strict=True), start=1):
        assert problem.startswith(f"line {number}"), problem
        assert fragment in problem, problem


def test_load_candidates_incomplete_last_line(unicode_tmp_path: Path) -> None:
    path = unicode_tmp_path / "ball_raw.jsonl"
    _write_lines(path, ['{"frame": 0, "cands": []}', '{"frame": 1, "ca'], newline_at_end=False)
    with pytest.raises(BallInputError, match="still being"):
        load_candidates(path)
    assert list(load_candidates(path, allow_incomplete_tail=True)) == [0]


def test_inputs_are_checked_and_every_problem_listed(scene: Scene) -> None:
    bad_shots = shots_table([(0, 50), (60, N_MAIN - 1)])
    bad_tracks = scene.tracks.drop(columns=["foot_u"])
    bad_calib = scene.calib.copy()
    bad_calib.at[0, "R"] = (2.0 * np.eye(3)).ravel().tolist()
    bad_candidates = {**scene.candidates, N_MAIN + 5: [[1.0, 1.0, 9.0, 9.0, 0.5]]}
    with pytest.raises(BallInputError) as info:
        track_ball(bad_candidates, bad_tracks, bad_calib, bad_shots, "25.0", N_MAIN)
    text = "\n".join(info.value.problems)
    for fragment in ("foot_u", "starts at 60", "fps", "calib frame 0", f"frame {N_MAIN + 5}"):
        assert fragment in text, fragment
    no_size = scene.calib.drop(columns=["width", "height"])
    with pytest.raises(BallInputError, match="frame_size"):
        track_ball({}, scene.tracks, no_size, scene.shots, FPS, N_MAIN)


def test_params_are_checked() -> None:
    with pytest.raises(BallInputError) as info:
        BallParams(
            aspect_min=2.0,
            missing_cost=-1.0,
            event_speed_ratio=0.5,
            player_roles=["x"],
            blur_max_step=0.3,
        )
    text = "\n".join(info.value.problems)
    for fragment in (
        "aspect_min",
        "missing_cost",
        "event_speed_ratio",
        "player_roles",
        "blur_min_step (0.4) must be < blur_max_step",
    ):
        assert fragment in text, fragment
    assert CONTACT_PARAMS.event_speed_floor_bh_s < DEFAULT_PARAMS.event_speed_floor_bh_s


# ----------------------------------------------------------------------------------------
# ball_events on hand-made tracks (15.5 px ball: 1 bh = 100 px, so 4 px/frame = 1 bh/s)
# ----------------------------------------------------------------------------------------

NO_PLAYERS = pd.DataFrame({column: [] for column in box_row(0, 0, 0, 0, 1, 1, None)})


def ball_track_from(
    points: dict[int, tuple[float, float]], n: int, radius: dict[int, float] | None = None
) -> pd.DataFrame:
    detected = [f in points for f in range(n)]
    return pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "u": [points[f][0] if f in points else math.nan for f in range(n)],
            "v": [points[f][1] if f in points else math.nan for f in range(n)],
            "radius_px": [
                (radius or {}).get(f, 7.75) if d else math.nan for f, d in enumerate(detected)
            ],
            "conf": np.where(detected, 0.5, np.nan).astype(np.float32),
            "status": ["detected" if d else "missing" for d in detected],
            "candidate_count": np.ones(n, dtype=np.int32),
            "pitch_x": np.full(n, np.nan),
            "pitch_y": np.full(n, np.nan),
            "ground_assumption": np.zeros(n, dtype=bool),
        }
    )


def polyline(
    n: int,
    start: tuple[float, float],
    v1: tuple[float, float],
    turn: int,
    v2: tuple[float, float],
    noise: float = 0.0,
    seed: int = 0,
) -> dict[int, tuple[float, float]]:
    """Positions moving at v1 (px per frame) up to frame ``turn`` and at v2 after it."""
    rng = np.random.default_rng(seed)
    points = {}
    for f in range(n):
        a, b = min(f, turn), max(0, f - turn)
        du, dv = rng.normal(0.0, noise, 2) if noise else (0.0, 0.0)
        points[f] = (
            start[0] + v1[0] * a + v2[0] * b + du,
            start[1] + v1[1] * a + v2[1] * b + dv,
        )
    return points


def test_direction_change_is_found_at_the_exact_frame() -> None:
    points = polyline(40, (300.0, 600.0), (8.0, 0.0), 20, (0.0, -8.0))
    shots = shots_table([(0, 39)])
    events = ball_events(ball_track_from(points, 40), NO_PLAYERS, FPS, shots=shots)
    assert tuple(events.columns) == BALL_EVENT_COLUMNS
    assert events["frame"].tolist() == [20]
    event = events.iloc[0]
    assert event["kind"] == "direction_change"
    assert event["angle_deg"] == pytest.approx(90.0)
    assert event["speed_before"] == pytest.approx(2.0)
    assert event["speed_after"] == pytest.approx(2.0)
    assert event["magnitude"] == pytest.approx(2.0 * math.sqrt(2.0))
    assert event["nearest_track_id"] == -1
    assert math.isnan(event["nearest_dist_box_h"])
    assert not event["at_foot"]


@pytest.mark.parametrize(
    ("v1", "v2", "kind"),
    [((0.0, 0.0), (12.0, 0.0), "speed_up"), ((12.0, 3.0), (1.0, 0.25), "slow_down")],
)
def test_speed_changes_are_found_at_the_exact_frame(
    v1: tuple[float, float], v2: tuple[float, float], kind: str
) -> None:
    shots = shots_table([(0, 39)])
    for seed in range(5):
        points = polyline(40, (300.0, 600.0), v1, 17, v2, noise=0.3, seed=seed)
        events = ball_events(ball_track_from(points, 40), NO_PLAYERS, FPS, shots=shots)
        assert events["frame"].tolist() == [17], seed
        assert events["kind"].iloc[0] == kind


@pytest.mark.parametrize("velocity", [(8.0, 3.0), (0.0, 0.0)])
def test_steady_motion_and_jitter_give_no_event(velocity: tuple[float, float]) -> None:
    shots = shots_table([(0, 59)])
    for seed in range(5):
        points = polyline(60, (300.0, 600.0), velocity, 60, velocity, noise=0.5, seed=seed)
        events = ball_events(ball_track_from(points, 60), NO_PLAYERS, FPS, shots=shots)
        assert events.empty, seed
        assert tuple(events.columns) == BALL_EVENT_COLUMNS


def test_no_event_across_a_cut_or_a_reacquisition() -> None:
    before = polyline(20, (300.0, 600.0), (8.0, 0.0), 20, (8.0, 0.0))
    after = {f: (460.0, 600.0 - 8.0 * (f - 20)) for f in range(20, 40)}  # turns at 20
    track = ball_track_from({**before, **after}, 40)
    merged = ball_events(track, NO_PLAYERS, FPS, shots=shots_table([(0, 39)]))
    assert merged["frame"].tolist() == [20]  # in one shot the turn is an event ...
    cut = ball_events(track, NO_PLAYERS, FPS, shots=shots_table([(0, 19), (20, 39)]))
    assert cut.empty  # ... across a cut it is not
    jump = {**before, **{f: (1500.0, 200.0 - 8.0 * (f - 20)) for f in range(20, 40)}}
    shots = shots_table([(0, 39)])
    teleport = ball_events(ball_track_from(jump, 40), NO_PLAYERS, FPS, shots=shots)
    assert teleport.empty  # a 1000 px jump is another object, not a kick


def test_no_event_where_the_track_was_reacquired_on_an_object_of_another_size() -> None:
    """Every other frame detected, so windows hold two samples and fit any line. The
    15.5 px ball moves right; from frame 20 the track holds a 40 px object that goes on
    from where the ball would be and moves up. Only the size jump shows that these are two
    objects (no speed limit is broken), and no turn may be reported there."""
    first = {f: (300.0 + 20.0 * f, 600.0) for f in range(0, 20, 2)}
    second = {f: (300.0 + 20.0 * 20, 600.0 - 20.0 * (f - 20)) for f in range(20, 40, 2)}
    radius = {f: 20.0 for f in second}
    track = ball_track_from({**first, **second}, 40, radius)
    events = ball_events(track, NO_PLAYERS, FPS, shots=shots_table([(0, 39)]))
    assert events.empty
    same_size = ball_track_from({**first, **second}, 40)
    turn = ball_events(same_size, NO_PLAYERS, FPS, shots=shots_table([(0, 39)]))
    assert turn["frame"].tolist() == [20]  # with one size it is one ball turning


def test_nearest_player_is_measured_from_the_foot_in_box_heights() -> None:
    points = polyline(40, (300.0, 600.0), (8.0, 0.0), 20, (0.0, -8.0))
    u, v = points[20]
    rows = [
        box_row(20, 7, u - 20.0, v - 90.0, u + 20.0, v + 10.0, "right"),  # foot 10 px away
        box_row(20, 8, u - 70.0, v - 300.0, u - 10.0, v - 100.0, "left"),
        {**box_row(20, 9, u - 20.0, v - 95.0, u + 20.0, v + 5.0, None), "role": "referee"},
    ]
    shots = shots_table([(0, 39)])
    events = ball_events(ball_track_from(points, 40), pd.DataFrame(rows), FPS, shots=shots)
    event = events.iloc[0]
    assert event["nearest_track_id"] == 7  # the referee's foot is nearer, but he is no player
    assert event["nearest_team"] == "right"
    assert event["nearest_dist_box_h"] == pytest.approx(0.1)
    assert event["at_foot"]


def test_contact_params_find_a_small_touch_the_defaults_ignore() -> None:
    points = polyline(40, (300.0, 600.0), (2.0, 0.0), 20, (0.0, -2.0))  # 0.5 bh/s, 90 degrees
    track = ball_track_from(points, 40)
    shots = shots_table([(0, 39)])
    assert ball_events(track, NO_PLAYERS, FPS, shots=shots).empty
    contact = ball_events(track, NO_PLAYERS, FPS, shots=shots, params=CONTACT_PARAMS)
    assert contact["frame"].tolist() == [20]


def test_ball_events_requires_the_shots(scene: Scene, tracked: pd.DataFrame) -> None:
    with pytest.raises(TypeError):
        ball_events(tracked, scene.tracks, FPS)  # type: ignore[call-arg]
    with pytest.raises(BallInputError, match="status"):
        ball_events(tracked.assign(status="seen"), scene.tracks, FPS, shots=scene.shots)


def rect_cand(u: float, v: float, w: float, h: float, conf: float) -> list[float]:
    return [u - w / 2, v - h / 2, u + w / 2, v + h / 2, conf]


def _kick_out_scene(*, blurred_v: float = 548.0) -> tuple[Scene, dict[int, tuple[float, float]]]:
    """A round 15.5 px ball at a player's feet (frames 0-19), then kicked away at 40 px per
    frame (10 bh/s) as 35.5 x 15.5 px motion-blurred boxes at conf 0.85 (frames 20-29),
    while a round 10 px spare ball sits still far away at conf 0.55."""
    path = {f: (720.0 + 2.0 * f, 548.0) for f in range(20)}
    fast = {f: (758.0 - 40.0 * (f - 19), blurred_v + 3.0 * (f - 19)) for f in range(20, 30)}
    extra = {f: [rect_cand(*fast[f], 35.5, 15.5, 0.85)] for f in fast}
    for f in range(30):
        extra.setdefault(f, []).append(image_cand(200.0, 300.0, 10.0, 0.55))
    return image_scene(path, 30, extra=extra), fast


def test_motion_blurred_kick_is_tracked_instead_of_a_spare_ball() -> None:
    scene, fast = _kick_out_scene()
    tracked, gates = _run(scene)
    for f, (u, v) in fast.items():
        assert tracked["status"][f] == "detected", f
        assert tracked["u"][f] == pytest.approx(u)
        assert tracked["v"][f] == pytest.approx(v)
    blurred = gates[gates["frame"].isin(list(fast)) & np.isclose(gates["conf"], 0.85, atol=1e-6)]
    assert len(blurred) == len(fast)
    assert (blurred["reason"] == "ok").all()
    assert blurred["diameter_px"].to_numpy() == pytest.approx(15.5)  # the short side
    events = ball_events(tracked, scene.tracks, FPS, shots=scene.shots)
    assert 19 in events["frame"].tolist()  # the kick, at the last frame at the feet


def test_blur_exception_can_be_switched_off() -> None:
    scene, fast = _kick_out_scene()
    params = dataclasses.replace(DEFAULT_PARAMS, blur_min_conf=1.0)
    tracked, gates = _run(scene, params)
    assert not (tracked["status"][list(fast)] == "detected").any() or not np.allclose(
        tracked["u"][list(fast)].to_numpy(dtype=float), [u for u, _ in fast.values()]
    )
    blurred = gates[gates["frame"].isin(list(fast)) & np.isclose(gates["conf"], 0.85, atol=1e-6)]
    assert len(blurred) == len(fast)
    assert (blurred["reason"] == "aspect").all()


def test_still_or_edge_cut_elongated_boxes_are_not_blur() -> None:
    still = {f: [rect_cand(900.0, 700.0, 60.0, 25.0, 0.9)] for f in range(20)}
    _, gates = _run(image_scene({}, 20, extra=still))
    assert (gates["reason"] == "aspect").all()
    scene, fast = _kick_out_scene(blurred_v=1080.0 - 7.0)  # boxes reach the bottom edge
    _, gates = _run(scene)
    cut = gates[gates["frame"].isin(list(fast)) & np.isclose(gates["conf"], 0.85, atol=1e-6)]
    assert (cut["y2"] > 1078.0).any()
    assert (cut.loc[cut["y2"] > 1078.0, "reason"] == "aspect").all()


@pytest.mark.parametrize(
    ("long_sides", "moves"),
    [
        (10.0, False),
        (1.1 * DEFAULT_PARAMS.blur_max_step, False),
        (0.9 * DEFAULT_PARAMS.blur_max_step, True),
    ],
)
def test_a_distant_candidate_gives_a_still_box_no_motion(long_sides: float, moves: bool) -> None:
    """A still 60 x 25 px box at conf 0.9, and on every even frame an unrelated weak round
    candidate ``long_sides`` box lengths away along the box's axis. Only within
    blur_max_step can that be the box's own motion; 600 px away it is another object."""
    extra = {f: [rect_cand(900.0, 700.0, 60.0, 25.0, 0.9)] for f in range(20)}
    for f in range(0, 20, 2):
        extra[f].append(image_cand(900.0 + 60.0 * long_sides, 700.0, 15.5, 0.25))
    _, gates = _run(image_scene({}, 20, extra=extra))
    box = gates[np.isclose(gates["conf"], 0.9, atol=1e-6)]
    assert len(box) == 20
    odd = (box["frame"] % 2 == 1).to_numpy()
    assert (box.loc[~odd, "reason"] == "aspect").all()  # nothing on the frames beside
    assert (box.loc[odd, "reason"] == ("ok" if moves else "aspect")).all()


def test_a_candidate_across_a_cut_gives_a_box_no_motion() -> None:
    """Two 60 x 25 px boxes one long side apart on frames 9 and 10 move like a blurred ball
    within one shot; with a cut between the two frames they are two views, not motion."""
    extra = {
        9: [rect_cand(900.0, 700.0, 60.0, 25.0, 0.9)],
        10: [rect_cand(960.0, 700.0, 60.0, 25.0, 0.9)],
    }
    _, gates = _run(image_scene({}, 20, extra=extra))
    assert gates["frame"].tolist() == [9, 10]
    assert (gates["reason"] == "ok").all()
    _, gates = _run(image_scene({}, 20, bounds=[(0, 9), (10, 19)], extra=extra))
    assert gates["shot_id"].tolist() == [0, 1]
    assert (gates["reason"] == "aspect").all()


@pytest.mark.parametrize("long_sides", [0.5, 1.0, 0.95 * DEFAULT_PARAMS.blur_max_step])
def test_a_fast_ball_within_the_step_cap_stays_blurred(long_sides: float) -> None:
    """A blurred 60 x 25 px ball at conf 0.8 moving ``long_sides`` box lengths per frame
    (and 3 px down) on frames 5-14: every box passes as a blurred ball and is tracked."""
    step = 60.0 * long_sides
    fast = {f: (300.0 + step * (f - 5), 600.0 + 3.0 * (f - 5)) for f in range(5, 15)}
    extra = {f: [rect_cand(*fast[f], 60.0, 25.0, 0.8)] for f in fast}
    tracked, gates = _run(image_scene({}, 20, extra=extra))
    assert len(gates) == len(fast)
    assert (gates["reason"] == "ok").all()
    assert gates["diameter_px"].to_numpy() == pytest.approx(25.0)
    for f, (u, v) in fast.items():
        assert tracked["status"][f] == "detected", f
        assert tracked["u"][f] == pytest.approx(u)
        assert tracked["v"][f] == pytest.approx(v)
