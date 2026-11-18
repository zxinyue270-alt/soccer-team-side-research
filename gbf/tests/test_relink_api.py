"""Tests for gbf.relink.inputs and gbf.relink.api: the inputs of a shot, the method on a
track table, the shots and the identifiers.

Everything here is synthetic and seeded. The scene is a wide shot of 150 frames at 1280x720
and 25 fps from the synthetic main camera (gbf.geometry): eight players and a referee walk
slowly on the pitch; every detection box is the projection of a 1.8 m person, and every
embedding is the person's own direction plus the team's plus a little noise. Player "a" is
tracked as piece 5 until frame 59, is lost, and comes back as piece 9 from frame 90 on: the
one return every mode must find. The assertions are on decisions that are far from their
thresholds (the linker's similarity is about 1.8 where 1.1 is needed), never on float bits.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gbf import relink
from gbf.geometry import project, synthetic_broadcast_camera
from gbf.relink import api, inputs
from gbf.relink.cuts import SUB_STRIDE

WIDTH, HEIGHT, FPS, N_FRAMES, DIM = 1280, 720, 25.0, 150, 32

# person: (team, role, start (x, y) in metres, velocity in metres per frame)
PEOPLE: dict[str, tuple[str | None, str, tuple[float, float], tuple[float, float]]] = {
    "a": ("left", "player", (-10.0, -12.0), (0.03, 0.01)),
    "b": ("left", "player", (-4.0, 6.0), (0.02, -0.01)),
    "c": ("left", "player", (3.0, -4.0), (-0.02, 0.02)),
    "d": ("left", "player", (9.0, 4.0), (0.01, 0.02)),
    "e": ("right", "player", (-7.0, -2.0), (0.02, 0.0)),
    "f": ("right", "player", (2.0, 10.0), (0.01, -0.01)),
    "g": ("right", "player", (6.0, -10.0), (0.0, 0.02)),
    "h": ("right", "player", (13.0, -8.0), (-0.02, 0.01)),
    "r": (None, "referee", (1.0, -14.0), (0.02, 0.0)),
}

# (piece id, plugin id, person, first frame, last frame)
Stint = tuple[int, int, str, int, int]
STINTS: tuple[Stint, ...] = (
    (1, 1, "b", 0, 149),
    (2, 2, "c", 0, 149),
    (3, 3, "d", 0, 149),
    (5, 5, "a", 0, 59),
    (9, 9, "a", 90, 149),
    (11, 11, "e", 0, 149),
    (12, 12, "f", 0, 149),
    (13, 13, "g", 0, 149),
    (14, 14, "h", 0, 149),
    (30, 30, "r", 0, 149),
)
# The tracker switch: piece 2 follows "c" until frame 74 and "h" from frame 75 on, and the
# piece that followed "h" before (14) ends at frame 74.
SWITCH = (2, 75, "h")
SWITCH_STINTS: tuple[Stint, ...] = (
    *(stint for stint in STINTS if stint[0] != 14),
    (14, 14, "h", 0, 74),
)


@dataclasses.dataclass(frozen=True)
class Scene:
    tracks: pd.DataFrame
    calib: pd.DataFrame
    emb: np.ndarray
    shots: pd.DataFrame
    size: tuple[int, int]

    def relink(self, mode: str = "strict", **changes: Any) -> relink.RelinkResult:
        settings: dict[str, Any] = {"fps": FPS, "frame_size": self.size, "mode": mode}
        settings.update(changes)
        return relink.relink_tracks(self.tracks, self.calib, self.emb, self.shots, **settings)


def make_scene(
    stints: tuple[Stint, ...] = STINTS,
    *,
    seed: int = 0,
    scale: float = 1.0,
    switch: tuple[int, int, str] | None = None,
    first_frame: int = 0,
    id_offset: int = 0,
    walked: int = 0,
) -> Scene:
    """The scene as the refine stage's tables: tracks sorted by (frame, track_id), one
    calibration row per frame, the stored (float32, not normalised) embeddings of the track
    rows and the shot table. ``scale`` scales the picture (1.5 = 1920x1080); ``switch``
    (piece, frame, person) lets a piece follow another person from a frame on;
    ``first_frame`` and ``id_offset`` move the scene in time and in the ids, and ``walked``
    is the number of frames the people have already walked when it starts."""
    rng = np.random.default_rng(seed)
    width, height = round(WIDTH * scale), round(HEIGHT * scale)
    camera = synthetic_broadcast_camera(width, height)
    team_direction = {}
    for team in ("left", "right", None):
        vector = rng.normal(size=DIM)
        team_direction[team] = vector / np.linalg.norm(vector)
    prototype = {}
    for name, (team, _role, _start, _velocity) in PEOPLE.items():
        own = rng.normal(size=DIM)
        vector = team_direction[team] + 0.8 * own / np.linalg.norm(own)
        prototype[name] = vector / np.linalg.norm(vector)
    rows: list[dict[str, Any]] = []
    emb: list[np.ndarray] = []
    for piece, plugin, person, first, last in stints:
        for frame in range(first, last + 1):
            who = person
            if switch is not None and piece == switch[0] and frame >= switch[1]:
                who = switch[2]
            team, role, (x0, y0), (vx, vy) = PEOPLE[who]
            x, y = x0 + vx * (frame + walked), y0 + vy * (frame + walked)
            foot, _depth, foot_ok = project(camera, [x, y, 0.0])
            head, _depth, head_ok = project(camera, [x, y, -1.8])
            assert bool(foot_ok)
            assert bool(head_ok)
            tall = float(foot[1] - head[1])
            rows.append(
                {
                    "frame": frame + first_frame,
                    "track_id": piece + id_offset,
                    "x1": float(foot[0]) - 0.2 * tall,
                    "y1": float(head[1]),
                    "x2": float(foot[0]) + 0.2 * tall,
                    "y2": float(foot[1]),
                    "conf": 0.9,
                    "role": role,
                    "team": team,
                    "pitch_usable": True,
                    "pitch_x": x,
                    "pitch_y": y,
                    "in_pitch": True,
                    "source_track_id": plugin + id_offset,
                }
            )
            emb.append(3.0 * (prototype[who] + 0.05 * rng.normal(size=DIM)))
    tracks = pd.DataFrame(rows)
    order = np.lexsort((tracks["track_id"].to_numpy(), tracks["frame"].to_numpy()))
    tracks = tracks.iloc[order].reset_index(drop=True)
    tracks["team"] = tracks["team"].astype("string")  # a missing team reads as NA, as in parquet
    calib = pd.DataFrame(
        {
            "frame_index": np.arange(first_frame, first_frame + N_FRAMES),
            "K": [camera.K.ravel().tolist()] * N_FRAMES,
            "R": [camera.R.ravel().tolist()] * N_FRAMES,
            "C": [camera.C.tolist()] * N_FRAMES,
        }
    )
    shots = pd.DataFrame(
        {"shot_id": [0], "start_frame": [first_frame], "end_frame": [first_frame + N_FRAMES - 1]}
    )
    return Scene(tracks, calib, np.asarray(emb, dtype=np.float32)[order], shots, (width, height))


def join(*scenes: Scene) -> Scene:
    """Several scenes as one table with one shot each."""
    shots = pd.concat([scene.shots for scene in scenes], ignore_index=True)
    shots["shot_id"] = np.arange(len(shots))
    return Scene(
        pd.concat([scene.tracks for scene in scenes], ignore_index=True),
        pd.concat([scene.calib for scene in scenes], ignore_index=True),
        np.concatenate([scene.emb for scene in scenes]),
        shots,
        scenes[0].size,
    )


@pytest.fixture(scope="module")
def base() -> Scene:
    return make_scene()


@pytest.fixture(scope="module")
def switched() -> Scene:
    return make_scene(SWITCH_STINTS, switch=SWITCH)


# --------------------------------------------------------------------------- inputs


def test_unit_embeddings_are_float64_unit_rows_and_unusable_rows_are_nan() -> None:
    rng = np.random.default_rng(1)
    raw = (4.0 * rng.normal(size=(6, 8))).astype(np.float32)
    raw[2, 3] = np.nan
    raw[4] = 0.0
    raw[5, 0] = np.inf
    unit = inputs.unit_embeddings(raw)
    assert unit.dtype == np.float64
    assert unit.shape == raw.shape
    good = [0, 1, 3]
    np.testing.assert_allclose(np.linalg.norm(unit[good], axis=1), 1.0, rtol=0, atol=1e-12)
    wide = raw[good].astype(np.float64)
    expected = wide / np.linalg.norm(wide, axis=1, keepdims=True)
    np.testing.assert_allclose(unit[good], expected, rtol=0, atol=1e-15)
    assert np.isnan(unit[[2, 4, 5]]).all()
    assert raw.dtype == np.float32  # the input is not changed
    assert raw[4, 0] == 0.0
    with pytest.raises(ValueError, match="embeddings must be"):
        inputs.unit_embeddings(np.zeros(4, np.float32))


def test_has_embedding_is_the_rule_of_the_unit_embeddings_block_by_block() -> None:
    block = inputs._HAS_BLOCK_ROWS
    rng = np.random.default_rng(3)
    raw = (4.0 * rng.normal(size=(block + 37, 4))).astype(np.float32)
    without = [0, 5, block - 1, block, len(raw) - 1]
    raw[0, 1] = np.nan
    raw[5] = 0.0  # a vector of length zero has no direction
    raw[block - 1, 3] = np.inf
    raw[block] = np.nan
    raw[len(raw) - 1] = 0.0
    has = inputs.has_embedding(raw)
    assert has.dtype == np.bool_
    assert np.flatnonzero(~has).tolist() == without
    assert has.tolist() == np.isfinite(inputs.unit_embeddings(raw)).all(axis=1).tolist()
    assert inputs.has_embedding(raw[:0]).tolist() == []
    with pytest.raises(ValueError, match="embeddings must be"):
        inputs.has_embedding(np.zeros(4, np.float32))


def _boxes(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of two frames at 1280x720 times ``scale``: (frame, box, has an embedding)."""
    rows = [
        # frame 0
        (0, [100.0, 100.0, 130.0, 160.0], True),  # clean
        (0, [300.0, 100.0, 330.0, 139.0], True),  # 39 px: too short
        (0, [400.0, 100.0, 430.0, 140.0], True),  # 40 px: tall enough
        (0, [1.0, 100.0, 31.0, 160.0], True),  # at the left border
        (0, [1.5, 300.0, 31.5, 360.0], True),  # just inside
        (0, [1248.0, 100.0, 1278.0, 160.0], True),  # at the right border (1280 - 2)
        (0, [1247.0, 300.0, 1277.5, 360.0], True),  # just inside
        (0, [600.0, 0.5, 630.0, 60.0], True),  # at the top
        (0, [700.0, 660.0, 730.0, 718.0], True),  # at the bottom (720 - 2)
        (0, [800.0, 100.0, 830.0, 160.0], False),  # no embedding
        (0, [900.0, 100.0, 930.0, 160.0], True),  # overlapped by the next one
        (0, [925.0, 150.0, 955.0, 210.0], True),
        # frame 1: the same place as the overlapping pair, alone
        (1, [900.0, 100.0, 930.0, 160.0], True),
        (1, [1000.0, 100.0, 1030.0, 160.0], True),  # touches the next one: no overlap
        (1, [1030.0, 100.0, 1060.0, 160.0], True),
    ]
    frame = np.array([r[0] for r in rows], dtype=np.int64)
    box = scale * np.array([r[1] for r in rows], dtype=np.float64)
    return frame, box, np.array([r[2] for r in rows], dtype=bool)


def test_clean_rows_at_720_lines_and_scaled_to_1080() -> None:
    expected = [
        *(True, False, True, False, True, False, True, False, False, False, False, False),
        *(True, True, True),
    ]
    frame, box, has = _boxes(1.0)
    assert inputs.clean_rows(frame, box, has, 1280.0, 720.0).tolist() == expected
    # the border test uses width - 1 - e: at 1920x1080 the right and bottom limits are not
    # the 720-line ones times 1.5, so the two rows at those borders are built for each size
    frame, box, has = _boxes(1.5)
    box[5] = [1880.0, 150.0, 1917.5, 240.0]  # x2 = 1920 - 1 - 1.5
    box[6] = [1870.0, 450.0, 1917.0, 540.0]
    box[8] = [1050.0, 990.0, 1095.0, 1077.5]  # y2 = 1080 - 1 - 1.5
    assert inputs.clean_rows(frame, box, has, 1920.0, 1080.0).tolist() == expected
    # an overlap with a row of any kind counts, whatever the row order
    order = np.random.default_rng(2).permutation(len(frame))
    frame, box, has = _boxes(1.0)
    shuffled = inputs.clean_rows(frame[order], box[order], has[order], 1280.0, 720.0)
    assert shuffled.tolist() == [expected[i] for i in order]
    assert inputs.clean_rows(frame[:0], box[:0], has[:0], 1280.0, 720.0).tolist() == []
    with pytest.raises(ValueError, match="clean_rows"):
        inputs.clean_rows(frame, box[:, :3], has, 1280.0, 720.0)


def test_cameras_are_the_projection_matrices_of_the_shots_frames() -> None:
    camera = synthetic_broadcast_camera(WIDTH, HEIGHT)
    K, R, C = camera.K.ravel().tolist(), camera.R.ravel().tolist(), camera.C.tolist()
    calib = pd.DataFrame(
        {
            "frame_index": [12, 10, 11, 13, 14, 20],
            "K": [K, K, None, K, [float("nan")] * 9, K],
            "R": [R, R, None, R, R, R],
            "C": [C, C, None, C, C, C],
        }
    )
    P = inputs.cameras(calib, 10, 6)
    assert P.shape == (6, 3, 4)
    assert np.isfinite(P[[0, 2, 3]]).all()
    assert np.isnan(P[[1, 4, 5]]).all()  # no calibration, not finite, no row
    expected = camera.K @ camera.R @ np.hstack([np.eye(3), -camera.C[:, None]])
    np.testing.assert_allclose(P[0], expected, rtol=1e-12)
    point = np.array([5.0, -3.0, 0.0])
    u, v, w = P[0] @ np.append(point, 1.0)
    uv, _depth, valid = project(camera, point)
    assert bool(valid)
    np.testing.assert_allclose([u / w, v / w], uv, rtol=0, atol=1e-6)


def test_build_clip_takes_the_shots_rows_and_counts_frames_from_its_start() -> None:
    scene = make_scene(first_frame=40)
    tracks = scene.tracks
    inp, rows = inputs.build_clip(
        tracks, scene.calib, scene.emb, start_frame=60, end_frame=99, fps=FPS, frame_size=scene.size
    )
    inside = (tracks["frame"] >= 60) & (tracks["frame"] <= 99)
    assert rows.tolist() == np.flatnonzero(inside.to_numpy()).tolist()
    assert inp.n_frames == 40
    assert inp.frame.tolist() == (tracks["frame"].to_numpy()[rows] - 60).tolist()
    assert inp.piece.tolist() == tracks["track_id"].to_numpy()[rows].tolist()
    assert inp.source.dtype == np.int64
    assert inp.cam_P.shape == (40, 3, 4)
    assert np.isfinite(inp.cam_P).all()
    assert (inp.fps, inp.width, inp.height, inp.px_scale) == (25.0, 1280.0, 720.0, 1.0)
    assert set(inp.team.tolist()) == {"left", "right", None}  # a missing label is None
    assert set(inp.role.tolist()) == {"player", "referee"}
    np.testing.assert_allclose(np.linalg.norm(inp.emb, axis=1), 1.0, rtol=0, atol=1e-12)
    assert inp.clean.all()  # nobody overlaps or touches a border in this scene
    assert inp.pitch_xy.shape == (len(rows), 2)

    twice = pd.concat([tracks, tracks.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="more than one row"):
        inputs.build_clip(
            twice,
            scene.calib,
            np.concatenate([scene.emb, scene.emb[:1]]),
            start_frame=40,
            end_frame=189,
            fps=FPS,
            frame_size=scene.size,
        )
    with pytest.raises(ValueError, match="relink inputs") as caught:
        inputs.build_clip(
            tracks.drop(columns=["source_track_id", "in_pitch"]),
            scene.calib.drop(columns="K"),
            scene.emb[:-1],
            start_frame=9,
            end_frame=3,
            fps=0.0,
            frame_size=scene.size,
        )
    message = str(caught.value)
    for part in ("source_track_id", "in_pitch", "column K", "one row per track row", "no frame"):
        assert part in message
    assert "fps must be positive" in message


# --------------------------------------------------------------------------- the method


def test_strict_joins_the_one_certain_return(base: Scene) -> None:
    result = base.relink("strict")
    assert result.mode == "strict"
    assert result.cuts == ()
    assert result.core_links == ((5, 9),)
    assert result.leaf_links == ()
    assert result.links == ((5, 9),)
    assert result.refused == ()
    piece = base.tracks["track_id"].to_numpy()
    assert result.sub_id.tolist() == piece.tolist()
    assert result.track_id.tolist() == np.where(piece == 9, 5, piece).tolist()

    report = result.report
    assert (report["mode"], report["applied"], report["reason"]) == ("strict", True, None)
    assert report["setting"] == {"margin": 0.6, "leaves": False}
    assert report["params"] == relink.DEFAULT_PARAMS.as_dict()
    assert (report["fps"], report["fitted_fps"], report["px_scale"]) == (25.0, 25.0, 1.0)
    assert report["frame_size"] == [1280, 720]
    assert report["counts"] == {
        "shots": 1,
        "shots_relinked": 1,
        "pieces": 10,
        "cuts": 0,
        "sub_pieces": 10,
        "solid": 10,
        "short": 0,
        "core_links": 1,
        "leaf_links": 0,
        "identities": 9,
        "rows_renamed": 60,
        "rows_without_embedding": 0,
        "links_refused_same_frame": 0,
    }
    assert (report["cuts"], report["links"], report["leaf_links"]) == ([], [[5, 9]], [])
    assert report["identities"] == [{"track_id": 5, "pieces": [5, 9]}]
    (merge,) = report["merges"]
    assert (merge["g"], merge["h"]) == ([5], [9])
    assert merge["s"] > 1.5  # appearance about 1.0 plus a position bonus of about 0.8
    assert merge["bonus"] > 0.5


def test_relink_clip_is_the_method_of_one_shot(base: Scene) -> None:
    inp, _rows = inputs.build_clip(
        base.tracks,
        base.calib,
        base.emb,
        start_frame=0,
        end_frame=N_FRAMES - 1,
        fps=FPS,
        frame_size=base.size,
    )
    log: list[dict[str, Any]] = []
    links = relink.relink_clip(inp, "strict", log=log)
    assert links == api.ClipLinks(cuts=(), core_links=((5, 9),), leaf_links=())
    assert len(log) == 1
    assert log[0]["stage"] == 1
    for mode in ("off", "loose"):
        with pytest.raises(relink.RelinkError, match="mode must be one of"):
            relink.relink_clip(inp, mode)


def test_the_result_does_not_depend_on_the_row_order(base: Scene) -> None:
    order = np.random.default_rng(5).permutation(len(base.tracks))
    shuffled = Scene(
        base.tracks.iloc[order].reset_index(drop=True),
        base.calib.iloc[::-1].reset_index(drop=True),
        base.emb[order],
        base.shots,
        base.size,
    )
    result, expected = shuffled.relink(), base.relink()
    assert (result.cuts, result.links) == (expected.cuts, expected.links)
    assert result.track_id.tolist() == expected.track_id[order].tolist()


def test_the_same_scene_at_1080_lines_gives_the_same_links() -> None:
    result = make_scene(scale=1.5).relink()
    assert (result.cuts, result.links) == ((), ((5, 9),))
    assert result.report["frame_size"] == [1920, 1080]
    assert result.report["px_scale"] == 1.5


def test_a_tracker_switch_is_cut_and_its_tail_joins_its_own_person(switched: Scene) -> None:
    result = switched.relink()
    tail = 2 + SUB_STRIDE
    assert result.cuts == ((2, 75),)
    assert set(result.links) == {(5, 9), (14, tail)}
    tracks = switched.tracks
    piece, frame = tracks["track_id"].to_numpy(), tracks["frame"].to_numpy()
    cut_off = (piece == 2) & (frame >= 75)
    assert result.sub_id[cut_off].tolist() == [tail] * 75
    assert result.sub_id[~cut_off].tolist() == piece[~cut_off].tolist()
    # an identity takes the id of its earliest member: 14 starts before the cut-off tail
    expected = np.where(cut_off, 14, np.where(piece == 9, 5, piece))
    assert result.track_id.tolist() == expected.tolist()
    assert result.report["identities"] == [
        {"track_id": 5, "pieces": [5, 9]},
        {"track_id": 14, "pieces": [14, tail]},
    ]
    counts = result.report["counts"]
    assert (counts["cuts"], counts["sub_pieces"], counts["rows_renamed"]) == (1, 11, 135)


def test_a_cut_off_tail_without_a_partner_keeps_its_sub_piece_id() -> None:
    stints = tuple(stint for stint in STINTS if stint[0] != 14)  # "h" was not tracked before
    scene = make_scene(stints, switch=SWITCH)
    result = scene.relink()
    assert result.cuts == ((2, 75),)
    assert result.links == ((5, 9),)
    piece, frame = scene.tracks["track_id"].to_numpy(), scene.tracks["frame"].to_numpy()
    tail = (piece == 2) & (frame >= 75)
    assert set(result.track_id[tail].tolist()) == {2 + SUB_STRIDE}
    assert set(result.track_id[(piece == 2) & ~tail].tolist()) == {2}


def test_without_embeddings_nothing_is_cut_or_joined(base: Scene) -> None:
    empty = Scene(base.tracks, base.calib, np.full_like(base.emb, np.nan), base.shots, base.size)
    result = empty.relink()
    assert (result.cuts, result.links) == ((), ())
    assert result.track_id.tolist() == base.tracks["track_id"].tolist()
    assert result.report["counts"]["rows_without_embedding"] == len(base.tracks)
    assert result.report["counts"]["rows_renamed"] == 0


def test_a_piece_without_embeddings_is_neither_cut_nor_joined_and_its_rows_are_counted(
    base: Scene, switched: Scene
) -> None:
    # the piece of the return has vectors of length zero, which are no embeddings either
    emb = base.emb.copy()
    returned = (base.tracks["track_id"] == 9).to_numpy()
    emb[returned] = 0.0
    emb[np.flatnonzero(~returned)[:3], 0] = np.nan  # and three rows of other pieces have none
    result = Scene(base.tracks, base.calib, emb, base.shots, base.size).relink()
    assert (result.cuts, result.links) == ((), ())
    assert result.track_id.tolist() == base.tracks["track_id"].tolist()
    assert result.report["counts"]["rows_without_embedding"] == int(returned.sum()) + 3
    assert base.relink().report["counts"]["rows_without_embedding"] == 0
    # the piece with the tracker switch has no embedding: it is not cut, so its tail cannot
    # be joined to the piece that followed that person before; the return is still found
    emb = switched.emb.copy()
    emb[(switched.tracks["track_id"] == 2).to_numpy()] = np.nan
    result = Scene(switched.tracks, switched.calib, emb, switched.shots, switched.size).relink()
    assert result.cuts == ()
    assert result.links == ((5, 9),)


# --------------------------------------------------------------------------- shots


def test_a_track_outside_every_shot_is_left_alone_and_one_half_outside_is_refused(
    base: Scene,
) -> None:
    # the shot table ends at frame 149. Track 40 lies after it, with the rows and embeddings
    # of track 1: inside a shot it would be a perfect partner
    rows = base.tracks[base.tracks["track_id"] == 1].iloc[:5]
    beside = rows.assign(frame=np.arange(200, 205), track_id=40, source_track_id=40)
    tracks = pd.concat([base.tracks, beside], ignore_index=True)
    emb = np.concatenate([base.emb, base.emb[rows.index.to_numpy()]])
    result = Scene(tracks, base.calib, emb, base.shots, base.size).relink()
    assert result.links == ((5, 9),)
    assert result.track_id[-5:].tolist() == [40] * 5
    assert result.sub_id[-5:].tolist() == [40] * 5
    counts = result.report["counts"]
    assert counts["pieces"] == base.relink().report["counts"]["pieces"] + 1
    assert (counts["shots"], counts["shots_relinked"]) == (1, 1)
    renamed = relink.apply_ids(tracks, result)
    assert renamed["track_id"].iloc[-5:].tolist() == [40] * 5
    # a track with rows in the shot and rows beside it was not split at the shots: refused
    half = pd.concat([base.tracks, rows.assign(frame=np.arange(200, 205))], ignore_index=True)
    with pytest.raises(relink.RelinkError, match="beside the shots"):
        Scene(half, base.calib, emb, base.shots, base.size).relink()


def test_every_shot_is_relinked_on_its_own_and_cut_frames_are_canonical(switched: Scene) -> None:
    second = make_scene(SWITCH_STINTS, switch=SWITCH, seed=0, first_frame=150, id_offset=100)
    lone = pd.DataFrame(
        {
            **{name: second.tracks[name].iloc[:5].to_numpy() for name in second.tracks.columns},
            "frame": np.arange(300, 305),
            "track_id": 300,
            "source_track_id": 300,
        }
    )
    two = join(switched, second)
    scene = Scene(
        pd.concat([two.tracks, lone], ignore_index=True),
        two.calib,
        np.concatenate([two.emb, second.emb[:5]]),
        pd.concat(
            [
                two.shots,
                pd.DataFrame(
                    {"shot_id": [2, 3], "start_frame": [300, 305], "end_frame": [304, 305]}
                ),
            ],
            ignore_index=True,
        ),
        two.size,
    )
    result = scene.relink()
    # the second shot holds the same people with the same embeddings under other ids: every
    # piece of it would be a perfect partner for a piece of the first shot
    assert result.cuts == ((2, 75), (102, 225))
    assert result.core_links == ((5, 9), (14, 2 + SUB_STRIDE), (105, 109), (114, 102 + SUB_STRIDE))
    counts = result.report["counts"]
    assert (counts["shots"], counts["shots_relinked"]) == (4, 2)  # one piece; no row at all
    piece = scene.tracks["track_id"].to_numpy()
    first_shot = scene.tracks["frame"].to_numpy() < 150
    assert set(result.track_id[first_shot].tolist()) <= set(piece[first_shot].tolist())
    assert set(result.track_id[piece == 300].tolist()) == {300}

    alone = switched.relink()
    assert result.track_id[first_shot].tolist() == alone.track_id.tolist()


def test_no_link_joins_two_shots_even_where_one_shot_would_be_joined(base: Scene) -> None:
    # the second shot goes on where the first ends: the same people one frame later, at
    # the same places, under other ids
    second = make_scene(first_frame=N_FRAMES, id_offset=100, walked=N_FRAMES)
    two = join(base, second)
    apart = two.relink()
    assert apart.links == ((5, 9), (105, 109))
    one_shot = pd.DataFrame({"shot_id": [0], "start_frame": [0], "end_frame": [2 * N_FRAMES - 1]})
    across = Scene(two.tracks, two.calib, two.emb, one_shot, two.size).relink()
    assert {(1, 101), (2, 102), (9, 105)} <= set(across.links)


# --------------------------------------------------------------------------- identifiers


def test_apply_ids_rewrites_track_id_and_keeps_the_old_one_in_piece_id(switched: Scene) -> None:
    tracks = switched.tracks
    before = tracks.copy()
    result = switched.relink()
    renamed = relink.apply_ids(tracks, result)
    pd.testing.assert_frame_equal(tracks, before)  # the input is not changed
    assert list(renamed.columns) == [*tracks.columns, "piece_id"]
    assert renamed["piece_id"].dtype == np.int64
    assert renamed["track_id"].dtype == np.int64
    assert renamed["piece_id"].tolist() == tracks["track_id"].tolist()
    assert renamed["track_id"].tolist() == result.track_id.tolist()
    for name in tracks.columns:
        if name != "track_id":
            pd.testing.assert_series_equal(renamed[name], tracks[name])
    assert not renamed.duplicated(["frame", "track_id"]).any()
    # the plugin's id stays where it was
    assert renamed["source_track_id"].tolist() == tracks["source_track_id"].tolist()

    skipped = relink.apply_ids(tracks, None)  # a mode that is on but did not run
    assert list(skipped.columns) == [*tracks.columns, "piece_id"]
    assert skipped["track_id"].tolist() == tracks["track_id"].tolist()
    assert skipped["piece_id"].tolist() == tracks["track_id"].tolist()


def test_rows_without_a_track_keep_their_id(base: Scene) -> None:
    tracks = base.tracks.copy()
    loose = (tracks["track_id"] == 3) & (tracks["frame"] % 7 == 0)
    tracks.loc[loose, "track_id"] = -1
    scene = Scene(tracks, base.calib, base.emb, base.shots, base.size)
    result = scene.relink()
    assert result.links == ((5, 9),)
    assert (result.track_id[loose.to_numpy()] == -1).all()
    renamed = relink.apply_ids(tracks, result)
    assert (renamed.loc[loose, ["track_id", "piece_id"]] == -1).all().all()


def test_apply_ids_refuses_a_result_that_does_not_fit_the_table(base: Scene) -> None:
    result = base.relink()
    tracks = base.tracks
    with pytest.raises(relink.RelinkError, match="already has a piece_id"):
        relink.apply_ids(relink.apply_ids(tracks, result), result)
    with pytest.raises(relink.RelinkError, match="expected"):
        relink.apply_ids(tracks.iloc[:-1], result)
    with pytest.raises(relink.RelinkError, match="column track_id is missing"):
        relink.apply_ids(tracks.drop(columns="track_id"), result)
    # two rows of one frame with one id: pieces 1 and 2 are both seen in every frame
    clash = np.where(result.track_id == 2, 1, result.track_id)
    with pytest.raises(relink.RelinkError, match="would repeat"):
        relink.apply_ids(tracks, dataclasses.replace(result, track_id=clash))
    untracked = result.track_id.copy()
    untracked[0] = -1
    with pytest.raises(relink.RelinkError, match="does not belong"):
        relink.apply_ids(tracks, dataclasses.replace(result, track_id=untracked))


def test_final_ids_name_an_identity_after_its_earliest_member_and_refuse_a_shared_frame() -> None:
    sub = np.array([7, 7, 4, 4, 9, 9, -1, 12, 3], dtype=np.int64)
    frame = np.array([5, 6, 6, 7, 20, 21, 5, 30, 5], dtype=np.int64)
    #   7: frames 5-6    4: frames 6-7    9: frames 20-21    12: frame 30    3: frame 5
    final, refused, identities = api._final_ids(sub, frame, ((7, 4), (4, 9), (9, 12), (7, 9)))
    assert refused == ((7, 4), (7, 9))  # 7 and 4 share frame 6; then 7 and {4, 9, 12} do
    assert final.tolist() == [7, 7, 4, 4, 4, 4, -1, 4, 3]
    assert identities == [{"track_id": 4, "pieces": [4, 9, 12]}]
    # equal first frames: the smaller id names the identity; a repeated link changes nothing
    final, refused, identities = api._final_ids(
        np.array([7, 3, 9], dtype=np.int64), np.array([5, 5, 8], dtype=np.int64), ((9, 7), (7, 9))
    )
    assert (final.tolist(), refused) == ([7, 3, 7], ())
    with pytest.raises(relink.RelinkError, match="names a sub-piece without rows"):
        api._final_ids(sub, frame, ((7, 99),))


def test_unusable_inputs_are_refused_with_every_problem(base: Scene) -> None:
    for mode in ("off", "balanced"):
        with pytest.raises(relink.RelinkError, match="mode must be one of"):
            base.relink(mode)

    big = base.tracks.copy()
    big.loc[big["track_id"] == 14, "track_id"] = SUB_STRIDE
    with pytest.raises(relink.RelinkError) as caught:
        relink.relink_tracks(
            big.drop(columns="pitch_x"),
            base.calib.drop(columns="C"),
            base.emb[:-3],
            base.shots.drop(columns="end_frame"),
            fps=-1.0,
            frame_size=(0, 720),
            mode="strict",
        )
    message = str(caught.value)
    for part in (
        "tracks: column pitch_x is missing",
        "calib: column C is missing",
        "shots: column end_frame is missing",
        "embeddings: expected one row per track row",
        f"track ids must be below {SUB_STRIDE}",
        "fps must be a positive number",
        "frame_size must be two positive numbers",
    ):
        assert part in message

    # a track with rows in two shots: relink needs the ids the refine stage split at cuts
    halves = pd.DataFrame({"shot_id": [0, 1], "start_frame": [0, 75], "end_frame": [74, 149]})
    with pytest.raises(relink.RelinkError, match="rows in more than one shot"):
        relink.relink_tracks(
            base.tracks, base.calib, base.emb, halves, fps=FPS, frame_size=base.size, mode="strict"
        )
    overlapping = pd.DataFrame({"shot_id": [0, 1], "start_frame": [0, 70], "end_frame": [74, 149]})
    with pytest.raises(relink.RelinkError, match="shots overlap"):
        relink.relink_tracks(
            base.tracks,
            base.calib,
            base.emb,
            overlapping,
            fps=FPS,
            frame_size=base.size,
            mode="strict",
        )
    twice = pd.concat([base.tracks, base.tracks.iloc[[3]]], ignore_index=True)
    with pytest.raises(relink.RelinkError, match="more than one row"):
        relink.relink_tracks(
            twice,
            base.calib,
            np.concatenate([base.emb, base.emb[[3]]]),
            base.shots,
            fps=FPS,
            frame_size=base.size,
            mode="strict",
        )


def test_fps_may_be_any_real_number_but_not_a_bool_a_string_or_infinite(base: Scene) -> None:
    def problems(fps: Any) -> list[str]:
        return api._input_problems(base.tracks, base.calib, base.emb, base.shots, fps, base.size)

    for fps in (25, 25.0, Fraction(30000, 1001), np.float32(25.0), np.float64(50.0), np.int64(25)):
        assert problems(fps) == [], repr(fps)
    for fps in (True, "25", None, float("nan"), float("inf"), 0, -25.0, Fraction(0, 1)):
        assert problems(fps) == [f"fps must be a positive number, got {fps!r}"], repr(fps)
    result = base.relink(fps=Fraction(25, 1))
    assert result.links == ((5, 9),)
    assert result.report["fps"] == 25.0
    assert type(result.report["fps"]) is float


def test_modes_settings_and_skip_sections() -> None:
    assert relink.MODES[:2] == ("off", "strict")
    assert relink.setting("strict") == {"margin": 0.6, "leaves": False}
    assert relink.setting("strict") == {
        "margin": relink.DEFAULT_PARAMS.link.margin,
        "leaves": False,
    }
    with pytest.raises(relink.RelinkError):
        relink.setting("off")
    assert relink.skipped("strict", "no_embeddings") == {
        "mode": "strict",
        "applied": False,
        "reason": "no_embeddings",
    }
    with pytest.raises(relink.RelinkError, match="unknown skip reason"):
        relink.skipped("strict", "tired")
    params = relink.DEFAULT_PARAMS.as_dict()
    assert sorted(params) == ["appear", "cuts", "link", "space"]
    assert params["cuts"]["p_cut"] == 0.7
    assert params["link"]["n_solid"] == 5
    assert (
        relink.RelinkParams(
            cuts=relink.cuts.CutParams(**params["cuts"]),
            space=relink.space.SpaceParams(**params["space"]),
        ).as_dict()
        == params
    )


# --------------------------------------------------------------------------- recall

SHORT_STINTS: tuple[Stint, ...] = (*STINTS, (20, 20, "a", 70, 72))
"""Player "a" is also seen for three frames while lost: a piece too short to be solid."""


def test_recall_attaches_the_short_piece_as_a_leaf_and_strict_leaves_it_alone() -> None:
    scene = make_scene(SHORT_STINTS)
    strict = scene.relink("strict")
    assert strict.links == ((5, 9),)
    assert (strict.report["counts"]["solid"], strict.report["counts"]["short"]) == (10, 1)

    recall = scene.relink("recall")
    assert relink.MODES == ("off", "strict", "recall")
    assert recall.core_links == ((5, 9),)
    assert len(recall.leaf_links) == 1
    assert recall.leaf_links[0] in {(5, 20), (20, 9)}  # joined to a solid neighbour in time
    assert recall.links == recall.core_links + recall.leaf_links
    assert recall.refused == ()
    report = recall.report
    assert report["setting"] == {"margin": 0.2, "leaves": True}
    assert report["setting"] == relink.setting("recall")
    assert report["leaf_links"] == [list(recall.leaf_links[0])]
    assert report["identities"] == [{"track_id": 5, "pieces": [5, 20, 9]}]
    assert (report["counts"]["core_links"], report["counts"]["leaf_links"]) == (1, 1)
    piece = scene.tracks["track_id"].to_numpy()
    expected = np.where(np.isin(piece, [9, 20]), 5, piece)
    assert recall.track_id.tolist() == expected.tolist()
    stages = [merge["stage"] for merge in report["merges"]]
    assert stages == [1, 2]


def test_a_short_piece_without_embeddings_is_not_attached_as_a_leaf() -> None:
    scene = make_scene(SHORT_STINTS)
    emb = scene.emb.copy()
    short = (scene.tracks["track_id"] == 20).to_numpy()
    emb[short] = 0.0  # vectors of length zero: no embeddings
    result = Scene(scene.tracks, scene.calib, emb, scene.shots, scene.size).relink("recall")
    assert result.core_links == ((5, 9),)
    assert result.leaf_links == ()
    assert result.track_id[short].tolist() == [20] * 3
    assert result.report["counts"]["rows_without_embedding"] == 3
    assert result.report["counts"]["short"] == 1


def test_recall_finds_the_strict_links_of_the_plain_scenes(base: Scene, switched: Scene) -> None:
    assert base.relink("recall").links == ((5, 9),)
    result = switched.relink("recall")
    assert result.cuts == ((2, 75),)
    assert set(result.links) == {(5, 9), (14, 2 + SUB_STRIDE)}
    assert result.leaf_links == ()


@pytest.mark.parametrize("mode", ["strict", "recall"])
def test_occlusion_feature_change_without_independent_evidence_keeps_the_identity(
    mode: str,
) -> None:
    scene = make_scene()
    tracks, emb = scene.tracks.copy(), scene.emb.copy()
    row = tracks.index[(tracks.track_id == 9) & (tracks.frame == 90)][0]
    other = tracks.index[(tracks.track_id == 11) & (tracks.frame == 90)][0]
    tracks.loc[row, ["x1", "y1", "x2", "y2"]] = tracks.loc[other, ["x1", "y1", "x2", "y2"]]
    emb[row] = emb[other]  # foreground occluder dominates ReID; this is not proof of a swap
    result = Scene(tracks, scene.calib, emb, scene.shots, scene.size).relink(mode)
    assert result.track_id[row] == 5
    assert result.report["isolation"]["rows"] == 0


@pytest.mark.parametrize("mode", ["strict", "recall"])
def test_conflicting_overlap_rows_cannot_contaminate_a_returned_identity(mode: str) -> None:
    scene = make_scene()
    tracks, emb = scene.tracks.copy(), scene.emb.copy()
    bad = tracks.index[(tracks.track_id == 9) & tracks.frame.isin([90, 91])].to_numpy()
    colours = _shirt_lab(scene)
    for row in bad:
        other = tracks.index[(tracks.track_id == 11) & (tracks.frame == tracks.at[row, "frame"])][0]
        tracks.loc[row, ["x1", "y1", "x2", "y2"]] = tracks.loc[other, ["x1", "y1", "x2", "y2"]]
        emb[row] = emb[other]  # the tracker momentarily put the other person in this id
        colours[row] = colours[other]
    result = Scene(tracks, scene.calib, emb, scene.shots, scene.size).relink(
        mode, shirt_lab=colours
    )
    good = (tracks.track_id == 9) & ~tracks.index.isin(bad)
    assert set(result.track_id[good]) == {5}  # the actual return still works
    assert len(set(result.track_id[bad])) == len(bad)
    assert not set(result.track_id[bad]) & set(result.track_id[~tracks.index.isin(bad)])
    assert result.report["isolation"]["rows"] == len(bad)
    renamed = relink.apply_ids(tracks, result)
    pd.testing.assert_frame_equal(
        renamed.drop(columns=["track_id", "piece_id"]), tracks.drop(columns="track_id")
    )


@pytest.mark.parametrize("missing", [False, True])
def test_overlap_alone_or_missing_features_do_not_trigger_isolation(missing: bool) -> None:
    scene = make_scene()
    tracks, emb = scene.tracks.copy(), scene.emb.copy()
    row = tracks.index[(tracks.track_id == 9) & (tracks.frame == 90)][0]
    other = tracks.index[(tracks.track_id == 11) & (tracks.frame == 90)][0]
    tracks.loc[row, ["x1", "y1", "x2", "y2"]] = tracks.loc[other, ["x1", "y1", "x2", "y2"]]
    if missing:
        emb[row] = np.nan
    result = Scene(tracks, scene.calib, emb, scene.shots, scene.size).relink()
    assert result.track_id[row] == 5
    assert result.report["isolation"]["rows"] == 0


def test_isolation_needs_clean_support_and_is_scoped_to_cut_subpieces(base: Scene) -> None:
    inp, _ = inputs.build_clip(
        base.tracks,
        base.calib,
        base.emb,
        start_frame=0,
        end_frame=149,
        fps=FPS,
        frame_size=base.size,
        shirt_lab=_shirt_lab(base),
    )
    unsupported = dataclasses.replace(inp, clean=np.zeros(len(inp.frame), dtype=bool))
    assert relink.relink_clip(unsupported, "strict").isolated_rows == ()
    switched = make_scene(stints=SWITCH_STINTS, switch=SWITCH)
    assert switched.relink().report["isolation"]["rows"] == 0


def test_isolated_observations_are_stable_under_row_permutations() -> None:
    scene = make_scene()
    tracks, emb = scene.tracks.copy(), scene.emb.copy()
    bad = tracks.index[(tracks.track_id == 9) & tracks.frame.isin([90, 91])].to_numpy()
    colours = _shirt_lab(scene)
    for row in bad:
        other = tracks.index[(tracks.track_id == 11) & (tracks.frame == tracks.at[row, "frame"])][0]
        tracks.loc[row, ["x1", "y1", "x2", "y2"]] = tracks.loc[other, ["x1", "y1", "x2", "y2"]]
        emb[row] = emb[other]
        colours[row] = colours[other]
    expected = Scene(tracks, scene.calib, emb, scene.shots, scene.size).relink(shirt_lab=colours)
    order = np.random.default_rng(48).permutation(len(tracks))
    actual = Scene(tracks.iloc[order], scene.calib, emb[order], scene.shots, scene.size).relink(
        shirt_lab=colours[order]
    )
    assert actual.track_id.tolist() == expected.track_id[order].tolist()
    assert actual.report["isolation"] == expected.report["isolation"]


def _shirt_lab(scene: Scene) -> np.ndarray:
    """Separate, noise-free blue/white/yellow pixel evidence of the synthetic people."""
    return np.array(
        [
            [35.0, 10.0, -40.0]
            if team == "left"
            else [85.0, 0.0, 0.0]
            if team == "right"
            else [80.0, 0.0, 80.0]
            for team in scene.tracks.team.fillna("unknown")
        ]
    )


@pytest.mark.parametrize("mode", ["strict", "recall"])
@pytest.mark.parametrize("matching_samples", [1, 3])
def test_observed_own_shirt_variation_vetoes_isolation(mode: str, matching_samples: int) -> None:
    scene = make_scene()
    tracks, emb = scene.tracks.copy(), scene.emb.copy()
    row = tracks.index[(tracks.track_id == 9) & (tracks.frame == 120)][0]
    other = tracks.index[(tracks.track_id == 11) & (tracks.frame == 120)][0]
    tracks.loc[row, ["x1", "y1", "x2", "y2"]] = tracks.loc[other, ["x1", "y1", "x2", "y2"]]
    emb[row] = emb[other]
    colour = _shirt_lab(scene)
    colour[row] = colour[other]
    # The same person's crop has already shown this colour (sleeves, print or light).
    # A majority-colour prototype must not erase that evidence and claim a swap.
    own_samples = tracks.index[
        (tracks.track_id == 9) & tracks.frame.between(121, 120 + matching_samples)
    ]
    colour[own_samples] = colour[row]
    result = Scene(tracks, scene.calib, emb, scene.shots, scene.size).relink(mode, shirt_lab=colour)
    assert result.track_id[row] == 5
    assert result.report["isolation"]["rows"] == 0


@pytest.mark.parametrize("mode", ["strict", "recall"])
@pytest.mark.parametrize("evidence", ["same_shirt", "missing", "far_from_every_shirt"])
def test_corrupted_reid_is_not_a_swap_without_a_clear_other_shirt(mode: str, evidence: str) -> None:
    scene = make_scene()
    tracks, emb = scene.tracks.copy(), scene.emb.copy()
    row = tracks.index[(tracks.track_id == 9) & (tracks.frame == 90)][0]
    other = tracks.index[(tracks.track_id == 11) & (tracks.frame == 90)][0]
    tracks.loc[row, ["x1", "y1", "x2", "y2"]] = tracks.loc[other, ["x1", "y1", "x2", "y2"]]
    emb[row] = emb[other]
    colour = _shirt_lab(scene)
    if evidence == "missing":
        colour[row] = np.nan
    elif evidence == "far_from_every_shirt":
        colour[row] = [55.0, 60.0, 10.0]
    result = Scene(tracks, scene.calib, emb, scene.shots, scene.size).relink(mode, shirt_lab=colour)
    assert result.track_id[row] == 5
    assert result.report["isolation"]["rows"] == 0
    assert result.report["isolation"]["colour_evidence_supplied"] is True


@pytest.mark.parametrize("bad", [np.zeros((1, 3)), np.zeros((0, 4)), [[np.inf, 0, 0]]])
def test_misaligned_or_infinite_shirt_evidence_is_rejected(base: Scene, bad: Any) -> None:
    if np.shape(bad) == (1, 3) and np.isinf(bad).any():
        bad = np.tile(bad, (len(base.tracks), 1))
    with pytest.raises(relink.RelinkError, match="shirt_lab"):
        base.relink(shirt_lab=bad)


@pytest.mark.parametrize("foreign_kit", [False, True])
def test_rgb_corroboration_separates_occlusion_from_foreign_kit(foreign_kit: bool) -> None:
    scene = make_scene(scale=1.5)
    tracks, emb = scene.tracks.copy(), scene.emb.copy()
    row = tracks.index[(tracks.track_id == 9) & (tracks.frame == 90)][0]
    other = tracks.index[(tracks.track_id == 11) & (tracks.frame == 90)][0]
    tracks.loc[row, ["x1", "y1", "x2", "y2"]] = tracks.loc[other, ["x1", "y1", "x2", "y2"]]
    emb[row] = emb[other]
    boxes = tracks[["x1", "y1", "x2", "y2"]].to_numpy().astype(int)
    colours = {"left": (20, 50, 180), "right": (230, 230, 230), "unknown": (220, 200, 20)}
    teams = tracks.team.fillna("unknown").tolist()

    def frames():
        for frame in range(N_FRAMES):
            image = np.full((1080, 1920, 3), (20, 100, 20), dtype=np.uint8)
            for pos in np.flatnonzero(tracks.frame.to_numpy() == frame):
                x1, y1, x2, y2 = boxes[pos]
                image[max(y1, 0) : y2, max(x1, 0) : x2] = colours[teams[pos]]
            if frame == 90:
                x1, y1, x2, y2 = boxes[row]
                image[y1:y2, x1:x2] = colours["right" if foreign_kit else "left"]
            yield frame, image

    lab = relink.measure_shirt_colours(tracks, frames(), frame_size=scene.size)
    result = Scene(tracks, scene.calib, emb, scene.shots, scene.size).relink(shirt_lab=lab)
    if foreign_kit:
        assert (result.track_id == result.track_id[row]).sum() == 1
        assert result.report["isolation"]["rows"] == 1
    else:
        assert result.track_id[row] == 5
        assert result.report["isolation"]["rows"] == 0
