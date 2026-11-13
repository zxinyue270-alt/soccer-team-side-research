"""Tests for gbf.kinematics: segments, Savitzky-Golay velocities, validity, peaks and the
per-frame helpers.

Trajectories are synthetic and known exactly. The first test runs the whole chain from
boxes projected by a synthetic camera through gbf.sngs_import; the others build tracks
tables directly so each rule can be tested on its own.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbf.geometry import project, synthetic_broadcast_camera
from gbf.kinematics import (
    KINEMATICS_COLUMNS,
    REJECTION_COLUMNS,
    SEGMENTS_COLUMNS,
    KinematicsInputError,
    chord_half_frames,
    common_mode_steps,
    compute_kinematics,
    kinematics_with_rejections,
    nearest_opponent,
    rejected_positions,
    team_shape,
)
from gbf.shots_grammar import build_shots
from gbf.sngs_import import import_sngs, load_calib, load_tracks
from gbf.util.hashing import sha256_bytes

FPS = 25.0


def _tracks(
    paths: dict[int, list[tuple[int, float, float]]],
    heights: dict[tuple[int, int], float] | None = None,
) -> pd.DataFrame:
    """A tracks table from {track_id: [(frame, x, y), ...]}: every row measured, every box
    40 x 100 px unless ``heights`` gives {(track_id, frame): height}."""
    rows = [(frame, track, x, y) for track, samples in paths.items() for frame, x, y in samples]
    frame = pd.DataFrame(rows, columns=["frame", "track_id", "pitch_x", "pitch_y"])
    frame["pitch_usable"] = True
    frame["in_pitch"] = True
    height = [
        (heights or {}).get((track, f), 100.0)
        for f, track in zip(frame["frame"], frame["track_id"], strict=True)
    ]
    frame["x1"] = 500.0
    frame["x2"] = 540.0
    frame["y2"] = 600.0
    frame["y1"] = frame["y2"] - np.asarray(height, dtype=np.float64)
    return frame


def _shots(*bounds: tuple[int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": list(range(len(bounds))),
            "start_frame": [start for start, _ in bounds],
            "end_frame": [end for _, end in bounds],
        }
    )


def _run(track: int, frames: range, x0: float, y0: float, vx: float, vy: float):
    """Samples of a constant-velocity run; velocity in m/s."""
    return [(f, x0 + vx * f / FPS, y0 + vy * f / FPS) for f in frames]


def _export(folder: Path, paths: dict[int, list[tuple[int, float, float]]], n: int) -> Path:
    """A plugin-sngs export whose boxes are the synthetic camera's projections of paths."""
    camera = synthetic_broadcast_camera(1920, 1080)
    detections = []
    for track, samples in paths.items():
        for frame, x, y in samples:
            uv, _, _ = project(camera, [x, y, 0.0])
            u, v = float(uv[0]), float(uv[1])
            box = [u - 20.0, v - 100.0, u + 20.0, v]
            team = "left" if track % 2 else "right"
            detections.append(
                {
                    "frame": frame,
                    "track_id": track,
                    "box": box,
                    "kf_box": None,
                    "conf": 0.9,
                    "role": "player",
                    "team": team,
                    "jersey": None,
                    "jersey_conf": None,
                }
            )
    calib = [
        {
            "frame": f,
            "calib_status": "fresh",
            "calib_age": 0,
            "K": camera.K.ravel().tolist(),
            "R": camera.R.ravel().tolist(),
            "C": camera.C.tolist(),
            "width": 1920,
            "height": 1080,
            "rep_err_px": 2.0,
            "rep_err_n_keypoints": 9,
        }
        for f in range(n)
    ]
    folder.mkdir(parents=True)
    files = {
        "detections.jsonl": "".join(json.dumps(r) + "\n" for r in detections).encode(),
        "calib.jsonl": "".join(json.dumps(r) + "\n" for r in calib).encode(),
    }
    for name, data in files.items():
        (folder / name).write_bytes(data)
    receipt = {
        "outputs": [{"name": k, "sha256": sha256_bytes(v)} for k, v in files.items()],
        "restrictions": [],
        "video_frames": n,
    }
    (folder / "plugin-receipt.json").write_bytes(json.dumps(receipt).encode())
    return folder


def test_constant_run_through_the_import(unicode_tmp_path: Path) -> None:
    paths = {
        1: _run(1, range(100), -30.0, 10.0, 6.0, 0.0),  # 6 m/s along the touchline
        2: _run(2, range(100), 5.0, -20.0, 0.0, 0.0),  # standing still
    }
    summary = import_sngs(_export(unicode_tmp_path / "export", paths, 100), unicode_tmp_path / "o")
    tracks = load_tracks(summary.tracks_path)
    shots = build_shots([], 100, tracks, load_calib(summary.calib_path))
    kinematics, segments = compute_kinematics(tracks, shots, "25/1")

    assert list(kinematics.columns) == list(KINEMATICS_COLUMNS)
    assert list(segments.columns) == list(SEGMENTS_COLUMNS)
    runner = kinematics[(kinematics["track_id"] == 1) & kinematics["valid"]]
    # A speed is the 10-frame chord centred on its frame, and both chord ends must be at
    # least 5 frames from the segment ends (one-sided estimates there): 10 frames at each end
    # have no valid speed.
    assert runner["frame"].tolist() == list(range(10, 90))
    np.testing.assert_allclose(runner["speed"], 6.0, rtol=0.01)
    np.testing.assert_allclose(runner["vx"], 6.0, rtol=0.01)
    np.testing.assert_allclose(runner["vy"], 0.0, atol=0.06)
    standing = kinematics[(kinematics["track_id"] == 2) & kinematics["valid"]]
    np.testing.assert_allclose(standing["speed"], 0.0, atol=1e-6)
    peaks = segments.set_index("track_id")["peak_speed"]
    assert peaks[1] == pytest.approx(6.0, rel=0.01)
    assert segments["n_frames"].tolist() == [100, 100]


def test_planted_sprint_peak_is_found() -> None:
    frames = np.arange(150)
    speed = 3.0 + 6.0 * np.exp(-0.5 * ((frames - 75) / 12.0) ** 2)  # 9 m/s at frame 75
    distance = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / FPS)])
    heading = math.radians(30.0)
    xs = -20.0 + distance * math.cos(heading)
    ys = -10.0 + distance * math.sin(heading)
    tracks = _tracks({7: list(zip(frames.tolist(), xs.tolist(), ys.tolist(), strict=True))})
    kinematics, segments = compute_kinematics(tracks, _shots((0, 149)), FPS)
    peak = segments.iloc[0]
    assert peak["peak_speed"] == pytest.approx(9.0, rel=0.02)
    assert abs(int(peak["peak_frame"]) - 75) <= 1
    at_peak = kinematics.set_index("frame").loc[75]
    assert math.degrees(math.atan2(at_peak["vy"], at_peak["vx"])) == pytest.approx(30.0, abs=0.5)
    quiet = kinematics.set_index("frame").loc[10]
    assert quiet["speed"] == pytest.approx(3.0, rel=0.01)


def test_segments_break_at_shot_boundaries() -> None:
    # The cut moves the camera; the same track id jumps 40 m but is never smoothed across.
    samples = [(f, 0.1 * f + (40.0 if f >= 30 else 0.0), 0.0) for f in range(60)]
    kinematics, segments = compute_kinematics(_tracks({3: samples}), _shots((0, 29), (30, 59)), FPS)
    assert segments[["shot_id", "start_frame", "end_frame"]].values.tolist() == [
        [0, 0, 29],
        [1, 30, 59],
    ]
    assert kinematics.groupby("segment_id")["frame"].agg(["min", "max"]).values.tolist() == [
        [0, 29],
        [30, 59],
    ]
    valid = kinematics[kinematics["valid"]]
    np.testing.assert_allclose(valid["speed"], 2.5, rtol=1e-6)
    # 10 frames at each end of a 30-frame segment: chord half (5) plus filter half (5).
    assert (segments["n_invalid"] == 20).all()


def test_short_gaps_are_interpolated_and_long_gaps_split() -> None:
    short = [s for s in _run(1, range(40), 0.0, 0.0, 4.0, 0.0) if not 20 <= s[0] <= 22]
    long = [s for s in _run(2, range(40), 0.0, 5.0, 4.0, 0.0) if not 20 <= s[0] <= 23]
    kinematics, segments = compute_kinematics(_tracks({1: short, 2: long}), _shots((0, 39)), FPS)
    by_track = segments.groupby("track_id")
    assert by_track.size().to_dict() == {1: 1, 2: 2}
    one = kinematics[kinematics["track_id"] == 1].set_index("frame")
    assert one.index.tolist() == list(range(40))
    assert one["interpolated"].sum() == 3
    assert one.loc[[20, 21, 22], "interpolated"].all()
    np.testing.assert_allclose(one.loc[[20, 21, 22], "x"], [3.2, 3.36, 3.52], atol=1e-9)
    assert segments.set_index("track_id").loc[1, "n_interpolated"] == 3
    two = segments[segments["track_id"] == 2]
    assert two[["start_frame", "end_frame"]].values.tolist() == [[0, 19], [24, 39]]


def test_implausible_speeds_are_flagged_with_their_neighbourhood() -> None:
    # An identity switch: from frame 50 on the track follows someone 5 m away. The jump
    # test would split the segment there (see test_id_jump_splits_the_segment), so it is
    # off here to test the plausibility rule on its own. Speeds are 0.4 s chords now: the
    # chords across the switch read more than 12 m/s, and every chord with an end within
    # half a filter window of the implausible filter derivative is invalid as well, so the
    # invalid frames form one block around the switch (before: the derivative's own burst
    # plus 5 frames on each side).
    run = _run(4, range(100), 0, 0, 5, 0)
    samples = [(f, x, y + (5.0 if f >= 50 else 0.0)) for f, x, y in run]
    for options in ({"outlier_threshold_m": None}, {}):
        kinematics, segments = compute_kinematics(
            _tracks({4: samples}), _shots((0, 99)), FPS, jump_threshold_m=None, **options
        )
        rows = kinematics.set_index("frame")
        burst = rows.index[rows["speed"] > 12.0].tolist()
        assert burst, "the switch must produce implausible speeds"
        assert min(burst) < 50 <= max(burst)
        invalid = [f for f in rows.index[~rows["valid"]] if 10 <= f <= 89]
        assert invalid == list(range(invalid[0], invalid[-1] + 1))  # one block
        assert invalid[0] < min(burst)
        assert max(burst) < invalid[-1]
        assert len(invalid) <= 30  # the switch costs at most 1.2 s of speeds
        clean = rows[rows["valid"]]
        np.testing.assert_allclose(clean["speed"], 5.0, rtol=1e-6)
        assert segments.iloc[0]["peak_speed"] == pytest.approx(5.0, rel=1e-6)


def test_single_frame_position_outlier_is_rejected() -> None:
    run = _run(5, range(80), 0, 0, 5, 0)
    samples = [(f, x, y + (3.0 if f == 40 else 0.0)) for f, x, y in run]
    tracks = _tracks({5: samples})
    kinematics, segments = compute_kinematics(tracks, _shots((0, 79)), FPS)
    rows = kinematics.set_index("frame")
    assert bool(rows.loc[40, "interpolated"])
    assert rows.loc[40, "y"] == pytest.approx(0.0, abs=1e-9)
    assert segments.iloc[0]["n_outliers"] == 1
    np.testing.assert_allclose(rows.loc[rows["valid"], "speed"], 5.0, rtol=1e-6)

    # The excursion test alone rejects it too: a one-frame excursion, out and back.
    jumps_only, jump_segments = compute_kinematics(
        tracks, _shots((0, 79)), FPS, outlier_threshold_m=None
    )
    assert bool(jumps_only.set_index("frame").loc[40, "interpolated"])
    assert jump_segments["n_outliers"].tolist() == [1]
    assert rejected_positions(tracks, _shots((0, 79)), FPS, outlier_threshold_m=None)[
        "test"
    ].tolist() == ["excursion"]
    unguarded, _ = compute_kinematics(
        tracks, _shots((0, 79)), FPS, outlier_threshold_m=None, jump_threshold_m=None
    )
    # Without both tests the spike stays in the smoothed positions. The 0.4 s chord reads it
    # as about 5.24 m/s (the filter derivative used before read more than 5.9 m/s).
    assert unguarded.loc[unguarded["valid"], "speed"].max() > 5.2


def _crowd(n_frames: int) -> dict[int, list[tuple[int, float, float]]]:
    """Five other players (tracks 10-14) jogging at 1 m/s in lanes of their own."""
    return {
        track: _run(track, range(n_frames), -30.0, -25.0 + 4.0 * track, 1.0, 0.0)
        for track in range(10, 15)
    }


def test_box_that_loses_the_legs_is_rejected() -> None:
    # Track 1 jogs at 3 m/s. For three frames its box loses the legs (70 px instead of
    # 100 px): the foot point rises in the image and the ground position moves 0.55 m away
    # from the camera. Too small for the 1.5 m median test and the 0.6 m jump test.
    run = _run(1, range(120), 0.0, 0.0, 3.0, 0.0)
    samples = [(f, x, y - (0.55 if 60 <= f <= 62 else 0.0)) for f, x, y in run]
    shrunk = {(1, f): 70.0 for f in (60, 61, 62)}
    tracks = _tracks({1: samples, **_crowd(120)}, heights=shrunk)
    kinematics, segments = compute_kinematics(tracks, _shots((0, 119)), FPS)
    rows = kinematics[kinematics["track_id"] == 1].set_index("frame")
    assert rows.loc[[60, 61, 62], "interpolated"].all()
    one = segments[segments["track_id"] == 1]
    assert one["n_outliers"].tolist() == [3]
    assert one.iloc[0]["peak_speed"] == pytest.approx(3.0, rel=1e-6)
    np.testing.assert_allclose(rows.loc[rows["valid"], "speed"], 3.0, rtol=1e-6)

    unguarded, unguarded_segments = compute_kinematics(
        tracks, _shots((0, 119)), FPS, max_box_height_change=None
    )
    fake = unguarded_segments.set_index("track_id").loc[1, "peak_speed"]
    # The dip reads as a burst of speed: about 3.11 m/s over a 0.4 s chord (the filter
    # derivative used before read more than 3.3 m/s).
    assert fake > 3.1


def test_box_width_test_is_off_by_default_and_can_be_turned_on() -> None:
    run = _run(1, range(60), 0.0, 0.0, 3.0, 0.0)
    tracks = _tracks({1: run})
    tracks.loc[tracks["frame"] == 30, "x2"] = 580.0  # 80 px wide instead of 40 px
    _, default = compute_kinematics(tracks, _shots((0, 59)), FPS)
    assert default["n_outliers"].tolist() == [0]
    _, on = compute_kinematics(tracks, _shots((0, 59)), FPS, max_box_width_change=0.2)
    assert on["n_outliers"].tolist() == [1]


@pytest.mark.parametrize("missing", [(), (57, 58, 59)])
def test_id_jump_splits_the_segment(missing: tuple[int, ...]) -> None:
    # Track 1 jogs at 1 m/s beside track 2, 2.5 m away. From frame 60 on track 1's box
    # follows track 2's player. Smoothed across, the 2.5 m step reads as a burst of about
    # 8.5 m/s, below the 12 m/s plausibility limit; across a 3-frame gap (frames 57-59
    # missing) it would be interpolated into the same burst.
    first = _run(1, range(120), 0.0, 0.0, 1.0, 0.0)
    second = _run(2, range(120), 0.0, 2.5, 1.0, 0.0)
    switched = [(f, x, y + (2.5 if f >= 60 else 0.0)) for f, x, y in first if f not in missing]
    tracks = _tracks({1: switched, 2: second, **_crowd(120)})
    kinematics, segments = compute_kinematics(tracks, _shots((0, 119)), FPS)
    one = segments[segments["track_id"] == 1]
    assert one["start_frame"].tolist() == [0, 60]
    assert one["n_outliers"].tolist() == [0, 0]
    np.testing.assert_allclose(one["peak_speed"], 1.0, rtol=1e-6)
    rows = kinematics[kinematics["track_id"] == 1]
    np.testing.assert_allclose(rows.loc[rows["valid"], "speed"], 1.0, rtol=1e-6)

    _, unguarded = compute_kinematics(tracks, _shots((0, 119)), FPS, jump_threshold_m=None)
    assert unguarded.set_index("track_id").loc[1, "peak_speed"] > 6.0  # a fake sprint


@pytest.mark.parametrize("size", [1.1, 1.2])
@pytest.mark.parametrize("dipped", [(50, 51), (50, 51, 52)])
def test_short_excursion_is_rejected_and_the_segment_kept(
    size: float, dipped: tuple[int, ...]
) -> None:
    # For two or three frames the box dips 1.1-1.2 m onto a neighbour and comes back: below
    # the 1.5 m median test, but more than the 1 m excursion test. The dip's positions are
    # tested together, against the trends around all of them; tested one at a time, each sat
    # in the other's trend fit, and the 2-frame dips were kept (the jogger read 1.49 and
    # 1.57 m/s) and the 3-frame dips lost only their middle position (1.85 and 1.97 m/s).
    run = _run(1, range(100), 0.0, 0.0, 1.0, 0.0)
    samples = [(f, x, y + (size if f in dipped else 0.0)) for f, x, y in run]
    tracks = _tracks({1: samples, **_crowd(100)})
    kinematics, segments = compute_kinematics(tracks, _shots((0, 99)), FPS)
    one = segments[segments["track_id"] == 1]
    assert one[["start_frame", "end_frame", "n_outliers"]].values.tolist() == [[0, 99, len(dipped)]]
    rows = kinematics[kinematics["track_id"] == 1].set_index("frame")
    assert rows.loc[list(dipped), "interpolated"].all()
    np.testing.assert_allclose(rows.loc[rows["valid"], "speed"], 1.0, rtol=1e-6)


@pytest.mark.parametrize("dipped", [(50, 51), (50, 51, 52)])
def test_dip_along_a_runners_motion_is_rejected(dipped: tuple[int, ...]) -> None:
    # A 6 m/s runner's box jumps 1.2 m ahead for two or three frames and comes back. Kept,
    # the dip read as a burst of 7.21 and 7.69 m/s (+20 and +28 %).
    run = _run(1, range(100), -40.0, 0.0, 6.0, 0.0)
    samples = [(f, x + (1.2 if f in dipped else 0.0), y) for f, x, y in run]
    kinematics, segments = compute_kinematics(
        _tracks({1: samples, **_crowd(100)}), _shots((0, 99)), FPS
    )
    one = segments[segments["track_id"] == 1]
    assert one[["start_frame", "end_frame", "n_outliers"]].values.tolist() == [[0, 99, len(dipped)]]
    rows = kinematics[kinematics["track_id"] == 1]
    np.testing.assert_allclose(rows.loc[rows["valid"], "speed"], 6.0, rtol=1e-6)


@pytest.mark.parametrize("along", [True, False])
@pytest.mark.parametrize(
    "dipped",
    [(50, 51, 55, 56), (50, 51, 56, 57), (50, 51, 52, 55, 56, 57), (50, 51, 52, 56, 57, 58)],
)
def test_two_short_dips_close_together_are_rejected(dipped: tuple[int, ...], along: bool) -> None:
    # Two dips of 1.2 m a few frames apart, ahead of a 6 m/s runner or sideways on a 1 m/s
    # jogger. Each dip lies in the side windows of the other and pulled the trend fitted
    # there towards itself, so neither was more than 1 m off its trends: every position was
    # kept, and the runner read 7.58 and 7.21 m/s (two 2-frame dips) and 7.81 and 7.69 m/s
    # (two 3-frame dips), the jogger 1.87, 1.57, 2.07 and 1.97 m/s. The side fits are made
    # again without the positions more than half a metre off their lines.
    speed = 6.0 if along else 1.0
    dx, dy = (1.2, 0.0) if along else (0.0, 1.2)
    samples = [
        (f, x + (dx if f in dipped else 0.0), y + (dy if f in dipped else 0.0))
        for f, x, y in _run(1, range(100), -40.0, 0.0, speed, 0.0)
    ]
    tracks = _tracks({1: samples, **_crowd(100)})
    rejected = rejected_positions(tracks, _shots((0, 99)), FPS)
    assert rejected[["frame", "track_id", "test"]].values.tolist() == [
        [f, 1, "excursion"] for f in dipped
    ]
    kinematics, segments = compute_kinematics(tracks, _shots((0, 99)), FPS)
    one = segments[segments["track_id"] == 1]
    assert one[["start_frame", "end_frame", "n_outliers"]].values.tolist() == [[0, 99, len(dipped)]]
    rows = kinematics[kinematics["track_id"] == 1]
    np.testing.assert_allclose(rows.loc[rows["valid"], "speed"], speed, rtol=1e-6)


@pytest.mark.parametrize("along", [True, False])
def test_dips_of_two_and_three_frames_close_together_are_rejected(along: bool) -> None:
    # Dips of 1.5 m at frames 50-51 and 54-56, ahead of a 6 m/s runner or sideways on a
    # 1 m/s jogger. Pulled towards the 3-frame dip, the first fit around the 2-frame dip puts
    # the positions between them (52, 53) more than half a metre off its after line too, on
    # the other side from the block: they do not continue the block, so the refit leaves
    # them out with the other dip, and as they lie on its final lines, only the other dip
    # must be rejected with the block. Both dips were kept and the jump tests split the
    # track at them (7.66 and 1.94 m/s).
    speed = 6.0 if along else 1.0
    dx, dy = (1.5, 0.0) if along else (0.0, 1.5)
    dipped = (50, 51, 54, 55, 56)
    samples = [
        (f, x + (dx if f in dipped else 0.0), y + (dy if f in dipped else 0.0))
        for f, x, y in _run(1, range(100), -40.0, 0.0, speed, 0.0)
    ]
    tracks = _tracks({1: samples, **_crowd(100)})
    rejected = rejected_positions(tracks, _shots((0, 99)), FPS)
    assert rejected[["frame", "track_id", "test"]].values.tolist() == [
        [f, 1, "excursion"] for f in dipped
    ]
    kinematics, segments = compute_kinematics(tracks, _shots((0, 99)), FPS)
    one = segments[segments["track_id"] == 1]
    assert one[["start_frame", "end_frame", "n_outliers"]].values.tolist() == [[0, 99, 5]]
    rows = kinematics[kinematics["track_id"] == 1]
    np.testing.assert_allclose(rows.loc[rows["valid"], "speed"], speed, rtol=1e-6)


def test_dip_next_to_another_one_is_not_rejected_alone() -> None:
    # A 6 m/s runner's box dips 1.2 m ahead at frames 50-51 and 1 m at 53-55, which is not
    # more than 1 m: no excursion. Fitted without the 1 m dip, the 1.2 m dip is more than 1 m
    # off its trends, but rejecting it alone left the other dip in the track (7.50 m/s); the
    # jump tests split the two off together (6.21 m/s: the chords of the segment that starts
    # at the dips do not reach them).
    offsets = {50: 1.2, 51: 1.2, 53: 1.0, 54: 1.0, 55: 1.0}
    samples = [
        (f, x + offsets.get(f, 0.0), y) for f, x, y in _run(1, range(100), -40.0, 0.0, 6.0, 0.0)
    ]
    tracks = _tracks({1: samples, **_crowd(100)})
    assert rejected_positions(tracks, _shots((0, 99)), FPS).empty
    kinematics, segments = compute_kinematics(tracks, _shots((0, 99)), FPS)
    assert segments.loc[segments["track_id"] == 1, "start_frame"].tolist() == [0, 50]
    rows = kinematics[(kinematics["track_id"] == 1) & kinematics["valid"]]
    assert rows["speed"].max() < 6.3


@pytest.mark.parametrize(
    ("speed", "dipped"), [(6.0, (60, 61, 63, 64, 65)), (1.0, (60, 61, 62, 64, 65))]
)
def test_dips_one_position_apart_are_left_whole_to_the_jump_tests(
    speed: float, dipped: tuple[int, ...]
) -> None:
    # A runner's box dips 1.6 m ahead twice with one position of the track between the
    # dips. The refits leave out the side positions far off their lines, except the run of
    # them next to the block that lies off on its side, which continues it: the second dip
    # continues the block made of the first dip and the position between them, so on that
    # block's refit trend the position lies more than 1 m off, and the first dip has no
    # neighbour on its trend. Neither dip is an isolated excursion, and the six frames are
    # left whole to the jump tests, which cut them off as segments too short for a valid
    # speed. Without that exception each dip's refits left the other out, both dips were
    # rejected and the position between them stayed alone in the track (the speeds are
    # true either way).
    samples = [
        (f, x + (1.6 if f in dipped else 0.0), y)
        for f, x, y in _run(1, range(150), -40.0, 0.0, speed, 0.0)
    ]
    tracks = _tracks({1: samples, **_crowd(150)})
    assert rejected_positions(tracks, _shots((0, 149)), FPS).empty
    kinematics, segments = compute_kinematics(tracks, _shots((0, 149)), FPS)
    assert segments.loc[segments["track_id"] == 1, "start_frame"].tolist() == [0, 60, 63, 66]
    rows = kinematics[(kinematics["track_id"] == 1) & kinematics["valid"]]
    np.testing.assert_allclose(rows["speed"], speed, rtol=1e-6)


def test_longer_excursion_is_split_off_whole() -> None:
    # A 5-frame dip is longer than the gap interpolation bridges (max_gap_frames, 3), so it
    # is not rejected in part: positions of it left at the end of a segment would bend the
    # smoothed positions the last valid speeds there use. The jump tests cut it off whole as
    # a segment too short for a valid speed.
    run = _run(1, range(150), -40.0, 0.0, 6.0, 0.0)
    samples = [(f, x + (1.2 if 70 <= f <= 74 else 0.0), y) for f, x, y in run]
    kinematics, segments = compute_kinematics(
        _tracks({1: samples, **_crowd(150)}), _shots((0, 149)), FPS
    )
    one = segments[segments["track_id"] == 1]
    assert one["start_frame"].tolist() == [0, 70, 75]
    assert one["n_outliers"].sum() == 0
    rows = kinematics[kinematics["track_id"] == 1]
    np.testing.assert_allclose(rows.loc[rows["valid"], "speed"], 6.0, rtol=1e-6)


def test_common_motion_raises_the_jump_threshold() -> None:
    # A calibration wobble moves everybody 0.3 m every frame (below the 0.5 m calibration
    # jump test). The fast player's single steps alternate 0.65 m and 0.05 m, but the
    # trend fits average the wobble out: no jump and no excursion, whatever jump_factor.
    wobble = {f: 0.15 * (-1) ** f for f in range(80)}
    paths = {
        track: [(f, x + wobble[f], y) for f, x, y in samples]
        for track, samples in {
            1: _run(1, range(80), -20.0, 0.0, 8.75, 0.0),
            **_crowd(80),
        }.items()
    }
    for factor in (5.0, 1.0):
        _, segments = compute_kinematics(_tracks(paths), _shots((0, 79)), FPS, jump_factor=factor)
        assert (segments["track_id"] == 1).sum() == 1
        assert segments["n_outliers"].sum() == 0
    # The old test checked the factor with that wobble, which no longer looks like a jump.
    # A common step the calibration guard misses (here the guard is off): from frame 40 on
    # everybody is 1.2 m further along x. Every track's trend steps 1.2 m there, more than
    # the 1 m threshold but not 5 times the other players' median, so it is no jump; with a
    # factor of 0.5 it is one.
    shifted = {
        track: [
            (f, x + (1.2 if f >= 40 else 0.0), y)
            for f, x, y in _run(track, range(80), -30.0, -25.0 + 4.0 * track, 1.0, 0.0)
        ]
        for track in range(1, 7)
    }
    tracks = _tracks(shifted)
    _, loose = compute_kinematics(tracks, _shots((0, 79)), FPS, max_common_step_m=None)
    assert len(loose) == 6
    _, strict = compute_kinematics(
        tracks, _shots((0, 79)), FPS, max_common_step_m=None, jump_factor=0.5
    )
    assert len(strict) == 12
    assert set(strict["start_frame"]) == {0, 40}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noisy_sprint_keeps_its_peak(seed: int) -> None:
    # The planted 9 m/s sprint with 3 cm of position noise: no jump splits it and the
    # peak stays within 5 %.
    frames = np.arange(150)
    speed = 3.0 + 6.0 * np.exp(-0.5 * ((frames - 75) / 12.0) ** 2)
    distance = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / FPS)])
    noise = np.random.default_rng(seed).normal(0.0, 0.03, size=(2, frames.size))
    xs = -20.0 + distance + noise[0]
    ys = -10.0 + noise[1]
    tracks = _tracks(
        {7: list(zip(frames.tolist(), xs.tolist(), ys.tolist(), strict=True)), **_crowd(150)}
    )
    _, segments = compute_kinematics(tracks, _shots((0, 149)), FPS)
    sprint = segments[segments["track_id"] == 7]
    assert len(sprint) == 1
    assert sprint.iloc[0]["n_outliers"] == 0
    assert sprint.iloc[0]["peak_speed"] == pytest.approx(9.0, rel=0.05)


def _gaussian_sprint(
    top: float, *, hold: int = 1, noise: float = 0.0, seed: int = 0
) -> list[tuple[int, float, float]]:
    """150 frames along x from 3 m/s up to ``top`` m/s at frame 75 and back (sigma 12
    frames). With ``hold`` the position advances only every hold-th frame (as a broadcast
    foot point does in steps); ``noise`` adds that much Gaussian noise (metres) to x and y."""
    frames = np.arange(150)
    speed = 3.0 + (top - 3.0) * np.exp(-0.5 * ((frames - 75) / 12.0) ** 2)
    distance = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / FPS)])
    xs = (-20.0 + distance)[(frames // hold) * hold]
    jitter = np.random.default_rng(seed).normal(0.0, noise, size=(2, frames.size))
    return list(
        zip(frames.tolist(), (xs + jitter[0]).tolist(), (-10.0 + jitter[1]).tolist(), strict=True)
    )


def _sprint_segments(path: list[tuple[int, float, float]]) -> pd.DataFrame:
    tracks = _tracks({7: path, **_crowd(150)})
    _, segments = compute_kinematics(tracks, _shots((0, 149)), FPS)
    return segments[segments["track_id"] == 7]


@pytest.mark.parametrize("hold", [2, 3])
@pytest.mark.parametrize("top", [8.0, 9.0])
def test_sprint_with_held_positions_keeps_its_peak(top: float, hold: int) -> None:
    # The position updates only every second (or third) frame, so the single steps of an
    # 8-9 m/s sprint alternate 0 and 0.64-0.72 m (up to 1.08 m). The old single-step jump
    # test (0.6 m) cut the sprint in two, and the peak read 6.6 m/s at 8 m/s and 6.2 m/s at
    # 9 m/s; the trend tests take the steps for neither a jump nor an acceleration.
    sprint = _sprint_segments(_gaussian_sprint(top, hold=hold))
    assert len(sprint) == 1
    assert sprint.iloc[0]["n_outliers"] == 0
    assert sprint.iloc[0]["peak_speed"] == pytest.approx(top, rel=0.05)


def _accelerating_run(v0: float, v1: float, acceleration: float, hold: int) -> list:
    """150 frames along x: v0 m/s until frame 40, then a constant acceleration (m/s^2,
    signed) to v1 m/s, then v1, both changes abrupt; positions held for ``hold`` frames."""
    frames = np.arange(150)
    t = (frames - 40) / FPS
    speed = np.clip(v0 + acceleration * np.maximum(t, 0.0), min(v0, v1), max(v0, v1))
    distance = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / FPS)])
    xs = (-40.0 + distance)[(frames // hold) * hold]
    return list(zip(frames.tolist(), xs.tolist(), [5.0] * 150, strict=True))


@pytest.mark.parametrize("hold", [1, 2, 3])
@pytest.mark.parametrize(("v0", "v1", "acceleration"), [(1.0, 8.5, 7.0), (8.5, 1.0, -7.0)])
def test_hard_acceleration_keeps_one_segment(
    v0: float, v1: float, acceleration: float, hold: int
) -> None:
    # A sprinter speeds up from 1 to 8.5 m/s at 7 m/s^2, or slows down as hard, starting
    # and ending abruptly, with the position held for up to three frames: one segment, no
    # position rejected, and the speeds of the sprint are kept.
    sprint = _sprint_segments(_accelerating_run(v0, v1, acceleration, hold))
    assert len(sprint) == 1
    assert sprint.iloc[0]["n_outliers"] == 0
    assert sprint.iloc[0]["peak_speed"] == pytest.approx(8.5, rel=0.05)


@pytest.mark.parametrize("hold", [1, 2])
def test_turn_and_sudden_stop_keep_one_segment(hold: int) -> None:
    # A 6 m/s runner turns by 90 degrees at 7 m/s^2 (a curve of 5.1 m radius), and an
    # 8 m/s runner's foot point stops from one frame to the next, as a planted foot does:
    # neither is a jump, and no position is rejected (the excursion test, one position at a
    # time, rejected the stop's corner).
    frames = np.arange(150)
    heading = np.clip((frames - 50) / FPS * 7.0 / 6.0, 0.0, math.pi / 2)
    vx, vy = 6.0 * np.cos(heading), 6.0 * np.sin(heading)
    xs = np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) / FPS)])
    ys = np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) / FPS)])
    held = (frames // hold) * hold
    turn = list(zip(frames.tolist(), (xs[held] - 40).tolist(), ys[held].tolist(), strict=True))
    stop = [(f, -40.0 + 8.0 * min(f, 75) / FPS, 5.0) for f in frames.tolist()]
    for path in (turn, stop):
        segments = _sprint_segments(path)
        assert len(segments) == 1
        assert segments.iloc[0]["n_outliers"] == 0


@pytest.mark.parametrize("seed", range(6))
def test_sprint_with_10cm_noise_keeps_its_peak(seed: int) -> None:
    # 10 cm of noise on a 9 m/s sprint: a single step scatters by 0.14 m around 0.36 m, so
    # the old single-step test cut it into 2-4 pieces with four of these seeds and rejected
    # three good positions with a fifth.
    sprint = _sprint_segments(_gaussian_sprint(9.0, noise=0.1, seed=seed))
    assert len(sprint) == 1
    assert sprint.iloc[0]["n_outliers"] == 0
    assert sprint.iloc[0]["peak_speed"] == pytest.approx(9.0, rel=0.05)


@pytest.mark.parametrize(
    ("distance", "missing"), [(1.5, (73, 74)), (1.8, (72, 73, 74)), (2.0, (74,))]
)
def test_id_switch_across_missing_frames_splits_the_segment(
    distance: float, missing: tuple[int, ...]
) -> None:
    # A jogger (1 m/s) is lost for 1-3 frames and the track comes back on a neighbour
    # 1.5-2 m away. The old allowance grew by 12 m/s for each missing frame (2.04 m for 3),
    # so the 1.5 m and 1.8 m switches were bridged and smoothed into a fake run of about
    # 5 m/s.
    jogger = [
        (f, x, y + (distance if f >= 75 else 0.0))
        for f, x, y in _run(1, range(150), 0.0, 0.0, 1.0, 0.0)
        if f not in missing
    ]
    neighbour = _run(2, range(150), 0.0, distance, 1.0, 0.0)
    _, segments = compute_kinematics(
        _tracks({1: jogger, 2: neighbour, **_crowd(150)}), _shots((0, 149)), FPS
    )
    one = segments[segments["track_id"] == 1]
    assert one["start_frame"].tolist() == [0, 75]
    np.testing.assert_allclose(one["peak_speed"], 1.0, rtol=1e-6)


@pytest.mark.parametrize(
    ("distance", "frames_sliding", "by_step"),
    [
        (1.5, 5, False),
        (1.5, 6, False),
        (1.5, 8, False),
        (2.4, 5, True),
        (2.4, 8, True),
        (2.4, 10, False),
        (2.4, 12, False),
        (3.0, 12, False),
    ],
)
def test_slide_onto_a_neighbour_splits_the_segment(
    distance: float, frames_sliding: int, by_step: bool
) -> None:
    # A jogger's box slides 1.5-3 m onto a neighbour over 5-12 frames and stays there
    # (every single step below the old 0.6 m test, which kept 2.4 m over 5 and 8 frames as
    # one segment peaking at 6.9-7.7 m/s). The two-sided trend fit with one velocity took
    # the slide for a change of velocity, and 1.5 m over 5/6/8 frames read 4.07/4.01/3.80
    # m/s, 2.4 m over 10/12 frames 5.50/5.01 m/s and 3 m over 12 frames 6.21 m/s. Moving
    # off the trend and back onto it that fast takes more than 8 m/s^2: the acceleration
    # test cuts the slide off.
    slide = {75 + i: distance * (i + 1) / frames_sliding for i in range(frames_sliding)}
    jogger = [
        (f, x, y + (distance if f >= 75 + frames_sliding else slide.get(f, 0.0)))
        for f, x, y in _run(1, range(150), 0.0, 0.0, 1.0, 0.0)
    ]
    neighbour = _run(2, range(150), 0.0, distance, 1.0, 0.0)
    tracks = _tracks({1: jogger, 2: neighbour, **_crowd(150)})
    kinematics, segments = compute_kinematics(tracks, _shots((0, 149)), FPS)
    one = segments[segments["track_id"] == 1]
    assert len(one) >= 2
    assert one["peak_speed"].max() == pytest.approx(1.0, rel=0.05)
    rows = kinematics[(kinematics["track_id"] == 1) & kinematics["valid"]]
    assert rows["speed"].max() <= 1.05
    # by_step: the trend step test alone splits the slide too
    _, unguarded = compute_kinematics(tracks, _shots((0, 149)), FPS, max_acceleration_mps2=None)
    assert (unguarded.loc[unguarded["track_id"] == 1, "peak_speed"].max() < 1.05) == by_step


@pytest.mark.parametrize("along", [False, True])
@pytest.mark.parametrize("distance", [0.6, 0.8, 1.0])
def test_small_switch_splits_the_segment(distance: float, along: bool) -> None:
    # From frame 75 on, a jogger's box is on a neighbour 0.6-1 m to the side (or ahead): a
    # trend step below the 1 m jump threshold. Kept, the 0.4 s chords across it read
    # 1.91/2.39/2.89 m/s sideways and 2.63/3.17 m/s ahead (0.6/0.8 m). No velocity explains
    # the step, and moving that far off the trend and back within the 0.48 s between the
    # side centroids takes more than 8 m/s^2 (4 * 0.6 m / 0.48^2 s^2 is 10.4 m/s^2).
    dx, dy = (distance, 0.0) if along else (0.0, distance)
    jogger = [
        (f, x + (dx if f >= 75 else 0.0), y + (dy if f >= 75 else 0.0))
        for f, x, y in _run(1, range(150), 0.0, 0.0, 1.0, 0.0)
    ]
    neighbour = _run(2, range(150), dx, dy, 1.0, 0.0)
    kinematics, segments = compute_kinematics(
        _tracks({1: jogger, 2: neighbour, **_crowd(150)}), _shots((0, 149)), FPS
    )
    one = segments[segments["track_id"] == 1]
    assert one["start_frame"].tolist() == [0, 75]
    np.testing.assert_allclose(one["peak_speed"], 1.0, rtol=1e-6)
    rows = kinematics[(kinematics["track_id"] == 1) & kinematics["valid"]]
    np.testing.assert_allclose(rows["speed"], 1.0, rtol=1e-6)


def test_speed_is_the_chord_of_the_smoothed_positions() -> None:
    # Every speed is the 0.4 s chord (10 frames at 25 fps) of the smoothed positions centred
    # on its frame; gbf.render's labels, the gbf.highlights speed cue and gbf.measure's
    # peak-speed all read it, so they show one number and none exceeds the peak.
    frames = np.arange(120)
    xs = 4.0 * np.sin(frames / 20.0) + 0.05 * frames
    ys = 3.0 * np.cos(frames / 25.0)
    tracks = _tracks({3: list(zip(frames.tolist(), xs.tolist(), ys.tolist(), strict=True))})
    kinematics, segments = compute_kinematics(tracks, _shots((0, 119)), FPS)
    rows = kinematics.set_index("frame")
    k = chord_half_frames(FPS)
    assert k == 5
    centre = np.arange(k, 120 - k)
    vx = (rows.loc[centre + k, "x"].to_numpy() - rows.loc[centre - k, "x"].to_numpy()) * FPS / 10
    vy = (rows.loc[centre + k, "y"].to_numpy() - rows.loc[centre - k, "y"].to_numpy()) * FPS / 10
    np.testing.assert_allclose(rows.loc[centre, "vx"], vx, atol=1e-12)
    np.testing.assert_allclose(rows.loc[centre, "vy"], vy, atol=1e-12)
    np.testing.assert_allclose(rows.loc[centre, "speed"], np.hypot(vx, vy), atol=1e-12)
    assert rows.loc[: k - 1, "speed"].isna().all()
    assert rows.loc[120 - k :, "speed"].isna().all()
    valid = rows[rows["valid"]]
    assert valid.index.tolist() == list(range(10, 110))
    assert segments.iloc[0]["peak_speed"] == valid["speed"].max()
    assert segments.iloc[0]["peak_frame"] == valid["speed"].idxmax()


def test_chord_half_frames() -> None:
    rates = (25, 24, 30, 50, "30000/1001", Fraction(24000, 1001))
    assert [chord_half_frames(fps) for fps in rates] == [5, 5, 6, 10, 6, 5]
    assert chord_half_frames(25, 1.0) == 13  # 12.5 frames round half up
    with pytest.raises(KinematicsInputError, match="fps"):
        chord_half_frames(0)
    with pytest.raises(KinematicsInputError, match="chord_s"):
        chord_half_frames(25, -0.4)


def _every_rejection() -> pd.DataFrame:
    """Track 1 jogs at 3 m/s. A 3 m spike at frame 30 (local-median test); for frames 60-64
    its box loses the legs (box test), a gap longer than max_gap_frames, so these frames lie
    between two segments; a 1.3 m dip at frames 90-91 (excursion test)."""

    def offset(frame: int) -> float:
        if frame == 30:
            return 3.0
        if 60 <= frame <= 64:
            return -0.55
        return 1.3 if frame in (90, 91) else 0.0

    samples = [(f, x, y + offset(f)) for f, x, y in _run(1, range(120), 0.0, 0.0, 3.0, 0.0)]
    shrunk = {(1, f): 70.0 for f in range(60, 65)}
    return _tracks({1: samples, **_crowd(120)}, heights=shrunk)


def test_rejected_positions_counts_every_test() -> None:
    tracks = _every_rejection()
    rejected = rejected_positions(tracks, _shots((0, 119)), FPS)
    assert list(rejected.columns) == list(REJECTION_COLUMNS)
    assert rejected.values.tolist() == [
        [30, 1, "local_median"],
        *([f, 1, "box_height"] for f in range(60, 65)),
        [90, 1, "excursion"],
        [91, 1, "excursion"],
    ]
    _, segments = compute_kinematics(tracks, _shots((0, 119)), FPS)
    one = segments[segments["track_id"] == 1]
    assert one[["start_frame", "end_frame"]].values.tolist() == [[0, 59], [65, 119]]
    # segments.parquet counts only the rejections inside a segment, not the 5 between them.
    assert one["n_outliers"].tolist() == [1, 2]
    assert len(rejected) == 8


@pytest.mark.parametrize("options", [{}, {"jump_window_frames": 10, "max_acceleration_mps2": None}])
def test_kinematics_with_rejections_equals_the_separate_calls(options: dict) -> None:
    # The pipeline takes the tables and the rejections from one run of the outlier tests,
    # which take most of the time, instead of running them twice.
    tracks, shots = _every_rejection(), _shots((0, 119))
    kinematics, segments, rejected = kinematics_with_rejections(tracks, shots, FPS, **options)
    separate = compute_kinematics(tracks, shots, FPS, **options)
    pd.testing.assert_frame_equal(kinematics, separate[0])
    pd.testing.assert_frame_equal(segments, separate[1])
    pd.testing.assert_frame_equal(rejected, rejected_positions(tracks, shots, FPS, **options))
    assert len(rejected) == 8
    with pytest.raises(KinematicsInputError, match="kinematics_with_rejections"):
        kinematics_with_rejections(tracks, _shots((0, 19)), FPS)


def test_excursion_test_in_chunks_of_tracks_gives_the_same_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The excursion test fits its blocks a chunk of whole tracks at a time, so that its side
    # arrays (a few kB per position) stay small on long inputs, while its allowances still
    # compare the blocks of every track: here every player dips 1.2 m at frames 90-91 (the
    # calibration guard is off), which the other players' median allows, and tracks 1-3 dip
    # on their own (a short dip, two dips close together and two dips one position apart).
    # Chunks of one or two tracks give the tables and rejections of a single chunk.
    rng = np.random.default_rng(11)
    own = {1: (40, 41), 2: (50, 51, 55, 56), 3: (60, 61, 63, 64, 65)}
    paths = {}
    for track in range(1, 9):
        noise = rng.normal(0.0, 0.05 if track > 3 else 0.0, size=(120, 2))
        paths[track] = [
            (
                f,
                x + noise[f, 0] + (1.2 if f in own.get(track, ()) + (90, 91) else 0.0),
                y + noise[f, 1],
            )
            for f, x, y in _run(track, range(120), -40.0, 4.0 * track, 1.0 + track, 0.0)
        ]
    tracks, shots = _tracks(paths), _shots((0, 119))
    results = []
    for size in (10**9, 1, 300):
        monkeypatch.setattr("gbf.kinematics._EXCURSION_CHUNK", size)
        results.append(kinematics_with_rejections(tracks, shots, FPS, max_common_step_m=None))
    for chunked in results[1:]:
        for table, expected in zip(chunked, results[0], strict=True):
            pd.testing.assert_frame_equal(table, expected)
    rejected = results[0][2]
    assert rejected[["frame", "track_id"]].values.tolist() == [
        [f, track] for track, frames in own.items() for f in frames
    ]


def test_calibration_jump_splits_every_track() -> None:
    # Five players walking at 1 m/s; from frame 30 the calibration puts everyone 1 m further.
    paths = {
        track: [
            (f, x + (1.0 if f >= 30 else 0.0), y)
            for f, x, y in _run(track, range(60), -40.0 + 10 * track, 2.0 * track, 1.0, 0.0)
        ]
        for track in range(5)
    }
    tracks = _tracks(paths)
    steps = common_mode_steps(tracks).set_index("frame")
    assert steps.loc[30, "step_m"] == pytest.approx(1.04)
    assert steps.loc[30, "n_players"] == 5
    assert steps.drop(index=30)["step_m"].max() == pytest.approx(0.04)

    kinematics, segments = compute_kinematics(tracks, _shots((0, 59)), FPS)
    assert segments.groupby("track_id").size().tolist() == [2] * 5
    assert set(segments["start_frame"]) == {0, 30}
    np.testing.assert_allclose(kinematics.loc[kinematics["valid"], "speed"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(segments["peak_speed"], 1.0, rtol=1e-6)

    unguarded, unguarded_segments = compute_kinematics(
        tracks, _shots((0, 59)), FPS, max_common_step_m=None
    )
    assert len(unguarded_segments) == 5
    assert unguarded_segments["peak_speed"].min() > 3.0  # the jump reads as a fake sprint


def test_short_segments_have_no_velocity_or_peak() -> None:
    paths = {
        1: _run(1, range(8), 0, 0, 3, 0),
        2: _run(2, range(20), 10, 0, 3, 0),
        3: _run(3, range(30), 20, 0, 3, 0),
    }
    kinematics, segments = compute_kinematics(_tracks(paths), _shots((0, 29)), FPS)
    first = kinematics[kinematics["track_id"] == 1]  # shorter than the filter window
    assert first["speed"].isna().all()
    assert not first["valid"].any()
    np.testing.assert_allclose(first["x"], [3 * f / FPS for f in range(8)])
    # 20 frames: chords fit at frames 5-14, but a segment shorter than min_segment_frames
    # (25) has no valid speed any more (before, its speeds were valid but gave no peak), so
    # the speeds shown always come from segments that can give the card's peak.
    second = kinematics[kinematics["track_id"] == 2]
    assert second.loc[second["speed"].notna(), "frame"].tolist() == list(range(5, 15))
    assert not second["valid"].any()
    third = kinematics[kinematics["track_id"] == 3]
    assert third.loc[third["valid"], "frame"].tolist() == list(range(10, 20))
    rows = segments.set_index("track_id")
    assert rows.loc[1, "n_invalid"] == 8
    assert math.isnan(rows.loc[2, "peak_speed"])
    assert rows.loc[2, "peak_frame"] == -1
    assert rows.loc[3, "peak_speed"] == pytest.approx(3.0)


def test_rows_that_are_not_measured_are_ignored() -> None:
    tracks = _tracks({1: _run(1, range(30), 0, 0, 2, 0)})
    extra = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3],
            "track_id": [-1, 9, 9, 9],
            "pitch_x": [1.0, 1.0, np.nan, 1.0],
            "pitch_y": [1.0, 1.0, 1.0, 1.0],
            "pitch_usable": [True, False, True, True],
            "in_pitch": [True, True, True, False],
        }
    )
    kinematics, segments = compute_kinematics(pd.concat([tracks, extra]), _shots((0, 29)), FPS)
    assert segments["track_id"].tolist() == [1]
    assert kinematics["track_id"].unique().tolist() == [1]


def test_empty_input_gives_empty_tables() -> None:
    kinematics, segments = compute_kinematics(_tracks({}), _shots((0, 9)), FPS)
    assert kinematics.empty
    assert segments.empty
    assert list(kinematics.columns) == list(KINEMATICS_COLUMNS)
    assert kinematics["valid"].dtype == bool
    assert segments["peak_frame"].dtype == np.int64


def test_invalid_arguments_are_all_listed() -> None:
    tracks = _tracks({1: _run(1, range(30), 0, 0, 2, 0)})
    with pytest.raises(KinematicsInputError) as caught:
        compute_kinematics(
            tracks,
            _shots((0, 29)),
            "29.97",
            sg_window_frames=10,
            sg_poly=12,
            max_gap_frames=-1,
            min_segment_frames=0,
            max_speed_mps=-1.0,
            max_common_step_m=0.0,
            outlier_threshold_m=float("nan"),
            max_box_height_change=0.0,
            max_box_width_change=-0.2,
            jump_threshold_m=float("inf"),
            jump_factor=0,
            jump_window_frames=2,
            max_acceleration_mps2=0.0,
            speed_chord_s=0.0,
        )
    text = "\n".join(caught.value.problems)
    for expected in (
        "fps:",
        "sg_window_frames must be odd",
        "sg_poly (12) must be smaller than sg_window_frames (10)",
        "max_gap_frames must be >= 0",
        "min_segment_frames must be >= 1",
        "max_speed_mps must be finite and > 0",
        "max_common_step_m must be finite and > 0",
        "outlier_threshold_m must be finite and > 0",
        "max_box_height_change must be finite and > 0",
        "max_box_width_change must be finite and > 0",
        "jump_threshold_m must be finite and > 0",
        "jump_factor must be finite and > 0",
        "jump_window_frames must be >= 3",
        "max_acceleration_mps2 must be finite and > 0",
        "speed_chord_s must be finite and > 0",
    ):
        assert expected in text, expected

    with pytest.raises(KinematicsInputError, match="lacks column"):
        compute_kinematics(tracks.drop(columns="in_pitch"), _shots((0, 29)), FPS)
    # The box columns are needed only while a box-size test is on.
    no_boxes = tracks.drop(columns=["x1", "y1", "x2", "y2"])
    with pytest.raises(KinematicsInputError, match=r"lacks column\(s\) x1, y1, x2, y2"):
        compute_kinematics(no_boxes, _shots((0, 29)), FPS)
    _, segments = compute_kinematics(no_boxes, _shots((0, 29)), FPS, max_box_height_change=None)
    assert len(segments) == 1
    duplicated = pd.concat([tracks, tracks.iloc[:1]])
    with pytest.raises(KinematicsInputError, match="duplicate"):
        compute_kinematics(duplicated, _shots((0, 29)), FPS)
    with pytest.raises(KinematicsInputError, match="lie in no shot"):
        compute_kinematics(tracks, _shots((0, 19)), FPS)


def _frame_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [7] * 6,
            "track_id": [1, 2, 3, 4, 5, 6],
            "team": ["left", "left", "right", "right", None, "right"],
            "role": ["player", "player", "player", "goalkeeper", "referee", "player"],
            "pitch_x": [0.0, 10.0, 3.0, 10.0, 0.5, np.nan],
            "pitch_y": [0.0, 0.0, 4.0, 2.0, 0.5, 0.0],
        }
    )


def test_nearest_opponent() -> None:
    result = nearest_opponent(_frame_rows())
    assert result["track_id"].tolist() == [1, 2, 3, 4]  # referee and NaN position left out
    assert result["opponent_track_id"].tolist() == [3, 4, 1, 2]
    np.testing.assert_allclose(result["distance_m"], [5.0, 2.0, 5.0, 2.0])
    alone = nearest_opponent(_frame_rows().iloc[:2])
    assert alone["opponent_track_id"].tolist() == [-1, -1]
    assert alone["distance_m"].isna().all()
    smoothed = _frame_rows().rename(columns={"pitch_x": "x", "pitch_y": "y"})
    assert nearest_opponent(smoothed, x_col="x", y_col="y")["distance_m"].iloc[0] == 5.0
    with pytest.raises(KinematicsInputError, match="single frame"):
        nearest_opponent(pd.concat([_frame_rows(), _frame_rows().assign(frame=8)]))


def test_team_shape() -> None:
    rows = pd.DataFrame(
        {
            "track_id": [1, 2, 3, 4, 5, 6],
            "team": ["left", "left", "left", "left", "right", "right"],
            "role": ["player", "player", "player", "goalkeeper", "player", "player"],
            "pitch_x": [-30.0, -20.0, -25.0, -50.0, 10.0, 20.0],
            "pitch_y": [-10.0, 5.0, 20.0, 0.0, -3.0, 9.0],
        }
    )
    left = team_shape(rows, "left")
    assert (left.n_outfield, left.deepest_track_id) == (3, 1)
    assert left.deepest_x == -30.0  # the goalkeeper at -50 does not count
    assert (left.width_m, left.length_m) == (30.0, 10.0)
    assert (left.centroid_x, left.centroid_y) == (-25.0, 5.0)
    right = team_shape(rows, "right")
    assert (right.deepest_x, right.deepest_track_id, right.width_m) == (20.0, 6, 12.0)
    empty = team_shape(rows[rows["team"] == "left"], "right")
    assert empty.n_outfield == 0
    assert empty.deepest_track_id == -1
    assert math.isnan(empty.width_m)
    with pytest.raises(KinematicsInputError) as caught:
        team_shape(rows.drop(columns="role"), "middle")
    assert len(caught.value.problems) == 2
