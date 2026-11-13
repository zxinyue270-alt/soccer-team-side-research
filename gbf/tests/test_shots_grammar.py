"""Tests for gbf.shots_grammar: shot bounds, shot types and cut detection.

Tracks and calibration tables are built by hand (the import has its own tests); the cut
test encodes a tiny synthetic video with ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbf.shots_grammar import (
    SHOTS_COLUMNS,
    ShotsInputError,
    build_shots,
    classify_shot,
    detect_cut_frames,
    shot_bounds,
    shot_ids,
)
from gbf.synth.video import write_frames_mp4


def _tracks(spec: dict[int, list[tuple[str, float]]]) -> pd.DataFrame:
    """tracks rows from {frame: [(role, box height px), ...]}."""
    rows = []
    for frame, boxes in spec.items():
        for index, (role, height) in enumerate(boxes):
            rows.append(
                {
                    "frame": frame,
                    "track_id": index,
                    "role": role,
                    "x1": 100.0,
                    "y1": 500.0 - height,
                    "x2": 140.0,
                    "y2": 500.0,
                }
            )
    return pd.DataFrame(rows, columns=["frame", "track_id", "role", "x1", "y1", "x2", "y2"])


def _calib(n_frames: int, usable: set[int], height: int = 1080) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": np.arange(n_frames, dtype=np.int64),
            "height": np.full(n_frames, height, dtype=np.int32),
            "pitch_usable": [frame in usable for frame in range(n_frames)],
        }
    )


def test_shot_bounds_are_inclusive_and_ordered() -> None:
    assert shot_bounds([5, 2], 10) == [(0, 1), (2, 4), (5, 9)]
    assert shot_bounds([], 3) == [(0, 2)]
    assert shot_bounds(np.array([1]), 2) == [(0, 0), (1, 1)]


def test_shot_bounds_lists_every_problem() -> None:
    with pytest.raises(ShotsInputError) as caught:
        shot_bounds([0, 4, 4, 10, 2.5, True], 10)
    text = "\n".join(caught.value.problems)
    assert "cut frame 0 must lie in 1..9" in text
    assert "cut frame 4 is listed more than once" in text
    assert "cut frame 10 must lie in 1..9" in text
    assert "cut frame 2.5 is not an integer" in text
    assert "cut frame True is not an integer" in text
    with pytest.raises(ShotsInputError, match="n_frames must be a positive integer"):
        shot_bounds([], 0)


@pytest.mark.parametrize(
    ("median_box_h", "fresh_share", "expected"),
    [
        (250.0, 0.9, "close"),
        (220.0, 1.0, "close"),
        (219.9, 0.9, "medium"),
        (160.0, 0.1, "close"),
        (160.0, 0.2, "medium"),
        (149.9, 0.0, "medium"),
        (100.0, 0.6, "wide"),
        (129.9, 0.5, "wide"),
        (130.0, 0.5, "medium"),
        (100.0, 0.49, "medium"),
        (float("nan"), 0.0, "close"),
        (float("nan"), 0.6, "medium"),
    ],
)
def test_classify_shot_rules(median_box_h: float, fresh_share: float, expected: str) -> None:
    assert classify_shot(median_box_h, fresh_share) == expected


def test_classify_shot_rejects_bad_values() -> None:
    with pytest.raises(ShotsInputError) as caught:
        classify_shot("tall", 1.5)  # type: ignore[arg-type]
    assert len(caught.value.problems) == 2


def test_build_shots_types_and_cues() -> None:
    small = [("player", 100.0)] * 12 + [("referee", 104.0)]
    spec: dict[int, list[tuple[str, float]]] = {}
    for frame in range(10):  # wide: small boxes, every frame usable
        spec[frame] = small
    for frame in range(10, 15):  # close: one huge box, no calibration
        spec[frame] = [("player", 600.0), ("other", 40.0)]
    for frame in range(15, 20):  # medium: mid-size boxes, 3 of 5 frames usable
        spec[frame] = [("goalkeeper", 180.0), ("player", 170.0), ("player", 190.0)]
    usable = set(range(10)) | {15, 16, 17}
    shots = build_shots([10, 15], 20, _tracks(spec), _calib(20, usable))

    assert list(shots.columns) == list(SHOTS_COLUMNS)
    assert shots["shot_id"].tolist() == [0, 1, 2]
    assert shots["start_frame"].tolist() == [0, 10, 15]
    assert shots["end_frame"].tolist() == [9, 14, 19]
    assert shots["shot_type"].tolist() == ["wide", "close", "medium"]
    np.testing.assert_allclose(shots["fresh_share"], [1.0, 0.0, 0.6])
    np.testing.assert_allclose(shots["median_box_h"], [100.0, 600.0, 180.0])
    assert shots["shot_id"].dtype == np.int64
    assert shots["median_box_h"].dtype == np.float64


def test_median_box_height_counts_every_frame_once() -> None:
    # One frame with 20 small people, three frames with one large person each: the
    # pooled median of all boxes would be small, the median of frame medians is large.
    spec = {0: [("player", 90.0)] * 20, 1: [("player", 400.0)], 2: [("player", 410.0)]}
    spec[3] = [("player", 420.0)]
    shots = build_shots([], 4, _tracks(spec), _calib(4, set()))
    assert shots["median_box_h"].iloc[0] == pytest.approx(405.0)
    assert shots["shot_type"].iloc[0] == "close"


def test_box_heights_are_normalised_to_1080_lines() -> None:
    spec = {0: [("player", 60.0)], 1: [("player", 60.0)]}
    shots = build_shots([], 2, _tracks(spec), _calib(2, {0, 1}, height=540))
    assert shots["median_box_h"].iloc[0] == pytest.approx(120.0)
    assert shots["shot_type"].iloc[0] == "wide"
    explicit = build_shots([], 2, _tracks(spec), _calib(2, {0, 1}), frame_height=540)
    assert explicit["median_box_h"].iloc[0] == pytest.approx(120.0)


def test_shot_without_people_and_frames_without_calib_rows() -> None:
    calib = _calib(3, {0, 1, 2}).iloc[:2]  # frame 2 has no calib row: not usable
    shots = build_shots([2], 3, _tracks({0: [("player", 100.0)]}), calib)
    assert shots["fresh_share"].tolist() == [1.0, 0.0]
    assert np.isnan(shots["median_box_h"].iloc[1])
    assert shots["shot_type"].tolist() == ["wide", "close"]


def test_build_shots_lists_every_problem() -> None:
    tracks = _tracks({0: [("player", 100.0)], 12: [("player", 100.0)]})
    calib = pd.concat([_calib(3, set()), _calib(1, set())])  # frame 0 twice
    with pytest.raises(ShotsInputError) as caught:
        build_shots([1], 10, tracks, calib)
    text = "\n".join(caught.value.problems)
    assert "tracks_df has 1 row(s) with frame outside 0..9" in text
    assert "calib_df has duplicate frame_index values" in text

    with pytest.raises(ShotsInputError) as caught:
        build_shots([0], 10, tracks.drop(columns="role"), "calib")  # type: ignore[arg-type]
    text = "\n".join(caught.value.problems)
    assert "cut frame 0 must lie in 1..9" in text
    assert "tracks_df lacks column(s) role" in text
    assert "calib_df must be a pandas DataFrame" in text

    two_heights = _calib(2, set())
    two_heights.loc[1, "height"] = 720
    with pytest.raises(ShotsInputError, match="several frame heights"):
        build_shots([], 2, tracks.iloc[:1], two_heights)


def test_shot_ids_maps_frames_to_shots() -> None:
    shots = pd.DataFrame({"shot_id": [0, 1], "start_frame": [0, 5], "end_frame": [4, 9]})
    np.testing.assert_array_equal(shot_ids(shots, [0, 4, 5, 9, 10, -1]), [0, 0, 1, 1, -1, -1])
    gap = pd.DataFrame({"shot_id": [7, 8], "start_frame": [0, 6], "end_frame": [3, 9]})
    np.testing.assert_array_equal(shot_ids(gap, [3, 4, 5, 6]), [7, -1, -1, 8])
    overlapping = pd.DataFrame({"shot_id": [0, 1], "start_frame": [0, 4], "end_frame": [4, 9]})
    with pytest.raises(ShotsInputError, match="overlapping"):
        shot_ids(overlapping, [1])


def test_detect_cut_frames_finds_a_hard_cut(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    frames = []
    for index in range(12):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        if index < 7:
            frame[:] = (40, 140, 60)  # grass
            frame[10:20, 4 + 3 * index : 12 + 3 * index] = 250  # a small moving player
        else:
            frame[:] = (20, 30, 120)  # another shot: a dark blue close-up
            frame[24:40, 20:44] = (200, 170, 140)
        frames.append(frame)
    video = unicode_tmp_path / "cut clip.mp4"
    write_frames_mp4(frames, video, crf=10)
    assert detect_cut_frames(video) == [7]
