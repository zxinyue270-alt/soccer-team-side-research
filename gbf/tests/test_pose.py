"""Tests for gbf.pose: Halpe-26 constants, box/keypoint bookkeeping and pose.parquet.

The network is never loaded: a fake runner stands in for rtmlib's RTMPose. The constants are
compared with rtmlib's own Halpe-26 skeleton definition (rtmlib is a default dependency;
importing it loads no model and touches no network).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from gbf import models
from gbf.pose import (
    CENTRE_RGB,
    DEFAULT_INPUT_CHANNELS,
    HALPE26_EDGE_COLORS_RGB,
    HALPE26_EDGES,
    HALPE26_FLIP_PAIRS,
    HALPE26_INDEX,
    HALPE26_KEYPOINT_COLORS_RGB,
    HALPE26_KEYPOINTS,
    LEFT_RGB,
    N_KEYPOINTS,
    POSE_COLUMNS,
    POSE_SCHEMA,
    RIGHT_RGB,
    PoseEstimator,
    PoseInputError,
    PoseRuntimeError,
    RtmlibRunner,
    draw_pose,
    keypoints_array,
    pose_table,
    read_pose_parquet,
    write_pose_parquet,
)

ZERO_SCORE_KEYPOINT = 7  # the fake gives this keypoint score 0 (no location)


class FakeRunner:
    """Deterministic stand-in for rtmlib: keypoint j sits at a known place inside each box."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append((image.copy(), boxes.copy()))
        steps = (np.arange(N_KEYPOINTS) + 1.0) / (N_KEYPOINTS + 1.0)
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        u = boxes[:, :1] + width[:, None] * steps
        v = boxes[:, 1:2] + height[:, None] * steps[::-1]
        scores = (
            np.tile(0.2 + 0.7 * steps, (len(boxes), 1)) + 0.001 * np.arange(len(boxes))[:, None]
        )
        scores[:, ZERO_SCORE_KEYPOINT] = 0.0
        return np.stack([u, v], axis=-1), scores


def expected_pose(box: tuple[float, float, float, float], position: int = 0) -> np.ndarray:
    steps = (np.arange(N_KEYPOINTS) + 1.0) / (N_KEYPOINTS + 1.0)
    x1, y1, x2, y2 = box
    pose = np.column_stack(
        [x1 + (x2 - x1) * steps, y1 + (y2 - y1) * steps[::-1], 0.2 + 0.7 * steps + 0.001 * position]
    )
    pose[ZERO_SCORE_KEYPOINT] = (np.nan, np.nan, 0.0)
    return pose


def make_estimator(**kwargs) -> tuple[PoseEstimator, FakeRunner]:
    runner = FakeRunner()
    estimator = PoseEstimator("unused/fake-model.onnx", (288, 384), runner=runner, **kwargs)
    return estimator, runner


def frame_of(index: int, height: int = 120, width: int = 160) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = index % 256  # red carries the frame index
    frame[..., 2] = 200  # blue is constant
    return frame


# ------------------------------------------------------------------------- constants


def rtmlib_halpe26() -> dict:
    """rtmlib's halpe26 skeleton dict, loaded from its file without importing rtmlib
    (whose package import pulls in OpenCV and onnxruntime, about two seconds)."""
    spec = importlib.util.find_spec("rtmlib")
    assert spec is not None
    assert spec.submodule_search_locations
    path = Path(spec.submodule_search_locations[0]) / "visualization" / "skeleton" / "halpe26.py"
    namespace: dict = {}
    exec(compile(path.read_bytes().decode("utf-8"), str(path), "exec"), namespace)  # noqa: S102
    return namespace["halpe26"]


def test_halpe26_constants_match_rtmlib() -> None:
    halpe26 = rtmlib_halpe26()
    info = halpe26["keypoint_info"]
    assert tuple(info[i]["name"] for i in range(len(info))) == HALPE26_KEYPOINTS
    assert N_KEYPOINTS == 26
    links = [halpe26["skeleton_info"][i] for i in range(len(halpe26["skeleton_info"]))]
    assert HALPE26_EDGES == tuple(
        (HALPE26_INDEX[link["link"][0]], HALPE26_INDEX[link["link"][1]]) for link in links
    )
    assert HALPE26_EDGE_COLORS_RGB == tuple(tuple(link["color"]) for link in links)
    swaps = {
        (HALPE26_INDEX[item["name"]], HALPE26_INDEX[item["swap"]])
        for item in info.values()
        if item["swap"]
    }
    assert {pair for pair in swaps if pair[0] < pair[1]} == set(HALPE26_FLIP_PAIRS)


def test_keypoint_colours_follow_the_body_side() -> None:
    for name, colour in zip(HALPE26_KEYPOINTS, HALPE26_KEYPOINT_COLORS_RGB, strict=True):
        expected = (
            LEFT_RGB
            if name.startswith("left_")
            else RIGHT_RGB
            if name.startswith("right_")
            else CENTRE_RGB
        )
        assert colour == expected, name
    assert LEFT_RGB == (0, 255, 0)  # green: mmpose's left colour, read as RGB
    assert RIGHT_RGB == (255, 128, 0)  # orange
    assert DEFAULT_INPUT_CHANNELS == "rgb"


# -------------------------------------------------------------------------- estimate


def test_estimate_returns_frame_pixels_and_nan_for_zero_scores() -> None:
    estimator, runner = make_estimator()
    boxes = [(10.0, 20.0, 50.0, 100.0), (60.5, 5.0, 90.0, 115.0)]
    poses = estimator.estimate(frame_of(3), boxes)
    assert poses.shape == (2, 26, 3)
    assert poses.dtype == np.float64
    for position, box in enumerate(boxes):
        np.testing.assert_allclose(poses[position], expected_pose(box, position), equal_nan=True)
    assert len(runner.calls) == 1
    np.testing.assert_array_equal(runner.calls[0][1], np.array(boxes))


def test_no_boxes_means_no_backend_call() -> None:
    estimator, runner = make_estimator()
    for empty in ([], np.empty((0, 4))):
        poses = estimator.estimate(frame_of(0), empty)
        assert poses.shape == (0, 26, 3)
    assert runner.calls == []


@pytest.mark.parametrize(("channels", "first_value"), [("rgb", 7), ("bgr", 200)])
def test_input_channel_order(channels: str, first_value: int) -> None:
    estimator, runner = make_estimator(input_channels=channels)
    frame = frame_of(7)
    estimator.estimate(frame, [(0.0, 0.0, 10.0, 10.0)])
    image = runner.calls[0][0]
    assert image.shape == frame.shape
    assert image[0, 0, 0] == first_value
    assert image.flags["C_CONTIGUOUS"]
    assert frame[0, 0, 0] == 7  # the caller's frame is never modified


def test_estimate_lists_every_input_problem() -> None:
    estimator, _ = make_estimator()
    with pytest.raises(PoseInputError) as error:
        estimator.estimate(
            np.zeros((10, 10), dtype=np.float32),
            [(0.0, 0.0, -1.0, 5.0), (0.0, 0.0, 5.0, 0.0), (np.nan, 0.0, 1.0, 1.0)],
        )
    problems = "\n".join(error.value.problems)
    assert "dtype must be uint8" in problems
    assert "shape must be (height, width, 3)" in problems
    assert "boxes[0]: x2 must be > x1" in problems
    assert "boxes[1]: y2 must be > y1" in problems
    assert "boxes[2]: non-finite" in problems
    with pytest.raises(PoseInputError, match="expected shape"):
        estimator.estimate(frame_of(0), [(1.0, 2.0, 3.0)])


def test_malformed_backend_output_is_an_error() -> None:
    def short(image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((len(boxes), 17, 2)), np.zeros((len(boxes), 17))

    def nan_scores(image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((len(boxes), 26, 2)), np.full((len(boxes), 26), np.nan)

    for runner in (short, nan_scores):
        estimator = PoseEstimator("fake.onnx", (192, 256), runner=runner)
        with pytest.raises(PoseRuntimeError):
            estimator.estimate(frame_of(0), [(0.0, 0.0, 10.0, 10.0)])


def test_constructor_checks() -> None:
    with pytest.raises(PoseInputError) as error:
        PoseEstimator("x.onnx", (0, 384), runner=FakeRunner(), input_channels="hsv", model_id="")
    assert len(error.value.problems) == 3
    estimator, _ = make_estimator()
    assert estimator.model_id == "fake-model"
    assert estimator.input_size == (288, 384)
    named, _ = make_estimator(model_id="rtmpose-x-halpe26-384x288")
    assert named.model_id == "rtmpose-x-halpe26-384x288"


def test_rtmlib_runner_refuses_a_missing_model_without_downloading(unicode_tmp_path) -> None:
    missing = unicode_tmp_path / "no-such-model.onnx"
    with pytest.raises(PoseInputError) as error:
        RtmlibRunner(missing, (288, 384), backend="tensorflow", device="")
    problems = "\n".join(error.value.problems)
    assert "is not a file" in problems
    assert "backend" in problems
    assert "device" in problems
    with pytest.raises(PoseInputError):
        PoseEstimator(missing, (288, 384))  # builds an RtmlibRunner, which refuses


def test_from_pinned_needs_a_fetched_model(unicode_tmp_path) -> None:
    with pytest.raises(models.ModelMissingError):
        PoseEstimator.from_pinned("performance", unicode_tmp_path)


# ------------------------------------------------------------------- estimate_frames


def test_estimate_frames_builds_the_pose_table() -> None:
    estimator, runner = make_estimator(model_id="m1")
    boxes = {
        5: [(3, (10.0, 10.0, 40.0, 90.0)), (1, (50.0, 12.0, 80.0, 100.0))],
        7: [(-1, (5.0, 5.0, 20.0, 50.0)), (-1, (1.0, 5.0, 20.0, 50.0))],
    }
    frames = ((index, frame_of(index)) for index in range(10))
    table = estimator.estimate_frames(frames, boxes)

    assert list(table.columns) == list(POSE_COLUMNS)
    assert table["frame"].tolist() == [5, 5, 7, 7]
    assert table["track_id"].tolist() == [1, 3, -1, -1]
    assert table["x1"].tolist() == [50.0, 10.0, 1.0, 5.0]  # sorted, not in input order
    assert table["model_id"].tolist() == ["m1"] * 4
    assert table["frame"].dtype == np.int64
    assert table["track_id"].dtype == np.int64
    assert table["mean_score"].dtype == np.float64
    poses = keypoints_array(table)
    assert poses.shape == (4, 26, 3)
    np.testing.assert_allclose(poses[2], expected_pose((1.0, 5.0, 20.0, 50.0), 0), equal_nan=True)
    assert table["mean_score"].iloc[0] == pytest.approx(float(np.mean(poses[0, :, 2])))
    assert [call[0][0, 0, 0] for call in runner.calls] == [5, 7]  # each frame its own image


def test_estimate_frames_stops_after_the_last_needed_frame() -> None:
    estimator, _ = make_estimator()
    yielded: list[int] = []

    def frames():
        for index in range(100):
            yielded.append(index)
            yield index, frame_of(index)

    estimator.estimate_frames(frames(), {2: [(1, (0.0, 0.0, 9.0, 9.0))]})
    assert yielded == [0, 1, 2]


def test_estimate_frames_accepts_a_tracks_table() -> None:
    estimator, _ = make_estimator()
    tracks = pd.DataFrame(
        {
            "frame": [4, 4, 6],
            "track_id": [2, 9, 2],
            "x1": [1.0, 30.0, 2.0],
            "y1": [1.0, 2.0, 3.0],
            "x2": [20.0, 60.0, 21.0],
            "y2": [50.0, 80.0, 52.0],
            "conf": [0.9, 0.8, 0.7],
        }
    )
    table = estimator.estimate_frames(((i, frame_of(i)) for i in range(8)), tracks)
    assert table[["frame", "track_id"]].values.tolist() == [[4, 2], [4, 9], [6, 2]]
    with pytest.raises(PoseInputError, match="missing column"):
        estimator.estimate_frames([], tracks.drop(columns=["y2"]))


def test_estimate_frames_reports_missing_frames_and_bad_boxes() -> None:
    estimator, _ = make_estimator()
    boxes = {3: [(1, (0.0, 0.0, 9.0, 9.0))], 50: [(1, (0.0, 0.0, 9.0, 9.0))]}
    with pytest.raises(PoseInputError, match="never yielded: 50"):
        estimator.estimate_frames(((i, frame_of(i)) for i in range(10)), boxes)
    partial = estimator.estimate_frames(
        ((i, frame_of(i)) for i in range(10)), boxes, require_all=False
    )
    assert partial["frame"].tolist() == [3]
    with pytest.raises(PoseInputError) as error:
        estimator.estimate_frames(
            [],
            {
                1: [(4, (0.0, 0.0, 9.0, 9.0)), (4, (1.0, 0.0, 9.0, 9.0))],
                2: [(True, (0.0, 0.0, 9.0, 9.0)), (5, (5.0, 0.0, 1.0, 9.0))],
                -3: [(1, (0.0, 0.0, 9.0, 9.0))],
            },
        )
    problems = "\n".join(error.value.problems)
    assert "track_id 4 appears twice" in problems
    assert "track_id True" in problems
    assert "x2 must be > x1" in problems
    assert "frame -3" in problems


# --------------------------------------------------------------------------- parquet


def test_pose_parquet_round_trip(unicode_tmp_path) -> None:
    estimator, _ = make_estimator(model_id="rtmpose-x-halpe26-384x288")
    boxes = {0: [(1, (10.0, 10.0, 40.0, 90.0))], 1: [(2, (11.0, 10.0, 41.0, 90.0))]}
    table = estimator.estimate_frames(((i, frame_of(i)) for i in range(2)), boxes)
    path = unicode_tmp_path / "pose.parquet"
    write_pose_parquet(table, path)
    stored = pq.read_table(path)
    assert stored.schema.equals(POSE_SCHEMA)
    assert stored.schema.metadata is None
    back = read_pose_parquet(path)
    assert back[["frame", "track_id", "model_id"]].equals(table[["frame", "track_id", "model_id"]])
    np.testing.assert_array_equal(keypoints_array(back), keypoints_array(table))
    assert math.isnan(keypoints_array(back)[0, ZERO_SCORE_KEYPOINT, 0])


def test_pose_table_order_never_depends_on_row_order() -> None:
    def row(frame: int, track: int, x1: float) -> dict:
        return {
            "frame": frame,
            "track_id": track,
            "model_id": "m",
            "x1": x1,
            "y1": 0.0,
            "x2": x1 + 1.0,
            "y2": 1.0,
            "keypoints": [0.0] * 78,
            "mean_score": 0.5,
        }

    rows = [row(2, 5, 0.0), row(1, 9, 0.0), row(2, -1, 7.0), row(2, -1, 3.0), row(1, 4, 0.0)]
    table = pose_table(rows)
    assert table[["frame", "track_id", "x1"]].values.tolist() == [
        [1, 4, 0.0],
        [1, 9, 0.0],
        [2, -1, 3.0],
        [2, -1, 7.0],
        [2, 5, 0.0],
    ]
    assert pose_table(rows[::-1]).equals(table)


def test_write_pose_parquet_checks_columns_and_lengths(unicode_tmp_path) -> None:
    rows = [
        {
            "frame": 0,
            "track_id": 1,
            "model_id": "m",
            "x1": 0.0,
            "y1": 0.0,
            "x2": 1.0,
            "y2": 1.0,
            "keypoints": [0.0] * 77,
            "mean_score": 0.0,
        }
    ]
    with pytest.raises(PoseInputError, match="78 values"):
        write_pose_parquet(pose_table(rows), unicode_tmp_path / "bad.parquet")
    with pytest.raises(PoseInputError, match="missing column"):
        write_pose_parquet(pose_table(rows).drop(columns=["mean_score"]), unicode_tmp_path / "x")
    empty = pose_table([])
    assert list(empty.columns) == list(POSE_COLUMNS)
    write_pose_parquet(empty, unicode_tmp_path / "empty.parquet")
    assert pq.read_table(unicode_tmp_path / "empty.parquet").num_rows == 0


# --------------------------------------------------------------------------- drawing


def test_draw_pose_paints_visible_keypoints_only() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    pose = np.full((26, 3), np.nan)
    pose[:, 2] = 0.0
    pose[HALPE26_INDEX["left_knee"]] = (20.0, 30.0, 0.9)
    pose[HALPE26_INDEX["left_ankle"]] = (20.0, 70.0, 0.9)
    pose[HALPE26_INDEX["right_knee"]] = (70.0, 30.0, 0.1)  # below the threshold
    drawn = draw_pose(image, pose, score_threshold=0.3, radius=2, width=1)
    assert tuple(drawn[30, 20]) == LEFT_RGB
    assert tuple(drawn[50, 20]) == LEFT_RGB  # the left shank edge
    assert tuple(drawn[30, 70]) == (0, 0, 0)
    assert image.sum() == 0  # the input is untouched
    with pytest.raises(PoseInputError):
        draw_pose(image, np.zeros((25, 3)))
