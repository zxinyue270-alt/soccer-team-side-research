"""2D body pose for player boxes: RTMPose Halpe-26 through rtmlib, with boxes we supply.

The pose stage runs a top-down RTMPose model (see :mod:`gbf.models`) on boxes that come from
the tracker (``tracks.parquet`` / plugin-sngs ``detections.jsonl``). No person detector is
run: rtmlib's ``RTMPose`` is called directly with our boxes. Output is ``pose.parquet``
(docs/design/m1-highlights.md): one row per (frame, box) with 26 keypoints in Halpe-26
order as ``[u, v, score]``.

Conventions
    Frames are ``(height, width, 3)`` uint8 **RGB** arrays, as :func:`gbf.ingest.iter_frames`
    yields them. Boxes are ``[x1, y1, x2, y2]`` in pixels of that frame, used as given (not
    clipped; rtmlib pads outside the image with black). Keypoint coordinates ``(u, v)`` are
    in the same pixel grid with pixel centres at integer coordinates (gbf.geometry): rtmlib
    crops with ``cv2.warpAffine`` and maps its SimCC output back through the inverse of that
    affine map, both in OpenCV's pixel-centre convention. The SimCC resolution is half an
    input pixel, i.e. ``1.25 * box_height / (2 * input_height)`` frame pixels for a box that
    fills the crop vertically. A keypoint whose score is <= 0 has no location: rtmlib marks
    it -1 in the crop; here its u and v are NaN.

Scores
    The score is rtmlib's SimCC confidence: the mean of the maxima of the x and y SimCC
    responses. It is typically within 0..1 but is not a calibrated probability and can
    exceed 1. ``mean_score`` is the plain mean of the 26 scores.

Colour order (checked, not assumed)
    rtmlib normalises whatever image it is given with the ImageNet RGB mean and std
    (123.675, 116.28, 103.53) / (58.395, 57.12, 57.375) and never reorders channels; its
    examples pass ``cv2.imread`` / ``VideoCapture`` frames, i.e. BGR. The mmdeploy
    ``pipeline.json`` shipped inside both pinned zips declares ``"to_rgb": true`` for that
    normalisation: the networks were trained on RGB input. ``input_channels`` chooses what
    the network receives; the default is RGB (our frames unchanged), see
    :data:`DEFAULT_INPUT_CHANNELS`.

Skeleton and colours
    :data:`HALPE26_KEYPOINTS` and :data:`HALPE26_EDGES` reproduce rtmlib's
    ``visualization/skeleton/halpe26.py`` (rtmlib 0.0.16) index for index, edge for edge.
    rtmlib's colour triples follow mmpose (left green, right orange, centre blue, in RGB) but
    rtmlib draws them with OpenCV on BGR images, so its own renders show the right side blue
    and the face orange. The colours here are RGB as mmpose intends; keypoints are coloured
    by body side (rtmlib also paints the left toes and heel in the right-side colour).

The model weights are experimental and for private use only (training-data licences not
verified; see :mod:`gbf.models`).
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

HALPE26_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
)
"""Halpe-26 keypoint names in network output order. ``head`` is the top of the head,
``neck`` the base of the neck and ``hip`` the pelvis centre; left/right are the person's."""

N_KEYPOINTS = len(HALPE26_KEYPOINTS)

HALPE26_INDEX: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(HALPE26_KEYPOINTS)}
)
"""Keypoint name to index."""

HALPE26_EDGES: tuple[tuple[int, int], ...] = (
    (15, 13),  # left_ankle - left_knee
    (13, 11),  # left_knee - left_hip
    (11, 19),  # left_hip - hip
    (16, 14),  # right_ankle - right_knee
    (14, 12),  # right_knee - right_hip
    (12, 19),  # right_hip - hip
    (17, 18),  # head - neck
    (18, 19),  # neck - hip
    (18, 5),  # neck - left_shoulder
    (5, 7),  # left_shoulder - left_elbow
    (7, 9),  # left_elbow - left_wrist
    (18, 6),  # neck - right_shoulder
    (6, 8),  # right_shoulder - right_elbow
    (8, 10),  # right_elbow - right_wrist
    (1, 2),  # left_eye - right_eye
    (0, 1),  # nose - left_eye
    (0, 2),  # nose - right_eye
    (1, 3),  # left_eye - left_ear
    (2, 4),  # right_eye - right_ear
    (3, 5),  # left_ear - left_shoulder
    (4, 6),  # right_ear - right_shoulder
    (15, 20),  # left_ankle - left_big_toe
    (15, 22),  # left_ankle - left_small_toe
    (15, 24),  # left_ankle - left_heel
    (16, 21),  # right_ankle - right_big_toe
    (16, 23),  # right_ankle - right_small_toe
    (16, 25),  # right_ankle - right_heel
)
"""The 27 links of rtmlib's halpe26 ``skeleton_info``, in its order."""

HALPE26_FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (20, 21),
    (22, 23),
    (24, 25),
)
"""(left, right) index pairs; swapping them mirrors the person's sides."""

LEFT_RGB: tuple[int, int, int] = (0, 255, 0)
RIGHT_RGB: tuple[int, int, int] = (255, 128, 0)
CENTRE_RGB: tuple[int, int, int] = (51, 153, 255)

HALPE26_SIDES: tuple[str, ...] = tuple(
    "left" if name.startswith("left_") else "right" if name.startswith("right_") else "centre"
    for name in HALPE26_KEYPOINTS
)
"""Body side of each keypoint: "left", "right" or "centre"."""

HALPE26_KEYPOINT_COLORS_RGB: tuple[tuple[int, int, int], ...] = tuple(
    {"left": LEFT_RGB, "right": RIGHT_RGB, "centre": CENTRE_RGB}[side] for side in HALPE26_SIDES
)
"""Drawing colour of each keypoint, RGB, by body side."""

HALPE26_EDGE_COLORS_RGB: tuple[tuple[int, int, int], ...] = (
    *(LEFT_RGB,) * 3,
    *(RIGHT_RGB,) * 3,
    *(CENTRE_RGB,) * 2,
    *(LEFT_RGB,) * 3,
    *(RIGHT_RGB,) * 3,
    *(CENTRE_RGB,) * 7,
    *(LEFT_RGB,) * 3,
    *(RIGHT_RGB,) * 3,
)
"""Drawing colour of each edge in :data:`HALPE26_EDGES`, RGB (rtmlib's link colours)."""

POSE_COLUMNS: tuple[str, ...] = (
    "frame",
    "track_id",
    "model_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "keypoints",
    "mean_score",
)
"""Columns of ``pose.parquet`` in order (docs/design/m1-highlights.md)."""

POSE_SCHEMA = pa.schema(
    [
        pa.field("frame", pa.int64(), nullable=False),
        pa.field("track_id", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("x1", pa.float64(), nullable=False),
        pa.field("y1", pa.float64(), nullable=False),
        pa.field("x2", pa.float64(), nullable=False),
        pa.field("y2", pa.float64(), nullable=False),
        pa.field("keypoints", pa.list_(pa.float64()), nullable=False),
        pa.field("mean_score", pa.float64(), nullable=False),
    ]
)
"""Arrow schema of ``pose.parquet``. ``keypoints`` holds 78 floats: 26 x [u, v, score]."""

InputChannels = Literal["rgb", "bgr"]

DEFAULT_INPUT_CHANNELS: InputChannels = "rgb"
"""Channel order handed to the network: RGB, as the zips' pipeline.json says it was trained.

A paired comparison on development footage (2026-10-21, rtmpose-x, the same boxes fed both
ways; numbers in the M1 pose task report, not here) gave slightly higher keypoint scores
for RGB in most boxes and sub-pixel differences in keypoint positions: the network is fairly
robust to the swap, and the documented training order is the default.
"""

RTMLIB_BACKENDS: tuple[str, ...] = ("onnxruntime", "opencv", "openvino")
"""Inference backends rtmlib 0.0.16 implements."""

UNKNOWN_TRACK_ID = -1
"""track_id of a detection without a track (plugin-sngs convention); may repeat per frame."""


class PoseInputError(ValueError):
    """Malformed input to the pose stage. ``problems`` lists every problem found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


class PoseRuntimeError(RuntimeError):
    """The pose backend returned something other than (k, 26, 2) keypoints and (k, 26) scores."""


class PoseRunner(Protocol):
    """A top-down pose network run on one image and its boxes.

    ``image`` is ``(height, width, 3)`` uint8 in the channel order the estimator chose and
    ``boxes`` a ``(k, 4)`` float64 array of ``[x1, y1, x2, y2]`` pixels with k >= 1. It
    returns ``(keypoints, scores)``: ``(k, 26, 2)`` pixel coordinates in the image's grid
    and ``(k, 26)`` scores. Tests implement it with a fake; :class:`RtmlibRunner` is the
    real one.
    """

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class RtmlibRunner:
    """rtmlib's ``RTMPose`` on an ONNX file that is already on disk (no download, no detector).

    rtmlib downloads any model path that does not exist (and falls back to a mirror), so a
    missing file is refused here before rtmlib is imported. rtmlib processes boxes one at a
    time (batch 1) and prints a load message, which is captured into ``load_message``.
    ``input_size`` is the network input (width, height) in pixels, rtmlib's order.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int],
        backend: str = "onnxruntime",
        device: str = "cpu",
    ) -> None:
        path = Path(model_path)
        problems = []
        if not path.is_file():
            problems.append(
                f"model_path: {path} is not a file (rtmlib would try to download it); "
                "fetch it with `python -m gbf models fetch`"
            )
        if backend not in RTMLIB_BACKENDS:
            problems.append(f"backend: {backend!r} is not one of {', '.join(RTMLIB_BACKENDS)}")
        if not isinstance(device, str) or not device:
            problems.append(f"device: expected a non-empty string, got {device!r}")
        size = _input_size(input_size, problems)
        if problems or size is None:
            raise PoseInputError("RtmlibRunner", problems)
        from rtmlib import RTMPose  # imported late: it pulls in cv2 and onnxruntime

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            self._model = RTMPose(
                onnx_model=str(path), model_input_size=size, backend=backend, device=device
            )
        self.load_message = captured.getvalue().strip()
        self.model_path = path
        self.input_size = size
        self.backend = backend
        self.device = device

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keypoints, scores = self._model(image, bboxes=[[float(v) for v in box] for box in boxes])
        return np.asarray(keypoints, dtype=np.float64), np.asarray(scores, dtype=np.float64)


def _input_size(value: Any, problems: list[str]) -> tuple[int, int] | None:
    try:
        items = list(value)
    except TypeError:
        problems.append(f"input_size: expected (width, height), got {value!r}")
        return None
    if len(items) != 2 or not all(
        isinstance(v, int | np.integer) and not isinstance(v, bool) and v > 0 for v in items
    ):
        problems.append(f"input_size: expected (width, height) positive integers, got {value!r}")
        return None
    return int(items[0]), int(items[1])


def _checked_frame(frame: Any, problems: list[str]) -> np.ndarray | None:
    if not isinstance(frame, np.ndarray):
        problems.append(f"frame_rgb: expected a numpy array, got {type(frame).__name__}")
        return None
    found = len(problems)
    if frame.dtype != np.uint8:
        problems.append(f"frame_rgb: dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
        problems.append(f"frame_rgb: shape must be (height, width, 3), got {frame.shape}")
    return frame if len(problems) == found else None


def _checked_boxes(boxes: Any, problems: list[str], name: str = "boxes") -> np.ndarray | None:
    try:
        array = np.asarray(boxes, dtype=np.float64)
    except (TypeError, ValueError) as error:
        problems.append(f"{name}: cannot be read as numbers ({error})")
        return None
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        problems.append(f"{name}: expected shape (k, 4) [x1, y1, x2, y2], got {array.shape}")
        return None
    found = len(problems)
    for row in np.flatnonzero(~np.isfinite(array).all(axis=1)).tolist():
        problems.append(f"{name}[{row}]: non-finite coordinates")
    finite = np.isfinite(array).all(axis=1)
    for row in np.flatnonzero(finite & ~(array[:, 2] > array[:, 0])).tolist():
        problems.append(f"{name}[{row}]: x2 must be > x1, got {array[row].tolist()}")
    for row in np.flatnonzero(finite & ~(array[:, 3] > array[:, 1])).tolist():
        problems.append(f"{name}[{row}]: y2 must be > y1, got {array[row].tolist()}")
    return array if len(problems) == found else None


class PoseEstimator:
    """Halpe-26 pose for given boxes, one frame at a time.

    Args:
        model_path: the ``.onnx`` network (from ``python -m gbf models fetch``).
        input_size: network input (width, height) in pixels: (288, 384) for rtmpose-x,
            (192, 256) for rtmpose-m (``gbf.models`` pins record it).
        backend, device: rtmlib backend ("onnxruntime", "opencv", "openvino") and device
            ("cpu", "cuda", ...).
        model_id: written to ``pose.parquet``; defaults to the file stem, which is the pinned
            model_id for files fetched by gbf.models.
        input_channels: "rgb" passes frames unchanged, "bgr" reverses the channels first
            (what rtmlib's own examples feed); see the module docstring.
        runner: a :class:`PoseRunner`; when given, the network is not loaded (tests use a
            fake runner). Otherwise an :class:`RtmlibRunner` is built.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int],
        backend: str = "onnxruntime",
        device: str = "cpu",
        *,
        model_id: str | None = None,
        input_channels: InputChannels = DEFAULT_INPUT_CHANNELS,
        runner: PoseRunner | None = None,
    ) -> None:
        problems: list[str] = []
        size = _input_size(input_size, problems)
        if input_channels not in ("rgb", "bgr"):
            problems.append(f"input_channels: expected 'rgb' or 'bgr', got {input_channels!r}")
        if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
            problems.append(f"model_id: expected a non-empty string, got {model_id!r}")
        if problems or size is None:
            raise PoseInputError("PoseEstimator", problems)
        self.model_path = Path(model_path)
        self.model_id = model_id if model_id is not None else self.model_path.stem
        self.input_size = size
        self.input_channels: InputChannels = input_channels
        self.runner: PoseRunner = (
            runner if runner is not None else RtmlibRunner(self.model_path, size, backend, device)
        )

    @classmethod
    def from_pinned(
        cls,
        model: str | None = None,
        models_dir: str | Path | None = None,
        *,
        backend: str = "onnxruntime",
        device: str = "cpu",
        verify_hash: bool = True,
        input_channels: InputChannels = DEFAULT_INPUT_CHANNELS,
    ) -> PoseEstimator:
        """An estimator for a pinned model (id or alias; default rtmpose-x), hash-checked."""
        from gbf import models

        pin = models.resolve_pin(model if model is not None else models.DEFAULT_MODEL_ID)
        path = models.onnx_path(pin.model_id, models_dir, verify_hash=verify_hash)
        return cls(
            path,
            pin.input_size,
            backend,
            device,
            model_id=pin.model_id,
            input_channels=input_channels,
        )

    def estimate(self, frame_rgb: np.ndarray, boxes: Any) -> np.ndarray:
        """Keypoints for each box: array (k, 26, 3) of [u, v, score], frame pixels.

        ``boxes`` is (k, 4) [x1, y1, x2, y2] in the frame's pixels. With k = 0 the backend is
        not called and an empty (0, 26, 3) array comes back (rtmlib would otherwise run on
        the whole frame). u and v are NaN where the score is <= 0. Raises PoseInputError
        listing every problem with the frame or the boxes, and PoseRuntimeError when the
        backend returns malformed arrays.
        """
        problems: list[str] = []
        frame = _checked_frame(frame_rgb, problems)
        array = _checked_boxes(boxes, problems)
        if problems or frame is None or array is None:
            raise PoseInputError("PoseEstimator.estimate", problems)
        count = array.shape[0]
        if count == 0:
            return np.empty((0, N_KEYPOINTS, 3), dtype=np.float64)
        image = frame if self.input_channels == "rgb" else np.ascontiguousarray(frame[..., ::-1])
        keypoints, scores = self.runner(image, array)
        keypoints = np.asarray(keypoints, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        if keypoints.shape != (count, N_KEYPOINTS, 2) or scores.shape != (count, N_KEYPOINTS):
            raise PoseRuntimeError(
                f"pose backend returned keypoints {keypoints.shape} and scores {scores.shape}"
                f" for {count} box(es); expected ({count}, {N_KEYPOINTS}, 2) and"
                f" ({count}, {N_KEYPOINTS})"
            )
        if not np.isfinite(scores).all():
            raise PoseRuntimeError("pose backend returned non-finite scores")
        result = np.concatenate([keypoints, scores[..., None]], axis=-1)
        result[scores <= 0.0, :2] = np.nan
        return result

    def estimate_frames(
        self,
        frames: Iterable[tuple[int, np.ndarray]],
        boxes_by_frame: Mapping[int, Sequence[tuple[int, Sequence[float]]]] | pd.DataFrame,
        *,
        require_all: bool = True,
    ) -> pd.DataFrame:
        """Run :meth:`estimate` over frames and return the ``pose.parquet`` table.

        Args:
            frames: ``(frame_index, frame_rgb)`` pairs in any order, e.g.
                ``gbf.ingest.iter_frames(path, start, stop)``. Frames without boxes are
                skipped; iteration stops once every frame with boxes has been seen.
            boxes_by_frame: either ``{frame: [(track_id, (x1, y1, x2, y2)), ...]}`` or a
                DataFrame with columns frame, track_id, x1, y1, x2, y2 (tracks.parquet).
                A track_id may appear once per frame, except -1 (no track).
            require_all: raise PoseInputError when a frame with boxes never came out of
                ``frames``.

        Returns a DataFrame with :data:`POSE_COLUMNS`, sorted by frame, track_id and box,
        with dtypes frame/track_id int64, model_id str, boxes float64, keypoints a list of
        78 floats (26 x [u, v, score]) and mean_score float64.
        """
        wanted = _boxes_by_frame(boxes_by_frame)
        remaining = set(wanted)
        rows: list[dict[str, Any]] = []
        for index, frame in frames:
            entries = wanted.get(int(index))
            if not entries or int(index) not in remaining:
                continue
            remaining.discard(int(index))
            poses = self.estimate(frame, np.array([box for _, box in entries], dtype=np.float64))
            for (track_id, box), pose in zip(entries, poses, strict=True):
                rows.append(
                    {
                        "frame": int(index),
                        "track_id": track_id,
                        "model_id": self.model_id,
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                        "keypoints": pose.reshape(-1).tolist(),
                        "mean_score": float(np.mean(pose[:, 2])),
                    }
                )
            if not remaining:
                break
        if remaining and require_all:
            missing = sorted(remaining)
            shown = ", ".join(str(index) for index in missing[:20])
            more = f" and {len(missing) - 20} more" if len(missing) > 20 else ""
            raise PoseInputError(
                "PoseEstimator.estimate_frames",
                [f"frames with boxes that the frame iterator never yielded: {shown}{more}"],
            )
        return pose_table(rows)


def _boxes_by_frame(
    boxes_by_frame: Mapping[int, Sequence[tuple[int, Sequence[float]]]] | pd.DataFrame,
) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    """Validated ``{frame: [(track_id, box), ...]}`` with every problem reported at once."""
    problems: list[str] = []
    result: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    if isinstance(boxes_by_frame, pd.DataFrame):
        needed = ["frame", "track_id", "x1", "y1", "x2", "y2"]
        absent = [name for name in needed if name not in boxes_by_frame.columns]
        if absent:
            raise PoseInputError("boxes_by_frame", [f"missing column(s): {', '.join(absent)}"])
        items: Iterable[tuple[Any, Any]] = (
            (
                frame,
                list(
                    zip(
                        group["track_id"].tolist(),
                        group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64),
                        strict=True,
                    )
                ),
            )
            for frame, group in boxes_by_frame.groupby("frame", sort=True)
        )
    elif isinstance(boxes_by_frame, Mapping):
        items = boxes_by_frame.items()
    else:
        raise PoseInputError(
            "boxes_by_frame",
            [f"expected a mapping or a DataFrame, got {type(boxes_by_frame).__name__}"],
        )
    for frame, entries in items:
        if isinstance(frame, bool) or not isinstance(frame, int | np.integer) or frame < 0:
            problems.append(f"frame {frame!r}: frame indices must be integers >= 0")
            continue
        seen: set[int] = set()
        checked: list[tuple[int, tuple[float, float, float, float]]] = []
        for position, entry in enumerate(entries):
            try:
                track_id, box = entry
            except (TypeError, ValueError):
                problems.append(f"frame {frame} entry {position}: expected (track_id, box)")
                continue
            if isinstance(track_id, bool) or not isinstance(track_id, int | np.integer):
                problems.append(f"frame {frame} entry {position}: track_id {track_id!r}")
                continue
            box_problems: list[str] = []
            array = _checked_boxes([box], box_problems, name=f"frame {frame} track {track_id}")
            if box_problems or array is None or array.shape[0] != 1:
                problems.extend(box_problems or [f"frame {frame} track {track_id}: bad box"])
                continue
            if int(track_id) != UNKNOWN_TRACK_ID and int(track_id) in seen:
                problems.append(f"frame {frame}: track_id {int(track_id)} appears twice")
                continue
            seen.add(int(track_id))
            x1, y1, x2, y2 = (float(v) for v in array[0])
            checked.append((int(track_id), (x1, y1, x2, y2)))
        if checked:
            checked.sort(key=lambda item: (item[0], item[1]))
            result[int(frame)] = checked
    if problems:
        raise PoseInputError("boxes_by_frame", problems)
    return result


def pose_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """``pose.parquet`` rows as a DataFrame with :data:`POSE_COLUMNS` and fixed dtypes.

    Rows are sorted by (frame, track_id, x1, y1, x2, y2), so the order never depends on the
    order the boxes were given in.
    """
    frame = pd.DataFrame(list(rows), columns=list(POSE_COLUMNS))
    frame = frame.astype(
        {
            "frame": "int64",
            "track_id": "int64",
            "x1": "float64",
            "y1": "float64",
            "x2": "float64",
            "y2": "float64",
            "mean_score": "float64",
        }
    )
    frame["model_id"] = frame["model_id"].astype(str)
    frame = frame.sort_values(["frame", "track_id", "x1", "y1", "x2", "y2"], kind="stable")
    return frame.reset_index(drop=True)


def keypoints_array(pose: pd.DataFrame) -> np.ndarray:
    """The ``keypoints`` column of a pose table as an (n, 26, 3) float64 array."""
    values = [np.asarray(item, dtype=np.float64) for item in pose["keypoints"]]
    bad = [row for row, item in enumerate(values) if item.shape != (N_KEYPOINTS * 3,)]
    if bad:
        raise PoseInputError(
            "keypoints_array", [f"row {row}: expected {N_KEYPOINTS * 3} values" for row in bad]
        )
    if not values:
        return np.empty((0, N_KEYPOINTS, 3), dtype=np.float64)
    return np.stack(values).reshape(-1, N_KEYPOINTS, 3)


def write_pose_parquet(pose: pd.DataFrame, path: str | Path) -> None:
    """Write a pose table with :data:`POSE_SCHEMA` (the columns are checked first)."""
    absent = [name for name in POSE_COLUMNS if name not in pose.columns]
    if absent:
        raise PoseInputError("write_pose_parquet", [f"missing column(s): {', '.join(absent)}"])
    keypoints_array(pose)  # every row must hold 78 values
    table = pa.Table.from_pandas(
        pose[list(POSE_COLUMNS)], schema=POSE_SCHEMA, preserve_index=False
    ).replace_schema_metadata(None)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)


def read_pose_parquet(path: str | Path) -> pd.DataFrame:
    """Read ``pose.parquet`` back into the form :func:`pose_table` returns."""
    table = pq.read_table(Path(path))
    frame = table.to_pandas()
    frame["keypoints"] = [list(map(float, item)) for item in frame["keypoints"]]
    return pose_table(frame.to_dict("records"))


def draw_pose(
    image_rgb: np.ndarray,
    keypoints: np.ndarray,
    *,
    score_threshold: float = 0.3,
    radius: float = 3.0,
    width: int = 2,
) -> np.ndarray:
    """A copy of ``image_rgb`` with Halpe-26 skeletons drawn (Pillow, RGB colours).

    ``keypoints`` is (26, 3) or (k, 26, 3) [u, v, score] in the image's pixel grid; points
    and edges whose scores are below ``score_threshold`` (or NaN) are not drawn.
    """
    problems: list[str] = []
    image = _checked_frame(image_rgb, problems)
    poses = np.asarray(keypoints, dtype=np.float64)
    if poses.ndim == 2:
        poses = poses[None]
    if poses.ndim != 3 or poses.shape[1:] != (N_KEYPOINTS, 3):
        problems.append(f"keypoints: expected (26, 3) or (k, 26, 3), got {poses.shape}")
    if problems or image is None:
        raise PoseInputError("draw_pose", problems)
    canvas = Image.fromarray(np.ascontiguousarray(image))
    pen = ImageDraw.Draw(canvas)
    for pose in poses:
        shown = np.isfinite(pose).all(axis=1) & (pose[:, 2] >= score_threshold)
        for (a, b), color in zip(HALPE26_EDGES, HALPE26_EDGE_COLORS_RGB, strict=True):
            if shown[a] and shown[b]:
                pen.line(
                    [(pose[a, 0], pose[a, 1]), (pose[b, 0], pose[b, 1])], fill=color, width=width
                )
        for index in np.flatnonzero(shown).tolist():
            u, v = pose[index, 0], pose[index, 1]
            pen.ellipse(
                [(u - radius, v - radius), (u + radius, v + radius)],
                fill=HALPE26_KEYPOINT_COLORS_RGB[index],
            )
    return np.asarray(canvas)
