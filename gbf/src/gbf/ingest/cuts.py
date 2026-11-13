"""Hard-cut detection from colour histograms of consecutive frames.

Each frame is subsampled to roughly :data:`SAMPLE_ROWS` rows (a fixed pixel stride), its
pixels are binned into a joint RGB histogram with :data:`BINS_PER_CHANNEL` bins per channel,
and the histogram is normalised to sum to 1. The score of frame ``i`` is the L1 distance
between the histograms of frames ``i - 1`` and ``i``; it lies in ``[0, 2]``. Frame ``i``
starts a new shot when its score exceeds the threshold.

:data:`DEFAULT_CUT_THRESHOLD` is an empirical starting value, not a validated one. It has to
be validated in P1b on GSR and on the private development clip (five hard cuts by frame
inspection; see prereg/README.md, erratum E1). Known limits: a cut between two shots of
the same pitch can score low because the colours are similar (on the development clip one
such cut scored 0.52, only just above 0.5), and gradual
transitions (dissolves, wipes) are not detected because consecutive frames differ little.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

#: Empirical starting threshold on the L1 histogram distance (range 0..2); validate in P1b.
DEFAULT_CUT_THRESHOLD = 0.5
#: Histogram bins per RGB channel (a power of two); the joint histogram has its cube.
BINS_PER_CHANNEL = 8
#: Frames are subsampled with an integer stride to about this many rows before binning.
SAMPLE_ROWS = 72

FrameItem = np.ndarray | tuple[int, np.ndarray]


def cut_detection_settings(threshold: float = DEFAULT_CUT_THRESHOLD) -> dict[str, Any]:
    """Method and parameters of :func:`detect_cuts`, as recorded in ``video.json``."""
    return {
        "method": "rgb_histogram_l1",
        "threshold": float(_check_threshold(threshold)),
        "bins_per_channel": BINS_PER_CHANNEL,
        "sample_rows": SAMPLE_ROWS,
        "validated": False,
    }


def _check_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, int | float | np.floating):
        raise ValueError(f"threshold must be a number, got {threshold!r}")
    value = float(threshold)
    if not math.isfinite(value) or not 0.0 < value < 2.0:
        raise ValueError(f"threshold must be a finite number between 0 and 2, got {threshold!r}")
    return value


def frame_histogram(frame: np.ndarray) -> np.ndarray:
    """Normalised joint RGB histogram (``BINS_PER_CHANNEL ** 3`` float64 bins) of a frame."""
    problems = []
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"frame must be a numpy array, got {type(frame).__name__}")
    if frame.dtype != np.uint8:
        problems.append(f"dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        problems.append(f"shape must be (height, width, 3), got {frame.shape}")
    elif frame.shape[0] < 1 or frame.shape[1] < 1:
        problems.append(f"frame must not be empty, got shape {frame.shape}")
    if problems:
        raise ValueError("frame_histogram: " + "; ".join(problems))
    step = max(1, min(frame.shape[0], frame.shape[1]) // SAMPLE_ROWS)
    sample = frame[::step, ::step].astype(np.intp) // (256 // BINS_PER_CHANNEL)
    codes = (sample[..., 0] * BINS_PER_CHANNEL + sample[..., 1]) * BINS_PER_CHANNEL + sample[..., 2]
    counts = np.bincount(codes.ravel(), minlength=BINS_PER_CHANNEL**3).astype(np.float64)
    return counts / counts.sum()


def _indexed(frames: Iterable[FrameItem]) -> Iterator[tuple[int, np.ndarray]]:
    """Accept bare frames (indexed 0, 1, ...) or ``(index, frame)`` pairs from iter_frames."""
    mode: str | None = None
    expected: int | None = None
    for position, item in enumerate(frames):
        if isinstance(item, tuple):
            if mode == "bare":
                raise ValueError(f"item {position}: mixes (index, frame) pairs with bare frames")
            mode = "pairs"
            if len(item) != 2:
                raise ValueError(f"item {position}: expected an (index, frame) pair")
            index, frame = item
            if isinstance(index, bool) or not isinstance(index, int | np.integer):
                raise ValueError(f"item {position}: frame index must be an integer, got {index!r}")
            index = int(index)
            if expected is not None and index != expected:
                raise ValueError(
                    f"frames must be consecutive: index {index} follows {expected - 1}; "
                    "a cut is only defined between neighbouring frames"
                )
        else:
            if mode == "pairs":
                raise ValueError(f"item {position}: mixes bare frames with (index, frame) pairs")
            mode = "bare"
            index, frame = position, item
        expected = index + 1
        yield index, frame


def cut_scores(frames: Iterable[FrameItem]) -> list[tuple[int, float]]:
    """``(i, d)`` for every frame after the first: ``d`` is the L1 histogram distance (0..2)
    between frames ``i - 1`` and ``i``. Useful for choosing and validating the threshold."""
    scores: list[tuple[int, float]] = []
    previous: np.ndarray | None = None
    for index, frame in _indexed(frames):
        histogram = frame_histogram(frame)
        if previous is not None:
            scores.append((index, float(np.abs(histogram - previous).sum())))
        previous = histogram
    return scores


def detect_cuts(frames: Iterable[FrameItem], threshold: float = DEFAULT_CUT_THRESHOLD) -> list[int]:
    """Indices of frames that start a new shot (the first frame is never reported).

    ``frames`` is either a sequence of ``(height, width, 3)`` uint8 RGB arrays, indexed from
    0, or the ``(index, frame)`` pairs of :func:`gbf.ingest.frames.iter_frames`, whose
    indices are then reported (they must be consecutive). ``threshold`` is the L1 distance
    above which a frame counts as a cut; see the module docstring for its status.
    """
    limit = _check_threshold(threshold)
    return [index for index, score in cut_scores(frames) if score > limit]
