"""Optional RGB corroboration for row isolation; colour is not person identity.

The caller supplies canonical frames. No model, team label or annotation is read here.
This uses the existing upper-central CIELAB measurement, allowing an upright arms-out
box (height >= width). Reference rows still have to pass relink's clean-box rule.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np
import pandas as pd

from gbf import team_check

MEASUREMENT_PARAMS = replace(team_check.DEFAULT_PARAMS, min_aspect=1.0)
COLOUR_MARGIN = team_check.DEFAULT_PARAMS.margin
MAX_COLOUR_DISTANCE = team_check.DEFAULT_PARAMS.margin


def measure_shirt_colours(
    tracks: pd.DataFrame,
    frames: Iterable[tuple[int, np.ndarray]],
    *,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """One CIELAB row per track row, NaN when the crop is unavailable.

    Boxes are in ``frame_size`` coordinates; frames can be a different resolution, as
    allowed by plugin exports. Each frame is kept only while measuring its rows.
    Missing frames are errors, not silent evidence loss. An unusable crop is unknown.
    """
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("shirt colours: frame_size must be positive")
    result = np.full((len(tracks), 3), np.nan)
    boxes = tracks[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
    indices = tracks["frame"].to_numpy(dtype=np.int64)
    requested = {int(f): np.flatnonzero(indices == f) for f in np.unique(indices)}
    if not requested:
        return result
    for frame, image in frames:
        rows = requested.pop(int(frame), None)
        if rows is not None:
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("shirt colours: frames must be uint8 RGB")
            scale = np.array([image.shape[1] / width, image.shape[0] / height] * 2)
            for row in rows:
                colour = team_check.shirt_colour(image, boxes[row] * scale, MEASUREMENT_PARAMS)
                if colour is not None:
                    result[row] = colour
        if not requested:
            break
    if requested:
        raise ValueError(f"shirt colours: missing requested frame {min(requested)}")
    return result


def checked_colours(raw: np.ndarray | None, n_rows: int) -> np.ndarray | None:
    """Validate row alignment; incomplete rows are unavailable, never guessed."""
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (n_rows, 3):
        raise ValueError(f"shirt_lab: expected ({n_rows}, 3), got {values.shape}")
    if np.isinf(values).any():
        raise ValueError("shirt_lab: infinite colour values")
    values = values.copy()
    values[~np.isfinite(values).all(axis=1)] = np.nan
    return values
