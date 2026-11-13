"""Conversion between the SoccerNet world frame (z down) and the internal pitch_zup frame.

SoccerNet: metres, origin at the centre spot, x along the touchlines, the far touchline at
y = -34, z pointing down (grass at z = 0, points above it have z < 0).

pitch_zup = (x, -y, -z): a 180 degree rotation about the x axis. Its matrix diag(1, -1, -1)
has determinant +1, so nothing is mirrored: a right-handed frame stays right-handed and a
left foot stays a left foot. In pitch_zup z points up and the far touchline is at y = +34.
The rotation is its own inverse, and points and direction vectors convert the same way.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from gbf.geometry._checks import GeometryInputError, trailing_dim

PITCH_ZUP_FROM_SOCCERNET: np.ndarray = np.diag([1.0, -1.0, -1.0])
"""Rotation taking SoccerNet coordinates to pitch_zup (and back: it is its own inverse)."""
PITCH_ZUP_FROM_SOCCERNET.setflags(write=False)

_AXIS_SIGNS = np.array([1.0, -1.0, -1.0])


def _rotate_about_x(points: ArrayLike, where: str) -> np.ndarray:
    problems: list[str] = []
    array = trailing_dim("points", points, 3, problems)
    if problems or array is None:
        raise GeometryInputError(where, problems)
    return array * _AXIS_SIGNS


def soccernet_to_pitch_zup(points: ArrayLike) -> np.ndarray:
    """SoccerNet (x, y, z) with z down to pitch_zup (x, -y, -z) with z up; shape (..., 3).

    Works for points and direction vectors alike. Non-finite entries pass through unchanged.
    """
    return _rotate_about_x(points, "soccernet_to_pitch_zup")


def pitch_zup_to_soccernet(points: ArrayLike) -> np.ndarray:
    """pitch_zup (x, y, z) with z up back to SoccerNet (x, -y, -z) with z down; shape (..., 3).

    The same 180 degree rotation about x as soccernet_to_pitch_zup (it is an involution).
    """
    return _rotate_about_x(points, "pitch_zup_to_soccernet")
