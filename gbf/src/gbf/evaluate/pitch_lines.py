"""Ground lines of the SoccerNet pitch model, by the line names of SN-GSR pitch annotations.

Metres in the SoccerNet world frame: origin at the centre spot, x along the touchlines,
y = -34 at the far ("top") touchline, z = 0 on the ground. Dimensions checked against
SoccerNet 0.1.55 (``SoccerNet/Evaluation/utils_calibration.py``, class ``SoccerPitch``, MIT)
and written here, not imported. Goal posts, crossbars and the "unknown" lines are left out:
they are off the ground or have no fixed place.
"""

from __future__ import annotations

import math

import numpy as np

HALF_LENGTH = 52.5
HALF_WIDTH = 34.0
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_HALF_WIDTH = 40.32 / 2
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_HALF_WIDTH = 18.32 / 2
CIRCLE_RADIUS = 9.15
PENALTY_MARK_DISTANCE = 11.0
STEP_M = 0.05


def _segment(a: tuple[float, float], b: tuple[float, float], step: float) -> np.ndarray:
    start, stop = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = max(2, math.ceil(float(np.linalg.norm(stop - start)) / step) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return start + t * (stop - start)


def _arc(centre: tuple[float, float], start: float, stop: float, step: float) -> np.ndarray:
    n = max(2, math.ceil(CIRCLE_RADIUS * abs(stop - start) / step) + 1)
    angle = np.linspace(start, stop, n)
    return np.column_stack(
        [centre[0] + CIRCLE_RADIUS * np.cos(angle), centre[1] + CIRCLE_RADIUS * np.sin(angle)]
    )


def ground_lines(step: float = STEP_M) -> dict[str, np.ndarray]:
    """Line name -> (n, 2) ground points (x, y) along it, at most ``step`` metres apart."""
    half_l, half_w = HALF_LENGTH, HALF_WIDTH
    lines = {
        "Side line top": _segment((-half_l, -half_w), (half_l, -half_w), step),
        "Side line bottom": _segment((-half_l, half_w), (half_l, half_w), step),
        "Side line left": _segment((-half_l, -half_w), (-half_l, half_w), step),
        "Side line right": _segment((half_l, -half_w), (half_l, half_w), step),
        "Middle line": _segment((0.0, -half_w), (0.0, half_w), step),
        "Circle central": _arc((0.0, 0.0), 0.0, 2 * math.pi, step),
    }
    boxes = (
        ("Big rect.", PENALTY_AREA_LENGTH, PENALTY_AREA_HALF_WIDTH),
        ("Small rect.", GOAL_AREA_LENGTH, GOAL_AREA_HALF_WIDTH),
    )
    for side, sign in (("left", -1.0), ("right", 1.0)):
        goal_x = sign * half_l
        for name, length, half in boxes:
            inner_x = goal_x - sign * length
            lines[f"{name} {side} top"] = _segment((goal_x, -half), (inner_x, -half), step)
            lines[f"{name} {side} bottom"] = _segment((goal_x, half), (inner_x, half), step)
            lines[f"{name} {side} main"] = _segment((inner_x, -half), (inner_x, half), step)
        mark_x = goal_x - sign * PENALTY_MARK_DISTANCE
        half_angle = math.acos((PENALTY_AREA_LENGTH - PENALTY_MARK_DISTANCE) / CIRCLE_RADIUS)
        towards_centre = 0.0 if sign < 0 else math.pi
        lines[f"Circle {side}"] = _arc(
            (mark_x, 0.0), towards_centre - half_angle, towards_centre + half_angle, step
        )
    return lines
