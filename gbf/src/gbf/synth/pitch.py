"""Markings of a 105 x 68 m pitch (Laws of the Game, Law 1) in the SoccerNet world frame.

Frame (see ``gbf.geometry``): metres, origin at the centre spot, x along the touchlines in
[-52.5, 52.5], y across the pitch with the FAR touchline at y = -34 and the near (main
camera) touchline at y = +34, z pointing DOWN (grass at z = 0, the crossbar at z = -2.44).
"Left" means x < 0.

Every marking is a polyline of points on the line's centreline; arcs are sampled at a fixed
angular step that always includes their end points and every multiple of 90 degrees, so the
named landmarks below lie exactly on a polyline vertex or segment. Line width is not
modelled here: the renderer draws lines with a pixel width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
HALF_LENGTH_M = PITCH_LENGTH_M / 2.0
HALF_WIDTH_M = PITCH_WIDTH_M / 2.0
CENTRE_CIRCLE_RADIUS_M = 9.15
GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44
GOAL_AREA_DEPTH_M = 5.5
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_MARK_DISTANCE_M = 11.0
PENALTY_ARC_RADIUS_M = 9.15
CORNER_ARC_RADIUS_M = 1.0
SPOT_RADIUS_M = 0.11
"""Radius of the centre and penalty marks (22 cm across)."""

_HALF_GOAL = GOAL_WIDTH_M / 2.0
_GOAL_AREA_HALF_WIDTH = _HALF_GOAL + GOAL_AREA_DEPTH_M
_PENALTY_AREA_HALF_WIDTH = _HALF_GOAL + PENALTY_AREA_DEPTH_M
_PENALTY_AREA_X = HALF_LENGTH_M - PENALTY_AREA_DEPTH_M
_GOAL_AREA_X = HALF_LENGTH_M - GOAL_AREA_DEPTH_M
_PENALTY_MARK_X = HALF_LENGTH_M - PENALTY_MARK_DISTANCE_M


@dataclass(frozen=True)
class Marking:
    """One marking: a polyline (N, 3) of world points in metres along the line's centreline.

    ``on_ground`` is False for the goal frames (posts and crossbar), which stand above the
    grass (z < 0).
    """

    name: str
    points: np.ndarray
    on_ground: bool = True

    def __post_init__(self) -> None:
        array = np.array(self.points, dtype=np.float64)
        problems = []
        if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 2:
            problems.append(f"points must have shape (N >= 2, 3), got {array.shape}")
        elif not np.isfinite(array).all():
            problems.append("points must be finite")
        if problems:
            raise ValueError(f"Marking {self.name!r}: " + "; ".join(problems))
        array.setflags(write=False)
        object.__setattr__(self, "points", array)


def _segment(name: str, start: tuple[float, float], end: tuple[float, float]) -> Marking:
    return Marking(name, np.array([[start[0], start[1], 0.0], [end[0], end[1], 0.0]]))


_SNAP_DEG = 1e-9
_AXIS_UNIT = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def arc_angles(start_deg: float, stop_deg: float, step_deg: float) -> list[float]:
    """Angles in degrees from ``start_deg`` to ``stop_deg`` (both included, in that order),
    at most ``step_deg`` apart, always including every multiple of 90 degrees in between."""
    if not step_deg > 0.0:
        raise ValueError(f"step_deg must be > 0, got {step_deg}")
    count = max(1, math.ceil(abs(stop_deg - start_deg) / step_deg))
    angles: set[float] = set()
    for value in np.linspace(start_deg, stop_deg, count + 1).tolist():
        nearest = round(value / 90.0) * 90.0
        angles.add(float(nearest) if abs(value - nearest) <= _SNAP_DEG else float(value))
    low, high = sorted((start_deg, stop_deg))
    first_quarter = math.ceil(low / 90.0 - _SNAP_DEG)
    last_quarter = math.floor(high / 90.0 + _SNAP_DEG)
    angles.update(90.0 * quarter for quarter in range(first_quarter, last_quarter + 1))
    ordered = sorted(angles)
    return ordered if stop_deg >= start_deg else ordered[::-1]


def _arc(
    name: str,
    centre: tuple[float, float],
    radius: float,
    start_deg: float,
    stop_deg: float,
    step_deg: float,
) -> Marking:
    points = []
    for angle in arc_angles(start_deg, stop_deg, step_deg):
        if angle % 90.0 == 0.0:  # exact points on the axes (cos and sin are not exact there)
            dx, dy = _AXIS_UNIT[int(angle // 90.0) % 4]
        else:
            dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        points.append((centre[0] + radius * dx, centre[1] + radius * dy, 0.0))
    return Marking(name, np.array(points))


def penalty_arc_half_angle_deg() -> float:
    """Half the opening angle of a penalty arc (degrees), about 53.13: the part of the
    9.15 m circle around the penalty mark that lies outside the penalty area."""
    inside = PENALTY_AREA_DEPTH_M - PENALTY_MARK_DISTANCE_M
    return math.degrees(math.acos(inside / PENALTY_ARC_RADIUS_M))


def pitch_markings(arc_step_deg: float = 2.0) -> tuple[Marking, ...]:
    """Every painted line of the pitch, centre and penalty marks excluded (see spots())."""
    L, W = HALF_LENGTH_M, HALF_WIDTH_M
    markings = [
        _segment("touchline_far", (-L, -W), (L, -W)),
        _segment("touchline_near", (-L, W), (L, W)),
        _segment("goal_line_left", (-L, -W), (-L, W)),
        _segment("goal_line_right", (L, -W), (L, W)),
        _segment("halfway_line", (0.0, -W), (0.0, W)),
        _arc("centre_circle", (0.0, 0.0), CENTRE_CIRCLE_RADIUS_M, 0.0, 360.0, arc_step_deg),
    ]
    for side, sign in (("left", -1.0), ("right", 1.0)):
        x_goal = sign * L
        x_penalty = sign * _PENALTY_AREA_X
        x_goal_area = sign * _GOAL_AREA_X
        markings += [
            _segment(
                f"penalty_area_{side}_front",
                (x_penalty, -_PENALTY_AREA_HALF_WIDTH),
                (x_penalty, _PENALTY_AREA_HALF_WIDTH),
            ),
            _segment(
                f"penalty_area_{side}_far",
                (x_goal, -_PENALTY_AREA_HALF_WIDTH),
                (x_penalty, -_PENALTY_AREA_HALF_WIDTH),
            ),
            _segment(
                f"penalty_area_{side}_near",
                (x_goal, _PENALTY_AREA_HALF_WIDTH),
                (x_penalty, _PENALTY_AREA_HALF_WIDTH),
            ),
            _segment(
                f"goal_area_{side}_front",
                (x_goal_area, -_GOAL_AREA_HALF_WIDTH),
                (x_goal_area, _GOAL_AREA_HALF_WIDTH),
            ),
            _segment(
                f"goal_area_{side}_far",
                (x_goal, -_GOAL_AREA_HALF_WIDTH),
                (x_goal_area, -_GOAL_AREA_HALF_WIDTH),
            ),
            _segment(
                f"goal_area_{side}_near",
                (x_goal, _GOAL_AREA_HALF_WIDTH),
                (x_goal_area, _GOAL_AREA_HALF_WIDTH),
            ),
        ]
        half = penalty_arc_half_angle_deg()
        towards_centre = 0.0 if sign < 0 else 180.0
        markings.append(
            _arc(
                f"penalty_arc_{side}",
                (sign * _PENALTY_MARK_X, 0.0),
                PENALTY_ARC_RADIUS_M,
                towards_centre - half,
                towards_centre + half,
                arc_step_deg,
            )
        )
    corners = (
        ("corner_arc_far_left", (-L, -W), 0.0, 90.0),
        ("corner_arc_far_right", (L, -W), 90.0, 180.0),
        ("corner_arc_near_right", (L, W), 180.0, 270.0),
        ("corner_arc_near_left", (-L, W), 270.0, 360.0),
    )
    for name, centre, start, stop in corners:
        markings.append(_arc(name, centre, CORNER_ARC_RADIUS_M, start, stop, arc_step_deg * 5))
    return tuple(markings)


def goal_frames() -> tuple[Marking, ...]:
    """Posts and crossbars as polylines (post foot, post top, other post top, other foot)."""
    frames = []
    for side, sign in (("left", -1.0), ("right", 1.0)):
        x = sign * HALF_LENGTH_M
        top = -GOAL_HEIGHT_M
        points = np.array(
            [
                [x, -_HALF_GOAL, 0.0],
                [x, -_HALF_GOAL, top],
                [x, _HALF_GOAL, top],
                [x, _HALF_GOAL, 0.0],
            ]
        )
        frames.append(Marking(f"goal_{side}", points, on_ground=False))
    return tuple(frames)


def spots() -> dict[str, np.ndarray]:
    """Centre and penalty marks (discs of radius SPOT_RADIUS_M on the grass), by name."""
    return {
        "centre_spot": np.array([0.0, 0.0, 0.0]),
        "penalty_spot_left": np.array([-_PENALTY_MARK_X, 0.0, 0.0]),
        "penalty_spot_right": np.array([_PENALTY_MARK_X, 0.0, 0.0]),
    }


def landmarks() -> dict[str, np.ndarray]:
    """Named world points that lie exactly on a marking's centreline (or a spot centre).

    A camera that rendered a frame must project each visible landmark onto a line pixel of
    that frame; the L0 smoke run checks this after a calibration round trip.
    """
    L, W, R = HALF_LENGTH_M, HALF_WIDTH_M, CENTRE_CIRCLE_RADIUS_M
    points: dict[str, tuple[float, float, float]] = {
        "halfway_far": (0.0, -W, 0.0),
        "halfway_near": (0.0, W, 0.0),
        "centre_circle_far": (0.0, -R, 0.0),
        "centre_circle_near": (0.0, R, 0.0),
        "centre_circle_left": (-R, 0.0, 0.0),
        "centre_circle_right": (R, 0.0, 0.0),
        "corner_far_left": (-L, -W, 0.0),
        "corner_far_right": (L, -W, 0.0),
        "corner_near_left": (-L, W, 0.0),
        "corner_near_right": (L, W, 0.0),
    }
    for side, sign in (("left", -1.0), ("right", 1.0)):
        points[f"penalty_area_{side}_far_corner"] = (
            sign * _PENALTY_AREA_X,
            -_PENALTY_AREA_HALF_WIDTH,
            0.0,
        )
        points[f"penalty_area_{side}_near_corner"] = (
            sign * _PENALTY_AREA_X,
            _PENALTY_AREA_HALF_WIDTH,
            0.0,
        )
        points[f"goal_area_{side}_far_corner"] = (sign * _GOAL_AREA_X, -_GOAL_AREA_HALF_WIDTH, 0.0)
        points[f"goal_area_{side}_near_corner"] = (sign * _GOAL_AREA_X, _GOAL_AREA_HALF_WIDTH, 0.0)
        points[f"crossbar_{side}_far_end"] = (sign * L, -_HALF_GOAL, -GOAL_HEIGHT_M)
        points[f"crossbar_{side}_near_end"] = (sign * L, _HALF_GOAL, -GOAL_HEIGHT_M)
    named = {name: np.array(point) for name, point in points.items()}
    named.update(spots())
    return dict(sorted(named.items()))
