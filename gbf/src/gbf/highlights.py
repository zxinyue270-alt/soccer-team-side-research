"""Highlight cues per frame and the highlight detector (M1, docs/design/m1-highlights.md).

:func:`compute_signals` turns the per-frame artifacts of one clip into one row per frame:
seven cues in [0, 1], their weighted ``score``, and the evidence behind every cue.
:func:`detect_highlights` picks the highlight moments from those rows and returns the
``highlights.json`` structure: a window per highlight, its type, the precise peak frame,
the players involved, the cues that made it (so the choice can be audited) and a Chinese
title and summary with frame-accurate times.

Conventions
    Frames are canonical CFR indices ``0 .. N - 1``, where ``N`` is the number of frames that
    ``shots`` covers (the shots must tile the clip without gaps or overlaps). Seconds are
    ``frame / fps``, and ``fps`` is an exact rational (``"25/1"``, a ``Fraction`` or an int).
    Pitch coordinates are SoccerNet world metres: origin at the centre spot, x along the
    touchlines, goal lines at x = -52.5 and x = +52.5. Team ``"left"`` defends x < 0.
    ``attack_direction`` gives the sign of the x direction each team attacks (default
    ``{"left": 1, "right": -1}``), so a team attacks the goal at x = 52.5 * sign. Speeds are
    m/s; km/h only appear in the Chinese text. Any team label other than ``"left"`` or
    ``"right"`` (None, NaN, ``"nan"``) counts as unknown.

Pitch frames
    Positions in metres feed a cue only on pitch frames: frames inside a shot of type
    ``"wide"`` whose ``fresh_share`` is at least :data:`MIN_PITCH_FRESH_SHARE`, and on which
    the tracks rows are ``pitch_usable`` (the calibration the refine stage kept: a refitted
    frame with a refit error of at most 8 px in a camera group that was not rejected, else
    the plugin's fresh calibration with rep_err_px <= 8, gate G6).
    A close or medium shot can carry "fresh" calibrations fitted to the few lines it shows,
    with a low reprojection error and an implausible camera, so such shots stay image-plane
    only here. Every other frame uses image-plane evidence, never metres.

Ball in sight
    The attack cues ``box_pressure`` and ``ball_goal`` count only near a detected ball: a
    ``ball_track`` row with status ``"detected"`` within :data:`ATTACK_BALL_SEEN_S` seconds
    of the frame, inside the same shot. During a stoppage (a corner or a free kick being
    prepared, an injury) players stand in the box while the ball is out of the picture for
    seconds; that is not an attack. Elsewhere both cues are 0 and their evidence is missing
    (``ev_box_attackers`` 0, ``ev_box_team``, ``ev_ball_goal_m`` and ``ev_ball_team``
    unset); without a ball track that is every frame.

Ball at the box
    The two attack cues count for the type ``attack_in_box`` only where the ball is in the
    penalty area the attacking side attacks, or within :data:`BALL_AT_BOX_MARGIN_M` of it
    (the error a pitch-usable calibration may have there), at some frame within
    :data:`ATTACK_BALL_SEEN_S` in the same shot: players running into the box while the ball
    is still 20 m out are no attack in the box. ``ev_ball_box_m`` holds that distance: on a
    frame where the attack cues name a side (``ev_box_team``, else ``ev_ball_team``), the
    smallest ground distance from the ball (``ball_track`` rows detected or interpolated, on
    pitch frames) to the penalty area that side attacks, over those frames (0 inside, NaN
    without a ball position). ``box_pressure`` is meant to see an attack in the box near the
    ball, so the same condition holds for it in the score: elsewhere it is 0, while its
    evidence (``ev_box_attackers``, ``ev_box_team``) still counts the players in the box.
    ``ball_goal`` counts in the score as it is: it measures the ball's own distance to the
    goal, so a ball in front of the box within 25 m of the goal is what it is meant to see;
    only the type asks for the ball at the box.

Attack direction
    The names ``"left"`` and ``"right"`` come from the tracking plugin, which names its two
    team clusters by their mean pitch x over the clip; :mod:`gbf.side_check` checks them
    against where the teams stand. When it cannot confirm them (``attack_known`` False), the
    goal each team attacks is unknown: ``box_pressure`` and ``ball_goal`` are 0 on every frame
    and their evidence (``ev_box_attackers`` 0, ``ev_box_team``, ``ev_ball_goal_m``,
    ``ev_ball_box_m``) is missing, so no ``attack_in_box`` highlight is found; the other cues
    and evidence, the players nearest the ball and their team (``ev_ball_team``) included, do
    not change.

Cues (float64 in [0, 1]; 0 when their evidence is missing)
    ``cut_emphasis``  broadcast grammar: :data:`CUT_PRE_VALUE` on the last :data:`CUT_PRE_S`
        seconds of a wide shot that is followed by a close or medium shot (the director cut
        away to emphasise what just happened), :data:`CUT_POST_VALUE` on the first
        :data:`FOLLOW_MAX_S` seconds of that following shot.
    ``audio``  ``clip(excitement_z, 0, AUDIO_Z_MAX) / AUDIO_Z_MAX`` (the z-score clipped to
        [0, 4], scaled to [0, 1]); 0 on frames without audio.
    ``speed``  the highest smoothed speed of a player or goalkeeper on a pitch frame (rows of
        ``kinematics`` with ``valid``; speeds above :data:`SPEED_PLAUSIBLE_MAX_MPS` are taken
        as tracking errors and ignored), mapped linearly from 4 m/s (0) to 9 m/s (1).
    ``box_pressure``  outfield players (role ``player``, inside the pitch) of one team inside
        the penalty area they attack (16.5 m deep, 40.32 m wide), on pitch frames with the
        ball in sight and at the box; the larger of the two teams' counts divided by
        :data:`BOX_FULL_ATTACKERS`, capped at 1.
    ``ball_goal``  on pitch frames with the ball in sight and its ground-plane position: the
        player or goalkeeper nearest the ball decides the side in possession; the ball's
        distance to the goal that side attacks maps linearly from 25 m (0) to 11 m, the
        penalty spot (1).
    ``duel``  the nearest pair of opposing players, both near the ball. On pitch frames with
        the ball's ground position: both within :data:`DUEL_BALL_RADIUS_M` of the ball, pair
        distance in metres mapped from 4 m (0) to 1 m (1). Elsewhere, in the image: foot
        points within :data:`DUEL_BALL_RADIUS_H` own box heights of the ball, pair distance in
        mean box heights (one box height stands for about 1.8 m) mapped from 2.0 (0) to 0.5 (1);
        not in shots whose ``median_box_h`` exceeds :data:`DUEL_IMAGE_MAX_BOX_H_PX`.
    ``strike``  ball events nearby in time: ``max(strength * (1 - |frame - event| / H))`` over
        the events, a triangle of half width ``H = STRIKE_HALF_WIDTH_S * fps`` frames. An
        event's strength is its ``strength`` (or ``conf`` or ``score``) clipped to [0, 1]
        when the table has such a column; otherwise, for the events of gbf.ball, it is
        ``min(1, magnitude / STRIKE_FULL_MAGNITUDE_BH_S)`` times a foot weight (1 when the
        nearest player's foot point is within :data:`STRIKE_FOOT_NEAR_H` box heights of the
        ball, falling to 0 at :data:`STRIKE_FOOT_FAR_H`, :data:`STRIKE_UNKNOWN_FOOT_WEIGHT`
        without a player) times :data:`STRIKE_KIND_WEIGHTS` of its ``kind``; each factor is
        1 when its column is absent.
    ``score`` is the weighted mean ``sum(w * cue) / sum(w)`` with :data:`CUE_WEIGHTS`.

Evidence columns (all named ``ev_*``; -1 for a missing id or frame, NaN for a missing
number, None for a missing label): ``ev_pitch`` (bool, the frame is a pitch frame),
``ev_cut_frame``, ``ev_audio_z``, ``ev_speed_mps``, ``ev_speed_id``, ``ev_box_attackers``,
``ev_box_team``, ``ev_ball_near_id`` (player nearest the ball), ``ev_ball_opp_id`` (nearest
player of the other team), ``ev_ball_goal_m``, ``ev_ball_team``, ``ev_ball_box_m`` (see "Ball
at the box"), ``ev_duel_dist``, ``ev_duel_unit`` (``"m"`` or ``"box_h"``), ``ev_duel_id_a``
(team left), ``ev_duel_id_b`` (team right) and ``ev_strike_frame``.

Every weight, mapping and threshold below is a documented starting value chosen from
football facts (pitch dimensions, sprint speeds, typical duel distances), not tuned to a
clip, and none of them is validated yet.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pandas as pd

from gbf.ingest.rates import format_rate, parse_rate
from gbf.kinematics import DEFAULT_SPEED_CHORD_S, chord_half_frames
from gbf.util.clock import ClockError, format_clock

# --------------------------------------------------------------------------- constants

#: The cue columns of the signals table, in order.
CUE_COLUMNS = ("cut_emphasis", "audio", "speed", "box_pressure", "ball_goal", "duel", "strike")
#: Weight of every cue in the score (the score divides by their sum, so it stays in [0, 1]).
CUE_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "cut_emphasis": 1.0,
        "audio": 1.5,
        "speed": 0.75,
        "box_pressure": 1.0,
        "ball_goal": 1.0,
        "duel": 1.0,
        "strike": 1.25,
    }
)
#: Highlight types of the contract.
HIGHLIGHT_TYPES = ("duel_tackle", "attack_in_box", "ball_strike", "sprint", "closeup_reaction")
#: Play cue groups that decide the type, in tie-break priority order.
PLAY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ball_strike", ("strike",)),
    ("attack_in_box", ("box_pressure", "ball_goal")),
    ("duel_tackle", ("duel",)),
    ("sprint", ("speed",)),
)
SHOT_TYPES = ("wide", "medium", "close")
#: Evidence columns of the signals table, in order.
EVIDENCE_COLUMNS = (
    "ev_pitch",
    "ev_cut_frame",
    "ev_audio_z",
    "ev_speed_mps",
    "ev_speed_id",
    "ev_box_attackers",
    "ev_box_team",
    "ev_ball_near_id",
    "ev_ball_opp_id",
    "ev_ball_goal_m",
    "ev_ball_team",
    "ev_ball_box_m",
    "ev_duel_dist",
    "ev_duel_unit",
    "ev_duel_id_a",
    "ev_duel_id_b",
    "ev_strike_frame",
)
SIGNAL_COLUMNS = ("frame", "shot_id", *CUE_COLUMNS, "score", *EVIDENCE_COLUMNS)
#: The cues of the attack_in_box group, which count for that type only with the ball at the box.
ATTACK_CUES = dict(PLAY_GROUPS)["attack_in_box"]

# Broadcast grammar
#: Seconds at the end of a wide shot that a following close or medium shot emphasises.
CUT_PRE_S = 1.5
CUT_PRE_VALUE = 1.0
#: Value on the following close or medium shot (a lower weight than the action before it).
CUT_POST_VALUE = 0.5
#: Longest stretch (s) of a following close shot that gets emphasis or joins a window.
FOLLOW_MAX_S = 10.0
EMPHASIS_FROM_TYPES = frozenset({"wide"})
EMPHASIS_TO_TYPES = frozenset({"close", "medium"})

# Audio
AUDIO_Z_MAX = 4.0

# Players and pitch (SoccerNet world frame, metres)
PLAYER_ROLES = frozenset({"player", "goalkeeper"})
OUTFIELD_ROLES = frozenset({"player"})
PITCH_SHOT_TYPES = frozenset({"wide"})
MIN_PITCH_FRESH_SHARE = 0.5
HALF_LENGTH_M = 52.5
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_HALF_WIDTH_M = 20.16

# Speed (m/s)
SPEED_LOW_MPS = 4.0
SPEED_HIGH_MPS = 9.0
SPEED_PLAUSIBLE_MAX_MPS = 12.0

# Box pressure
BOX_FULL_ATTACKERS = 3

# Attack cues (box pressure, ball and goal)
#: The attack cues need a detected ball (ball_track status "detected"; interpolated rows do
#: not count) within this many seconds of the frame, in the same shot (see "Ball in sight");
#: for box_pressure, and for the type attack_in_box, the ball must also be at the box within
#: this time ("Ball at the box").
ATTACK_BALL_SEEN_S = 1.0
#: The ball counts as at the box this close (m) to the penalty area. The margin stands for
#: the calibration error at the box, not for attacks from outside it: on the development
#: clip's pitch-usable frames with a penalty-area line in view, the 8 px reprojection error
#: that pitch-usable frames may have (gbf.sngs_import.DEFAULT_REP_ERR_MAX_PX) is 1.1 m
#: (median) and at most 1.4 m on the ground along the most foreshortened direction at those
#: lines. A ball in the air is placed where its ray meets the ball-centre plane, farther from
#: the camera than it is: no margin covers that.
BALL_AT_BOX_MARGIN_M = 1.5

# Ball and goal (m)
BALL_GOAL_FAR_M = 25.0
BALL_GOAL_NEAR_M = 11.0

# Duel
DUEL_BALL_RADIUS_M = 4.0
DUEL_NEAR_M = 1.0
DUEL_FAR_M = 4.0
DUEL_BALL_RADIUS_H = 2.0
DUEL_NEAR_H = 0.5
DUEL_FAR_H = 2.0
#: No image-plane duel in a shot whose median person box (shots.median_box_h, 1080-line
#: pixels) is taller than half the frame: such close-ups show upper bodies, whose box
#: bottoms are the frame edge rather than feet.
DUEL_IMAGE_MAX_BOX_H_PX = 540.0

# Strike
STRIKE_HALF_WIDTH_S = 0.6
#: Columns that give an event's strength in [0, 1] directly, in order of preference.
STRIKE_STRENGTH_COLUMNS = ("strength", "conf", "score")
#: Velocity change (box heights per second, as gbf.ball reports ``magnitude``) of a full
#: strike: about 18 m/s for a 1.8 m box, a firm pass or a shot.
STRIKE_FULL_MAGNITUDE_BH_S = 10.0
#: The nearest foot point within this many box heights of the ball: a kick (weight 1) ...
STRIKE_FOOT_NEAR_H = 1.0
#: ... and beyond this, no player kicked it (weight 0: a bounce or a tracking artefact).
STRIKE_FOOT_FAR_H = 2.0
#: Weight of an event whose nearest player is unknown (no player box near the ball).
STRIKE_UNKNOWN_FOOT_WEIGHT = 0.5
#: Weight by event kind; a slow-down is a reception or a block rather than a strike.
STRIKE_KIND_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {"speed_up": 1.0, "direction_change": 1.0, "slow_down": 0.5}
)

# Detection
#: Gaussian smoothing of the score and the cues before peak picking (sigma, seconds).
SMOOTH_SIGMA_S = 0.5
#: Smallest smoothed score a peak needs (starting value, not validated).
MIN_PEAK_SCORE = 0.08
WINDOW_BEFORE_S = 4.0
WINDOW_AFTER_S = 2.0
#: A window edge closer than this (s) to the boundary of its shot snaps to the boundary.
SNAP_S = 0.5
#: The precise peak is searched within this many seconds of the score peak (same shot).
REFINE_S = 1.0
#: Detection cues are averaged over this many seconds (5 frames at 25 fps) before the
#: precise peak is chosen, so a one-frame flicker cannot become the peak.
REFINE_SMOOTH_S = 0.2
#: A play group competes for the type only when its strongest smoothed cue reaches this;
#: when none does, the type is ``closeup_reaction``.
PLAY_EVIDENCE_MIN = 0.2
#: Without the ball near goal, attackers in the box alone make an attack_in_box only when
#: the raw box_pressure reaches this near the peak (two of BOX_FULL_ATTACKERS attackers):
#: one player standing in the box while the ball is out of play is not an attack. Applied
#: before the groups compete, so a lone attacker never hides a duel or a sprint.
BOX_PRESSURE_ALONE_MIN = 0.5
#: Contribution (to the [0, 1] score) from which audio or editing is named in a title;
#: below it the title names neither.
BROADCAST_CUE_MIN = 0.02
#: Smallest mean jersey confidence for a jersey number to be printed.
JERSEY_MIN_CONF = 0.5
MAX_TRACK_IDS = 6
DEFAULT_TEAM_NAMES: Mapping[str, str] = MappingProxyType({"left": "左侧球队", "right": "右侧球队"})
#: Chinese names of the highlight types. They say only what the evidence shows: a
#: ball_strike rests on an abrupt change of the ball's image velocity near a player (often a
#: first touch, not necessarily a kick), and a closeup_reaction is what is left when no play
#: cue is strong enough, in a shot that is often a full-body framing rather than a close-up
#: (its title names sound or editing only when they carried it).
TYPE_NAMES_ZH: Mapping[str, str] = MappingProxyType(
    {
        "duel_tackle": "贴身争抢",
        "attack_in_box": "禁区进攻",
        "ball_strike": "触球（球速突变）",
        "sprint": "高速冲刺",
        "closeup_reaction": "球员反应",
    }
)
SHOT_TYPE_NAMES_ZH: Mapping[str, str] = MappingProxyType(
    {"wide": "全景", "medium": "中景", "close": "特写"}
)
#: The text names a shot by the size of its players (shots.median_box_h, 1080-line pixels).
#: gbf.shots_grammar.classify_shot types a shot "close" from a median box of 220 px (150 px
#: when under a fifth of its frames are pitch-usable), where players are still seen from head
#: to foot: such a shot is a 中景 (in the usual shot scale 近景 is chest-up). From this height
#: on, half the frame, the box bottom is the frame edge rather than the feet (the bound of
#: the image-plane duel, DUEL_IMAGE_MAX_BOX_H_PX, and of gbf.measure's
#: image_feet_max_median_box_px): upper bodies, 特写. A close shot without any person box
#: (a crowd, a bench, a replay graphic) keeps the name 特写.
SHOT_UPPER_BODY_MIN_BOX_H_PX = DUEL_IMAGE_MAX_BOX_H_PX

_TEAMS = ("left", "right")


class HighlightInputError(ValueError):
    """Inputs of :func:`compute_signals` or :func:`detect_highlights` are unusable.

    ``problems`` lists every problem found.
    """

    def __init__(self, what: str, problems: Sequence[str]):
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{what}: {len(self.problems)} problem(s)\n  - {joined}")


# --------------------------------------------------------------------------- input tables

_KINDS = ("int", "id", "float", "bool", "str")


def _rate(fps: object, problems: list[str]) -> Fraction | None:
    if isinstance(fps, bool):
        problems.append(f"fps must be a rate such as '25/1', got {fps!r}")
        return None
    if isinstance(fps, int | np.integer):
        if int(fps) > 0:
            return Fraction(int(fps))
        problems.append(f"fps must be positive, got {fps!r}")
        return None
    if isinstance(fps, float | np.floating):
        value = float(fps)
        if math.isfinite(value) and value > 0 and value == int(value):
            return Fraction(int(value))
        problems.append(
            f"fps must be a positive rational such as '30000/1001'; a float is only accepted "
            f"when it is a whole number, got {fps!r}"
        )
        return None
    try:
        return parse_rate(fps)  # type: ignore[arg-type]
    except ValueError as exc:
        problems.append(f"fps: {exc}")
        return None


def _column(frame: pd.DataFrame, table: str, name: str, kind: str, problems: list[str]) -> Any:
    """Column ``name`` of ``frame`` as a numpy array of ``kind`` (see :data:`_KINDS`).

    ``int``: an integer column without missing values (int64). ``id``: whole numbers where a
    missing value becomes -1 (int64). ``float``: numeric, missing -> NaN (float64). ``bool``:
    missing -> False. ``str``: an object array with None for anything that is not a string.
    """
    series = frame[name]
    if kind == "id" and not pd.api.types.is_integer_dtype(series.dtype):
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            problems.append(f"{table}.{name} must be a numeric id column, got {series.dtype}")
            return None
        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        finite = np.isfinite(values)
        if np.any(values[finite] != np.round(values[finite])):
            problems.append(f"{table}.{name} must hold whole numbers")
            return None
        out = np.full(values.shape, -1, dtype=np.int64)
        out[finite] = values[finite].astype(np.int64)
        return out
    if kind == "id":
        return series.fillna(-1).to_numpy(dtype=np.int64)
    if kind == "int":
        if not pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            problems.append(f"{table}.{name} must be an integer column, got {series.dtype}")
            return None
        if bool(series.isna().any()):
            problems.append(f"{table}.{name} has missing values")
            return None
        return series.to_numpy(dtype=np.int64)
    if kind == "float":
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            problems.append(f"{table}.{name} must be a numeric column, got {series.dtype}")
            return None
        return series.to_numpy(dtype=np.float64, na_value=np.nan)
    if kind == "bool":
        try:
            return series.astype("boolean").fillna(False).to_numpy(dtype=bool)
        except (TypeError, ValueError):
            problems.append(f"{table}.{name} must be a boolean column, got {series.dtype}")
            return None
    values = series.to_numpy(dtype=object, na_value=None)
    return np.array([v if isinstance(v, str) else None for v in values], dtype=object)


def _table(
    frame: object,
    table: str,
    required: Mapping[str, str],
    optional: Mapping[str, str] | None,
    problems: list[str],
) -> dict[str, Any] | None:
    """Arrays of the columns of ``frame``; ``None`` (with problems) when it is unusable.

    ``frame=None`` stands for an empty table. Optional columns that are absent are left out.
    """
    if frame is None:
        frame = pd.DataFrame(
            {name: pd.Series(dtype=_EMPTY_DTYPES[kind]) for name, kind in required.items()}
        )
    if not isinstance(frame, pd.DataFrame):
        problems.append(f"{table} must be a pandas DataFrame or None, got {type(frame).__name__}")
        return None
    missing = [name for name in required if name not in frame.columns]
    if missing:
        problems.append(f"{table} lacks the column(s): {', '.join(missing)}")
        return None
    before = len(problems)
    arrays: dict[str, Any] = {}
    for name, kind in {**required, **(optional or {})}.items():
        if name in frame.columns:
            arrays[name] = _column(frame, table, name, kind, problems)
    if len(problems) > before:
        return None
    arrays["_rows"] = len(frame)
    return arrays


_EMPTY_DTYPES = {"int": "int64", "id": "int64", "float": "float64", "bool": "bool", "str": "object"}


def _frame_range_problems(frames: np.ndarray, table: str, n: int) -> list[str]:
    bad = frames[(frames < 0) | (frames >= n)]
    if bad.size == 0:
        return []
    shown = ", ".join(str(int(v)) for v in np.unique(bad)[:10])
    return [f"{table} has {bad.size} row(s) with frames outside 0..{n - 1}: {shown}"]


def _duplicate_frame_problems(frames: np.ndarray, table: str) -> list[str]:
    values, counts = np.unique(frames, return_counts=True)
    dup = values[counts > 1]
    if dup.size == 0:
        return []
    shown = ", ".join(str(int(v)) for v in dup[:10])
    return [f"{table} has more than one row for frame(s) {shown}"]


@dataclass(frozen=True)
class ShotTable:
    """Shots sorted by start frame; ``of_frame[f]`` is the row index of the shot of frame f."""

    shot_id: np.ndarray
    start: np.ndarray
    end: np.ndarray
    shot_type: tuple[str, ...]
    fresh_share: np.ndarray
    median_box_h: np.ndarray  # NaN when unknown
    of_frame: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.end[-1]) + 1


def _shot_table(shots: object, problems: list[str]) -> ShotTable | None:
    arrays = _table(
        shots,
        "shots",
        {
            "shot_id": "int",
            "start_frame": "int",
            "end_frame": "int",
            "shot_type": "str",
            "fresh_share": "float",
        },
        {"median_box_h": "float"},
        problems,
    )
    if arrays is None:
        return None
    if arrays["_rows"] == 0:
        problems.append("shots is empty; it must cover every frame of the clip")
        return None
    order = np.lexsort((arrays["end_frame"], arrays["start_frame"]))
    start = arrays["start_frame"][order]
    end = arrays["end_frame"][order]
    ids = arrays["shot_id"][order]
    types = tuple(arrays["shot_type"][order])
    # A missing fresh share counts as 0: the shot is then image-plane only.
    fresh = np.nan_to_num(arrays["fresh_share"][order], nan=0.0)
    box_h = arrays.get("median_box_h")
    box_h = np.full(len(order), np.nan) if box_h is None else box_h[order]
    found = []
    if start[0] != 0:
        found.append(f"shots must start at frame 0, the first starts at {int(start[0])}")
    for i in range(len(order)):
        if end[i] < start[i]:
            found.append(
                f"shot {int(ids[i])} ends ({int(end[i])}) before it starts ({int(start[i])})"
            )
        if i and start[i] != end[i - 1] + 1:
            found.append(
                f"shot {int(ids[i])} starts at {int(start[i])}, not right after the previous "
                f"shot's end {int(end[i - 1])} (shots must tile the clip)"
            )
        if types[i] not in SHOT_TYPES:
            found.append(f"shot {int(ids[i])} has shot_type {types[i]!r}, not one of {SHOT_TYPES}")
    if len(np.unique(ids)) != len(ids):
        found.append("shot_id values are not unique")
    if found:
        problems.extend(found)
        return None
    n = int(end[-1]) + 1
    of_frame = np.repeat(np.arange(len(order)), (end - start + 1).astype(np.int64))
    return ShotTable(ids, start, end, types, fresh, box_h, of_frame[:n])


def _attack(attack_direction: object, problems: list[str]) -> dict[str, int] | None:
    if attack_direction is None:
        return {"left": 1, "right": -1}
    if not isinstance(attack_direction, Mapping):
        problems.append(
            f"attack_direction must be a mapping, got {type(attack_direction).__name__}"
        )
        return None
    found = []
    if set(attack_direction) != set(_TEAMS):
        keys = sorted(map(str, attack_direction))
        found.append(f"attack_direction must have exactly the keys 'left' and 'right', got {keys}")
    else:
        for team in _TEAMS:
            value = attack_direction[team]
            if isinstance(value, bool) or value not in (1, -1):
                found.append(f"attack_direction[{team!r}] must be 1 or -1, got {value!r}")
        if not found and attack_direction["left"] == attack_direction["right"]:
            found.append("attack_direction: the two teams must attack opposite goals")
    if found:
        problems.extend(found)
        return None
    return {team: int(attack_direction[team]) for team in _TEAMS}


def _team_code(values: np.ndarray) -> np.ndarray:
    """1 for "left", 2 for "right", 0 for anything else."""
    return np.array([1 if v == "left" else 2 if v == "right" else 0 for v in values], dtype=np.int8)


def _majority_role(track_ids: np.ndarray, roles: np.ndarray) -> dict[int, str | None]:
    """Most frequent role per track (ties: alphabetical), ignoring missing roles."""
    counts: dict[int, dict[str, int]] = {}
    for track, role in zip(track_ids.tolist(), roles.tolist(), strict=True):
        if role is None:
            continue
        bucket = counts.setdefault(int(track), {})
        bucket[role] = bucket.get(role, 0) + 1
    return {
        track: sorted(bucket.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for track, bucket in counts.items()
    }


# --------------------------------------------------------------------------- per-frame helpers


def _first_per_frame(frames: np.ndarray, *keys: np.ndarray) -> np.ndarray:
    """Row index of the first row of every frame after sorting by ``keys`` (then frame).

    ``keys`` are given from the most to the least significant after the frame.
    """
    if frames.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort(tuple(reversed(keys)) + (frames,))
    _, first = np.unique(frames[order], return_index=True)
    return order[first]


def _ramp(values: np.ndarray, zero_at: float, one_at: float) -> np.ndarray:
    """Linear map with ``zero_at -> 0`` and ``one_at -> 1``, clipped to [0, 1]; NaN -> 0."""
    out = (values - zero_at) / (one_at - zero_at)
    return np.nan_to_num(np.clip(out, 0.0, 1.0), nan=0.0)


def _exact_seconds(seconds: float) -> Fraction:
    """A decimal number of seconds as an exact fraction (0.6 -> 3/5, not the binary float)."""
    return Fraction(repr(float(seconds)))


def _round_frames(seconds: float, rate: Fraction) -> int:
    """``seconds * fps`` rounded to the nearest frame (halves away from zero)."""
    exact = _exact_seconds(seconds) * rate
    return (
        int(math.floor(exact + Fraction(1, 2)))
        if exact >= 0
        else -int(math.floor(-exact + Fraction(1, 2)))
    )


# --------------------------------------------------------------------------- cues


def _cut_emphasis(shots: ShotTable, rate: Fraction, n: int) -> tuple[np.ndarray, np.ndarray]:
    cue = np.zeros(n)
    cut_frame = np.full(n, -1, dtype=np.int64)
    pre = _round_frames(CUT_PRE_S, rate)
    follow = _round_frames(FOLLOW_MAX_S, rate)
    for i in range(len(shots.start) - 1):
        if shots.shot_type[i] not in EMPHASIS_FROM_TYPES:
            continue
        if shots.shot_type[i + 1] not in EMPHASIS_TO_TYPES:
            continue
        cut = int(shots.start[i + 1])
        lo = max(int(shots.start[i]), cut - pre)
        cue[lo:cut] = np.maximum(cue[lo:cut], CUT_PRE_VALUE)
        cut_frame[lo:cut] = cut
        hi = min(int(shots.end[i + 1]) + 1, cut + follow)
        cue[cut:hi] = np.maximum(cue[cut:hi], CUT_POST_VALUE)
        cut_frame[cut:hi] = cut
    return cue, cut_frame


def _audio_cue(audio: dict[str, Any], n: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.full(n, np.nan)
    frames = audio["frame"]
    values = audio["excitement_z"].copy()
    if "has_audio" in audio:
        values[~audio["has_audio"]] = np.nan
    z[frames] = values
    cue = np.clip(np.nan_to_num(z, nan=0.0), 0.0, AUDIO_Z_MAX) / AUDIO_Z_MAX
    return cue, z


@dataclass
class _Players:
    """Rows of tracks that belong to players or goalkeepers with a known team."""

    frame: np.ndarray
    track_id: np.ndarray
    team: np.ndarray  # 1 left, 2 right
    outfield: np.ndarray
    x: np.ndarray  # NaN unless a pitch frame with a usable position
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    h: np.ndarray
    in_pitch: np.ndarray


def _players(tracks: dict[str, Any], pitch_frame: np.ndarray) -> _Players:
    role = tracks["role"]
    team = _team_code(tracks["team"])
    keep = np.array([r in PLAYER_ROLES for r in role], dtype=bool) & (team > 0)
    frame = tracks["frame"][keep]
    usable = tracks["pitch_usable"][keep] & pitch_frame[frame]
    x = np.where(usable, tracks["pitch_x"][keep], np.nan)
    y = np.where(usable, tracks["pitch_y"][keep], np.nan)
    return _Players(
        frame=frame,
        track_id=tracks["track_id"][keep],
        team=team[keep],
        outfield=np.array([r in OUTFIELD_ROLES for r in role[keep]], dtype=bool),
        x=x,
        y=y,
        u=tracks["foot_u"][keep],
        v=tracks["foot_v"][keep],
        h=tracks["y2"][keep] - tracks["y1"][keep],
        in_pitch=tracks["in_pitch"][keep],
    )


def _speed_cue(
    kinematics: dict[str, Any], roles: dict[int, str | None], pitch_frame: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed_mps = np.full(n, np.nan)
    speed_id = np.full(n, -1, dtype=np.int64)
    frames = kinematics["frame"]
    speed = kinematics["speed"]
    track = kinematics["track_id"]
    is_player = np.array([roles.get(int(t)) in PLAYER_ROLES for t in track], dtype=bool)
    keep = (
        kinematics["valid"]
        & np.isfinite(speed)
        & (speed >= 0)
        & (speed <= SPEED_PLAUSIBLE_MAX_MPS)
        & is_player
        & pitch_frame[frames]
    )
    rows = np.flatnonzero(keep)
    first = _first_per_frame(frames[rows], -speed[rows], track[rows])
    chosen = rows[first]
    speed_mps[frames[chosen]] = speed[chosen]
    speed_id[frames[chosen]] = track[chosen]
    return _ramp(speed_mps, SPEED_LOW_MPS, SPEED_HIGH_MPS), speed_mps, speed_id


def _ball_seen(ball: dict[str, Any], shots: ShotTable, rate: Fraction, n: int) -> np.ndarray:
    """Frames with a detected ball within :data:`ATTACK_BALL_SEEN_S` in the same shot."""
    detected = np.zeros(n, dtype=bool)
    rows = np.array([status == "detected" for status in ball["status"]], dtype=bool)
    detected[ball["frame"][rows]] = True
    counts = np.concatenate(([0], np.cumsum(detected)))
    reach = _round_frames(ATTACK_BALL_SEEN_S, rate)
    frames = np.arange(n)
    lo = np.maximum(frames - reach, shots.start[shots.of_frame])
    hi = np.minimum(frames + reach, shots.end[shots.of_frame])
    return (counts[hi + 1] - counts[lo]) > 0


def _box_pressure(
    players: _Players, attack: dict[str, int], ball_seen: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sign = np.where(players.team == 1, attack["left"], attack["right"])
    ahead = sign * players.x
    inside = (
        ball_seen[players.frame]
        & players.outfield
        & players.in_pitch
        & np.isfinite(players.x)
        & np.isfinite(players.y)
        & (ahead >= HALF_LENGTH_M - PENALTY_AREA_DEPTH_M)
        & (ahead <= HALF_LENGTH_M)
        & (np.abs(players.y) <= PENALTY_AREA_HALF_WIDTH_M)
    )
    counts = np.zeros((n, 3), dtype=np.int64)
    np.add.at(counts, (players.frame[inside], players.team[inside].astype(np.int64)), 1)
    attackers = counts[:, 1:].max(axis=1)
    team = np.full(n, None, dtype=object)
    team[(attackers > 0) & (counts[:, 1] >= counts[:, 2])] = "left"
    team[(attackers > 0) & (counts[:, 2] > counts[:, 1])] = "right"
    cue = np.minimum(1.0, attackers / BOX_FULL_ATTACKERS)
    return cue, attackers, team


@dataclass
class _BallSide:
    near_id: np.ndarray
    opp_id: np.ndarray
    goal_m: np.ndarray
    team: np.ndarray
    cue: np.ndarray
    metric: np.ndarray  # the frame's ball distances are in metres
    dist: np.ndarray  # per player row: distance to the ball (m or box heights), NaN if none


def _ball_arrays(ball: dict[str, Any], n: int) -> dict[str, np.ndarray]:
    out = {name: np.full(n, np.nan) for name in ("u", "v", "x", "y")}
    frames = ball["frame"]
    present = np.array([s != "missing" for s in ball["status"]], dtype=bool)
    for name, column in (("u", "u"), ("v", "v"), ("x", "pitch_x"), ("y", "pitch_y")):
        values = np.where(present, ball[column], np.nan)
        out[name][frames] = values
    return out


def _ball_side(
    players: _Players,
    ball: dict[str, np.ndarray],
    pitch_frame: np.ndarray,
    attack: dict[str, int],
    ball_seen: np.ndarray,
    n: int,
) -> _BallSide:
    """Players nearest the ball, and the ball_goal cue (only where ``ball_seen``)."""
    metric_frame = pitch_frame & np.isfinite(ball["x"]) & np.isfinite(ball["y"])
    image_frame = ~metric_frame & np.isfinite(ball["u"]) & np.isfinite(ball["v"])
    f = players.frame
    metric_rows = metric_frame[f] & np.isfinite(players.x) & np.isfinite(players.y)
    image_rows = (
        image_frame[f]
        & np.isfinite(players.u)
        & np.isfinite(players.v)
        & np.isfinite(players.h)
        & (players.h > 0)
    )
    dist = np.full(f.shape, np.nan)
    dist[metric_rows] = np.hypot(
        players.x[metric_rows] - ball["x"][f[metric_rows]],
        players.y[metric_rows] - ball["y"][f[metric_rows]],
    )
    dist[image_rows] = (
        np.hypot(
            players.u[image_rows] - ball["u"][f[image_rows]],
            players.v[image_rows] - ball["v"][f[image_rows]],
        )
        / players.h[image_rows]
    )
    rows = np.flatnonzero(np.isfinite(dist))
    near_id = np.full(n, -1, dtype=np.int64)
    near_team = np.zeros(n, dtype=np.int8)
    first = rows[_first_per_frame(f[rows], dist[rows], players.track_id[rows])]
    near_id[f[first]] = players.track_id[first]
    near_team[f[first]] = players.team[first]
    opp_rows = rows[(players.team[rows] != near_team[f[rows]]) & (near_team[f[rows]] > 0)]
    opp_id = np.full(n, -1, dtype=np.int64)
    first_opp = opp_rows[_first_per_frame(f[opp_rows], dist[opp_rows], players.track_id[opp_rows])]
    opp_id[f[first_opp]] = players.track_id[first_opp]

    goal_m = np.full(n, np.nan)
    team = np.full(n, None, dtype=object)
    side = metric_frame & (near_team > 0) & ball_seen
    sign = np.where(near_team == 1, attack["left"], attack["right"])
    goal_x = sign * HALF_LENGTH_M
    goal_m[side] = np.hypot(ball["x"][side] - goal_x[side], ball["y"][side])
    team[side & (near_team == 1)] = "left"
    team[side & (near_team == 2)] = "right"
    cue = _ramp(goal_m, BALL_GOAL_FAR_M, BALL_GOAL_NEAR_M)
    return _BallSide(near_id, opp_id, goal_m, team, cue, metric_frame, dist)


def _box_distance(x: np.ndarray, y: np.ndarray, sign: int) -> np.ndarray:
    """Ground distance (m) from (x, y) to the penalty area in front of the goal at
    x = HALF_LENGTH_M * sign, 0 inside it; NaN where a coordinate is NaN."""
    ahead = sign * x
    depth = np.where(
        ahead > HALF_LENGTH_M,
        ahead - HALF_LENGTH_M,
        np.maximum(0.0, HALF_LENGTH_M - PENALTY_AREA_DEPTH_M - ahead),
    )
    width = np.maximum(0.0, np.abs(y) - PENALTY_AREA_HALF_WIDTH_M)
    return np.hypot(depth, width)


def _ball_box(
    ball: dict[str, np.ndarray],
    pitch_frame: np.ndarray,
    box_team: np.ndarray,
    ball_team: np.ndarray,
    attack: dict[str, int],
    shots: ShotTable,
    rate: Fraction,
    n: int,
) -> np.ndarray:
    """``ev_ball_box_m`` (see "Ball at the box" in the module docstring)."""
    x = np.where(pitch_frame, ball["x"], np.nan)
    y = np.where(pitch_frame, ball["y"], np.nan)
    to_box = {team: _box_distance(x, y, attack[team]) for team in _TEAMS}
    reach = _round_frames(ATTACK_BALL_SEEN_S, rate)
    out = np.full(n, np.nan)
    for frame in range(n):
        team = box_team[frame] if box_team[frame] in _TEAMS else ball_team[frame]
        if team not in _TEAMS:
            continue
        shot = shots.of_frame[frame]
        lo = max(frame - reach, int(shots.start[shot]))
        hi = min(frame + reach, int(shots.end[shot]))
        chunk = to_box[team][lo : hi + 1]
        if np.isfinite(chunk).any():
            out[frame] = float(np.nanmin(chunk))
    return out


def _at_box(ball_box_m: np.ndarray) -> np.ndarray:
    """Frames whose ``ev_ball_box_m`` is at most :data:`BALL_AT_BOX_MARGIN_M` (NaN: no)."""
    distance = np.nan_to_num(np.asarray(ball_box_m, dtype=np.float64), nan=np.inf)
    return distance <= BALL_AT_BOX_MARGIN_M


def _duel(
    players: _Players, side: _BallSide, image_ok: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest opposing pair near the ball; ``image_ok[f]`` allows the image plane on f."""
    f = players.frame
    metric = side.metric[f]
    radius = np.where(metric, DUEL_BALL_RADIUS_M, DUEL_BALL_RADIUS_H)
    near = np.isfinite(side.dist) & (side.dist <= radius) & (metric | image_ok[f])
    table = pd.DataFrame(
        {
            "frame": f[near],
            "track_id": players.track_id[near],
            "team": players.team[near],
            "metric": metric[near],
            "x": players.x[near],
            "y": players.y[near],
            "u": players.u[near],
            "v": players.v[near],
            "h": players.h[near],
        }
    )
    pairs = table[table["team"] == 1].merge(
        table[table["team"] == 2], on="frame", suffixes=("_a", "_b")
    )
    cue = np.zeros(n)
    dist_out = np.full(n, np.nan)
    unit = np.full(n, None, dtype=object)
    id_a = np.full(n, -1, dtype=np.int64)
    id_b = np.full(n, -1, dtype=np.int64)
    if pairs.empty:
        return cue, dist_out, unit, id_a, id_b
    is_metric = pairs["metric_a"].to_numpy(dtype=bool)
    metres = np.hypot(
        pairs["x_a"].to_numpy() - pairs["x_b"].to_numpy(),
        pairs["y_a"].to_numpy() - pairs["y_b"].to_numpy(),
    )
    heights = np.hypot(
        pairs["u_a"].to_numpy() - pairs["u_b"].to_numpy(),
        pairs["v_a"].to_numpy() - pairs["v_b"].to_numpy(),
    ) / ((pairs["h_a"].to_numpy() + pairs["h_b"].to_numpy()) / 2.0)
    dist = np.where(is_metric, metres, heights)
    closeness = np.where(
        is_metric,
        _ramp(dist, DUEL_FAR_M, DUEL_NEAR_M),
        _ramp(dist, DUEL_FAR_H, DUEL_NEAR_H),
    )
    frames = pairs["frame"].to_numpy(dtype=np.int64)
    a = pairs["track_id_a"].to_numpy(dtype=np.int64)
    b = pairs["track_id_b"].to_numpy(dtype=np.int64)
    ok = np.isfinite(dist)
    rows = np.flatnonzero(ok)
    first = rows[_first_per_frame(frames[rows], -closeness[rows], dist[rows], a[rows], b[rows])]
    chosen = frames[first]
    cue[chosen] = closeness[first]
    dist_out[chosen] = dist[first]
    unit[chosen] = np.where(is_metric[first], "m", "box_h")
    id_a[chosen] = a[first]
    id_b[chosen] = b[first]
    return cue, dist_out, unit, id_a, id_b


def _event_strengths(events: dict[str, Any]) -> np.ndarray:
    """Strength in [0, 1] of every ball event (see the ``strike`` cue in the module doc)."""
    for name in STRIKE_STRENGTH_COLUMNS:
        if name in events:
            return np.clip(np.nan_to_num(events[name], nan=0.0), 0.0, 1.0)
    strength = np.ones(events["_rows"])
    if "magnitude" in events:
        magnitude = np.nan_to_num(events["magnitude"], nan=0.0)
        strength = np.clip(magnitude / STRIKE_FULL_MAGNITUDE_BH_S, 0.0, 1.0)
    if "nearest_dist_box_h" in events:
        distance = events["nearest_dist_box_h"]
        foot = np.where(
            np.isfinite(distance),
            _ramp(distance, STRIKE_FOOT_FAR_H, STRIKE_FOOT_NEAR_H),
            STRIKE_UNKNOWN_FOOT_WEIGHT,
        )
        strength = strength * foot
    if "kind" in events:
        strength = strength * np.array(
            [STRIKE_KIND_WEIGHTS.get(kind, 1.0) if kind else 1.0 for kind in events["kind"]]
        )
    return strength


def _event_track_ids(events: dict[str, Any]) -> np.ndarray:
    """Player id of every ball event (``nearest_track_id`` or ``track_id``; -1 unknown)."""
    for name in ("nearest_track_id", "track_id"):
        if name in events:
            return events[name]
    return np.full(events["_rows"], -1, dtype=np.int64)


def _strike(events: dict[str, Any], rate: Fraction, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangle cue around every ball event; ``event_frame`` names the event that set it.

    Where two events give the same value, the nearer event wins, then the earlier one.
    """
    cue = np.zeros(n)
    event_frame = np.full(n, -1, dtype=np.int64)
    half_exact = _exact_seconds(STRIKE_HALF_WIDTH_S) * rate
    half = float(half_exact)
    reach = int(math.floor(half_exact))
    frames = events["frame"]
    strength = _event_strengths(events)
    for row in np.lexsort((-strength, frames)):
        e = int(frames[row])
        lo, hi = max(0, e - reach), min(n, e + reach + 1)
        if lo >= hi:
            continue
        span = np.arange(lo, hi)
        values = strength[row] * np.maximum(0.0, 1.0 - np.abs(span - e) / half)
        held = event_frame[lo:hi]
        new_gap = np.abs(span - e)
        old_gap = np.where(held >= 0, np.abs(span - held), np.iinfo(np.int64).max)
        tie_wins = (new_gap < old_gap) | ((new_gap == old_gap) & (e < held))
        replace = (values > 0) & ((values > cue[lo:hi]) | ((values == cue[lo:hi]) & tie_wins))
        cue[lo:hi] = np.where(replace, values, cue[lo:hi])
        event_frame[lo:hi] = np.where(replace, e, held)
    return cue, event_frame


# --------------------------------------------------------------------------- compute_signals


_TRACK_COLUMNS = {
    "frame": "int",
    "track_id": "int",
    "y1": "float",
    "y2": "float",
    "role": "str",
    "team": "str",
    "foot_u": "float",
    "foot_v": "float",
    "pitch_usable": "bool",
    "pitch_x": "float",
    "pitch_y": "float",
    "in_pitch": "bool",
}
_KINEMATICS_COLUMNS = {"frame": "int", "track_id": "int", "speed": "float", "valid": "bool"}
_BALL_COLUMNS = {
    "frame": "int",
    "u": "float",
    "v": "float",
    "status": "str",
    "pitch_x": "float",
    "pitch_y": "float",
}
_EVENT_COLUMNS = {"frame": "int"}
_EVENT_OPTIONAL = {
    "track_id": "id",
    "nearest_track_id": "id",
    "strength": "float",
    "conf": "float",
    "score": "float",
    "magnitude": "float",
    "nearest_dist_box_h": "float",
    "kind": "str",
}
_AUDIO_COLUMNS = {"frame": "int", "excitement_z": "float"}


def compute_signals(
    tracks: pd.DataFrame | None,
    kinematics: pd.DataFrame | None,
    ball_track: pd.DataFrame | None,
    ball_events: pd.DataFrame | None,
    shots: pd.DataFrame,
    audio: pd.DataFrame | None,
    fps: str | Fraction | int,
    *,
    attack_direction: Mapping[str, int] | None = None,
    attack_known: bool = True,
) -> pd.DataFrame:
    """One row per frame with the highlight cues, the score and their evidence.

    Inputs follow the M1 contract (``tracks.parquet``, ``kinematics.parquet``,
    ``ball_track.parquet``, ``shots.parquet``, ``audio.parquet``). ``ball_events`` needs a
    ``frame`` column; it is normally the table of :func:`gbf.ball.ball_events`
    (``magnitude``, ``kind``, ``nearest_track_id``, ``nearest_dist_box_h`` are used), and a
    ``strength``, ``conf`` or ``score`` column in [0, 1] and a ``track_id`` column are
    accepted instead. ``None`` stands for an empty table: the cues that need it are 0.
    ``attack_direction`` defaults to ``{"left": 1, "right": -1}``. ``attack_known`` False
    (gbf.side_check could not confirm the tracking plugin's left/right names) turns the
    cues that need the goal a team attacks off: ``box_pressure`` and ``ball_goal`` are 0 and
    their evidence (``ev_box_attackers`` 0, ``ev_box_team``, ``ev_ball_goal_m``,
    ``ev_ball_box_m``) missing; the other cues and evidence do not change. The columns are
    :data:`SIGNAL_COLUMNS`; see the module docstring for every cue. Raises
    :class:`HighlightInputError` listing every problem with the inputs.
    """
    problems: list[str] = []
    rate = _rate(fps, problems)
    attack = _attack(attack_direction, problems)
    if not isinstance(attack_known, bool):
        problems.append(f"attack_known must be a bool, got {attack_known!r}")
    shot_table = _shot_table(shots, problems)
    track_arrays = _table(tracks, "tracks", _TRACK_COLUMNS, None, problems)
    kin = _table(kinematics, "kinematics", _KINEMATICS_COLUMNS, None, problems)
    ball = _table(ball_track, "ball_track", _BALL_COLUMNS, None, problems)
    events = _table(ball_events, "ball_events", _EVENT_COLUMNS, _EVENT_OPTIONAL, problems)
    sound = _table(audio, "audio", _AUDIO_COLUMNS, {"has_audio": "bool"}, problems)
    if shot_table is not None:
        n = shot_table.n_frames
        for name, arrays in (
            ("tracks", track_arrays),
            ("kinematics", kin),
            ("ball_track", ball),
            ("ball_events", events),
            ("audio", sound),
        ):
            if arrays is not None:
                problems.extend(_frame_range_problems(arrays["frame"], name, n))
        for name, arrays in (("ball_track", ball), ("audio", sound)):
            if arrays is not None:
                problems.extend(_duplicate_frame_problems(arrays["frame"], name))
    if problems:
        raise HighlightInputError("compute_signals", problems)
    # Each helper returns None only after recording a problem, so all are set here.
    return _signals(
        cast(Fraction, rate),
        cast(dict[str, int], attack),
        cast(ShotTable, shot_table),
        cast(dict[str, Any], track_arrays),
        cast(dict[str, Any], kin),
        cast(dict[str, Any], ball),
        cast(dict[str, Any], events),
        cast(dict[str, Any], sound),
        attack_known,
    )


def _signals(
    rate: Fraction,
    attack: dict[str, int],
    shots: ShotTable,
    tracks: dict[str, Any],
    kin: dict[str, Any],
    ball: dict[str, Any],
    events: dict[str, Any],
    sound: dict[str, Any],
    attack_known: bool,
) -> pd.DataFrame:
    n = shots.n_frames
    shot_index = shots.of_frame
    pitch_shot = np.array(
        [
            shots.shot_type[i] in PITCH_SHOT_TYPES and shots.fresh_share[i] >= MIN_PITCH_FRESH_SHARE
            for i in range(len(shots.start))
        ],
        dtype=bool,
    )
    usable_rows = tracks["pitch_usable"]
    usable_frame = np.bincount(tracks["frame"][usable_rows], minlength=n)[:n] > 0
    pitch_frame = pitch_shot[shot_index] & usable_frame

    cut_cue, cut_frame = _cut_emphasis(shots, rate, n)
    audio_cue, audio_z = _audio_cue(sound, n)
    roles = _majority_role(tracks["track_id"], tracks["role"])
    speed_cue, speed_mps, speed_id = _speed_cue(kin, roles, pitch_frame, n)
    players = _players(tracks, pitch_frame)
    ball_seen = _ball_seen(ball, shots, rate, n)
    box_cue, box_attackers, box_team = _box_pressure(players, attack, ball_seen, n)
    ball_pos = _ball_arrays(ball, n)
    side = _ball_side(players, ball_pos, pitch_frame, attack, ball_seen, n)
    ball_box = _ball_box(ball_pos, pitch_frame, box_team, side.team, attack, shots, rate, n)
    # Attackers in the box count only with the ball at the box; their evidence stays.
    box_cue = np.where(_at_box(ball_box), box_cue, 0.0)
    if not attack_known:
        # The goal each team attacks is unknown: the cues that need it are off and their
        # evidence is missing; the duel reads only the distances to the ball.
        box_cue = np.zeros(n)
        box_attackers = np.zeros(n, dtype=np.int64)
        box_team = np.full(n, None, dtype=object)
        ball_box = np.full(n, np.nan)
        side = replace(side, cue=np.zeros(n), goal_m=np.full(n, np.nan))
    # Unknown box heights (NaN) do not block the image plane.
    image_shot = ~(shots.median_box_h > DUEL_IMAGE_MAX_BOX_H_PX)
    duel_cue, duel_dist, duel_unit, duel_a, duel_b = _duel(players, side, image_shot[shot_index], n)
    strike_cue, strike_frame = _strike(events, rate, n)

    cues = {
        "cut_emphasis": cut_cue,
        "audio": audio_cue,
        "speed": speed_cue,
        "box_pressure": box_cue,
        "ball_goal": side.cue,
        "duel": duel_cue,
        "strike": strike_cue,
    }
    total = sum(CUE_WEIGHTS.values())
    score = sum(CUE_WEIGHTS[name] * cues[name] for name in CUE_COLUMNS) / total
    data: dict[str, Any] = {
        "frame": np.arange(n, dtype=np.int64),
        "shot_id": shots.shot_id[shot_index].astype(np.int64),
    }
    data.update({name: cues[name].astype(np.float64) for name in CUE_COLUMNS})
    data["score"] = np.asarray(score, dtype=np.float64)
    data.update(
        {
            "ev_pitch": pitch_frame.astype(bool),
            "ev_cut_frame": cut_frame,
            "ev_audio_z": audio_z,
            "ev_speed_mps": speed_mps,
            "ev_speed_id": speed_id,
            "ev_box_attackers": box_attackers.astype(np.int64),
            "ev_box_team": pd.array(box_team, dtype="object"),
            "ev_ball_near_id": side.near_id,
            "ev_ball_opp_id": side.opp_id,
            "ev_ball_goal_m": side.goal_m,
            "ev_ball_team": pd.array(side.team, dtype="object"),
            "ev_ball_box_m": ball_box,
            "ev_duel_dist": duel_dist,
            "ev_duel_unit": pd.array(duel_unit, dtype="object"),
            "ev_duel_id_a": duel_a,
            "ev_duel_id_b": duel_b,
            "ev_strike_frame": strike_frame,
        }
    )
    return pd.DataFrame(data, columns=list(SIGNAL_COLUMNS))


# --------------------------------------------------------------------------- detection

_SIGNAL_REQUIRED = {
    "frame": "int",
    "shot_id": "int",
    **{name: "float" for name in CUE_COLUMNS},
    "score": "float",
}
_EVIDENCE_KINDS = {
    "ev_pitch": "bool",
    "ev_cut_frame": "id",
    "ev_audio_z": "float",
    "ev_speed_mps": "float",
    "ev_speed_id": "id",
    "ev_box_attackers": "id",
    "ev_box_team": "str",
    "ev_ball_near_id": "id",
    "ev_ball_opp_id": "id",
    "ev_ball_goal_m": "float",
    "ev_ball_team": "str",
    "ev_ball_box_m": "float",
    "ev_duel_dist": "float",
    "ev_duel_unit": "str",
    "ev_duel_id_a": "id",
    "ev_duel_id_b": "id",
    "ev_strike_frame": "id",
}
_EVIDENCE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {"bool": False, "id": -1, "float": np.nan, "str": None}
)
_EVIDENCE_DTYPES: Mapping[str, Any] = MappingProxyType(
    {"bool": bool, "id": np.int64, "float": np.float64, "str": object}
)
_TRACK_INFO_REQUIRED = {"track_id": "int"}
_TRACK_INFO_OPTIONAL = {
    "frame": "int",
    "team": "str",
    "jersey": "str",
    "jersey_conf": "float",
    "role": "str",
    "y1": "float",
    "y2": "float",
}
#: How the precise peak frame was chosen, with its Chinese wording for the summary. A ball
#: event is an abrupt change of the ball's image velocity (often a first touch, not a
#: strike); the duel distance is averaged over REFINE_SMOOTH_S before its minimum is taken.
REFINE_METHODS_ZH: Mapping[str, str] = MappingProxyType(
    {
        "strike_event": "以球的像面速度突变的一帧为准",
        "duel_min_distance": "球在两人脚下时距离最近的一帧",
        "speed_peak": "以速度最高的一帧为准",
        "attack_peak": "以禁区进攻线索最强的一帧为准",
        "score_peak": "以综合评分峰值为准",
    }
)


def gaussian_smooth(values: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Gaussian moving average (truncated at 3 sigma), renormalised at the clip edges.

    NaN counts as 0. ``sigma_frames <= 0`` returns a copy of the input.
    """
    data = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    if data.size == 0 or sigma_frames <= 0:
        return data.copy()
    radius = max(1, math.ceil(3.0 * sigma_frames))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_frames) ** 2)
    padded = np.pad(data, radius)
    weights = np.pad(np.ones_like(data), radius)
    return np.convolve(padded, kernel, mode="valid") / np.convolve(weights, kernel, mode="valid")


def format_time(frame: int, fps: str | Fraction | int) -> str:
    """``mm:ss.cc`` of ``frame / fps``: minutes, seconds and hundredths of a second.

    The one clock text of every title, summary, key moment and table
    (:func:`gbf.util.clock.format_clock`: hundredths floored, exact rationals), so at 25 fps
    every frame is exact (0.04 s); the frame number is always printed next to it as the
    exact reference. Raises :class:`HighlightInputError` for an invalid frame or rate.
    """
    problems: list[str] = []
    rate = _rate(fps, problems)
    if problems or rate is None:
        raise HighlightInputError("format_time", problems)
    try:
        return format_clock(frame, rate)
    except ClockError as exc:
        raise HighlightInputError("format_time", exc.problems) from exc


def _team_names(team_names: object, problems: list[str]) -> dict[str, str]:
    names = dict(DEFAULT_TEAM_NAMES)
    if team_names is None:
        return names
    if not isinstance(team_names, Mapping):
        problems.append(f"team_names must be a mapping or None, got {type(team_names).__name__}")
        return names
    for key, value in team_names.items():
        if key not in _TEAMS:
            problems.append(f"team_names has the key {key!r}; only 'left' and 'right' exist")
        elif not isinstance(value, str) or not value.strip():
            problems.append(f"team_names[{key!r}] must be a non-empty string, got {value!r}")
        else:
            names[key] = value.strip()
    return names


def _seconds(value: object, name: str, problems: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        problems.append(f"{name} must be a number of seconds, got {value!r}")
        return 0.0
    if not math.isfinite(float(value)) or float(value) < 0:
        problems.append(f"{name} must be finite and >= 0, got {value!r}")
        return 0.0
    return float(value)


def _signal_problems(sig: dict[str, Any], shots: ShotTable) -> list[str]:
    n = shots.n_frames
    frames = sig["frame"]
    if frames.size != n or not np.array_equal(np.sort(frames), np.arange(n)):
        return [
            f"signals must hold exactly one row per frame 0..{n - 1} of the shots, got "
            f"{frames.size} row(s)"
        ]
    found = []
    order = np.argsort(frames, kind="stable")
    expected = shots.shot_id[shots.of_frame]
    wrong = np.flatnonzero(sig["shot_id"][order] != expected)
    if wrong.size:
        shown = ", ".join(str(int(v)) for v in wrong[:10])
        found.append(f"signals.shot_id disagrees with shots on {wrong.size} frame(s): {shown}")
    for name in (*CUE_COLUMNS, "score"):
        values = sig[name]
        outside = np.isfinite(values) & ((values < -1e-9) | (values > 1.0 + 1e-9))
        if np.any(outside):
            found.append(
                f"signals.{name} must lie in [0, 1]; {int(np.count_nonzero(outside))} value(s) "
                f"do not, e.g. {float(values[outside][0])!r}"
            )
    return found


@dataclass(frozen=True)
class _TrackInfo:
    team: str | None
    jersey: str | None


def _track_info(
    info: dict[str, Any], lo: int | None = None, hi: int | None = None
) -> dict[int, _TrackInfo]:
    """Majority team and a printable jersey number per track id.

    The team is the label most rows carry (None on a tie). The jersey is the number read
    most often (ties: higher mean confidence, then text order), printed only when its mean
    ``jersey_conf`` reaches :data:`JERSEY_MIN_CONF`. With ``lo`` and ``hi`` (and a ``frame``
    column) only the rows of frames ``lo..hi`` vote: a tracker can reuse an id after a cut.
    """
    ids = info["track_id"]
    teams = info.get("team")
    jerseys = info.get("jersey")
    confs = info.get("jersey_conf")
    rows = np.arange(ids.size)
    if lo is not None and hi is not None and "frame" in info:
        rows = np.flatnonzero((info["frame"] >= lo) & (info["frame"] <= hi))
    team_votes: dict[int, dict[str, int]] = {}
    jersey_votes: dict[int, dict[str, list[float]]] = {}
    for row, track in zip(rows.tolist(), ids[rows].tolist(), strict=True):
        if teams is not None and teams[row] in _TEAMS:
            votes = team_votes.setdefault(track, {})
            votes[teams[row]] = votes.get(teams[row], 0) + 1
        if jerseys is not None and jerseys[row] is not None and jerseys[row].strip():
            conf = float(confs[row]) if confs is not None and np.isfinite(confs[row]) else 0.0
            jersey_votes.setdefault(track, {}).setdefault(jerseys[row].strip(), []).append(conf)
    result: dict[int, _TrackInfo] = {}
    for track in sorted(set(ids[rows].tolist())):
        votes = team_votes.get(track, {})
        team = None
        if votes:
            ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
            if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                team = ranked[0][0]
        jersey = None
        options = jersey_votes.get(track, {})
        if options:
            best, values = sorted(
                options.items(),
                key=lambda item: (-len(item[1]), -float(np.mean(item[1])), item[0]),
            )[0]
            if float(np.mean(values)) >= JERSEY_MIN_CONF:
                jersey = best
        result[track] = _TrackInfo(team, jersey)
    return result


def _subject(info: dict[str, Any], lo: int, hi: int) -> tuple[int, int]:
    """Subject of a close shot over frames ``lo..hi``: ``(track_id, frame)`` or ``(-1, -1)``.

    The player or goalkeeper whose box heights add up to the most over those frames (a
    person seen large for long, not a passer-by filling one frame); ties go to the lower
    id. ``frame`` is where that box is tallest (ties: the earlier frame).
    """
    if not all(name in info for name in ("frame", "y1", "y2")) or hi < lo:
        return -1, -1
    rows = np.flatnonzero((info["frame"] >= lo) & (info["frame"] <= hi))
    if "role" in info and rows.size:
        rows = rows[np.array([info["role"][r] in PLAYER_ROLES for r in rows], dtype=bool)]
    heights = info["y2"][rows] - info["y1"][rows]
    ok = np.isfinite(heights) & (heights > 0) & (info["track_id"][rows] >= 0)
    if not np.any(ok):
        return -1, -1
    rows, heights = rows[ok], heights[ok]
    tracks = info["track_id"][rows]
    ids, inverse = np.unique(tracks, return_inverse=True)
    totals = np.bincount(inverse, weights=heights)
    track = int(ids[np.lexsort((ids, -totals))[0]])
    mine = tracks == track
    frames = info["frame"][rows][mine]
    best = np.lexsort((frames, -heights[mine]))[0]
    return track, int(frames[best])


def _best_frame(
    values: np.ndarray, lo: int, hi: int, centre: int, *, positive: bool = True
) -> int | None:
    """Frame in ``[lo, hi]`` with the largest finite value (ties: nearest ``centre``, earlier).

    With ``positive`` the value must also be above 0. ``None`` when no frame qualifies.
    """
    if hi < lo:
        return None
    span = np.arange(lo, hi + 1)
    chunk = values[lo : hi + 1]
    ok = np.isfinite(chunk)
    if positive:
        ok &= chunk > 0
    if not np.any(ok):
        return None
    span, chunk = span[ok], chunk[ok]
    best = np.lexsort((span, np.abs(span - centre), -chunk))[0]
    return int(span[best])


def _short_mean(values: np.ndarray, half_width: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean over ``[i - half_width, i + half_width]`` (edges truncated) of the masked values.

    Frames outside ``mask`` (default: the finite values) neither count nor get a value (NaN).
    """
    data = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(data) if mask is None else (mask & np.isfinite(data))
    filled = np.where(keep, data, 0.0)
    sums = np.concatenate(([0.0], np.cumsum(filled)))
    counts = np.concatenate(([0], np.cumsum(keep.astype(np.int64))))
    index = np.arange(data.size)
    lo = np.clip(index - half_width, 0, data.size)
    hi = np.clip(index + half_width + 1, 0, data.size)
    count = counts[hi] - counts[lo]
    out = np.full(data.size, np.nan)
    ok = keep & (count > 0)
    out[ok] = (sums[hi] - sums[lo])[ok] / count[ok]
    return out


def _min_duel_frame(
    data: dict[str, Any], cues: dict[str, np.ndarray], lo: int, hi: int, centre: int, half: int
) -> int | None:
    """Frame of the smallest duel distance in ``[lo, hi]``, averaged over ``2 half + 1`` frames.

    Only frames with a duel (cue above 0) count, in the unit most of them use (metres win a
    tie); ties go to the frame nearest ``centre``, then the earlier one. None without a duel.
    """
    if hi < lo:
        return None
    span = np.arange(lo, hi + 1)
    dist = data["ev_duel_dist"][lo : hi + 1]
    units = data["ev_duel_unit"][lo : hi + 1]
    ok = np.isfinite(dist) & (cues["duel"][lo : hi + 1] > 0)
    if not np.any(ok):
        return None
    metres = int(np.count_nonzero(ok & (units == "m")))
    heights = int(np.count_nonzero(ok & (units == "box_h")))
    ok &= units == ("m" if metres >= heights else "box_h")
    smoothed = _short_mean(dist, half, ok)
    rows = np.flatnonzero(ok)
    best = rows[np.lexsort((span[rows], np.abs(span[rows] - centre), smoothed[rows]))[0]]
    return int(span[best])


def _nearest_valid(ids: np.ndarray, lo: int, hi: int, centre: int) -> int:
    """The id nearest in time to ``centre`` within ``[lo, hi]`` that is not -1, else -1."""
    if hi < lo:
        return -1
    span = np.arange(lo, hi + 1)
    chunk = ids[lo : hi + 1]
    ok = chunk >= 0
    if not np.any(ok):
        return -1
    span = span[ok]
    best = np.lexsort((span, np.abs(span - centre)))[0]
    return int(ids[span[best]])


@dataclass(frozen=True)
class _Pick:
    score_peak: int
    peak: int
    kind: str
    method: str


def _attack_value(cues: Mapping[str, np.ndarray], lo: int, hi: int) -> np.ndarray:
    """Weighted sum of the attack cues on frames ``lo..hi``: above 0 where the attack type's
    own condition holds, given cues that count only with the ball at the box."""
    return np.asarray(sum(CUE_WEIGHTS[name] * cues[name][lo : hi + 1] for name in ATTACK_CUES))


def _classify(
    smooth_cues: dict[str, np.ndarray],
    frame: int,
    cues: dict[str, np.ndarray],
    lo: int,
    hi: int,
) -> str:
    """Type of a highlight from the smoothed cues at its score peak (see detect_highlights).

    ``smooth_cues`` and the raw ``cues`` hold the attack cues only where the ball is at the
    box ("Ball at the box" in the module docstring); ``lo..hi`` are the frames near the peak
    in its shot. Before the groups compete, box pressure leaves the attack group unless its
    raw value reaches :data:`BOX_PRESSURE_ALONE_MIN` there (one attacker alone is no
    attack), and the attack group counts only when one of its raw cues is above 0 there (its
    key frame must show the condition). A group competes when its strongest remaining cue
    reaches :data:`PLAY_EVIDENCE_MIN`; the largest weighted contribution wins (ties in
    PLAY_GROUPS order). Without a competing group the type is closeup_reaction.
    """
    total = sum(CUE_WEIGHTS.values())
    best_kind, best_contribution = "closeup_reaction", -1.0
    for kind, members in PLAY_GROUPS:
        values = {m: float(smooth_cues[m][frame]) for m in members}
        if kind == "attack_in_box":
            if float(np.max(cues["box_pressure"][lo : hi + 1])) < BOX_PRESSURE_ALONE_MIN:
                values["box_pressure"] = 0.0
            if not np.any(_attack_value(cues, lo, hi) > 0):
                values = dict.fromkeys(members, 0.0)
        if max(values.values()) < PLAY_EVIDENCE_MIN:
            continue
        contribution = sum(CUE_WEIGHTS[m] * values[m] for m in members) / total
        if contribution > best_contribution:
            best_kind, best_contribution = kind, contribution
    return best_kind


def _event_frames(events: dict[str, Any], lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Frames and strengths of the ball events in ``[lo, hi]`` whose strength is above 0."""
    frames = events["frame"]
    strength = _event_strengths(events)
    inside = (frames >= lo) & (frames <= hi) & (strength > 0)
    return frames[inside], strength[inside]


def _refine(
    kind: str,
    score_peak: int,
    lo: int,
    hi: int,
    cues: dict[str, np.ndarray],
    short: dict[str, np.ndarray],
    data: dict[str, Any],
    events: dict[str, Any],
    half: int,
) -> tuple[int, str]:
    """Precise peak frame within ``[lo, hi]`` for a highlight type, and how it was chosen.

    ``short`` holds the cues averaged over :data:`REFINE_SMOOTH_S`, so that a one-frame
    flicker of a detection cannot become the peak; event frames are used as they are. An
    attack_in_box keeps only frames where its own condition holds (an attack cue above 0
    with the ball at the box, as ``cues`` hold them): a ball event elsewhere is not used.
    """
    if kind in ("ball_strike", "attack_in_box"):
        frames, strength = _event_frames(events, lo, hi)
        holds = _attack_value(cues, lo, hi) > 0
        if kind == "attack_in_box":
            frames, strength = frames[holds[frames - lo]], strength[holds[frames - lo]]
        if frames.size:
            best = np.lexsort((frames, np.abs(frames - score_peak), -strength))[0]
            return int(frames[best]), "strike_event"
        if kind == "ball_strike":
            found = _best_frame(cues["strike"], lo, hi, score_peak)
            if found is not None:
                return found, "strike_event"
        else:
            attack = np.full(len(cues["box_pressure"]), np.nan)
            attack[lo : hi + 1] = np.where(holds, _attack_value(short, lo, hi), np.nan)
            found = _best_frame(attack, lo, hi, score_peak)
            if found is not None:
                return found, "attack_peak"
    elif kind == "duel_tackle":
        found = _min_duel_frame(data, cues, lo, hi, score_peak, half)
        if found is None:
            found = _best_frame(short["duel"], lo, hi, score_peak)
        if found is not None:
            return found, "duel_min_distance"
    elif kind == "sprint":
        found = _best_frame(data["ev_speed_mps"], lo, hi, score_peak)
        if found is None:
            found = _best_frame(short["speed"], lo, hi, score_peak)
        if found is not None:
            return found, "speed_peak"
    return score_peak, "score_peak"


def _window(
    peak: int, shots: ShotTable, cut_cue: np.ndarray, rate: Fraction
) -> tuple[int, int, int | None]:
    """Window around ``peak`` snapped to its shot, and the index of a joined following shot."""
    i = int(shots.of_frame[peak])
    first, last = int(shots.start[i]), int(shots.end[i])
    snap = _round_frames(SNAP_S, rate)
    start = max(first, peak - _round_frames(WINDOW_BEFORE_S, rate))
    if start - first <= snap:
        start = first
    end = min(last, peak + _round_frames(WINDOW_AFTER_S, rate))
    if last - end <= snap:
        end = last
    joined = None
    if (
        end == last
        and i + 1 < len(shots.start)
        and shots.shot_type[i] in EMPHASIS_FROM_TYPES
        and shots.shot_type[i + 1] in EMPHASIS_TO_TYPES
        and cut_cue[last] > 0
    ):
        follow = _round_frames(FOLLOW_MAX_S, rate)
        end = min(int(shots.end[i + 1]), int(shots.start[i + 1]) + follow - 1)
        joined = i + 1
    return start, end, joined


def _num(value: object, digits: int = 4) -> float | None:
    """A finite float rounded to ``digits`` for JSON, or None."""
    if value is None:
        return None
    result = float(value)  # type: ignore[arg-type]
    return round(result, digits) if math.isfinite(result) else None


def _int_or_none(value: object) -> int | None:
    result = int(value)  # type: ignore[call-overload]
    return result if result >= 0 else None


def detect_highlights(
    signals: pd.DataFrame,
    shots: pd.DataFrame,
    tracks: pd.DataFrame | None,
    ball_events: pd.DataFrame | None,
    fps: str | Fraction | int,
    *,
    min_separation_s: float = 4.0,
    max_highlights: int = 6,
    team_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Pick the highlight moments of a clip from its signals (the ``highlights.json`` dict).

    1. The score and every cue are smoothed with a Gaussian of :data:`SMOOTH_SIGMA_S`.
    2. Peaks are local maxima of the smoothed score of at least :data:`MIN_PEAK_SCORE`,
       taken in descending order (ties: the earlier frame) with non-maximum suppression: a
       peak closer than ``min_separation_s`` to an accepted one, before or after refinement,
       is dropped, until ``max_highlights`` are accepted.
    3. Type: among the play groups (ball_strike: strike; attack_in_box: box_pressure +
       ball_goal; duel_tackle: duel; sprint: speed) whose strongest smoothed cue at the peak
       reaches :data:`PLAY_EVIDENCE_MIN`, the one with the largest weighted smoothed
       contribution (ties in that order); when none reaches it, ``closeup_reaction``. The
       attack cues count here only with the ball at the box ("Ball at the box" in the
       module docstring), and box pressure only when two attackers
       (:data:`BOX_PRESSURE_ALONE_MIN`) were in the box near the peak, both before the
       groups compete.
    4. The peak frame is refined within :data:`REFINE_S` of the score peak, inside its shot,
       to the most precise local event: the strongest nearby ball event (ball_strike;
       attack_in_box takes one only on a frame where its own condition holds, else the
       strongest box_pressure + ball_goal there), the minimum duel distance, or the speed
       peak. Detection cues are averaged over :data:`REFINE_SMOOTH_S` first, so a one-frame
       flicker cannot become the peak.
    5. Window: ``[peak - 4 s, peak + 2 s]`` clamped to the peak's shot, an edge within
       :data:`SNAP_S` of a cut snapped to it; when the window reaches the end of a wide shot
       whose cut emphasis fired and a close or medium shot follows, the window is extended
       over that shot (at most :data:`FOLLOW_MAX_S`), and the strongest play moment inside
       that shot (a ball event or the closest duel, else the player it shows) is reported
       in ``cues["cut_emphasis"]["joined_shot_event"]`` and in the summary.
    6. ``track_ids``: the striker, the duel pair, the players nearest the ball and the
       fastest player near the peak (first the subject of the close-up for a
       closeup_reaction in a close shot), ordered by relevance for the type, then the
       players of the joined shot's moment; without -1 and duplicates, at most
       :data:`MAX_TRACK_IDS`.
    7. ``title_zh`` and ``summary_zh`` are composed from the cues and their evidence, with
       times as ``mm:ss.cc`` (:func:`format_time`) plus the frame number. Teams are named
       by ``team_names`` (fallback 左侧球队 / 右侧球队) from the track's rows inside the
       window (a tracker may reuse an id after a cut); jersey numbers appear only when
       recognised with confidence; the summary names every player with the track id
       (:func:`player_name_zh`); shots are named by the size of their players (全景, 中景,
       特写, see :data:`SHOT_UPPER_BODY_MIN_BOX_H_PX`). A closeup_reaction outside a close
       or medium shot names sound or editing in its title only when that cue's contribution
       reaches :data:`BROADCAST_CUE_MIN`, else no cue (综合评分入选的瞬间). Numbers in the
       text are marked as unverified, with the tier of the matching measurement where there
       is one, and name the frame they were read on.

    ``signals`` comes from :func:`compute_signals` (the ``ev_*`` columns are optional);
    ``tracks`` (``track_id`` plus optional ``frame``, ``team``, ``jersey``, ``jersey_conf``,
    ``role``, ``y1``, ``y2``) and ``ball_events`` may be None. Highlights are listed in time
    order with ids ``h01``, ``h02``, ...; each carries the contract keys plus
    ``score_peak_frame`` and ``refined_by``, and ``cues`` maps every cue to its smoothed
    ``value`` at the score peak, ``weight``, ``contribution`` (the contributions add up to
    ``score``), its strongest ``frame`` near the peak and that frame's evidence. Raises
    :class:`HighlightInputError` listing every problem with the inputs. Deterministic.
    """
    problems: list[str] = []
    rate = _rate(fps, problems)
    shot_table = _shot_table(shots, problems)
    sig = _table(signals, "signals", _SIGNAL_REQUIRED, _EVIDENCE_KINDS, problems)
    info = _table(tracks, "tracks", _TRACK_INFO_REQUIRED, _TRACK_INFO_OPTIONAL, problems)
    events = _table(ball_events, "ball_events", _EVENT_COLUMNS, _EVENT_OPTIONAL, problems)
    names = _team_names(team_names, problems)
    separation_s = _seconds(min_separation_s, "min_separation_s", problems)
    if (
        isinstance(max_highlights, bool)
        or not isinstance(max_highlights, int | np.integer)
        or max_highlights < 1
    ):
        problems.append(f"max_highlights must be an integer >= 1, got {max_highlights!r}")
    if sig is not None and shot_table is not None:
        problems.extend(_signal_problems(sig, shot_table))
    if events is not None and shot_table is not None:
        problems.extend(_frame_range_problems(events["frame"], "ball_events", shot_table.n_frames))
    if problems:
        raise HighlightInputError("detect_highlights", problems)
    return _detect(
        cast(dict[str, Any], sig),
        cast(ShotTable, shot_table),
        cast(dict[str, Any], info),
        cast(dict[str, Any], events),
        cast(Fraction, rate),
        separation_s,
        int(max_highlights),
        names,
    )


def _detect(
    sig: dict[str, Any],
    shots: ShotTable,
    info: dict[str, Any],
    events: dict[str, Any],
    rate: Fraction,
    separation_s: float,
    max_highlights: int,
    names: dict[str, str],
) -> dict[str, Any]:
    n = shots.n_frames
    order = np.argsort(sig["frame"], kind="stable")
    data: dict[str, Any] = {}
    for name in (*_SIGNAL_REQUIRED, *_EVIDENCE_KINDS):
        if name in sig:
            data[name] = sig[name][order]
        else:
            kind = _EVIDENCE_KINDS[name]
            data[name] = np.full(n, _EVIDENCE_DEFAULTS[kind], dtype=_EVIDENCE_DTYPES[kind])
    cues = {name: np.nan_to_num(data[name], nan=0.0) for name in CUE_COLUMNS}
    sigma = float(_exact_seconds(SMOOTH_SIGMA_S) * rate)
    smooth_score = gaussian_smooth(data["score"], sigma)
    smooth_cues = {name: gaussian_smooth(cues[name], sigma) for name in CUE_COLUMNS}
    half = int(math.floor(_exact_seconds(REFINE_SMOOTH_S) * rate / 2))
    short = {name: _short_mean(cues[name], half) for name in CUE_COLUMNS}
    # The cues that decide the type: the attack cues only where the ball is at the box.
    at_box = _at_box(data["ev_ball_box_m"])
    type_cues = {
        name: np.where(at_box, cues[name], 0.0) if name in ATTACK_CUES else cues[name]
        for name in CUE_COLUMNS
    }
    type_smooth = {
        name: gaussian_smooth(type_cues[name], sigma) if name in ATTACK_CUES else smooth_cues[name]
        for name in CUE_COLUMNS
    }
    type_short = {
        name: _short_mean(type_cues[name], half) if name in ATTACK_CUES else short[name]
        for name in CUE_COLUMNS
    }

    rising = np.concatenate(([True], smooth_score[1:] >= smooth_score[:-1]))
    falling = np.concatenate((smooth_score[:-1] >= smooth_score[1:], [True]))
    candidates = np.flatnonzero(rising & falling & (smooth_score >= MIN_PEAK_SCORE))
    candidates = candidates[np.lexsort((candidates, -smooth_score[candidates]))]

    separation = _round_frames(separation_s, rate)
    reach = _round_frames(REFINE_S, rate)
    picks: list[_Pick] = []
    for score_peak in candidates.tolist():
        if len(picks) >= max_highlights:
            break
        if any(abs(score_peak - p.score_peak) < separation for p in picks):
            continue
        shot = int(shots.of_frame[score_peak])
        lo = max(int(shots.start[shot]), score_peak - reach)
        hi = min(int(shots.end[shot]), score_peak + reach)
        kind = _classify(type_smooth, score_peak, type_cues, lo, hi)
        peak, method = _refine(kind, score_peak, lo, hi, type_cues, type_short, data, events, half)
        if any(abs(peak - p.peak) < separation for p in picks):
            continue
        picks.append(_Pick(score_peak, peak, kind, method))

    context = _Context(
        shots=shots,
        data=data,
        cues=cues,
        short=short,
        type_short=type_short,
        smooth_cues=smooth_cues,
        smooth_score=smooth_score,
        info=info,
        events=events,
        track_info=_track_info(info),
        names=names,
        rate=rate,
        reach=reach,
        half=half,
    )
    ordered = sorted(picks, key=lambda p: p.peak)
    highlights = [_describe(number, pick, context) for number, pick in enumerate(ordered, 1)]
    return {
        "schema_version": 1,
        "fps": format_rate(rate),
        "frame_count": n,
        "highlights": highlights,
    }


def player_name_zh(track: int, team_name: str | None = None, jersey: str | None = None) -> str:
    """How a summary names a player: the team and a confident jersey number when known, and
    always the track id, so that two players of one team are told apart
    (甲队 9 号（轨迹 7）, 甲队球员（轨迹 7）, 9 号球员（轨迹 7）, 轨迹 7 球员)."""
    if team_name and jersey:
        return f"{team_name} {jersey} 号（轨迹 {track}）"
    if team_name:
        return f"{team_name}球员（轨迹 {track}）"
    if jersey:
        return f"{jersey} 号球员（轨迹 {track}）"
    return f"轨迹 {track} 球员"


def player_names_zh(
    tracks: pd.DataFrame | None,
    track_ids: Sequence[int],
    lo: int,
    hi: int,
    team_names: Mapping[str, str] | None = None,
) -> dict[int, str]:
    """:func:`player_name_zh` of each of ``track_ids`` from its rows of ``tracks`` in frames
    ``lo..hi`` (majority team, None on a tie; a jersey number read with confidence), as the
    summaries of :func:`detect_highlights` name them. Raises :class:`HighlightInputError`."""
    problems: list[str] = []
    names = _team_names(team_names, problems)
    info = _table(tracks, "tracks", _TRACK_INFO_REQUIRED, _TRACK_INFO_OPTIONAL, problems)
    if problems or info is None:
        raise HighlightInputError("player_names_zh", problems)
    known = _track_info(info, lo, hi)
    result = {}
    for track in track_ids:
        entry = known.get(int(track), _TrackInfo(None, None))
        team = names.get(entry.team) if entry.team else None
        result[int(track)] = player_name_zh(int(track), team, entry.jersey)
    return result


@dataclass(frozen=True)
class _Context:
    shots: ShotTable
    data: dict[str, Any]
    cues: dict[str, np.ndarray]
    short: dict[str, np.ndarray]
    type_short: dict[str, np.ndarray]
    smooth_cues: dict[str, np.ndarray]
    smooth_score: np.ndarray
    info: dict[str, Any]
    events: dict[str, Any]
    track_info: dict[int, _TrackInfo]
    names: dict[str, str]
    rate: Fraction
    reach: int
    half: int

    def label(self, track: int) -> str | None:
        """Chinese label of a player in a title: team name and confident jersey number when
        known."""
        if track < 0:
            return None
        info = self.track_info.get(track, _TrackInfo(None, None))
        team = self.names.get(info.team) if info.team else None
        if team and info.jersey:
            return f"{team} {info.jersey} 号"
        if team:
            return f"{team}球员"
        if info.jersey:
            return f"{info.jersey} 号球员"
        return "一名球员"

    def name(self, track: int) -> str:
        """How the summary names a player (:func:`player_name_zh`), with the track id."""
        info = self.track_info.get(track, _TrackInfo(None, None))
        team = self.names.get(info.team) if info.team else None
        return player_name_zh(track, team, info.jersey)

    def team_name(self, team: object) -> str | None:
        return self.names[team] if isinstance(team, str) and team in _TEAMS else None


def _describe(number: int, pick: _Pick, ctx: _Context) -> dict[str, Any]:
    """The highlights.json entry of one pick: window, ids, cue evidence and Chinese text."""
    data, cues, rate = ctx.data, ctx.cues, ctx.rate
    peak = pick.peak
    start, end, joined_index = _window(peak, ctx.shots, cues["cut_emphasis"], rate)
    joined = None if joined_index is None else int(ctx.shots.shot_id[joined_index])
    # Name players from their rows inside this window (ids can be reused after a cut).
    ctx = replace(ctx, track_info={**ctx.track_info, **_track_info(ctx.info, start, end)})
    # The evidence is read near the precise peak and inside its own shot.
    shot = int(ctx.shots.of_frame[peak])
    lo = max(start, peak - ctx.reach, int(ctx.shots.start[shot]))
    hi = min(end, peak + ctx.reach, int(ctx.shots.end[shot]))
    total = sum(CUE_WEIGHTS.values())

    # Evidence frame of every cue: its strongest frame near the precise peak (detection
    # cues averaged over REFINE_SMOOTH_S, as for the refinement); an attack_in_box names the
    # attack cues where they counted for its type, with the ball at the box.
    evidence_frame: dict[str, int | None] = {}
    for name in CUE_COLUMNS:
        values = ctx.short[name] if name in ("speed", "box_pressure", "ball_goal") else cues[name]
        if pick.kind == "attack_in_box" and name in ATTACK_CUES:
            values = ctx.type_short[name]
        evidence_frame[name] = _best_frame(values, lo, hi, peak)
    duel_at = _min_duel_frame(data, cues, lo, hi, peak, ctx.half)
    evidence_frame["duel"] = duel_at if duel_at is not None else evidence_frame["duel"]
    fastest_frame = _best_frame(data["ev_speed_mps"], lo, hi, peak)
    if fastest_frame is not None:
        evidence_frame["speed"] = fastest_frame

    def at(column: str, cue: str) -> Any:
        frame = evidence_frame[cue]
        return None if frame is None else data[column][frame]

    striker = -1
    if pick.method == "strike_event":
        rows = np.flatnonzero(ctx.events["frame"] == peak)
        if rows.size:
            strength = _event_strengths(ctx.events)[rows]
            best = rows[np.lexsort((rows, -strength))[0]]
            striker = int(_event_track_ids(ctx.events)[best])
    duel_frame = evidence_frame["duel"]
    duel_a = int(data["ev_duel_id_a"][duel_frame]) if duel_frame is not None else -1
    duel_b = int(data["ev_duel_id_b"][duel_frame]) if duel_frame is not None else -1
    near = _nearest_valid(data["ev_ball_near_id"], lo, hi, peak)
    opp = _nearest_valid(data["ev_ball_opp_id"], lo, hi, peak)
    fastest = int(data["ev_speed_id"][fastest_frame]) if fastest_frame is not None else -1
    shot_type = ctx.shots.shot_type[shot]
    close_up = pick.kind == "closeup_reaction" and shot_type in EMPHASIS_TO_TYPES
    shot_lo = max(start, int(ctx.shots.start[shot]))
    shot_hi = min(end, int(ctx.shots.end[shot]))
    subject = _subject(ctx.info, shot_lo, shot_hi)[0] if close_up else -1
    by_kind = {
        "ball_strike": [striker, near, opp, duel_a, duel_b, fastest],
        "attack_in_box": [near, striker, opp, duel_a, duel_b, fastest],
        "duel_tackle": [duel_a, duel_b, near, opp, striker, fastest],
        "sprint": [fastest, near, opp, duel_a, duel_b, striker],
        "closeup_reaction": [subject, near, opp, duel_a, duel_b, striker, fastest],
    }
    moment = None if joined_index is None else _joined_moment(ctx, joined_index, end)
    track_ids: list[int] = []
    for track in by_kind[pick.kind] + ([] if moment is None else moment["track_ids"]):
        if track >= 0 and track not in track_ids:
            track_ids.append(int(track))
    track_ids = track_ids[:MAX_TRACK_IDS]

    cue_json: dict[str, Any] = {}
    reasons: list[tuple[float, str]] = []
    for name in CUE_COLUMNS:
        value = float(ctx.smooth_cues[name][pick.score_peak])
        contribution = CUE_WEIGHTS[name] * value / total
        entry: dict[str, Any] = {
            "value": _num(value),
            "weight": CUE_WEIGHTS[name],
            "contribution": _num(contribution),
            "frame": evidence_frame[name],
        }
        text = _cue_evidence(name, entry, at, ctx, joined, (duel_a, duel_b))
        cue_json[name] = entry
        if text is not None and contribution >= 0.005:
            reasons.append((contribution, text))
    reasons.sort(key=lambda item: -item[0])
    cue_json["cut_emphasis"]["joined_shot_event"] = moment

    when = f"{format_time(peak, rate)}（第 {peak} 帧）"
    attack_team = ctx.team_name(cue_json["box_pressure"]["team"]) or ctx.team_name(
        cue_json["ball_goal"]["team"]
    )
    if pick.kind == "duel_tackle" and duel_a >= 0 and duel_b >= 0:
        title = f"{when}{ctx.label(duel_a)}与{ctx.label(duel_b)}贴身争抢"
    elif pick.kind == "duel_tackle":
        title = f"{when}双方球员贴身争抢"
    elif pick.kind == "attack_in_box":
        title = f"{when}{attack_team or '进攻方'}在对方禁区内进攻"
    elif pick.kind == "ball_strike":
        who = ctx.label(striker) or ctx.label(near)
        title = f"{when}{who or ''}{TYPE_NAMES_ZH['ball_strike']}"
    elif pick.kind == "sprint":
        title = f"{when}{ctx.label(fastest) or '球员'}高速冲刺"
    elif shot_type in EMPHASIS_TO_TYPES:
        who = ctx.label(subject)
        name = _shot_name(ctx.shots, shot)
        title = f"{when}{name}：{who}的反应" if who else f"{when}{name}镜头中的现场反应"
    else:
        # No play cue was strong enough: name a broadcast cue only when it carried the
        # moment, else claim none.
        sound = (cue_json["audio"]["contribution"] or 0.0) >= BROADCAST_CUE_MIN
        editing = (cue_json["cut_emphasis"]["contribution"] or 0.0) >= BROADCAST_CUE_MIN
        if sound and editing:
            title = f"{when}声音与导播线索标记的精彩瞬间"
        elif editing:
            title = f"{when}导播切镜强调的瞬间"
        elif sound:
            title = f"{when}现场声音响度变化明显的瞬间"
        else:
            title = f"{when}综合评分入选的瞬间"

    parts = [
        f"{TYPE_NAMES_ZH[pick.kind]}。片段 {format_time(start, rate)} 至 "
        f"{format_time(end, rate)}（第 {start}–{end} 帧），关键帧 {format_time(peak, rate)}"
        f"（第 {peak} 帧，{REFINE_METHODS_ZH[pick.method]}）。"
    ]
    if joined_index is not None:
        parts.append(_joined_sentence(ctx, joined_index, end, moment))
    if reasons:
        parts.append("入选依据：" + "；".join(text for _, text in reasons[:4]) + "。")
    if track_ids:
        parts.append(f"涉及球员：{'、'.join(ctx.name(track) for track in track_ids)}。")
    parts.append(
        f"评分 {float(ctx.smooth_score[pick.score_peak]):.2f}，仅用于挑选片段；"
        "文中数值均为自动估计，误差未测量（未验证）。"
    )
    return {
        "id": f"h{number:02d}",
        "type": pick.kind,
        "start_frame": int(start),
        "peak_frame": int(peak),
        "end_frame": int(end),
        "score": _num(ctx.smooth_score[pick.score_peak]),
        "cues": cue_json,
        "track_ids": track_ids,
        "title_zh": title,
        "summary_zh": "".join(parts),
        "score_peak_frame": int(pick.score_peak),
        "refined_by": pick.method,
    }


def _joined_moment(ctx: _Context, shot_index: int, end: int) -> dict[str, Any] | None:
    """The strongest play moment inside a joined close or medium shot, or None.

    The strongest ball event (weighted strength) competes with the closest duel (weighted
    cue averaged over REFINE_SMOOTH_S); the winner needs a value of at least
    :data:`PLAY_EVIDENCE_MIN`. Without one, the player seen largest for longest names its
    subject. The result is JSON-ready: ``frame``, ``cue`` (``strike``, ``duel`` or
    ``subject``), ``track_ids`` and, for a duel, ``distance`` and ``unit``.
    """
    lo = int(ctx.shots.start[shot_index])
    hi = min(int(ctx.shots.end[shot_index]), end)
    options: list[tuple[float, int, str]] = []
    frames, strength = _event_frames(ctx.events, lo, hi)
    strike_row = -1
    if frames.size:
        strike_row = int(np.lexsort((frames, -strength))[0])
        if strength[strike_row] >= PLAY_EVIDENCE_MIN:
            weighted = CUE_WEIGHTS["strike"] * float(strength[strike_row])
            options.append((weighted, int(frames[strike_row]), "strike"))
    duel_frame = _min_duel_frame(ctx.data, ctx.cues, lo, hi, lo, ctx.half)
    if duel_frame is not None and ctx.short["duel"][duel_frame] >= PLAY_EVIDENCE_MIN:
        options.append(
            (CUE_WEIGHTS["duel"] * float(ctx.short["duel"][duel_frame]), duel_frame, "duel")
        )
    if options:
        _, frame, cue = sorted(options, key=lambda option: (-option[0], option[1]))[0]
        if cue == "strike":
            rows = np.flatnonzero(ctx.events["frame"] == frame)
            best = rows[np.lexsort((rows, -_event_strengths(ctx.events)[rows]))[0]]
            striker = int(_event_track_ids(ctx.events)[best])
            return {
                "frame": frame,
                "cue": "strike",
                "strength": _num(_event_strengths(ctx.events)[best]),
                "track_ids": [striker] if striker >= 0 else [],
            }
        pair = [int(ctx.data["ev_duel_id_a"][frame]), int(ctx.data["ev_duel_id_b"][frame])]
        unit = ctx.data["ev_duel_unit"][frame]
        return {
            "frame": frame,
            "cue": "duel",
            "distance": _num(ctx.data["ev_duel_dist"][frame]),
            "unit": unit if unit in ("m", "box_h") else None,
            "track_ids": [track for track in pair if track >= 0],
        }
    subject, frame = _subject(ctx.info, lo, hi)
    if subject >= 0:
        return {"frame": frame, "cue": "subject", "track_ids": [subject]}
    return None


def shot_name_zh(shot_type: str, median_box_h: float | None) -> str:
    """Chinese name of a shot by the size of its players: 全景, 中景 or 特写.

    ``median_box_h`` is the shot's ``shots.median_box_h`` (1080-line pixels; None or NaN
    when unknown): a close shot whose players are below :data:`SHOT_UPPER_BODY_MIN_BOX_H_PX`
    is a 中景. Other types come back unchanged.
    """
    height = math.nan if median_box_h is None else float(median_box_h)
    if shot_type == "close" and math.isfinite(height) and height < SHOT_UPPER_BODY_MIN_BOX_H_PX:
        return SHOT_TYPE_NAMES_ZH["medium"]
    return SHOT_TYPE_NAMES_ZH.get(shot_type, shot_type)


def _shot_name(shots: ShotTable, index: int) -> str:
    return shot_name_zh(shots.shot_type[index], float(shots.median_box_h[index]))


def _joined_sentence(
    ctx: _Context, shot_index: int, end: int, moment: dict[str, Any] | None
) -> str:
    """Chinese sentence about the close or medium shot that joined a window."""
    name = _shot_name(ctx.shots, shot_index)
    lo = int(ctx.shots.start[shot_index])
    head = f"片段延伸到导播随后切入的{name}（第 {lo}–{end} 帧）"
    if moment is None:
        return head + "。"
    frame = int(moment["frame"])
    when = f"{format_time(frame, ctx.rate)}（第 {frame} 帧）"
    ids = moment["track_ids"]
    if moment["cue"] == "duel" and moment["distance"] is not None and len(ids) == 2:
        pair = f"{ctx.name(ids[0])}与{ctx.name(ids[1])}"
        if moment["unit"] == "m":
            size = f"这一帧相距约 {moment['distance']:.1f} 米（A 档，未验证）"
        else:
            size = f"这一帧相距约 {moment['distance']:.1f} 个身高（像面估计，未验证）"
        return f"{head}：{pair}平滑后的距离在 {when}最近，{size}。"
    if moment["cue"] == "strike":
        who = ctx.name(ids[0]) if ids else ""
        return f"{head}：{when}{who}触球（球的像面速度突变）。"
    if moment["cue"] == "subject" and ids:
        return f"{head}，画面主体是{ctx.name(ids[0])}。"
    return head + "。"


def _chord_s_text(rate: Fraction) -> str:
    """Duration of the chord a speed is measured over, as the data card's label states it:
    2 * gbf.kinematics.chord_half_frames frames ("0.4" at 25 fps, "0.42" at 24 fps)."""
    seconds = float(Fraction(2 * chord_half_frames(rate, DEFAULT_SPEED_CHORD_S)) / rate)
    return f"{seconds:.2f}".rstrip("0").rstrip(".")


def _cue_evidence(
    name: str,
    entry: dict[str, Any],
    at: Any,
    ctx: _Context,
    joined: int | None,
    duel_pair: tuple[int, int],
) -> str | None:
    """Fill the evidence fields of one cue entry; return its Chinese reason or None.

    A number read on one frame names that frame (``entry["frame"]``): the measurements of
    the report are taken on other frames (the key frame, or a window around it). Numbers
    are stored with gbf.measure's four decimals and rounded once for the text, as the
    measurement table rounds them; a speed is the speed on its frame (the chord around it),
    neither called the player's peak nor said not to be one (nothing here compares it with
    the player's other frames). Players are named with their track id
    (:func:`player_name_zh`).
    """
    rate = ctx.rate
    at_frame = f"第 {entry['frame']} 帧"
    if name == "cut_emphasis":
        cut = at("ev_cut_frame", name)
        entry["cut_frame"] = None if cut is None else _int_or_none(cut)
        entry["extended_into_shot_id"] = joined
        if entry["cut_frame"] is None:
            return None
        cut_frame = int(entry["cut_frame"])
        after = int(ctx.shots.of_frame[cut_frame])
        before = _shot_name(ctx.shots, after - 1) if after > 0 else SHOT_TYPE_NAMES_ZH["wide"]
        return (
            f"导播在 {format_time(cut_frame, rate)}（第 {cut_frame} 帧）由{before}切到"
            f"{_shot_name(ctx.shots, after)}"
        )
    if name == "audio":
        z = at("ev_audio_z", name)
        entry["excitement_z"] = None if z is None else _num(z, 2)
        if entry["excitement_z"] is None:
            return None
        return f"现场声音响度变化 z = {entry['excitement_z']:.1f}"
    if name == "speed":
        mps = at("ev_speed_mps", name)
        track = at("ev_speed_id", name)
        entry["speed_mps"] = None if mps is None else _num(mps)
        entry["track_id"] = None if track is None else _int_or_none(track)
        if entry["speed_mps"] is None:
            return None
        who = ctx.name(entry["track_id"]) if entry["track_id"] is not None else "一名球员"
        return (
            f"{at_frame}{who}的速度约 {entry['speed_mps'] * 3.6:.1f} km/h"
            f"（该帧前后 {_chord_s_text(rate)} 秒平均；A 档，未验证）"
        )
    if name in ATTACK_CUES:
        box = at("ev_ball_box_m", name)
        entry["ball_box_m"] = None if box is None else _num(box)
    if name == "box_pressure":
        count = at("ev_box_attackers", name)
        team = at("ev_box_team", name)
        entry["attackers"] = 0 if count is None else max(0, int(count))
        entry["team"] = team if isinstance(team, str) and team in _TEAMS else None
        if entry["attackers"] <= 0:
            return None
        side = ctx.team_name(entry["team"]) or "进攻方"
        return f"{at_frame}{side} {entry['attackers']} 名球员位于对方禁区内（B 档，实验，未验证）"
    if name == "ball_goal":
        metres = at("ev_ball_goal_m", name)
        team = at("ev_ball_team", name)
        entry["distance_m"] = None if metres is None else _num(metres)
        entry["team"] = team if isinstance(team, str) and team in _TEAMS else None
        if entry["distance_m"] is None:
            return None
        side = ctx.team_name(entry["team"]) or "控球方"
        return f"{at_frame}球距{side}进攻的球门约 {entry['distance_m']:.0f} 米（地面近似，未验证）"
    if name == "duel":
        dist = at("ev_duel_dist", name)
        unit = at("ev_duel_unit", name)
        entry["distance"] = None if dist is None else _num(dist)
        entry["unit"] = unit if unit in ("m", "box_h") else None
        entry["track_ids"] = [int(t) for t in duel_pair if t >= 0]
        if entry["distance"] is None or entry["unit"] is None:
            return None
        pair = "与".join(ctx.name(t) if t >= 0 else "一名球员" for t in duel_pair)
        if entry["unit"] == "m":
            return f"{at_frame}{pair}相距约 {entry['distance']:.1f} 米（A 档，未验证）"
        return f"{at_frame}{pair}相距约 {entry['distance']:.1f} 个身高（像面估计，未验证）"
    # strike
    event = at("ev_strike_frame", name)
    entry["event_frame"] = None if event is None else _int_or_none(event)
    if entry["event_frame"] is None:
        return None
    return f"第 {entry['event_frame']} 帧检测到触球（球的像面速度突变）"
