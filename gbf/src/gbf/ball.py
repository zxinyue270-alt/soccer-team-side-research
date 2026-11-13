"""Ball tracking on the original frames, and ball-strike events (milestone M1).

The ball is tracked in the image of the original broadcast video. A pitch position is only a
ground-plane approximation, filled on pitch-usable frames. Column contracts follow
docs/design/m1-highlights.md (``ball_track.parquet``).

Inputs
    ``ball_raw.jsonl`` from the external ball detector, one line per processed frame:
    ``{"frame": n, "cands": [[x1, y1, x2, y2, conf], ...]}``: COCO "sports ball" boxes in
    pixels of the video frame with a detector confidence in [0, 1]. A frame missing from the
    file (not processed) and a frame with an empty list both give the tracker no candidate.
    ``tracks``, ``calib`` and ``shots``: DataFrames with the contract columns of
    ``tracks.parquet``, ``calib.parquet`` (calib.v1 of gbf.plugins.calib plus
    ``pitch_usable``) and ``shots.parquet``. The columns read here are TRACKS_COLUMNS_READ,
    CALIB_COLUMNS_READ (``frame`` is accepted in place of ``frame_index``, as in
    ``calib.jsonl``; ``width``/``height`` are optional) and SHOTS_COLUMNS_READ (plus the
    optional ``median_box_h``, 1080-line pixels). K and R are 9 numbers (row-major) or 3x3,
    C is 3 numbers; only rows with ``pitch_usable`` True are read, and their camera is
    rescaled (Camera.resized) when the calibrated width x height differs from the video.

Conventions
    Pixels: (u, v) with u right and v down. Box coordinates of the ball detector and of
    ``tracks`` are used unchanged, in the pixel grid of the exports, as gbf.sngs_import
    does for the foot points (foot_u = (x1 + x2) / 2, foot_v = y2): the calibration comes
    from the same pixel grid, so rays through these pixels stay exact; relative to
    gbf.geometry's integer pixel centres they may sit half a pixel off, far below detection
    noise. The ball centre is the box centre ((x1 + x2) / 2, (y1 + y2) / 2), its diameter
    in pixels is the mean of the box width and height, and radius_px is half of it.
    World coordinates are SoccerNet metres (origin at the centre spot, far touchline at
    y = -34, z down). A ball on the grass has its centre at z = -BallParams.ball_radius_m.
    Frames are canonical CFR indices. Durations are parameters in seconds, converted to
    frames as round(seconds * fps); the frame counts quoted below are those at 25 fps.
    "bh" (box heights) measure image lengths in units of the local player box height, which
    takes out the perspective scale. A box is about 1.8 m tall, so at 25 fps 0.7 bh per
    frame is about 30 m/s, and 1 bh/s about 1.8 m/s. Ball motion is measured against the
    box height at the ball's own depth, ball box diameter / 0.155 (BallParams.
    ball_to_box_height, measured): in close and low-angle shots the players beside the
    ball in the image often stand much nearer or farther.
    A player is "at the ball" when the ball centre lies inside the player's box grown by
    0.3 of its height to the left, the right and below (roles player and goalkeeper).

Pipeline
    1. load_candidates reads ``ball_raw.jsonl``.
    2. gate_candidates tests every raw candidate; the first failing test, in this order,
       names the ``reason``, and every test also has its own column:
       ``aspect``: box width / height outside [0.6, 1.6], unless the box is a
       motion-blurred fast ball: detector conf >= 0.6, long side <= 3 x short side, clear
       of the frame edges, and a candidate on the previous or next frame of the same shot
       0.4 to 2 long sides away along the long axis. Such a box's diameter is its short
       side; every other box's is (w + h) / 2.
       ``size``: diameter / median box height of the players whose box centre is within 3
       of their own box heights horizontally outside [0.05, 0.3]; with no such player,
       diameter outside [4, 120] px scaled to 1080 lines.
       ``off_pitch`` (pitch-usable frames only): the ray through the box centre meets the
       plane of a grounded ball's centre outside |x| <= 55 m, |y| <= 36.5 m, or not at all.
       ``static``: a screen-fixed distractor such as a broadcast graphic. A screen-fixed
       run is a chain of candidates within 3 px of its first position on at least 25 frames
       (gaps of at most 3 frames), built over the whole clip. It is ``static`` when the
       camera moved during the run: the run crosses a cut, or the scene point under it
       moved by at least 10 px on at least 3 frames, measured with the calibration when at
       least 4 of its frames are pitch-usable, otherwise with the tracked players' median
       image motion. Players also run, so with that weaker evidence the frames where a
       player is at the ball stay eligible.
       ``off_field_static``: a world-fixed object outside the field lines, such as the
       spare balls on their stands at the ad boards (pitch-usable frames only): the
       candidate's ground point lies more than 0.5 m outside the field lines, and the
       candidates of the same shot within 1.5 m of it come from at least 6 frames spanning
       at least 1 s, with their median more than 1 m outside.
       ``parked``: a slow chain, each member within 3 px (at 1080 lines) per frame of the
       previous one, on at least 25 frames (gaps of at most 3), on the frames where no
       player is at the ball: a spare ball or a cone under a still or slowly drifting
       camera, or a graphic under a still one. A match ball waiting for a set piece is
       parked too until a player comes to it, which keeps the frames before a kick.
       Both chain tests first set aside second detector boxes on the same object (within
       half a diameter of a more confident box of the same frame), which take the verdict
       of that box; otherwise parallel chains would starve each other.
    3. track_ball runs one Viterbi per shot over the states {gated candidates, missing}.
       The path maximises the sum of: ln(conf) + a log-normal size prior (expected box
       diameter from the calibration on pitch-usable frames, 1.2 x the projected 0.22 m,
       sigma 0.35 in log; else 0.155 x the nearby box height, sigma 0.7) for each chosen
       candidate; minus a link cost between successive chosen candidates (speed in bh,
       quadratic up to 0.7 bh per frame and linear beyond, plus a size-continuity term;
       links span at most 8 frames and 1.2 bh per frame); minus a constant cost (2.0) per
       missing frame; minus a re-acquisition cost (3.0) when the ball is picked up with no
       link. A chosen detection with no other one within 3 frames on both sides and a
       confidence below 0.6 is then dropped (a stepping stone across an occlusion). Gaps
       of at most 6 frames between two LINKED detections are interpolated linearly
       (status ``interpolated``) when the implied speed is plausible; a re-acquisition and
       a cut are never bridged. pitch_x, pitch_y are filled on pitch-usable frames for
       detected and interpolated rows by assuming the ball is on the grass
       (ground_assumption True).
    4. ball_events finds abrupt changes of the ball's image velocity from detected samples
       only, within a shot (see its docstring).

Known limits
    - A ball in the air projects beyond its true ground position. Near the far touchline
      (and when it is very high anywhere in the far half) it can therefore fail the
      ``off_pitch`` gate, which is there because ad boards and stands start just beyond
      y = -34, and be lost while it is high; its pitch_x, pitch_y are wrong whenever it is
      off the grass. The kick itself happens on the grass, so ball-strike events survive.
    - The camera-motion verdict from players is weak (players run together), and a
      perfectly still camera leaves a broadcast graphic ``parked`` rather than ``static``.
    - With a still camera and no player at it, the match ball at rest is ``parked``.
    - Image velocity includes camera motion. Pans are smooth compared with a kick, and the
      event thresholds include a minimum velocity change, but a fast pan start or stop
      next to a slow ball can still be reported.
    - All thresholds are empirical starting values (BallParams), not validated ones.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from gbf.geometry import Camera, GeometryInputError, pixel_to_ground, project, world_to_camera
from gbf.ingest.rates import parse_rate

__all__ = [
    "BALL_EVENT_COLUMNS",
    "BALL_TRACK_COLUMNS",
    "CALIB_COLUMNS_READ",
    "CONTACT_PARAMS",
    "DEFAULT_PARAMS",
    "DETECTED",
    "EVENT_KINDS",
    "GATE_COLUMNS",
    "GATE_REASONS",
    "INTERPOLATED",
    "MISSING",
    "SHOTS_COLUMNS_READ",
    "STATUSES",
    "TRACKS_COLUMNS_READ",
    "BallInputError",
    "BallParams",
    "ball_events",
    "gate_candidates",
    "load_candidates",
    "track_ball",
]

DETECTED = "detected"
INTERPOLATED = "interpolated"
MISSING = "missing"
STATUSES: tuple[str, ...] = (DETECTED, INTERPOLATED, MISSING)

GATE_REASONS: tuple[str, ...] = (
    "ok",
    "aspect",
    "size",
    "off_pitch",
    "static",
    "off_field_static",
    "parked",
)
"""Values of the gate table's ``reason`` column: "ok" or the first test that failed."""

EVENT_KINDS: tuple[str, ...] = ("direction_change", "speed_up", "slow_down")

TRACKS_COLUMNS_READ: tuple[str, ...] = (
    "frame",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "role",
    "team",
    "foot_u",
    "foot_v",
)
CALIB_COLUMNS_READ: tuple[str, ...] = ("frame_index", "pitch_usable", "K", "R", "C")
SHOTS_COLUMNS_READ: tuple[str, ...] = ("shot_id", "start_frame", "end_frame")

FIELD_HALF_LENGTH_M = 52.5
FIELD_HALF_WIDTH_M = 34.0

BALL_TRACK_COLUMNS: tuple[str, ...] = (
    "frame",
    "u",
    "v",
    "radius_px",
    "conf",
    "status",
    "candidate_count",
    "pitch_x",
    "pitch_y",
    "ground_assumption",
)
"""Columns of track_ball's result (``ball_track.parquet``), in order.

frame int64; u, v float64 (pixels, NaN when missing); radius_px float64 (px, NaN when
missing); conf float32 (the detector confidence of the chosen candidate, NaN unless
detected); status string (detected, interpolated, missing); candidate_count int32 (raw
detector candidates in that frame, before gating); pitch_x, pitch_y float64 (SoccerNet
metres, ground-plane approximation, NaN unless the frame is pitch-usable and the ball was
detected or interpolated); ground_assumption bool (True exactly where pitch_x is set).
"""

BALL_EVENT_COLUMNS: tuple[str, ...] = (
    "frame",
    "kind",
    "magnitude",
    "u",
    "v",
    "nearest_track_id",
    "nearest_team",
    "nearest_dist_box_h",
    "shot_id",
    "angle_deg",
    "speed_before",
    "speed_after",
    "at_foot",
)
"""Columns of ball_events' result, in order.

frame int64 (the vertex: the last frame on the incoming path and the first on the outgoing
one); kind string (EVENT_KINDS); magnitude float64 (|v_after - v_before| in bh/s); u, v
float64 (ball position at that frame, pixels); nearest_track_id int64 (-1 if none);
nearest_team string or missing; nearest_dist_box_h float64 (distance from the ball centre to
that player's foot point, in that player's box heights; NaN if none); shot_id int64;
angle_deg float64 (between the incoming and outgoing velocity, NaN if either is zero);
speed_before, speed_after float64 (bh/s, the ball's own scale); at_foot bool
(nearest_dist_box_h <= event_at_foot_bh). The first eight columns are the contract; the
rest are evidence for auditing an event.
"""

GATE_COLUMNS: tuple[str, ...] = (
    "frame",
    "cand_index",
    "shot_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "conf",
    "u",
    "v",
    "diameter_px",
    "aspect",
    "n_nearby",
    "ref_box_h",
    "size_ratio",
    "pitch_usable",
    "pitch_x",
    "pitch_y",
    "expected_diameter_px",
    "size_prior",
    "scale_px",
    "near_player",
    "static_run",
    "aspect_ok",
    "size_ok",
    "pitch_ok",
    "static_ok",
    "field_static_ok",
    "parked_ok",
    "reason",
)
"""Columns of gate_candidates' result: one row per raw candidate, ordered by frame and
cand_index (the candidate's position in its ``cands`` list).

n_nearby / ref_box_h: players within ``nearby_box_heights`` horizontally and their median
box height (NaN when none); size_ratio = diameter_px / ref_box_h; pitch_x, pitch_y: where
the ray through the box centre meets the plane z = -ball_radius_m (NaN unless pitch-usable
or when the ray misses); expected_diameter_px: the size prior's centre (NaN when unknown);
size_prior: its log-likelihood term; scale_px: diameter_px / ball_to_box_height, the box
height at the ball's depth that normalises motion; near_player: a player is at the ball;
static_run: members of the anchored screen-fixed chain the candidate belongs to (1 when
alone; a duplicate box on the same object reports its representative's chain); the
``*_ok`` columns: each test on its own; reason: "ok" or the first failure (GATE_REASONS).
"""


class BallInputError(ValueError):
    """Malformed input to gbf.ball. ``problems`` lists every problem found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


@dataclass(frozen=True)
class BallParams:
    """Every threshold of gbf.ball. All of them are empirical starting values.

    Durations are seconds (frames at 25 fps in brackets), image speeds are in bh per
    second (see the module docstring), costs are in units of natural-log likelihood.
    """

    # --- gating -------------------------------------------------------------------------
    aspect_min: float = 0.6
    """Smallest box width / height of a ball (a blurred fast ball is elongated)."""
    aspect_max: float = 1.6
    """Largest box width / height of a ball."""
    blur_min_conf: float = 0.6
    """A box outside [aspect_min, aspect_max] still passes the aspect test as a
    motion-blurred fast ball when the detector is at least this confident ..."""
    blur_aspect_max: float = 3.0
    """... and its long side is at most this many times its short side. The diameter of
    such a box is its short side: blur stretches the box only along the motion (on the
    development clip a fast ball gave boxes twice as long as wide at high confidence for a
    few frames, and a spare ball was tracked instead until this exception)."""
    blur_min_step: float = 0.4
    """A blurred box also needs motion: a candidate on the previous or next frame of the
    same shot (never across a cut) at least this many of its long sides away, ..."""
    blur_max_step: float = 2.0
    """... and at most this many: blur stretches a ball of diameter d moving s px per frame
    to a long side L = d + e s, e being the share of the frame interval the shutter is
    open, so the ball moves (1 - d / L) / e long sides per frame, at the longest accepted
    blur (blur_aspect_max) 4/3 with e = 1/2 and 2 with e = 1/3. A farther candidate is
    another object (without this cap any weak candidate on the next frame made a still
    box move). The candidate also lies ..."""
    blur_axis_tol_deg: float = 30.0
    """... in a direction within this angle of the box's long axis, ..."""
    blur_neighbour_min_conf: float = 0.2
    """... with at least this confidence. Elongated still objects (the digits of a shirt
    number in a close-up reached conf 0.88) do not move like that."""
    blur_edge_px: float = 2.0
    """A box closer than this to a frame edge is cut by it, so its shape says nothing about
    blur: it never takes the blur exception (the shirt digits above were cut by the
    bottom edge)."""
    nearby_box_heights: float = 3.0
    """A player is near a candidate when their box centres are at most this many of the
    player's box heights apart horizontally."""
    size_ratio_min: float = 0.05
    """Smallest ball diameter / median nearby box height (tracked balls: about 0.155)."""
    size_ratio_max: float = 0.3
    """Largest ball diameter / median nearby box height."""
    fallback_diameter_min_px: float = 4.0
    """Smallest ball diameter when no player is near, px at ``reference_lines`` lines."""
    fallback_diameter_max_px: float = 120.0
    """Largest ball diameter when no player is near, px at ``reference_lines`` lines."""
    reference_lines: int = 1080
    """Frame height (px) the fallback diameters refer to; scaled to the actual height."""
    pitch_half_length_m: float = 55.0
    """|x| limit of the off-pitch gate (the goal lines are at 52.5 m)."""
    pitch_half_width_m: float = 36.5
    """|y| limit of the off-pitch gate (the touchlines are at 34 m)."""
    ball_radius_m: float = 0.11
    """Ball radius: a ball on the grass has its centre this high (z = -0.11)."""
    near_player_margin_bh: float = 0.3
    """A player is at the ball when the ball centre is inside the player's box grown by
    this fraction of its height to the left, the right and below, not above: a ball in
    the hands is inside the box, while above the head the image shows what lies behind
    the player (a spare ball at the ad boards)."""
    static_radius_px: float = 3.0
    """A screen-fixed run stays within this distance of its first position."""
    static_min_s: float = 1.0
    """Screen-fixed and slow chains need members on at least this many seconds of frames
    (25); a world-fixed spot must span at least this long."""
    static_max_gap_s: float = 0.12
    """Largest gap inside a screen-fixed or slow chain (3 frames)."""
    static_camera_shift_px: float = 10.0
    """The camera "moved" when the scene point under the run moved at least this far ..."""
    static_min_moved_frames: int = 3
    """... on at least this many frames of the run."""
    static_min_common_tracks: int = 3
    """Tracks present on both frames needed for a median player-motion step."""
    parked_step_px: float = 3.0
    """A slow chain moves at most this many pixels (at reference_lines lines) per frame of
    gap between members: detector jitter (about 1 px) plus a slowly drifting camera (a
    resting spare ball drifts by a pixel or two per frame). Absolute pixels, not bh:
    a close-shot camera follows the ball, which then moves slowly in bh but several pixels
    per frame. A lone ball rolling slower than this (about 1.3 m/s in a wide shot) for a
    second is parked too."""
    world_static_radius_m: float = 1.5
    """Single-linkage radius (metres) grouping ground points into one world-fixed spot:
    with the per-frame calibration a spare ball near the far touchline scatters over more
    than a metre within one shot."""
    world_static_min_members: int = 6
    """A world-fixed spot needs candidates from at least this many frames, spanning at
    least static_min_s (far spare balls are detected on some frames only)."""
    off_field_margin_m: float = 1.0
    """A world-fixed spot is off the field when the median of its supporting ground points
    lies more than this far outside the field lines (|x| > 52.5 m + margin or
    |y| > 34 m + margin) ..."""
    off_field_point_margin_m: float = 0.5
    """Only ground points more than this outside the lines join such a spot (a ball in
    play has its centre at most 0.11 m beyond the line; single points of a spare ball
    near the far touchline scatter by more than a metre with the calibration)."""
    off_field_carry_px: float = 4.0
    """On frames without a usable calibration, a candidate within this distance (plus half
    its diameter) of an off_field_static candidate ..."""
    off_field_carry_frames: int = 2
    """... on one of this many neighbouring frames on each side is off_field_static too."""

    # --- Viterbi ------------------------------------------------------------------------
    conf_floor: float = 1e-3
    """Confidences are clipped to at least this before taking the log."""
    size_sigma_log: float = 0.35
    """Standard deviation of ln(diameter / expected diameter) in the size prior when the
    expected diameter comes from a pitch-usable calibration."""
    size_sigma_log_players: float = 0.7
    """The same when it comes from the nearby box heights: weak, because in close and
    low-angle shots the players near the ball in u stand at other depths."""
    ball_diameter_m: float = 0.22
    """Ball diameter for the calibrated size prior (FIFA size 5: 0.22 m)."""
    ball_box_scale: float = 1.2
    """Detector box diameter / projected ball diameter (measured 1.17-1.21 on calibrated
    frames of the development clip: the boxes include blur and margin)."""
    ball_to_box_height: float = 0.155
    """Detector box diameter / player box height: converts box heights to ball diameters
    and back. Measured 0.14-0.17 (median 0.155) on the development clip; the physical
    ratio 0.22 m / 1.8 m is 0.12, the detector boxes are larger and players bend."""
    motion_ref_bh_per_s: float = 17.5
    """Image speed at which the motion cost per frame reaches motion_weight (0.7 bh per
    frame at 25 fps, about 30 m/s). Motion is normalised by the ball's own scale,
    diameter / ball_to_box_height, the local box height at the ball's depth."""
    motion_weight: float = 0.5
    """Motion cost per frame at motion_ref_bh_per_s."""
    max_link_speed_bh_per_s: float = 30.0
    """Fastest image speed (1.2 bh per frame at 25 fps, about 54 m/s with camera motion) a
    motion link may imply; a faster jump is another object, reachable only through
    re-acquisition."""
    size_change_sigma_log: float = 0.15
    """Standard deviation of ln(diameter ratio) between linked detections one frame apart
    (times sqrt(gap)); a jump to an object at another depth costs."""
    max_link_gap_s: float = 0.32
    """Longest jump (8 frames) between two chosen candidates that keeps a motion link."""
    isolated_gap_s: float = 0.12
    """A chosen detection with no other one within this time (3 frames) on both sides is
    isolated: a first-order path happily uses a weak candidate as a stepping stone across
    an occlusion ..."""
    isolated_min_conf: float = 0.6
    """... so an isolated detection below this confidence is dropped after the Viterbi,
    and the links through it are cut."""
    missing_cost: float = 2.0
    """Cost of a frame without a chosen candidate: a candidate beats "missing" only from
    conf = exp(-2) = 0.135 up. On the development clip 94% of the tracked detections have
    conf >= 0.2, while the rejected clutter has a median conf of 0.15; a skipped weak
    detection inside a chain is interpolated instead."""
    reacquire_cost: float = 3.0
    """Cost of picking the ball up with no motion link (start of a shot or after a loss):
    a short chain of weak candidates must earn this before it is believed."""
    # --- interpolation ------------------------------------------------------------------
    max_interp_gap_s: float = 0.24
    """Longest gap (6 frames) filled by linear interpolation between linked detections."""
    max_interp_speed_bh_per_s: float = 20.0
    """A linked gap implying a faster image speed (about 36 m/s) stays missing: a straight
    line through it would be a guess."""

    # --- events -------------------------------------------------------------------------
    event_window_s: float = 0.12
    """Detected samples within this time (3 frames) on each side of a vertex estimate the
    incoming and outgoing velocity (least squares through the vertex sample); events
    closer than this are merged, keeping the larger magnitude."""
    event_angle_deg: float = 35.0
    """A direction change larger than this is an event when both speeds reach the floor."""
    event_speed_ratio: float = 2.0
    """A speed ratio larger than this is an event when the faster speed reaches the floor
    and the velocity changes by at least event_min_change_bh_s."""
    event_speed_floor_bh_s: float = 1.5
    """Speed floor, bh/s (about 2.7 m/s)."""
    event_min_change_bh_s: float = 2.0
    """Smallest |v_after - v_before| of a speed-ratio event, bh/s (about 3.6 m/s); keeps
    smooth camera accelerations next to a slow ball out."""
    event_max_speed_bh_s: float = 30.0
    """A window implying a faster image speed spans two different objects: no event."""
    event_at_foot_bh: float = 0.6
    """``at_foot`` is True when the nearest player's foot point is at most this many of
    their box heights from the ball centre (contacts measured 0.06-0.44 on the
    development clip, bounces and flight apexes 1.3 and more)."""
    event_max_fit_rms_bh: float = 0.08
    """A window whose samples miss a straight line by more than this RMS (in bh, at least
    event_min_fit_rms_px) is not one ball moving steadily: no event."""
    event_min_fit_rms_px: float = 2.0
    """Floor of the fit tolerance in pixels (detector jitter)."""
    player_roles: tuple[str, ...] = ("player", "goalkeeper")
    """Roles that can be "at the ball" and that are reported as the nearest player."""

    def __post_init__(self) -> None:
        problems: list[str] = []
        non_negative = {"motion_weight", "reacquire_cost"}
        integers = {
            "reference_lines",
            "static_min_moved_frames",
            "static_min_common_tracks",
            "world_static_min_members",
            "off_field_carry_frames",
        }
        good: set[str] = set()
        for spec in fields(self):
            name = spec.name
            value = getattr(self, name)
            if name == "player_roles":
                if not isinstance(value, tuple) or not all(isinstance(r, str) for r in value):
                    problems.append(f"{name}: expected a tuple of strings, got {value!r}")
                continue
            if name in integers:
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    problems.append(f"{name}: expected an integer >= 1, got {value!r}")
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                problems.append(f"{name}: expected a number, got {value!r}")
            elif not math.isfinite(value):
                problems.append(f"{name}: must be finite, got {value!r}")
            elif name in non_negative and value < 0:
                problems.append(f"{name}: must be >= 0, got {value!r}")
            elif name not in non_negative and value <= 0:
                problems.append(f"{name}: must be > 0, got {value!r}")
            else:
                good.add(name)
        for low, high in (
            ("aspect_min", "aspect_max"),
            ("blur_min_step", "blur_max_step"),
            ("size_ratio_min", "size_ratio_max"),
            ("fallback_diameter_min_px", "fallback_diameter_max_px"),
        ):
            if {low, high} <= good and getattr(self, low) >= getattr(self, high):
                problems.append(f"{low} ({getattr(self, low)}) must be < {high}")
        if {"aspect_max", "blur_aspect_max"} <= good and self.blur_aspect_max < self.aspect_max:
            problems.append(
                f"blur_aspect_max ({self.blur_aspect_max}) must be >= aspect_max"
                f" ({self.aspect_max})"
            )
        if "blur_min_conf" in good and self.blur_min_conf > 1.0:
            problems.append(f"blur_min_conf must be <= 1, got {self.blur_min_conf}")
        if "event_angle_deg" in good and self.event_angle_deg >= 180.0:
            problems.append(f"event_angle_deg must be < 180, got {self.event_angle_deg}")
        if "event_speed_ratio" in good and self.event_speed_ratio <= 1.0:
            problems.append(f"event_speed_ratio must be > 1, got {self.event_speed_ratio}")
        if "conf_floor" in good and self.conf_floor >= 1.0:
            problems.append(f"conf_floor must be < 1, got {self.conf_floor}")
        if problems:
            raise BallInputError("BallParams", problems)


DEFAULT_PARAMS = BallParams()
"""Default thresholds; ball_events with them reports strikes: kicks, passes, shots, traps."""

CONTACT_PARAMS = BallParams(
    event_speed_floor_bh_s=0.3,
    event_min_change_bh_s=0.5,
    event_angle_deg=45.0,
)
"""Lower event thresholds for ball contact frames inside a highlight (dribbling touches,
duels). In a close shot the camera follows the ball, so image speeds there are 0.1-1 bh/s
and a touch changes them by 0.5-1.1 bh/s (velocity noise about 0.05 bh/s). On the
development clip these thresholds gave 55 events, 44 of them at a foot (at_foot); filter
with at_foot. Empirical, like every threshold here."""


# ----------------------------------------------------------------------------------------
# Candidates file
# ----------------------------------------------------------------------------------------


def load_candidates(
    path: str | Path, *, allow_incomplete_tail: bool = False
) -> dict[int, np.ndarray]:
    """Read ``ball_raw.jsonl`` into {frame: float64 array of shape (k, 5)}, sorted by frame.

    Each row of an array is [x1, y1, x2, y2, conf] (detector pixels, see the module
    docstring); k may be 0. Blank lines are ignored. A last line without a newline is a
    line still being written: by default that is an error, with
    ``allow_incomplete_tail=True`` it is dropped (for reading a file that is growing).

    Raises BallInputError listing every problem: unreadable or non-UTF-8 file, invalid
    JSON, a line that is not {"frame": int >= 0, "cands": [...]}, a duplicated frame, a
    candidate that is not 5 finite numbers, x2 <= x1, y2 <= y1, or conf outside [0, 1].
    """
    path = Path(path)
    where = f"load_candidates({path.name})"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BallInputError(where, [f"cannot read the file: {exc}"]) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BallInputError(where, [f"not UTF-8: {exc}"]) from exc
    lines = text.split("\n")
    tail = lines.pop()
    problems: list[str] = []
    if tail.strip() and not allow_incomplete_tail:
        problems.append(
            f"line {len(lines) + 1}: no newline at the end of the file (a line still being "
            "written?); pass allow_incomplete_tail=True to drop it"
        )
    result: dict[int, np.ndarray] = {}
    seen: set[int] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(record, dict) or "frame" not in record or "cands" not in record:
            problems.append(f'line {number}: expected an object with "frame" and "cands"')
            continue
        frame = record["frame"]
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            problems.append(f"line {number}: frame must be an integer >= 0, got {frame!r}")
            continue
        if frame in seen:
            problems.append(f"line {number}: frame {frame} appears more than once")
            continue
        seen.add(frame)
        array = _candidate_array(record["cands"], f"line {number} (frame {frame})", problems)
        if array is not None:
            result[frame] = array
    if problems:
        raise BallInputError(where, problems)
    return dict(sorted(result.items()))


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(
        value, bool | np.bool_
    )


def _candidate_array(value: Any, label: str, problems: list[str]) -> np.ndarray | None:
    """(k, 5) float64 array of valid candidates, or None after recording every problem."""
    if isinstance(value, np.ndarray):
        items: list[Any] = list(value) if value.ndim >= 1 else [value]
    elif isinstance(value, list | tuple):
        items = list(value)
    else:
        problems.append(f"{label}: cands must be a list, got {type(value).__name__}")
        return None
    rows: list[list[float]] = []
    found = len(problems)
    for index, item in enumerate(items):
        numbers = list(item) if isinstance(item, list | tuple | np.ndarray) else None
        if numbers is None or len(numbers) != 5 or not all(_is_number(x) for x in numbers):
            problems.append(f"{label}: cands[{index}] must be 5 numbers, got {item!r}")
            continue
        x1, y1, x2, y2, conf = (float(x) for x in numbers)
        if not all(math.isfinite(x) for x in (x1, y1, x2, y2, conf)):
            problems.append(f"{label}: cands[{index}] has non-finite values {numbers!r}")
            continue
        if x2 <= x1 or y2 <= y1:
            problems.append(f"{label}: cands[{index}] needs x2 > x1 and y2 > y1, got {numbers!r}")
            continue
        if not 0.0 <= conf <= 1.0:
            problems.append(f"{label}: cands[{index}] conf {conf!r} is outside [0, 1]")
            continue
        rows.append([x1, y1, x2, y2, conf])
    if len(problems) > found:
        return None
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


# ----------------------------------------------------------------------------------------
# Input checks
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shot:
    shot_id: int
    start: int
    end: int  # inclusive


@dataclass(frozen=True)
class _FramePlayers:
    """Person boxes of one frame (all roles), arrays of equal length."""

    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    cu: np.ndarray  # box centre u, px
    cv: np.ndarray  # box centre v, px
    h: np.ndarray  # box height, px
    foot_u: np.ndarray
    foot_v: np.ndarray
    track_id: np.ndarray  # int64
    team: tuple[str | None, ...]
    role: tuple[str, ...]


@dataclass
class _Context:
    fps: float
    n_frames: int
    width: int
    height: int
    params: BallParams
    shots: list[_Shot]
    shot_of_frame: np.ndarray  # (n_frames,) index into shots
    players: dict[int, _FramePlayers]
    cameras: dict[int, Camera]
    candidates: dict[int, np.ndarray]
    _motion: np.ndarray | None = field(default=None, repr=False)

    def frames(self, seconds: float) -> int:
        """round(seconds * fps), at least 1."""
        return max(1, int(round(seconds * self.fps)))

    def player_motion(self) -> np.ndarray:
        """(n_frames, 2) median image motion of the tracked players from frame t - 1 to t.

        NaN where t is the first frame of a shot or fewer than static_min_common_tracks
        tracks (track_id >= 0) are on both frames. Computed once.
        """
        if self._motion is None:
            self._motion = _median_player_motion(
                self.players,
                self.shot_of_frame,
                self.n_frames,
                self.params.static_min_common_tracks,
            )
        return self._motion


def _is_int(value: Any) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_)


def _check_n_frames(n_frames: Any, problems: list[str]) -> int | None:
    if not _is_int(n_frames) or int(n_frames) <= 0:
        problems.append(f"n_frames must be an integer > 0, got {n_frames!r}")
        return None
    return int(n_frames)


def _check_fps(fps: Any, problems: list[str]) -> float | None:
    """fps as a float: a positive finite number, a Fraction or a rate string ("25/1")."""
    if isinstance(fps, str | Fraction):
        try:
            return float(parse_rate(fps))
        except ValueError as exc:
            problems.append(f"fps: {exc}")
            return None
    if not _is_number(fps):
        problems.append(f"fps must be a positive number or a rate string, got {fps!r}")
        return None
    value = float(fps)
    if not math.isfinite(value) or value <= 0.0:
        problems.append(f"fps must be finite and > 0, got {fps!r}")
        return None
    return value


def _missing_columns(frame: Any, name: str, required: Sequence[str], problems: list[str]) -> bool:
    """True (after recording problems) when ``frame`` is not a DataFrame with ``required``."""
    if not isinstance(frame, pd.DataFrame):
        problems.append(f"{name}: expected a pandas DataFrame, got {type(frame).__name__}")
        return True
    missing = [column for column in required if column not in frame.columns]
    if missing:
        problems.append(f"{name}: missing column(s) {', '.join(missing)}")
        return True
    return False


def _float_column(
    frame: pd.DataFrame, table: str, column: str, problems: list[str]
) -> np.ndarray | None:
    """float64 values of a numeric column (missing values become NaN), or None."""
    series = frame[column]
    if pd.api.types.is_bool_dtype(series.dtype):
        problems.append(f"{table}.{column}: expected numbers, got booleans")
        return None
    try:
        numeric = pd.to_numeric(series, errors="raise")
        return numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        problems.append(f"{table}.{column}: expected numbers ({exc})")
        return None


def _integer_column(
    frame: pd.DataFrame, table: str, column: str, problems: list[str]
) -> np.ndarray | None:
    """int64 values of a column holding integers (integral floats accepted), or None."""
    found = len(problems)
    values = _float_column(frame, table, column, problems)
    if values is None:
        return None
    finite = np.isfinite(values)
    if not finite.all():
        problems.append(f"{table}.{column}: {int((~finite).sum())} missing or non-finite value(s)")
    elif (np.floor(values) != values).any():
        problems.append(f"{table}.{column}: expected integers, got fractional values")
    if len(problems) > found:
        return None
    return values.astype(np.int64)


def _text(value: Any) -> str | None:
    """A string cell as str, or None for a missing value."""
    return value if isinstance(value, str) else None


def _check_shots(shots: Any, n_frames: int | None, problems: list[str]) -> list[_Shot] | None:
    """Shots sorted by start frame; they must partition [0, n_frames) without gaps.

    With n_frames None the partition must start at 0 and its end defines n_frames.
    """
    if _missing_columns(shots, "shots", SHOTS_COLUMNS_READ, problems):
        return None
    if len(shots) == 0:
        problems.append("shots: empty; at least one shot is needed")
        return None
    found = len(problems)
    ids = _integer_column(shots, "shots", "shot_id", problems)
    starts = _integer_column(shots, "shots", "start_frame", problems)
    ends = _integer_column(shots, "shots", "end_frame", problems)
    if ids is None or starts is None or ends is None or len(problems) > found:
        return None
    if len(np.unique(ids)) != len(ids):
        problems.append("shots.shot_id: values must be unique")
    order = np.argsort(starts, kind="stable")
    expected_start = 0
    for position in order.tolist():
        start, end = int(starts[position]), int(ends[position])
        if end < start:
            problems.append(f"shots: shot {ids[position]} ends ({end}) before it starts ({start})")
        if start != expected_start:
            problems.append(
                f"shots: shot {ids[position]} starts at {start}, expected {expected_start} "
                "(shots must cover every frame once, without gaps or overlaps)"
            )
        expected_start = end + 1
    if n_frames is not None and expected_start != n_frames:
        problems.append(
            f"shots: the last shot ends at {expected_start - 1}, expected {n_frames - 1}"
        )
    if len(problems) > found:
        return None
    return [_Shot(int(ids[i]), int(starts[i]), int(ends[i])) for i in order.tolist()]


def _check_tracks(
    tracks: Any, n_frames: int | None, problems: list[str]
) -> dict[int, _FramePlayers] | None:
    if _missing_columns(tracks, "tracks", TRACKS_COLUMNS_READ, problems):
        return None
    if len(tracks) == 0:
        return {}
    found = len(problems)
    frames = _integer_column(tracks, "tracks", "frame", problems)
    track_ids = _integer_column(tracks, "tracks", "track_id", problems)
    values = {
        c: _float_column(tracks, "tracks", c, problems)
        for c in ("x1", "y1", "x2", "y2", "foot_u", "foot_v")
    }
    if len(problems) > found or frames is None or track_ids is None:
        return None
    x1, y1, x2, y2 = values["x1"], values["y1"], values["x2"], values["y2"]
    foot_u, foot_v = values["foot_u"], values["foot_v"]
    if x1 is None or y1 is None or x2 is None or y2 is None or foot_u is None or foot_v is None:
        return None
    if n_frames is not None:
        outside = (frames < 0) | (frames >= n_frames)
        if outside.any():
            problems.append(
                f"tracks.frame: {int(outside.sum())} row(s) outside [0, {n_frames - 1}]"
            )
    finite = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(x2) & np.isfinite(y2)
    if not finite.all():
        problems.append(f"tracks: {int((~finite).sum())} row(s) with non-finite box coordinates")
    degenerate = finite & ((x2 <= x1) | (y2 <= y1))
    if degenerate.any():
        problems.append(f"tracks: {int(degenerate.sum())} row(s) with x2 <= x1 or y2 <= y1")
    if len(problems) > found:
        return None
    roles = [r if isinstance(r, str) else "unknown" for r in tracks["role"].tolist()]
    teams = [_text(t) for t in tracks["team"].tolist()]
    order = np.argsort(frames, kind="stable")
    boundaries = np.flatnonzero(np.diff(frames[order])) + 1
    players: dict[int, _FramePlayers] = {}
    for rows in np.split(order, boundaries):
        if rows.size == 0:
            continue
        players[int(frames[rows[0]])] = _FramePlayers(
            x1=x1[rows],
            y1=y1[rows],
            x2=x2[rows],
            y2=y2[rows],
            cu=(x1[rows] + x2[rows]) / 2.0,
            cv=(y1[rows] + y2[rows]) / 2.0,
            h=y2[rows] - y1[rows],
            foot_u=foot_u[rows],
            foot_v=foot_v[rows],
            track_id=track_ids[rows],
            team=tuple(teams[i] for i in rows.tolist()),
            role=tuple(roles[i] for i in rows.tolist()),
        )
    return players


def _check_frame_size(frame_size: Any, calib: Any, problems: list[str]) -> tuple[int, int] | None:
    """(width, height) of the video frames: given, or the single size found in ``calib``."""
    if frame_size is not None:
        if (
            not isinstance(frame_size, tuple | list)
            or len(frame_size) != 2
            or not all(_is_int(x) and int(x) > 0 for x in frame_size)
        ):
            problems.append(
                f"frame_size must be (width, height) with positive integers, got {frame_size!r}"
            )
            return None
        return int(frame_size[0]), int(frame_size[1])
    if (
        isinstance(calib, pd.DataFrame)
        and {"width", "height"} <= set(calib.columns)
        and len(calib) > 0
    ):
        widths = pd.to_numeric(calib["width"], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        heights = pd.to_numeric(calib["height"], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        known = np.isfinite(widths) & np.isfinite(heights)
        sizes = sorted(
            {(int(w), int(h)) for w, h in zip(widths[known], heights[known], strict=True)}
        )
        if len(sizes) == 1 and sizes[0][0] > 0 and sizes[0][1] > 0:
            return sizes[0]
        if len(sizes) > 1:
            problems.append(
                f"calib: several frame sizes {sizes}; pass frame_size=(width, height) of the video"
            )
            return None
    problems.append(
        "frame_size not given and calib has no width/height; pass frame_size=(width, height)"
    )
    return None


def _matrix(value: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    """A cell holding a list / array of numbers as a float64 array of ``shape``, or None."""
    if value is None or isinstance(value, str | bytes):
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.size != int(np.prod(shape)):
        return None
    return array.reshape(shape)


def _check_calib(
    calib: Any, n_frames: int | None, frame_size: tuple[int, int] | None, problems: list[str]
) -> dict[int, Camera] | None:
    """Cameras of the pitch-usable frames, for frames of size frame_size."""
    frame_column = "frame_index"
    if isinstance(calib, pd.DataFrame) and "frame_index" not in calib.columns:
        frame_column = "frame"  # rows as in calib.jsonl
    required = (frame_column, *CALIB_COLUMNS_READ[1:])
    if _missing_columns(calib, "calib", required, problems):
        return None
    if len(calib) == 0:
        return {}
    found = len(problems)
    frames = _integer_column(calib, "calib", frame_column, problems)
    if frames is None:
        return None
    if len(np.unique(frames)) != len(frames):
        problems.append(f"calib.{frame_column}: values must be unique (one row per frame)")
    if n_frames is not None:
        outside = (frames < 0) | (frames >= n_frames)
        if outside.any():
            problems.append(
                f"calib.{frame_column}: {int(outside.sum())} row(s) outside [0, {n_frames - 1}]"
            )
    usable_values = calib["pitch_usable"].tolist()
    if not all(isinstance(x, bool | np.bool_) for x in usable_values):
        problems.append("calib.pitch_usable: expected booleans (no missing values)")
        return None
    if len(problems) > found or frame_size is None:
        return None
    has_size = {"width", "height"} <= set(calib.columns)
    cameras: dict[int, Camera] = {}
    for row in np.flatnonzero(np.asarray(usable_values, dtype=bool)).tolist():
        frame = int(frames[row])
        K = _matrix(calib["K"].iloc[row], (3, 3))
        R = _matrix(calib["R"].iloc[row], (3, 3))
        C = _matrix(calib["C"].iloc[row], (3,))
        bad = [n for n, a in (("K", K), ("R", R), ("C", C)) if a is None]
        if bad:
            problems.append(
                f"calib frame {frame}: pitch_usable but {', '.join(bad)} is not 9 / 9 / 3 numbers"
            )
            continue
        width, height = frame_size
        if has_size:
            w, h = calib["width"].iloc[row], calib["height"].iloc[row]
            if _is_number(w) and math.isfinite(float(w)):
                width = int(w)
            if _is_number(h) and math.isfinite(float(h)):
                height = int(h)
        try:
            camera = Camera(K=K, R=R, C=C, width=width, height=height)
            if (width, height) != frame_size:
                camera = camera.resized(*frame_size)
        except GeometryInputError as exc:
            problems.extend(f"calib frame {frame}: {p}" for p in exc.problems)
            continue
        cameras[frame] = camera
    if len(problems) > found:
        return None
    return cameras


def _check_candidates(
    candidates: Any, n_frames: int | None, problems: list[str]
) -> dict[int, np.ndarray] | None:
    if not isinstance(candidates, Mapping):
        problems.append(
            f"candidates: expected a mapping frame -> (k, 5) array (load_candidates), "
            f"got {type(candidates).__name__}"
        )
        return None
    found = len(problems)
    result: dict[int, np.ndarray] = {}
    for frame, value in candidates.items():
        if not _is_int(frame) or int(frame) < 0:
            problems.append(f"candidates: frame keys must be integers >= 0, got {frame!r}")
            continue
        if n_frames is not None and int(frame) >= n_frames:
            problems.append(f"candidates: frame {frame} is beyond the last frame {n_frames - 1}")
            continue
        if isinstance(value, np.ndarray) and value.size == 0:
            result[int(frame)] = np.zeros((0, 5))
            continue
        array = _candidate_array(value, f"candidates[{frame}]", problems)
        if array is not None:
            result[int(frame)] = array
    if len(problems) > found:
        return None
    return dict(sorted(result.items()))


def _prepare(
    where: str,
    candidates: Any,
    tracks: Any,
    calib: Any,
    shots: Any,
    fps: Any,
    n_frames: Any,
    frame_size: Any,
    params: Any,
) -> _Context:
    problems: list[str] = []
    if not isinstance(params, BallParams):
        problems.append(f"params: expected BallParams, got {type(params).__name__}")
    rate = _check_fps(fps, problems)
    count = _check_n_frames(n_frames, problems)
    size = _check_frame_size(frame_size, calib, problems)
    shot_list = _check_shots(shots, count, problems)
    players = _check_tracks(tracks, count, problems)
    cameras = _check_calib(calib, count, size, problems)
    cands = _check_candidates(candidates, count, problems)
    if (
        problems
        or rate is None
        or count is None
        or size is None
        or shot_list is None
        or players is None
        or cameras is None
        or cands is None
    ):
        raise BallInputError(where, problems)
    shot_of_frame = np.empty(count, dtype=np.int64)
    for index, shot in enumerate(shot_list):
        shot_of_frame[shot.start : shot.end + 1] = index
    return _Context(
        fps=rate,
        n_frames=count,
        width=size[0],
        height=size[1],
        params=params,
        shots=shot_list,
        shot_of_frame=shot_of_frame,
        players=players,
        cameras=cameras,
        candidates=cands,
    )


# ----------------------------------------------------------------------------------------
# Scene helpers
# ----------------------------------------------------------------------------------------


def _median_player_motion(
    players: Mapping[int, _FramePlayers],
    shot_of_frame: np.ndarray,
    n_frames: int,
    min_common: int,
) -> np.ndarray:
    """(n_frames, 2) per-axis median box-centre motion of tracks seen on t - 1 and t."""
    motion = np.full((n_frames, 2), np.nan)
    for t in range(1, n_frames):
        if shot_of_frame[t] != shot_of_frame[t - 1]:
            continue
        now, before = players.get(t), players.get(t - 1)
        if now is None or before is None:
            continue
        keep_now = now.track_id >= 0
        keep_before = before.track_id >= 0
        ids_now, idx_now = np.unique(now.track_id[keep_now], return_index=True)
        ids_before, idx_before = np.unique(before.track_id[keep_before], return_index=True)
        _common, a, b = np.intersect1d(ids_now, ids_before, return_indices=True)
        if a.size < min_common:
            continue
        rows_now = np.flatnonzero(keep_now)[idx_now[a]]
        rows_before = np.flatnonzero(keep_before)[idx_before[b]]
        motion[t, 0] = float(np.median(now.cu[rows_now] - before.cu[rows_before]))
        motion[t, 1] = float(np.median(now.cv[rows_now] - before.cv[rows_before]))
    return motion


def _scene_shift(reference: Camera, camera: Camera, uv: np.ndarray) -> float:
    """Pixels the scene point under ``uv`` in the reference frame moves to in ``camera``.

    The point is taken on the grass (z = 0), which both calibrations were fitted to; when
    the ray misses the grass, the rotation-only (infinite) homography is used instead.
    """
    point, valid = pixel_to_ground(reference, uv, 0.0)
    if bool(valid):
        mapped, _depth, ok = project(camera, point)
        if bool(ok):
            return float(np.hypot(*(mapped - uv)))
    homography = camera.K @ camera.R @ reference.R.T @ np.linalg.inv(reference.K)
    p = homography @ np.array([uv[0], uv[1], 1.0])
    if p[2] <= 0.0:
        return math.inf
    return float(np.hypot(p[0] / p[2] - uv[0], p[1] / p[2] - uv[1]))


def _nearby(players: _FramePlayers | None, u: float, box_heights: float) -> np.ndarray:
    """Box heights of the players whose box centre is within box_heights * h of u."""
    if players is None or players.h.size == 0:
        return np.zeros(0)
    mask = np.abs(players.cu - u) <= box_heights * players.h
    return players.h[mask]


def _at_ball(
    players: _FramePlayers | None, u: np.ndarray, v: np.ndarray, p: BallParams
) -> np.ndarray:
    """True per (u, v) when a player of p.player_roles is at the ball (grown box)."""
    result = np.zeros(u.shape, dtype=bool)
    if players is None or players.h.size == 0:
        return result
    roles = set(p.player_roles)
    keep = np.array([r in roles for r in players.role], dtype=bool)
    if not keep.any():
        return result
    margin = p.near_player_margin_bh * players.h[keep]
    left = (players.x1[keep] - margin)[None, :]
    right = (players.x2[keep] + margin)[None, :]
    top = players.y1[keep][None, :]  # no margin above: there stands what is behind him
    bottom = (players.y2[keep] + margin)[None, :]
    inside = (
        (u[:, None] >= left) & (u[:, None] <= right) & (v[:, None] >= top) & (v[:, None] <= bottom)
    )
    return inside.any(axis=1)


# ----------------------------------------------------------------------------------------
# Gating
# ----------------------------------------------------------------------------------------


def gate_candidates(
    candidates: Mapping[int, Any],
    tracks: pd.DataFrame,
    calib: pd.DataFrame,
    shots: pd.DataFrame,
    fps: float | str | Fraction,
    n_frames: int,
    *,
    frame_size: tuple[int, int] | None = None,
    params: BallParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Test every raw candidate against the gates; one row per candidate (GATE_COLUMNS).

    Arguments are those of track_ball; the tests are described in the module docstring.
    Raises BallInputError listing every input problem.
    """
    ctx = _prepare(
        "gate_candidates", candidates, tracks, calib, shots, fps, n_frames, frame_size, params
    )
    return _gate(ctx)


def _blur_moves(ctx: _Context, frame: int, u: float, v: float, width: float, height: float) -> bool:
    """Whether an elongated box moves like a blurred ball (see BallParams.blur_min_step)."""
    p = ctx.params
    long_side = max(width, height)
    cos_tol = math.cos(math.radians(p.blur_axis_tol_deg))
    for neighbour in (frame - 1, frame + 1):
        if not 0 <= neighbour < ctx.n_frames:
            continue
        if ctx.shot_of_frame[neighbour] != ctx.shot_of_frame[frame]:
            continue  # across a cut the image shows another view, not motion
        other = ctx.candidates.get(neighbour)
        if other is None or other.shape[0] == 0:
            continue
        du = (other[:, 0] + other[:, 2]) / 2.0 - u
        dv = (other[:, 1] + other[:, 3]) / 2.0 - v
        dist = np.hypot(du, dv)
        along = np.abs(du) if width >= height else np.abs(dv)
        moving = (
            (other[:, 4] >= p.blur_neighbour_min_conf)
            & (dist >= p.blur_min_step * long_side)
            & (dist <= p.blur_max_step * long_side)
            & (along >= cos_tol * dist)
        )
        if moving.any():
            return True
    return False


def _gate(ctx: _Context) -> pd.DataFrame:
    p = ctx.params
    scale_lines = p.reference_lines / ctx.height
    parts: dict[str, list[np.ndarray]] = {}

    def add(name: str, array: np.ndarray) -> None:
        parts.setdefault(name, []).append(array)

    for frame, cands in ctx.candidates.items():
        k = cands.shape[0]
        if k == 0:
            continue
        shot = ctx.shots[int(ctx.shot_of_frame[frame])]
        x1, y1, x2, y2, conf = (cands[:, i] for i in range(5))
        width, height = x2 - x1, y2 - y1
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        aspect = width / height
        round_box = (aspect >= p.aspect_min) & (aspect <= p.aspect_max)
        elongation = np.maximum(aspect, 1.0 / aspect)
        inside = (
            (x1 >= p.blur_edge_px)
            & (y1 >= p.blur_edge_px)
            & (x2 <= ctx.width - p.blur_edge_px)
            & (y2 <= ctx.height - p.blur_edge_px)
        )
        blurred = (
            ~round_box & inside & (conf >= p.blur_min_conf) & (elongation <= p.blur_aspect_max)
        )
        for i in np.flatnonzero(blurred):
            blurred[i] = _blur_moves(
                ctx, frame, float(u[i]), float(v[i]), float(width[i]), float(height[i])
            )
        diameter = np.where(blurred, np.minimum(width, height), (width + height) / 2.0)
        players = ctx.players.get(frame)
        n_nearby = np.zeros(k, dtype=np.int32)
        ref_h = np.full(k, np.nan)
        for i in range(k):
            heights = _nearby(players, float(u[i]), p.nearby_box_heights)
            n_nearby[i] = heights.size
            if heights.size:
                ref_h[i] = float(np.median(heights))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = diameter / ref_h
        aspect_ok = round_box | blurred
        scaled = diameter * scale_lines
        size_ok = np.where(
            n_nearby > 0,
            (ratio >= p.size_ratio_min) & (ratio <= p.size_ratio_max),
            (scaled >= p.fallback_diameter_min_px) & (scaled <= p.fallback_diameter_max_px),
        )
        camera = ctx.cameras.get(frame)
        pitch_x = np.full(k, np.nan)
        pitch_y = np.full(k, np.nan)
        calibrated = np.full(k, np.nan)
        pitch_ok = np.ones(k, dtype=bool)
        if camera is not None:
            uv = np.stack([u, v], axis=-1)
            points, valid = pixel_to_ground(camera, uv, -p.ball_radius_m)
            pitch_x = np.where(valid, points[:, 0], np.nan)
            pitch_y = np.where(valid, points[:, 1], np.nan)
            pitch_ok = (
                valid
                & (np.abs(np.nan_to_num(pitch_x, nan=np.inf)) <= p.pitch_half_length_m)
                & (np.abs(np.nan_to_num(pitch_y, nan=np.inf)) <= p.pitch_half_width_m)
            )
            depth = world_to_camera(camera, np.where(valid[:, None], points, 0.0))[:, 2]
            focal = 0.5 * (camera.K[0, 0] + camera.K[1, 1])
            size_m = p.ball_diameter_m * p.ball_box_scale
            with np.errstate(invalid="ignore", divide="ignore"):
                calibrated = np.where(valid & (depth > 0.0), focal * size_m / depth, np.nan)
        from_camera = np.isfinite(calibrated)
        expected = np.where(from_camera, calibrated, p.ball_to_box_height * ref_h)
        sigma = np.where(from_camera, p.size_sigma_log, p.size_sigma_log_players)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.log(diameter / expected) / sigma
        size_prior = np.where(np.isfinite(z), -0.5 * z * z, 0.0)
        add("frame", np.full(k, frame, dtype=np.int64))
        add("cand_index", np.arange(k, dtype=np.int32))
        add("shot_id", np.full(k, shot.shot_id, dtype=np.int64))
        for name, array in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2), ("conf", conf)):
            add(name, array)
        add("u", u)
        add("v", v)
        add("diameter_px", diameter)
        add("aspect", aspect)
        add("n_nearby", n_nearby)
        add("ref_box_h", ref_h)
        add("size_ratio", ratio)
        add("pitch_usable", np.full(k, camera is not None))
        add("pitch_x", pitch_x)
        add("pitch_y", pitch_y)
        add("expected_diameter_px", expected)
        add("size_prior", size_prior)
        add("scale_px", diameter / p.ball_to_box_height)
        add("near_player", _at_ball(players, u, v, p))
        add("aspect_ok", aspect_ok)
        add("size_ok", size_ok)
        add("pitch_ok", pitch_ok)
    integer_columns = {"frame": np.int64, "cand_index": np.int32, "shot_id": np.int64}
    bool_columns = {"pitch_usable", "near_player", "aspect_ok", "size_ok", "pitch_ok"}
    data: dict[str, Any] = {}
    for name in GATE_COLUMNS:
        if name in ("static_run", "static_ok", "field_static_ok", "parked_ok", "reason"):
            continue
        if name == "n_nearby":
            dtype: Any = np.int32
        elif name in integer_columns:
            dtype = integer_columns[name]
        elif name in bool_columns:
            dtype = bool
        else:
            dtype = np.float64
        chunks = parts.get(name, [])
        data[name] = np.concatenate(chunks).astype(dtype) if chunks else np.zeros(0, dtype=dtype)
    run, static, parked = _screen_fixed(data, ctx)
    data["static_run"] = run
    data["static_ok"] = ~static
    data["field_static_ok"] = ~_world_fixed_off_field(data, ctx)
    data["parked_ok"] = ~parked
    n_rows = data["frame"].size
    reason = np.full(n_rows, "ok", dtype=object)
    for name, test in (
        ("parked", data["parked_ok"]),
        ("off_field_static", data["field_static_ok"]),
        ("static", data["static_ok"]),
        ("off_pitch", data["pitch_ok"]),
        ("size", data["size_ok"]),
        ("aspect", data["aspect_ok"]),
    ):
        reason[~test] = name  # later assignments win: aspect is the first test
    data["reason"] = reason.tolist()
    return pd.DataFrame({name: data[name] for name in GATE_COLUMNS})


@dataclass
class _Run:
    anchor: np.ndarray
    last: int
    members: list[int]


def _chain_runs(
    frames: np.ndarray,
    points: np.ndarray,
    radius: np.ndarray | float,
    max_gap: int,
    rows: np.ndarray,
    *,
    follow: bool = False,
) -> list[list[int]]:
    """Chains of candidate rows that stay at one image spot.

    ``rows`` (sorted by frame) are the rows considered. Anchored (``follow`` False): a row
    joins the open chain whose FIRST point is nearest and within ``radius``. Following:
    a row joins the chain whose LAST point is nearest and within ``radius`` times the
    frame gap (the spot may drift slowly). ``radius`` is a number or one value per row
    (the joining row's). One row per chain and frame, nearest pairs first; a chain closes
    after more than ``max_gap`` frames without a member. Returns the member rows of every
    chain, singletons included.
    """
    runs: list[_Run] = []
    open_runs: list[int] = []
    if rows.size == 0:
        return []
    limits = np.broadcast_to(np.asarray(radius, dtype=np.float64), frames.shape)
    row_frames = frames[rows]
    starts = np.flatnonzero(np.r_[True, row_frames[1:] != row_frames[:-1]])
    ends = np.r_[starts[1:], rows.size]
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        frame = int(row_frames[start])
        here = rows[start:end].tolist()
        open_runs = [r for r in open_runs if frame - runs[r].last <= max_gap + 1]
        pairs: list[tuple[float, int, int]] = []
        for r in open_runs:
            scale = frame - runs[r].last if follow else 1
            for row in here:
                distance = float(np.hypot(*(points[row] - runs[r].anchor)))
                if distance <= limits[row] * scale:
                    pairs.append((distance, r, row))
        pairs.sort()
        used_runs: set[int] = set()
        used_rows: set[int] = set()
        for _distance, r, row in pairs:
            if r in used_runs or row in used_rows:
                continue
            used_runs.add(r)
            used_rows.add(row)
            runs[r].members.append(row)
            runs[r].last = frame
            if follow:
                runs[r].anchor = points[row].copy()
        for row in here:
            if row not in used_rows:
                runs.append(_Run(points[row].copy(), frame, [row]))
                open_runs.append(len(runs) - 1)
    return [run.members for run in runs]


def _screen_fixed(data: Mapping[str, np.ndarray], ctx: _Context) -> tuple[np.ndarray, ...]:
    """(screen-fixed run length, static, parked) per gate row; see the module docstring.

    static: anchored chains (within static_radius_px of their first point) on at least
    static_min_s of frames during which the camera moved. parked: slow chains (each member
    within parked_step_px, scaled to the frame height, per frame of gap of the previous
    one) on at least static_min_s of frames, on the frames where no player is at the ball.
    """
    p = ctx.params
    frames = data["frame"]
    n = frames.size
    points = np.stack([data["u"], data["v"]], axis=-1) if n else np.zeros((0, 2))
    near = data["near_player"]
    length = np.ones(n, dtype=np.int32)
    static = np.zeros(n, dtype=bool)
    parked = np.zeros(n, dtype=bool)
    min_frames = ctx.frames(p.static_min_s)
    max_gap = ctx.frames(p.static_max_gap_s)
    keep, representative = _deduplicate(data)
    for chain in _chain_runs(frames, points, p.static_radius_px, max_gap, keep):
        members = np.asarray(chain)
        length[members] = members.size
        if members.size < min_frames:
            continue
        verdict = _camera_motion(frames[members], points[members[0]], ctx)
        if verdict in ("cut", "calibration"):
            static[members] = True
        elif verdict == "players":
            static[members] = ~near[members]
    step = p.parked_step_px * ctx.height / p.reference_lines
    for chain in _chain_runs(frames, points, step, max_gap, keep, follow=True):
        if len(chain) >= min_frames:
            members = np.asarray(chain)
            parked[members] = ~near[members] & ~static[members]
    return length[representative], static[representative], parked[representative]


def _deduplicate(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(rows kept for chaining, representative row of every row).

    A second detector box on the same object (centre within half the larger diameter of
    a more confident box of the same frame) would start a parallel chain that competes
    for single detections and starves; it is left out of the chains and takes the
    verdicts of its representative instead.
    """
    frames, conf = data["frame"], data["conf"]
    u, v, diameter = data["u"], data["v"], data["diameter_px"]
    representative = np.arange(frames.size)
    starts = np.flatnonzero(np.r_[True, frames[1:] != frames[:-1]]) if frames.size else []
    ends = np.r_[starts[1:], frames.size] if frames.size else []
    for start, end in zip(np.asarray(starts).tolist(), np.asarray(ends).tolist(), strict=True):
        order = start + np.argsort(-conf[start:end], kind="stable")
        kept: list[int] = []
        for row in order.tolist():
            for other in kept:
                reach = 0.5 * max(diameter[row], diameter[other])
                if math.hypot(u[row] - u[other], v[row] - v[other]) <= reach:
                    representative[row] = other
                    break
            else:
                kept.append(row)
    return np.flatnonzero(representative == np.arange(frames.size)), representative


def _camera_motion(member_frames: np.ndarray, anchor: np.ndarray, ctx: _Context) -> str | None:
    """How we know the camera moved during a screen-fixed run, or None if it did not.

    "cut" when the run spans two shots; "calibration" or "players" when the scene point
    under the run moved by at least static_camera_shift_px on static_min_moved_frames
    frames, measured with the calibration if more than static_min_moved_frames of the
    run's frames are pitch-usable, else with the players' cumulative median motion.
    """
    p = ctx.params
    shots = ctx.shot_of_frame[member_frames]
    if shots.min() != shots.max():
        return "cut"
    usable = [int(f) for f in member_frames if int(f) in ctx.cameras]
    if len(usable) > p.static_min_moved_frames:
        reference = ctx.cameras[usable[0]]
        shifts = [_scene_shift(reference, ctx.cameras[f], anchor) for f in usable[1:]]
        moved = sum(shift >= p.static_camera_shift_px for shift in shifts)
        return "calibration" if moved >= p.static_min_moved_frames else None
    motion = ctx.player_motion()
    first, last = int(member_frames[0]), int(member_frames[-1])
    steps = np.nan_to_num(motion[first + 1 : last + 1], nan=0.0)
    shift = np.zeros((last - first + 1, 2))
    shift[1:] = np.cumsum(steps, axis=0)
    distances = np.hypot(shift[:, 0], shift[:, 1])[member_frames - first]
    moved = int(np.count_nonzero(distances >= p.static_camera_shift_px))
    return "players" if moved >= p.static_min_moved_frames else None


def _world_fixed_off_field(data: Mapping[str, np.ndarray], ctx: _Context) -> np.ndarray:
    """True per gate row for candidates on a world-fixed spot outside the field lines.

    1. On pitch-usable frames, the ground points lying more than off_field_point_margin_m
       outside the field lines are grouped per shot by single linkage at
       world_static_radius_m (order-free, so a ball passing a spare ball does not split
       it). A group from at least world_static_min_members distinct frames spanning at
       least static_min_s whose median lies more than off_field_margin_m outside is a
       world-fixed object off the field, and all its members are rejected.
    2. A candidate on a frame without a usable calibration is rejected too when a rejected
       candidate lies within off_field_carry_px (plus half its diameter) on one of the
       off_field_carry_frames neighbouring frames, repeatedly, so the label follows a
       spare ball through frames whose calibration failed.
    """
    p = ctx.params
    frames = data["frame"]
    x, y = data["pitch_x"], data["pitch_y"]
    result = np.zeros(frames.size, dtype=bool)
    finite = np.isfinite(x) & np.isfinite(y)
    ax = np.abs(np.where(finite, x, 0.0))
    ay = np.abs(np.where(finite, y, 0.0))
    outside = finite & (
        (ax > FIELD_HALF_LENGTH_M + p.off_field_point_margin_m)
        | (ay > FIELD_HALF_WIDTH_M + p.off_field_point_margin_m)
    )
    if not outside.any():
        return result
    min_span = ctx.frames(p.static_min_s)
    shot_ids = data["shot_id"]
    for shot_id in np.unique(shot_ids[outside]).tolist():
        rows = np.flatnonzero(outside & (shot_ids == shot_id))
        points = np.stack([x[rows], y[rows]], axis=-1)
        pairs = cKDTree(points).query_pairs(p.world_static_radius_m, output_type="ndarray")
        _count, labels = connected_components(
            coo_matrix(
                (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                shape=(rows.size, rows.size),
            ),
            directed=False,
        )
        for label in np.unique(labels).tolist():
            members = rows[labels == label]
            member_frames = np.unique(frames[members])
            if (
                member_frames.size < p.world_static_min_members
                or int(member_frames[-1] - member_frames[0]) + 1 < min_span
            ):
                continue
            mx = abs(float(np.median(x[members])))
            my = abs(float(np.median(y[members])))
            if (
                mx > FIELD_HALF_LENGTH_M + p.off_field_margin_m
                or my > FIELD_HALF_WIDTH_M + p.off_field_margin_m
            ):
                result[members] = True
    _carry_to_uncalibrated(result, data, p)
    return result


def _carry_to_uncalibrated(
    flags: np.ndarray, data: Mapping[str, np.ndarray], p: BallParams
) -> None:
    """Extend ``flags`` (in place) to candidates of frames without a usable calibration,
    within the same shot."""
    frames, shot_ids = data["frame"], data["shot_id"]
    open_rows = np.flatnonzero(~data["pitch_usable"])
    if open_rows.size == 0 or not flags.any():
        return
    u, v, diameter = data["u"], data["v"], data["diameter_px"]
    by_frame: dict[int, list[int]] = {}
    for row, frame in enumerate(frames.tolist()):
        by_frame.setdefault(frame, []).append(row)
    reach = p.off_field_carry_frames
    changed = True
    while changed:
        changed = False
        for row in open_rows.tolist():
            if flags[row]:
                continue
            frame = int(frames[row])
            limit = p.off_field_carry_px + 0.5 * float(diameter[row])
            for other_frame in range(frame - reach, frame + reach + 1):
                for other in by_frame.get(other_frame, ()):
                    if (
                        flags[other]
                        and shot_ids[other] == shot_ids[row]
                        and math.hypot(u[other] - u[row], v[other] - v[row]) <= limit
                    ):
                        flags[row] = True
                        changed = True
                        break
                if flags[row]:
                    break


# ----------------------------------------------------------------------------------------
# Tracking
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Options:
    """Gated candidates of one frame (arrays of equal length)."""

    rows: np.ndarray  # row in the gate table
    u: np.ndarray
    v: np.ndarray
    scale: np.ndarray  # ball diameter / ball_to_box_height, px
    log_d: np.ndarray  # ln(ball diameter / px)
    emission: np.ndarray

    @property
    def n(self) -> int:
        return int(self.rows.size)


def _link_cost(
    before: _Options, after: _Options, gap: int, ref_bh_per_frame: float, ctx: _Context
) -> np.ndarray:
    """(before.n, after.n) cost of linking candidates ``gap`` frames apart.

    Motion cost (speed in bh per frame, normalised by the mean ball scale of the two
    candidates; quadratic up to ref_bh_per_frame, linear beyond, times motion_weight and
    the gap) plus size continuity (0.5 z^2 with z = ln(d_after / d_before) /
    (size_change_sigma_log sqrt(gap))). Infinite above max_link_speed_bh_per_s.
    """
    p = ctx.params
    du = after.u[None, :] - before.u[:, None]
    dv = after.v[None, :] - before.v[:, None]
    scale = 0.5 * (before.scale[:, None] + after.scale[None, :])
    speed = np.hypot(du, dv) / (gap * scale)  # box heights per frame
    r = speed / ref_bh_per_frame
    per_frame = np.where(r <= 1.0, r * r, 2.0 * r - 1.0)
    z = (after.log_d[None, :] - before.log_d[:, None]) / (p.size_change_sigma_log * math.sqrt(gap))
    cost = p.motion_weight * gap * per_frame + 0.5 * z * z
    return np.where(speed * ctx.fps > p.max_link_speed_bh_per_s, np.inf, cost)


def _viterbi_shot(
    start: int, end: int, options: Mapping[int, _Options], ctx: _Context
) -> tuple[dict[int, int], dict[int, int]]:
    """Best path through one shot.

    Returns ({frame: index into that frame's options}, {frame: link gap}) for the chosen
    detections; the link gap is the number of frames back to the detection it is linked to
    by a motion link, or 0 when the ball was (re-)acquired there.
    """
    p = ctx.params
    max_gap = ctx.frames(p.max_link_gap_s)
    ref = p.motion_ref_bh_per_s / ctx.fps
    length = end - start + 1
    empty_f = np.zeros(0)
    empty_i = np.zeros(0, dtype=np.int64)
    score: list[np.ndarray] = []  # best path ending on candidate c at i
    link_gap: list[np.ndarray] = []  # 0: picked up from "missing"; g: linked g frames back
    link_from: list[np.ndarray] = []  # candidate index g frames back
    lost = np.empty(length)  # best path with frame i missing
    lost_from = np.empty(length, dtype=np.int64)  # -1: frame i - 1 missing too; else c
    for i in range(length):
        current = options.get(start + i)
        previous_lost = lost[i - 1] if i > 0 else 0.0
        if current is None or current.n == 0:
            score.append(empty_f)
            link_gap.append(empty_i)
            link_from.append(empty_i)
        else:
            best = previous_lost - p.reacquire_cost + current.emission
            gap_of = np.zeros(current.n, dtype=np.int64)
            from_of = np.full(current.n, -1, dtype=np.int64)
            for gap in range(1, min(max_gap, i) + 1):
                j = i - gap
                if score[j].size == 0:
                    continue
                before = options[start + j]
                total = score[j][:, None] - _link_cost(before, current, gap, ref, ctx)
                arg = np.argmax(total, axis=0)
                value = (
                    total[arg, np.arange(current.n)] - (gap - 1) * p.missing_cost + current.emission
                )
                better = value > best
                best = np.where(better, value, best)
                gap_of = np.where(better, gap, gap_of)
                from_of = np.where(better, arg, from_of)
            score.append(best)
            link_gap.append(gap_of)
            link_from.append(from_of)
        if i > 0 and score[i - 1].size and float(score[i - 1].max()) > previous_lost:
            lost_from[i] = int(np.argmax(score[i - 1]))
            lost[i] = float(score[i - 1].max()) - p.missing_cost
        else:
            lost_from[i] = -1
            lost[i] = previous_lost - p.missing_cost
    chosen: dict[int, int] = {}
    links: dict[int, int] = {}
    i = length - 1
    on: int | None = None  # candidate index, or None for "missing"
    if score[i].size and float(score[i].max()) > lost[i]:
        on = int(np.argmax(score[i]))
    while i >= 0:
        if on is None:
            back = int(lost_from[i])
            i -= 1
            on = back if back >= 0 else None
        else:
            chosen[start + i] = on
            gap = int(link_gap[i][on])
            links[start + i] = gap
            if gap == 0:
                i -= 1
                on = None
            else:
                on = int(link_from[i][on])
                i -= gap
    return chosen, links


def track_ball(
    candidates: Mapping[int, Any],
    tracks: pd.DataFrame,
    calib: pd.DataFrame,
    shots: pd.DataFrame,
    fps: float | str | Fraction,
    n_frames: int,
    *,
    frame_size: tuple[int, int] | None = None,
    params: BallParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Track the ball through the clip; one row per frame 0..n_frames-1 (BALL_TRACK_COLUMNS).

    Args:
        candidates: {frame: (k, 5) [x1, y1, x2, y2, conf]}, as from load_candidates.
        tracks: player boxes (TRACKS_COLUMNS_READ).
        calib: per-frame calibration (CALIB_COLUMNS_READ, optional width/height); only
            pitch-usable rows are used.
        shots: shot table (SHOTS_COLUMNS_READ, optional median_box_h); the shots must cover
            every frame once. The tracker never links, interpolates or carries anything
            across a cut.
        fps: frame rate (number, Fraction or "25/1"), for the durations in params.
        n_frames: number of video frames.
        frame_size: (width, height) of the video frames; defaults to the single size in
            calib's width/height columns.
        params: thresholds, see BallParams.

    The method is described in the module docstring. Raises BallInputError listing every
    input problem.
    """
    ctx = _prepare(
        "track_ball", candidates, tracks, calib, shots, fps, n_frames, frame_size, params
    )
    track, _gates, _links = _track(ctx)
    return track


def _track(ctx: _Context) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    """(track, gate table, {detected frame: link gap}) for a prepared context."""
    gates = _gate(ctx)
    p = ctx.params
    passed = np.flatnonzero((gates["reason"] == "ok").to_numpy())
    frame_col = gates["frame"].to_numpy()
    u_col = gates["u"].to_numpy()
    v_col = gates["v"].to_numpy()
    scale_col = gates["scale_px"].to_numpy()
    conf_col = gates["conf"].to_numpy()
    diameter_col = gates["diameter_px"].to_numpy()
    emission_col = np.log(np.maximum(conf_col, p.conf_floor)) + gates["size_prior"].to_numpy()
    options: dict[int, _Options] = {}
    if passed.size:
        cuts = np.flatnonzero(np.diff(frame_col[passed])) + 1
        for rows in np.split(passed, cuts):
            options[int(frame_col[rows[0]])] = _Options(
                rows=rows,
                u=u_col[rows],
                v=v_col[rows],
                scale=scale_col[rows],
                log_d=np.log(diameter_col[rows]),
                emission=emission_col[rows],
            )
    n = ctx.n_frames
    u = np.full(n, np.nan)
    v = np.full(n, np.nan)
    radius = np.full(n, np.nan)
    conf = np.full(n, np.nan, dtype=np.float32)
    status = np.full(n, MISSING, dtype=object)
    max_interp = ctx.frames(p.max_interp_gap_s)
    all_links: dict[int, int] = {}
    isolation = ctx.frames(p.isolated_gap_s)
    for shot in ctx.shots:
        chosen, links = _viterbi_shot(shot.start, shot.end, options, ctx)
        detected = sorted(chosen)
        dropped = set()
        for index, frame in enumerate(detected):
            before = frame - detected[index - 1] if index > 0 else math.inf
            after = detected[index + 1] - frame if index + 1 < len(detected) else math.inf
            row = int(options[frame].rows[chosen[frame]])
            if min(before, after) >= isolation and conf_col[row] < p.isolated_min_conf:
                dropped.add(frame)
        for frame in dropped:
            del chosen[frame]
            del links[frame]
        for frame, gap in list(links.items()):
            if gap and frame - gap in dropped:
                links[frame] = 0
        all_links.update(links)
        for frame, index in chosen.items():
            row = int(options[frame].rows[index])
            u[frame], v[frame] = u_col[row], v_col[row]
            radius[frame] = diameter_col[row] / 2.0
            conf[frame] = conf_col[row]
            status[frame] = DETECTED
        for frame, gap in links.items():
            if not 2 <= gap <= max_interp + 1:
                continue
            first = frame - gap
            scale = (radius[first] + radius[frame]) / p.ball_to_box_height  # mean d / ratio
            step = math.hypot(u[frame] - u[first], v[frame] - v[first]) / gap
            if step / scale * ctx.fps > p.max_interp_speed_bh_per_s:
                continue
            weights = np.arange(1, gap) / gap
            inside = np.arange(first + 1, frame)
            u[inside] = u[first] + weights * (u[frame] - u[first])
            v[inside] = v[first] + weights * (v[frame] - v[first])
            radius[inside] = radius[first] + weights * (radius[frame] - radius[first])
            status[inside] = INTERPOLATED
    counts = np.zeros(n, dtype=np.int32)
    for frame, cands in ctx.candidates.items():
        counts[frame] = cands.shape[0]
    pitch_x = np.full(n, np.nan)
    pitch_y = np.full(n, np.nan)
    for frame, camera in ctx.cameras.items():
        if status[frame] == MISSING:
            continue
        point, valid = pixel_to_ground(camera, np.array([u[frame], v[frame]]), -p.ball_radius_m)
        if bool(valid):
            pitch_x[frame], pitch_y[frame] = float(point[0]), float(point[1])
    track = pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "u": u,
            "v": v,
            "radius_px": radius,
            "conf": conf,
            "status": status.tolist(),
            "candidate_count": counts,
            "pitch_x": pitch_x,
            "pitch_y": pitch_y,
            "ground_assumption": np.isfinite(pitch_x),
        }
    )
    return track, gates, all_links


# ----------------------------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------------------------


def _fit(times: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares velocity (px per frame, (2,)) of points (n, 2) at times (n,), and the
    RMS distance (px) of the points from the fitted line (0 for two points)."""
    centred = times - times.mean()
    mean = points.mean(axis=0)
    velocity = (centred[:, None] * (points - mean)).sum(axis=0) / float((centred * centred).sum())
    residual = points - (mean + centred[:, None] * velocity)
    return velocity, float(np.sqrt((residual * residual).sum(axis=1).mean()))


def _segments(
    frames: np.ndarray, points: np.ndarray, diameters: np.ndarray, rate: float, p: BallParams
) -> np.ndarray:
    """Segment id per detected sample (sorted by frame, one shot): a new segment starts
    where the tracker could not have linked two consecutive samples (more than
    max_link_gap frames apart, faster than max_link_speed_bh_per_s at their mean ball
    scale, or a diameter jump beyond 3 sigma of size_change_sigma_log), i.e. where the
    track was re-acquired on another object."""
    segment = np.zeros(frames.size, dtype=np.int64)
    max_gap = max(1, int(round(p.max_link_gap_s * rate)))
    for i in range(1, frames.size):
        gap = int(frames[i] - frames[i - 1])
        d0, d1 = float(diameters[i - 1]), float(diameters[i])
        scale = 0.5 * (d0 + d1) / p.ball_to_box_height
        step = float(np.hypot(*(points[i] - points[i - 1]))) / gap
        jump = abs(math.log(d1 / d0)) if d0 > 0.0 and d1 > 0.0 else math.inf
        broken = (
            gap > max_gap
            or not (math.isfinite(scale) and scale > 0.0)
            or step / scale * rate > p.max_link_speed_bh_per_s
            or jump > 3.0 * p.size_change_sigma_log * math.sqrt(gap)
        )
        segment[i] = segment[i - 1] + int(broken)
    return segment


def _check_ball_track(ball_track: Any, problems: list[str]) -> pd.DataFrame | None:
    required = ("frame", "u", "v", "status", "radius_px")
    if _missing_columns(ball_track, "ball_track", required, problems):
        return None
    found = len(problems)
    frames = _integer_column(ball_track, "ball_track", "frame", problems)
    u = _float_column(ball_track, "ball_track", "u", problems)
    v = _float_column(ball_track, "ball_track", "v", problems)
    radius = _float_column(ball_track, "ball_track", "radius_px", problems)
    if frames is None or u is None or v is None or radius is None:
        return None
    if len(np.unique(frames)) != len(frames):
        problems.append("ball_track.frame: values must be unique")
    statuses = ball_track["status"].tolist()
    unknown = sorted({str(s) for s in statuses if s not in STATUSES})
    if unknown:
        problems.append(f"ball_track.status: unknown value(s) {unknown}; expected {STATUSES}")
    detected = np.array([s == DETECTED for s in statuses], dtype=bool)
    if detected.size and not (np.isfinite(u[detected]) & np.isfinite(v[detected])).all():
        problems.append("ball_track: detected rows need finite u and v")
    if len(problems) > found:
        return None
    return pd.DataFrame(
        {"frame": frames, "u": u, "v": v, "radius_px": radius, "detected": detected}
    )


def ball_events(
    ball_track: pd.DataFrame,
    tracks: pd.DataFrame,
    fps: float | str | Fraction,
    *,
    shots: pd.DataFrame,
    params: BallParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Abrupt changes of the ball's image velocity (BALL_EVENT_COLUMNS), sorted by frame.

    Only detected samples are used, and only within one shot (``shots`` is required for
    that: without it a cut would look like a kick). For each detected frame k with detected
    samples within event_window frames on both sides, the incoming velocity is the
    least-squares slope through the samples in [k - W, k] and the outgoing one through
    [k, k + W]. Speeds are normalised by the ball's own scale (median diameter of the
    window's samples / ball_to_box_height, the box height at the ball's depth) and
    expressed in bh/s. Windows stop where the tracker could not have linked two
    consecutive samples (more than max_link_gap frames apart, faster than
    max_link_speed_bh_per_s, or a diameter jump beyond 3 sigma): the track was
    re-acquired on another object there. A window that implies a speed above
    event_max_speed_bh_s or whose samples miss a straight line by more than
    event_max_fit_rms_bh is not one ball moving steadily (a second touch inside the
    window) and gives no event.
    k is an event when the direction changes by more than event_angle_deg with both speeds
    at least event_speed_floor_bh_s ("direction_change"), or when the faster / slower speed
    ratio exceeds event_speed_ratio with the faster speed at least the floor and the change
    at least event_min_change_bh_s ("speed_up" or "slow_down"). Of events closer than the
    window, the one with the largest magnitude stays. The nearest player is the one of
    player_roles with the smallest distance from the ball centre to the foot point, in
    their own box heights; ``at_foot`` marks events within event_at_foot_bh of it (ball
    strikes, traps and touches; the others are mostly bounces and the top of a lofted
    ball's flight). All thresholds are empirical (BallParams).

    The vertex must itself be detected: a ball hidden at the contact frame gives the event
    on a neighbouring detected frame, or none. Raises BallInputError listing every problem.
    """
    problems: list[str] = []
    if not isinstance(params, BallParams):
        problems.append(f"params: expected BallParams, got {type(params).__name__}")
    rate = _check_fps(fps, problems)
    track = _check_ball_track(ball_track, problems)
    shot_list = _check_shots(shots, None, problems)
    n_frames = shot_list[-1].end + 1 if shot_list else None
    players = _check_tracks(tracks, n_frames, problems)
    if track is not None and n_frames is not None:
        outside = (track["frame"] < 0) | (track["frame"] >= n_frames)
        if outside.any():
            problems.append(
                f"ball_track.frame: {int(outside.sum())} row(s) outside the shots "
                f"[0, {n_frames - 1}]"
            )
    if (
        problems
        or rate is None
        or track is None
        or shot_list is None
        or players is None
        or not isinstance(params, BallParams)
    ):
        raise BallInputError("ball_events", problems)
    p = params
    window = max(1, int(round(p.event_window_s * rate)))
    roles = set(p.player_roles)
    detected = track[track["detected"]].sort_values("frame", kind="stable")
    all_frames = detected["frame"].to_numpy()
    all_points = detected[["u", "v"]].to_numpy()
    all_radius = detected["radius_px"].to_numpy()
    rows: list[dict[str, Any]] = []
    for shot in shot_list:
        inside = (all_frames >= shot.start) & (all_frames <= shot.end)
        frames = all_frames[inside]
        points = all_points[inside]
        radius = all_radius[inside]
        segment = _segments(frames, points, 2.0 * radius, rate, p)
        found: list[dict[str, Any]] = []
        for index, k in enumerate(frames.tolist()):
            low = int(np.searchsorted(frames, k - window, side="left"))
            high = int(np.searchsorted(frames, k + window, side="right"))
            same = segment[low:high] == segment[index]
            low += int(np.argmax(same))  # first sample of the vertex's segment in the window
            high = low + int(np.count_nonzero(segment[low:high] == segment[index]))
            if low == index or high == index + 1:
                continue  # no detected sample of the same segment on one side
            times_in = frames[low : index + 1].astype(np.float64)
            times_out = frames[index:high].astype(np.float64)
            v_in, rms_in = _fit(times_in, points[low : index + 1])
            v_out, rms_out = _fit(times_out, points[index:high])
            u_k, v_k = float(points[index, 0]), float(points[index, 1])
            frame_players = players.get(k)
            diameters = 2.0 * radius[low:high]
            diameters = diameters[np.isfinite(diameters) & (diameters > 0.0)]
            if diameters.size == 0:
                continue
            scale = float(np.median(diameters)) / p.ball_to_box_height
            tolerance = max(p.event_min_fit_rms_px, p.event_max_fit_rms_bh * scale)
            if rms_in > tolerance or rms_out > tolerance:
                continue
            to_bh_s = rate / scale
            speed_in = float(np.hypot(*v_in))
            speed_out = float(np.hypot(*v_out))
            if max(speed_in, speed_out) * to_bh_s > p.event_max_speed_bh_s:
                continue
            change = float(np.hypot(*(v_out - v_in))) * to_bh_s
            slow, fast = sorted((speed_in, speed_out))
            angle = math.nan
            if speed_in > 0.0 and speed_out > 0.0:
                cosine = float(np.dot(v_in, v_out)) / (speed_in * speed_out)
                angle = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
            turns = (
                math.isfinite(angle)
                and angle > p.event_angle_deg
                and slow * to_bh_s >= p.event_speed_floor_bh_s
            )
            ratio = fast / slow if slow > 0.0 else math.inf
            jumps = (
                ratio > p.event_speed_ratio
                and fast * to_bh_s >= p.event_speed_floor_bh_s
                and change >= p.event_min_change_bh_s
            )
            if not (turns or jumps):
                continue
            if turns:
                kind = "direction_change"
            else:
                kind = "speed_up" if speed_out > speed_in else "slow_down"
            nearest_id, nearest_team, nearest_dist = -1, None, math.nan
            if frame_players is not None:
                keep = np.array([r in roles for r in frame_players.role], dtype=bool)
                keep &= np.isfinite(frame_players.foot_u) & np.isfinite(frame_players.foot_v)
                if keep.any():
                    distance = (
                        np.hypot(frame_players.foot_u - u_k, frame_players.foot_v - v_k)
                        / frame_players.h
                    )
                    distance = np.where(keep, distance, np.inf)
                    best = int(np.argmin(distance))
                    nearest_id = int(frame_players.track_id[best])
                    nearest_team = frame_players.team[best]
                    nearest_dist = float(distance[best])
            found.append(
                {
                    "frame": int(k),
                    "kind": kind,
                    "magnitude": change,
                    "u": u_k,
                    "v": v_k,
                    "nearest_track_id": nearest_id,
                    "nearest_team": nearest_team,
                    "nearest_dist_box_h": nearest_dist,
                    "shot_id": shot.shot_id,
                    "angle_deg": angle,
                    "speed_before": speed_in * to_bh_s,
                    "speed_after": speed_out * to_bh_s,
                    "at_foot": bool(nearest_dist <= p.event_at_foot_bh),
                }
            )
        found.sort(key=lambda e: (-e["magnitude"], e["frame"]))
        kept: list[dict[str, Any]] = []
        for event in found:
            if all(abs(event["frame"] - other["frame"]) > window for other in kept):
                kept.append(event)
        rows.extend(kept)
    rows.sort(key=lambda e: e["frame"])
    result = pd.DataFrame(rows, columns=list(BALL_EVENT_COLUMNS))
    return result.astype(
        {
            "frame": np.int64,
            "magnitude": np.float64,
            "u": np.float64,
            "v": np.float64,
            "nearest_track_id": np.int64,
            "nearest_dist_box_h": np.float64,
            "shot_id": np.int64,
            "angle_deg": np.float64,
            "speed_before": np.float64,
            "speed_after": np.float64,
            "at_foot": bool,
        }
    )
