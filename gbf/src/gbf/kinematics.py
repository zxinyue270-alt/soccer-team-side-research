"""Player kinematics on pitch-usable frames: track segments, smoothing, velocity and peaks.

Inputs are tracks.parquet rows (gbf.sngs_import) and shots.parquet rows (gbf.shots_grammar).
Positions and velocities are in the SoccerNet world frame: metres, origin at the centre
spot, x along the touchlines, far touchline at y = -34; velocities in m/s (km/h only on
screen). Times are canonical frame indices; seconds = frame / fps.

Segments
    For every track with ``track_id >= 0``, a frame is *measured* when it is pitch-usable,
    its foot point has a ground position (finite pitch_x, pitch_y) and that position lies
    in the extended pitch (``in_pitch``; a point more than 3 m outside the lines is a
    projection error or not a player). Measured frames are split into segments of
    consecutive frames inside one shot. A gap of at most ``max_gap_frames`` missing frames
    is bridged by linear interpolation and flagged (``interpolated``); a longer gap or a
    shot boundary starts a new segment. A cut changes the camera, so a track is never
    smoothed across it.

Calibration jumps
    Every fresh frame has its own calibration, and consecutive calibrations can disagree:
    a per-frame fit can trade focal length against camera distance and still reproject the
    few visible pitch keypoints within a pixel or two. Such a jump moves every player on
    the frame by about the same amount. :func:`common_mode_steps` measures it: for each
    frame t whose predecessor t - 1 is measured too, the median over players present in
    both frames of their step p(t) - p(t - 1). Real collective motion stays far below
    ``max_common_step_m`` (0.5 m per frame, i.e. a whole team moving at 12.5 m/s), so a
    larger common step is a calibration jump and every segment is split there instead of
    being smoothed across it. Frames whose predecessor is not measured are not checked.

Position outliers
    A box that merges with a neighbour or loses the feet to an occlusion moves the foot
    point, and so the ground position, by metres for a frame or two. Before segmenting,
    three tests reject measured positions, which are then treated as missing (interpolated
    when the gap is short), and two more split a track where its trend jumps. Each works
    inside one run: the track's measured frames in one shot between two calibration jumps.
    :func:`rejected_positions` lists every rejection with the test that made it.

    - Box size: a frame whose box height differs from the median box height of its track's
      measured frames within ``outlier_radius_frames`` (5) frames by more than
      ``max_box_height_change`` (20 %) of that median is rejected, and likewise for the box
      width with ``max_box_width_change`` (off by default, see
      :data:`DEFAULT_MAX_BOX_WIDTH_CHANGE`). A box that loses the legs to an occlusion or
      the image edge, or reaches down to a neighbour's feet, moves its bottom edge, and the
      ground position follows it: a few pixels of the foot point are tens of centimetres in
      depth on a wide shot.
    - Distance from the local median: of the remaining positions, one farther than
      ``outlier_threshold_m`` (1.5 m) from the component-wise median of its track's
      positions within ``outlier_radius_frames`` frames is rejected. The median of a steady
      run is its centre, so real motion stays well below the threshold.
    - Excursions and jumps: the remaining positions of a run are cut into pieces at gaps of
      more than ``max_gap_frames`` missing frames. A broadcast foot point does not move
      smoothly: the bottom of a running player's box follows the planted foot, which stands
      still while it is on the ground and moves on by up to about half a step at the next
      foot strike, so the single step between two frames of a real sprint can exceed any
      fixed per-frame limit, and noise of a few centimetres adds to it. These tests
      therefore compare trends, never single steps. The excursion and step tests fit the
      positions of two sides by least squares with one shared velocity and a step between
      the sides (a vector), each side spanning ``jump_window_frames`` frames. A constant
      velocity, a constant acceleration and a kink of the velocity between the sides all
      leave the step at zero when both sides are sampled alike, and missing frames only
      lengthen the extrapolation, so no allowance grows across a gap.

      An excursion is a block of 1 to ``max_gap_frames`` consecutive positions (spanning
      at most that many frames) all farther than max(``jump_threshold_m``, ``jump_factor`` *
      m) from both lines of such a fit of the positions before and after the block (at
      least 2 on each side; m is the other players' median of this distance over the same
      frames, used when at least ``min_common_players`` have one), while the positions just
      before and after the block lie within that distance of the line of their own side,
      fitted with the block left out too: a box that dips onto a neighbour or loses the legs
      for a frame or a few. Testing the dip's positions together keeps each out of the
      others' trend fits. A second short dip in the side windows still pulls the fitted
      trends towards itself, so that neither dip need lie 1 m off them: each block is tested
      again on side fits made twice more without the side positions more than half of
      ``jump_threshold_m`` off the line of their side (except the run of them next to the
      block that lie off on the block's side, which continues it, so that two dips one
      position apart can stay one excursion for the jump tests), and it is an excursion
      when either test finds it. A block that only the second test finds stands only when
      the side positions its fits leave out, and that stay that far off the final lines,
      are rejected too, as the other dip. A position next to a jump lies on the line of its
      own side, so it is kept, and a block inside a longer excursion is left to the jump
      tests below, which cut the whole excursion off (rejecting part of it would leave the
      rest at the end of a segment). Excursions are rejected (the gap they leave is bridged
      like any other) and the test runs again on what it kept, up to 3 times, so that an
      excursion hidden at first by another one in its fits is found too.

      Then a jump is a step |s| > max(``jump_threshold_m``, ``jump_factor`` * m) between two
      consecutive positions a and b of a piece (up to ``max_gap_frames`` missing frames
      between them), fitted on the positions in the ``jump_window_frames`` frames ending at
      a and those starting at b (at least 3 on each side); m is the median |s| of the other
      players whose consecutive positions are the same two frames, and follows camera and
      calibration motion that moves every player at once.

      The shared velocity takes a box that slides onto a neighbour over several frames, or
      jumps less than ``jump_threshold_m``, for motion, so a junction is a jump too when the
      acceleration test finds it. The positions of ``jump_window_frames`` frames on each
      side, a guard of g = 0, 2, ... jump_window_frames / 2 frames next to the junction
      left out (at least ``jump_window_frames`` - ``max_gap_frames`` positions on each side),
      each get a line of their own, with velocities v0 and v1 at their centroids T apart.
      The displacement e between the lines that their velocities do not explain is first
      reduced by |v1 - v0| T / 2, the most one sudden change of velocity anywhere between
      the centroids explains (a planted foot stops the foot point at once), and by 3
      standard errors of e from the scatter of the positions about their lines. Leaving the
      trend by what is left and coming back within T takes an acceleration of at least
      4 e / T^2; above max(``max_acceleration_mps2`` (8 m/s^2), ``jump_factor`` * m) for any
      guard, the junction is a jump (with a guard, the 2g frames left out are cut off on
      both sides as a short segment of their own). Sprinters start, stop and curve at up to
      about 6-7 m/s^2, so their positions stay one segment, held for up to 3 frames or with
      10 cm of noise. A burst of speed from 3 to 9 m/s and back with a Gaussian sigma of
      0.24 s or less, which takes 15 m/s^2 or more, is cut off too (at 0.32 s it is kept).

      A jump raises the statistics of the junctions next to it too, so a piece is split at
      its largest jump first and each part is fitted again on its own. A jump starts a new
      segment: an identity switch onto a neighbour is never smoothed into a sprint.

      What stays, measured with the defaults at 25 fps on synthetic 6 s tracks without
      other players, of a runner at 1, 3, 6 or 9 m/s with the artefact along the motion or
      at 1 or 6 m/s with the artefact sideways ("adds" is the highest valid speed less the
      true one; noise is per axis). Figures with noise are the largest seen in the sample
      named, not bounds. A switch from one frame to the next stays up to 0.46 m on clean
      positions (0.54, 0.62 and 0.72 m across 1, 2 and 3 missing frames; sizes in 1 cm
      steps), and noise lets larger ones stay. In 60 tracks per size (10 seeds; 0.3-2 m in
      5 cm steps) with 5, 10, 20 and 30 cm of noise, half of them keep switches up to 0.6,
      0.75, 1 and 1 m, and the largest seen to stay are 0.7, 0.95, 1.35 and 1.45 m; across
      1, 2 or 3 missing frames (60 tracks per size for each) they are 1.05, 1.15, 1.4 and
      1.6 m. The filter overshoots a step by 8.4 %, so a switch of s metres that stays adds
      up to 1.084 s / 0.4 s along the motion (1.25 m/s at 0.46 m) and less across it, plus
      what the noise adds: in that sample up to 1.72, 2.27, 3.08 and 3.36 m/s in all (2.71,
      2.89, 3.37 and 3.94 m/s across missing frames). Another sample (5 seeds x 6 runners
      per size and missing-frame count) keeps switches up to 1.2 m at 10 cm and 1.45 m at
      20 cm across 1-3 missing frames, adding up to 2.97 and 3.40 m/s respectively.
      On clean positions a slide onto a
      neighbour over k frames is split when k is at most 1 for 0.5 m, 5 for 1 m, 10 for
      1.5 m, 13 for 2 m, 17 for 3 m, 19 for 3.5-4 m, 21 for 4.5-6 m, 23 for 6.5-9.5 m and 25
      for 10 m (0.5-10 m in 0.5 m steps over 1-40 frames; some slides of 9.5 m over 25
      frames are split too). With noise some of these stay (12 tracks per slide of 1, 2, 3,
      5 or 8 m over every third k of 2-59: 1 m over 2 and 5 frames, 3 m over 17 and 5 m over
      20 with 10 cm, and also 2 m over 11 and 3 m over 11 and 14 with 30 cm). A slower slide
      looks like a change of speed and stays: the chords over it read the slide's own speed
      on top of the player's, along the motion or in quadrature across it (at most 2.7 % or
      0.15 m/s more), bounded only by ``max_speed_mps`` (a 1 m/s jogger sliding 9.5 m in 24
      frames reads 11.04 m/s, a 6 m/s runner sliding 8 m ahead in 34 frames 11.97 m/s), and
      so do the ramps of an excursion out and back (0.5-8 m over 1-16 frames each way, held
      0, 3 or 12 frames: up to 5.7 m/s added). An out-and-back dip that stays adds up to
      2.4 m/s on clean positions (0.3-2.5 m in 0.1 m steps for 1-14 frames; 1 m for 8 frames
      ahead of a jogger, which the jump tests split but do not cut off), as much below 1 m
      (2.38 m/s for 0.99 m for 8 frames; 0.85-1.15 m in 1 cm steps for 5-12 frames) and
      2.0 m/s for 0.7 m for 9-10 frames ahead of a 3-9 m/s run; with 10 and 20 cm of noise
      up to 2.57 and 3.07 m/s (12 tracks per size and length, 2 seeds). Two dips of 1-3
      frames and 1.05-2 m, 1-14 frames apart (along the motion of runners at 1, 6 and 9 m/s
      or sideways to that of runners at 1 and 6 m/s), lose every position in 3127 of 3150
      clean pairs, and the others add up to 0.46 m/s (two 3-frame dips of 2 m 2 frames
      apart, split off in part); with 5 and 10 cm of noise parts of 833 and 2167 of 6300
      pairs stay (2 seeds) and add up to 2.02 m/s, and in 882 other pairs per level (1.3 and
      2 m, 1-12 frames apart, 7 runners, one seed) up to 2.19 m/s with 10 cm. Three 3-frame
      dips of 1.2 m 1-3 frames apart are not rejected whole and add up to 2.55, 2.23 and
      1.69 m/s 1, 2 and 3 frames apart (clean, 7 runners). Clean pairs of dips of 1-4
      frames, 1-8 frames apart, add up to 1.94 m/s below 1 m and, where one lasts 4 frames,
      which the jump tests leave whole, up to 2.28 m/s (1.2 m). Noise alone raises the
      highest valid speed by a median of 0.4 m/s and at most 0.6 m/s per 10 cm (120 tracks
      per level of 5-30 cm).

    The box and median tests need at least 3 values in the window.

Smoothing and velocity
    x and y of each segment are smoothed with a Savitzky-Golay filter
    (``scipy.signal.savgol_filter``, window ``sg_window_frames``, polynomial order
    ``sg_poly``, ``mode="interp"``). Inside the segment the filter is a symmetric FIR
    filter, i.e. zero-phase: peaks are not delayed. With the defaults (11 frames, order 2,
    25 fps) the window spans 0.44 s. Within h = (window - 1) / 2 frames of a segment end
    the filter extrapolates a one-sided polynomial fit, which is noisier. Segments shorter
    than the window keep their raw (interpolated) positions.

    The velocity of frame t is a chord of the smoothed positions centred on it:
    v(t) = (p(t + k) - p(t - k)) / (2k / fps) with k = :func:`chord_half_frames` of
    ``speed_chord_s`` (0.4 s: k = 5, 10 frames at 25 fps; at 24 fps the chord is 10 frames,
    0.417 s), i.e. the mean velocity over the 2k / fps seconds around t. The chord follows
    the foot point over about a whole stride, so the planted-foot steps above average out.
    An out-and-back burst of the positions moves it by up to 1.17 times the burst's size
    divided by the chord (0.21 times for a burst of one frame, 0.92 for five, 1.17 for nine
    or ten, where the smoothing overshoots it, 1.084 for a step); the peak of a real sprint
    reads as its 0.4 s mean, slightly below the instantaneous peak. vx, vy and speed are NaN
    where t - k or t + k lies outside the segment and in segments shorter than the window.

Validity and peaks
    ``valid`` is True only where all of these hold:

    - both chord ends t - k and t + k are measured (not interpolated) positions at least h
      frames from both segment ends, where the filter is symmetric (one-sided estimates
      near the ends are markedly noisier, and segment ends often sit next to the very
      artefact that ended the segment);
    - no Savitzky-Golay first derivative (``deriv=1``) within h frames of either chord end
      exceeds ``max_speed_mps`` (12 m/s, above the fastest recorded match sprints of about
      10.5-11 m/s): one bad position spreads over the whole filter window, and the
      derivative is its most sensitive detector;
    - the chord speed itself is at most ``max_speed_mps``;
    - the segment is at least ``min_segment_frames`` frames long.

    A valid speed therefore lies at least h + k frames (10 with the defaults) from both
    segment ends. Invalid samples stay in the table. Each segment gets ``peak_speed``, its
    largest valid speed, and ``peak_frame``, that chord's centre (NaN and -1 when no speed
    is valid). gbf.render's speed labels, the gbf.highlights speed cue and gbf.measure's
    peak-speed all read the valid speeds of this table, so every speed shown is the same
    chord, and the card's peak is the highest of them in its window.

Errors: every function validates its input and raises :class:`KinematicsInputError`
listing every problem. Outputs are deterministic (sorted by track, then frame).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from gbf.ingest import parse_rate
from gbf.shots_grammar import ShotsInputError, shot_ids

KINEMATICS_COLUMNS: tuple[str, ...] = (
    "frame",
    "track_id",
    "segment_id",
    "x",
    "y",
    "vx",
    "vy",
    "speed",
    "valid",
    "interpolated",
)
"""kinematics.parquet: one row per frame of every segment. x, y: the smoothed position
(metres). vx, vy, speed: the chord velocity centred on the frame (m/s): the smoothed
positions ``chord_half_frames`` frames after and before it, their difference divided by
the time between them (0.4 s with the defaults); NaN where the chord leaves the segment.
valid: the speed may be shown (both chord ends measured, away from the segment ends and
from implausible derivatives, the chord at most max_speed_mps, the segment at least
min_segment_frames long; see the module docstring). interpolated: True on frames whose
position was missing or rejected as an outlier and is linearly interpolated."""

SEGMENTS_COLUMNS: tuple[str, ...] = (
    "track_id",
    "segment_id",
    "shot_id",
    "start_frame",
    "end_frame",
    "n_frames",
    "n_interpolated",
    "n_outliers",
    "n_invalid",
    "peak_speed",
    "peak_frame",
)
"""segments.parquet: the contract columns plus shot_id, n_interpolated (frames filled by
interpolation, rejected outliers included), n_outliers (measured positions between the
segment's first and last frame rejected as outliers; rejections before, between or after
segments are not counted anywhere here, so the sum is not the number of rejections, which
:func:`rejected_positions` gives) and n_invalid (frames whose speed is not valid). start_frame
and end_frame are inclusive; peak_speed is the largest valid speed and peak_frame its
frame, -1 when peak_speed is NaN."""

COMMON_STEP_COLUMNS: tuple[str, ...] = ("frame", "n_players", "dx", "dy", "step_m")
"""common_mode_steps: frame t, players measured in t - 1 and t, the median step (dx, dy)
in metres and its length step_m."""

REJECTION_TESTS: tuple[str, ...] = ("box_height", "box_width", "local_median", "excursion")
"""The tests that reject measured positions, in the order they run (module docstring)."""

REJECTION_COLUMNS: tuple[str, ...] = ("frame", "track_id", "test")
"""rejected_positions: one row per rejected measured position and the test that rejected
it (one of :data:`REJECTION_TESTS`), sorted by track, then frame."""

PLAYER_ROLES: tuple[str, ...] = ("player", "goalkeeper")
TEAMS: tuple[str, ...] = ("left", "right")

DEFAULT_MAX_COMMON_STEP_M = 0.5
"""Largest plausible common-mode step between consecutive frames, metres (12.5 m/s for a
whole team); larger steps are calibration jumps."""

DEFAULT_OUTLIER_THRESHOLD_M = 1.5
"""A measured position farther than this from the median of its track's positions within
outlier_radius_frames (default 5) frames is rejected. A player at 9 m/s covers 1.8 m in
5 frames, but the median of a steady run is its centre, so real motion stays far below;
a merged or occluded box typically moves the foot point by several metres."""

DEFAULT_MAX_BOX_HEIGHT_CHANGE = 0.2
"""A measured frame whose box height differs from the median height of its track's boxes
within outlier_radius_frames frames by more than this share of that median is rejected:
the foot point is the bottom edge of the box, so a box that loses the legs or reaches down
to a neighbour's feet moves the ground position in depth, where a pixel of a distant
player is several centimetres."""

DEFAULT_MAX_BOX_WIDTH_CHANGE: float | None = None
"""The same test on the box width; off by default. Arms and stride change the width of a
running player's box by more than 20 % from one frame to the next without moving the foot
point much (the bottom centre moves sideways by half the width change), so the test would
mostly throw away good positions of fast players. Pass a share to turn it on."""

DEFAULT_JUMP_THRESHOLD_M = 1.0
"""Smallest step of a track's trend (the least-squares step between the two sides of a
junction) that counts as a jump, and smallest distance from the local trend that makes a
position an excursion (module docstring). The planted-foot steps of a running player's foot
point stay below it once a trend is fitted over a stride, while an identity switch onto a
player 1.5 m away exceeds it. Smaller switches from one frame to the next are left to the
acceleration test, which on clean synthetic runs of 1-9 m/s keeps them up to 0.46 m (0.72 m
across 3 missing frames), while noise lets larger ones stay: the largest seen to stay in 60
tracks per size (10 seeds x 6 runners) are 0.7 m with 5 cm of noise per axis, 0.95 m with
10 cm, 1.35 m with 20 cm and 1.45 m with 30 cm, and 1.05-1.6 m across 1-3 missing frames.
One that stays adds up to 1.084 s / 0.4 s to a 0.4 s chord speed along the motion (the
filter overshoots a step by 8.4 %), plus what the noise adds: in that sample up to 2.27 m/s
in all with 10 cm and 3.36 m/s with 30 cm, and 2.89 and 3.94 m/s across missing frames.
Another 5-seed sample keeps 1.2 m with 10 cm and 1.45 m with 20 cm across missing frames,
adding up to 2.97 and 3.40 m/s (module docstring; these are sample maxima, not bounds)."""

DEFAULT_JUMP_FACTOR = 5.0
"""A jump or an excursion must also exceed this multiple of the other players' median of
the same statistic, so that camera and calibration motion that moves everybody at once is
not taken for a jump."""

DEFAULT_JUMP_WINDOW_FRAMES = 12
"""Frames on each side of a junction (or of a position) whose positions the jump and
excursion tests fit: 0.48 s at 25 fps, about a stride (two foot strikes) of a sprinting
player, so that the planted-foot steps of the foot point average out."""

DEFAULT_MAX_ACCELERATION_MPS2 = 8.0
"""Largest acceleration (m/s^2) the acceleration test lets the trends on the two sides of a
junction imply (module docstring). Sprinters start and stop at up to about 6-7 m/s^2, and so
does a curve; a box that jumps 0.6 m onto a neighbour from one frame to the next, or slides
1.5 m onto one in 8 frames, needs more. A slower slide (on clean synthetic runs of 1-9 m/s,
more than 5 frames for 1 m, 13 for 2 m, 17 for 3 m, 21 for 5 m or 23 for 8 m) looks like a
change of speed and is kept: the chords over it read the slide's own speed on top of the
player's (at most 2.7 % or 0.15 m/s more), bounded only by max_speed_mps; a 1 m/s jogger
sliding 9.5 m in 24 frames reads 11.04 m/s (module docstring)."""

DEFAULT_SPEED_CHORD_S = 0.4
"""Nominal duration of the chord a speed is measured over; the chord spans
2 * chord_half_frames(fps, DEFAULT_SPEED_CHORD_S) frames (10 at 25 fps)."""

_MIN_JUMP_SIDE = 3
"""Positions the jump test needs on each side of a junction."""

_MIN_EXCURSION_SIDE = 2
"""Positions the excursion test needs on each side of a position."""

_ACCELERATION_NOISE_SE = 3.0
"""Standard errors (from the scatter of the positions about the side trends) by which the
acceleration test first reduces the displacement between the side trends, so that noise and
the planted-foot steps of the foot point are not taken for an acceleration."""

_EXCURSION_ROUNDS = 3
"""Times the excursion test runs again on the positions it kept."""

_EXCURSION_FIT_CUT = 0.5
"""Share of jump_threshold_m beyond which a position of a side window is left out of the
excursion test's second fit (unless it continues the block). Real positions mostly stay
within it (a sprint's positions held for 3 frames lie within 0.43 m of their line at
10.5 m/s, and 10 cm of noise on each axis puts a position 0.5 m off a known line once in
about 270000), and a block that only the second fit finds stands only when the test
rejects the positions that fit left out as well (they are the other dip). Measured with the
defaults at 25 fps: on 6972 synthetic sprints (up to 10.5 m/s, positions held for 1-4
frames), accelerations, stops, turns and reversals with 0-20 cm of noise, the excursion test
rejects the same positions, splits the same segments and reads the same peaks as with the
first fit alone; on 15750 pairs of dips of 1-3 frames and 1.05-2 m, 1-14 frames apart, along
the motion of runners at 1, 6 and 9 m/s or sideways to that of runners at 1 and 6 m/s, it
rejects every dipped position of 3127 of the 3150 clean pairs (the first fit alone 1987),
5467 and 4133 of the 6300 with 5 and 10 cm of noise (3882 and 3451), and no track reads more
than 0.05 m/s faster than with the first fit alone. That last holds on this grid only: of
3528 other noisy pairs (dips of 1.3 and 2 m, 5-30 cm of noise, one seed each), 9 read more
than 0.05 and up to 0.33 m/s faster, all of them pairs this fit rejects whole where the first
fit alone split the track."""

_EXCURSION_REFITS = 2
"""Times the excursion test's second fit is made again on the side positions within
_EXCURSION_FIT_CUT * jump_threshold_m of the previous fit's lines."""

_EXCURSION_CHUNK = 20000
"""Most samples (whole tracks, unless one track has more) whose blocks the excursion test
fits at once. Its side arrays take about 3-4 kB per sample, so a long input is fitted a chunk
of tracks at a time: on 87272 synthetic samples rejected_positions peaked at 331 MB (traced)
in one chunk and at 129 MB in chunks of 20000, and at 161 MB on twice as many samples. The
allowances still compare the blocks of every track, so the result does not depend on the
chunk size."""

_TRACK_COLUMNS = ("frame", "track_id", "pitch_usable", "pitch_x", "pitch_y", "in_pitch")
_BOX_COLUMNS = ("x1", "y1", "x2", "y2")

# Codes of the rejection tests in the per-sample arrays: 0 is kept, i + 1 is
# REJECTION_TESTS[i].
_KEPT = 0
_BOX_HEIGHT, _BOX_WIDTH, _LOCAL_MEDIAN, _EXCURSION = 1, 2, 3, 4


class KinematicsInputError(ValueError):
    """Invalid input to a kinematics function; ``problems`` lists every problem found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


def _int_at_least(name: str, value: Any, minimum: int, problems: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        problems.append(f"{name} must be an integer, got {value!r}")
        return None
    if value < minimum:
        problems.append(f"{name} must be >= {minimum}, got {int(value)}")
        return None
    return int(value)


def _positive_float(name: str, value: Any, problems: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        problems.append(f"{name} must be a number, got {value!r}")
        return None
    if not (math.isfinite(float(value)) and float(value) > 0.0):
        problems.append(f"{name} must be finite and > 0, got {value!r}")
        return None
    return float(value)


def _rate(fps: Any, problems: list[str]) -> float | None:
    """Frames per second as a float from a positive number, a Fraction or '25/1' text."""
    if isinstance(fps, str | Fraction):
        try:
            return float(parse_rate(fps))
        except ValueError as exc:
            problems.append(f"fps: {exc}")
            return None
    return _positive_float("fps", fps, problems)


def chord_half_frames(fps: float | str | Fraction, chord_s: float = DEFAULT_SPEED_CHORD_S) -> int:
    """Half the chord a speed is measured over, in frames: chord_s * fps / 2 rounded half up,
    at least 1. The chord spans twice this many frames, i.e. 2 * k / fps seconds, which can
    differ from ``chord_s`` by up to 1 / fps (10 frames, 0.417 s, at 24 fps for 0.4 s).
    Raises KinematicsInputError for an invalid rate or duration."""
    problems: list[str] = []
    rate = _rate(fps, problems)
    seconds = _positive_float("chord_s", chord_s, problems)
    if problems or rate is None or seconds is None:
        raise KinematicsInputError("chord_half_frames", problems)
    return max(1, math.floor(seconds * rate / 2.0 + 0.5))


def _missing_columns(name: str, frame: Any, needed: Sequence[str], problems: list[str]) -> bool:
    if not isinstance(frame, pd.DataFrame):
        problems.append(f"{name} must be a pandas DataFrame, got {type(frame).__name__}")
        return True
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        problems.append(f"{name} lacks column(s) {', '.join(missing)}")
        return True
    return False


@dataclass(frozen=True)
class _Settings:
    fps: float
    window: int
    poly: int
    max_gap: int
    min_frames: int
    max_speed: float
    max_common_step: float | None
    min_common_players: int
    outlier_radius: int
    outlier_threshold: float | None
    max_height_change: float | None
    max_width_change: float | None
    jump_threshold: float | None
    jump_factor: float
    jump_window: int
    max_acceleration: float | None
    chord_half: int

    @property
    def boxes(self) -> bool:
        """True when a box-size test is on (tracks_df then needs the box columns)."""
        return self.max_height_change is not None or self.max_width_change is not None

    @property
    def acceleration_side(self) -> int:
        """Positions the acceleration test needs on each side of a junction: the window less
        one bridged gap, so that both side velocities rest on most of a stride."""
        return max(_MIN_JUMP_SIDE, self.jump_window - self.max_gap)

    @property
    def excursion_block(self) -> int:
        """Most consecutive positions the excursion test rejects together: a gap they leave
        is still bridged."""
        return max(1, self.max_gap)


def _settings(
    fps: Any,
    sg_window_frames: Any,
    sg_poly: Any,
    max_gap_frames: Any,
    min_segment_frames: Any,
    max_speed_mps: Any,
    max_common_step_m: Any,
    min_common_players: Any,
    outlier_radius_frames: Any,
    outlier_threshold_m: Any,
    max_box_height_change: Any,
    max_box_width_change: Any,
    jump_threshold_m: Any,
    jump_factor: Any,
    jump_window_frames: Any,
    max_acceleration_mps2: Any,
    speed_chord_s: Any,
    problems: list[str],
) -> _Settings | None:
    rate = _rate(fps, problems)
    window = _int_at_least("sg_window_frames", sg_window_frames, 3, problems)
    poly = _int_at_least("sg_poly", sg_poly, 1, problems)
    if window is not None and window % 2 == 0:
        problems.append(f"sg_window_frames must be odd (a centred window), got {window}")
    if window is not None and poly is not None and poly >= window:
        problems.append(f"sg_poly ({poly}) must be smaller than sg_window_frames ({window})")
    max_gap = _int_at_least("max_gap_frames", max_gap_frames, 0, problems)
    min_frames = _int_at_least("min_segment_frames", min_segment_frames, 1, problems)
    speed = _positive_float("max_speed_mps", max_speed_mps, problems)
    common = None
    if max_common_step_m is not None:
        common = _positive_float("max_common_step_m", max_common_step_m, problems)
    players = _int_at_least("min_common_players", min_common_players, 1, problems)
    radius = _int_at_least("outlier_radius_frames", outlier_radius_frames, 1, problems)
    outlier = None
    if outlier_threshold_m is not None:
        outlier = _positive_float("outlier_threshold_m", outlier_threshold_m, problems)
    height = None
    if max_box_height_change is not None:
        height = _positive_float("max_box_height_change", max_box_height_change, problems)
    width = None
    if max_box_width_change is not None:
        width = _positive_float("max_box_width_change", max_box_width_change, problems)
    jump = None
    if jump_threshold_m is not None:
        jump = _positive_float("jump_threshold_m", jump_threshold_m, problems)
    factor = _positive_float("jump_factor", jump_factor, problems)
    jump_window = _int_at_least("jump_window_frames", jump_window_frames, _MIN_JUMP_SIDE, problems)
    acceleration = None
    if max_acceleration_mps2 is not None:
        acceleration = _positive_float("max_acceleration_mps2", max_acceleration_mps2, problems)
    chord = _positive_float("speed_chord_s", speed_chord_s, problems)
    if (
        problems
        or rate is None
        or window is None
        or poly is None
        or max_gap is None
        or min_frames is None
        or speed is None
        or players is None
        or radius is None
        or factor is None
        or jump_window is None
        or chord is None
    ):
        return None
    return _Settings(
        rate,
        window,
        poly,
        max_gap,
        min_frames,
        speed,
        common,
        players,
        radius,
        outlier,
        height,
        width,
        jump,
        factor,
        jump_window,
        acceleration,
        chord_half_frames(rate, chord),
    )


def _measured_rows(tracks_df: pd.DataFrame) -> np.ndarray:
    """Row positions of the measured rows of ``tracks_df``, sorted by track then frame."""
    frames = tracks_df["frame"].to_numpy(dtype=np.int64)
    tracks = tracks_df["track_id"].to_numpy(dtype=np.int64)
    keep = (
        (tracks >= 0)
        & tracks_df["pitch_usable"].to_numpy(dtype=bool)
        & tracks_df["in_pitch"].to_numpy(dtype=bool)
        & np.isfinite(tracks_df["pitch_x"].to_numpy(dtype=np.float64))
        & np.isfinite(tracks_df["pitch_y"].to_numpy(dtype=np.float64))
    )
    rows = np.flatnonzero(keep)
    return rows[np.lexsort((frames[rows], tracks[rows]))]


def _measured(tracks_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(frames, track_ids, x, y) of the measured rows, sorted by track then frame."""
    rows = _measured_rows(tracks_df)
    return (
        tracks_df["frame"].to_numpy(dtype=np.int64)[rows],
        tracks_df["track_id"].to_numpy(dtype=np.int64)[rows],
        tracks_df["pitch_x"].to_numpy(dtype=np.float64)[rows],
        tracks_df["pitch_y"].to_numpy(dtype=np.float64)[rows],
    )


def _box_sizes(tracks_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(height, width) in pixels of the boxes of the measured rows, in :func:`_measured`
    order."""
    rows = _measured_rows(tracks_df)
    x1, y1, x2, y2 = (tracks_df[name].to_numpy(dtype=np.float64)[rows] for name in _BOX_COLUMNS)
    return y2 - y1, x2 - x1


def _common_steps(
    frames: np.ndarray, tracks: np.ndarray, xs: np.ndarray, ys: np.ndarray, min_players: int
) -> pd.DataFrame:
    """common_mode_steps on measured rows sorted by (track, frame) without duplicates."""
    follows = (tracks[1:] == tracks[:-1]) & (frames[1:] == frames[:-1] + 1)
    steps = pd.DataFrame(
        {
            "frame": frames[1:][follows],
            "dx": (xs[1:] - xs[:-1])[follows],
            "dy": (ys[1:] - ys[:-1])[follows],
        }
    )
    grouped = steps.groupby("frame", sort=True).agg(
        n_players=("dx", "size"), dx=("dx", "median"), dy=("dy", "median")
    )
    grouped = grouped[grouped["n_players"] >= min_players]
    result = pd.DataFrame(
        {
            "frame": grouped.index.to_numpy(dtype=np.int64),
            "n_players": grouped["n_players"].to_numpy(dtype=np.int64),
            "dx": grouped["dx"].to_numpy(dtype=np.float64),
            "dy": grouped["dy"].to_numpy(dtype=np.float64),
        }
    )
    result["step_m"] = np.hypot(result["dx"].to_numpy(), result["dy"].to_numpy())
    return result[list(COMMON_STEP_COLUMNS)]


def common_mode_steps(tracks_df: pd.DataFrame, *, min_players: int = 4) -> pd.DataFrame:
    """Median step of all players between consecutive frames (calibration-jump diagnostic).

    Uses the measured rows of ``tracks_df`` (see the module docstring). For every frame t
    such that at least ``min_players`` tracks are measured in both t - 1 and t, returns the
    component-wise median of their steps p(t) - p(t - 1) in metres (dx, dy), its length
    step_m and the number of players, sorted by frame (:data:`COMMON_STEP_COLUMNS`). Raises
    KinematicsInputError for missing columns or duplicate (frame, track_id) rows.
    """
    problems: list[str] = []
    _missing_columns("tracks_df", tracks_df, _TRACK_COLUMNS, problems)
    players = _int_at_least("min_players", min_players, 1, problems)
    if problems or players is None:
        raise KinematicsInputError("common_mode_steps", problems)
    frames, tracks, xs, ys = _measured(tracks_df)
    _duplicates(frames, tracks, problems)
    if problems:
        raise KinematicsInputError("common_mode_steps", problems)
    return _common_steps(frames, tracks, xs, ys, players)


def _duplicates(frames: np.ndarray, tracks: np.ndarray, problems: list[str]) -> None:
    same = (tracks[1:] == tracks[:-1]) & (frames[1:] == frames[:-1])
    if np.any(same):
        repeated = zip(tracks[1:][same].tolist(), frames[1:][same].tolist(), strict=True)
        pairs = sorted(set(repeated))
        shown = ", ".join(f"track {t} frame {f}" for t, f in pairs[:10])
        problems.append(f"tracks_df has {len(pairs)} duplicate (frame, track_id) row(s): {shown}")


@dataclass(frozen=True)
class _Cleaned:
    """The measured samples (sorted by track, frame) and what the outlier tests made of
    them: ``test`` is 0 for a kept sample, else 1 + the index of the rejecting test in
    :data:`REJECTION_TESTS`; ``piece`` numbers the pieces between calibration jumps, shot
    and track changes and trend jumps (a new segment starts where it changes)."""

    frames: np.ndarray
    tracks: np.ndarray
    shots: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    test: np.ndarray
    piece: np.ndarray


def _prepare(
    where: str,
    tracks_df: Any,
    shots_df: Any,
    fps: Any,
    options: dict[str, Any],
) -> tuple[_Settings, _Cleaned]:
    """Validate the inputs of compute_kinematics, rejected_positions or
    kinematics_with_rejections (``where``) and run the outlier tests. Raises
    KinematicsInputError listing every problem."""
    problems: list[str] = []
    settings = _settings(fps, **options, problems=problems)
    boxes = (
        options["max_box_height_change"] is not None or options["max_box_width_change"] is not None
    )
    needed = _TRACK_COLUMNS + (_BOX_COLUMNS if boxes else ())
    tracks_bad = _missing_columns("tracks_df", tracks_df, needed, problems)
    shots_bad = _missing_columns(
        "shots_df", shots_df, ("shot_id", "start_frame", "end_frame"), problems
    )
    if problems or settings is None or tracks_bad or shots_bad:
        raise KinematicsInputError(where, problems)

    frames, tracks, xs, ys = _measured(tracks_df)
    _duplicates(frames, tracks, problems)
    try:
        shots = shot_ids(shots_df, frames)
    except ShotsInputError as exc:
        problems.extend(exc.problems)
        shots = np.zeros(frames.shape, dtype=np.int64)
    uncovered = np.unique(frames[shots < 0])
    if uncovered.size:
        shown = ", ".join(str(f) for f in uncovered[:10].tolist())
        problems.append(f"{uncovered.size} measured frame(s) lie in no shot of shots_df: {shown}")
    if problems:
        raise KinematicsInputError(where, problems)

    jump_frames = np.zeros(0, dtype=np.int64)
    if settings.max_common_step is not None:
        steps = _common_steps(frames, tracks, xs, ys, settings.min_common_players)
        jumps = steps["step_m"].to_numpy() > settings.max_common_step
        jump_frames = steps["frame"].to_numpy(dtype=np.int64)[jumps]
    sizes = _box_sizes(tracks_df) if settings.boxes else None
    test, piece = _rejections(frames, tracks, shots, xs, ys, sizes, jump_frames, settings)
    return settings, _Cleaned(frames, tracks, shots, xs, ys, test, piece)


def compute_kinematics(
    tracks_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    fps: float | str | Fraction,
    *,
    sg_window_frames: int = 11,
    sg_poly: int = 2,
    max_gap_frames: int = 3,
    min_segment_frames: int = 25,
    max_speed_mps: float = 12.0,
    max_common_step_m: float | None = DEFAULT_MAX_COMMON_STEP_M,
    min_common_players: int = 4,
    outlier_radius_frames: int = 5,
    outlier_threshold_m: float | None = DEFAULT_OUTLIER_THRESHOLD_M,
    max_box_height_change: float | None = DEFAULT_MAX_BOX_HEIGHT_CHANGE,
    max_box_width_change: float | None = DEFAULT_MAX_BOX_WIDTH_CHANGE,
    jump_threshold_m: float | None = DEFAULT_JUMP_THRESHOLD_M,
    jump_factor: float = DEFAULT_JUMP_FACTOR,
    jump_window_frames: int = DEFAULT_JUMP_WINDOW_FRAMES,
    max_acceleration_mps2: float | None = DEFAULT_MAX_ACCELERATION_MPS2,
    speed_chord_s: float = DEFAULT_SPEED_CHORD_S,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smoothed positions, chord velocities and segment peaks of every track.

    Args:
        tracks_df: tracks.parquet rows; needs frame, track_id, pitch_usable, pitch_x,
            pitch_y (metres) and in_pitch, and the box x1, y1, x2, y2 (pixels) while a
            box-size test is on.
        shots_df: shots.parquet rows; needs shot_id, start_frame, end_frame (inclusive).
            Every measured frame must lie in a shot.
        fps: frame rate (a positive number, a Fraction or text such as "25/1").
        sg_window_frames: Savitzky-Golay window, an odd number of frames.
        sg_poly: Savitzky-Golay polynomial order, 1 <= sg_poly < sg_window_frames.
        max_gap_frames: longest run of missing frames bridged by linear interpolation.
        min_segment_frames: shortest segment (frames, interpolated ones included) whose
            speeds can be valid.
        max_speed_mps: fastest plausible speed; a faster chord is invalid, and so is every
            chord with an end within half a smoothing window of a faster derivative.
        max_common_step_m: common-mode step (metres per frame) above which a frame
            transition counts as a calibration jump and splits every segment; None turns
            the guard off.
        min_common_players: players needed in both frames to measure a common-mode step,
            and other players needed for the jump test's common-mode allowance.
        outlier_radius_frames: half-width, in frames, of the local medians a measured
            position and its box size are compared with.
        outlier_threshold_m: distance in metres from that local median above which a
            measured position is rejected (then treated as missing); None turns the test
            off.
        max_box_height_change: share of the local median box height by which a box
            height may differ before its position is rejected; None turns the test off.
        max_box_width_change: the same for the box width; None (the default) turns it off.
        jump_threshold_m: smallest trend step (metres) that is a jump, and smallest
            distance from the local trend that makes a position an excursion (see the
            module docstring); None turns the jump, acceleration and excursion tests off.
        jump_factor: a jump, an acceleration or an excursion must also exceed this
            multiple of the other players' median of the same statistic between the same
            frames.
        jump_window_frames: frames on each side whose positions the jump, acceleration and
            excursion tests fit (at least 3).
        max_acceleration_mps2: largest acceleration (m/s^2) the trends on the two sides of
            a junction may imply before the junction counts as a jump (the acceleration
            test, module docstring); None turns that test off.
        speed_chord_s: nominal duration of the chord a velocity is measured over
            (:func:`chord_half_frames` gives the frames).

    Returns (kinematics_df, segments_df) with :data:`KINEMATICS_COLUMNS` and
    :data:`SEGMENTS_COLUMNS`; see the module docstring. segment_id numbers segments
    0, 1, ... ordered by (track_id, start_frame). Raises KinematicsInputError listing every
    problem (arguments, missing columns, duplicate (frame, track_id) rows, measured frames
    outside every shot, overlapping shots).
    """
    options = dict(
        sg_window_frames=sg_window_frames,
        sg_poly=sg_poly,
        max_gap_frames=max_gap_frames,
        min_segment_frames=min_segment_frames,
        max_speed_mps=max_speed_mps,
        max_common_step_m=max_common_step_m,
        min_common_players=min_common_players,
        outlier_radius_frames=outlier_radius_frames,
        outlier_threshold_m=outlier_threshold_m,
        max_box_height_change=max_box_height_change,
        max_box_width_change=max_box_width_change,
        jump_threshold_m=jump_threshold_m,
        jump_factor=jump_factor,
        jump_window_frames=jump_window_frames,
        max_acceleration_mps2=max_acceleration_mps2,
        speed_chord_s=speed_chord_s,
    )
    return _tables(*_prepare("compute_kinematics", tracks_df, shots_df, fps, options))


def _tables(settings: _Settings, cleaned: _Cleaned) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(kinematics_df, segments_df) of the cleaned samples (compute_kinematics)."""
    outliers = cleaned.test != _KEPT
    rejected: dict[int, np.ndarray] = {}
    for track in np.unique(cleaned.tracks[outliers]).tolist():
        rejected[int(track)] = cleaned.frames[outliers & (cleaned.tracks == track)]
    keep = ~outliers
    frames, tracks, shots = cleaned.frames[keep], cleaned.tracks[keep], cleaned.shots[keep]
    xs, ys, piece = cleaned.xs[keep], cleaned.ys[keep], cleaned.piece[keep]

    starts_new = np.ones(frames.size, dtype=bool)
    if frames.size > 1:
        starts_new[1:] = (piece[1:] != piece[:-1]) | (
            frames[1:] - frames[:-1] > settings.max_gap + 1
        )
    bounds = np.flatnonzero(starts_new).tolist() + [frames.size]
    kin_parts: list[dict[str, np.ndarray]] = []
    segment_rows: list[dict[str, Any]] = []
    for segment_id, (first, stop) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        part, summary = _segment(
            frames[first:stop], xs[first:stop], ys[first:stop], settings, segment_id
        )
        track = int(tracks[first])
        part["track_id"] = np.full(part["frame"].size, track, dtype=np.int64)
        kin_parts.append(part)
        dropped = rejected.get(track, np.zeros(0, dtype=np.int64))
        inside = (dropped >= summary["start_frame"]) & (dropped <= summary["end_frame"])
        segment_rows.append(
            {
                "track_id": track,
                "shot_id": int(shots[first]),
                "n_outliers": int(np.count_nonzero(inside)),
                **summary,
            }
        )
    return _kinematics_frame(kin_parts), _segments_frame(segment_rows)


def rejected_positions(
    tracks_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    fps: float | str | Fraction,
    *,
    sg_window_frames: int = 11,
    sg_poly: int = 2,
    max_gap_frames: int = 3,
    min_segment_frames: int = 25,
    max_speed_mps: float = 12.0,
    max_common_step_m: float | None = DEFAULT_MAX_COMMON_STEP_M,
    min_common_players: int = 4,
    outlier_radius_frames: int = 5,
    outlier_threshold_m: float | None = DEFAULT_OUTLIER_THRESHOLD_M,
    max_box_height_change: float | None = DEFAULT_MAX_BOX_HEIGHT_CHANGE,
    max_box_width_change: float | None = DEFAULT_MAX_BOX_WIDTH_CHANGE,
    jump_threshold_m: float | None = DEFAULT_JUMP_THRESHOLD_M,
    jump_factor: float = DEFAULT_JUMP_FACTOR,
    jump_window_frames: int = DEFAULT_JUMP_WINDOW_FRAMES,
    max_acceleration_mps2: float | None = DEFAULT_MAX_ACCELERATION_MPS2,
    speed_chord_s: float = DEFAULT_SPEED_CHORD_S,
) -> pd.DataFrame:
    """Every measured position that :func:`compute_kinematics` with the same arguments
    rejects, with the test that rejected it.

    Returns a DataFrame with :data:`REJECTION_COLUMNS` (frame and track_id int64, test one
    of :data:`REJECTION_TESTS`), sorted by track, then frame, so that
    ``value_counts("test")`` is an honest count by test: unlike the n_outliers column of
    segments.parquet it also holds rejections that lie outside every segment. Raises
    KinematicsInputError like compute_kinematics.
    """
    options = dict(
        sg_window_frames=sg_window_frames,
        sg_poly=sg_poly,
        max_gap_frames=max_gap_frames,
        min_segment_frames=min_segment_frames,
        max_speed_mps=max_speed_mps,
        max_common_step_m=max_common_step_m,
        min_common_players=min_common_players,
        outlier_radius_frames=outlier_radius_frames,
        outlier_threshold_m=outlier_threshold_m,
        max_box_height_change=max_box_height_change,
        max_box_width_change=max_box_width_change,
        jump_threshold_m=jump_threshold_m,
        jump_factor=jump_factor,
        jump_window_frames=jump_window_frames,
        max_acceleration_mps2=max_acceleration_mps2,
        speed_chord_s=speed_chord_s,
    )
    _, cleaned = _prepare("rejected_positions", tracks_df, shots_df, fps, options)
    return _rejection_table(cleaned)


def kinematics_with_rejections(
    tracks_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    fps: float | str | Fraction,
    *,
    sg_window_frames: int = 11,
    sg_poly: int = 2,
    max_gap_frames: int = 3,
    min_segment_frames: int = 25,
    max_speed_mps: float = 12.0,
    max_common_step_m: float | None = DEFAULT_MAX_COMMON_STEP_M,
    min_common_players: int = 4,
    outlier_radius_frames: int = 5,
    outlier_threshold_m: float | None = DEFAULT_OUTLIER_THRESHOLD_M,
    max_box_height_change: float | None = DEFAULT_MAX_BOX_HEIGHT_CHANGE,
    max_box_width_change: float | None = DEFAULT_MAX_BOX_WIDTH_CHANGE,
    jump_threshold_m: float | None = DEFAULT_JUMP_THRESHOLD_M,
    jump_factor: float = DEFAULT_JUMP_FACTOR,
    jump_window_frames: int = DEFAULT_JUMP_WINDOW_FRAMES,
    max_acceleration_mps2: float | None = DEFAULT_MAX_ACCELERATION_MPS2,
    speed_chord_s: float = DEFAULT_SPEED_CHORD_S,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(kinematics_df, segments_df, rejected_df): what :func:`compute_kinematics` and
    :func:`rejected_positions` return for the same arguments, from one run of the outlier
    tests (they take most of the time). Raises KinematicsInputError like
    compute_kinematics."""
    options = dict(
        sg_window_frames=sg_window_frames,
        sg_poly=sg_poly,
        max_gap_frames=max_gap_frames,
        min_segment_frames=min_segment_frames,
        max_speed_mps=max_speed_mps,
        max_common_step_m=max_common_step_m,
        min_common_players=min_common_players,
        outlier_radius_frames=outlier_radius_frames,
        outlier_threshold_m=outlier_threshold_m,
        max_box_height_change=max_box_height_change,
        max_box_width_change=max_box_width_change,
        jump_threshold_m=jump_threshold_m,
        jump_factor=jump_factor,
        jump_window_frames=jump_window_frames,
        max_acceleration_mps2=max_acceleration_mps2,
        speed_chord_s=speed_chord_s,
    )
    settings, cleaned = _prepare("kinematics_with_rejections", tracks_df, shots_df, fps, options)
    return *_tables(settings, cleaned), _rejection_table(cleaned)


def _rejection_table(cleaned: _Cleaned) -> pd.DataFrame:
    """rejected_positions' table of the cleaned samples."""
    rows = np.flatnonzero(cleaned.test != _KEPT)
    names = np.asarray(REJECTION_TESTS, dtype=object)
    return pd.DataFrame(
        {
            "frame": cleaned.frames[rows].astype(np.int64),
            "track_id": cleaned.tracks[rows].astype(np.int64),
            "test": pd.Series(names[cleaned.test[rows] - 1], dtype=object),
        },
        columns=list(REJECTION_COLUMNS),
    )


def _piece_starts(
    frames: np.ndarray,
    tracks: np.ndarray,
    shots: np.ndarray,
    jump_frames: np.ndarray,
    max_gap: int | None,
) -> np.ndarray:
    """True where a sample (sorted by track, frame) starts a new piece: a new track, a new
    shot, a calibration jump frame j with previous < j <= current, or (when ``max_gap`` is
    given) more than ``max_gap`` missing frames since the previous sample."""
    starts = np.ones(frames.size, dtype=bool)
    if frames.size < 2:
        return starts
    jumped = np.searchsorted(jump_frames, frames[1:], side="right") > np.searchsorted(
        jump_frames, frames[:-1], side="right"
    )
    new = (tracks[1:] != tracks[:-1]) | (shots[1:] != shots[:-1]) | jumped
    if max_gap is not None:
        new |= frames[1:] - frames[:-1] > max_gap + 1
    starts[1:] = new
    return starts


def _window_medians(
    frames: np.ndarray,
    starts: np.ndarray,
    radius: int,
    values: Sequence[np.ndarray],
    usable: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """For every sample (sorted by piece, then frame; pieces begin where ``starts`` is True)
    the median of each of ``values`` over the usable samples of its piece within ``radius``
    frames (itself included), NaN where there is none, and how many there are."""
    if not frames.size:
        return [np.zeros(0) for _ in values], np.zeros(0, dtype=np.int64)
    # one increasing key, with a gap of more than 2 * radius frames between pieces
    stride = int(frames.max()) - int(frames.min()) + 2 * radius + 2
    key = (np.cumsum(starts) - 1) * stride + (frames - frames.min())
    lo = np.searchsorted(key, key - radius, side="left")
    hi = np.searchsorted(key, key + radius, side="right")
    index = lo[:, None] + np.arange(2 * radius + 1)
    inside = index < hi[:, None]
    index = np.minimum(index, frames.size - 1)
    inside &= usable[index]
    count = inside.sum(axis=1)
    rows = np.arange(frames.size)
    lower, upper = np.maximum(count - 1, 0) // 2, count // 2
    medians = []
    for value in values:
        ranked = np.sort(np.where(inside, value[index], np.nan), axis=1)  # NaN last
        middle = 0.5 * (ranked[rows, lower] + ranked[rows, upper])
        medians.append(np.where(count > 0, middle, np.nan))
    return medians, count


def _position_outliers(
    frames: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    starts: np.ndarray,
    radius: int,
    threshold: float,
) -> np.ndarray:
    """True for samples farther than ``threshold`` metres from the component-wise median of
    their piece's samples within ``radius`` frames (themselves included; at least 3)."""
    usable = np.ones(frames.size, dtype=bool)
    (centre_x, centre_y), count = _window_medians(frames, starts, radius, (xs, ys), usable)
    return (count >= 3) & (np.hypot(xs - centre_x, ys - centre_y) > threshold)


def _size_outliers(
    frames: np.ndarray, sizes: np.ndarray, starts: np.ndarray, radius: int, limit: float
) -> np.ndarray:
    """True for samples whose box size (height or width, pixels) differs from the median
    size of their piece's samples within ``radius`` frames (themselves included; at least 3
    finite positive sizes) by more than ``limit`` times that median."""
    usable = np.isfinite(sizes) & (sizes > 0)
    (centre,), count = _window_medians(frames, starts, radius, (sizes,), usable)
    return usable & (count >= 3) & (np.abs(sizes - centre) > limit * centre)


class _PieceSums:
    """Cumulative sums of one piece's positions on its frame grid (frames first..last, one
    cell per frame, empty where a frame is missing), for the least-squares fits of the jump
    and acceleration tests over arbitrary frame ranges. Positions are summed relative to
    ``origin`` (the piece's first position), which keeps the sums of squares precise."""

    def __init__(self, frames: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
        self.first = int(frames[0])
        self.origin = np.array([xs[0], ys[0]], dtype=np.float64)
        size = int(frames[-1]) - self.first + 1
        cells = frames - self.first
        count = np.zeros(size)
        count[cells] = 1.0
        x = np.zeros(size)
        x[cells] = xs - self.origin[0]
        y = np.zeros(size)
        y[cells] = ys - self.origin[1]
        t = np.arange(size, dtype=np.float64)
        self.size = size
        self.sums = {
            name: np.concatenate([[0.0], np.cumsum(values)])
            for name, values in (
                ("n", count),
                ("t", count * t),
                ("tt", count * t * t),
                ("x", x),
                ("tx", t * x),
                ("xx", x * x),
                ("y", y),
                ("ty", t * y),
                ("yy", y * y),
            )
        }

    def over(self, lo: np.ndarray, hi: np.ndarray) -> dict[str, np.ndarray]:
        """Sums over the grid cells lo..hi (inclusive, clipped to the piece)."""
        start = np.clip(lo, 0, self.size)
        stop = np.maximum(np.clip(hi + 1, 0, self.size), start)
        return {name: values[stop] - values[start] for name, values in self.sums.items()}


def _centred(sums: dict[str, np.ndarray], centre: np.ndarray) -> dict[str, np.ndarray]:
    """The sums with time measured from ``centre`` (cells) instead of the grid origin."""
    n, t = sums["n"], sums["t"]
    return {
        "n": n,
        "t": t - centre * n,
        "tt": sums["tt"] - 2.0 * centre * t + centre * centre * n,
        "x": sums["x"],
        "tx": sums["tx"] - centre * sums["x"],
        "xx": sums["xx"],
        "y": sums["y"],
        "ty": sums["ty"] - centre * sums["y"],
        "yy": sums["yy"],
    }


def _subset(sums: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[rows] for name, values in sums.items()}


def _two_sided(
    before: dict[str, np.ndarray], after: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares fit of the positions of two sides (sums with centred time, at least 2
    positions on each side) with one shared velocity and a step between the sides. Returns
    (level, velocity, step), each of shape (N, 2): the before side's line passes through
    ``level`` (metres from the piece's origin) at time 0 and the after side's through
    ``level + step``; the velocity is in metres per frame."""
    n = before["n"] + after["n"]
    t = before["t"] + after["t"]
    tt = before["tt"] + after["tt"]
    normal = np.stack(
        [
            np.stack([n, t, after["n"]], axis=-1),
            np.stack([t, tt, after["t"]], axis=-1),
            np.stack([after["n"], after["t"], after["n"]], axis=-1),
        ],
        axis=-2,
    )
    rhs = np.stack(
        [
            np.stack([before["x"] + after["x"], before["tx"] + after["tx"], after["x"]], axis=-1),
            np.stack([before["y"] + after["y"], before["ty"] + after["ty"], after["y"]], axis=-1),
        ],
        axis=-1,
    )
    solution = np.linalg.solve(normal, rhs)  # (N, 3 parameters, 2 axes)
    return solution[:, 0, :], solution[:, 1, :], solution[:, 2, :]


def _side_trends(
    sums: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A line of its own through the positions of each side (sums over at least 3 positions
    at distinct frames): (n, centroid time in frames, mean position (N, 2), velocity in
    metres per frame (N, 2), sum of the squared time offsets, residual sum of squares of
    both axes)."""
    n = sums["n"]
    t_mean = sums["t"] / n
    spread_t = sums["tt"] - sums["t"] * t_mean
    mean = np.stack([sums["x"], sums["y"]], axis=-1) / n[:, None]
    cross = np.stack([sums["tx"] - t_mean * sums["x"], sums["ty"] - t_mean * sums["y"]], axis=-1)
    velocity = cross / spread_t[:, None]
    spread = np.stack(
        [sums["xx"] - sums["x"] * mean[:, 0], sums["yy"] - sums["y"] * mean[:, 1]], -1
    )
    residual = np.maximum(np.sum(spread - cross * velocity, axis=-1), 0.0)
    return n, t_mean, mean, velocity, spread_t, residual


def _trend_steps(piece: _PieceSums, cells: np.ndarray, window: int) -> np.ndarray:
    """|s| of the jump test at the junctions between consecutive samples of one piece, at
    grid cells cells[:-1] (a) and cells[1:] (b): the step of a two-sided fit of the
    positions in the ``window`` cells ending at a and the ``window`` cells starting at b.
    NaN where a side has fewer than _MIN_JUMP_SIDE positions."""
    a, b = cells[:-1], cells[1:]
    centre = 0.5 * (a + b)
    before = _centred(piece.over(a - window + 1, a), centre)
    after = _centred(piece.over(b, b + window - 1), centre)
    result = np.full(a.size, np.nan)
    enough = (before["n"] >= _MIN_JUMP_SIDE) & (after["n"] >= _MIN_JUMP_SIDE)
    if np.any(enough):
        _, _, step = _two_sided(_subset(before, enough), _subset(after, enough))
        result[enough] = np.hypot(step[:, 0], step[:, 1])
    return result


def _trend_accelerations(
    piece: _PieceSums, cells: np.ndarray, window: int, min_side: int, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """The acceleration test's statistic (m/s^2) at the junctions between consecutive samples
    of one piece, at grid cells cells[:-1] (a) and cells[1:] (b). For each guard g of
    0, 2, ... up to window / 2 frames, the positions in the ``window`` cells ending g cells
    before a and those in the ``window`` cells starting g cells after b each get a line of
    their own, with velocities v0 and v1 at their centroids T apart. The displacement
    between the lines that their velocities do not explain, e = |p1 - p0 - (v0 + v1) T / 2|,
    is reduced by |v1 - v0| T / 2, the most that one sudden change of velocity anywhere
    between the centroids explains (a planted foot stops the foot point at once), and by
    _ACCELERATION_NOISE_SE standard errors of e (from the scatter of the positions about
    their lines). Moving off the trend and back onto it by what is left, e', within T takes
    an acceleration of at least 4 e' / T^2; the statistic is its largest value over the
    guards, so that a slide onto a neighbour over up to ``window`` frames falls between the
    sides of one of them. Returns (statistic, the guard of its largest value): NaN and -1
    where no guard has ``min_side`` positions on each side."""
    a, b = cells[:-1], cells[1:]
    result = np.full(a.size, np.nan)
    chosen = np.full(a.size, -1, dtype=np.int64)
    for guard in range(0, window // 2 + 1, 2):
        before = piece.over(a - guard - window + 1, a - guard)
        after = piece.over(b + guard, b + guard + window - 1)
        enough = (before["n"] >= min_side) & (after["n"] >= min_side)
        if not np.any(enough):
            continue
        n0, t0, p0, v0, spread0, residual0 = _side_trends(_subset(before, enough))
        n1, t1, p1, v1, spread1, residual1 = _side_trends(_subset(after, enough))
        gap = t1 - t0  # frames between the side centroids
        excess = (p1 - p0) - 0.5 * (v0 + v1) * gap[:, None]
        change = v1 - v0
        sigma = np.sqrt((residual0 + residual1) / (2.0 * (n0 + n1 - 4.0)))
        excess_se = sigma * np.sqrt(
            1.0 / n0 + 1.0 / n1 + 0.25 * gap**2 * (1.0 / spread0 + 1.0 / spread1)
        )
        left = (
            np.hypot(excess[:, 0], excess[:, 1])
            - 0.5 * gap * np.hypot(change[:, 0], change[:, 1])
            - _ACCELERATION_NOISE_SE * excess_se
        )
        need = 4.0 * np.maximum(left, 0.0) / gap**2 * fps * fps
        rows = np.flatnonzero(enough)
        larger = ~(need <= result[rows])  # also where no smaller guard had enough sides
        chosen[rows[larger]] = guard
        result[rows[larger]] = need[larger]
    return result, chosen


def _side_sums(times: np.ndarray, points: np.ndarray, use: np.ndarray) -> dict[str, np.ndarray]:
    """The sums _two_sided needs over the side positions marked in ``use`` (shape (N, W)):
    ``times`` (N, W) in frames from the time origin, ``points`` (N, W, 2) in metres."""
    weight = use.astype(np.float64)
    timed = weight * times
    return {
        "n": weight.sum(axis=1),
        "t": timed.sum(axis=1),
        "tt": (timed * times).sum(axis=1),
        "x": (weight * points[..., 0]).sum(axis=1),
        "tx": (timed * points[..., 0]).sum(axis=1),
        "y": (weight * points[..., 1]).sum(axis=1),
        "ty": (timed * points[..., 1]).sum(axis=1),
    }


def _block_distances(
    frames: np.ndarray,
    pieces: np.ndarray,
    points: np.ndarray,
    window: int,
    length: int,
    cut: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """For every block of ``length`` consecutive samples of one piece (samples sorted by
    piece, then frame; ``pieces`` numbers them, ``points`` (n, 2) holds their positions),
    indexed by its first sample: the distance of each of its positions from the before and
    from the after line of a two-sided fit of the piece's positions within ``window`` frames
    before and after the block, two arrays of shape (blocks, length), for two fits. The
    second is made again (_EXCURSION_REFITS times) without the side positions farther than
    ``cut`` metres from the previous fit's line of their side, except the run of them next
    to the block that lie off on the block's side, which continues it (where each side
    keeps at least _MIN_EXCURSION_SIDE positions, else the previous fit stays), so that
    another short dip near the block does not pull the trends towards itself. Rows are NaN
    where the block spans two pieces or a side has fewer than _MIN_EXCURSION_SIDE positions.
    Returns ([first fit's (to_before, to_after), second fit's], side, left_out): the side
    samples of every block, shape (blocks, 2 * window), and which of them the second fit
    left out and are still farther than ``cut`` from its lines."""
    n = frames.size
    count = max(n - length + 1, 0)
    empty = [(np.full((count, length), np.nan), np.full((count, length), np.nan)) for _ in "ab"]
    left_out = np.zeros((count, 2 * window), dtype=bool)
    if count == 0:
        return empty, np.zeros((0, 2 * window), dtype=np.int64), left_out
    first, last, piece = frames[:count], frames[length - 1 :], pieces[:count]
    centre = 0.5 * (first + last)
    # side samples by index, nearest to the block first
    offsets = np.arange(window)
    before = np.arange(count)[:, None] - 1 - offsets
    after = np.arange(length, count + length)[:, None] + offsets
    before_in, after_in = before >= 0, after < n
    before, after = np.clip(before, 0, n - 1), np.clip(after, 0, n - 1)
    before_in &= (pieces[before] == piece[:, None]) & (frames[before] >= (first - window)[:, None])
    after_in &= (pieces[after] == piece[:, None]) & (frames[after] <= (last + window)[:, None])
    side = np.concatenate([before, after], axis=1)
    starts = np.flatnonzero(
        (pieces[length - 1 :] == piece)
        & (before_in.sum(axis=1) >= _MIN_EXCURSION_SIDE)
        & (after_in.sum(axis=1) >= _MIN_EXCURSION_SIDE)
    )
    if not starts.size:
        return empty, side, left_out
    t0, t1 = (frames[side[starts]] - centre[starts, None] for side in (before, after))
    p0, p1 = points[before[starts]], points[after[starts]]
    in0, in1 = before_in[starts], after_in[starts]
    use0, use1 = in0.copy(), in1.copy()
    level, velocity, step = _two_sided(_side_sums(t0, p0, use0), _side_sums(t1, p1, use1))
    times = frames[starts[:, None] + np.arange(length)] - centre[starts, None]
    members = points[starts[:, None] + np.arange(length)]

    def distances(fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        level, velocity, step = (value[:, None, :] for value in fit)
        off_before = members - level - velocity * times[..., None]
        off_after = off_before - step
        to_before, to_after = np.full((count, length), np.nan), np.full((count, length), np.nan)
        to_before[starts] = np.hypot(off_before[..., 0], off_before[..., 1])
        to_after[starts] = np.hypot(off_after[..., 0], off_after[..., 1])
        result.append((to_before, to_after))

    result: list[tuple[np.ndarray, np.ndarray]] = []
    distances((level, velocity, step))
    for _refit in range(_EXCURSION_REFITS):
        off0 = p0 - level[:, None, :] - velocity[:, None, :] * t0[..., None]
        off1 = p1 - (level + step)[:, None, :] - velocity[:, None, :] * t1[..., None]
        # Far positions are left out, but not the run of them next to the block off their
        # line on the block's side, which continue it (a longer excursion stays whole for
        # the jump tests); a position between the block and another dip lies off the line
        # on the other side while the fit is pulled towards that dip.
        own = members - level[:, None, :] - velocity[:, None, :] * times[..., None]
        towards = (own - 0.5 * step[:, None, :]).mean(axis=1)[:, None, :]
        far0 = in0 & (np.hypot(off0[..., 0], off0[..., 1]) > cut)
        far1 = in1 & (np.hypot(off1[..., 0], off1[..., 1]) > cut)
        run0 = np.cumprod(far0 & (np.sum(off0 * towards, axis=-1) > 0.0), axis=1, dtype=bool)
        run1 = np.cumprod(far1 & (np.sum(off1 * towards, axis=-1) > 0.0), axis=1, dtype=bool)
        near0 = in0 & ~(far0 & ~run0)
        near1 = in1 & ~(far1 & ~run1)
        refit = np.flatnonzero(
            (near0.sum(axis=1) >= _MIN_EXCURSION_SIDE)
            & (near1.sum(axis=1) >= _MIN_EXCURSION_SIDE)
            & (np.any(near0 != use0, axis=1) | np.any(near1 != use1, axis=1))
        )
        if not refit.size:
            break
        use0[refit], use1[refit] = near0[refit], near1[refit]
        level[refit], velocity[refit], step[refit] = _two_sided(
            _side_sums(t0[refit], p0[refit], use0[refit]),
            _side_sums(t1[refit], p1[refit], use1[refit]),
        )
    distances((level, velocity, step))
    # left out: still far from the final lines (a position between two dips can look far
    # only while the fit is pulled towards them)
    off0 = p0 - level[:, None, :] - velocity[:, None, :] * t0[..., None]
    off1 = p1 - (level + step)[:, None, :] - velocity[:, None, :] * t1[..., None]
    left_out[starts] = np.concatenate(
        [
            in0 & ~use0 & (np.hypot(off0[..., 0], off0[..., 1]) > cut),
            in1 & ~use1 & (np.hypot(off1[..., 0], off1[..., 1]) > cut),
        ],
        axis=1,
    )
    return result, side, left_out


def _allowances(
    values: np.ndarray,
    groups: np.ndarray,
    tracks: np.ndarray,
    settings: _Settings,
    threshold: float,
) -> np.ndarray:
    """max(threshold, jump_factor * the median of the other tracks' values in the same
    group) for every (finite) value, the median used when at least min_common_players other
    tracks have one; ``groups`` holds an integer key per value (a span of frames, or a
    junction)."""
    allowed = np.full(values.size, threshold)
    if not values.size:
        return allowed
    # Sorted by group, then value: the others of the value of rank r in its group are the
    # group's sorted values without the r-th, so their k-th is the group's k-th below r and
    # its (k + 1)-th from r on.
    order = np.lexsort((values, groups))
    keys = groups[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    sizes = np.diff(np.r_[starts, order.size])
    first = np.repeat(starts, sizes)
    usable = np.flatnonzero(np.repeat(sizes, sizes) - 1 >= settings.min_common_players)
    first = first[usable]
    rank = usable - first
    others = np.repeat(sizes, sizes)[usable] - 1
    ranked = values[order]

    def other(k: np.ndarray) -> np.ndarray:
        return ranked[first + k + (k >= rank)]

    median = 0.5 * (other((others - 1) // 2) + other(others // 2))
    allowed[order[usable]] = np.maximum(threshold, settings.jump_factor * median)
    return allowed


def _excursion_blocks(
    frames: np.ndarray, pieces: np.ndarray, points: np.ndarray, settings: _Settings
) -> list[tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]]:
    """The candidate excursions among samples sorted by piece, then frame (``pieces``
    numbers them, ``points`` (n, 2) holds their positions), for each block length from 1 to
    excursion_block: (the blocks' members, sample indices of shape (blocks, length); for
    each of the two fits of _block_distances, the smallest distance of a block's positions
    from the nearer of the trends around it and the larger distance of its two neighbours
    in its piece from the trend of their own side, fitted with the block left out as well,
    NaN where that fit fails; the side samples of the block's fit and of its neighbours'
    fits, shape (blocks, 6 * jump_window), and which of them their second fits left out).
    A position next to a jump lies on the trend of its own side. Blocks spanning more than
    excursion_block frames, or whose own fit fails, are left out."""
    window, block = settings.jump_window, settings.excursion_block
    cut = _EXCURSION_FIT_CUT * float(settings.jump_threshold or 0.0)
    distances = {
        length: _block_distances(frames, pieces, points, window, length, cut)
        for length in range(1, block + 2)
    }
    found = []
    for length in range(1, block + 1):
        count = frames.size - length + 1
        if count <= 0:
            break
        (fits, side, left_out), (wider_fits, wider_side, wider_left) = (
            distances[length],
            distances[length + 1],
        )
        tests = []
        for (own_before, own_after), (wider_before, wider_after) in zip(
            fits, wider_fits, strict=True
        ):
            nearest = np.min(np.minimum(own_before, own_after), axis=1)
            # The neighbour before the block starting at i is the first member of the wider
            # block starting at i - 1, the neighbour after it the last member of the wider
            # block starting at i (NaN where that block spans two pieces).
            before = np.full(count, np.nan)
            after = np.full(count, np.nan)
            before[1 : wider_before.shape[0] + 1] = wider_before[:, 0]
            after[: wider_after.shape[0]] = wider_after[:, length]
            tests.append((nearest, np.fmax(before, after)))
        index = np.arange(count)
        keep = np.isfinite(tests[0][0]) & (frames[index + length - 1] - frames[index] < block)
        starts = index[keep]
        members = np.stack([starts + k for k in range(length)], axis=-1)
        # the neighbours' wider blocks start at i - 1 and i (none where out of range)
        wider_count = wider_side.shape[0]
        lower = np.clip(starts - 1, 0, max(wider_count - 1, 0))
        upper = np.clip(starts, 0, max(wider_count - 1, 0))
        has_lower = (starts >= 1) & (starts - 1 < wider_count)
        has_upper = starts < wider_count
        if wider_count:
            sides = np.concatenate([side[starts], wider_side[lower], wider_side[upper]], axis=1)
            outs = np.concatenate(
                [
                    left_out[starts],
                    wider_left[lower] & has_lower[:, None],
                    wider_left[upper] & has_upper[:, None],
                ],
                axis=1,
            )
        else:
            sides, outs = side[starts], left_out[starts]
        found.append(
            (
                members,
                [(nearest[keep], neighbours[keep]) for nearest, neighbours in tests],
                sides,
                outs,
            )
        )
    return found


def _track_chunks(tracks: np.ndarray, size: int) -> list[tuple[int, int]]:
    """Ranges [lo, hi) of samples sorted by track that hold whole tracks, each at most
    ``size`` samples unless one track has more."""
    if not tracks.size:
        return []
    chunks: list[tuple[int, int]] = []
    lo = last = 0
    for end in (np.flatnonzero(np.r_[tracks[1:] != tracks[:-1], True]) + 1).tolist():
        if end - lo > size and last > lo:
            chunks.append((lo, last))
            lo = last
        last = end
    if last > lo:
        chunks.append((lo, last))
    return chunks


def _chunked_blocks(
    rows: np.ndarray,
    frames: np.ndarray,
    tracks: np.ndarray,
    pieces: np.ndarray,
    points: np.ndarray,
    settings: _Settings,
) -> list[tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], dict[int, np.ndarray]]]:
    """_excursion_blocks of the samples ``rows`` (sorted by track, then frame), fitted a
    chunk of whole tracks at a time (_EXCURSION_CHUNK) and joined in sample order, for each
    block length: (the blocks' members as samples, the two tests, and the samples the second
    fits left out by block index, kept only for the blocks those fits put farther than
    jump_threshold_m from their trends: no other block can be found by them alone)."""
    threshold = float(settings.jump_threshold or 0.0)
    parts: dict[int, list[tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], dict]]] = {}
    for lo, hi in _track_chunks(tracks[rows], _EXCURSION_CHUNK):
        chunk = rows[lo:hi]
        blocks = _excursion_blocks(frames[chunk], pieces[chunk], points[chunk], settings)
        for length, (members, tests, sides, outs) in enumerate(blocks, start=1):
            candidates = np.flatnonzero(tests[1][0] > threshold).tolist()
            left = {index: chunk[sides[index][outs[index]]] for index in candidates}
            parts.setdefault(length, []).append((chunk[members], tests, left))
    joined = []
    for length in sorted(parts):
        members = np.concatenate([part[0] for part in parts[length]])
        tests = [
            (
                np.concatenate([part[1][fit][0] for part in parts[length]]),
                np.concatenate([part[1][fit][1] for part in parts[length]]),
            )
            for fit in (0, 1)
        ]
        left: dict[int, np.ndarray] = {}
        offset = 0
        for part_members, _, part_left in parts[length]:
            left.update({offset + index: samples for index, samples in part_left.items()})
            offset += part_members.shape[0]
        joined.append((members, tests, left))
    return joined


def _excursions(
    frames: np.ndarray,
    tracks: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    starts: np.ndarray,
    settings: _Settings,
) -> np.ndarray:
    """The excursion test (module docstring) on samples sorted by (track, frame) whose
    pieces begin where ``starts`` is True: every block of 1 to max_gap_frames consecutive
    positions (spanning at most max_gap_frames frames) is tested against the trends around
    it, so that the positions of a short dip never hide each other, and again against
    trends fitted without the side positions far off them, so that a second dip near the
    block does not hide it either. A block is an excursion when, for either fit, all its
    positions are off the trends while its two neighbours are on theirs; one that only the
    second fits find stands when the positions they left out are rejected too (they are the
    other dip), so that no part of a longer excursion is rejected alone: that is left to
    the jump tests, which split the whole of it off. Every position of an excursion is
    rejected, and the test runs again on what it keeps (at most _EXCURSION_ROUNDS times), so
    that an excursion hidden at first by another one in its windows is found too. The
    blocks are fitted a chunk of tracks at a time (_EXCURSION_CHUNK). Returns the samples to
    reject."""
    n = frames.size
    rejected = np.zeros(n, dtype=bool)
    threshold = float(settings.jump_threshold or 0.0)
    pieces = np.cumsum(starts) - 1
    points = np.stack([xs, ys], axis=-1)
    scale = int(frames.max()) + 2 if n else 1
    for _round in range(_EXCURSION_ROUNDS):
        rows = np.flatnonzero(~rejected)
        found = np.zeros(n, dtype=bool)
        # (members, samples their second fits left out) of the blocks only those fits find
        second: list[tuple[np.ndarray, np.ndarray]] = []
        for members, tests, left_out in _chunked_blocks(
            rows, frames, tracks, pieces, points, settings
        ):
            if not members.size:
                continue
            span = frames[members[:, 0]] * scale + frames[members[:, -1]]
            hits = []
            for nearest, neighbours in tests:
                allowed = _allowances(nearest, span, tracks[members[:, 0]], settings, threshold)
                isolated = ~(neighbours > allowed)  # a neighbour without a fit is on trend
                hits.append((nearest > allowed) & isolated)
            found[members[hits[0]].ravel()] = True
            for index in np.flatnonzero(hits[1] & ~hits[0]).tolist():
                second.append((members[index], left_out[index]))
        while second:
            rejecting = found.copy()
            for own, _ in second:
                rejecting[own] = True
            consistent = [(own, out) for own, out in second if rejecting[out].all()]
            if len(consistent) == len(second):
                break
            second = consistent
        for own, _ in second:
            found[own] = True
        if not np.any(found):
            break
        rejected |= found
    return rejected


def _jumps(
    frames: np.ndarray,
    tracks: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    starts: np.ndarray,
    settings: _Settings,
) -> np.ndarray:
    """The jump and acceleration tests (module docstring) on samples sorted by (track,
    frame) whose pieces begin where ``starts`` is True. A jump raises the statistics of the
    junctions around it too, so the pieces are split one junction at a time: at the junction
    whose trend step or acceleration exceeds its allowance by the largest factor, then again
    inside each part with the statistics fitted on that part alone, until no junction
    exceeds its allowances. A junction that the acceleration test finds with a guard of g
    frames is cut on both sides of the frames its fits left out, which form a part of their
    own (too short for a valid speed). The allowances come from the first fit of every
    track. Returns the samples that a jump separates from the sample before them."""
    n = frames.size
    breaks = np.zeros(n, dtype=bool)
    threshold, window = float(settings.jump_threshold or 0.0), settings.jump_window
    limit = settings.max_acceleration

    def statistics_of(first: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        piece = _PieceSums(frames[first:stop], xs[first:stop], ys[first:stop])
        cells = frames[first:stop] - piece.first
        steps = _trend_steps(piece, cells, window)
        if limit is None:
            return steps, np.full(steps.size, np.nan), np.full(steps.size, -1, dtype=np.int64)
        side = settings.acceleration_side
        return steps, *_trend_accelerations(piece, cells, window, side, settings.fps)

    # steps[i], needs[i], guards[i]: the statistics of the junction between samples i - 1
    # and i
    steps = np.full(n, np.nan)
    needs = np.full(n, np.nan)
    guards = np.full(n, -1, dtype=np.int64)
    bounds = np.flatnonzero(starts).tolist() + [n]
    pieces = [(first, stop) for first, stop in zip(bounds[:-1], bounds[1:], strict=True)]
    pieces = [(first, stop) for first, stop in pieces if stop - first > 1]
    for first, stop in pieces:
        found = statistics_of(first, stop)
        steps[first + 1 : stop], needs[first + 1 : stop], guards[first + 1 : stop] = found
    junction = np.zeros(n, dtype=np.int64)
    if n > 1:
        junction[1:] = frames[:-1] * (int(frames.max()) + 2) + frames[1:]
    allowed_step = np.full(n, np.inf)
    allowed_need = np.full(n, np.inf)
    for values, allowed, floor in ((steps, allowed_step, threshold), (needs, allowed_need, limit)):
        tested = np.flatnonzero(np.isfinite(values))
        if tested.size and floor is not None:
            allowed[tested] = _allowances(
                values[tested], junction[tested], tracks[tested], settings, floor
            )

    def ratios(
        first: int, stop: int, step: np.ndarray, need: np.ndarray, guard: np.ndarray
    ) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        by_step = np.where(np.isfinite(step), step, 0.0) / allowed_step[first + 1 : stop]
        by_need = np.where(np.isfinite(need), need, 0.0) / allowed_need[first + 1 : stop]
        return first, stop, np.maximum(by_step, by_need), by_need > by_step, guard

    queue = [
        ratios(
            first, stop, steps[first + 1 : stop], needs[first + 1 : stop], guards[first + 1 : stop]
        )
        for first, stop in pieces
    ]
    while queue:
        first, stop, ratio, by_need, guard = queue.pop()
        if not np.any(ratio > 1.0):
            continue
        best = int(np.argmax(ratio))
        cuts = {first + 1 + best}
        if by_need[best] and guard[best] > 0:
            # both sides of the frames the acceleration test's fits left out
            part = frames[first:stop]
            left = int(part[best]) - int(guard[best])
            right = int(part[best + 1]) + int(guard[best])
            cuts = {
                first + int(np.searchsorted(part, left, side="right")),
                first + int(np.searchsorted(part, right, side="left")),
            }
        edges = [first, *sorted(cuts), stop]
        for cut in edges[1:-1]:
            breaks[cut] = True
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            if hi - lo > 1:
                queue.append(ratios(lo, hi, *statistics_of(lo, hi)))
    return breaks


def _rejections(
    frames: np.ndarray,
    tracks: np.ndarray,
    shots: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    sizes: tuple[np.ndarray, np.ndarray] | None,
    jump_frames: np.ndarray,
    settings: _Settings,
) -> tuple[np.ndarray, np.ndarray]:
    """(test, piece) of the measured samples sorted by (track, frame): the code of the test
    that rejects each position (0 when it is kept), the box-size, local-median and
    excursion tests running in that order, each on what the previous ones kept; and a piece
    number that changes at every track, shot, calibration jump and trend jump (the jump
    test runs last, on the positions every test kept)."""
    test = np.zeros(frames.size, dtype=np.int64)
    radius = settings.outlier_radius
    if sizes is not None:
        runs = _piece_starts(frames, tracks, shots, jump_frames, None)
        limits = (settings.max_height_change, settings.max_width_change)
        for values, limit, code in zip(sizes, limits, (_BOX_HEIGHT, _BOX_WIDTH), strict=True):
            if limit is not None:
                flags = _size_outliers(frames, values, runs, radius, limit)
                test[flags & (test == _KEPT)] = code
    if settings.outlier_threshold is not None:
        kept = np.flatnonzero(test == _KEPT)
        runs = _piece_starts(frames[kept], tracks[kept], shots[kept], jump_frames, None)
        flags = _position_outliers(
            frames[kept], xs[kept], ys[kept], runs, radius, settings.outlier_threshold
        )
        test[kept[flags]] = _LOCAL_MEDIAN
    breaks = np.zeros(frames.size, dtype=bool)
    if settings.jump_threshold is not None:
        kept = np.flatnonzero(test == _KEPT)
        pieces = _piece_starts(
            frames[kept], tracks[kept], shots[kept], jump_frames, settings.max_gap
        )
        flags = _excursions(frames[kept], tracks[kept], xs[kept], ys[kept], pieces, settings)
        test[kept[flags]] = _EXCURSION
        kept = np.flatnonzero(test == _KEPT)
        pieces = _piece_starts(
            frames[kept], tracks[kept], shots[kept], jump_frames, settings.max_gap
        )
        flags = _jumps(frames[kept], tracks[kept], xs[kept], ys[kept], pieces, settings)
        breaks[kept[flags]] = True
    # Piece numbers on every sample: a new piece at each run boundary and each jump.
    runs = _piece_starts(frames, tracks, shots, jump_frames, None) | breaks
    return test, np.cumsum(runs) - 1


def _segment(
    frames: np.ndarray, xs: np.ndarray, ys: np.ndarray, settings: _Settings, segment_id: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Kinematics of one segment from its measured frames (sorted, gaps already bounded)."""
    full = np.arange(frames[0], frames[-1] + 1, dtype=np.int64)
    interpolated = ~np.isin(full, frames)
    x = np.interp(full, frames, xs)
    y = np.interp(full, frames, ys)
    n = full.size
    h = (settings.window - 1) // 2  # filter half-window
    k = settings.chord_half  # chord half, frames
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    ends_ok = np.zeros(n, dtype=bool)
    if n >= settings.window:
        dt = 1.0 / settings.fps
        window, poly = settings.window, settings.poly
        x_s = savgol_filter(x, window, poly, mode="interp")
        y_s = savgol_filter(y, window, poly, mode="interp")
        derivative = np.hypot(
            savgol_filter(x, window, poly, deriv=1, delta=dt, mode="interp"),
            savgol_filter(y, window, poly, deriv=1, delta=dt, mode="interp"),
        )
        # One bad position spreads over the smoothing window, so every position within
        # half a window of an implausible derivative is tainted too.
        tainted = np.zeros(n, dtype=bool)
        for index in np.flatnonzero(derivative > settings.max_speed).tolist():
            tainted[max(0, index - h) : index + h + 1] = True
        position = np.arange(n)
        anchor = ~interpolated & (position >= h) & (position <= n - 1 - h) & ~tainted
        if n > 2 * k:
            lo = np.arange(n - 2 * k)
            hi = lo + 2 * k
            seconds = 2 * k / settings.fps
            vx[lo + k] = (x_s[hi] - x_s[lo]) / seconds
            vy[lo + k] = (y_s[hi] - y_s[lo]) / seconds
            ends_ok[lo + k] = anchor[lo] & anchor[hi]
    else:
        x_s, y_s = x, y
    speed = np.hypot(vx, vy)
    valid = (
        np.isfinite(speed) & ends_ok & (speed <= settings.max_speed) & (n >= settings.min_frames)
    )
    peak_speed = math.nan
    peak_frame = -1
    if np.any(valid):
        best = int(np.argmax(np.where(valid, speed, -np.inf)))
        peak_speed = float(speed[best])
        peak_frame = int(full[best])
    part = {
        "frame": full,
        "segment_id": np.full(n, segment_id, dtype=np.int64),
        "x": np.asarray(x_s, dtype=np.float64),
        "y": np.asarray(y_s, dtype=np.float64),
        "vx": vx,
        "vy": vy,
        "speed": np.asarray(speed, dtype=np.float64),
        "valid": valid,
        "interpolated": interpolated,
    }
    summary = {
        "segment_id": segment_id,
        "start_frame": int(full[0]),
        "end_frame": int(full[-1]),
        "n_frames": n,
        "n_interpolated": int(np.count_nonzero(interpolated)),
        "n_invalid": int(np.count_nonzero(~valid)),
        "peak_speed": peak_speed,
        "peak_frame": peak_frame,
    }
    return part, summary


_KIN_DTYPES: dict[str, Any] = {
    "frame": np.int64,
    "track_id": np.int64,
    "segment_id": np.int64,
    "x": np.float64,
    "y": np.float64,
    "vx": np.float64,
    "vy": np.float64,
    "speed": np.float64,
    "valid": bool,
    "interpolated": bool,
}
_SEGMENT_DTYPES: dict[str, Any] = {
    "track_id": np.int64,
    "segment_id": np.int64,
    "shot_id": np.int64,
    "start_frame": np.int64,
    "end_frame": np.int64,
    "n_frames": np.int64,
    "n_interpolated": np.int64,
    "n_outliers": np.int64,
    "n_invalid": np.int64,
    "peak_speed": np.float64,
    "peak_frame": np.int64,
}


def _kinematics_frame(parts: list[dict[str, np.ndarray]]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(
            {name: np.array([], dtype=_KIN_DTYPES[name]) for name in KINEMATICS_COLUMNS}
        )
    columns = {
        name: np.concatenate([part[name] for part in parts]).astype(_KIN_DTYPES[name])
        for name in KINEMATICS_COLUMNS
    }
    return pd.DataFrame(columns, columns=list(KINEMATICS_COLUMNS))


def _segments_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            {name: np.array([], dtype=_SEGMENT_DTYPES[name]) for name in SEGMENTS_COLUMNS}
        )
    frame = pd.DataFrame(rows, columns=list(SEGMENTS_COLUMNS))
    return frame.astype(_SEGMENT_DTYPES)


# --- per-frame helpers ----------------------------------------------------------------------


def _single_frame_players(
    frame_df: Any, x_col: str, y_col: str, problems: list[str]
) -> pd.DataFrame | None:
    """Players (team left/right, role player or goalkeeper when a role column exists) with
    finite positions, sorted by track_id."""
    if _missing_columns("frame_df", frame_df, ("track_id", "team", x_col, y_col), problems):
        return None
    if "frame" in frame_df.columns and frame_df["frame"].nunique() > 1:
        problems.append(f"frame_df must hold a single frame, got {frame_df['frame'].nunique()}")
    track_ids = frame_df["track_id"].to_numpy(dtype=np.int64)
    known = track_ids[track_ids >= 0]
    if np.unique(known).size != known.size:
        problems.append("frame_df lists a track_id more than once")
    if problems:
        return None
    keep = (
        frame_df["team"].isin(TEAMS).to_numpy(dtype=bool)
        & np.isfinite(frame_df[x_col].to_numpy(dtype=np.float64))
        & np.isfinite(frame_df[y_col].to_numpy(dtype=np.float64))
    )
    if "role" in frame_df.columns:
        keep = keep & frame_df["role"].isin(PLAYER_ROLES).to_numpy(dtype=bool)
    players = frame_df.loc[keep]
    return players.sort_values("track_id", kind="stable")


def nearest_opponent(
    frame_df: pd.DataFrame, *, x_col: str = "pitch_x", y_col: str = "pitch_y"
) -> pd.DataFrame:
    """For each player of one frame, the nearest player of the other team.

    Args:
        frame_df: rows of a single frame with track_id, team ("left" | "right" | null) and
            positions in metres (``x_col``/``y_col``; default the tracks.parquet columns,
            pass "x"/"y" for smoothed kinematics positions joined with team). When a
            ``role`` column exists only players and goalkeepers count (not referees).

    Returns a DataFrame sorted by track_id with columns track_id (int64), team,
    opponent_track_id (int64, -1 when the other team has nobody with a position) and
    distance_m (float64 metres on the pitch plane, NaN when there is no opponent). Rows
    without a team or a finite position are left out. Raises KinematicsInputError.
    """
    problems: list[str] = []
    players = _single_frame_players(frame_df, x_col, y_col, problems)
    if problems or players is None:
        raise KinematicsInputError("nearest_opponent", problems)
    ids = players["track_id"].to_numpy(dtype=np.int64)
    teams = players["team"].to_numpy(dtype=object)
    points = np.column_stack(
        [players[x_col].to_numpy(dtype=np.float64), players[y_col].to_numpy(dtype=np.float64)]
    )
    opponent = np.full(ids.size, -1, dtype=np.int64)
    distance = np.full(ids.size, np.nan)
    if ids.size:
        gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        gaps[teams[:, None] == teams[None, :]] = np.inf
        best = np.argmin(gaps, axis=1)
        nearest = gaps[np.arange(ids.size), best]
        found = np.isfinite(nearest)
        opponent[found] = ids[best[found]]
        distance[found] = nearest[found]
    return pd.DataFrame(
        {
            "track_id": ids,
            "team": pd.Series(teams, dtype=object),
            "opponent_track_id": opponent,
            "distance_m": distance,
        }
    )


@dataclass(frozen=True)
class TeamShape:
    """Shape of one team's outfield players in one frame (SoccerNet metres).

    Attributes:
        team: the defending team, "left" (defends x < 0) or "right" (defends x > 0).
        n_outfield: outfield players with a position (goalkeepers excluded).
        deepest_x: x of the outfield player nearest to the team's own goal line (the
            smallest x for "left", the largest for "right"): the defensive line. NaN when
            n_outfield is 0.
        deepest_track_id: that player's track_id, -1 when n_outfield is 0.
        width_m: max y - min y over the outfield players (0 for one player, NaN for none).
        length_m: max x - min x over the outfield players (same rules).
        centroid_x, centroid_y: mean position of the outfield players (NaN for none).
    """

    team: str
    n_outfield: int
    deepest_x: float
    deepest_track_id: int
    width_m: float
    length_m: float
    centroid_x: float
    centroid_y: float


def team_shape(
    frame_df: pd.DataFrame,
    defending_team: str,
    *,
    x_col: str = "pitch_x",
    y_col: str = "pitch_y",
) -> TeamShape:
    """Deepest outfield defender, width, length and centroid of ``defending_team``.

    ``frame_df`` holds one frame's rows (track_id, team, role, positions in metres). Team
    "left" defends the goal at x = -52.5 and "right" the one at x = +52.5 (the plugin's
    convention). Outfield players are rows with role "player"; goalkeepers are excluded.
    Only players visible in the frame count, so on a broadcast frame the shape describes
    the visible part of the team. Raises KinematicsInputError listing every problem.
    """
    problems: list[str] = []
    if defending_team not in TEAMS:
        problems.append(f"defending_team must be 'left' or 'right', got {defending_team!r}")
    if isinstance(frame_df, pd.DataFrame) and "role" not in frame_df.columns:
        problems.append("frame_df lacks column role (needed to leave out goalkeepers)")
    players = _single_frame_players(frame_df, x_col, y_col, problems)
    if problems or players is None:
        raise KinematicsInputError("team_shape", problems)
    outfield = players.loc[
        (players["team"] == defending_team).to_numpy(dtype=bool)
        & (players["role"] == "player").to_numpy(dtype=bool)
    ]
    n = len(outfield)
    if n == 0:
        nan = float("nan")
        return TeamShape(defending_team, 0, nan, -1, nan, nan, nan, nan)
    x = outfield[x_col].to_numpy(dtype=np.float64)
    y = outfield[y_col].to_numpy(dtype=np.float64)
    ids = outfield["track_id"].to_numpy(dtype=np.int64)
    deepest = int(np.argmin(x)) if defending_team == "left" else int(np.argmax(x))
    return TeamShape(
        team=defending_team,
        n_outfield=n,
        deepest_x=float(x[deepest]),
        deepest_track_id=int(ids[deepest]),
        width_m=float(np.max(y) - np.min(y)),
        length_m=float(np.max(x) - np.min(x)),
        centroid_x=float(np.mean(x)),
        centroid_y=float(np.mean(y)),
    )
