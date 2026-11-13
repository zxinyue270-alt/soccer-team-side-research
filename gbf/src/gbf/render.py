"""Analysis drawn on the ORIGINAL broadcast frames, H.264 output and highlight edits (M1).

The analysis is shown where it was measured: on the broadcast frames themselves. 3D results
appear only as numbers or as projections back onto the frame.

Entry points
------------
- :class:`FrameAnnotations` (players, skeletons, ball, arrows, lines, texts, minimap, banner,
  badges, data card) and :func:`draw_annotations`, which draws them on one frame.
- :class:`VideoWriter` / :func:`write_video`: verified H.264 output, with original audio for
  a time range (:class:`AudioSpan`, :class:`Silence`).
- :func:`plan_highlight_edit` and :func:`render_edit`: the highlight edit (normal-speed
  window, slow motion by repeated frames around the key frame, a freeze-frame card) and its
  ``<clip>.render.json`` manifest (segment table, hashes, every text with its evidence).
- :class:`HighlightScene` over :class:`SceneTables`: the per-frame overlay of one highlight,
  built from the run's tables (tracks, calib, ball track, kinematics, pose, and the 3D lift's
  facing, drawn back on the frame as a reference).
- :func:`render_highlight`: clip, manifest and key-frame images of one highlight.

Conventions
-----------
Frames
    ``(height, width, 3)`` ``uint8`` RGB numpy arrays, as :func:`gbf.ingest.iter_frames`
    yields them. :func:`draw_annotations` returns a new array and never modifies its input.
Pixels
    ``(u, v)`` with u to the right and v down, pixel centres at integer coordinates (pixel
    (i, j) covers [i - 0.5, i + 0.5) x [j - 0.5, j + 0.5)), exactly as in
    :mod:`gbf.geometry`. Boxes are ``(x1, y1, x2, y2)`` in the same coordinates with
    x1 <= x2 and y1 <= y2; a player's foot point is the bottom centre ((x1 + x2) / 2, y2).
Pitch (minimap)
    SoccerNet metres: origin at the centre spot, x in [-52.5, 52.5] along the touchlines,
    y in [-34, 34] with the FAR touchline at y = -34. The minimap is drawn as the main
    camera sees the pitch: x to the right, the far touchline at the top.
Sizes
    Every size in :class:`Style` and in the annotation classes is in pixels of a 1080-line
    frame and is multiplied by the UI scale (frame height / 1080 unless
    ``Style.ui_scale`` is set), so a 720p frame gets the same layout, smaller.
Colours
    RGB tuples of ints in 0..255; an alpha, where present, is 0..255.
Drawing
    The overlay is drawn on a copy of the frame upsampled ``Style.supersample`` times
    (nearest neighbour) and reduced back with a box filter: shapes and text are
    anti-aliased, and every pixel the overlay does not touch stays bit-identical.
Video
    :class:`VideoWriter` streams frames into one ffmpeg process (rawvideo rgb24 on stdin;
    H.264 yuv420p, BT.709 limited range, CRF 18 by default, at the source frame rate) and
    verifies the written file with ffprobe -- decoded frame count, constant frame rate,
    size, codec and audio -- before moving it into place. Slow motion repeats source frames
    (no interpolation), so every output frame is a real broadcast frame; the edit manifest
    maps every output frame back to its canonical source frame.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.PngImagePlugin import PngInfo

import gbf
from gbf.fonts import FontInfo, find_cjk_fonts, load_font
from gbf.geometry import Camera, GeometryInputError, pitch_zup_to_soccernet, pixel_rays, project
from gbf.highlights import (
    DUEL_BALL_RADIUS_H,
    DUEL_BALL_RADIUS_M,
    DUEL_FAR_H,
    DUEL_FAR_M,
    TYPE_NAMES_ZH,
)
from gbf.ingest import FFmpegError, format_rate, iter_frames, parse_rate, probe, read_frames
from gbf.ingest.rates import round_half_away
from gbf.ingest.tools import (
    LOGLEVEL,
    StderrCollector,
    ffmpeg_path,
    ffmpeg_version,
    ffprobe_path,
    media_arg,
    run_tool,
)
from gbf.kinematics import chord_half_frames
from gbf.measure import LIFTED, FacingGates, facing_frame_ok, facing_image_dx, image_trunk_deg
from gbf.moments import BALL_MOMENT_KINDS
from gbf.track_tools import (
    EDGE_TOUCH_PER_BOX_HEIGHT,
    MIN_PLAYER_GAP_M,
    box_touches_edge,
    off_pitch_at_camera_edge,
    same_spot,
    speed_chord_safe,
)
from gbf.util.clock import ClockError
from gbf.util.clock import format_clock as _clock_text
from gbf.util.hashing import sha256_file
from gbf.util.io import write_json

Colour = tuple[int, int, int]
RGBA = tuple[int, int, int, int]
Point = tuple[float, float]
Emphasis = Literal["focus", "normal", "context"]
EMPHASES: tuple[str, ...] = ("focus", "normal", "context")
BadgeKind = Literal[
    "unverified",
    "image_plane",
    "reference_3d",
    "experimental",
    "unavailable",
    "official",
    "measured",
    "private",
    "info",
]
BADGE_KINDS: tuple[str, ...] = (
    "unverified",
    "image_plane",
    "reference_3d",
    "experimental",
    "unavailable",
    "official",
    "measured",
    "private",
    "info",
)
SegmentKind = Literal["normal", "slow", "freeze"]
SEGMENT_KINDS: tuple[str, ...] = ("normal", "slow", "freeze")

#: Ground ellipse under a player without a calibrated ring: half-width as a fraction of the
#: box width and height as a fraction of the half-width (a camera above the pitch sees the
#: ground foreshortened), each with a floor of 10 and 4 pixels at 1080p.
RING_RX_PER_BOX_WIDTH = 0.55
RING_RY_PER_RX = 0.26

#: Without a calibrated ground ring, a player whose box is taller than this share of the
#: frame height is in a close-up and gets no ground ellipse, nor does a player whose box
#: touches the bottom edge: the ellipse would cross the body (and the banner), not the grass.
CLOSEUP_BOX_SHARE = 0.5

#: Pitch size of the minimap and of SoccerNet's template, metres.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

#: The minimap does not draw a ball more than this many metres outside the pitch lines: a
#: ground-plane position out there is mostly a ball in the air projected past the line, and
#: the minimap shows only the pitch.
MINIMAP_BALL_MARGIN_M = 1.0

#: Output audio format of every rendered clip.
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"

#: A highlight window with less of its own audio than this (seconds) is not repeated under
#: a replay that has run past the end of the source audio: the replay is silent instead.
REPLAY_LOOP_MIN_S = 0.5

#: Video encoder settings (x264's output depends on its thread count, so it is fixed).
VIDEO_CODEC = "libx264"
VIDEO_PIX_FMT = "yuv420p"
DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"
X264_THREADS = 8

#: Halpe-26 keypoint names in index order (RTMPose through rtmlib). left/right are the
#: person's own anatomical sides.
HALPE26_NAMES: tuple[str, ...] = (
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

#: Limbs drawn for a Halpe-26 skeleton: (joint a, joint b, side). Face links are left out
#: on purpose: at broadcast sizes they are a few pixels long and only add clutter.
HALPE26_LIMBS: tuple[tuple[int, int, str], ...] = (
    (17, 18, "centre"),
    (18, 19, "centre"),
    (18, 5, "left"),
    (5, 7, "left"),
    (7, 9, "left"),
    (18, 6, "right"),
    (6, 8, "right"),
    (8, 10, "right"),
    (19, 11, "left"),
    (11, 13, "left"),
    (13, 15, "left"),
    (15, 20, "left"),
    (15, 24, "left"),
    (19, 12, "right"),
    (12, 14, "right"),
    (14, 16, "right"),
    (16, 21, "right"),
    (16, 25, "right"),
)
_JOINT_SIDE: dict[int, str] = {
    5: "left",
    7: "left",
    9: "left",
    11: "left",
    13: "left",
    15: "left",
    20: "left",
    24: "left",
    6: "right",
    8: "right",
    10: "right",
    12: "right",
    14: "right",
    16: "right",
    21: "right",
    25: "right",
    17: "centre",
    18: "centre",
    19: "centre",
}

#: Halpe-26 joints of the lower body (hips, pelvis, knees, ankles, toes, heels): not drawn
#: for a person whose box touches the bottom edge of the frame (see :class:`Skeleton`).
HALPE26_LOWER_BODY: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25)


class RenderError(ValueError):
    """Invalid annotations, frames, settings or output. ``problems`` lists every problem."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


# --------------------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------------------
def _is_number(value: Any) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool)


def _finite(name: str, value: Any, problems: list[str]) -> float | None:
    if not _is_number(value) or not math.isfinite(float(value)):
        problems.append(f"{name} must be a finite number, got {value!r}")
        return None
    return float(value)


def _point(name: str, value: Any, problems: list[str]) -> Point | None:
    try:
        items = list(value)
    except TypeError:
        problems.append(f"{name} must be a (u, v) pair, got {value!r}")
        return None
    if len(items) != 2:
        problems.append(f"{name} must be a (u, v) pair, got {len(items)} values")
        return None
    u = _finite(f"{name}[0]", items[0], problems)
    v = _finite(f"{name}[1]", items[1], problems)
    if u is None or v is None:
        return None
    return (u, v)


def _colour(name: str, value: Any, problems: list[str]) -> Colour | None:
    try:
        items = list(value)
    except TypeError:
        problems.append(f"{name} must be an (r, g, b) tuple, got {value!r}")
        return None
    if len(items) != 3 or not all(
        isinstance(c, int | np.integer) and not isinstance(c, bool) and 0 <= int(c) <= 255
        for c in items
    ):
        problems.append(f"{name} must be three ints in 0..255, got {value!r}")
        return None
    return (int(items[0]), int(items[1]), int(items[2]))


def _optional_text(name: str, value: Any, problems: list[str]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        problems.append(f"{name} must be a string or None, got {type(value).__name__}")
        return None
    return value


def _ids(name: str, value: Any, problems: list[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        problems.append(f"{name} must be a sequence of evidence ids, not one string")
        return ()
    try:
        items = tuple(value)
    except TypeError:
        problems.append(f"{name} must be a sequence of strings, got {value!r}")
        return ()
    bad = [item for item in items if not isinstance(item, str) or not item]
    if bad:
        problems.append(f"{name} must hold non-empty strings, got {bad!r}")
        return ()
    return items


def _emphasis(name: str, value: Any, problems: list[str]) -> None:
    if value not in EMPHASES:
        problems.append(f"{name} must be one of {EMPHASES}, got {value!r}")


def _raise(where: str, problems: list[str]) -> None:
    if problems:
        raise RenderError(where, problems)


# --------------------------------------------------------------------------------------
# annotation types
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Badge:
    """A small status chip (coloured dot + text), e.g. 未验证, 仅画面平面, 3D 参考, 实验, 不可用.

    ``kind`` picks the dot colour from :attr:`Style.badge_colours`. :meth:`of` infers the
    kind from the usual Chinese labels. A ``compact`` badge is drawn as its text alone in
    the kind's colour, without the chip and the dot, e.g. the 未验证 marker of a number on
    a secondary label. ``reference_3d`` (3D 参考) marks a value of the 3D lift (gbf.lift3d:
    facing, trunk lean), a reference beside what the original frames show.
    """

    text: str
    kind: BadgeKind = "info"
    compact: bool = False

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.text, str) or not self.text.strip():
            problems.append(f"text must be a non-empty string, got {self.text!r}")
        if self.kind not in BADGE_KINDS:
            problems.append(f"kind must be one of {BADGE_KINDS}, got {self.kind!r}")
        if not isinstance(self.compact, bool):
            problems.append(f"compact must be a bool, got {self.compact!r}")
        _raise("Badge", problems)

    @classmethod
    def of(cls, value: Badge | str) -> Badge:
        """``value`` itself, or a Badge for the text with its kind inferred."""
        if isinstance(value, Badge):
            return value
        if not isinstance(value, str):
            raise RenderError("Badge.of", [f"expected a Badge or a string, got {value!r}"])
        kinds: tuple[tuple[str, BadgeKind], ...] = (
            ("未验证", "unverified"),
            ("仅画面平面", "image_plane"),
            ("画面平面", "image_plane"),
            ("3D 参考", "reference_3d"),
            ("实验", "experimental"),
            ("不可用", "unavailable"),
            ("官方", "official"),
            ("已测量", "measured"),
            ("私有", "private"),
            ("禁止分享", "private"),
        )
        for marker, kind in kinds:
            if marker in value:
                return cls(value, kind)
        return cls(value, "info")


def _badges(name: str, value: Any, problems: list[str]) -> tuple[Badge, ...]:
    if isinstance(value, str | Badge):
        value = (value,)
    try:
        items = tuple(value)
    except TypeError:
        problems.append(f"{name} must be a sequence of Badge or str, got {value!r}")
        return ()
    badges: list[Badge] = []
    for index, item in enumerate(items):
        try:
            badges.append(Badge.of(item))
        except RenderError as exc:
            problems.extend(f"{name}[{index}]: {p}" for p in exc.problems)
    return tuple(badges)


@dataclass(frozen=True)
class PlayerMark:
    """One player on the frame.

    Attributes:
        box: (x1, y1, x2, y2) in pixels.
        colour: identity colour; ``None`` uses the team colour of ``team`` from the style.
        team: team key ("left", "right", "referee", ...) for the style's team colours.
        label: text above the head, e.g. "#10 · 27.4 km/h"; ``None`` draws no label.
        emphasis: "focus" (glowing ring under the feet, corner brackets, pointer, bold
            label), "normal" (thin ring, light brackets) or "context" (faint ring only).
        ring: optional ground ring in pixels (for example :func:`ground_circle_px` of a
            calibrated frame); ``None`` draws an ellipse derived from the box, except for a
            close-up or a box at the bottom edge (:data:`CLOSEUP_BOX_SHARE`).
        badges: chips drawn after the label text.
        evidence_ids: evidence records behind any number in ``label`` or ``label_held``.
        label_held: text drawn right after ``label`` in a grey, for a value held from an
            earlier frame (a speed over a few frames without a valid one); ``None`` for none.
    """

    box: tuple[float, float, float, float]
    colour: Colour | None = None
    team: str | None = None
    label: str | None = None
    emphasis: Emphasis = "normal"
    ring: tuple[Point, ...] | None = None
    badges: tuple[Badge, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    label_held: str | None = None

    def __post_init__(self) -> None:
        problems: list[str] = []
        try:
            values = list(self.box)
        except TypeError:
            values = []
        if len(values) != 4:
            problems.append(f"box must be (x1, y1, x2, y2), got {self.box!r}")
        else:
            box = [_finite(f"box[{i}]", v, problems) for i, v in enumerate(values)]
            if None not in box:
                x1, y1, x2, y2 = (float(v) for v in box)  # type: ignore[arg-type]
                if x2 < x1 or y2 < y1:
                    problems.append(f"box must have x1 <= x2 and y1 <= y2, got {self.box!r}")
                object.__setattr__(self, "box", (x1, y1, x2, y2))
        if self.colour is not None:
            colour = _colour("colour", self.colour, problems)
            if colour is not None:
                object.__setattr__(self, "colour", colour)
        _optional_text("team", self.team, problems)
        _optional_text("label", self.label, problems)
        _optional_text("label_held", self.label_held, problems)
        if self.label_held is not None and not self.label:
            problems.append("label_held needs a label it follows")
        _emphasis("emphasis", self.emphasis, problems)
        if self.ring is not None:
            ring = _polyline("ring", self.ring, problems, allow_nan=True)
            if ring is not None and len(ring) < 3:
                problems.append("ring needs at least 3 points")
            object.__setattr__(self, "ring", ring)
        object.__setattr__(self, "badges", _badges("badges", self.badges, problems))
        object.__setattr__(self, "evidence_ids", _ids("evidence_ids", self.evidence_ids, problems))
        _raise("PlayerMark", problems)


def _polyline(
    name: str, value: Any, problems: list[str], *, allow_nan: bool = False
) -> tuple[Point, ...] | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        problems.append(f"{name} must be a sequence of (u, v) points")
        return None
    if array.size == 0:
        return ()
    if array.ndim != 2 or array.shape[1] != 2:
        problems.append(f"{name} must have shape (n, 2), got {array.shape}")
        return None
    if not allow_nan and not np.isfinite(array).all():
        problems.append(f"{name} must hold finite coordinates")
        return None
    if np.isinf(array).any():
        problems.append(f"{name} must not hold infinite coordinates")
        return None
    return tuple((float(u), float(v)) for u, v in array)


def _box(name: str, value: Any, problems: list[str]) -> tuple[float, float, float, float] | None:
    try:
        values = list(value)
    except TypeError:
        values = []
    if len(values) != 4:
        problems.append(f"{name} must be (x1, y1, x2, y2), got {value!r}")
        return None
    box = [_finite(f"{name}[{i}]", v, problems) for i, v in enumerate(values)]
    if None in box:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)  # type: ignore[arg-type]
    if x2 < x1 or y2 < y1:
        problems.append(f"{name} must have x1 <= x2 and y1 <= y2, got {value!r}")
        return None
    return (x1, y1, x2, y2)


@dataclass(frozen=True)
class Skeleton:
    """A 2D Halpe-26 skeleton (keypoints in :data:`HALPE26_NAMES` order).

    Keypoints outside the frame are not drawn, nor the limbs that use them: they are the
    pose model's guesses about the part of the body the picture does not show. Limbs are
    never dropped for their angle alone (a falling player can lie flat).

    Attributes:
        keypoints: (26, 3) array of (u, v, score); NaN marks a missing keypoint.
        min_score: keypoints scoring below this are not drawn, nor limbs touching them.
        emphasis: "context" draws the skeleton fainter.
        colour: optional single colour for every limb (default: left/right/centre colours
            of the style, by the person's anatomical side).
        box: the person's box (x1, y1, x2, y2) or ``None``. When it touches the bottom edge
            of the frame (:data:`EDGE_TOUCH_PER_BOX_HEIGHT`) the legs are out of the
            picture, and the lower body (:data:`HALPE26_LOWER_BODY`) is not drawn.
    """

    keypoints: np.ndarray
    min_score: float = 0.3
    emphasis: Emphasis = "normal"
    colour: Colour | None = None
    box: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        problems: list[str] = []
        try:
            array = np.array(self.keypoints, dtype=np.float64)
        except (TypeError, ValueError):
            array = np.empty((0,))
        if array.shape != (26, 3):
            problems.append(f"keypoints must have shape (26, 3), got {array.shape}")
        elif np.isinf(array).any():
            problems.append("keypoints must not hold infinite values")
        else:
            array.setflags(write=False)
            object.__setattr__(self, "keypoints", array)
        _finite("min_score", self.min_score, problems)
        _emphasis("emphasis", self.emphasis, problems)
        if self.colour is not None:
            colour = _colour("colour", self.colour, problems)
            if colour is not None:
                object.__setattr__(self, "colour", colour)
        if self.box is not None:
            object.__setattr__(self, "box", _box("box", self.box, problems))
        _raise("Skeleton", problems)


@dataclass(frozen=True)
class BallMark:
    """The ball: a ring around ``position`` and a fading trail of earlier positions.

    Attributes:
        position: (u, v) pixels of the ball centre, or ``None`` (trail only).
        radius: ball radius in source pixels (the ring is drawn just outside it).
        trail: earlier positions, OLDEST FIRST; NaN points break the trail.
        interpolated: the position is interpolated, not detected: drawn as a dashed ring.
        label: optional text next to the ball.
        evidence_ids: evidence behind any number in ``label``.
    """

    position: Point | None
    radius: float = 9.0
    trail: tuple[Point, ...] = ()
    interpolated: bool = False
    label: str | None = None
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.position is not None:
            object.__setattr__(self, "position", _point("position", self.position, problems))
        radius = _finite("radius", self.radius, problems)
        if radius is not None and radius <= 0:
            problems.append(f"radius must be > 0, got {radius}")
        trail = _polyline("trail", self.trail, problems, allow_nan=True)
        object.__setattr__(self, "trail", trail if trail is not None else ())
        if not isinstance(self.interpolated, bool):
            problems.append(f"interpolated must be a bool, got {self.interpolated!r}")
        _optional_text("label", self.label, problems)
        object.__setattr__(self, "evidence_ids", _ids("evidence_ids", self.evidence_ids, problems))
        _raise("BallMark", problems)


@dataclass(frozen=True)
class Arrow:
    """An arrow from ``start`` to ``end`` (pixels) with an optional label just past its
    head; an arrow whose label is left out (:func:`draw_annotations`) is not drawn either.

    ``width`` is the shaft width in 1080p pixels (default 4); ``colour`` defaults to the
    style's accent; ``dashed`` draws the shaft dashed (an estimate, such as the 3D facing).
    """

    start: Point
    end: Point
    label: str | None = None
    colour: Colour | None = None
    width: float = 4.0
    dashed: bool = False
    badges: tuple[Badge, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        problems: list[str] = []
        start = _point("start", self.start, problems)
        end = _point("end", self.end, problems)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if start is not None and end is not None and start == end:
            problems.append("start and end must differ")
        _optional_text("label", self.label, problems)
        if self.colour is not None:
            object.__setattr__(self, "colour", _colour("colour", self.colour, problems))
        width = _finite("width", self.width, problems)
        if width is not None and width <= 0:
            problems.append(f"width must be > 0, got {width}")
        object.__setattr__(self, "badges", _badges("badges", self.badges, problems))
        object.__setattr__(self, "evidence_ids", _ids("evidence_ids", self.evidence_ids, problems))
        _raise("Arrow", problems)


@dataclass(frozen=True)
class LinkLine:
    """A line between two points with a label, e.g. the distance to the nearest opponent.

    Drawn dashed by default with dots at both ends; ``colour`` defaults to white.
    """

    start: Point
    end: Point
    label: str | None = None
    colour: Colour | None = None
    dashed: bool = True
    badges: tuple[Badge, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        problems: list[str] = []
        object.__setattr__(self, "start", _point("start", self.start, problems))
        object.__setattr__(self, "end", _point("end", self.end, problems))
        _optional_text("label", self.label, problems)
        if self.colour is not None:
            object.__setattr__(self, "colour", _colour("colour", self.colour, problems))
        object.__setattr__(self, "badges", _badges("badges", self.badges, problems))
        object.__setattr__(self, "evidence_ids", _ids("evidence_ids", self.evidence_ids, problems))
        _raise("LinkLine", problems)


_ANCHORS = {f"{h}{v}" for h in "lmr" for v in "atmsbd"}


@dataclass(frozen=True)
class TextLabel:
    """Free text at ``position`` (pixels) with a Pillow ``anchor`` (e.g. "lm", "mb").

    ``size`` is the font size in 1080p pixels; ``background`` puts the text on the
    semi-transparent label panel.
    """

    position: Point
    text: str
    anchor: str = "lm"
    size: float = 24.0
    colour: Colour = (255, 255, 255)
    background: bool = True
    bold: bool = False
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        problems: list[str] = []
        object.__setattr__(self, "position", _point("position", self.position, problems))
        if not isinstance(self.text, str) or not self.text:
            problems.append(f"text must be a non-empty string, got {self.text!r}")
        if self.anchor not in _ANCHORS:
            problems.append(
                f"anchor must be a Pillow text anchor such as 'lm', got {self.anchor!r}"
            )
        size = _finite("size", self.size, problems)
        if size is not None and size <= 0:
            problems.append(f"size must be > 0, got {size}")
        object.__setattr__(self, "colour", _colour("colour", self.colour, problems))
        object.__setattr__(self, "evidence_ids", _ids("evidence_ids", self.evidence_ids, problems))
        _raise("TextLabel", problems)


@dataclass(frozen=True)
class MinimapDot:
    """One player on the minimap: SoccerNet metres (x, y), a team key and an emphasis."""

    x: float
    y: float
    team: str | None = None
    emphasis: Emphasis = "normal"

    def __post_init__(self) -> None:
        problems: list[str] = []
        _finite("x", self.x, problems)
        _finite("y", self.y, problems)
        _optional_text("team", self.team, problems)
        _emphasis("emphasis", self.emphasis, problems)
        _raise("MinimapDot", problems)


@dataclass(frozen=True)
class Minimap:
    """A 105 x 68 m top-down pitch in a corner of the frame.

    Attributes:
        players: :class:`MinimapDot` items or ``(x, y, team)`` / ``(x, y, team, emphasis)``
            tuples in SoccerNet metres.
        ball: ball (x, y) in SoccerNet metres (ground-plane approximation) or ``None``; not
            drawn when it is more than :data:`MINIMAP_BALL_MARGIN_M` off the pitch.
        view: the pitch area the camera sees, a polygon in metres (see
            :func:`camera_footprint`), drawn as a light wash; empty for none.
        paths: polylines in metres (for example a sprint path), drawn in the accent colour.
        note: a sentence shown over the pitch instead of positions, e.g. for a frame
            without calibration ("本帧没有可用标定，不显示球场位置").
        title: small caption above the pitch.
    """

    players: tuple[MinimapDot, ...] = ()
    ball: tuple[float, float] | None = None
    view: tuple[Point, ...] = ()
    paths: tuple[tuple[Point, ...], ...] = ()
    note: str | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        problems: list[str] = []
        dots: list[MinimapDot] = []
        for index, item in enumerate(self.players):
            if isinstance(item, MinimapDot):
                dots.append(item)
                continue
            try:
                parts = tuple(item)
            except TypeError:
                problems.append(f"players[{index}] must be a MinimapDot or (x, y, team) tuple")
                continue
            if len(parts) not in (3, 4):
                problems.append(f"players[{index}] must have 3 or 4 items, got {len(parts)}")
                continue
            try:
                dots.append(MinimapDot(*parts))
            except RenderError as exc:
                problems.extend(f"players[{index}]: {p}" for p in exc.problems)
        object.__setattr__(self, "players", tuple(dots))
        if self.ball is not None:
            object.__setattr__(self, "ball", _point("ball", self.ball, problems))
        view = _polyline("view", self.view, problems)
        object.__setattr__(self, "view", view if view is not None else ())
        paths: list[tuple[Point, ...]] = []
        for index, path in enumerate(self.paths):
            line = _polyline(f"paths[{index}]", path, problems)
            if line is not None:
                paths.append(line)
        object.__setattr__(self, "paths", tuple(paths))
        _optional_text("note", self.note, problems)
        _optional_text("title", self.title, problems)
        _raise("Minimap", problems)


@dataclass(frozen=True)
class Banner:
    """The lower-third title: ``title``, a time pill (e.g. "00:19.04") and a subtitle.

    The banner keeps clear of the minimap. A title too long for its first line goes on to a
    second line, between clauses where it can; one too long for two lines is set smaller,
    and only then loses its last clauses, marked "…" (the report keeps every title whole).
    :func:`annotation_texts` with the frame size lists the title as drawn.
    """

    title: str
    time: str | None = None
    subtitle: str | None = None

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.title, str) or not self.title.strip():
            problems.append(f"title must be a non-empty string, got {self.title!r}")
        _optional_text("time", self.time, problems)
        _optional_text("subtitle", self.subtitle, problems)
        _raise("Banner", problems)


@dataclass(frozen=True)
class CardRow:
    """One line of an :class:`InfoCard`: label, value, unit, badges and evidence ids. A
    unit of :data:`TIGHT_UNITS` (° and %) is written right after the value, others after a
    space."""

    label: str
    value: str
    unit: str = ""
    badges: tuple[Badge, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name in ("label", "value", "unit"):
            if not isinstance(getattr(self, name), str):
                problems.append(f"{name} must be a string, got {getattr(self, name)!r}")
        object.__setattr__(self, "badges", _badges("badges", self.badges, problems))
        object.__setattr__(self, "evidence_ids", _ids("evidence_ids", self.evidence_ids, problems))
        _raise("CardRow", problems)


@dataclass(frozen=True)
class InfoCard:
    """A data card (title, rows, footer) on one side of the frame, e.g. on a freeze frame.

    Rows that do not fit on the frame (clear of the minimap, the banner and its badges,
    inside the title-safe area) are left out from the end; labels under the card are not
    drawn (:func:`draw_annotations`). ``omitted`` counts the rows the card was given
    without (the report lists them), and the footer adds them to the rows left out for
    room: "另有 N 项见报告". :func:`annotation_texts` with the frame size lists what is drawn.
    """

    title: str
    rows: tuple[CardRow, ...] = ()
    footer: str | None = None
    side: Literal["right", "left"] = "right"
    omitted: int = 0

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.title, str) or not self.title.strip():
            problems.append(f"title must be a non-empty string, got {self.title!r}")
        rows = tuple(self.rows)
        for index, row in enumerate(rows):
            if not isinstance(row, CardRow):
                problems.append(f"rows[{index}] must be a CardRow, got {type(row).__name__}")
        object.__setattr__(self, "rows", rows)
        _optional_text("footer", self.footer, problems)
        if self.side not in ("right", "left"):
            problems.append(f"side must be 'right' or 'left', got {self.side!r}")
        omitted = self.omitted
        if isinstance(omitted, bool) or not isinstance(omitted, int | np.integer) or omitted < 0:
            problems.append(f"omitted must be an int >= 0, got {omitted!r}")
        else:
            object.__setattr__(self, "omitted", int(omitted))
        _raise("InfoCard", problems)


@dataclass(frozen=True)
class FrameAnnotations:
    """Everything drawn on one frame. Lists may be given as any sequence.

    ``dim`` (0..1) darkens the frame under the overlay, except inside ``spotlight``
    ((u, v, radius) in pixels), for freeze frames. ``arrow_badges`` are frame badges that
    only the arrows call for: they join the badge row, before ``badges``, when an arrow is
    drawn, and not when every arrow is left out with its label (:func:`draw_annotations`).
    """

    players: tuple[PlayerMark, ...] = ()
    skeletons: tuple[Skeleton, ...] = ()
    ball: BallMark | None = None
    arrows: tuple[Arrow, ...] = ()
    lines: tuple[LinkLine, ...] = ()
    texts: tuple[TextLabel, ...] = ()
    minimap: Minimap | None = None
    banner: Banner | None = None
    badges: tuple[Badge, ...] = ()
    card: InfoCard | None = None
    dim: float = 0.0
    spotlight: tuple[float, float, float] | None = None
    arrow_badges: tuple[Badge, ...] = ()

    def __post_init__(self) -> None:
        problems: list[str] = []
        typed: tuple[tuple[str, type], ...] = (
            ("players", PlayerMark),
            ("skeletons", Skeleton),
            ("arrows", Arrow),
            ("lines", LinkLine),
            ("texts", TextLabel),
        )
        for name, kind in typed:
            items = tuple(getattr(self, name))
            for index, item in enumerate(items):
                if not isinstance(item, kind):
                    problems.append(
                        f"{name}[{index}] must be a {kind.__name__}, got {type(item).__name__}"
                    )
            object.__setattr__(self, name, items)
        single: tuple[tuple[str, type], ...] = (
            ("ball", BallMark),
            ("minimap", Minimap),
            ("banner", Banner),
            ("card", InfoCard),
        )
        for name, kind in single:
            value = getattr(self, name)
            if value is not None and not isinstance(value, kind):
                problems.append(f"{name} must be a {kind.__name__} or None")
        object.__setattr__(self, "badges", _badges("badges", self.badges, problems))
        object.__setattr__(
            self, "arrow_badges", _badges("arrow_badges", self.arrow_badges, problems)
        )
        dim = _finite("dim", self.dim, problems)
        if dim is not None and not 0.0 <= dim <= 1.0:
            problems.append(f"dim must be in [0, 1], got {dim}")
        if self.spotlight is not None:
            try:
                spot = [float(v) for v in self.spotlight]
            except (TypeError, ValueError):
                spot = []
            if len(spot) != 3 or not all(math.isfinite(v) for v in spot) or spot[2] <= 0:
                problems.append(f"spotlight must be (u, v, radius > 0), got {self.spotlight!r}")
            else:
                object.__setattr__(self, "spotlight", (spot[0], spot[1], spot[2]))
        _raise("FrameAnnotations", problems)


def merge_annotations(base: FrameAnnotations, extra: FrameAnnotations) -> FrameAnnotations:
    """``base`` with ``extra`` drawn on top.

    Lists are concatenated (``extra`` last, so on top); ball, minimap, banner, card and
    spotlight come from ``extra`` when it sets them; ``dim`` is the larger of the two.
    """
    return FrameAnnotations(
        players=base.players + extra.players,
        skeletons=base.skeletons + extra.skeletons,
        ball=extra.ball if extra.ball is not None else base.ball,
        arrows=base.arrows + extra.arrows,
        lines=base.lines + extra.lines,
        texts=base.texts + extra.texts,
        minimap=extra.minimap if extra.minimap is not None else base.minimap,
        banner=extra.banner if extra.banner is not None else base.banner,
        badges=base.badges + extra.badges,
        card=extra.card if extra.card is not None else base.card,
        dim=max(base.dim, extra.dim),
        spotlight=extra.spotlight if extra.spotlight is not None else base.spotlight,
        arrow_badges=base.arrow_badges + extra.arrow_badges,
    )


def annotation_texts(
    annotations: FrameAnnotations,
    frame_size: tuple[int, int] | None = None,
    style: Style | None = None,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Every text the annotations draw, as (kind, text, evidence ids), in drawing order.

    Render manifests list these so that every number on screen can be traced to its
    evidence record (texts without digits need none). With ``frame_size`` (width, height)
    the list holds exactly what :func:`draw_annotations` draws on a frame of that size with
    ``style`` (default :class:`Style`): the data card's rows and footer as it lays them out
    (a card that does not fit beside the minimap, the banner or inside the title-safe area
    leaves out its last rows and says so in its footer), the banner's title, time and
    subtitle as fitted (a title on two lines has a line break between them), and no label
    or free text that is left out because it would lie under a panel drawn over it (the
    card, the minimap, the banner and its badges, the watermark). Without it, every label,
    row, the card's footer (with the rows it was given without) and the banner's texts as
    given are listed.
    """
    layout = None
    if frame_size is not None:
        size = list(frame_size) if isinstance(frame_size, Sequence) else []
        if len(size) != 2 or not all(
            isinstance(v, int | np.integer) and not isinstance(v, bool) and v > 0 for v in size
        ):
            raise RenderError(
                "annotation_texts", [f"frame_size must be (width, height) > 0, got {frame_size!r}"]
            )
        resolved = (style or Style()).with_fonts()
        layout = _layout(_geometry(int(size[0]), int(size[1]), resolved), annotations)

    def drawn(label: _Label | None) -> bool:
        return label is not None and label.spot is not None

    texts: list[tuple[str, str, tuple[str, ...]]] = []
    for index, player in enumerate(annotations.players):
        if layout is not None and not drawn(layout.players[index]):
            continue
        if player.label:
            label = player.label + (player.label_held or "")
            texts.append(("player_label", label, player.evidence_ids))
        texts.extend(("badge", badge.text, player.evidence_ids) for badge in player.badges)
    ball = annotations.ball
    if ball is not None and ball.label and (layout is None or drawn(layout.ball)):
        texts.append(("ball_label", ball.label, ball.evidence_ids))
    for index, line in enumerate(annotations.lines):
        if layout is not None and not drawn(layout.lines[index]):
            continue
        if line.label:
            texts.append(("line_label", line.label, line.evidence_ids))
        texts.extend(("badge", badge.text, line.evidence_ids) for badge in line.badges)
    for index, arrow in enumerate(annotations.arrows):
        if layout is not None and not drawn(layout.arrows[index]):
            continue
        if arrow.label:
            texts.append(("arrow_label", arrow.label, arrow.evidence_ids))
        texts.extend(("badge", badge.text, arrow.evidence_ids) for badge in arrow.badges)
    for index, item in enumerate(annotations.texts):
        if layout is None or layout.texts[index]:
            texts.append(("text", item.text, item.evidence_ids))
    card = annotations.card
    if card is not None:
        plan = layout.card if layout is not None else None
        rows, footer = (
            (plan.rows, plan.footer) if plan is not None else (card.rows, _card_footer(card, 0))
        )
        texts.append(("card_title", card.title, ()))
        for row in rows:
            texts.append(("card_row", _card_row_text(row), row.evidence_ids))
            texts.extend(("badge", badge.text, row.evidence_ids) for badge in row.badges)
        if footer:
            texts.append(("card_footer", footer, ()))
    minimap = annotations.minimap
    if minimap is not None:
        if minimap.title:
            texts.append(("minimap_title", minimap.title, ()))
        if minimap.note:
            texts.append(("minimap_note", minimap.note, ()))
    banner = annotations.banner
    if banner is not None:
        fitted = layout.banner if layout is not None else None
        if fitted is not None:  # as drawn: the title's lines, the subtitle as fitted
            title, subtitle = "\n".join(fitted.title), fitted.subtitle
        else:
            title, subtitle = banner.title, banner.subtitle
        texts.append(("banner_title", title, ()))
        if banner.time:
            texts.append(("banner_time", banner.time, ()))
        if subtitle:
            texts.append(("banner_subtitle", subtitle, ()))
    # the frame's badge row: with the layout, the arrows' badges only when an arrow is drawn
    badges = layout.badges if layout is not None else annotations.arrow_badges + annotations.badges
    texts.extend(("badge", badge.text, ()) for badge in badges)
    return texts


# --------------------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------------------
DEFAULT_TEAM_COLOURS: tuple[tuple[str, Colour], ...] = (
    ("left", (108, 182, 240)),
    ("right", (46, 78, 168)),
    ("referee", (238, 238, 238)),
    ("goalkeeper", (250, 204, 21)),
)
"""Display colours by team key: light blue for "left" and a visible navy for "right" by
default (pass a Style with other team colours to match the kits), white for referees. Each
mark also gets a dark or light outline, so a dark team colour stays visible on grass."""

DEFAULT_BADGE_COLOURS: tuple[tuple[str, Colour], ...] = (
    ("unverified", (250, 178, 25)),
    ("image_plane", (144, 133, 233)),
    ("reference_3d", (57, 135, 229)),
    ("experimental", (236, 131, 90)),
    ("unavailable", (137, 135, 129)),
    ("official", (12, 163, 12)),
    ("measured", (12, 163, 12)),
    ("private", (208, 59, 59)),
    ("info", (240, 239, 236)),
)
"""Dot colours of badge chips (the status palette of the project's charts): amber for
unverified, violet for image-plane-only, blue for a 3D reference, orange for experimental,
grey for unavailable, green for official/measured, red for private material, near-white for
information."""


@dataclass(frozen=True)
class Style:
    """Colours, fonts and layout of the overlay. Sizes are 1080p pixels (see module doc).

    Attributes:
        team_colours: (team key, colour) pairs; unknown teams use ``neutral``.
        neutral: colour of players without a known team.
        accent: focus ring, pointer, arrows and paths.
        text: main text colour; ``text_secondary`` for subtitles and card labels;
            ``text_held`` for a value a label holds from an earlier frame (greyed).
        panel: RGBA of the semi-transparent label, banner and card panels.
        skeleton_left, skeleton_right, skeleton_centre: limb colours by anatomical side.
        ball, trail: ball ring and ball trail colours.
        badge_colours: (badge kind, dot colour) pairs.
        font, font_bold: fonts to use; ``None`` finds system CJK fonts
            (:func:`gbf.fonts.find_cjk_fonts`). Without a bold font, bold text uses the
            regular font with a thin stroke.
        ui_scale: overrides frame height / 1080.
        supersample: anti-aliasing factor (1 to 4).
        minimap_corner: "br", "bl", "tr" or "tl", not the banner's corner ("bl" with a
            bottom_left banner, "tl" with a top_left one).
        minimap_width: minimap pitch width in pixels (3.2 px per metre by default).
        banner_position: "bottom_left" (lower third) or "top_left".
        watermark: text drawn at the top centre of every frame, e.g. "私有素材 · 禁止分享".
        watermark_repeat: also draw the watermark, fainter, at the bottom centre (in the
            margin below the title-safe area, clear of every panel), so that a frame cut
            off at the top still carries it.
        safe_margin: fraction of the frame width/height kept free at the edges (title-safe).
    """

    team_colours: tuple[tuple[str, Colour], ...] = DEFAULT_TEAM_COLOURS
    neutral: Colour = (203, 213, 225)
    accent: Colour = (255, 214, 10)
    text: Colour = (255, 255, 255)
    text_secondary: Colour = (203, 213, 225)
    text_held: Colour = (140, 148, 160)
    panel: RGBA = (10, 14, 22, 190)
    skeleton_left: Colour = (255, 159, 67)
    skeleton_right: Colour = (72, 219, 251)
    skeleton_centre: Colour = (245, 246, 250)
    ball: Colour = (255, 255, 255)
    trail: Colour = (255, 214, 10)
    badge_colours: tuple[tuple[str, Colour], ...] = DEFAULT_BADGE_COLOURS
    font: FontInfo | None = None
    font_bold: FontInfo | None = None
    ui_scale: float | None = None
    supersample: int = 2
    minimap_corner: Literal["br", "bl", "tr", "tl"] = "br"
    minimap_width: float = 336.0
    banner_position: Literal["bottom_left", "top_left"] = "bottom_left"
    watermark: str | None = None
    watermark_repeat: bool = False
    safe_margin: float = 0.05

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name in (
            "neutral",
            "accent",
            "text",
            "text_secondary",
            "text_held",
            "skeleton_left",
            "skeleton_right",
            "skeleton_centre",
            "ball",
            "trail",
        ):
            colour = _colour(name, getattr(self, name), problems)
            if colour is not None:
                object.__setattr__(self, name, colour)
        for name in ("team_colours", "badge_colours"):
            pairs: list[tuple[str, Colour]] = []
            for index, pair in enumerate(getattr(self, name)):
                try:
                    key, value = pair
                except (TypeError, ValueError):
                    problems.append(f"{name}[{index}] must be a (key, colour) pair")
                    continue
                colour = _colour(f"{name}[{index}]", value, problems)
                if not isinstance(key, str):
                    problems.append(f"{name}[{index}] key must be a string")
                elif colour is not None:
                    pairs.append((key, colour))
            object.__setattr__(self, name, tuple(pairs))
        panel = tuple(self.panel)
        if len(panel) != 4 or not all(isinstance(c, int) and 0 <= c <= 255 for c in panel):
            problems.append(f"panel must be four ints in 0..255 (RGBA), got {self.panel!r}")
        for name in ("font", "font_bold"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, FontInfo):
                problems.append(f"{name} must be a gbf.fonts.FontInfo or None")
        if self.ui_scale is not None:
            scale = _finite("ui_scale", self.ui_scale, problems)
            if scale is not None and scale <= 0:
                problems.append(f"ui_scale must be > 0, got {scale}")
        if (
            isinstance(self.supersample, bool)
            or not isinstance(self.supersample, int)
            or not 1 <= self.supersample <= 4
        ):
            problems.append(f"supersample must be an int in 1..4, got {self.supersample!r}")
        if self.minimap_corner not in ("br", "bl", "tr", "tl"):
            problems.append(f"minimap_corner must be br, bl, tr or tl, got {self.minimap_corner!r}")
        width = _finite("minimap_width", self.minimap_width, problems)
        if width is not None and width < 105:
            problems.append(f"minimap_width must be >= 105 px (1 px per metre), got {width}")
        if self.banner_position not in ("bottom_left", "top_left"):
            problems.append(
                f"banner_position must be bottom_left or top_left, got {self.banner_position!r}"
            )
        elif (self.minimap_corner, self.banner_position) in (
            ("bl", "bottom_left"),
            ("tl", "top_left"),
        ):
            problems.append(
                f"minimap_corner {self.minimap_corner!r} and banner_position "
                f"{self.banner_position!r} share a corner: the banner and its badges would "
                "cover the minimap's text"
            )
        _optional_text("watermark", self.watermark, problems)
        if not isinstance(self.watermark_repeat, bool):
            problems.append(f"watermark_repeat must be a bool, got {self.watermark_repeat!r}")
        margin = _finite("safe_margin", self.safe_margin, problems)
        if margin is not None and not 0.0 <= margin < 0.25:
            problems.append(f"safe_margin must be in [0, 0.25), got {margin}")
        _raise("Style", problems)

    def team_colour(self, team: str | None) -> Colour:
        """Display colour of ``team`` (``neutral`` when unknown or None)."""
        for key, colour in self.team_colours:
            if key == team:
                return colour
        return self.neutral

    def badge_colour(self, kind: str) -> Colour:
        for key, colour in self.badge_colours:
            if key == kind:
                return colour
        return self.text

    def with_fonts(self) -> Style:
        """This style with ``font`` (and ``font_bold`` when one exists) filled in."""
        if self.font is not None:
            return self
        regular, bold = _system_fonts()
        return replace(self, font=regular, font_bold=self.font_bold or bold)

    def manifest(self) -> dict[str, Any]:
        """JSON form for render manifests (fonts by name and sha256, no paths)."""
        resolved = self.with_fonts()
        return {
            "team_colours": [[key, list(colour)] for key, colour in self.team_colours],
            "accent": list(self.accent),
            "panel": list(self.panel),
            "supersample": self.supersample,
            "ui_scale": self.ui_scale,
            "minimap_corner": self.minimap_corner,
            "minimap_width": self.minimap_width,
            "banner_position": self.banner_position,
            "watermark": self.watermark,
            "watermark_repeat": self.watermark_repeat,
            "font": resolved.font.manifest() if resolved.font is not None else None,
            "font_bold": resolved.font_bold.manifest() if resolved.font_bold else None,
        }


@lru_cache(maxsize=1)
def _system_fonts() -> tuple[FontInfo, FontInfo | None]:
    return find_cjk_fonts()


def _style_font(style: Style, size_px: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    """The style's font at ``size_px`` canvas pixels: the bold font for bold text when the
    style has one, else the regular font."""
    info = style.font_bold if bold and style.font_bold is not None else None
    if info is None:
        info = style.font
    if info is None:  # pragma: no cover - draw_annotations resolves fonts first
        raise RenderError("draw", ["no font resolved"])
    return load_font(info, size_px)


# --------------------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------------------
@dataclass
class _Geometry:
    """The supersampled canvas of one frame without its pixels: its size and the coordinate
    and size conversions. The overlay's layout is computed on it (:func:`_layout`)."""

    s: int
    ui: float
    width: int
    height: int
    style: Style

    @property
    def canvas_w(self) -> int:
        return self.width * self.s

    @property
    def canvas_h(self) -> int:
        return self.height * self.s

    def p(self, u: float, v: float) -> tuple[float, float]:
        """Frame pixel coordinates (centres at integers) to canvas coordinates."""
        half = (self.s - 1) / 2.0
        return (u * self.s + half, v * self.s + half)

    def k(self, size_1080: float) -> float:
        """A 1080p size in canvas pixels."""
        return size_1080 * self.ui * self.s

    def w(self, size_1080: float) -> int:
        """A 1080p line width in whole canvas pixels (at least 1)."""
        return max(1, int(round(self.k(size_1080))))

    def font(self, size_1080: float, bold: bool = False) -> ImageFont.FreeTypeFont:
        return _style_font(self.style, self.k(size_1080), bold)

    def faux_bold(self, bold: bool) -> int:
        """Stroke width that imitates bold when no bold font exists."""
        return max(1, int(round(self.k(0.6)))) if bold and self.style.font_bold is None else 0

    @property
    def margin_x(self) -> float:
        return self.width * self.style.safe_margin * self.s

    @property
    def margin_y(self) -> float:
        return self.height * self.style.safe_margin * self.s


def _geometry(width: int, height: int, style: Style) -> _Geometry:
    """The canvas of a ``width`` x ``height`` frame drawn with ``style``."""
    ui = style.ui_scale if style.ui_scale is not None else height / 1080.0
    return _Geometry(s=style.supersample, ui=ui, width=width, height=height, style=style)


@dataclass
class _Canvas(_Geometry):
    """The supersampled frame being drawn on, with its geometry."""

    image: Image.Image
    draw: ImageDraw.ImageDraw


#: Measures text boxes for the layout exactly as a canvas's RGBA draw does (Pillow's text
#: box depends on the draw's mode, not on its image).
_MEASURE = ImageDraw.Draw(Image.new("RGBA", (1, 1)))


def _rgba(colour: Colour, alpha: int) -> RGBA:
    return (colour[0], colour[1], colour[2], max(0, min(255, int(alpha))))


def _text_width(font: ImageFont.FreeTypeFont, text: str) -> float:
    return float(font.getlength(text))


def _line_height(font: ImageFont.FreeTypeFont) -> float:
    ascent, descent = font.getmetrics()
    return float(ascent + descent)


def _fit(font: ImageFont.FreeTypeFont, text: str, max_width: float) -> str:
    """``text`` cut with an ellipsis so that it is at most ``max_width`` wide."""
    if _text_width(font, text) <= max_width:
        return text
    ellipsis = "…"
    cut = text
    while cut and _text_width(font, cut + ellipsis) > max_width:
        cut = cut[:-1]
    return cut + ellipsis if cut else ellipsis


def _text(
    c: _Canvas,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: RGBA,
    anchor: str = "la",
    stroke: int = 0,
    stroke_fill: RGBA | None = None,
) -> None:
    """Draw text; a fill alpha below 255 is honoured through a mask (Pillow draws text
    opaque even on a blending draw)."""
    if fill[3] >= 255:
        c.draw.text(
            xy,
            text,
            fill=fill,
            font=font,
            anchor=anchor,
            stroke_width=stroke,
            stroke_fill=stroke_fill if stroke_fill is not None else fill,
        )
        return
    left, top, right, bottom = c.draw.textbbox(
        xy, text, font=font, anchor=anchor, stroke_width=stroke
    )
    x0, y0 = int(math.floor(left)), int(math.floor(top))
    width, height = int(math.ceil(right)) - x0 + 1, int(math.ceil(bottom)) - y0 + 1
    if width <= 0 or height <= 0:
        return
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).text(
        (xy[0] - x0, xy[1] - y0), text, fill=255, font=font, anchor=anchor, stroke_width=stroke
    )
    mask = mask.point(lambda value: value * fill[3] // 255)
    c.image.paste(fill[:3], (x0, y0, x0 + width, y0 + height), mask)


def _dashed(
    c: _Canvas,
    a: tuple[float, float],
    b: tuple[float, float],
    fill: RGBA,
    width: int,
    dash: float,
    gap: float,
) -> None:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length <= 0:
        return
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    position = 0.0
    while position < length:
        end = min(length, position + dash)
        c.draw.line(
            [(a[0] + ux * position, a[1] + uy * position), (a[0] + ux * end, a[1] + uy * end)],
            fill=fill,
            width=width,
        )
        position = end + gap


def _chip_width(c: _Geometry, badge: Badge, font: ImageFont.FreeTypeFont) -> float:
    """Width in canvas pixels of a badge chip (of its text alone when compact)."""
    text_w = _text_width(font, badge.text)
    if badge.compact:
        return c.k(2) + text_w
    pad_x, dot_r, gap = c.k(7), c.k(4), c.k(5)
    return pad_x + 2 * dot_r + gap + text_w + pad_x


def _chip(
    c: _Canvas,
    x: float,
    y_mid: float,
    badge: Badge,
    font: ImageFont.FreeTypeFont,
    background: RGBA,
) -> float:
    """Draw a badge chip whose left edge is ``x`` and vertical centre ``y_mid``; returns
    its width in canvas pixels. A compact badge is its text in the kind's colour."""
    width = _chip_width(c, badge, font)
    if badge.compact:
        colour = _rgba(c.style.badge_colour(badge.kind), 255)
        _text(c, (x + c.k(2), y_mid), badge.text, font, colour, "lm")
        return width
    pad_x, dot_r, gap = c.k(7), c.k(4), c.k(5)
    height = _line_height(font) + c.k(4)
    c.draw.rounded_rectangle(
        [x, y_mid - height / 2, x + width, y_mid + height / 2],
        radius=height / 2,
        fill=background,
    )
    dot_x = x + pad_x + dot_r
    c.draw.ellipse(
        [dot_x - dot_r, y_mid - dot_r, dot_x + dot_r, y_mid + dot_r],
        fill=_rgba(c.style.badge_colour(badge.kind), 255),
    )
    _text(c, (dot_x + dot_r + gap, y_mid), badge.text, font, _rgba(c.style.text, 255), "lm")
    return width


def _striped_panel(
    c: _Canvas,
    box: tuple[float, float, float, float],
    radius: float,
    stripe: Colour | None,
    stripe_width: float,
) -> None:
    """A rounded semi-transparent panel; with ``stripe``, its left ``stripe_width`` pixels
    are an opaque coloured stripe. The two parts do not overlap, so the translucent panel
    never tints the stripe."""
    x0, y0, x1, y1 = box
    if stripe is None or stripe_width <= 0:
        c.draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=c.style.panel)
        return
    split = int(round(x0 + stripe_width))
    corner = min(radius, stripe_width, (y1 - y0) / 2)
    c.draw.rounded_rectangle(
        [x0, y0, split - 1, y1],
        radius=corner,
        fill=_rgba(stripe, 255),
        corners=(True, False, False, True),
    )
    c.draw.rounded_rectangle(
        [split, y0, x1, y1], radius=radius, fill=c.style.panel, corners=(False, True, True, False)
    )


def _chips_width(c: _Geometry, badges: Sequence[Badge], font: ImageFont.FreeTypeFont) -> float:
    return sum(c.k(6) + _chip_width(c, badge, font) for badge in badges)


def _overlaps(box: tuple[float, float, float, float], others: Sequence[tuple[float, ...]]) -> bool:
    x0, y0, x1, y1 = box
    return any(x0 < o[2] and o[0] < x1 and y0 < o[3] and o[1] < y1 for o in others)


def _clamp_spot(
    g: _Geometry, x: float, y: float, width: float, height: float
) -> tuple[float, float]:
    """The top-left corner (x, y) of a label of this size moved inside the title-safe area."""
    x = min(max(x, g.margin_x * 0.5), g.canvas_w - g.margin_x * 0.5 - width)
    y = min(max(y, g.margin_y * 0.5), g.canvas_h - g.margin_y * 0.5 - height)
    return x, y


def _free_spot(
    g: _Geometry,
    anchor: tuple[float, float],
    x0: float,
    y0: float,
    width: float,
    height: float,
    place: str,
    occupied: Sequence[tuple[float, float, float, float]],
    panels: Sequence[tuple[float, float, float, float]],
    keep_off: Sequence[tuple[float, float, float, float]] = (),
    *,
    strict: bool = False,
    beside: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Top-left corner for a label of this size whose wanted spot is (``x0``, ``y0``), or
    None when the label is left out.

    The candidates are the wanted spot, kept in the title-safe area, and the spots up to
    three label heights from it (up first for labels above a player); then, except for a
    ``strict`` label, spots beside what the label is about, right and left of ``beside``
    (its canvas x extent, by default the ``anchor``) at the wanted height and, for a label
    above a player, a step lower, next to the head; then spots farther out, up to three
    half label widths across and six steps up or three down, the nearest first (a crowded
    box can leave no room nearer). A candidate under one of ``panels``
    (drawn over the labels, so its text would show through) is never taken; of the
    others, the first that overlaps none of the ``occupied`` boxes (earlier labels, the
    ball) and ``keep_off`` boxes (players a label should not hide), else the first that
    overlaps no ``occupied`` box, else (not for a ``strict`` label, an arrow's, which is
    left out rather than drawn over another label) the first. A label is also left out
    when every candidate lies under a panel, or the ``anchor`` (the point the label is
    about) does."""
    gap = g.k(3)
    u, v = anchor
    if any(p[0] <= u < p[2] and p[1] <= v < p[3] for p in panels):
        return None
    step = height + gap
    offsets = (0, -1, -2, -3, 1, 2) if place == "above" else (0, 1, -1, 2, -2, 3)
    spots = [_clamp_spot(g, x0, y0 + offset * step, width, height) for offset in offsets]
    if not strict:
        left, right = beside if beside is not None else (u, u)
        levels = (0, 1) if place == "above" else (0,)
        spots += [
            _clamp_spot(g, x, y0 + level * step, width, height)
            for level in levels
            for x in (right + 2 * gap, left - 2 * gap - width)
        ]
        # then farther out, the nearest first: half a label width across, a step up or down
        farther = [
            (x0 + across * width / 2, y0 + down * step)
            for across in range(-3, 4)
            for down in range(-6, 4)
        ]
        farther.sort(key=lambda spot: math.hypot(spot[0] - x0, spot[1] - y0))
        spots += [_clamp_spot(g, x, y, width, height) for x, y in farther]
    clear = [(x, y) for x, y in spots if not _overlaps((x, y, x + width, y + height), panels)]

    def free(x: float, y: float, boxes: Sequence[tuple[float, float, float, float]]) -> bool:
        return not _overlaps((x - gap, y - gap, x + width + gap, y + height + gap), boxes)

    avoid = [*occupied, *keep_off]
    for x, y in clear:
        if free(x, y, avoid):
            return x, y
    # keeping off the players is a wish, keeping off the other labels is not
    off_labels = next(((x, y) for x, y in clear if free(x, y, occupied)), None)
    if strict or off_labels is not None:
        return off_labels
    return clear[0] if clear else None


@dataclass(frozen=True)
class _Label:
    """A label panel of the overlay as placed on the canvas: text, stripe colour, badges,
    weight and size (1080p), the top-left corner and the panel's size in canvas pixels.
    ``spot`` is None for a label left out (:func:`_free_spot`). ``held`` is drawn after the
    text in a grey; ``leader`` is the canvas point a thin leader line runs to from a label
    moved a step or more off its wanted spot (None for a label in place)."""

    text: str
    stripe: Colour | None
    badges: tuple[Badge, ...]
    bold: bool
    size: float
    spot: tuple[float, float] | None
    width: float
    height: float
    held: str = ""
    leader: tuple[float, float] | None = None

    @property
    def box(self) -> tuple[float, float, float, float] | None:
        if self.spot is None:
            return None
        x0, y0 = self.spot
        return (x0, y0, x0 + self.width, y0 + self.height)


def _label_size(
    g: _Geometry,
    text: str,
    *,
    stripe: Colour | None,
    badges: Sequence[Badge] = (),
    bold: bool = False,
    size: float = 22.0,
    held: str = "",
) -> tuple[float, float]:
    """(width, height) in canvas pixels of a label panel (:func:`_place_label`)."""
    font = g.font(size, bold)
    chip_font = g.font(size * 0.78)
    stripe_w = g.k(5) if stripe is not None else 0.0
    pad_x, pad_y = g.k(9), g.k(5)
    height = _line_height(font) + 2 * pad_y
    text_w = _text_width(font, text) + (_text_width(font, held) if held else 0.0)
    width = stripe_w + pad_x + text_w + _chips_width(g, badges, chip_font) + pad_x
    return width, height


def _place_label(
    g: _Geometry,
    anchor: tuple[float, float],
    text: str,
    occupied: list[tuple[float, float, float, float]],
    panels: Sequence[tuple[float, float, float, float]],
    *,
    stripe: Colour | None,
    badges: Sequence[Badge] = (),
    bold: bool = False,
    size: float = 22.0,
    place: Literal["above", "below", "centre"] = "above",
    about: tuple[float, float] | None = None,
    keep_off: Sequence[tuple[float, float, float, float]] = (),
    strict: bool = False,
    held: str = "",
    beside: tuple[float, float] | None = None,
) -> _Label:
    """A semi-transparent label panel with a coloured stripe on its left edge, placed.

    ``anchor`` (canvas coordinates) is the bottom centre of the panel for "above", its top
    centre for "below" and its centre for "centre". The panel is moved inside the
    title-safe area when needed, off the ``occupied`` boxes, to which it adds its own, off
    the ``keep_off`` boxes where it can be (up or down, or beside the canvas x extent
    ``beside`` of what it is about), and off ``panels``; it is left out when it cannot keep
    clear of the panels (a ``strict`` one also when it cannot keep clear of the
    ``occupied`` boxes), or the point it is about (``about``, by default the anchor) lies
    under one (:func:`_free_spot`). A panel moved a label height or more from its wanted
    spot gets a leader line to that point, so that it is never read as another player's.
    """
    width, height = _label_size(
        g, text, stripe=stripe, badges=badges, bold=bold, size=size, held=held
    )
    x0 = anchor[0] - width / 2
    if place == "above":
        y0 = anchor[1] - height
    elif place == "below":
        y0 = anchor[1]
    else:
        y0 = anchor[1] - height / 2
    point = about if about is not None else anchor
    spot = _free_spot(
        g,
        point,
        x0,
        y0,
        width,
        height,
        place,
        occupied,
        panels,
        keep_off,
        strict=strict,
        beside=beside,
    )
    leader = None
    if spot is not None:
        wanted = _clamp_spot(g, x0, y0, width, height)
        if math.hypot(spot[0] - wanted[0], spot[1] - wanted[1]) >= height:
            leader = (float(point[0]), float(point[1]))
    label = _Label(
        text, stripe, tuple(badges), bold, size, spot, width, height, held=held, leader=leader
    )
    if label.box is not None:
        occupied.append(label.box)
    return label


def _draw_label(c: _Canvas, label: _Label) -> None:
    """Draw a placed label: its panel, text, held text in a grey and badge chips (nothing
    when left out)."""
    if label.spot is None:
        return
    x0, y0 = label.spot
    font = c.font(label.size, label.bold)
    chip_font = c.font(label.size * 0.78)
    stripe_w = c.k(5) if label.stripe is not None else 0.0
    pad_x = c.k(9)
    box = (x0, y0, x0 + label.width, y0 + label.height)
    _striped_panel(c, box, c.k(6), label.stripe, stripe_w)
    mid = y0 + label.height / 2
    x = x0 + stripe_w + pad_x
    _text(c, (x, mid), label.text, font, _rgba(c.style.text, 255), "lm", c.faux_bold(label.bold))
    x += _text_width(font, label.text)
    if label.held:
        grey = _rgba(c.style.text_held, 255)
        _text(c, (x, mid), label.held, font, grey, "lm", c.faux_bold(label.bold))
        x += _text_width(font, label.held)
    x += c.k(6)
    for badge in label.badges:
        x += _chip(c, x, mid, badge, chip_font, (255, 255, 255, 36)) + c.k(6)


def _draw_leader(c: _Canvas, label: _Label) -> None:
    """A thin line from a moved label's panel to the point it is about, with a dot there,
    in the label's stripe colour (nothing for a label in place or left out)."""
    box = label.box
    if box is None or label.leader is None:
        return
    u, v = label.leader
    start = (min(max(u, box[0]), box[2]), min(max(v, box[1]), box[3]))
    if math.hypot(u - start[0], v - start[1]) < c.k(3):
        return
    colour = label.stripe or (255, 255, 255)
    c.draw.line([start, (u, v)], fill=(0, 0, 0, 120), width=c.w(3.2))
    c.draw.line([start, (u, v)], fill=_rgba(colour, 230), width=c.w(1.6))
    radius = c.k(3)
    c.draw.ellipse(
        [u - radius, v - radius, u + radius, v + radius],
        fill=_rgba(colour, 255),
        outline=(0, 0, 0, 150),
        width=c.w(1),
    )


def _ellipse_points(cx: float, cy: float, rx: float, ry: float, n: int = 72) -> list[Point]:
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in angles]


def _stroke(
    c: _Canvas,
    points: Sequence[tuple[float, float]],
    colour: Colour,
    alpha: int,
    width: int,
    *,
    closed: bool = False,
    joint: Literal["curve"] | None = None,
) -> None:
    """A polyline in ``colour`` at uniform opacity ``alpha``.

    Drawn opaque into a mask that covers only its bounding box and composited once, so a
    translucent line does not darken where its segments overlap at the joints.
    """
    pts = list(points)
    if len(pts) < 2 or width < 1:
        return
    pad = width + 2
    canvas_w, canvas_h = c.image.size
    x0 = max(0, int(math.floor(min(x for x, _ in pts) - pad)))
    y0 = max(0, int(math.floor(min(y for _, y in pts) - pad)))
    x1 = min(canvas_w, int(math.ceil(max(x for x, _ in pts) + pad)))
    y1 = min(canvas_h, int(math.ceil(max(y for _, y in pts) + pad)))
    if x1 <= x0 or y1 <= y0:
        return
    local = [(x - x0, y - y0) for x, y in pts]
    if closed:
        local += local[:2]
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(mask).line(local, fill=max(0, min(255, int(alpha))), width=width, joint=joint)
    c.image.paste(colour, (x0, y0, x1, y1), mask)


def _outline_for(colour: Colour) -> Colour:
    """Black for light colours, white for dark ones: the halo that keeps a mark visible."""
    luminance = 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]
    return (255, 255, 255) if luminance < 100 else (0, 0, 0)


def _touches_bottom(c: _Canvas, box: tuple[float, float, float, float]) -> bool:
    """True when ``box`` (frame pixels) ends at the bottom edge of the frame, within
    :data:`EDGE_TOUCH_PER_BOX_HEIGHT` of its height (at least 2 pixels at 1080p)."""
    _x1, y1, _x2, y2 = box
    return y2 >= c.height - 0.5 - max(EDGE_TOUCH_PER_BOX_HEIGHT * (y2 - y1), 2.0 * c.ui)


def _draw_player_ring(c: _Canvas, mark: PlayerMark) -> None:
    x1, y1, x2, y2 = mark.box
    colour = mark.colour or c.style.team_colour(mark.team)
    ring = mark.ring
    if ring is not None and all(math.isfinite(u) and math.isfinite(v) for u, v in ring):
        frame_points = list(ring)
    else:
        if y2 - y1 > CLOSEUP_BOX_SHARE * c.height or _touches_bottom(c, mark.box):
            return  # a close-up, or the feet are out of the picture (CLOSEUP_BOX_SHARE)
        box_w = x2 - x1
        rx = max(RING_RX_PER_BOX_WIDTH * box_w, 10.0 * c.ui)
        ry = max(rx * RING_RY_PER_RX, 4.0 * c.ui)
        frame_points = _ellipse_points((x1 + x2) / 2.0, y2, rx, ry)
    points = [c.p(u, v) for u, v in frame_points]
    halo = _outline_for(colour)
    if mark.emphasis == "focus":
        _stroke(c, points, c.style.accent, 70, c.w(9), closed=True)
        _stroke(c, points, (0, 0, 0), 120, c.w(5), closed=True)
        _stroke(c, points, c.style.accent, 255, c.w(3), closed=True)
        cx = sum(u for u, _ in frame_points) / len(frame_points)
        cy = sum(v for _, v in frame_points) / len(frame_points)
        inner = [c.p(cx + (u - cx) * 0.74, cy + (v - cy) * 0.74) for u, v in frame_points]
        _stroke(c, inner, halo, 110, c.w(4), closed=True)
        _stroke(c, inner, colour, 240, c.w(2), closed=True)
    elif mark.emphasis == "normal":
        _stroke(c, points, halo, 120, c.w(4.5), closed=True)
        _stroke(c, points, colour, 235, c.w(2.4), closed=True)
    else:
        _stroke(c, points, colour, 140, c.w(1.6), closed=True)


def _draw_brackets(c: _Canvas, mark: PlayerMark) -> None:
    if mark.emphasis == "context":
        return
    x1, y1, x2, y2 = mark.box
    focus = mark.emphasis == "focus"
    colour = c.style.accent if focus else (mark.colour or c.style.team_colour(mark.team))
    alpha = 255 if focus else 200
    width = c.w(3 if focus else 2)
    halo = _outline_for(colour)
    pad = 3.0 * c.ui
    left, top, right, bottom = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    arm = min(max(0.22 * min(right - left, bottom - top), 5.0 * c.ui), 26.0 * c.ui)
    corners = (
        ((left, top + arm), (left, top), (left + arm, top)),
        ((right - arm, top), (right, top), (right, top + arm)),
        ((left, bottom - arm), (left, bottom), (left + arm, bottom)),
        ((right - arm, bottom), (right, bottom), (right, bottom - arm)),
    )
    for corner in corners:
        points = [c.p(u, v) for u, v in corner]
        _stroke(c, points, halo, 110, width + c.w(2), joint="curve")
        _stroke(c, points, colour, alpha, width, joint="curve")


def _player_label(
    g: _Geometry,
    mark: PlayerMark,
    occupied: list[tuple[float, float, float, float]],
    panels: Sequence[tuple[float, float, float, float]],
    bodies: Sequence[tuple[float, float, float, float]] = (),
) -> _Label | None:
    """The label above a player, placed (None for a mark without a label), off the
    ``bodies`` (canvas boxes of the players a label should not hide) where it can be: up,
    or beside the player's head."""
    if not mark.label:
        return None
    x1, y1, x2, _y2 = mark.box
    colour = mark.colour or g.style.team_colour(mark.team)
    top = g.p((x1 + x2) / 2.0, y1 - 6.0 * g.ui)
    bold = mark.emphasis == "focus"
    return _place_label(
        g,
        top,
        mark.label,
        occupied,
        panels,
        stripe=colour,
        badges=mark.badges,
        bold=bold,
        keep_off=bodies,
        held=mark.label_held or "",
        beside=(g.p(x1, y1)[0], g.p(x2, y1)[0]),
    )


def _draw_player_label(c: _Canvas, mark: PlayerMark, label: _Label | None) -> None:
    """The player's label and, for the focus, the pointer above it (above the head when
    the player has no label, or its label is left out)."""
    x1, y1, x2, _y2 = mark.box
    top = c.p((x1 + x2) / 2.0, y1 - 6.0 * c.ui)
    focus = mark.emphasis == "focus"
    if label is not None and label.spot is not None:
        _draw_label(c, label)
        pointer_y = label.spot[1]
    else:
        pointer_y = top[1]
    if focus:
        tip_x = top[0]
        size = c.k(9)
        tip_y = pointer_y - c.k(4)
        triangle = [(tip_x - size, tip_y - size * 1.2), (tip_x + size, tip_y - size * 1.2)]
        triangle.append((tip_x, tip_y))
        if tip_y - size * 1.2 > c.margin_y * 0.5:
            c.draw.polygon(triangle, fill=_rgba(c.style.accent, 255), outline=(0, 0, 0, 140))


def _skeleton_colour(c: _Canvas, skeleton: Skeleton, side: str) -> Colour:
    if skeleton.colour is not None:
        return skeleton.colour
    if side == "left":
        return c.style.skeleton_left
    if side == "right":
        return c.style.skeleton_right
    return c.style.skeleton_centre


def _draw_skeleton(c: _Canvas, skeleton: Skeleton) -> None:
    kp = skeleton.keypoints
    ok = np.isfinite(kp).all(axis=1) & (np.nan_to_num(kp[:, 2], nan=-1.0) >= skeleton.min_score)
    u, v = kp[:, 0], kp[:, 1]
    ok &= (u >= -0.5) & (u <= c.width - 0.5) & (v >= -0.5) & (v <= c.height - 0.5)
    if skeleton.box is not None and _touches_bottom(c, skeleton.box):
        ok[list(HALPE26_LOWER_BODY)] = False
    if not ok.any():
        return
    visible = kp[ok]
    span = float(visible[:, 1].max() - visible[:, 1].min()) if len(visible) > 1 else 0.0
    scale = min(max(span / (170.0 * c.ui), 0.55), 1.5)
    limb_w = c.w(3.2 * scale)
    alpha = 235 if skeleton.emphasis != "context" else 130
    for a, b, side in HALPE26_LIMBS:
        if not (ok[a] and ok[b]):
            continue
        pa, pb = c.p(kp[a, 0], kp[a, 1]), c.p(kp[b, 0], kp[b, 1])
        c.draw.line([pa, pb], fill=(0, 0, 0, alpha // 2), width=limb_w + c.w(2))
        c.draw.line([pa, pb], fill=_rgba(_skeleton_colour(c, skeleton, side), alpha), width=limb_w)
    radius = max(c.k(2.4 * scale), 1.5)
    for index, side in _JOINT_SIDE.items():
        if not ok[index]:
            continue
        x, y = c.p(kp[index, 0], kp[index, 1])
        c.draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=_rgba(_skeleton_colour(c, skeleton, side), alpha),
            outline=(0, 0, 0, alpha // 2),
            width=max(1, c.w(1)),
        )


def _draw_trail(c: _Canvas, ball: BallMark) -> None:
    points: list[Point] = list(ball.trail)
    if ball.position is not None:
        points.append(ball.position)
    n = len(points)
    if n < 2:
        return
    for index in range(n - 1):
        a, b = points[index], points[index + 1]
        if not all(math.isfinite(v) for v in (*a, *b)):
            continue
        t = (index + 1) / (n - 1)
        alpha = int(25 + 205 * t**1.4)
        width = c.w(1.0 + 3.5 * t)
        c.draw.line([c.p(*a), c.p(*b)], fill=_rgba(c.style.trail, alpha), width=width)


def _draw_ball(c: _Canvas, ball: BallMark) -> None:
    if ball.position is None:
        return
    x, y = c.p(*ball.position)
    radius = ball.radius * c.s + c.k(5)
    box = [x - radius, y - radius, x + radius, y + radius]
    if ball.interpolated:
        points = _ellipse_points(x, y, radius, radius, 24)
        for index in range(0, 24, 2):
            c.draw.line([points[index], points[index + 1]], fill=(255, 255, 255, 190), width=c.w(2))
    else:
        c.draw.ellipse(box, outline=(0, 0, 0, 150), width=c.w(4.5))
        c.draw.ellipse(box, outline=_rgba(c.style.ball, 240), width=c.w(2.2))


def _ball_box(g: _Geometry, ball: BallMark) -> tuple[float, float, float, float] | None:
    """Canvas box of the ball ring (labels keep out of it)."""
    if ball.position is None:
        return None
    x, y = g.p(*ball.position)
    radius = ball.radius * g.s + g.k(5) + g.k(3)
    return (x - radius, y - radius, x + radius, y + radius)


def _ball_label(
    g: _Geometry,
    ball: BallMark,
    occupied: list[tuple[float, float, float, float]],
    panels: Sequence[tuple[float, float, float, float]],
) -> _Label | None:
    """The label next to the ball, placed (None without a label or a ball position)."""
    box = _ball_box(g, ball)
    if box is None or not ball.label:
        return None
    anchor = (box[2] + g.k(8), (box[1] + box[3]) / 2)
    return _place_label(
        g, anchor, ball.label, occupied, panels, stripe=None, size=18.0, place="centre"
    )


def _draw_link(c: _Canvas, line: LinkLine) -> None:
    colour = line.colour or (255, 255, 255)
    a, b = c.p(*line.start), c.p(*line.end)
    if line.dashed:
        _dashed(c, a, b, (0, 0, 0, 110), c.w(4.5), c.k(11), c.k(7))
        _dashed(c, a, b, _rgba(colour, 235), c.w(2.4), c.k(11), c.k(7))
    else:
        c.draw.line([a, b], fill=(0, 0, 0, 110), width=c.w(4.5))
        c.draw.line([a, b], fill=_rgba(colour, 235), width=c.w(2.4))
    radius = c.k(4)
    for x, y in (a, b):
        c.draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=_rgba(colour, 255),
            outline=(0, 0, 0, 150),
            width=c.w(1),
        )


def _line_label(
    g: _Geometry,
    line: LinkLine,
    occupied: list[tuple[float, float, float, float]],
    panels: Sequence[tuple[float, float, float, float]],
    bodies: Sequence[tuple[float, float, float, float]] = (),
) -> _Label | None:
    """The label at the middle of a line, placed (None for a line without a label), off
    the ``bodies`` where it can be."""
    if not line.label:
        return None
    a, b = g.p(*line.start), g.p(*line.end)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    colour = line.colour or (255, 255, 255)
    return _place_label(
        g,
        mid,
        line.label,
        occupied,
        panels,
        stripe=colour,
        badges=line.badges,
        size=20.0,
        place="centre",
        keep_off=bodies,
    )


def _draw_arrow(c: _Canvas, arrow: Arrow) -> None:
    colour = arrow.colour or c.style.accent
    a, b = c.p(*arrow.start), c.p(*arrow.end)
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    head = min(max(c.k(arrow.width * 4.0), c.k(12)), length * 0.6)
    shaft_end = (b[0] - ux * head * 0.8, b[1] - uy * head * 0.8)
    width = c.w(arrow.width)
    if arrow.dashed:
        _dashed(c, a, shaft_end, (0, 0, 0, 110), width + c.w(2.5), c.k(13), c.k(8))
        _dashed(c, a, shaft_end, _rgba(colour, 240), width, c.k(13), c.k(8))
    else:
        c.draw.line([a, shaft_end], fill=(0, 0, 0, 110), width=width + c.w(2.5))
        c.draw.line([a, shaft_end], fill=_rgba(colour, 240), width=width)
    px, py = -uy, ux
    base = (b[0] - ux * head, b[1] - uy * head)
    triangle = [
        b,
        (base[0] + px * head * 0.55, base[1] + py * head * 0.55),
        (base[0] - px * head * 0.55, base[1] - py * head * 0.55),
    ]
    c.draw.polygon(triangle, fill=_rgba(colour, 245), outline=(0, 0, 0, 130), width=c.w(1))


def _arrow_label(
    g: _Geometry,
    arrow: Arrow,
    occupied: list[tuple[float, float, float, float]],
    panels: Sequence[tuple[float, float, float, float]],
    bodies: Sequence[tuple[float, float, float, float]] = (),
) -> _Label | None:
    """The label just past an arrow's head, in the arrow's direction, placed (None without
    a label): clear of the arrow, and of the ``bodies`` (the players' boxes) where it can
    be, so that it hides neither the direction it names nor the player. An arrow that
    points up the image (more up than sideways) gets its label below its start instead: an
    arrow on the grass from a player's feet points up into the player. The label is left
    out when the head lies under a panel, or no spot keeps clear of the other labels."""
    if not arrow.label:
        return None
    colour = arrow.colour or g.style.accent
    a, b = g.p(*arrow.start), g.p(*arrow.end)
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    size = 20.0
    width, height = _label_size(g, arrow.label, stripe=colour, badges=arrow.badges, size=size)
    gap = g.k(6)
    if uy < -0.5:
        anchor = (a[0], a[1] + height / 2 + gap)
    else:
        # the distance from the panel's centre to its edge along the arrow, plus a gap
        reach = min(
            width / 2 / abs(ux) if abs(ux) > 1e-9 else math.inf,
            height / 2 / abs(uy) if abs(uy) > 1e-9 else math.inf,
        )
        anchor = (b[0] + ux * (reach + gap), b[1] + uy * (reach + gap))
    return _place_label(
        g,
        anchor,
        arrow.label,
        occupied,
        panels,
        stripe=colour,
        badges=arrow.badges,
        size=size,
        place="centre",
        about=b,
        keep_off=bodies,
        strict=True,
    )


def _text_box(g: _Geometry, item: TextLabel) -> tuple[float, float, float, float]:
    """Canvas box of a free text: its panel, or the outlined text without one."""
    font = g.font(item.size, item.bold)
    xy = g.p(*item.position)
    if item.background:
        left, top, right, bottom = _MEASURE.textbbox(xy, item.text, font=font, anchor=item.anchor)
        pad_x, pad_y = g.k(8), g.k(4)
        return (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y)
    stroke = max(g.w(1.5), g.faux_bold(item.bold))
    return _MEASURE.textbbox(xy, item.text, font=font, anchor=item.anchor, stroke_width=stroke)


def _draw_text(c: _Canvas, item: TextLabel) -> None:
    font = c.font(item.size, item.bold)
    xy = c.p(*item.position)
    if item.background:
        c.draw.rounded_rectangle(list(_text_box(c, item)), radius=c.k(6), fill=c.style.panel)
        _text(c, xy, item.text, font, _rgba(item.colour, 255), item.anchor, c.faux_bold(item.bold))
    else:
        _text(
            c,
            xy,
            item.text,
            font,
            _rgba(item.colour, 255),
            item.anchor,
            stroke=max(c.w(1.5), c.faux_bold(item.bold)),
            stroke_fill=(0, 0, 0, 255),
        )


#: Units written right after the number, without a space and in the number's font (90°,
#: 50%); every other unit follows the number after a gap, in a smaller font.
TIGHT_UNITS: tuple[str, ...] = ("°", "%")


def _tight_unit(unit: str) -> bool:
    return unit.startswith(TIGHT_UNITS)


def _card_row_text(row: CardRow) -> str:
    """The text of a card row as render manifests list it: "<label> <value> <unit>", with
    no space before a unit of :data:`TIGHT_UNITS` ("角度 90°")."""
    value = f"{row.value}{row.unit}" if _tight_unit(row.unit) else f"{row.value} {row.unit}"
    return f"{row.label} {value.strip()}".strip()


@dataclass(frozen=True)
class _CardMetrics:
    """Fonts and sizes of the data card in canvas pixels."""

    title_font: ImageFont.FreeTypeFont
    label_font: ImageFont.FreeTypeFont
    value_font: ImageFont.FreeTypeFont
    unit_font: ImageFont.FreeTypeFont
    chip_font: ImageFont.FreeTypeFont
    footer_font: ImageFont.FreeTypeFont
    width: float
    pad: float
    stripe: float
    gap: float
    title_h: float
    label_h: float
    value_ascent: float
    row_h: float


@dataclass(frozen=True)
class _CardPlan:
    """Where the data card goes on the canvas and what it shows: the panel ``box``
    (x0, y0, x1, y1), the title, the first rows of the card that fit, and the footer as
    drawn (cut with "…" to the card's width)."""

    box: tuple[float, float, float, float]
    title: str
    rows: tuple[CardRow, ...]
    footer: str | None
    metrics: _CardMetrics


def _card_footer(card: InfoCard, dropped: int, note_first: bool = False) -> str | None:
    """The footer of ``card`` when it leaves out its last ``dropped`` rows: its own footer
    and how many rows the report has beyond those shown, counting the rows the card was
    given without (:attr:`InfoCard.omitted`), the count last or with ``note_first`` first."""
    more = card.omitted + dropped
    if not more:
        return card.footer
    note = f"另有 {more} 项见报告"
    if not card.footer:
        return note
    return f"{note}；{card.footer}" if note_first else f"{card.footer}；{note}"


def _plan_card(
    g: _Geometry,
    annotations: FrameAnnotations,
    panels: Sequence[tuple[float, float, float, float]] = (),
) -> _CardPlan | None:
    """The layout of the data card on the canvas ``g``, part of :func:`_layout` (so a
    render manifest lists exactly the rows that are drawn). None without a card.

    The card sits on its side, centred a little above the middle of the frame and at least
    110 px (1080p) below the top title-safe margin. It keeps clear of the ``panels`` drawn
    over it (the minimap, the banner and its badge row) on its side, ending above those in
    the lower half of the frame and starting below those in the upper half; when it still
    does not fit, or runs past the bottom title-safe margin, it drops its last rows. The
    footer says how many rows the report has beyond those drawn, these and the ones the card
    was given without ("另有 N 项见报告", :func:`_card_footer`), first when the footer is too
    long for the card and cut with "…". The style of ``g`` must have its fonts resolved.
    """
    card = annotations.card
    if card is None:
        return None
    style = g.style
    unit = g.k(1)
    canvas_w, canvas_h = g.canvas_w, g.canvas_h
    margin_x, margin_y = g.margin_x, g.margin_y
    label_font = _style_font(style, 19 * unit)
    value_font = _style_font(style, 32 * unit, bold=True)
    title_font = _style_font(style, 28 * unit, bold=True)
    footer_font = _style_font(style, 15 * unit)
    label_h = _line_height(label_font)
    gap = 8 * unit
    metrics = _CardMetrics(
        title_font=title_font,
        label_font=label_font,
        value_font=value_font,
        unit_font=_style_font(style, 20 * unit),
        chip_font=_style_font(style, 16 * unit),
        footer_font=footer_font,
        width=560 * unit,
        pad=18 * unit,
        stripe=6 * unit,
        gap=gap,
        title_h=_line_height(title_font) + 8 * unit,
        label_h=label_h,
        value_ascent=float(value_font.getmetrics()[0]),
        row_h=gap + label_h + _line_height(value_font) + gap,
    )
    m = metrics
    x0 = canvas_w - margin_x - m.width if card.side == "right" else margin_x
    top_min = margin_y + 110 * unit
    bottom_max = canvas_h - margin_y
    beside = [box for box in panels if x0 < box[2] and x0 + m.width > box[0]]  # on its side
    below = [box for box in beside if box[1] > canvas_h / 2]
    above = [box for box in beside if box[1] <= canvas_h / 2]
    clear = 12 * unit
    n = len(card.rows)
    for dropped in range(n + 1):
        footer = _card_footer(card, dropped)
        footer_h = _line_height(footer_font) + 10 * unit if footer else 0.0
        card_h = m.pad + m.title_h + m.row_h * (n - dropped) + footer_h + m.pad
        y0 = max(top_min, (canvas_h - card_h) / 2 - 40 * unit)
        end = bottom_max
        if below:  # panels in the lower half: end above them
            end = min(bottom_max, min(box[1] for box in below) - clear)
            y0 = max(top_min, min(y0, end - card_h))
        if above:  # in the upper half: start below them
            y0 = max(y0, max(box[3] for box in above) + clear)
        if y0 + card_h <= end:
            break
    inner_w = m.width - m.stripe - 2 * m.pad
    if footer and footer != card.footer and _text_width(footer_font, footer) > inner_w:
        footer = _card_footer(card, dropped, note_first=True)  # the count is never cut off
    return _CardPlan(
        box=(x0, y0, x0 + m.width, y0 + card_h),
        title=card.title,
        rows=tuple(card.rows[: n - dropped]),
        footer=_fit(footer_font, footer, inner_w) if footer else None,  # as drawn
        metrics=metrics,
    )


def _draw_card(c: _Canvas, plan: _CardPlan) -> None:
    """The data card as ``plan`` lays it out: title, then per row a small label line above
    a large value + unit, with the row's badges right-aligned on the label line, then the
    footer."""
    m = plan.metrics
    x0, y0, x1, y1 = plan.box
    _striped_panel(c, (x0, y0, x1, y1), c.k(12), c.style.accent, m.stripe)
    inner_x = x0 + m.stripe + m.pad
    inner_w = m.width - m.stripe - 2 * m.pad
    y = y0 + m.pad
    _text(
        c,
        (inner_x, y),
        _fit(m.title_font, plan.title, inner_w),
        m.title_font,
        _rgba(c.style.text, 255),
        "la",
        stroke=c.faux_bold(True),
    )
    y += m.title_h
    for row in plan.rows:
        c.draw.line([(inner_x, y), (inner_x + inner_w, y)], fill=(255, 255, 255, 46), width=c.w(1))
        chips_w = _chips_width(c, row.badges, m.chip_font)
        label_top = y + m.gap
        _text(
            c,
            (inner_x, label_top),
            _fit(m.label_font, row.label, max(inner_w - chips_w - c.k(8), c.k(40))),
            m.label_font,
            _rgba(c.style.text_secondary, 255),
            "la",
        )
        x = inner_x + inner_w - chips_w
        for badge in row.badges:
            x += c.k(6)
            x += _chip(c, x, label_top + m.label_h / 2, badge, m.chip_font, (255, 255, 255, 36))
        baseline = label_top + m.label_h + m.value_ascent
        tight = _tight_unit(row.unit)
        if tight:  # 90°: the unit is part of the number
            unit_w = _text_width(m.value_font, row.unit)
        else:
            unit_w = _text_width(m.unit_font, row.unit) + c.k(8) if row.unit else 0.0
        value_text = _fit(m.value_font, row.value, inner_w - unit_w)
        _text(
            c,
            (inner_x, baseline),
            value_text + row.unit if tight else value_text,
            m.value_font,
            _rgba(c.style.text, 255),
            "ls",
            stroke=c.faux_bold(True),
        )
        if row.unit and not tight:
            _text(
                c,
                (inner_x + _text_width(m.value_font, value_text) + c.k(8), baseline),
                row.unit,
                m.unit_font,
                _rgba(c.style.text_secondary, 255),
                "ls",
            )
        y += m.row_h
    if plan.footer:
        _text(
            c,
            (inner_x, y + c.k(8)),
            _fit(m.footer_font, plan.footer, inner_w),
            m.footer_font,
            _rgba(c.style.text_secondary, 255),
            "la",
        )


def _pitch_lines() -> list[tuple[str, Any]]:
    """Markings of a 105 x 68 m pitch in metres (SoccerNet), as drawing primitives."""
    half_l, half_w = PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2
    lines: list[tuple[str, Any]] = [
        ("rect", (-half_l, -half_w, half_l, half_w)),
        ("line", ((0.0, -half_w), (0.0, half_w))),
        ("circle", (0.0, 0.0, 9.15)),
        ("dot", (0.0, 0.0)),
    ]
    for sign in (-1.0, 1.0):
        goal_x = sign * half_l
        lines.append(("rect", _span(goal_x, goal_x - sign * 16.5, -20.16, 20.16)))
        lines.append(("rect", _span(goal_x, goal_x - sign * 5.5, -9.16, 9.16)))
        lines.append(("rect", _span(goal_x, goal_x + sign * 2.0, -3.66, 3.66)))
        lines.append(("dot", (goal_x - sign * 11.0, 0.0)))
        # the arc of the penalty area: the part of the 9.15 m circle around the spot that
        # lies outside the box (|x| < 36 m); its half-angle is acos(5.5 / 9.15)
        half_angle = math.degrees(math.acos(5.5 / 9.15))
        centre_angle = 180.0 if sign > 0 else 0.0
        lines.append(
            (
                "arc",
                (
                    goal_x - sign * 11.0,
                    0.0,
                    9.15,
                    centre_angle - half_angle,
                    centre_angle + half_angle,
                ),
            )
        )
    return lines


def _span(x_a: float, x_b: float, y_a: float, y_b: float) -> tuple[float, float, float, float]:
    return (min(x_a, x_b), min(y_a, y_b), max(x_a, x_b), max(y_a, y_b))


def _near_pitch(x: float, y: float, margin_m: float) -> bool:
    """True when (x, y), SoccerNet metres, lies within ``margin_m`` of the pitch lines."""
    return abs(x) <= PITCH_LENGTH_M / 2 + margin_m and abs(y) <= PITCH_WIDTH_M / 2 + margin_m


def _clip_polygon(
    points: Sequence[Point], x_min: float, y_min: float, x_max: float, y_max: float
) -> list[Point]:
    """Sutherland-Hodgman clipping of a polygon against an axis-aligned rectangle."""

    def clip(
        poly: list[Point], inside: Callable[[Point], bool], cut: Callable[[Point, Point], Point]
    ) -> list[Point]:
        out: list[Point] = []
        for index, current in enumerate(poly):
            previous = poly[index - 1]
            if inside(current):
                if not inside(previous):
                    out.append(cut(previous, current))
                out.append(current)
            elif inside(previous):
                out.append(cut(previous, current))
        return out

    def at_x(x: float) -> Callable[[Point, Point], Point]:
        def cut(a: Point, b: Point) -> Point:
            t = (x - a[0]) / (b[0] - a[0])
            return (x, a[1] + t * (b[1] - a[1]))

        return cut

    def at_y(y: float) -> Callable[[Point, Point], Point]:
        def cut(a: Point, b: Point) -> Point:
            t = (y - a[1]) / (b[1] - a[1])
            return (a[0] + t * (b[0] - a[0]), y)

        return cut

    poly = list(points)
    for inside, cut in (
        (lambda p: p[0] >= x_min, at_x(x_min)),
        (lambda p: p[0] <= x_max, at_x(x_max)),
        (lambda p: p[1] >= y_min, at_y(y_min)),
        (lambda p: p[1] <= y_max, at_y(y_max)),
    ):
        if not poly:
            break
        poly = clip(poly, inside, cut)
    return poly


@dataclass(frozen=True)
class MinimapGeometry:
    """Where the minimap sits, in the coordinates it was computed for.

    ``panel`` is (x0, y0, width, height) of the panel, ``scale`` pixels per metre and
    ``origin`` the pixel of the centre spot. :meth:`pixel` maps SoccerNet metres to pixels:
    x grows to the right, y downwards (the far touchline, y = -34, at the top).
    """

    panel: tuple[float, float, float, float]
    scale: float
    origin: tuple[float, float]

    def pixel(self, x: float, y: float) -> tuple[float, float]:
        return (self.origin[0] + x * self.scale, self.origin[1] + y * self.scale)


def _minimap_geometry(
    canvas_w: float,
    canvas_h: float,
    unit: float,
    margin_x: float,
    margin_y: float,
    style: Style,
    has_title: bool,
) -> MinimapGeometry:
    """Minimap layout on a canvas of ``canvas_w`` x ``canvas_h``; ``unit`` is canvas pixels
    per 1080p pixel and the margins are the title-safe margins in canvas pixels."""
    pitch_w = style.minimap_width * unit
    scale = pitch_w / PITCH_LENGTH_M
    pad = 12 * unit
    title_h = 26 * unit if has_title else 0.0
    width = pitch_w + 2 * pad + 6 * unit
    height = PITCH_WIDTH_M * scale + 2 * pad + title_h
    corner = style.minimap_corner
    x0 = canvas_w - margin_x - width if corner in ("br", "tr") else margin_x
    y0 = canvas_h - margin_y - height if corner in ("br", "bl") else margin_y + 120 * unit
    origin = (
        x0 + pad + 3 * unit + PITCH_LENGTH_M / 2 * scale,
        y0 + title_h + pad + PITCH_WIDTH_M / 2 * scale,
    )
    return MinimapGeometry(panel=(x0, y0, width, height), scale=scale, origin=origin)


def minimap_geometry(
    frame_width: int, frame_height: int, minimap: Minimap, style: Style | None = None
) -> MinimapGeometry:
    """The minimap layout in FRAME pixels for a frame of this size (the drawn minimap is
    within a quarter pixel of it), e.g. to place extra marks on the minimap."""
    style = style or Style()
    ui = style.ui_scale if style.ui_scale is not None else frame_height / 1080.0
    return _minimap_geometry(
        frame_width,
        frame_height,
        ui,
        frame_width * style.safe_margin,
        frame_height * style.safe_margin,
        style,
        bool(minimap.title),
    )


def _draw_minimap(c: _Canvas, minimap: Minimap) -> None:
    canvas_w, canvas_h = c.image.size
    geometry = _minimap_geometry(
        canvas_w, canvas_h, c.k(1), c.margin_x, c.margin_y, c.style, bool(minimap.title)
    )
    x0, y0, width, height = geometry.panel
    pad = c.k(12)
    title_h = c.k(26) if minimap.title else 0.0
    c.draw.rounded_rectangle(
        [x0, y0, x0 + width, y0 + height], radius=c.k(10), fill=(8, 22, 14, 200)
    )
    m = geometry.pixel

    if minimap.title:
        font = c.font(15)
        _text(
            c,
            (x0 + pad, y0 + pad * 0.5 + title_h / 2),
            _fit(font, minimap.title, width - 2 * pad),
            font,
            _rgba(c.style.text_secondary, 255),
            "lm",
        )
    pitch_box = [
        *m(-PITCH_LENGTH_M / 2, -PITCH_WIDTH_M / 2),
        *m(PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2),
    ]
    c.draw.rectangle(pitch_box, fill=(34, 102, 58, 150))
    if minimap.view:
        clipped = _clip_polygon(
            minimap.view,
            -PITCH_LENGTH_M / 2 - 3,
            -PITCH_WIDTH_M / 2 - 3,
            PITCH_LENGTH_M / 2 + 3,
            PITCH_WIDTH_M / 2 + 3,
        )
        if len(clipped) >= 3:
            c.draw.polygon(
                [m(x, y) for x, y in clipped],
                fill=(255, 255, 255, 34),
                outline=(255, 255, 255, 110),
                width=c.w(1),
            )
    line_fill = (255, 255, 255, 190)
    line_w = c.w(1.3)
    for kind, spec in _pitch_lines():
        if kind == "rect":
            xa, ya, xb, yb = spec
            c.draw.rectangle([*m(xa, ya), *m(xb, yb)], outline=line_fill, width=line_w)
        elif kind == "line":
            c.draw.line([m(*spec[0]), m(*spec[1])], fill=line_fill, width=line_w)
        elif kind == "circle":
            cx, cy, r = spec
            c.draw.ellipse(
                [*m(cx - r, cy - r), *m(cx + r, cy + r)], outline=line_fill, width=line_w
            )
        elif kind == "dot":
            cx, cy = m(*spec)
            r = c.k(1.6)
            c.draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=line_fill)
        elif kind == "arc":
            cx, cy, r, start, end = spec
            c.draw.arc(
                [*m(cx - r, cy - r), *m(cx + r, cy + r)], start, end, fill=line_fill, width=line_w
            )
    for path in minimap.paths:
        if len(path) >= 2:
            c.draw.line(
                [m(x, y) for x, y in path],
                fill=_rgba(c.style.accent, 230),
                width=c.w(2),
                joint="curve",
            )
    if minimap.note:
        font = c.font(17)
        centre = m(0.0, 0.0)
        text = _fit(font, minimap.note, width - 4 * pad)
        left, top, right, bottom = c.draw.textbbox(centre, text, font=font, anchor="mm")
        c.draw.rounded_rectangle(
            [left - c.k(10), top - c.k(6), right + c.k(10), bottom + c.k(6)],
            radius=c.k(6),
            fill=(0, 0, 0, 170),
        )
        _text(c, centre, text, font, _rgba(c.style.text, 255), "mm")
    ordered = sorted(minimap.players, key=lambda dot: EMPHASES.index(dot.emphasis), reverse=True)
    for dot in ordered:
        if not (abs(dot.x) <= PITCH_LENGTH_M / 2 + 5 and abs(dot.y) <= PITCH_WIDTH_M / 2 + 5):
            continue
        cx, cy = m(dot.x, dot.y)
        colour = c.style.team_colour(dot.team)
        if dot.emphasis == "focus":
            r = c.k(7)
            c.draw.ellipse(
                [cx - r - c.k(3), cy - r - c.k(3), cx + r + c.k(3), cy + r + c.k(3)],
                fill=_rgba(c.style.accent, 255),
            )
            c.draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=_rgba(colour, 255),
                outline=(255, 255, 255, 255),
                width=c.w(1.5),
            )
        else:
            r = c.k(5 if dot.emphasis == "normal" else 4)
            alpha = 255 if dot.emphasis == "normal" else 150
            c.draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=_rgba(colour, alpha),
                outline=(255, 255, 255, alpha),
                width=c.w(1.5),
            )
    if minimap.ball is not None and _near_pitch(*minimap.ball, MINIMAP_BALL_MARGIN_M):
        cx, cy = m(*minimap.ball)
        r = c.k(4)
        c.draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(255, 255, 255, 255),
            outline=(0, 0, 0, 255),
            width=c.w(1.5),
        )


_NO_LINE_START = frozenset("，。、；：！？）》」』”’…·%°")
"""Characters a wrapped line of a banner title never starts with: closing punctuation and
units, as Chinese text is broken."""

_NO_LINE_END = frozenset("（《「『“‘")
"""Characters a wrapped line of a banner title never ends with: opening brackets."""

_CLAUSE_ENDS = frozenset("：，；。！？")
"""Punctuation that ends a clause of a title outside brackets: a banner title goes on to its
second line after one where it can, and loses only whole clauses (:func:`_banner_title`)."""

_BANNER_TITLE_SIZES = (34.0, 28.0)
"""Sizes of a banner title (1080p pixels): the first at which the title fits on two lines."""


def _latin(char: str) -> bool:
    """A character of a run that is never broken across lines: Latin letters, digits and
    the signs inside numbers and times ("45", "0.42", "00:19.04")."""
    return char.isascii() and (char.isalnum() or char in ".:,'%/+-_")


def _line_breaks(text: str) -> list[int]:
    """Positions where ``text`` may go on to a new line, as Chinese text is broken: between
    two characters, but never inside a run of Latin letters and digits, before closing
    punctuation or a unit, after an opening bracket, or at a space next to a Chinese
    character ("第 12 帧" stays together); a space between two Latin words is a break."""
    breaks: list[int] = []
    for i in range(1, len(text)):
        before, after = text[i - 1], text[i]
        if after == " ":
            continue
        if before == " ":
            if i >= 2 and _latin(text[i - 2]) and _latin(after):
                breaks.append(i)
            continue
        if _latin(before) and _latin(after):
            continue
        if after in _NO_LINE_START or before in _NO_LINE_END:
            continue
        breaks.append(i)
    return breaks


def _bracket_depths(text: str) -> list[int]:
    """How many brackets are open at each position of ``text`` (after ``text[:i]``)."""
    depths = [0]
    for char in text:
        depth = depths[-1] + (char in "（(") - (char in "）)")
        depths.append(max(0, depth))
    return depths


def _clause_ends(text: str) -> list[int]:
    """Positions right after the clause ends of ``text`` (:data:`_CLAUSE_ENDS` outside
    brackets: the comma in "（画面平面，第 12 帧）" ends no clause)."""
    depths = _bracket_depths(text)
    return [i for i in range(1, len(text) + 1) if text[i - 1] in _CLAUSE_ENDS and not depths[i]]


def _wrap_two(
    font: ImageFont.FreeTypeFont, text: str, first_w: float, second_w: float
) -> tuple[str, ...] | None:
    """``text`` on one line at most ``first_w`` wide, else on two lines with the second at
    most ``second_w`` wide, broken after the last clause end that lets both lines fit, else
    at the last break outside brackets that does, else at the last break that does
    (:func:`_line_breaks`); None when two lines are too few."""
    text = text.strip()
    if _text_width(font, text) <= first_w:
        return (text,)
    fits = [i for i in _line_breaks(text) if _text_width(font, text[:i].rstrip()) <= first_w]
    ends, depths = set(_clause_ends(text)), _bracket_depths(text)
    for i in sorted(fits, key=lambda i: (i not in ends, depths[i] > 0, -i)):
        rest = text[i:].strip()
        if _text_width(font, rest) <= second_w:
            return (text[:i].rstrip(), rest)
    return None


def _banner_title(
    g: _Geometry, title: str, first_w: float, second_w: float
) -> tuple[tuple[str, ...], float]:
    """The lines of a banner ``title`` as drawn and their size (1080p).

    The title takes the first size of :data:`_BANNER_TITLE_SIZES` at which it fits on its
    first line (``first_w`` wide, beside the time pill) or on two lines (the second
    ``second_w`` wide, :func:`_wrap_two`). A title too long for two lines at the smallest
    size keeps its clauses up to the last whole one that fits and ends with "…" in place
    of that clause's punctuation, so no claim is cut in the middle; only a first clause
    longer than two lines is cut inside, with "…".
    """
    for size in _BANNER_TITLE_SIZES:
        lines = _wrap_two(g.font(size, bold=True), title, first_w, second_w)
        if lines is not None:
            return lines, size
    size = _BANNER_TITLE_SIZES[-1]
    font = g.font(size, bold=True)
    text = title.strip()
    for i in reversed([i for i in _clause_ends(text) if i < len(text)]):
        lines = _wrap_two(font, text[: i - 1].rstrip() + "…", first_w, second_w)
        if lines is not None:
            return lines, size
    fits = [i for i in _line_breaks(text) if _text_width(font, text[:i].rstrip()) <= first_w]
    if not fits:
        return (_fit(font, text, first_w),), size
    return (text[: fits[-1]].rstrip(), _fit(font, text[fits[-1] :].strip(), second_w)), size


@dataclass(frozen=True)
class _BannerPlan:
    """The banner and its badge row on the canvas: the banner's panel ``box`` (None without
    a banner) with its title's lines as drawn (one or two) and their size (1080p), the time
    pill and the subtitle as fitted, the heights of the first line, of the title's second
    line and of the subtitle's line, and the badge row: its centre line, each chip's left
    edge and badge, and the row's box (None without badges).
    """

    box: tuple[float, float, float, float] | None
    title: tuple[str, ...]
    title_size: float
    time: str | None
    time_w: float
    subtitle: str | None
    line1_h: float
    more_h: float
    sub_h: float
    chip_y: float
    chips: tuple[tuple[float, Badge], ...]
    chips_box: tuple[float, float, float, float] | None


def _plan_banner(
    g: _Geometry,
    banner: Banner | None,
    badges: Sequence[Badge],
    beside: Sequence[tuple[float, float, float, float]] = (),
) -> _BannerPlan | None:
    """The layout of the banner (lower third, or top left by :attr:`Style.banner_position`)
    and of the badge row next to it on the canvas ``g``; None with neither.

    The banner ends at least 16 px (1080p) left of the panels ``beside`` it in its half of
    the frame (the minimap), and its first line, the title and the time pill, spans at most
    62 % of the frame width plus the pill, as does its second line. The title takes the
    room the pill leaves on the first line and goes on to a second line where it does not
    fit (:func:`_banner_title`); the subtitle is cut with "…" to the second line's width.
    """
    if banner is None and not badges:
        return None
    time_font = g.font(28, bold=True)
    sub_font = g.font(21)
    chip_font = g.font(17)
    pad_x, pad_y, stripe = g.k(20), g.k(12), g.k(7)
    bottom = g.style.banner_position == "bottom_left"
    x0 = g.margin_x
    box = None
    title: tuple[str, ...] = ()
    size = _BANNER_TITLE_SIZES[0]
    subtitle = None
    time_w = line1_h = more_h = sub_h = 0.0
    if banner is not None:
        right = g.canvas_w - g.margin_x
        half = g.canvas_h / 2
        for panel in beside:  # a panel right of the banner's start, in the banner's half
            if panel[0] > x0 and (panel[3] > half if bottom else panel[1] < half):
                right = min(right, panel[0] - g.k(16))
        time_w = _text_width(time_font, banner.time) + g.k(24) if banner.time else 0.0
        pill = g.k(16) + time_w if banner.time else 0.0
        line_w = min(g.canvas_w * 0.62 + pill, right - x0 - stripe - 2 * pad_x)
        title, size = _banner_title(g, banner.title, line_w - pill, line_w)
        title_font = g.font(size, bold=True)
        subtitle = _fit(sub_font, banner.subtitle, line_w) if banner.subtitle else None
        line1_h = max(_line_height(title_font), _line_height(time_font) + g.k(6))
        more_h = _line_height(title_font)
        sub_h = _line_height(sub_font) + g.k(4) if subtitle else 0.0
        text_w = max(
            _text_width(title_font, title[0]) + pill,
            *(_text_width(title_font, line) for line in title[1:]),
            _text_width(sub_font, subtitle) if subtitle else 0.0,
        )
        height = pad_y + line1_h + more_h * (len(title) - 1) + sub_h + pad_y
        y0 = g.canvas_h - g.margin_y - height if bottom else g.margin_y
        box = (x0, y0, x0 + stripe + pad_x + text_w + pad_x, y0 + height)
    chip_h = _line_height(chip_font) + g.k(4)
    if box is None:
        chip_y = g.canvas_h - g.margin_y - chip_h / 2
    elif bottom:
        chip_y = box[1] - g.k(10) - chip_h / 2
    else:
        chip_y = box[3] + g.k(10) + chip_h / 2
    chips: list[tuple[float, Badge]] = []
    x = x0
    for badge in badges:
        chips.append((x, badge))
        x += _chip_width(g, badge, chip_font) + g.k(8)
    chips_box = (x0, chip_y - chip_h / 2, x - g.k(8), chip_y + chip_h / 2) if chips else None
    return _BannerPlan(
        box=box,
        title=title,
        title_size=size,
        time=banner.time if banner is not None else None,
        time_w=time_w,
        subtitle=subtitle,
        line1_h=line1_h,
        more_h=more_h,
        sub_h=sub_h,
        chip_y=chip_y,
        chips=tuple(chips),
        chips_box=chips_box,
    )


def _draw_banner(c: _Canvas, plan: _BannerPlan) -> None:
    """The banner and its badge row as ``plan`` lays them out."""
    title_font = c.font(plan.title_size, bold=True)
    time_font = c.font(28, bold=True)
    sub_font = c.font(21)
    chip_font = c.font(17)
    pad_x, pad_y, stripe = c.k(20), c.k(12), c.k(7)
    if plan.box is not None:
        x0, y0, _x1, _y1 = plan.box
        _striped_panel(c, plan.box, c.k(10), c.style.accent, stripe)
        line1_mid = y0 + pad_y + plan.line1_h / 2
        tx = x0 + stripe + pad_x
        white = _rgba(c.style.text, 255)
        _text(c, (tx, line1_mid), plan.title[0], title_font, white, "lm", stroke=c.faux_bold(True))
        if plan.time:
            px0 = tx + _text_width(title_font, plan.title[0]) + c.k(16)
            pill_h = _line_height(time_font) + c.k(4)
            c.draw.rounded_rectangle(
                [px0, line1_mid - pill_h / 2, px0 + plan.time_w, line1_mid + pill_h / 2],
                radius=c.k(6),
                fill=_rgba(c.style.accent, 255),
            )
            _text(
                c,
                (px0 + plan.time_w / 2, line1_mid),
                plan.time,
                time_font,
                (12, 14, 20, 255),
                "mm",
                stroke=c.faux_bold(True),
            )
        y = y0 + pad_y + plan.line1_h
        for line in plan.title[1:]:
            _text(c, (tx, y + plan.more_h / 2), line, title_font, white, "lm", c.faux_bold(True))
            y += plan.more_h
        if plan.subtitle:
            _text(
                c,
                (tx, y + plan.sub_h / 2),
                plan.subtitle,
                sub_font,
                _rgba(c.style.text_secondary, 255),
                "lm",
            )
    for x, badge in plan.chips:
        _chip(c, x, plan.chip_y, badge, chip_font, (10, 14, 22, 200))


def _watermark_centres(g: _Geometry) -> list[tuple[float, float]]:
    """Canvas centres of the watermark: the top centre, inside the top margin, and with
    :attr:`Style.watermark_repeat` the bottom centre, inside the bottom margin (no panel
    reaches into the margins)."""
    centres = [(g.canvas_w / 2, g.margin_y * 0.55)]
    if g.style.watermark_repeat:
        centres.append((g.canvas_w / 2, g.canvas_h - g.margin_y * 0.55))
    return centres


def _watermark_boxes(g: _Geometry, text: str) -> list[tuple[float, float, float, float]]:
    """Canvas boxes of the watermark's panels (its text is centred in each), in the order of
    :func:`_watermark_centres`."""
    boxes = []
    for centre in _watermark_centres(g):
        left, top, right, bottom = _MEASURE.textbbox(centre, text, font=g.font(19), anchor="mm")
        boxes.append((left - g.k(12), top - g.k(5), right + g.k(12), bottom + g.k(5)))
    return boxes


def _draw_watermark(c: _Canvas, text: str) -> None:
    """The watermark: the text on a dark panel at the top centre, and a fainter copy at the
    bottom centre with :attr:`Style.watermark_repeat`."""
    font = c.font(19)
    marks = zip(_watermark_boxes(c, text), _watermark_centres(c), strict=True)
    for index, (box, centre) in enumerate(marks):
        faint = index > 0
        c.draw.rounded_rectangle(list(box), radius=c.k(8), fill=(0, 0, 0, 70 if faint else 110))
        _text(c, centre, text, font, (255, 255, 255, 150 if faint else 200), "mm")


def _apply_dim(c: _Canvas, dim: float, spotlight: tuple[float, float, float] | None) -> None:
    if dim <= 0.0:
        return
    canvas_w, canvas_h = c.image.size
    factor = 8
    small = (max(1, canvas_w // factor), max(1, canvas_h // factor))
    mask = Image.new("L", small, int(round(255 * dim)))
    if spotlight is not None:
        u, v, radius = spotlight
        x, y = c.p(u, v)
        r = radius * c.s / factor
        x, y = x / factor, y / factor
        ImageDraw.Draw(mask).ellipse([x - r, y - r, x + r, y + r], fill=0)
        mask = mask.filter(ImageFilter.GaussianBlur(max(1.0, r * 0.35)))
    mask = mask.resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
    c.image.paste((0, 0, 0), (0, 0, canvas_w, canvas_h), mask)


def check_frame(frame: Any, where: str = "frame") -> np.ndarray:
    """``frame`` as a C-contiguous (height, width, 3) uint8 array, or RenderError."""
    if not isinstance(frame, np.ndarray):
        raise RenderError(where, [f"frame must be a numpy array, got {type(frame).__name__}"])
    problems = []
    if frame.dtype != np.uint8:
        problems.append(f"dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        problems.append(f"shape must be (height, width, 3), got {frame.shape}")
    elif frame.shape[0] < 1 or frame.shape[1] < 1:
        problems.append(f"frame must not be empty, got shape {frame.shape}")
    _raise(where, problems)
    return np.ascontiguousarray(frame)


@dataclass(frozen=True)
class _Layout:
    """Where the texts and panels of one frame's overlay go on its canvas, and which texts
    are left out: the one computation behind :func:`draw_annotations` and
    :func:`annotation_texts`, so a render manifest lists exactly the texts that are drawn.

    ``players``, ``lines`` and ``arrows`` hold each mark's label in the order of the
    annotations (None for a mark without one); ``texts`` says which free texts are drawn;
    ``badges`` is the frame's badge row as drawn (``arrow_badges`` only with an arrow).
    """

    players: tuple[_Label | None, ...]
    lines: tuple[_Label | None, ...]
    arrows: tuple[_Label | None, ...]
    ball: _Label | None
    texts: tuple[bool, ...]
    card: _CardPlan | None
    banner: _BannerPlan | None
    badges: tuple[Badge, ...] = ()


_LABEL_PRIORITY: Mapping[str, int] = MappingProxyType({"focus": 0, "normal": 1, "context": 2})
"""Player labels are placed in this order of emphasis, the focus player's first."""


def _label_order(players: Sequence[PlayerMark]) -> list[int]:
    return sorted(range(len(players)), key=lambda index: _LABEL_PRIORITY[players[index].emphasis])


def _layout(g: _Geometry, annotations: FrameAnnotations) -> _Layout:
    """The layout of ``annotations`` on the canvas ``g`` (fonts resolved).

    No text lies under a panel drawn over it, where it would show through the panel: the
    card keeps clear of the minimap, the banner and the badge row (:func:`_plan_card`); a
    free text under one of those, the card or the watermark is left out; a label keeps
    clear of them and of the free texts drawn, and is left out when the point it is about
    (above a player's head, the middle of a line, an arrow's head) lies under one, or it
    finds no spot clear of them (:func:`_free_spot`). Labels are placed by priority (the
    players' by emphasis, focus first, then the lines', the arrows' and the ball's), each
    off earlier labels and the ball where it can be; the players' and the lines' labels also
    keep off the bodies of the focus and the involved players (the marks that are not
    context) where they can, an arrow's label off every player's box (:func:`_arrow_label`),
    and a label moved a step or more gets a leader line to its player or line. The badge
    row holds ``arrow_badges`` only when an arrow is drawn; its room is kept either way.
    """
    style = g.style
    over: list[tuple[float, float, float, float]] = []  # panels drawn over the free texts
    if annotations.minimap is not None:
        mx, my, mw, mh = _minimap_geometry(
            g.canvas_w,
            g.canvas_h,
            g.k(1),
            g.margin_x,
            g.margin_y,
            style,
            bool(annotations.minimap.title),
        ).panel
        over.append((mx, my, mx + mw, my + mh))
    banner_avoid = tuple(over)
    planned = annotations.arrow_badges + annotations.badges
    banner = _plan_banner(g, annotations.banner, planned, banner_avoid)
    if banner is not None:
        over.extend(box for box in (banner.box, banner.chips_box) if box is not None)
    card = _plan_card(g, annotations, over)
    if card is not None:
        over.append(card.box)
    if style.watermark:
        over.extend(_watermark_boxes(g, style.watermark))
    boxes = [_text_box(g, item) for item in annotations.texts]
    texts = tuple(not _overlaps(box, over) for box in boxes)
    panels = over + [box for box, shown in zip(boxes, texts, strict=True) if shown]
    occupied: list[tuple[float, float, float, float]] = []
    ball = annotations.ball
    ball_box = _ball_box(g, ball) if ball is not None else None
    if ball_box is not None:
        occupied.append(ball_box)
    bodies = [
        (*g.p(x1, y1), *g.p(x2, y2)) for x1, y1, x2, y2 in (p.box for p in annotations.players)
    ]
    involved = [
        body
        for body, mark in zip(bodies, annotations.players, strict=True)
        if mark.emphasis != "context"
    ]
    players: list[_Label | None] = [None] * len(annotations.players)
    for index in _label_order(annotations.players):
        players[index] = _player_label(g, annotations.players[index], occupied, panels, involved)
    lines = tuple(_line_label(g, line, occupied, panels, involved) for line in annotations.lines)
    arrows = tuple(_arrow_label(g, arrow, occupied, panels, bodies) for arrow in annotations.arrows)
    badges = annotations.badges
    if any(
        not arrow.label or (label is not None and label.spot is not None)
        for arrow, label in zip(annotations.arrows, arrows, strict=True)
    ):
        badges = planned
    if badges != planned:
        banner = _plan_banner(g, annotations.banner, badges, banner_avoid)
    return _Layout(
        players=tuple(players),
        lines=lines,
        arrows=arrows,
        ball=_ball_label(g, ball, occupied, panels) if ball is not None else None,
        texts=texts,
        card=card,
        banner=banner,
        badges=badges,
    )


def draw_annotations(
    frame_rgb: np.ndarray, annotations: FrameAnnotations, style: Style | None = None
) -> np.ndarray:
    """Return a new frame with ``annotations`` drawn on ``frame_rgb`` (never modified).

    Drawing order, bottom to top: dimming, ground rings, ball trail, lines, arrows,
    skeletons, corner brackets, ball; then the leader lines of moved labels and the labels
    of players (focus first), lines, arrows and the ball, each moved off earlier labels and
    the ball when they would cover them, and off the focus and involved players where they
    can be; then free texts, card, minimap, banner and badges, watermark. No text is drawn
    under a panel drawn after it, where it would show through: a label or free text there
    is left out, and the card keeps clear of the minimap, the banner and the badges (see
    :func:`annotation_texts`, which lists what is drawn). An arrow with a label is drawn
    only with it: the label says what the arrow shows, and ``arrow_badges`` join the
    badge row only when an arrow is drawn.

    Raises :class:`RenderError` for a malformed frame or annotations and
    :class:`gbf.fonts.FontNotFoundError` when no CJK font exists.
    """
    frame = check_frame(frame_rgb, "draw_annotations")
    if not isinstance(annotations, FrameAnnotations):
        raise RenderError(
            "draw_annotations",
            [f"annotations must be FrameAnnotations, got {type(annotations).__name__}"],
        )
    style = (style or Style()).with_fonts()
    height, width = frame.shape[:2]
    s = style.supersample
    base = Image.fromarray(frame)  # Pillow copies RGB data into its own buffer
    canvas_image = (
        base.resize((width * s, height * s), Image.Resampling.NEAREST) if s > 1 else base.copy()
    )
    canvas = _Canvas(
        image=canvas_image,
        draw=ImageDraw.Draw(canvas_image, "RGBA"),
        s=s,
        ui=style.ui_scale if style.ui_scale is not None else height / 1080.0,
        width=width,
        height=height,
        style=style,
    )
    layout = _layout(canvas, annotations)
    _apply_dim(canvas, annotations.dim, annotations.spotlight)
    for player in annotations.players:
        _draw_player_ring(canvas, player)
    if annotations.ball is not None:
        _draw_trail(canvas, annotations.ball)
    for line in annotations.lines:
        _draw_link(canvas, line)
    for arrow, arrow_label in zip(annotations.arrows, layout.arrows, strict=True):
        if not arrow.label or (arrow_label is not None and arrow_label.spot is not None):
            _draw_arrow(canvas, arrow)
    for skeleton in annotations.skeletons:
        _draw_skeleton(canvas, skeleton)
    for player in annotations.players:
        _draw_brackets(canvas, player)
    if annotations.ball is not None:
        _draw_ball(canvas, annotations.ball)
    for label in (*layout.players, *layout.lines, *layout.arrows):
        if label is not None:
            _draw_leader(canvas, label)
    for index in _label_order(annotations.players):
        _draw_player_label(canvas, annotations.players[index], layout.players[index])
    for label in (*layout.lines, *layout.arrows, layout.ball):
        if label is not None:
            _draw_label(canvas, label)
    for item, shown in zip(annotations.texts, layout.texts, strict=True):
        if shown:
            _draw_text(canvas, item)
    if layout.card is not None:
        _draw_card(canvas, layout.card)
    if annotations.minimap is not None:
        _draw_minimap(canvas, annotations.minimap)
    if layout.banner is not None:
        _draw_banner(canvas, layout.banner)
    if style.watermark:
        _draw_watermark(canvas, style.watermark)
    result = canvas.image.reduce(s) if s > 1 else canvas.image
    out = np.array(result, dtype=np.uint8)
    if out.shape != frame.shape:  # pragma: no cover - reduce() of an exact multiple
        raise RenderError("draw_annotations", [f"output shape {out.shape} != {frame.shape}"])
    return out


RESTRICTIONS_COMMENT_PREFIX = "gbf restrictions: "
"""Start of the comment that carries a rendered file's restriction tags in the file itself
(:func:`restrictions_comment`)."""


def _restriction_tags(where: str, restrictions: Iterable[str]) -> list[str]:
    """``restrictions`` as a list of tags, or RenderError naming every one that is not a
    non-empty string (a bare string is refused: its characters are no tags)."""
    if isinstance(restrictions, str | bytes):
        raise RenderError(where, [f"restrictions must be a list of tags, got {restrictions!r}"])
    tags = list(restrictions)
    bad = [tag for tag in tags if not isinstance(tag, str) or not tag]
    if bad:
        raise RenderError(where, [f"restrictions must be non-empty strings, got {bad!r}"])
    return tags


def restrictions_comment(restrictions: Iterable[str]) -> str | None:
    """The comment that a rendered clip or image carries in its own metadata: the sorted
    tags after :data:`RESTRICTIONS_COMMENT_PREFIX`, joined by commas ("gbf restrictions:
    hl3,no_share"), so that a file copied out of its run still says what it may be used
    for; None without tags. Raises RenderError for a tag that is not a string."""
    tags = sorted(set(_restriction_tags("restrictions_comment", restrictions)))
    return RESTRICTIONS_COMMENT_PREFIX + ",".join(tags) if tags else None


def save_png(path: str | Path, frame_rgb: np.ndarray, *, restrictions: Sequence[str] = ()) -> str:
    """Write an RGB frame as PNG (Pillow, any path) and return the file's sha256.

    With ``restrictions`` the file carries them in a tEXt chunk "Comment"
    (:func:`restrictions_comment`), which image viewers show.
    """
    frame = check_frame(frame_rgb, "save_png")
    comment = restrictions_comment(_restriction_tags("save_png", restrictions))
    info = None
    if comment is not None:
        info = PngInfo()
        info.add_text("Comment", comment)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(target, format="PNG", pnginfo=info)
    return sha256_file(target)


def stamp_watermark(image_rgb: np.ndarray, text: str) -> np.ndarray:
    """A copy of ``image_rgb`` (an RGB chart or frame, e.g. the report's timeline) with the
    watermark badge of the rendered frames in its top-right corner, where a chart's legend
    leaves room: the same white text on a dark panel at its 1080p size, the panel darker so
    that the text reads on a light chart too. Raises RenderError for a malformed image or
    an empty text."""
    image = check_frame(image_rgb, "stamp_watermark")
    if not isinstance(text, str) or not text.strip():
        raise RenderError("stamp_watermark", [f"text must be a non-empty string, got {text!r}"])
    style = Style(ui_scale=1.0).with_fonts()
    height, width = image.shape[:2]
    canvas_image = Image.fromarray(image).resize(
        (width * style.supersample, height * style.supersample), Image.Resampling.NEAREST
    )
    c = _Canvas(
        image=canvas_image,
        draw=ImageDraw.Draw(canvas_image, "RGBA"),
        s=style.supersample,
        ui=1.0,
        width=width,
        height=height,
        style=style,
    )
    font = c.font(19)
    left, top, right, bottom = _MEASURE.textbbox((0, 0), text, font=font, anchor="mm")
    inset = c.k(8)
    centre = (c.canvas_w - inset - c.k(12) - right, inset + c.k(5) - top)
    box = (
        centre[0] + left - c.k(12),
        centre[1] + top - c.k(5),
        centre[0] + right + c.k(12),
        centre[1] + bottom + c.k(5),
    )
    c.draw.rounded_rectangle(list(box), radius=c.k(8), fill=(0, 0, 0, 200))
    _text(c, centre, text, font, (255, 255, 255, 255), "mm")
    return np.array(canvas_image.reduce(style.supersample), dtype=np.uint8)


# --------------------------------------------------------------------------------------
# geometry helpers for overlays
# --------------------------------------------------------------------------------------
def ground_circle_px(
    camera: Camera, centre_xy: tuple[float, float], radius_m: float, n: int = 72
) -> np.ndarray:
    """Pixels (n, 2) of a circle of ``radius_m`` metres on the pitch (z = 0) around
    ``centre_xy`` (SoccerNet x, y metres): a perspective-correct ring under a player.

    Points behind the camera are NaN (:class:`PlayerMark` then falls back to an ellipse).
    """
    problems: list[str] = []
    centre = _point("centre_xy", centre_xy, problems)
    radius = _finite("radius_m", radius_m, problems)
    if radius is not None and radius <= 0:
        problems.append(f"radius_m must be > 0, got {radius}")
    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        problems.append(f"n must be an int >= 3, got {n!r}")
    if problems or centre is None or radius is None:
        raise RenderError("ground_circle_px", problems)
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    points = np.stack(
        [centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles), np.zeros(n)],
        axis=-1,
    )
    uv, _depth, _valid = project(camera, points)
    return uv


def camera_footprint(
    camera: Camera,
    *,
    samples_per_edge: int = 24,
    limit: tuple[float, float] = (60.0, 40.0),
    far_m: float = 1000.0,
) -> list[Point]:
    """The pitch-plane area the camera sees, as a polygon of SoccerNet (x, y) metres.

    The image border is sampled ``samples_per_edge`` times per edge and every border ray is
    intersected with the grass (z = 0). A ray that misses the grass (above the horizon) or
    meets it farther than ``far_m`` is replaced by a point ``far_m`` away in its horizontal
    direction; the polygon is then clipped to |x| <= limit[0], |y| <= limit[1].
    """
    width, height = camera.width, camera.height
    t = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
    left, top, right, bottom = -0.5, -0.5, width - 0.5, height - 0.5
    border = np.concatenate(
        [
            np.stack([left + t * (right - left), np.full_like(t, top)], -1),
            np.stack([np.full_like(t, right), top + t * (bottom - top)], -1),
            np.stack([right - t * (right - left), np.full_like(t, bottom)], -1),
            np.stack([np.full_like(t, left), bottom - t * (bottom - top)], -1),
        ]
    )
    rays = pixel_rays(camera, border)
    centre = np.asarray(camera.C, dtype=np.float64)
    points: list[Point] = []
    for direction in rays:
        horizontal = math.hypot(direction[0], direction[1])
        hit = None
        if direction[2] > 1e-12:
            distance = -centre[2] / direction[2]
            ground = centre + distance * direction
            if math.hypot(ground[0] - centre[0], ground[1] - centre[1]) <= far_m:
                hit = (float(ground[0]), float(ground[1]))
        if hit is None:
            if horizontal <= 1e-12:
                continue
            hit = (
                float(centre[0] + direction[0] / horizontal * far_m),
                float(centre[1] + direction[1] / horizontal * far_m),
            )
        points.append(hit)
    return _clip_polygon(points, -limit[0], -limit[1], limit[0], limit[1])


# --------------------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioSourceFacts:
    """Facts about the file that carries the original audio (all times in seconds on the
    file's own timeline, as ffprobe reports them)."""

    file_name: str
    codec_name: str
    sample_rate: int
    channels: int
    audio_start_s: float
    video_start_s: float | None
    format_start_s: float
    duration_s: float | None

    @property
    def default_offset_s(self) -> float:
        """Audio-file time (as ffmpeg's filters see it) minus canonical time.

        ffmpeg shifts every input so that its earliest stream starts at 0; canonical frame 0
        is the first video frame of the source, so canonical time t is filter time
        ``t + video_start - format_start``. Without a video stream the audio is assumed to
        start together with the video (offset 0).
        """
        if self.video_start_s is None:
            return 0.0
        return self.video_start_s - self.format_start_s

    def manifest(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "codec_name": self.codec_name,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "audio_start_s": self.audio_start_s,
            "video_start_s": self.video_start_s,
            "format_start_s": self.format_start_s,
            "duration_s": self.duration_s,
        }


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@lru_cache(maxsize=16)
def _probe_audio_cached(path: str, size: int, mtime_ns: int) -> AudioSourceFacts:
    what = f"probe audio of {Path(path).name}"
    args = [
        ffprobe_path(),
        "-v",
        "error",
        "-hide_banner",
        "-show_entries",
        "format=start_time,duration:stream=index,codec_type,codec_name,sample_rate,channels,"
        "start_time,duration",
        "-of",
        "json=compact=1",
        media_arg(path),
    ]
    proc = run_tool(args, what=what)
    data = json.loads(proc.stdout.decode("utf-8"))
    streams = data.get("streams") or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if audio is None:
        raise RenderError(what, ["the file has no audio stream"])
    fmt = data.get("format") or {}
    return AudioSourceFacts(
        file_name=Path(path).name,
        codec_name=str(audio.get("codec_name") or "unknown"),
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
        audio_start_s=_float_or_none(audio.get("start_time")) or 0.0,
        video_start_s=_float_or_none(video.get("start_time")) if video is not None else None,
        format_start_s=_float_or_none(fmt.get("start_time")) or 0.0,
        duration_s=_float_or_none(audio.get("duration")) or _float_or_none(fmt.get("duration")),
    )


def probe_audio(path: str | Path) -> AudioSourceFacts:
    """ffprobe facts of the first audio stream of ``path`` (cached per file version).

    Raises FileNotFoundError, :class:`RenderError` when there is no audio stream and
    :class:`gbf.ingest.FFmpegError` when ffprobe fails.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"not a file: {source}")
    stat = source.stat()
    return _probe_audio_cached(str(source.resolve()), stat.st_size, stat.st_mtime_ns)


def first_sample_offset(source: str | Path, offset_s: float) -> float:
    """The ``offset_s`` of :class:`AudioSpan` and :func:`render_edit` for a source whose
    audio is ``offset_s`` seconds in at canonical frame 0, counted from its first audio
    sample as :mod:`gbf.audio` counts it (the pipeline's ``--audio-offset-s``).

    That is ``offset_s`` plus the time of the first audio sample on ffmpeg's timeline (the
    audio stream's start minus the file's start). A clip rendered with it plays at every
    frame the sound gbf.audio measured the frame's excitement cue on, also when the file's
    audio starts later than its video. Raises as :func:`probe_audio` does, and RenderError
    for an offset that is not a finite number.
    """
    problems: list[str] = []
    offset = _finite("offset_s", offset_s, problems)
    if offset is None:
        raise RenderError("first_sample_offset", problems)
    facts = probe_audio(source)
    return offset + facts.audio_start_s - facts.format_start_s


def _seconds(name: str, value: Any, problems: list[str]) -> Fraction | None:
    if isinstance(value, bool):
        problems.append(f"{name} must be a number of seconds, got {value!r}")
        return None
    if isinstance(value, Fraction | int):
        result = Fraction(value)
    elif isinstance(value, float) and math.isfinite(value):
        result = Fraction(value).limit_denominator(10**9)
    else:
        problems.append(f"{name} must be a finite number of seconds, got {value!r}")
        return None
    if result < 0:
        problems.append(f"{name} must be >= 0, got {float(result)}")
        return None
    return result


@dataclass(frozen=True)
class AudioSpan:
    """Original audio from ``source``: ``duration_s`` seconds from canonical time ``start_s``.

    Canonical time is frame / fps of the canonical video. ``offset_s`` is the audio-file
    time minus canonical time; ``None`` uses :attr:`AudioSourceFacts.default_offset_s`.
    The span is padded with silence where the source has no audio (past its end), so its
    length is always exactly ``duration_s`` (at 48 kHz). ``gain_db`` scales it; 15 ms fades
    at both ends avoid clicks.
    """

    source: Path
    start_s: Fraction | float
    duration_s: Fraction | float
    gain_db: float = 0.0
    offset_s: float | None = None

    def __post_init__(self) -> None:
        problems: list[str] = []
        object.__setattr__(self, "source", Path(self.source))
        start = _seconds("start_s", self.start_s, problems)
        duration = _seconds("duration_s", self.duration_s, problems)
        if duration is not None and duration <= 0:
            problems.append("duration_s must be > 0")
        _finite("gain_db", self.gain_db, problems)
        if self.offset_s is not None:
            _finite("offset_s", self.offset_s, problems)
        _raise("AudioSpan", problems)
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "duration_s", duration)


@dataclass(frozen=True)
class Silence:
    """``duration_s`` seconds of silence in the output audio track."""

    duration_s: Fraction | float

    def __post_init__(self) -> None:
        problems: list[str] = []
        duration = _seconds("duration_s", self.duration_s, problems)
        if duration is not None and duration <= 0:
            problems.append("duration_s must be > 0")
        _raise("Silence", problems)
        object.__setattr__(self, "duration_s", duration)


AudioPart = AudioSpan | Silence


def _samples(seconds: Fraction) -> int:
    return round_half_away(Fraction(seconds) * AUDIO_SAMPLE_RATE)


def _offset(facts: AudioSourceFacts, offset_s: float | None) -> Fraction:
    """The audio-file time minus canonical time of a span (``offset_s``, else the file's
    :attr:`AudioSourceFacts.default_offset_s`) as the nearest fraction with a denominator up
    to 10**9, as :func:`_seconds` reads seconds. The edit's sample counts
    (:func:`_audio_extent`) and the filter graph use this one value: recomputed in floats,
    a span that starts on the first audio sample could start a float's last bit before the
    audio file (an offset of -(0.1 + 0.2) s, say)."""
    offset = facts.default_offset_s if offset_s is None else float(offset_s)
    return Fraction(offset).limit_denominator(10**9)


def _audio_graph(
    parts: Sequence[AudioPart], inputs: dict[Path, int]
) -> tuple[str, list[dict[str, Any]]]:
    """filter_complex text that builds the output track, and the parts' manifest rows."""
    chains: list[str] = []
    rows: list[dict[str, Any]] = []
    fade = int(0.015 * AUDIO_SAMPLE_RATE)
    common = f"aformat=sample_fmts=fltp:sample_rates={AUDIO_SAMPLE_RATE}:channel_layouts=stereo"
    for index, part in enumerate(parts):
        n = _samples(Fraction(part.duration_s))
        if n < 1:
            raise RenderError("audio", [f"part {index} is shorter than one sample"])
        label = f"[a{index}]"
        if isinstance(part, Silence):
            chains.append(
                f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_SAMPLE_RATE},"
                f"atrim=end_sample={n},{common}{label}"
            )
            rows.append({"kind": "silence", "samples": n})
            continue
        facts = probe_audio(part.source)
        offset = facts.default_offset_s if part.offset_s is None else float(part.offset_s)
        start = Fraction(part.start_s) + _offset(facts, part.offset_s)  # as _audio_extent
        if start < 0:
            raise RenderError(
                "audio", [f"part {index} starts before the audio file (t = {float(start):.6f} s)"]
            )
        f = min(fade, n // 4)
        # Pad the source with endless silence BEFORE trimming: a span that starts after the
        # end of the source audio still gets exactly n samples (atrim on an exhausted
        # stream would give none, and apad cannot pad an empty stream).
        chain = (
            f"[{inputs[part.source]}:a:0]aresample={AUDIO_SAMPLE_RATE},{common},apad,"
            f"atrim=start={float(start):.6f},asetpts=PTS-STARTPTS,atrim=end_sample={n}"
        )
        if f > 0:
            chain += f",afade=t=in:ss=0:ns={f},afade=t=out:ss={n - f}:ns={f}"
        if part.gain_db:
            chain += f",volume={part.gain_db:.3f}dB"
        chains.append(chain + label)
        rows.append(
            {
                "kind": "source",
                "source": facts.file_name,
                "start_s": float(Fraction(part.start_s)),
                "offset_s": offset,
                "samples": n,
                "gain_db": float(part.gain_db),
            }
        )
    if len(parts) == 1:
        chains.append("[a0]anull[aout]")
    else:
        joined = "".join(f"[a{i}]" for i in range(len(parts)))
        chains.append(f"{joined}concat=n={len(parts)}:v=0:a=1[aout]")
    return ";".join(chains), rows


# --------------------------------------------------------------------------------------
# video writer
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WrittenVideo:
    """A verified H.264 file written by :class:`VideoWriter`."""

    path: Path
    frames: int
    fps: str
    width: int
    height: int
    sha256: str
    bytes: int
    audio: dict[str, Any] | None
    audio_parts: tuple[dict[str, Any], ...]
    settings: dict[str, Any]
    comment: str | None = None

    @property
    def duration_s(self) -> float:
        return float(self.frames / parse_rate(self.fps))

    def manifest(self) -> dict[str, Any]:
        """JSON form for manifests (file name only, no path)."""
        return {
            "file_name": self.path.name,
            "frames": self.frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration_s": self.duration_s,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "audio": self.audio,
            "audio_parts": list(self.audio_parts),
            "settings": dict(self.settings),
            "comment": self.comment,
        }


class VideoWriter:
    """Stream RGB frames into a verified H.264 mp4 through one ffmpeg rawvideo pipe.

    Usage::

        with VideoWriter(out, 1920, 1080, "25/1", audio=[AudioSpan(src, 10, 4)]) as writer:
            for frame in frames:
                writer.write(frame)
        result = writer.result

    With ``restrictions`` the file carries them in its own metadata, as the comment of
    :func:`restrictions_comment` (ffprobe shows it as the format tag "comment").

    The file is written to ``<name>.partial`` and moved into place by :meth:`close` only
    after ffprobe confirms: decoded frame count == frames written, CFR at ``fps``, frame
    size, h264/yuv420p, (with audio) one AAC stereo 48 kHz stream whose length matches the
    video within 0.1 s, (with restrictions) the comment; and, if ``max_bytes`` is set, the
    size. Any failure removes the partial file and raises (:class:`RenderError` or
    :class:`gbf.ingest.FFmpegError`). Leaving the ``with`` block through an exception aborts
    the file.
    """

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: str | Fraction = "25/1",
        *,
        crf: int = DEFAULT_CRF,
        preset: str = DEFAULT_PRESET,
        audio: Sequence[AudioPart] = (),
        overwrite: bool = False,
        max_bytes: int | None = None,
        restrictions: Sequence[str] = (),
    ) -> None:
        self.path = Path(path)
        problems: list[str] = []
        self.comment: str | None = None
        try:
            self.comment = restrictions_comment(restrictions)
        except RenderError as exc:
            problems.extend(exc.problems)
        if self.path.suffix.lower() != ".mp4":
            problems.append(f"output must end in .mp4, got {self.path.name!r}")
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                problems.append(f"{name} must be an int >= 2, got {value!r}")
            elif value % 2:
                problems.append(f"{name} must be even for yuv420p, got {value}")
        try:
            self.rate = parse_rate(fps)
        except ValueError as exc:
            problems.append(str(exc))
            self.rate = Fraction(25)
        if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
            problems.append(f"crf must be an int in 0..51, got {crf!r}")
        if not isinstance(preset, str) or not preset.isalpha():
            problems.append(f"preset must be an x264 preset name, got {preset!r}")
        if max_bytes is not None and (not isinstance(max_bytes, int) or max_bytes <= 0):
            problems.append(f"max_bytes must be a positive int or None, got {max_bytes!r}")
        parts = tuple(audio)
        for index, part in enumerate(parts):
            if not isinstance(part, AudioSpan | Silence):
                problems.append(f"audio[{index}] must be an AudioSpan or Silence")
            elif isinstance(part, AudioSpan) and not part.source.is_file():
                problems.append(f"audio[{index}] source is not a file: {part.source.name}")
        _raise(f"VideoWriter({self.path.name})", problems)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} exists; pass overwrite=True to replace it")
        self.width, self.height = int(width), int(height)
        self.crf, self.preset, self.max_bytes = crf, preset, max_bytes
        self.audio_parts = parts
        self.frames_written = 0
        self.result: WrittenVideo | None = None
        self._tmp = self.path.with_name(self.path.name + ".partial")
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        inputs: dict[Path, int] = {}
        for part in parts:
            if isinstance(part, AudioSpan) and part.source not in inputs:
                inputs[part.source] = len(inputs) + 1
        graph, self._audio_rows = _audio_graph(parts, inputs) if parts else ("", [])
        self._args = self._command(inputs, graph)
        self._proc = subprocess.Popen(
            self._args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if self._proc.stdin is None or self._proc.stderr is None:  # pragma: no cover
            self._proc.kill()
            raise FFmpegError("start ffmpeg", None, "could not open pipes to ffmpeg")
        self._stdin = self._proc.stdin
        self._stderr = StderrCollector(self._proc.stderr)

    def settings(self) -> dict[str, Any]:
        """Encoder settings recorded in manifests."""
        return {
            "codec": VIDEO_CODEC,
            "preset": self.preset,
            "crf": self.crf,
            "pix_fmt": VIDEO_PIX_FMT,
            "colour": "bt709 limited range",
            "threads": X264_THREADS,
            "fps": format_rate(self.rate),
            "audio_codec": AUDIO_CODEC if self.audio_parts else None,
            "audio_bitrate": AUDIO_BITRATE if self.audio_parts else None,
        }

    def _command(self, inputs: dict[Path, int], graph: str) -> list[str]:
        args = [
            ffmpeg_path(),
            "-hide_banner",
            "-nostats",
            "-loglevel",
            LOGLEVEL,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-framerate",
            format_rate(self.rate),
            "-thread_queue_size",
            "512",
            "-i",
            "pipe:0",
        ]
        for source in inputs:
            args += ["-i", media_arg(source)]
        if graph:
            args += ["-filter_complex", graph]
        args += ["-map", "0:v:0"]
        if graph:
            args += ["-map", "[aout]"]
        args += [
            "-vf",
            # RGB -> BT.709 limited-range YUV, and every frame tagged so (setparams), so
            # players decode the colours the overlay was drawn in.
            f"scale=out_color_matrix=bt709:out_range=limited,format={VIDEO_PIX_FMT},"
            "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-c:v",
            VIDEO_CODEC,
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            VIDEO_PIX_FMT,
            "-threads",
            str(X264_THREADS),
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
        ]
        if graph:
            args += [
                "-c:a",
                AUDIO_CODEC,
                "-b:a",
                AUDIO_BITRATE,
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                "-ac",
                str(AUDIO_CHANNELS),
            ]
        args += ["-map_metadata", "-1"]
        if self.comment is not None:
            args += ["-metadata", f"comment={self.comment}"]
        args += [
            "-movflags",
            "+faststart",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
        ]
        if graph:
            args += ["-flags:a", "+bitexact"]
        args += ["-f", "mp4", media_arg(self._tmp)]
        return args

    def write(self, frame: np.ndarray, repeat: int = 1) -> None:
        """Append ``frame`` ((height, width, 3) uint8 RGB) ``repeat`` times."""
        if self._closed:
            raise RenderError("VideoWriter.write", ["the writer is closed"])
        array = check_frame(frame, "VideoWriter.write")
        problems = []
        if array.shape != (self.height, self.width, 3):
            problems.append(
                f"frame shape {array.shape} != ({self.height}, {self.width}, 3) of the writer"
            )
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            problems.append(f"repeat must be an int >= 1, got {repeat!r}")
        _raise("VideoWriter.write", problems)
        stdin = self._stdin
        data = memoryview(array).cast("B")
        try:
            for _ in range(repeat):
                stdin.write(data)
        except OSError as exc:
            returncode = self._proc.wait()
            stderr = self._stderr.text()
            self._cleanup()
            raise FFmpegError(
                f"write frames to {self.path.name}", returncode, f"{exc}\n{stderr}"
            ) from exc
        self.frames_written += repeat

    def abort(self) -> None:
        """Stop ffmpeg and remove the partial file (safe to call more than once)."""
        if self._closed:
            return
        self._closed = True
        if self._proc.poll() is None:
            self._proc.kill()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:  # pragma: no cover - only if the OS refuses
            pass
        try:
            self._stdin.close()
        except OSError:
            pass
        self._stderr.text(timeout=10)
        self._tmp.unlink(missing_ok=True)

    def _cleanup(self) -> None:
        self._closed = True
        self._tmp.unlink(missing_ok=True)

    def close(self) -> WrittenVideo:
        """Finish the file, verify it with ffprobe and move it into place."""
        if self.result is not None:
            return self.result
        if self._closed:
            raise RenderError("VideoWriter.close", ["the writer was aborted"])
        what = f"write {self.frames_written} frame(s) to {self.path.name}"
        if self.frames_written == 0:
            self.abort()
            raise RenderError(what, ["no frames were written"])
        try:
            self._stdin.close()
        except OSError:
            pass
        returncode = self._proc.wait()
        stderr = self._stderr.text()
        self._closed = True
        try:
            if returncode != 0 or stderr.strip():
                raise FFmpegError(what, returncode, stderr)
            result = self._verify(what)
            os.replace(self._tmp, self.path)
        finally:
            self._tmp.unlink(missing_ok=True)
        self.result = replace(result, path=self.path)
        return self.result

    def _verify(self, what: str) -> WrittenVideo:
        facts = probe(self._tmp)
        problems: list[str] = []
        if facts.nb_frames_decoded != self.frames_written:
            problems.append(
                f"decodes to {facts.nb_frames_decoded} frames, {self.frames_written} written"
            )
        if not facts.is_cfr_at(self.rate):
            problems.append(
                f"not CFR at {format_rate(self.rate)}: r_frame_rate={facts.r_frame_rate}, "
                f"avg_frame_rate={facts.avg_frame_rate}, timestamps_cfr={facts.timestamps_cfr}"
            )
        if (facts.frame_width, facts.frame_height) != (self.width, self.height):
            problems.append(
                f"frames are {facts.frame_width}x{facts.frame_height}, "
                f"expected {self.width}x{self.height}"
            )
        if facts.codec_name != "h264" or facts.pix_fmt != VIDEO_PIX_FMT:
            problems.append(f"stream is {facts.codec_name}/{facts.pix_fmt}, expected h264/yuv420p")
        audio, tags = _probe_output(self._tmp)
        if self.comment is not None and tags.get("comment") != self.comment:
            problems.append(
                f"the file's comment is {tags.get('comment')!r}, expected {self.comment!r}"
            )
        duration = float(self.frames_written / self.rate)
        if self.audio_parts:
            if audio is None:
                problems.append("no audio stream in the output")
            else:
                if audio["codec_name"] != AUDIO_CODEC:
                    problems.append(f"audio codec is {audio['codec_name']}, expected aac")
                if audio["sample_rate"] != AUDIO_SAMPLE_RATE or audio["channels"] != 2:
                    problems.append(
                        f"audio is {audio['sample_rate']} Hz x {audio['channels']}, "
                        "expected 48000 Hz stereo"
                    )
                planned = sum(row["samples"] for row in self._audio_rows) / AUDIO_SAMPLE_RATE
                got = audio.get("duration_s")
                if got is None or abs(got - planned) > 0.1:
                    problems.append(f"audio lasts {got} s, planned {planned:.3f} s")
                if abs(planned - duration) > 0.1:
                    problems.append(
                        f"audio plan lasts {planned:.3f} s but the video {duration:.3f} s"
                    )
        elif audio is not None:
            problems.append("unexpected audio stream in the output")
        size = self._tmp.stat().st_size
        if self.max_bytes is not None and size > self.max_bytes:
            problems.append(f"file has {size} bytes, more than max_bytes={self.max_bytes}")
        _raise(what, problems)
        return WrittenVideo(
            path=self._tmp,
            frames=self.frames_written,
            fps=format_rate(self.rate),
            width=self.width,
            height=self.height,
            sha256=sha256_file(self._tmp),
            bytes=size,
            audio=audio,
            audio_parts=tuple(self._audio_rows),
            settings=self.settings(),
            comment=self.comment,
        )

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            self.abort()
            return
        if self.result is None and not self._closed:
            self.close()


def _probe_output(path: Path) -> tuple[dict[str, Any] | None, dict[str, str]]:
    """The audio stream of a written file (None without one) and the file's metadata tags."""
    args = [
        ffprobe_path(),
        "-v",
        "error",
        "-hide_banner",
        "-show_entries",
        "stream=codec_type,codec_name,sample_rate,channels,duration:format_tags",
        "-of",
        "json=compact=1",
        media_arg(path),
    ]
    proc = run_tool(args, what=f"probe {path.name}")
    data = json.loads(proc.stdout.decode("utf-8"))
    fmt = data.get("format") or {}
    tags = {str(key): str(value) for key, value in (fmt.get("tags") or {}).items()}
    streams = [s for s in data.get("streams") or [] if s.get("codec_type") == "audio"]
    if not streams:
        return None, tags
    if len(streams) > 1:
        raise RenderError(f"probe audio of {path.name}", [f"{len(streams)} audio streams"])
    stream = streams[0]
    audio = {
        "codec_name": str(stream.get("codec_name") or "unknown"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "duration_s": _float_or_none(stream.get("duration")),
    }
    return audio, tags


def write_video(
    path: str | Path,
    frames: Iterable[np.ndarray],
    fps: str | Fraction = "25/1",
    **options: Any,
) -> WrittenVideo:
    """Write ``frames`` (all the same size) with :class:`VideoWriter`; returns its result."""
    iterator = iter(frames)
    try:
        first = check_frame(next(iterator), "write_video")
    except StopIteration:
        raise RenderError("write_video", ["no frames given"]) from None
    height, width = first.shape[:2]
    writer = VideoWriter(path, width, height, fps, **options)
    try:
        writer.write(first)
        for frame in iterator:
            writer.write(frame)
    except BaseException:
        writer.abort()
        raise
    return writer.close()


# --------------------------------------------------------------------------------------
# highlight edits
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EditSegment:
    """A run of output frames: ``kind`` and the canonical source frame of each output frame
    (a source frame appears ``repeat`` times in a row for slow motion and freezes)."""

    kind: SegmentKind
    source_frames: tuple[int, ...]
    repeat: int = 1

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.kind not in SEGMENT_KINDS:
            problems.append(f"kind must be one of {SEGMENT_KINDS}, got {self.kind!r}")
        frames = tuple(self.source_frames)
        if not frames:
            problems.append("a segment needs at least one frame")
        if any(isinstance(f, bool) or not isinstance(f, int | np.integer) or f < 0 for f in frames):
            problems.append("source_frames must be non-negative ints")
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, int) or self.repeat < 1:
            problems.append(f"repeat must be an int >= 1, got {self.repeat!r}")
        _raise("EditSegment", problems)
        object.__setattr__(self, "source_frames", tuple(int(f) for f in frames))

    @property
    def n_frames(self) -> int:
        """Output frames of this segment."""
        return len(self.source_frames) * self.repeat


@dataclass(frozen=True)
class EditPlan:
    """An edit: segments in output order, at the source frame rate ``fps``."""

    fps: Fraction
    segments: tuple[EditSegment, ...]
    key_frame: int

    @property
    def n_frames(self) -> int:
        return sum(segment.n_frames for segment in self.segments)

    def source_frames(self) -> list[int]:
        """Distinct canonical source frames used, sorted."""
        return sorted({f for segment in self.segments for f in segment.source_frames})

    def table(self) -> list[dict[str, Any]]:
        """The segment table: output frame range to source frame range, per segment."""
        rows: list[dict[str, Any]] = []
        out = 0
        for segment in self.segments:
            rows.append(
                {
                    "kind": segment.kind,
                    "out_start": out,
                    "out_end": out + segment.n_frames - 1,
                    "src_start": segment.source_frames[0],
                    "src_end": segment.source_frames[-1],
                    "repeat": segment.repeat,
                    "frames": segment.n_frames,
                    "seconds": float(segment.n_frames / self.fps),
                }
            )
            out += segment.n_frames
        return rows

    def output_to_source(self) -> list[int]:
        """Canonical source frame of every output frame, in output order."""
        mapping: list[int] = []
        for segment in self.segments:
            for frame in segment.source_frames:
                mapping.extend([frame] * segment.repeat)
        return mapping


def plan_highlight_edit(
    start_frame: int,
    end_frame: int,
    key_frame: int,
    fps: str | Fraction = "25/1",
    *,
    slow_factor: int = 4,
    slow_before_s: float = 0.6,
    slow_after_s: float = 0.4,
    freeze_s: float = 2.0,
    normal: bool = True,
    freeze_inside_slow: bool = True,
    cuts: Sequence[int] = (),
) -> EditPlan:
    """Plan a highlight edit around ``key_frame`` (canonical frames, inclusive window).

    Segments, in output order:

    1. "normal": every frame of [start_frame, end_frame] at normal speed (if ``normal``);
    2. "slow": frames from key_frame - round(slow_before_s * fps) up to the key frame, each
       repeated ``slow_factor`` times (no interpolation);
    3. "freeze": the key frame held for round(freeze_s * fps) frames;
    4. "slow": the frames after the key frame up to key_frame + round(slow_after_s * fps).

    The slow range is clipped to the window and to the shot of the key frame: ``cuts`` are
    the first frames of new shots (as in cuts.json), and slow motion never crosses one.
    With ``freeze_inside_slow=False`` the slow part runs through the key frame and the
    freeze follows it. ``slow_factor=1`` or ``freeze_s=0`` leaves that part out. Raises
    :class:`RenderError` listing every problem.
    """
    problems: list[str] = []
    for name, value in (
        ("start_frame", start_frame),
        ("end_frame", end_frame),
        ("key_frame", key_frame),
    ):
        if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0:
            problems.append(f"{name} must be a non-negative int, got {value!r}")
    try:
        rate = parse_rate(fps)
    except ValueError as exc:
        problems.append(str(exc))
        rate = Fraction(25)
    if isinstance(slow_factor, bool) or not isinstance(slow_factor, int) or slow_factor < 1:
        problems.append(f"slow_factor must be an int >= 1, got {slow_factor!r}")
    for name, value in (
        ("slow_before_s", slow_before_s),
        ("slow_after_s", slow_after_s),
        ("freeze_s", freeze_s),
    ):
        number = _finite(name, value, problems)
        if number is not None and number < 0:
            problems.append(f"{name} must be >= 0, got {number}")
    bad_cuts = [
        cut
        for cut in cuts
        if isinstance(cut, bool) or not isinstance(cut, int | np.integer) or cut < 0
    ]
    if bad_cuts:
        problems.append(f"cuts must be non-negative ints, got {bad_cuts!r}")
    if not problems and not start_frame <= key_frame <= end_frame:
        problems.append(
            f"key_frame {key_frame} must lie in the window [{start_frame}, {end_frame}]"
        )
    _raise("plan_highlight_edit", problems)
    start, end, key = int(start_frame), int(end_frame), int(key_frame)
    segments: list[EditSegment] = []
    if normal:
        segments.append(EditSegment("normal", tuple(range(start, end + 1))))
    before = round_half_away(Fraction(slow_before_s).limit_denominator(10**6) * rate)
    after = round_half_away(Fraction(slow_after_s).limit_denominator(10**6) * rate)
    hold = round_half_away(Fraction(freeze_s).limit_denominator(10**6) * rate)
    # the key frame's shot: from the last cut at or before it to the frame before the next
    shot_first = max((int(cut) for cut in cuts if cut <= key), default=start)
    shot_last = min((int(cut) - 1 for cut in cuts if cut > key), default=end)
    slow_first = max(start, shot_first, key - before)
    slow_last = min(end, shot_last, key + after)
    use_slow = slow_factor > 1
    if freeze_inside_slow:
        if use_slow:
            segments.append(EditSegment("slow", tuple(range(slow_first, key + 1)), slow_factor))
        if hold > 0:
            segments.append(EditSegment("freeze", (key,), hold))
        if use_slow and slow_last > key:
            segments.append(EditSegment("slow", tuple(range(key + 1, slow_last + 1)), slow_factor))
    else:
        if use_slow:
            segments.append(
                EditSegment("slow", tuple(range(slow_first, slow_last + 1)), slow_factor)
            )
        if hold > 0:
            segments.append(EditSegment("freeze", (key,), hold))
    if not segments:
        raise RenderError("plan_highlight_edit", ["the plan has no segment"])
    return EditPlan(fps=rate, segments=tuple(segments), key_frame=key)


AnnotationSource = Callable[[int, str], FrameAnnotations | None]
"""Callback (canonical source frame, segment kind) -> annotations for that output frame."""


@dataclass(frozen=True)
class EditResult:
    """A rendered and verified edit: the video, its plan and the manifest written beside it."""

    video: WrittenVideo
    plan: EditPlan
    manifest_path: Path
    manifest: dict[str, Any]


def _audio_extent(source: Path, offset_s: float | None) -> tuple[Fraction, Fraction | None]:
    """Canonical times of the first audio sample of ``source`` and of the end of its audio
    (None when ffprobe reports no duration), for the offset an :class:`AudioSpan` uses."""
    facts = probe_audio(source)
    audio_start = Fraction(facts.audio_start_s - facts.format_start_s).limit_denominator(10**9)
    first = audio_start - _offset(facts, offset_s)
    if facts.duration_s is None:
        return first, None
    return first, first + Fraction(facts.duration_s).limit_denominator(10**9)


def _edit_audio(
    plan: EditPlan,
    source: Path,
    replay_audio: Literal["continue", "silence"],
    replay_gain_db: float,
    offset_s: float | None,
) -> list[AudioPart]:
    """Audio parts that match the plan's segments exactly (see :func:`render_edit`).

    Every part is cut to the source audio: what it would read before the first or after the
    last audio sample is a :class:`Silence` part. With ``"continue"``, a replay part that
    would read past the end of the audio plays the window's own audio instead (the last
    normal segment, else the plan's source frames): from the window's start, on through the
    following replay parts and over again from the start when the window's audio runs out.
    A window with less than :data:`REPLAY_LOOP_MIN_S` of audio leaves the replay silent.
    """
    rate = plan.fps
    first, end = _audio_extent(source, offset_s)

    def read(start: Fraction, n: int, gain_db: float) -> list[AudioPart]:
        """``n`` samples from canonical time ``start``: the source where it has audio."""
        # rounded up, so that a source span never starts before the first audio sample
        # (an offset off the 1/48000 s grid would otherwise put it up to half a sample early)
        lead = min(n, max(0, math.ceil((first - start) * AUDIO_SAMPLE_RATE)))
        stop = n if end is None else min(n, max(lead, _samples(end - start)))
        pieces: list[AudioPart] = []
        if lead:
            pieces.append(Silence(Fraction(lead, AUDIO_SAMPLE_RATE)))
        if stop > lead:
            at = start + Fraction(lead, AUDIO_SAMPLE_RATE)
            length = Fraction(stop - lead, AUDIO_SAMPLE_RATE)
            pieces.append(AudioSpan(source, at, length, gain_db, offset_s))
        if n > stop:
            pieces.append(Silence(Fraction(n - stop, AUDIO_SAMPLE_RATE)))
        return pieces

    frames = plan.source_frames()
    window = (Fraction(frames[0]) / rate, Fraction(frames[-1] + 1) / rate)
    parts: list[AudioPart] = []
    cursor: Fraction | None = None  # where the replay continues after the window
    replayed: int | None = None  # samples of the window's audio replayed, once it is used
    for segment in plan.segments:
        duration = Fraction(segment.n_frames) / rate
        n = _samples(duration)
        if segment.kind == "normal":
            start = Fraction(segment.source_frames[0]) / rate
            parts += read(start, n, 0.0)
            cursor = Fraction(segment.source_frames[-1] + 1) / rate
            window, replayed = (start, cursor), None
            continue
        if replay_audio == "silence":
            parts.append(Silence(Fraction(n, AUDIO_SAMPLE_RATE)))
            continue
        if cursor is None:
            cursor = Fraction(segment.source_frames[0]) / rate
        if replayed is None and (end is None or cursor + duration <= end):
            parts += read(cursor, n, replay_gain_db)
            cursor += duration
            continue
        # The audio after the window has run out: replay the window's own audio.
        replayed = replayed or 0
        own_start = max(window[0], first)
        own = max(0, _samples(min(window[1], end) - own_start)) if end is not None else 0
        if own < REPLAY_LOOP_MIN_S * AUDIO_SAMPLE_RATE:
            parts.append(Silence(Fraction(n, AUDIO_SAMPLE_RATE)))
            continue
        while n > 0:
            take = min(n, own - replayed)
            at = own_start + Fraction(replayed, AUDIO_SAMPLE_RATE)
            length = Fraction(take, AUDIO_SAMPLE_RATE)
            parts.append(AudioSpan(source, at, length, replay_gain_db, offset_s))
            replayed = (replayed + take) % own
            n -= take
    return parts


def render_edit(
    video: str | Path,
    out: str | Path,
    plan: EditPlan,
    *,
    annotate: AnnotationSource | None = None,
    freeze_layer: FrameAnnotations | None = None,
    style: Style | None = None,
    audio_source: str | Path | None = None,
    audio_offset_s: float | None = None,
    replay_audio: Literal["continue", "silence"] = "continue",
    replay_gain_db: float = -6.0,
    part_badges: bool = True,
    source_fps: str | None = None,
    source_sha256: str | None = None,
    restrictions: Sequence[str] = (),
    crf: int = DEFAULT_CRF,
    overwrite: bool = False,
    max_buffer_frames: int = 100,
) -> EditResult:
    """Render ``plan`` from the canonical ``video`` into the verified mp4 ``out``.

    Frames are decoded once, sequentially (:func:`gbf.ingest.iter_frames`); frames that a
    later segment reuses (slow motion, freeze) are kept in memory, at most
    ``max_buffer_frames`` of them. Each output frame shows ``annotate(source_frame, kind)``;
    freeze frames add ``freeze_layer`` on top; with ``part_badges`` slow-motion frames get a
    "慢放 ×n" badge and freeze frames a "定格" badge.

    Audio (when ``audio_source`` is given): the normal segment carries the original audio
    of its own frames; slow-motion and freeze segments carry, with ``replay_audio=
    "continue"``, the original audio that follows the window (as a live broadcast keeps its
    sound running under a replay), ``replay_gain_db`` quieter, or silence. Where the source
    audio ends before a replay part does, that part and the following ones replay the
    window's own audio from its start (over again when it runs out). Time that no source
    audio covers is a silence part, and the manifest lists it as one. ``audio_offset_s``
    aligns the sound as :class:`AudioSpan` says (:func:`first_sample_offset` converts the
    pipeline's offset); the manifest records the offset used.

    Writes ``out`` (carrying ``restrictions`` in its metadata, see :class:`VideoWriter`)
    and ``<out stem>.render.json`` (the manifest: segment table, output hash, fonts, style,
    audio parts, every text drawn with its evidence ids, restrictions, tool versions; no
    absolute paths). Returns an :class:`EditResult`.
    """
    video = Path(video)
    out = Path(out)
    style = (style or Style()).with_fonts()
    problems: list[str] = []
    if not isinstance(plan, EditPlan):
        problems.append(f"plan must be an EditPlan, got {type(plan).__name__}")
    if not video.is_file():
        problems.append(f"video is not a file: {video}")
    if audio_source is not None and not Path(audio_source).is_file():
        problems.append(f"audio_source is not a file: {audio_source}")
    if replay_audio not in ("continue", "silence"):
        problems.append(f"replay_audio must be 'continue' or 'silence', got {replay_audio!r}")
    if audio_offset_s is not None:
        _finite("audio_offset_s", audio_offset_s, problems)
    try:
        _restriction_tags("render_edit", restrictions)
    except RenderError as exc:
        problems.extend(exc.problems)
    _raise(f"render_edit({out.name})", problems)
    manifest_path = out.with_name(out.stem + ".render.json")
    if not overwrite:
        existing = [p.name for p in (out, manifest_path) if p.exists()]
        if existing:
            raise FileExistsError(f"{', '.join(existing)} exist; pass overwrite=True")

    needed = plan.source_frames()
    first, last = needed[0], needed[-1]
    # The first segment is written while decoding when it shows frames in decode order
    # once each; every frame a later segment needs is kept in memory until then.
    head = plan.segments[0]
    frames_in_order = all(
        a < b for a, b in zip(head.source_frames, head.source_frames[1:], strict=False)
    )
    stream_head = head.repeat == 1 and frames_in_order
    streamed = set(head.source_frames) if stream_head else set()
    later = {f for segment in plan.segments[1:] for f in segment.source_frames}
    if not stream_head:
        later |= set(head.source_frames)
    if len(later) > max_buffer_frames:
        raise RenderError(
            f"render_edit({out.name})",
            [
                f"{len(later)} frames would be kept in memory for slow motion and freezes, "
                f"more than max_buffer_frames={max_buffer_frames}; shorten the slow range"
            ],
        )

    audio_parts: list[AudioPart] = []
    if audio_source is not None:
        audio_parts = _edit_audio(
            plan, Path(audio_source), replay_audio, float(replay_gain_db), audio_offset_s
        )
    texts: dict[tuple[str, str, tuple[str, ...]], int] = {}

    def annotations_for(frame_index: int, kind: str) -> FrameAnnotations:
        base = annotate(frame_index, kind) if annotate is not None else None
        result = base if base is not None else FrameAnnotations()
        if not isinstance(result, FrameAnnotations):
            raise RenderError("render_edit", [f"annotate returned {type(result).__name__}"])
        if kind == "freeze" and freeze_layer is not None:
            result = merge_annotations(result, freeze_layer)
        if part_badges and kind == "slow":
            factor = next(s.repeat for s in plan.segments if s.kind == "slow")
            result = merge_annotations(result, FrameAnnotations(badges=(Badge(f"慢放 ×{factor}"),)))
        if part_badges and kind == "freeze":
            result = merge_annotations(result, FrameAnnotations(badges=(Badge("定格"),)))
        return result

    def record(annotations: FrameAnnotations, count: int, size: tuple[int, int]) -> None:
        # the texts as drawn on a frame of this size (the card lays out its rows for it)
        for entry in annotation_texts(annotations, size, style):
            texts[entry] = texts.get(entry, 0) + count

    buffer: dict[int, np.ndarray] = {}
    writer: VideoWriter | None = None
    frames = iter_frames(video, first, last + 1, fps=source_fps)
    try:
        for index, frame in frames:
            if writer is None:
                height, width = frame.shape[:2]
                writer = VideoWriter(
                    out,
                    width,
                    height,
                    plan.fps,
                    crf=crf,
                    audio=audio_parts,
                    overwrite=overwrite,
                    restrictions=restrictions,
                )
            if index in later:
                buffer[index] = frame
            if index in streamed:
                annotations = annotations_for(index, head.kind)
                record(annotations, 1, (writer.width, writer.height))
                writer.write(draw_annotations(frame, annotations, style))
        if writer is None:  # pragma: no cover - iter_frames raises for an empty range
            raise RenderError("render_edit", ["no frames decoded"])
        for segment in plan.segments[1 if stream_head else 0 :]:
            for frame_index in segment.source_frames:
                annotations = annotations_for(frame_index, segment.kind)
                record(annotations, segment.repeat, (writer.width, writer.height))
                drawn = draw_annotations(buffer[frame_index], annotations, style)
                writer.write(drawn, repeat=segment.repeat)
        written = writer.close()
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()  # stops the decoder if we left the loop early
    if written.frames != plan.n_frames:  # pragma: no cover - VideoWriter verified the count
        raise RenderError(
            "render_edit", [f"wrote {written.frames} frames, plan has {plan.n_frames}"]
        )

    manifest = {
        "schema": "gbf.render-manifest/1",
        "kind": "highlight_edit",
        "output": written.manifest(),
        "source": {
            "file_name": video.name,
            "sha256": source_sha256 if source_sha256 is not None else sha256_file(video),
            "first_frame": first,
            "last_frame": last,
            "fps_mapping": source_fps,
        },
        "key_frame": plan.key_frame,
        "segments": plan.table(),
        "slow_motion": {
            "method": "frame_repeat",
            "interpolated_frames": 0,
        },
        "audio": {
            "source": probe_audio(audio_source).manifest() if audio_source is not None else None,
            "offset_s": (
                float(_offset(probe_audio(audio_source), audio_offset_s))
                if audio_source is not None
                else None
            ),
            "replay_audio": replay_audio if audio_source is not None else None,
            "replay_gain_db": replay_gain_db if audio_source is not None else None,
        },
        "style": style.manifest(),
        "texts": [
            {"kind": kind, "text": text, "evidence_ids": list(ids), "frames": count}
            for (kind, text, ids), count in texts.items()
        ],
        "restrictions": sorted(set(restrictions)),
        "tools": {
            "ffmpeg": ffmpeg_version(),
            "gbf": gbf.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
    }
    write_json(manifest_path, manifest)
    return EditResult(video=written, plan=plan, manifest_path=manifest_path, manifest=manifest)


def format_clock(frame: int, fps: str | Fraction | int = "25/1") -> str:
    """Canonical frame as "mm:ss.cc", cc hundredths of a second (floored, exact rationals).

    The one clock text of titles, summaries, cards, banners and the report
    (:func:`gbf.util.clock.format_clock`): frame 476 at 25/1 is 19.04 s, "00:19.04", and a
    frame shows the same time everywhere at 30000/1001 too. Raises RenderError listing
    every problem with the frame and the rate.
    """
    try:
        return _clock_text(frame, fps)
    except ClockError as exc:
        raise RenderError("format_clock", exc.problems) from exc


# --------------------------------------------------------------------------------------
# scenes: per-frame annotations of a highlight, built from the run's tables
# --------------------------------------------------------------------------------------
HIGHLIGHT_TYPES_ZH: Mapping[str, str] = TYPE_NAMES_ZH
"""Chinese names of the highlight types of docs/design/m1-highlights.md, the same mapping
as gbf.highlights.TYPE_NAMES_ZH (titles and summaries), so a type has one name everywhere.
A name says no more than the type's evidence: a ball_strike is an abrupt change of the ball's image
velocity near a player (a shot, a pass, a touch or a deflection alike), a duel_tackle two
opposing players close together near the ball (who wins the ball is not measured), and a
closeup_reaction's shot need not be a close-up."""

TEAM_NAMES_ZH: tuple[tuple[str, str], ...] = (("left", "左侧球队"), ("right", "右侧球队"))
"""Default Chinese team names, by the side of the pitch the team defends (x < 0 is "left"),
the same words highlights.json uses. Pass kit colours or club names when they are known."""

#: In a duel (type ``duel_tackle``), the line between the duel pair (:data:`LINK_LABELS_ZH`)
#: is drawn only within this many seconds of a moment of that pair: the duel cue's frame
#: for the cue's pair, the joined shot event's frame for its pair (in that event's shot),
#: and the key frame for gbf.moments' ``key_pair`` (in the key frame's shot), whichever is
#: nearer; the contest, not the seconds of approach before it. Pairs are compared as sets.
#: With a ball track, also only while the duel cue's rule holds for the pair on the frame
#: (gbf.highlights): the ball within DUEL_BALL_RADIUS_M of both players on the grass and
#: the pair closer than DUEL_FAR_M, or, without ground positions, within DUEL_BALL_RADIUS_H
#: and DUEL_FAR_H box heights in the image; once the ball has gone, it is no duel.
#: Outside them, on pitch frames, the focus player's nearest opponent is linked instead.
#: No other highlight draws a pair's line, as its pairs are not what it is about: gbf.moments
#: gives ``key_pair`` to any highlight whose key moments used a pair, and turns a highlight
#: whose pair it finds in contact or falling into a duel. On every frame of the others,
#: their key frame included, the line is the focus player's nearest opponent, the one the
#: data card's 最近对手距离 is about.
DUEL_LINK_WINDOW_S = 1.0

LINK_LABELS_ZH: Mapping[str, str] = MappingProxyType({"duel": "争抢", "nearest": "最近对手"})
"""What a scene's line links, the first word of its label: the duel pair of a duel (争抢)
or the focus player's nearest opponent (最近对手, the nearest outfield player of the other
team, as gbf.measure counts it for the data card, among the players observed on the frame;
a box on the focus player's own spot or a person off the pitch is no opponent,
gbf.track_tools). On pitch frames the distance in metres follows ("争抢 · 1.6 m",
"最近对手 · 4.2 m"): measured on the frame, or held over a frame on which the focus player
is filled in (:data:`TRACK_HOLD_MAX_GAP`); never one below gbf.track_tools'
MIN_PLAYER_GAP_M, and never to a partner drawn at a filled-in position: then the line says
only what it links."""

#: In a merged or extended highlight (``merged_parts``, or a shot joined after a cut), on
#: the frames of the shots before the key frame's, the focus player is marked only while
#: within this many metres of the ball on the grass (the minimap's ball): seconds before the
#: cut the play there can be about other players than the ones the key frame is about.
FOCUS_BALL_MAX_M = 5.0

#: The facing arrow of a scene (:meth:`HighlightScene.annotate`), the 3D lift's facing of the
#: focus player drawn back on the original frame, is this many metres long on the grass,
#: from the player's foot point: it shows a direction, not a distance, and reaches past the
#: focus ring (0.9 m) so that it reads at broadcast sizes.
FACING_ARROW_LENGTH_M = 2.0

FACING_LABEL_ZH = "朝向（3D 参考）"
"""Label of the facing arrow: the direction the height-prior skeleton of gbf.lift3d faces, a
3D reference drawn back on the frame (with the 未验证 badge, as its error is unmeasured)."""

#: The scene puts the ball on the minimap only while the ground assumption behind its
#: pitch position holds: the ball table's ``ground_assumption`` is set, and the position
#: moves no faster than this over the frames up to BALL_SPEED_SPAN_FRAMES on either side in
#: the same shot (the first and the last of them). A ball rolling on the grass is slower; a
#: ball in the air projects onto the grass ahead of itself and races across the minimap.
BALL_GROUND_MAX_SPEED_MPS = 30.0
BALL_SPEED_SPAN_FRAMES = 2

_PLAYER_ROLES = frozenset({"player", "goalkeeper"})
_TRACK_COLUMNS = ("frame", "track_id", "x1", "y1", "x2", "y2", "role", "team")
_CALIB_COLUMNS = ("K", "R", "C", "width", "height")
_BALL_COLUMNS = ("frame", "u", "v", "status")
_KINEMATICS_COLUMNS = ("frame", "track_id", "speed", "valid")
_POSE_COLUMNS = ("frame", "track_id", "keypoints")
_LIFT_COLUMNS = ("frame", "track_id", "method", "facing_deg", "facing_conf", "facing_sigma_deg")
_TITLE_TIME_PREFIX = re.compile(r"^\s*\d+:\d{2}\.\d{2}\s*（第\s*\d+\s*帧）\s*")


def _missing_columns(name: str, table: Any, columns: Sequence[str], problems: list[str]) -> None:
    if table is None:
        return
    if not isinstance(table, pd.DataFrame):
        problems.append(f"{name} must be a pandas DataFrame, got {type(table).__name__}")
        return
    missing = [column for column in columns if column not in table.columns]
    if missing:
        problems.append(f"{name} misses column(s): {', '.join(missing)}")


@dataclass(frozen=True)
class SceneTables:
    """The run tables a highlight render reads, as pandas DataFrames with the columns of
    docs/design/m1-highlights.md (extra columns are ignored, optional tables may be None).

    Attributes:
        tracks: tracks.parquet: frame, track_id, x1, y1, x2, y2, role, team and, when
            present, jersey, pitch_x and pitch_y (NaN unless pitch-usable).
        calib: calib.parquet: frame_index (or frame), K, R, C (9, 9 and 3 numbers), width
            and height; gives perspective-correct ground rings and the camera's view on the
            minimap.
        ball: ball_track.parquet: frame, u, v, status (detected, interpolated, missing) and,
            when present, radius_px, pitch_x and pitch_y.
        kinematics: kinematics.parquet: frame, track_id, speed (m/s) and valid.
        pose: pose.parquet: frame, track_id and keypoints (26 x [u, v, score], flattened).
        lift: lift3d.parquet: frame, track_id, method, facing_deg, facing_conf and
            facing_sigma_deg (gbf.lift3d); the focus player's facing is drawn back on pitch
            frames (:class:`HighlightScene`, ``facing_gates``).
        team_verdicts: optional (track_id, claimed team) to team-check verdict; unchecked
            people without ground positions or with cut-off boxes stay neutral context.
    """

    tracks: pd.DataFrame
    calib: pd.DataFrame | None = None
    ball: pd.DataFrame | None = None
    kinematics: pd.DataFrame | None = None
    pose: pd.DataFrame | None = None
    lift: pd.DataFrame | None = None
    team_verdicts: Mapping[tuple[int, str], str] | None = None

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.tracks is None:
            problems.append("tracks is required")
        _missing_columns("tracks", self.tracks, _TRACK_COLUMNS, problems)
        _missing_columns("calib", self.calib, _CALIB_COLUMNS, problems)
        if isinstance(self.calib, pd.DataFrame) and not (
            "frame_index" in self.calib.columns or "frame" in self.calib.columns
        ):
            problems.append("calib misses column frame_index (or frame)")
        _missing_columns("ball", self.ball, _BALL_COLUMNS, problems)
        _missing_columns("kinematics", self.kinematics, _KINEMATICS_COLUMNS, problems)
        _missing_columns("pose", self.pose, _POSE_COLUMNS, problems)
        _missing_columns("lift", self.lift, _LIFT_COLUMNS, problems)
        _raise("SceneTables", problems)


@dataclass(frozen=True)
class _Facing:
    """One ``lift`` row as the overlay reads it (NaN where the table has no value)."""

    lifted: bool
    facing_deg: float
    conf: float
    sigma_deg: float


def _clean_label(value: Any) -> str | None:
    """A text cell as a string, or None for None, NaN, <NA> and empty strings."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in ("nan", "none", "<na>") else None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _track_id(value: Any) -> int | None:
    """``value`` as a track id (an int >= 0, not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0:
        return None
    return int(value)


def _key_roles(
    highlight: Mapping[str, Any], problems: list[str]
) -> tuple[tuple[int, int] | None, int | None]:
    """(``key_pair``, ``key_focus_track``) of a highlight: the duel pair gbf.moments
    analysed and the player its key frame is about, each None when absent or null."""
    pair: tuple[int, int] | None = None
    value = highlight.get("key_pair")
    if value is not None:
        items = list(value) if isinstance(value, Sequence) and not isinstance(value, str) else []
        ids = [_track_id(item) for item in items]
        if len(ids) != 2 or ids[0] is None or ids[1] is None or ids[0] == ids[1]:
            problems.append(f"highlight key_pair must be two different track ids, got {value!r}")
        else:
            pair = (ids[0], ids[1])
    focus = _track_id(highlight.get("key_focus_track"))
    if highlight.get("key_focus_track") is not None and focus is None:
        problems.append(
            "highlight key_focus_track must be a track id (an int >= 0), "
            f"got {highlight.get('key_focus_track')!r}"
        )
    return pair, focus


#: For display only, a track missing for at most this many frames between two of its rows
#: in the same shot is filled in by :class:`HighlightScene` (box, pitch position and
#: skeleton interpolated, team, role and jersey held): slow motion repeats every frame, so
#: a gap of a third of a second (8 frames at 25 fps, over a second at 4x slow motion) would
#: otherwise blink on screen. Never across a cut, never for boxes without a track (id -1:
#: those are not one person), and the tables are not changed. No number shown is measured
#: on a filled-in position, and none is chosen by one: the nearest opponent is chosen among
#: the players observed on the frame; over a frame where the focus player is filled in, its
#: line holds the distance last measured to the opponent that was nearest then (at most this
#: many frames before, with that frame's evidence), a partner drawn at a filled-in position
#: gets no distance, and a filled-in box shows no speed, even where ``kinematics`` has a
#: (then interpolated) row for the frame. An observed player's speed label holds its last
#: valid speed, greyed, over at most this many frames without a valid one. A filled-in
#: player is drawn and counts as on its frame (for the duel pair's line, the key frame's
#: focus and the freeze frame's spotlight).
TRACK_HOLD_MAX_GAP = 8


def _lerp(a: Sequence[float], b: Sequence[float], weight: float) -> tuple[float, ...]:
    return tuple(x + (y - x) * weight for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class _Player:
    track_id: int
    box: tuple[float, float, float, float]
    role: str | None
    team: str | None
    jersey: str | None
    pitch: tuple[float, float] | None
    observed: bool = True
    """False on a frame filled in over a gap of the track (:data:`TRACK_HOLD_MAX_GAP`)."""


@dataclass(frozen=True)
class _ShotRoles:
    """The roles on the frames of one shot of a :class:`HighlightScene`."""

    focus: int | None
    key_frame_focus: int | None
    """In the key frame's shot, the key frame's own focus; elsewhere ``focus``."""
    pair: tuple[int, int] | None
    involved: frozenset[int]


class HighlightScene:
    """The overlay of one highlight, frame by frame, built from :class:`SceneTables`.

    :meth:`annotate` has the :data:`AnnotationSource` signature of :func:`render_edit`, so
    ``render_edit(video, out, plan, annotate=scene.annotate,
    freeze_layer=scene.freeze_layer(rows))`` renders the highlight on the original frames.
    What a frame shows:

    - every player and goalkeeper of the frame: the focus player with a glowing ground
      ring, corner brackets, a pointer and a label; the other involved players with a ring
      and a label; everyone else as a faint ring (``show_context``). On pitch frames the
      rings are circles of 0.9 m (focus) or 0.7 m on the grass, projected with the frame's
      camera; elsewhere they are ellipses under the box. A box without a track (id -1) and,
      on pitch frames, a person off the pitch (:meth:`_off_pitch`: a coach, a steward)
      only ever get a faint ring without a team colour: no role, label, line or minimap
      dot;
    - labels: team name and jersey (or track id) and, on pitch frames, the speed in km/h
      from ``kinematics`` where it is valid, each speed with its own 未验证 marker (a chip
      on the focus label, the compact marker on the others); over at most
      :data:`TRACK_HOLD_MAX_GAP` frames without a valid speed, the last one, greyed. The
      speeds of a player whose peak speed gbf.measure withholds (``peak_speed_withheld``,
      the track ids of the highlight's unavailable peak-speed records) instead name the
      observed peak lower bound, 峰值≥25.2 km/h, with that peak's frame evidence. Labels keep
      off the focus and the involved players where they can, and
      a label moved a step or more gets a leader line back to its player
      (:func:`draw_annotations`);
    - in a duel, a line between the duel pair around that pair's own moments
      (:data:`DUEL_LINK_WINDOW_S`) while the ball is at the pair (the duel cue's rule),
      labelled 争抢 and, on pitch frames, with the distance (争抢 · 1.6 m); otherwise, and on
      every frame of the other highlights, on pitch frames, a line from the focus player to
      the nearest outfield player of the other team (最近对手 · 4.2 m,
      :data:`LINK_LABELS_ZH`), chosen among the players observed on the frame;
    - on pitch frames the minimap with every player, the ball's ground-plane position and the
      camera's view; on other frames the minimap says that positions are not shown, and the
      仅画面平面 badge is set;
    - the ball with a ring (dashed when interpolated) and a fading trail of up to
      ``trail_frames`` earlier positions in the same shot (a trail never crosses a cut);
    - 2D skeletons of the involved players from ``pose``;
    - on pitch frames, the focus player's facing from the 3D lift (``lift``), drawn back on
      the frame: a dashed arrow on the grass from the player's foot point,
      :data:`FACING_ARROW_LENGTH_M` long along the facing, both ends projected with the
      frame's camera as the rings are, labelled 朝向（3D 参考） with the 未验证 badge (the
      frame's 未验证 joins the badge row only where the arrow is drawn). Only where the
      frame's lifted facing passes gbf.measure's per-frame gates of body-orientation
      (``facing_gates``, :func:`gbf.measure.facing_frame_ok`, the front-back reversal
      check against the frame's 2D trunk lean included), the focus player has a row of its
      own on the frame (not filled in) and its pose of that frame, the one the facing was
      lifted from, is in ``pose``. The 3D lift reaches the frame only this way, as a
      reference; everything else is drawn from the 2D tables;
    - the lower-third banner: the highlight title, the frame's time and frame number.

    The roles (focus, duel pair, involved players) are set per shot, and :meth:`roles_at`
    gives them for any frame:

    - in the highlight's own shots, the focus is ``focus_track`` when it is given, else by
      type: the sprinter of a sprint (the speed cue's track), the first of the duel cue's
      pair of a duel, the attacking team's first involved player of an attack in the box,
      otherwise (and when the cue names nobody) the first of ``track_ids``; a duel whose
      key frame is in one of its own shots takes that frame's ``key_focus_track`` instead,
      where it applies (below). The pair is the duel cue's pair, the involved players are
      ``track_ids``;
    - in the shot a highlight extends into after a cut, the players of
      ``cut_emphasis.joined_shot_event``: its first player is the focus, its two players (or
      the first with its ``partner_track_id``) are the pair;
    - in the shot of the key frame (``peak_frame``), gbf.moments' key fields take the roles
      where that shot has track rows of their players: ``key_pair`` (both players) becomes
      the pair, ``key_focus_track`` the focus. A null ``key_focus_track`` leaves the focus as
      it is, except on the key frame of a ball event (a key moment of a ball kind at
      ``peak_frame``): that frame, its freeze and its data card are about the event's
      player;
    - in a merged highlight (``merged_parts``), a shot after the key frame's without any of
      the key frame's players (``key_pair``, ``key_focus_track``, the ball event's player,
      the key frame's focus) is the later part's: nobody is its focus. In a merged or
      extended one, on the frames before the key frame's shot, the focus is marked only
      within :data:`FOCUS_BALL_MAX_M` of the ball.

    A shot without track rows of its focus (on the pitch) takes the first focus candidate
    it has such rows of (with none, nobody is marked as the focus there). The key frame
    takes the first candidate it shows (with a row, or filled in, see below), so that the
    freeze frame's spotlight and data card are about a player the frame shows (with none,
    the focus of its shot); that holds for the key frame only, the rest of its shot keeps
    the shot's focus.
    The candidates, in order: ``key_focus_track`` where it applies (in its shot, and in a
    duel's own shots when the key frame is in one of them), the shot's own focus (by type,
    or the joined event's), then the involved players: ``track_ids`` (in the joined event's
    shot, the event's players) and, in the key frame's shot, ``key_pair``, the pair first
    when ``key_focus_track`` applies there. A given ``focus_track`` is never replaced in the
    highlight's own shots. The focus of any frame of a shot, and the key fields' players
    where they apply, are involved on every frame of that shot. :attr:`focus` is the focus
    the highlight's own shots start from: ``focus_track`` when given, else the type's
    focus or, for a duel whose key frame is in one of them, that ``key_focus_track`` where
    it applies; never a player chosen because a frame shows them.

    Pitch frames are ``pitch_frames`` when given (for example the frames where
    signals.parquet has ``ev_pitch``), otherwise the frames whose calibration row is
    ``pitch_usable``. Every number drawn, and the facing arrow, carries an evidence locator
    ``"<table>.parquet#frame=<f>&track_id=<t>"`` into the run table it came from, so the
    render manifest can be traced back to its source rows. ``facing_gates`` is required
    with a ``lift`` table.

    For display only: a track missing for at most :data:`TRACK_HOLD_MAX_GAP` frames within
    a shot is filled in, with its skeleton (it measures nothing: no distance, no speed, no
    choice of an opponent); the minimap shows the ball only while its ground-plane position
    is plausible (:data:`MINIMAP_BALL_MARGIN_M`, :data:`BALL_GROUND_MAX_SPEED_MPS`).
    Invalid arguments raise :class:`RenderError` listing every problem.
    """

    def __init__(
        self,
        tables: SceneTables,
        highlight: Mapping[str, Any],
        fps: str | Fraction = "25/1",
        *,
        cuts: Sequence[int] = (),
        pitch_frames: Iterable[int] | None = None,
        team_names: Sequence[tuple[str, str]] = TEAM_NAMES_ZH,
        focus_track: int | None = None,
        trail_frames: int = 15,
        show_context: bool = True,
        frame_size: tuple[int, int] | None = None,
        facing_gates: FacingGates | None = None,
        peak_speed_withheld: Iterable[int] = (),
    ) -> None:
        problems: list[str] = []
        withheld = list(peak_speed_withheld)
        if any(_track_id(track) is None for track in withheld):
            problems.append(
                f"peak_speed_withheld must hold track ids (ints >= 0), got {withheld!r}"
            )
        if not isinstance(tables, SceneTables):
            problems.append(f"tables must be SceneTables, got {type(tables).__name__}")
        elif tables.lift is not None and facing_gates is None:
            problems.append(
                "facing_gates is required with tables.lift: a lifted facing is drawn only "
                "where it passes gbf.measure's display gates of body-orientation"
            )
        if facing_gates is not None and not isinstance(facing_gates, FacingGates):
            problems.append(
                f"facing_gates must be FacingGates or None, got {type(facing_gates).__name__}"
            )
        key_pair: tuple[int, int] | None = None
        key_focus: int | None = None
        peak = 0
        if not isinstance(highlight, Mapping):
            problems.append("highlight must be a mapping (one entry of highlights.json)")
        else:
            for key in ("id", "start_frame", "peak_frame", "end_frame"):
                value = highlight.get(key)
                if key not in highlight:
                    problems.append(f"highlight misses {key}")
                elif key != "id" and (
                    isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0
                ):
                    problems.append(f"highlight {key} must be a non-negative int, got {value!r}")
                elif key == "peak_frame":
                    peak = int(value)
            key_pair, key_focus = _key_roles(highlight, problems)
        if focus_track is not None and _track_id(focus_track) is None:
            problems.append(
                f"focus_track must be a track id (an int >= 0) or None, got {focus_track!r}"
            )
        try:
            self.fps = parse_rate(fps)
        except ValueError as exc:
            problems.append(str(exc))
            self.fps = Fraction(25)
        if isinstance(trail_frames, bool) or not isinstance(trail_frames, int) or trail_frames < 0:
            problems.append(f"trail_frames must be an int >= 0, got {trail_frames!r}")
        bad_cuts = [cut for cut in cuts if isinstance(cut, bool) or not isinstance(cut, int)]
        if bad_cuts:
            problems.append(f"cuts must be ints, got {bad_cuts!r}")
        _raise("HighlightScene", problems)
        self.tables = tables
        self.highlight = highlight
        self.cuts = sorted(int(cut) for cut in cuts)
        self.team_names = dict(team_names)
        self.trail_frames = trail_frames
        self.show_context = show_context
        self.frame_size = frame_size
        self.facing_gates = facing_gates
        self.track_ids = [int(t) for t in highlight.get("track_ids") or []]
        self.key_pair = key_pair
        self.key_focus_track = key_focus
        self._peak = peak
        self._key_shot = self._shot(peak)
        self._players = self._index_tracks(tables.tracks)
        self._track_team: dict[int, str] = {}
        for frame_players in self._players.values():
            for player in frame_players:
                if player.team and player.track_id not in self._track_team:
                    self._track_team[player.track_id] = player.team
        cues = highlight.get("cues")
        self._cues: Mapping[str, Any] = cues if isinstance(cues, Mapping) else {}
        self._joined = self._joined_event()
        self.duel_pair = self._duel_pair(highlight)
        self._focus_given = focus_track is not None
        self._shot_ids = self._index_shot_ids()
        # the key fields apply only where the key frame's shot has rows of their players
        in_shot = self._ids_in_shot(self._key_shot)
        self._key_pair = key_pair if key_pair is not None and set(key_pair) <= in_shot else None
        self._key_focus = key_focus if key_focus is not None and key_focus in in_shot else None
        # without key_focus_track, a key frame chosen at a ball event is about that event's
        # player: the freeze frame's spotlight and the data card are about them
        self._event_focus = self._ball_event_player() if key_focus is None else None
        self._withheld_peaks = frozenset(int(track) for track in withheld)
        parts = highlight.get("merged_parts")
        self._merged = isinstance(parts, Sequence) and len(parts) >= 2
        self._extended = (
            self._joined is not None
            or self._cue("cut_emphasis").get("extended_into_shot_id") is not None
        )
        self._frame_focus: dict[int, int | None] = {}
        self._eligible_by_shot: dict[int, frozenset[int]] | None = None
        # a duel is about the player its key frame is about, unless that frame is in the
        # joined shot after a cut, whose players are not in the duel's own shots
        key_in_own = self._joined is None or self._joined[0] != self._key_shot
        if focus_track is not None:
            self.focus = _track_id(focus_track)
        elif highlight.get("type") == "duel_tackle" and key_in_own and self._key_focus is not None:
            self.focus = self._key_focus
        else:
            self.focus = self._type_focus()
        self._shot_roles: dict[int, _ShotRoles] = {}
        self._calib = self._index_calib(tables.calib)
        self._ball = self._index_ball(tables.ball)
        self._speed = self._index_kinematics(tables.kinematics)
        self._pose, self._posed = self._index_pose(tables.pose)
        self._facing = self._index_facing(tables.lift)
        if pitch_frames is not None:
            self._pitch = {int(f) for f in pitch_frames}
        else:
            self._pitch = {f for f, row in self._calib.items() if bool(row.get("pitch_usable"))}
        self._cameras: dict[int, Camera | None] = {}
        unsafe = {
            (frame, player.track_id)
            for frame, players in self._players.items()
            for player in players
            if player.observed and not self._eligible(player, frame)
        }
        half = chord_half_frames(self.fps)
        self._speed = {
            (frame, track): speed
            for (frame, track), speed in self._speed.items()
            if speed_chord_safe(frame, track, unsafe, half)
        }
        self._peak_bounds: dict[int, tuple[int, float]] = {}
        for (frame, track), speed in sorted(self._speed.items()):
            if not int(highlight["start_frame"]) <= frame <= int(highlight["end_frame"]):
                continue
            if track not in self._withheld_peaks or frame not in self._pitch:
                continue
            if not any(
                p.track_id == track and p.observed and self._eligible(p, frame)
                for p in self._players.get(frame, ())
            ):
                continue
            if track not in self._peak_bounds or speed > self._peak_bounds[track][1]:
                self._peak_bounds[track] = (frame, speed)
        self._set_focus_window(
            plan_highlight_edit(
                int(highlight["start_frame"]),
                int(highlight["end_frame"]),
                peak,
                self.fps,
                cuts=self.cuts,
            )
        )

    def _set_focus_window(self, plan: EditPlan) -> None:
        """Keep a ball event's focus throughout the edit's key slow-motion window."""
        self._event_frames = {
            frame
            for segment in plan.segments
            if segment.kind in {"slow", "freeze"}
            for frame in segment.source_frames
        }
        self._frame_focus.clear()

    # ---------------------------------------------------------------- roles
    def _cue(self, name: str) -> Mapping[str, Any]:
        cue = self._cues.get(name)
        return cue if isinstance(cue, Mapping) else {}

    def _attacking_team(self) -> str | None:
        team = self._cue("box_pressure").get("team") or self._cue("ball_goal").get("team")
        return str(team) if isinstance(team, str) else None

    def _pick(self, candidates: Sequence[int]) -> int | None:
        """The attacking team's first candidate in attack highlights, else the first."""
        if not candidates:
            return None
        if self.highlight.get("type") == "attack_in_box":
            team = self._attacking_team()
            for track in candidates:
                if team is not None and self._track_team.get(track) == team:
                    return track
        return candidates[0]

    def _type_focus(self) -> int | None:
        """The focus by the highlight's type: the sprinter of a sprint (the speed cue's
        track), the first of the duel cue's pair of a duel, the attacking team's first
        involved player of an attack in the box; otherwise, and when the cue names nobody,
        the first of ``track_ids`` (gbf.highlights lists the striker first)."""
        kind = self.highlight.get("type")
        speed_track = _track_id(self._cue("speed").get("track_id"))
        if kind == "sprint" and speed_track is not None:
            return speed_track
        duel = self._cue("duel").get("track_ids")
        if kind == "duel_tackle" and isinstance(duel, Sequence) and duel:
            first = _track_id(duel[0])
            if first is not None:
                return first
        return self._pick(self.track_ids)

    def _joined_event(
        self,
    ) -> tuple[int, int | None, tuple[int, int] | None, tuple[int, ...], int] | None:
        """(shot, focus, pair, involved, frame) of the event in the shot that the highlight
        extends into after a cut (``cut_emphasis.joined_shot_event``), or None. The event's
        first player is the focus; a single player's ``partner_track_id`` (the nearest
        opponent gbf.moments found) makes the pair with it. ``involved`` is in the event's
        order."""
        event = self._cue("cut_emphasis").get("joined_shot_event")
        if not isinstance(event, Mapping) or not isinstance(event.get("frame"), int):
            return None
        ids = [int(t) for t in event.get("track_ids") or [] if isinstance(t, int) and t >= 0]
        if not ids:
            return None
        partner = _track_id(event.get("partner_track_id"))
        involved = list(dict.fromkeys(ids))
        if len(ids) == 1 and partner is not None and partner != ids[0]:
            pair: tuple[int, int] | None = (ids[0], partner)
            involved.append(partner)
        else:
            pair = (ids[0], ids[1]) if len(ids) == 2 else None
        frame = int(event["frame"])
        return (self._shot(frame), self._pick(ids), pair, tuple(involved), frame)

    def _ball_event_player(self) -> int | None:
        """The player of the key frame's ball moment (gbf.moments' ``key_moments`` of a ball
        kind at ``peak_frame``: a touch, the ball leaving their feet), or None."""
        moments = self.highlight.get("key_moments")
        for moment in moments if isinstance(moments, Sequence) else ():
            if (
                not isinstance(moment, Mapping)
                or moment.get("frame") != self._peak
                or moment.get("kind") not in BALL_MOMENT_KINDS
            ):
                continue
            ids = moment.get("track_ids")
            for value in ids if isinstance(ids, Sequence) else ():
                track = _track_id(value)
                if track is not None:
                    return track
        return None

    def _index_shot_ids(self) -> dict[int, frozenset[int]]:
        """Track ids with a player or goalkeeper row, by shot (boxes without a track,
        id -1, are no one's)."""
        ids: dict[int, set[int]] = {}
        for frame, players in self._players.items():
            ids.setdefault(self._shot(frame), set()).update(
                player.track_id
                for player in players
                if (player.role or "player") in _PLAYER_ROLES and player.track_id >= 0
            )
        return {shot: frozenset(tracks) for shot, tracks in ids.items()}

    def _ids_in_shot(self, shot: int) -> frozenset[int]:
        """Track ids with a player or goalkeeper row in ``shot``."""
        return self._shot_ids.get(shot, frozenset())

    def _candidates(self, shot: int) -> tuple[tuple[int, ...], tuple[int, int] | None, set[int]]:
        """(focus candidates in order, duel pair, the involved tracks whatever the focus) of
        ``shot``, with the key fields that apply in the key frame's shot (see the class
        docstring)."""
        joined = self._joined if self._joined is not None and self._joined[0] == shot else None
        if joined is not None:
            own: tuple[int | None, ...] = (joined[1],)
            pair, listed = joined[2], joined[3]
        else:
            # self.focus is a duel's key focus where it applies, else the type's focus
            own = (self.focus, self._type_focus())
            pair, listed = self.duel_pair, tuple(self.track_ids)
        key_focus: int | None = None
        key_pair: tuple[int, int] | None = None
        if shot == self._key_shot:
            key_focus, key_pair = self._key_focus, self._key_pair
        involved = set(listed)
        if key_pair is not None:
            pair = key_pair
            involved.update(key_pair)
        if key_focus is not None:
            involved.add(key_focus)
        if self._focus_given and joined is None:
            ranked: tuple[int | None, ...] = own[:1]  # focus_track, never replaced
        elif key_focus is not None:
            ranked = (key_focus, *own, *(key_pair or ()), *listed)
        else:
            ranked = (*own, *listed, *(key_pair or ()))
        return tuple(dict.fromkeys(t for t in ranked if t is not None)), pair, involved

    def _roles_of_shot(self, shot: int) -> _ShotRoles:
        """The roles on the frames of ``shot``: the first focus candidate with track rows in
        the shot, else the first candidate (then nobody is marked as the focus); in the key
        frame's shot also the key frame's own focus, the player of the key frame's ball
        event (without ``key_focus_track``) or else the first candidate that frame shows
        (with a row, or filled in), else the shot's. A merged highlight has no focus in a
        shot after the key frame's that shows none of the key frame's players."""
        ranked, pair, involved = self._candidates(shot)
        here = self._eligible_ids(shot)
        focus = next((t for t in ranked if t in here), ranked[0] if ranked else None)
        key_frame_focus = focus
        if shot == self._key_shot:
            peak = self._peak
            shown = {p.track_id for p in self._players_in(peak) if self._eligible(p, peak)}
            event = self._event_focus if not self._focus_given else None
            first = (event,) if event is not None else ()
            key_frame_focus = next((t for t in (*first, *ranked) if t in shown), focus)
        elif self._merged and shot > self._key_shot and not here & self._key_players():
            # a later part of a merged highlight, about other players than its key frame:
            # its players stay involved, but nobody is marked as the highlight's focus
            return _ShotRoles(None, None, pair, frozenset(involved))
        involved.update(t for t in (focus, key_frame_focus) if t is not None)
        return _ShotRoles(focus, key_frame_focus, pair, frozenset(involved))

    def _eligible_ids(self, shot: int) -> frozenset[int]:
        """Track ids with a player or goalkeeper row in ``shot`` on which they may take a role
        (:meth:`_eligible`): a person off the pitch on every pitch frame of the shot is
        nobody's focus."""
        if self._eligible_by_shot is None:
            ids: dict[int, set[int]] = {}
            for frame, players in self._players.items():
                ids.setdefault(self._shot(frame), set()).update(
                    player.track_id
                    for player in players
                    if (player.role or "player") in _PLAYER_ROLES and self._eligible(player, frame)
                )
            self._eligible_by_shot = {key: frozenset(tracks) for key, tracks in ids.items()}
        return self._eligible_by_shot.get(shot, frozenset())

    def _key_players(self) -> set[int]:
        """The players the key frame is about: ``key_pair``, ``key_focus_track``, the key
        frame's ball event's player and the key frame's focus."""
        players = {*(self.key_pair or ()), self.key_focus_track, self._event_focus}
        players.add(self._roles(self._peak)[0])
        return {track for track in players if track is not None}

    def _roles(self, frame: int) -> tuple[int | None, tuple[int, int] | None, set[int]]:
        """(focus, duel pair, involved tracks) on ``frame``: those of its shot (the key
        frame's shot, the joined event's shot after a cut, or one of the highlight's own
        shots), with the key frame's own focus on the key frame (see the class docstring).
        Before the key frame's shot in a merged or extended highlight, a focus player farther
        than :data:`FOCUS_BALL_MAX_M` from the ball is not marked on that frame."""
        shot = self._shot(frame)
        roles = self._shot_roles.get(shot)
        if roles is None:
            roles = self._shot_roles[shot] = self._roles_of_shot(shot)
        if frame not in self._frame_focus:
            event_window = self._event_focus is not None and frame in self._event_frames
            focus = roles.key_frame_focus if frame == self._peak or event_window else roles.focus
            subject = next((p for p in self._players_in(frame) if p.track_id == focus), None)
            if subject is not None and not self._eligible(subject, frame):
                focus = None
            if focus is not None and self._away_from_the_ball(frame, focus):
                focus = None
            self._frame_focus[frame] = focus
        return self._frame_focus[frame], roles.pair, set(roles.involved)

    def _away_from_the_ball(self, frame: int, track: int) -> bool:
        """True on a pitch frame before the key frame's shot of a merged or extended
        highlight where ``track`` is farther than :data:`FOCUS_BALL_MAX_M` from the ball's
        ground position (the minimap's ball); False wherever either is unknown."""
        if not (self._merged or self._extended) or self._shot(frame) >= self._key_shot:
            return False
        if not self.is_pitch_frame(frame):
            return False
        ball = self._minimap_ball(frame)
        player = next((p for p in self._players.get(frame, ()) if p.track_id == track), None)
        if ball is None or player is None or player.pitch is None:
            return False
        return math.hypot(player.pitch[0] - ball[0], player.pitch[1] - ball[1]) > FOCUS_BALL_MAX_M

    def roles_at(self, frame: int) -> tuple[int | None, tuple[int, int] | None, set[int]]:
        """(focus, duel pair, involved tracks) on canonical ``frame``, as the overlay marks
        them. ``roles_at(peak_frame)[0]`` is the player the key frame is about: the freeze
        frame's spotlight is on that player, and the data card's values should be its own.
        It is a player the key frame shows (with a row, or filled in) whenever one of the
        focus candidates is (see the class docstring).
        """
        return self._roles(frame)

    # ---------------------------------------------------------------- indexing
    @staticmethod
    def _duel_pair(highlight: Mapping[str, Any]) -> tuple[int, int] | None:
        cue = (highlight.get("cues") or {}).get("duel")
        ids = cue.get("track_ids") if isinstance(cue, Mapping) else None
        if isinstance(ids, Sequence) and len(ids) == 2:
            return (int(ids[0]), int(ids[1]))
        return None

    def _index_tracks(self, tracks: pd.DataFrame) -> dict[int, list[_Player]]:
        """Players by frame, with each gap of at most :data:`TRACK_HOLD_MAX_GAP` frames
        between two rows of a track in the same shot filled in for display: box and pitch
        position interpolated, role, team and jersey held, ``observed`` False. Boxes
        without a track (id -1) are not one person's: they are never filled in."""
        columns = tracks.columns
        index: dict[int, list[_Player]] = {}
        for row in tracks.itertuples(index=False):
            record = row._asdict()
            pitch = None
            if "pitch_x" in columns and "pitch_y" in columns:
                x, y = _number(record["pitch_x"]), _number(record["pitch_y"])
                if x is not None and y is not None:
                    pitch = (x, y)
            box = tuple(_number(record[k]) for k in ("x1", "y1", "x2", "y2"))
            if any(v is None for v in box):
                continue
            index.setdefault(int(record["frame"]), []).append(
                _Player(
                    track_id=int(record["track_id"]),
                    box=box,  # type: ignore[arg-type]
                    role=_clean_label(record["role"]),
                    team=_clean_label(record["team"]),
                    jersey=_clean_label(record.get("jersey")),
                    pitch=pitch,
                )
            )
        by_track: dict[int, list[tuple[int, _Player]]] = {}
        for frame, players in index.items():
            for player in players:
                if player.track_id >= 0:
                    by_track.setdefault(player.track_id, []).append((frame, player))
        for rows in by_track.values():
            rows.sort(key=lambda item: item[0])
            for (f0, a), (f1, b) in zip(rows, rows[1:], strict=False):
                if not 1 < f1 - f0 <= TRACK_HOLD_MAX_GAP + 1 or self._shot(f0) != self._shot(f1):
                    continue
                for frame in range(f0 + 1, f1):
                    weight = (frame - f0) / (f1 - f0)
                    pitch = None
                    if a.pitch is not None and b.pitch is not None:
                        pitch = _lerp(a.pitch, b.pitch, weight)
                    index.setdefault(frame, []).append(
                        _Player(
                            track_id=a.track_id,
                            box=_lerp(a.box, b.box, weight),  # type: ignore[arg-type]
                            role=a.role,
                            team=a.team,
                            jersey=a.jersey,
                            pitch=pitch,  # type: ignore[arg-type]
                            observed=False,
                        )
                    )
        return index

    @staticmethod
    def _index_calib(calib: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
        if calib is None:
            return {}
        key = "frame_index" if "frame_index" in calib.columns else "frame"
        return {int(row[key]): row for row in calib.to_dict("records")}

    @staticmethod
    def _index_ball(ball: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
        if ball is None:
            return {}
        return {int(row["frame"]): row for row in ball.to_dict("records")}

    @staticmethod
    def _index_kinematics(kinematics: pd.DataFrame | None) -> dict[tuple[int, int], float]:
        if kinematics is None:
            return {}
        valid = kinematics[kinematics["valid"].astype(bool)]
        speeds: dict[tuple[int, int], float] = {}
        for frame, track, speed in zip(
            valid["frame"].to_numpy(),
            valid["track_id"].to_numpy(),
            valid["speed"].to_numpy(),
            strict=True,
        ):
            value = _number(speed)
            if value is not None:
                speeds[(int(frame), int(track))] = value
        return speeds

    def _index_pose(
        self, pose: pd.DataFrame | None
    ) -> tuple[dict[tuple[int, int], np.ndarray], frozenset[tuple[int, int]]]:
        """Keypoints by (frame, track), with each gap of at most :data:`TRACK_HOLD_MAX_GAP`
        frames between two poses of a track in the same shot filled in for display: (u, v)
        interpolated, the score the lower of the two; and the (frame, track) pairs of the
        table's own rows (the poses the lift was computed from)."""
        if pose is None:
            return {}, frozenset()
        keypoints: dict[tuple[int, int], np.ndarray] = {}
        for frame, track, points in zip(
            pose["frame"].to_numpy(), pose["track_id"].to_numpy(), pose["keypoints"], strict=True
        ):
            array = np.asarray(points, dtype=np.float64).reshape(-1)
            if array.size == 26 * 3:
                keypoints[(int(frame), int(track))] = array.reshape(26, 3)
        observed = frozenset(keypoints)
        by_track: dict[int, list[int]] = {}
        for frame, track in keypoints:
            if track >= 0:  # a box without a track is no one's: no pose is filled in
                by_track.setdefault(track, []).append(frame)
        for track, frames in by_track.items():
            frames.sort()
            for f0, f1 in zip(frames, frames[1:], strict=False):
                if not 1 < f1 - f0 <= TRACK_HOLD_MAX_GAP + 1 or self._shot(f0) != self._shot(f1):
                    continue
                a, b = keypoints[(f0, track)], keypoints[(f1, track)]
                for frame in range(f0 + 1, f1):
                    held = a + (b - a) * ((frame - f0) / (f1 - f0))
                    held[:, 2] = np.minimum(a[:, 2], b[:, 2])
                    keypoints[(frame, track)] = held
        return keypoints, observed

    @staticmethod
    def _index_facing(lift: pd.DataFrame | None) -> dict[tuple[int, int], _Facing]:
        """The facing columns of ``lift`` by (frame, track)."""
        if lift is None:
            return {}

        def value(cell: Any) -> float:
            number = _number(cell)
            return number if number is not None else math.nan

        facings: dict[tuple[int, int], _Facing] = {}
        for frame, track, method, facing, conf, sigma in zip(
            lift["frame"].to_numpy(),
            lift["track_id"].to_numpy(),
            lift["method"].to_numpy(dtype=object),
            lift["facing_deg"].to_numpy(dtype=object),
            lift["facing_conf"].to_numpy(dtype=object),
            lift["facing_sigma_deg"].to_numpy(dtype=object),
            strict=True,
        ):
            facings[(int(frame), int(track))] = _Facing(
                lifted=str(method) == LIFTED,
                facing_deg=value(facing),
                conf=value(conf),
                sigma_deg=value(sigma),
            )
        return facings

    # ---------------------------------------------------------------- per frame
    def is_pitch_frame(self, frame: int) -> bool:
        """True when positions in metres may be shown on ``frame``."""
        return frame in self._pitch

    def camera(self, frame: int) -> Camera | None:
        """The frame's own camera from ``calib`` (None without a usable row)."""
        if frame not in self._cameras:
            row = self._calib.get(frame)
            camera = None
            if row is not None:
                try:
                    camera = Camera(
                        K=np.asarray(row["K"], dtype=np.float64).reshape(3, 3),
                        R=np.asarray(row["R"], dtype=np.float64).reshape(3, 3),
                        C=np.asarray(row["C"], dtype=np.float64).reshape(3),
                        width=int(row["width"]),
                        height=int(row["height"]),
                    )
                except (GeometryInputError, TypeError, ValueError):
                    camera = None
                if camera is not None and self.frame_size is not None:
                    if camera.size != tuple(self.frame_size):
                        camera = camera.resized(*self.frame_size)
            self._cameras[frame] = camera
        return self._cameras[frame]

    def _shot(self, frame: int) -> int:
        return bisect.bisect_right(self.cuts, frame)

    def _team_name(self, team: str | None) -> str:
        return self.team_names.get(team or "", "未知球队")

    def _emphasis(self, track: int, frame: int) -> Emphasis:
        focus, _pair, involved = self._roles(frame)
        if track == focus:
            return "focus"
        return "normal" if track in involved else "context"

    def _label(
        self, frame: int, player: _Player, pitch: bool
    ) -> tuple[str, str | None, tuple[str, ...]]:
        """(label, held, evidence) of a player: team and name, and on pitch frames the
        speed, or, over at most :data:`TRACK_HOLD_MAX_GAP` frames without a valid one, the
        last valid speed held (``held``, drawn greyed). A player whose peak speed gbf.measure
        withholds gets an explicit observed-peak lower bound, with its measured frame."""
        name = f"{player.jersey}号" if player.jersey else f"轨迹 {player.track_id}"
        _focus, pair, _involved = self._roles(frame)
        if self._emphasis(player.track_id, frame) == "focus" or (
            pair is not None and player.track_id in pair
        ):
            name = f"{self._team_name(player.team)} · {name}"
        # a filled-in box (observed False) measures nothing: no speed on it
        if not (pitch and player.observed):
            return name, None, ()
        speed = self._speed.get((frame, player.track_id))
        peak = self._peak_bounds.get(player.track_id)
        if peak is not None and (speed is not None or self._held_speed(frame, player.track_id)):
            measured, value = peak
            evidence = (f"kinematics.parquet#frame={measured}&track_id={player.track_id}",)
            label = f" · 峰值≥{value * 3.6:.1f} km/h"
            return (name + label, None, evidence) if speed is not None else (name, label, evidence)
        if speed is not None:
            evidence = (f"kinematics.parquet#frame={frame}&track_id={player.track_id}",)
            return f"{name} · {speed * 3.6:.1f} km/h", None, evidence
        held = self._held_speed(frame, player.track_id)
        if held is None:
            return name, None, ()
        measured, speed = held
        evidence = (f"kinematics.parquet#frame={measured}&track_id={player.track_id}",)
        return name, f" · {speed * 3.6:.1f} km/h", evidence

    def _held_speed(self, frame: int, track: int) -> tuple[int, float] | None:
        """(frame, speed) of the latest valid speed of ``track`` at most
        :data:`TRACK_HOLD_MAX_GAP` frames before ``frame`` in its shot, measured on a frame
        where the track has a row of its own; None when there is none."""
        shot = self._shot(frame)
        for earlier in range(frame - 1, max(-1, frame - TRACK_HOLD_MAX_GAP - 1), -1):
            if self._shot(earlier) != shot:
                break
            speed = self._speed.get((earlier, track))
            if speed is None or not self.is_pitch_frame(earlier):
                continue
            if any(p.track_id == track and p.observed for p in self._players.get(earlier, ())):
                return earlier, speed
        return None

    def _ring(
        self, frame: int, player: _Player, pitch: bool, emphasis: Emphasis
    ) -> tuple[Point, ...] | None:
        camera = self.camera(frame) if pitch else None
        if camera is None or player.pitch is None:
            return None
        radius = 0.9 if emphasis == "focus" else 0.7
        ring = ground_circle_px(camera, player.pitch, radius, n=48)
        if not np.isfinite(ring).all():
            return None
        return tuple((float(u), float(v)) for u, v in ring)

    def _facing_arrow(self, frame: int, players: Sequence[_Player], pitch: bool) -> Arrow | None:
        """The focus player's lifted facing on ``frame`` drawn back on the frame (see the
        class docstring), or None. ``facing_deg`` is pitch_zup degrees from +x,
        counter-clockwise (gbf.lift3d: +90 faces the far touchline); its direction is turned
        into SoccerNet metres (far touchline at y = -34) before the arrow is projected. The
        frame's facing must pass gbf.measure's per-frame gates of body-orientation
        (:func:`gbf.measure.facing_frame_ok`), the front-back reversal check included: the
        trunk angle of the frame's own pose against the facing placed in the picture."""
        gates = self.facing_gates
        focus, _pair, _involved = self._roles(frame)
        if not pitch or gates is None or focus is None:
            return None
        player = next((p for p in players if p.track_id == focus), None)
        facing = self._facing.get((frame, focus))
        if (
            player is None
            or not player.observed
            or not self._eligible(player, frame)
            or player.pitch is None
            or (frame, focus) not in self._posed
            or facing is None
        ):
            return None
        camera = self.camera(frame)
        if camera is None or not math.isfinite(facing.facing_deg):
            return None

        def to_image(points: np.ndarray) -> np.ndarray | None:
            grass = np.column_stack([points, np.zeros(len(points))])
            uv, _depth, valid = project(camera, grass)
            return uv if valid.all() else None

        passes = facing_frame_ok(
            gates,
            lifted=facing.lifted,
            facing_deg=facing.facing_deg,
            facing_conf=facing.conf,
            facing_sigma_deg=facing.sigma_deg,
            box_height_px=player.box[3] - player.box[1],
            image_trunk_deg=image_trunk_deg(self._pose[(frame, focus)]),
            image_dx=facing_image_dx(to_image, player.pitch, facing.facing_deg),
        )
        if not passes:
            return None
        heading = math.radians(facing.facing_deg)
        along = pitch_zup_to_soccernet(np.array([math.cos(heading), math.sin(heading), 0.0]))
        x, y = player.pitch
        reach = FACING_ARROW_LENGTH_M
        ends = np.array([[x, y, 0.0], [x + reach * along[0], y + reach * along[1], 0.0]])
        uv, _depth, valid = project(camera, ends)
        if not valid.all():
            return None
        start, end = (float(uv[0, 0]), float(uv[0, 1])), (float(uv[1, 0]), float(uv[1, 1]))
        if start == end:  # pragma: no cover - a ground arrow is never along a camera ray
            return None
        return Arrow(
            start,
            end,
            label=FACING_LABEL_ZH,
            dashed=True,
            badges=(Badge("未验证", "unverified"),),
            evidence_ids=(f"lift3d.parquet#frame={frame}&track_id={focus}",),
        )

    def _players_in(self, frame: int) -> list[_Player]:
        return [p for p in self._players.get(frame, []) if (p.role or "player") in _PLAYER_ROLES]

    def _eligible(self, player: _Player, frame: int) -> bool:
        """True when ``player`` may take a role, a team colour, a label, a minimap dot or a
        line on ``frame``: a box with a track (id >= 0, as gbf.measure counts players) that
        is not off the pitch (:meth:`_off_pitch`: no coach, steward or substitute). Every
        other person is drawn as a faint ring only."""
        if player.track_id < 0:
            return False
        verdicts = self.tables.team_verdicts
        unchecked = (
            verdicts is not None
            and verdicts.get((player.track_id, player.team or ""), "unchecked") != "consistent"
        )
        size = self.frame_size
        if size is None and (camera := self.camera(frame)) is not None:
            size = camera.size
        if unchecked and (player.pitch is None or box_touches_edge(player.box, size)):
            return False
        return not self._off_pitch(player, frame)

    def _off_pitch(self, player: _Player, frame: int) -> bool:
        """True on a pitch frame for a person off the pitch: a foot point beyond a line by
        more than gbf.track_tools' OFF_PITCH_MARGIN_M (:func:`gbf.track_tools.off_pitch`),
        or a box cut off by the bottom of the frame (within :data:`EDGE_TOUCH_PER_BOX_HEIGHT`
        of its height) whose ground point lies beyond a line at all, a line the frame's
        camera stands beyond too. That ground point is the frame edge's under the person; the
        feet, below the frame, are nearer the camera, so farther beyond the line: a coach in
        the technical area under the camera, whose box the frame cuts off."""
        if not self.is_pitch_frame(frame) or player.pitch is None:
            return False
        camera = self.camera(frame)
        return off_pitch_at_camera_edge(
            player.box,
            player.pitch,
            self.frame_size or (camera.size if camera is not None else None),
            camera.C if camera is not None else None,
        )

    def _link(self, frame: int, players: Sequence[_Player], pitch: bool) -> LinkLine | None:
        """The frame's line (see :data:`LINK_LABELS_ZH`): the duel pair's while the duel is
        on, else, on pitch frames, the focus player's to its nearest opponent."""
        shown = {p.track_id: p for p in players if self._eligible(p, frame)}
        focus_id, duel_pair, _involved = self._roles(frame)
        if (
            self.highlight.get("type") == "duel_tackle"
            and duel_pair is not None
            and all(t in shown for t in duel_pair)
            and self._in_duel_time(frame, duel_pair)
            and self._duel_on(frame, shown[duel_pair[0]], shown[duel_pair[1]], pitch)
        ):
            a, b = shown[duel_pair[0]], shown[duel_pair[1]]
            return self._pair_line(frame, a, b, LINK_LABELS_ZH["duel"], pitch, focus_id)
        if not pitch or focus_id not in shown or shown[focus_id].pitch is None:
            return None
        focus = shown[focus_id]
        what = LINK_LABELS_ZH["nearest"]
        if focus.observed:
            nearest = self._nearest_opponent(frame, focus)
            return None if nearest is None else self._pair_line(frame, focus, nearest, what, pitch)
        # The focus player is filled in: its line goes on to the opponent that was nearest
        # on the last frame it was observed, with the distance measured there.
        held = self._held_nearest(frame, focus_id)
        if held is None or held[2].track_id not in shown:
            return None
        partner = shown[held[2].track_id]
        if self._nearer_uncertain(frame, focus, partner):
            return None
        return self._pair_line(frame, focus, partner, what, pitch, focus_id, held=held)

    def _nearer_uncertain(self, frame: int, focus: _Player, partner: _Player) -> bool:
        """Drawn uncertain positions may veto a nearest claim, never select or measure."""
        if focus.pitch is None or partner.pitch is None:
            return True
        distance = math.dist(focus.pitch, partner.pitch)
        return any(
            p.pitch is not None
            and p.track_id != focus.track_id
            and (p.role or "player") == "player"
            and self._eligible(p, frame)
            and not same_spot(focus.box, p.box, focus.pitch, p.pitch)
            and (not p.team or (not p.observed and focus.team and p.team != focus.team))
            and math.dist(focus.pitch, p.pitch) < distance
            for p in self._players_in(frame)
        )

    def _nearest_opponent(self, frame: int, focus: _Player) -> _Player | None:
        """The nearest opponent of the observed ``focus`` on ``frame`` among the players
        observed there (filled-in positions measure nothing), as gbf.measure's
        nearest-opponent-distance (the data card's 最近对手距离) counts them: outfield
        players of the other team on the pitch, goalkeepers left out, and never a box on the
        focus player's own spot (gbf.track_tools.same_spot: the focus detected twice, or a
        box around it and the player behind). None when there is none, or when an outfield
        player without a team label (gbf.team_check clears labels that contradict the shirt
        colours), or a drawn filled-in opponent, is nearer: then the nearest opponent is
        not known. A filled position only withholds; it never selects or measures."""
        candidates = [
            p
            for p in self._players_in(frame)
            if p.pitch is not None
            and p.track_id != focus.track_id
            and (p.role or "player") == "player"
            and self._eligible(p, frame)
            and not same_spot(focus.box, p.box, focus.pitch, p.pitch)
        ]
        opponents = [
            p for p in candidates if p.observed and p.team and focus.team and p.team != focus.team
        ]

        def gap(p: _Player) -> float:
            return math.hypot(
                p.pitch[0] - focus.pitch[0],  # type: ignore[index]
                p.pitch[1] - focus.pitch[1],  # type: ignore[index]
            )

        if not opponents:
            return None
        nearest = min(opponents, key=gap)
        if self._nearer_uncertain(frame, focus, nearest):
            return None
        return nearest

    def _held_nearest(self, frame: int, track: int) -> tuple[int, _Player, _Player] | None:
        """(frame, focus, opponent) of the last pitch frame, at most
        :data:`TRACK_HOLD_MAX_GAP` before ``frame`` in its shot, on which ``track`` was
        observed, with its nearest opponent there; None when there is none or that frame's
        nearest opponent is not known."""
        shot = self._shot(frame)
        for earlier in range(frame - 1, max(-1, frame - TRACK_HOLD_MAX_GAP - 1), -1):
            if self._shot(earlier) != shot:
                break
            if not self.is_pitch_frame(earlier):
                continue
            focus = next((p for p in self._players.get(earlier, ()) if p.track_id == track), None)
            if focus is None or not focus.observed:
                continue
            if focus.pitch is None or not self._eligible(focus, earlier):
                return None
            nearest = self._nearest_opponent(earlier, focus)
            return None if nearest is None else (earlier, focus, nearest)
        return None

    def _pair_line(
        self,
        frame: int,
        a: _Player,
        b: _Player,
        what: str,
        pitch: bool,
        focus_id: int | None = None,
        *,
        held: tuple[int, _Player, _Player] | None = None,
    ) -> LinkLine:
        """The line between ``a`` and ``b`` (their foot points on the frame), labelled
        ``what`` and, on pitch frames, the ground distance: measured on the frame when both
        are observed there; else ``held`` (or, for a duel pair, the last frame on which both
        were observed, at most :data:`TRACK_HOLD_MAX_GAP` before) when the only end drawn at
        a filled-in position is the focus player's (``focus_id``). A partner drawn at a
        filled-in position gets no distance, and neither does a distance below
        gbf.track_tools.MIN_PLAYER_GAP_M (two boxes on one spot are no two players)."""
        foot_a = ((a.box[0] + a.box[2]) / 2, a.box[3])
        foot_b = ((b.box[0] + b.box[2]) / 2, b.box[3])
        plain = LinkLine(foot_a, foot_b, what)
        if not pitch or a.pitch is None or b.pitch is None:
            return plain
        measured, first, second = frame, a, b
        if not (a.observed and b.observed):
            if any(not p.observed and p.track_id != focus_id for p in (a, b)):
                return plain  # a partner at a filled-in position: nothing measured there
            last = held or self._last_observed_pair(frame, a.track_id, b.track_id)
            if last is None:
                return plain
            measured, first, second = last
        metres = math.hypot(
            first.pitch[0] - second.pitch[0],  # type: ignore[index]
            first.pitch[1] - second.pitch[1],  # type: ignore[index]
        )
        if metres < MIN_PLAYER_GAP_M:
            return plain
        ids = f"{first.track_id},{second.track_id}"
        held_label = "（沿用）" if measured != frame else ""
        return LinkLine(
            foot_a,
            foot_b,
            f"{what} · {metres:.1f} m{held_label}",
            badges=(Badge("未验证", "unverified"),),
            evidence_ids=(f"refine/tracks.parquet#frame={measured}&track_id={ids}",),
        )

    def _last_observed_pair(
        self, frame: int, a: int, b: int
    ) -> tuple[int, _Player, _Player] | None:
        """The latest pitch frame at most :data:`TRACK_HOLD_MAX_GAP` before ``frame`` in its
        shot where tracks ``a`` and ``b`` both have a row with a pitch position, and those
        rows; None when there is none."""
        shot = self._shot(frame)
        for earlier in range(frame - 1, max(-1, frame - TRACK_HOLD_MAX_GAP - 1), -1):
            if self._shot(earlier) != shot:
                break
            if not self.is_pitch_frame(earlier):
                continue
            seen = {
                p.track_id: p
                for p in self._players.get(earlier, ())
                if p.observed and p.pitch is not None
            }
            if a in seen and b in seen:
                return earlier, seen[a], seen[b]
        return None

    def _in_duel_time(self, frame: int, pair: tuple[int, int]) -> bool:
        """True when ``frame`` is within :data:`DUEL_LINK_WINDOW_S` of a moment of ``pair``
        (pairs compared as sets): the duel cue's frame for the cue's pair, in the joined
        event's shot the event's frame for its pair, and in the key frame's shot the key
        frame for ``key_pair``, whichever is nearer. A pair without such a moment (a duel
        cue without a frame) is not limited in time."""
        moments: list[int] = []
        cue_frame = self._cue("duel").get("frame")
        if (
            self.duel_pair is not None
            and set(pair) == set(self.duel_pair)
            and isinstance(cue_frame, int | np.integer)
            and not isinstance(cue_frame, bool)
        ):
            moments.append(int(cue_frame))
        shot = self._shot(frame)
        joined = self._joined
        if (
            joined is not None
            and joined[2] is not None
            and set(pair) == set(joined[2])
            and shot == joined[0]
        ):
            moments.append(joined[4])
        if (
            self._key_pair is not None
            and set(pair) == set(self._key_pair)
            and shot == self._key_shot
        ):
            moments.append(self._peak)
        if not moments:
            return True
        window = Fraction(DUEL_LINK_WINDOW_S).limit_denominator(10**6) * self.fps
        return min(abs(frame - moment) for moment in moments) <= round_half_away(window)

    def _duel_on(self, frame: int, a: _Player, b: _Player, pitch: bool) -> bool:
        """True while the duel cue's rule holds for ``a`` and ``b`` on ``frame`` (see
        :data:`DUEL_LINK_WINDOW_S`): on a pitch frame with the ball's ground position, both
        within DUEL_BALL_RADIUS_M of it and the pair closer than DUEL_FAR_M; else, with the
        ball in the picture, both foot points within DUEL_BALL_RADIUS_H own box heights of
        it and the pair closer than DUEL_FAR_H mean box heights. Without a ball on the frame
        it is no duel; without a ball track at all, the time window alone decides."""
        if self.tables.ball is None:
            return True
        ground = self._ground_ball(frame) if pitch else None
        at_a, at_b = a.pitch, b.pitch
        if ground is not None and at_a is not None and at_b is not None:
            near = all(
                math.hypot(x - ground[0], y - ground[1]) <= DUEL_BALL_RADIUS_M
                for x, y in (at_a, at_b)
            )
            return near and math.hypot(at_a[0] - at_b[0], at_a[1] - at_b[1]) < DUEL_FAR_M
        row = self._ball.get(frame)
        if row is None or str(row.get("status")) not in ("detected", "interpolated"):
            return False
        u, v = _number(row.get("u")), _number(row.get("v"))
        heights = [p.box[3] - p.box[1] for p in (a, b)]
        if u is None or v is None or min(heights) <= 0:
            return False
        feet = [((p.box[0] + p.box[2]) / 2, p.box[3]) for p in (a, b)]
        near = all(
            math.hypot(x - u, y - v) <= DUEL_BALL_RADIUS_H * height
            for (x, y), height in zip(feet, heights, strict=True)
        )
        apart = math.hypot(feet[0][0] - feet[1][0], feet[0][1] - feet[1][1])
        return near and apart < DUEL_FAR_H * (heights[0] + heights[1]) / 2

    def _ball_mark(self, frame: int) -> BallMark | None:
        row = self._ball.get(frame)
        position = None
        interpolated = False
        radius = 9.0
        if row is not None and str(row.get("status")) in ("detected", "interpolated"):
            u, v = _number(row.get("u")), _number(row.get("v"))
            if u is not None and v is not None:
                position = (u, v)
                interpolated = str(row.get("status")) == "interpolated"
                radius = _number(row.get("radius_px")) or radius
        shot = self._shot(frame)
        trail: list[Point] = []
        for earlier in range(max(0, frame - self.trail_frames), frame):
            if self._shot(earlier) != shot:
                continue
            past = self._ball.get(earlier)
            if past is None or str(past.get("status")) not in ("detected", "interpolated"):
                trail.append((math.nan, math.nan))
                continue
            u, v = _number(past.get("u")), _number(past.get("v"))
            trail.append((u, v) if u is not None and v is not None else (math.nan, math.nan))
        if position is None and not any(math.isfinite(u) for u, _ in trail):
            return None
        return BallMark(position, max(radius, 1.0), tuple(trail), interpolated)

    def _minimap(self, frame: int, players: Sequence[_Player], pitch: bool) -> Minimap:
        if not pitch:
            # pitch frames are decided frame by frame, so the note speaks of this frame only
            return Minimap(note="本帧没有可用标定，不显示球场位置", title="俯视图")
        dots = [
            MinimapDot(p.pitch[0], p.pitch[1], p.team, self._emphasis(p.track_id, frame))
            for p in players
            if p.pitch is not None and self._eligible(p, frame)
        ]
        camera = self.camera(frame)
        view = tuple(camera_footprint(camera)) if camera is not None else ()
        return Minimap(
            players=tuple(dots),
            ball=self._minimap_ball(frame),
            view=view,
            title="俯视图（自洽标定，未验证）",
        )

    def _ground_ball(self, frame: int) -> tuple[float, float] | None:
        """The ball's ground-plane position (pitch_x, pitch_y) on ``frame``, where the ball
        table has one for a detected or interpolated ball with ``ground_assumption`` set."""
        row = self._ball.get(frame)
        if row is None or str(row.get("status")) not in ("detected", "interpolated"):
            return None
        held = row.get("ground_assumption", True)
        if not isinstance(held, bool | np.bool_) or not held:
            return None
        x, y = _number(row.get("pitch_x")), _number(row.get("pitch_y"))
        return (x, y) if x is not None and y is not None else None

    def _minimap_ball(self, frame: int) -> tuple[float, float] | None:
        """The ball for the minimap: its ground-plane position, unless that is more than
        :data:`MINIMAP_BALL_MARGIN_M` off the pitch or the ground assumption does not hold
        (:data:`BALL_GROUND_MAX_SPEED_MPS`)."""
        here = self._ground_ball(frame)
        if here is None or not _near_pitch(*here, MINIMAP_BALL_MARGIN_M):
            return None
        shot = self._shot(frame)
        near = [
            (other, point)
            for other in range(frame - BALL_SPEED_SPAN_FRAMES, frame + BALL_SPEED_SPAN_FRAMES + 1)
            if self._shot(other) == shot and (point := self._ground_ball(other)) is not None
        ]
        (first, a), (last, b) = near[0], near[-1]
        if last - first >= 2:
            speed = math.hypot(b[0] - a[0], b[1] - a[1]) / (last - first) * float(self.fps)
            if speed > BALL_GROUND_MAX_SPEED_MPS:
                return None
        return here

    def _banner(self, frame: int, pitch: bool) -> Banner:
        title = str(self.highlight.get("title_zh") or self.highlight.get("id"))
        title = _TITLE_TIME_PREFIX.sub("", title) or title
        kind = HIGHLIGHT_TYPES_ZH.get(str(self.highlight.get("type")), "高光")
        plane = "球场坐标" if pitch else "画面平面"
        return Banner(title, format_clock(frame, self.fps), f"{kind} · 第 {frame} 帧 · {plane}")

    def annotate(self, frame: int, kind: str = "normal") -> FrameAnnotations:
        """Annotations of canonical ``frame`` (``kind`` is the edit segment kind; the
        overlay is the same for every kind, :func:`render_edit` adds the part badges)."""
        pitch = self.is_pitch_frame(frame)
        players = self._players_in(frame)
        marks: list[PlayerMark] = []
        numbers = False
        order = {"context": 0, "normal": 1, "focus": 2}

        def emphasis_of(player: _Player) -> Emphasis:
            # a box without a track, or a person off the pitch: a faint ring, no team colour
            eligible = self._eligible(player, frame)
            return self._emphasis(player.track_id, frame) if eligible else "context"

        for player in sorted(players, key=lambda p: order[emphasis_of(p)]):
            emphasis = emphasis_of(player)
            if emphasis == "context" and not self.show_context:
                continue
            label, held, evidence = None, None, ()
            if emphasis != "context":
                label, held, evidence = self._label(frame, player, pitch)
                numbers = numbers or bool(evidence)
            badges: tuple[Badge, ...] = ()
            if evidence:  # every number carries its own marker; a compact one off focus
                badges = (Badge("未验证", "unverified", compact=emphasis != "focus"),)
            marks.append(
                PlayerMark(
                    player.box,
                    team=player.team if self._eligible(player, frame) else None,
                    label=label,
                    emphasis=emphasis,
                    ring=self._ring(frame, player, pitch, emphasis),
                    badges=badges,
                    evidence_ids=evidence,
                    label_held=held,
                )
            )
        _focus, _pair, involved = self._roles(frame)
        skeletons = tuple(
            Skeleton(
                self._pose[(frame, track)],
                emphasis=self._emphasis(track, frame),
                box=next((p.box for p in players if p.track_id == track), None),
            )
            for track in sorted(involved)
            if (frame, track) in self._pose
            and any(p.track_id == track and self._eligible(p, frame) for p in players)
        )
        link = self._link(frame, players, pitch)
        numbers = numbers or (link is not None and bool(link.evidence_ids))
        arrow = self._facing_arrow(frame, players, pitch)
        unverified = Badge("未验证", "unverified")
        badges: list[Badge] = [unverified] if numbers else []
        if not pitch:
            badges.append(Badge("仅画面平面", "image_plane"))
        # the arrow's estimate calls for the frame's 未验证 only where the arrow is drawn
        arrow_badges = (unverified,) if arrow is not None and not numbers else ()
        return FrameAnnotations(
            players=tuple(marks),
            skeletons=skeletons,
            ball=self._ball_mark(frame),
            arrows=(arrow,) if arrow is not None else (),
            lines=(link,) if link is not None else (),
            minimap=self._minimap(frame, players, pitch),
            banner=self._banner(frame, pitch),
            badges=tuple(badges),
            arrow_badges=arrow_badges,
        )

    def _frame_width(self, frame: int) -> float:
        """Frame width in pixels: ``frame_size``, else the calibration's, else 1920."""
        if self.frame_size is not None:
            return float(self.frame_size[0])
        row = self._calib.get(frame)
        width = _number(row.get("width")) if row is not None else None
        return width if width is not None else 1920.0

    def freeze_layer(
        self, rows: Sequence[CardRow], *, title: str | None = None, dim: float = 0.35
    ) -> FrameAnnotations:
        """The extra layer of the freeze frame: a data card with ``rows`` on the side away
        from the focus player, and a spotlight on that player at the key frame (the rest of
        the frame darkened by ``dim``)."""
        key = self._peak
        focus, _pair, _involved = self._roles(key)
        spotlight = None
        side: Literal["right", "left"] = "right"
        width = self._frame_width(key)
        for player in self._players_in(key):
            if player.track_id == focus and self._eligible(player, key):
                x1, y1, x2, y2 = player.box
                radius = max((y2 - y1) * 1.1, 90.0)
                spotlight = ((x1 + x2) / 2, (y1 + y2) / 2, radius)
                side = "left" if (x1 + x2) / 2 > width / 2 else "right"
        footer = "数值为自动估计，误差未测量（未验证）"
        return FrameAnnotations(
            card=InfoCard(title or f"关键帧 · 第 {key} 帧", tuple(rows), footer=footer, side=side),
            dim=dim if spotlight is not None else min(dim, 0.2),
            spotlight=spotlight,
        )


@dataclass(frozen=True)
class HighlightRender:
    """What :func:`render_highlight` wrote for one highlight; paths are relative to its
    ``out_dir`` (POSIX), ready for :func:`gbf.report.write_report`."""

    highlight_id: str
    clip: str
    manifest: str
    keyframe: str
    keyframe_card: str
    edit: EditResult


def render_highlight(
    video: str | Path,
    out_dir: str | Path,
    scene: HighlightScene,
    *,
    card_rows: Sequence[CardRow] = (),
    card_omitted: int = 0,
    style: Style | None = None,
    audio_source: str | Path | None = None,
    audio_offset_s: float | None = None,
    source_sha256: str | None = None,
    restrictions: Sequence[str] = (),
    slow_factor: int = 4,
    slow_before_s: float = 0.6,
    slow_after_s: float = 0.4,
    freeze_s: float = 2.0,
    overwrite: bool = False,
) -> HighlightRender:
    """Render one highlight of ``scene`` into ``out_dir``.

    Writes ``clips/<id>.mp4`` (the normal-speed window, slow motion around the peak frame
    inside the peak's shot of the scene's ``cuts``, a freeze-frame card, see
    :func:`plan_highlight_edit` and :func:`render_edit`, with ``audio_offset_s`` as there)
    with its ``clips/<id>.render.json`` manifest, ``keyframes/<id>.png`` (the peak frame
    with the overlay) and ``keyframes/<id>-card.png`` (the same frame with the data card of
    ``card_rows``, as the freeze shows it, whose footer counts the ``card_omitted`` rows the
    caller left out with those the frame has no room for). The clip and the images carry
    ``restrictions`` in their own metadata. The highlight's ``id`` must be a plain file name.
    """
    out = Path(out_dir)
    highlight = scene.highlight
    ident = str(highlight["id"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", ident):
        raise RenderError(
            "render_highlight", [f"highlight id {ident!r} cannot be used as a file name"]
        )
    style = (style or Style()).with_fonts()
    start, peak, end = (
        int(highlight["start_frame"]),
        int(highlight["peak_frame"]),
        int(highlight["end_frame"]),
    )
    plan = plan_highlight_edit(
        start,
        end,
        peak,
        scene.fps,
        slow_factor=slow_factor,
        slow_before_s=slow_before_s,
        slow_after_s=slow_after_s,
        freeze_s=freeze_s,
        cuts=scene.cuts,
    )
    scene._set_focus_window(plan)
    freeze = scene.freeze_layer(card_rows)
    if card_omitted and freeze.card is not None:
        freeze = replace(freeze, card=replace(freeze.card, omitted=card_omitted))
    clip = out / "clips" / f"{ident}.mp4"
    keyframe = out / "keyframes" / f"{ident}.png"
    keyframe_card = out / "keyframes" / f"{ident}-card.png"
    if not overwrite:
        existing = [path.name for path in (keyframe, keyframe_card) if path.exists()]
        if existing:
            raise FileExistsError(f"{', '.join(existing)} exist; pass overwrite=True")
    edit = render_edit(
        video,
        clip,
        plan,
        annotate=scene.annotate,
        freeze_layer=freeze,
        style=style,
        audio_source=audio_source,
        audio_offset_s=audio_offset_s,
        source_sha256=source_sha256,
        restrictions=restrictions,
        overwrite=overwrite,
    )
    frame = next(iter(read_frames(video, [peak]).values()))
    base = scene.annotate(peak, "normal")
    save_png(keyframe, draw_annotations(frame, base, style), restrictions=restrictions)
    card_frame = draw_annotations(frame, merge_annotations(base, freeze), style)
    save_png(keyframe_card, card_frame, restrictions=restrictions)
    return HighlightRender(
        highlight_id=ident,
        clip=f"clips/{ident}.mp4",
        manifest=f"clips/{edit.manifest_path.name}",
        keyframe=f"keyframes/{ident}.png",
        keyframe_card=f"keyframes/{ident}-card.png",
        edit=edit,
    )


__all__ = [
    "AUDIO_SAMPLE_RATE",
    "BADGE_KINDS",
    "BALL_GROUND_MAX_SPEED_MPS",
    "BALL_SPEED_SPAN_FRAMES",
    "CLOSEUP_BOX_SHARE",
    "DEFAULT_BADGE_COLOURS",
    "DEFAULT_CRF",
    "DEFAULT_TEAM_COLOURS",
    "DUEL_LINK_WINDOW_S",
    "EDGE_TOUCH_PER_BOX_HEIGHT",
    "EMPHASES",
    "FACING_ARROW_LENGTH_M",
    "FACING_LABEL_ZH",
    "HALPE26_LIMBS",
    "HALPE26_LOWER_BODY",
    "HALPE26_NAMES",
    "HIGHLIGHT_TYPES_ZH",
    "LINK_LABELS_ZH",
    "MINIMAP_BALL_MARGIN_M",
    "REPLAY_LOOP_MIN_S",
    "RESTRICTIONS_COMMENT_PREFIX",
    "RING_RX_PER_BOX_WIDTH",
    "RING_RY_PER_RX",
    "TEAM_NAMES_ZH",
    "TIGHT_UNITS",
    "TRACK_HOLD_MAX_GAP",
    "AnnotationSource",
    "Arrow",
    "AudioPart",
    "AudioSourceFacts",
    "AudioSpan",
    "Badge",
    "BallMark",
    "Banner",
    "CardRow",
    "EditPlan",
    "EditResult",
    "EditSegment",
    "FacingGates",
    "FrameAnnotations",
    "HighlightRender",
    "HighlightScene",
    "InfoCard",
    "LinkLine",
    "Minimap",
    "MinimapDot",
    "MinimapGeometry",
    "PlayerMark",
    "RenderError",
    "SceneTables",
    "Silence",
    "Skeleton",
    "Style",
    "TextLabel",
    "VideoWriter",
    "WrittenVideo",
    "annotation_texts",
    "camera_footprint",
    "check_frame",
    "draw_annotations",
    "first_sample_offset",
    "format_clock",
    "ground_circle_px",
    "merge_annotations",
    "minimap_geometry",
    "plan_highlight_edit",
    "probe_audio",
    "render_edit",
    "render_highlight",
    "restrictions_comment",
    "save_png",
    "stamp_watermark",
    "write_video",
]
