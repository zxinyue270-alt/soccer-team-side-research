"""Deterministic broadcast-like frames rendered from known cameras (plan P0, L0 smoke run).

What a frame shows, back to front:

1. Background, per pixel, from where the pixel's ray meets the grass (the closed form of
   gbf.geometry.pixel_to_ground, see ground_from_pixel_matrix): sky above the horizon; grass
   with mowing stripes on the pitch and a 5 m apron; beyond the apron, one flat stand colour
   per side (far, near, behind the goals).
2. Markings of the 105 x 68 m pitch (gbf.synth.pitch) and the goal frames, drawn as lines of
   constant pixel width (line_radius_px). Each 3D segment is clipped to camera depth
   >= NEAR_CLIP_M before projection, so nothing behind the camera is drawn. Centre and
   penalty marks are discs.
3. Two teams (light blue against dark blue, goalkeepers in their own colours), a referee
   and a ball, as upright billboards painted far to near.
4. The frame-number barcode (gbf.synth.barcode), top left.

Pixel convention: as in gbf.geometry, pixel (column i, row j) has its centre at (u, v) =
(i, j) and covers [i - 0.5, i + 0.5) x [j - 0.5, j + 0.5). Rendering is pure numpy with
fixed inputs, so the same arguments always give the same pixels on one machine. Every frame
also has a label image (LABEL_* values) saying what each pixel shows; tests use it.

render_sequence() renders a sequence with a hard cut: frames before cut_at come from the
main camera (frame 0 is exactly gbf.geometry.synthetic_broadcast_camera; the camera then
pans and zooms a little every frame, so no two frames share K, R and C); frames from cut_at
on come from a clearly different high reverse-angle camera. People and the ball never cover
a visible pitch landmark (gbf.synth.pitch.landmarks) in any frame, so a landmark check
cannot fail because of them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gbf.geometry import (
    REFERENCE_CENTRE_M,
    REFERENCE_FOCAL_PX,
    REFERENCE_SIZE,
    Camera,
    look_at,
    project,
    scale_intrinsics,
    world_to_camera,
)
from gbf.synth import barcode, pitch

FPS = "25/1"
"""Frame rate of every synthetic sequence (exact rational, frames per second)."""

LABEL_SKY = 0
LABEL_STAND = 1
LABEL_GRASS = 2
LABEL_LINE = 3
LABEL_PERSON = 4
LABEL_BALL = 5
LABEL_BARCODE = 6
LABEL_NAMES = ("sky", "stand", "grass", "line", "person", "ball", "barcode")

Colour = tuple[int, int, int]

# Colours (RGB). Large areas use the centres of 32-level bins, so lossy coding does not move
# them across the bins of gbf.ingest's cut detector. Only lines and the ball are near white.
SKY: Colour = (112, 144, 208)
STAND_FAR: Colour = (80, 80, 112)
STAND_NEAR: Colour = (144, 48, 48)
STAND_END: Colour = (112, 112, 112)
GRASS_LIGHT: Colour = (80, 144, 48)
GRASS_DARK: Colour = (48, 112, 48)
LINE: Colour = (240, 240, 240)
SKIN: Colour = (208, 144, 112)
BALL_COLOUR: Colour = (240, 240, 240)
KITS: dict[str, tuple[Colour, Colour, Colour]] = {
    # role: (shirt, shorts, socks)
    "team_a": ((112, 176, 240), (48, 80, 144), (112, 176, 240)),
    "team_b": ((16, 48, 112), (16, 16, 48), (16, 48, 112)),
    "goalkeeper_a": ((240, 144, 16), (16, 16, 16), (240, 144, 16)),
    "goalkeeper_b": ((208, 48, 176), (16, 16, 16), (208, 48, 176)),
    "referee": ((240, 208, 16), (16, 16, 16), (16, 16, 16)),
}

APRON_M = 5.0
"""Grass beyond the touchlines and goal lines, metres."""
STRIPE_M = 5.25
"""Width of one mowing stripe along x, metres (20 stripes over the pitch length)."""
NEAR_CLIP_M = 0.5
"""Segments are clipped to camera-frame depth >= this (metres) before projection."""
LINE_RADIUS_PX = 1.0
"""Half-width of a painted line at 720 rows, pixels; see line_radius_px."""
PERSON_HEIGHT_M = 1.8
BALL_RADIUS_M = 0.11
MAX_RUN_M_PER_FRAME = 0.24
"""Fastest a person moves: 6 m/s at 25 fps."""
# Body parts as (bottom, top, width) in metres above the grass, and what colours them.
_BODY: tuple[tuple[float, float, float, str], ...] = (
    (0.00, 0.80, 0.32, "socks"),
    (0.80, 1.02, 0.42, "shorts"),
    (1.02, 1.56, 0.48, "shirt"),
    (1.56, 1.80, 0.22, "skin"),
)
_BODY_HALF_WIDTH_M = 0.5 * max(width for _, _, width, _ in _BODY)
_CHUNK_ROWS = 256
_PIECE_PX = 24.0
_KEEP_OUT_PX = 3.0
_MAX_PLACEMENT_TRIES = 200


class SynthError(ValueError):
    """Invalid arguments for the synthetic renderer. ``problems`` lists every problem."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


@dataclass(frozen=True)
class ShotCamera:
    """A broadcast camera for one shot, defined at the 1920x1080 reference size.

    Frame j of the shot (j = 0, 1, ...) looks at ``target + j * pan_m`` (world metres) with
    focal length ``focal_px * (1 + j * zoom_per_frame)`` pixels at 1920x1080, square pixels,
    zero skew, principal point at the image centre and zero roll (gbf.geometry.look_at).
    Other frame sizes use the same pose with K rescaled by gbf.geometry.scale_intrinsics, so
    a 720p frame is the 1080p frame's geometry, downscaled.
    """

    name: str
    centre: tuple[float, float, float]
    target: tuple[float, float, float]
    focal_px: float
    pan_m: tuple[float, float, float]
    zoom_per_frame: float

    def camera(self, j: int, width: int, height: int) -> Camera:
        """The camera of frame ``j`` of the shot for frames of width x height pixels."""
        focal = self.focal_px * (1.0 + j * self.zoom_per_frame)
        ref_width, ref_height = REFERENCE_SIZE
        reference_K = np.array(
            [
                [focal, 0.0, (ref_width - 1) / 2.0],
                [0.0, focal, (ref_height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        target = np.asarray(self.target, dtype=np.float64) + j * np.asarray(self.pan_m)
        centre = np.asarray(self.centre, dtype=np.float64)
        return Camera(
            K=scale_intrinsics(reference_K, REFERENCE_SIZE, (width, height)),
            R=look_at(centre, target),
            C=centre,
            width=width,
            height=height,
        )


MAIN_CAMERA = ShotCamera(
    name="main",
    centre=REFERENCE_CENTRE_M,
    target=(0.0, 0.0, 0.0),
    focal_px=REFERENCE_FOCAL_PX,
    pan_m=(0.4, -0.15, 0.0),
    zoom_per_frame=0.006,
)
"""The main camera: 36.3 m outside the near touchline, 17.4 m up, aimed at the centre spot."""

REVERSE_CAMERA = ShotCamera(
    name="reverse_high",
    centre=(-6.0, -61.0, -13.0),
    target=(-25.0, 9.0, 0.0),
    focal_px=2300.0,
    pan_m=(-0.3, 0.1, 0.0),
    zoom_per_frame=-0.004,
)
"""A high camera 27 m behind the FAR touchline looking back at the left penalty area: the
near stand and the sky fill the top of its frames, the far stand fills the main camera's."""

SHOT_CAMERAS = (MAIN_CAMERA, REVERSE_CAMERA)


@dataclass(frozen=True)
class Person:
    """A player, goalkeeper or referee; ``role`` is a key of KITS. The ground position at
    frame f is start + f * velocity (metres, metres per frame)."""

    role: str
    start: tuple[float, float]
    velocity: tuple[float, float]

    def ground_point(self, frame_index: int) -> np.ndarray:
        """World point (3,) under the person's feet at ``frame_index``, metres."""
        return np.array(
            [
                self.start[0] + frame_index * self.velocity[0],
                self.start[1] + frame_index * self.velocity[1],
                0.0,
            ]
        )


@dataclass(frozen=True)
class Ball:
    """The ball, rolling on the grass: centre BALL_RADIUS_M above it (z = -0.11)."""

    start: tuple[float, float]
    velocity: tuple[float, float]

    def centre(self, frame_index: int) -> np.ndarray:
        """World point (3,) of the ball's centre at ``frame_index``, metres."""
        return np.array(
            [
                self.start[0] + frame_index * self.velocity[0],
                self.start[1] + frame_index * self.velocity[1],
                -BALL_RADIUS_M,
            ]
        )


@dataclass(frozen=True)
class Scene:
    """Everything on the pitch. ``ball`` is None and ``dropped`` counts people (and the
    ball) that found no place keeping every visible landmark uncovered (see make_scene)."""

    persons: tuple[Person, ...]
    ball: Ball | None
    dropped: int = 0


@dataclass(frozen=True)
class FrameTruth:
    """Ground truth of one rendered frame: its index, its shot and the camera that drew it."""

    frame_index: int
    shot_id: int
    camera: Camera

    def to_dict(self) -> dict[str, Any]:
        """JSON form: K and R as nested row-major lists, C as a list."""
        return {
            "frame_index": self.frame_index,
            "shot_id": self.shot_id,
            "width": self.camera.width,
            "height": self.camera.height,
            "K": self.camera.K.tolist(),
            "R": self.camera.R.tolist(),
            "C": self.camera.C.tolist(),
        }


@dataclass(frozen=True)
class RenderedSequence:
    """Rendered frames (H, W, 3) uint8 RGB and labels (H, W) uint8 (read-only), with the
    camera of every frame and the scene."""

    width: int
    height: int
    fps: str
    seed: int
    cut_at: int | None
    frames: tuple[np.ndarray, ...]
    labels: tuple[np.ndarray, ...]
    truth: tuple[FrameTruth, ...]
    scene: Scene

    @property
    def cuts(self) -> list[int]:
        """Frame indices that start a new shot (frame 0 is never listed)."""
        return [] if self.cut_at is None else [self.cut_at]

    def truth_record(self) -> dict[str, Any]:
        """The camera of every frame as one JSON-ready document (``cameras.json``)."""
        return {
            "format": "gbf.synth.cameras.v1",
            "conventions": (
                "SoccerNet world frame (metres, z down); P = K R [I | -C]; K in pixels of "
                "width x height with pixel centres at integer coordinates; frame_index is the "
                "canonical frame index of the rendered video."
            ),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "seed": self.seed,
            "cuts": self.cuts,
            "frames": [item.to_dict() for item in self.truth],
        }


def line_radius_px(height: int) -> float:
    """Half-width of painted lines in pixels: 1.0 at 720 rows, scaled with the height,
    never below 1.0 (so the pixel holding any point of a line's centreline is a line pixel)."""
    return max(LINE_RADIUS_PX, LINE_RADIUS_PX * height / 720.0)


def shot_of(frame_index: int, cut_at: int | None) -> tuple[int, int]:
    """(shot id, index of the frame within its shot) for a sequence cut at ``cut_at``."""
    if cut_at is None or frame_index < cut_at:
        return 0, frame_index
    return 1, frame_index - cut_at


def sequence_cameras(
    width: int, height: int, n_frames: int, cut_at: int | None
) -> list[FrameTruth]:
    """The camera of every frame of a sequence (see the module docstring)."""
    truth = []
    for frame_index in range(n_frames):
        shot_id, j = shot_of(frame_index, cut_at)
        camera = SHOT_CAMERAS[shot_id].camera(j, width, height)
        truth.append(FrameTruth(frame_index, shot_id, camera))
    return truth


def visible_landmarks(camera: Camera, *, margin_px: float = 3.0) -> dict[str, np.ndarray]:
    """Landmarks (gbf.synth.pitch.landmarks) ``camera`` sees, as name -> (u, v) in pixels.

    A landmark counts when it is in front of the camera, at least ``margin_px`` inside the
    image, and at least ``margin_px`` away from the barcode and its quiet zone.
    """
    named = pitch.landmarks()
    names = list(named)
    uv, _, valid = project(camera, np.array([named[name] for name in names]))
    top, bottom, left, right = barcode.barcode_layout(camera.width, camera.height).outer_box()
    found = {}
    for name, (u, v), ok in zip(names, uv, valid, strict=True):
        if not ok:
            continue
        inside = (
            -0.5 + margin_px <= u <= camera.width - 0.5 - margin_px
            and -0.5 + margin_px <= v <= camera.height - 0.5 - margin_px
        )
        on_barcode = (
            left - 0.5 - margin_px <= u <= right - 0.5 + margin_px
            and top - 0.5 - margin_px <= v <= bottom - 0.5 + margin_px
        )
        if inside and not on_barcode:
            found[name] = np.array([u, v])
    return found


# --- scene ---------------------------------------------------------------------------
_TEAM_A_SHAPE: tuple[tuple[str, float, float], ...] = (
    ("goalkeeper_a", -47.0, 2.0),
    ("team_a", -33.0, -22.0),
    ("team_a", -36.0, -8.0),
    ("team_a", -36.0, 7.0),
    ("team_a", -32.0, 21.0),
    ("team_a", -20.0, -14.0),
    ("team_a", -22.0, 1.0),
    ("team_a", -19.0, 15.0),
    ("team_a", -7.0, -19.0),
    ("team_a", -5.0, -3.0),
    ("team_a", -8.0, 12.0),
)
FORMATION: tuple[tuple[str, float, float], ...] = (
    *_TEAM_A_SHAPE,
    *((role[:-1] + "b", 2.0 - x, -y) for role, x, y in _TEAM_A_SHAPE),
    ("referee", 6.0, -6.0),
)
"""(role, x, y) starting spots in metres before jitter: 11 per team and a referee."""
BALL_SPOT = (-12.0, -10.0)


def _ball_radius_px(camera: Camera, depth: float) -> float:
    return max(1.2, BALL_RADIUS_M * float(camera.K[0, 0]) / depth)


def _box(
    camera: Camera, points: np.ndarray, half_width_m: float
) -> tuple[float, float, float, float] | None:
    """(left, top, right, bottom) pixels around projected points, widened by half_width_m at
    the depth of the first point; None when a point is behind the camera."""
    uv, depth, valid = project(camera, points)
    if not valid.all():
        return None
    half = half_width_m * float(camera.K[0, 0]) / float(depth[0])
    return (
        float(uv[:, 0].min()) - half,
        float(uv[:, 1].min()) - half,
        float(uv[:, 0].max()) + half,
        float(uv[:, 1].max()) + half,
    )


def _covers(box: tuple[float, float, float, float] | None, marks: dict[str, np.ndarray]) -> bool:
    if box is None:
        return False
    left, top, right, bottom = box
    return any(
        left - _KEEP_OUT_PX <= u <= right + _KEEP_OUT_PX
        and top - _KEEP_OUT_PX <= v <= bottom + _KEEP_OUT_PX
        for u, v in marks.values()
    )


def _in_pitch(x: float, y: float) -> bool:
    return abs(x) <= pitch.HALF_LENGTH_M - 0.5 and abs(y) <= pitch.HALF_WIDTH_M - 0.5


def make_scene(seed: int, cameras: Sequence[FrameTruth]) -> Scene:
    """People and ball for a sequence, drawn from ``seed`` (numpy default_rng).

    Each person starts at its FORMATION spot plus N(0, 2.5 m) jitter per axis and runs in a
    uniformly random direction at up to MAX_RUN_M_PER_FRAME; the ball starts near
    BALL_SPOT (N(0, 1.5 m)). A draw that leaves the pitch during the sequence, or whose
    on-screen box comes within 3 px of a visible landmark in any frame, is drawn again, up
    to 200 times; whatever finds no place is left out and counted in Scene.dropped. Draws
    happen in a fixed order, so one seed always gives the same scene for the same cameras.
    """
    rng = np.random.default_rng(seed)
    marks = [visible_landmarks(item.camera, margin_px=0.0) for item in cameras]
    last_frame = max(item.frame_index for item in cameras)

    def draw(spot: tuple[float, float], spread: float) -> tuple[tuple, tuple] | None:
        dx, dy = rng.normal(0.0, spread, size=2)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        speed = rng.uniform(0.0, MAX_RUN_M_PER_FRAME)
        start = (float(spot[0] + dx), float(spot[1] + dy))
        velocity = (float(speed * math.cos(angle)), float(speed * math.sin(angle)))
        end = (start[0] + last_frame * velocity[0], start[1] + last_frame * velocity[1])
        if _in_pitch(*start) and _in_pitch(*end):
            return start, velocity
        return None

    def person_clear(person: Person) -> bool:
        for item, frame_marks in zip(cameras, marks, strict=True):
            foot = person.ground_point(item.frame_index)
            head = foot + np.array([0.0, 0.0, -PERSON_HEIGHT_M])
            if _covers(_box(item.camera, np.stack([foot, head]), _BODY_HALF_WIDTH_M), frame_marks):
                return False
        return True

    def ball_clear(ball: Ball) -> bool:
        for item, frame_marks in zip(cameras, marks, strict=True):
            centre = ball.centre(item.frame_index)[None, :]
            uv, depth, valid = project(item.camera, centre)
            if not valid.all():
                continue
            radius = _ball_radius_px(item.camera, float(depth[0]))
            box = (uv[0, 0] - radius, uv[0, 1] - radius, uv[0, 0] + radius, uv[0, 1] + radius)
            if _covers(box, frame_marks):
                return False
        return True

    persons: list[Person] = []
    dropped = 0
    for role, x, y in FORMATION:
        placed = None
        for _ in range(_MAX_PLACEMENT_TRIES):
            candidate = draw((x, y), 2.5)
            if candidate is not None and person_clear(Person(role, *candidate)):
                placed = Person(role, *candidate)
                break
        if placed is None:
            dropped += 1
        else:
            persons.append(placed)
    ball = None
    for _ in range(_MAX_PLACEMENT_TRIES):
        candidate = draw(BALL_SPOT, 1.5)
        if candidate is not None and ball_clear(Ball(*candidate)):
            ball = Ball(*candidate)
            break
    if ball is None:
        dropped += 1
    return Scene(persons=tuple(persons), ball=ball, dropped=dropped)


# --- drawing -------------------------------------------------------------------------
# Background codes, their colours and labels (indexed by code).
_SKY, _FAR, _NEAR, _END, _LIGHT, _DARK = range(6)
_BACKGROUND_RGB = np.array(
    [SKY, STAND_FAR, STAND_NEAR, STAND_END, GRASS_LIGHT, GRASS_DARK], np.uint8
)
_BACKGROUND_LABEL = np.array(
    [LABEL_SKY, LABEL_STAND, LABEL_STAND, LABEL_STAND, LABEL_GRASS, LABEL_GRASS], np.uint8
)


def ground_from_pixel_matrix(camera: Camera) -> np.ndarray:
    """A (3, 3) with [x, y, 1] proportional to A [u, v, 1] for the grass point (x, y, 0)
    seen at pixel (u, v); the pixel's ray meets the grass in front of the camera exactly when
    the third component of A [u, v, 1] is positive.

    A is the inverse of the ground homography K [r1 r2 -R C] (r1, r2 the first two columns of
    R): for a grass point, K R (X - C) = K (x r1 + y r2 - R C), whose third component is the
    point's depth. This is gbf.geometry.pixel_to_ground for z0 = 0 in closed form, cheap
    enough to evaluate at every pixel; tests pin the two against each other.
    """
    homography = camera.K @ np.column_stack([camera.R[:, 0], camera.R[:, 1], -camera.R @ camera.C])
    return np.linalg.inv(homography)


def ground_points(camera: Camera, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(xy (..., 2) metres, valid (...)) of the grass points seen at pixels uv (..., 2);
    xy is NaN where the ray does not meet the grass in front of the camera."""
    pixels = np.asarray(uv, dtype=np.float64)
    a = ground_from_pixel_matrix(camera)
    u, v = pixels[..., 0], pixels[..., 1]
    w = a[2, 0] * u + a[2, 1] * v + a[2, 2]
    valid = w > 0.0
    safe = np.where(valid, w, 1.0)
    x = np.where(valid, (a[0, 0] * u + a[0, 1] * v + a[0, 2]) / safe, np.nan)
    y = np.where(valid, (a[1, 0] * u + a[1, 1] * v + a[1, 2]) / safe, np.nan)
    return np.stack([x, y], axis=-1), valid


def _plane(coefficients: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """c0 * u + c1 * v + c2 over the grid of rows v and columns u, with one allocation."""
    values = np.add.outer(coefficients[1] * v, coefficients[0] * u)
    values += coefficients[2]
    return values


def _paint_background(camera: Camera, rgb: np.ndarray, labels: np.ndarray) -> None:
    """Sky, stands and striped grass for every pixel, from where its ray meets the grass."""
    grass_x = pitch.HALF_LENGTH_M + APRON_M
    grass_y = pitch.HALF_WIDTH_M + APRON_M
    a = ground_from_pixel_matrix(camera)
    u = np.arange(camera.width, dtype=np.float64)
    for top in range(0, camera.height, _CHUNK_ROWS):
        bottom = min(camera.height, top + _CHUNK_ROWS)
        v = np.arange(top, bottom, dtype=np.float64)
        w = _plane(a[2], u, v)
        sky = w <= 0.0
        w[sky] = 1.0  # x and y of sky pixels are meaningless; their code is set last
        x = _plane(a[0], u, v)
        x /= w
        y = _plane(a[1], u, v)
        y /= w
        grass = (x >= -grass_x) & (x <= grass_x) & (y >= -grass_y) & (y <= grass_y)
        code = np.full(w.shape, _END, dtype=np.uint8)
        code[y < -grass_y] = _FAR
        code[y > grass_y] = _NEAR
        stripe = np.floor((x[grass] + pitch.HALF_LENGTH_M) / STRIPE_M) % 2.0
        code[grass] = np.where(stripe == 0.0, _LIGHT, _DARK)
        code[sky] = _SKY
        np.take(_BACKGROUND_RGB, code, axis=0, out=rgb[top:bottom])
        np.take(_BACKGROUND_LABEL, code, out=labels[top:bottom])


def clipped_segments(camera: Camera, points: np.ndarray) -> np.ndarray:
    """Pixel segments (S, 2, 2) of a world polyline (N, 3), clipped to depth >= NEAR_CLIP_M.

    The camera-frame depth is affine along a world segment, so a segment that crosses the
    near plane is cut where its depth equals NEAR_CLIP_M; segments entirely nearer than that
    (or behind the camera) are dropped. Only the kept parts are projected.
    """
    depth = world_to_camera(camera, points)[:, 2]
    a, b = points[:-1], points[1:]
    da, db = depth[:-1], depth[1:]
    keep = (da >= NEAR_CLIP_M) | (db >= NEAR_CLIP_M)
    a, b, da, db = a[keep], b[keep], da[keep], db[keep]
    start = a.copy()
    end = b.copy()
    cut_a = da < NEAR_CLIP_M
    t = (NEAR_CLIP_M - da[cut_a]) / (db[cut_a] - da[cut_a])
    start[cut_a] = a[cut_a] + t[:, None] * (b[cut_a] - a[cut_a])
    cut_b = db < NEAR_CLIP_M
    t = (NEAR_CLIP_M - db[cut_b]) / (da[cut_b] - db[cut_b])
    end[cut_b] = b[cut_b] + t[:, None] * (a[cut_b] - b[cut_b])
    if not len(start):
        return np.empty((0, 2, 2))
    uv_start, _, ok_start = project(camera, start)
    uv_end, _, ok_end = project(camera, end)
    ok = ok_start & ok_end
    return np.stack([uv_start[ok], uv_end[ok]], axis=1)


def _clip_to_box(
    a: np.ndarray, b: np.ndarray, box: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Liang-Barsky clip of the 2D segment a-b to box = (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = box
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-d[0], a[0] - xmin),
        (d[0], xmax - a[0]),
        (-d[1], a[1] - ymin),
        (d[1], ymax - a[1]),
    ):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return a + t0 * d, a + t1 * d


def _capsule(
    shape: tuple[int, ...], a: np.ndarray, b: np.ndarray, radius: float
) -> tuple[slice, slice, np.ndarray] | None:
    """(rows, columns, inside) for the pixels whose centres lie within ``radius`` pixels of
    the segment a-b: ``inside`` is a bool array over the bounding box rows x columns of an
    image of ``shape``; None when the box misses the image."""
    height, width = shape[0], shape[1]
    x_lo = max(0, math.floor(min(a[0], b[0]) - radius))
    x_hi = min(width - 1, math.ceil(max(a[0], b[0]) + radius))
    y_lo = max(0, math.floor(min(a[1], b[1]) - radius))
    y_hi = min(height - 1, math.ceil(max(a[1], b[1]) + radius))
    if x_lo > x_hi or y_lo > y_hi:
        return None
    xs = np.arange(x_lo, x_hi + 1, dtype=np.float64)[None, :]
    ys = np.arange(y_lo, y_hi + 1, dtype=np.float64)[:, None]
    d = b - a
    length2 = float(d @ d)
    if length2 > 0.0:
        t = np.clip(((xs - a[0]) * d[0] + (ys - a[1]) * d[1]) / length2, 0.0, 1.0)
    else:
        t = np.zeros((1, 1))
    dx = xs - (a[0] + t * d[0])
    dy = ys - (a[1] + t * d[1])
    inside = dx * dx + dy * dy <= radius * radius
    return slice(y_lo, y_hi + 1), slice(x_lo, x_hi + 1), inside


def _stamp_capsule(mask: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> None:
    """Set every pixel of ``mask`` whose centre lies within ``radius`` of the segment a-b."""
    found = _capsule(mask.shape, a, b, radius)
    if found is not None:
        rows, columns, inside = found
        mask[rows, columns] |= inside


def _stamp_disc(mask: np.ndarray, centre: np.ndarray, radius: float) -> None:
    _stamp_capsule(mask, centre, centre, radius)


def line_mask(camera: Camera) -> np.ndarray:
    """Boolean (H, W) mask of the painted lines, goal frames and marks seen by ``camera``."""
    mask = np.zeros((camera.height, camera.width), dtype=bool)
    radius = line_radius_px(camera.height)
    margin = radius + 2.0
    box = (-margin, -margin, camera.width - 1 + margin, camera.height - 1 + margin)
    for marking in (*pitch.pitch_markings(), *pitch.goal_frames()):
        for a, b in clipped_segments(camera, marking.points):
            clipped = _clip_to_box(a, b, box)
            if clipped is None:
                continue
            start, stop = clipped
            pieces = max(1, math.ceil(float(np.hypot(*(stop - start))) / _PIECE_PX))
            for k in range(pieces):
                _stamp_capsule(
                    mask,
                    start + (stop - start) * (k / pieces),
                    start + (stop - start) * ((k + 1) / pieces),
                    radius,
                )
    spot_points = np.array(list(pitch.spots().values()))
    uv, depth, valid = project(camera, spot_points)
    for centre, z, ok in zip(uv, depth, valid, strict=True):
        if ok and z >= NEAR_CLIP_M:
            spot_radius = pitch.SPOT_RADIUS_M * float(camera.K[0, 0]) / float(z)
            _stamp_disc(mask, centre, max(1.6 * radius, spot_radius))
    return mask


def _fill_rect(
    rgb: np.ndarray,
    labels: np.ndarray,
    u0: float,
    v0: float,
    u1: float,
    v1: float,
    colour: Colour,
    label: int,
) -> None:
    """Fill pixels whose centres lie in [u0, u1] x [v0, v1]; at least the nearest pixel."""
    height, width = labels.shape
    i0, i1 = math.ceil(u0), math.floor(u1)
    j0, j1 = math.ceil(v0), math.floor(v1)
    if i0 > i1:
        i0 = i1 = math.floor(0.5 * (u0 + u1) + 0.5)
    if j0 > j1:
        j0 = j1 = math.floor(0.5 * (v0 + v1) + 0.5)
    i0, i1 = max(i0, 0), min(i1, width - 1)
    j0, j1 = max(j0, 0), min(j1, height - 1)
    if i0 > i1 or j0 > j1:
        return
    rgb[j0 : j1 + 1, i0 : i1 + 1] = colour
    labels[j0 : j1 + 1, i0 : i1 + 1] = label


def _paint_person(
    camera: Camera, rgb: np.ndarray, labels: np.ndarray, person: Person, frame_index: int
) -> None:
    foot = person.ground_point(frame_index)
    heights = sorted({h for bottom, top, _, _ in _BODY for h in (bottom, top)})
    points = np.array([foot + np.array([0.0, 0.0, -h]) for h in heights])
    uv, depth, valid = project(camera, points)
    if not valid.all():
        return
    shirt, shorts, socks = KITS[person.role]
    colours = {"socks": socks, "shorts": shorts, "shirt": shirt, "skin": SKIN}
    scale = float(camera.K[0, 0]) / float(depth[0])
    for bottom, top, width_m, part in _BODY:
        low, high = uv[heights.index(bottom)], uv[heights.index(top)]
        centre_u = 0.5 * (low[0] + high[0])
        half = 0.5 * width_m * scale
        _fill_rect(
            rgb,
            labels,
            centre_u - half,
            high[1],
            centre_u + half,
            low[1],
            colours[part],
            LABEL_PERSON,
        )


def _paint_ball(
    camera: Camera, rgb: np.ndarray, labels: np.ndarray, ball: Ball, frame_index: int
) -> None:
    uv, depth, valid = project(camera, ball.centre(frame_index)[None, :])
    if not valid.all():
        return
    found = _capsule(labels.shape, uv[0], uv[0], _ball_radius_px(camera, float(depth[0])))
    if found is not None:
        rows, columns, inside = found
        rgb[rows, columns][inside] = BALL_COLOUR
        labels[rows, columns][inside] = LABEL_BALL


def render_frame(camera: Camera, scene: Scene, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Render one frame: (rgb (H, W, 3) uint8, labels (H, W) uint8).

    ``frame_index`` positions the people and the ball and is written into the barcode.
    """
    rgb = np.empty((camera.height, camera.width, 3), dtype=np.uint8)
    labels = np.empty((camera.height, camera.width), dtype=np.uint8)
    _paint_background(camera, rgb, labels)
    lines = line_mask(camera)
    rgb[lines] = LINE
    labels[lines] = LABEL_LINE
    drawables: list[tuple[float, int, Person | Ball]] = []
    for order, person in enumerate(scene.persons):
        depth = float(world_to_camera(camera, person.ground_point(frame_index))[2])
        drawables.append((depth, order, person))
    if scene.ball is not None:
        depth = float(world_to_camera(camera, scene.ball.centre(frame_index))[2])
        drawables.append((depth, len(drawables), scene.ball))
    for _, _, item in sorted(drawables, key=lambda entry: (-entry[0], entry[1])):
        if isinstance(item, Person):
            _paint_person(camera, rgb, labels, item, frame_index)
        else:
            _paint_ball(camera, rgb, labels, item, frame_index)
    layout = barcode.draw_barcode(rgb, frame_index)
    top, bottom, left, right = layout.outer_box()
    labels[top:bottom, left:right] = LABEL_BARCODE
    return rgb, labels


def _as_int(value: object) -> int | None:
    """``value`` as an int if it is an integer (bools excluded), else None."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        return None
    return int(value)


def _check_args(
    width: object, height: object, n_frames: object, cut_at: object, seed: object
) -> list[str]:
    problems = []
    for name, value in (("width", width), ("height", height), ("n_frames", n_frames)):
        number = _as_int(value)
        if number is None or number < 1:
            problems.append(f"{name} must be a positive integer, got {value!r}")
    frames = _as_int(n_frames)
    if frames is not None and frames > barcode.MAX_VALUE + 1:
        problems.append(
            f"n_frames must be at most {barcode.MAX_VALUE + 1} (the barcode's range), got {frames}"
        )
    if cut_at is not None:
        cut = _as_int(cut_at)
        if cut is None:
            problems.append(f"cut_at must be an integer or None, got {cut_at!r}")
        elif frames is not None and frames >= 1 and not 1 <= cut <= frames - 1:
            problems.append(f"cut_at must lie in 1..n_frames-1 = 1..{frames - 1}, got {cut}")
    number = _as_int(seed)
    if number is None or number < 0:
        problems.append(f"seed must be a non-negative integer, got {seed!r}")
    return problems


def render_sequence(
    width: int, height: int, n_frames: int = 5, cut_at: int | None = 3, seed: int = 0
) -> RenderedSequence:
    """Render ``n_frames`` frames of width x height pixels at 25 fps with a hard cut.

    Frames 0 .. cut_at - 1 are shot 0 (MAIN_CAMERA), frames cut_at .. n_frames - 1 shot 1
    (REVERSE_CAMERA); cut_at=None renders one shot. ``seed`` picks the people's positions
    and runs (make_scene). Each frame carries its index in its barcode. Raises SynthError
    listing every invalid argument, and gbf.synth.barcode.BarcodeError when the frame is too
    small for the barcode (gbf.synth.barcode.barcode_layout; 472 x 56 pixels at 720 rows).
    """
    problems = _check_args(width, height, n_frames, cut_at, seed)
    if problems:
        raise SynthError("render_sequence", problems)
    barcode.barcode_layout(width, height)
    cameras = sequence_cameras(int(width), int(height), int(n_frames), cut_at)
    scene = make_scene(int(seed), cameras)
    frames, labels = [], []
    for item in cameras:
        rgb, label_image = render_frame(item.camera, scene, item.frame_index)
        rgb.setflags(write=False)
        label_image.setflags(write=False)
        frames.append(rgb)
        labels.append(label_image)
    return RenderedSequence(
        width=int(width),
        height=int(height),
        fps=FPS,
        seed=int(seed),
        cut_at=None if cut_at is None else int(cut_at),
        frames=tuple(frames),
        labels=tuple(labels),
        truth=tuple(cameras),
        scene=scene,
    )
