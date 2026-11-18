"""Stitch audit: which pieces of the track table could be re-joined as one person.

gbf splits a track id where a player left the picture and came back (gbf.track_tools
split_tracks), and the plugin itself gives a fresh id to a player it lost for a moment, so one
person can hold two ids. For every pair of pieces of one shot, an earlier piece A and a later
piece B, this module decides whether B can be PROVEN to be A's person, and records the decision
with its evidence. It never changes the track table: the refine stage runs it in audit mode
(docs/design/m1-highlights.md, "Stitch audit"), and nothing downstream reads it until its
assumptions are measured and certified.

Two proofs, either of which stitches; everything else keeps the pieces apart.

Path G, geometric closure. A's anchor is its last clean row (frame t_A, ground point p_A; a
clean row is tall enough, inside the frame, on a pitch-usable frame and overlaps no other box).
At frame t A's person stands within r(t) = max_speed * (t - t_A) / fps + 2 * position_error
of p_A: the reach disc. On a frame t of the first ``window_s`` after B's first frame, the disc
is covered when it lies inside the pitch less ``pitch_margin_m``, and the image of its rim at
the grass and ``head_height_m`` above it lies in front of the camera and inside the image less
``cover_margin_px_1080`` and the graphics bands, with a person of ``person_height_m`` at least
``min_person_px_1080`` tall all along it. A person standing in a covered disc is then wholly in
the picture. The rows whose boxes meet the image of the disc (the convex hull of the two rims;
any id, -1 included, any role, any team) must be B's row alone: a decision frame when that row
is clean and its box has the shape of one person. Any other row meeting the image, a second row
of B, or a row of B whose box misses the image makes a contradiction frame; a covered frame on
which no box meets the image is a missing frame (the person who must stand in the disc has no
row: S3 visibly failed). On every frame of the window, a row of B placed beyond r (ground point
known, box not at a frame edge) is a contradiction. B is A's person when there are at least
``min_decision_frames`` decision frames spanning ``min_decision_span_s``, the first at most
``first_decision_max_frames`` after B's first clean row, no contradiction frame and at most
``max_missing_frames`` missing frames.

The proof rests on assumptions that must be measured before the rule may change anything:
(S1) nobody runs faster than ``max_speed_mps``; (S2) ground points lie within
``position_error_m`` of the truth; (S3) a person wholly in the picture, on the pitch, tall
enough and not under broadcast graphics has a row; (S5) within a shot the camera moves
continuously, checked here: every frame from t_A on has a calibration whose rotation, focal
length and centre change by no more than the per-frame limits since its last change. Given
them, A's person stands in the covered disc and so has a row meeting its image, and the only
such row is B's. Against S3 failing by occlusion or a merged box, any other piece of the shot
that ended within ``rival_horizon_s`` before B began and could have reached B (or whose
position is unknown) makes the pair abstain.

Path J, jersey key. Both pieces read the same number on their own rows (the plugin's per-row
reads, which gbf.track_tools.confident_jerseys later rewrites) often, consistently and over a
span of time; both teams are verified by the shirt-colour check and equal; no other piece of
that team (or without a team) reads the number while A or B is seen, or reads a number one edit
away; and when both pieces have positions, B is within reach. Off by default
(``jersey_enabled``): the plugin's per-row reads may be votes over its tracklet, and its OCR
precision is unmeasured. Passing pairs are listed as ``stitch_path_disabled``.

Vetoes for both paths: majority roles that differ or are not player/goalkeeper; a piece that
jumps (two consecutive clean rows further apart than one person can run in the time between
them: an id switch inside it); a team contradiction (the team labels left in the table differ,
both teams verified and different, or one piece's own colour evidence against the other's
verified team); and two different confident jersey numbers. Every pair a path passes is a
partner of its two pieces; a piece with several partners on one side keeps them only when they
form a chain whose links are enabled proofs that agree with the plugin's id (the nearest link
stays, the others are ``implied_by_chain``), otherwise all of them are ``ambiguous``. When B
continues the plugin id of another earlier piece, a proof that B is A contradicts the plugin:
the pair is listed under ``conflicts_with_plugin``, not as a stitch.

It is pure: the caller passes the refine stage's tables and reports and stores the result.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from gbf.geometry import Camera, project
from gbf.sngs_import import cameras_by_frame
from gbf.track_tools import PITCH_HALF_LENGTH_M, PITCH_HALF_WIDTH_M, boxes_touch_edge

COLUMNS: tuple[str, ...] = (
    "frame",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "role",
    "team",
    "jersey",
    "pitch_usable",
    "pitch_x",
    "pitch_y",
)
"""Columns of the track table the audit reads (``source_track_id`` too when present)."""

READ_COLUMNS: tuple[str, ...] = ("frame", "track_id", "source_track_id", "jersey", "jersey_conf")
"""Columns of the per-row jersey reads (:func:`raw_reads`)."""

STITCH_ROLES: tuple[str, ...] = ("player", "goalkeeper")
DECISIONS: tuple[str, ...] = (
    "stitch",
    "stitch_path_disabled",
    "implied_by_chain",
    "conflicts_with_plugin",
    "ambiguous",
    "abstain",
)
"""Decisions of a pair, in the order the report counts them."""


class StitchError(ValueError):
    """Invalid input to gbf.stitch; ``problems`` lists everything found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        super().__init__(f"{where}: " + "; ".join(self.problems))


def _number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float | np.integer | np.floating)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class StitchParams:
    """Settings of the stitch audit (hand-set starting values; none is validated).

    Attributes:
        max_speed_mps: no player runs faster (S1); gbf.kinematics uses the same bound.
        position_error_m: bound on the error of one ground point (S2); the reach disc adds it
            twice, once for the anchor and once for the frame looked at.
        window_s: path G looks at the frames of this long after B's first frame.
        anchor_max_age_s: A's anchor must lie this close to A's last row.
        min_decision_frames, min_decision_span_s: decision frames path G needs, and the
            time from the first to the last of them.
        max_missing_frames: covered frames on which no box meets the image of the disc (the
            person who must stand in it was not detected: S3 visibly failed) that path G
            tolerates in its window.
        first_decision_max_frames: the first decision frame lies at most this many frames
            after B's first clean row, so B's start is checked too.
        boundary_points: points sampled on each rim of the reach disc.
        head_height_m: the upper rim, above any head (a jump included), so a person standing
            in a covered disc is wholly in the picture.
        person_height_m, min_person_px_1080: a person of this height must be at least this
            tall in pixels (at 1080 lines) everywhere on the disc (S3 holds only for persons
            the detector can see); also the expected height of B's box.
        cover_margin_px_1080: the disc's image keeps this far from the image edges (pixels
            at 1080 lines).
        graphics_top_share, graphics_bottom_share: bands of the image height where
            broadcast graphics (score, captions) may hide a person; the disc keeps out.
        pitch_margin_m: the disc keeps this far inside the touchlines and goal lines, where
            the plugin may drop people.
        clean_min_box_px_1080: a clean row's box is at least this tall (pixels at 1080
            lines; the team check's 50 px at 1080).
        max_width_per_height, box_height_ratio: B's box on a decision frame has the shape of
            one person: width at most this share of its height, height between these
            multiples of the expected height of a person standing on its ground point.
        role_majority: share of a piece's rows its role needs to count as the piece's role.
        rival_horizon_s: pieces that ended this shortly before B began are rivals.
        candidate_max_gap_s: pairs with a longer gap are not considered.
        max_rotation_deg_per_frame, max_focal_change_per_frame,
        max_centre_move_m_per_frame: the camera counts as continuous (S5) when its
            calibration changes by at most these per frame since its last change.
        min_prototype_distance: a verified team needs team prototypes at least this far
            apart (CIE76 delta E).
        jersey_enabled: whether path J may stitch.
        jersey_min_reads, jersey_min_agreement, jersey_min_mean_conf, jersey_min_span_s:
            a piece's number for path J: read at least this often, on this share of its
            reads, with this mean confidence, first and last read this far apart.
        jersey_rival_min_reads, jersey_rival_min_conf: a number another piece reads at least
            this often with at least this confidence counts against path J.
        jersey_voted_share: when at least this share of the plugin tracks with
            ``jersey_min_reads`` reads repeat one number with one confidence on every row,
            the reads look like votes over the tracklet.
        max_listed_abstentions: the report lists at most this many abstaining pairs (all are
            counted).
    """

    max_speed_mps: float = 12.0
    position_error_m: float = 2.0
    window_s: float = 2.0
    anchor_max_age_s: float = 2.0
    min_decision_frames: int = 8
    min_decision_span_s: float = 0.3
    first_decision_max_frames: int = 5
    max_missing_frames: int = 0
    boundary_points: int = 64
    head_height_m: float = 2.4
    person_height_m: float = 1.8
    min_person_px_1080: float = 50.0
    cover_margin_px_1080: float = 15.0
    graphics_top_share: float = 0.12
    graphics_bottom_share: float = 0.08
    pitch_margin_m: float = 3.0
    clean_min_box_px_1080: float = 50.0
    max_width_per_height: float = 0.8
    box_height_ratio: tuple[float, float] = (0.6, 1.5)
    role_majority: float = 0.8
    rival_horizon_s: float = 2.0
    candidate_max_gap_s: float = 10.0
    max_rotation_deg_per_frame: float = 3.0
    max_focal_change_per_frame: float = 0.05
    max_centre_move_m_per_frame: float = 5.0
    min_prototype_distance: float = 20.0
    jersey_enabled: bool = False
    jersey_min_reads: int = 10
    jersey_min_agreement: float = 0.95
    jersey_min_mean_conf: float = 0.85
    jersey_min_span_s: float = 1.0
    jersey_rival_min_reads: int = 2
    jersey_rival_min_conf: float = 0.5
    jersey_voted_share: float = 0.9
    max_listed_abstentions: int = 1000

    def __post_init__(self) -> None:
        if isinstance(self.box_height_ratio, list):  # as_dict() stores it as a list
            object.__setattr__(self, "box_height_ratio", tuple(self.box_height_ratio))
        problems: list[str] = []
        positive = (
            "max_speed_mps",
            "window_s",
            "anchor_max_age_s",
            "head_height_m",
            "person_height_m",
            "max_width_per_height",
            "candidate_max_gap_s",
            "max_rotation_deg_per_frame",
            "max_focal_change_per_frame",
            "max_centre_move_m_per_frame",
        )
        for name in positive:
            value = getattr(self, name)
            if not _number(value) or value <= 0:
                problems.append(f"{name} must be a finite number > 0, got {value!r}")
        non_negative = (
            "position_error_m",
            "min_decision_span_s",
            "min_person_px_1080",
            "cover_margin_px_1080",
            "pitch_margin_m",
            "clean_min_box_px_1080",
            "rival_horizon_s",
            "min_prototype_distance",
            "jersey_min_span_s",
        )
        for name in non_negative:
            value = getattr(self, name)
            if not _number(value) or value < 0:
                problems.append(f"{name} must be a finite number >= 0, got {value!r}")
        shares = (
            "graphics_top_share",
            "graphics_bottom_share",
            "role_majority",
            "jersey_min_agreement",
            "jersey_min_mean_conf",
            "jersey_rival_min_conf",
            "jersey_voted_share",
        )
        for name in shares:
            value = getattr(self, name)
            if not _number(value) or not 0 <= value <= 1:
                problems.append(f"{name} must be a number in [0, 1], got {value!r}")
        counts = (
            ("min_decision_frames", 1),
            ("first_decision_max_frames", 0),
            ("max_missing_frames", 0),
            ("boundary_points", 8),
            ("jersey_min_reads", 1),
            ("jersey_rival_min_reads", 1),
            ("max_listed_abstentions", 0),
        )
        for name, least in counts:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                problems.append(f"{name} must be an int >= {least}, got {value!r}")
        if not isinstance(self.jersey_enabled, bool):
            problems.append(f"jersey_enabled must be a bool, got {self.jersey_enabled!r}")
        ratio = self.box_height_ratio
        if (
            not isinstance(ratio, tuple)
            or len(ratio) != 2
            or not all(_number(v) for v in ratio)
            or not 0 < ratio[0] < ratio[1]
        ):
            problems.append("box_height_ratio must be a tuple (low, high) with 0 < low < high")
        if (
            _number(self.graphics_top_share)
            and _number(self.graphics_bottom_share)
            and self.graphics_top_share + self.graphics_bottom_share >= 1
        ):
            problems.append("graphics_top_share + graphics_bottom_share must be < 1")
        if problems:
            raise StitchError("StitchParams", problems)

    def as_dict(self) -> dict[str, Any]:
        """JSON form for reports and receipts."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


DEFAULT_PARAMS = StitchParams()


# --------------------------------------------------------------------------- geometry


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Vertices (k, 2) of the convex hull of ``points`` (n, 2) in order (Andrew's monotone
    chain); fewer than three distinct points come back as they are."""
    pts = np.unique(np.asarray(points, dtype=np.float64).reshape(-1, 2), axis=0)
    if len(pts) < 3:
        return pts

    def turn(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def boxes_meet_polygon(boxes: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Whether each box (n, 4) x1 y1 x2 y2 shares at least one point with the convex polygon
    (k, 2); touching counts. Separating-axis test over the two image axes and the polygon's
    edge normals."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(boxes) == 0 or len(polygon) == 0:
        return np.zeros(len(boxes), dtype=bool)
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    axes = np.vstack([[1.0, 0.0], [0.0, 1.0], normals])
    poly = polygon @ axes.T
    p_min, p_max = poly.min(axis=0), poly.max(axis=0)
    corners = np.stack(
        [boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1
    )
    proj = corners @ axes.T
    b_min, b_max = proj.min(axis=1), proj.max(axis=1)
    separated = (b_max < p_min) | (b_min > p_max)
    return ~separated.any(axis=1)


def reach_image(
    camera: Camera, centre: Sequence[float], radius: float, params: StitchParams | None = None
) -> tuple[str, np.ndarray | None]:
    """Whether the reach disc of ``radius`` around ground point ``centre`` is covered on the
    camera's frame (module docstring), and the convex hull of its image.

    Returns ("covered", hull) or (reason, None) with reason "near_pitch_line" (the disc does
    not lie inside the pitch less ``pitch_margin_m``), "behind_camera", "outside_view" (a rim
    point outside the image less the margin and the graphics bands) or "too_small" (a person
    on the disc would be shorter than ``min_person_px_1080``). The rims are the polygons of
    ``boundary_points`` corners circumscribed about the disc: a convex polygon that contains
    the disc and lies wholly in front of the camera images to the convex polygon of its
    corners' images, so the hull contains the image of the disc and testing the corners
    suffices.
    """
    params = DEFAULT_PARAMS if params is None else params
    x0, y0 = float(centre[0]), float(centre[1])
    n = params.boundary_points
    outer = radius / math.cos(math.pi / n)
    margin = params.pitch_margin_m
    if (
        abs(x0) + outer > PITCH_HALF_LENGTH_M - margin
        or abs(y0) + outer > PITCH_HALF_WIDTH_M - margin
    ):
        return "near_pitch_line", None
    angles = 2.0 * math.pi * np.arange(n) / n
    rim = np.column_stack([x0 + outer * np.cos(angles), y0 + outer * np.sin(angles)])
    zeros = np.zeros((n, 1))
    points = np.vstack(
        [
            np.hstack([rim, zeros]),
            np.hstack([rim, zeros - params.head_height_m]),
            np.hstack([rim, zeros - params.person_height_m]),
        ]
    )
    uv, _, valid = project(camera, points)
    if not bool(valid.all()):
        return "behind_camera", None
    width, height = float(camera.width), float(camera.height)
    scale = height / 1080.0
    pad = params.cover_margin_px_1080 * scale
    left, right = pad, width - 1.0 - pad
    top = params.graphics_top_share * height + pad
    bottom = height - 1.0 - params.graphics_bottom_share * height - pad
    ground, head, body = uv[:n], uv[n : 2 * n], uv[2 * n :]
    rims = np.vstack([ground, head])
    if (
        (rims[:, 0] < left).any()
        or (rims[:, 0] > right).any()
        or (rims[:, 1] < top).any()
        or (rims[:, 1] > bottom).any()
    ):
        return "outside_view", None
    if float(np.hypot(*(ground - body).T).min()) < params.min_person_px_1080 * scale:
        return "too_small", None
    return "covered", convex_hull(rims)


def _one_person(box: np.ndarray, ground: np.ndarray, camera: Camera, params: StitchParams) -> bool:
    """Whether ``box`` has the shape of one person standing on ``ground`` (no merged box)."""
    width, height = float(box[2] - box[0]), float(box[3] - box[1])
    if height <= 0 or width / height > params.max_width_per_height:
        return False
    foot = [float(ground[0]), float(ground[1]), 0.0]
    uv, _, valid = project(camera, np.array([foot, [foot[0], foot[1], -params.person_height_m]]))
    if not bool(valid.all()):
        return False
    expected = float(np.hypot(*(uv[0] - uv[1])))
    low, high = params.box_height_ratio
    return expected > 0 and low <= height / expected <= high


# --------------------------------------------------------------------------- inputs


def raw_reads(imported: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """The plugin's own per-row jersey reads next to the refined piece ids (:data:`READ_COLUMNS`).

    gbf.track_tools.confident_jerseys rewrites the jersey column, so the reads come from the
    imported table, row for row: gbf.enhance.clean_tracks keeps the rows and their order.
    Raises StitchError when the two tables do not hold the same rows in the same order.
    """
    problems = []
    for name, table in (("imported", imported), ("cleaned", cleaned)):
        if not isinstance(table, pd.DataFrame):
            problems.append(f"{name} must be a DataFrame")
    if problems:
        raise StitchError("raw_reads", problems)
    need = {
        "imported": ("frame", "track_id", "jersey", "jersey_conf"),
        "cleaned": ("frame", "track_id"),
    }
    for name, table in (("imported", imported), ("cleaned", cleaned)):
        missing = [c for c in need[name] if c not in table.columns]
        if missing:
            problems.append(f"{name} misses column(s) {missing}")
    if problems:
        raise StitchError("raw_reads", problems)
    source = (
        cleaned["source_track_id"] if "source_track_id" in cleaned.columns else cleaned["track_id"]
    ).to_numpy(dtype=np.int64)
    if (
        len(imported) != len(cleaned)
        or not np.array_equal(imported["frame"].to_numpy(), cleaned["frame"].to_numpy())
        or not np.array_equal(imported["track_id"].to_numpy(dtype=np.int64), source)
    ):
        raise StitchError(
            "raw_reads", ["the imported and cleaned tables do not hold the same rows in order"]
        )
    jersey = imported["jersey"].astype(object)
    return pd.DataFrame(
        {
            "frame": cleaned["frame"].to_numpy(dtype=np.int64),
            "track_id": cleaned["track_id"].to_numpy(dtype=np.int64),
            "source_track_id": source,
            # an object column keeps None (a plain array of str and None would become a
            # string column with NaN)
            "jersey": pd.Series(jersey.where(jersey.notna(), None).to_list(), dtype=object),
            "jersey_conf": imported["jersey_conf"].to_numpy(dtype=np.float64),
        }
    )


@dataclass(frozen=True)
class _Point:
    frame: int
    x: float
    y: float


@dataclass(frozen=True)
class _Piece:
    track_id: int
    source: int
    shot: int
    first: int
    last: int
    role: str | None
    team: str | None
    jersey: str | None
    anchor: _Point | None
    first_clean: _Point | None
    jump: bool


@dataclass(frozen=True)
class _Reads:
    number: str | None
    count: int
    total: int
    mean_conf: float
    first: int
    last: int
    strong: frozenset[str]


def _rounded(value: float) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), 3)


def _one_edit(a: str, b: str) -> bool:
    """Whether ``b`` is one insertion, deletion or substitution away from ``a``."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


class _Scene:
    """The tables of one audit, indexed for the pair tests."""

    def __init__(
        self,
        tracks: pd.DataFrame,
        calib: pd.DataFrame,
        shots: pd.DataFrame,
        fps: float,
        team_report: Mapping[str, Any],
        role_report: Mapping[str, Any],
        reads: pd.DataFrame | None,
        params: StitchParams,
    ) -> None:
        self.params = params
        self.fps = fps
        self.frame = tracks["frame"].to_numpy(dtype=np.int64)
        self.track = tracks["track_id"].to_numpy(dtype=np.int64)
        source = tracks["source_track_id"] if "source_track_id" in tracks.columns else None
        self.source = (source if source is not None else tracks["track_id"]).to_numpy(np.int64)
        self.boxes = tracks[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
        self.ground = tracks[["pitch_x", "pitch_y"]].to_numpy(dtype=np.float64)
        self.rows_at: dict[int, np.ndarray] = {
            int(f): np.asarray(rows, dtype=np.int64)
            for f, rows in pd.DataFrame({"f": self.frame}).groupby("f").indices.items()
        }
        self.cameras = cameras_by_frame(calib, usable_only=False)
        self.usable = {f: self.cameras[f] for f in cameras_by_frame(calib, usable_only=True)}
        n_frames = 1 + max(
            int(calib["frame_index"].max()) if len(calib) else -1,
            int(self.frame.max()) if len(self.frame) else -1,
        )
        self.breaks = self._breaks(n_frames, calib)
        self.break_count = np.cumsum(self.breaks)
        self.clean = self._clean(tracks, calib)
        self.shot_end = {int(s.shot_id): int(s.end_frame) for s in shots.itertuples()}
        self.verified, self.entries = self._teams(team_report, role_report)
        self.pieces = self._pieces(tracks, shots)
        self.by_shot: dict[int, list[_Piece]] = defaultdict(list)
        for piece in self.pieces.values():
            self.by_shot[piece.shot].append(piece)
        self.by_last: dict[int, list[_Piece]] = {}
        self.lasts: dict[int, list[int]] = {}
        for shot, pieces in self.by_shot.items():
            pieces.sort(key=lambda p: (p.first, p.track_id))
            self.by_last[shot] = sorted(pieces, key=lambda p: (p.last, p.track_id))
            self.lasts[shot] = [p.last for p in self.by_last[shot]]
        self.parent = self._parents()
        self.reads, self.read_check = self._reads(reads)

    # ------------------------------------------------------------------ indexing
    def frames(self, seconds: float) -> int:
        return int(round(seconds * self.fps))

    def reach(self, frames: int) -> float:
        return self.params.max_speed_mps * frames / self.fps + 2.0 * self.params.position_error_m

    def continuous(self, start: int, end: int) -> bool:
        """No camera break on frames start + 1 .. end."""
        if end <= start:
            return True
        if end >= len(self.break_count):
            return False
        return int(self.break_count[end] - self.break_count[start]) == 0

    def _breaks(self, n_frames: int, calib: pd.DataFrame) -> np.ndarray:
        """breaks[f]: the camera is not known to move continuously into frame f (S5).

        A frame without a calibration breaks. A held frame repeats an earlier calibration
        and says nothing new. Every other calibrated frame is compared with the last one
        before it that was not held: rotation, focal length and centre may change by at most
        the per-frame limits times the frames between them (the camera may have moved while
        the calibration was held)."""
        params = self.params
        status = {
            int(f): str(s) for f, s in zip(calib["frame_index"], calib["calib_status"], strict=True)
        }
        out = np.zeros(max(n_frames, 0), dtype=bool)
        last: Camera | None = None
        last_frame = 0
        for f in range(len(out)):
            camera = self.cameras.get(f)
            if camera is None:
                out[f] = True
                last = None
                continue
            if last is not None and status.get(f) == "held":
                continue
            if last is not None:
                steps = f - last_frame
                turn = camera.R @ last.R.T
                angle = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(turn) - 1) / 2))))
                focal = abs(math.log(camera.K[0, 0] / last.K[0, 0]))
                moved = float(np.linalg.norm(camera.C - last.C))
                if (
                    angle > params.max_rotation_deg_per_frame * steps
                    or focal > math.log1p(params.max_focal_change_per_frame) * steps
                    or moved > params.max_centre_move_m_per_frame * steps
                ):
                    out[f] = True
            last, last_frame = camera, f
        return out

    def _clean(self, tracks: pd.DataFrame, calib: pd.DataFrame) -> np.ndarray:
        sizes = Counter(
            (int(w), int(h)) for w, h in zip(calib["width"], calib["height"], strict=True)
        )
        if not sizes:
            self.edge = np.ones(len(self.frame), dtype=bool)
            return np.zeros(len(self.frame), dtype=bool)
        size = sizes.most_common(1)[0][0]
        least = self.params.clean_min_box_px_1080 * size[1] / 1080.0
        tall = (self.boxes[:, 3] - self.boxes[:, 1]) >= least
        self.edge = boxes_touch_edge(self.boxes, size)
        inside = ~self.edge
        placed = tracks["pitch_usable"].to_numpy(dtype=bool) & np.isfinite(self.ground).all(axis=1)
        usable_frame = np.isin(self.frame, np.fromiter(self.usable, dtype=np.int64))
        alone = np.ones(len(self.frame), dtype=bool)
        for rows in self.rows_at.values():
            if len(rows) < 2:
                continue
            b = self.boxes[rows]
            w = np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0])
            h = np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1])
            overlap = (w > 0) & (h > 0)
            np.fill_diagonal(overlap, False)
            alone[rows] = ~overlap.any(axis=1)
        return tall & inside & placed & usable_frame & alone

    def _teams(
        self, team_report: Mapping[str, Any], role_report: Mapping[str, Any]
    ) -> tuple[dict[int, str], dict[int, list[Mapping[str, Any]]]]:
        """(verified team per track id, team-check entries per track id)."""
        entries: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for entry in team_report.get("tracks", []) or []:
            entries[int(entry["track_id"])].append(entry)
        verified: dict[int, str] = {}
        prototypes = team_report.get("prototypes") or {}
        distance = team_report.get("prototype_distance")
        if (
            role_report.get("verdict") == "implausible"
            or any(prototypes.get(team) is None for team in ("left", "right"))
            or distance is None
            or float(distance) < self.params.min_prototype_distance
        ):
            return verified, entries
        for track, found in entries.items():
            if len(found) != 1:
                continue
            entry = found[0]
            if (
                entry.get("verdict") == "consistent"
                and entry.get("nearer_team") == entry.get("team")
                and entry.get("samples_against") == 0
                and not entry.get("cleared")
            ):
                verified[track] = str(entry["team"])
        return verified, entries

    def against(self, track: int, team: str) -> bool:
        """Whether the piece's own shirt-colour evidence speaks against ``team``."""
        for entry in self.entries.get(track, []):
            nearer = entry.get("nearer_team")
            if nearer is not None and nearer != team:
                return True
            if entry.get("team") == team and (
                (entry.get("samples_against") or 0) > 0 or entry.get("verdict") == "inconsistent"
            ):
                return True
        return False

    def _pieces(self, tracks: pd.DataFrame, shots: pd.DataFrame) -> dict[int, _Piece]:
        params = self.params
        starts = shots.sort_values("start_frame")
        start_frames = starts["start_frame"].to_numpy(dtype=np.int64)
        shot_ids = starts["shot_id"].to_numpy(dtype=np.int64)
        roles = tracks["role"].astype(object).to_numpy(dtype=object)
        teams = tracks["team"].astype(object).where(tracks["team"].notna(), None).to_numpy(object)
        jerseys = tracks["jersey"].astype(object).where(tracks["jersey"].notna(), None)
        jerseys = jerseys.to_numpy(dtype=object)
        anchor_age = self.frames(params.anchor_max_age_s)
        pieces: dict[int, _Piece] = {}
        groups = pd.DataFrame({"t": self.track}).groupby("t").indices
        for track, rows in groups.items():
            if int(track) < 0:
                continue
            rows = np.asarray(rows, dtype=np.int64)
            rows = rows[np.argsort(self.frame[rows], kind="stable")]
            frames = self.frame[rows]
            first, last = int(frames[0]), int(frames[-1])
            shot_index = np.searchsorted(start_frames, [first, last], side="right") - 1
            if shot_index[0] < 0 or shot_index[0] != shot_index[1]:
                continue  # outside every shot or across a cut: never an endpoint
            role_counts = Counter(roles[rows])
            role, n_role = role_counts.most_common(1)[0]
            team_counts = Counter(t for t in teams[rows] if t is not None)
            jersey = next((j for j in jerseys[rows] if j not in (None, "")), None)
            clean_rows = rows[self.clean[rows]]
            anchor = first_clean = None
            if len(clean_rows):
                end_row, start_row = clean_rows[-1], clean_rows[0]
                if last - int(self.frame[end_row]) <= anchor_age:
                    anchor = _Point(int(self.frame[end_row]), *map(float, self.ground[end_row]))
                first_clean = _Point(
                    int(self.frame[start_row]), *map(float, self.ground[start_row])
                )
            pieces[int(track)] = _Piece(
                track_id=int(track),
                source=int(self.source[rows[0]]),
                shot=int(shot_ids[shot_index[0]]),
                first=first,
                last=last,
                role=str(role) if n_role >= params.role_majority * len(rows) else None,
                team=team_counts.most_common(1)[0][0] if team_counts else None,
                jersey=str(jersey) if jersey is not None else None,
                anchor=anchor,
                first_clean=first_clean,
                jump=self._jump(clean_rows),
            )
        return pieces

    def _jump(self, clean_rows: np.ndarray) -> bool:
        """Whether two consecutive clean rows of a piece lie further apart than one person can
        run in the time between them: the piece holds two people (an id switch)."""
        if len(clean_rows) < 2:
            return False
        frames = self.frame[clean_rows]
        steps = np.diff(frames)
        distance = np.hypot(*np.diff(self.ground[clean_rows], axis=0).T)
        reach = self.params.max_speed_mps * steps / self.fps + 2.0 * self.params.position_error_m
        return bool((distance[steps > 0] > reach[steps > 0]).any())

    def _reads(self, reads: pd.DataFrame | None) -> tuple[dict[int, _Reads], dict[str, Any]]:
        """Per-piece number reads, and whether the plugin's reads look like tracklet votes."""
        check: dict[str, Any] = {"plugin_tracks_checked": 0, "constant_share": None, "vary": None}
        if reads is None or not len(reads):
            return {}, check
        params = self.params
        numbers = reads["jersey"].astype(object).where(reads["jersey"].notna(), None)
        numbers = numbers.map(lambda v: str(v).strip() if v is not None else "")
        conf = reads["jersey_conf"].to_numpy(dtype=np.float64)
        table = pd.DataFrame(
            {
                "frame": reads["frame"].to_numpy(dtype=np.int64),
                "track_id": reads["track_id"].to_numpy(dtype=np.int64),
                "source": reads["source_track_id"].to_numpy(dtype=np.int64),
                "number": numbers.to_numpy(dtype=object),
                "conf": np.where(np.isfinite(conf), conf, 0.0),
            }
        )
        # a shirt number is one to three digits; anything else (empty, "nan", "?") is no read
        digits = table["number"].map(lambda s: s.isascii() and s.isdigit() and len(s) <= 3)
        table = table[digits.to_numpy(dtype=bool) & (table["track_id"] >= 0).to_numpy()]
        out: dict[int, _Reads] = {}
        for track, group in table.groupby("track_id"):
            counts = Counter(group["number"])
            number, count = counts.most_common(1)[0]
            top = group[group["number"] == number]
            confident = group[group["conf"] >= params.jersey_rival_min_conf]
            strong = Counter(confident["number"])
            out[int(track)] = _Reads(
                number=str(number),
                count=int(count),
                total=len(group),
                mean_conf=float(top["conf"].mean()),
                first=int(top["frame"].min()),
                last=int(top["frame"].max()),
                strong=frozenset(
                    str(n) for n, c in strong.items() if c >= params.jersey_rival_min_reads
                ),
            )
        eligible = constant = 0
        for _, group in table.groupby("source"):
            if len(group) < params.jersey_min_reads:
                continue
            eligible += 1
            constant += int(group["number"].nunique() == 1 and group["conf"].nunique() == 1)
        check["plugin_tracks_checked"] = eligible
        if eligible:
            share = constant / eligible
            check["constant_share"] = round(share, 3)
            check["vary"] = share < params.jersey_voted_share
        return out, check

    def strong(self, reads: _Reads | None) -> bool:
        params = self.params
        return (
            reads is not None
            and reads.count >= params.jersey_min_reads
            and reads.count >= params.jersey_min_agreement * reads.total
            and reads.mean_conf >= params.jersey_min_mean_conf
            and reads.last - reads.first >= self.frames(params.jersey_min_span_s)
        )

    def _parents(self) -> dict[int, int | None]:
        """Per piece, the piece of the same shot whose plugin id it continues (the latest one
        of that id ending before it starts), or None: ids are already split at cuts, so the
        plugin claims no identity across one."""
        groups: dict[tuple[int, int], list[_Piece]] = defaultdict(list)
        for piece in self.pieces.values():
            groups[(piece.shot, piece.source)].append(piece)
        parents: dict[int, int | None] = {}
        for group in groups.values():
            group.sort(key=lambda p: (p.last, p.track_id))
            lasts = [p.last for p in group]
            for b in group:
                k = bisect.bisect_left(lasts, b.first) - 1
                while k >= 0 and group[k].track_id == b.track_id:
                    k -= 1
                parents[b.track_id] = group[k].track_id if k >= 0 else None
        return parents

    def ended_between(self, shot: int, start: int, stop: int) -> list[_Piece]:
        """Pieces of ``shot`` whose last frame lies in [start, stop)."""
        lasts = self.lasts.get(shot)
        if lasts is None:
            return []
        lo, hi = bisect.bisect_left(lasts, start), bisect.bisect_left(lasts, stop)
        return self.by_last[shot][lo:hi]

    # ------------------------------------------------------------------ pair tests
    def plugin_relation(self, a: _Piece, b: _Piece) -> str:
        """The plugin's own view of the pair: "agrees" when B continues A's plugin id, "fresh"
        when B's plugin id starts with B in this shot, "conflict" when B continues the plugin
        id of another earlier piece."""
        parent = self.parent.get(b.track_id)
        if parent is None:
            return "fresh"
        return "agrees" if parent == a.track_id else "conflict"

    def veto(self, a: _Piece, b: _Piece) -> str | None:
        if a.role is None or a.role != b.role or a.role not in STITCH_ROLES:
            return "role"
        if a.jump or b.jump:
            return "piece_jump"
        if a.team is not None and b.team is not None and a.team != b.team:
            return "team_conflict"  # the labels left after the checks disagree
        team_a, team_b = self.verified.get(a.track_id), self.verified.get(b.track_id)
        if team_a is not None and team_b is not None and team_a != team_b:
            return "team_conflict"
        if (team_a is not None and self.against(b.track_id, team_a)) or (
            team_b is not None and self.against(a.track_id, team_b)
        ):
            return "team_conflict"
        if a.jersey is not None and b.jersey is not None and a.jersey != b.jersey:
            return "jersey_conflict"
        return None

    def rivals(self, a: _Piece, b: _Piece) -> list[int]:
        """Other pieces that ended shortly before B began and could have reached B's first
        clean point (or whose position is unknown)."""
        horizon = self.frames(self.params.rival_horizon_s)
        found = []
        for p in self.ended_between(b.shot, b.first - horizon, b.first):
            if p.track_id in (a.track_id, b.track_id):
                continue
            if p.anchor is None or b.first_clean is None:
                found.append(p.track_id)
                continue
            gap = b.first_clean.frame - p.anchor.frame
            distance = math.hypot(b.first_clean.x - p.anchor.x, b.first_clean.y - p.anchor.y)
            if distance <= self.reach(gap):
                found.append(p.track_id)
        return sorted(found)

    def path_g(self, a: _Piece, b: _Piece) -> tuple[str | None, dict[str, Any]]:
        params = self.params
        if a.anchor is None:
            return "no_anchor", {}
        if b.first_clean is None:
            return "no_clean_row", {}
        anchor = a.anchor
        if self.reach(b.first - anchor.frame) > PITCH_HALF_WIDTH_M - params.pitch_margin_m:
            return "gap_too_long", {"radius_m": _rounded(self.reach(b.first - anchor.frame))}
        rivals = self.rivals(a, b)
        if rivals:
            return "rivals", {"rivals": rivals}
        if not self.continuous(anchor.frame, b.first):
            return "camera_discontinuity", {}
        end = min(b.first + self.frames(params.window_s), self.shot_end.get(b.shot, b.first))
        decisions: list[int] = []
        contradictions: list[dict[str, Any]] = []
        missing: list[int] = []
        uncovered: Counter[str] = Counter()
        covered = 0
        empty = np.zeros(0, dtype=np.int64)
        for tau in range(b.first, end + 1):
            if not self.continuous(anchor.frame, tau):
                uncovered["camera_discontinuity"] += 1
                break
            radius = self.reach(tau - anchor.frame)
            rows = self.rows_at.get(tau, empty)
            mine = rows[self.track[rows] == b.track_id]
            # B's own rows, wherever they are: one placed beyond A's reach cannot be A's person
            placed = mine[np.isfinite(self.ground[mine]).all(axis=1) & ~self.edge[mine]]
            far = np.hypot(*(self.ground[placed] - (anchor.x, anchor.y)).T) > radius
            if far.any():
                contradictions.append({"frame": tau, "track_ids": [], "why": "outside_reach"})
                continue
            camera = self.usable.get(tau)
            if camera is None:
                uncovered["not_pitch_usable"] += 1
                continue
            state, hull = reach_image(camera, (anchor.x, anchor.y), radius, params)
            if state != "covered" or hull is None:
                uncovered[state] += 1
                continue
            covered += 1
            meets = boxes_meet_polygon(self.boxes[rows], hull)
            others = rows[meets & (self.track[rows] != b.track_id)]
            if len(others) or len(mine) > 1:
                contradictions.append(
                    {
                        "frame": tau,
                        "track_ids": sorted({int(t) for t in self.track[others]}),
                        **({} if len(others) else {"why": "two_rows_of_b"}),
                    }
                )
                continue
            if len(mine) == 0:
                missing.append(tau)  # A's person stands in the covered disc and has no row
                continue
            row = int(mine[0])
            if not meets[np.flatnonzero(rows == row)[0]]:
                contradictions.append({"frame": tau, "track_ids": [], "why": "outside_disc"})
                continue
            if not self.clean[row]:
                continue  # B there but its box overlaps another or is cut: no decision
            if not _one_person(self.boxes[row], self.ground[row], camera, params):
                contradictions.append({"frame": tau, "track_ids": [], "why": "box_not_one_person"})
                continue
            decisions.append(tau)
        evidence: dict[str, Any] = {
            "anchor": {"frame": anchor.frame, "x": _rounded(anchor.x), "y": _rounded(anchor.y)},
            "window": [b.first, end],
            "covered_frames": covered,
            "decision_frames": len(decisions),
            "first_decision": decisions[0] if decisions else None,
            "last_decision": decisions[-1] if decisions else None,
            "radius_m": [
                _rounded(self.reach(decisions[0] - anchor.frame)),
                _rounded(self.reach(decisions[-1] - anchor.frame)),
            ]
            if decisions
            else None,
            "missing_frames": len(missing),
            "uncovered": dict(sorted(uncovered.items())),
        }
        if contradictions:
            evidence["contradictions"] = contradictions[:5]
            evidence["contradiction_frames"] = len(contradictions)
            return "contradiction", evidence
        if len(missing) > params.max_missing_frames:
            evidence["missing"] = missing[:5]
            return "missing_frames", evidence
        if not decisions:
            return "no_decision_frames", evidence
        if len(decisions) < params.min_decision_frames:
            return "too_few_decision_frames", evidence
        if decisions[-1] - decisions[0] < math.ceil(params.min_decision_span_s * self.fps - 1e-9):
            return "decision_span_short", evidence
        if decisions[0] - b.first_clean.frame > params.first_decision_max_frames:
            return "late_first_decision", evidence
        return None, evidence

    def path_j(self, a: _Piece, b: _Piece) -> tuple[str | None, dict[str, Any]]:
        ra, rb = self.reads.get(a.track_id), self.reads.get(b.track_id)
        if not (self.strong(ra) and self.strong(rb)) or ra is None or rb is None:
            return "reads_weak", {}
        if ra.number != rb.number:
            return "numbers_differ", {}
        team = self.verified.get(a.track_id)
        if team is None or self.verified.get(b.track_id) != team:
            return "team_unverified", {}
        if a.source == b.source and self.read_check.get("vary") is not True:
            return "reads_may_be_votes", {}
        key = str(ra.number)
        for p in self.by_shot.get(a.shot, []):
            if p.track_id in (a.track_id, b.track_id) or p.team not in (team, None):
                continue
            rp = self.reads.get(p.track_id)
            if rp is None:
                continue
            if any(_one_edit(n, key) for n in rp.strong):
                return "confusable_number", {"piece": p.track_id}
            seen = (p.first <= a.last and a.first <= p.last) or (
                p.first <= b.last and b.first <= p.last
            )
            if key in rp.strong and seen:
                return "number_on_two_pieces", {"piece": p.track_id}
        evidence: dict[str, Any] = {"number": key, "reads": [ra.count, rb.count]}
        if a.anchor is not None and b.first_clean is not None:
            gap = b.first_clean.frame - a.anchor.frame
            distance = math.hypot(b.first_clean.x - a.anchor.x, b.first_clean.y - a.anchor.y)
            evidence["distance_m"] = _rounded(distance)
            if distance > self.reach(gap):
                return "too_far", evidence
        return None, evidence


# --------------------------------------------------------------------------- audit


def audit(
    tracks: pd.DataFrame,
    calib: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    fps: float,
    team_report: Mapping[str, Any] | None = None,
    role_report: Mapping[str, Any] | None = None,
    reads: pd.DataFrame | None = None,
    params: StitchParams | None = None,
) -> dict[str, Any]:
    """Decide for every pair of pieces of one shot whether the later is the earlier's person.

    ``tracks`` is the refine stage's track table (:data:`COLUMNS`), ``calib`` its calibration
    (gbf.sngs_import CALIB_SCHEMA columns), ``shots`` the shot table, ``team_report`` and
    ``role_report`` the reports of gbf.team_check (as the refine stage stores it) and
    gbf.role_check, ``reads`` the per-row jersey reads (:func:`raw_reads`; without them path J
    has nothing to read). Returns a JSON-ready report: mode "audit" (nothing applied),
    params, the paths enabled, diagnostics, counts (pairs, decisions, vetoes and the reasons
    of each path), and the pairs by decision: ``stitches`` (decision stitch or
    stitch_path_disabled, with evidence), ``conflicts_with_plugin``, ``ambiguous``, and the
    first ``max_listed_abstentions`` of ``abstentions`` with ``abstentions_total``.
    """
    params = DEFAULT_PARAMS if params is None else params
    problems = []
    if not isinstance(tracks, pd.DataFrame):
        problems.append("tracks must be a DataFrame")
    else:
        missing = [c for c in COLUMNS if c not in tracks.columns]
        if missing:
            problems.append(f"tracks misses column(s) {missing}")
    if not isinstance(shots, pd.DataFrame) or not {"shot_id", "start_frame", "end_frame"} <= set(
        shots.columns
    ):
        problems.append("shots must be a DataFrame with shot_id, start_frame and end_frame")
    if not _number(fps) or fps <= 0:
        problems.append(f"fps must be a finite number > 0, got {fps!r}")
    if reads is not None:
        missing = [c for c in READ_COLUMNS if c not in reads.columns]
        if missing:
            problems.append(f"reads misses column(s) {missing}")
    if problems:
        raise StitchError("audit", problems)
    scene = _Scene(
        tracks, calib, shots, float(fps), team_report or {}, role_report or {}, reads, params
    )
    max_gap = scene.frames(params.candidate_max_gap_s)
    pairs: list[dict[str, Any]] = []
    for shot in sorted(scene.by_shot):
        for b in scene.by_shot[shot]:
            for a in scene.ended_between(shot, b.first - max_gap, b.first):
                pairs.append(_pair(scene, a, b))
    _resolve(pairs, params)
    return _report(scene, pairs, params)


def _pair(scene: _Scene, a: _Piece, b: _Piece) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "earlier": a.track_id,
        "later": b.track_id,
        "shot_id": a.shot,
        "earlier_last": a.last,
        "later_first": b.first,
        "gap_frames": b.first - a.last - 1,
        "plugin": scene.plugin_relation(a, b),
    }
    veto = scene.veto(a, b)
    if veto is not None:
        entry.update(decision="abstain", veto=veto, geometry=None, jersey=None)
        return entry
    g_reason, g_evidence = scene.path_g(a, b)
    j_reason, j_evidence = scene.path_j(a, b)
    entry["geometry"] = g_reason or "pass"
    entry["jersey"] = j_reason or "pass"
    if g_evidence:
        entry["geometry_evidence"] = g_evidence
    if j_reason is None:
        entry["jersey_evidence"] = j_evidence
    entry["decision"] = "abstain"
    return entry


def _path(entry: Mapping[str, Any]) -> str | None:
    passed = [name for name in ("geometry", "jersey") if entry.get(name) == "pass"]
    return "+".join(passed) if passed else None


def _resolve(pairs: list[dict[str, Any]], params: StitchParams) -> None:
    """Set each pair's decision.

    Every pair a path passes is a partner of its two pieces, whether its path is enabled or
    not and whether it agrees with the plugin's id or not. A piece with several partners on
    one side keeps them only when they form a chain whose links are enabled proofs that agree
    with the plugin (one player who left the picture twice, A -> B -> C, also proves A -> C):
    the link to the nearest partner stays, the others are ``implied_by_chain``. Any other set
    of partners makes all of them ``ambiguous``. Then a pair against the plugin's id is
    ``conflicts_with_plugin``, one proven by an enabled path a ``stitch``, the rest
    ``stitch_path_disabled``."""
    proven = []
    for entry in pairs:
        path = _path(entry)
        if path is None:
            continue
        entry["path"] = path
        proven.append(entry)

    def enabled(entry: Mapping[str, Any]) -> bool:
        return entry["geometry"] == "pass" or (params.jersey_enabled and entry["jersey"] == "pass")

    links = {(e["earlier"], e["later"]) for e in proven if enabled(e) and e["plugin"] != "conflict"}
    by_later: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_earlier: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in proven:
        by_later[entry["later"]].append(entry)
        by_earlier[entry["earlier"]].append(entry)
    marks: dict[tuple[int, int], str] = {}

    def settle(group: list[dict[str, Any]], end: str, keep_last: bool) -> None:
        if len(group) < 2:
            return
        ids = [e[end] for e in group]
        if all((x, y) in links for x, y in zip(ids, ids[1:], strict=False)):
            kept = group[-1] if keep_last else group[0]
            for entry in group:
                if entry is not kept:
                    key = (entry["earlier"], entry["later"])
                    marks.setdefault(key, "implied_by_chain")
        else:
            for entry in group:
                marks[(entry["earlier"], entry["later"])] = "ambiguous"

    for group in by_later.values():
        settle(sorted(group, key=lambda e: (e["earlier_last"], e["earlier"])), "earlier", True)
    for group in by_earlier.values():
        settle(sorted(group, key=lambda e: (e["later_first"], e["later"])), "later", False)
    for entry in proven:
        mark = marks.get((entry["earlier"], entry["later"]))
        if mark == "ambiguous":
            entry["decision"] = "ambiguous"
        elif entry["plugin"] == "conflict":
            entry["decision"] = "conflicts_with_plugin"
        elif mark is not None:
            entry["decision"] = mark
        elif enabled(entry):
            entry["decision"] = "stitch"
        else:
            entry["decision"] = "stitch_path_disabled"


def _report(scene: _Scene, pairs: list[dict[str, Any]], params: StitchParams) -> dict[str, Any]:
    decisions = Counter(e["decision"] for e in pairs)
    abstentions = [e for e in pairs if e["decision"] == "abstain"]
    return {
        "mode": "audit",
        "applied": False,
        "params": params.as_dict(),
        "paths_enabled": {"geometry": True, "jersey": params.jersey_enabled},
        "diagnostics": {
            "pieces": len(scene.pieces),
            "frames_with_camera": len(scene.cameras),
            "frames_pitch_usable": len(scene.usable),
            "camera_breaks": int(scene.breaks.sum()),
            "verified_team_pieces": len(scene.verified),
            "jersey_reads": scene.read_check,
        },
        "counts": {
            "pairs": len(pairs),
            "decisions": {d: int(decisions.get(d, 0)) for d in DECISIONS},
            "vetoes": dict(sorted(Counter(e["veto"] for e in pairs if e.get("veto")).items())),
            "geometry": dict(
                sorted(Counter(e["geometry"] for e in pairs if e.get("geometry")).items())
            ),
            "jersey": dict(sorted(Counter(e["jersey"] for e in pairs if e.get("jersey")).items())),
            "stitches_by_path": dict(
                sorted(
                    Counter(
                        e["path"]
                        for e in pairs
                        if e["decision"] in ("stitch", "stitch_path_disabled")
                    ).items()
                )
            ),
        },
        "stitches": [e for e in pairs if e["decision"] in ("stitch", "stitch_path_disabled")],
        "conflicts_with_plugin": [e for e in pairs if e["decision"] == "conflicts_with_plugin"],
        "ambiguous": [e for e in pairs if e["decision"] == "ambiguous"],
        "implied_by_chain": [e for e in pairs if e["decision"] == "implied_by_chain"],
        "abstentions": abstentions[: params.max_listed_abstentions],
        "abstentions_total": len(abstentions),
    }
