"""Shirt-colour check of the plugin's team labels.

The tracking plugin labels each player track "left" or "right" (the side the team defends,
gbf.sngs_import) from re-identification features, and some labels are wrong: a track that
wears one team's shirt carries the other team's label. Everything downstream trusts the
label, so such a track puts a player in the wrong team (an opponent counted as an attacker
in the box, a nearest-opponent line that skips a closer opponent). This module checks the
labels against the shirt colours seen on the original frames. It is pure: the caller reads
the frames (gbf.ingest) and stores the results.

1. :func:`plan_samples` chooses the frames to read. Every track with role ``player`` and a
   team gets up to ``samples_per_track`` samples, at most one in each of that many equal
   parts of its life (first to last row), from rows whose box is at least
   ``min_box_height_px`` tall and ``min_aspect`` times as tall as wide (an upright whole
   body), stays ``edge_px`` inside the frame (when the frame size is given) and has at
   most ``max_overlap`` of its torso crop covered by the other boxes of its frame. Frames
   are picked greedily, the one that fills the most open (track, part) slots first, so
   frames shared by many tracks come first, up to ``max_frames`` distinct frames.
   Goalkeepers, referees and unlabelled rows are not sampled.
2. :func:`measure_shirts` crops the torso of every planned box (``torso_x``, ``torso_y``:
   fractions of the box from its left and top edges), drops grass-like pixels (green above
   red and blue by ``grass_margin``), near-black pixels and near-white ones (shirt print,
   pitch lines), and takes the per-channel median in CIELAB of the rest as the sample's
   colour; a crop with less than ``min_shirt_share`` of its pixels left gives none. A
   track's colour is the per-channel median of its samples' colours.
3. :func:`check_teams` takes as each team's prototype the per-channel median of its
   tracks' colours, weighted by their sample counts, over the tracks with at least
   ``min_samples`` samples (a team needs ``min_team_tracks`` of them). Such a track is
   inconsistent when its colour is closer to the other team's prototype than to its own by
   more than ``margin``: its team becomes missing on its player rows, with the missing value
   the table already uses. It is never flipped to the other team; the colour says that the
   label is wrong, not who the player is. Tracks with fewer samples keep their label and are
   reported as unchecked.

A track here is one ``track_id`` with one team label (a track whose rows carry both labels
is checked as two). The check judges whole tracks: a track in which the tracker switched
from one player to another keeps its label when most of its samples agree with it (the
report counts the samples that do not, ``samples_against``). Colours are CIELAB (D65) from
sRGB values (:func:`srgb_to_lab`) and distances are CIE76 delta E. Boxes are in the pixels
of the frames passed in. Every threshold is an empirical starting value
(:class:`TeamCheckParams`), not a validated one.

Typical use, with the canonical video whose frames the boxes were detected on (one decode
pass that ends at the last planned frame, one frame in memory at a time)::

    plan = plan_samples(tracks, frame_size=(width, height))
    with contextlib.closing(gbf.ingest.iter_frames(video)) as frames:
        colours, samples = measure_shirts(plan, frames)
    checked, report = check_teams(tracks, colours, samples=samples)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

TEAMS: tuple[str, ...] = ("left", "right")
BOX_COLUMNS: tuple[str, ...] = ("x1", "y1", "x2", "y2")
TRACK_COLUMNS: tuple[str, ...] = ("frame", "track_id", *BOX_COLUMNS, "role", "team")
"""Columns of the track table (gbf.sngs_import tracks.parquet) the check reads."""

PLAN_COLUMNS: tuple[str, ...] = ("frame", "track_id", "team", *BOX_COLUMNS, "part", "overlap")
""":func:`plan_samples`: one planned sample per row; ``part`` is the part of the track's life
it stands for, ``overlap`` the share of its torso crop covered by other boxes."""

SAMPLE_COLUMNS: tuple[str, ...] = (*PLAN_COLUMNS, "pixels", "L", "a", "b")
"""Per-sample table of :func:`measure_shirts`: the plan plus the shirt pixels used and the
colour (NaN when the crop gives none)."""

COLOUR_COLUMNS: tuple[str, ...] = ("track_id", "team", "planned", "samples", "L", "a", "b")
"""Per-track table of :func:`measure_shirts`: planned samples, samples with a colour and the
median colour (NaN without any)."""

VERDICTS: tuple[str, ...] = ("consistent", "inconsistent", "unchecked")

MIN_CROP_PIXELS = 24
"""A torso crop with fewer pixels gives no colour, whatever its content."""

MAX_CROP_PIXELS = 4096
"""Larger torso crops (close-ups) are subsampled on a regular grid to about this many pixels."""

_INTEGER_COLUMNS = ("frame", "track_id", "part", "pixels", "planned", "samples")

_D65_WHITE = np.array([0.95047, 1.0, 1.08883])
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


class TeamCheckError(ValueError):
    """Invalid input to gbf.team_check; ``problems`` lists everything found."""

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
class TeamCheckParams:
    """Thresholds of the team check (empirical starting values, not validated).

    Attributes:
        samples_per_track: most samples a track gets: its life is cut into this many equal
            parts and each part gives at most one sample.
        min_box_height_px: shorter boxes are not sampled; their torso is a few dozen
            blurred pixels.
        min_aspect: boxes whose height is less than this many times their width are not
            sampled: the torso fractions assume a whole upright body, and in an
            upper-body close-up or around a sitting player they land on the head or the
            background.
        torso_x, torso_y: the torso crop as fractions of the box from its left and top
            edges: the middle of the body across (arms and the grass beside the body stay
            out) and from below the shoulders to above the shorts.
        max_overlap: a row is sampled only when the other boxes of its frame cover at most
            this share of its torso crop (their intersections summed).
        edge_px: boxes closer than this to the frame edge are not sampled: a box cut off by
            the frame puts the torso fractions on the wrong part of the body.
        max_frames: most distinct frames a plan reads.
        grass_margin: a pixel is grass-like when its green value exceeds its red and its
            blue value by at least this much (0-255).
        dark_l: pixels darker than this CIELAB lightness are dropped (deep shadow, hair).
        white_l, white_chroma: pixels lighter than ``white_l`` with a chroma below
            ``white_chroma`` are dropped (white print on shirts, pitch lines, glare).
        min_shirt_share: a sample gives a colour only when at least this share of its
            torso crop is left after the pixel tests.
        min_samples: a track needs this many samples with a colour for a verdict.
        min_team_tracks: a team's prototype needs this many tracks with a verdict-ready
            colour; without both prototypes no track gets a verdict.
        margin: a track is inconsistent when its colour is closer to the other team's
            prototype than to its own by more than this (delta E, CIELAB units).
    """

    samples_per_track: int = 8
    min_box_height_px: float = 50.0
    min_aspect: float = 1.5
    torso_x: tuple[float, float] = (0.3, 0.7)
    torso_y: tuple[float, float] = (0.2, 0.45)
    max_overlap: float = 0.1
    edge_px: float = 2.0
    max_frames: int = 200
    grass_margin: float = 10.0
    dark_l: float = 10.0
    white_l: float = 85.0
    white_chroma: float = 12.0
    min_shirt_share: float = 0.3
    min_samples: int = 3
    min_team_tracks: int = 3
    margin: float = 10.0

    def __post_init__(self) -> None:
        for name in ("torso_x", "torso_y"):  # as_dict() stores them as lists
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))
        problems: list[str] = []
        for name in ("samples_per_track", "max_frames", "min_samples", "min_team_tracks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                problems.append(f"{name} must be an int >= 1, got {value!r}")
        for name in (
            "min_box_height_px",
            "min_aspect",
            "edge_px",
            "grass_margin",
            "dark_l",
            "white_l",
            "white_chroma",
            "margin",
        ):
            value = getattr(self, name)
            if not _number(value) or value < 0:
                problems.append(f"{name} must be a finite number >= 0, got {value!r}")
        for name in ("max_overlap", "min_shirt_share"):
            value = getattr(self, name)
            if not _number(value) or not 0 <= value <= 1:
                problems.append(f"{name} must be a number in [0, 1], got {value!r}")
        for name in ("torso_x", "torso_y"):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or not all(_number(v) for v in value)
                or not 0 <= value[0] < value[1] <= 1
            ):
                problems.append(f"{name} must be a tuple (start, end) with 0 <= start < end <= 1")
        if not problems and self.min_samples > self.samples_per_track:
            problems.append("min_samples must not exceed samples_per_track")
        if problems:
            raise TeamCheckError("TeamCheckParams", problems)

    def as_dict(self) -> dict[str, Any]:
        """JSON form for reports and receipts."""
        return {name: list(v) if isinstance(v, tuple) else v for name, v in asdict(self).items()}


DEFAULT_PARAMS = TeamCheckParams()


def srgb_to_lab(rgb: Any) -> np.ndarray:
    """CIELAB (D65) of sRGB values in 0..255, shape ``(..., 3)`` to ``(..., 3)`` float64."""
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    if values.shape[-1:] != (3,):
        raise TeamCheckError("srgb_to_lab", [f"expected shape (..., 3), got {values.shape}"])
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    t = (linear @ _SRGB_TO_XYZ.T) / _D65_WHITE
    epsilon, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(t > epsilon, np.cbrt(t), (kappa * t + 16.0) / 116.0)
    return np.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )


# --------------------------------------------------------------------------- inputs


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    """A table without rows: integer ids and counts, object team, float everything else."""

    def dtype(name: str) -> Any:
        if name in _INTEGER_COLUMNS:
            return np.int64
        return object if name == "team" else np.float64

    return pd.DataFrame({name: pd.Series([], dtype=dtype(name)) for name in columns})


def _check_columns(table: Any, name: str, columns: Sequence[str], where: str) -> None:
    if not isinstance(table, pd.DataFrame):
        raise TeamCheckError(where, [f"{name} must be a DataFrame, got {type(table).__name__}"])
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise TeamCheckError(where, [f"{name} misses column(s) {missing}"])


def _labels(column: pd.Series, where: str) -> np.ndarray:
    """Team labels as an object array of "left", "right" or None; anything else is refused."""
    values = column.astype(object).where(column.notna(), None).to_numpy(dtype=object)
    unknown = sorted({str(v) for v in values if v is not None and v not in TEAMS})
    if unknown:
        raise TeamCheckError(where, [f"team must be 'left', 'right' or missing, got {unknown}"])
    return values


def _player_rows(tracks: pd.DataFrame, where: str) -> tuple[np.ndarray, np.ndarray]:
    """(mask of player rows with a team and a track id, team labels)."""
    labels = _labels(tracks["team"], where)
    ids = tracks["track_id"].to_numpy(dtype=np.int64)
    roles = tracks["role"].astype(object).to_numpy(dtype=object)
    labelled = np.array([v is not None for v in labels], dtype=bool)
    return (roles == "player") & (ids >= 0) & labelled, labels


def _torso(boxes: np.ndarray, params: TeamCheckParams) -> np.ndarray:
    """Torso crops ``(n, 4)`` [x0, y0, x1, y1] of boxes ``(n, 4)`` [x1, y1, x2, y2]."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.stack(
        [
            boxes[:, 0] + params.torso_x[0] * w,
            boxes[:, 1] + params.torso_y[0] * h,
            boxes[:, 0] + params.torso_x[1] * w,
            boxes[:, 1] + params.torso_y[1] * h,
        ],
        axis=1,
    )


def _samplable(
    boxes: np.ndarray, params: TeamCheckParams, size: tuple[float, float] | None
) -> np.ndarray:
    """Which boxes ``(n, 4)`` the torso crop applies to: finite, tall enough, upright and,
    when the frame size is known, inside the frame."""
    finite = np.isfinite(boxes).all(axis=1)
    safe = np.where(finite[:, None], boxes, 0.0)
    w = safe[:, 2] - safe[:, 0]
    h = safe[:, 3] - safe[:, 1]
    ok = finite & (w > 0) & (h >= params.min_box_height_px) & (h >= params.min_aspect * w)
    if size is not None:
        edge = params.edge_px
        ok &= (safe[:, 0] >= edge) & (safe[:, 1] >= edge)
        ok &= (safe[:, 2] <= size[0] - edge) & (safe[:, 3] <= size[1] - edge)
    return ok


def _covered_share(
    frames: np.ndarray, boxes: np.ndarray, rows: np.ndarray, params: TeamCheckParams
) -> np.ndarray:
    """Share of the torso crop of each row in ``rows`` covered by the other boxes of its frame
    (intersection areas summed, so it can exceed 1)."""
    torso = _torso(boxes[rows], params)
    area = (torso[:, 2] - torso[:, 0]) * (torso[:, 3] - torso[:, 1])
    others = np.flatnonzero(np.isfinite(boxes).all(axis=1))
    left = pd.DataFrame(
        {"i": rows, "frame": frames[rows], **{f"t{k}": torso[:, k] for k in range(4)}}
    )
    right = pd.DataFrame({"j": others, "frame": frames[others]})
    for k, name in enumerate(BOX_COLUMNS):
        right[name] = boxes[others, k]
    pairs = left.merge(right, on="frame")
    pairs = pairs[pairs["i"].to_numpy() != pairs["j"].to_numpy()]
    w = np.minimum(pairs["t2"], pairs["x2"]) - np.maximum(pairs["t0"], pairs["x1"])
    h = np.minimum(pairs["t3"], pairs["y2"]) - np.maximum(pairs["t1"], pairs["y1"])
    covered = (w.clip(lower=0) * h.clip(lower=0)).groupby(pairs["i"]).sum()
    return covered.reindex(rows, fill_value=0.0).to_numpy(dtype=np.float64) / area


# --------------------------------------------------------------------------- plan


def plan_samples(
    tracks: pd.DataFrame,
    params: TeamCheckParams | None = None,
    *,
    frame_size: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """The samples to measure, one row per (frame, track): :data:`PLAN_COLUMNS`.

    ``tracks`` needs :data:`TRACK_COLUMNS` (refine/tracks.parquet). ``frame_size`` is the
    (width, height) of the frames the boxes refer to; with it, boxes within ``edge_px`` of
    the frame edge are not planned (:func:`measure_shirts` drops them anyway). The plan
    holds at most ``max_frames`` distinct frames, sorted by frame and track.
    """
    params = DEFAULT_PARAMS if params is None else params
    where = "plan_samples"
    _check_columns(tracks, "tracks", TRACK_COLUMNS, where)
    if frame_size is not None and (
        not isinstance(frame_size, tuple)
        or len(frame_size) != 2
        or not all(_number(v) and v > 0 for v in frame_size)
    ):
        raise TeamCheckError(where, [f"frame_size must be (width, height), got {frame_size!r}"])
    player, labels = _player_rows(tracks, where)
    frames = tracks["frame"].to_numpy(dtype=np.int64)
    ids = tracks["track_id"].to_numpy(dtype=np.int64)
    boxes = tracks[list(BOX_COLUMNS)].to_numpy(dtype=np.float64)
    if not player.any():
        return _empty(PLAN_COLUMNS)

    # Life parts: unit = (track id, team), part k of n covers [first + k L / n, ...).
    unit = np.where(labels == "right", 1, 0) + 2 * ids
    rows = np.flatnonzero(player)
    life = pd.DataFrame({"unit": unit[rows], "frame": frames[rows]}).groupby("unit")["frame"]
    first = life.min().reindex(unit[rows]).to_numpy()
    span = life.max().reindex(unit[rows]).to_numpy() - first + 1
    n = params.samples_per_track
    part = np.minimum((frames[rows] - first) * n // span, n - 1)
    codes = pd.factorize(unit[rows])[0]
    slot = codes * n + part

    # Rows that may be sampled.
    size = None if frame_size is None else (float(frame_size[0]), float(frame_size[1]))
    ok = _samplable(boxes[rows], params, size)
    share = np.full(len(rows), np.inf)
    share[ok] = _covered_share(frames, boxes, rows[ok], params)
    ok &= share <= params.max_overlap
    candidates = pd.DataFrame(
        {"pos": np.flatnonzero(ok), "frame": frames[rows[ok]], "slot": slot[ok]}
    )
    candidates = candidates.drop_duplicates(["frame", "slot"])
    pos = candidates["pos"].to_numpy()
    cand_slot = candidates["slot"].to_numpy()
    frame_code, frame_values = pd.factorize(candidates["frame"].to_numpy(), sort=True)

    # Greedy: the frame that fills the most open slots first (ties: the earliest frame).
    filled = np.zeros(int(slot.max()) + 1 if len(slot) else 0, dtype=bool)
    picked: list[np.ndarray] = []
    for _ in range(params.max_frames):
        active = ~filled[cand_slot]
        if not active.any():
            break
        best = int(np.argmax(np.bincount(frame_code[active], minlength=len(frame_values))))
        take = active & (frame_code == best)
        filled[cand_slot[take]] = True
        picked.append(pos[take])
    if not picked:
        return _empty(PLAN_COLUMNS)
    chosen = np.sort(np.concatenate(picked))
    source = rows[chosen]
    plan = pd.DataFrame(
        {
            "frame": frames[source],
            "track_id": ids[source],
            "team": labels[source],
            **{name: boxes[source, k] for k, name in enumerate(BOX_COLUMNS)},
            "part": part[chosen].astype(np.int64),
            "overlap": share[chosen],
        }
    )
    return plan.sort_values(["frame", "track_id", "team"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- measure


def _crop_colour(
    image: np.ndarray, box: Sequence[float], params: TeamCheckParams
) -> tuple[np.ndarray | None, int]:
    """(CIELAB colour or None, shirt pixels used) of the torso crop of ``box``."""
    height, width = image.shape[:2]
    values = np.asarray(box, dtype=np.float64).reshape(1, 4)
    if not _samplable(values, params, (float(width), float(height)))[0]:
        return None, 0
    x0, top, x3, bottom = (int(math.floor(v + 0.5)) for v in _torso(values, params)[0])
    crop = image[max(top, 0) : max(bottom, 0), max(x0, 0) : max(x3, 0)]
    count = crop.shape[0] * crop.shape[1]
    if count < MIN_CROP_PIXELS:
        return None, 0
    step = max(1, math.ceil(math.sqrt(count / MAX_CROP_PIXELS)))
    pixels = crop[::step, ::step].reshape(-1, 3)
    rgb = pixels.astype(np.int16)
    grass = (rgb[:, 1] - rgb[:, 0] >= params.grass_margin) & (
        rgb[:, 1] - rgb[:, 2] >= params.grass_margin
    )
    lab = srgb_to_lab(pixels)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    white = (lab[:, 0] > params.white_l) & (chroma < params.white_chroma)
    dark = lab[:, 0] < params.dark_l
    keep = ~(grass | white | dark)
    if int(keep.sum()) < params.min_shirt_share * len(pixels):
        # Too little left without the near-white pixels: the shirt itself is white (a white
        # kit, often floodlit), not white print on a coloured shirt. Keep them.
        keep = ~(grass | dark)
    used = int(keep.sum())
    if used == 0 or used < params.min_shirt_share * len(pixels):
        return None, used
    return np.median(lab[keep], axis=0), used


def shirt_colour(
    frame: np.ndarray, box: Sequence[float], params: TeamCheckParams | None = None
) -> np.ndarray | None:
    """CIELAB colour ``(3,)`` of the shirt in ``box`` (x1, y1, x2, y2) of an RGB frame.

    None when the box is shorter than ``min_box_height_px``, less than ``min_aspect`` times
    as tall as wide or within ``edge_px`` of the frame edge, or when less than
    ``min_shirt_share`` of its torso crop passes the pixel tests (grass, near-black,
    near-white).
    """
    params = DEFAULT_PARAMS if params is None else params
    colour, _ = _crop_colour(_rgb_frame(frame, "shirt_colour"), box, params)
    return colour


def _rgb_frame(image: Any, where: str) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        shape = getattr(image, "shape", None)
        raise TeamCheckError(where, [f"frames must be (height, width, 3) uint8 RGB, got {shape}"])
    return image


def measure_shirts(
    plan: pd.DataFrame,
    frames: Mapping[int, np.ndarray] | Iterable[tuple[int, np.ndarray]],
    params: TeamCheckParams | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shirt colours of the planned samples: ``(colours, samples)``.

    ``frames`` is ``{index: frame}`` (gbf.ingest.read_frames) or ``(index, frame)`` pairs in
    any order (gbf.ingest.iter_frames); frames the plan does not use are skipped, and
    iteration stops once every planned frame has been seen. A planned frame that never comes
    raises :class:`TeamCheckError`. ``samples`` has :data:`SAMPLE_COLUMNS`, ``colours``
    :data:`COLOUR_COLUMNS` with one row per (track_id, team) of the plan.
    """
    params = DEFAULT_PARAMS if params is None else params
    where = "measure_shirts"
    _check_columns(plan, "plan", PLAN_COLUMNS, where)
    boxes = plan[list(BOX_COLUMNS)].to_numpy(dtype=np.float64)
    wanted = {int(f): positions for f, positions in plan.groupby("frame").indices.items()}
    remaining = set(wanted)
    lab = np.full((len(plan), 3), np.nan)
    pixels = np.zeros(len(plan), dtype=np.int64)
    if remaining:
        for index, image in frames.items() if isinstance(frames, Mapping) else frames:
            key = int(index)
            if key not in remaining:
                continue
            remaining.discard(key)
            image = _rgb_frame(image, where)
            for pos in wanted[key]:
                colour, used = _crop_colour(image, boxes[pos], params)
                pixels[pos] = used
                if colour is not None:
                    lab[pos] = colour
            if not remaining:
                break
    if remaining:
        shown = ", ".join(str(f) for f in sorted(remaining)[:20])
        raise TeamCheckError(where, [f"{len(remaining)} planned frame(s) not given: {shown}"])
    samples = plan[list(PLAN_COLUMNS)].copy()
    samples["pixels"] = pixels
    for k, name in enumerate(("L", "a", "b")):
        samples[name] = lab[:, k]
    grouped = samples.groupby(["track_id", "team"], sort=True)
    colours = grouped[["L", "a", "b"]].median()
    colours.insert(0, "samples", grouped["L"].count())
    colours.insert(0, "planned", grouped.size())
    colours = colours.reset_index()
    if colours.empty:
        colours = _empty(COLOUR_COLUMNS)
    return colours[list(COLOUR_COLUMNS)], samples


# --------------------------------------------------------------------------- check


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _missing_value(column: pd.Series) -> Any:
    """The missing value ``column`` already uses: its dtype's, else that of a missing entry."""
    na_value = getattr(column.dtype, "na_value", None)
    if na_value is not None:
        return na_value
    missing = column[column.isna()]
    return missing.iloc[0] if len(missing) else None


def _rounded(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), 2)


def _samples_against(
    samples: pd.DataFrame, prototypes: Mapping[str, Any], margin: float
) -> dict[tuple[int, str], int]:
    """Per (track_id, team): samples whose colour is nearer the other team's prototype by
    more than ``margin``."""
    lab = samples[["L", "a", "b"]].to_numpy(dtype=np.float64)
    teams = samples["team"].astype(object).to_numpy(dtype=object)
    near = {team: np.linalg.norm(lab - prototypes[team], axis=1) for team in TEAMS}
    own = np.where(teams == "left", near["left"], near["right"])
    other = np.where(teams == "left", near["right"], near["left"])
    against = np.isfinite(lab).all(axis=1) & (own - other > margin)
    keys = zip(samples["track_id"].to_numpy(dtype=np.int64), teams, against, strict=True)
    counts: dict[tuple[int, str], int] = {}
    for track, team, flag in keys:
        key = (int(track), str(team))
        counts[key] = counts.get(key, 0) + int(flag)
    return counts


def check_teams(
    tracks: pd.DataFrame,
    colours: pd.DataFrame,
    params: TeamCheckParams | None = None,
    *,
    samples: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check the team labels against the shirt colours: ``(tracks, report)``.

    ``tracks`` is the table the plan came from, ``colours`` the per-track table of
    :func:`measure_shirts`. Returns a copy of ``tracks`` in which the player rows of every
    inconsistent track have a missing team (the table's own missing value; nothing is
    flipped), and a JSON-ready report: ``params``, ``prototypes`` (per team its colour and
    the tracks and samples behind it, or None), ``prototype_distance``, ``tracks`` (per
    labelled player track: team, rows, planned and measured samples, colour, distances to
    its own and the other team's prototype, the nearer team and the verdict, with a
    ``reason`` when unchecked, and ``cleared``: "inconsistent" when its label was cleared
    here, else None, for a caller that clears others to mark them) and ``counts``
    (``rows_cleared``: the player rows whose label was cleared).

    "consistent" means not contradicted: the colour is not clearly nearer the other team,
    which includes a colour far from both (a box on someone who is not a player). With the
    per-sample table of :func:`measure_shirts` as ``samples``, every checked track also gets
    ``samples_against``, its samples that are on their own clearly nearer the other team:
    a consistent track with some may hold two players (the tracker switched identities),
    and the check leaves such a track's label alone.
    """
    params = DEFAULT_PARAMS if params is None else params
    where = "check_teams"
    _check_columns(tracks, "tracks", ("track_id", "role", "team"), where)
    _check_columns(colours, "colours", COLOUR_COLUMNS, where)
    if samples is not None:
        _check_columns(samples, "samples", ("track_id", "team", "L", "a", "b"), where)
    player, labels = _player_rows(tracks, where)
    ids = tracks["track_id"].to_numpy(dtype=np.int64)
    units = (
        pd.DataFrame({"track_id": ids[player], "team": labels[player].astype(object)})
        .groupby(["track_id", "team"], sort=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    measured = colours[list(COLOUR_COLUMNS)].astype({"team": object})
    stray = measured.merge(units, on=["track_id", "team"], how="left", indicator=True)
    stray = stray[stray["_merge"] == "left_only"]
    if len(stray):
        listed = ", ".join(
            f"{int(t)}/{s}" for t, s in zip(stray["track_id"], stray["team"], strict=True)
        )
        raise TeamCheckError(where, [f"colours has tracks without player rows: {listed}"])
    table = units.merge(measured, on=["track_id", "team"], how="left")
    table["planned"] = table["planned"].fillna(0).astype(np.int64)
    table["samples"] = table["samples"].fillna(0).astype(np.int64)
    lab = table[["L", "a", "b"]].to_numpy(dtype=np.float64)
    measured_n = table["samples"].to_numpy()
    ready = (measured_n >= params.min_samples) & np.isfinite(lab).all(axis=1)
    team_of = table["team"].to_numpy(dtype=object)

    prototypes: dict[str, np.ndarray | None] = {}
    report_prototypes: dict[str, Any] = {}
    for team in TEAMS:
        members = ready & (team_of == team)
        if members.sum() < params.min_team_tracks:
            prototypes[team] = None
            report_prototypes[team] = None
            continue
        weights = measured_n[members].astype(np.float64)
        centre = np.array([_weighted_median(lab[members, k], weights) for k in range(3)])
        prototypes[team] = centre
        report_prototypes[team] = {
            "lab": [_rounded(v) for v in centre],
            "tracks": int(members.sum()),
            "samples": int(measured_n[members].sum()),
        }
    both = all(prototypes[team] is not None for team in TEAMS)
    gap = float(np.linalg.norm(prototypes["left"] - prototypes["right"])) if both else None
    against = None
    if samples is not None and both:
        against = _samples_against(samples, prototypes, params.margin)

    entries: list[dict[str, Any]] = []
    inconsistent: set[tuple[int, str]] = set()
    for i, row in enumerate(table.itertuples(index=False)):
        team = str(row.team)
        entry: dict[str, Any] = {
            "track_id": int(row.track_id),
            "team": team,
            "rows": int(row.rows),
            "planned": int(row.planned),
            "samples": int(row.samples),
            "lab": [_rounded(v) for v in lab[i]] if np.isfinite(lab[i]).all() else None,
            "distance_own": None,
            "distance_other": None,
            "nearer_team": None,
            "samples_against": None,
            "verdict": "unchecked",
            "reason": None,
            "cleared": None,
        }
        if not ready[i]:
            entry["reason"] = "too_few_samples"
        elif not both:
            entry["reason"] = "no_prototypes"
        else:
            other = "right" if team == "left" else "left"
            own_d = float(np.linalg.norm(lab[i] - prototypes[team]))
            other_d = float(np.linalg.norm(lab[i] - prototypes[other]))
            entry["distance_own"] = _rounded(own_d)
            entry["distance_other"] = _rounded(other_d)
            entry["nearer_team"] = team if own_d <= other_d else other
            if against is not None:
                entry["samples_against"] = against.get((int(row.track_id), team), 0)
            if own_d - other_d > params.margin:
                entry["verdict"] = "inconsistent"
                entry["cleared"] = "inconsistent"
                inconsistent.add((int(row.track_id), team))
            else:
                entry["verdict"] = "consistent"
        entries.append(entry)

    out = tracks.copy()
    clear = np.zeros(len(out), dtype=bool)
    for track, team in inconsistent:
        clear |= player & (ids == track) & (labels == team)
    if clear.any():
        out.loc[clear, "team"] = _missing_value(out["team"])
    skipped = set(ids[ids >= 0].tolist()) - set(units["track_id"].tolist())
    counts = {verdict: sum(e["verdict"] == verdict for e in entries) for verdict in VERDICTS}
    report = {
        "params": params.as_dict(),
        "prototypes": report_prototypes,
        "prototype_distance": _rounded(gap),
        "tracks": entries,
        "counts": {
            "player_tracks": len(entries),
            **counts,
            "consistent_with_samples_against": None
            if against is None
            else sum(e["verdict"] == "consistent" and e["samples_against"] > 0 for e in entries),
            "rows_cleared": int(clear.sum()),
            "skipped_tracks": len(skipped),
        },
    }
    return out, report
