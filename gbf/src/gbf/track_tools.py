"""Clean-ups of the plugin's track table that the analysis relies on.

- :func:`split_tracks_at_cuts`: a tracker that does not know about cuts may give an id
  that was used before a cut to someone else after it (on the development clip two players
  of different teams in consecutive shots shared one id).
  Every (track id, shot) piece after the first one of an id gets a new id, so an id never
  spans a cut.
- :func:`split_tracks`: the same, and with the frame size also at a re-entry: within a
  shot, a track missing for at least :data:`REENTRY_MIN_GAP_S` whose box touches a frame
  edge just before or just after the gap left the picture (a player running out of it, or
  the camera panning away) and came back. While the player is out of the picture the
  tracker has no box to follow, so the id it gives on the way back comes from appearance
  alone, and players of one team look alike: the plugin may hand the returning player the
  id of another player (and with it that player's team and jersey), or the returning
  player's id to someone else. The piece after such a gap gets a new id, so its team and
  jersey are judged on its own rows (gbf.team_check, :func:`confident_jerseys`), never
  inherited across the gap.
- :func:`confident_jerseys`: the plugin writes a jersey guess on almost every row, mostly
  with confidence near 0. Only a number read with a mean confidence of at least
  ``min_mean_conf`` over at least ``min_reads`` rows of the same track, and agreed by at
  least ``min_agreement`` of those rows, is kept; every other row loses its jersey, and so
  does every row without a track (negative id: those boxes are not one person).
- :func:`same_spot`: two person boxes of one frame that cannot be two players standing apart
  (one person detected twice, or one box around two players that ends at the feet of one of
  them). gbf.measure's nearest opponent and gbf.render's line to it never take such a box.
- :func:`off_pitch`: a ground point beyond the pitch lines by more than
  :data:`OFF_PITCH_MARGIN_M` (a coach in the technical area, a steward): no player in play.
"""

from __future__ import annotations

import math
from collections.abc import Sequence, Set

import numpy as np
import pandas as pd

#: new ids for later pieces are old id + k * ID_STRIDE (k = 1, 2, ...)
ID_STRIDE = 100_000

REENTRY_MIN_GAP_S = 0.4
"""A track missing for at least this long within a shot, next to a frame edge, left the
picture and came back (:func:`split_tracks`): longer than the frame or two a detector drops
a person half cut off by the edge (10 frames at 25 fps, against the 3 frames gbf.kinematics
bridges), so the player was out of sight and the tracker's id on the way back rests on
appearance alone."""

SAME_SPOT_IOU = 0.75
"""Two person boxes that overlap more than this (intersection over union) are one detection
twice: two players overlap that much only when one stands right behind the other, and then
the detector sees one of them."""

MIN_PLAYER_GAP_M = 0.3
"""No two players' foot points (the bottom centres of their boxes) are closer on the grass:
a box whose foot point lies nearer to a player's is that player again, or a box around two
players that ends at that player's feet. A distance below it is never shown."""

SAME_SPOT_FOOT_BOX_H = 0.05
"""Without ground positions, foot points closer than this many mean box heights in the
image (9 cm at the players' own scale) are one person's, as :data:`MIN_PLAYER_GAP_M` on the
grass."""

OFF_PITCH_MARGIN_M = 0.5
"""A ground point more than this many metres beyond a touchline or a goal line is off the
pitch: the Laws of the Game end the technical area 1 m from the touchline, so a coach stands
at least twice this far out, while a player taking a throw-in stands on or just behind the
line."""

PITCH_HALF_LENGTH_M = 52.5
PITCH_HALF_WIDTH_M = 34.0
"""SoccerNet's 105 x 68 m pitch, origin at the centre spot."""

EDGE_TOUCH_PER_BOX_HEIGHT = 0.03
"""A box this close to a frame edge may have its feet or body cut off."""


class TrackToolsError(ValueError):
    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__(f"{where}: " + "; ".join(self.problems))


def _need(frame: pd.DataFrame, name: str, columns: Sequence[str], where: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise TrackToolsError(where, [f"{name} misses column(s) {missing}"])


SPLIT_REASONS: tuple[str, ...] = ("first", "cut", "reentry")
"""Why a piece of :func:`split_tracks` begins: the id's first piece, a cut, a re-entry."""


def split_tracks_at_cuts(
    tracks: pd.DataFrame, shots: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give every (track id, shot) piece after the first one of an id a new id.

    Returns the new table (same rows and order, ``track_id`` rewritten, the original in
    ``source_track_id``) and the mapping ``source_track_id, shot_id, track_id``. Negative
    ids (no track) are left alone.
    """
    out, mapping = split_tracks(tracks, shots, where="split_tracks_at_cuts")
    return out, mapping[["source_track_id", "shot_id", "track_id"]]


def boxes_touch_edge(boxes: np.ndarray, size: tuple[float, float]) -> np.ndarray:
    """:func:`box_touches_edge` of every box ``(n, 4)`` (x1, y1, x2, y2); False when a box
    is not finite."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    width, height = float(size[0]), float(size[1])
    finite = np.isfinite(boxes).all(axis=1)
    safe = np.where(finite[:, None], boxes, 0.0)
    edge = np.maximum(EDGE_TOUCH_PER_BOX_HEIGHT * (safe[:, 3] - safe[:, 1]), 2.0 * height / 1080)
    inner = np.minimum.reduce(
        [safe[:, 0], safe[:, 1], width - 0.5 - safe[:, 2], height - 0.5 - safe[:, 3]]
    )
    return finite & (inner <= edge)


def split_tracks(
    tracks: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    frame_size: tuple[float, float] | None = None,
    fps: float | None = None,
    where: str = "split_tracks",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give every piece of a track id after its first one a new id: a piece ends at a cut
    and, with ``frame_size`` (width, height of the boxes' pixel grid) and ``fps``, at a
    re-entry (module docstring; ``tracks`` then needs x1, y1, x2, y2).

    A re-entry is a gap of at least ``REENTRY_MIN_GAP_S * fps`` (rounded, at least 1)
    frames without a row of the id between two of its rows in one shot, where a box of the
    id on the frame before the gap or on the frame after it touches a frame edge
    (:func:`boxes_touch_edge`). A gap away from every edge (a player hidden behind another,
    missed by the detector) keeps the id.

    Returns the new table (same rows and order, ``track_id`` rewritten, the original in
    ``source_track_id``) and the mapping, one row per piece in id and frame order:
    ``source_track_id, shot_id, start_frame, end_frame, track_id, reason``
    (:data:`SPLIT_REASONS`; ``gap_frames``, the frames missing before a re-entry piece, 0
    otherwise). Piece k of an id (from 0, in frame order) gets ``id + k * ID_STRIDE``.
    Negative ids (no track) are left alone.
    """
    _need(tracks, "tracks", ("frame", "track_id"), where)
    _need(shots, "shots", ("shot_id", "start_frame", "end_frame"), where)
    reentry = frame_size is not None
    if reentry:
        _need(tracks, "tracks", ("x1", "y1", "x2", "y2"), where)
        if fps is None or not math.isfinite(float(fps)) or float(fps) <= 0:
            raise TrackToolsError(where, [f"fps must be a positive number, got {fps!r}"])
    starts = shots.sort_values("start_frame")
    frames = tracks["frame"].to_numpy(dtype=np.int64)
    index = np.searchsorted(starts["start_frame"].to_numpy(), frames, side="right") - 1
    if (index < 0).any() or (frames > starts["end_frame"].to_numpy()[index]).any():
        raise TrackToolsError(where, ["some track rows lie outside every shot"])
    shot_of = starts["shot_id"].to_numpy()[index]
    ids = tracks["track_id"].to_numpy(dtype=np.int64)
    if ids.max(initial=0) >= ID_STRIDE:
        raise TrackToolsError(where, [f"track ids must be < {ID_STRIDE}"])

    rows = pd.DataFrame(
        {"pos": np.arange(len(ids)), "source_track_id": ids, "shot_id": shot_of, "frame": frames}
    )
    if reentry:
        boxes = tracks[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
        rows["edge"] = boxes_touch_edge(boxes, frame_size)
    else:
        rows["edge"] = False
    rows = rows[rows["source_track_id"] >= 0]
    # One entry per (id, frame): its shot, and whether any of its boxes there touches an edge.
    per_frame = (
        rows.groupby(["source_track_id", "frame"], as_index=False, sort=True)
        .agg(shot_id=("shot_id", "first"), edge=("edge", "any"))
        .reset_index(drop=True)
    )
    same_id = per_frame["source_track_id"].eq(per_frame["source_track_id"].shift())
    new_shot = per_frame["shot_id"].ne(per_frame["shot_id"].shift())
    gap = (per_frame["frame"] - per_frame["frame"].shift() - 1).fillna(0).astype(np.int64)
    reason = np.where(~same_id, "first", np.where(new_shot, "cut", ""))
    if reentry:
        min_gap = max(1, round(REENTRY_MIN_GAP_S * float(fps)))
        at_edge = per_frame["edge"] | per_frame["edge"].shift(fill_value=False)
        back = same_id & ~new_shot & (gap >= min_gap) & at_edge
        reason = np.where(back, "reentry", reason)
    per_frame["reason"] = reason
    per_frame["gap_frames"] = np.where(reason == "reentry", gap, 0)
    per_frame["piece"] = (per_frame["reason"] != "").cumsum()

    pieces = per_frame.groupby("piece", sort=True).agg(
        source_track_id=("source_track_id", "first"),
        shot_id=("shot_id", "first"),
        start_frame=("frame", "min"),
        end_frame=("frame", "max"),
        reason=("reason", "first"),
        gap_frames=("gap_frames", "first"),
    )
    k = pieces.groupby("source_track_id").cumcount().to_numpy(dtype=np.int64)
    pieces["track_id"] = pieces["source_track_id"].to_numpy(dtype=np.int64) + k * ID_STRIDE
    piece_of = per_frame.set_index(["source_track_id", "frame"])["piece"]
    keys = pd.MultiIndex.from_arrays([rows["source_track_id"], rows["frame"]])
    new_ids = ids.copy()
    new_ids[rows["pos"].to_numpy()] = (
        pieces["track_id"].reindex(piece_of.reindex(keys).to_numpy()).to_numpy(dtype=np.int64)
    )
    mapping = pieces[
        ["source_track_id", "shot_id", "start_frame", "end_frame", "track_id", "reason"]
    ].reset_index(drop=True)
    mapping["gap_frames"] = pieces["gap_frames"].to_numpy(dtype=np.int64)
    for name in ("source_track_id", "shot_id", "start_frame", "end_frame", "track_id"):
        mapping[name] = mapping[name].astype(np.int64)
    out = tracks.copy()
    out["source_track_id"] = ids
    out["track_id"] = new_ids
    return out, mapping


def confident_jerseys(
    tracks: pd.DataFrame,
    *,
    min_mean_conf: float = 0.8,
    min_reads: int = 5,
    min_agreement: float = 0.7,
) -> pd.DataFrame:
    """Keep a track's jersey only when it is read confidently and consistently; rows
    without a track (negative id) never keep one."""
    _need(tracks, "tracks", ("track_id", "jersey", "jersey_conf"), "confident_jerseys")
    out = tracks.copy()
    jersey = out["jersey"].astype(object).where(out["jersey"].notna(), None)
    conf = out["jersey_conf"].to_numpy(dtype=float)
    keep: dict[int, str] = {}
    for track, rows in out.groupby("track_id").indices.items():
        if int(track) < 0:
            continue
        values = [(jersey.iloc[i], conf[i]) for i in rows if jersey.iloc[i] not in (None, "")]
        if len(values) < min_reads:
            continue
        numbers = pd.Series([v for v, _ in values])
        top = numbers.value_counts()
        best = top.index[0]
        agreement = top.iloc[0] / len(values)
        mean_conf = float(np.nanmean([c for v, c in values if v == best]))
        if agreement >= min_agreement and mean_conf >= min_mean_conf:
            keep[int(track)] = str(best)
    out["jersey"] = [keep.get(int(t)) for t in out["track_id"]]
    out.loc[out["jersey"].isna(), "jersey_conf"] = np.nan
    return out


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two boxes (x1, y1, x2, y2); 0 when neither has an area."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, width) * max(0.0, height)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_spot(
    box_a: Sequence[float],
    box_b: Sequence[float],
    ground_a: Sequence[float] | None = None,
    ground_b: Sequence[float] | None = None,
) -> bool:
    """True when ``box_b`` holds no second person standing apart from the person of
    ``box_a``: the boxes overlap more than :data:`SAME_SPOT_IOU`, or the foot points (bottom
    centres) lie less than :data:`MIN_PLAYER_GAP_M` apart on the grass (both ground positions
    given, SoccerNet metres), else less than :data:`SAME_SPOT_FOOT_BOX_H` mean box heights
    apart in the image. A box around two players that ends at the feet of one of them (the
    detector merging a player with the one behind) is such a box: its foot point is that
    player's own, while the boxes overlap less."""
    if box_iou(box_a, box_b) > SAME_SPOT_IOU:
        return True
    if ground_a is not None and ground_b is not None:
        gap = math.hypot(ground_a[0] - ground_b[0], ground_a[1] - ground_b[1])
        return gap < MIN_PLAYER_GAP_M
    feet = math.hypot((box_a[0] + box_a[2] - box_b[0] - box_b[2]) / 2, box_a[3] - box_b[3])
    height = (box_a[3] - box_a[1] + box_b[3] - box_b[1]) / 2
    return height > 0 and feet < SAME_SPOT_FOOT_BOX_H * height


def off_pitch(ground: Sequence[float] | None) -> bool:
    """True when the ground point (SoccerNet metres) lies more than :data:`OFF_PITCH_MARGIN_M`
    beyond a touchline or a goal line; False without one."""
    if ground is None:
        return False
    x, y = float(ground[0]), float(ground[1])
    return (
        abs(x) > PITCH_HALF_LENGTH_M + OFF_PITCH_MARGIN_M
        or abs(y) > PITCH_HALF_WIDTH_M + OFF_PITCH_MARGIN_M
    )


def box_touches_edge(box: Sequence[float], size: tuple[int, int] | None) -> bool:
    """Whether any frame edge cuts a box (with a small height-scaled tolerance)."""
    if size is None:
        return False
    return bool(boxes_touch_edge(np.asarray([box], dtype=np.float64), size)[0])


def speed_chord_safe(frame: int, track: int, unsafe: Set[tuple[int, int]], half: int) -> bool:
    """No explicitly unsafe observed person at a speed centre or either chord end.

    A missing row alone stays bridgeable: absence supplies no unsafe-person evidence.
    """
    return not any((frame + delta, track) in unsafe for delta in (-half, 0, half))


def off_pitch_at_camera_edge(
    box: Sequence[float],
    ground: Sequence[float] | None,
    size: tuple[int, int] | None,
    camera_centre: Sequence[float] | None,
) -> bool:
    """A point beyond the pitch margin, or cut-off feet beyond the camera-side line.

    At the bottom edge the ground point is the edge's, while the actual feet lie nearer
    the camera, still farther outside. A complete box within the margin stays eligible.
    """
    if off_pitch(ground):
        return True
    if ground is None or size is None or camera_centre is None:
        return False
    _x1, y1, _x2, y2 = box
    edge = max(EDGE_TOUCH_PER_BOX_HEIGHT * (y2 - y1), 2.0 * size[1] / 1080.0)
    if y2 < size[1] - 0.5 - edge:
        return False
    x, y = ground
    cx, cy = camera_centre[:2]
    return (abs(y) > PITCH_HALF_WIDTH_M and abs(cy) > PITCH_HALF_WIDTH_M and y * cy > 0) or (
        abs(x) > PITCH_HALF_LENGTH_M and abs(cx) > PITCH_HALF_LENGTH_M and x * cx > 0
    )
