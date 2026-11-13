"""Clean-ups of the plugin's track table that the analysis relies on.

- :func:`split_tracks_at_cuts`: a tracker that does not know about cuts may give an id
  that was used before a cut to someone else after it (on the development clip two players
  of different teams in consecutive shots shared one id).
  Every (track id, shot) piece after the first one of an id gets a new id, so an id never
  spans a cut.
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


def split_tracks_at_cuts(
    tracks: pd.DataFrame, shots: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give every (track id, shot) piece after the first one of an id a new id.

    Returns the new table (same rows and order, ``track_id`` rewritten, the original in
    ``source_track_id``) and the mapping ``source_track_id, shot_id, track_id``. Negative
    ids (no track) are left alone.
    """
    _need(tracks, "tracks", ("frame", "track_id"), "split_tracks_at_cuts")
    _need(shots, "shots", ("shot_id", "start_frame", "end_frame"), "split_tracks_at_cuts")
    starts = shots.sort_values("start_frame")
    frames = tracks["frame"].to_numpy(dtype=np.int64)
    index = np.searchsorted(starts["start_frame"].to_numpy(), frames, side="right") - 1
    if (index < 0).any() or (frames > starts["end_frame"].to_numpy()[index]).any():
        raise TrackToolsError("split_tracks_at_cuts", ["some track rows lie outside every shot"])
    shot_of = starts["shot_id"].to_numpy()[index]
    ids = tracks["track_id"].to_numpy(dtype=np.int64)
    if ids.max(initial=0) >= ID_STRIDE:
        raise TrackToolsError("split_tracks_at_cuts", [f"track ids must be < {ID_STRIDE}"])
    pieces = (
        pd.DataFrame({"source_track_id": ids, "shot_id": shot_of, "frame": frames})
        .loc[lambda d: d["source_track_id"] >= 0]
        .groupby(["source_track_id", "shot_id"], as_index=False)["frame"]
        .min()
        .sort_values(["source_track_id", "frame"], kind="stable")
    )
    pieces["k"] = pieces.groupby("source_track_id").cumcount()
    pieces["track_id"] = pieces["source_track_id"] + pieces["k"] * ID_STRIDE
    mapping = pieces[["source_track_id", "shot_id", "track_id"]].reset_index(drop=True)
    lookup = {(int(a), int(s)): int(t) for a, s, t in mapping.itertuples(index=False, name=None)}
    new_ids = np.array(
        [lookup.get((int(t), int(s)), int(t)) for t, s in zip(ids, shot_of, strict=True)],
        dtype=np.int64,
    )
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
    width, height = size
    x1, y1, x2, y2 = box
    edge = max(EDGE_TOUCH_PER_BOX_HEIGHT * (y2 - y1), 2.0 * height / 1080.0)
    return min(x1, y1, width - 0.5 - x2, height - 0.5 - y2) <= edge


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
