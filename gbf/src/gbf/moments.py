"""Frame-precise key moments inside a highlight, from 2D pose on the original frames.

These measurements live in the image plane: they need no calibration, so they also work on
the low-angle and close-up shots where most duels are shown. Angles are measured against
the image vertical, which is close to the true vertical for a level broadcast camera; the
report says "画面平面" for every value from here.

- :func:`body_angles`: per pose row, the trunk angle from the image vertical (the line from
  the hip midpoint to the shoulder midpoint, :func:`gbf.lift3d.image_trunk_lean_deg`, the
  angle the report shows as 躯干倾斜（画面）, unsigned), the hip height above the ankles in
  torso lengths, and the pose quality.
- :func:`fall_events`: a player going to ground: the trunk angle rises from upright to at
  least :data:`FALL_DOWN_DEG` within :data:`FALL_MAX_S` and the hips drop toward the
  ankles. ``onset_frame`` is the last upright frame before the fall. The text says what
  this measures (the trunk starts to tilt, then passes 45° in the image), not that the
  player lost balance or reached the ground.
- :func:`contact_series`: per frame, the smallest distance between the keypoints of two
  players in body heights; :func:`contact_intervals` turns it into contact phases. A
  contact phase is this closeness in the image, not a measured touch: the text says the
  two are close (身体贴近（画面平面）), never that they touched.
- :func:`duel_timeline`: the ordered key moments of a duel (start of the longest contact
  phase, touches, the ball leaving the pair's feet, fall onset, the trunk passing 45°
  (``on_ground``), end of the longest contact phase), each with its frame and a short
  Chinese sentence.
- :func:`confirmed_touches`: the ball events gbf.measure matches to a foot keypoint (the
  criterion of the measurement table's contact frames); the only touches counted here.
- :func:`arm_gestures` and :func:`reaction_timeline`: in a close-up, the subject raising a
  hand above the shoulder, with a straight or a bent arm, and the frame the hand is
  highest; whether the arm is straight or bent on that frame is read from its elbow there.

:func:`add_key_moments` writes these fields into every highlight of highlights.json:

- ``key_moments``: frame, time, kind (:data:`MOMENT_KINDS`), track_ids, text_zh and
  evidence of each moment, in frame order;
- ``touches``: per player of ``key_pair``, the touches :func:`confirmed_touches` confirms
  inside the highlight's final window (``start_frame``..``end_frame`` after any change of
  the window): track_id, count, frames, mean_interval_s, label_zh. Given the same ball
  events, tracks and pose, a count is the number of gbf.measure's ``foot_keypoint``
  ``ball-contact-frame`` records of that player in that highlight, and the ``touch``
  moments are the same touches. A soft touch that barely changes the ball's velocity is
  missed, so a count is a lower bound;
- ``key_pair``: ``[a, b]``, the duel pair whose :func:`duel_timeline` is in
  ``key_moments``, or null;
- ``key_focus_track``: the player the key frame (``peak_frame``) is about, or null: the
  subject of a close-up reaction whose peak moved to a gesture; the player of the primary
  moment of a highlight that became a duel (the faller of ``fall_onset``, the nearest
  player of ``ball_release``); the first of ``key_pair`` for any other ``duel_tackle``.
  Never a player without a track row (without the track table: a pose row) at
  ``peak_frame``: when the pair is a joined shot's, say, whose track ids do not exist in
  the shot of the key frame gbf.highlights chose (track ids are split at cuts), it stays
  null rather than name a player who is not on the key frame. Null for the other
  highlights: their key frame is still the one gbf.highlights chose, about the players it
  lists first;
- ``previous_peak``, only when the peak moved: the peak frame, type, refinement, title and
  window (start_frame, end_frame) before the move. A close-up reaction's window then starts
  at most :data:`REACTION_LEAD_S` before the cut into the close-up; merged parts that lie
  wholly outside the new window leave ``merged_from`` and ``merged_parts`` (both keys go
  when fewer than two parts are left).

Titles and summaries name a shot by the size of its players (gbf.highlights.shot_name_zh,
from the shot table), and say :data:`SHOT_UNKNOWN_ZH` without one. A title is about its key
frame: a moment it names on another frame is said to come later (随后) or earlier (此前),
with its frame. Titles name players by team (左侧球队球员); key-moment texts and summaries
add the track id (gbf.highlights.player_name_zh), and every summary ends with the players
involved: the highlight's ``track_ids``, to which the players of ``key_pair`` and of the
key moments are added when missing.

Thresholds are starting values checked on one development clip; they are not validated.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd

from gbf.highlights import format_time, player_names_zh, shot_name_zh
from gbf.lift3d import image_trunk_lean_deg
from gbf.measure import (
    DEFAULT_PARAMS,
    UPRIGHT_DEG,
    ContactMatch,
    MeasureParams,
    MeasureTables,
    match_contacts,
)
from gbf.pose import HALPE26_INDEX, keypoints_array

#: keypoints below this score are ignored
MIN_SCORE = 0.3
#: a fall reaches at least this trunk angle ...
FALL_DOWN_DEG = 45.0
#: ... within this time from the last upright frame
FALL_MAX_S = 1.0
#: and the hips end up at most this many torso lengths above the ankles
FALL_HIP_RATIO = 0.8
#: two players are in contact when keypoints come closer than this many body heights in the
#: image (close together; whether they touch is not measured)
CONTACT_BODY_H = 0.12
#: contact phases separated by fewer frames than this are merged
CONTACT_GAP_FRAMES = 3
#: below this median body height (px, head to ankle) the image plane cannot tell contact
#: from one player standing in front of another, nor a fall from a crouch: only touches
#: and the ball leaving the feet (``ball_away``) are reported (in wide shots players
#: overlap in the image without touching)
MIN_BODY_PX = 150.0
#: the line whose angle from the image vertical is the trunk angle of a fall (the evidence
#: of fall_onset and on_ground names it; gbf.lift3d.image_trunk_lean_deg)
TRUNK_LINE = "hip_midpoint_to_shoulder_midpoint"


class MomentsInputError(ValueError):
    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__(f"{where}: " + "; ".join(self.problems))


def _midpoint(k: np.ndarray, a: str, b: str) -> np.ndarray:
    pa, pb = k[:, HALPE26_INDEX[a]], k[:, HALPE26_INDEX[b]]
    ok = (pa[:, 2] >= MIN_SCORE) & (pb[:, 2] >= MIN_SCORE)
    mid = (pa[:, :2] + pb[:, :2]) / 2.0
    mid[~ok] = np.nan
    return mid


def body_angles(pose: pd.DataFrame) -> pd.DataFrame:
    """Per pose row: frame, track_id, trunk_deg (0 = upright in the image; the absolute
    value of :func:`gbf.lift3d.image_trunk_lean_deg`), hip_ratio (ankle-to-hip height over
    torso length; about 1.2-1.6 standing), body_h_px (head to ankle) and mean_score. Values
    are NaN when the needed keypoints are below MIN_SCORE."""
    missing = [c for c in ("frame", "track_id", "keypoints") if c not in pose.columns]
    if missing:
        raise MomentsInputError("body_angles", [f"pose misses column(s) {missing}"])
    k = keypoints_array(pose) if len(pose) else np.zeros((0, 26, 3))
    hip = _midpoint(k, "left_hip", "right_hip")
    shoulder = _midpoint(k, "left_shoulder", "right_shoulder")
    ankle = _midpoint(k, "left_ankle", "right_ankle")
    trunk = shoulder - hip
    torso = np.hypot(trunk[:, 0], trunk[:, 1])
    angle = np.abs(image_trunk_lean_deg(k, MIN_SCORE))
    head = k[:, HALPE26_INDEX["head"]] if len(k) else np.zeros((0, 3))
    top = np.where((head[:, 2] >= MIN_SCORE)[:, None], head[:, :2], shoulder)
    body_h = ankle[:, 1] - top[:, 1]
    return (
        pd.DataFrame(
            {
                "frame": pose["frame"].to_numpy(dtype=np.int64),
                "track_id": pose["track_id"].to_numpy(dtype=np.int64),
                "trunk_deg": angle,
                "hip_ratio": (ankle[:, 1] - hip[:, 1]) / np.where(torso > 1e-6, torso, np.nan),
                "body_h_px": body_h,
                "mean_score": (
                    pose["mean_score"].to_numpy(dtype=float)
                    if "mean_score" in pose
                    else k[:, :, 2].mean(axis=1)
                ),
            }
        )
        .sort_values(["track_id", "frame"], kind="stable")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class FallEvent:
    track_id: int
    onset_frame: int  # last upright frame before the fall
    down_frame: int  # first frame with trunk >= FALL_DOWN_DEG
    max_trunk_deg: float
    min_hip_ratio: float
    evidence: dict[str, Any] = field(default_factory=dict)


def fall_events(pose: pd.DataFrame, fps: int | float | Fraction | str) -> list[FallEvent]:
    """Players going to ground; see the module docstring."""
    rate = float(Fraction(str(fps)))
    max_frames = max(1, int(round(FALL_MAX_S * rate)))
    angles = body_angles(pose)
    events: list[FallEvent] = []
    for track_id, rows in angles.groupby("track_id", sort=True):
        frames = rows["frame"].to_numpy()
        trunk = rows["trunk_deg"].to_numpy()
        hip = rows["hip_ratio"].to_numpy()
        i = 0
        while i < len(frames):
            if not (np.isfinite(trunk[i]) and trunk[i] >= FALL_DOWN_DEG):
                i += 1
                continue
            # Walk back to the last upright frame within the time limit.
            onset = None
            for j in range(i - 1, -1, -1):
                if frames[i] - frames[j] > max_frames:
                    break
                if np.isfinite(trunk[j]) and trunk[j] <= UPRIGHT_DEG:
                    onset = j
                    break
            end = i
            while (
                end + 1 < len(frames)
                and frames[end + 1] == frames[end] + 1
                and (not np.isfinite(trunk[end + 1]) or trunk[end + 1] >= UPRIGHT_DEG)
            ):
                end += 1
            window = slice(i, end + 1)
            low_hip = np.nanmin(hip[window]) if np.isfinite(hip[window]).any() else np.nan
            if onset is not None and np.isfinite(low_hip) and low_hip <= FALL_HIP_RATIO:
                events.append(
                    FallEvent(
                        track_id=int(track_id),
                        onset_frame=int(frames[onset]),
                        down_frame=int(frames[i]),
                        max_trunk_deg=float(np.nanmax(trunk[window])),
                        min_hip_ratio=float(low_hip),
                        evidence={
                            "trunk_deg": {
                                int(f): round(float(a), 1)
                                for f, a in zip(
                                    frames[onset : end + 1], trunk[onset : end + 1], strict=True
                                )
                                if np.isfinite(a)
                            }
                        },
                    )
                )
            i = end + 1
    return events


def contact_series(pose: pd.DataFrame, track_a: int, track_b: int) -> pd.DataFrame:
    """Per frame where both players have a pose: the smallest keypoint distance between
    them in body heights (mean of the two), and the pair of keypoints that are closest."""
    rows = []
    a = pose[pose["track_id"] == track_a].set_index("frame")
    b = pose[pose["track_id"] == track_b].set_index("frame")
    names = list(HALPE26_INDEX)
    for frame in sorted(set(a.index) & set(b.index)):
        ka = np.asarray(a.loc[frame, "keypoints"], dtype=float).reshape(26, 3)
        kb = np.asarray(b.loc[frame, "keypoints"], dtype=float).reshape(26, 3)
        ma, mb = ka[:, 2] >= MIN_SCORE, kb[:, 2] >= MIN_SCORE
        if not ma.any() or not mb.any():
            continue
        d = np.linalg.norm(ka[ma, None, :2] - kb[None, mb, :2], axis=-1)
        ia, ib = np.unravel_index(np.argmin(d), d.shape)
        heights = [float(np.nanmax(k[m, 1]) - np.nanmin(k[m, 1])) for k, m in ((ka, ma), (kb, mb))]
        body = max(1.0, float(np.mean(heights)))
        rows.append(
            {
                "frame": int(frame),
                "distance_body_h": float(d[ia, ib]) / body,
                "keypoint_a": names[np.flatnonzero(ma)[ia]],
                "keypoint_b": names[np.flatnonzero(mb)[ib]],
            }
        )
    return pd.DataFrame(rows, columns=["frame", "distance_body_h", "keypoint_a", "keypoint_b"])


def contact_intervals(
    series: pd.DataFrame, *, threshold: float = CONTACT_BODY_H, gap_frames: int = CONTACT_GAP_FRAMES
) -> list[tuple[int, int]]:
    """Inclusive frame intervals where the players are in contact (gaps merged)."""
    frames = series.loc[series["distance_body_h"] <= threshold, "frame"].to_numpy()
    if not len(frames):
        return []
    intervals = [[int(frames[0]), int(frames[0])]]
    for f in frames[1:]:
        if f - intervals[-1][1] <= gap_frames:
            intervals[-1][1] = int(f)
        else:
            intervals.append([int(f), int(f)])
    return [(a, b) for a, b in intervals]


#: key-moment kinds, in the order the moments of one frame are listed: duels
#: (``ball_release`` is the ball leaving the pair's feet after the two came close in the
#: image, a contact phase; ``ball_away`` the ball leaving a player's feet without one), then
#: close-up reactions
MOMENT_KINDS = (
    "contact_start",
    "touch",
    "ball_release",
    "ball_away",
    "fall_onset",
    "on_ground",
    "contact_end",
    "hand_up",
    "arm_top",
    "hand_top",
)
#: the kinds read from the ball (the others come from pose alone)
BALL_MOMENT_KINDS = ("touch", "ball_release", "ball_away")
#: how the touches are found, said with every count of them
TOUCHES_NOTE_ZH = "球速突变且脚在球边；轻触可能漏检"


@dataclass(frozen=True)
class Moment:
    frame: int
    kind: str  # one of MOMENT_KINDS
    track_ids: tuple[int, ...]
    text_zh: str
    evidence: dict[str, Any] = field(default_factory=dict)


def _in_order(moments: Sequence[Moment]) -> list[Moment]:
    return sorted(moments, key=lambda m: (m.frame, MOMENT_KINDS.index(m.kind)))


def _rounded(value: Any, digits: int) -> float | None:
    number = float(value) if value is not None else math.nan
    return round(number, digits) if math.isfinite(number) else None


#: the ball is at a player's feet when it is at most this many body heights from an ankle or
#: a big toe ...
RELEASE_NEAR_BODY_H = 0.15
#: ... and it has left them fast when the first detection more than RELEASE_AWAY_BODY_H body
#: heights away comes at most RELEASE_WITHIN_S after the last such frame, and the ball's
#: distance from the feet grew by at least RELEASE_MIN_SPEED_BH_S body heights per second on
#: average between the two frames (a ball seen again only later, or rolling off slowly, has
#: not; one detection is enough, so a sparse ball track still shows a ball kicked away)
RELEASE_AWAY_BODY_H = 0.3
RELEASE_WITHIN_S = 0.2
RELEASE_MIN_SPEED_BH_S = 2.5


def _ball_release(
    ball_track: pd.DataFrame | None,
    pose: pd.DataFrame,
    pair: tuple[int, int],
    start: int,
    end: int,
    fps: int | float | Fraction | str,
) -> tuple[int, dict[str, Any]] | None:
    """The last frame the ball is at the pair's feet, when it leaves them fast afterwards
    (see :data:`RELEASE_AWAY_BODY_H`), with the nearest player, the first frame it is away,
    its mean speed away from the feet until then and the distances as evidence; None
    otherwise."""
    if ball_track is None or not len(ball_track):
        return None
    rate = float(Fraction(str(fps)))
    within = max(1, round(RELEASE_WITHIN_S * rate))
    ball = ball_track[(ball_track["frame"] >= start) & (ball_track["frame"] <= end)]
    ball = ball[ball["status"] == "detected"].set_index("frame")
    feet: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for frame, rows in pose[pose["track_id"].isin(pair)].groupby("frame"):
        k = keypoints_array(rows)
        for track, one in zip(rows["track_id"].to_numpy(), k, strict=True):
            ok = one[:, 2] >= MIN_SCORE
            pts = [
                one[HALPE26_INDEX[name], :2]
                for name in ("left_ankle", "right_ankle", "left_big_toe", "right_big_toe")
                if one[HALPE26_INDEX[name], 2] >= MIN_SCORE
            ]
            if ok.any() and pts:
                body = float(np.nanmax(one[ok, 1]) - np.nanmin(one[ok, 1]))
                feet.setdefault(int(frame), []).append((int(track), np.asarray(pts), body))
    near = []
    for frame in sorted(feet):
        if frame not in ball.index:
            continue
        uv = ball.loc[frame, ["u", "v"]].to_numpy(dtype=float)
        best = min(
            (float(np.min(np.linalg.norm(pts - uv, axis=1))) / max(body, 1.0), track)
            for track, pts, body in feet[frame]
        )
        near.append((frame, best[0], best[1]))
    at_feet = [(f, d, t) for f, d, t in near if d <= RELEASE_NEAR_BODY_H]
    if not at_feet:
        return None
    last_near, near_dist, nearest = at_feet[-1]
    away = next(
        (
            (f, d)
            for f, d, _ in near
            if last_near < f <= last_near + within and d > RELEASE_AWAY_BODY_H
        ),
        None,
    )
    if away is None:
        return None
    speed = (away[1] - near_dist) * rate / (away[0] - last_near)
    if speed < RELEASE_MIN_SPEED_BH_S:
        return None
    return last_near, {
        "nearest_track_id": nearest,
        "away_frame": away[0],
        "away_speed_bh_s": round(speed, 2),
        "ball_to_feet_body_h": {
            int(f): round(d, 3)
            for f, d, _ in near
            if last_near - 3 <= f <= max(away[0], last_near + 3)
        },
    }


def confirmed_touches(
    events: pd.DataFrame | None,
    tracks: pd.DataFrame | None,
    pose: pd.DataFrame | None,
    params: MeasureParams = DEFAULT_PARAMS,
) -> list[ContactMatch]:
    """Ball events that touch a player's foot by gbf.measure's criterion, in frame order.

    :func:`gbf.measure.match_contacts` (the ``ball-contact-frame`` records of the
    measurement table) matches a ball event (an abrupt change of the ball's image velocity,
    gbf.ball) to the candidate whose confident foot keypoint (ankle, toe or heel) is
    nearest the ball centre, within ``params.contact_radius_bh`` of that player's box
    height. Only these keypoint matches are kept: its fallback to a box's foot point, for a
    player without a usable pose, confirms nothing. A soft touch that changes the ball's
    velocity too little to be an event is missed, so counts are lower bounds.

    ``events`` are gbf.ball events of one or several profiles (MeasureTables.ball_events),
    ``tracks`` the track table (MeasureTables.tracks columns) and ``pose`` has frame,
    track_id, keypoints and mean_score. Events are matched frame by frame, so one shot
    spanning every frame stands in for the shot table that MeasureTables needs.
    """
    if events is None or not len(events) or pose is None or not len(pose):
        return []
    if tracks is None:
        raise MomentsInputError("confirmed_touches", ["touches need the track table (boxes)"])
    last = max(int(events["frame"].max()), int(pose["frame"].max()))
    shots = pd.DataFrame({"shot_id": [0], "start_frame": [0], "end_frame": [last]})
    tables = MeasureTables(
        tracks=tracks, shots=shots, ball_events=events, pose=pose, pitch_frames=frozenset()
    )
    matches, _examined = match_contacts(tables, 0, last, params)
    return [m for m in matches if m.method == "foot_keypoint"]


def _touch_moments(
    touches: Sequence[ContactMatch] | None,
    pair: tuple[int, int],
    start: int,
    end: int,
    label: Mapping[int, str],
) -> list[Moment]:
    """``touch`` moments of the confirmed touches (:func:`confirmed_touches`) of either
    player of ``pair`` inside frames ``start..end``."""
    moments = []
    for touch in touches or []:
        if touch.method != "foot_keypoint" or touch.track_id not in pair:
            continue
        if not start <= touch.frame <= end:
            continue
        moments.append(
            Moment(
                touch.frame,
                "touch",
                (touch.track_id,),
                f"{label[touch.track_id]}触球",
                {
                    "kind": str(touch.event.get("kind")),
                    "magnitude_bh_s": _rounded(touch.event.get("magnitude"), 2),
                    "keypoint": touch.keypoint,
                    "distance_box_h": round(touch.distance_box_h, 3),
                },
            )
        )
    return moments


def duel_timeline(
    pose: pd.DataFrame,
    pair: tuple[int, int],
    start: int,
    end: int,
    fps: int | float | Fraction | str,
    *,
    ball_track: pd.DataFrame | None = None,
    names: Mapping[int, str] | None = None,
    touches: Sequence[ContactMatch] | None = None,
) -> list[Moment]:
    """Key moments of a duel between ``pair`` inside frames ``start..end`` (inclusive).

    ``touches`` are confirmed ball contacts (:func:`confirmed_touches`): the ones of either
    player inside ``start..end`` become ``touch`` moments (:func:`add_key_moments` passes
    none and adds them over the whole highlight window instead). The ball leaving the
    pair's feet is ``ball_release`` when the two came close (a contact phase), else
    ``ball_away``. A fall's ``fall_onset`` names the frame of its ``on_ground`` in its
    evidence (``down_frame``).
    """
    names = dict(names or {})
    label = {t: names.get(t, f"轨迹 {t}") for t in pair}
    window = pose[(pose["frame"] >= start) & (pose["frame"] <= end)]
    moments = _touch_moments(touches, pair, start, end, label)
    body = body_angles(window[window["track_id"].isin(pair)]) if len(window) else None
    large = body is not None and float(np.nanmedian(body["body_h_px"])) >= MIN_BODY_PX
    series = contact_series(window, *pair)
    phases = contact_intervals(series) if large else []
    if phases:
        longest = max(phases, key=lambda ab: ab[1] - ab[0])
        moments.append(
            Moment(
                longest[0],
                "contact_start",
                pair,
                f"{label[pair[0]]}与{label[pair[1]]}身体贴近（画面平面）",
                {"contact_phases": phases},
            )
        )
        last_pose = int(window["frame"].max())
        if longest[1] < last_pose - CONTACT_GAP_FRAMES:  # else the shot ended, not the contact
            moments.append(
                Moment(longest[1], "contact_end", pair, "身体贴近结束", {"contact_phases": phases})
            )
    release = _ball_release(ball_track, window, pair, start, end, fps)
    if release is not None:
        frame, evidence = release
        if phases:  # the players were close: the ball left the duel
            moments.append(Moment(frame, "ball_release", pair, "球离开两人脚下", evidence))
        else:  # a pass or a clearance, say: no duel
            nearest = int(evidence["nearest_track_id"])
            text = f"球从{label[nearest]}脚下快速离开"
            moments.append(Moment(frame, "ball_away", (nearest,), text, evidence))
    for event in fall_events(window, fps) if large else []:
        if event.track_id not in pair:
            continue
        who = label[event.track_id]
        moments.append(
            Moment(
                event.onset_frame,
                "fall_onset",
                (event.track_id,),
                f"{who}躯干开始倾倒",
                {**event.evidence, "trunk_line": TRUNK_LINE, "down_frame": event.down_frame},
            )
        )
        moments.append(
            Moment(
                event.down_frame,
                "on_ground",
                (event.track_id,),
                f"{who}躯干倾倒超过 {FALL_DOWN_DEG:.0f}°（画面平面）",
                {
                    "max_trunk_deg": round(event.max_trunk_deg, 1),
                    "min_hip_ratio": round(event.min_hip_ratio, 2),
                    "trunk_line": TRUNK_LINE,
                },
            )
        )
    return _in_order(moments)


def touch_summary(moments: Sequence[Moment], fps: int | float | Fraction | str) -> list[dict]:
    """Per player: number of detected touches (a lower bound, see :func:`confirmed_touches`),
    their frames and the mean interval between them in seconds."""
    rate = float(Fraction(str(fps)))
    by_player: dict[int, list[int]] = {}
    for m in moments:
        if m.kind == "touch":
            by_player.setdefault(m.track_ids[0], []).append(m.frame)
    result = []
    for track, frames in sorted(by_player.items()):
        frames = sorted(frames)
        gaps = np.diff(frames) / rate if len(frames) > 1 else np.zeros(0)
        result.append(
            {
                "track_id": track,
                "count": len(frames),
                "frames": frames,
                "mean_interval_s": round(float(gaps.mean()), 3) if gaps.size else None,
            }
        )
    return result


# ---------------------------------------------------------------------------------------
# Reactions in a close-up: the subject's raised hands.

#: a hand is raised when the wrist is at least this many shoulder widths above its own
#: shoulder (image plane) ...
RAISE_SHOULDER_W = 0.25
#: ... and the arm is straight when the elbow angle is at least this
ARM_STRAIGHT_DEG = 140.0
#: a gesture lasts at least this long, and a straight-arm gesture has at least as many
#: straight frames (a brief left/right swap of the arm keypoints can look like a straight
#: arm)
GESTURE_MIN_S = 0.3
#: raised frames at most this far apart belong to one gesture (a hand dipping just below the
#: threshold for a few frames is the same gesture)
GESTURE_GAP_S = 0.2
#: gestures are read only when the player's median shoulder width is at least this (px):
#: a close-up or a near medium shot (about a whole body of MIN_BODY_PX), never a wide shot
GESTURE_MIN_SHOULDER_PX = 35.0


@dataclass(frozen=True)
class Gesture:
    track_id: int
    start_frame: int
    end_frame: int
    top_frame: int  # the frame the wrist is highest above its shoulder
    max_raise: float  # shoulder widths
    straight: bool
    arm: str  # the person's "left" or "right" arm at top_frame
    evidence: dict[str, Any] = field(default_factory=dict)


def _arm_raise(k: np.ndarray, side: str, width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: how far the wrist is above the shoulder (shoulder widths) and the elbow
    angle (degrees, 180 = straight); NaN where a keypoint is below MIN_SCORE."""
    s, e, w = (k[:, HALPE26_INDEX[f"{side}_{part}"]] for part in ("shoulder", "elbow", "wrist"))
    ok = (s[:, 2] >= MIN_SCORE) & (e[:, 2] >= MIN_SCORE) & (w[:, 2] >= MIN_SCORE)
    ok &= np.isfinite(width) & (width > 1e-6)
    lift = np.full(len(k), np.nan)
    lift[ok] = (s[ok, 1] - w[ok, 1]) / width[ok]
    upper, lower = s[:, :2] - e[:, :2], w[:, :2] - e[:, :2]
    norm = np.hypot(upper[:, 0], upper[:, 1]) * np.hypot(lower[:, 0], lower[:, 1])
    ok &= norm > 1e-9
    angle = np.full(len(k), np.nan)
    cos = (upper[ok] * lower[ok]).sum(axis=1) / norm[ok]
    angle[ok] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return lift, angle


def arm_gestures(
    pose: pd.DataFrame, track: int, fps: int | float | Fraction | str
) -> list[Gesture]:
    """Raised-hand gestures of one player, in time order; see the constants above. Empty
    when the player has no pose or is too small in the image."""
    rate = float(Fraction(str(fps)))
    min_frames = max(1, math.ceil(GESTURE_MIN_S * rate - 1e-9))
    max_gap = max(1, round(GESTURE_GAP_S * rate))
    rows = pose[pose["track_id"] == track].sort_values("frame", kind="stable")
    rows = rows.drop_duplicates("frame")
    if rows.empty:
        return []
    k = keypoints_array(rows)
    frames = rows["frame"].to_numpy(dtype=np.int64)
    ls, rs = k[:, HALPE26_INDEX["left_shoulder"]], k[:, HALPE26_INDEX["right_shoulder"]]
    width = np.hypot(ls[:, 0] - rs[:, 0], ls[:, 1] - rs[:, 1])
    width[(ls[:, 2] < MIN_SCORE) | (rs[:, 2] < MIN_SCORE)] = np.nan
    if not np.isfinite(width).any() or float(np.nanmedian(width)) < GESTURE_MIN_SHOULDER_PX:
        return []
    lift: dict[str, np.ndarray] = {}
    angle: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        lift[side], angle[side] = _arm_raise(k, side, width)
    up = {side: np.nan_to_num(lift[side], nan=-np.inf) >= RAISE_SHOULDER_W for side in lift}
    straight = {
        side: up[side] & (np.nan_to_num(angle[side], nan=0.0) >= ARM_STRAIGHT_DEG) for side in up
    }
    raised = up["left"] | up["right"]
    runs: list[list[int]] = []
    for i in np.flatnonzero(raised):
        if runs and frames[i] - frames[runs[-1][1]] <= max_gap:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    best_lift = np.fmax(lift["left"], lift["right"])
    gestures: list[Gesture] = []
    for i0, i1 in runs:
        if frames[i1] - frames[i0] + 1 < min_frames:
            continue
        part = slice(i0, i1 + 1)
        straight_at = np.flatnonzero(straight["left"][part] | straight["right"][part])
        n_straight = int(straight_at.size)
        top = i0 + int(np.nanargmax(best_lift[part]))
        arm = (
            "left"
            if np.nan_to_num(lift["left"][top], nan=-np.inf)
            >= np.nan_to_num(lift["right"][top], nan=-np.inf)
            else "right"
        )
        gestures.append(
            Gesture(
                track_id=int(track),
                start_frame=int(frames[i0]),
                end_frame=int(frames[i1]),
                top_frame=int(frames[top]),
                max_raise=float(best_lift[top]),
                straight=n_straight >= min_frames,
                arm=arm,
                evidence={
                    "straight_frames": n_straight,
                    "first_straight_frame": (
                        int(frames[i0 + straight_at[0]]) if straight_at.size else None
                    ),
                    "shoulder_width_px": round(float(np.nanmedian(width[part])), 1),
                    "raise_shoulder_w": {
                        int(f): round(float(v), 2)
                        for f, v in zip(frames[part], best_lift[part], strict=True)
                        if np.isfinite(v) and abs(int(f) - int(frames[top])) <= 3
                    },
                    "elbow_deg_at_top": (
                        round(float(angle[arm][top]), 1) if np.isfinite(angle[arm][top]) else None
                    ),
                    "straight_at_top": (
                        bool(angle[arm][top] >= ARM_STRAIGHT_DEG)
                        if np.isfinite(angle[arm][top])
                        else None
                    ),
                },
            )
        )
    return gestures


def reaction_timeline(
    pose: pd.DataFrame,
    track: int,
    start: int,
    end: int,
    fps: int | float | Fraction | str,
    *,
    label: str | None = None,
) -> list[Moment]:
    """Key moments of a close-up subject's reaction inside frames ``start..end``: each
    gesture's start (``hand_up``), and the frame the hand is highest in the reaction's main
    gesture, the straight-arm gesture raised highest, else the longest bent-arm one. That
    frame is ``arm_top`` when the main gesture is a straight-arm one and its elbow is still
    straight there, else ``hand_top`` (see :func:`_top_words` for what each says).
    ``hand_up`` gives how long an arm is straight (in all, over the gesture's straight
    frames) and whether it straightens only after the hand went up; for a bent-arm gesture,
    how long the hand stays up, with 屈肘 only when the elbow is bent on every raised frame
    (an arm straight too briefly to count, perhaps on the top frame, is said to be so)."""
    who = label or f"轨迹 {track}"
    rate = float(Fraction(str(fps)))
    window = pose[(pose["frame"] >= start) & (pose["frame"] <= end)]
    gestures = arm_gestures(window, track, fps)
    if not gestures:
        return []
    straight = [g for g in gestures if g.straight]
    if straight:
        main = max(straight, key=lambda g: (g.max_raise, -g.start_frame))
    else:
        main = max(gestures, key=lambda g: (g.end_frame - g.start_frame, -g.start_frame))
    moments: list[Moment] = []
    for g in gestures:
        seconds = (g.end_frame - g.start_frame + 1) / rate
        straight_s = g.evidence["straight_frames"] / rate
        evidence = {
            "gesture_frames": [g.start_frame, g.end_frame],
            "raised_s": round(seconds, 2),
            "straight_s": round(straight_s, 2),
            "max_raise_shoulder_w": round(g.max_raise, 2),
            "straight_arm": g.straight,
            "arm": g.arm,
            **g.evidence,
        }
        if not g.straight and not g.evidence["straight_frames"]:
            shape = f"屈肘，持续 {seconds:.1f} 秒"
        elif not g.straight:  # straight on too few frames to count, maybe on the top one
            shape = f"持续 {seconds:.1f} 秒，手臂伸直不到 {GESTURE_MIN_S:g} 秒"
        elif g.evidence["first_straight_frame"] == g.start_frame:
            shape = f"手臂伸直约 {straight_s:.1f} 秒"
        else:  # the arm straightens after the hand went up
            shape = f"随后手臂伸直约 {straight_s:.1f} 秒"
        moments.append(
            Moment(g.start_frame, "hand_up", (track,), f"{who}抬手过肩（{shape}）", evidence)
        )
        if g is main:
            kind = "arm_top" if g.straight and g.evidence["straight_at_top"] else "hand_top"
            text = _top_words(kind, g.evidence["straight_at_top"])[0]
            moments.append(Moment(g.top_frame, kind, (track,), f"{who}{text}", evidence))
    return _in_order(moments)


def _top_words(kind: str, straight_at_top: bool | None) -> tuple[str, str]:
    """(moment text, title words) for the frame the hand is highest: a straight arm only
    for ``arm_top`` (a straight-arm gesture still straight there), a bent elbow only when
    the elbow is bent on that frame, else nothing about the elbow (an elbow not measured,
    or straight there in a gesture whose arm was straight too briefly to count, see
    :data:`GESTURE_MIN_S`)."""
    if kind == "arm_top":
        return "手臂伸直举过肩，举到最高", "手臂伸直举过肩"
    if straight_at_top is False:
        return "手举到最高（屈肘）", "手抬过肩（屈肘）"
    return "手举到最高", "手抬过肩"


def _reaction_peak(moments: Sequence[Moment]) -> Moment | None:
    """The top of the reaction's main gesture (see :func:`reaction_timeline`)."""
    return next((m for m in moments if m.kind in ("arm_top", "hand_top")), None)


# ---------------------------------------------------------------------------------------
# Post-processing of highlights.json: merge windows that meet at a cut, add key moments,
# rank.

#: highlights whose windows touch (the second starts at most this many frames after the
#: first ends, on a cut frame) are merged: often one play seen in consecutive shots, but
#: nothing beyond that adjacency is checked ...
MERGE_GAP_FRAMES = 1
#: ... unless the merged window would be longer than this
MERGE_MAX_S = 15.0
#: a close-up reaction whose peak moves to a gesture keeps at most this much (s) of the
#: shot before the cut into the close-up; its old window stays in ``previous_peak``
REACTION_LEAD_S = 2.0
#: a highlight is "main" when its score is at least this share of the best score ...
MAIN_SHARE = 0.5
#: ... and it ranks within the first this many
MAIN_MAX = 3
#: key-moment kinds that can become the peak of a highlight, in order of preference
PEAK_KINDS = ("ball_release", "fall_onset")
#: a duel pair candidate replaces an earlier one only when its timeline has more moments of
#: these kinds (the ones that can move the peak, and the ball leaving a player's feet
#: without contact); other moments, touches among them, do not rank pairs
PAIR_RANK_KINDS = (*PEAK_KINDS, "ball_away")
#: the Chinese name of a closeup_reaction highlight, which starts its summary
REACTION_NAME_ZH = "球员反应"
#: the ids a merged highlight records for its parts are the ids the parts had before the
#: merge, with this prefix: the merge renumbers the highlights, so a bare part id could name
#: another, final highlight
PART_ID_PREFIX = "detect-"
#: what the text calls a shot whose size is not known (no shot table)
SHOT_UNKNOWN_ZH = "近镜头"


def merge_adjacent_highlights(
    doc: Mapping[str, Any], cut_frames: Sequence[int], fps: int | float | Fraction | str
) -> dict[str, Any]:
    """Merge highlights whose windows meet at a cut (see :data:`MERGE_GAP_FRAMES`).

    The merged highlight keeps the fields of the highest-scoring part, spans every window,
    lists the track ids of all parts (that part's first) and records ``merged_from`` (the
    parts' ids) and ``merged_parts`` (each part's id, type, window, peak, score and title).
    Ids are renumbered h01, h02, ... in time order; a part keeps the id it had before, with
    :data:`PART_ID_PREFIX` (detect-h05), so it never names another highlight.
    """
    rate = float(Fraction(str(fps)))
    cuts = {int(c) for c in cut_frames}
    items = sorted((dict(h) for h in doc.get("highlights", [])), key=lambda h: h["start_frame"])
    keys = ("id", "type", "start_frame", "peak_frame", "end_frame", "score", "title_zh")

    def part_id(h: Mapping[str, Any]) -> str:
        ident = str(h["id"])
        return ident if ident.startswith(PART_ID_PREFIX) else PART_ID_PREFIX + ident

    def parts(h: Mapping[str, Any]) -> list[dict[str, Any]]:
        return list(h.get("merged_parts") or [{**{k: h.get(k) for k in keys}, "id": part_id(h)}])

    merged: list[dict[str, Any]] = []
    for h in items:
        if merged:
            prev = merged[-1]
            gap = h["start_frame"] - prev["end_frame"]
            span = (max(prev["end_frame"], h["end_frame"]) - prev["start_frame"] + 1) / rate
            if gap <= MERGE_GAP_FRAMES and h["start_frame"] in cuts and span <= MERGE_MAX_S:
                best, other = (prev, h) if prev["score"] >= h["score"] else (h, prev)
                combined = dict(best)
                combined["start_frame"] = prev["start_frame"]
                combined["end_frame"] = max(prev["end_frame"], h["end_frame"])
                ids = list(best.get("track_ids", []))
                ids += [t for t in other.get("track_ids", []) if t not in ids]
                combined["track_ids"] = ids
                combined["merged_from"] = list(prev.get("merged_from") or [part_id(prev)]) + list(
                    h.get("merged_from") or [part_id(h)]
                )
                combined["merged_parts"] = parts(prev) + parts(h)
                merged[-1] = combined
                continue
        merged.append(h)
    for number, h in enumerate(merged, start=1):
        h["id"] = f"h{number:02d}"
    out = dict(doc)
    out["highlights"] = merged
    return out


def _team_label(
    tracks: pd.DataFrame | None, track: int, lo: int, hi: int, team_names: Mapping[str, str]
) -> str:
    """'<team>球员' from the track's majority team inside frames lo..hi, else the id (the
    way a title names a player)."""
    if tracks is None or "team" not in tracks.columns:
        return f"轨迹 {track} 球员"
    rows = tracks[(tracks["track_id"] == track) & (tracks["frame"] >= lo) & (tracks["frame"] <= hi)]
    team = _majority_team(rows)
    if team is None:
        return f"轨迹 {track} 球员"
    return f"{team_names.get(team, team)}球员"


def _text_names(
    tracks: pd.DataFrame | None,
    track_ids: Sequence[int],
    lo: int,
    hi: int,
    team_names: Mapping[str, str],
) -> dict[int, str]:
    """How key-moment texts and summaries name players: team, a confident jersey number and
    the track id, from the rows of frames lo..hi (gbf.highlights.player_names_zh)."""
    if tracks is not None:
        tracks = tracks[(tracks["frame"] >= lo) & (tracks["frame"] <= hi)]
    return player_names_zh(tracks, track_ids, lo, hi, team_names)


def _majority_team(rows: pd.DataFrame) -> str | None:
    """The team label most of ``rows`` carry ("left" or "right"), None without one or on a
    tie."""
    if "team" not in rows.columns:
        return None
    teams = rows["team"]
    counts = teams[teams.isin(["left", "right"])].value_counts()
    if counts.empty or (len(counts) > 1 and counts.iloc[0] == counts.iloc[1]):
        return None
    return str(counts.index[0])


#: a duel partner of a joined shot's single subject stands at most this many box heights
#: away (median over the frames both are seen, foot to foot) ...
PARTNER_MAX_BOX_H = 1.0
#: ... and is seen on at least this share of the subject's frames
PARTNER_MIN_SHARE = 0.5


def add_duel_partners(doc: Mapping[str, Any], tracks: pd.DataFrame) -> dict[str, Any]:
    """Give a joined shot's single subject (a strike or the player the close-up shows) its
    nearest opponent as ``partner_track_id``, and list the partner in ``track_ids``.

    gbf.highlights reports one player for a ball event in the shot a highlight extends
    into; the duel partner is needed so that pose and key moments cover both players.
    The partner is the tracked player (negative ids are boxes without a track) of the other
    team seen on at least ``PARTNER_MIN_SHARE`` of the subject's frames inside the window
    whose median foot-to-foot distance, in mean box heights, is the smallest and at most
    ``PARTNER_MAX_BOX_H``. Both teams must be known: a subject without a team (no label,
    or as many rows of each) gets no partner, and only rows labelled with the other team
    count for a candidate, so two players are never paired without labels that differ.
    """
    out = dict(doc)
    result = []
    for original in doc.get("highlights", []):
        h = dict(original)
        cues = dict(h.get("cues") or {})
        emphasis = dict(cues.get("cut_emphasis") or {})
        joined = dict(emphasis.get("joined_shot_event") or {})
        ids = [int(t) for t in joined.get("track_ids") or []]
        if len(ids) != 1 or ids[0] < 0:
            result.append(h)
            continue
        subject = ids[0]
        lo, hi = int(h["start_frame"]), int(h["end_frame"])
        rows = tracks[(tracks["frame"] >= lo) & (tracks["frame"] <= hi)]
        mine = rows[rows["track_id"] == subject]
        if mine.empty:
            result.append(h)
            continue
        team = _majority_team(mine)
        if team is None:  # a teammate could not be told from an opponent
            result.append(h)
            continue
        others = rows[
            (rows["track_id"] != subject)
            & (rows["track_id"] >= 0)  # negative ids are boxes without a track
            & rows["frame"].isin(set(mine["frame"]))
            & rows["role"].isin(["player", "goalkeeper"])
            & rows["team"].isin(["left", "right"])
            & (rows["team"] != team)
        ]
        me = mine.set_index("frame")
        best = None
        for track, part in others.groupby("track_id"):
            if len(part) < PARTNER_MIN_SHARE * len(mine):
                continue
            a = me.loc[part["frame"]]
            dist = np.hypot(
                part["foot_u"].to_numpy() - a["foot_u"].to_numpy(),
                part["foot_v"].to_numpy() - a["foot_v"].to_numpy(),
            )
            heights = ((part["y2"] - part["y1"]).to_numpy() + (a["y2"] - a["y1"]).to_numpy()) / 2
            median = float(np.median(dist / np.maximum(heights, 1.0)))
            if median <= PARTNER_MAX_BOX_H and (best is None or median < best[1]):
                best = (int(track), median)
        if best is not None:
            joined["partner_track_id"] = best[0]
            joined["partner_distance_box_h"] = round(best[1], 3)
            emphasis["joined_shot_event"] = joined
            cues["cut_emphasis"] = emphasis
            h["cues"] = cues
            listed = [int(t) for t in h.get("track_ids") or []]
            if best[0] not in listed:
                at = listed.index(subject) + 1 if subject in listed else len(listed)
                listed.insert(at, best[0])
            h["track_ids"] = listed
        result.append(h)
    out["highlights"] = result
    return out


def _duel_pairs(h: Mapping[str, Any]) -> list[tuple[int, int]]:
    """Duel pair candidates: the joined shot's duel (or its subject and the partner found by
    :func:`add_duel_partners`) first, then the duel cue's pair. Negative ids (boxes without
    a track) are never part of one."""
    pairs: list[tuple[int, int]] = []
    cues = h.get("cues") or {}
    joined = (cues.get("cut_emphasis") or {}).get("joined_shot_event") or {}
    if joined.get("cue") == "duel" and len(joined.get("track_ids") or []) == 2:
        pairs.append((int(joined["track_ids"][0]), int(joined["track_ids"][1])))
    if joined.get("partner_track_id") is not None and len(joined.get("track_ids") or []) == 1:
        pairs.append((int(joined["track_ids"][0]), int(joined["partner_track_id"])))
    duel = cues.get("duel") or {}
    if len(duel.get("track_ids") or []) == 2:
        pairs.append((int(duel["track_ids"][0]), int(duel["track_ids"][1])))
    unique: list[tuple[int, int]] = []
    for pair in pairs:
        if min(pair) >= 0 and not any(set(pair) == set(seen) for seen in unique):
            unique.append(pair)
    return unique


def _moment_dicts(moments: Sequence[Moment], fps: int | float | Fraction | str) -> list[dict]:
    return [
        {
            "frame": m.frame,
            "time": format_time(m.frame, fps),
            "kind": m.kind,
            "track_ids": list(m.track_ids),
            "text_zh": m.text_zh,
            "evidence": m.evidence,
        }
        for m in moments
    ]


def _previous_peak(h: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame": h["peak_frame"],
        "type": h["type"],
        "refined_by": h.get("refined_by"),
        "title_zh": h.get("title_zh"),
        "start_frame": h["start_frame"],
        "end_frame": h["end_frame"],
    }


def _touch_rows(
    moments: Sequence[Moment], labels: Mapping[int, str], fps: int | float | Fraction | str
) -> list[dict]:
    return [
        {**row, "label_zh": labels.get(row["track_id"], f"轨迹 {row['track_id']}")}
        for row in touch_summary(moments, fps)
    ]


def _touch_text(rows: Sequence[Mapping[str, Any]], texts: Mapping[int, str]) -> str:
    """One Chinese sentence per player: the detected touches, a lower bound, naming the
    player as ``texts`` does (with the track id)."""
    parts = []
    for t in rows:
        frames = t["frames"]
        where = (
            f"在第 {frames[0]} 帧" if len(frames) == 1 else f"在第 {frames[0]}–{frames[-1]} 帧之间"
        )
        gap = (
            f"，检测到的触球平均间隔 {t['mean_interval_s']:.2f} 秒" if t["mean_interval_s"] else ""
        )
        who = texts.get(int(t["track_id"]), t["label_zh"])
        parts.append(f"{where}检测到{who}触球 {t['count']} 次（{TOUCHES_NOTE_ZH}）{gap}。")
    return "".join(parts)


#: the sentence of a summary that lists the players involved (gbf.highlights writes it)
_INVOLVED = re.compile(r"涉及球员：[^。]*。")


def _with_involved(
    h: dict[str, Any], tracks: pd.DataFrame | None, names: Mapping[str, str]
) -> None:
    """Add the players of ``key_pair`` and of the key moments to ``track_ids`` when missing,
    and write the summary's 涉及球员 sentence from them (gbf.highlights wrote it before the
    merge, the duel partner and the key moments), so it lists every player the title and
    the key moments name, with track ids."""
    ids = [int(t) for t in h.get("track_ids") or []]
    named = list(h.get("key_pair") or [])
    named += [t for m in h.get("key_moments") or [] for t in m.get("track_ids") or []]
    for track in named:
        if int(track) >= 0 and int(track) not in ids:
            ids.append(int(track))
    h["track_ids"] = ids
    if not ids:
        return
    texts = _text_names(tracks, ids, int(h["start_frame"]), int(h["end_frame"]), names)
    sentence = f"涉及球员：{'、'.join(texts[t] for t in ids)}。"
    summary = str(h.get("summary_zh") or "")
    if _INVOLVED.search(summary):
        h["summary_zh"] = _INVOLVED.sub(lambda _match: sentence, summary, count=1)
    elif "评分 " in summary:
        at = summary.index("评分 ")
        h["summary_zh"] = summary[:at] + sentence + summary[at:]
    else:
        h["summary_zh"] = summary + sentence


def _source_text(moments: Sequence[Moment]) -> str:
    """What the key moments are read from: the ball only when one of them is about it."""
    with_ball = any(m.kind in BALL_MOMENT_KINDS for m in moments)
    return "2D 姿态和球的位置" if with_ball else "2D 姿态"


def _shot_name(shots: pd.DataFrame | None, frame: int) -> str:
    """Chinese name of the shot of ``frame`` by the size of its players
    (gbf.highlights.shot_name_zh); :data:`SHOT_UNKNOWN_ZH` without a shot table, or when
    no shot covers the frame."""
    if shots is None:
        return SHOT_UNKNOWN_ZH
    rows = shots[(shots["start_frame"] <= frame) & (shots["end_frame"] >= frame)]
    if rows.empty:
        return SHOT_UNKNOWN_ZH
    box_h = rows["median_box_h"].iat[0] if "median_box_h" in rows.columns else None
    return shot_name_zh(str(rows["shot_type"].iat[0]), None if pd.isna(box_h) else float(box_h))


def _on_frame(table: pd.DataFrame | None, track: int, frame: int) -> bool:
    """Whether ``table`` (tracks or pose) has a row of ``track`` at ``frame``."""
    if table is None or not len(table):
        return False
    return bool(((table["frame"] == frame) & (table["track_id"] == track)).any())


def _drop_parts_outside(h: dict[str, Any], start: int, end: int) -> None:
    """Remove the merged parts (:func:`merge_adjacent_highlights`) that lie wholly outside
    frames ``start..end`` from ``merged_parts`` and ``merged_from``; both keys go when fewer
    than two parts are left. A highlight without ``merged_parts`` is left as it is."""
    parts = h.get("merged_parts")
    if not parts:
        return
    kept = [p for p in parts if int(p["end_frame"]) >= start and int(p["start_frame"]) <= end]
    if len(kept) == len(parts):
        return
    if len(kept) < 2:
        h.pop("merged_parts", None)
        h.pop("merged_from", None)
        return
    h["merged_parts"] = kept
    h["merged_from"] = [p["id"] for p in kept]


@dataclass(frozen=True)
class _Pair:
    """The duel pair kept for a highlight: its players, how a title names them (``labels``,
    by team) and how key-moment texts and summaries name them (``texts``, with track ids)."""

    a: int
    b: int
    labels: dict[int, str]
    texts: dict[int, str]


def _with_reaction(
    h: dict[str, Any],
    pose: pd.DataFrame | None,
    fps: int | float | Fraction | str,
    tracks: pd.DataFrame | None,
    shots: pd.DataFrame | None,
    names: Mapping[str, str],
    duel: Sequence[Moment],
    pair: _Pair | None,
) -> dict[str, Any] | None:
    """A close-up reaction with its subject's gestures as key moments, next to the moments
    of the duel ``pair`` (which change nothing on their own), and the peak moved to the
    reaction's peak (:func:`_reaction_peak`). The window then starts at most
    :data:`REACTION_LEAD_S` before the cut into the close-up, and pair moments and merged
    parts before that are dropped. None when the highlight is no close-up reaction or its
    subject shows no gesture."""
    if h.get("type") != "closeup_reaction" or pose is None or not len(pose):
        return None
    emphasis = (h.get("cues") or {}).get("cut_emphasis") or {}
    joined = emphasis.get("joined_shot_event") or {}
    ids = [int(t) for t in joined.get("track_ids") or []]
    if joined.get("cue") != "subject" or len(ids) != 1 or ids[0] < 0:
        return None
    lo, hi = int(h["start_frame"]), int(h["end_frame"])
    cut = emphasis.get("cut_frame")
    first = max(lo, int(cut)) if cut is not None else lo
    track = ids[0]
    label = _team_label(tracks, track, first, hi, names)
    texts = dict(pair.texts) if pair is not None else {}
    texts.update(_text_names(tracks, [track], first, hi, names))
    gestures = reaction_timeline(pose, track, first, hi, fps, label=texts[track])
    peak = _reaction_peak(gestures)
    if peak is None:
        return None
    lead = round(REACTION_LEAD_S * float(Fraction(str(fps))))
    start = max(lo, int(cut) - lead) if cut is not None else lo
    duel = [m for m in duel if start <= m.frame <= hi]
    moments = _in_order([*gestures, *duel])
    out = dict(h)
    out["key_moments"] = _moment_dicts(moments, fps)
    if duel and pair is not None:
        out["touches"] = _touch_rows(duel, pair.labels, fps)
        out["key_pair"] = [pair.a, pair.b]
    out["key_focus_track"] = track
    out["previous_peak"] = _previous_peak(h)
    out["start_frame"] = start
    out["peak_frame"] = peak.frame
    out["refined_by"] = "pose_key_moment"
    _drop_parts_outside(out, start, hi)
    shape = _top_words(peak.kind, peak.evidence.get("straight_at_top"))[1]
    out["title_zh"] = (
        f"{format_time(peak.frame, fps)}（第 {peak.frame} 帧）"
        f"{_shot_name(shots, peak.frame)}中{label}{shape}"
    )
    steps = "；".join(
        f"{m['time']}（第 {m['frame']} 帧）{m['text_zh']}"
        for m in out["key_moments"]
        if m["kind"] != "touch"
    )
    old = h.get("summary_zh") or ""
    reasons = old[old.index("入选依据") :] if "入选依据" in old else ""
    shot = f"，导播在第 {int(cut)} 帧切到{_shot_name(shots, int(cut))}" if cut is not None else ""
    out["summary_zh"] = (
        f"{REACTION_NAME_ZH}。关键时刻（由原始画面上的 {_source_text(moments)}逐帧确定，"
        f"画面平面，未验证）：{steps}。{_touch_text(out['touches'], texts)}"
        f"片段 {format_time(start, fps)} 至 {format_time(hi, fps)}（第 {start}–{hi} 帧{shot}）。"
        f"{reasons}"
    )
    return out


def _as_duel(
    h: dict[str, Any],
    moments: Sequence[Moment],
    pair: _Pair,
    primary: Moment,
    fps: int | float | Fraction | str,
) -> dict[str, Any]:
    """The highlight turned into a duel: ``primary`` is the new peak, and the title and the
    summary say what happened, frame by frame. The duel is called 争抢 as in the type's name
    (贴身争抢) and in 球脱离争抢; the title says 贴身争抢 only when the two came close in the
    image (``contact_start``): 贴身 says they were right next to each other, which the
    keypoint distance measures, not that they touched. A fall is what
    :func:`fall_events` measures: the trunk starts to tilt, then passes
    :data:`FALL_DOWN_DEG` in the image (画面平面); the title is about the key frame, so it
    says when the trunk passes that angle (随后 or 此前, with the frame) unless that is the
    key frame."""
    a, b, labels = pair.a, pair.b, pair.labels
    lo, hi = int(h["start_frame"]), int(h["end_frame"])
    out = dict(h)
    out["key_moments"] = _moment_dicts(moments, fps)
    out["touches"] = _touch_rows(moments, labels, fps)
    out["key_pair"] = [a, b]
    if primary.kind == "ball_release":
        out["key_focus_track"] = int(primary.evidence["nearest_track_id"])
    else:
        out["key_focus_track"] = int(primary.track_ids[0])
    fall = next((m for m in moments if m.kind == "fall_onset"), None)
    parts = []
    if primary.kind == "ball_release":
        parts.append("球脱离争抢")
    if fall is not None:
        who, down = labels[fall.track_ids[0]], int(fall.evidence["down_frame"])
        tilt = f"过 {FALL_DOWN_DEG:.0f}°"
        if primary.kind == "fall_onset":  # the key frame is the last upright one
            parts.append(f"{who}躯干开始倾倒，随后{tilt}（画面平面，第 {down} 帧）")
        elif down == primary.frame:
            parts.append(f"{who}躯干倾倒{tilt}（画面平面）")
        else:
            when = "随后" if down > primary.frame else "此前"
            parts.append(f"{who}躯干{when}倾倒{tilt}（画面平面，第 {down} 帧）")
    duel = "贴身争抢" if any(m.kind == "contact_start" for m in moments) else "争抢"
    out["previous_peak"] = _previous_peak(h)
    out["peak_frame"] = primary.frame
    out["type"] = "duel_tackle"
    out["refined_by"] = "pose_key_moment"
    out["title_zh"] = (
        f"{format_time(primary.frame, fps)}（第 {primary.frame} 帧）"
        f"{labels[a]}与{labels[b]}{duel}：{'，'.join(parts)}"
    )
    steps = "；".join(
        f"{m['time']}（第 {m['frame']} 帧）{m['text_zh']}"
        for m in out["key_moments"]
        if m["kind"] != "touch"
    )
    old = h.get("summary_zh") or ""
    reasons = old[old.index("入选依据") :] if "入选依据" in old else ""
    merged = h.get("merged_from") or []
    span = (
        f"片段 {format_time(lo, fps)} 至 {format_time(hi, fps)}（第 {lo}–{hi} 帧"
        + (f"，由 {len(merged)} 段合并：切镜前后相接的镜头" if merged else "")
        + "）。"
    )
    out["summary_zh"] = (
        f"关键时刻（由原始画面上的 {_source_text(moments)}逐帧确定，画面平面，"
        f"未验证）：{steps}。{_touch_text(out['touches'], pair.texts)}{span}{reasons}"
    )
    return out


def _key_moments_of(
    h: dict[str, Any],
    pose: pd.DataFrame | None,
    fps: int | float | Fraction | str,
    *,
    ball_track: pd.DataFrame | None,
    tracks: pd.DataFrame | None,
    shots: pd.DataFrame | None,
    names: Mapping[str, str],
    contacts: Sequence[ContactMatch],
) -> dict[str, Any]:
    """One highlight of :func:`add_key_moments`, before its ``key_focus_track`` is checked
    against the key frame."""
    h.update({"key_moments": [], "touches": [], "key_pair": None, "key_focus_track": None})
    best: list[Moment] = []
    best_pair: _Pair | None = None
    best_rank = -1
    lo, hi = int(h["start_frame"]), int(h["end_frame"])
    for a, b in _duel_pairs(h) if pose is not None and len(pose) else []:
        window = pose[(pose["frame"] >= lo) & (pose["frame"] <= hi) & pose["track_id"].isin([a, b])]
        both = set(window.loc[window["track_id"] == a, "frame"]) & set(
            window.loc[window["track_id"] == b, "frame"]
        )
        if len(both) < 5:
            continue
        f0, f1 = min(both), max(both)
        labels = {t: _team_label(tracks, t, f0, f1, names) for t in (a, b)}
        texts = _text_names(tracks, (a, b), f0, f1, names)
        timeline = duel_timeline(window, (a, b), f0, f1, fps, ball_track=ball_track, names=texts)
        rank = sum(m.kind in PAIR_RANK_KINDS for m in timeline)
        if rank > best_rank:  # an earlier candidate wins ties
            best, best_pair, best_rank = timeline, _Pair(a, b, labels, texts), rank
    if best_pair is not None:
        # Touches count over the whole window, as gbf.measure's contact records do, also
        # where only one of the pair has a pose; a shortened window drops some below.
        pair = (best_pair.a, best_pair.b)
        best = _in_order([*best, *_touch_moments(contacts, pair, lo, hi, best_pair.texts)])
    primary = next((m for kind in PEAK_KINDS for m in best if m.kind == kind), None)
    engaged = any(m.kind in ("contact_start", "fall_onset") for m in best)
    if best_pair is not None and primary is not None and engaged:
        return _as_duel(h, best, best_pair, primary, fps)
    reaction = _with_reaction(h, pose, fps, tracks, shots, names, best, best_pair)
    if reaction is not None:
        return reaction
    if best_pair is not None and best:
        h["key_moments"] = _moment_dicts(best, fps)
        h["touches"] = _touch_rows(best, best_pair.labels, fps)
        h["key_pair"] = [best_pair.a, best_pair.b]
        if h.get("type") == "duel_tackle":  # gbf.highlights' key frame of the duel
            h["key_focus_track"] = best_pair.a
    return h


def add_key_moments(
    doc: Mapping[str, Any],
    pose: pd.DataFrame | None,
    fps: int | float | Fraction | str,
    *,
    ball_track: pd.DataFrame | None = None,
    tracks: pd.DataFrame | None = None,
    team_names: Mapping[str, str] | None = None,
    touches: pd.DataFrame | None = None,
    shots: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Add ``key_moments``, ``touches``, ``key_pair`` and ``key_focus_track`` (see the module
    docstring) to every highlight.

    Every duel pair candidate (:func:`_duel_pairs`) with a pose for both players on at
    least 5 frames gets a :func:`duel_timeline`; the first candidate is kept unless a later
    one has more moments of :data:`PAIR_RANK_KINDS`. When the players were in contact (close
    in the image) or one of them fell, and a ball release or the fall is found, that moment
    becomes ``peak_frame`` (``refined_by`` = ``pose_key_moment``; the old peak is kept in
    ``previous_peak``), the type becomes ``duel_tackle``, and the title and summary say what
    happened, frame by frame. Otherwise the pair's moments are listed and change nothing
    (a ball leaving a player's feet without contact is a pass or a clearance, not a duel),
    except in a close-up reaction: there the subject's raised hands
    (:func:`reaction_timeline`) join them and the highest one becomes the peak. ``touches``
    are gbf.ball events; the ones :func:`confirmed_touches` confirms (it needs ``tracks``)
    for a player of the kept pair inside the final window become ``touch`` moments and the
    per-player ``touches`` summary. ``key_focus_track`` is cleared when that player has no
    row of ``tracks`` (without it: of ``pose``) at the final ``peak_frame``. ``shots``
    (shots.parquet: start_frame, end_frame, shot_type and median_box_h) names the shots in
    the text by their size; without it they are :data:`SHOT_UNKNOWN_ZH`. Last, the players
    of ``key_pair`` and of the key moments join ``track_ids`` when missing, and the
    summary's list of the players involved is written from ``track_ids``.
    """
    if shots is not None:
        missing = [c for c in ("start_frame", "end_frame", "shot_type") if c not in shots.columns]
        if missing:
            raise MomentsInputError("add_key_moments", [f"shots misses column(s) {missing}"])
    names = {"left": "左侧球队", "right": "右侧球队", **dict(team_names or {})}
    with_pose = pose is not None and len(pose) > 0
    contacts = confirmed_touches(touches, tracks, pose) if with_pose else []
    present = tracks if tracks is not None else pose  # the rows that put a player on a frame
    out = dict(doc)
    result = []
    for original in doc.get("highlights", []):
        h = _key_moments_of(
            dict(original),
            pose,
            fps,
            ball_track=ball_track,
            tracks=tracks,
            shots=shots,
            names=names,
            contacts=contacts,
        )
        focus = h["key_focus_track"]
        if focus is not None and not _on_frame(present, int(focus), int(h["peak_frame"])):
            h["key_focus_track"] = None  # the key frame is about the players it shows
        _with_involved(h, tracks, names)
        result.append(h)
    out["highlights"] = result
    return out


def rank_highlights(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Add ``rank`` (1 = highest score) and ``tier`` (``main`` or ``candidate``)."""
    out = dict(doc)
    items = [dict(h) for h in doc.get("highlights", [])]
    if items:
        top = max(h["score"] for h in items)
        order = sorted(
            range(len(items)), key=lambda i: (-items[i]["score"], items[i]["peak_frame"])
        )
        for rank, i in enumerate(order, start=1):
            items[i]["rank"] = rank
            main = rank <= MAIN_MAX and items[i]["score"] >= MAIN_SHARE * top
            items[i]["tier"] = "main" if main else "candidate"
    out["highlights"] = items
    return out
