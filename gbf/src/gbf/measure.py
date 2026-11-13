"""Evidence records for the highlights of a clip (M1): ``measurements.jsonl``.

Every number shown for a highlight is one evidence record (schema
``urn:gbf:schema:measurement``, ADR 0004): metric id and tier, claim type, value, unit and
display text, the frames and players it comes from, quality and coverage, the error status,
the method with its parameters and the code version, the restrictions of its inputs and the
sha256 of every input. A value that cannot be measured is a record too, with status
``unavailable``, value null and a ``reasonCode``; it is never written as zero. C-tier
metrics (ball speed, foot speed, acceleration, angular velocities, ...) are never computed:
:func:`build_record` refuses a metric the registry blocks, and the schema has no C tier.

What :func:`measure_highlight` measures inside one highlight window
    The involved players are the highlight's ``track_ids`` (gbf.highlights: striker, duel
    pair, players nearest the ball, fastest player, the players of a joined close shot).
    Each player has a reference frame: the key frame (``peak_frame``) when the player is
    on screen within ``key_half_window`` frames of it in the key frame's shot, else the
    frame of the joined shot's moment (``cues.cut_emphasis.joined_shot_event``) when the
    player belongs to it.

    - ``ball-contact-frame`` (tier B, an estimate): every ball event of the window (abrupt
      change of the ball's image velocity, gbf.ball) whose ball centre lies within
      ``contact_radius_bh`` box heights of a confident foot keypoint (ankle, toe or heel)
      of a player; without a pose for the players at that frame, the nearest player's foot
      point (bottom centre of the box) within ``contact_fallback_bh`` box heights is used
      instead and the record says so. Frame-accurate at the video rate, not sub-frame, and
      not validated (gate G14 not passed). A window without any matched contact gets one
      ``unavailable`` record.
    - ``ball-image-speed`` (tier image_plane): for each matched contact whose ball leaves
      faster or in a new direction, the ball's image speed after the contact in box heights
      per second at the ball's own scale (gbf.ball ``speed_after``). Never converted to
      km/h: there is no ground model for a ball in flight.
    - On pitch frames (fresh calibration with rep_err <= 8 px inside a wide shot, the
      ``ev_pitch`` frames of signals.parquet), per involved player:
      ``peak-speed`` (A, labelled 峰值速度（0.4秒平均）, the label stating the chord's
      actual duration): the highest valid speed of kinematics.parquet on the window's
      pitch frames, in segments of at least ``min_segment_frames`` frames. A gbf.kinematics
      speed is the chord of the smoothed positions centred on its frame, over
      ``peak_speed_window_s`` (2 * :func:`gbf.kinematics.chord_half_frames` frames, 10 at
      25 fps), valid only where both chord ends are measured, away from the segment ends
      and plausible; the record checks that the speed it reports is that chord. gbf.render
      labels and the gbf.highlights speed cue show the same valid speeds, so none exceeds
      the card's peak. Stretches without valid speeds may hide the real peak, so it is
      withheld when valid speeds cover less than ``peak_speed_min_coverage`` of the frames
      on which the player was tracked on the pitch in the window, leaving out those within
      half the smoothing window plus the chord half (``sg_window_frames``; 10 frames at
      25 fps) of the player's first and last tracked frame in each shot, which no speed can
      cover (``speed_coverage_split``, ``_rejected``, ``_short`` or ``_implausible``, after
      the commonest reason the other frames have no valid speed). It is withheld as
      ``speed_peak_hidden`` when the highest valid speed lies within half a chord (5 frames
      at 25 fps) of an end of its run of valid speeds beyond which the player was tracked on
      frames that no valid chord spans (track ends left out), or when the player crosses
      such a stretch, one with a gap of more than ``max_gap_frames`` missing or rejected
      positions, more than :data:`PEAK_SPEED_GAP_MARGIN` faster than that speed (between the
      smoothed positions at the chord ends around it; across a shorter gap the segments
      were split by a jump test and need not be one player). Either way the real peak may
      lie there: the card shows no value, and the highest valid speed stays in quality as a
      lower bound for audit; the reason says so with the record's numbers (how many frames
      carry a trusted speed, that highest speed and its frame,
      :func:`withheld_speed_reason_zh`);
      ``nearest-opponent-distance`` (A, labelled with its window, 最近对手距离（±2帧中位数）,
      since the overlay line shows the distance of one frame): median over the reference
      frame +/- ``key_half_window`` frames of the distance in metres to the nearest
      outfield player of the other team (goalkeepers reported separately in ``quality``);
      ``body-orientation`` (A_candidate): the angle between the player's facing (gbf.lift3d
      height-prior skeleton) and the direction to the goal the player's team attacks, from
      the circular mean of the facings over the reference frame +/- ``key_half_window``,
      where the reference is the player's own contact frame when one was matched;
      ``trunk-lean-at-contact`` (A_candidate): median sagittal trunk lean (+ forward) over
      the same frames.
      The two A_candidate metrics pass display gates first: lifted rows only and box
      height at least the registry's ``min_bbox_px``. Orientation requires facing_conf >=
      ``facing_min_conf`` and facing_sigma_deg < ``facing_max_sigma_deg``; lean requires
      trunk_lean_sigma_deg < ``lean_max_sigma_deg``. Both also leave out a frame whose
      facing points to the other side of the picture than the trunk visibly leans (a
      front-back reversal, :func:`facing_reversed`), since reversing the forward axis reverses
      signed sagittal lean. Lean keeps its own sigma gate, without the facing confidence
      and sigma gates. The orientation's per-frame gates are :func:`facing_frame_ok`, which
      gbf.render's facing arrow passes too, and its label states the window
      (朝向夹角（±2帧均值）).
      ``attackers-in-box`` (B, per highlight): outfield players of the attacking team inside
      the penalty area they attack (visible players only), on the frame of the highlight's
      box-pressure cue (the frame its summary names) when the window has one, else on the
      key frame; withheld (``team_unknown``) when outfield players are visible on that frame
      but none has a team label, and (``unknown_team_in_box``) when one without a team label
      is inside the box it counts for.
      Both metrics that need the goal a team attacks, ``attackers-in-box`` and
      ``body-orientation``, are withheld (``attack_direction_unknown``) when
      ``MeasureContext.attack_direction_known`` is False: gbf.side_check could not confirm
      the tracking plugin's left/right team names. Only ``team_unknown`` and the reasons a
      metric has before the direction matters (no pose, uncertain facing) come first.
    - On other frames (no pitch-usable calibration: close-ups, low angles), per involved
      player: ``nearest-opponent-image-distance`` (foot points, in box heights, the median
      over the same frames, labelled 画面最近对手距离（±2帧中位数）) and the
      image-plane angles ``image-trunk-lean``, ``image-knee-angle-left`` and
      ``image-knee-angle-right`` (tier image_plane, labelled 仅画面平面). They are image
      measurements, not body angles.

Conventions
    Frames are canonical CFR indices. Pitch coordinates are SoccerNet metres (origin at the
    centre spot, x along the touchlines, far touchline at y = -34, z down); team "left"
    defends x < 0 and attacks +x (``attack_direction`` overrides). Facing angles are
    pitch_zup degrees from +x, counter-clockwise (gbf.lift3d). Speeds are stored in m/s.
    Pixels are those of the exports (tracks, pose). Every threshold here is an empirical
    starting value (:class:`MeasureParams`), not a validated one.

Determinism
    Records hold no timestamps; floats are rounded to :data:`FLOAT_DECIMALS` decimals before
    hashing, records are ordered deterministically, and :func:`write_measurements` writes
    canonical JSON lines, so the same inputs give the same bytes.
"""

from __future__ import annotations

import json
import math
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from gbf import contracts
from gbf.ingest.rates import format_rate, parse_rate
from gbf.kinematics import DEFAULT_SPEED_CHORD_S, chord_half_frames
from gbf.lift3d import image_trunk_lean_deg
from gbf.pose import HALPE26_INDEX, HALPE26_KEYPOINTS, N_KEYPOINTS
from gbf.track_tools import (
    box_touches_edge,
    off_pitch,
    off_pitch_at_camera_edge,
    same_spot,
    speed_chord_safe,
)
from gbf.util.clock import format_clock
from gbf.util.hashing import canonical_json, canonical_sha256
from gbf.util.io import write_text
from gbf.util.paths import RepoRootNotFoundError, repo_root

SCHEMA_NAME = "measurement"
SCHEMA_VERSION = 1
MEASUREMENTS_NAME = "measurements.jsonl"

FLOAT_DECIMALS = 4
"""Floats in records are rounded to this many decimals (stable evidence ids)."""

RECORD_TIERS: tuple[str, ...] = ("official", "A", "A_candidate", "B", "image_plane")
EXPLICIT_TIERS: tuple[str, ...] = ("B", "image_plane")
UNITS: tuple[str, ...] = ("m/s", "m", "deg", "count", "frame", "box_h", "bh/s", "")
CLAIM_TYPES: tuple[str, ...] = (
    "measurement",
    "estimate",
    "image_measurement",
    "provider_model",
    "official_fact",
)
ERROR_KINDS: tuple[str, ...] = ("measured", "literature", "inferred", "unmeasured")
PLANES: tuple[str, ...] = ("pitch", "image")
STATUSES: tuple[str, ...] = ("ok", "unavailable")
UNAVAILABLE_DISPLAY = "—"

PLAYER_ROLES: tuple[str, ...] = ("player", "goalkeeper")
TEAMS: tuple[str, ...] = ("left", "right")
DEFAULT_ATTACK_DIRECTION: Mapping[str, int] = MappingProxyType({"left": 1, "right": -1})
"""Sign of the x direction each team attacks: "left" defends x < 0 (plugin convention)."""

HALF_LENGTH_M = 52.5
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_HALF_WIDTH_M = 20.16

FOOT_KEYPOINTS: tuple[str, ...] = (
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
)
"""Halpe-26 keypoints that can touch the ball for a contact match."""
_FOOT_INDEX = tuple(HALPE26_INDEX[name] for name in FOOT_KEYPOINTS)

LIFTED = "height_prior_calibrated"
"""gbf.lift3d method of a lifted row."""

REFERENCE_HEIGHT_PX = 1080
"""Box heights are compared with registry minimums in pixels of a 1080-line frame."""


class MeasureError(ValueError):
    """Invalid input to gbf.measure, or a record that violates its contract.

    ``problems`` lists every problem found.
    """

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


# --------------------------------------------------------------------------- metric specs


@dataclass(frozen=True)
class MetricSpec:
    """How a metric is labelled and stored.

    ``explicit_tier`` is None for metrics whose tier comes from registries/metrics.toml
    (they must be listed there); otherwise it is "B" or "image_plane".
    """

    metric_id: str
    label_zh: str
    unit: str
    plane: str
    claim_type: str
    decimals: int | None
    explicit_tier: str | None
    method_id: str
    summary: str


METRICS: Mapping[str, MetricSpec] = MappingProxyType(
    {
        spec.metric_id: spec
        for spec in (
            MetricSpec(
                "peak-speed",
                "峰值速度（0.4秒平均）",
                "m/s",
                "pitch",
                "measurement",
                1,
                None,
                "gbf.measure.peak_speed",
                "Highest valid gbf.kinematics speed on the highlight window's pitch frames, "
                "in segments of at least min_segment_frames frames. Each speed is the chord "
                "of the player's smoothed ground positions (Savitzky-Golay) centred on its "
                "frame: their distance chord_frames apart (2 * round(peak_speed_window_s * "
                "fps / 2), 10 frames at 25 fps) divided by chord_s, valid where both ends are "
                "measured (not interpolated) positions away from the segment ends and the "
                "speed is plausible; the overlay labels and the sprint cue show the same "
                "speeds. Positions are cleaned first (box size, local median, excursions "
                "from the local trend), and a jump of the trend or an acceleration of more "
                "than 8 m/s^2 between the trends around a junction splits the segment. What "
                "the cleaning keeps still adds to the speeds (synthetic runs of 1-9 m/s at "
                "25 fps, gbf.kinematics module docstring): an identity switch from one frame "
                "to the next stays up to 0.46 m on clean positions (with noise, the largest "
                "seen to stay in 60 tracks per size are 0.95 m with 10 cm per axis and "
                "1.45 m with 30 cm, 1.15 and 1.6 m across 1-3 missing frames) and adds up "
                "to 1.084 times its size over chord_s along the motion, plus what the noise "
                "adds (in that sample up to 2.27 and 3.36 m/s in all, 2.89 and 3.94 m/s "
                "across missing frames; another 5-seed sample keeps 1.2 m at 10 cm and "
                "1.45 m at 20 cm across missing frames, adding up to 2.97 and 3.40 m/s; "
                "these are sample maxima, not bounds); on clean positions a slide onto a "
                "neighbour slower "
                "than 1 m in 5 frames, 2 m in 13 or 8 m in 23 reads as its own speed on top "
                "of the player's, up to max_speed (with noise some faster ones do too); "
                "an out-and-back dip that stays adds up to 2.4 m/s on clean positions and, "
                "the largest seen in 12 tracks per size and length, 3.07 m/s with 20 cm of "
                "noise; noise alone adds a median of 0.4 m/s and at "
                "most 0.6 m/s per 10 cm. An out-and-back burst of the positions moves a "
                "chord by up to 1.17 times its size over chord_s (the smoothing overshoots a "
                "burst of 9 or 10 frames); a real sprint reads as its chord_s mean, slightly "
                "below the instantaneous peak. Unavailable when valid speeds cover less than "
                "peak_speed_min_coverage of the frames the player was tracked on the pitch "
                "(those within the chord margin of the track's start and end in a shot left "
                "out; reasonCode speed_coverage_* after the commonest cause), or when the "
                "highest valid speed lies within half a chord of a stretch without valid "
                "speeds, or the player crosses such a stretch with a long gap faster "
                "(speed_peak_hidden): the real peak may lie there.",
            ),
            MetricSpec(
                "nearest-opponent-distance",
                "最近对手距离（±2帧中位数）",
                "m",
                "pitch",
                "measurement",
                1,
                None,
                "gbf.measure.nearest_opponent",
                "Median over the reference frame +/- key_half_window frames of the ground "
                "distance from the player's foot point to the nearest outfield player of the "
                "other team (tracks.parquet pitch_x/pitch_y, pitch frames only); in-frame "
                "players only; goalkeepers reported separately in quality. A box on the "
                "player's own spot (gbf.track_tools.same_spot: the player detected twice, or "
                "one box around the player and the one behind) and a person off the pitch "
                "(gbf.track_tools.off_pitch) are no opponent.",
            ),
            MetricSpec(
                "body-orientation",
                "朝向夹角（±2帧均值）",
                "deg",
                "pitch",
                "estimate",
                0,
                None,
                "gbf.measure.body_orientation",
                "Unsigned angle between the circular mean facing of the gbf.lift3d "
                "height-prior skeleton over the reference frame +/- key_half_window (gated "
                "rows only) and the direction from the player to the centre of the goal the "
                "player's team attacks.",
            ),
            MetricSpec(
                "trunk-lean-at-contact",
                "躯干前倾",
                "deg",
                "pitch",
                "estimate",
                0,
                None,
                "gbf.measure.trunk_lean",
                "Median sagittal trunk lean (+ forward) of the gbf.lift3d height-prior "
                "skeleton over the reference frame +/- key_half_window (gated rows only); the "
                "other monocular solution is kept in quality.",
            ),
            MetricSpec(
                "attackers-in-box",
                "禁区内进攻球员",
                "count",
                "pitch",
                "measurement",
                None,
                "B",
                "gbf.measure.attackers_in_box",
                "Outfield players (role player) of the attacking team whose ground position "
                "lies inside the penalty area they attack, on the frame of the highlight's "
                "box-pressure cue when it has one in its window, else at the key frame; "
                "players outside the image are not counted.",
            ),
            MetricSpec(
                "ball-contact-frame",
                "触球帧（估计）",
                "frame",
                "image",
                "estimate",
                None,
                "B",
                "gbf.measure.contact_frame",
                "Vertex frame of an abrupt change of the ball's image velocity (gbf.ball) "
                "whose ball centre lies within contact_radius_bh box heights of a confident "
                "foot keypoint, or, without a pose, within contact_fallback_bh box heights of "
                "the nearest player's foot point. Frame-accurate, not sub-frame; not "
                "validated (gate G14).",
            ),
            MetricSpec(
                "ball-image-speed",
                "触球后球的像面速度",
                "bh/s",
                "image",
                "image_measurement",
                1,
                "image_plane",
                "gbf.measure.ball_image_speed",
                "The ball's image speed after a matched contact (gbf.ball speed_after: "
                "least-squares image velocity over the event window divided by the ball's "
                "own scale, box heights per second). Image plane only; never km/h.",
            ),
            MetricSpec(
                "nearest-opponent-image-distance",
                "画面最近对手距离（±2帧中位数）",
                "box_h",
                "image",
                "image_measurement",
                2,
                "image_plane",
                "gbf.measure.nearest_opponent_image",
                "Median over the reference frame +/- key_half_window frames of the image "
                "distance between the player's foot point and the nearest foot point of the "
                "other team, divided by the mean box height of the two players; a box on the "
                "player's own spot (gbf.track_tools.same_spot) is no opponent.",
            ),
            MetricSpec(
                "image-trunk-lean",
                "躯干倾斜（画面）",
                "deg",
                "image",
                "image_measurement",
                0,
                "image_plane",
                "gbf.measure.image_angles",
                "Angle of the hip-midpoint to shoulder-midpoint image vector from image up "
                "(+ towards image right), from the 2D pose at the reference frame (gbf.lift3d "
                "image_trunk_lean_deg, the trunk angle gbf.moments reads falls from).",
            ),
            MetricSpec(
                "image-knee-angle-left",
                "左膝夹角（画面）",
                "deg",
                "image",
                "image_measurement",
                0,
                "image_plane",
                "gbf.measure.image_angles",
                "Image angle at the left knee between thigh and shank (180 = straight), from "
                "the 2D pose at the reference frame.",
            ),
            MetricSpec(
                "image-knee-angle-right",
                "右膝夹角（画面）",
                "deg",
                "image",
                "image_measurement",
                0,
                "image_plane",
                "gbf.measure.image_angles",
                "Image angle at the right knee between thigh and shank (180 = straight), from "
                "the 2D pose at the reference frame.",
            ),
        )
    }
)
"""Every metric gbf.measure writes."""

REASONS_ZH: Mapping[str, str] = MappingProxyType(
    {
        "not_pitch_usable": "参考帧没有可用于球场坐标的标定（本镜头只给画面平面量）",
        "no_pitch_frames": "高光窗口内没有可用于球场坐标的帧",
        "out_of_frame": "该球员不在参考帧附近的画面内",
        "no_kinematics": "窗口内没有该球员的球场轨迹",
        "no_valid_speed": "窗口内没有可信的速度样本（连续可用帧太少或速度异常）",
        "speed_coverage_split": (
            "可信速度覆盖的帧太少，其余帧多在轨迹断开处前后（漏检或剔除的帧太多、身份或标定"
            "跳变）：测得的最高速度只是下限，不作为峰值"
        ),
        "speed_coverage_rejected": (
            "可信速度覆盖的帧太少，其余帧的位置多缺失或被判为异常而剔除：测得的最高速度只是"
            "下限，不作为峰值"
        ),
        "speed_coverage_short": (
            "可信速度覆盖的帧太少，其余帧多在太短的连续轨迹段上：测得的最高速度只是下限，不作为峰值"
        ),
        "speed_coverage_implausible": (
            "可信速度覆盖的帧太少，其余帧的速度多不合理（超过上限或紧挨着这样的速度）："
            "测得的最高速度只是下限，不作为峰值"
        ),
        "speed_peak_hidden": (
            "测得的最高可信速度紧挨着一段测不出速度的轨迹，或比跨过这段轨迹时的平均速度慢："
            "真实峰值可能在那段里，测得的最高速度只是下限，不作为峰值"
        ),
        "team_unknown": "球队归属未知",
        "attack_direction_unknown": "进攻方向不确定：插件给两队定的左右未能核实",
        "no_opponent": "画面内没有对方球员",
        "unknown_team_nearer": "附近有球队未知的球员（标签被球衣核对清除），最近对手无法确定",
        "unknown_team_in_box": "禁区里有球队未知的球员，进攻人数无法确定",
        "no_pose": "没有该球员的可用姿态估计",
        "not_lifted": "姿态无法提升到三维",
        "box_too_small": "目标框太小",
        "facing_uncertain": "身体前后方向不确定、前后帧不一致或与可见倾斜方向相反",
        "lean_uncertain": "躯干前倾的不确定度过大（多为正对或背对机位）",
        "low_keypoint_confidence": "关键点置信度不足",
        "feet_not_visible": "特写镜头里脚不在画面内（框底不是脚）",
        "no_contact": "窗口内没有检测到靠近脚的触球",
        "not_measured": "没有可用的输入表",
    }
)
"""Chinese text of every reasonCode gbf.measure writes. A withheld peak-speed record says
more: how many of its frames carry a trusted speed, and the highest one as a lower bound
(:func:`withheld_speed_reason_zh`)."""

_COVERAGE_CAUSES_ZH: Mapping[str, str] = MappingProxyType(
    {
        "speed_coverage_split": "其余帧多在轨迹断开处前后（漏检或剔除的帧太多、身份或标定跳变）",
        "speed_coverage_rejected": "其余帧的位置多缺失或被判为异常而剔除",
        "speed_coverage_short": "其余帧多在太短的连续轨迹段上",
        "speed_coverage_implausible": "其余帧的速度多不合理（超过上限或紧挨着这样的速度）",
    }
)
"""Why the frames of a peak-speed window without a trusted speed have none, by reasonCode."""


@dataclass(frozen=True)
class RegistryMetric:
    """The registry facts gbf.measure needs about one metric (registries/metrics.toml)."""

    metric_id: str
    tier: str
    blocked: bool
    min_bbox_px: int
    needs_fresh_calibration: bool
    needs_contact_frame: bool
    error_status: str


def default_registry_path() -> Path:
    """registries/metrics.toml of the gbf checkout this code runs from."""
    try:
        return repo_root() / "registries" / "metrics.toml"
    except RepoRootNotFoundError as exc:
        raise MeasureError(
            "default_registry_path", ["gbf does not run from a checkout; pass registry_path"]
        ) from exc


def load_metric_registry(path: str | Path | None = None) -> dict[str, RegistryMetric]:
    """Read and validate registries/metrics.toml (schema ``registry_metrics``).

    Returns ``{metric_id: RegistryMetric}``. Raises MeasureError listing every problem.
    """
    source = Path(path) if path is not None else default_registry_path()
    try:
        with open(source, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MeasureError("load_metric_registry", [f"cannot read {source.name}: {exc}"]) from exc
    found = contracts.problems(data, "registry_metrics")
    if found:
        raise MeasureError("load_metric_registry", found)
    result: dict[str, RegistryMetric] = {}
    duplicates = []
    for entry in data["metrics"]:
        ident = str(entry["id"])
        if ident in result:
            duplicates.append(f"metric id {ident!r} is listed twice")
            continue
        result[ident] = RegistryMetric(
            metric_id=ident,
            tier=str(entry["tier"]),
            blocked=bool(entry["blocked"]),
            min_bbox_px=int(entry["min_bbox_px"]),
            needs_fresh_calibration=bool(entry["needs_fresh_calibration"]),
            needs_contact_frame=bool(entry["needs_contact_frame"]),
            error_status=str(entry["error_status"]),
        )
    if duplicates:
        raise MeasureError("load_metric_registry", duplicates)
    return result


# --------------------------------------------------------------------------- parameters

CARD_LABEL_MAX_CHARS = 16
"""The freeze card shows labels of at most this many characters (it cuts longer ones)."""

PEAK_SPEED_GAP_MARGIN = 0.05
"""A stretch without valid speeds hides a faster peak when the player's mean speed across it
exceeds the highest valid speed by more than this share (module docstring)."""

_WINDOW_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "peak-speed": "峰值速度（{chord}秒平均）",
        "nearest-opponent-distance": "最近对手距离（±{key_half_window}帧中位数）",
        "nearest-opponent-image-distance": "画面最近对手距离（±{key_half_window}帧中位数）",
        "body-orientation": "朝向夹角（±{key_half_window}帧均值）",
    }
)
"""Labels that state the metric's window, so the card says what the number is: the median
over the reference frame +/- key_half_window frames (the overlay line shows one frame), the
mean facing over the same frames (the overlay's arrow shows one frame's facing; the record's
frame is its reference frame even when only a neighbour passes the gates), the duration of
the chord a speed is measured over."""


def _window_label(metric_id: str, key_half_window: int, chord_s: float) -> str | None:
    """The label of a metric whose label states its window (None for the others). Every
    median label says 中位数, so a card never names one statistic two ways; a window whose
    label would not fit the card (two-digit key_half_window for the image distance) raises
    MeasureError."""
    template = _WINDOW_LABELS.get(metric_id)
    if template is None:
        return None
    seconds = f"{chord_s:.2f}".rstrip("0").rstrip(".")
    label = template.format(key_half_window=key_half_window, chord=seconds)
    if len(label) > CARD_LABEL_MAX_CHARS:
        raise MeasureError(
            "metric_label",
            [
                f"the label {label!r} of {metric_id} is longer than {CARD_LABEL_MAX_CHARS} "
                "characters, which the card cuts"
            ],
        )
    return label


@dataclass(frozen=True)
class MeasureParams:
    """Thresholds of gbf.measure (empirical starting values, not validated).

    Attributes:
        key_half_window: frames on each side of a reference frame used for medians (the
            distance labels state it, so it must keep them within the card's 16 characters:
            at most 9).
        contact_radius_bh: ball centre to foot keypoint, in the player's box heights, for a
            contact: 0.15 box heights is about 0.27 m for a 1.80 m box, the ball's radius
            (0.11 m) plus room for keypoint and box noise. Chosen by eye on the development
            clip's contact sheets (matches between 0.2 and 0.28 box heights showed the foot
            visibly away from the ball); a regression value, not a validated one.
        contact_fallback_bh: ball centre to the nearest foot point (box bottom centre) when
            the player has no pose (gbf.ball's at_foot radius).
        contact_candidates: players nearest the ball whose pose is requested for a contact.
        contact_candidate_max_bh: only players within this many box heights of the ball
            (foot point) are contact candidates.
        keypoint_min_score: RTMPose score a foot keypoint needs.
        pose_min_mean_score: mean keypoint score a pose row needs (lower is a non-person box).
        facing_min_conf, facing_max_sigma_deg: display gates of body-orientation.
        lean_max_sigma_deg: display gate of trunk-lean-at-contact.
        facing_min_resultant: the facings of the frames used must agree (mean resultant
            length of their unit vectors) at least this much.
        min_segment_frames: kinematics segments shorter than this give no peak speed.
        max_gap_frames: the longest gap gbf.kinematics bridged (its ``max_gap_frames``):
            two segments of a track fewer frames apart were split by a jump test, so
            their positions need not be one player's.
        sg_window_frames: the smoothing window gbf.kinematics used (its
            ``sg_window_frames``): a valid speed lies at least (sg_window_frames - 1) / 2
            plus the chord half frames from a segment end, so that many frames at the start
            and end of a player's track in a shot never have one (they are left out of the
            coverage).
        peak_speed_window_s: the chord gbf.kinematics measured its speeds over
            (``speed_chord_s``, by default :data:`gbf.kinematics.DEFAULT_SPEED_CHORD_S`):
            2 * chord_half_frames(fps, this) frames (10 at 25 fps). The label states the
            chord's actual duration, and the record checks that the speed it reports is the
            chord of that many frames.
        peak_speed_min_coverage: peak-speed is withheld when the frames with a valid speed
            are fewer than this share of the frames on which the player was tracked on the
            pitch in the window, the frames at the start and end of the player's track in
            each shot that no speed can cover left out (stretches that were rejected or
            split off may hide the real peak).
        image_angle_min_box_px: smallest box (1080-line pixels) for image-plane angles (the
            B-tier general gate).
        image_feet_max_median_box_px: no image-plane foot distance in shots whose median
            person box (shots.median_box_h, 1080 lines) is taller than this (the box bottom
            is the frame edge in upper-body close-ups; gbf.highlights uses the same value).
        attack_direction: sign of the x direction each team attacks.
    """

    key_half_window: int = 2
    contact_radius_bh: float = 0.15
    contact_fallback_bh: float = 0.6
    contact_candidates: int = 3
    contact_candidate_max_bh: float = 2.0
    keypoint_min_score: float = 0.3
    pose_min_mean_score: float = 0.4
    facing_min_conf: float = 0.5
    facing_max_sigma_deg: float = 15.0
    lean_max_sigma_deg: float = 8.0
    facing_min_resultant: float = 0.7
    min_segment_frames: int = 25
    max_gap_frames: int = 3
    sg_window_frames: int = 11
    peak_speed_window_s: float = DEFAULT_SPEED_CHORD_S
    peak_speed_min_coverage: float = 0.5
    image_angle_min_box_px: float = 150.0
    image_feet_max_median_box_px: float = 540.0
    attack_direction: Mapping[str, int] = DEFAULT_ATTACK_DIRECTION

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name in (
            "key_half_window",
            "contact_candidates",
            "min_segment_frames",
            "max_gap_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                problems.append(f"{name} must be an int >= 0, got {value!r}")
        sg_window = self.sg_window_frames
        if (
            isinstance(sg_window, bool)
            or not isinstance(sg_window, int)
            or sg_window < 3
            or sg_window % 2 == 0
        ):
            problems.append(f"sg_window_frames must be an odd int >= 3, got {sg_window!r}")
        window = self.peak_speed_window_s
        if (
            isinstance(window, bool)
            or not isinstance(window, int | float)
            or not math.isfinite(window)
            or window <= 0
        ):
            problems.append(f"peak_speed_window_s must be a finite number > 0, got {window!r}")
        coverage = self.peak_speed_min_coverage
        if (
            isinstance(coverage, bool)
            or not isinstance(coverage, int | float)
            or not 0.0 <= coverage <= 1.0
        ):
            problems.append(f"peak_speed_min_coverage must be a number in [0, 1], got {coverage!r}")
        for name in (
            "contact_radius_bh",
            "contact_fallback_bh",
            "contact_candidate_max_bh",
            "keypoint_min_score",
            "pose_min_mean_score",
            "facing_min_conf",
            "facing_max_sigma_deg",
            "lean_max_sigma_deg",
            "facing_min_resultant",
            "image_angle_min_box_px",
            "image_feet_max_median_box_px",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                problems.append(f"{name} must be a finite number >= 0, got {value!r}")
        direction = self.attack_direction
        if not isinstance(direction, Mapping) or set(direction) != set(TEAMS):
            problems.append("attack_direction must map 'left' and 'right' to +1 or -1")
        elif any(direction[team] not in (1, -1) for team in TEAMS):
            problems.append("attack_direction values must be +1 or -1")
        if not problems:
            for metric_id in _WINDOW_LABELS:
                try:
                    _window_label(metric_id, self.key_half_window, float(window))
                except MeasureError as exc:
                    problems.extend(exc.problems)
        if problems:
            raise MeasureError("MeasureParams", problems)
        object.__setattr__(self, "attack_direction", MappingProxyType(dict(direction)))

    def as_dict(self) -> dict[str, Any]:
        """JSON form for evidence records and receipts."""
        result: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            result[item.name] = dict(value) if isinstance(value, Mapping) else value
        return result


DEFAULT_PARAMS = MeasureParams()


# --------------------------------------------------------------------------- facing gate

UPRIGHT_DEG = 20.0
FACING_LEAN_DEG = UPRIGHT_DEG
"""A trunk the picture shows leaning more than this from upright (the 2D pose's
hip-to-shoulder line, gbf.lift3d.image_trunk_lean_deg; gbf.moments' bound of an upright
trunk) leans to that side of the picture: a lunge, a fall or a sprint goes that way."""

FACING_SIDE_SHARE = 0.5
"""A facing points to one side of the picture when the horizontal part of its direction there
is at least this share of it (30 degrees or more off the picture's vertical)."""


@dataclass(frozen=True)
class FacingGates:
    """The per-frame gates of a lifted facing (:func:`facing_frame_ok`): body-orientation
    averages only frames that pass them, and gbf.render draws the facing arrow only on such
    frames, so the arrow shows a facing of the kind the data card's value is made of.

    ``min_conf`` and ``max_sigma_deg`` are :class:`MeasureParams`' ``facing_min_conf`` and
    ``facing_max_sigma_deg``, ``min_box_px`` the registry's ``min_bbox_px`` of
    body-orientation (:meth:`of`). The card's last gate, that the facings of the frames
    around its reference frame agree (``facing_min_resultant``), is about the mean over
    several frames; one frame's arrow shows that frame's facing.
    """

    min_conf: float
    max_sigma_deg: float
    min_box_px: float

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name in ("min_conf", "max_sigma_deg", "min_box_px"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                problems.append(f"{name} must be a finite number >= 0, got {value!r}")
            else:
                object.__setattr__(self, name, float(value))
        if problems:
            raise MeasureError("FacingGates", problems)

    @classmethod
    def of(cls, params: MeasureParams, registry: Mapping[str, RegistryMetric]) -> FacingGates:
        """The gates of ``params`` and the registry's body-orientation entry."""
        entry = registry.get("body-orientation")
        return cls(
            min_conf=params.facing_min_conf,
            max_sigma_deg=params.facing_max_sigma_deg,
            min_box_px=float(entry.min_bbox_px) if entry is not None else 0.0,
        )


def facing_image_dx(
    to_image: Callable[[np.ndarray], np.ndarray | None],
    ground: Sequence[float],
    facing_deg: float,
) -> float | None:
    """The horizontal part of the facing's direction in the picture, as a share of it (-1
    to 1, + to the right), at the ground point ``ground`` (SoccerNet metres): the image of a
    short step along ``facing_deg`` (pitch_zup degrees). ``to_image`` maps (n, 2) ground points
    to (n, 2) pixels (the frame's camera, or its ground homography); None when it cannot."""
    heading = math.radians(facing_deg)
    x, y = float(ground[0]), float(ground[1])
    step = 0.01  # metres; pitch_zup (x, y) is SoccerNet (x, -y)
    points = np.array([[x, y], [x + step * math.cos(heading), y - step * math.sin(heading)]])
    pixels = to_image(points)
    if pixels is None:
        return None
    image = np.asarray(pixels, dtype=np.float64)
    du, dv = image[1] - image[0]
    length = math.hypot(du, dv)
    if not math.isfinite(length) or length <= 0.0:
        return None
    return float(du / length)


def facing_reversed(image_trunk_deg: float, image_dx: float | None) -> bool:
    """True when the picture shows the trunk leaning to one side, more than
    :data:`FACING_LEAN_DEG` from upright, and the facing points to the other side (by at
    least :data:`FACING_SIDE_SHARE`, :func:`facing_image_dx`) or cannot be placed in the
    picture. Players lunge, fall and sprint the way they face; the height-prior lift follows
    the 2D pose's left/right labels and can reverse front and back on exactly such frames,
    with no more doubt in its facing_conf or facing_sigma_deg than on a correct one. A
    trunk without both hips and both shoulders seen (NaN) is not checked."""
    if not math.isfinite(image_trunk_deg) or abs(image_trunk_deg) <= FACING_LEAN_DEG:
        return False
    if image_dx is None or not math.isfinite(image_dx):
        return True
    return image_dx * math.copysign(1.0, image_trunk_deg) <= -FACING_SIDE_SHARE


def facing_frame_ok(
    gates: FacingGates,
    *,
    lifted: bool,
    facing_deg: float,
    facing_conf: float,
    facing_sigma_deg: float,
    box_height_px: float,
    image_trunk_deg: float,
    image_dx: float | None,
) -> bool:
    """True when one frame's facing passes the per-frame gates of body-orientation: a lifted
    row (method :data:`LIFTED`), a finite facing_deg, ``facing_conf >= min_conf``, a finite
    ``facing_sigma_deg < max_sigma_deg``, the player's box at least ``min_box_px`` tall, and
    no front-back reversal (:func:`facing_reversed`, from the frame's 2D trunk angle
    ``image_trunk_deg`` and the facing's ``image_dx``). gbf.measure's body-orientation and
    gbf.render's facing arrow both use it."""
    return (
        bool(lifted)
        and math.isfinite(facing_deg)
        and facing_conf >= gates.min_conf
        and math.isfinite(facing_sigma_deg)
        and facing_sigma_deg < gates.max_sigma_deg
        and box_height_px >= gates.min_box_px
        and not facing_reversed(image_trunk_deg, image_dx)
    )


def image_trunk_deg(keypoints: Any) -> float:
    """The 2D trunk angle of one pose's (26, 3) keypoints, degrees from image up (+ to the
    right), as gbf.lift3d.image_trunk_lean_deg gives it; NaN without both hips and shoulders."""
    return float(image_trunk_lean_deg(np.asarray(keypoints, dtype=np.float64))[0])


def _chord_frames(params: MeasureParams, rate: Fraction) -> tuple[int, float]:
    """(half, seconds): a peak-speed chord joins the frames centre - half and centre + half,
    2 * half frames (``seconds``) apart, half = chord_half_frames(rate,
    peak_speed_window_s) as in gbf.kinematics."""
    half = chord_half_frames(rate, params.peak_speed_window_s)
    return half, float(Fraction(2 * half) / rate)


def metric_label(
    metric_id: str, params: MeasureParams = DEFAULT_PARAMS, rate: Fraction | None = None
) -> str:
    """The Chinese label of a metric, with the windows of ``params`` for the metrics whose
    label states one (:data:`METRICS` holds the labels of :data:`DEFAULT_PARAMS` at 25 fps).
    With ``rate``, the peak-speed label states the duration of the chord actually measured
    (2 * chord_half_frames frames: 0.42 s for 0.4 s at 24 fps), else peak_speed_window_s.
    Labels stay within :data:`CARD_LABEL_MAX_CHARS` characters."""
    spec = METRICS.get(metric_id)
    if spec is None:
        raise MeasureError("metric_label", [f"unknown metric {metric_id!r}"])
    chord_s = float(params.peak_speed_window_s)
    if rate is not None:
        chord_s = _chord_frames(params, rate)[1]
    label = _window_label(metric_id, params.key_half_window, chord_s)
    return spec.label_zh if label is None else label


# --------------------------------------------------------------------------- context


@dataclass(frozen=True)
class MeasureContext:
    """Facts every record carries: rate, restrictions, input hashes, code and registry.

    Attributes:
        fps: exact frame rate ("25/1", a Fraction or an int).
        restrictions: union of the inputs' restriction tags (ADR 0002).
        video_sha256: sha256 of the canonical video.
        inputs: (name, sha256) of every run artifact the records are computed from; names
            are file names or run-relative POSIX paths, never absolute paths.
        code: {"gbf_version", "git_commit", "git_dirty"}.
        registry: metric registry (:func:`load_metric_registry`).
        error_receipts: metric_id -> {"kind", "receiptId", "value", "ci95", "dataset",
            "conditions"} for metrics whose error a receipt measured; every other metric's
            error is "unmeasured".
        attack_direction_known: False when gbf.side_check could not confirm the tracking
            plugin's left/right team names: the goal each team attacks is unknown, and the
            metrics that need it are withheld (``attack_direction_unknown``).
    """

    fps: Fraction | str | int
    restrictions: tuple[str, ...]
    video_sha256: str
    inputs: tuple[tuple[str, str], ...]
    code: Mapping[str, Any]
    registry: Mapping[str, RegistryMetric]
    error_receipts: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    attack_direction_known: bool = True

    def __post_init__(self) -> None:
        problems: list[str] = []
        rate: Fraction | None = None
        fps = self.fps
        if isinstance(fps, bool):
            problems.append(f"fps: expected a rate such as '25/1', got {fps!r}")
        elif isinstance(fps, int):
            if fps > 0:
                rate = Fraction(fps)
            else:
                problems.append(f"fps must be positive, got {fps}")
        else:
            try:
                rate = parse_rate(fps)
            except ValueError as exc:
                problems.append(f"fps: {exc}")
        tags = sorted(set(self.restrictions))
        problems.extend(
            f"restrictions {problem}"
            for problem in contracts.def_problems(tags, "restrictions", "restrictions")
        )
        if not (
            isinstance(self.video_sha256, str)
            and len(self.video_sha256) == 64
            and all(char in "0123456789abcdef" for char in self.video_sha256)
        ):
            problems.append("video_sha256 must be 64 lower-case hex digits")
        inputs = []
        for position, item in enumerate(self.inputs):
            try:
                name, digest = item
            except (TypeError, ValueError):
                problems.append(f"inputs[{position}] must be a (name, sha256) pair")
                continue
            if not isinstance(name, str) or not name or "\\" in name or name.startswith("/"):
                problems.append(f"inputs[{position}]: name must be a relative POSIX name")
            if not (isinstance(digest, str) and len(digest) == 64):
                problems.append(f"inputs[{position}]: sha256 must be 64 hex digits")
            inputs.append((str(name), str(digest)))
        code = dict(self.code) if isinstance(self.code, Mapping) else {}
        if set(code) != {"gbf_version", "git_commit", "git_dirty"}:
            problems.append("code must hold exactly gbf_version, git_commit and git_dirty")
        if not isinstance(self.registry, Mapping):
            problems.append("registry must be a mapping (load_metric_registry)")
        if not isinstance(self.attack_direction_known, bool):
            problems.append(
                f"attack_direction_known must be a bool, got {self.attack_direction_known!r}"
            )
        if problems:
            raise MeasureError("MeasureContext", problems)
        object.__setattr__(self, "fps", rate)
        object.__setattr__(self, "restrictions", tuple(tags))
        object.__setattr__(self, "inputs", tuple(sorted(set(inputs))))
        object.__setattr__(self, "code", MappingProxyType(code))
        object.__setattr__(
            self, "error_receipts", MappingProxyType(dict(self.error_receipts or {}))
        )

    @property
    def rate(self) -> Fraction:
        return Fraction(self.fps)  # normalised to a Fraction in __post_init__


# --------------------------------------------------------------------------- records


def _json_value(value: Any) -> Any:
    """``value`` as plain JSON: numpy scalars unwrapped, floats rounded, NaN/inf as None."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        if not math.isfinite(number):
            return None
        rounded = round(number, FLOAT_DECIMALS)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [_json_value(item) for item in value]
    raise MeasureError("record", [f"value of type {type(value).__name__} is not JSON"])


def format_frame_time(frame: int, fps: Fraction) -> str:
    """Canonical frame as "mm:ss.cc" (hundredths of a second, floored, exact rationals): the
    clock text of every title, summary and table (:func:`gbf.util.clock.format_clock`)."""
    return format_clock(int(frame), fps)


def display_text(value: Any, unit: str, decimals: int | None, fps: Fraction) -> str:
    """The display text of a value: km/h for speeds, a time for frames, an em dash for None."""
    if value is None:
        return UNAVAILABLE_DISPLAY
    if isinstance(value, bool):
        return "是" if value else "否"
    number = float(value)
    places = 1 if decimals is None else decimals
    if unit == "m/s":
        return f"{number * 3.6:.1f} km/h"
    if unit == "m":
        return f"{number:.{places}f} m"
    if unit == "deg":
        return f"{number:.{places}f}°"
    if unit == "count":
        return f"{int(round(number))} 人"
    if unit == "frame":
        return f"第 {int(round(number))} 帧（{format_frame_time(int(round(number)), fps)}）"
    if unit == "box_h":
        return f"{number:.{places}f} 个身高"
    if unit == "bh/s":
        return f"{number:.{places}f} 个身高/秒"
    return f"{number:.{places}f}"


def resolve_tier(
    spec: MetricSpec, registry: Mapping[str, RegistryMetric]
) -> tuple[str, str, RegistryMetric | None]:
    """(tier, tier_source, registry entry) of a metric; refuses blocked and C-tier metrics."""
    entry = registry.get(spec.metric_id)
    if entry is not None:
        if entry.blocked or entry.tier == "C":
            raise MeasureError(
                "resolve_tier", [f"{spec.metric_id} is a blocked C-tier metric (ADR 0004)"]
            )
        if entry.tier not in RECORD_TIERS:
            raise MeasureError("resolve_tier", [f"{spec.metric_id}: unknown tier {entry.tier!r}"])
        return entry.tier, "registry", entry
    if spec.explicit_tier not in EXPLICIT_TIERS:
        raise MeasureError(
            "resolve_tier",
            [
                f"{spec.metric_id} is not in registries/metrics.toml, so its tier must be set "
                f"explicitly to one of {EXPLICIT_TIERS}"
            ],
        )
    return str(spec.explicit_tier), "explicit", None


def _error_block(metric_id: str, context: MeasureContext) -> dict[str, Any]:
    receipt = context.error_receipts.get(metric_id)
    if receipt is None:
        return {
            "kind": "unmeasured",
            "value": None,
            "ci95": None,
            "dataset": None,
            "conditions": None,
            "receiptId": None,
        }
    return {
        "kind": receipt.get("kind", "measured"),
        "value": receipt.get("value"),
        "ci95": list(receipt["ci95"]) if receipt.get("ci95") is not None else None,
        "dataset": receipt.get("dataset"),
        "conditions": receipt.get("conditions"),
        "receiptId": receipt.get("receiptId"),
    }


def evidence_id(record: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``record`` without its evidenceId."""
    return canonical_sha256({key: value for key, value in record.items() if key != "evidenceId"})


@lru_cache(maxsize=1)
def _validator() -> Any:
    """The measurement schema validator, built (and the schema checked) once."""
    return contracts.validator(SCHEMA_NAME)


def record_problems(record: Any) -> list[str]:
    """Every problem of one record: the schema (which has no C tier) and the evidenceId."""
    found = []
    for error in _validator().iter_errors(record):
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        found.append(f"{location}: {error.message}")
    found.sort()
    if isinstance(record, Mapping):
        if "evidenceId" in record and not found:
            try:
                expected = evidence_id(record)
            except (TypeError, ValueError) as exc:
                found.append(f"evidenceId: the record is not canonical JSON ({exc})")
            else:
                if record["evidenceId"] != expected:
                    found.append(f"evidenceId: {record['evidenceId']} != computed {expected}")
    return found


def validate_record(record: Any) -> None:
    """Raise MeasureError listing every problem of one evidence record."""
    found = record_problems(record)
    if found:
        raise MeasureError("validate_record", found)


def build_record(
    metric_id: str,
    *,
    context: MeasureContext,
    status: str,
    value: Any,
    frame: int | None,
    frames: Iterable[int],
    highlight_id: str | None,
    track_id: int | None,
    track_ids: Iterable[int] = (),
    quality: Mapping[str, Any] | None = None,
    coverage: Mapping[str, Any],
    parameters: Mapping[str, Any] | None = None,
    reason_code: str | None = None,
    label_zh: str | None = None,
    reason_zh: str | None = None,
) -> dict[str, Any]:
    """One validated evidence record with its evidenceId.

    The tier comes from the registry when ``metric_id`` is listed there (blocked C-tier
    metrics are refused) and from :data:`METRICS` otherwise. ``status`` "unavailable" needs
    a ``reason_code`` from :data:`REASONS_ZH` and ``value`` None; ``reason_zh`` replaces
    that code's text with one that gives the record's own numbers. ``label_zh`` replaces the
    label of :data:`METRICS` (:func:`metric_label` for other parameters). Raises
    MeasureError.
    """
    problems: list[str] = []
    spec = METRICS.get(metric_id)
    if spec is None:
        raise MeasureError("build_record", [f"unknown metric {metric_id!r}"])
    tier, tier_source, _entry = resolve_tier(spec, context.registry)
    label = spec.label_zh if label_zh is None else label_zh
    if not isinstance(label, str) or not label.strip():
        problems.append(f"label_zh must be a non-empty string, got {label_zh!r}")
    if status not in STATUSES:
        problems.append(f"status must be one of {STATUSES}, got {status!r}")
    if status == "unavailable":
        if value is not None:
            problems.append("an unavailable record has value None")
        if reason_code not in REASONS_ZH:
            problems.append(f"reason_code must be one of {sorted(REASONS_ZH)}, got {reason_code!r}")
        if reason_zh is not None and (not isinstance(reason_zh, str) or not reason_zh.strip()):
            problems.append(f"reason_zh must be a non-empty string, got {reason_zh!r}")
    elif status == "ok":
        if value is None:
            problems.append("an ok record needs a value")
        if reason_code is not None:
            problems.append("an ok record has no reason_code")
        if reason_zh is not None:
            problems.append("an ok record has no reason_zh")
    if problems:
        raise MeasureError(f"build_record({metric_id})", problems)
    clean_value = _json_value(value)
    if isinstance(clean_value, float) and spec.unit in ("count", "frame"):
        clean_value = int(round(clean_value))
    ids: list[int] = []
    for track in [track_id, *track_ids]:
        if track is not None and int(track) >= 0 and int(track) not in ids:
            ids.append(int(track))
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_id": spec.metric_id,
        "label_zh": label,
        "tier": tier,
        "tier_source": tier_source,
        "claimType": spec.claim_type,
        "status": status,
        "value": clean_value,
        "unit": spec.unit,
        "display": display_text(clean_value, spec.unit, spec.decimals, context.rate),
        "frame": None if frame is None else int(frame),
        "frames": sorted({int(f) for f in frames}),
        "highlight_id": highlight_id,
        "track_id": None if track_id is None or int(track_id) < 0 else int(track_id),
        "track_ids": ids,
        "plane": spec.plane,
        "quality": _json_value(dict(quality or {})),
        "error": _json_value(_error_block(spec.metric_id, context)),
        "coverage": _json_value(dict(coverage)),
        "method": {
            "id": spec.method_id,
            "summary": spec.summary,
            "parameters": _json_value(dict(parameters or {})),
            "code": _json_value(dict(context.code)),
        },
        "restrictions": list(context.restrictions),
        "source": {
            "video_sha256": context.video_sha256,
            "fps": format_rate(context.rate),
            "inputs": [{"name": name, "sha256": digest} for name, digest in context.inputs],
        },
    }
    if spec.decimals is not None:
        record["decimals"] = spec.decimals
    if status == "unavailable":
        record["reasonCode"] = reason_code
        record["reason_zh"] = REASONS_ZH[str(reason_code)] if reason_zh is None else reason_zh
    record["evidenceId"] = evidence_id(record)
    validate_record(record)
    return record


def write_measurements(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write records as canonical JSON lines (UTF-8, LF), after validating every one."""
    problems: list[str] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        problems.extend(f"records[{index}]: {problem}" for problem in record_problems(record))
        ident = record.get("evidenceId") if isinstance(record, Mapping) else None
        if isinstance(ident, str):
            if ident in seen:
                problems.append(f"records[{index}]: evidenceId {ident} appears twice")
            seen.add(ident)
    if problems:
        raise MeasureError("write_measurements", problems)
    write_text(path, "".join(canonical_json(dict(record)) + "\n" for record in records))


def read_measurements(path: str | Path) -> list[dict[str, Any]]:
    """Read measurements.jsonl and validate every record (MeasureError lists all problems)."""
    source = Path(path)
    data = source.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MeasureError("read_measurements", [f"{source.name} is not UTF-8: {exc}"]) from exc
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {number}: invalid JSON ({exc.msg})")
            continue
        problems.extend(f"line {number}: {problem}" for problem in record_problems(record))
        records.append(record)
    if problems:
        raise MeasureError("read_measurements", problems)
    return records


# --------------------------------------------------------------------------- input tables


@dataclass(frozen=True)
class MeasureTables:
    """The run tables gbf.measure reads (columns of docs/design/m1-highlights.md).

    Attributes:
        tracks: tracks.parquet (frame, track_id, x1, y1, x2, y2, role, team, foot_u,
            foot_v, pitch_x, pitch_y, in_pitch).
        shots: shots.parquet (shot_id, start_frame, end_frame, median_box_h).
        kinematics: kinematics.parquet (frame, track_id, segment_id, x, y, speed, valid,
            interpolated), or None.
        ball_events: gbf.ball events (frame, kind, magnitude, u, v, nearest_track_id,
            nearest_dist_box_h, speed_before, speed_after, angle_deg, at_foot) of one or
            several profiles; an optional ``profile`` column names them ("strike",
            "contact"). None when there is no ball track.
        pose: pose.parquet (frame, track_id, keypoints, mean_score), or None.
        lift: lift3d.parquet (frame, track_id, method, reason, facing_deg, facing_conf,
            facing_sigma_deg, trunk_lean_deg, trunk_lean_alt_deg, trunk_lean_sigma_deg,
            trunk_tilt_deg, image_lean_deg, image_knee_left_deg, image_knee_right_deg), or
            None.
        pitch_frames: frames on which metres may be measured (signals.parquet ev_pitch);
            None means the frames whose tracks rows are pitch_usable.
        team_verdicts: optional team-check verdicts keyed by (track_id, claimed team).
            Unchecked tracks without ground positions or with cut-off boxes are excluded.
        frame_size: optional (width, height) of the original frames for cut-off boxes.
        calib: optional refined calibration rows (frame_index or frame, C, width, height)
            for boxes cut off past the camera-side pitch boundary.
    """

    tracks: pd.DataFrame
    shots: pd.DataFrame
    kinematics: pd.DataFrame | None = None
    ball_events: pd.DataFrame | None = None
    pose: pd.DataFrame | None = None
    lift: pd.DataFrame | None = None
    pitch_frames: frozenset[int] | None = None
    team_verdicts: Mapping[tuple[int, str], str] | None = None
    frame_size: tuple[int, int] | None = None
    calib: pd.DataFrame | None = None


_TRACK_COLUMNS = (
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
    "pitch_x",
    "pitch_y",
    "in_pitch",
)
_SHOT_COLUMNS = ("shot_id", "start_frame", "end_frame")
_KIN_COLUMNS = ("frame", "track_id", "segment_id", "x", "y", "speed", "valid", "interpolated")
_EVENT_COLUMNS = (
    "frame",
    "kind",
    "magnitude",
    "u",
    "v",
    "speed_before",
    "speed_after",
)
_POSE_COLUMNS = ("frame", "track_id", "keypoints", "mean_score")
_LIFT_COLUMNS = (
    "frame",
    "track_id",
    "method",
    "reason",
    "facing_deg",
    "facing_conf",
    "facing_sigma_deg",
    "trunk_lean_deg",
    "trunk_lean_alt_deg",
    "trunk_lean_sigma_deg",
    "trunk_tilt_deg",
    "image_lean_deg",
    "image_knee_left_deg",
    "image_knee_right_deg",
)


def _missing(name: str, table: Any, columns: Sequence[str], problems: list[str]) -> None:
    if table is None:
        return
    if not isinstance(table, pd.DataFrame):
        problems.append(f"{name} must be a pandas DataFrame or None, got {type(table).__name__}")
        return
    absent = [column for column in columns if column not in table.columns]
    if absent:
        problems.append(f"{name} lacks column(s) {', '.join(absent)}")


def _text(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    return text if text and text.lower() not in ("nan", "none", "<na>") else None


@dataclass(frozen=True)
class _Player:
    track_id: int
    role: str | None
    team: str | None
    box: tuple[float, float, float, float]
    foot: tuple[float, float]
    pitch: tuple[float, float] | None

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


class _Index:
    """Per-frame lookups built once for all highlights of a run."""

    def __init__(self, tables: MeasureTables) -> None:
        problems: list[str] = []
        if not isinstance(tables, MeasureTables):
            raise MeasureError("measure", [f"tables must be MeasureTables, got {tables!r}"])
        if not isinstance(tables.tracks, pd.DataFrame):
            problems.append("tracks is required")
        if not isinstance(tables.shots, pd.DataFrame):
            problems.append("shots is required")
        _missing("tracks", tables.tracks, _TRACK_COLUMNS, problems)
        _missing("shots", tables.shots, _SHOT_COLUMNS, problems)
        _missing("kinematics", tables.kinematics, _KIN_COLUMNS, problems)
        _missing("ball_events", tables.ball_events, _EVENT_COLUMNS, problems)
        _missing("pose", tables.pose, _POSE_COLUMNS, problems)
        _missing("lift", tables.lift, _LIFT_COLUMNS, problems)
        if problems:
            raise MeasureError("measure", problems)

        shots = tables.shots.sort_values("start_frame", kind="stable")
        self.shot_start = shots["start_frame"].to_numpy(dtype=np.int64)
        self.shot_end = shots["end_frame"].to_numpy(dtype=np.int64)
        self.shot_id = shots["shot_id"].to_numpy(dtype=np.int64)
        if "median_box_h" in shots.columns:
            self.shot_box_h = shots["median_box_h"].to_numpy(dtype=np.float64)
        else:
            self.shot_box_h = np.full(self.shot_id.size, np.nan)
        if self.shot_start.size == 0:
            raise MeasureError("measure", ["shots is empty"])
        if np.any(self.shot_start[1:] != self.shot_end[:-1] + 1) or self.shot_start[0] != 0:
            raise MeasureError("measure", ["shots must tile 0..N-1 without gaps or overlaps"])
        self.n_frames = int(self.shot_end[-1]) + 1

        self.players: dict[int, list[_Player]] = {}
        self.unsafe: set[tuple[int, int]] = set()
        tracks = tables.tracks
        columns = {
            name: tracks[name].to_numpy()
            for name in ("frame", "track_id", "x1", "y1", "x2", "y2", "foot_u", "foot_v")
        }
        pitch_x = tracks["pitch_x"].to_numpy(dtype=np.float64)
        pitch_y = tracks["pitch_y"].to_numpy(dtype=np.float64)
        in_pitch = tracks["in_pitch"].to_numpy(dtype=bool)
        roles = tracks["role"].to_numpy(dtype=object)
        teams = tracks["team"].to_numpy(dtype=object)
        cameras = {}
        if tables.calib is not None:
            key = "frame_index" if "frame_index" in tables.calib else "frame"
            cameras = {int(row[key]): row for row in tables.calib.to_dict("records")}
        for row in range(len(tracks)):
            role = _text(roles[row])
            if role not in PLAYER_ROLES:
                continue
            track = int(columns["track_id"][row])
            if track < 0:
                continue
            ground = None
            if in_pitch[row] and math.isfinite(pitch_x[row]) and math.isfinite(pitch_y[row]):
                ground = (float(pitch_x[row]), float(pitch_y[row]))
            team = _text(teams[row])
            frame = int(columns["frame"][row])
            box = tuple(float(columns[name][row]) for name in ("x1", "y1", "x2", "y2"))
            camera = cameras.get(frame, {})
            size = tables.frame_size
            if size is None and camera.get("width") and camera.get("height"):
                size = (int(camera["width"]), int(camera["height"]))
            unchecked = (
                tables.team_verdicts is not None
                and tables.team_verdicts.get((track, team or ""), "unchecked") != "consistent"
            )
            if unchecked and (ground is None or box_touches_edge(box, size)):
                self.unsafe.add((frame, track))
                continue
            if off_pitch_at_camera_edge(box, ground, size, camera.get("C")):
                self.unsafe.add((frame, track))
                continue
            self.players.setdefault(frame, []).append(
                _Player(
                    track_id=track,
                    role=role,
                    team=team if team in TEAMS else None,
                    box=box,
                    foot=(float(columns["foot_u"][row]), float(columns["foot_v"][row])),
                    pitch=ground,
                )
            )
        if tables.pitch_frames is not None:
            self.pitch = frozenset(int(f) for f in tables.pitch_frames)
        else:
            usable = tracks.loc[tracks["pitch_usable"].to_numpy(dtype=bool), "frame"]
            self.pitch = frozenset(int(f) for f in usable.to_numpy())
        # pitch frames on which each player has a ground position, sorted
        grounded: dict[int, list[int]] = {}
        for frame, players in self.players.items():
            if frame in self.pitch:
                for player in players:
                    if player.pitch is not None:
                        grounded.setdefault(player.track_id, []).append(frame)
        self.ground_frames = {
            track: np.unique(np.asarray(frames, dtype=np.int64))
            for track, frames in grounded.items()
        }

        self.kin: dict[int, pd.DataFrame] = {}
        self.segment_length: dict[tuple[int, int], int] = {}
        if tables.kinematics is not None and len(tables.kinematics):
            kin = tables.kinematics
            counts = kin.groupby(["track_id", "segment_id"]).size()
            self.segment_length = {(int(t), int(s)): int(n) for (t, s), n in counts.items()}
            for track, group in kin.groupby("track_id", sort=True):
                self.kin[int(track)] = group.sort_values("frame", kind="stable")

        self.pose: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
        if tables.pose is not None:
            for frame, track, points, mean in zip(
                tables.pose["frame"].to_numpy(),
                tables.pose["track_id"].to_numpy(),
                tables.pose["keypoints"],
                tables.pose["mean_score"].to_numpy(dtype=np.float64),
                strict=True,
            ):
                array = np.asarray(points, dtype=np.float64).reshape(-1)
                if array.size == N_KEYPOINTS * 3 and int(track) >= 0:
                    self.pose[(int(frame), int(track))] = (array.reshape(N_KEYPOINTS, 3), mean)

        self.lift: dict[tuple[int, int], dict[str, Any]] = {}
        if tables.lift is not None:
            for row in tables.lift[list(_LIFT_COLUMNS)].to_dict("records"):
                if int(row["track_id"]) >= 0:
                    self.lift[(int(row["frame"]), int(row["track_id"]))] = row

        self._ground_maps: dict[int, np.ndarray | None] = {}

        self.events = pd.DataFrame(columns=list(_EVENT_COLUMNS))
        if tables.ball_events is not None and len(tables.ball_events):
            events = tables.ball_events.copy()
            if "profile" not in events.columns:
                events["profile"] = "strike"
            priority = events["profile"].map({"strike": 0, "contact": 1}).fillna(2)
            events = events.assign(_priority=priority.to_numpy())
            events = events.sort_values(["frame", "_priority"], kind="stable")
            events = events.drop_duplicates("frame", keep="first").drop(columns="_priority")
            self.events = events.reset_index(drop=True)

    def shot_index(self, frame: int) -> int:
        position = int(np.searchsorted(self.shot_start, frame, side="right")) - 1
        return max(position, 0)

    def same_shot_frames(self, centre: int, half: int) -> list[int]:
        shot = self.shot_index(centre)
        lo = max(int(self.shot_start[shot]), centre - half)
        hi = min(int(self.shot_end[shot]), centre + half)
        return list(range(lo, hi + 1))

    def player(self, frame: int, track: int) -> _Player | None:
        for player in self.players.get(frame, ()):
            if player.track_id == track:
                return player
        return None

    def team_of(self, track: int, frames: Iterable[int]) -> str | None:
        votes = Counter(
            player.team
            for frame in frames
            if (player := self.player(frame, track)) is not None and player.team is not None
        )
        if not votes:
            return None
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        return ranked[0][0]

    def ground_map(self, frame: int) -> Callable[[np.ndarray], np.ndarray | None] | None:
        """The frame's map from the grass (SoccerNet x, y) to its pixels: the homography of
        its players' ground positions and foot points (tracks.parquet maps the foot points to
        the grass with the frame's calibration), or None (:func:`_fit_homography`)."""
        if frame not in self._ground_maps:
            grounded = [p for p in self.players.get(frame, ()) if p.pitch is not None]
            self._ground_maps[frame] = _fit_homography(
                np.array([p.pitch for p in grounded], dtype=np.float64).reshape(-1, 2),
                np.array([p.foot for p in grounded], dtype=np.float64).reshape(-1, 2),
            )
        homography = self._ground_maps[frame]
        if homography is None:
            return None

        def to_image(points: np.ndarray) -> np.ndarray | None:
            ground = np.column_stack([points, np.ones(len(points))])
            mapped = ground @ homography.T
            if np.any(mapped[:, 2] <= 0.0):
                return None
            return mapped[:, :2] / mapped[:, 2:3]

        return to_image


def _normaliser(points: np.ndarray) -> np.ndarray | None:
    """The similarity that moves ``points`` to mean 0 and mean distance sqrt(2) (Hartley)."""
    centre = points.mean(axis=0)
    spread = float(np.sqrt(((points - centre) ** 2).sum(axis=1)).mean())
    if not math.isfinite(spread) or spread <= 0.0:
        return None
    scale = math.sqrt(2.0) / spread
    return np.array(
        [[scale, 0.0, -scale * centre[0]], [0.0, scale, -scale * centre[1]], [0.0, 0.0, 1.0]]
    )


def _fit_homography(ground: np.ndarray, pixels: np.ndarray) -> np.ndarray | None:
    """The homography ``H`` with ``pixels ~ H [x, y, 1]`` of the (n, 2) correspondences, a
    direct linear fit on normalised points; None with fewer than four, with points that lie
    near a line, or when they do not fit one homography to a pixel (then they did not come
    from one camera)."""
    if len(ground) < 4 or not (np.isfinite(ground).all() and np.isfinite(pixels).all()):
        return None
    for points in (ground, pixels):
        spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if spread[1] <= 1e-3 * spread[0]:
            return None
    to_unit, from_pixels = _normaliser(ground), _normaliser(pixels)
    if to_unit is None or from_pixels is None:
        return None
    ones = np.ones(len(ground))
    g = np.column_stack([ground, ones]) @ to_unit.T
    p = np.column_stack([pixels, ones]) @ from_pixels.T
    rows = []
    for (x, y, _), (u, v, _) in zip(g, p, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    homography = np.linalg.inv(from_pixels) @ vt[-1].reshape(3, 3) @ to_unit
    mapped = np.column_stack([ground, ones]) @ homography.T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        return None
    if mapped[0, 2] < 0.0:
        homography, mapped = -homography, -mapped  # points in front of the camera map to w > 0
    residual = np.hypot(*(mapped[:, :2] / mapped[:, 2:3] - pixels).T)
    if not np.isfinite(residual).all() or float(residual.max()) > 1.0:
        return None
    return homography


# --------------------------------------------------------------------------- contacts


@dataclass(frozen=True)
class ContactMatch:
    """A ball event matched to a player's foot.

    Attributes:
        frame: canonical frame of the event (the velocity vertex).
        track_id: the player.
        method: "foot_keypoint" (pose) or "foot_point_fallback" (box bottom centre).
        distance_box_h: ball centre to the matched foot point, in the player's box heights.
        keypoint: name of the nearest foot keypoint ("" for the fallback).
        box_h_px: the player's box height in pixels.
        event: the event row (frame, kind, magnitude, u, v, speed_before, speed_after, ...).
    """

    frame: int
    track_id: int
    method: str
    distance_box_h: float
    keypoint: str
    box_h_px: float
    event: Mapping[str, Any]


def _foot_distance(
    pose: np.ndarray, u: float, v: float, box_h: float, min_score: float
) -> tuple[float, str] | None:
    best: tuple[float, str] | None = None
    for index in _FOOT_INDEX:
        pu, pv, score = pose[index]
        if not (math.isfinite(pu) and math.isfinite(pv) and score >= min_score):
            continue
        distance = math.hypot(pu - u, pv - v) / box_h
        name = HALPE26_KEYPOINTS[index]
        if best is None or distance < best[0]:
            best = (distance, name)
    return best


def contact_candidates(
    frame: int, u: float, v: float, players: Sequence[_Player], params: MeasureParams
) -> list[_Player]:
    """Players nearest the ball (foot point, in their box heights), nearest first."""
    scored = []
    for player in players:
        if player.height <= 0:
            continue
        distance = math.hypot(player.foot[0] - u, player.foot[1] - v) / player.height
        if distance <= params.contact_candidate_max_bh:
            scored.append((distance, player.track_id, player))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [player for _, _, player in scored[: params.contact_candidates]]


def _match_event(
    event: Mapping[str, Any], index: _Index, params: MeasureParams
) -> ContactMatch | None:
    frame = int(event["frame"])
    u, v = float(event["u"]), float(event["v"])
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    candidates = contact_candidates(frame, u, v, index.players.get(frame, ()), params)
    with_pose: list[tuple[float, int, str, _Player]] = []
    without_pose: list[tuple[float, int, _Player]] = []
    for player in candidates:
        entry = index.pose.get((frame, player.track_id))
        if entry is not None and entry[1] >= params.pose_min_mean_score:
            found = _foot_distance(entry[0], u, v, player.height, params.keypoint_min_score)
            if found is not None:
                with_pose.append((found[0], player.track_id, found[1], player))
                continue
        distance = math.hypot(player.foot[0] - u, player.foot[1] - v) / player.height
        without_pose.append((distance, player.track_id, player))
    with_pose.sort(key=lambda item: (item[0], item[1]))
    if with_pose and with_pose[0][0] <= params.contact_radius_bh:
        distance, track, keypoint, player = with_pose[0]
        return ContactMatch(
            frame, track, "foot_keypoint", distance, keypoint, player.height, dict(event)
        )
    without_pose.sort(key=lambda item: (item[0], item[1]))
    if without_pose and without_pose[0][0] <= params.contact_fallback_bh:
        distance, track, player = without_pose[0]
        return ContactMatch(
            frame, track, "foot_point_fallback", distance, "", player.height, dict(event)
        )
    return None


def _match_window(
    index: _Index, start: int, end: int, params: MeasureParams
) -> tuple[list[ContactMatch], int]:
    events = index.events
    inside = events[(events["frame"] >= start) & (events["frame"] <= end)]
    matches = []
    for event in inside.to_dict("records"):
        match = _match_event(event, index, params)
        if match is not None:
            matches.append(match)
    return matches, len(inside)


def match_contacts(
    tables: MeasureTables, start: int, end: int, params: MeasureParams = DEFAULT_PARAMS
) -> tuple[list[ContactMatch], int]:
    """Ball events in [start, end] matched to a player's foot (see :class:`ContactMatch`),
    in frame order, and the number of events examined.

    A ball event is matched to the candidate (one of the
    :attr:`MeasureParams.contact_candidates` players nearest the ball) whose confident foot
    keypoint lies nearest the ball centre, within ``contact_radius_bh`` of that player's box
    height. When none matches that way, the nearest candidate WITHOUT a usable pose is
    tried with its foot point (box bottom centre) and ``contact_fallback_bh``: a missing
    pose falls back to the box, a pose that shows the feet away from the ball does not.
    """
    return _match_window(_Index(tables), int(start), int(end), params)


def pose_requests(
    highlights: Sequence[Mapping[str, Any]],
    tracks: pd.DataFrame,
    ball_events: pd.DataFrame | None,
    params: MeasureParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """The boxes gbf.measure needs a 2D pose for (frame, track_id, x1, y1, x2, y2).

    Involved players (``track_ids`` and the joined shot's players) on every frame of their
    highlight's window, plus the :attr:`MeasureParams.contact_candidates` players nearest
    the ball on every ball-event frame inside a window. Only players and goalkeepers with a
    track (track_id >= 0); one row per (frame, track_id), sorted.
    """
    problems: list[str] = []
    _missing("tracks", tracks, ("frame", "track_id", "x1", "y1", "x2", "y2", "role"), problems)
    if ball_events is not None:
        _missing("ball_events", ball_events, ("frame", "u", "v"), problems)
    if problems:
        raise MeasureError("pose_requests", problems)
    people = tracks.loc[
        tracks["role"].isin(PLAYER_ROLES).to_numpy(dtype=bool)
        & (tracks["track_id"].to_numpy() >= 0)
    ]
    wanted: set[tuple[int, int]] = set()
    by_frame = {int(frame): group for frame, group in people.groupby("frame", sort=True)}
    for highlight in highlights:
        start, end = int(highlight["start_frame"]), int(highlight["end_frame"])
        involved = {int(t) for t in highlight.get("track_ids") or [] if int(t) >= 0}
        joined = ((highlight.get("cues") or {}).get("cut_emphasis") or {}).get("joined_shot_event")
        if isinstance(joined, Mapping):
            involved |= {int(t) for t in joined.get("track_ids") or [] if int(t) >= 0}
        for frame in range(start, end + 1):
            group = by_frame.get(frame)
            if group is None:
                continue
            for track in group["track_id"].to_numpy():
                if int(track) in involved:
                    wanted.add((frame, int(track)))
        if ball_events is None:
            continue
        inside = ball_events[(ball_events["frame"] >= start) & (ball_events["frame"] <= end)]
        for frame, u, v in zip(
            inside["frame"].to_numpy(),
            inside["u"].to_numpy(dtype=np.float64),
            inside["v"].to_numpy(dtype=np.float64),
            strict=True,
        ):
            group = by_frame.get(int(frame))
            if group is None or not (math.isfinite(u) and math.isfinite(v)):
                continue
            players = [
                _Player(
                    int(row["track_id"]),
                    None,
                    None,
                    (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                    ((float(row["x1"]) + float(row["x2"])) / 2.0, float(row["y2"])),
                    None,
                )
                for row in group.to_dict("records")
            ]
            for player in contact_candidates(int(frame), u, v, players, params):
                wanted.add((int(frame), player.track_id))
    keys = pd.DataFrame(sorted(wanted), columns=["frame", "track_id"]).astype(np.int64)
    boxes = people.drop_duplicates(["frame", "track_id"], keep="first")[
        ["frame", "track_id", "x1", "y1", "x2", "y2"]
    ].astype({"frame": np.int64, "track_id": np.int64})
    result = keys.merge(boxes, on=["frame", "track_id"], how="inner", validate="one_to_one")
    return result.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- measuring


def _wrap_deg(angle: float) -> float:
    wrapped = math.fmod(angle + 180.0, 360.0)
    if wrapped <= 0.0:
        wrapped += 360.0
    return wrapped - 180.0


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


@dataclass
class _Context:
    highlight: Mapping[str, Any]
    hid: str
    start: int
    end: int
    key: int
    index: _Index
    context: MeasureContext
    params: MeasureParams
    records: list[dict[str, Any]]

    def coverage(self, considered: int, used: int) -> dict[str, Any]:
        share = None if considered == 0 else min(1.0, used / considered)
        return {
            "window": [self.start, self.end],
            "frames_considered": int(considered),
            "frames_used": int(used),
            "share": share,
        }

    def emit(
        self,
        metric_id: str,
        *,
        value: Any,
        frame: int | None,
        frames: Iterable[int],
        track_id: int | None,
        track_ids: Iterable[int] = (),
        quality: Mapping[str, Any] | None = None,
        considered: int = 0,
        used: int = 0,
        reason: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        reason_zh: str | None = None,
    ) -> None:
        status = "unavailable" if reason is not None else "ok"
        self.records.append(
            build_record(
                metric_id,
                context=self.context,
                status=status,
                value=None if reason is not None else value,
                frame=frame,
                frames=frames,
                highlight_id=self.hid,
                track_id=track_id,
                track_ids=track_ids,
                quality=quality,
                coverage=self.coverage(considered, used),
                parameters=parameters,
                reason_code=reason,
                label_zh=metric_label(metric_id, self.params, self.context.rate),
                reason_zh=reason_zh,
            )
        )


def _present_near(index: _Index, track: int, centre: int, half: int) -> list[int]:
    return [f for f in index.same_shot_frames(centre, half) if index.player(f, track) is not None]


def _reference_frame(ctx: _Context, track: int) -> int | None:
    """The key frame if the player is on screen near it, else the joined moment's frame."""
    half = ctx.params.key_half_window
    if _present_near(ctx.index, track, ctx.key, half):
        return ctx.key
    joined = ((ctx.highlight.get("cues") or {}).get("cut_emphasis") or {}).get("joined_shot_event")
    if isinstance(joined, Mapping) and isinstance(joined.get("frame"), int):
        ids = [int(t) for t in joined.get("track_ids") or []]
        frame = int(joined["frame"])
        if track in ids and _present_near(ctx.index, track, frame, half):
            return frame
    return None


def _check_chord(
    kin: pd.DataFrame, track: int, centre: int, half: int, seconds: float, speed: float
) -> None:
    """Raise MeasureError unless ``speed``, the kinematics speed of ``track`` at ``centre``,
    is the chord of its smoothed positions ``half`` frames before and after it in the same
    segment over ``seconds``: the record states that chord, so the table must have been
    computed with it (speed_chord_s equal to peak_speed_window_s)."""
    rows = kin.set_index("frame")
    found = all(frame in rows.index for frame in (centre - half, centre, centre + half))
    problem = None
    if not found:
        problem = f"has no rows at frames {centre - half} and {centre + half} for its chord"
    else:
        first, middle, last = (rows.loc[f] for f in (centre - half, centre, centre + half))
        same = int(first["segment_id"]) == int(middle["segment_id"]) == int(last["segment_id"])
        chord = math.hypot(last["x"] - first["x"], last["y"] - first["y"]) / seconds
        if not same:
            problem = "spans two segments"
        elif not math.isclose(chord, speed, rel_tol=1e-6, abs_tol=1e-6):
            problem = f"is {speed:.4f} m/s, but the {2 * half}-frame chord is {chord:.4f} m/s"
    if problem is not None:
        raise MeasureError(
            "peak-speed",
            [
                f"the kinematics speed of track {track} at frame {centre} {problem}; "
                "kinematics.parquet must be computed with speed_chord_s equal to "
                "peak_speed_window_s"
            ],
        )


def _ground(index: _Index, frame: int, track: int) -> tuple[float, float] | None:
    """The player's measured ground position on ``frame`` (tracks.parquet), or None."""
    player = index.player(frame, track)
    return None if player is None else player.pitch


_COVERAGE_REASONS: Mapping[str, str] = MappingProxyType(
    {
        "split": "speed_coverage_split",
        "rejected": "speed_coverage_rejected",
        "short_segment": "speed_coverage_short",
        "implausible": "speed_coverage_implausible",
    }
)
"""The reasonCode of a peak-speed withheld for low coverage, by the commonest cause of the
frames without a valid speed (:func:`_uncovered_causes`)."""


def _natural_margins(ctx: _Context, track: int, kin: pd.DataFrame, margin: int) -> set[int]:
    """Frames that no valid speed can cover however clean the track: those within ``margin``
    frames (the filter half-window plus the chord half) of the player's first and last
    tracked frame in each shot the window touches, where the track starts or ends or the
    camera cuts."""
    index = ctx.index
    frames = kin["frame"].to_numpy(dtype=np.int64)
    ground = index.ground_frames.get(track, np.zeros(0, dtype=np.int64))
    result: set[int] = set()
    for shot in range(index.shot_index(ctx.start), index.shot_index(ctx.end) + 1):
        lo, hi = int(index.shot_start[shot]), int(index.shot_end[shot])
        inside = np.concatenate(
            [frames[(frames >= lo) & (frames <= hi)], ground[(ground >= lo) & (ground <= hi)]]
        )
        if inside.size:
            first, last = int(inside.min()), int(inside.max())
            result.update(range(first, first + margin))
            result.update(range(last - margin + 1, last + 1))
    return result


def _uncovered_causes(
    kin: pd.DataFrame, uncovered: set[int], min_frames: int, margin: int, half: int
) -> dict[str, int]:
    """Why each of the ``uncovered`` frames (tracked, no valid speed, natural margins left
    out) has no valid speed, counted by cause: "rejected" (its position, or an end of its
    chord, is missing or rejected and bridged, or lies outside every segment), "short_segment"
    (its segment is shorter than ``min_frames``), "split" (it lies within ``margin`` frames
    of a segment end where the track was split) or "implausible" (the speed, or a filter
    derivative near its chord ends, is above the limit)."""
    counts = dict.fromkeys(_COVERAGE_REASONS, 0)
    rows = kin.set_index("frame")
    segments = kin.groupby("segment_id")["frame"].agg(["min", "max", "size"])
    for frame in sorted(uncovered):
        if frame not in rows.index:
            counts["rejected"] += 1
            continue
        row = rows.loc[frame]
        first, last, size = segments.loc[int(row["segment_id"])]
        ends = [f for f in (frame - half, frame + half) if f in rows.index]
        if size < min_frames:
            counts["short_segment"] += 1
        elif bool(row["interpolated"]) or any(bool(rows.loc[f, "interpolated"]) for f in ends):
            counts["rejected"] += 1
        elif frame - first < margin or last - frame < margin:
            counts["split"] += 1
        else:
            counts["implausible"] += 1
    return counts


def _hidden_peak(
    ctx: _Context,
    kin: pd.DataFrame,
    tracked: set[int],
    natural: set[int],
    centres: np.ndarray,
    speeds: np.ndarray,
    half: int,
) -> dict[str, Any] | None:
    """Evidence that the real peak may lie in a stretch of the window without valid speeds,
    or None (module docstring); ``centres`` and ``speeds`` are the valid speeds. A tracked
    frame is hidden when no valid chord spans it and it is no natural margin. The peak is
    next to such a stretch when it lies within ``half`` frames (half a chord) of an end of
    its run of valid speeds and the frames beyond that end hold a hidden frame before the
    next chord spans one; a speed reached again away from such ends (a plateau) is no
    peak at an edge. A stretch of the window that no
    chord spans, at least a chord long, with a hidden frame and a gap of more than
    max_gap_frames missing or rejected positions, is crossed at the mean speed between the
    positions at its ends (the last and first chord ends around it); faster than the peak by
    more than PEAK_SPEED_GAP_MARGIN hides it too. A split that a jump test made across a
    shorter gap is no such stretch: its two sides need not be one player."""
    value = float(np.max(speeds))
    usable = {int(c) for c in centres}
    spanned: set[int] = set()
    for chord in usable:
        spanned.update(range(chord - half, chord + half + 1))
    hidden = {f for f in tracked if f not in spanned and f not in natural}
    if not hidden:
        return None

    def borders(frames: Iterable[int]) -> bool:
        for frame in frames:
            if frame in spanned:
                return False
            if frame in hidden:
                return True
        return False

    def edge(centre: int) -> str | None:
        first = last = centre
        while first - 1 in usable:
            first -= 1
        while last + 1 in usable:
            last += 1
        if centre - first <= half and borders(range(first - half - 1, ctx.start - 1, -1)):
            return "before"
        if last - centre <= half and borders(range(last + half + 1, ctx.end + 1)):
            return "after"
        return None

    found: dict[str, Any] = {}
    sides = [edge(int(c)) for c in centres[speeds >= value - 1e-6]]
    if all(sides):
        found["peak_next_to_uncovered"] = sides[0]
    rows = kin.set_index("frame")
    present = rows.index.to_numpy(dtype=np.int64)
    inside = sorted(f for f in spanned if ctx.start <= f <= ctx.end)
    fastest: tuple[float, int, int] | None = None
    for a, b in zip(inside[:-1], inside[1:], strict=True):
        if b - a < 2 * half or not any(f in hidden for f in range(a + 1, b)):
            continue
        kept = present[(present >= a) & (present <= b)]
        if int(np.max(np.diff(kept))) - 1 <= ctx.params.max_gap_frames:
            continue
        start, stop = rows.loc[a], rows.loc[b]
        seconds = float(Fraction(b - a) / ctx.context.rate)
        across = math.hypot(stop["x"] - start["x"], stop["y"] - start["y"]) / seconds
        if fastest is None or across > fastest[0]:
            fastest = (across, a, b)
    if fastest is not None:
        found["speed_across_uncovered_mps"] = fastest[0]
        found["uncovered_between_frames"] = [fastest[1], fastest[2]]
    faster = fastest is not None and fastest[0] > value * (1.0 + PEAK_SPEED_GAP_MARGIN)
    return found if "peak_next_to_uncovered" in found or faster else None


def withheld_speed_reason_zh(reason: str, quality: Mapping[str, Any], rate: Fraction) -> str:
    """The ``reason_zh`` of a peak-speed record withheld as ``reason``, with its own numbers
    from ``quality``: for low coverage, how many of the window's frames carry a trusted
    speed and why the others do not; for a hidden peak, where the real peak may lie. Either
    way the highest trusted speed is only a lower bound, written as the table writes
    speeds. Other reasons keep their text (:data:`REASONS_ZH`)."""
    if reason not in _COVERAGE_CAUSES_ZH and reason != "speed_peak_hidden":
        return REASONS_ZH[reason]
    highest = float(quality["highest_valid_speed_mps"])
    speed = display_text(_json_value(highest), "m/s", 1, rate)
    frame = int(quality["highest_valid_speed_frame"])
    if reason in _COVERAGE_CAUSES_ZH:
        used = int(quality["speed_frames"])
        considered = int(quality["tracked_frames"]) - int(quality["margin_frames"])
        return (
            f"可信速度只覆盖 {used}/{considered} 帧，{_COVERAGE_CAUSES_ZH[reason]}："
            f"测得的最高 {speed}（第 {frame} 帧）只是下限，不作为峰值"
        )
    clauses = []
    if quality.get("peak_next_to_uncovered"):
        clauses.append("紧挨着一段测不出速度的轨迹")
    across = quality.get("speed_across_uncovered_mps")
    if across is not None and float(across) > highest * (1.0 + PEAK_SPEED_GAP_MARGIN):
        first, last = quality["uncovered_between_frames"]
        clauses.append(f"比跨过第 {first}–{last} 帧这段测不出速度的轨迹时的平均速度慢")
    return (
        f"测得的最高可信速度 {speed}（第 {frame} 帧）{'，也'.join(clauses)}："
        f"真实峰值可能在那段里，{speed} 只是下限，不作为峰值"
    )


def _peak_speed(ctx: _Context, track: int) -> None:
    index, params = ctx.index, ctx.params
    window = range(ctx.start, ctx.end + 1)
    pitch = [f for f in window if f in index.pitch]
    half, seconds = _chord_frames(params, ctx.context.rate)
    margin = (params.sg_window_frames - 1) // 2 + half
    parameters = {
        "min_segment_frames": params.min_segment_frames,
        "max_gap_frames": params.max_gap_frames,
        "sg_window_frames": params.sg_window_frames,
        "peak_speed_window_s": params.peak_speed_window_s,
        "peak_speed_min_coverage": params.peak_speed_min_coverage,
        "peak_speed_gap_margin": PEAK_SPEED_GAP_MARGIN,
        "chord_frames": 2 * half,
        "chord_s": seconds,
    }
    present = [f for f in window if index.player(f, track) is not None]
    if not pitch:
        ctx.emit(
            "peak-speed",
            value=None,
            frame=None,
            frames=[],
            track_id=track,
            considered=len(present),
            reason="no_pitch_frames",
            parameters=parameters,
        )
        return
    kin = index.kin.get(track)
    rows = None
    if kin is not None:
        frames = kin["frame"].to_numpy(dtype=np.int64)
        rows = kin[(frames >= ctx.start) & (frames <= ctx.end)]
        rows = rows[rows["frame"].isin(index.pitch).to_numpy(dtype=bool)]
    if kin is None or rows is None or not len(rows):
        ctx.emit(
            "peak-speed",
            value=None,
            frame=None,
            frames=[],
            track_id=track,
            considered=len(present),
            reason="no_kinematics",
            parameters=parameters,
        )
        return
    # Tracked frames: the window's pitch frames with a measured ground position of the
    # player, or with a kinematics row (a position bridged or rejected inside a segment).
    # Those within the chord margin of the player's first and last tracked frame in a shot
    # can never have a valid speed and are left out of the coverage.
    row_frames = rows["frame"].to_numpy(dtype=np.int64)
    measured = {f for f in pitch if _ground(index, f, track) is not None}
    tracked = measured | set(row_frames.tolist())
    natural = _natural_margins(ctx, track, kin, margin)
    lengths = np.array(
        [index.segment_length.get((track, int(s)), 0) for s in rows["segment_id"].to_numpy()],
        dtype=np.int64,
    )
    speed = rows["speed"].to_numpy(dtype=np.float64)
    # Explicitly unsafe observed people invalidate their speed centres and chord ends.
    # A merely missing row may be a legitimate bridged position and stays eligible.
    usable = (
        rows["valid"].to_numpy(dtype=bool)
        & np.isfinite(speed)
        & (lengths >= params.min_segment_frames)
        & np.array(
            [speed_chord_safe(int(frame), track, index.unsafe, half) for frame in row_frames],
            dtype=bool,
        )
    )
    centres, speeds = row_frames[usable], speed[usable]
    uncovered = tracked - natural - set(centres.tolist())
    considered = len(uncovered) + int(centres.size)
    share = float(centres.size) / considered if considered else 0.0
    quality: dict[str, Any] = {
        "kinematics_rows": int(len(rows)),
        "tracked_frames": len(tracked),
        "margin_frames": len(tracked) - considered,
        "speed_frames": int(centres.size),
        "speed_coverage": share,
    }
    if not centres.size:
        ctx.emit(
            "peak-speed",
            value=None,
            frame=None,
            frames=row_frames.tolist(),
            track_id=track,
            considered=considered,
            quality=quality,
            reason="no_valid_speed",
            parameters=parameters,
        )
        return
    best = int(np.lexsort((centres, -speeds))[0])
    centre, value = int(centres[best]), float(speeds[best])
    segments = sorted({int(s) for s in rows["segment_id"].to_numpy()[usable]})
    quality.update(
        {
            "chord_start_frame": centre - half,
            "chord_end_frame": centre + half,
            "segments": segments,
            "shortest_segment_frames": min(index.segment_length[(track, s)] for s in segments),
        }
    )
    reason = None
    if share < params.peak_speed_min_coverage:
        # Stretches without valid speeds cover too much of the window: the real peak may lie
        # in them, so the highest speed measured is no peak.
        causes = _uncovered_causes(kin, uncovered, params.min_segment_frames, margin, half)
        quality.update({f"uncovered_{cause}_frames": n for cause, n in causes.items()})
        reason = _COVERAGE_REASONS[max(causes, key=lambda cause: (causes[cause], cause))]
    else:
        hidden = _hidden_peak(ctx, kin, tracked, natural, centres, speeds, half)
        if hidden is not None:
            quality.update(hidden)
            reason = "speed_peak_hidden"
    if reason is not None:
        # kept for audit only: the real peak may be higher
        quality["highest_valid_speed_mps"] = value
        quality["highest_valid_speed_frame"] = centre
        ctx.emit(
            "peak-speed",
            value=None,
            frame=None,
            frames=centres.tolist(),
            track_id=track,
            considered=considered,
            used=int(centres.size),
            quality=quality,
            reason=reason,
            parameters=parameters,
            reason_zh=withheld_speed_reason_zh(reason, quality, ctx.context.rate),
        )
        return
    _check_chord(kin, track, centre, half, seconds, value)
    ctx.emit(
        "peak-speed",
        value=value,
        frame=centre,
        frames=centres.tolist(),
        track_id=track,
        quality=quality,
        considered=considered,
        used=int(centres.size),
        parameters=parameters,
    )


def _emit_unknown_team_nearer(
    ctx: _Context,
    metric: str,
    track: int,
    ref: int,
    frames: list[int],
    team: str,
    unknown: set[int],
    parameters: dict[str, Any],
) -> None:
    """Withhold a nearest-opponent value: in some frame an outfield player without a team
    label (gbf.team_check clears labels that contradict the shirt colours) was nearer than
    every labelled opponent, so the nearest opponent is not known."""
    ctx.emit(
        metric,
        value=None,
        frame=ref,
        frames=frames,
        track_id=track,
        considered=len(frames),
        quality={"team": team, "unknown_team_track_ids": sorted(unknown)},
        reason="unknown_team_nearer",
        parameters=parameters,
    )


def _nearest_opponent_pitch(ctx: _Context, track: int, ref: int) -> None:
    index, params = ctx.index, ctx.params
    frames = index.same_shot_frames(ref, params.key_half_window)
    parameters = {"key_half_window": params.key_half_window}
    usable = [f for f in frames if f in index.pitch]
    team = index.team_of(track, frames)
    if team is None:
        ctx.emit(
            "nearest-opponent-distance",
            value=None,
            frame=ref,
            frames=frames,
            track_id=track,
            considered=len(frames),
            reason="team_unknown",
            parameters=parameters,
        )
        return
    distances: list[float] = []
    keeper: list[float] = []
    opponents: list[int] = []
    used: list[int] = []
    unknown_nearer: set[int] = set()
    for frame in usable:
        subject = index.player(frame, track)
        if subject is None or subject.pitch is None:
            continue
        best: tuple[float, int] | None = None
        best_keeper: float | None = None
        unknown: tuple[float, int] | None = None
        for other in index.players.get(frame, ()):
            if other.track_id == track or other.pitch is None or off_pitch(other.pitch):
                continue  # off the pitch: a coach or a steward, no player in play
            if same_spot(subject.box, other.box, subject.pitch, other.pitch):
                continue  # the subject twice, or one box around it and the player behind
            gap = math.hypot(other.pitch[0] - subject.pitch[0], other.pitch[1] - subject.pitch[1])
            if other.team is None:
                if other.role == "player" and (unknown is None or (gap, other.track_id) < unknown):
                    unknown = (gap, other.track_id)
                continue
            if other.team == team:
                continue
            if other.role == "goalkeeper":
                best_keeper = gap if best_keeper is None else min(best_keeper, gap)
                continue
            if best is None or (gap, other.track_id) < best:
                best = (gap, other.track_id)
        if unknown is not None and (best is None or unknown[0] < best[0]):
            unknown_nearer.add(unknown[1])
        if best is None:
            continue
        distances.append(best[0])
        opponents.append(best[1])
        used.append(frame)
        if best_keeper is not None:
            keeper.append(best_keeper)
    if unknown_nearer:
        _emit_unknown_team_nearer(
            ctx, "nearest-opponent-distance", track, ref, frames, team, unknown_nearer, parameters
        )
        return
    if not used:
        ctx.emit(
            "nearest-opponent-distance",
            value=None,
            frame=ref,
            frames=frames,
            track_id=track,
            considered=len(frames),
            quality={"team": team},
            reason="no_opponent",
            parameters=parameters,
        )
        return
    common = sorted(Counter(opponents).items(), key=lambda item: (-item[1], item[0]))[0][0]
    ctx.emit(
        "nearest-opponent-distance",
        value=_median(distances),
        frame=ref,
        frames=used,
        track_id=track,
        track_ids=[common],
        quality={
            "team": team,
            "opponent_track_id": common,
            "frames_min_m": min(distances),
            "frames_max_m": max(distances),
            "goalkeeper_distance_m": _median(keeper) if keeper else None,
            "in_frame_only": True,
        },
        considered=len(frames),
        used=len(used),
        parameters=parameters,
    )


def _nearest_opponent_image(ctx: _Context, track: int, ref: int) -> None:
    index, params = ctx.index, ctx.params
    frames = index.same_shot_frames(ref, params.key_half_window)
    parameters = {
        "key_half_window": params.key_half_window,
        "image_feet_max_median_box_px": params.image_feet_max_median_box_px,
    }
    shot = index.shot_index(ref)
    box_h = float(index.shot_box_h[shot])
    if math.isfinite(box_h) and box_h > params.image_feet_max_median_box_px:
        ctx.emit(
            "nearest-opponent-image-distance",
            value=None,
            frame=ref,
            frames=frames,
            track_id=track,
            considered=len(frames),
            quality={"shot_median_box_h_px": box_h},
            reason="feet_not_visible",
            parameters=parameters,
        )
        return
    team = index.team_of(track, frames)
    if team is None:
        ctx.emit(
            "nearest-opponent-image-distance",
            value=None,
            frame=ref,
            frames=frames,
            track_id=track,
            considered=len(frames),
            reason="team_unknown",
            parameters=parameters,
        )
        return
    distances: list[float] = []
    opponents: list[int] = []
    used: list[int] = []
    unknown_nearer: set[int] = set()
    for frame in frames:
        subject = index.player(frame, track)
        if subject is None or subject.height <= 0:
            continue
        best: tuple[float, int] | None = None
        unknown: tuple[float, int] | None = None
        for other in index.players.get(frame, ()):
            if other.track_id == track or other.height <= 0:
                continue
            if same_spot(subject.box, other.box):
                continue  # the subject twice, or one box around it and the player behind
            gap = math.hypot(other.foot[0] - subject.foot[0], other.foot[1] - subject.foot[1])
            scaled = gap / (0.5 * (other.height + subject.height))
            if other.team is None:
                if other.role == "player" and (
                    unknown is None or (scaled, other.track_id) < unknown
                ):
                    unknown = (scaled, other.track_id)
                continue
            if other.team == team:
                continue
            if best is None or (scaled, other.track_id) < best:
                best = (scaled, other.track_id)
        if unknown is not None and (best is None or unknown[0] < best[0]):
            unknown_nearer.add(unknown[1])
        if best is None:
            continue
        distances.append(best[0])
        opponents.append(best[1])
        used.append(frame)
    if unknown_nearer:
        _emit_unknown_team_nearer(
            ctx,
            "nearest-opponent-image-distance",
            track,
            ref,
            frames,
            team,
            unknown_nearer,
            parameters,
        )
        return
    if not used:
        ctx.emit(
            "nearest-opponent-image-distance",
            value=None,
            frame=ref,
            frames=frames,
            track_id=track,
            considered=len(frames),
            quality={"team": team},
            reason="no_opponent",
            parameters=parameters,
        )
        return
    common = sorted(Counter(opponents).items(), key=lambda item: (-item[1], item[0]))[0][0]
    ctx.emit(
        "nearest-opponent-image-distance",
        value=_median(distances),
        frame=ref,
        frames=used,
        track_id=track,
        track_ids=[common],
        quality={
            "team": team,
            "opponent_track_id": common,
            "shot_median_box_h_px": box_h if math.isfinite(box_h) else None,
            "team_labels_in_close_shots": "unreliable",
        },
        considered=len(frames),
        used=len(used),
        parameters=parameters,
    )


def _lift_rows(
    ctx: _Context, track: int, frames: Sequence[int]
) -> tuple[list[tuple[int, dict[str, Any], _Player]], str | None]:
    """Lifted rows of the player on ``frames`` (with their box), or a reason code."""
    index = ctx.index
    pose_seen = False
    reasons: list[str] = []
    lifted = []
    for frame in frames:
        player = index.player(frame, track)
        entry = index.pose.get((frame, track))
        row = index.lift.get((frame, track))
        if player is None:
            continue
        if entry is None or entry[1] < ctx.params.pose_min_mean_score:
            continue
        pose_seen = True
        if row is None or str(row.get("method")) != LIFTED:
            reasons.append(str(row.get("reason")) if row is not None else "not_lifted")
            continue
        lifted.append((frame, row, player))
    if lifted:
        return lifted, None
    if not pose_seen:
        return [], "no_pose"
    if reasons and all(reason == "no_calibration" for reason in reasons):
        return [], "not_pitch_usable"
    return [], "not_lifted"


def _contact_reference(
    ctx: _Context, track: int, ref: int, contacts: Sequence[ContactMatch]
) -> tuple[int, str]:
    """The player's matched contact frame nearest the key frame (on a pitch frame), else ref."""
    own = [m.frame for m in contacts if m.track_id == track and m.frame in ctx.index.pitch]
    if own:
        best = min(own, key=lambda frame: (abs(frame - ctx.key), frame))
        return best, "contact_estimate"
    return ref, "highlight_key_frame" if ref == ctx.key else "joined_shot_moment"


def _box_ok(ctx: _Context, player: _Player, metric_id: str) -> bool:
    entry = ctx.context.registry.get(metric_id)
    minimum = entry.min_bbox_px if entry is not None else 0
    return player.height >= minimum


def _row_facing_reversed(
    index: _Index, frame: int, track: int, row: Mapping[str, Any], player: _Player
) -> bool:
    """The shared front-back check, independent of either metric's other display gates."""
    entry = index.pose.get((frame, track))
    trunk = image_trunk_deg(entry[0]) if entry is not None else math.nan
    image_dx = None
    to_image = index.ground_map(frame)
    if to_image is not None and player.pitch is not None:
        image_dx = facing_image_dx(to_image, player.pitch, float(row["facing_deg"]))
    return facing_reversed(trunk, image_dx)


def _facing_passes(
    index: _Index,
    gates: FacingGates,
    frame: int,
    track: int,
    row: Mapping[str, Any],
    player: _Player,
) -> tuple[bool, bool]:
    """(passes, reversed): whether the player's lifted row of ``frame`` passes
    :func:`facing_frame_ok` with the frame's 2D trunk angle (its pose) and the facing placed
    in the picture with the frame's ground map, and whether it fails only as a front-back
    reversal (:func:`facing_reversed`)."""
    facing = float(row["facing_deg"])
    checks: dict[str, Any] = {
        "lifted": str(row.get("method")) == LIFTED,
        "facing_deg": facing,
        "facing_conf": float(row["facing_conf"]),
        "facing_sigma_deg": float(row["facing_sigma_deg"]),
        "box_height_px": player.height,
    }
    if not facing_frame_ok(gates, **checks, image_trunk_deg=math.nan, image_dx=None):
        return False, False  # a gate other than the reversal check (NaN skips that)
    reversal = _row_facing_reversed(index, frame, track, row, player)
    return not reversal, reversal


def _body_orientation(ctx: _Context, track: int, ref: int, reference_kind: str) -> None:
    index, params = ctx.index, ctx.params
    frames = index.same_shot_frames(ref, params.key_half_window)
    entry = ctx.context.registry.get("body-orientation")
    parameters = {
        "key_half_window": params.key_half_window,
        "facing_min_conf": params.facing_min_conf,
        "facing_max_sigma_deg": params.facing_max_sigma_deg,
        "facing_min_resultant": params.facing_min_resultant,
        "min_bbox_px": entry.min_bbox_px if entry is not None else 0,
        "facing_lean_deg": FACING_LEAN_DEG,
        "facing_side_share": FACING_SIDE_SHARE,
        "attack_direction": dict(params.attack_direction),
    }
    base_quality: dict[str, Any] = {
        "reference": reference_kind,
        "contact_frame_confirmed": False,
    }
    if ref not in index.pitch:
        ctx.emit(
            "body-orientation",
            value=None,
            frame=ref,
            frames=frames,
            track_id=track,
            considered=len(frames),
            quality=base_quality,
            reason="not_pitch_usable",
            parameters=parameters,
        )
        return
    rows, reason = _lift_rows(ctx, track, [f for f in frames if f in index.pitch])
    if reason is not None:
        ctx.emit(
            "body-orientation",
            value=None,
            frame=ref,
            frames=frames,
            track_id=track,
            considered=len(frames),
            quality=base_quality,
            reason=reason,
            parameters=parameters,
        )
        return
    big = [(f, row, p) for f, row, p in rows if _box_ok(ctx, p, "body-orientation")]
    if not big:
        ctx.emit(
            "body-orientation",
            value=None,
            frame=ref,
            frames=[f for f, _, _ in rows],
            track_id=track,
            considered=len(frames),
            quality={**base_quality, "box_h_px": max(p.height for _, _, p in rows)},
            reason="box_too_small",
            parameters=parameters,
        )
        return
    gates = FacingGates.of(params, ctx.context.registry)
    gated = []
    reversed_rows = 0
    for f, row, p in big:
        passes, reversal = _facing_passes(index, gates, f, track, row, p)
        if passes:
            gated.append((f, row, p))
        reversed_rows += reversal
    angles = np.radians([float(row["facing_deg"]) for _, row, _ in gated])
    resultant = float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles)))) if gated else 0.0
    if not gated or resultant < params.facing_min_resultant:
        ctx.emit(
            "body-orientation",
            value=None,
            frame=ref,
            frames=[f for f, _, _ in big],
            track_id=track,
            considered=len(frames),
            quality={
                **base_quality,
                "rows_passing_gates": len(gated),
                "rows_reversed": reversed_rows,
                "resultant": resultant,
            },
            reason="facing_uncertain",
            parameters=parameters,
        )
        return
    facing = math.degrees(math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    team = index.team_of(track, frames)
    positions = [p.pitch for _, _, p in gated if p.pitch is not None]
    quality = {
        **base_quality,
        "facing_deg_pitch_zup": facing,
        "rows_passing_gates": len(gated),
        "rows_reversed": reversed_rows,
        "resultant": resultant,
        "facing_conf_min": min(float(row["facing_conf"]) for _, row, _ in gated),
        "facing_sigma_deg_max": max(float(row["facing_sigma_deg"]) for _, row, _ in gated),
        "box_h_px_min": min(p.height for _, _, p in gated),
        "team": team,
    }
    used = [f for f, _, _ in gated]
    known = ctx.context.attack_direction_known
    if team is None or not known or not positions:
        ctx.emit(
            "body-orientation",
            value=None,
            frame=ref,
            frames=used,
            track_id=track,
            considered=len(frames),
            quality=quality,
            reason=(
                "team_unknown"
                if team is None
                else "attack_direction_unknown"
                if not known
                else "not_pitch_usable"
            ),
            parameters=parameters,
        )
        return
    sign = params.attack_direction[team]
    x = _median([position[0] for position in positions])
    y = _median([position[1] for position in positions])
    # pitch_zup: (x, -y); the attacked goal's centre is (52.5 * sign, 0).
    goal_direction = math.degrees(math.atan2(0.0 - (-y), HALF_LENGTH_M * sign - x))
    signed = _wrap_deg(facing - goal_direction)
    quality.update({"goal_direction_deg": goal_direction, "signed_angle_deg": signed})
    ctx.emit(
        "body-orientation",
        value=abs(signed),
        frame=ref,
        frames=used,
        track_id=track,
        quality=quality,
        considered=len(frames),
        used=len(used),
        parameters=parameters,
    )


def _trunk_lean(ctx: _Context, track: int, ref: int, reference_kind: str) -> None:
    index, params = ctx.index, ctx.params
    frames = index.same_shot_frames(ref, params.key_half_window)
    entry = ctx.context.registry.get("trunk-lean-at-contact")
    parameters = {
        "key_half_window": params.key_half_window,
        "lean_max_sigma_deg": params.lean_max_sigma_deg,
        "min_bbox_px": entry.min_bbox_px if entry is not None else 0,
        "facing_lean_deg": FACING_LEAN_DEG,
        "facing_side_share": FACING_SIDE_SHARE,
    }
    base_quality: dict[str, Any] = {
        "reference": reference_kind,
        "contact_frame_confirmed": False,
    }
    if ref not in index.pitch:
        ctx.emit(
            "trunk-lean-at-contact",
            value=None,
            frame=ref,
            frames=frames,
            track_id=track,
            considered=len(frames),
            quality=base_quality,
            reason="not_pitch_usable",
            parameters=parameters,
        )
        return
    rows, reason = _lift_rows(ctx, track, [f for f in frames if f in index.pitch])
    if reason is not None:
        ctx.emit(
            "trunk-lean-at-contact",
            value=None,
            frame=ref,
            frames=frames,
            track_id=track,
            considered=len(frames),
            quality=base_quality,
            reason=reason,
            parameters=parameters,
        )
        return
    big = [(f, row, p) for f, row, p in rows if _box_ok(ctx, p, "trunk-lean-at-contact")]
    if not big:
        ctx.emit(
            "trunk-lean-at-contact",
            value=None,
            frame=ref,
            frames=[f for f, _, _ in rows],
            track_id=track,
            considered=len(frames),
            quality={**base_quality, "box_h_px": max(p.height for _, _, p in rows)},
            reason="box_too_small",
            parameters=parameters,
        )
        return
    lean_rows = [
        (f, row, p)
        for f, row, p in big
        if math.isfinite(float(row["trunk_lean_deg"]))
        and math.isfinite(float(row["trunk_lean_sigma_deg"]))
        and float(row["trunk_lean_sigma_deg"]) < params.lean_max_sigma_deg
    ]
    gated = [
        (f, row) for f, row, p in lean_rows if not _row_facing_reversed(index, f, track, row, p)
    ]
    reversed_rows = len(lean_rows) - len(gated)
    if not gated:
        sigmas = [float(row["trunk_lean_sigma_deg"]) for _, row, _ in big]
        finite = [s for s in sigmas if math.isfinite(s)]
        ctx.emit(
            "trunk-lean-at-contact",
            value=None,
            frame=ref,
            frames=[f for f, _, _ in big],
            track_id=track,
            considered=len(frames),
            quality={
                **base_quality,
                "sigma_deg_min": min(finite) if finite else None,
                "rows_passing_gates": 0,
                "rows_reversed": reversed_rows,
            },
            reason="facing_uncertain" if reversed_rows else "lean_uncertain",
            parameters=parameters,
        )
        return
    leans = [float(row["trunk_lean_deg"]) for _, row in gated]
    alternatives = [
        float(row["trunk_lean_alt_deg"])
        for _, row in gated
        if math.isfinite(float(row["trunk_lean_alt_deg"]))
    ]
    used = [f for f, _ in gated]
    ctx.emit(
        "trunk-lean-at-contact",
        value=_median(leans),
        frame=ref,
        frames=used,
        track_id=track,
        quality={
            **base_quality,
            "rows_passing_gates": len(gated),
            "rows_reversed": reversed_rows,
            "frames_min_deg": min(leans),
            "frames_max_deg": max(leans),
            "other_solution_deg": _median(alternatives) if alternatives else None,
            "sigma_deg_max": max(float(row["trunk_lean_sigma_deg"]) for _, row in gated),
            "tilt_deg": _median([float(row["trunk_tilt_deg"]) for _, row in gated]),
        },
        considered=len(frames),
        used=len(used),
        parameters=parameters,
    )


_IMAGE_ANGLES = (
    ("image-trunk-lean", "image_lean_deg"),
    ("image-knee-angle-left", "image_knee_left_deg"),
    ("image-knee-angle-right", "image_knee_right_deg"),
)


def _image_angles(ctx: _Context, track: int, ref: int) -> None:
    index, params = ctx.index, ctx.params
    frames = index.same_shot_frames(ref, params.key_half_window)
    ordered = sorted(frames, key=lambda frame: (abs(frame - ref), frame))
    parameters = {
        "key_half_window": params.key_half_window,
        "pose_min_mean_score": params.pose_min_mean_score,
        "image_angle_min_box_px": params.image_angle_min_box_px,
    }
    for metric_id, column in _IMAGE_ANGLES:
        chosen: tuple[int, float, _Player, float] | None = None
        pose_seen = False
        small = False
        for frame in ordered:
            player = index.player(frame, track)
            entry = index.pose.get((frame, track))
            row = index.lift.get((frame, track))
            if player is None or entry is None or entry[1] < params.pose_min_mean_score:
                continue
            pose_seen = True
            if player.height < params.image_angle_min_box_px:
                small = True
                continue
            value = float(row[column]) if row is not None else math.nan
            if math.isfinite(value):
                chosen = (frame, value, player, float(entry[1]))
                break
        if chosen is None:
            reason = (
                "no_pose"
                if not pose_seen
                else "box_too_small"
                if small
                else ("low_keypoint_confidence")
            )
            ctx.emit(
                metric_id,
                value=None,
                frame=ref,
                frames=frames,
                track_id=track,
                considered=len(frames),
                reason=reason,
                parameters=parameters,
            )
            continue
        frame, value, player, mean_score = chosen
        ctx.emit(
            metric_id,
            value=value,
            frame=frame,
            frames=[frame],
            track_id=track,
            quality={
                "box_h_px": player.height,
                "pose_mean_score": mean_score,
                "offset_from_reference_frames": frame - ref,
                "plane": "image",
            },
            considered=len(frames),
            used=1,
            parameters=parameters,
        )


def _attackers_in_box(ctx: _Context) -> None:
    index, params = ctx.index, ctx.params
    cues = ctx.highlight.get("cues") or {}
    # the frame the highlight's summary names for its box-pressure cue, so the card and the
    # summary count the same players; else the key frame
    box = cues.get("box_pressure") if isinstance(cues, Mapping) else None
    cue_frame = box.get("frame") if isinstance(box, Mapping) else None
    from_cue = (
        isinstance(cue_frame, int | np.integer)
        and not isinstance(cue_frame, bool)
        and ctx.start <= int(cue_frame) <= ctx.end
    )
    key = int(cue_frame) if from_cue else ctx.key  # type: ignore[arg-type]
    parameters = {
        "penalty_area_depth_m": PENALTY_AREA_DEPTH_M,
        "penalty_area_half_width_m": PENALTY_AREA_HALF_WIDTH_M,
        "attack_direction": dict(params.attack_direction),
    }
    pitch_frame = key in index.pitch
    players = [p for p in index.players.get(key, ()) if p.pitch is not None and p.role == "player"]
    if pitch_frame and players and not any(p.team in TEAMS for p in players):
        # no outfield player on the frame has a team label (gbf.role_check clears them all
        # when the plugin's roles are implausible): any team's count would be 0 by construction
        ctx.emit(
            "attackers-in-box",
            value=None,
            frame=key,
            frames=[key],
            track_id=None,
            quality={"outfield_players_without_team": len(players)},
            considered=1,
            reason="team_unknown",
            parameters=parameters,
        )
        return
    if not ctx.context.attack_direction_known or not pitch_frame:
        # an unknown attack direction (gbf.side_check could not confirm the left/right names)
        # holds for every frame and comes first; else a frame without pitch positions
        ctx.emit(
            "attackers-in-box",
            value=None,
            frame=key,
            frames=[key],
            track_id=None,
            considered=1,
            reason=(
                "attack_direction_unknown"
                if not ctx.context.attack_direction_known
                else "not_pitch_usable"
            ),
            parameters=parameters,
        )
        return
    # Where the attacking team comes from: the highlight's attack cues, else the first
    # involved player's label, else the team with more players in the box it attacks.
    team, source = None, "largest_count"
    for name in ("box_pressure", "ball_goal"):
        cue = cues.get(name) if isinstance(cues, Mapping) else None
        if isinstance(cue, Mapping) and cue.get("team") in TEAMS:
            team, source = str(cue["team"]), "cue"
            break
    track_ids = [int(t) for t in ctx.highlight.get("track_ids") or []]
    if team is None and track_ids:
        team = index.team_of(track_ids[0], index.same_shot_frames(key, params.key_half_window))
        source = "largest_count" if team is None else "first_player"

    def inside(player: _Player, attacking: str) -> bool:
        sign = params.attack_direction[attacking]
        x = player.pitch[0] * sign  # type: ignore[index]
        y = player.pitch[1]  # type: ignore[index]
        return (
            HALF_LENGTH_M - PENALTY_AREA_DEPTH_M <= x <= HALF_LENGTH_M
            and abs(y) <= PENALTY_AREA_HALF_WIDTH_M
        )

    counts = {
        attacking: sorted(
            p.track_id for p in players if p.team == attacking and inside(p, attacking)
        )
        for attacking in TEAMS
    }
    chosen = team
    if chosen is None:
        chosen = sorted(TEAMS, key=lambda t: (-len(counts[t]), t))[0]
    other = "right" if chosen == "left" else "left"
    unlabelled = [p for p in players if p.team not in TEAMS and inside(p, chosen)]
    if unlabelled:
        # a player without a team label in this box may be an attacker: the count would only
        # be a lower bound (nearest-opponent-distance is withheld alike, unknown_team_nearer)
        ctx.emit(
            "attackers-in-box",
            value=None,
            frame=key,
            frames=[key],
            track_id=None,
            quality={
                "team": chosen,
                "team_source": source,
                "reference": "box_pressure_cue" if from_cue else "highlight_key_frame",
                "attackers_with_team": len(counts[chosen]),
                "players_without_team_in_box": len(unlabelled),
            },
            considered=1,
            reason="unknown_team_in_box",
            parameters=parameters,
        )
        return
    defenders = sorted(p.track_id for p in players if p.team == other and inside(p, chosen))
    ctx.emit(
        "attackers-in-box",
        value=len(counts[chosen]),
        frame=key,
        frames=[key],
        track_id=None,
        track_ids=counts[chosen],
        quality={
            "team": chosen,
            "team_source": source,
            "reference": "box_pressure_cue" if from_cue else "highlight_key_frame",
            "defenders_in_box": len(defenders),
            "visible_outfield_players": sum(1 for p in players if p.team == chosen),
            "in_frame_only": True,
        },
        considered=1,
        used=1,
        parameters=parameters,
    )


def _contacts(ctx: _Context) -> list[ContactMatch]:
    params = ctx.params
    matches, examined = _match_window(ctx.index, ctx.start, ctx.end, params)
    parameters = {
        "contact_radius_bh": params.contact_radius_bh,
        "contact_fallback_bh": params.contact_fallback_bh,
        "contact_candidates": params.contact_candidates,
        "keypoint_min_score": params.keypoint_min_score,
        "pose_min_mean_score": params.pose_min_mean_score,
    }
    window = ctx.end - ctx.start + 1
    if not matches:
        ctx.emit(
            "ball-contact-frame",
            value=None,
            frame=None,
            frames=[],
            track_id=None,
            considered=window,
            quality={"ball_events_examined": examined},
            reason="no_contact",
            parameters=parameters,
        )
        return []
    for match in matches:
        event = match.event
        ctx.emit(
            "ball-contact-frame",
            value=match.frame,
            frame=match.frame,
            frames=[match.frame],
            track_id=match.track_id,
            quality={
                "match": match.method,
                "distance_box_h": match.distance_box_h,
                "keypoint": match.keypoint or None,
                "box_h_px": match.box_h_px,
                "event_kind": str(event.get("kind")),
                "event_profile": str(event.get("profile", "strike")),
                "velocity_change_bh_s": event.get("magnitude"),
                "estimate": True,
                "contact_frame_confirmed": False,
            },
            considered=window,
            used=1,
            parameters=parameters,
        )
        speed_after = float(event.get("speed_after", math.nan))
        if str(event.get("kind")) == "slow_down" or not math.isfinite(speed_after):
            continue
        ctx.emit(
            "ball-image-speed",
            value=speed_after,
            frame=match.frame,
            frames=[match.frame],
            track_id=match.track_id,
            quality={
                "speed_before_bh_s": event.get("speed_before"),
                "angle_deg": event.get("angle_deg"),
                "event_kind": str(event.get("kind")),
                "plane": "image",
                "never_km_h": True,
            },
            considered=window,
            used=1,
            parameters={"scale": "ball box diameter / 0.155 (gbf.ball ball_to_box_height)"},
        )
    return matches


# Order in which a highlight's records are listed (the freeze card shows the first ones).
_TYPE_ORDER: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "ball_strike": (
            "ball-contact-frame",
            "ball-image-speed",
            "trunk-lean-at-contact",
            "body-orientation",
            "nearest-opponent-distance",
            "nearest-opponent-image-distance",
            "peak-speed",
            "attackers-in-box",
        ),
        "attack_in_box": (
            "attackers-in-box",
            "nearest-opponent-distance",
            "peak-speed",
            "body-orientation",
            "ball-contact-frame",
            "ball-image-speed",
            "trunk-lean-at-contact",
            "nearest-opponent-image-distance",
        ),
        "duel_tackle": (
            "nearest-opponent-distance",
            "nearest-opponent-image-distance",
            "ball-contact-frame",
            "body-orientation",
            "trunk-lean-at-contact",
            "peak-speed",
            "ball-image-speed",
            "attackers-in-box",
        ),
        "sprint": (
            "peak-speed",
            "nearest-opponent-distance",
            "body-orientation",
            "trunk-lean-at-contact",
            "ball-contact-frame",
            "ball-image-speed",
            "nearest-opponent-image-distance",
            "attackers-in-box",
        ),
        "closeup_reaction": (
            "nearest-opponent-image-distance",
            "image-trunk-lean",
            "image-knee-angle-left",
            "image-knee-angle-right",
            "ball-contact-frame",
            "ball-image-speed",
            "nearest-opponent-distance",
            "peak-speed",
        ),
    }
)


def _metric_rank(metric: str, order: Sequence[str]) -> int:
    return order.index(metric) if metric in order else len(order) + sorted(METRICS).index(metric)


def _sort_key(
    record: Mapping[str, Any], order: Sequence[str], players: Sequence[int], key: int
) -> tuple[int, int, int, int, int, str]:
    """Available first; then the type's metric order; then the player order of track_ids
    (highlight-level records first); then closeness to the key frame."""
    metric = str(record["metric_id"])
    track = record.get("track_id")
    player_rank = (
        -1 if track is None else players.index(track) if track in players else len(players)
    )
    frame = record.get("frame")
    distance = 0 if frame is None else abs(int(frame) - key)
    return (
        int(record["status"] != "ok"),
        _metric_rank(metric, order),
        player_rank,
        distance,
        -1 if frame is None else int(frame),
        str(record["evidenceId"]),
    )


def card_records(
    records: Sequence[Mapping[str, Any]],
    highlight: Mapping[str, Any],
    limit: int = 5,
    *,
    prefer_track: int | None = None,
    key_frame: int | None = None,
) -> list[Mapping[str, Any]]:
    """The records a freeze-frame data card shows for ``highlight``: following the metric
    order of the highlight's type, one available record per metric, at most ``limit``.

    For each metric the record of ``prefer_track`` (the player the overlay focuses on)
    nearest ``key_frame`` (the frozen frame; default the highlight's peak_frame) wins; when
    that player has none, the record of any player nearest the key frame. Ties keep the
    order of :func:`measure_highlight` (by player, then closeness to the key frame); a
    record without a frame comes last. Unavailable records never fill the card."""
    order = _TYPE_ORDER.get(str(highlight.get("type")), tuple(sorted(METRICS)))
    hid = str(highlight.get("id"))
    key = key_frame if key_frame is not None else highlight.get("peak_frame")

    def rank(position: int, record: Mapping[str, Any]) -> tuple[int, float, int]:
        focus = 0 if prefer_track is not None and record.get("track_id") == prefer_track else 1
        frame = record.get("frame")
        distance = math.inf if frame is None or key is None else abs(int(frame) - int(key))
        return focus, distance, position

    chosen: dict[str, tuple[tuple[int, float, int], Mapping[str, Any]]] = {}
    for position, record in enumerate(records):
        if record.get("highlight_id") != hid or record.get("status") != "ok":
            continue
        metric = str(record["metric_id"])
        candidate = rank(position, record)
        if metric not in chosen or candidate < chosen[metric][0]:
            chosen[metric] = (candidate, record)
    ranked = sorted(chosen.items(), key=lambda item: _metric_rank(item[0], order))
    return [record for _, (_, record) in ranked[: max(0, int(limit))]]


def measure_highlight(
    highlight: Mapping[str, Any],
    tables: MeasureTables,
    context: MeasureContext,
    params: MeasureParams = DEFAULT_PARAMS,
    *,
    _index: _Index | None = None,
) -> list[dict[str, Any]]:
    """Evidence records of one highlight (see the module docstring for every metric).

    Records come in display order: available values first, then the unavailable ones; each
    group follows a metric order that depends on the highlight type (:func:`card_records`
    picks the freeze card's rows from it), then the player order of ``track_ids``
    (highlight-level records first), then closeness to the key frame. Raises MeasureError
    for malformed inputs.
    """
    problems: list[str] = []
    if not isinstance(highlight, Mapping):
        raise MeasureError("measure_highlight", ["highlight must be a mapping"])
    for name in ("id", "start_frame", "peak_frame", "end_frame"):
        if name not in highlight:
            problems.append(f"highlight lacks {name}")
    if not isinstance(context, MeasureContext):
        problems.append("context must be a MeasureContext")
    if not isinstance(params, MeasureParams):
        problems.append("params must be MeasureParams")
    if problems:
        raise MeasureError("measure_highlight", problems)
    index = _index if _index is not None else _Index(tables)
    start, key, end = (int(highlight[k]) for k in ("start_frame", "peak_frame", "end_frame"))
    if not 0 <= start <= key <= end < index.n_frames:
        raise MeasureError(
            "measure_highlight",
            [f"need 0 <= start <= peak <= end < {index.n_frames}, got {start}, {key}, {end}"],
        )
    ctx = _Context(
        highlight=highlight,
        hid=str(highlight["id"]),
        start=start,
        end=end,
        key=key,
        index=index,
        context=context,
        params=params,
        records=[],
    )
    players: list[int] = []
    for track in highlight.get("track_ids") or []:
        if int(track) >= 0 and int(track) not in players:
            players.append(int(track))
    contacts = _contacts(ctx)
    _attackers_in_box(ctx)
    for track in players:
        _peak_speed(ctx, track)
        ref = _reference_frame(ctx, track)
        if ref is None:
            frames = index.same_shot_frames(key, params.key_half_window)
            ctx.emit(
                "nearest-opponent-distance",
                value=None,
                frame=key,
                frames=frames,
                track_id=track,
                considered=len(frames),
                reason="out_of_frame",
            )
            continue
        if ref in index.pitch:
            _nearest_opponent_pitch(ctx, track, ref)
            contact_ref, kind = _contact_reference(ctx, track, ref, contacts)
            _body_orientation(ctx, track, contact_ref, kind)
            _trunk_lean(ctx, track, contact_ref, kind)
        else:
            ctx.emit(
                "nearest-opponent-distance",
                value=None,
                frame=ref,
                frames=[ref],
                track_id=track,
                considered=1,
                reason="not_pitch_usable",
            )
            _nearest_opponent_image(ctx, track, ref)
            _image_angles(ctx, track, ref)
    order = _TYPE_ORDER.get(str(highlight.get("type")), tuple(sorted(METRICS)))
    return sorted(ctx.records, key=lambda record: _sort_key(record, order, players, key))


def measure_highlights(
    highlights: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    tables: MeasureTables,
    context: MeasureContext,
    params: MeasureParams = DEFAULT_PARAMS,
) -> list[dict[str, Any]]:
    """Records of every highlight (a highlights.json document or its list), in time order."""
    items = highlights.get("highlights") if isinstance(highlights, Mapping) else highlights
    if not isinstance(items, Sequence):
        raise MeasureError("measure_highlights", ["highlights must be a list or highlights.json"])
    index = _Index(tables)
    ordered = sorted(items, key=lambda item: (int(item["peak_frame"]), str(item["id"])))
    records: list[dict[str, Any]] = []
    for highlight in ordered:
        records.extend(measure_highlight(highlight, tables, context, params, _index=index))
    return records


__all__ = [
    "CARD_LABEL_MAX_CHARS",
    "DEFAULT_ATTACK_DIRECTION",
    "DEFAULT_PARAMS",
    "EXPLICIT_TIERS",
    "FACING_LEAN_DEG",
    "FACING_SIDE_SHARE",
    "FOOT_KEYPOINTS",
    "LIFTED",
    "MEASUREMENTS_NAME",
    "METRICS",
    "PEAK_SPEED_GAP_MARGIN",
    "REASONS_ZH",
    "RECORD_TIERS",
    "SCHEMA_NAME",
    "UNITS",
    "ContactMatch",
    "FacingGates",
    "MeasureContext",
    "MeasureError",
    "MeasureParams",
    "MeasureTables",
    "MetricSpec",
    "RegistryMetric",
    "build_record",
    "card_records",
    "contact_candidates",
    "default_registry_path",
    "display_text",
    "evidence_id",
    "facing_frame_ok",
    "facing_image_dx",
    "facing_reversed",
    "format_frame_time",
    "image_trunk_deg",
    "load_metric_registry",
    "match_contacts",
    "measure_highlight",
    "measure_highlights",
    "metric_label",
    "pose_requests",
    "read_measurements",
    "record_problems",
    "resolve_tier",
    "validate_record",
    "withheld_speed_reason_zh",
    "write_measurements",
]
