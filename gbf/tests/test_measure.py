"""Tests for gbf.measure: evidence records and the per-highlight measurements.

Everything is synthetic: small hand-built tables in the column contracts of
docs/design/m1-highlights.md (tracks, shots, kinematics, ball events, pose, lift3d).

Scene used by most tests (25 fps):
- shot 0, frames 0-49: a wide, calibrated shot (all frames are pitch frames). Track 1
  (team left, attacks +x) stands at (40, 0) m; track 2 (right) at (43, 0) m; track 3
  (right) at (45, 6) m; the right goalkeeper 9 at (50, 0) m; a referee (track 20). Boxes are
  120 px tall, foot points at (40 x, 600) px (40 px per metre, so 3 m is one box height).
- shot 1, frames 50-79: an uncalibrated close shot of track 7 (left) and track 8 (right).
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gbf import measure
from gbf.geometry import project, synthetic_broadcast_camera
from gbf.kinematics import compute_kinematics
from gbf.measure import (
    METRICS,
    MeasureContext,
    MeasureError,
    MeasureParams,
    MeasureTables,
    MetricSpec,
    build_record,
    card_records,
    display_text,
    evidence_id,
    load_metric_registry,
    match_contacts,
    measure_highlight,
    measure_highlights,
    metric_label,
    pose_requests,
    read_measurements,
    record_problems,
    resolve_tier,
    validate_record,
    write_measurements,
)
from gbf.pose import HALPE26_KEYPOINTS
from gbf.track_tools import OFF_PITCH_MARGIN_M
from gbf.util.hashing import canonical_sha256

REGISTRY = load_metric_registry()
HEIGHT = 120.0
PITCH_END = 49
CLOSE_START, CLOSE_END = 50, 79


def _context(**changes: Any) -> MeasureContext:
    settings: dict[str, Any] = {
        "fps": "25/1",
        "restrictions": ("no_share", "hl3"),
        "video_sha256": "a" * 64,
        "inputs": (("tracks.parquet", "b" * 64), ("pose.parquet", "c" * 64)),
        "code": {"gbf_version": "0.0.1", "git_commit": "unknown", "git_dirty": True},
        "registry": REGISTRY,
    }
    settings.update(changes)
    return MeasureContext(**settings)


# --------------------------------------------------------------------------- tables


def _foot(x: float) -> tuple[float, float]:
    """Image foot point of a player standing at pitch x (a simple linear stand-in)."""
    return 40.0 * x, 600.0


def _track_row(
    frame: int,
    track: int,
    team: str | None,
    *,
    role: str = "player",
    pitch: tuple[float, float] | None = None,
    foot: tuple[float, float] | None = None,
    height: float = HEIGHT,
    usable: bool = True,
) -> dict[str, Any]:
    if foot is None:
        foot = _foot(pitch[0] if pitch is not None else 0.0)
    u, v = foot
    half = 0.2 * height
    has_ground = usable and pitch is not None
    return {
        "frame": frame,
        "track_id": track,
        "x1": u - half,
        "y1": v - height,
        "x2": u + half,
        "y2": v,
        "conf": np.float32(0.9),
        "role": role,
        "team": team,
        "jersey": None,
        "jersey_conf": np.nan,
        "foot_u": u,
        "foot_v": v,
        "calib_status": "fresh" if usable else "none",
        "pitch_usable": usable,
        "pitch_x": pitch[0] if has_ground else np.nan,
        "pitch_y": pitch[1] if has_ground else np.nan,
        "in_pitch": has_ground,
    }


WIDE_PLAYERS = {
    1: ("left", "player", (40.0, 0.0)),
    2: ("right", "player", (43.0, 0.0)),
    3: ("right", "player", (45.0, 6.0)),
    9: ("right", "goalkeeper", (50.0, 0.0)),
    20: (None, "referee", (30.0, 0.0)),
}


def _tracks(overrides: dict[tuple[int, int], dict[str, Any]] | None = None) -> pd.DataFrame:
    rows = []
    for frame in range(0, PITCH_END + 1):
        for track, (team, role, pitch) in WIDE_PLAYERS.items():
            row = _track_row(frame, track, team, role=role, pitch=pitch)
            row.update((overrides or {}).get((frame, track), {}))
            rows.append(row)
    for frame in range(CLOSE_START, CLOSE_END + 1):
        rows.append(_track_row(frame, 7, "left", foot=(800.0, 1000.0), height=400.0, usable=False))
        rows.append(
            _track_row(frame, 8, "right", foot=(1000.0, 1000.0), height=400.0, usable=False)
        )
    frame = pd.DataFrame(rows)
    frame["team"] = frame["team"].astype(object)
    return frame


def _shots(close_box_h: float = 300.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [0, 1],
            "start_frame": [0, CLOSE_START],
            "end_frame": [PITCH_END, CLOSE_END],
            "shot_type": ["wide", "close"],
            "fresh_share": [1.0, 0.0],
            "median_box_h": [HEIGHT, close_box_h],
        }
    )


def _kinematics(
    rows: list[tuple[int, int, int, float, bool, bool]],
    xs: dict[tuple[int, int], float] | None = None,
) -> pd.DataFrame:
    """(frame, track, segment, speed, valid, interpolated) rows on the line y = 0. x is
    ``xs`` {(frame, track): metres} where given, else each track moves along x by its
    speed column (the running sum of speed / 25 over its rows in frame order)."""
    table = pd.DataFrame(
        {
            "frame": [r[0] for r in rows],
            "track_id": [r[1] for r in rows],
            "segment_id": [r[2] for r in rows],
            "speed": [r[3] for r in rows],
            "valid": [r[4] for r in rows],
            "interpolated": [r[5] for r in rows],
        }
    )
    table = table.sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True)
    running = table.groupby("track_id")["speed"].cumsum() / 25.0
    table["x"] = [
        (xs or {}).get((f, t), float(x))
        for f, t, x in zip(table["frame"], table["track_id"], running, strict=True)
    ]
    table["y"] = 0.0
    table["vx"] = table["speed"]
    table["vy"] = 0.0
    return table


def _chord_rows(
    track: int,
    segment: int,
    frames: range,
    *,
    valid: set[int],
    steps: dict[int, float] | None = None,
    override: dict[int, float] | None = None,
) -> pd.DataFrame:
    """One kinematics segment on the line y = 0 as gbf.kinematics stores it: x advances by
    ``steps`` {frame: metres} into a frame (0.25 m, 6.25 m/s, elsewhere) and the speed of a
    frame is the 10-frame chord (x(f + 5) - x(f - 5)) / 0.4 s, NaN where the chord leaves the
    segment; valid on ``valid`` frames. ``override`` {frame: m/s} replaces chord speeds (a
    table that does not hold chords)."""
    index = list(frames)
    x = np.cumsum([0.0] + [(steps or {}).get(f, 0.25) for f in index[1:]])
    at = dict(zip(index, x.tolist(), strict=True))
    speed = [
        (at[f + 5] - at[f - 5]) / 0.4 if f - 5 in at and f + 5 in at else math.nan for f in index
    ]
    speed = [(override or {}).get(f, s) for f, s in zip(index, speed, strict=True)]
    return pd.DataFrame(
        {
            "frame": index,
            "track_id": track,
            "segment_id": segment,
            "x": x,
            "y": 0.0,
            "vx": speed,
            "vy": 0.0,
            "speed": speed,
            "valid": [f in valid for f in index],
            "interpolated": False,
        }
    )


_TEMPLATE = {
    "nose": (0.5, 0.07),
    "left_eye": (0.54, 0.055),
    "right_eye": (0.46, 0.055),
    "left_ear": (0.58, 0.07),
    "right_ear": (0.42, 0.07),
    "left_shoulder": (0.7, 0.2),
    "right_shoulder": (0.3, 0.2),
    "left_elbow": (0.76, 0.36),
    "right_elbow": (0.24, 0.36),
    "left_wrist": (0.78, 0.49),
    "right_wrist": (0.22, 0.49),
    "left_hip": (0.6, 0.5),
    "right_hip": (0.4, 0.5),
    "left_knee": (0.6, 0.72),
    "right_knee": (0.4, 0.72),
    "left_ankle": (0.6, 0.94),
    "right_ankle": (0.4, 0.94),
    "head": (0.5, 0.0),
    "neck": (0.5, 0.15),
    "hip": (0.5, 0.5),
    "left_big_toe": (0.66, 1.0),
    "right_big_toe": (0.34, 1.0),
    "left_small_toe": (0.7, 0.99),
    "right_small_toe": (0.3, 0.99),
    "left_heel": (0.58, 0.98),
    "right_heel": (0.42, 0.98),
}
FEET = ("ankle", "toe", "heel")


def _pose_row(
    frame: int,
    track_row: dict[str, Any],
    *,
    score: float = 0.9,
    feet_shift: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    x1, y1, x2, y2 = (track_row[k] for k in ("x1", "y1", "x2", "y2"))
    points = []
    for name in HALPE26_KEYPOINTS:
        fx, fy = _TEMPLATE[name]
        u, v = x1 + fx * (x2 - x1), y1 + fy * (y2 - y1)
        if any(part in name for part in FEET):
            u, v = u + feet_shift[0], v + feet_shift[1]
        points += [u, v, score]
    return {
        "frame": frame,
        "track_id": int(track_row["track_id"]),
        "model_id": "fake",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "keypoints": points,
        "mean_score": score,
    }


def _pose(tracks: pd.DataFrame, keys: list[tuple[int, int]], **options: Any) -> pd.DataFrame:
    rows = []
    for frame, track in keys:
        match = tracks[(tracks["frame"] == frame) & (tracks["track_id"] == track)]
        rows.append(_pose_row(frame, match.iloc[0].to_dict(), **options))
    return pd.DataFrame(rows)


def _lift_row(frame: int, track: int, **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame": frame,
        "track_id": track,
        "method": "height_prior_calibrated",
        "joints3d": [0.0] * 78,
        "stature_m": 1.8,
        "facing_deg": 0.0,
        "facing_conf": 0.9,
        "trunk_lean_deg": 10.0,
        "image_lean_deg": 4.0,
        "reproj_err_px": 1.0,
        "reason": "lifted",
        "facing_sigma_deg": 5.0,
        "trunk_tilt_deg": 11.0,
        "trunk_lean_alt_deg": 18.0,
        "trunk_lean_sigma_deg": 3.0,
        "image_knee_left_deg": 160.0,
        "image_knee_right_deg": 150.0,
    }
    row.update(values)
    return row


def _event(frame: int, u: float, v: float, **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame": frame,
        "kind": "speed_up",
        "magnitude": 8.0,
        "u": u,
        "v": v,
        "nearest_track_id": 1,
        "nearest_team": "left",
        "nearest_dist_box_h": 0.1,
        "shot_id": 0,
        "angle_deg": 12.0,
        "speed_before": 2.0,
        "speed_after": 10.0,
        "at_foot": True,
        "profile": "strike",
    }
    row.update(values)
    return row


HIGHLIGHT: dict[str, Any] = {
    "id": "h01",
    "type": "ball_strike",
    "start_frame": 10,
    "peak_frame": 30,
    "end_frame": 79,
    "score": 0.4,
    "track_ids": [1, 2, 7],
    "cues": {
        "cut_emphasis": {"joined_shot_event": {"frame": 60, "cue": "duel", "track_ids": [7, 8]}},
        "box_pressure": {"team": "left"},
    },
}


def _records(
    tables: MeasureTables, highlight: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    return measure_highlight(highlight or HIGHLIGHT, tables, _context())


def _find(records: list[dict[str, Any]], metric: str, track: int | None = None) -> dict:
    found = [
        r for r in records if r["metric_id"] == metric and (track is None or r["track_id"] == track)
    ]
    assert len(found) == 1, [(r["metric_id"], r["track_id"]) for r in records]
    return found[0]


# --------------------------------------------------------------------------- records


def test_record_id_is_the_hash_of_the_record_without_it() -> None:
    record = build_record(
        "peak-speed",
        context=_context(),
        status="ok",
        value=np.float64(7.123456789),
        frame=25,
        frames=[25, 24, 25],
        highlight_id="h01",
        track_id=1,
        coverage={"window": [10, 79], "frames_considered": 40, "frames_used": 30, "share": 0.75},
    )
    assert record["evidenceId"] == evidence_id(record)
    assert record["evidenceId"] == canonical_sha256(
        {key: value for key, value in record.items() if key != "evidenceId"}
    )
    assert record["value"] == 7.1235
    assert record["frames"] == [24, 25]
    assert record["tier"] == "A"
    assert record["tier_source"] == "registry"
    assert record["display"] == "25.6 km/h"
    assert record["unit"] == "m/s"
    assert record["restrictions"] == ["hl3", "no_share"]
    assert record["error"]["kind"] == "unmeasured"
    assert record["error"]["receiptId"] is None
    validate_record(record)
    tampered = {**record, "value": 9.0}
    with pytest.raises(MeasureError, match="evidenceId"):
        validate_record(tampered)


def test_unavailable_needs_a_reason_and_never_a_value() -> None:
    common: dict[str, Any] = {
        "context": _context(),
        "frame": 30,
        "frames": [30],
        "highlight_id": "h01",
        "track_id": 1,
        "coverage": {"window": [10, 79], "frames_considered": 1, "frames_used": 0, "share": 0.0},
    }
    record = build_record(
        "body-orientation", status="unavailable", value=None, reason_code="no_pose", **common
    )
    assert record["value"] is None
    assert record["display"] == "—"
    assert record["reasonCode"] == "no_pose"
    assert record["reason_zh"] == measure.REASONS_ZH["no_pose"]
    assert record["tier"] == "A_candidate"
    with pytest.raises(MeasureError, match="reason_code"):
        build_record("body-orientation", status="unavailable", value=None, **common)
    with pytest.raises(MeasureError, match="value None"):
        build_record(
            "body-orientation", status="unavailable", value=0.0, reason_code="no_pose", **common
        )
    with pytest.raises(MeasureError, match="needs a value"):
        build_record("body-orientation", status="ok", value=None, **common)
    with pytest.raises(MeasureError, match="unknown metric"):
        build_record("ball-speed", status="ok", value=1.0, **common)


def test_c_tier_and_unlisted_metrics_are_refused() -> None:
    blocked = MetricSpec("ball-speed", "球速", "m/s", "pitch", "measurement", 1, "B", "x.y", "s")
    with pytest.raises(MeasureError, match="C-tier"):
        resolve_tier(blocked, REGISTRY)
    unlisted = MetricSpec("made-up", "x", "m", "pitch", "measurement", 1, None, "x.y", "s")
    with pytest.raises(MeasureError, match="explicitly"):
        resolve_tier(unlisted, REGISTRY)
    wrong = MetricSpec("made-up", "x", "m", "pitch", "measurement", 1, "A", "x.y", "s")
    with pytest.raises(MeasureError, match="explicitly"):
        resolve_tier(wrong, REGISTRY)
    for spec in METRICS.values():
        tier, source, _ = resolve_tier(spec, REGISTRY)
        assert tier != "C"
        assert (source == "registry") == (spec.metric_id in REGISTRY)
    # The schema itself has no C tier.
    record = build_record(
        "ball-image-speed",
        context=_context(),
        status="ok",
        value=10.0,
        frame=30,
        frames=[30],
        highlight_id="h01",
        track_id=1,
        coverage={"window": [10, 79], "frames_considered": 1, "frames_used": 1, "share": 1.0},
    )
    as_c = {**record, "tier": "C"}
    as_c["evidenceId"] = evidence_id(as_c)
    assert any("tier" in problem for problem in record_problems(as_c))


def test_image_plane_records_never_show_metres_or_km_h() -> None:
    record = build_record(
        "ball-image-speed",
        context=_context(),
        status="ok",
        value=12.34,
        frame=30,
        frames=[30],
        highlight_id="h01",
        track_id=1,
        coverage={"window": [10, 79], "frames_considered": 1, "frames_used": 1, "share": 1.0},
    )
    assert record["tier"] == "image_plane"
    assert record["tier_source"] == "explicit"
    assert record["plane"] == "image"
    assert record["unit"] == "bh/s"
    assert record["display"] == "12.3 个身高/秒"
    assert display_text(40, "frame", None, _context().rate) == "第 40 帧（00:01.60）"
    assert display_text(5.0, "m/s", 1, _context().rate) == "18.0 km/h"
    assert display_text(None, "m", 1, _context().rate) == "—"


def test_write_and_read_are_byte_stable(unicode_tmp_path: Path) -> None:
    tables = MeasureTables(tracks=_tracks(), shots=_shots(), kinematics=None)
    records = _records(tables)
    first = unicode_tmp_path / "a" / "measurements.jsonl"
    second = unicode_tmp_path / "b" / "measurements.jsonl"
    write_measurements(first, records)
    write_measurements(second, measure_highlight(HIGHLIGHT, tables, _context()))
    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()
    assert read_measurements(first) == records
    with pytest.raises(MeasureError, match="appears twice"):
        write_measurements(unicode_tmp_path / "dup.jsonl", [records[0], records[0]])
    broken = unicode_tmp_path / "broken.jsonl"
    lines = first.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["value"] = 123
    broken.write_text(json.dumps(record) + "\n{not json\n", encoding="utf-8", newline="\n")
    with pytest.raises(MeasureError) as excinfo:
        read_measurements(broken)
    assert len(excinfo.value.problems) >= 2


def test_context_and_params_report_every_problem() -> None:
    with pytest.raises(MeasureError) as excinfo:
        MeasureContext(
            fps="29.97",
            restrictions=("no_share", "made_up"),
            video_sha256="xyz",
            inputs=(("C:\\abs\\tracks.parquet", "short"),),
            code={"gbf_version": "0"},
            registry=REGISTRY,
        )
    assert len(excinfo.value.problems) >= 5
    with pytest.raises(MeasureError) as excinfo:
        MeasureParams(contact_radius_bh=-1.0, key_half_window=-2, attack_direction={"left": 2})
    assert len(excinfo.value.problems) == 3


# --------------------------------------------------------------------------- metrics


RUN_FRAMES = 150
RUNNER: dict[str, Any] = {
    "id": "r01",
    "type": "sprint",
    "start_frame": 0,
    "peak_frame": 75,
    "end_frame": RUN_FRAMES - 1,
    "score": 0.5,
    "track_ids": [1],
    "cues": {},
}


def _run_tables(
    paths: dict[int, list[tuple[int, float, float]]],
    heights: dict[tuple[int, int], float] | None = None,
    **kinematics_options: Any,
) -> MeasureTables:
    """One 150-frame wide shot whose players move along ``paths`` {track: [(frame, x, y)]}
    (boxes of HEIGHT px unless ``heights`` {(track, frame): px}); kinematics from
    gbf.kinematics with its defaults unless ``kinematics_options`` change them."""
    rows = []
    for track, samples in paths.items():
        team = "left" if track % 2 else "right"
        for frame, x, y in samples:
            height = (heights or {}).get((track, frame), HEIGHT)
            rows.append(_track_row(frame, track, team, pitch=(x, y), height=height))
    tracks = pd.DataFrame(rows)
    tracks["team"] = tracks["team"].astype(object)
    shots = pd.DataFrame(
        {
            "shot_id": [0],
            "start_frame": [0],
            "end_frame": [RUN_FRAMES - 1],
            "shot_type": ["wide"],
            "fresh_share": [1.0],
            "median_box_h": [HEIGHT],
        }
    )
    kinematics, _ = compute_kinematics(tracks, shots, "25/1", **kinematics_options)
    return MeasureTables(tracks=tracks, shots=shots, kinematics=kinematics)


def _line(speed: float, y: float = 0.0, x0: float = 0.0) -> list[tuple[int, float, float]]:
    """A player running along x at ``speed`` m/s for the whole shot."""
    return [(f, x0 + speed * f / 25.0, y) for f in range(RUN_FRAMES)]


def _others() -> dict[int, list[tuple[int, float, float]]]:
    """Four players jogging at 1 m/s in lanes of their own (the jump test's reference)."""
    return {track: _line(1.0, y=-20.0 + 5.0 * track, x0=-30.0) for track in (3, 4, 5, 6)}


def _runner_speed(tables: MeasureTables) -> dict[str, Any]:
    return _find(measure_highlight(RUNNER, tables, _context()), "peak-speed", 1)


def test_peak_speed_ignores_a_box_that_loses_the_legs() -> None:
    # Track 1 jogs at 3 m/s; for three frames its box loses the legs (70 % of its height)
    # and the ground position moves 0.55 m away from the camera.
    path = [(f, x, y - (0.55 if 70 <= f <= 72 else 0.0)) for f, x, y in _line(3.0)]
    shrunk = {(1, f): 0.7 * HEIGHT for f in (70, 71, 72)}
    record = _runner_speed(_run_tables({1: path, **_others()}, shrunk))
    assert record["value"] == pytest.approx(3.0, rel=0.05)
    # Without the box test the dip stays in the positions; the 0.4 s chord, which is now the
    # only speed there is (the smoothed derivative this test also read is gone), reads it
    # as at most a few per cent.
    unguarded = _runner_speed(
        _run_tables({1: path, **_others()}, shrunk, max_box_height_change=None)
    )
    assert unguarded["value"] == pytest.approx(3.0, rel=0.05)


def test_peak_speed_ignores_an_id_jump() -> None:
    # Track 1 jogs at 1 m/s next to track 2, 2.5 m away; from frame 75 on, track 1's box
    # is on track 2's player. Smoothed across, the step reads as a sprint of 8.5 m/s.
    jumped = [(f, x, y + (2.5 if f >= 75 else 0.0)) for f, x, y in _line(1.0)]
    tables = _run_tables({1: jumped, 2: _line(1.0, y=2.5), **_others()})
    record = _runner_speed(tables)
    assert record["value"] == pytest.approx(1.0, rel=0.05)
    assert len(record["quality"]["segments"]) == 2  # split at the jump


def test_peak_speed_keeps_a_real_sprint() -> None:
    frames = np.arange(RUN_FRAMES)
    speed = 3.0 + 6.0 * np.exp(-0.5 * ((frames - 75) / 12.0) ** 2)  # 9 m/s at frame 75
    distance = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / 25.0)])
    path = [(int(f), -30.0 + float(d), 5.0) for f, d in zip(frames, distance, strict=True)]
    tables = _run_tables({1: path, **_others()})
    record = _runner_speed(tables)
    assert record["value"] == pytest.approx(9.0, rel=0.05)
    assert abs(record["frame"] - 75) <= 1
    # One speed for everything shown (the smoothed peak this test compared with is gone):
    # the record is the highest valid kinematics speed, which the overlay labels and the
    # sprint cue read too, so none of them shows more than the card.
    kinematics = tables.kinematics
    shown = kinematics[(kinematics["track_id"] == 1) & kinematics["valid"]]
    assert record["value"] == pytest.approx(shown["speed"].max(), abs=1e-4)
    assert record["frame"] == int(shown.loc[shown["speed"].idxmax(), "frame"])
    assert record["quality"]["speed_coverage"] > 0.8


@pytest.mark.parametrize("hold", [1, 2])
def test_peak_speed_keeps_a_sprint_whose_positions_advance_in_steps(hold: int) -> None:
    # A broadcast foot point advances in steps: with the position held every second frame
    # the old single-step jump test split this 8 m/s sprint and the card read 5.5 m/s.
    frames = np.arange(RUN_FRAMES)
    speed = 3.0 + 5.0 * np.exp(-0.5 * ((frames - 75) / 12.0) ** 2)  # 8 m/s at frame 75
    distance = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / 25.0)])
    held = distance[(frames // hold) * hold]
    path = [(int(f), -30.0 + float(d), 5.0) for f, d in zip(frames, held, strict=True)]
    record = _runner_speed(_run_tables({1: path, **_others()}))
    assert record["value"] == pytest.approx(8.0, rel=0.05)
    assert len(record["quality"]["segments"]) == 1


def test_peak_speed_ignores_an_id_switch_across_missing_frames() -> None:
    # Track 1 jogs at 1 m/s and is lost for frames 72-74; it comes back on track 2's player
    # 1.8 m away. The old allowance grew by 12 m/s per missing frame and read 4.9 m/s.
    jumped = [(f, x, y + (1.8 if f >= 75 else 0.0)) for f, x, y in _line(1.0) if not 72 <= f <= 74]
    record = _runner_speed(_run_tables({1: jumped, 2: _line(1.0, y=1.8), **_others()}))
    assert record["value"] == pytest.approx(1.0, rel=0.05)
    assert len(record["quality"]["segments"]) == 2


def test_peak_speed_is_the_highest_valid_speed_in_the_window() -> None:
    # gbf.kinematics stores every speed as the 0.4 s chord (10 frames at 25 fps) of the
    # smoothed positions with its validity, and peak-speed is the highest valid one on the
    # window's pitch frames (before, gbf.measure built its own chords from the positions and
    # the old test planted a one-frame burst in a smoothed speed column that no longer
    # exists). Track 1 runs at 6.25 m/s and at 7.5 m/s into frames 21-30, so the chord centred
    # on frame 25 reads 7.5 m/s. Higher speeds that are invalid (frame 33), before the window
    # (frame 7) or in a 10-frame segment (track 2) never count.
    rows = _chord_rows(
        1,
        0,
        range(0, 50),
        steps={f: 0.3 for f in range(21, 31)},
        valid=set(range(5, 45)) - {33, 34},
        override={7: 9.0, 33: 11.0},
    )
    short = _chord_rows(2, 1, range(20, 30), valid=set(range(20, 30)), override={25: 6.0})
    kinematics = pd.concat([rows, short], ignore_index=True)
    tables = MeasureTables(tracks=_tracks(), shots=_shots(), kinematics=kinematics)
    records = _records(tables)
    striker = _find(records, "peak-speed", 1)
    assert striker["status"] == "ok"
    assert striker["value"] == pytest.approx(7.5)
    assert striker["display"] == "27.0 km/h"
    assert striker["label_zh"] == "峰值速度（0.4秒平均）"
    assert striker["frame"] == 25
    quality = striker["quality"]
    assert (quality["chord_start_frame"], quality["chord_end_frame"]) == (20, 30)
    used = [f for f in range(10, 45) if f not in (33, 34)]  # valid frames in the window
    assert striker["frames"] == used
    # Tracked: the 40 pitch frames 10-49 of the window on which track 1 has a position. The
    # track ends with the shot at frame 49, and no valid speed can lie within 10 frames
    # (filter half-window plus chord half) of that end: frames 45-49 are left out of the
    # coverage.
    assert (quality["speed_frames"], quality["tracked_frames"]) == (len(used), 40)
    assert quality["margin_frames"] == 5
    assert quality["speed_coverage"] == pytest.approx(len(used) / 35, abs=1e-4)
    assert (striker["coverage"]["frames_considered"], striker["coverage"]["frames_used"]) == (
        35,
        len(used),
    )
    assert striker["method"]["parameters"]["chord_frames"] == 10
    assert striker["method"]["parameters"]["chord_s"] == 0.4
    assert striker["coverage"]["window"] == [10, 79]
    assert _find(records, "peak-speed", 2)["reasonCode"] == "no_valid_speed"
    close = _find(records, "peak-speed", 7)
    assert close["reasonCode"] == "no_kinematics"
    in_close_shot = {**HIGHLIGHT, "start_frame": 55, "peak_frame": 60, "end_frame": 70}
    only_close = _records(tables, in_close_shot)
    assert _find(only_close, "peak-speed", 1)["reasonCode"] == "no_pitch_frames"


def _fragmented() -> list[tuple[int, float, float]]:
    """Track 1 at 5 m/s, lost for frames 25-29, 55-59 and 85-89, and 6 m/s at frame 120."""
    frames = np.arange(RUN_FRAMES)
    speed = 5.0 + np.exp(-0.5 * ((frames - 120) / 6.0) ** 2)
    distance = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / 25.0)])
    lost = {f for start in (25, 55, 85) for f in range(start, start + 5)}
    return [
        (int(f), -40.0 + float(d), 5.0)
        for f, d in zip(frames, distance, strict=True)
        if int(f) not in lost
    ]


def test_peak_speed_is_withheld_when_valid_speeds_cover_too_little() -> None:
    # Track 1 is lost three times for 5 frames, more than gbf.kinematics bridges, so it is
    # split into four segments, and no valid speed lies within 10 frames of a split: valid
    # speeds cover 55 of the 115 tracked frames that are not within 10 frames of the
    # track's start and end. The player may have run faster in the rest, so the highest
    # speed found is no peak; it stays in quality, and the reason names the splits.
    tables = _run_tables({1: _fragmented(), **_others()})
    record = _runner_speed(tables)
    assert record["status"] == "unavailable"
    assert record["value"] is None
    assert record["display"] == "—"
    assert record["reasonCode"] == "speed_coverage_split"
    quality = record["quality"]
    assert (quality["tracked_frames"], quality["margin_frames"]) == (135, 20)
    assert quality["speed_coverage"] == pytest.approx(55 / 115, abs=1e-4)
    assert quality["uncovered_split_frames"] == 60
    assert quality["uncovered_rejected_frames"] == 0
    assert quality["highest_valid_speed_mps"] == pytest.approx(6.0, rel=0.02)
    # The reason gives the record's own numbers: the trusted frames, and the highest speed
    # as a lower bound, written as the table writes speeds.
    lower = display_text(quality["highest_valid_speed_mps"], "m/s", 1, Fraction(25))
    frame = quality["highest_valid_speed_frame"]
    assert record["reason_zh"] == (
        "可信速度只覆盖 55/115 帧，其余帧多在轨迹断开处前后（漏检或剔除的帧太多、身份或标定"
        f"跳变）：测得的最高 {lower}（第 {frame} 帧）只是下限，不作为峰值"
    )
    assert record["method"]["parameters"]["peak_speed_min_coverage"] == 0.5
    validate_record(record)
    lenient = measure_highlight(
        RUNNER, tables, _context(), MeasureParams(peak_speed_min_coverage=0.3)
    )
    assert _find(lenient, "peak-speed", 1)["value"] == pytest.approx(6.0, rel=0.02)
    assert all(
        r["metric_id"] != "peak-speed"
        for r in card_records(measure_highlight(RUNNER, tables, _context()), RUNNER)
    )


@pytest.mark.parametrize("frames", [25, 30, 39])
def test_peak_speed_of_a_clean_short_track_is_shown(frames: int) -> None:
    # A player is tracked for 25-39 frames only, cleanly, at 5 m/s. No valid speed can lie
    # within 10 frames (the filter half-window plus the chord half) of a track's start or
    # end, so those frames are not counted against the coverage: it was (n - 20) / n, below
    # 0.5 for every track shorter than 40 frames, and the reason blamed rejected positions
    # or a broken track.
    path = [(f, -30.0 + 5.0 * f / 25.0, 5.0) for f in range(50, 50 + frames)]
    record = _runner_speed(_run_tables({1: path, **_others()}))
    assert record["status"] == "ok"
    assert record["value"] == pytest.approx(5.0, rel=1e-6)
    assert record["quality"]["margin_frames"] == 20
    assert record["quality"]["speed_coverage"] == pytest.approx(1.0)


def _sprint_path(top: float, lost: range = range(0)) -> list[tuple[int, float, float]]:
    """A run along x from 3 m/s up to ``top`` m/s at frame 75 and back (sigma 12 frames),
    the frames of ``lost`` missing."""
    frames = np.arange(RUN_FRAMES)
    speed = 3.0 + (top - 3.0) * np.exp(-0.5 * ((frames - 75) / 12.0) ** 2)
    distance = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / 25.0)])
    return [
        (int(f), -40.0 + float(d), 5.0)
        for f, d in zip(frames, distance, strict=True)
        if int(f) not in lost
    ]


@pytest.mark.parametrize("top", [8.0, 9.0])
def test_peak_speed_is_withheld_when_the_peak_may_lie_in_a_gap(top: float) -> None:
    # The box of an 8-9 m/s sprinter loses the legs for 4 frames at the peak: the box test
    # rejects them, a gap longer than kinematics bridges, and no speed is valid for 24
    # frames around the peak. The highest valid speed (about 6 m/s at the edge of the gap)
    # was shown as the peak (6.03 and 6.64 m/s, coverage 0.71); the speed across the gap
    # is the sprint's. The record is withheld, the lower bound kept for audit.
    lost = range(73, 77)
    path = [(f, x, y - (0.5 if f in lost else 0.0)) for f, x, y in _sprint_path(top)]
    shrunk = {(1, f): 0.7 * HEIGHT for f in lost}
    tables = _run_tables({1: path, **_others()}, shrunk)
    record = _runner_speed(tables)
    assert record["status"] == "unavailable"
    assert record["reasonCode"] == "speed_peak_hidden"
    quality = record["quality"]
    assert quality["speed_coverage"] >= 0.5  # not withheld for the coverage
    assert quality["highest_valid_speed_mps"] < 0.8 * top
    assert quality["peak_next_to_uncovered"] in ("before", "after")
    assert quality["speed_across_uncovered_mps"] > quality["highest_valid_speed_mps"]
    # The reason says that the highest speed measured is only a lower bound, and why.
    lower = display_text(quality["highest_valid_speed_mps"], "m/s", 1, Fraction(25))
    first, last = quality["uncovered_between_frames"]
    assert record["reason_zh"] == (
        f"测得的最高可信速度 {lower}（第 {quality['highest_valid_speed_frame']} 帧）"
        f"紧挨着一段测不出速度的轨迹，也比跨过第 {first}–{last} 帧这段测不出速度的轨迹时的"
        f"平均速度慢：真实峰值可能在那段里，{lower} 只是下限，不作为峰值"
    )
    validate_record(record)
    assert all(
        r["metric_id"] != "peak-speed"
        for r in card_records(measure_highlight(RUNNER, tables, _context()), RUNNER)
    )


@pytest.mark.parametrize(
    ("sprint", "gaps"),
    [((60, 90), [(73, 77)]), ((60, 90), [(72, 78)]), ((62, 88), [(72, 76), (85, 89)])],
    ids=["4 frames", "6 frames", "twice 4 frames"],
)
def test_peak_speed_is_withheld_when_tracking_is_lost_in_a_sprint(
    sprint: tuple[int, int], gaps: list[tuple[int, int]]
) -> None:
    # A player jogs at 3 m/s and sprints at 8 m/s for about a second, and the track loses
    # the player for 4 or 6 frames in the middle of the sprint (or twice for 4 frames):
    # the highest valid speed was shown as the peak and read 6.81, 6.31 and 5.23 m/s.
    frames = np.arange(RUN_FRAMES)
    speed = np.where((frames >= sprint[0]) & (frames < sprint[1]), 8.0, 3.0)
    distance = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / 25.0)])
    missing = {f for start, stop in gaps for f in range(start, stop)}
    path = [
        (int(f), -40.0 + float(d), 5.0)
        for f, d in zip(frames, distance, strict=True)
        if int(f) not in missing
    ]
    record = _runner_speed(_run_tables({1: path, **_others()}))
    assert record["status"] == "unavailable"
    assert record["reasonCode"] == "speed_peak_hidden"
    assert record["quality"]["highest_valid_speed_mps"] < 7.0


def test_peak_speed_of_a_steady_run_next_to_a_gap_is_shown() -> None:
    # A steady 5 m/s run that the track loses for frames 15-18 at its start: the frames
    # before the gap are too few for a valid speed. The highest valid speed is reached all
    # along the run, not only next to the gap, so it is no peak hidden by the gap.
    path = [step for step in _line(5.0, x0=-40.0) if not 15 <= step[0] <= 18]
    record = _runner_speed(_run_tables({1: path, **_others()}))
    assert record["status"] == "ok"
    assert record["value"] == pytest.approx(5.0, rel=1e-6)


def test_peak_speed_is_kept_when_a_gap_lies_away_from_the_peak() -> None:
    # A 4-frame gap in the slow part of the run (frames 120-123): the peak lies well inside
    # its run of valid speeds, and the player crosses the gap at 3 m/s.
    record = _runner_speed(_run_tables({1: _sprint_path(8.0, range(120, 124)), **_others()}))
    assert record["status"] == "ok"
    assert record["value"] == pytest.approx(7.84, rel=0.02)
    assert record["quality"]["segments"] == [0, 1]


def test_peak_speed_refuses_speeds_that_are_not_its_chord() -> None:
    # A kinematics table whose top speed is not the 10-frame chord of its positions (another
    # chord, or a smoothed derivative) would make the record state a chord it does not hold.
    rows = _chord_rows(1, 0, range(0, 50), valid=set(range(10, 40)), override={25: 9.0})
    tables = MeasureTables(tracks=_tracks(), shots=_shots(), kinematics=rows)
    with pytest.raises(MeasureError, match="chord"):
        _records(tables)
    with pytest.raises(MeasureError, match="chord"):
        measure_highlight(HIGHLIGHT, tables, _context(), MeasureParams(peak_speed_window_s=0.8))


def test_peak_speed_label_states_the_chord_measured() -> None:
    # At 24 fps the 0.4 s chord rounds to 10 frames, 0.417 s: the label says 0.42 s, not 0.4.
    assert metric_label("peak-speed") == "峰值速度（0.4秒平均）"
    assert metric_label("peak-speed", rate=Fraction(25)) == "峰值速度（0.4秒平均）"
    assert metric_label("peak-speed", rate=Fraction(24)) == "峰值速度（0.42秒平均）"
    assert metric_label("peak-speed", rate=Fraction(30000, 1001)) == "峰值速度（0.4秒平均）"
    frames = range(RUN_FRAMES)
    rows = [_track_row(f, 1, "left", pitch=(-30.0 + 6.0 * f / 24.0, 0.0)) for f in frames]
    tracks = pd.DataFrame(rows)
    tracks["team"] = tracks["team"].astype(object)
    shots = pd.DataFrame(
        {
            "shot_id": [0],
            "start_frame": [0],
            "end_frame": [RUN_FRAMES - 1],
            "median_box_h": [HEIGHT],
        }
    )
    kinematics, _ = compute_kinematics(tracks, shots, "24/1")
    tables = MeasureTables(tracks=tracks, shots=shots, kinematics=kinematics)
    record = _find(measure_highlight(RUNNER, tables, _context(fps="24/1")), "peak-speed", 1)
    assert record["label_zh"] == "峰值速度（0.42秒平均）"
    assert record["method"]["parameters"]["chord_frames"] == 10
    assert record["method"]["parameters"]["chord_s"] == pytest.approx(10 / 24, abs=1e-4)
    assert record["value"] == pytest.approx(6.0)
    assert (record["quality"]["chord_end_frame"] - record["quality"]["chord_start_frame"]) == 10


def test_card_prefers_the_records_nearest_the_key_frame() -> None:
    def contact(track: int, frame: int) -> dict[str, Any]:
        return build_record(
            "ball-contact-frame",
            context=_context(),
            status="ok",
            value=frame,
            frame=frame,
            frames=[frame],
            highlight_id="h01",
            track_id=track,
            coverage={"window": [10, 79], "frames_considered": 70, "frames_used": 1, "share": 1},
        )

    def shown(card: list[Any]) -> tuple[int, int]:
        (record,) = [r for r in card if r["metric_id"] == "ball-contact-frame"]
        return record["track_id"], record["frame"]

    # In player order: another player's contact far before the key frame (30) comes first.
    records = [contact(1, 12), contact(2, 20), contact(3, 29), contact(2, 31)]
    # The focus player's contact nearest the key frame, not its first one.
    assert shown(card_records(records, HIGHLIGHT, prefer_track=2)) == (2, 31)
    # A focus player without a contact: the contact nearest the key frame, whoever made it.
    assert shown(card_records(records, HIGHLIGHT, prefer_track=8)) == (3, 29)
    assert shown(card_records(records, HIGHLIGHT)) == (3, 29)
    # The key frame defaults to the highlight's peak frame; key_frame overrides it.
    assert shown(card_records(records, HIGHLIGHT, key_frame=13)) == (1, 12)
    assert shown(card_records(records, HIGHLIGHT, prefer_track=2, key_frame=13)) == (2, 20)


def test_distance_labels_state_the_median_window() -> None:
    # The overlay line shows one frame's distance; the record is a median over +/-2 frames.
    assert METRICS["nearest-opponent-distance"].label_zh == "最近对手距离（±2帧中位数）"
    assert METRICS["nearest-opponent-image-distance"].label_zh == "画面最近对手距离（±2帧中位数）"
    for metric, spec in METRICS.items():
        assert len(spec.label_zh) <= 16, metric  # the card cuts longer labels
        assert metric_label(metric) == spec.label_zh
    tables = MeasureTables(tracks=_tracks(), shots=_shots())
    records = _records(tables)
    pitch = _find(records, "nearest-opponent-distance", 1)
    assert pitch["label_zh"] == "最近对手距离（±2帧中位数）"
    image = _find(records, "nearest-opponent-image-distance", 7)
    assert image["label_zh"] == "画面最近对手距离（±2帧中位数）"
    wider = measure_highlight(HIGHLIGHT, tables, _context(), MeasureParams(key_half_window=3))
    record = _find(wider, "nearest-opponent-distance", 1)
    assert record["label_zh"] == "最近对手距离（±3帧中位数）"
    assert record["frames"] == [27, 28, 29, 30, 31, 32, 33]
    validate_record(record)


@pytest.mark.parametrize("half", [1, 5, 9])
def test_window_labels_fit_the_card(half: int) -> None:
    # Every median label says 中位数, so one card never names the statistic two ways (the
    # image label said 中值 for two-digit windows, while the pitch label kept 中位数).
    params = MeasureParams(key_half_window=half)
    labels = {metric: metric_label(metric, params) for metric in METRICS}
    assert all(len(label) <= measure.CARD_LABEL_MAX_CHARS for label in labels.values())
    assert labels["nearest-opponent-distance"] == f"最近对手距离（±{half}帧中位数）"
    assert labels["nearest-opponent-image-distance"] == f"画面最近对手距离（±{half}帧中位数）"
    assert not any("中值" in label for label in labels.values())


@pytest.mark.parametrize("half", [10, 25, 100])
def test_window_labels_that_do_not_fit_are_refused(half: int) -> None:
    # With a two-digit window the image label would be 17 characters, one more than the
    # card shows; such a window is refused rather than labelled with another word.
    with pytest.raises(MeasureError, match="16 characters"):
        MeasureParams(key_half_window=half)


def test_nearest_opponent_is_the_median_over_the_key_frames() -> None:
    overrides = {(30, 2): _track_row(30, 2, "right", pitch=(41.0, 0.0))}
    tables = MeasureTables(tracks=_tracks(overrides), shots=_shots())
    records = _records(tables)
    record = _find(records, "nearest-opponent-distance", 1)
    assert record["status"] == "ok"
    assert record["value"] == 3.0  # median of 3, 3, 1, 3, 3
    assert record["frames"] == [28, 29, 30, 31, 32]
    assert record["quality"]["frames_min_m"] == 1.0
    assert record["quality"]["opponent_track_id"] == 2
    assert record["quality"]["goalkeeper_distance_m"] == 10.0
    assert record["track_ids"] == [1, 2]
    assert record["tier"] == "A"
    unknown = {(f, 1): {"team": None} for f in range(0, 50)}
    tables = MeasureTables(tracks=_tracks(unknown), shots=_shots())
    assert _find(_records(tables), "nearest-opponent-distance", 1)["reasonCode"] == "team_unknown"


def _with_rows(tracks: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    joined = pd.concat([tracks, pd.DataFrame(rows)], ignore_index=True)
    joined["team"] = joined["team"].astype(object)
    return joined


def test_nearest_opponent_is_withheld_when_a_player_without_a_team_is_nearer() -> None:
    # gbf.team_check clears team labels that contradict the shirt colours: a player without
    # a label may be the nearest opponent, so no value when one is nearer than every
    # labelled opponent.
    def cleared(x: float) -> list[dict[str, Any]]:
        return [_track_row(f, 4, None, pitch=(x, 0.0)) for f in range(0, PITCH_END + 1)]

    near = MeasureTables(tracks=_with_rows(_tracks(), cleared(41.5)), shots=_shots())
    record = _find(_records(near), "nearest-opponent-distance", 1)
    assert record["status"] == "unavailable"
    assert record["reasonCode"] == "unknown_team_nearer"
    assert record["quality"]["unknown_team_track_ids"] == [4]
    validate_record(record)
    far = MeasureTables(tracks=_with_rows(_tracks(), cleared(70.0)), shots=_shots())
    assert _find(_records(far), "nearest-opponent-distance", 1)["value"] == 3.0
    # In the close shot: a player without a label between tracks 7 and 8 ...
    between = [
        _track_row(f, 10, None, foot=(900.0, 1000.0), height=400.0, usable=False)
        for f in range(CLOSE_START, CLOSE_END + 1)
    ]
    tables = MeasureTables(tracks=_with_rows(_tracks(), between), shots=_shots())
    image = _find(_records(tables), "nearest-opponent-image-distance", 7)
    assert image["reasonCode"] == "unknown_team_nearer"
    assert image["quality"]["unknown_team_track_ids"] == [10]
    # ... and the only opponent in view losing its label (was "no_opponent").
    tracks = _tracks()
    tracks.loc[tracks["track_id"] == 8, "team"] = None
    alone = _find(
        _records(MeasureTables(tracks=tracks, shots=_shots())),
        "nearest-opponent-image-distance",
        7,
    )
    assert alone["reasonCode"] == "unknown_team_nearer"
    assert alone["quality"]["unknown_team_track_ids"] == [8]


def test_nearest_opponent_never_takes_a_box_on_the_players_own_spot() -> None:
    # A box around two players that ends at the feet of track 1 (the detector merging it with
    # the player behind) has track 1's foot point: it is no opponent standing 0.0 m away,
    # labelled or not, on the grass or in the image.
    def merged(team: str | None) -> list[dict[str, Any]]:
        return [
            _track_row(f, 41, team, pitch=(40.03, 0.0), foot=(1601.2, 600.0), height=170.0)
            for f in range(28, 33)
        ]

    for team in ("right", None):
        tables = MeasureTables(tracks=_with_rows(_tracks(), merged(team)), shots=_shots())
        record = _find(_records(tables), "nearest-opponent-distance", 1)
        assert (record["status"], record["value"]) == ("ok", 3.0), team
        assert record["quality"]["opponent_track_id"] == 2
    close = [
        _track_row(f, 42, "right", foot=(800.5, 1000.0), height=450.0, usable=False)
        for f in range(CLOSE_START, CLOSE_END + 1)
    ]
    tables = MeasureTables(tracks=_with_rows(_tracks(), close), shots=_shots())
    image = _find(_records(tables), "nearest-opponent-image-distance", 7)
    assert image["quality"]["opponent_track_id"] == 8
    assert image["value"] == pytest.approx(0.5, abs=1e-4)  # 200 px over 400 px boxes


def test_nearest_opponent_is_never_a_person_off_the_pitch() -> None:
    # Track 1 near the touchline; a coach of the other team stands in the technical area,
    # a metre beyond the margin: nearer than any player, but no player in play.
    near_line = {(f, 1): {"pitch_y": 33.0} for f in range(0, PITCH_END + 1)}
    beyond = 34.0 + OFF_PITCH_MARGIN_M + 1.0
    coach = [
        _track_row(f, 30, "right", pitch=(40.0, beyond), foot=(1500.0, 700.0))
        for f in range(0, PITCH_END + 1)
    ]
    tables = MeasureTables(tracks=_with_rows(_tracks(near_line), coach), shots=_shots())
    record = _find(_records(tables), "nearest-opponent-distance", 1)
    assert record["quality"]["opponent_track_id"] == 3
    assert record["value"] == pytest.approx(math.hypot(5.0, 27.0), abs=1e-4)


@pytest.mark.parametrize("checked", [False, True])
def test_nearest_image_opponent_ignores_unchecked_without_ground(checked: bool) -> None:
    bystanders = [
        _track_row(f, 10, "right", foot=(880.0, 1000.0), height=400.0, usable=False)
        for f in range(CLOSE_START, CLOSE_END + 1)
    ]
    tables = MeasureTables(
        tracks=_with_rows(_tracks(), bystanders),
        shots=_shots(),
        team_verdicts={
            (7, "left"): "consistent",
            (8, "right"): "consistent",
            (10, "right"): "consistent" if checked else "unchecked",
        },
        frame_size=(1920, 1080),
    )
    record = _find(_records(tables), "nearest-opponent-image-distance", 7)
    assert record["quality"]["opponent_track_id"] == (10 if checked else 8)


@pytest.mark.parametrize("cut_off", [False, True])
def test_nearest_pitch_opponent_respects_camera_edge(cut_off: bool) -> None:
    tracks = _tracks({(f, 1): {"pitch_y": 33.0} for f in range(0, PITCH_END + 1)})
    bystanders = [
        _track_row(f, 30, "right", pitch=(40.0, 34.2), foot=(1500.0, 1079.5 if cut_off else 1000.0))
        for f in range(0, PITCH_END + 1)
    ]
    frames = list(range(PITCH_END + 1))
    calib = pd.DataFrame({"frame_index": frames, "C": [[0.0, 60.0, -20.0]] * len(frames)})
    tables = MeasureTables(
        tracks=_with_rows(tracks, bystanders),
        shots=_shots(),
        calib=calib,
        frame_size=(1920, 1080),
    )
    record = _find(_records(tables), "nearest-opponent-distance", 1)
    assert record["quality"]["opponent_track_id"] == (3 if cut_off else 30)


def _camera_tracks(camera: Any, players: dict[int, tuple[str, str, tuple[float, float]]]) -> Any:
    """Track rows of a wide shot (frames 0-49) whose foot points are the ground positions
    seen by ``camera``, so that one homography maps the grass to the frame."""
    rows = []
    for frame in range(0, PITCH_END + 1):
        for track, (team, role, (x, y)) in players.items():
            foot, _, _ = project(camera, np.array([x, y, 0.0]))
            head, _, _ = project(camera, np.array([x, y, -1.8]))
            rows.append(
                _track_row(
                    frame, track, team, role=role, pitch=(x, y),
                    foot=(float(foot[0]), float(foot[1])), height=float(foot[1] - head[1]),
                )
            )  # fmt: skip
    table = pd.DataFrame(rows)
    table["team"] = table["team"].astype(object)
    return table


def _leaning(row: dict[str, Any], degrees: float) -> dict[str, Any]:
    """A pose row whose shoulders lean ``degrees`` from upright in the image (+ right)."""
    points = np.asarray(row["keypoints"], dtype=np.float64).reshape(-1, 3)
    rise = (row["y2"] - row["y1"]) * 0.3  # hips at 0.5, shoulders at 0.2 of the box
    for name in ("left_shoulder", "right_shoulder"):
        points[HALPE26_KEYPOINTS.index(name), 0] += rise * math.tan(math.radians(degrees))
    return {**row, "keypoints": points.reshape(-1).tolist()}


def test_body_orientation_leaves_out_a_facing_reversed_against_the_lean() -> None:
    # Track 1 leans 30 degrees to the left of the picture (a lunge that way) on every frame
    # around the key frame; the lift faces it left (pitch_zup 180, the picture's left for
    # the main camera) except on frame 30, where it faces right: a front-back reversal. That
    # frame is left out of the mean, so the four others agree.
    camera = synthetic_broadcast_camera(1920, 1080, target=(5.0, 20.0, 0.0))
    players = {
        1: ("left", "player", (5.0, 20.0)), 2: ("right", "player", (8.0, 20.0)),
        3: ("right", "player", (10.0, 26.0)), 4: ("right", "player", (0.0, 15.0)),
        9: ("right", "goalkeeper", (14.0, 18.0)),
    }  # fmt: skip
    tracks = _camera_tracks(camera, players)
    facings = {28: 180.0, 29: 180.0, 30: 0.0, 31: 180.0, 32: 180.0}
    lift = pd.DataFrame([_lift_row(f, 1, facing_deg=angle) for f, angle in facings.items()])

    def orientation(lean: float) -> dict[str, Any]:
        pose = _pose(tracks, [(f, 1) for f in facings])
        pose = pd.DataFrame([_leaning(row, lean) for row in pose.to_dict("records")])
        tables = MeasureTables(tracks=tracks, shots=_shots(), pose=pose, lift=lift)
        return _find(_records(tables), "body-orientation", 1)

    record = orientation(-30.0)
    assert record["status"] == "ok"
    assert record["frames"] == [28, 29, 31, 32]
    assert record["quality"]["rows_reversed"] == 1
    assert record["quality"]["facing_deg_pitch_zup"] == pytest.approx(180.0, abs=1e-6)
    validate_record(record)
    # Upright, nothing tells which way is right: the frames disagree, so no value.
    upright = orientation(0.0)
    assert (upright["reasonCode"], upright["quality"]["rows_reversed"]) == ("facing_uncertain", 0)


@pytest.mark.parametrize(
    ("image_lean", "facings", "leans", "expected", "reversed_count"),
    [
        (-30.0, [0.0, 0.0, 0.0, 180.0, 180.0], [-40.0, -40.0, -40.0, 10.0, 20.0], 15.0, 3),
        (-30.0, [0.0] * 5, [-20.0] * 5, None, 5),
        (0.0, [0.0] * 5, [12.0] * 5, 12.0, 0),
        (-30.0, [180.0] * 5, [12.0] * 5, 12.0, 0),
    ],
)
def test_trunk_lean_excludes_explicit_facing_reversals_only(
    image_lean: float,
    facings: list[float],
    leans: list[float],
    expected: float | None,
    reversed_count: int,
) -> None:
    camera = synthetic_broadcast_camera(1920, 1080, target=(5.0, 20.0, 0.0))
    tracks = _camera_tracks(
        camera,
        {1: ("left", "player", (5.0, 20.0)), 2: ("right", "player", (8.0, 20.0)),
         3: ("right", "player", (10.0, 26.0)), 4: ("right", "player", (0.0, 15.0))},
    )  # fmt: skip
    frames = list(range(28, 33))
    pose = _pose(tracks, [(f, 1) for f in frames])
    pose = pd.DataFrame([_leaning(row, image_lean) for row in pose.to_dict("records")])
    # Lean has its own sigma gate: poor facing confidence/sigma must not reject it.
    lift = pd.DataFrame(
        [
            _lift_row(
                f, 1, facing_deg=facing, facing_conf=0.1, facing_sigma_deg=50.0, trunk_lean_deg=lean
            )
            for f, facing, lean in zip(frames, facings, leans, strict=True)
        ]
    )
    record = _find(
        _records(MeasureTables(tracks=tracks, shots=_shots(), pose=pose, lift=lift)),
        "trunk-lean-at-contact",
        1,
    )
    assert record["quality"]["rows_reversed"] == reversed_count
    assert record["method"]["parameters"]["facing_lean_deg"] == measure.FACING_LEAN_DEG
    assert record["method"]["parameters"]["facing_side_share"] == measure.FACING_SIDE_SHARE
    if expected is None:
        assert record["status"] == "unavailable"
        assert record["reasonCode"] == "facing_uncertain"
        assert record["value"] is None
        assert record["quality"]["rows_passing_gates"] == 0
    else:
        assert record["status"] == "ok"
        assert record["value"] == expected
        assert record["frames"] == [
            f
            for f, facing in zip(frames, facings, strict=True)
            if image_lean == 0.0 or facing == 180.0
        ]
    validate_record(record)


def test_the_facing_reversal_check() -> None:
    lean = measure.FACING_LEAN_DEG
    side = measure.FACING_SIDE_SHARE
    assert not measure.facing_reversed(-lean, 1.0), "upright enough: not checked"
    assert measure.facing_reversed(-lean - 0.1, side), "leans left, faces right"
    assert not measure.facing_reversed(-lean - 0.1, side - 0.01), "faces mostly up or down"
    assert not measure.facing_reversed(lean + 0.1, 0.9), "leans right, faces right"
    assert measure.facing_reversed(lean + 0.1, None), "a facing that cannot be placed"
    assert not measure.facing_reversed(math.nan, -1.0), "no trunk seen: not checked"
    gates = measure.FacingGates(0.5, 15.0, 80.0)
    ok = {"lifted": True, "facing_deg": 0.0, "facing_conf": 0.9, "facing_sigma_deg": 5.0,
          "box_height_px": 80.0, "image_trunk_deg": -30.0, "image_dx": -1.0}  # fmt: skip
    assert measure.facing_frame_ok(gates, **ok)
    assert not measure.facing_frame_ok(gates, **{**ok, "image_dx": 1.0})
    assert not measure.facing_frame_ok(gates, **{**ok, "box_height_px": 79.9})
    assert not measure.facing_frame_ok(gates, **{**ok, "lifted": False})
    camera = synthetic_broadcast_camera(1920, 1080)

    def to_image(points: np.ndarray) -> np.ndarray:
        return project(camera, np.column_stack([points, np.zeros(len(points))]))[0]

    assert measure.facing_image_dx(to_image, (0.0, 0.0), 0.0) == pytest.approx(1.0, abs=1e-3)
    assert measure.facing_image_dx(to_image, (0.0, 0.0), 180.0) == pytest.approx(-1.0, abs=1e-3)
    away = measure.facing_image_dx(to_image, (0.0, 0.0), 90.0)
    assert away is not None
    assert abs(away) < 0.05, "towards the far touchline: up the picture"


def test_the_orientation_label_states_its_window() -> None:
    # The arrow shows one frame's facing; the card's angle is the mean over +/-2 frames.
    assert METRICS["body-orientation"].label_zh == "朝向夹角（±2帧均值）"
    assert metric_label("body-orientation", MeasureParams(key_half_window=4)) == (
        "朝向夹角（±4帧均值）"
    )


def _lift_tables(lift_rows: list[dict[str, Any]], **changes: Any) -> MeasureTables:
    tracks = changes.pop("tracks", _tracks())
    keys = [(int(row["frame"]), int(row["track_id"])) for row in lift_rows]
    pose = _pose(tracks, keys)
    return MeasureTables(
        tracks=tracks, shots=_shots(), pose=pose, lift=pd.DataFrame(lift_rows), **changes
    )


def test_body_orientation_is_the_angle_to_the_attacked_goal() -> None:
    player = (40.0, 10.0)  # SoccerNet: 10 m towards the near touchline
    tracks = _tracks({(f, 1): {"pitch_y": player[1]} for f in range(0, 50)})
    rows = [_lift_row(f, 1, facing_deg=0.0) for f in range(28, 33)]
    records = _records(_lift_tables(rows, tracks=tracks))
    record = _find(records, "body-orientation", 1)
    # pitch_zup: player at (40, -10), goal at (52.5, 0): direction atan2(10, 12.5).
    expected = math.degrees(math.atan2(10.0, 12.5))
    assert record["status"] == "ok"
    assert record["value"] == pytest.approx(expected, abs=1e-3)
    assert record["quality"]["facing_deg_pitch_zup"] == 0.0
    assert record["quality"]["signed_angle_deg"] == pytest.approx(-expected, abs=1e-3)
    assert record["quality"]["reference"] == "highlight_key_frame"
    assert record["quality"]["contact_frame_confirmed"] is False
    assert record["tier"] == "A_candidate"
    assert record["claimType"] == "estimate"
    lean = _find(records, "trunk-lean-at-contact", 1)
    assert lean["value"] == 10.0
    assert lean["quality"]["other_solution_deg"] == 18.0


def test_orientation_and_lean_gates() -> None:
    noisy = [
        _lift_row(f, 1, facing_sigma_deg=25.0, trunk_lean_sigma_deg=30.0) for f in range(28, 33)
    ]
    records = _records(_lift_tables(noisy))
    assert _find(records, "body-orientation", 1)["reasonCode"] == "facing_uncertain"
    assert _find(records, "trunk-lean-at-contact", 1)["reasonCode"] == "lean_uncertain"
    assert _find(records, "trunk-lean-at-contact", 1)["quality"]["rows_reversed"] == 0
    flipping = [_lift_row(f, 1, facing_deg=0.0 if f % 2 else 180.0) for f in range(28, 33)]
    records = _records(_lift_tables(flipping))
    assert _find(records, "body-orientation", 1)["reasonCode"] == "facing_uncertain"
    small = _tracks({(f, 1): {"y1": 600.0 - 60.0} for f in range(0, 50)})
    rows = [_lift_row(f, 1) for f in range(28, 33)]
    records = _records(_lift_tables(rows, tracks=small))
    assert _find(records, "body-orientation", 1)["reasonCode"] == "box_too_small"
    assert _find(records, "trunk-lean-at-contact", 1)["reasonCode"] == "box_too_small"
    not_lifted = [
        _lift_row(f, 1, method="image_plane_only", reason="feet_low_confidence")
        for f in range(28, 33)
    ]
    records = _records(_lift_tables(not_lifted))
    assert _find(records, "body-orientation", 1)["reasonCode"] == "not_lifted"
    no_pose = MeasureTables(tracks=_tracks(), shots=_shots())
    assert _find(_records(no_pose), "trunk-lean-at-contact", 1)["reasonCode"] == "no_pose"


def test_close_shot_players_get_image_plane_values_only() -> None:
    rows = [
        _lift_row(f, 7, method="image_plane_only", reason="no_calibration", image_lean_deg=-12.0)
        for f in range(58, 63)
    ]
    records = _records(_lift_tables(rows))
    lean = _find(records, "image-trunk-lean", 7)
    assert lean["value"] == -12.0
    assert lean["frame"] == 60  # the joined shot's moment: track 7 is not near frame 30
    assert lean["tier"] == "image_plane"
    assert lean["plane"] == "image"
    assert _find(records, "image-knee-angle-left", 7)["value"] == 160.0
    assert _find(records, "nearest-opponent-distance", 7)["reasonCode"] == "not_pitch_usable"
    image = _find(records, "nearest-opponent-image-distance", 7)
    assert image["value"] == pytest.approx(0.5)  # 200 px between feet / 400 px boxes
    assert image["quality"]["opponent_track_id"] == 8
    for record in records:
        if record["track_id"] == 7 and record["plane"] == "pitch":
            assert record["status"] == "unavailable"
    upper_body = MeasureTables(tracks=_tracks(), shots=_shots(close_box_h=700.0))
    found = _find(_records(upper_body), "nearest-opponent-image-distance", 7)
    assert found["reasonCode"] == "feet_not_visible"


def test_contacts_match_foot_keypoints_then_fall_back_to_the_box() -> None:
    tracks = _tracks()
    striker = tracks[(tracks["frame"] == 30) & (tracks["track_id"] == 1)].iloc[0].to_dict()
    toe_u = striker["x1"] + 0.66 * (striker["x2"] - striker["x1"])
    event = _event(30, toe_u, striker["y2"] + 8.0)  # 8 px below the left big toe
    pose = _pose(tracks, [(30, 1), (30, 2)])
    tables = MeasureTables(
        tracks=tracks, shots=_shots(), ball_events=pd.DataFrame([event]), pose=pose
    )
    matches, examined = match_contacts(tables, 10, 79)
    assert examined == 1
    assert len(matches) == 1
    assert matches[0].track_id == 1
    assert matches[0].method == "foot_keypoint"
    assert matches[0].keypoint == "left_big_toe"
    assert matches[0].distance_box_h == pytest.approx(8.0 / HEIGHT, abs=1e-6)

    far_feet = _pose(tracks, [(30, 1)], feet_shift=(0.0, -60.0))
    tables = MeasureTables(
        tracks=tracks, shots=_shots(), ball_events=pd.DataFrame([event]), pose=far_feet
    )
    assert match_contacts(tables, 10, 79)[0] == []  # a pose that shows the feet away wins

    weak = _pose(tracks, [(30, 1)], score=0.2)
    tables = MeasureTables(
        tracks=tracks, shots=_shots(), ball_events=pd.DataFrame([event]), pose=weak
    )
    fallback = match_contacts(tables, 10, 79)[0]
    assert [m.method for m in fallback] == ["foot_point_fallback"]
    assert fallback[0].keypoint == ""

    away = _event(30, 100.0, 100.0)
    tables = MeasureTables(tracks=tracks, shots=_shots(), ball_events=pd.DataFrame([away]))
    assert match_contacts(tables, 10, 79)[0] == []
    outside = MeasureTables(tracks=tracks, shots=_shots(), ball_events=pd.DataFrame([event]))
    assert match_contacts(outside, 31, 79) == ([], 0)


def test_contact_records_and_ball_image_speed() -> None:
    tracks = _tracks()
    striker = tracks[(tracks["frame"] == 30) & (tracks["track_id"] == 1)].iloc[0].to_dict()
    events = pd.DataFrame(
        [
            _event(30, striker["foot_u"] + 5.0, striker["foot_v"] - 5.0),
            _event(
                40,
                striker["foot_u"] + 3.0,
                striker["foot_v"],
                kind="slow_down",
                profile="contact",
                speed_after=0.5,
            ),
            _event(45, 50.0, 50.0, profile="contact"),
        ]
    )
    rows = [_lift_row(f, 1, facing_deg=30.0) for f in range(26, 35)]
    tables = MeasureTables(
        tracks=tracks,
        shots=_shots(),
        ball_events=events,
        pose=_pose(tracks, [(f, 1) for f in range(26, 45)]),
        lift=pd.DataFrame(rows),
    )
    records = _records(tables)
    contacts = [r for r in records if r["metric_id"] == "ball-contact-frame"]
    assert [r["value"] for r in contacts] == [30, 40]
    assert all(r["status"] == "ok" and r["track_id"] == 1 for r in contacts)
    assert contacts[0]["quality"]["event_profile"] == "strike"
    assert contacts[1]["quality"]["event_kind"] == "slow_down"
    speeds = [r for r in records if r["metric_id"] == "ball-image-speed"]
    assert [(r["frame"], r["value"]) for r in speeds] == [(30, 10.0)]  # no speed after a trap
    orientation = _find(records, "body-orientation", 1)
    assert orientation["quality"]["reference"] == "contact_estimate"
    assert orientation["frame"] == 30
    nothing = MeasureTables(tracks=tracks, shots=_shots(), ball_events=events.iloc[2:3])
    missing = _find(_records(nothing), "ball-contact-frame")
    assert missing["status"] == "unavailable"
    assert missing["reasonCode"] == "no_contact"
    assert missing["quality"]["ball_events_examined"] == 1


def test_attackers_in_box_counts_the_attacking_team() -> None:
    tracks = _tracks()
    more = pd.DataFrame(
        [
            _track_row(30, 4, "left", pitch=(38.0, -10.0)),
            _track_row(30, 5, "left", pitch=(30.0, 0.0)),  # outside the box
            _track_row(30, 6, "left", pitch=(44.0, 25.0)),  # wide of the box
        ]
    )
    tracks = pd.concat([tracks, more], ignore_index=True)
    tracks["team"] = tracks["team"].astype(object)
    records = _records(MeasureTables(tracks=tracks, shots=_shots()))
    record = _find(records, "attackers-in-box")
    assert record["value"] == 2  # tracks 1 and 4
    assert record["track_ids"] == [1, 4]
    assert record["quality"]["team"] == "left"
    assert record["quality"]["defenders_in_box"] == 2  # tracks 2 and 3, not the goalkeeper
    assert record["tier"] == "B"
    assert record["display"] == "2 人"
    assert record["frame"] == 30
    assert record["quality"]["reference"] == "highlight_key_frame"
    away = {**HIGHLIGHT, "cues": {}, "track_ids": [2]}
    record = _find(_records(MeasureTables(tracks=tracks, shots=_shots()), away), "attackers-in-box")
    assert record["quality"]["team"] == "right"
    assert record["value"] == 0
    # The frame the summary names for the box-pressure cue (track 4 is in the box at frame
    # 30 only): the card and the summary count the same players.
    cued = {**HIGHLIGHT, "cues": {"box_pressure": {"team": "left", "frame": 31}}}
    record = _find(_records(MeasureTables(tracks=tracks, shots=_shots()), cued), "attackers-in-box")
    assert (record["frame"], record["value"], record["track_ids"]) == (31, 1, [1])
    assert record["quality"]["reference"] == "box_pressure_cue"
    outside = {**HIGHLIGHT, "cues": {"box_pressure": {"team": "left", "frame": 5}}}
    record = _find(
        _records(MeasureTables(tracks=tracks, shots=_shots()), outside), "attackers-in-box"
    )
    assert record["frame"] == 30, "a cue frame outside the window is not used"


def test_pose_requests_cover_involved_players_and_contact_candidates() -> None:
    tracks = _tracks()
    events = pd.DataFrame([_event(30, 1725.0, 600.0)])  # next to track 2 at x = 43 m
    highlight = {**HIGHLIGHT, "start_frame": 28, "end_frame": 52, "track_ids": [1]}
    requests = pose_requests([highlight], tracks, events)
    keys = set(zip(requests["frame"], requests["track_id"], strict=True))
    assert {(f, 1) for f in range(28, 50)} <= keys
    assert {(f, 7) for f in range(50, 53)} <= keys  # the joined shot's players
    assert (30, 2) in keys  # nearest the ball at the event
    assert (30, 3) in keys  # second nearest
    assert (30, 9) not in keys  # more than two box heights away
    assert not any(track == 20 for _, track in keys)  # referees never
    assert all(28 <= frame <= 52 for frame, _ in keys)
    assert requests.columns.tolist() == ["frame", "track_id", "x1", "y1", "x2", "y2"]
    assert len(requests) == len(keys)


def test_card_records_follow_the_highlight_type() -> None:
    tracks = _tracks()
    striker = tracks[(tracks["frame"] == 30) & (tracks["track_id"] == 1)].iloc[0].to_dict()
    events = pd.DataFrame([_event(30, striker["foot_u"], striker["foot_v"])])
    rows = [(f, 1, 0, 5.0, True, False) for f in range(0, 50)]
    tables = MeasureTables(
        tracks=tracks, shots=_shots(), ball_events=events, kinematics=_kinematics(rows)
    )
    records = _records(tables)
    card = card_records(records, HIGHLIGHT)
    metrics = [record["metric_id"] for record in card]
    assert metrics[:2] == ["ball-contact-frame", "ball-image-speed"]
    assert all(record["status"] == "ok" for record in card)
    assert len(card) <= 5
    assert len(set(metrics)) == len(metrics)
    assert card_records(records, HIGHLIGHT, limit=1) == card[:1]
    sprint = {**HIGHLIGHT, "type": "sprint"}
    assert card_records(measure_highlight(sprint, tables, _context()), sprint)[0]["metric_id"] == (
        "peak-speed"
    )
    # Records list available values first, in the type's metric order.
    statuses = [record["status"] for record in records]
    assert statuses == sorted(statuses, key=lambda s: s != "ok")
    # The overlay's focus player wins a metric when that player has a value.
    first = next(r for r in card if r["metric_id"] == "nearest-opponent-distance")
    assert first["track_id"] == 1
    preferred = card_records(records, HIGHLIGHT, prefer_track=2)
    chosen = next(r for r in preferred if r["metric_id"] == "nearest-opponent-distance")
    assert chosen["track_id"] == 2
    assert card_records(records, HIGHLIGHT, prefer_track=999) == card


def test_contact_radius_is_a_parameter_with_a_tight_default() -> None:
    tracks = _tracks()
    striker = tracks[(tracks["frame"] == 30) & (tracks["track_id"] == 1)].iloc[0].to_dict()
    toe_u = striker["x1"] + 0.66 * (striker["x2"] - striker["x1"])
    # 0.2 box heights below the left big toe: near the foot, not touching the ball.
    event = _event(30, toe_u, striker["y2"] + 0.2 * HEIGHT)
    tables = MeasureTables(
        tracks=tracks,
        shots=_shots(),
        ball_events=pd.DataFrame([event]),
        pose=_pose(tracks, [(30, 1), (30, 2), (30, 3)]),
    )
    assert measure.DEFAULT_PARAMS.contact_radius_bh == 0.15
    assert match_contacts(tables, 10, 79)[0] == []
    loose = match_contacts(tables, 10, 79, MeasureParams(contact_radius_bh=0.3))[0]
    assert [(m.track_id, m.method) for m in loose] == [(1, "foot_keypoint")]


def test_measure_highlights_is_deterministic_and_traceable() -> None:
    tables = MeasureTables(tracks=_tracks(), shots=_shots())
    document = {"highlights": [HIGHLIGHT, {**HIGHLIGHT, "id": "h00", "peak_frame": 20}]}
    first = measure_highlights(document, tables, _context())
    second = measure_highlights(document, tables, _context())
    assert first == second
    assert [r["highlight_id"] for r in first][0] == "h00"  # time order
    other_input = measure_highlights(
        document, tables, _context(inputs=(("tracks.parquet", "d" * 64),))
    )
    assert {r["evidenceId"] for r in first}.isdisjoint({r["evidenceId"] for r in other_input})
    for record in first:
        assert record["metric_id"] in METRICS
        assert record_problems(record) == []
        if record["status"] == "unavailable":
            assert record["value"] is None
            assert record["reasonCode"] in measure.REASONS_ZH
        assert record["source"]["inputs"][0]["name"] == "pose.parquet"


def test_bad_inputs_are_reported_together() -> None:
    with pytest.raises(MeasureError) as excinfo:
        measure_highlight(
            HIGHLIGHT,
            MeasureTables(
                tracks=pd.DataFrame({"frame": [0]}),
                shots=pd.DataFrame({"shot_id": [0]}),
                pose=pd.DataFrame({"frame": [0]}),
            ),
            _context(),
        )
    assert len(excinfo.value.problems) == 3
    tables = MeasureTables(tracks=_tracks(), shots=_shots())
    with pytest.raises(MeasureError, match="start <= peak"):
        measure_highlight({**HIGHLIGHT, "peak_frame": 5}, tables, _context())
    gap = _shots()
    gap.loc[1, "start_frame"] = 52
    with pytest.raises(MeasureError, match="tile"):
        measure_highlight(HIGHLIGHT, MeasureTables(tracks=_tracks(), shots=gap), _context())


def test_registry_problems_are_reported(unicode_tmp_path: Path) -> None:
    broken = unicode_tmp_path / "metrics.toml"
    broken.write_text('schema_version = 1\n[[metrics]]\nid = "x"\n', encoding="utf-8")
    with pytest.raises(MeasureError) as excinfo:
        load_metric_registry(broken)
    assert len(excinfo.value.problems) >= 1
    with pytest.raises(MeasureError, match="cannot read"):
        load_metric_registry(unicode_tmp_path / "missing.toml")


@pytest.mark.parametrize("unsafe", ["all", "isolated", "missing"])
def test_peak_speed_rejects_unsafe_observed_subject_without_breaking_bridges(unsafe: str) -> None:
    tracks = _tracks()
    if unsafe == "all":
        tracks.loc[tracks.track_id == 1, "y2"] = 1079.5
    elif unsafe == "isolated":
        tracks.loc[(tracks.track_id == 1) & (tracks.frame == 20), "y2"] = 1079.5
    else:
        tracks = tracks[~((tracks.track_id == 1) & (tracks.frame == 20))]
    rows = _chord_rows(1, 0, range(50), valid=set(range(10, 40)))
    tables = MeasureTables(
        tracks=tracks,
        shots=_shots(),
        kinematics=rows,
        team_verdicts={(1, "left"): "unchecked"},
        frame_size=(1920, 1080),
    )
    record = _find(_records(tables), "peak-speed", 1)
    if unsafe == "all":
        assert record["status"] == "unavailable"
        assert record["reasonCode"] == "no_valid_speed"
    else:
        assert record["status"] == "ok"
        assert record["value"] == pytest.approx(6.25)
        assert (20 in record["frames"]) == (unsafe == "missing")
        if unsafe == "isolated":
            assert not {15, 20, 25} & set(record["frames"]), "unsafe chord endpoints too"


def test_peak_speed_withheld_for_short_segments_says_how_many_frames_are_trusted() -> None:
    # Track 1 is tracked on frames 10-39 of the window (40-49 lie within the chord margin of
    # its end): frames 10-21 in a 22-frame segment, too short for a valid speed, frames
    # 22-31 at the start of the next segment, valid speeds (6.25 m/s) on frames 32-39 only.
    # The reason said 测不出可信速度 although it had trusted speeds on 8 frames.
    rows = pd.concat(
        [
            _chord_rows(1, 0, range(0, 22), valid=set()),
            _chord_rows(1, 1, range(22, 50), valid=set(range(32, 40))),
        ],
        ignore_index=True,
    )
    tables = MeasureTables(tracks=_tracks(), shots=_shots(), kinematics=rows)
    record = _find(_records(tables), "peak-speed", 1)
    assert (record["status"], record["reasonCode"]) == ("unavailable", "speed_coverage_short")
    quality = record["quality"]
    considered = quality["tracked_frames"] - quality["margin_frames"]
    assert (quality["speed_frames"], considered) == (8, 30)
    short, split = quality["uncovered_short_segment_frames"], quality["uncovered_split_frames"]
    assert (short, split) == (12, 10)
    assert record["reason_zh"] == (
        "可信速度只覆盖 8/30 帧，其余帧多在太短的连续轨迹段上：测得的最高 22.5 km/h（第 32 帧）"
        "只是下限，不作为峰值"
    )
    assert record["reason_zh"] == measure.withheld_speed_reason_zh(
        "speed_coverage_short", quality, Fraction(25)
    )
    validate_record(record)
    # No coverage reason claims that nothing could be measured: each has a speed to show.
    for code in ("split", "rejected", "short", "implausible"):
        text = measure.REASONS_ZH[f"speed_coverage_{code}"]
        assert "测不出可信速度" not in text
        assert "下限" in text
    assert "下限" in measure.REASONS_ZH["speed_peak_hidden"]


def test_attackers_in_box_record_where_their_team_comes_from() -> None:
    tracks = _tracks()
    tables = MeasureTables(tracks=tracks, shots=_shots())
    sources = {}
    for name, highlight in (
        ("cue", HIGHLIGHT),  # the box-pressure cue names the team
        ("player", {**HIGHLIGHT, "cues": {}, "track_ids": [2]}),  # track 2's label (right)
        ("count", {**HIGHLIGHT, "cues": {}, "track_ids": []}),  # the most players in the box
    ):
        record = _find(_records(tables, highlight), "attackers-in-box")
        sources[name] = (record["quality"]["team"], record["quality"]["team_source"])
    assert sources == {
        "cue": ("left", "cue"),
        "player": ("right", "first_player"),
        "count": ("left", "largest_count"),
    }


def test_attackers_in_box_is_withheld_when_no_player_on_its_frame_has_a_team() -> None:
    # gbf.role_check clears every team label of a run whose role labels are implausible: a
    # count of one team's players would then be 0 whatever the picture shows.
    tracks = _tracks()
    tracks["team"] = None
    tables = MeasureTables(tracks=tracks, shots=_shots())
    for name, highlight in (
        ("cue", HIGHLIGHT),  # the cue still names a side
        ("count", {**HIGHLIGHT, "cues": {}, "track_ids": []}),
    ):
        record = _find(_records(tables, highlight), "attackers-in-box")
        assert (record["status"], record.get("reasonCode"), record["value"]) == (
            "unavailable",
            "team_unknown",
            None,
        ), name
        assert record["frame"] == 30
        assert record["quality"]["outfield_players_without_team"] == 3  # tracks 1, 2 and 3


def test_attack_measures_are_withheld_when_the_attack_direction_is_unknown() -> None:
    # gbf.side_check could not confirm the plugin's left/right names: the goal a team attacks
    # is unknown, so the attackers in the box and the orientation to the attacked goal give no
    # value. A record withheld before the direction matters (no pose, no team) keeps its
    # reason; every other record stays as it is.
    tracks = _tracks({(f, 1): {"pitch_y": 10.0} for f in range(0, 50)})
    rows = [_lift_row(f, 1, facing_deg=0.0) for f in range(28, 33)]
    tables = _lift_tables(rows, tracks=tracks)
    known = measure_highlight(HIGHLIGHT, tables, _context())
    unknown = measure_highlight(HIGHLIGHT, tables, _context(attack_direction_known=False))
    withheld = {"attackers-in-box", "body-orientation"}
    measured = {
        (r["metric_id"], r["track_id"])
        for r in known
        if r["metric_id"] in withheld and r["status"] == "ok"
    }
    assert measured == {("attackers-in-box", None), ("body-orientation", 1)}
    by_key = {(r["metric_id"], r["track_id"]): r for r in unknown}
    assert len(by_key) == len(unknown) == len(known)
    for before in known:
        key = (before["metric_id"], before["track_id"])
        after = by_key[key]
        if key in measured:
            assert (after["status"], after.get("reasonCode"), after["value"]) == (
                "unavailable",
                "attack_direction_unknown",
                None,
            ), key
            assert after["reason_zh"] == measure.REASONS_ZH["attack_direction_unknown"]
        elif before["metric_id"] in withheld:
            assert before.get("reasonCode") is not None, key
            assert after.get("reasonCode") == before.get("reasonCode"), key
        else:
            assert after == before, key
    tracks["team"] = None
    tables = _lift_tables(rows, tracks=tracks)
    for record in measure_highlight(HIGHLIGHT, tables, _context(attack_direction_known=False)):
        if (record["metric_id"], record["track_id"]) in measured:
            assert record.get("reasonCode") == "team_unknown", record["metric_id"]
    with pytest.raises(MeasureError) as caught:
        _context(attack_direction_known="no")
    assert "attack_direction_known" in str(caught.value)


def test_attackers_in_box_is_withheld_when_a_player_in_that_box_has_no_team() -> None:
    # A player without a team label in the box the count is for may be an attacker, so the
    # count would only be a lower bound; the nearest opponent is withheld in the same way when
    # a player without a team is nearer (unknown_team_nearer). One elsewhere changes nothing.
    tracks = _tracks()
    unlabelled_outside = pd.DataFrame([_track_row(30, 4, None, pitch=(30.0, 0.0))])
    tables = MeasureTables(
        tracks=pd.concat([tracks, unlabelled_outside], ignore_index=True), shots=_shots()
    )
    record = _find(_records(tables), "attackers-in-box")
    assert (record["status"], record["value"], record["track_ids"]) == ("ok", 1, [1])
    unlabelled_inside = pd.DataFrame([_track_row(30, 4, None, pitch=(38.0, -10.0))])
    tables = MeasureTables(
        tracks=pd.concat([tracks, unlabelled_inside], ignore_index=True), shots=_shots()
    )
    record = _find(_records(tables), "attackers-in-box")
    assert (record["status"], record.get("reasonCode"), record["value"]) == (
        "unavailable",
        "unknown_team_in_box",
        None,
    )
    assert record["reason_zh"] == measure.REASONS_ZH["unknown_team_in_box"]
    assert record["frame"] == 30
    assert record["quality"]["team"] == "left"
    assert record["quality"]["attackers_with_team"] == 1  # track 1: a lower bound
    assert record["quality"]["players_without_team_in_box"] == 1  # track 4
