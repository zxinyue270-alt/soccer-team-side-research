"""Conservative image-plane context for replay; no semantic action or identity claims."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import pandas as pd

from gbf.ball import GATE_REASONS

POLICY = "ball-and-people-context/2"


def associated_ids(highlight: dict[str, Any]) -> list[int]:
    values = list(highlight.get("track_ids") or []) + list(highlight.get("key_pair") or [])
    if highlight.get("key_focus_track") is not None:
        values.append(highlight["key_focus_track"])
    for moment in highlight.get("key_moments") or []:
        values.extend(moment.get("track_ids") or [])
    return sorted({int(value) for value in values})


def context_window(highlight, shot, frame_count: int, fps: Fraction, cuts=()) -> dict:
    """Fixed context budget bounded by the peak's shot, source and explicit cuts."""
    key = highlight["peak_frame"]
    lo = max(0, int(shot["start_frame"]), max((c for c in cuts if c <= key), default=0))
    hi = min(
        frame_count - 1,
        int(shot["end_frame"]),
        min((c - 1 for c in cuts if c > key), default=frame_count - 1),
    )
    pad = math.ceil(2 * fps)
    requested = (highlight["start_frame"] - pad, highlight["end_frame"] + pad)
    first, last = max(lo, requested[0]), min(hi, requested[1])
    return dict(
        start_frame=first,
        end_frame=last,
        candidate_start_frame=highlight["start_frame"],
        candidate_end_frame=highlight["end_frame"],
        context_budget_frames=pad,
        shot_start_frame=lo,
        shot_end_frame=hi,
        start_limit="shot_or_source" if first == lo else "context_budget",
        end_limit="shot_or_source" if last == hi else "context_budget",
        action_boundaries_verified=False,
    )


def _box(row, size):
    box = [float(row[c]) for c in ("x1", "y1", "x2", "y2")]
    w, h = size
    if (
        all(math.isfinite(v) for v in box)
        and 0 <= box[0] < box[2] <= w
        and 0 <= box[1] < box[3] <= h
    ):
        return box
    return None


def ball_selection_evidence(row, group, *, require_unique=True):
    """Raw multiplicity is not the number of surviving gated alternatives."""
    raw = row.get("candidate_count")
    if pd.isna(raw) or isinstance(raw, bool):
        return None, "ball_candidate_evidence_invalid"
    raw = float(raw)
    if not math.isfinite(raw) or raw < 1 or raw != int(raw):
        return None, "ball_candidate_evidence_invalid"
    count = int(raw)
    if group is None:
        if count == 1:
            return "single_raw_candidate", None
        return None, "ball_candidates_ambiguous"
    required = {"cand_index", "u", "v", "diameter_px", "reason"}
    if not required <= set(group) or len(group) != count:
        return None, "ball_candidate_evidence_invalid"
    indexes = group.cand_index.tolist()
    if (
        any(pd.isna(v) or isinstance(v, bool) or not isinstance(v, (int, float)) for v in indexes)
        or sorted(indexes) != list(range(count))
        or not group.reason.isin(GATE_REASONS).all()
    ):
        return None, "ball_candidate_evidence_invalid"
    passed = group[group.reason == "ok"]
    if len(passed) == 0 or (require_unique and len(passed) != 1):
        return None, "ball_gated_candidates_ambiguous"
    matching = passed[
        (passed.u == row.u) & (passed.v == row.v) & (passed.diameter_px / 2 == row.radius_px)
    ]
    if len(matching) != 1:
        return None, "ball_selected_candidate_mismatch"
    return (
        "unique_matching_gated_candidate"
        if len(passed) == 1
        else "matching_gated_candidate_among_alternatives"
    ), None


def plan_focus(
    tracks: pd.DataFrame,
    ball: pd.DataFrame,
    ids: list[int],
    first: int,
    last: int,
    size: tuple[int, int],
    *,
    candidates: pd.DataFrame | None = None,
) -> dict[int, dict]:
    """Frame-local geometry only: retain full view whenever evidence is insufficient.

    Associated IDs are proposals. A nearby box need not be the same physical person
    in consecutive frames. No interpolation, ID repair, possession or contact inference.
    """
    people = {int(f): g for f, g in tracks[tracks.frame.between(first, last)].groupby("frame")}
    balls = (
        {int(f): g for f, g in ball[ball.frame.between(first, last)].groupby("frame")}
        if "frame" in ball
        else {}
    )
    options = (
        {int(f): g for f, g in candidates.groupby("frame")}
        if candidates is not None and "frame" in candidates
        else {}
    )
    w, h = size
    result = {}
    for frame in range(first, last + 1):
        entry = dict(
            crop=None,
            boxes=[],
            track_ids=[],
            associated_missing_ids=[],
            ball=None,
            reason="ball_missing",
            policy=POLICY,
        )
        valid, ambiguous = {}, set()
        if frame in people:
            for tid, rows in people[frame].groupby("track_id"):
                tid = int(tid)
                box = _box(rows.iloc[0], size) if len(rows) == 1 else None
                if box is None:
                    ambiguous.add(tid)
                else:
                    valid[tid] = box
        entry["scene_people"] = [
            {"track_id": tid, "bbox_xyxy": valid[tid]} for tid in sorted(valid)
        ]
        entry["invalid_or_duplicate_scene_ids"] = sorted(ambiguous)
        entry["associated_missing_ids"] = sorted(set(ids) - valid.keys())
        result[frame] = entry
        group = balls.get(frame)
        if group is None:
            continue
        if len(group) != 1:
            entry["reason"] = "ball_ambiguous"
            continue
        row = group.iloc[0]
        if row.get("status") != "detected":
            entry["reason"] = "ball_not_direct"
            continue
        u, v, radius = (float(row.get(c, float("nan"))) for c in ("u", "v", "radius_px"))
        if not (
            all(math.isfinite(x) for x in (u, v, radius))
            and 0 <= u < w
            and 0 <= v < h
            and radius >= 0
        ):
            entry["reason"] = "ball_invalid_geometry"
            continue
        selection, failure = ball_selection_evidence(
            row, options.get(frame, pd.DataFrame()) if candidates is not None else None
        )
        if failure:
            entry["reason"] = failure
            continue
        entry["ball"] = dict(
            u=u,
            v=v,
            radius_px=radius,
            status="detected",
            candidate_count=int(row.candidate_count),
            selection_evidence=selection,
            identity_verified=False,
        )
        if set(ids) & ambiguous:
            entry["reason"] = "associated_box_ambiguous_or_invalid"
            continue
        near = {
            tid
            for tid, b in valid.items()
            if math.hypot(max(b[0] - u, 0, u - b[2]), max(b[1] - v, 0, v - b[3]))
            <= 2 * (b[3] - b[1])
        }
        if not near:
            entry["reason"] = "no_nearby_person"
            continue
        selected = sorted(near | (set(ids) & valid.keys()))
        boxes = [valid[tid] for tid in selected]
        pad = max(b[3] - b[1] for b in boxes) * 0.5
        left = max(0, math.floor(min([b[0] for b in boxes] + [u - radius]) - pad))
        top = max(0, math.floor(min([b[1] for b in boxes] + [v - radius]) - pad))
        right = min(w, math.ceil(max([b[2] for b in boxes] + [u + radius]) + pad))
        bottom = min(h, math.ceil(max([b[3] for b in boxes] + [v + radius]) + pad))
        rw, rh = max(right - left, w / 4), max(bottom - top, h / 4)
        left = max(0, min(w - rw, (left + right - rw) / 2))
        top = max(0, min(h - rh, (top + bottom - rh) / 2))
        crop = [
            math.floor(left),
            math.floor(top),
            min(w, math.ceil(left + rw)),
            min(h, math.ceil(top + rh)),
        ]
        if (crop[2] - crop[0]) * (crop[3] - crop[1]) >= 0.8 * w * h:
            entry["reason"] = "region_too_wide"
            continue
        entry.update(crop=crop, boxes=boxes, track_ids=selected, reason="ball_and_people")
    return result


def stabilize_replay(focus: dict[int, dict], frames, fps: Fraction, size) -> None:
    """Fix the crop within each eligible replay run; never bridge missing evidence.

    Half a second is an editorial minimum, not an action or identity accuracy threshold.
    Counts refer to distinct source frames, not repeated slow-motion output frames.
    """
    for entry in focus.values():
        entry["proposed_crop"] = entry["crop"]
        entry["replay_component"] = None
    runs = []
    for frame in sorted(set(frames)):
        if focus[frame]["crop"] is None:
            continue
        if not runs or runs[-1][-1] != frame - 1:
            runs.append([])
        runs[-1].append(frame)
    minimum = math.ceil(Fraction(1, 2) * fps)
    w, h = size
    for component, run in enumerate(runs):
        crops = [focus[f]["crop"] for f in run]
        region = [
            min(c[0] for c in crops),
            min(c[1] for c in crops),
            max(c[2] for c in crops),
            max(c[3] for c in crops),
        ]
        reason = (
            "replay_component_too_short"
            if len(run) < minimum
            else "replay_region_too_wide"
            if (region[2] - region[0]) * (region[3] - region[1]) >= 0.8 * w * h
            else None
        )
        for frame in run:
            entry = focus[frame]
            entry["replay_component"] = component
            entry["crop"] = None if reason else region
            if reason:
                entry["reason"] = reason
