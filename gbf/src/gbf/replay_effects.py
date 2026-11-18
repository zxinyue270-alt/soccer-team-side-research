"""Editorial emphasis on existing events and observed image-plane ball estimates."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from gbf.action_context import ball_selection_evidence
from gbf.render import plan_highlight_edit

POLICY = "event-emphasis-and-observed-trail/2"


def plan_emphasis(highlight, first: int, last: int, fps: Fraction, *, cuts=()):
    """Prefer a release event to a running peak, without asserting it is a shot."""
    peak = highlight["peak_frame"]
    events = [
        m
        for m in highlight.get("key_moments", [])
        if m.get("kind") == "ball_away"
        and type(m.get("frame")) is int
        and max(first, highlight["start_frame"]) <= m["frame"] <= min(last, highlight["end_frame"])
        and not any(min(peak, m["frame"]) < c <= max(peak, m["frame"]) for c in cuts)
    ]
    event = min(events, key=lambda m: (abs(m["frame"] - peak), m["frame"])) if events else None
    key = event["frame"] if event else peak
    emphasis = {
        "policy": POLICY,
        "candidate_peak_frame": peak,
        "key_frame": key,
        "source": "key_moments.ball_away" if event else "candidate_peak",
        "label": "关键出球" if event else "关键动作",
        "semantic_action_verified": False,
        "event": event,
    }
    edit = plan_highlight_edit(first, last, key, fps, cuts=cuts, slow_after_s=2.0 if event else 0.4)
    return emphasis, edit


def plan_ball_trails(ball, first, last, fps, size, *, cuts=(), candidates=None, camera_motion=None):
    """Show the selected tracking estimate; never fill a missing observation.

    Multiple gated alternatives are retained as uncertainty, rather than claimed
    to be resolved. This display does not relax the stricter crop eligibility.
    """
    rate = float(fps)
    if rate <= 0 or min(size) <= 0:
        raise ValueError("Invalid trail frame grid")
    history = max(1, math.floor(0.28 * rate))
    max_step = 1.5 * math.hypot(*size) / rate
    groups = {int(f): g for f, g in ball.groupby("frame")} if "frame" in ball else {}
    gates = (
        {int(f): g for f, g in candidates.groupby("frame")}
        if candidates is not None and "frame" in candidates
        else {}
    )
    result, points = {}, []
    for frame in range(first, last + 1):
        entry = {"points": [], "reason": "ball_missing", "identity_verified": False}
        result[frame] = entry
        group = groups.get(frame)
        if frame in cuts:
            points = []
        motion = camera_motion.get(frame, {}) if camera_motion is not None else None
        if motion is not None:
            entry["camera_motion"] = motion
            if motion.get("valid"):
                matrix = np.asarray(motion["matrix"], dtype=float)
                for p in points:
                    moved = matrix @ [p["display_u"], p["display_v"], 1]
                    p["display_u"], p["display_v"] = (
                        float(moved[0] / moved[2]),
                        float(moved[1] / moved[2]),
                    )
            else:
                points = []
        if group is None or len(group) != 1:
            points = []
            continue
        row = group.iloc[0]
        u, v, radius = [float(row.get(c, float("nan"))) for c in ("u", "v", "radius_px")]
        if (
            row.get("status") != "detected"
            or not all(math.isfinite(x) for x in (u, v, radius))
            or not (radius > 0 and 0 <= u < size[0] and 0 <= v < size[1])
        ):
            points = []
            entry["reason"] = "ball_not_direct_or_invalid"
            continue
        options = gates.get(frame, pd.DataFrame()) if candidates is not None else None
        selection, failure = ball_selection_evidence(row, options, require_unique=False)
        if failure:
            points = []
            entry["reason"] = failure
            continue
        jump = bool(
            points
            and math.hypot(u - points[-1]["display_u"], v - points[-1]["display_v"]) > max_step
        )
        if jump:
            points = []
        points = [p for p in points if frame - p["frame"] <= history]
        points.append(
            {
                "frame": frame,
                "u": u,
                "v": v,
                "radius_px": radius,
                "display_u": u,
                "display_v": v,
                "conf": float(row.conf)
                if math.isfinite(float(row.get("conf", float("nan"))))
                else None,
                "selection_evidence": selection,
                "alternative_count": int((options.reason == "ok").sum())
                if options is not None
                else int(row.candidate_count),
            }
        )
        if len(points) >= 2:
            delta = np.array([u - points[-2]["display_u"], v - points[-2]["display_v"]])
            entry["direction"] = delta.tolist()
            if np.linalg.norm(delta) < 0.15 * radius:
                points = points[-1:]
            else:
                ahead = [
                    i
                    for i, p in enumerate(points[:-1])
                    if np.dot([p["display_u"] - u, p["display_v"] - v], delta) > 0
                ]
                if ahead:
                    points = points[max(ahead) + 1 :]
        # Copies matter: subsequent camera transforms must not mutate earlier plans.
        entry.update(
            points=[dict(p) for p in points],
            reason="jump_restart" if jump else "observed_tracking_estimate",
        )
    return result


def draw_ball_trail(source: Image.Image, trail: dict) -> Image.Image:
    """Composite a fading trail on a copy; raw model pixels remain untouched."""
    points = trail.get("points", [])
    if not points:
        return source.copy()
    scale = max(0.5, source.width / 1920)
    glow = Image.new("RGBA", source.size)
    core = Image.new("RGBA", source.size)
    gd, cd = ImageDraw.Draw(glow), ImageDraw.Draw(core)
    for i, (a, b) in enumerate(zip(points, points[1:], strict=False), 1):
        strength = i / max(1, len(points) - 1)
        coords = [(a["display_u"], a["display_v"]), (b["display_u"], b["display_v"])]
        alpha = round(45 + 190 * strength)
        gd.line(
            coords, fill=(255, 164, 38, alpha), width=max(2, round((4 + 13 * strength) * scale))
        )
        cd.line(
            coords, fill=(255, 222, 118, alpha), width=max(1, round((1 + 4 * strength) * scale))
        )
    head = points[-1]
    x, y = head["u"], head["v"]
    radius = max(head["radius_px"] + 4 * scale, 7 * scale)
    cd.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(255, 238, 173, 235),
        width=max(1, round(3 * scale)),
    )
    rendered = Image.alpha_composite(
        source.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(4 * scale))
    )
    return Image.alpha_composite(rendered, core).convert("RGB")
