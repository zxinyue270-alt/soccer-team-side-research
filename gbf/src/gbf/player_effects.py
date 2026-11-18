"""Highlight proposed action participants with observed, associated 2D body poses."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from gbf.pose import HALPE26_EDGES

POLICY = "observed-actor-glow-and-halpe26/1"
COLORS = [(255, 202, 93), (92, 235, 224), (190, 155, 255), (160, 226, 117)]
BOX = ["x1", "y1", "x2", "y2"]


def plan_player_effects(highlight, tracks, poses, first, last, key, size, fps):
    """Select stable proposed subjects at the anchor; never invent a missing pose."""
    at = {}
    for tid, group in tracks[tracks.frame == key].groupby("track_id"):
        if len(group) == 1:
            box = group.iloc[0][BOX].to_numpy(dtype=float)
            if (
                np.isfinite(box).all()
                and 0 <= box[0] < box[2] <= size[0]
                and 0 <= box[1] < box[3] <= size[1]
            ):
                at[int(tid)] = box
    primary = [
        int(tid)
        for m in highlight.get("key_moments", [])
        if m.get("kind") == "ball_away" and m.get("frame") == key
        for tid in m.get("track_ids", [])
    ]
    pair = (
        highlight.get("key_pair")
        or highlight.get("cues", {}).get("duel", {}).get("track_ids")
        or []
    )
    focus = [highlight["key_focus_track"]] if highlight.get("key_focus_track") is not None else []
    seeds = list(dict.fromkeys(int(t) for t in [*primary, *pair, *focus] if int(t) in at))
    related = [int(t) for t in highlight.get("track_ids", []) if int(t) in at]
    if not seeds:
        seeds = related[:1]
    selected = seeds.copy()
    for tid in related:
        a = at[tid]
        if any(
            math.hypot(
                max(a[0] - at[t][2], at[t][0] - a[2], 0), max(a[1] - at[t][3], at[t][1] - a[3], 0)
            )
            <= 3 * max(a[3] - a[1], at[t][3] - at[t][1])
            for t in seeds
        ):
            if tid not in selected:
                selected.append(tid)
    selected = selected[:4]
    by_frame = {
        int(f): g
        for f, g in tracks[
            tracks.frame.between(first, last) & tracks.track_id.isin(selected)
        ].groupby("frame")
    }
    by_pose = (
        {tuple(map(int, key)): group for key, group in poses.groupby(["frame", "track_id"])}
        if {"frame", "track_id"} <= set(poses)
        else {}
    )
    result = {}
    for frame in range(first, last + 1):
        entry = {
            "actors": [],
            "selected_track_ids": selected,
            "near_key": abs(frame - key) <= float(fps),
            "policy": POLICY,
        }
        result[frame] = entry
        for tid in selected:
            group = by_frame.get(frame)
            rows = group[group.track_id == tid] if group is not None else None
            if rows is None or len(rows) != 1:
                continue
            box = rows.iloc[0][BOX].to_numpy(dtype=float)
            if not (
                np.isfinite(box).all()
                and 0 <= box[0] < box[2] <= size[0]
                and 0 <= box[1] < box[3] <= size[1]
            ):
                continue
            actor = {
                "track_id": tid,
                "bbox_xyxy": box.tolist(),
                "label": chr(65 + selected.index(tid)),
                "color": list(COLORS[selected.index(tid)]),
                "keypoints": [None] * 26,
                "edges": [],
                "pose_reason": "missing_pose",
                "identity_verified": False,
            }
            entry["actors"].append(actor)
            ps = by_pose.get((frame, tid))
            if ps is None:
                continue
            if len(ps) != 1:
                actor["pose_reason"] = "duplicate_pose"
                continue
            pose = ps.iloc[0]
            if not set(BOX + ["keypoints"]).issubset(ps):
                actor["pose_reason"] = "invalid_pose"
                continue
            if not np.allclose(pose[BOX].to_numpy(dtype=float), box, rtol=0, atol=1e-6):
                actor["pose_reason"] = "pose_box_mismatch"
                continue
            joints = np.asarray(pose.keypoints, dtype=float)
            if joints.size != 78:
                actor["pose_reason"] = "invalid_pose"
                continue
            joints = joints.reshape(26, 3)
            pad = 0.2 * (box[3] - box[1])
            valid = np.isfinite(joints).all(axis=1) & (joints[:, 2] >= 0.4)
            valid &= (joints[:, 0] >= max(0, box[0] - pad)) & (
                joints[:, 0] < min(size[0], box[2] + pad)
            )
            valid &= (joints[:, 1] >= max(0, box[1] - pad)) & (
                joints[:, 1] < min(size[1], box[3] + pad)
            )
            actor.update(
                keypoints=[j.tolist() if valid[i] else None for i, j in enumerate(joints)],
                model_id=str(pose.get("model_id", "unknown")),
                pose_reason="associated_pose",
            )
            actor["edges"] = [
                [a, b]
                for a, b in HALPE26_EDGES
                if a > 4
                and b > 4
                and valid[a]
                and valid[b]
                and np.linalg.norm(joints[a, :2] - joints[b, :2])
                <= 1.2 * max(box[2] - box[0], box[3] - box[1])
            ]
    return result


def draw_player_effects(source, plan, *, skeleton=True):
    """Soft body aura, foot spotlight and high-contrast observed limb lines."""
    if not plan["actors"]:
        return source.copy()
    aura = Image.new("RGBA", source.size)
    line = Image.new("RGBA", source.size)
    ad, ld = ImageDraw.Draw(aura), ImageDraw.Draw(line)
    scale = max(0.5, source.width / 1920)
    for actor in plan["actors"]:
        x1, y1, x2, y2 = actor["bbox_xyxy"]
        w, h = x2 - x1, y2 - y1
        color = tuple(actor["color"])
        # This is a soft spotlight within the observed box, not a segmentation mask.
        ad.ellipse((x1 - 0.08 * w, y1, x2 + 0.08 * w, y2), fill=(*color, 38))
        foot = (x1 - 0.1 * w, y2 - 0.045 * h, x2 + 0.1 * w, y2 + 0.065 * h)
        ad.ellipse(foot, fill=(*color, 115))
        ld.ellipse(foot, outline=(*color, 225), width=max(1, round(3 * scale)))
        # Short shoulder accents make subjects readable without boxing in the action.
        length = min(w * 0.25, h * 0.12)
        for x, sign in [(x1, -1), (x2, 1)]:
            ld.line(
                [(x, y1 + length), (x, y1), (x - sign * length, y1)],
                fill=(*color, 230),
                width=max(1, round(3 * scale)),
            )
        if not skeleton:
            continue
        points = actor["keypoints"]
        used = set()
        for a, b in actor["edges"]:
            coords = [tuple(points[a][:2]), tuple(points[b][:2])]
            side_color = (
                (255, 180, 70)
                if a in {5, 7, 9, 11, 13, 15, 20, 22, 24}
                else (75, 228, 255)
                if a in {6, 8, 10, 12, 14, 16, 21, 23, 25}
                else (250, 250, 240)
            )
            ld.line(coords, fill=(4, 12, 20, 230), width=max(3, round(7 * scale)))
            ld.line(coords, fill=(*side_color, 250), width=max(2, round(4 * scale)))
            used.update([a, b])
        for i in used:
            x, y = points[i][:2]
            r = max(1.5, 2.5 * scale)
            ld.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 240, 245))
    result = Image.alpha_composite(
        source.convert("RGBA"), aura.filter(ImageFilter.GaussianBlur(5 * scale))
    )
    return Image.alpha_composite(result, line).convert("RGB")
