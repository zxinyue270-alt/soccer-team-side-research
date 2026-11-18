"""Conservative adjacent-frame background alignment for decorative image overlays."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


class BackgroundMotion:
    """A causal similarity estimate, never a metric camera calibration."""

    def __init__(self):
        self.previous = None
        self.mask = None
        self.frame = None

    def update(self, frame, raw, boxes=(), ball=None, *, cut=False):
        try:
            import cv2
        except ImportError:
            return {"valid": False, "reason": "opencv_unavailable"}
        height, width = raw.shape[:2]
        small_width = min(640, width)
        small_height = round(height * small_width / width)
        gray = cv2.cvtColor(
            np.asarray(
                Image.fromarray(raw).resize((small_width, small_height), Image.Resampling.BILINEAR)
            ),
            cv2.COLOR_RGB2GRAY,
        )
        sx, sy = small_width / width, small_height / height
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        mask[: round(small_height * 0.12)] = 0
        mask[round(small_height * 0.92) :] = 0
        mask[:, : round(small_width * 0.03)] = 0
        mask[:, round(small_width * 0.97) :] = 0
        for box in boxes:
            pad = 0.15 * (box[3] - box[1])
            a, b = max(0, int((box[0] - pad) * sx)), max(0, int((box[1] - pad) * sy))
            c, d = (
                min(small_width, math.ceil((box[2] + pad) * sx)),
                min(small_height, math.ceil((box[3] + pad) * sy)),
            )
            mask[b:d, a:c] = 0
        if ball is not None:
            x, y, r = ball
            if all(math.isfinite(v) for v in (x, y, r)):
                cv2.circle(mask, (round(x * sx), round(y * sy)), max(5, round(r * 3 * sx)), 0, -1)
        previous, previous_mask, previous_frame = self.previous, self.mask, self.frame
        self.previous, self.mask, self.frame = gray, mask, frame
        if cut or previous is None or frame != previous_frame + 1 or previous.shape != gray.shape:
            return {"valid": False, "reason": "boundary"}
        try:
            points = cv2.goodFeaturesToTrack(previous, 600, 0.01, 8, mask=previous_mask)
            if points is None or len(points) < 25:
                return {"valid": False, "reason": "few_background_features"}
            forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
            backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, forward, None)
            p, q = points.reshape(-1, 2), forward.reshape(-1, 2)
            valid = (status[:, 0] == 1) & (back_status[:, 0] == 1)
            valid &= np.linalg.norm(p - backward.reshape(-1, 2), axis=1) < 1
            valid &= np.isfinite(q).all(axis=1)
            valid &= (
                (q[:, 0] >= 0)
                & (q[:, 0] < small_width - 1)
                & (q[:, 1] >= 0)
                & (q[:, 1] < small_height - 1)
            )
            indexes = np.flatnonzero(valid)
            indexes = indexes[mask[q[indexes, 1].astype(int), q[indexes, 0].astype(int)] > 0]
            p, q = p[indexes], q[indexes]
            if len(p) < 25:
                return {"valid": False, "reason": "few_background_matches"}
            affine, inliers = cv2.estimateAffinePartial2D(
                p, q, method=cv2.RANSAC, ransacReprojThreshold=1.5
            )
            if affine is None:
                return {"valid": False, "reason": "alignment_failed"}
            inliers = inliers[:, 0].astype(bool)
            fraction = float(inliers.mean())
            error = float(
                np.median(
                    np.linalg.norm(p[inliers] @ affine[:, :2].T + affine[:, 2] - q[inliers], axis=1)
                )
            )
            spread = np.ptp(p[inliers], axis=0) if inliers.any() else np.zeros(2)
            zoom = math.hypot(affine[0, 0], affine[1, 0])
            rotation = abs(math.atan2(affine[1, 0], affine[0, 0]))
            reliable = (
                inliers.sum() >= 25
                and fraction >= 0.65
                and error <= 1.5
                and spread[0] >= 0.3 * small_width
                and spread[1] >= 0.2 * small_height
                and 0.95 <= zoom <= 1.05
                and rotation <= math.radians(3)
                and np.linalg.norm(affine[:, 2]) <= 0.15 * math.hypot(small_width, small_height)
            )
            small = np.vstack([affine, [0, 0, 1]])
            scale = np.diag([sx, sy, 1.0])
            full = np.linalg.inv(scale) @ small @ scale
            return {
                "valid": bool(reliable),
                "reason": "background_similarity" if reliable else "alignment_uncertain",
                "matrix": full.tolist() if reliable else None,
                "inlier_fraction": fraction,
                "inliers": int(inliers.sum()),
                "median_error_small_px": error,
            }
        except cv2.error:
            return {"valid": False, "reason": "optical_flow_error"}
