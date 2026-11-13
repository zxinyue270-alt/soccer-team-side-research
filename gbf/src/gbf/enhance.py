"""Pipeline steps that sit between the M1 modules (one call each, so the pipeline stays
readable):

- :func:`clean_tracks` -- split track ids at cuts, keep only confident jersey numbers
  (gbf.track_tools).
- :func:`refine_from_plugin` -- if the plugin exported pitch keypoints
  (``keypoints.jsonl`` listed with its sha256 in ``keypoints-receipt.json``), refit the
  calibration with a fixed camera centre per camera (gbf.calib_refine) and recompute the
  ground positions of the tracks; otherwise return the inputs unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from gbf import calib_refine, track_tools
from gbf.util.hashing import sha256_file

KEYPOINTS_NAME = "keypoints.jsonl"
KEYPOINTS_RECEIPT_NAME = "keypoints-receipt.json"


class EnhanceError(ValueError):
    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__(f"{where}: " + "; ".join(self.problems))


def clean_tracks(tracks: pd.DataFrame, shots: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Split ids at cuts and drop unconfident jersey numbers; returns (tracks, counts)."""
    split, mapping = track_tools.split_tracks_at_cuts(tracks, shots)
    cleaned = track_tools.confident_jerseys(split)
    counts = {
        "track_ids_before": int(tracks.loc[tracks["track_id"] >= 0, "track_id"].nunique()),
        "track_ids_after": int(cleaned.loc[cleaned["track_id"] >= 0, "track_id"].nunique()),
        "pieces_renamed": int((mapping["track_id"] != mapping["source_track_id"]).sum()),
        "rows_with_jersey_before": int(tracks["jersey"].notna().sum()) if "jersey" in tracks else 0,
        "rows_with_jersey_after": int(cleaned["jersey"].notna().sum()),
    }
    return cleaned, counts


def _verified_keypoints(sngs_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    path = sngs_dir / KEYPOINTS_NAME
    receipt_path = sngs_dir / KEYPOINTS_RECEIPT_NAME
    if not path.is_file():
        return None
    if not receipt_path.is_file():
        raise EnhanceError(
            "refine_from_plugin", [f"{KEYPOINTS_NAME} has no {KEYPOINTS_RECEIPT_NAME}"]
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    listed = [o for o in receipt.get("outputs", []) if o.get("name") == KEYPOINTS_NAME]
    problems = []
    if not listed:
        problems.append(f"{KEYPOINTS_RECEIPT_NAME} does not list {KEYPOINTS_NAME}")
    else:
        if listed[0].get("sha256") != sha256_file(path):
            problems.append(f"{KEYPOINTS_NAME} sha256 differs from the receipt")
        if listed[0].get("bytes") != path.stat().st_size:
            problems.append(f"{KEYPOINTS_NAME} size differs from the receipt")
    if problems:
        raise EnhanceError("refine_from_plugin", problems)
    return path, receipt


def refine_from_plugin(
    sngs_dir: str | Path,
    tracks: pd.DataFrame,
    calib: pd.DataFrame,
    shots: pd.DataFrame,
    settings: calib_refine.RefineSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    """(tracks, calib, report): refined when the plugin exported keypoints, else as given.

    The report carries the refinement report plus the ground jitter of the tracks before
    and after (gbf.calib_refine.ground_jitter) and the keypoint receipt's restrictions.
    """
    found = _verified_keypoints(Path(sngs_dir))
    if found is None:
        return tracks, calib, None
    path, receipt = found
    keypoints = calib_refine.load_keypoints(path)
    result = calib_refine.refine_calibration(calib, keypoints, shots, settings)
    new_tracks = calib_refine.reproject_tracks(tracks, result.calib)
    report = dict(result.report)
    report["ground_jitter"] = {
        "plugin": calib_refine.ground_jitter(tracks),
        "refined": calib_refine.ground_jitter(new_tracks),
    }
    report["keypoints"] = {
        "file": KEYPOINTS_NAME,
        "sha256": sha256_file(path),
        "restrictions": list(receipt.get("restrictions", [])),
    }
    return new_tracks, result.calib, report
