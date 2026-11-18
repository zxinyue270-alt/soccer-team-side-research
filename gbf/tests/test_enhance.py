"""Tests for gbf.enhance (the glue between the M1 modules)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbf import enhance
from gbf.util.hashing import sha256_file
from test_calib_refine import TRUE_C, _players, _scene


def _write_keypoints(folder: Path, kps: dict[int, np.ndarray], *, tamper: bool = False) -> None:
    path = folder / enhance.KEYPOINTS_NAME
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for frame in sorted(kps):
            handle.write(json.dumps({"frame": frame, "points": kps[frame].tolist()}) + "\n")
    digest = sha256_file(path)
    if tamper:
        digest = "0" * 64
    receipt = {
        "restrictions": ["no_share"],
        "outputs": [{"name": path.name, "sha256": digest, "bytes": path.stat().st_size}],
    }
    (folder / enhance.KEYPOINTS_RECEIPT_NAME).write_text(
        json.dumps(receipt), encoding="utf-8", newline="\n"
    )


def test_without_keypoints_inputs_come_back_unchanged(unicode_tmp_path: Path) -> None:
    calib, _kps, shots, true_cams = _scene(seed=11, shots=((20, -3.0, 3.0, TRUE_C),))
    tracks, _ = _players(true_cams)
    t, c, report = enhance.refine_from_plugin(unicode_tmp_path, tracks, calib, shots)
    assert report is None
    assert t is tracks
    assert c is calib


def test_keypoints_refine_and_reproject(unicode_tmp_path: Path) -> None:
    calib, kps, shots, true_cams = _scene(seed=12, shots=((30, -6.0, 6.0, TRUE_C),))
    tracks, truth = _players(true_cams)
    tracks = enhance.calib_refine.reproject_tracks(tracks, calib)  # as sngs_import would
    _write_keypoints(unicode_tmp_path, kps)
    t, c, report = enhance.refine_from_plugin(
        unicode_tmp_path,
        tracks,
        calib,
        shots,
        enhance.calib_refine.RefineSettings(holdout_fraction=0.0),
    )
    assert report is not None
    assert c["refined"].all()
    assert report["keypoints"]["restrictions"] == ["no_share"]
    assert report["ground_jitter"]["refined"]["p90_m"] < report["ground_jitter"]["plugin"]["p90_m"]
    errors = [
        math.hypot(
            row.pitch_x - truth[(row.frame, row.track_id)][0],
            row.pitch_y - truth[(row.frame, row.track_id)][1],
        )
        for row in t.itertuples()
    ]
    assert np.median(errors) < 0.3


def test_tampered_keypoints_are_refused(unicode_tmp_path: Path) -> None:
    calib, kps, shots, true_cams = _scene(seed=13, shots=((10, -2.0, 2.0, TRUE_C),))
    tracks, _ = _players(true_cams)
    _write_keypoints(unicode_tmp_path, kps, tamper=True)
    with pytest.raises(enhance.EnhanceError, match="sha256"):
        enhance.refine_from_plugin(unicode_tmp_path, tracks, calib, shots)


def test_clean_tracks_counts() -> None:
    tracks = pd.DataFrame(
        {
            "frame": [0, 1, 12, 13],
            "track_id": [5, 5, 5, 6],
            "jersey": ["9", "9", None, "3"],
            "jersey_conf": [0.01, 0.02, math.nan, 0.99],
        }
    )
    shots = pd.DataFrame({"shot_id": [0, 1], "start_frame": [0, 10], "end_frame": [9, 19]})
    cleaned, counts = enhance.clean_tracks(tracks, shots)
    assert cleaned["track_id"].tolist() == [5, 5, 100_005, 6]
    assert cleaned["jersey"].isna().all()
    assert counts["pieces_renamed"] == 1
    assert counts["rows_with_jersey_after"] == 0


def test_clean_tracks_judges_a_reentered_piece_on_its_own() -> None:
    # Track 5 is read as "9" until it runs out of the picture at the right edge; the box the
    # tracker calls 5 after the gap is read as "4" twice only: too few reads of its own, so
    # it keeps no jersey rather than the "9" of the player who had the id before.
    edge, middle = (1250.0, 300.0, 1279.0, 380.0), (600.0, 300.0, 630.0, 380.0)
    frames = [0, 1, 2, 3, 4, 30, 31]
    boxes = [middle] * 4 + [edge, middle, middle]
    tracks = pd.DataFrame(
        {
            "frame": frames,
            "track_id": [5] * 7,
            **{name: [b[k] for b in boxes] for k, name in enumerate(("x1", "y1", "x2", "y2"))},
            "jersey": ["9"] * 5 + ["4", "4"],
            "jersey_conf": [0.95] * 7,
        }
    )
    shots = pd.DataFrame({"shot_id": [0], "start_frame": [0], "end_frame": [99]})
    cleaned, counts = enhance.clean_tracks(tracks, shots, frame_size=(1280, 720), fps=25.0)
    assert cleaned["track_id"].tolist() == [5] * 5 + [100_005] * 2
    assert cleaned["jersey"].iloc[:5].tolist() == ["9"] * 5
    assert cleaned["jersey"].iloc[5:].isna().all()
    assert counts["pieces_split_at_reentry"] == 1
    assert counts["pieces_split_at_cuts"] == 0
    assert counts["reentry_pieces"] == [
        {"track_id": 100_005, "source_track_id": 5, "start_frame": 30, "gap_frames": 25}
    ]
    # Without the frame size the track stays whole and the majority "9" covers every row.
    whole, _ = enhance.clean_tracks(tracks, shots)
    assert whole["track_id"].tolist() == [5] * 7
    assert whole["jersey"].tolist() == ["9"] * 7
