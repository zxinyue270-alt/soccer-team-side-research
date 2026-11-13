"""gbf.evaluate.gsr_dev on synthetic data (prereg/eval-gsr-dev-v1.md)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbf.evaluate import gsr_dev, pitch_lines
from test_evaluate_gsr import FAILING_RUNNER, FAKE_RUNNER, _synthetic_run, _tiny_video

# --------------------------------------------------------------------------- jump windows (P3)


def _raw_people(n_people: int, jump: float, at: int = 10, frames: int = 30) -> pd.DataFrame:
    """People walking 0.125 m per frame (exact in binary), all jumping by ``jump`` at ``at``."""
    rows = [
        {
            "frame": f,
            "ref_id": ref_id,
            "pitch_x_raw": 0.125 * f + ref_id + (jump if f >= at else 0.0),
            "pitch_y_raw": 0.0,
        }
        for ref_id in range(n_people)
        for f in range(frames)
    ]
    return pd.DataFrame(rows)


def test_a_common_jump_of_five_people_is_a_jump_frame() -> None:
    assert gsr_dev.jump_frames(_raw_people(5, 2.875)) == [10]  # step 3.0


def test_four_people_or_a_step_of_exactly_two_metres_is_no_jump() -> None:
    assert gsr_dev.jump_frames(_raw_people(4, 2.875)) == []
    assert gsr_dev.jump_frames(_raw_people(6, 1.875)) == []  # step exactly 2.0, not above


def test_windows_stay_inside_the_clip() -> None:
    expected = {0, 1, 2, 3, 4, 5, 37, 38, 39, 40, 41, 42, 43}
    assert gsr_dev.jump_windows([2, 40], 45, half=3) == frozenset(expected)


# --------------------------------------------------------------------------- reprojection (R)

K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
CENTRE = (0.0, 60.0, -15.0)


def _look_at(centre: tuple[float, float, float]) -> np.ndarray:
    """Rotation of a camera at ``centre`` looking at the centre spot (world z points down)."""
    c = np.asarray(centre, dtype=float)
    forward = -c / np.linalg.norm(c)
    right = np.cross([0.0, 0.0, 1.0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack([right, down, forward])


def _document(points: dict[str, list[tuple[float, float]]]) -> dict:
    lines = {name: [{"x": u / 1920, "y": v / 1080} for u, v in uv] for name, uv in points.items()}
    return {
        "images": [{"image_id": "1", "file_name": "000001.jpg", "width": 1920, "height": 1080}],
        "annotations": [{"supercategory": "pitch", "image_id": "1", "lines": lines}],
    }


def _calib(usable: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [0],
            "K": [K.ravel()],
            "R": [_look_at(CENTRE).ravel()],
            "C": [np.array(CENTRE)],
            "pitch_usable": [usable],
        }
    )


def test_projected_model_points_have_no_reprojection_error() -> None:
    model = pitch_lines.ground_lines()["Middle line"][::200]
    uv = gsr_dev.project_ground(model, K, _look_at(CENTRE), CENTRE)
    rows, skipped = gsr_dev.reprojection_rows(
        _document({"Middle line": [tuple(p) for p in uv]}), _calib(), 1920, 1080
    )
    assert len(rows) == len(model)
    assert rows["distance_px"].max() < 1e-6
    assert skipped == {
        "frame_not_usable": 0,
        "line_not_on_ground_model": 0,
        "line_not_projected": 0,
    }


def test_a_three_pixel_shift_reads_three_pixels_and_other_lines_are_counted() -> None:
    far = np.array([[0.0, -34.0]])  # the far touchline projects horizontally for this camera
    u, v = gsr_dev.project_ground(far, K, _look_at(CENTRE), CENTRE)[0]
    document = _document({"Side line top": [(u, v + 3.0)], "Goal left crossbar": [(u, v)]})
    rows, skipped = gsr_dev.reprojection_rows(document, _calib(), 1920, 1080)
    assert 2.99 < rows["distance_px"].iloc[0] < 3.05
    assert skipped["line_not_on_ground_model"] == 1


def test_frames_without_a_usable_calibration_are_counted_not_measured() -> None:
    rows, skipped = gsr_dev.reprojection_rows(
        _document({"Middle line": [(960.0, 600.0)]}), _calib(usable=False), 1920, 1080
    )
    assert rows.empty
    assert skipped["frame_not_usable"] == 1


def test_the_point_looked_at_projects_to_the_principal_point() -> None:
    centre_spot = np.array([[0.0, 0.0]])
    uv = gsr_dev.project_ground(centre_spot, K, _look_at(CENTRE), CENTRE)[0]
    assert uv == pytest.approx([960.0, 540.0])


def test_points_behind_the_camera_are_not_projected() -> None:
    behind = np.array([[0.0, 80.0]])  # beyond the camera, which stands at y = 60
    assert np.isnan(gsr_dev.project_ground(behind, K, _look_at(CENTRE), CENTRE)).all()


# --------------------------------------------------------------------------- nearest opponent (N)


def _people() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [  # frame 0: ref 1 (left) at the origin
        (0, 1, "player", "left", 0.0, 0.0),
        (0, 2, "player", "left", 1.0, 0.0),  # team mate: no opponent
        (0, 3, "player", "right", 3.0, 4.0),  # 5 m: the nearest opponent
        (0, 4, "player", "right", 10.0, 0.0),
        (0, 5, "goalkeeper", "right", 0.5, 0.0),  # keepers are not outfield players
    ]
    people = pd.DataFrame(
        [(f, r, role, x, y) for f, r, role, _, x, y in rows],
        columns=["frame", "ref_id", "role", "pitch_x", "pitch_y"],
    )
    teams = pd.DataFrame(
        [(f, r, t) for f, r, _, t, _, _ in rows], columns=["frame", "ref_id", "team"]
    )
    return people, teams


def test_reference_nearest_opponent_counts_outfield_players_of_the_other_team() -> None:
    people, teams = _people()
    assert gsr_dev.reference_nearest_opponent(people, teams, 1, [0]) == pytest.approx(5.0)
    assert gsr_dev.reference_nearest_opponent(people, teams, 1, [7]) is None


def test_nearest_opponent_rows_pair_gbf_values_with_the_reference() -> None:
    people, teams = _people()
    shown = [
        {"metric_id": "nearest-opponent-distance", "value": 4.5, "frames": [0], "track_id": 9},
        {"metric_id": "peak-speed", "value": 7.0, "frames": [0], "track_id": 9},
    ]
    pairs = pd.DataFrame({"frame": [0], "ref_id": [1], "track_id": [9]})
    table = gsr_dev.nearest_opponent_rows(shown, pairs, people, teams)
    assert len(table) == 1
    assert table.iloc[0]["deviation_m"] == pytest.approx(-0.5)
    ambiguous = pd.DataFrame({"frame": [0, 0], "ref_id": [1, 2], "track_id": [9, 9]})
    row = gsr_dev.nearest_opponent_rows(shown, ambiguous, people, teams).iloc[0]
    assert row["reference_m"] is None


def test_reference_teams_read_the_attribute() -> None:
    document = {
        "images": [{"image_id": "7", "file_name": "000003.jpg"}],
        "annotations": [
            {
                "supercategory": "object",
                "image_id": "7",
                "track_id": 4,
                "attributes": {"role": "player", "team": "left"},
            },
            {
                "supercategory": "object",
                "image_id": "7",
                "track_id": 9,
                "attributes": {"role": "ball"},
            },
        ],
    }
    teams = gsr_dev.reference_teams(document)
    assert teams.to_dict("records") == [{"frame": 2, "ref_id": 4, "team": "left"}]


# --------------------------------------------------------------------------- pooling


def test_pooling_resamples_matches() -> None:
    out = gsr_dev.pooled({"A": [1.0, 1.0], "B": [3.0], "C": [5.0]}, {"A": "7", "B": "7", "C": "9"})
    assert out["n"] == 4
    assert out["median"] == pytest.approx(2.0)
    assert (out["clusters"], out["n_clusters"]) == ("match", 2)
    assert out["per_clip"]["C"]["median"] == 5.0
    low, high = out["median_ci95"]
    assert 1.0 <= low <= high <= 5.0


def test_pooling_falls_back_to_clips_without_match_ids() -> None:
    out = gsr_dev.pooled({"A": [1.0], "B": [2.0]}, {"A": "7", "B": None})
    assert (out["clusters"], out["n_clusters"]) == ("clip", 2)


def test_pooling_nothing_gives_nulls() -> None:
    out = gsr_dev.pooled({}, {})
    assert (out["n"], out["median"], out["median_ci95"]) == (0, None, None)


# --------------------------------------------------------------------------- command line


def _clip(tmp_path: Path, seq: str, offset: tuple[float, float], game: str) -> dict:
    """A synthetic clip (two players, 60 frames; see test_evaluate_gsr._synthetic_run) with a
    game id, one nearest-opponent value gbf showed, and a 10-frame video for check 0."""
    folder = tmp_path / seq
    folder.mkdir()
    labels, run = _synthetic_run(folder, offset)
    document = json.loads(labels.read_text(encoding="utf-8"))
    document["info"] = {"game_id": game}
    labels.write_text(json.dumps(document), encoding="utf-8")
    shown = {"metric_id": "nearest-opponent-distance", "value": 2.0, "frames": [5], "track_id": 21}
    (run / "measurements.jsonl").write_text(json.dumps(shown) + "\n", encoding="utf-8")
    return {"seq": seq, "labels": str(labels), "run": str(run)}


def _manifest(tmp_path: Path, ffmpeg_bin: str, clips: list[dict]) -> Path:
    video, images = _tiny_video(tmp_path / "media", ffmpeg_bin)
    for clip in clips:
        clip.update({"video": str(video), "images": str(images)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": clips}), encoding="utf-8")
    return manifest


def _run(tmp_path: Path, manifest: Path, *extra: str) -> tuple[int, dict]:
    out = tmp_path / "out"
    code = gsr_dev.main(
        ["--manifest", str(manifest), "--out", str(out), "--align-frames", "2,5,7", *extra]
    )
    return code, json.loads((out / "results.json").read_text(encoding="utf-8"))


def test_cli_pools_two_clips_of_one_match(tmp_path: Path, ffmpeg_bin: str) -> None:
    clips = [_clip(tmp_path, "A", (0.3, -0.4), "5"), _clip(tmp_path, "B", (0.0, 0.0), "5")]
    code, result = _run(tmp_path, _manifest(tmp_path, ffmpeg_bin, clips))
    assert code == 0
    assert result["stopped_clips"] == []
    p_all = result["pooled"]["p_all"]
    assert p_all["n"] == 240  # 2 clips x 2 players x 60 frames
    assert p_all["median"] == pytest.approx(0.25)  # half the pairs 0.5 m, half 0 m
    assert (p_all["clusters"], p_all["n_clusters"]) == ("match", 1)
    assert p_all["per_clip"]["A"]["median"] == pytest.approx(0.5)
    assert result["pooled"]["s2_all"]["p90"] == pytest.approx(0.0, abs=1e-9)
    assert result["clips"]["A"]["jumps"]["frames"] == []
    assert result["nearest_opponent"]["summary"] == {"shown": 2, "compared": 0}
    assert result["tracking"] is None
    trackers = tmp_path / "out" / "trackeval" / "trackers" / "GSR-eval"
    assert (trackers / "gbf-refine" / "data" / "B.txt").exists()


def test_cli_leaves_a_stopped_clip_out_of_the_pool(tmp_path: Path, ffmpeg_bin: str) -> None:
    clips = [_clip(tmp_path, "A", (0.3, -0.4), "5"), _clip(tmp_path, "B", (30.0, 0.0), "6")]
    code, result = _run(tmp_path, _manifest(tmp_path, ffmpeg_bin, clips))
    assert code == 0
    assert result["stopped_clips"] == ["B"]
    assert result["pooled"]["p_all"]["n"] == 120
    assert "values" not in result["clips"]["B"]
    assert result["clips"]["B"]["per_clip"]["stopped"]


def test_cli_runs_trackeval_and_compares_import_with_refine(
    tmp_path: Path, ffmpeg_bin: str
) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text(FAKE_RUNNER, encoding="utf-8")
    manifest = _manifest(tmp_path, ffmpeg_bin, [_clip(tmp_path, "A", (0.0, 0.0), "5")])
    extra = ("--trackeval-python", sys.executable, "--trackeval-runner", str(runner))
    code, result = _run(tmp_path, manifest, *extra)
    assert code == 0
    assert result["tracking"]["t_secondary_equals_t_primary"] is True
    assert result["tracking"]["runner_sha256"] == hashlib.sha256(runner.read_bytes()).hexdigest()
    seq_info = tmp_path / "out" / "trackeval" / "seq_info.json"
    assert json.loads(seq_info.read_text(encoding="utf-8")) == {"A": 60}


def test_cli_records_a_failed_trackeval_run(tmp_path: Path, ffmpeg_bin: str) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text(FAILING_RUNNER, encoding="utf-8")
    manifest = _manifest(tmp_path, ffmpeg_bin, [_clip(tmp_path, "A", (0.0, 0.0), "5")])
    extra = ("--trackeval-python", sys.executable, "--trackeval-runner", str(runner))
    code, result = _run(tmp_path, manifest, *extra)
    assert code == 4
    assert result["tracking"]["stderr_tail"] == ["trackeval is not installed"]
