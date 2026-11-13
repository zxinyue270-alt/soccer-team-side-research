"""gbf.evaluate.gsr on synthetic annotations and runs (prereg/eval-sngs021-v1.md)."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from gbf import kinematics
from gbf.evaluate import gsr
from gbf.pipeline import KINEMATICS_PARAMS

W, H = 1920, 1080
ALIGNED = {"passed": True}
ALIGN_ARGS = ["--align-frames", "2,5,7"]


def _labels(
    path: Path,
    people: list[dict],
    n_frames: int,
    others: list[dict] | None = None,
    ignore: bool = False,
    no_camera: tuple[int, ...] = (),
) -> Path:
    """A GSR-shaped annotation file: people rows (frame, id, role, box, pitch) and others."""
    images = [
        {
            "is_labeled": True,
            "image_id": f"9001{n + 1:06d}",
            "file_name": f"{n + 1:06d}.jpg",
            "height": H,
            "width": W,
            "has_labeled_camera": n not in no_camera,
            "ignore_regions_x": [[1, 2]] if ignore and n == 0 else [],
            "ignore_regions_y": [],
        }
        for n in range(n_frames)
    ]
    annotations = []
    for i, p in enumerate(people + (others or [])):
        x1, y1, x2, y2 = p["box"]
        pitch = p.get("pitch", (math.nan, math.nan))
        annotations.append(
            {
                "id": str(i),
                "image_id": f"9001{p['frame'] + 1:06d}",
                "track_id": p.get("id", 0),
                "supercategory": "object",
                "category_id": 1,
                "attributes": {"role": p.get("role", "player"), "jersey": None, "team": None},
                "bbox_image": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                "bbox_pitch": {"x_bottom_middle": pitch[0], "y_bottom_middle": pitch[1]},
                "bbox_pitch_raw": {
                    "x_bottom_middle": pitch[0] + 0.1,
                    "y_bottom_middle": pitch[1],
                },
            }
        )
    annotations.append({"id": "pitch", "image_id": "9001000001", "supercategory": "pitch"})
    path.write_text(
        json.dumps({"info": {}, "images": images, "annotations": annotations}), encoding="utf-8"
    )
    return path


def _predictions(**columns: list) -> pd.DataFrame:
    n = len(columns["frame"])
    base = {
        "track_id": [5] * n,
        "x1": [0.0] * n,
        "y1": [0.0] * n,
        "x2": [10.0] * n,
        "y2": [10.0] * n,
        "pitch_x": [0.0] * n,
        "pitch_y": [0.0] * n,
        "pitch_usable": [True] * n,
    }
    return pd.DataFrame({**base, **columns})


# --------------------------------------------------------------------------- inputs


def test_frames_follow_image_names_and_only_people_are_tracked(tmp_path: Path) -> None:
    people = [
        {"frame": 0, "id": 3, "role": "player", "box": (10, 20, 30, 80), "pitch": (1.0, 2.0)},
        {"frame": 1, "id": 4, "role": "referee", "box": (10, 20, 30, 80), "pitch": (1.0, 2.0)},
    ]
    others = [
        {"frame": 1, "id": 9, "role": "ball", "box": (5, 5, 8, 8)},
        {"frame": 1, "id": 10, "role": "other", "box": (100, 100, 120, 160)},
    ]
    ref = gsr.load_reference(_labels(tmp_path / "l.json", people, 2, others, no_camera=(1,)))
    assert ref.people["frame"].tolist() == [0, 1]  # 000001.jpg is frame 0
    assert ref.people["ref_id"].tolist() == [3, 4]
    assert ref.people[["x1", "y1", "x2", "y2"]].iloc[0].tolist() == [10, 20, 30, 80]
    assert ref.people["pitch_x_raw"].iloc[0] == pytest.approx(1.1)
    assert len(ref.distractors) == 1  # the ball is no person and no distractor
    assert ref.image_numbers == (1, 2)
    assert ref.frames_without_camera == frozenset({1})
    assert (ref.n_frames, ref.width, ref.height) == (2, W, H)


def test_ignore_regions_are_refused(tmp_path: Path) -> None:
    path = _labels(tmp_path / "l.json", [], 1, ignore=True)
    with pytest.raises(gsr.GsrEvalError, match="ignore regions"):
        gsr.load_reference(path)


# --------------------------------------------------------------------------- check 0


def test_frame_alignment_finds_the_offset() -> None:
    rng = np.random.default_rng(0)
    images = {n: rng.integers(0, 255, size=(8, 8, 3)).astype(np.uint8) for n in range(1, 13)}
    aligned = {k: images[k + 1] for k in (2, 5, 7)}
    assert gsr.frame_alignment(aligned, images, (2, 5, 7))["passed"]
    shifted = {k: images[k + 2] for k in (2, 5, 7)}  # decoded frame k shows image k + 2
    result = gsr.frame_alignment(shifted, images, (2, 5, 7))
    assert not result["passed"]
    assert int(np.argmin(result["mean_abs_grey_difference"]["5"])) == 3  # offset +1


def _tiny_video(folder: Path, ffmpeg_bin: str, n: int = 10) -> tuple[Path, Path]:
    """An n-frame 64x48 noise video (H.264, CRF 18) and its source images 000001.jpg..."""
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 255, size=(n, 48, 64, 3)).astype(np.uint8)
    images = folder / "img1"
    images.mkdir(parents=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(images / f"{i + 1:06d}.jpg", quality=95)
    video = folder / "clip.mp4"
    subprocess.run(
        [
            ffmpeg_bin,
            *("-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x48"),
            *("-framerate", "25", "-i", "pipe:0", "-c:v", "libx264", "-crf", "18"),
            *("-pix_fmt", "yuv420p", str(video)),
        ],
        input=frames.tobytes(),
        check=True,
    )
    return video, images


def test_frame_alignment_on_a_real_encode(tmp_path: Path, ffmpeg_bin: str) -> None:
    video, images = _tiny_video(tmp_path, ffmpeg_bin)
    decoded, sources = gsr.load_alignment_frames(video, images, (2, 5, 7))
    assert gsr.frame_alignment(decoded, sources, (2, 5, 7))["passed"]


# --------------------------------------------------------------------------- matching


def test_box_iou() -> None:
    iou = gsr.box_iou(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [5, 0, 15, 10]]))
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(50 / 150)


def test_assignment_counts_sub_threshold_overlap_as_none() -> None:
    # raw IoU would pair r0-p1 and r1-p0 (0.55 + 0.45); with 0.45 set to 0, r0 takes p0
    iou = np.array([[0.6, 0.55], [0.45, 0.0]])
    assert gsr._assign(iou, 0.5) == [(0, 0, 0.6)]
    assert gsr._assign(np.array([[0.5]]), 0.5) == [(0, 0, 0.5)]  # the threshold itself counts


def test_matching_drops_predictions_on_distractors_from_precision(tmp_path: Path) -> None:
    people = [{"frame": 0, "id": 1, "box": (0, 0, 10, 10), "pitch": (0.0, 0.0)}]
    others = [{"frame": 0, "id": 9, "role": "other", "box": (100, 100, 110, 110)}]
    ref = gsr.load_reference(_labels(tmp_path / "l.json", people, 1, others))
    preds = _predictions(
        frame=[0, 0, 0],
        track_id=[7, 8, 9],
        x1=[0.0, 100.0, 500.0],
        y1=[0.0, 100.0, 500.0],
        x2=[10.0, 110.0, 510.0],
        y2=[10.0, 110.0, 510.0],
    )
    pairs, distractor = gsr.match(ref, preds)
    assert pairs[["ref_id", "track_id"]].values.tolist() == [[1, 7]]
    assert distractor.tolist() == [False, True, False]
    summary = gsr.detection_summary(ref, preds, pairs, distractor)
    assert summary["recall"] == 1.0
    assert summary["precision"] == pytest.approx(1 / 2)  # 3 predictions, 1 on a distractor


# --------------------------------------------------------------------------- position


def test_position_deviation_splits_along_the_camera_direction(tmp_path: Path) -> None:
    people = [{"frame": 0, "id": 1, "box": (0, 0, 10, 10), "pitch": (10.0, 20.0)}]
    ref = gsr.load_reference(_labels(tmp_path / "l.json", people, 1))
    preds = _predictions(frame=[0], pitch_x=[10.0], pitch_y=[21.0])
    pairs, _ = gsr.match(ref, preds)
    rows = gsr.position_rows(ref, preds, pairs, {0: (0.0, 70.0)})
    ux, uy = np.array([10.0, -50.0]) / math.hypot(10.0, 50.0)
    assert rows["deviation"].iloc[0] == pytest.approx(1.0)
    assert rows["radial"].iloc[0] == pytest.approx(1.0 * uy)
    assert rows["lateral"].iloc[0] == pytest.approx(1.0 * ux)
    assert rows["camera_distance"].iloc[0] == pytest.approx(math.hypot(10.0, 50.0))
    assert rows["raw_gap"].iloc[0] == pytest.approx(0.1)


def test_position_needs_a_usable_predicted_position(tmp_path: Path) -> None:
    people = [{"frame": 0, "id": 1, "box": (0, 0, 10, 10), "pitch": (1.0, 1.0)}]
    ref = gsr.load_reference(_labels(tmp_path / "l.json", people, 1))
    preds = _predictions(frame=[0], pitch_x=[1.0], pitch_y=[1.0], pitch_usable=[False])
    pairs, _ = gsr.match(ref, preds)
    assert len(pairs) == 1
    assert gsr.position_rows(ref, preds, pairs, {}).empty


def test_tercile_cut_points_are_numpy_quantiles() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    near, far = gsr.terciles(values)
    assert near == pytest.approx(np.quantile(values, 1 / 3))
    assert far == pytest.approx(np.quantile(values, 2 / 3))


# --------------------------------------------------------------------------- speed


def test_the_smoother_is_the_runs_smoother() -> None:
    assert gsr.SG_WINDOW == KINEMATICS_PARAMS["sg_window_frames"]
    assert gsr.SG_POLY == KINEMATICS_PARAMS["sg_poly"]
    assert gsr.MAX_GAP_FRAMES == KINEMATICS_PARAMS["max_gap_frames"]
    assert gsr.MIN_SEGMENT_FRAMES == KINEMATICS_PARAMS["min_segment_frames"]
    assert gsr.CHORD_S == kinematics.DEFAULT_SPEED_CHORD_S


@pytest.mark.parametrize("missing", [(), (30, 31)])
def test_reference_speeds_equal_gbf_kinematics_on_clean_tracks(missing: tuple[int, ...]) -> None:
    frames = [f for f in range(60) if f not in missing]
    t = np.asarray(frames, dtype=float) / 25.0
    x = 3.0 * t + 0.8 * t**2  # accelerating from 3 m/s
    y = 5.0 + 0.5 * np.sin(t)
    people = pd.DataFrame(
        {"frame": frames, "ref_id": 1, "role": "player", "pitch_x": x, "pitch_y": y}
    )
    ours = gsr.reference_kinematics(people)
    tracks = pd.DataFrame(
        {
            "frame": frames,
            "track_id": 1,
            "pitch_usable": True,
            "pitch_x": x,
            "pitch_y": y,
            "in_pitch": True,
        }
    )
    shots = pd.DataFrame({"shot_id": [0], "start_frame": [0], "end_frame": [59]})
    options = {**KINEMATICS_PARAMS, "max_box_height_change": None, "max_box_width_change": None}
    theirs, _ = kinematics.compute_kinematics(tracks, shots, 25.0, **options)
    both = ours.merge(theirs, on="frame", suffixes=("_ours", "_gbf"))
    use = both["defined"] & both["valid"]
    assert use.sum() > 30
    assert (both["defined"] == both["valid"]).all()
    np.testing.assert_allclose(both.loc[use, "speed_ours"], both.loc[use, "speed_gbf"], atol=1e-9)


def test_gaps_up_to_three_frames_are_bridged_and_longer_ones_split() -> None:
    bridged = [f for f in range(60) if f not in (20, 21, 22)]
    split = [f for f in range(60) if f not in (20, 21, 22, 23)]
    for frames, segments in ((bridged, 1), (split, 2)):
        people = pd.DataFrame(
            {"frame": frames, "ref_id": 1, "pitch_x": np.arange(len(frames)) * 0.2, "pitch_y": 0.0}
        )
        assert gsr.reference_kinematics(people)["segment"].nunique() == segments


def test_short_segments_give_no_reference_speed_or_position() -> None:
    people = pd.DataFrame(
        {"frame": range(24), "ref_id": 1, "pitch_x": np.arange(24) * 0.2, "pitch_y": 0.0}
    )
    kin = gsr.reference_kinematics(people)
    assert not kin["defined"].any()
    assert kin["x"].isna().all()


def test_reference_accelerations() -> None:
    n = 60
    t = np.arange(n) / 25.0
    people = pd.DataFrame({"frame": range(n), "ref_id": 1, "pitch_x": 2.0 * t**2, "pitch_y": 0.0})
    acc = gsr.reference_kinematics(people)["acceleration"].to_numpy(dtype=float)
    acc = acc[np.isfinite(acc)]
    assert acc.size > 20
    np.testing.assert_allclose(acc, 4.0, atol=1e-6)


def _per_frame(
    n: int, ref: np.ndarray, pred: np.ndarray, tracks: np.ndarray, acceleration: float = 0.0
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": 0,
            "ref_id": 1,
            "role": "player",
            "frame": np.arange(n),
            "speed": ref,
            "defined": np.isfinite(ref),
            "acceleration": acceleration,
            "track_id": tracks,
            "pred_speed": pred,
            "pred_valid": np.isfinite(pred),
            "usable": True,
            "in_f": np.isfinite(ref) & np.isfinite(pred) & (tracks >= 0),
        }
    )


def test_peak_deviation_uses_the_common_frames() -> None:
    n = 30
    ref = np.full(n, 5.0)
    ref[3] = 9.0  # the reference peak lies where the prediction has no valid speed
    pred = np.full(n, 5.5)
    pred[3] = np.nan
    pred[20] = 6.0
    row = gsr.segment_peaks(_per_frame(n, ref, pred, np.full(n, 7))).iloc[0]
    assert bool(row["comparable"])
    assert row["deviation"] == pytest.approx(6.0 - 5.0)
    assert row["deviation_all"] == pytest.approx(6.0 - 9.0)
    assert row["identity_exposure"] == 0.0
    assert not row["peak_in_f"]
    assert row["quiet_f"]


def test_comparable_needs_25_common_frames_and_switches_are_exposed() -> None:
    n = 30
    ref = np.full(n, 5.0)
    pred = np.full(n, np.nan)
    pred[:24] = 5.0
    assert not gsr.segment_peaks(_per_frame(n, ref, pred, np.full(n, 7))).iloc[0]["comparable"]
    pred[:25] = 5.0
    assert gsr.segment_peaks(_per_frame(n, ref, pred, np.full(n, 7))).iloc[0]["comparable"]
    tracks = np.where(np.arange(n) < 20, 7, 8)
    peaks = gsr.segment_peaks(_per_frame(n, ref, np.full(n, 5.0), tracks))
    assert peaks.iloc[0]["identity_exposure"] == pytest.approx(10 / 30)


def test_noisy_reference_frames_leave_the_sensitivity_set() -> None:
    n = 30
    frames = _per_frame(n, np.full(n, 5.0), np.full(n, 5.0), np.full(n, 7), acceleration=12.0)
    assert not gsr.segment_peaks(frames).iloc[0]["quiet_f"]


# --------------------------------------------------------------------------- statistics


def test_cluster_bootstrap_draws_whole_tracks_and_is_reproducible() -> None:
    groups = {1: np.array([1.0, 1.0]), 2: np.array([3.0]), 3: np.array([])}
    first = gsr.cluster_bootstrap(groups, np.median, resamples=200)
    assert first == gsr.cluster_bootstrap(groups, np.median, resamples=200)
    assert first is not None
    assert 1.0 <= first[0] <= first[1] <= 3.0
    assert gsr.cluster_bootstrap({}, np.median) is None


def test_results_have_no_nan() -> None:
    value = {"a": math.nan, "b": [np.float64(1.5), math.inf], "c": np.int64(3)}
    assert gsr._jsonable(value) == {"a": None, "b": [1.5, None], "c": 3}


# --------------------------------------------------------------------------- TrackEval files


def test_write_mot_files(tmp_path: Path) -> None:
    people = [{"frame": 0, "id": 4, "box": (10, 20, 30, 80), "pitch": (0.0, 0.0)}]
    ref = gsr.load_reference(_labels(tmp_path / "l.json", people, 3))
    preds = pd.DataFrame(
        {"frame": [2], "track_id": [11], "x1": [1.0], "y1": [2.0], "x2": [4.0], "y2": [8.0]}
    )
    gsr.write_mot(ref, preds, tmp_path / "te", "gbf-refine", "SEQ")
    gt = (tmp_path / "te" / "gt" / "GSR-eval" / "SEQ" / "gt" / "gt.txt").read_text()
    assert gt.splitlines() == ["1,4,10.000,20.000,20.000,60.000,1,1,1"]
    info = (tmp_path / "te" / "gt" / "GSR-eval" / "SEQ" / "seqinfo.ini").read_text()
    assert "seqLength=3" in info
    assert "imWidth=1920" in info
    data = tmp_path / "te" / "trackers" / "GSR-eval" / "gbf-refine" / "data" / "SEQ.txt"
    assert data.read_text().splitlines() == ["3,11,1.000,2.000,3.000,6.000,1,-1,-1,-1"]


# --------------------------------------------------------------------------- end to end


def _synthetic_run(
    tmp_path: Path, offset: tuple[float, float], size: tuple[int, int] = (W, H)
) -> tuple[Path, Path]:
    """Two players running for 60 frames; the prediction's boxes equal the reference boxes
    and its pitch positions are shifted by ``offset``; frame 10 has no labelled camera."""
    n = 60
    people, rows = [], []
    for ref_id, (x0, vx) in {1: (0.0, 4.0), 2: (10.0, 6.0)}.items():
        xs = x0 + vx * np.arange(n) / 25.0
        for f in range(n):
            box = (100.0 * ref_id + f, 300.0, 100.0 * ref_id + f + 40.0, 400.0)
            people.append({"frame": f, "id": ref_id, "box": box, "pitch": (xs[f], 5.0)})
            rows.append(
                {
                    "frame": f,
                    "track_id": 20 + ref_id,
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "pitch_x": xs[f] + offset[0],
                    "pitch_y": 5.0 + offset[1],
                    "pitch_usable": True,
                    "in_pitch": True,
                }
            )
    labels = _labels(tmp_path / "labels.json", people, n, no_camera=(10,))
    run = tmp_path / "run"
    for sub in ("refine", "import"):
        (run / sub).mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(run / sub / "tracks.parquet")
    calib = pd.DataFrame(
        {
            "frame_index": range(n),
            "width": size[0],
            "height": size[1],
            "C": [np.array([0.0, 70.0, -20.0])] * n,
            "pitch_usable": True,
        }
    )
    calib.to_parquet(run / "refine" / "calib.parquet")
    shots = pd.DataFrame({"shot_id": [0], "start_frame": [0], "end_frame": [n - 1]})
    options = {**KINEMATICS_PARAMS, "max_box_height_change": None, "max_box_width_change": None}
    kin, _ = kinematics.compute_kinematics(pd.DataFrame(rows), shots, 25.0, **options)
    kin.to_parquet(run / "kinematics.parquet")
    (run / "run.json").write_text("{}", encoding="utf-8")
    return labels, run


def test_evaluate_reports_a_known_shift_and_equal_speeds(tmp_path: Path) -> None:
    labels, run = _synthetic_run(tmp_path, (0.3, -0.4))
    result = gsr.evaluate(labels, run, ALIGNED)
    assert result["stopped"] == []
    assert result["restrictions"] == ["no_share"]
    refine = result["refine"]
    assert refine["matching"]["recall"] == 1.0
    assert refine["matching"]["precision"] == 1.0
    position = refine["position"]
    assert position["deviation"]["median"] == pytest.approx(0.5)
    assert position["coverage"] == 1.0
    left_out = position["deviation_without_frames_lacking_a_labelled_camera"]
    assert left_out["frames_left_out"] == [10]
    assert left_out["n"] == 2 * 59
    assert position["context_bbox_pitch_vs_raw"]["median"] == pytest.approx(0.1)
    assert refine["position_smoothed"]["median"] == pytest.approx(0.5)
    speed = refine["speed"]
    assert speed["comparable_segments"] == 2
    assert speed["s1_peak"]["abs"]["median"] == pytest.approx(0.0, abs=1e-9)
    assert speed["s1_peak"]["share_with_reference_peak_in_f"] == 1.0
    assert speed["s2_per_frame"]["abs"]["p90"] == pytest.approx(0.0, abs=1e-9)
    assert speed["s3_noise_floor"]["share_above_11_mps"] == 0.0
    assert result["import"]["boxes_and_ids_equal_refine"]


def test_a_failed_alignment_stops_before_any_measure(tmp_path: Path) -> None:
    labels, run = _synthetic_run(tmp_path, (0.0, 0.0))
    result = gsr.evaluate(labels, run, {"passed": False})
    assert result["stopped"] == ["check 0: frame alignment did not pass (or was not run)"]
    assert "refine" not in result
    assert "import" not in result


def test_a_different_image_size_stops(tmp_path: Path) -> None:
    labels, run = _synthetic_run(tmp_path, (0.0, 0.0), size=(1280, 720))
    assert any(stop.startswith("check 1") for stop in gsr.evaluate(labels, run, ALIGNED)["stopped"])


def _cli(
    tmp_path: Path, labels: Path, run: Path, ffmpeg_bin: str, *extra: str
) -> tuple[Path, Path, list[str]]:
    """The output folder, the video and the eval-gsr arguments for a synthetic run."""
    video, images = _tiny_video(tmp_path / "media", ffmpeg_bin)
    out = tmp_path / "out"
    args = ["--labels", str(labels), "--run", str(run), "--video", str(video)]
    return out, video, [*args, "--images", str(images), "--out", str(out), *ALIGN_ARGS, *extra]


def _results(out: Path) -> dict:
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


def test_cli_writes_results_and_trackeval_files(tmp_path: Path, ffmpeg_bin: str) -> None:
    labels, run = _synthetic_run(tmp_path, (0.0, 0.0))
    out, video, args = _cli(tmp_path, labels, run, ffmpeg_bin, "--seq", "S")
    assert gsr.main(args) == 0
    result = _results(out)
    assert result["stopped"] == []
    assert result["checks"]["check0_frame_alignment"]["passed"]
    assert result["checks"]["reference_frames"] == 60
    assert result["inputs"]["video"]["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert result["tracking"] is None  # no --trackeval-python
    assert (out / "trackeval" / "trackers" / "GSR-eval" / "gbf-import" / "data" / "S.txt").exists()


FAKE_RUNNER = """
import json
import sys

root, seq, n_frames, out = sys.argv[1:5]
row = {"HOTA": 0.5, "IDSW": 2, "message": "Success"}
trackers = {"gbf-refine": row, "gbf-import": dict(row)}
summary = {"tool": "sn-trackeval", "version": "0.4.0", "python": "3.9.25", "trackers": trackers}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(summary, handle)
"""

FAILING_RUNNER = """
import sys

sys.stderr.write("trackeval is not installed")
sys.exit(1)
"""


def test_cli_runs_trackeval_and_compares_import_with_refine(
    tmp_path: Path, ffmpeg_bin: str
) -> None:
    labels, run = _synthetic_run(tmp_path, (0.0, 0.0))
    runner = tmp_path / "runner.py"
    runner.write_text(FAKE_RUNNER, encoding="utf-8")
    extra = ("--trackeval-python", sys.executable, "--trackeval-runner", str(runner))
    out, _, args = _cli(tmp_path, labels, run, ffmpeg_bin, *extra)
    assert gsr.main(args) == 0
    tracking = _results(out)["tracking"]
    assert tracking["t_primary_refine"]["IDSW"] == 2
    assert tracking["t_secondary_equals_t_primary"] is True
    assert tracking["runner_sha256"] == hashlib.sha256(runner.read_bytes()).hexdigest()
    assert (tracking["version"], tracking["python"]) == ("0.4.0", "3.9.25")


def test_cli_records_a_failed_trackeval_run(tmp_path: Path, ffmpeg_bin: str) -> None:
    labels, run = _synthetic_run(tmp_path, (0.0, 0.0))
    runner = tmp_path / "runner.py"
    runner.write_text(FAILING_RUNNER, encoding="utf-8")
    extra = ("--trackeval-python", sys.executable, "--trackeval-runner", str(runner))
    out, _, args = _cli(tmp_path, labels, run, ffmpeg_bin, *extra)
    assert gsr.main(args) == 4
    assert _results(out)["tracking"]["stderr_tail"] == ["trackeval is not installed"]


def test_tracking_compares_every_number(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("", encoding="utf-8")
    trackers = {"gbf-refine": {"HOTA": 0.5, "message": "a"}, "gbf-import": {"HOTA": 0.4}}
    assert (
        gsr.tracking_block({"trackers": trackers}, runner)["t_secondary_equals_t_primary"] is False
    )
    del trackers["gbf-import"]
    assert (
        gsr.tracking_block({"trackers": trackers}, runner)["t_secondary_equals_t_primary"] is None
    )


def test_cli_stops_on_a_far_position_frame(tmp_path: Path, ffmpeg_bin: str) -> None:
    labels, run = _synthetic_run(tmp_path, (30.0, 0.0))
    out, _, args = _cli(tmp_path, labels, run, ffmpeg_bin)
    assert gsr.main(args) == 3
    result = _results(out)
    assert [stop[:7] for stop in result["stopped"]] == ["check 3"]
    assert "refine" not in result
    assert "tracking" not in result
    assert not (out / "trackeval").exists()


def test_cli_refuses_unusable_labels(tmp_path: Path, ffmpeg_bin: str) -> None:
    _, run = _synthetic_run(tmp_path, (0.0, 0.0))
    labels = _labels(tmp_path / "ignored.json", [], 60, ignore=True)
    out, _, args = _cli(tmp_path, labels, run, ffmpeg_bin)
    assert gsr.main(args) == 2
    assert not (out / "results.json").exists()
