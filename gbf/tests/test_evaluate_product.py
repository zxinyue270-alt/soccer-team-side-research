"""Product acceptance at the public scoring and offline CLI boundaries; synthetic only."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from itertools import product as combinations
from pathlib import Path

import pytest

from gbf.evaluate import product


def event(ident: str, start: int, peak: int, end: int, kind: str = "ball_strike") -> dict:
    return {"id": ident, "start": start, "peak": peak, "end": end, "types": [kind]}


def prediction(ident: str, start: int, peak: int, end: int, kind: str = "ball_strike") -> dict:
    return {
        "id": ident,
        "start_frame": start,
        "peak_frame": peak,
        "end_frame": end,
        "type": kind,
        "tier": "candidate",
    }


def test_matching_maximises_event_count_before_overlap() -> None:
    # A greedy first match A->X loses B. Only A->Y, B->X finds both events.
    labels = [event("A", 0, 4, 9), event("B", 5, 9, 14)]
    predictions = [prediction("X", 0, 4, 14), prediction("Y", 0, 4, 5)]
    result = product.score_events(labels, predictions, frame_count=25, fps="25/1")
    assert result["tp"] == 2
    assert [(r["event"], r["prediction"]) for r in result["pairs"]] == [("A", "Y"), ("B", "X")]


def test_duplicate_prediction_ids_are_rejected_instead_of_hiding_false_positives() -> None:
    with pytest.raises(product.ProductEvalError, match="duplicate"):
        product.score_events(
            [event("A", 0, 4, 9)],
            [prediction("X", 0, 4, 9), prediction("X", 0, 4, 9)],
            frame_count=25,
            fps="25/1",
        )


def test_duplicates_misses_and_uncertain_predictions_keep_their_denominators() -> None:
    result = product.score_events(
        [event("A", 0, 4, 9), event("B", 30, 35, 39)],
        [prediction("X", 0, 4, 9), prediction("Y", 0, 4, 9), prediction("Z", 60, 65, 69)],
        frame_count=100,
        fps="25/1",
        uncertain_intervals=[[60, 69], [65, 74]],
    )
    assert (result["tp"], result["fp"], result["fn"], result["unscorable"]) == (1, 1, 1, 1)
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
    assert result["uncertain_frames"] == 15


def test_type_does_not_bias_matching_and_id_ties_are_order_independent() -> None:
    labels = [event("A", 0, 4, 9)]
    predictions = [prediction("Z", 0, 4, 9), prediction("B", 0, 4, 9, "sprint")]
    first = product.score_events(labels, predictions, frame_count=25, fps="25/1")
    second = product.score_events(labels, predictions[::-1], frame_count=25, fps="25/1")
    assert first == second
    assert first["pairs"][0]["prediction"] == "B"
    assert first["type_correct"] == 0
    assert first["by_type"]["ball_strike"]["fn"] == 1
    assert first["by_type"]["sprint"]["fp"] == 1


def test_timing_breaks_equal_overlap_before_ids() -> None:
    result = product.score_events(
        [event("A", 0, 5, 9)],
        [prediction("A", 0, 0, 9), prediction("Z", 0, 5, 9)],
        frame_count=25,
        fps="25/1",
    )
    assert result["pairs"][0]["prediction"] == "Z"


def test_iou_boundary_is_inclusive_and_whole_clip_cannot_match_tiny_event() -> None:
    result = product.score_events(
        [event("A", 0, 1, 9)], [prediction("X", 0, 1, 2)], frame_count=100, fps="25/1"
    )
    assert result["tp"] == 1  # 3/10 exactly
    result = product.score_events(
        [event("A", 0, 1, 9)], [prediction("X", 0, 1, 99)], frame_count=100, fps="25/1"
    )
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)


def test_zero_denominators_are_missing_and_second_tolerance_uses_rational_fps() -> None:
    empty = product.score_events([], [], frame_count=100, fps="25/1")
    assert empty["precision"] is None
    assert empty["recall"] is None
    result = product.score_events(
        [event("A", 0, 40, 99)],
        [prediction("X", 0, 70, 99)],
        frame_count=100,
        fps="30000/1001",
    )
    assert result["peak_ok"] == 0  # 30 frames is longer than one second


@pytest.mark.parametrize(("field", "value"), [("start", -1), ("peak", True), ("end", 25)])
def test_invalid_label_frames_fail_explicitly(field: str, value: int) -> None:
    label = event("A", 0, 4, 9)
    label[field] = value
    with pytest.raises(product.ProductEvalError):
        product.score_events([label], [], frame_count=25, fps="25/1")


def test_failed_clip_events_remain_false_negatives_and_block_acceptance() -> None:
    scores = product.score_events([event("A", 0, 4, 9)], [], frame_count=25, fps="25/1")
    result = product.summarize(
        [{"id": "c", "match_id": "m", "stratum": "uniform", "run_ok": False, "scores": scores}],
        data_kind="independent",
    )
    assert result["counts"]["fn"] == 1
    assert result["status"] != "passed"
    assert result["checks"]["all_runs_completed"] is False


def test_preflight_reports_missing_materials_without_claiming_a_pass(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "data_kind": "independent", "clips": []}), encoding="utf-8"
    )
    output = tmp_path / "preflight"
    assert product.main(["--preflight", "--manifest", str(manifest), "--out", str(output)]) == 3
    result = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert "no_clips" in result["blockers"]
    assert "freeze_pending" in result["blockers"]
    assert not (output / "results.json").exists()


def artifact(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def put(path: Path, value: dict) -> dict:
    path.write_text(json.dumps(value), encoding="utf-8")
    return artifact(path)


def synthetic_manifest(tmp_path: Path) -> Path:
    video = tmp_path / "synthetic-video.bin"
    video.write_bytes(b"synthetic: not an actual video or an accuracy test")
    labels = put(
        tmp_path / "labels.json",
        {"frame_count": 25, "fps": "25/1", "exhaustive": True, "events": [event("A", 0, 4, 9)]},
    )
    highlights = put(
        tmp_path / "highlights.json",
        {"frame_count": 25, "fps": "25/1", "highlights": [prediction("X", 0, 4, 9)]},
    )
    verification = put(tmp_path / "verification.json", {"status": "passed", "source_frames": 25})
    run = put(
        tmp_path / "run.json",
        {
            "inputs": [{"name": "video", "sha256": artifact(video)["sha256"]}],
            "code": {"git_commit": "synthetic", "git_dirty": False},
        },
    )
    manifest = {
        "schema_version": 1,
        "data_kind": "synthetic",
        "core_commit": "synthetic",
        "clips": [
            {
                "id": "c",
                "match_id": "synthetic:m",
                "stratum": "uniform",
                "fps": "25/1",
                "frame_count": 25,
                "video": artifact(video),
                "labels": labels,
                "outputs": {
                    "status": "passed",
                    "highlights": highlights,
                    "verification": verification,
                    "run": run,
                },
            }
        ],
    }
    path = tmp_path / "manifest.json"
    put(path, manifest)
    return path


def test_cli_synthetic_highlights_never_certify(tmp_path: Path) -> None:
    manifest = synthetic_manifest(tmp_path)
    output = tmp_path / "evaluation"
    assert product.main(["--manifest", str(manifest), "--out", str(output)]) == 0
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert result["summary"]["status"] == "diagnostic_only"
    assert result["summary"]["counts"]["tp"] == 1
    assert result["summary"]["three_d_accuracy"]["status"] == "not_evaluated"
    assert result["clips"][0]["coverage"]["status"] == "not_evaluated_no_targets"
    assert (output / "summary-zh.md").is_file()


def test_missing_detections_stay_in_three_d_coverage_denominator() -> None:
    labels = {
        "events": [event("A", 0, 12, 24) | {"subjects": ["s"]}],
        "targets": [
            {"subject_id": "s", "frame": f, "visible": True, "box": [0, 0, 10, 100]}
            for f in range(25)
        ],
    }
    result = product.score_coverage(labels, [], [], [], frame_count=25)
    assert result["three_d"]["considered"] == 25
    assert result["three_d"]["used"] == 0
    assert result["three_d"]["share"] == 0
    assert result["three_d"]["reasons"] == {"no_detection": 25}
    assert result["three_d"]["by_box_height"]["80-149"]["considered"] == 25


def test_cli_rejects_hash_changes_before_comparing_predictions(tmp_path: Path) -> None:
    manifest = synthetic_manifest(tmp_path)
    (tmp_path / "highlights.json").write_text("{}", encoding="utf-8")
    assert (
        product.main(["--preflight", "--manifest", str(manifest), "--out", str(tmp_path / "out")])
        == 2
    )
    assert not (tmp_path / "out" / "results.json").exists()


@pytest.mark.parametrize(
    "change", ["fps", "frame_count", "run_video", "run_commit", "verification"]
)
def test_cli_rejects_inconsistent_native_provenance(tmp_path: Path, change: str) -> None:
    path = synthetic_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    outputs = manifest["clips"][0]["outputs"]
    if change in {"fps", "frame_count"}:
        doc = json.loads((tmp_path / "highlights.json").read_text(encoding="utf-8"))
        doc[change] = "24/1" if change == "fps" else 26
        outputs["highlights"] = put(tmp_path / "highlights.json", doc)
    elif change in {"run_video", "run_commit"}:
        doc = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
        if change == "run_video":
            doc["inputs"][0]["sha256"] = "0" * 64
        else:
            doc["code"]["git_commit"] = "wrong"
        outputs["run"] = put(tmp_path / "run.json", doc)
    else:
        outputs["verification"] = put(
            tmp_path / "verification.json", {"status": "failed", "source_frames": 25}
        )
    put(path, manifest)
    assert product.main(["--manifest", str(path), "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out" / "results.json").exists()


def test_independent_manifest_rejects_a_known_development_match(tmp_path: Path) -> None:
    path = synthetic_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["data_kind"] = "independent"
    manifest["clips"][0]["match_id"] = "soccernet-gsr:5"
    with pytest.raises(product.ProductEvalError, match="already exposed"):
        product.preflight(manifest, tmp_path)


def test_frozen_input_fingerprint_excludes_outputs_but_binds_labels(tmp_path: Path) -> None:
    path = synthetic_manifest(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    before = product.input_fingerprint(doc)
    doc["clips"][0]["outputs"]["status"] = "failed"
    assert product.input_fingerprint(doc) == before
    doc["clips"][0]["labels"]["sha256"] = "0" * 64
    assert product.input_fingerprint(doc) != before


def test_coverage_uses_human_box_and_retains_missing_frames() -> None:
    labels = {
        "events": [event("A", 0, 1, 2) | {"subjects": ["s"]}],
        "targets": [
            {"subject_id": "s", "frame": f, "visible": True, "box": [0, 0, 10, 160]}
            for f in range(3)
        ],
    }
    tracks = [
        {
            "frame": 1,
            "track_id": 7,
            "x1": 0,
            "y1": 0,
            "x2": 10,
            "y2": 160,
            "pitch_usable": True,
            "pitch_x": 1,
            "pitch_y": 2,
        }
    ]
    lifted = [
        {
            "frame": 1,
            "track_id": 7,
            "method": "height_prior_calibrated",
            "reason": "lifted",
            "joints3d": [0.0] * 78,
        }
    ]
    result = product.score_coverage(labels, tracks, lifted, [], frame_count=3)
    assert (result["three_d"]["used"], result["three_d"]["considered"]) == (1, 3)
    assert result["three_d"]["by_box_height"][">=150"]["share"] == pytest.approx(1 / 3)
    assert result["metrics"]["positions-trajectories"]["share"] == 1
    assert result["metrics"]["peak-speed"]["share"] == 0


def test_partial_visibility_labels_cannot_shrink_coverage_denominator() -> None:
    labels = {"events": [event("A", 0, 1, 2) | {"subjects": ["s"]}], "targets": []}
    with pytest.raises(product.ProductEvalError, match="incomplete"):
        product.score_coverage(labels, [], [], [], frame_count=3)


def test_recorded_results_are_never_overwritten(tmp_path: Path) -> None:
    manifest = synthetic_manifest(tmp_path)
    args = ["--manifest", str(manifest), "--out", str(tmp_path / "out")]
    assert product.main(args) == 0
    before = (tmp_path / "out" / "results.json").read_bytes()
    assert product.main(args) == 2
    assert (tmp_path / "out" / "results.json").read_bytes() == before


def test_exact_matcher_agrees_with_an_exhaustive_small_problem_oracle() -> None:
    # Enumerate every possible assignment independently of the residual graph algorithm.
    import random

    rng = random.Random(19)
    for _ in range(30):
        labels, predictions = [], []
        for i in range(3):
            a, b = sorted(rng.sample(range(20), 2))
            labels.append(event(str(i), a, (a + b) // 2, b))
            a, b = sorted(rng.sample(range(20), 2))
            predictions.append(prediction(str(i), a, (a + b) // 2, b))
        possibilities = []
        for assignment in combinations(range(-1, 3), repeat=3):
            chosen = [j for j in assignment if j != -1]
            if len(chosen) != len(set(chosen)):
                continue
            pairs, overlap, distance = [], Fraction(0), 0
            for i, j in enumerate(assignment):
                if j == -1:
                    continue
                a, b = labels[i], predictions[j]
                intersection = max(
                    0, min(a["end"], b["end_frame"]) - max(a["start"], b["start_frame"]) + 1
                )
                union = a["end"] - a["start"] + b["end_frame"] - b["start_frame"] + 2 - intersection
                iou = Fraction(intersection, union)
                if iou < Fraction(3, 10) or not b["start_frame"] <= a["peak"] <= b["end_frame"]:
                    break
                pairs.append((str(i), str(j)))
                overlap += iou
                distance += abs(a["peak"] - b["peak_frame"])
            else:
                possibilities.append(((-len(pairs), -overlap, distance, pairs), pairs))
        expected = min(possibilities)[1]
        actual = product.score_events(labels, predictions, frame_count=25, fps="25/1")
        assert [(p["event"], p["prediction"]) for p in actual["pairs"]] == expected


def test_acceptance_uses_exact_integer_thresholds_and_reports_by_match() -> None:
    clips = []
    for i in range(12):
        labels = [event("A", 0, 4, 9), event("B", 30, 34, 39)]
        predictions = [prediction("X", 0, 4, 9), prediction("Y", 30, 34, 39)]
        if i % 4 == 0:
            predictions += [prediction("D1", 0, 4, 9), prediction("D2", 0, 4, 9)]
        scores = product.score_events(labels, predictions, frame_count=750, fps="25/1")
        clips.append(
            {
                "id": str(i),
                "match_id": str(i // 4),
                "stratum": "event" if i % 4 < 2 else "uniform",
                "run_ok": True,
                "scores": scores,
            }
        )
    result = product.summarize(clips, data_kind="independent")
    assert result["status"] == "passed"  # pure arithmetic on synthetic counts, no material verdict
    assert result["counts"]["precision"] == 0.8  # 24/30 exactly
    assert len(result["by_match"]) == 3
    clips[0]["scores"]["fp"] += 1
    below = product.summarize(clips, data_kind="independent")
    assert below["status"] == "failed"
    assert below["checks"]["precision"] is False


@pytest.mark.parametrize("raw", ['{"schema_version":1,"schema_version":1}', '{"x":NaN}'])
def test_cli_rejects_ambiguous_or_nonfinite_json(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(raw, encoding="utf-8")
    assert (
        product.main(["--preflight", "--manifest", str(path), "--out", str(tmp_path / "out")]) == 2
    )


def test_frozen_manifest_cannot_change_inputs(tmp_path: Path) -> None:
    doc = {
        "schema_version": 1,
        "data_kind": "independent",
        "clips": [],
        "freeze": {"status": "frozen", "inputs_sha256": "0" * 64},
    }
    with pytest.raises(product.ProductEvalError, match="frozen inputs changed"):
        product.preflight(doc, tmp_path)


def test_independent_labels_require_real_distinct_blind_reviewers(tmp_path: Path) -> None:
    path = synthetic_manifest(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["data_kind"] = "independent"
    clip = doc["clips"][0]
    clip.update(match_id="unseen:match", frame_count=750)
    label_doc = {
        "frame_count": 750,
        "fps": "25/1",
        "exhaustive": True,
        "events": [],
        "locked_at": "2026-11-13T00:00:00+08:00",
        "annotators": [{"id": "one", "kind": "model"}, {"id": "one", "kind": "model"}],
    }
    clip["labels"] = put(tmp_path / "labels.json", label_doc)
    result = product.preflight(doc, tmp_path)
    assert "c:independent_blind_labels_missing" in result["blockers"]
    assert "sample_design_incomplete" in result["blockers"]


def test_cli_reads_native_parquet_and_measurement_coverage(tmp_path: Path) -> None:
    import pandas as pd

    path = synthetic_manifest(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    clip = doc["clips"][0]
    labels = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    labels["events"][0]["subjects"] = ["subject"]
    labels["targets"] = [
        {"subject_id": "subject", "frame": f, "visible": True, "box": [0, 0, 10, 100]}
        for f in range(17)
    ]
    clip["labels"] = put(tmp_path / "labels.json", labels)
    tracks = tmp_path / "tracks.parquet"
    pd.DataFrame(
        [
            {
                "frame": 4,
                "track_id": 7,
                "x1": 0,
                "y1": 0,
                "x2": 10,
                "y2": 100,
                "pitch_usable": True,
                "pitch_x": 1.0,
                "pitch_y": 2.0,
            }
        ]
    ).to_parquet(tracks)
    lift = tmp_path / "lift.parquet"
    pd.DataFrame(
        [
            {
                "frame": 4,
                "track_id": 7,
                "method": "height_prior_calibrated",
                "reason": "lifted",
                "joints3d": [0.0] * 78,
            }
        ]
    ).to_parquet(lift)
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text(
        json.dumps(
            {"frame": 4, "track_id": 7, "metric_id": "peak-speed", "status": "ok", "value": 5.0}
        )
        + "\n",
        encoding="utf-8",
    )
    clip["outputs"].update(
        tracks=artifact(tracks), lift=artifact(lift), measurements=artifact(measurements)
    )
    put(path, doc)
    out = tmp_path / "evaluation"
    assert product.main(["--manifest", str(path), "--out", str(out)]) == 0
    report = json.loads((out / "results.json").read_text(encoding="utf-8"))
    coverage = report["clips"][0]["coverage"]
    assert coverage["three_d"]["used"] == 1
    assert coverage["three_d"]["considered"] == 17
    assert coverage["metrics"]["peak-speed"]["share"] == 1
