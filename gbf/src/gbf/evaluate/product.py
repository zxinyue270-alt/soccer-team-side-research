"""Offline product acceptance. No video inference, network access or registry promotion.

Protocol: prereg/product-acceptance-v1.proposed.md. Source annotations and receipts stay
outside the repository. Missing references are never interpreted as zero error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

KINDS = frozenset({"duel_tackle", "attack_in_box", "ball_strike", "sprint", "closeup_reaction"})
Cost = tuple[Fraction, int, int]
ZERO: Cost = (Fraction(0), 0, 0)


class ProductEvalError(ValueError):
    """Invalid or contradictory evaluation inputs."""


@dataclass
class _Edge:
    target: int
    reverse: int
    capacity: int
    cost: Cost


def _add(a: Cost, b: Cost) -> Cost:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _edge(graph: list[list[_Edge]], a: int, b: int, cost: Cost) -> _Edge:
    forward = _Edge(b, len(graph[b]), 1, cost)
    backward = _Edge(a, len(graph[a]), 0, (-cost[0], -cost[1], -cost[2]))
    graph[a].append(forward)
    graph[b].append(backward)
    return forward


def _iou(a: dict, b: dict) -> Fraction:
    overlap = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]) + 1)
    union = a["end"] - a["start"] + b["end"] - b["start"] + 2 - overlap
    return Fraction(overlap, union)


def match_events(labels: list[dict], predictions: list[dict]) -> list[tuple[dict, dict]]:
    """Exact lexicographic matching: count, rational IoU sum, peak distance, then IDs.

    Successive shortest augmenting paths with tuple-valued costs avoid epsilon weights
    that could trade a real IoU improvement for an unrelated timing/ID tie-break.
    Inputs use start/peak/end and have already passed score_events validation.
    """
    labels = sorted(labels, key=lambda row: row["id"])
    predictions = sorted(predictions, key=lambda row: row["id"])
    candidates = [
        (i, j)
        for i, a in enumerate(labels)
        for j, b in enumerate(predictions)
        if _iou(a, b) >= Fraction(3, 10) and b["start"] <= a["peak"] <= b["end"]
    ]
    source, sink = len(labels) + len(predictions), len(labels) + len(predictions) + 1
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    for i in range(len(labels)):
        _edge(graph, source, i, ZERO)
    for j in range(len(predictions)):
        _edge(graph, len(labels) + j, sink, ZERO)
    links = []
    for rank, (i, j) in enumerate(candidates):
        a, b = labels[i], predictions[j]
        cost = (-_iou(a, b), abs(a["peak"] - b["peak"]), -(1 << (len(candidates) - rank)))
        links.append((i, j, _edge(graph, i, len(labels) + j, cost)))
    while True:
        distances: list[Cost | None] = [None] * len(graph)
        parents: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = ZERO
        for _ in range(len(graph) - 1):
            changed = False
            for u, edges in enumerate(graph):
                if distances[u] is None:
                    continue
                for index, edge in enumerate(edges):
                    if not edge.capacity:
                        continue
                    cost = _add(distances[u], edge.cost)
                    if distances[edge.target] is None or cost < distances[edge.target]:
                        distances[edge.target] = cost
                        parents[edge.target] = (u, index)
                        changed = True
            if not changed:
                break
        if distances[sink] is None:
            break
        v = sink
        while v != source:
            parent = parents[v]
            if parent is None:
                raise ProductEvalError("matching residual path is incomplete")
            u, index = parent
            edge = graph[u][index]
            edge.capacity = 0
            graph[v][edge.reverse].capacity = 1
            v = u
    return [(labels[i], predictions[j]) for i, j, edge in links if edge.capacity == 0]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductEvalError(message)


def _rows(rows: list[dict], frame_count: int, *, predicted: bool) -> None:
    _require(isinstance(rows, list) and len(rows) <= 128, "expected at most 128 events per clip")
    ids: set[str] = set()
    for row in rows:
        _require(isinstance(row, dict), "event must be an object")
        ident = row.get("id")
        _require(isinstance(ident, str) and bool(ident.strip()), "event needs a nonempty id")
        _require(ident not in ids, "duplicate event/prediction id")
        ids.add(ident)
        keys = ("start_frame", "peak_frame", "end_frame") if predicted else ("start", "peak", "end")
        values = [row.get(key) for key in keys]
        _require(all(type(v) is int for v in values), "frame numbers must be integers")
        start, peak, end = values
        _require(0 <= start <= peak <= end < frame_count, "event frames out of range or order")
        kinds = [row.get("type")] if predicted else row.get("types")
        _require(isinstance(kinds, list) and bool(kinds), "event types required")
        _require(all(isinstance(k, str) and k in KINDS for k in kinds), "unknown event type")


def score_events(
    labels: list[dict],
    predictions: list[dict],
    *,
    frame_count: int,
    fps: str,
    uncertain_intervals: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Score all final highlights, including candidates, against exhaustive labels."""
    try:
        rate = Fraction(fps)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise ProductEvalError("invalid fps") from exc
    if rate <= 0 or type(frame_count) is not int or frame_count < 1:
        raise ProductEvalError("positive fps and frame_count required")
    _rows(labels, frame_count, predicted=False)
    _rows(predictions, frame_count, predicted=True)
    normalised = [
        {
            "id": row["id"],
            "start": row["start_frame"],
            "peak": row["peak_frame"],
            "end": row["end_frame"],
            "type": row["type"],
        }
        for row in predictions
    ]
    intervals = uncertain_intervals or []
    _require(isinstance(intervals, list), "uncertain intervals must be a list")
    for interval in intervals:
        _require(
            isinstance(interval, list)
            and len(interval) == 2
            and all(type(v) is int for v in interval),
            "uncertain interval needs two integer frames",
        )
        _require(0 <= interval[0] <= interval[1] < frame_count, "uncertain interval out of range")
        _require(
            not any(interval[0] <= row["peak"] <= interval[1] for row in labels),
            "definite event peak lies in an uncertain interval",
        )
    pairs = match_events(labels, normalised)
    used = {b["id"] for _, b in pairs}
    unmatched = [row for row in normalised if row["id"] not in used]
    unscorable = [
        row["id"]
        for row in unmatched
        if any(row["start"] <= end and row["end"] >= start for start, end in intervals)
    ]
    rows = [
        {
            "event": a["id"],
            "prediction": b["id"],
            "type_agrees": b["type"] in a["types"],
            "iou": float(_iou(a, b)),
            "peak_error_frames": b["peak"] - a["peak"],
            "peak_within_one_second": abs(b["peak"] - a["peak"]) <= rate,
        }
        for a, b in pairs
    ]
    tp, unknown = len(pairs), len(unscorable)
    fp, fn = len(unmatched) - unknown, len(labels) - tp
    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "unscorable": unknown,
        "precision": _ratio(tp, tp + fp + unknown),
        "precision_excluding_unscorable": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "type_correct": sum(r["type_agrees"] for r in rows),
        "peak_ok": sum(r["peak_within_one_second"] for r in rows),
        "pairs": rows,
        "missed_events": sorted(
            a["id"] for a in labels if a["id"] not in {x["id"] for x, _ in pairs}
        ),
        "false_predictions": sorted(r["id"] for r in unmatched if r["id"] not in unscorable),
        "unscorable_predictions": sorted(unscorable),
        "uncertain_frames": len({f for start, end in intervals for f in range(start, end + 1)}),
        "frame_count": frame_count,
        "duration_s": float(frame_count / rate),
    }
    result["by_type"] = {
        kind: {
            "reference_events": sum(kind in a["types"] for a in labels),
            "detected_events": sum(kind in a["types"] for a, _ in pairs),
            "predictions": sum(b["type"] == kind for b in normalised),
            "unmatched_predictions": sum(b["type"] == kind for b in unmatched),
            "tp": sum(kind in a["types"] and b["type"] == kind for a, b in pairs),
            "fn": sum(kind in a["types"] for a in labels)
            - sum(kind in a["types"] and b["type"] == kind for a, b in pairs),
            "fp": sum(b["type"] == kind and b["id"] not in unscorable for b in unmatched)
            + sum(b["type"] == kind and kind not in a["types"] for a, b in pairs),
            "unscorable": sum(b["type"] == kind and b["id"] in unscorable for b in unmatched),
        }
        for kind in sorted(KINDS)
    }
    return result


def _counts(clips: list[dict]) -> dict[str, Any]:
    counts = {
        key: sum(clip["scores"][key] for clip in clips)
        for key in ("tp", "fp", "fn", "unscorable", "peak_ok", "type_correct")
    }
    tp, fp, fn, unknown = (counts[key] for key in ("tp", "fp", "fn", "unscorable"))
    seconds = sum(clip["scores"]["duration_s"] for clip in clips)
    counts.update(
        precision=_ratio(tp, tp + fp + unknown),
        recall=_ratio(tp, tp + fn),
        type_agreement=_ratio(counts["type_correct"], tp),
        peak_within_one_second=_ratio(counts["peak_ok"], tp),
        false_positives_per_minute=_ratio(fp * 60, seconds),
        unscorable_per_minute=_ratio(unknown * 60, seconds),
    )
    return counts


def _at_least(n: int, d: int, numerator: int, denominator: int) -> bool:
    return d > 0 and denominator * n >= numerator * d


def summarize(clips: list[dict], *, data_kind: str) -> dict[str, Any]:
    """Aggregate clips by match, retaining failed runs and every definite reference event."""
    _require(data_kind in {"synthetic", "development", "independent"}, "unknown data_kind")
    matches = sorted({clip["match_id"] for clip in clips})
    by_match = {match: _counts([c for c in clips if c["match_id"] == match]) for match in matches}
    by_stratum = {
        stratum: _counts([c for c in clips if c["stratum"] == stratum])
        for stratum in ("event", "uniform")
    }
    counts = _counts(clips)
    tp, fp, fn, unknown = (counts[key] for key in ("tp", "fp", "fn", "unscorable"))
    checks = {
        "all_runs_completed": bool(clips) and all(c["run_ok"] for c in clips),
        "design_complete": len(clips) == 12
        and len(matches) == 3
        and all(
            sum(c["match_id"] == match and c["stratum"] == stratum for c in clips) == 2
            for match in matches
            for stratum in ("event", "uniform")
        ),
        "enough_events": tp + fn >= 20 and len(matches) >= 3,
        "precision": _at_least(tp, tp + fp + unknown, 4, 5),
        "recall": _at_least(tp, tp + fn, 4, 5),
        "peak_timing": _at_least(counts["peak_ok"], tp, 4, 5),
        "per_match": all(
            (v["tp"] + v["fn"] == 0 or _at_least(v["tp"], v["tp"] + v["fn"], 3, 5))
            and (
                v["tp"] + v["fp"] + v["unscorable"] == 0
                or _at_least(v["tp"], v["tp"] + v["fp"] + v["unscorable"], 3, 5)
            )
            for v in by_match.values()
        ),
        "uncertainty": all(
            5 * c["scores"]["uncertain_frames"] <= c["scores"]["frame_count"] for c in clips
        ),
    }
    if data_kind != "independent":
        status = "diagnostic_only"
    elif not checks["enough_events"] or not checks["design_complete"]:
        status = "insufficient_evidence"
    else:
        status = "passed" if all(checks.values()) else "failed"
    intervals: dict[str, list[float] | None] = {"precision": None, "recall": None}
    if matches:
        rng = np.random.default_rng(0)
        samples: dict[str, list[float]] = {"precision": [], "recall": []}
        for _ in range(2000):
            selected = [by_match[matches[i]] for i in rng.integers(len(matches), size=len(matches))]
            a, b, c, u = (sum(v[k] for v in selected) for k in ("tp", "fp", "fn", "unscorable"))
            for key, value in (("precision", _ratio(a, a + b + u)), ("recall", _ratio(a, a + c))):
                if value is not None:
                    samples[key].append(value)
        intervals = {
            key: [float(v) for v in np.quantile(values, [0.025, 0.975])] if values else None
            for key, values in samples.items()
        }
    return {
        "status": status,
        "counts": counts,
        "checks": checks,
        "by_match": by_match,
        "by_stratum": by_stratum,
        "by_type": {
            kind: {
                key: sum(c["scores"]["by_type"][kind][key] for c in clips)
                for key in (
                    "reference_events",
                    "detected_events",
                    "predictions",
                    "unmatched_predictions",
                    "tp",
                    "fp",
                    "fn",
                    "unscorable",
                )
            }
            for kind in sorted(KINDS)
        },
        "match_bootstrap_ci95": intervals,
        "bootstrap_note": (
            "2000 resamples, seed 0; three matches give unstable intervals; "
            "not a lower-bound certification"
        ),
        "measurement_accuracy": {"status": "not_evaluated_no_reference"},
        "three_d_accuracy": {"status": "not_evaluated"},
    }


def _decode(text: str) -> dict:
    def invalid_constant(value: str) -> None:
        raise ProductEvalError(f"non-finite JSON constant: {value}")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(text, parse_constant=invalid_constant, object_pairs_hook=unique_keys)
    _require(isinstance(value, dict), "JSON document must be an object")
    return value


def _json(path: Path) -> dict:
    return _decode(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _artifact(ref: dict, base: Path) -> Path:
    _require(isinstance(ref, dict), "artifact needs path and sha256")
    _require(isinstance(ref.get("path"), str) and bool(ref["path"]), "artifact path missing")
    path = Path(ref["path"])
    path = path if path.is_absolute() else base / path
    _require(path.is_file(), f"artifact missing: {path}")
    _require(_digest(path) == ref.get("sha256"), f"artifact hash mismatch: {path}")
    return path


def input_fingerprint(manifest: dict) -> str:
    """Hash only pre-inference inputs, excluding the push receipt and later model outputs."""
    value = {k: v for k, v in manifest.items() if k not in {"freeze", "clips"}}
    value["clips"] = [{k: v for k, v in c.items() if k != "outputs"} for c in manifest["clips"]]
    canonical = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _time(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        _require(result.tzinfo is not None, "timestamps need a timezone")
        return result
    except (ValueError, AttributeError) as exc:
        raise ProductEvalError("invalid timestamp") from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _freeze_checks(manifest: dict, base: Path) -> list[str]:
    freeze = manifest.get("freeze", {})
    blockers = []
    if freeze.get("status") != "frozen":
        blockers.append("freeze_pending")
    if not _git("remote"):
        blockers.append("project_remote_missing")
    if not manifest.get("protocol"):
        blockers.append("protocol_missing")
    if not manifest.get("runtime"):
        blockers.append("runtime_hashes_missing")
    if freeze.get("status") != "frozen":
        return blockers
    _require(freeze.get("inputs_sha256") == input_fingerprint(manifest), "frozen inputs changed")
    receipt = _json(_artifact(freeze.get("receipt"), base))
    _require(
        all(
            receipt.get(k) == freeze.get(k) and receipt.get(k)
            for k in ("commit", "remote_url", "pushed_at", "inputs_sha256")
        ),
        "push receipt disagrees with freeze",
    )
    _time(freeze["pushed_at"])
    _require(_git("rev-parse", "HEAD") == freeze["commit"], "freeze commit is not checked out")
    _require(not _git("status", "--porcelain"), "core working tree is dirty")
    urls = _git("config", "--get-regexp", r"^remote\..*\.url$")
    _require(
        any(line.split(maxsplit=1)[-1] == freeze["remote_url"] for line in urls.splitlines()),
        "freeze remote is not configured",
    )
    _require(_git("rev-parse", freeze["commit"]) == freeze["commit"], "freeze commit unavailable")
    _require(
        _git("merge-base", manifest["core_commit"], freeze["commit"]) == manifest["core_commit"],
        "freeze is not descended from the tested code commit",
    )
    _require(
        not _git(
            "diff", manifest["core_commit"], freeze["commit"], "--", "src", "tools", "registries"
        ),
        "runtime or evaluator code changed after the tested commit",
    )
    marker = _decode(_git("show", freeze["commit"] + ":prereg/product-acceptance-v1-freeze.json"))
    _require(
        marker.get("inputs_sha256") == input_fingerprint(manifest)
        and marker.get("core_commit") == manifest["core_commit"],
        "tracked freeze marker mismatch",
    )
    return blockers


def _labels_checks(labels: dict, clip: dict, base: Path, *, independent: bool) -> list[str]:
    _require(labels.get("frame_count") == clip["frame_count"], "label frame_count mismatch")
    _require(Fraction(labels.get("fps", "0")) == Fraction(clip["fps"]), "label fps mismatch")
    score_events(
        labels.get("events"),
        [],
        frame_count=clip["frame_count"],
        fps=clip["fps"],
        uncertain_intervals=labels.get("uncertain_intervals", []),
    )
    blockers = []
    if labels.get("exhaustive") is not True:
        blockers.append(f"{clip['id']}:labels_not_exhaustive")
    if independent:
        people = labels.get("annotators", [])
        if (
            len(people) != 2
            or len({p.get("id") for p in people}) != 2
            or not all(p.get("id") and p.get("kind") == "human" for p in people)
            or labels.get("two_independent") is not True
            or labels.get("no_model_outputs_seen") is not True
        ):
            blockers.append(f"{clip['id']}:independent_blind_labels_missing")
        originals = labels.get("originals", [])
        if len(originals) != 2 or not labels.get("adjudication"):
            blockers.append(f"{clip['id']}:original_labels_or_adjudication_missing")
        else:
            for ref in [*originals, labels["adjudication"]]:
                _artifact(ref, base)
        _time(labels.get("locked_at"))
    return blockers


def preflight(manifest: dict, base: Path) -> dict:
    """Validate local evidence only; a recorded remote receipt is not a live server audit."""
    _require(
        type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
        "unsupported manifest schema",
    )
    kind = manifest.get("data_kind")
    _require(kind in {"independent", "synthetic", "development"}, "unknown data_kind")
    clips = manifest.get("clips")
    _require(isinstance(clips, list), "clips must be a list")
    independent = kind == "independent"
    blockers = _freeze_checks(manifest, base) if independent else []
    if not clips:
        blockers.append("no_clips")
    if independent:
        matches = {clip.get("match_id") for clip in clips}
        if (
            len(clips) != 12
            or len(matches) != 3
            or any(
                sum(c.get("match_id") == match and c.get("stratum") == layer for c in clips) != 2
                for match in matches
                for layer in ("event", "uniform")
            )
        ):
            blockers.append("sample_design_incomplete")
    if manifest.get("protocol"):
        _artifact(manifest["protocol"], base)
    for ref in manifest.get("runtime", []):
        _artifact(ref, base)
    exposed = set(manifest.get("exposed_matches", []))
    exposed |= {f"soccernet-gsr:{game}" for game in (2, 3, 4, 5, 6, 9)}
    if independent and manifest.get("exposure_register_complete") is not True:
        blockers.append("exposure_register_incomplete")
    ids: set[str] = set()
    sources: set[str] = set()
    for clip in clips:
        ident = clip.get("id")
        _require(
            isinstance(ident, str) and bool(ident) and ident not in ids, "missing/duplicate clip id"
        )
        ids.add(ident)
        _require(
            isinstance(clip.get("match_id"), str) and bool(clip["match_id"]), "match_id required"
        )
        _require(clip.get("stratum") in {"event", "uniform"}, "unknown stratum")
        if independent:
            _require(clip["match_id"] not in exposed, "match was already exposed")
            _require(
                Fraction(clip["frame_count"]) / Fraction(clip["fps"]) == 30,
                "independent clips must last 30 seconds",
            )
            if clip.get("local_use_rights_confirmed") is not True:
                blockers.append(f"{ident}:local_use_rights_unconfirmed")
            if clip.get("selection_locked_before_predictions") is not True:
                blockers.append(f"{ident}:selection_not_locked")
            if not clip.get("selection_receipt"):
                blockers.append(f"{ident}:selection_receipt_missing")
            else:
                _artifact(clip["selection_receipt"], base)
        video = _artifact(clip.get("video"), base)
        digest = _digest(video)
        _require(digest not in sources, "duplicate canonical video across clips")
        sources.add(digest)
        labels = _json(_artifact(clip.get("labels"), base))
        blockers.extend(_labels_checks(labels, clip, base, independent=independent))
        if independent:
            if not clip.get("frame_map"):
                blockers.append(f"{ident}:frame_alignment_missing")
            else:
                mapping = _json(_artifact(clip["frame_map"], base))
                _require(
                    mapping.get("status") == "passed"
                    and mapping.get("canonical_video_sha256") == digest
                    and mapping.get("frame_count") == clip["frame_count"]
                    and Fraction(mapping.get("fps", "0")) == Fraction(clip["fps"]),
                    "frame mapping verification mismatch",
                )
        outputs = clip.get("outputs", {})
        if outputs.get("status") not in {"passed", "failed"}:
            blockers.append(f"{ident}:model_outputs_missing")
        elif outputs["status"] == "failed":
            _require(bool(outputs.get("failure_reason")), "failed run needs its reason")
        # Hash predictions only after the independently recorded freeze and labels exist.
        elif not blockers:
            for key in ("highlights", "verification", "run"):
                _artifact(outputs.get(key), base)
        if independent and outputs.get("started_at"):
            started = _time(outputs["started_at"])
            _require(_time(labels["locked_at"]) < started, "labels locked after inference")
            if manifest.get("freeze", {}).get("status") == "frozen":
                _require(
                    _time(labels["locked_at"]) <= _time(manifest["freeze"]["pushed_at"]),
                    "labels locked after freeze",
                )
                _require(
                    _time(manifest["freeze"]["pushed_at"]) < started, "inference predates freeze"
                )
        elif independent:
            blockers.append(f"{ident}:inference_timestamp_missing")
    return {
        "status": "blocked" if blockers else "ready",
        "blockers": sorted(set(blockers)),
        "inputs_sha256": input_fingerprint(manifest),
        "data_kind": kind,
        "clips": len(clips),
        "remote_evidence": "local receipt consistency only; no live remote verification",
    }


def _box_iou(a: list[float], b: list[float]) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def _coverage_result(used: int, considered: int, reasons: Counter) -> dict:
    return {
        "used": used,
        "considered": considered,
        "share": _ratio(used, considered),
        "reasons": dict(sorted(reasons.items())),
    }


def score_coverage(
    labels: dict,
    tracks: list[dict] | None,
    lift: list[dict] | None,
    measurements: list[dict] | None,
    *,
    frame_count: int,
) -> dict:
    """Human-visible targets are the denominator, including those with no detection.

    This is output availability, never accuracy. Metric cards are checked on the exact
    human representative frame; output elsewhere does not silently fill that frame.
    """
    if "targets" not in labels:
        return {"status": "not_evaluated_no_targets"}
    if any(value is None for value in (tracks, lift, measurements)):
        return {"status": "not_evaluated_missing_artifacts"}
    expected = set()
    for event in labels["events"]:
        subjects = event.get("subjects")
        _require(
            isinstance(subjects, list) and bool(subjects),
            "every event needs annotated subjects for coverage",
        )
        for subject in subjects:
            _require(isinstance(subject, str) and bool(subject), "invalid subject id")
            expected.update(
                (subject, f)
                for f in range(max(0, event["peak"] - 12), min(frame_count, event["peak"] + 13))
            )
    targets = {}
    for target in labels["targets"]:
        _require(type(target.get("frame")) is int, "target frame must be an integer")
        key = (target.get("subject_id"), target.get("frame"))
        _require(key in expected and key not in targets, "unexpected/duplicate target frame")
        _require(type(target.get("visible")) is bool, "target visibility required")
        if target["visible"]:
            box = target.get("box")
            _require(
                isinstance(box, list)
                and len(box) == 4
                and all(type(v) in {int, float} and math.isfinite(v) for v in box)
                and box[2] > box[0]
                and box[3] > box[1],
                "visible target needs a finite positive box",
            )
        targets[key] = target
    _require(set(targets) == expected, "incomplete manual visibility targets")
    visible = {key: row for key, row in targets.items() if row["visible"]}
    associated = {}
    for frame in sorted({f for _, f in visible}):
        subjects = sorted(subject for subject, f in visible if f == frame)
        candidates = sorted([r for r in tracks if r["frame"] == frame], key=lambda r: r["track_id"])
        _require(
            len({r["track_id"] for r in candidates}) == len(candidates),
            "duplicate predicted track/frame",
        )
        if not candidates:
            continue
        ious = np.array(
            [
                [
                    _box_iou(visible[(s, frame)]["box"], [r[k] for k in ("x1", "y1", "x2", "y2")])
                    for r in candidates
                ]
                for s in subjects
            ]
        )
        weights = np.where(ious >= 0.5, len(subjects) + 1 + ious, 0)
        a, b = linear_sum_assignment(weights, maximize=True)
        for i, j in zip(a, b, strict=True):
            if ious[i, j] >= 0.5:
                associated[(subjects[i], frame)] = candidates[j]
    lifted = {}
    for row in lift:
        key = (row["track_id"], row["frame"])
        _require(key not in lifted, "duplicate lift track/frame")
        lifted[key] = row
    bins = {key: [0, 0, Counter()] for key in ("<80", "80-149", ">=150")}
    used, reasons = 0, Counter()
    for key, target in visible.items():
        track = associated.get(key)
        row = lifted.get((track["track_id"], key[1])) if track else None
        if not track:
            reason = "no_detection"
        elif not row:
            reason = "no_pose_output"
        elif row.get("method") != "height_prior_calibrated" or row.get("reason") != "lifted":
            reason = str(row.get("reason", "not_lifted"))
        else:
            joints = row.get("joints3d", [])
            reason = (
                "used"
                if len(joints) == 78 and all(math.isfinite(v) for v in joints)
                else "invalid_3d_output"
            )
        height = target["box"][3] - target["box"][1]
        bucket = bins["<80" if height < 80 else "80-149" if height < 150 else ">=150"]
        bucket[1] += 1
        if reason == "used":
            used += 1
            bucket[0] += 1
        else:
            reasons[reason] += 1
            bucket[2][reason] += 1
    metric_results = {}
    for metric in ("positions-trajectories", "peak-speed", "nearest-opponent-distance"):
        n_used, n, omitted, missing = 0, 0, 0, Counter()
        for event in labels["events"]:
            applies = event.get("metric_applicability", {}).get(
                metric, metric != "nearest-opponent-distance"
            )
            for subject in event["subjects"]:
                if applies is not True:
                    omitted += 1
                    continue
                n += 1
                track = associated.get((subject, event["peak"]))
                if track is None:
                    missing["no_detection_or_not_visible"] += 1
                    continue
                if metric == "positions-trajectories":
                    valid = track.get("pitch_usable") and all(
                        math.isfinite(track.get(k, math.nan)) for k in ("pitch_x", "pitch_y")
                    )
                    if valid:
                        n_used += 1
                    else:
                        missing["no_usable_pitch_position"] += 1
                else:
                    records = [
                        r
                        for r in measurements
                        if r.get("metric_id") == metric
                        and r.get("frame") == event["peak"]
                        and r.get("track_id") == track["track_id"]
                    ]
                    if any(
                        r.get("status") == "ok"
                        and isinstance(r.get("value"), int | float)
                        and math.isfinite(r["value"])
                        for r in records
                    ):
                        n_used += 1
                    else:
                        missing[
                            str(records[0].get("reasonCode", "unavailable"))
                            if records
                            else "not_reported_at_reference_frame"
                        ] += 1
        metric_results[metric] = _coverage_result(n_used, n, missing) | {
            "not_applicable_or_unchecked": omitted
        }
    return {
        "status": "evaluated_output_availability_only",
        "three_d": _coverage_result(used, len(visible), reasons)
        | {
            "by_box_height": {key: _coverage_result(*values) for key, values in bins.items()},
            "meaning": "finite lifted skeleton output, not certified 3D or technique accuracy",
        },
        "metrics": metric_results,
    }


def _output_rows(ref: dict | None, base: Path, *, lines: bool = False) -> list[dict] | None:
    if ref is None:
        return None
    path = _artifact(ref, base)
    if lines:
        rows = [
            _decode(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        _require(all(isinstance(row, dict) for row in rows), "measurement rows must be objects")
        return rows
    import pandas as pd

    return pd.read_parquet(path).to_dict("records")


def evaluate(manifest: dict, base: Path) -> dict:
    """Read native gbf output artifacts after a successful preflight."""
    clips = []
    for clip in manifest["clips"]:
        labels = _json(_artifact(clip["labels"], base))
        outputs = clip["outputs"]
        run_ok = outputs["status"] == "passed"
        predictions = []
        if run_ok:
            doc = _json(_artifact(outputs["highlights"], base))
            verification = _json(_artifact(outputs["verification"], base))
            run = _json(_artifact(outputs["run"], base))
            _require(
                doc.get("frame_count") == clip["frame_count"]
                and Fraction(doc.get("fps", "0")) == Fraction(clip["fps"]),
                "prediction frame_count/fps mismatch",
            )
            _require(
                verification.get("status") == "passed"
                and verification.get("source_frames") == clip["frame_count"],
                "operational verification mismatch",
            )
            _require(
                any(
                    r.get("name") == "video" and r.get("sha256") == clip["video"]["sha256"]
                    for r in run.get("inputs", [])
                ),
                "run video hash mismatch",
            )
            expected_commit = (
                manifest["freeze"]["commit"]
                if manifest["data_kind"] == "independent"
                else manifest.get("core_commit")
            )
            _require(
                run.get("code", {}).get("git_commit") == expected_commit
                and run["code"].get("git_dirty") is False,
                "run core provenance mismatch",
            )
            predictions = doc["highlights"]
            artifacts = [
                _output_rows(outputs.get(k), base, lines=k == "measurements")
                for k in ("tracks", "lift", "measurements")
            ]
        else:
            artifacts = [[], [], []]
        kwargs = {
            "frame_count": clip["frame_count"],
            "fps": clip["fps"],
            "uncertain_intervals": labels.get("uncertain_intervals", []),
        }
        scores = score_events(labels["events"], predictions, **kwargs)
        main_scores = score_events(
            labels["events"], [p for p in predictions if p.get("tier") == "main"], **kwargs
        )
        clips.append(
            {
                "id": clip["id"],
                "match_id": clip["match_id"],
                "stratum": clip["stratum"],
                "run_ok": run_ok,
                "failure_reason": outputs.get("failure_reason"),
                "scores": scores,
                "main_only": main_scores,
                "coverage": score_coverage(labels, *artifacts, frame_count=clip["frame_count"]),
            }
        )
    summary = summarize(clips, data_kind=manifest["data_kind"])
    return {
        "schema_version": 1,
        "data_kind": manifest["data_kind"],
        "inputs_sha256": input_fingerprint(manifest),
        "restrictions": ["no_share"],
        "summary": summary,
        "clips": clips,
        "provenance_limit": (
            "hashes and recorded attestations checked locally; no independent live audit "
            "of rights, annotator identity or remote server time"
        ),
    }


def _summary_zh(report: dict) -> str:
    summary = report["summary"]
    counts = summary["counts"]
    return (
        "# 产品验收结果\n\n"
        f"状态：`{summary['status']}`；数据：`{report['data_kind']}`。\n\n"
        f"TP {counts['tp']}；FP {counts['fp']}；FN {counts['fn']}；"
        f"不可评分预测 {counts['unscorable']}。\n\n"
        f"保守精度：{counts['precision']}；召回率：{counts['recall']}。零分母写 null。\n\n"
        "逐片段、逐比赛、分层、分类计数和缺失原因见 results.json；"
        "事件富集样本不能代表自然场景。\n\n"
        "位置/速度真实误差：未评估，缺少独立同步真值。3D 准确度：未评估；有输出不代表准确。\n\n"
        "合成/开发材料只检查计分器，不认证产品。冻结来源仅核对本地回执，未联网独立审计。\n"
    )


def _write(out: Path, name: str, document: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / name).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m gbf eval-product")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = _json(args.manifest)
        readiness = preflight(manifest, args.manifest.parent)
        _write(args.out, "preflight.json", readiness)
        if readiness["status"] != "ready":
            print("eval-product: blocked: " + ", ".join(readiness["blockers"]), file=sys.stderr)
            return 3
        if args.preflight:
            return 0
        report = evaluate(manifest, args.manifest.parent)
        _write(args.out, "results.json", report)
        with (args.out / "summary-zh.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_summary_zh(report))
        return {"passed": 0, "diagnostic_only": 0, "failed": 4, "insufficient_evidence": 5}[
            report["summary"]["status"]
        ]
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        print(f"eval-product: {exc}", file=sys.stderr)
        return 2
