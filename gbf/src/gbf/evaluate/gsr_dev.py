"""gates.v1 GSR evaluation of gbf on the dev clips (prereg/eval-gsr-dev-v1.md).

Runs the frozen per-clip evaluation of :mod:`gbf.evaluate.gsr` unchanged on every clip and adds
what the dev protocol adds: the reprojection error R of gbf's calibration against the annotated
pitch lines, the nearest-opponent comparison N, the reference jump windows of
prereg/README.md P3 (a secondary result), and statistics pooled over clips with intervals
that resample matches. Results are deviations from the GSR annotation (the reference).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gbf.evaluate import gsr, pitch_lines

JUMP_STEP_M = 2.0
JUMP_MIN_PEOPLE = 5
JUMP_HALF_WINDOW = 15
"""prereg/README.md P3: a jump frame and its window."""

MIN_ROWS_FOR_SUMMARY_N = 10
"""Measure N gets quantiles and intervals only from this many values on."""

POOLED = (
    "p_all",
    "p_outside_jumps",
    "s1_all",
    "s1_outside_jumps",
    "s2_all",
    "s2_outside_jumps",
    "r_px",
)
"""Pooled measures: P-primary |d| (m), |S1| and |S2| (m/s), each over all frames and outside
the P3 jump windows, and R (px)."""

TRACKEVAL_DEV_RUNNER = gsr.REPO_ROOT / "tools" / "run_trackeval_dev.py"


# --------------------------------------------------------------------------- jump windows (P3)


def jump_frames(
    people: pd.DataFrame, step_m: float = JUMP_STEP_M, min_people: int = JUMP_MIN_PEOPLE
) -> list[int]:
    """Frames f on which at least ``min_people`` reference people also appear on f - 1 with
    finite raw positions on both, and the median of their |bbox_pitch_raw(f) -
    bbox_pitch_raw(f - 1)| exceeds ``step_m``. Reads the reference only."""
    table = people.sort_values(["ref_id", "frame"])
    by_person = table.groupby("ref_id")
    step = np.hypot(by_person["pitch_x_raw"].diff(), by_person["pitch_y_raw"].diff())
    follows = by_person["frame"].diff() == 1
    steps = pd.DataFrame({"frame": table["frame"], "step": step})[follows & step.notna()]
    per_frame = steps.groupby("frame")["step"].agg(["median", "size"])
    hit = per_frame[(per_frame["size"] >= min_people) & (per_frame["median"] > step_m)]
    return sorted(int(frame) for frame in hit.index)


def jump_windows(
    jumps: Iterable[int], n_frames: int, half: int = JUMP_HALF_WINDOW
) -> frozenset[int]:
    """Frames f - half .. f + half of every jump frame f, inside 0 .. n_frames - 1."""
    return frozenset(
        f for jump in jumps for f in range(jump - half, jump + half + 1) if 0 <= f < n_frames
    )


# --------------------------------------------------------------------------- reprojection (R)


def project_ground(points: np.ndarray, K: Any, R: Any, C: Any) -> np.ndarray:
    """Pixels (n, 2) of ground points (n, 2) at z = 0 through x ~ K R (X - C); NaN behind
    the camera."""
    world = np.column_stack([np.asarray(points, dtype=float), np.zeros(len(points))])
    camera = (
        np.asarray(R, dtype=float).reshape(3, 3)
        @ (world - np.asarray(C, dtype=float).reshape(1, 3)).T
    )
    image = np.asarray(K, dtype=float).reshape(3, 3) @ camera
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (image[:2] / image[2]).T
    uv[camera[2] <= 0] = np.nan
    return uv


def _frame_of(document: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(image["image_id"]): int(Path(image["file_name"]).stem) - 1
        for image in document["images"]
    }


def reprojection_rows(
    document: Mapping[str, Any], calib: pd.DataFrame, width: int, height: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Measure R: for every annotated point of a ground line on a frame whose calibration is
    pitch_usable, the pixel distance to the nearest sample of that line projected with gbf's
    calibration. Returns the rows (frame, line, distance_px) and the counts of points left
    out (frame not usable, line not in the ground model, line behind the camera)."""
    frame_of = _frame_of(document)
    model = pitch_lines.ground_lines()
    cameras = {
        int(row.frame_index): row for row in calib.itertuples(index=False) if bool(row.pitch_usable)
    }
    rows: list[dict[str, Any]] = []
    skipped = {"frame_not_usable": 0, "line_not_on_ground_model": 0, "line_not_projected": 0}
    for annotation in document["annotations"]:
        if annotation.get("supercategory") != "pitch":
            continue
        frame = frame_of.get(str(annotation.get("image_id")))
        for name, points in (annotation.get("lines") or {}).items():
            if frame not in cameras:
                skipped["frame_not_usable"] += len(points)
                continue
            if name not in model:
                skipped["line_not_on_ground_model"] += len(points)
                continue
            camera = cameras[frame]
            projected = project_ground(model[name], camera.K, camera.R, camera.C)
            projected = projected[np.isfinite(projected).all(axis=1)]
            if not len(projected):
                skipped["line_not_projected"] += len(points)
                continue
            for point in points:
                u, v = float(point["x"]) * width, float(point["y"]) * height
                distance = np.hypot(projected[:, 0] - u, projected[:, 1] - v).min()
                rows.append({"frame": frame, "line": name, "distance_px": float(distance)})
    return pd.DataFrame(rows, columns=["frame", "line", "distance_px"]), skipped


# --------------------------------------------------------------------------- nearest opponent (N)

TEAMS = ("left", "right")


def reference_teams(document: Mapping[str, Any]) -> pd.DataFrame:
    """frame, ref_id and team ("left", "right" or None, as the file writes it) of every person
    row of a GSR label document."""
    frame_of = _frame_of(document)
    rows = [
        {
            "frame": frame_of[str(row["image_id"])],
            "ref_id": int(row["track_id"]),
            "team": (row.get("attributes") or {}).get("team"),
        }
        for row in document["annotations"]
        if row.get("supercategory") == "object"
        and (row.get("attributes") or {}).get("role") in gsr.PERSON_ROLES
        and str(row.get("image_id")) in frame_of
    ]
    return pd.DataFrame(rows, columns=["frame", "ref_id", "team"])


def reference_nearest_opponent(
    people: pd.DataFrame, teams: pd.DataFrame, ref_id: int, frames: Iterable[int]
) -> float | None:
    """Median over ``frames`` of the ground distance (bbox_pitch) from reference person
    ``ref_id`` to the nearest reference outfield player (role player) of the other team. A
    frame counts only when the person's team is known and both positions are finite; None
    when no frame counts."""
    table = people.merge(teams, on=["frame", "ref_id"], how="left")
    distances: list[float] = []
    for frame in frames:
        here = table[table["frame"] == frame]
        me = here[here["ref_id"] == ref_id]
        if me.empty or me["team"].iloc[0] not in TEAMS:
            continue
        mine = me[["pitch_x", "pitch_y"]].to_numpy(dtype=float)[0]
        other_team = TEAMS[1 - TEAMS.index(me["team"].iloc[0])]
        opponents = here[(here["role"] == "player") & (here["team"] == other_team)]
        points = opponents[["pitch_x", "pitch_y"]].to_numpy(dtype=float)
        points = points[np.isfinite(points).all(axis=1)]
        if not np.isfinite(mine).all() or not len(points):
            continue
        distances.append(float(np.hypot(*(points - mine).T).min()))
    return float(np.median(distances)) if distances else None


def nearest_opponent_rows(
    measurements: Iterable[Mapping[str, Any]],
    pairs: pd.DataFrame,
    people: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    """Measure N: every nearest-opponent-distance value gbf showed, against the same quantity
    of the reference person M pairs with gbf's track on the same frames (None when the track
    pairs with no single reference person there)."""
    rows = []
    for shown in measurements:
        if shown.get("metric_id") != "nearest-opponent-distance" or shown.get("value") is None:
            continue
        frames = [int(f) for f in shown.get("frames") or []]
        track = int(shown["track_id"])
        hit = pairs[(pairs["track_id"] == track) & pairs["frame"].isin(frames)]
        refs = sorted({int(r) for r in hit["ref_id"]})
        single = refs[0] if len(refs) == 1 else None
        reference = (
            None if single is None else reference_nearest_opponent(people, teams, single, frames)
        )
        rows.append(
            {
                "highlight_id": shown.get("highlight_id"),
                "track_id": track,
                "frames": frames,
                "ref_id": single,
                "gbf_m": float(shown["value"]),
                "reference_m": reference,
                "deviation_m": None if reference is None else float(shown["value"]) - reference,
            }
        )
    columns = ["highlight_id", "track_id", "frames", "ref_id", "gbf_m", "reference_m"]
    return pd.DataFrame(rows, columns=[*columns, "deviation_m"])


# --------------------------------------------------------------------------- pooling


def pooled(values_by_clip: Mapping[str, Any], game_of: Mapping[str, str | None]) -> dict[str, Any]:
    """gates.v1 reporting over clips: n, median, P90 and mean of all finite values, every
    clip's own quantiles, and cluster-bootstrap 95 % intervals of the median and P90 that
    resample matches (game_id), or clips when some clip has no game_id (``clusters`` says
    which). Quantiles and the bootstrap are the frozen ones of :mod:`gbf.evaluate.gsr`."""
    arrays = {clip: np.asarray(v, dtype=float).reshape(-1) for clip, v in values_by_clip.items()}
    everything = np.concatenate(list(arrays.values())) if arrays else np.array([])
    out = gsr._quantiles(everything)
    by_match = all(game_of.get(clip) is not None for clip in arrays)
    clusters: dict[int, list[np.ndarray]] = {}
    for index, (clip, values) in enumerate(sorted(arrays.items())):
        key = int(game_of[clip]) if by_match else index
        clusters.setdefault(key, []).append(values)
    groups = {key: np.concatenate(parts) for key, parts in clusters.items()}
    out["clusters"] = "match" if by_match else "clip"
    out["n_clusters"] = sum(bool(np.isfinite(v).any()) for v in groups.values())
    out["median_ci95"] = gsr.cluster_bootstrap(groups, np.median) if out["n"] else None
    out["p90_ci95"] = gsr.cluster_bootstrap(groups, gsr._p90) if out["n"] else None
    out["per_clip"] = {clip: gsr._quantiles(values) for clip, values in sorted(arrays.items())}
    return out


def strata(positions: pd.DataFrame) -> dict[str, Any]:
    """P-primary over all clips by distance-to-camera terciles (cut points from the pooled
    pairs) and by the depth direction, as gates.v1 reporting asks."""
    near, far = gsr.terciles(positions["camera_distance"].to_numpy())
    distance = positions["camera_distance"].to_numpy(dtype=float)
    bins = {
        "near": positions[distance <= near],
        "middle": positions[(distance > near) & (distance <= far)],
        "far": positions[distance > far],
    }
    return {
        "cut_points_m": [near, far],
        **{name: gsr._quantiles(rows["deviation"]) for name, rows in bins.items()},
        **{
            part: {
                "signed_median": gsr._signed_median(positions[part]),
                "abs": gsr._quantiles(np.abs(positions[part].to_numpy(dtype=float))),
            }
            for part in ("radial", "lateral")
        },
    }


# --------------------------------------------------------------------------- one clip


def clip_values(seq: str, labels: Path, run: Path, alignment: Mapping[str, Any]) -> dict[str, Any]:
    """One dev clip: the frozen per-clip evaluation (gsr.evaluate, with its section 8 stop
    rules) and, when it did not stop, the values the pooled statistics need, R, N and the
    P3 jump windows."""
    per_clip = gsr.evaluate(labels, run, alignment)
    document = json.loads(Path(labels).read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "seq": seq,
        "game_id": (document.get("info") or {}).get("game_id"),
        "per_clip": per_clip,
    }
    if per_clip["stopped"]:
        return out
    reference = gsr.load_reference(labels)
    refine = gsr.load_tracks(run / "refine" / "tracks.parquet")
    calib = pd.read_parquet(run / "refine" / "calib.parquet")
    pairs, _ = gsr.match(reference, refine)
    positions = gsr.position_rows(reference, refine, pairs, gsr.camera_ground_points(calib))
    usable = {
        int(f) for f, u in zip(calib["frame_index"], calib["pitch_usable"], strict=True) if bool(u)
    }
    kin = gsr.reference_kinematics(reference.people)
    frames = gsr.speed_frames(kin, pairs, pd.read_parquet(run / "kinematics.parquet"), usable)
    jumps = jump_frames(reference.people)
    windows = jump_windows(jumps, reference.n_frames)
    outside = frames.assign(in_f=frames["in_f"] & ~frames["frame"].isin(sorted(windows)))
    values: dict[str, np.ndarray] = {}
    for name, table in (("all", frames), ("outside_jumps", outside)):
        peaks = gsr.segment_peaks(table)
        comparable = peaks[peaks["comparable"]]
        on_f = table[table["in_f"] & table["segment"].isin(comparable["segment"])]
        values[f"s1_{name}"] = np.abs(comparable["deviation"].to_numpy(dtype=float))
        values[f"s2_{name}"] = np.abs((on_f["pred_speed"] - on_f["speed"]).to_numpy(dtype=float))
    keep = ~positions["frame"].isin(sorted(windows))
    values["p_all"] = positions["deviation"].to_numpy(dtype=float)
    values["p_outside_jumps"] = positions.loc[keep, "deviation"].to_numpy(dtype=float)
    reprojection, skipped = reprojection_rows(document, calib, reference.width, reference.height)
    values["r_px"] = reprojection["distance_px"].to_numpy(dtype=float)
    lines = (run / "measurements.jsonl").read_text(encoding="utf-8").splitlines()
    shown = [json.loads(line) for line in lines if line.strip()]
    nearest = nearest_opponent_rows(shown, pairs, reference.people, reference_teams(document))
    out["values"] = values
    out["positions"] = positions
    out["jumps"] = {
        "frames": jumps,
        "window_frames": len(windows),
        "share": len(windows) / reference.n_frames,
    }
    out["reprojection_skipped"] = skipped
    out["nearest_opponent"] = nearest.to_dict("records")
    return out


def secondary_equals_primary(summary: Mapping[str, Any]) -> bool | None:
    """Whether every number TrackEval reports for T-secondary (gbf-import) equals T-primary's
    (gbf-refine), per sequence and combined; None when either is missing."""
    trackers = summary.get("trackers") or {}
    primary = trackers.get(gsr.TRACKERS["refine"])
    secondary = trackers.get(gsr.TRACKERS["import"])
    if primary is None or secondary is None:
        return None
    return {k: v for k, v in primary.items() if k != "message"} == {
        k: v for k, v in secondary.items() if k != "message"
    }


def nearest_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure N over clips: counts, and quantiles of |deviation| only from
    MIN_ROWS_FOR_SUMMARY_N values on (N is reported value by value)."""
    deviations = [r["deviation_m"] for r in rows if r.get("deviation_m") is not None]
    out: dict[str, Any] = {"shown": len(rows), "compared": len(deviations)}
    if len(deviations) >= MIN_ROWS_FOR_SUMMARY_N:
        out["abs"] = gsr._quantiles(np.abs(np.asarray(deviations, dtype=float)))
        out["signed_median"] = gsr._signed_median(deviations)
    return out


# --------------------------------------------------------------------------- command line


def main(argv: list[str]) -> int:
    """``python -m gbf eval-gsr-dev --manifest M --out O [--trackeval-python P]``; the manifest
    lists ``{"clips": [{"seq", "labels", "run", "video", "images"}]}``. Writes
    ``O/results.json`` and the TrackEval files; returns 0, 2 for unusable input or 4 when
    TrackEval fails."""
    parser = argparse.ArgumentParser(
        prog="python -m gbf eval-gsr-dev",
        description="Deviation of gbf runs on the GSR dev clips (prereg/eval-gsr-dev-v1.md).",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--align-frames", default=",".join(str(k) for k in gsr.ALIGN_FRAMES))
    parser.add_argument(
        "--trackeval-python",
        type=Path,
        help="interpreter with sn-trackeval installed; without it measure T is not run",
    )
    parser.add_argument("--trackeval-runner", type=Path, default=TRACKEVAL_DEV_RUNNER)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        clips = json.loads(args.manifest.read_text(encoding="utf-8"))["clips"]
        frames = [int(part) for part in args.align_frames.split(",") if part.strip()]
        results = []
        for clip in clips:
            video, images = Path(clip["video"]), Path(clip["images"])
            decoded, sources = gsr.load_alignment_frames(video, images, frames)
            alignment = gsr.frame_alignment(decoded, sources, frames)
            result = clip_values(clip["seq"], Path(clip["labels"]), Path(clip["run"]), alignment)
            result["per_clip"]["inputs"]["video"] = {"sha256": gsr._sha256(video)}
            results.append(result)
    except (gsr.GsrEvalError, OSError, KeyError, ValueError) as exc:
        print(f"eval-gsr-dev: {exc}", file=sys.stderr)
        return 2
    kept = [r for r in results if not r["per_clip"]["stopped"]]
    games = {r["seq"]: r["game_id"] for r in kept}
    nearest = [row | {"seq": r["seq"]} for r in kept for row in r["nearest_opponent"]]
    report: dict[str, Any] = {
        "protocol": "prereg/eval-gsr-dev-v1.md",
        "restrictions": list(gsr.RESTRICTIONS),
        "wording": "deviation from the GSR annotation (reference), never error",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "clips_in_order": [r["seq"] for r in results],
        "stopped_clips": [r["seq"] for r in results if r["per_clip"]["stopped"]],
        "clips": {
            r["seq"]: {k: v for k, v in r.items() if k not in ("values", "positions")}
            for r in results
        },
        "pooled": {
            name: pooled({r["seq"]: r["values"][name] for r in kept}, games) for name in POOLED
        },
        "pooled_position_strata": strata(pd.concat([r["positions"] for r in kept]))
        if kept
        else None,
        "nearest_opponent": {"summary": nearest_summary(nearest), "rows": nearest},
        "deviations": [],
    }
    code = 0
    root = args.out / "trackeval"
    for r in kept:
        clip = next(c for c in clips if c["seq"] == r["seq"])
        reference = gsr.load_reference(Path(clip["labels"]))
        for name, tracker in gsr.TRACKERS.items():
            tracks = gsr.load_tracks(Path(clip["run"]) / name / "tracks.parquet")
            gsr.write_mot(reference, tracks, root, tracker, r["seq"])
    report["tracking"] = None
    if kept and args.trackeval_python is not None:
        seq_info = root / "seq_info.json"
        lengths = {r["seq"]: r["per_clip"]["checks"]["reference_frames"] for r in kept}
        seq_info.write_text(json.dumps(lengths), encoding="utf-8")
        try:
            summary = gsr.run_trackeval(
                args.trackeval_python, args.trackeval_runner, root, str(seq_info), 0
            )
            report["tracking"] = summary | {
                "runner_sha256": gsr._sha256(args.trackeval_runner),
                "t_secondary_equals_t_primary": secondary_equals_primary(summary),
            }
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", None) or ""
            report["tracking"] = {
                "error": str(exc),
                "stderr_tail": stderr.strip().splitlines()[-5:],
            }
            print(f"eval-gsr-dev: TrackEval failed: {exc}", file=sys.stderr)
            code = 4
    args.out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(gsr._jsonable(report), ensure_ascii=False, indent=2, allow_nan=False)
    (args.out / "results.json").write_text(text + "\n", encoding="utf-8")
    for seq in report["stopped_clips"]:
        print(f"eval-gsr-dev: STOP {seq}", file=sys.stderr)
    return code
