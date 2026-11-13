"""End-to-end test of gbf.pipeline on a tiny SYNTHETIC clip (no private data, no network).

The clip is generated here: 100 frames of 960x540 at 25 fps with a hard cut at frame 60.
Frames 0-59 are a wide shot from the synthetic main camera (gbf.geometry) with two teams
drawn as boxes on the grass; the striker (track 1, team left, attacking +x) dribbles and
kicks the ball towards goal at frame KICK. Frames 60-99 are an uncalibrated close-up of one
player. The audio is a quiet tone with a loud burst from 1.6 s to 2.8 s. The plugin export
(detections, calibration, receipt) and the ball candidates are written from the same truth,
so every stage sees a consistent clip. The pose network is replaced by a fake runner that
puts a standing skeleton into each box.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from gbf import contracts
from gbf import measure as measure_module
from gbf import pipeline as pipeline_module
from gbf import render as render_module
from gbf.audio import decode_mono
from gbf.geometry import Camera, project, synthetic_broadcast_camera
from gbf.ingest import probe
from gbf.kinematics import REJECTION_TESTS
from gbf.measure import DEFAULT_PARAMS, load_metric_registry, metric_label, read_measurements
from gbf.pipeline import (
    STAGES,
    PipelineConfig,
    PipelineError,
    _card_rows,
    _moment_rows,
    check_out_dir,
    display_record,
    main,
    read_restrictions,
    run_pipeline,
    summary_text,
)
from gbf.pose import HALPE26_KEYPOINTS, PoseEstimator
from gbf.render import FrameAnnotations, InfoCard, Style, draw_annotations, plan_highlight_edit
from gbf.sngs_import import load_tracks
from gbf.synth.video import write_frames_mp4
from gbf.util.hashing import sha256_bytes, sha256_file
from gbf.util.io import read_json
from gbf.util.paths import repo_root

WIDTH, HEIGHT = 960, 540
N_FRAMES = 100
CUT = 60
KICK = 40
FPS = "25/1"
PLUGIN_TAGS = ["hl3"]
TEAM_NAMES = {"left": "浅蓝队", "right": "深蓝队"}
AUDIO_FRAMES = 95

GRASS = (60, 128, 60)
STANDS = (70, 70, 110)
KIT = {"left": (120, 180, 240), "right": (20, 40, 100)}
KEEPER = (240, 150, 20)
BALL_WHITE = (245, 245, 245)

# track_id: (team, role, start (x, y) metres, velocity (vx, vy) metres per frame)
PLAYERS: dict[int, tuple[str, str, tuple[float, float], tuple[float, float]]] = {
    1: ("left", "player", (30.0, 3.0), (0.22, 0.0)),
    2: ("left", "player", (33.0, -6.0), (0.18, 0.02)),
    3: ("left", "player", (35.0, 10.0), (0.12, -0.05)),
    11: ("right", "player", (38.5, 1.0), (-0.04, 0.02)),
    12: ("right", "player", (41.0, -4.0), (0.05, 0.0)),
    13: ("right", "goalkeeper", (50.5, 0.0), (0.0, 0.01)),
}
STRIKER = 1
CLOSE_UP_TRACK = 31
BALL_LEAD_M = 0.3
KICK_SPEED_MPF = 0.8  # 20 m/s towards the goal


def _camera() -> Camera:
    return synthetic_broadcast_camera(WIDTH, HEIGHT, target=(40.0, 0.0, 0.0))


def _position(track: int, frame: int) -> tuple[float, float]:
    _team, _role, (x0, y0), (vx, vy) = PLAYERS[track]
    if track == STRIKER and frame > KICK:
        x_kick = x0 + vx * KICK
        return x_kick + 0.08 * (frame - KICK), y0
    return x0 + vx * frame, y0 + vy * frame


def _ball(frame: int) -> tuple[float, float]:
    sx, sy = _position(STRIKER, min(frame, KICK))
    start = (sx + BALL_LEAD_M, sy)
    if frame <= KICK:
        return start
    goal = (52.5, 1.0)
    direction = np.array(goal) - np.array(start)
    direction /= np.linalg.norm(direction)
    travel = KICK_SPEED_MPF * (frame - KICK)
    return start[0] + direction[0] * travel, start[1] + direction[1] * travel


def _pixel(camera: Camera, point: tuple[float, float, float]) -> tuple[float, float]:
    uv, _depth, valid = project(camera, list(point))
    assert bool(valid)
    return float(uv[0]), float(uv[1])


def _player_box(camera: Camera, x: float, y: float) -> list[float]:
    foot = _pixel(camera, (x, y, 0.0))
    head = _pixel(camera, (x, y, -1.8))
    height = foot[1] - head[1]
    half = 0.22 * height
    return [foot[0] - half, head[1], foot[0] + half, foot[1]]


def _ball_box(camera: Camera, x: float, y: float) -> list[float]:
    centre = _pixel(camera, (x, y, -0.11))
    top = _pixel(camera, (x, y, -0.33))
    radius = 0.5 * 1.2 * (centre[1] - top[1])  # detector boxes are about 1.2x the ball
    return [centre[0] - radius, centre[1] - radius, centre[0] + radius, centre[1] + radius]


CLOSE_BOX = [380.0, 140.0, 520.0, 520.0]


def _frames(camera: Camera) -> list[np.ndarray]:
    frames = []
    for frame in range(N_FRAMES):
        image = np.empty((HEIGHT, WIDTH, 3), np.uint8)
        if frame < CUT:
            image[:] = GRASS
            for track, (team, role, _start, _velocity) in PLAYERS.items():
                x1, y1, x2, y2 = (
                    int(round(v)) for v in _player_box(camera, *_position(track, frame))
                )
                image[y1:y2, x1:x2] = KEEPER if role == "goalkeeper" else KIT[team]
            bx1, by1, bx2, by2 = (int(round(v)) for v in _ball_box(camera, *_ball(frame)))
            image[by1:by2, bx1:bx2] = BALL_WHITE
        else:
            image[:] = STANDS
            x1, y1, x2, y2 = (int(v) for v in CLOSE_BOX)
            image[y1:y2, x1:x2] = KIT["left"]
        frames.append(image)
    return frames


def _detections(camera: Camera) -> list[dict[str, Any]]:
    rows = []
    for frame in range(N_FRAMES):
        if frame < CUT:
            for track, (team, role, _start, _velocity) in PLAYERS.items():
                box = _player_box(camera, *_position(track, frame))
                rows.append(_det(frame, track, box, role, team))
        else:
            rows.append(_det(frame, CLOSE_UP_TRACK, CLOSE_BOX, "player", "left"))
    return rows


def _det(frame: int, track: int, box: list[float], role: str, team: str) -> dict[str, Any]:
    return {
        "frame": frame,
        "state_frame": frame + 1,
        "track_id": track,
        "box": [round(v, 3) for v in box],
        "kf_box": [round(v, 3) for v in box],
        "conf": 0.9,
        "role": role,
        "team": team,
        "team_cluster": 0.0 if team == "left" else 1.0,
        "jersey": None,
        "jersey_conf": None,
    }


def _calib(camera: Camera) -> list[dict[str, Any]]:
    rows = []
    for frame in range(N_FRAMES):
        row: dict[str, Any] = {
            "frame": frame,
            "state_frame": frame + 1,
            "calib_status": "fresh" if frame < CUT else "none",
            "calib_age": 0,
            "n_keypoints": 12 if frame < CUT else 0,
            "n_line_classes": 5 if frame < CUT else 0,
        }
        if frame < CUT:
            row.update(
                {
                    "K": camera.K.ravel().tolist(),
                    "R": camera.R.ravel().tolist(),
                    "C": camera.C.tolist(),
                    "radial_distortion": [0.0] * 6,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "rep_err_px": 2.0,
                    "rep_err_n_keypoints": 10,
                }
            )
        rows.append(row)
    return rows


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")


def _write_wav(path: Path, seconds: float, rate: int = 48000) -> None:
    t = np.arange(int(round(seconds * rate))) / rate
    signal = 0.01 * np.sin(2 * np.pi * 300.0 * t)
    burst = (t >= 1.6) & (t < 2.8)
    signal[burst] = 0.5 * np.sin(2 * np.pi * 1000.0 * t[burst])
    pcm = np.round(np.clip(signal, -1, 1) * 32767).astype("<i2")
    stereo = np.stack([pcm, pcm], axis=1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(stereo.tobytes())


def make_inputs(folder: Path) -> dict[str, Path]:
    """Write the synthetic clip, its audio, plugin export and ball candidates."""
    camera = _camera()
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / "合成 片段.mp4"
    write_frames_mp4(_frames(camera), video, fps=FPS)
    audio = folder / "原声 audio.wav"
    _write_wav(audio, AUDIO_FRAMES / 25)  # the sound ends before the picture
    sngs = folder / "sngs export"
    sngs.mkdir()
    files = {"detections.jsonl": _jsonl(_detections(camera)), "calib.jsonl": _jsonl(_calib(camera))}
    for name, data in files.items():
        (sngs / name).write_bytes(data)
    receipt = {
        "plugin": "plugin-sngs-export",
        "version": "0.1.0",
        "license": "GPL-3.0-only",
        "outputs": [
            {"name": name, "sha256": sha256_bytes(data), "bytes": len(data)}
            for name, data in files.items()
        ],
        "restrictions": PLUGIN_TAGS,
        "video_frames": N_FRAMES,
    }
    (sngs / "plugin-receipt.json").write_bytes(json.dumps(receipt).encode("utf-8"))
    lines = []
    for frame in range(N_FRAMES):
        cands = []
        if frame < CUT:
            box = _ball_box(camera, *_ball(frame))
            cands = [[round(v, 2) for v in box] + [0.85]]
        lines.append({"frame": frame, "cands": cands})
    ball = folder / "ball_raw.jsonl"
    ball.write_bytes(_jsonl(lines)[:-1])  # the last line is still being written
    return {"video": video, "audio": audio, "sngs": sngs, "ball": ball}


# Standing figure inside a box: (fraction across from the left edge, fraction down from the
# top) per Halpe-26 keypoint; the player faces the camera (their left on the image right).
_TEMPLATE = {
    "nose": (0.5, 0.07),
    "left_eye": (0.54, 0.055),
    "right_eye": (0.46, 0.055),
    "left_ear": (0.58, 0.07),
    "right_ear": (0.42, 0.07),
    "left_shoulder": (0.7, 0.2),
    "right_shoulder": (0.3, 0.2),
    "left_elbow": (0.76, 0.36),
    "right_elbow": (0.24, 0.36),
    "left_wrist": (0.78, 0.49),
    "right_wrist": (0.22, 0.49),
    "left_hip": (0.6, 0.5),
    "right_hip": (0.4, 0.5),
    "left_knee": (0.6, 0.72),
    "right_knee": (0.4, 0.72),
    "left_ankle": (0.6, 0.94),
    "right_ankle": (0.4, 0.94),
    "head": (0.5, 0.0),
    "neck": (0.5, 0.15),
    "hip": (0.5, 0.5),
    "left_big_toe": (0.66, 0.995),
    "right_big_toe": (0.34, 0.995),
    "left_small_toe": (0.7, 0.99),
    "right_small_toe": (0.3, 0.99),
    "left_heel": (0.58, 0.98),
    "right_heel": (0.42, 0.98),
}


class TemplateRunner:
    """Fake PoseRunner: a standing skeleton scaled into every box, score 0.9."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        fractions = np.array([_TEMPLATE[name] for name in HALPE26_KEYPOINTS])
        points = []
        for x1, y1, x2, y2 in boxes:
            u = x1 + fractions[:, 0] * (x2 - x1)
            v = y1 + fractions[:, 1] * (y2 - y1)
            points.append(np.stack([u, v], axis=1))
        keypoints = np.stack(points)
        return keypoints, np.full(keypoints.shape[:2], 0.9)


def _estimator() -> PoseEstimator:
    return PoseEstimator(
        "fake-template.onnx", (192, 256), model_id="fake-standing-template", runner=TemplateRunner()
    )


def _config(inputs: dict[str, Path], out: Path, base: Path, **changes: Any) -> PipelineConfig:
    settings: dict[str, Any] = {
        "video": inputs["video"],
        "audio_source": inputs["audio"],
        "sngs_dir": inputs["sngs"],
        "ball_raw": inputs["ball"],
        "out_dir": out,
        "team_names": TEAM_NAMES,
        "pose_estimator": _estimator(),
        "allow_licenses": ("GPL-3.0-only",),  # the synthetic plugin's declared licence
        "repo_root": base / "not-the-repo",
        "git_ceiling": base,
        "log": lambda text: None,
    }
    settings.update(changes)
    return PipelineConfig(**settings)


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("pipeline") / "非ASCII 路径-ü"
    base.mkdir()
    inputs = make_inputs(base / "inputs")
    out = base / "run 输出"
    started = time.perf_counter()
    result = run_pipeline(_config(inputs, out, base))
    elapsed = time.perf_counter() - started
    return {"base": base, "inputs": inputs, "out": out, "result": result, "elapsed": elapsed}


# --------------------------------------------------------------------------- the run


@pytest.mark.parametrize("tags", [("no_share",), ("nda",), ("hl3",)])
def test_report_stage_passes_the_restriction_watermark(
    unicode_tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tags: tuple[str, ...]
) -> None:
    seen = []
    monkeypatch.setattr(pipeline_module.measure_mod, "read_measurements", lambda _path: [])
    monkeypatch.setattr(
        pipeline_module.pd,
        "read_parquet",
        lambda *_a, **_k: pd.DataFrame({"frame": [0], "score": [0.0]}),
    )
    monkeypatch.setattr(pipeline_module, "_REPORT_ARTIFACTS", ())
    monkeypatch.setattr(pipeline_module, "_calibration_summary", lambda _run: {})
    monkeypatch.setattr(pipeline_module, "_tool_versions", lambda _run: [])
    monkeypatch.setattr(
        pipeline_module.report_mod, "write_report", lambda *args: seen.append(args[-1])
    )
    run = SimpleNamespace(
        final_highlights=lambda: {"highlights": []},
        path=lambda name: unicode_tmp_path / name,
        table=lambda _name: pd.DataFrame(columns=["start_frame", "end_frame", "shot_type"]),
        facts=SimpleNamespace(width=320, height=180, sha256="ab" * 32),
        notes=[],
        config=SimpleNamespace(labels=None),
        video=Path("synthetic.mp4"),
        audio=Path("synthetic.wav"),
        has_audio_stream=True,
        rate=Fraction(25),
        n_frames=1,
        cuts=lambda: [],
        restrictions=tags,
        dir=unicode_tmp_path,
    )
    pipeline_module._stage_report(run)
    assert seen[0].watermark == (
        pipeline_module.WATERMARK if set(tags) & {"nda", "no_share"} else None
    )


def test_run_finishes_quickly_and_runs_every_stage(synthetic_run: dict[str, Any]) -> None:
    result = synthetic_run["result"]
    assert [stage.name for stage in result.stages] == list(STAGES)
    assert all(stage.status == "ran" for stage in result.stages)
    assert synthetic_run["elapsed"] < 90.0
    text = summary_text(result)
    assert "h01" in text
    assert "render" in text


def test_video_without_audio_finishes_without_inventing_sound(unicode_tmp_path: Path) -> None:
    inputs = make_inputs(unicode_tmp_path / "inputs")
    out = unicode_tmp_path / "soundless run"
    result = run_pipeline(_config(inputs, out, unicode_tmp_path, audio_source=inputs["video"]))
    assert all(stage.status == "ran" for stage in result.stages)
    audio = pd.read_parquet(out / "audio.parquet")
    assert len(audio) == N_FRAMES
    assert not audio["has_audio"].any()
    assert audio[["rms_dbfs", "flux", "excitement_z"]].isna().all().all()
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "无音轨" in report
    assert "原声来源" not in report
    assert (out / "clips/full.mp4").is_file()
    for path in (out / "clips").glob("*.render.json"):
        manifest = read_json(path)
        assert manifest["output"]["audio"] is None
        assert manifest["audio"]["source"] is None
    # Rebuilding the report from cached stages must retain the missing-audio note.
    run_pipeline(
        _config(
            inputs,
            out,
            unicode_tmp_path,
            audio_source=inputs["video"],
            stages=("report",),
            force=True,
        )
    )
    rebuilt = (out / "report.html").read_text(encoding="utf-8")
    assert "无音轨" in rebuilt
    assert "原声来源" not in rebuilt


def test_highlight_is_found_at_the_kick(synthetic_run: dict[str, Any]) -> None:
    document = read_json(synthetic_run["out"] / "highlights.json")
    assert document["frame_count"] == N_FRAMES
    assert document["fps"] == FPS
    assert sorted(document["restrictions"]) == ["hl3", "no_share"]
    highlights = document["highlights"]
    assert len(highlights) >= 1
    best = max(highlights, key=lambda item: item["score"])
    assert best["start_frame"] <= KICK <= best["end_frame"]
    assert abs(best["peak_frame"] - KICK) <= 12
    assert STRIKER in best["track_ids"]


def test_measurements_are_valid_and_find_the_contact(synthetic_run: dict[str, Any]) -> None:
    records = read_measurements(synthetic_run["out"] / "measurements.jsonl")
    assert records
    assert all(record["tier"] != "C" for record in records)
    assert all(record["restrictions"] == ["hl3", "no_share"] for record in records)
    assert all(record["error"]["kind"] == "unmeasured" for record in records)
    unavailable = [r for r in records if r["status"] == "unavailable"]
    assert all(r["value"] is None and r["reasonCode"] for r in unavailable)

    contacts = [
        r for r in records if r["metric_id"] == "ball-contact-frame" and r["status"] == "ok"
    ]
    kick = [r for r in contacts if abs(r["value"] - KICK) <= 1]
    assert kick, [r["value"] for r in contacts]
    assert kick[0]["track_id"] == STRIKER
    assert kick[0]["quality"]["match"] == "foot_keypoint"
    assert kick[0]["tier"] == "B"
    assert kick[0]["claimType"] == "estimate"

    speeds = [
        r
        for r in records
        if r["metric_id"] == "ball-image-speed"
        and r["status"] == "ok"
        and r["frame"] == kick[0]["frame"]
    ]
    assert speeds
    assert speeds[0]["unit"] == "bh/s"
    assert speeds[0]["tier"] == "image_plane"
    assert "km/h" not in speeds[0]["display"]

    peak = [
        r
        for r in records
        if r["metric_id"] == "peak-speed" and r["track_id"] == STRIKER and r["status"] == "ok"
    ]
    assert peak, "the striker runs at 5.5 m/s on calibrated frames"
    assert peak[0]["tier"] == "A"
    assert peak[0]["unit"] == "m/s"
    assert 4.5 <= peak[0]["value"] <= 6.5

    # The joined close-up has no calibration: metres are unavailable there, image-plane
    # values are labelled as such.
    close = [r for r in records if r["track_id"] == CLOSE_UP_TRACK]
    for record in close:
        if record["plane"] == "pitch":
            assert record["status"] == "unavailable"
        else:
            assert record["tier"] in ("image_plane", "B")


def test_rendered_clips_have_the_planned_frames_and_audio(synthetic_run: dict[str, Any]) -> None:
    out = synthetic_run["out"]
    document = read_json(out / "highlights_final.json")  # what the render stage reads
    for item in document["highlights"]:
        clip = out / "clips" / f"{item['id']}.mp4"
        plan = plan_highlight_edit(
            item["start_frame"],
            item["end_frame"],
            item["peak_frame"],
            FPS,
            cuts=read_json(out / "cuts.json")["cuts"],
        )
        facts = probe(clip)
        assert facts.nb_frames_decoded == plan.n_frames
        assert (facts.frame_width, facts.frame_height) == (WIDTH, HEIGHT)
        manifest = read_json(out / "clips" / f"{item['id']}.render.json")
        assert manifest["output"]["audio"] is not None
        assert sorted(manifest["restrictions"]) == ["hl3", "no_share"]
        assert (out / "keyframes" / f"{item['id']}.png").is_file()
        assert (out / "keyframes" / f"{item['id']}-card.png").is_file()
    full = probe(out / "clips" / "full.mp4")
    assert full.nb_frames_decoded == N_FRAMES


def test_audio_offset_moves_the_sound_of_the_clips_with_the_audio_cue(
    synthetic_run: dict[str, Any], tmp_path: Path
) -> None:
    # --audio-offset-s 0.5: canonical frame 0 is 0.5 s into the audio (counted from its first
    # sample), so the burst at 1.6-2.8 s of the audio is at 1.1-2.3 s of the clips, where the
    # audio cue now has it, and the sound ends at 3.3 s (the audio lasts 3.8 s): the frames
    # after it are silent, as the cue's has_audio says.
    out = tmp_path / "offset 输出"
    shutil.copytree(synthetic_run["out"], out)
    config = _config(
        synthetic_run["inputs"], out, synthetic_run["base"], audio_offset_s=0.5,
        stages=("audio", "render"),
    )  # fmt: skip
    statuses = {stage.name: stage.status for stage in run_pipeline(config).stages}
    assert statuses["audio"] == statuses["render"] == "ran"
    receipt = read_json(out / "receipts" / "render.json")
    assert receipt["result"]["parameters"]["audio_offset_s"] == 0.5
    end_s = AUDIO_FRAMES / 25 - 0.5  # the clip time where the audio ends
    full = read_json(out / "clips" / "full.render.json")
    assert full["audio"]["offset_s"] == 0.5
    assert full["output"]["audio_parts"] == [
        {"kind": "source", "source": synthetic_run["inputs"]["audio"].name, "start_s": 0.0,
         "offset_s": 0.5, "samples": round(end_s * 48000), "gain_db": 0.0},
        {"kind": "silence", "samples": round((N_FRAMES / 25 - end_s) * 48000)},
    ]  # fmt: skip
    for item in read_json(out / "highlights_final.json")["highlights"]:
        parts = read_json(out / "clips" / f"{item['id']}.render.json")["output"]["audio_parts"]
        sources = [part for part in parts if part["kind"] == "source"]
        assert sources, item["id"]
        assert {part["offset_s"] for part in sources} == {0.5}, item["id"]
    cue = pd.read_parquet(out / "audio.parquet")
    with_audio = [k for k in range(N_FRAMES) if k / 25 + 0.5 < AUDIO_FRAMES / 25]
    assert cue.loc[cue["has_audio"], "frame"].tolist() == with_audio
    loud = cue.loc[cue["rms_dbfs"] > -20.0, "frame"].tolist()
    assert set(range(28, 58)) <= set(loud) <= set(range(27, 59)), "1.1-2.3 s"
    sound = decode_mono(out / "clips" / "full.mp4", sample_rate=48000).astype(np.float64)

    def rms(start_s: float, end: float) -> float:
        return float(np.sqrt(np.mean(np.square(sound[int(start_s * 48000) : int(end * 48000)]))))

    assert rms(1.15, 1.5) > 0.2, "the burst plays where the cue has it"
    assert rms(2.35, 3.2) < 0.03
    assert rms(3.35, 3.95) < 1e-3, "silence after the end of the audio"


def test_report_and_receipts(synthetic_run: dict[str, Any]) -> None:
    out = synthetic_run["out"]
    report = (out / "report.html").read_text(encoding="utf-8")
    document = read_json(out / "highlights.json")
    for item in document["highlights"]:
        assert f'id="hl-{item["id"]}"' in report
    assert "clips/full.mp4" in report
    for forbidden in (str(synthetic_run["base"]), "\\", synthetic_run["base"].drive):
        if forbidden:
            assert forbidden not in report
    run = read_json(out / "run.json")
    contracts.validate(run, "receipt")
    assert sorted(run["restrictions"]) == ["hl3", "no_share"]
    assert [stage["name"] for stage in run["result"]["stages"]] == list(STAGES)
    for stage in STAGES:
        receipt = read_json(out / "receipts" / f"{stage}.json")
        contracts.validate(receipt, "receipt")
        assert receipt["result"]["stage"] == stage
        for entry in receipt["outputs"]:
            assert sha256_file(out / entry["relpath"]) == entry["sha256"]
        text = json.dumps(receipt, ensure_ascii=False)
        assert str(synthetic_run["base"]) not in text
    for name in (
        "signals.parquet",
        "pose.parquet",
        "lift3d.parquet",
        "ball_track.parquet",
        "import/tracks.parquet",
        "import/calib.parquet",
        "refine/tracks.parquet",
        "refine/calib.parquet",
    ):
        assert sorted(read_restrictions(out / name)) == ["hl3", "no_share"], name
    # The import tables keep their producer while gaining the run's restrictions.
    tracks_meta = pq.read_schema(out / "import" / "tracks.parquet").metadata
    assert tracks_meta[b"gbf.producer"] == b"gbf.sngs_import"
    assert load_tracks(out / "import" / "tracks.parquet").attrs["restrictions"] == [
        "hl3",
        "no_share",
    ]


def test_render_stage_draws_the_3d_facing_with_the_gates_of_gbf_measure(
    synthetic_run: dict[str, Any],
) -> None:
    # The render stage reads lift3d.parquet (so a new lift re-runs it) and draws the focus
    # player's facing back on the frames where it passes gbf.measure's per-frame gates of
    # body-orientation; the receipt states the gates and the arrow's length.
    out = synthetic_run["out"]
    receipt = read_json(out / "receipts" / "render.json")
    inputs = {entry["name"]: entry["sha256"] for entry in receipt["inputs"]}
    assert inputs["lift3d.parquet"] == sha256_file(out / "lift3d.parquet")
    body = load_metric_registry()["body-orientation"]
    assert receipt["result"]["parameters"]["facing_arrow"] == {
        "length_m": render_module.FACING_ARROW_LENGTH_M,
        "gates": {
            "min_conf": DEFAULT_PARAMS.facing_min_conf,
            "max_sigma_deg": DEFAULT_PARAMS.facing_max_sigma_deg,
            "min_box_px": float(body.min_bbox_px),
        },
    }
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "3D 只作参考，所有结论以原始画面为准" in report
    records = read_measurements(out / "measurements.jsonl")
    lifted = [r for r in records if r["metric_id"] in ("body-orientation", "trunk-lean-at-contact")]
    assert lifted, "the synthetic run measures facing and lean (or says why it cannot)"
    assert all(r["tier"] == "A_candidate" for r in lifted)
    assert report.count('<span class="chip reference_3d">3D 参考</span>') >= len(lifted)


def test_card_names_other_players_and_units_are_shown_in_chinese(
    synthetic_run: dict[str, Any],
) -> None:
    out = synthetic_run["out"]
    records = read_measurements(out / "measurements.jsonl")
    highlight = read_json(out / "highlights.json")["highlights"][0]
    speed = next(r for r in records if r["metric_id"] == "ball-image-speed")
    shown = display_record(speed)
    assert shown["unit"] == "个身高/秒"
    assert speed["unit"] == "bh/s"  # the stored record is never changed
    assert shown["evidenceId"] == speed["evidenceId"]
    as_focus = {row.label: row for row in _card_rows(records, highlight, STRIKER)}
    as_other = {row.label: row for row in _card_rows(records, highlight, 12345)}
    assert any(label.startswith(f"轨迹{STRIKER}·") for label in as_other)
    assert not any(label.startswith(f"轨迹{STRIKER}·") for label in as_focus)
    contact = [row for label, row in as_focus.items() if "触球帧" in label]
    assert contact
    assert "帧·" not in contact[0].label  # the frame is the value, not repeated
    for rows in (as_focus, as_other):
        assert len(rows) <= 5
        assert all(row.evidence_ids for row in rows.values())


def _card_record(
    metric: str, value: float, unit: str, frame: int, track: int, tier: str, plane: str
) -> dict[str, Any]:
    """A synthetic evidence record as gbf.measure writes it (only the fields the card uses)."""
    return {
        "evidenceId": f"ev-{metric}-{track}",
        "highlight_id": "h01",
        "metric_id": metric,
        "label_zh": metric_label(metric),
        "status": "ok",
        "value": value,
        "unit": unit,
        "decimals": 2,
        "tier": tier,
        "plane": plane,
        "error": {"kind": "unmeasured"},
        "frame": frame,
        "track_id": track,
    }


@pytest.fixture(scope="module")
def style() -> Style:
    """The default render style with the system CJK fonts resolved once."""
    return Style().with_fonts()


def test_measure_stage_passes_team_verdicts_and_camera(
    unicode_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (unicode_tmp_path / "refine").mkdir()
    (unicode_tmp_path / "refine/refine.json").write_text(
        json.dumps(
            {"teams": {"tracks": [{"track_id": 7, "team": "left", "verdict": "unchecked"}]}}
        ),
        encoding="utf-8",
    )
    calib = pd.DataFrame({"frame_index": [0], "C": [[0.0, 60.0, -20.0]]})
    run = SimpleNamespace(
        path=lambda name: unicode_tmp_path / name,
        tracks=lambda: pd.DataFrame(),
        table=lambda name: pd.DataFrame(),
        ball_events_all=lambda: pd.DataFrame(),
        pitch_frames=lambda: set(),
        calib=lambda: calib,
        final_highlights=lambda: {},
        facts=SimpleNamespace(width=640, height=360),
        result_extras={},
    )
    captured: list[dict] = []
    monkeypatch.setattr(
        pipeline_module.measure_mod, "MeasureTables", lambda **kw: captured.append(kw)
    )
    monkeypatch.setattr(pipeline_module.pose_mod, "read_pose_parquet", lambda path: pd.DataFrame())
    monkeypatch.setattr(pipeline_module.pd, "read_parquet", lambda path: pd.DataFrame())
    monkeypatch.setattr(pipeline_module, "_measure_context", lambda run: None)
    monkeypatch.setattr(pipeline_module.measure_mod, "measure_highlights", lambda *args: [])
    monkeypatch.setattr(pipeline_module.measure_mod, "write_measurements", lambda *args: None)
    pipeline_module._stage_measure(run)
    assert captured[0]["team_verdicts"] == {(7, "left"): "unchecked"}
    assert captured[0]["frame_size"] == (640, 360)
    assert captured[0]["calib"] is calib
    assert "refine/refine.json" in pipeline_module._MEASURE_INPUTS
    assert pipeline_module.CALIB in pipeline_module._MEASURE_INPUTS


def test_render_stage_passes_team_verdicts_and_withheld_peaks(
    unicode_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    highlights = [{"id": "h01"}, {"id": "h02"}]
    records = [
        {
            "highlight_id": hid,
            "metric_id": metric,
            "status": status,
            "reasonCode": reason,
            "track_id": track,
        }
        for hid, metric, status, reason, track in [
            ("h01", "peak-speed", "unavailable", "speed_peak_hidden", 7),
            ("h02", "peak-speed", "unavailable", "speed_coverage_short", 9),
            ("h01", "peak-speed", "ok", "speed_peak_hidden", 11),
            ("h01", "peak-speed", "unavailable", "no_pose", 12),
            ("h01", "body-orientation", "unavailable", "speed_peak_hidden", 13),
        ]
    ]
    (unicode_tmp_path / "refine").mkdir()
    (unicode_tmp_path / "refine/refine.json").write_text(
        json.dumps(
            {"teams": {"tracks": [{"track_id": 7, "team": "left", "verdict": "unchecked"}]}}
        ),
        encoding="utf-8",
    )
    run = SimpleNamespace(
        path=lambda name: unicode_tmp_path / name,
        final_highlights=lambda: {"highlights": highlights},
        tracks=lambda: pd.DataFrame(),
        calib=lambda: pd.DataFrame(),
        table=lambda name: pd.DataFrame(),
        cuts=lambda: [],
        pitch_frames=lambda: set(),
        facts=SimpleNamespace(sha256="synthetic", width=640, height=360),
        config=SimpleNamespace(full_clip=True, audio_offset_s=0.0),
        n_frames=40,
        rate="25/1",
        dir=unicode_tmp_path,
        video=unicode_tmp_path / "source.mp4",
        audio=None,
        has_audio_stream=False,
        restrictions=(),
        log=lambda text: None,
        result_extras={},
    )
    monkeypatch.setattr(pipeline_module.measure_mod, "read_measurements", lambda path: records)
    monkeypatch.setattr(
        pipeline_module.pose_mod, "read_pose_parquet", lambda path: pd.DataFrame({"mean_score": []})
    )
    monkeypatch.setattr(pipeline_module.pd, "read_parquet", lambda *args, **kw: pd.DataFrame())
    table_args: list[dict] = []
    scene_args: list[dict] = []
    monkeypatch.setattr(render_module, "SceneTables", lambda **kw: table_args.append(kw))
    monkeypatch.setattr(render_module, "HighlightScene", lambda *args, **kw: scene_args.append(kw))
    monkeypatch.setattr(pipeline_module, "_facing_gates", lambda: None)
    monkeypatch.setattr(pipeline_module, "_style", lambda run: Style())
    monkeypatch.setattr(render_module, "first_sample_offset", lambda source, offset: offset)
    monkeypatch.setattr(pipeline_module, "_render_team_names", lambda run: ())
    monkeypatch.setattr(pipeline_module, "_card", lambda *args: ((), 0))
    edit = SimpleNamespace(
        video=SimpleNamespace(frames=40),
        plan=SimpleNamespace(table=lambda: []),
        manifest_path=Path("full.render.json"),
    )
    result = SimpleNamespace(
        clip="clip.mp4",
        manifest="clip.render.json",
        keyframe="key.png",
        keyframe_card="card.png",
        edit=edit,
    )
    monkeypatch.setattr(render_module, "render_highlight", lambda *args, **kw: result)
    monkeypatch.setattr(render_module, "render_edit", lambda *args, **kw: edit)
    pipeline_module._stage_render(run)
    assert table_args[0]["team_verdicts"] == {(7, "left"): "unchecked"}
    assert [set(kw["peak_speed_withheld"]) for kw in scene_args] == [{7}, {9}, {7, 9}]
    assert all(kw["frame_size"] == (640, 360) for kw in scene_args)


@pytest.mark.parametrize(("frame", "track"), [(13, 7), (20, 42)])
def test_orientation_card_label_fits_with_prefix_and_badges(
    frame: int,
    track: int,
    style: Style,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _card_record("body-orientation", 45.0, "deg", frame, track, "A_candidate", "pitch")
    rows = _card_rows([record], {"id": "h01", "peak_frame": 20}, 7)
    drawn: list[str] = []
    original = render_module._text

    def capture(c: object, xy: object, text: str, *args: object, **kw: object) -> None:
        drawn.append(text)
        original(c, xy, text, *args, **kw)

    monkeypatch.setattr(render_module, "_text", capture)
    draw_annotations(
        np.zeros((1080, 1920, 3), np.uint8), FrameAnnotations(card=InfoCard("key", rows)), style
    )
    assert rows[0].label in drawn, "orientation window and mean must survive fitting"


def test_card_moment_rows_name_the_players_and_what_was_measured() -> None:
    # The report shows each key moment as a sentence; the card, where long labels are cut,
    # shows the players and the short name of what was measured, badged as an image-plane
    # estimate. A touch count names its player's track like the moments: the touches may
    # be another player's than the one in focus.
    highlight = {
        "key_moments": [
            {"kind": "contact_start", "track_ids": [5, 9], "frame": 40, "time": "00:01.60",
             "text_zh": "甲方球员与乙方球员身体贴近（画面平面）"},
            {"kind": "touch", "track_ids": [5], "frame": 42, "time": "00:01.68",
             "text_zh": "甲方球员触球"},
            {"kind": "fall_onset", "track_ids": [9], "frame": 50, "time": "00:02.00",
             "text_zh": "乙方球员躯干开始倾倒"},
        ],
        "touches": [{"label_zh": "甲方球员", "count": 2, "track_id": 5}],
    }  # fmt: skip
    rows = _moment_rows(highlight)
    assert [row.label for row in rows] == [
        "轨迹5、9·身体贴近",
        "轨迹9·躯干开始倾倒",
        "轨迹5·检测到的触球",
    ]
    assert [(row.value, row.unit) for row in rows[:2]] == [
        ("00:01.60", "第 40 帧"),
        ("00:02.00", "第 50 帧"),
    ]
    assert all({b.text for b in row.badges} == {"未验证", "仅画面平面"} for row in rows)


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720)])
@pytest.mark.parametrize(("other", "key", "earlier"), [(42, 2400, 2391), (100042, 150123, 150114)])
def test_card_labels_keep_the_player_and_the_frame_that_render_draws(
    size: tuple[int, int],
    other: int,
    key: int,
    earlier: int,
    style: Style,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Values of track ``other`` next to the focus player 7, on the key frame and on an
    # earlier frame; gbf.render cuts a label that does not fit at its end.
    focus = 7
    highlight = {"id": "h01", "type": "duel_tackle", "peak_frame": key, "track_ids": [focus, other]}
    records = [
        # image-plane values carry two badges, the least label room (316 px at 1080p): one on
        # another frame (the player and the frame), one on the key frame (the player only)
        _card_record(
            "nearest-opponent-image-distance", 0.85, "box_h", earlier, other, "image_plane", "image"
        ),
        _card_record("ball-image-speed", 1.3, "bh/s", key, other, "image_plane", "image"),
        # a value in metres on another frame: the player and the frame
        _card_record("nearest-opponent-distance", 2.4, "m", earlier, other, "A", "pitch"),
        # the focus player's own value on the key frame: neither
        _card_record("peak-speed", 7.6, "m/s", key, focus, "A", "pitch"),
    ]
    rows = {row.evidence_ids[0]: row for row in _card_rows(records, highlight, focus)}
    assert len(rows) == 4
    drawn: list[str] = []
    draw_text = render_module._text

    def recording(c: object, xy: object, text: str, *args: object, **kwargs: object) -> None:
        drawn.append(text)
        draw_text(c, xy, text, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(render_module, "_text", recording)
    width, height = size
    card = InfoCard(f"关键帧 · 第 {key} 帧", tuple(rows.values()))
    draw_annotations(np.zeros((height, width, 3), np.uint8), FrameAnnotations(card=card), style)

    def shown(evidence: str) -> str:
        """The label gbf.render drew for a row: all of it, or its start and an ellipsis when
        it did not fit (how much fits depends on the font, GBF_CJK_FONT, and the size)."""
        label = rows[evidence].label
        for text in drawn:
            if text == label or (text.endswith("…") and label.startswith(text[:-1])):
                return text
        raise AssertionError(f"{label!r} is not drawn: {drawn}")

    # Cut or not, the player and the frame are drawn.
    for metric in ("nearest-opponent-image-distance", "nearest-opponent-distance"):
        label = shown(f"ev-{metric}-{other}")
        assert label.startswith(f"轨迹{other}·第{earlier}帧"), label
    on_key = f"ev-ball-image-speed-{other}"
    assert shown(on_key).startswith(f"轨迹{other}"), shown(on_key)
    assert f"第{key}帧" not in rows[on_key].label, "a value of the key frame names no frame"
    own = shown(f"ev-peak-speed-{focus}")
    assert own == metric_label("peak-speed"), "the focus player's own value keeps its label"


def test_final_highlights_are_ranked_and_the_refine_stage_reports_why_it_did_nothing(
    synthetic_run: dict[str, Any],
) -> None:
    out = synthetic_run["out"]
    first = read_json(out / "highlights.json")
    final = read_json(out / "highlights_final.json")
    assert [h["id"] for h in final["highlights"]] == [h["id"] for h in first["highlights"]]
    assert sorted(final["restrictions"]) == ["hl3", "no_share"]
    ranks = sorted(h["rank"] for h in final["highlights"])
    assert ranks == list(range(1, len(ranks) + 1))
    assert all(h["tier"] in ("main", "candidate") for h in final["highlights"])
    assert all("key_moments" in h for h in final["highlights"])
    refine = read_json(out / "refine" / "refine.json")
    assert refine["calibration"] == {
        "refined": False,
        "reason": "the plugin export has no keypoints.jsonl",
    }
    assert sorted(refine["restrictions"]) == ["hl3", "no_share"]
    # The shirt-colour check of the team labels ran; the synthetic players are too small
    # for it to judge anything, and the report says so.
    assert refine["teams"]["counts"]["player_tracks"] >= 2
    assert refine["teams"]["counts"]["inconsistent"] == 0
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "球衣颜色核对没能进行" in report
    # The role labels were checked against the one referee on the pitch and kept.
    assert refine["roles"]["verdict"] == "plausible"
    assert refine["roles"]["rows_cleared"] == 0
    assert "角色标签不合理" not in report


def test_labels_are_compared_and_shown_in_the_report(
    synthetic_run: dict[str, Any], tmp_path: Path
) -> None:
    copy = tmp_path / "带标注 run"
    shutil.copytree(synthetic_run["out"], copy)
    best = max(read_json(copy / "highlights_final.json")["highlights"], key=lambda h: h["score"])
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "id": "synthetic-v1",
                "labelled_by": "the generator of the synthetic sprint clip",
                "blinding": "written from the generator, before any run",
                "moments": [
                    {
                        "id": "M1",
                        "type": best["type"],
                        "start": KICK - 10,
                        "peak": KICK,
                        "end": KICK + 10,
                        "peak_tolerance_frames": 12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = run_pipeline(
        _config(synthetic_run["inputs"], copy, synthetic_run["base"], labels=labels)
    )
    statuses = {stage.name: stage.status for stage in result.stages}
    assert statuses["highlights"] == "skipped"
    assert statuses["moments"] == "ran"
    evaluation = read_json(copy / "label_eval.json")
    assert sorted(evaluation["restrictions"]) == ["hl3", "no_share"]
    assert read_json(copy / "run.json")["result"]["restriction_sources"]["labels"] == ["no_share"]
    assert evaluation["n_labels"] == 1
    assert evaluation["n_matched"] == 1
    row = evaluation["rows"][0]
    assert row["highlight"] == best["id"]
    assert abs(row["peak_error_frames"]) <= 12
    assert evaluation["labels_source"] == {
        "labelled_by": "the generator of the synthetic sprint clip",
        "blinding": "written from the generator, before any run",
    }
    assert evaluation["source_zh"].startswith("标注来源（照录标注文件）：标注者")
    report = (copy / "report.html").read_text(encoding="utf-8")
    # Where the labels come from is quoted from the label file (its words kept: sprint is
    # not renamed as a type code would be); nothing more is claimed.
    section = report[report.index('id="evaluation"') : report.index('id="notes"')]
    assert "<h2>与标注对比</h2>" in section
    assert (
        "标注来源（照录标注文件）：标注者“the generator of the synthetic sprint clip”；"
        "盲法“written from the generator, before any run”。"
    ) in section
    assert "人工" not in section


def test_pose_was_run_only_for_the_needed_boxes(synthetic_run: dict[str, Any]) -> None:
    out = synthetic_run["out"]
    pose = pd.read_parquet(out / "pose.parquet")
    document = read_json(out / "highlights.json")
    windows = [(h["start_frame"], h["end_frame"]) for h in document["highlights"]]
    assert len(pose) > 0
    inside = pose["frame"].apply(lambda f: any(lo <= f <= hi for lo, hi in windows))
    assert bool(inside.all())
    involved = {t for h in document["highlights"] for t in h["track_ids"]}
    assert set(pose["track_id"]) <= involved | set(PLAYERS)
    lift = pd.read_parquet(out / "lift3d.parquet")
    assert len(lift) == len(pose)
    lifted = lift[lift["method"] == "height_prior_calibrated"]
    assert len(lifted) > 0
    assert lifted["frame"].max() < CUT


def test_input_gaps_become_report_notes(synthetic_run: dict[str, Any]) -> None:
    out = synthetic_run["out"]
    notes = read_json(out / "run.json")["result"]["notes_zh"]
    assert any(f"{AUDIO_FRAMES}/{N_FRAMES}" in note for note in notes)
    assert any("球检测文件不完整" in note and "未写完" in note for note in notes)
    report = (out / "report.html").read_text(encoding="utf-8")
    assert f"{AUDIO_FRAMES}/{N_FRAMES}" in report
    ball = read_json(out / "receipts" / "ball.json")["result"]["facts"]
    assert ball["incomplete_tail_dropped"] is True
    assert ball["frames_without_a_line"] == 1


def test_kinematics_facts_count_every_rejected_position(synthetic_run: dict[str, Any]) -> None:
    # segments.parquet n_outliers counts only rejections inside a segment's span; the run's
    # facts count every rejected position, by test.
    out = synthetic_run["out"]
    facts = read_json(out / "receipts" / "kinematics.json")["result"]["facts"]
    by_test = facts["rejected_by_test"]
    assert sorted(by_test) == sorted(REJECTION_TESTS)
    assert facts["rejected_positions"] == sum(by_test.values())
    inside = int(pd.read_parquet(out / "segments.parquet")["n_outliers"].sum())
    assert facts["rejected_positions"] >= inside


def test_second_run_skips_every_stage(synthetic_run: dict[str, Any]) -> None:
    out = synthetic_run["out"]
    before = {path: sha256_file(path) for path in sorted(out.rglob("*")) if path.is_file()}
    result = run_pipeline(_config(synthetic_run["inputs"], out, synthetic_run["base"]))
    statuses = {stage.name: stage.status for stage in result.stages}
    assert set(statuses.values()) == {"skipped"}
    after = {path: sha256_file(path) for path in sorted(out.rglob("*")) if path.is_file()}
    changed = [path.name for path in before if before[path] != after.get(path)]
    assert changed == ["run.json"]
    # Notes of skipped stages come back from their receipts.
    notes = read_json(out / "run.json")["result"]["notes_zh"]
    assert any(f"{AUDIO_FRAMES}/{N_FRAMES}" in note for note in notes)


def test_forced_rerun_gives_the_same_bytes(synthetic_run: dict[str, Any]) -> None:
    out = synthetic_run["out"]

    def artifacts() -> dict[str, str]:
        return {
            path.relative_to(out).as_posix(): sha256_file(path)
            for path in sorted(out.rglob("*"))
            if path.is_file() and path.parent.name != "receipts" and path.name != "run.json"
        }

    before = artifacts()
    result = run_pipeline(_config(synthetic_run["inputs"], out, synthetic_run["base"], force=True))
    assert {stage.status for stage in result.stages} == {"ran"}
    after = artifacts()
    assert sorted(after) == sorted(before)
    changed = sorted(name for name in before if before[name] != after[name])
    assert changed == []


def test_changed_parameter_reruns_only_what_depends_on_it(synthetic_run: dict[str, Any]) -> None:
    out = synthetic_run["out"]
    before = sha256_file(out / "highlights.json")
    result = run_pipeline(
        _config(synthetic_run["inputs"], out, synthetic_run["base"], max_highlights=5)
    )
    statuses = {stage.name: stage.status for stage in result.stages}
    for stage in ("import", "shots", "refine", "kinematics", "ball", "audio"):
        assert statuses[stage] == "skipped"
    assert statuses["highlights"] == "ran"
    unchanged = sha256_file(out / "highlights.json") == before
    for stage in ("pose", "lift3d", "moments", "measure", "render", "report"):
        assert statuses[stage] == ("skipped" if unchanged else "ran")


def test_unselected_stage_without_outputs_is_an_error(
    synthetic_run: dict[str, Any], tmp_path: Path
) -> None:
    base = synthetic_run["base"]
    config = _config(synthetic_run["inputs"], base / "fresh run", base, stages=("report",))
    with pytest.raises(PipelineError, match="not in --stages"):
        run_pipeline(config)
    run = read_json(base / "fresh run" / "run.json")
    assert run["status"] == "failed"


def test_every_json_artifact_carries_the_run_restrictions(synthetic_run: dict[str, Any]) -> None:
    out = synthetic_run["out"]
    found = 0
    for path in sorted(out.rglob("*")):
        if path.suffix == ".json":
            document = read_json(path)
            assert sorted(document.get("restrictions", [])) == ["hl3", "no_share"], path.name
            found += 1
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                assert sorted(json.loads(line)["restrictions"]) == ["hl3", "no_share"], path.name
    assert found >= 10
    sources = read_json(out / "run.json")["result"]["restriction_sources"]
    assert sources["plugin"] == list(PLUGIN_TAGS)
    assert sources["ball"] == ["no_share"], "a ball file without a receipt is private"
    assert "pose_model" not in sources, "the fake pose estimator carries no model tags"


def test_a_rerun_with_another_tag_writes_it_into_every_artifact(
    synthetic_run: dict[str, Any], tmp_path: Path
) -> None:
    # Re-running into an existing output folder with one more tag must not keep artifacts
    # of skipped stages with the old tags.
    out = tmp_path / "re-run 输出"
    shutil.copytree(synthetic_run["out"], out)
    tags = ("no_share", "research_nc")
    run_pipeline(_config(synthetic_run["inputs"], out, synthetic_run["base"], restrictions=tags))
    expected = ["hl3", "no_share", "research_nc"]
    checked = 0
    for path in sorted(out.rglob("*")):
        if path.suffix == ".json":
            assert sorted(read_json(path).get("restrictions", [])) == expected, path.name
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                assert sorted(json.loads(line)["restrictions"]) == expected, path.name
        elif path.suffix == ".parquet":
            assert sorted(read_restrictions(path)) == expected, path.name
        else:
            continue
        checked += 1
    assert checked >= 20


def test_ball_receipt_tags_reach_every_artifact_and_its_licence_needs_opting_in(
    synthetic_run: dict[str, Any], tmp_path: Path
) -> None:
    inputs = dict(synthetic_run["inputs"])
    ball_dir = tmp_path / "球 输入"
    ball_dir.mkdir()
    ball = ball_dir / inputs["ball"].name
    shutil.copyfile(inputs["ball"], ball)
    (ball_dir / "ball-receipt.json").write_text(
        json.dumps({"license": "AGPL-3.0-only", "restrictions": ["agpl_network_source"]}),
        encoding="utf-8",
    )
    inputs["ball"] = ball
    base = synthetic_run["base"]
    out = tmp_path / "ball run"
    with pytest.raises(PipelineError, match="AGPL-3.0-only"):
        run_pipeline(_config(inputs, out, base))
    assert not out.exists(), "a refused licence writes nothing"
    allowed = ("GPL-3.0-only", "AGPL-3.0-only")
    run_pipeline(_config(inputs, out, base, allow_licenses=allowed))
    run = read_json(out / "run.json")
    assert sorted(run["restrictions"]) == ["agpl_network_source", "hl3", "no_share"]
    assert run["result"]["restriction_sources"]["ball"] == ["agpl_network_source"]
    assert run["result"]["allow_licenses"] == list(allowed)
    assert sorted(read_restrictions(out / "import" / "tracks.parquet")) == [
        "agpl_network_source",
        "hl3",
        "no_share",
    ]


def test_plugin_licence_outside_the_allowlist_is_refused(
    synthetic_run: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "no opt-in"
    config = _config(synthetic_run["inputs"], out, synthetic_run["base"], allow_licenses=())
    with pytest.raises(PipelineError, match="--allow-license"):
        run_pipeline(config)
    assert not out.exists()


# --------------------------------------------------------------------------- refusals


def test_out_dir_inside_a_git_work_tree_is_refused(tmp_path: Path) -> None:
    work_tree = tmp_path / "其他 repo"
    (work_tree / ".git").mkdir(parents=True)
    problems = check_out_dir(
        work_tree / "runs" / "a", repo_root=tmp_path / "gbf", git_ceiling=tmp_path
    )
    assert len(problems) == 1
    assert "git work tree" in problems[0]
    inside = check_out_dir(tmp_path / "gbf" / "x", repo_root=tmp_path / "gbf", git_ceiling=tmp_path)
    assert len(inside) == 1
    assert "gbf checkout" in inside[0]
    assert (
        check_out_dir(tmp_path / "data" / "run", repo_root=tmp_path / "gbf", git_ceiling=tmp_path)
        == []
    )
    # The real checkout is refused by default.
    assert check_out_dir(repo_root() / "runs" / "x")


def test_cli_lists_every_problem_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = repo_root() / ".local" / "pipeline-cli-refusal-test"
    code = main(
        [
            "--video",
            str(tmp_path / "missing.mp4"),
            "--audio-source",
            str(tmp_path / "missing.wav"),
            "--sngs",
            str(tmp_path / "no export"),
            "--ball",
            str(tmp_path / "ball.jsonl"),
            "--out",
            str(out),
            "--stages",
            "import,nonsense",
            "--restrictions",
            "no_share,made_up",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    for fragment in ("--video", "--audio-source", "--sngs", "--ball", "nonsense", "made_up"):
        assert fragment in err
    assert "gbf checkout" in err
    assert not out.exists()


def test_cli_rejects_bad_team_names(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--video",
            "v",
            "--audio-source",
            "a",
            "--sngs",
            "s",
            "--ball",
            "b",
            "--out",
            "o",
            "--team-names",
            "left",
        ]
    )
    assert code == 2
    assert "left=NAME" in capsys.readouterr().err


# --------------------------------------------------------------------------- review fixes


def test_a_plugin_export_without_its_receipt_is_refused_and_nothing_is_written(
    synthetic_run: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The run's restrictions and licences come from the receipts: an export without one is
    # refused input (exit 2, nothing written), not a failed stage (exit 1, a run folder).
    inputs = dict(synthetic_run["inputs"])
    sngs = tmp_path / "导出 without receipt"
    sngs.mkdir()
    for name in ("detections.jsonl", "calib.jsonl"):
        shutil.copyfile(inputs["sngs"] / name, sngs / name)
    inputs["sngs"] = sngs
    out = tmp_path / "refused run"
    with pytest.raises(PipelineError, match="plugin-receipt.json is missing"):
        run_pipeline(_config(inputs, out, synthetic_run["base"]))
    assert not out.exists()
    argv = [
        "--video",
        str(inputs["video"]),
        "--audio-source",
        str(inputs["audio"]),
        "--sngs",
        str(sngs),
        "--ball",
        str(inputs["ball"]),
        "--out",
        str(out),
        "--allow-license",
        "GPL-3.0-only",
    ]
    assert main(argv) == 2
    assert "plugin-receipt.json is missing" in capsys.readouterr().err
    assert not out.exists()
    # Pitch keypoints that the refit would use need their own receipt as well.
    shutil.copyfile(
        synthetic_run["inputs"]["sngs"] / "plugin-receipt.json", sngs / "plugin-receipt.json"
    )
    (sngs / "keypoints.jsonl").write_bytes(b"")
    with pytest.raises(PipelineError, match="keypoints-receipt.json is missing"):
        run_pipeline(_config(inputs, out, synthetic_run["base"]))
    assert not out.exists()
    assert main(argv) == 2
    assert "keypoints-receipt.json is missing" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b'{"license": "GPL-3.0-only", "restrictions": [', "cannot be read as JSON"),
        (b"\xff\xfe not utf-8", "cannot be read as JSON"),
        (b'{"license": "GPL-3.0-only"}', "declares no restrictions list"),
        (b'{"license": "GPL-3.0-only", "restrictions": "no_share"}', "declares no restrictions"),
        (b'["no_share"]', "declares no restrictions list"),
        (b'{"license": "GPL-3.0-only", "restrictions": ["made_up"]}', "plugin-receipt.json: "),
    ],
)
def test_an_unusable_plugin_receipt_is_refused_and_nothing_is_written(
    unicode_tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes, expected: str
) -> None:
    # The run's restrictions are read from the receipt once the run folder exists: one that
    # does not parse, or gives no valid restrictions list, failed there (exit 1, a run
    # folder with a failed run.json) instead of being refused input (exit 2, nothing).
    inputs = {
        "video": unicode_tmp_path / "片段.mp4",
        "audio": unicode_tmp_path / "原声.wav",
        "ball": unicode_tmp_path / "ball_raw.jsonl",
        "sngs": unicode_tmp_path / "导出 export",
    }
    inputs["sngs"].mkdir()
    for name in ("video", "audio", "ball"):
        inputs[name].write_bytes(b"placeholder: only the receipt is read before refusing")
    for name in ("detections.jsonl", "calib.jsonl"):
        (inputs["sngs"] / name).write_bytes(b"")
    (inputs["sngs"] / "plugin-receipt.json").write_bytes(content)
    out = unicode_tmp_path / "refused run"
    with pytest.raises(PipelineError) as caught:
        run_pipeline(_config(inputs, out, unicode_tmp_path))
    assert any(problem.startswith("plugin-receipt.json") for problem in caught.value.problems)
    assert expected in str(caught.value)
    assert not out.exists()
    argv = [
        "--video",
        str(inputs["video"]),
        "--audio-source",
        str(inputs["audio"]),
        "--sngs",
        str(inputs["sngs"]),
        "--ball",
        str(inputs["ball"]),
        "--out",
        str(out),
        "--allow-license",
        "GPL-3.0-only",
    ]
    assert main(argv) == 2
    assert expected in capsys.readouterr().err
    assert not out.exists()


def _canned_team_check() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(tracks, tracks as gbf.team_check returns them, its report): tracks 1-6 consistent, 7
    contradicted (its rows cleared), 8 and 9 with too few shirt samples, and 100003, a piece
    of track 3 split off at a cut, unchecked too."""
    rows = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 6, 8: 4, 9: 3, 100003: 5}
    teams = dict.fromkeys((1, 2, 3, 7, 9, 100003), "left") | dict.fromkeys((4, 5, 6, 8), "right")
    tracks = pd.DataFrame(
        [
            {"frame": frame, "track_id": track, "role": "player", "team": teams[track]}
            for track, count in rows.items()
            for frame in range(count)
        ]
    )
    checked = tracks.assign(team=tracks["team"].where(tracks["track_id"] != 7, None))

    def entry(track: int, verdict: str, reason: str | None = None) -> dict[str, Any]:
        cleared = "inconsistent" if verdict == "inconsistent" else None
        return {"track_id": track, "team": teams[track], "rows": rows[track],
                "verdict": verdict, "reason": reason, "cleared": cleared}  # fmt: skip

    entries = [entry(track, "consistent") for track in range(1, 7)] + [entry(7, "inconsistent")]
    entries += [entry(track, "unchecked", "too_few_samples") for track in (8, 9, 100003)]
    report = {
        "prototypes": {"left": {"tracks": 3}, "right": {"tracks": 3}},
        "tracks": entries,
        "counts": {"player_tracks": 10, "consistent": 6, "inconsistent": 1, "unchecked": 3,
                   "rows_cleared": 6},
    }  # fmt: skip
    return tracks, checked, report


def test_the_team_note_counts_tracks_kept_unchecked_and_every_cleared_row_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracks, checked, report = _canned_team_check()

    def no_frames(video: Path) -> Any:
        yield from ()

    monkeypatch.setattr(pipeline_module, "iter_frames", no_frames)
    team_check = pipeline_module.team_check_mod
    monkeypatch.setattr(team_check, "plan_samples", lambda *args, **kwargs: None)
    monkeypatch.setattr(team_check, "measure_shirts", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(team_check, "check_teams", lambda *args, **kwargs: (checked, report))
    run = SimpleNamespace(facts=None, video=Path("synthetic.mp4"), notes=[])
    cleaned, result = pipeline_module._check_team_labels(run, tracks)  # type: ignore[arg-type]
    # The note says how many tracks could not be checked and that they keep the plugin's
    # label, next to the contradicted tracks and the split-off pieces.
    assert run.notes == [
        "球队标签按球衣颜色核对：7 条轨迹球衣样本足够，其中 1 条的球队标签与球衣颜色矛盾，"
        "改为未知球队（不翻转）；2 条轨迹球衣样本太少、未能核对，沿用插件的球队标签；"
        "1 条切镜后拆出的轨迹无法核对球衣，也不标球队。"
    ]
    teams = {
        track: set(group.astype(object).where(group.notna(), None))
        for track, group in cleaned.groupby("track_id")["team"]
    }
    assert teams[7] == teams[100003] == {None}
    assert (teams[8], teams[9]) == ({"right"}, {"left"}), "too few samples: the plugin's label"
    # refine.json counts the split-off piece's rows too, and marks its entry.
    counts = result["counts"]
    assert (counts["rows_cleared"], counts["rows_cleared_split_pieces"]) == (6 + 5, 5)
    marks = {entry["track_id"]: entry["cleared"] for entry in result["tracks"]}
    assert marks == dict.fromkeys(range(1, 10)) | {7: "inconsistent", 100003: "split_piece"}
    assert result["cleared_inherited_tracks"] == [100003]
    # Without both teams' colours nothing is judged or cleared, and the note says so.
    tracks, checked, report = _canned_team_check()
    report["prototypes"]["left"] = None
    kept, unjudged = pipeline_module._clear_split_pieces(checked, report)
    assert unjudged["counts"]["rows_cleared_split_pieces"] == 0
    assert (kept["team"] == "left").sum() == (checked["team"] == "left").sum()
    assert pipeline_module._team_note(unjudged).startswith("球衣颜色核对没能进行")
    # Nothing contradicted or unchecked: the note says that the labels were checked.
    agreeing = {**report, "prototypes": {"left": {}, "right": {}}, "tracks": report["tracks"][:6]}
    assert pipeline_module._team_note(agreeing) == (
        "球队标签按球衣颜色核对：6 条轨迹球衣样本足够，球队标签都没有与球衣颜色矛盾。"
    )


def test_implausible_role_labels_clear_every_team_label_and_the_note_says_why() -> None:
    people = ((1, "player", "left", -30.0), (2, "player", "right", 30.0),
              (3, "referee", None, -5.0), (4, "referee", None, 5.0))  # fmt: skip
    tracks = pd.DataFrame(
        [
            {
                "frame": frame,
                "track_id": track,
                "role": role,
                "team": team,
                "pitch_usable": True,
                "pitch_x": x,
                "pitch_y": 0.0,
            }
            for frame in range(30)
            for track, role, team, x in people
        ]
    )
    run = SimpleNamespace(notes=[])
    cleared, report = pipeline_module._check_role_labels(run, tracks)  # type: ignore[arg-type]
    assert report["verdict"] == "implausible"
    assert cleared["team"].isna().all()
    assert cleared["role"].tolist() == tracks["role"].tolist()
    assert run.notes == [
        "角色标签不合理：30 帧里有 30 帧（100%）同时有两人以上被插件标成裁判、站在场内，"
        "而规则只允许一名裁判在场内。插件只给标成球员的人分队，所以分队也不可信："
        "球队标签全部清空，依赖球队的数值（最近对手等）不再计算。角色沿用插件的，未经核对。"
    ]
    # One referee inside the pitch: the labels stay and there is no note.
    run = SimpleNamespace(notes=[])
    one_referee = tracks[tracks["track_id"] != 4]
    kept, report = pipeline_module._check_role_labels(run, one_referee)  # type: ignore[arg-type]
    assert (report["verdict"], run.notes) == ("plausible", [])
    assert kept["team"].notna().sum() == 60


def test_left_right_names_that_cannot_be_confirmed_get_a_note() -> None:
    # gbf.side_check: right minus left mean x is 1.1 - 0.5 = 0.6 m; resampling two tracks a
    # team draws the same one twice 1 time in 4, so the interval runs from -1 - 3 to 3.2 + 2.
    people = ((1, "left", -2.0), (2, "left", 3.0), (3, "right", -1.0), (4, "right", 3.2))
    tracks = pd.DataFrame(
        [
            {
                "frame": frame,
                "track_id": track,
                "role": "player",
                "team": team,
                "pitch_usable": True,
                "pitch_x": x,
                "pitch_y": 0.0,
            }
            for frame in range(30)
            for track, team, x in people
        ]
    )
    run = SimpleNamespace(notes=[])
    report = pipeline_module._check_sides(run, tracks)  # type: ignore[arg-type]
    assert (report["verdict"], report["attack_direction_known"]) == ("uncertain", False)
    assert run.notes == [
        "左右分队未能核实：插件只按两队全部检测的平均位置定左右。按 gbf 的标定，标为右侧的"
        "球队平均位置减去标为左侧的为 +0.6 米（95% 区间 -4.0 至 +5.2 米），区间跨过 0。"
        "所以不知道哪队进攻哪个球门：标题、说明和信息卡里两队只称「甲队」「乙队」"
        "（不用配置的队名），禁区压迫和射门方向两种线索不计，禁区进攻人数和相对进攻球门的"
        "身体朝向不给值；最近对手距离等只需分清两队的数值照常。"
    ]
    # Teams standing in their own halves: the names hold and there is no note.
    run = SimpleNamespace(notes=[])
    apart = tracks.assign(pitch_x=tracks["pitch_x"] + np.where(tracks["team"] == "left", -20, 20))
    report = pipeline_module._check_sides(run, apart)  # type: ignore[arg-type]
    assert (report["verdict"], run.notes) == ("consistent", [])


def test_team_names_are_neutral_when_the_attack_direction_is_unknown(tmp_path: Path) -> None:
    # The configured names are given per side, so they cannot be matched to the teams either.
    (tmp_path / "refine").mkdir()
    for configured in (TEAM_NAMES, None):
        run = SimpleNamespace(
            config=SimpleNamespace(team_names=configured), path=lambda name: tmp_path / name
        )
        for refine, expected in (
            ({"sides": {"attack_direction_known": False}}, {"left": "甲队", "right": "乙队"}),
            ({"sides": {"attack_direction_known": True}}, configured),
            ({}, configured),  # a refine stage from before the check
        ):
            (tmp_path / "refine" / "refine.json").write_text(json.dumps(refine), encoding="utf-8")
            assert pipeline_module._team_names(run) == expected  # type: ignore[arg-type]


def test_the_synthetic_run_cannot_confirm_its_left_right_names(
    synthetic_run: dict[str, Any],
) -> None:
    # Three left players and two right players within a few metres of each other: the run
    # treats the direction each team attacks as unknown, end to end.
    out = synthetic_run["out"]
    sides = read_json(out / "refine" / "refine.json")["sides"]
    assert (sides["verdict"], sides["tracks"]) == ("uncertain", {"left": 3, "right": 2})
    assert "左右分队未能核实" in (out / "report.html").read_text(encoding="utf-8")
    signals = pd.read_parquet(out / "signals.parquet")
    assert (signals["box_pressure"] == 0).all()
    assert (signals["ball_goal"] == 0).all()
    final = read_json(out / "highlights_final.json")["highlights"]
    texts = "".join(str(h["title_zh"]) + str(h["summary_zh"]) for h in final)
    assert "甲队" in texts or "乙队" in texts
    assert not [word for word in ("左侧", "右侧", *TEAM_NAMES.values()) if word in texts]
    records = read_measurements(out / "measurements.jsonl")
    attack = [r for r in records if r["metric_id"] in ("attackers-in-box", "body-orientation")]
    assert attack
    assert all(r["status"] == "unavailable" for r in attack)
    assert "attack_direction_unknown" in {r.get("reasonCode") for r in attack}
    parameters = read_json(out / "receipts" / "highlights.json")["result"]["parameters"]
    assert parameters["team_names"] == {"left": "甲队", "right": "乙队"}
    assert parameters["attack_known"] is False


def test_the_refine_note_says_when_no_frame_keeps_a_usable_calibration() -> None:
    shots = pd.DataFrame({"shot_id": [0, 1], "start_frame": [0, 60], "end_frame": [59, 99]})
    plugin = {"frames": 40, "p50_m": 0.31, "p90_m": 0.8694, "p99_m": 2.66}
    report: dict[str, Any] = {
        "counts": {"pitch_usable_before": 90, "pitch_usable_after": 70, "refined_frames": 100},
        "ground_jitter": {"plugin": plugin, "refined": {**plugin, "p90_m": 0.4321}},
        "groups": [{"shots": [0]}, {"shots": [1], "rejected": "held-out P90 30.0 px > 25 px"}],
        "settings": {"holdout_p90_max_px": 25.0},
    }
    assert pipeline_module._refine_note(report, shots) == (
        "标定按同一机位固定相机中心重拟合（插件导出的球场关键点；每帧旋转和焦距平滑）："
        "可用帧 90 → 70；球员位置逐帧抖动（二阶差分，各自可用帧上）P90 0.87 → 0.43 米。"
        "第 60–99 帧的机位的留出关键点误差 P90 超过 25 像素，整组不给米制数值。仍只是自洽检验。"
    )
    # Every group rejected: no frame keeps a usable calibration, so the refit has no jitter
    # (gbf.calib_refine.ground_jitter gives None), and the note says so instead of failing.
    none_left = {"frames": 0, "p50_m": None, "p90_m": None, "p99_m": None}
    rejected = {
        **report,
        "counts": {**report["counts"], "pitch_usable_after": 0},
        "ground_jitter": {"plugin": plugin, "refined": none_left},
        "groups": [{"shots": [0, 1], "rejected": "held-out P90 7391.6 px > 25 px"}],
    }
    assert pipeline_module._refine_note(rejected, shots) == (
        "标定按同一机位固定相机中心重拟合（插件导出的球场关键点；每帧旋转和焦距平滑）："
        "可用帧 90 → 0；球员位置逐帧抖动（二阶差分，各自可用帧上）P90 0.87 米 → 没有可用帧。"
        "第 0–59、60–99 帧的机位的留出关键点误差 P90 超过 25 像素，整组不给米制数值。"
        "仍只是自洽检验。"
    )
    no_refit = {**report, "counts": {**report["counts"], "refined_frames": 0}}
    assert pipeline_module._refine_note(no_refit, shots) == (
        "插件导出了球场关键点，但没有一帧按机位重拟合成功：标定沿用插件的单帧标定。"
    )


def test_the_refine_note_says_which_cameras_fell_back_to_the_plugin_calibration() -> None:
    shots = pd.DataFrame({"shot_id": [0, 1], "start_frame": [0, 60], "end_frame": [59, 99]})
    plugin = {"frames": 40, "p50_m": 0.31, "p90_m": 0.8694, "p99_m": 2.66}
    fell_back = {
        "shots": [1],
        "holdout": {"refined": {"p90_px": 7391.572}, "free_centre": {"p90_px": 30.038}},
        "fallback": "held-out P90 7391.6 px > 2 x 30.0 px of per-frame fits with a free centre",
    }
    report: dict[str, Any] = {
        "counts": {"pitch_usable_before": 90, "pitch_usable_after": 85, "refined_frames": 60},
        "ground_jitter": {"plugin": plugin, "refined": {**plugin, "p90_m": 0.4321}},
        "groups": [{"shots": [0]}, fell_back],
        "settings": {"holdout_p90_max_px": 25.0, "holdout_fallback_ratio": 2.0},
    }
    assert pipeline_module._refine_note(report, shots) == (
        "标定按同一机位固定相机中心重拟合（插件导出的球场关键点；每帧旋转和焦距平滑）："
        "可用帧 90 → 85；球员位置逐帧抖动（二阶差分，各自可用帧上）P90 0.87 → 0.43 米。"
        "第 60–99 帧的机位重拟合后留出关键点误差 P90 为 7391.6 像素，超过单帧拟合（30.0 像素）"
        "的 2 倍，整组改用插件的单帧标定。仍只是自洽检验。"
    )
    # Every group fell back: no frame is refitted, and the note says why.
    alone = {
        **report,
        "counts": {**report["counts"], "refined_frames": 0},
        "groups": [{**fell_back, "shots": [0, 1]}],
    }
    assert pipeline_module._refine_note(alone, shots) == (
        "插件导出了球场关键点，但没有一帧按机位重拟合成功：标定沿用插件的单帧标定。"
        "第 0–59、60–99 帧的机位重拟合后留出关键点误差 P90 为 7391.6 像素，超过单帧拟合"
        "（30.0 像素）的 2 倍，整组改用插件的单帧标定。"
    )


def test_the_refine_note_says_which_cameras_did_not_converge() -> None:
    shots = pd.DataFrame({"shot_id": [0, 1], "start_frame": [0, 60], "end_frame": [59, 99]})
    plugin = {"frames": 40, "p50_m": 0.31, "p90_m": 0.8694, "p99_m": 2.66}
    converged = {"shots": [0], "solver_status": 2, "solver_nfev": 143, "converged": True}
    stopped = {"shots": [1], "solver_status": 0, "solver_nfev": 2000, "converged": False}
    # A group whose refit did not converge keeps the plugin's calibration (gbf.calib_refine),
    # even when its held-out error looked fine; the note must not give the held-out reason.
    report: dict[str, Any] = {
        "counts": {"pitch_usable_before": 90, "pitch_usable_after": 85, "refined_frames": 60},
        "ground_jitter": {"plugin": plugin, "refined": {**plugin, "p90_m": 0.4321}},
        "groups": [
            converged,
            {
                **stopped,
                "holdout": {"refined": {"p90_px": 12.697}, "free_centre": {"p90_px": 13.988}},
                "fallback": "the refit stopped at 2000 evaluations without converging",
            },
        ],
        "settings": {"holdout_p90_max_px": 25.0, "holdout_fallback_ratio": 2.0},
    }
    assert pipeline_module._refine_note(report, shots) == (
        "标定按同一机位固定相机中心重拟合（插件导出的球场关键点；每帧旋转和焦距平滑）："
        "可用帧 90 → 85；球员位置逐帧抖动（二阶差分，各自可用帧上）P90 0.87 → 0.43 米。"
        "第 60–99 帧的机位重拟合用满 2000 次求解仍未收敛，整组改用插件的单帧标定。"
        "仍只是自洽检验。"
    )
    # A rejected group is still refitted (it gives no metric values): its note says both.
    rejected = {
        **report,
        "counts": {**report["counts"], "refined_frames": 100},
        "groups": [converged, {**stopped, "rejected": "held-out P90 30.0 px > 25 px"}],
    }
    assert pipeline_module._refine_note(rejected, shots) == (
        "标定按同一机位固定相机中心重拟合（插件导出的球场关键点；每帧旋转和焦距平滑）："
        "可用帧 90 → 85；球员位置逐帧抖动（二阶差分，各自可用帧上）P90 0.87 → 0.43 米。"
        "第 60–99 帧的机位重拟合用满 2000 次求解仍未收敛：这组标定可能还没到最好，"
        "换机器或线程数，结果也可能不同。"
        "第 60–99 帧的机位的留出关键点误差 P90 超过 25 像素，整组不给米制数值。仍只是自洽检验。"
    )


def test_the_source_fingerprint_covers_every_module_but_the_gate_and_the_web_page(
    synthetic_run: dict[str, Any], tmp_path: Path
) -> None:
    package = Path(pipeline_module.__file__).resolve().parent
    fingerprint = pipeline_module._source_fingerprint
    # Every receipt and run.json name the source they ran.
    out = synthetic_run["out"]
    assert read_json(out / "run.json")["result"]["source_sha256"] == fingerprint(package)
    assert read_json(out / "receipts" / "report.json")["result"]["source_sha256"] == (
        fingerprint(package)
    )
    copy = tmp_path / "源 copy" / "gbf"
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert fingerprint(copy) == fingerprint(package), "paths inside the package and bytes"
    before = fingerprint(copy)
    helper = copy / "geometry" / "ground.py"  # a helper no stage names
    helper.write_bytes(helper.read_bytes() + b"# edited\n")
    edited = fingerprint(copy)
    assert edited != before
    for left_out in pipeline_module.SOURCE_LEFT_OUT:
        (copy / left_out).mkdir(exist_ok=True)
        (copy / left_out / "page.py").write_bytes(b"VALUE = 1\n")
    (copy / "notes.txt").write_bytes(b"not code\n")
    assert fingerprint(copy) == edited, "the gate, the web page and non-code files stay out"
    (copy / "new_helper.py").write_bytes(b"VALUE = 2\n")
    assert fingerprint(copy) != edited


_FROM_COPY = """
import json
import sys
from pathlib import Path

from gbf.pipeline import PipelineConfig, run_pipeline

args = json.loads(sys.argv[1])
result = run_pipeline(
    PipelineConfig(
        video=Path(args["video"]),
        audio_source=Path(args["audio"]),
        sngs_dir=Path(args["sngs"]),
        ball_raw=Path(args["ball"]),
        out_dir=Path(args["out"]),
        team_names=args["team_names"],
        pose_model="none",
        stages=("report",),
        allow_licenses=("GPL-3.0-only",),
        repo_root=Path(args["base"]) / "not-the-repo",
        git_ceiling=Path(args["base"]),
        log=lambda text: None,
    )
)
print(json.dumps({stage.name: stage.status for stage in result.stages}))
"""
"""Runs the report stage of a run (the other stages kept) with the gbf on PYTHONPATH."""


def test_editing_a_helper_of_a_stage_reruns_it(
    synthetic_run: dict[str, Any], tmp_path: Path
) -> None:
    # The pipeline runs, as its own process, from a scratch copy of the gbf package on a
    # copy of the synthetic run. Unedited, the copy keeps the report. Then a helper the
    # report stage uses without naming it in its stage entry is edited (gbf.render's label
    # of the facing arrow, which the report's notes print): the stage must re-run, with the
    # edit, instead of keeping the report of the older code.
    run = tmp_path / "run"
    shutil.copytree(synthetic_run["out"], run)
    source = tmp_path / "src"
    package = Path(pipeline_module.__file__).resolve().parent
    shutil.copytree(package, source / "gbf", ignore=shutil.ignore_patterns("__pycache__"))
    inputs = synthetic_run["inputs"]
    args = {
        "video": str(inputs["video"]),
        "audio": str(inputs["audio"]),
        "sngs": str(inputs["sngs"]),
        "ball": str(inputs["ball"]),
        "out": str(run),
        "team_names": TEAM_NAMES,
        "base": str(tmp_path),
    }
    env = {**os.environ, "PYTHONPATH": str(source), "PYTHONUTF8": "1"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    def report_stage() -> str:
        done = subprocess.run(
            [sys.executable, "-c", _FROM_COPY, json.dumps(args)],
            capture_output=True,
            env=env,
            timeout=600,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        return json.loads(done.stdout.decode("utf-8").strip().splitlines()[-1])["report"]

    assert report_stage() == "skipped"
    render = source / "gbf" / "render.py"
    old = 'FACING_LABEL_ZH = "朝向（3D 参考）"'.encode()
    assert render.read_bytes().count(old) == 1
    render.write_bytes(
        render.read_bytes().replace(old, 'FACING_LABEL_ZH = "朝向（3D 参考，改）"'.encode())
    )
    assert report_stage() == "ran"
    assert "朝向（3D 参考，改）" in (run / "report.html").read_text(encoding="utf-8")


def test_measure_reruns_when_the_metric_registry_changes(
    synthetic_run: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # gbf.measure reads the metric tiers from registries/metrics.toml: an edit there re-runs
    # the stage even where the code and the git state stay the same (an already dirty tree).
    run = tmp_path / "run"
    shutil.copytree(synthetic_run["out"], run)
    registry = tmp_path / "registry" / "metrics.toml"
    registry.parent.mkdir()
    shutil.copyfile(measure_module.default_registry_path(), registry)
    monkeypatch.setattr(measure_module, "default_registry_path", lambda: registry)
    config = _config(synthetic_run["inputs"], run, synthetic_run["base"], stages=("measure",))

    def status() -> str:
        return {stage.name: stage.status for stage in run_pipeline(config).stages}["measure"]

    assert status() == "skipped"
    registry.write_bytes(registry.read_bytes() + b"\n# edited\n")
    assert status() == "ran"
    inputs = read_json(run / "receipts" / "measure.json")["inputs"]
    entry = next(item for item in inputs if item["name"] == "registries/metrics.toml")
    assert entry["sha256"] == sha256_file(registry)
    assert "relpath" not in entry, "the registry lies outside the run folder"
