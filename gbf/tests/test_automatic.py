"""Synthetic coverage of video-only preparation, local uploads and process cancellation."""

from __future__ import annotations

import http.client
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from gbf.ingest import FFmpegError, probe
from gbf.synth.video import write_frames_mp4
from gbf.util.io import read_json, write_json, write_text
from gbf.web.automatic import (
    AUTO_STAGES,
    MAX_UPLOAD_BYTES,
    RuntimeConfig,
    prepare_video,
    run,
    runtime_status,
)
from gbf.web.jobs import JobManager, JobRefused, JobRequest, JobRequestError, child_environment
from gbf.web.server import WebServer


@pytest.fixture
def config(unicode_tmp_path: Path) -> RuntimeConfig:
    path = unicode_tmp_path / "runtime.json"
    models = unicode_tmp_path / "models"
    models.mkdir()
    write_json(path, {"command": [sys.executable], "models_dir": str(models)})
    return RuntimeConfig.load(path)


def _video(folder: Path) -> Path:
    path = folder / "input.mp4"
    frames = [np.full((90, 160, 3), i * 20, dtype=np.uint8) for i in range(3)]
    write_frames_mp4(frames, path, fps="25/1")
    return path


def test_automatic_request_needs_only_video_and_output(unicode_tmp_path: Path) -> None:
    video = unicode_tmp_path / "video.mp4"
    video.touch()
    request = JobRequest.from_json({"mode": "auto", "video": str(video), "out_name": "new"})
    assert request.mode == "auto"
    assert request.sngs_dir is None
    assert request.ball is None
    assert request.as_json()["sngs_dir"] is None
    assert request.audio_source == video
    with pytest.raises(JobRequestError, match="必填"):
        JobRequest.from_json({"video": str(video), "out_name": "manual"})
    with pytest.raises(JobRequestError, match="自动模式不接受"):
        JobRequest.from_json(
            {"mode": "auto", "video": str(video), "out_name": "new", "labels": str(video)}
        )


def test_runtime_reports_missing_or_invalid_configuration(config: RuntimeConfig) -> None:
    assert not runtime_status({})["available"]
    assert runtime_status({"GBF_AUTO_CONFIG": str(config.path)})["available"]
    write_json(config.path, {"command": "shell text"})
    assert not runtime_status({"GBF_AUTO_CONFIG": str(config.path)})["available"]


def test_child_environment_drops_cross_interpreter_launcher() -> None:
    env = child_environment({"__PYVENV_LAUNCHER__": "other-python", "PYTHONHOME": "other-home"})
    assert "__PYVENV_LAUNCHER__" not in env
    assert "PYTHONHOME" not in env
    assert env["PYTHONUTF8"] == "1"


def test_prepare_normalizes_and_preserves_three_distinct_frames(unicode_tmp_path: Path) -> None:
    from PIL import Image

    video = _video(unicode_tmp_path)
    work = unicode_tmp_path / "work"
    work.mkdir()
    record = prepare_video(video, work, 60)
    normalized = probe(Path(record["video"]))
    assert normalized.is_cfr_at("25/1")
    assert (normalized.width, normalized.height, normalized.nb_frames_decoded) == (1920, 1080, 3)
    assert [row["index"] for row in record["frame_hashes"]] == [0, 1, 2]
    assert len({row["rgb_sha256"] for row in record["frame_hashes"]}) == 3
    means = []
    for path in sorted(Path(record["frames_dir"]).glob("*.png")):
        with Image.open(path) as image:
            means.append(np.asarray(image).mean())
    assert means[0] < means[1] < means[2]
    assert record["source_probe"]["sha256"] == probe(video).sha256


def test_invalid_media_and_overlong_input_fail_before_predictions(
    unicode_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = unicode_tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(FFmpegError, match="Invalid data"):
        prepare_video(bad, unicode_tmp_path, 60)
    from dataclasses import replace

    video = _video(unicode_tmp_path)
    facts = replace(probe(video), duration_s=61.0)
    monkeypatch.setattr("gbf.web.automatic.probe", lambda _: facts)
    with pytest.raises(ValueError, match="60 秒以内"):
        prepare_video(video, unicode_tmp_path, 60)


def test_video_only_orchestration_calls_external_predictor_and_core(
    config: RuntimeConfig, unicode_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = unicode_tmp_path / "predict.py"
    write_text(
        script,
        """import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
data=json.loads(Path(a.input).read_text());out=Path(a.output);out.mkdir()
for name in ('detections.jsonl','calib.jsonl','plugin-receipt.json',
             'ball_raw.jsonl','ball-receipt.json'):
    (out/name).write_text('{}')
(out/'frame-verification.json').write_text(json.dumps({'passed':True,'frames':data['frames']}))
""",
    )
    write_json(
        config.path,
        {"command": [sys.executable, str(script)], "models_dir": str(config.models_dir)},
    )
    config = RuntimeConfig.load(config.path)
    called = []

    def core(args: list[str]) -> int:
        called.extend(args)
        out = Path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--out=")))
        out.mkdir()
        write_json(out / "run.json", {"status": "done"})
        return 0

    monkeypatch.setattr("gbf.pipeline.main", core)
    out = unicode_tmp_path / "runs" / "fresh"
    out.parent.mkdir()
    assert run(config, _video(unicode_tmp_path), out, {}) == 0
    assert any(arg.startswith("--sngs=") for arg in called)
    assert read_json(out / "automatic.json")["accuracy_evaluated"] is False
    with pytest.raises(ValueError, match="已经存在"):
        run(config, unicode_tmp_path / "input.mp4", out, {})


def test_prediction_failure_never_runs_core(
    config: RuntimeConfig, unicode_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    config = replace(config, command=(sys.executable, "-c", "raise SystemExit(7)"))
    monkeypatch.setattr("gbf.pipeline.main", lambda _: pytest.fail("core must not run"))
    out = unicode_tmp_path / "runs" / "failed"
    out.parent.mkdir()
    assert run(config, _video(unicode_tmp_path), out, {}) == 1
    assert not out.exists()


class LocalWeb:
    def __init__(self, root: Path, config: RuntimeConfig) -> None:
        root.mkdir()
        self.manager = JobManager(root, environ={"GBF_AUTO_CONFIG": str(config.path)})
        self.server = WebServer(root, jobs=self.manager, log=lambda _: None)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def request(self, path: str, body: bytes, **changes: str) -> tuple[int, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=10)
        headers = {"X-GBF-Token": self.server.token, "Content-Type": "application/octet-stream"}
        headers.update(changes)
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def close(self) -> None:
        self.manager.close(grace_s=0)
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


def test_upload_is_local_bounded_authenticated_and_server_named(
    config: RuntimeConfig, unicode_tmp_path: Path
) -> None:
    web = LocalWeb(unicode_tmp_path / "runs", config)
    try:
        body = b"synthetic upload bytes"
        status, data = web.request("/api/uploads", body)
        assert status == 201
        saved = Path(data["video"])
        assert saved.read_bytes() == body
        assert saved.parent == web.manager.runs_dir / ".gbf-web" / "uploads"
        assert not list(saved.parent.glob("*.partial"))
        assert web.request("/api/uploads", body, **{"X-GBF-Token": "wrong"})[0] == 403
        assert web.request("/api/uploads", body, Origin="https://untrusted.invalid")[0] == 403
        assert web.request("/api/uploads", b"")[0] == 413
        assert (
            web.request("/api/uploads", b"", **{"Content-Length": str(MAX_UPLOAD_BYTES + 1)})[0]
            == 413
        )
        assert web.request("/api/uploads", body, **{"Content-Type": "text/plain"})[0] == 415
        assert len(list(saved.parent.glob("*.mp4"))) == 1
    finally:
        web.close()


def test_cancel_ends_nested_processes_and_persists_interruption(
    config: RuntimeConfig, unicode_tmp_path: Path
) -> None:
    folder = unicode_tmp_path / "runs"
    folder.mkdir()
    script = unicode_tmp_path / "parent.py"
    pulse = unicode_tmp_path / "pulse.txt"
    child = "import time; from pathlib import Path; p=Path(" + repr(str(pulse)) + "); "
    child += "\nwhile True:\n p.write_text(str(time.time())); time.sleep(.02)\n"
    write_text(
        script,
        "import subprocess, sys, time\nsubprocess.Popen([sys.executable,'-c',"
        + repr(child)
        + "])\nprint('[tracking] running',flush=True)\ntime.sleep(60)\n",
    )
    manager = JobManager(folder, command=[sys.executable, str(script)])
    video = unicode_tmp_path / "input.mp4"
    video.touch()
    request = JobRequest(video, video, unicode_tmp_path, video, "nested")
    try:
        job = manager.start(request)
        deadline = time.monotonic() + 15
        while not pulse.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pulse.exists()
        manager.cancel(job.id)
        assert job.status == "interrupted"
        assert manager.current() is None
        last = pulse.read_text()
        time.sleep(0.15)
        assert pulse.read_text() == last
        reloaded = JobManager(folder).get(job.id)
        assert reloaded is not None
        assert reloaded.status == "interrupted"
    finally:
        manager.close(grace_s=0)


def test_automatic_progress_and_reload_include_prediction_stages(
    config: RuntimeConfig, unicode_tmp_path: Path
) -> None:
    from gbf.web.jobs import Job

    video = unicode_tmp_path / "input.mp4"
    video.touch()
    request = JobRequest.from_json({"mode": "auto", "video": str(video), "out_name": "new"})
    log = unicode_tmp_path / "log.txt"
    write_text(log, "[prepare] running\n[prepare] done\n[tracking] running\n")
    job = Job("20260101-000000-abcdef", request, log, log.with_suffix(".json"), (), "now")
    job.known_status = "running"
    snapshot = job.snapshot(unicode_tmp_path)
    assert snapshot["current_stage"] == "tracking"
    assert snapshot["stages"][0]["state"] == "done"
    assert {row["name"] for row in snapshot["stages"]} >= set(AUTO_STAGES)
    write_json(job.record_path, job.record())
    folder = unicode_tmp_path / "runs"
    record_dir = folder / ".gbf-web" / "jobs"
    record_dir.mkdir(parents=True)
    write_json(record_dir / (job.id + ".json"), job.record())
    loaded = JobManager(folder).get(job.id)
    assert loaded is not None
    assert loaded.request.mode == "auto"
    assert loaded.status == "unknown"


def test_auto_is_refused_when_runtime_unavailable(unicode_tmp_path: Path) -> None:
    video = unicode_tmp_path / "input.mp4"
    video.touch()
    folder = unicode_tmp_path / "runs"
    folder.mkdir()
    request = JobRequest.from_json({"mode": "auto", "video": str(video), "out_name": "new"})
    with pytest.raises(JobRefused, match="尚未配置"):
        JobManager(folder, environ={}).start(request)
    assert not (folder / "new").exists()


def test_short_upload_is_removed_and_cancel_endpoint_requires_token(
    config: RuntimeConfig, unicode_tmp_path: Path
) -> None:
    web = LocalWeb(unicode_tmp_path / "runs", config)
    try:
        connection = http.client.HTTPConnection("127.0.0.1", web.server.port, timeout=5)
        connection.putrequest("POST", "/api/uploads")
        connection.putheader("X-GBF-Token", web.server.token)
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", "20")
        connection.endheaders(b"short")
        connection.sock.shutdown(1)
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        connection.close()
        uploads = web.manager.runs_dir / ".gbf-web" / "uploads"
        assert not list(uploads.iterdir())
        endpoint = "/api/jobs/20260101-000000-abcdef/cancel"
        assert (
            web.request(
                endpoint, b"{}", **{"Content-Type": "application/json", "X-GBF-Token": "wrong"}
            )[0]
            == 403
        )
        assert web.request(endpoint, b"{}", **{"Content-Type": "application/json"})[0] == 404
    finally:
        web.close()


def test_coordinator_crash_cleans_remaining_child(unicode_tmp_path: Path) -> None:
    folder = unicode_tmp_path / "runs"
    folder.mkdir()
    pulse = unicode_tmp_path / "crash-child.txt"
    script = unicode_tmp_path / "crash.py"
    child = "import time; from pathlib import Path; p=Path(" + repr(str(pulse)) + "); "
    child += "\nwhile True:\n p.write_text(str(time.time())); time.sleep(.02)\n"
    write_text(
        script,
        "import subprocess, sys, time, os\nsubprocess.Popen([sys.executable,'-c',"
        + repr(child)
        + "])\ntime.sleep(.5)\nos._exit(1)\n",
    )
    manager = JobManager(folder, command=[sys.executable, str(script)])
    video = unicode_tmp_path / "video.mp4"
    video.touch()
    try:
        job = manager.start(JobRequest(video, video, unicode_tmp_path, video, "crashed"))
        job.waiter.join(timeout=15)
        assert job.status == "failed"
        assert pulse.exists()
        before = pulse.read_text()
        time.sleep(0.15)
        assert pulse.read_text() == before
    finally:
        manager.close(grace_s=0)
