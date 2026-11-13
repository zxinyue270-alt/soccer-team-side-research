"""Tests for gbf.web: the local web page's server, its security rules, runs and jobs.

Everything is synthetic: a runs folder written by the tests (JSON documents, a tiny mp4
from gbf.synth.video, a PNG from Pillow), a server on a free port of 127.0.0.1 in a thread,
and a fake command in place of ``python -m gbf highlights``.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import socket
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pytest
from PIL import Image

import gbf
from gbf import __main__ as cli
from gbf.contracts import load_schema
from gbf.gate.content import check_abs_paths, check_credentials, check_eol
from gbf.gate.datafiles import check_media, check_restricted_data
from gbf.gate.imports import check_forbidden_imports
from gbf.gate.repo import Workspace
from gbf.measure import METRICS
from gbf.pipeline import STAGES, WATERMARK
from gbf.synth.video import write_frames_mp4
from gbf.util.io import write_json, write_text
from gbf.util.paths import repo_root
from gbf.web import jobs as jobs_mod
from gbf.web import server as server_mod
from gbf.web.jobs import JobManager, JobRefused, JobRequest, JobRequestError
from gbf.web.runs import (
    LIFT3D_METRICS,
    RunPathError,
    full_clip_seeker,
    measurement_row,
    resolve_artifact,
    segment_problem,
)
from gbf.web.server import WebServer, parse_range, runs_dir_from

FPS = "25/1"
RUN = "demo-run"
SECRET = "outside-the-run-marker"
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
    "X-Frame-Options": "DENY",
}
"""Written out here, not read from gbf.web.server, so that a header dropped there fails."""


# --------------------------------------------------------------------------- the runs folder


def _record(highlight: str, metric: str, tier: str, value: Any, unit: str, **extra: Any) -> dict:
    plane = extra.pop("plane", "pitch")
    record = {
        "claimType": "measurement",
        "evidenceId": f"{highlight}-{metric}",
        "highlight_id": highlight,
        "metric_id": metric,
        "label_zh": metric,
        "tier": tier,
        "value": value,
        "unit": unit,
        "plane": plane,
        "status": "ok" if value is not None else "unavailable",
        "error": {"kind": "unmeasured"},
        "frame": 12,
        "track_id": 3,
        "restrictions": ["no_share"],
    }
    record.update(extra)
    return record


def _write_run(folder: Path, *, labels: bool = True, created: str = "2026-01-02T03:04:05Z") -> None:
    """A small run as gbf.pipeline writes it (only the documents the page reads)."""
    inputs = [{"name": "video", "bytes": 1, "sha256": "ab" * 32}]
    if labels:
        inputs.append({"name": "labels", "bytes": 1, "sha256": "cd" * 32})
    write_json(
        folder / "run.json",
        {
            "kind": "run",
            "status": "passed",
            "created_at": created,
            "inputs": inputs,
            "restrictions": ["no_share", "research_nc"],
            "result": {
                "video": {
                    "file_name": "synthetic clip.mp4",
                    "fps": FPS,
                    "frames": 30,
                    "width": 64,
                    "height": 48,
                },
                "audio_source": "synthetic clip.mp4",
                "notes_zh": ["合成数据的说明。"],
                "error": None,
                "parameters": {"team_names": {"left": "左队", "right": "右队"}},
                "stages": [{"name": "import", "status": "ran", "wall_s": 0.5}],
                "wall_s": 12.5,
            },
        },
    )
    highlights = [
        {
            "id": "h02",
            "type": "duel_tackle",
            "start_frame": 10,
            "peak_frame": 20,
            "end_frame": 29,
            "score": 0.31,
            "rank": 1,
            "tier": "main",
            "title_zh": "第二个高光",
            "summary_zh": "合成的摘要。",
            "refined_by": "duel_min_distance",
            "key_moments": [
                {"frame": 18, "kind": "touch", "text_zh": "触球", "track_ids": [3]},
                {"frame": 21, "kind": "contact_start", "text_zh": "身体贴近", "track_ids": [3, 4]},
            ],
            "touches": [{"label_zh": "左队球员", "count": 2, "mean_interval_s": 0.4}],
        },
        {
            "id": "h01",
            "type": "sprint",
            "start_frame": 0,
            "peak_frame": 5,
            "end_frame": 9,
            "score": 0.2,
            "rank": 2,
            "tier": "candidate",
            "title_zh": "第一个高光",
            "summary_zh": "另一个摘要。",
            "key_moments": [],
        },
    ]
    write_json(
        folder / "highlights_final.json",
        {"fps": FPS, "frame_count": 30, "highlights": highlights, "restrictions": ["no_share"]},
    )
    records = [
        _record("h02", "peak-speed", "A", 7.5, "m/s"),
        _record("h02", "body-orientation", "A_candidate", 40.0, "deg", decimals=0),
        _record("h02", "image-trunk-lean", "image_plane", 12.0, "deg", plane="image"),
        _record("h02", "trunk-lean-at-contact", "A_candidate", None, "deg", reasonCode="no_pose"),
        _record("h02", "foot-speed", "C", 20.0, "m/s"),
        _record("h01", "nearest-opponent-distance", "A", 1.2, "m", frame=999),
    ]
    write_text(folder / "measurements.jsonl", "".join(json.dumps(r) + "\n" for r in records))
    write_json(
        folder / "label_eval.json",
        {
            "n_labels": 2,
            "n_highlights": 2,
            "n_matched": 1,
            "n_peak_within_tolerance": 1,
            "n_type_agrees": 1,
            "rows": [
                {
                    "label": "M1",
                    "label_type": "duel_tackle",
                    "label_peak": 21,
                    "matched": True,
                    "highlight": "h02",
                    "peak": 20,
                    "peak_error_frames": -1,
                    "tolerance_frames": 6,
                    "peak_within_tolerance": True,
                    "type_agrees": True,
                    "window_iou": 0.5,
                },
                {"label": "M2", "label_type": "sprint", "label_peak": 3, "matched": False},
            ],
            "summary_zh": ["标注 M1 → 检出 h02，duel_tackle 一致。"],
            "source_zh": "标注来源（照录标注文件）：标注者“示例”；盲法“未说明”。",
            "labels_source": {"labelled_by": "示例", "blinding": "未说明"},
            "unmatched_highlights": [{"highlight": "h01"}],
            "restrictions": ["no_share"],
        },
    )
    write_text(folder / "report.html", "<!doctype html><title>report</title><p>synthetic</p>\n")
    keyframes = folder / "keyframes"
    keyframes.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), (200, 30, 30)).save(keyframes / "h02.png")
    write_json(
        folder / "clips" / "full.render.json",
        {
            "output": {"fps": FPS, "frames": 30},
            "segments": [
                {"kind": "normal", "src_start": 0, "src_end": 29, "out_start": 0, "repeat": 1}
            ],
        },
    )
    (folder / "pose.parquet").write_bytes(b"not served")
    write_text(folder / ".hidden.json", "{}\n")


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory, ffmpeg_bin: str) -> Path:
    """A tiny mp4 (30 frames, 64x48) written once for the module."""
    folder = tmp_path_factory.mktemp("web-media")
    frames = [np.full((48, 64, 3), (i * 8) % 256, dtype=np.uint8) for i in range(30)]
    write_frames_mp4(frames, folder / "tiny.mp4", fps=FPS)
    return folder / "tiny.mp4"


@pytest.fixture
def runs_root(unicode_tmp_path: Path, media: Path) -> Path:
    """``<tmp>/runs`` with the run ``demo-run``, a folder without run.json, a hidden run, and a
    secret file outside the runs folder."""
    root = unicode_tmp_path / "runs"
    run = root / RUN
    _write_run(run)
    shutil.copyfile(media, run / "clips" / "full.mp4")
    shutil.copyfile(media, run / "clips" / "h02.mp4")
    (root / "not-a-run").mkdir()
    write_text(root / "not-a-run" / "notes.json", "{}\n")
    _write_run(root / ".hidden-run")
    outside = unicode_tmp_path / "outside"
    outside.mkdir()
    write_text(outside / "secret.json", json.dumps({"marker": SECRET}) + "\n")
    _write_run(outside / "elsewhere-run")
    return root


def _link_dir(link: Path, target: Path) -> None:
    """A directory link: a symbolic link, or on Windows without that privilege a junction."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


def _unlink_dir(link: Path) -> None:
    """Remove a link made by :func:`_link_dir` (never what it points to)."""
    if os.name == "nt":
        os.rmdir(link)
    else:
        link.unlink()


def _short_name(path: Path) -> str | None:
    """The Windows 8.3 short name of ``path`` (``HIDDEN~1`` for ``.hidden-run``) when the
    volume makes one, else None."""
    if os.name != "nt":
        return None
    import ctypes

    buffer = ctypes.create_unicode_buffer(1024)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer)):
        return None
    short = Path(buffer.value).name
    return short if short != path.name else None


def _record_file_system(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The paths Path.is_file and Path.is_dir are asked about from now on; they answer False
    without asking the file system."""
    asked: list[str] = []

    def record(self: Path) -> bool:
        asked.append(str(self))
        return False

    monkeypatch.setattr(Path, "is_file", record)
    monkeypatch.setattr(Path, "is_dir", record)
    return asked


# --------------------------------------------------------------------------- the fake command

FAKE_COMMAND = r"""
import json, os, sys, time, pathlib

args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
control = pathlib.Path(os.environ["GBF_WEB_FAKE_CONTROL"])
settings = json.loads(control.read_text(encoding="utf-8"))
(control.parent / "argv.json").write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
mode = settings["mode"]
print("video: probing synthetic ...", flush=True)
if mode == "refused":
    print("highlights: 1 problem(s)", file=sys.stderr)
    print("  - plugin input (plugin-receipt.json): license 'GPL-3.0-only' is not satisfied by "
          "the allowed licenses (not allowed: GPL-3.0-only); accept it explicitly with "
          "--allow-license <id>", file=sys.stderr, flush=True)
    sys.exit(2)
out = pathlib.Path(args["out"])  # like run_pipeline: the run folder exists once stages start
out.mkdir(parents=True, exist_ok=True)


def run_json(status):
    document = {"status": status, "created_at": "2026-01-03T00:00:00Z",
                "restrictions": ["no_share"], "result": {}}
    (out / "run.json").write_text(json.dumps(document), encoding="utf-8")


print("[import] running ...", flush=True)
print("[import] done in 0.1 s", flush=True)
print("[shots] skipped (inputs, parameters and code unchanged)", flush=True)
print("[refine] running ...", flush=True)
if mode == "failed":
    run_json("failed")
    print("highlights: failed: refine: 1 problem(s)", file=sys.stderr)
    print("  - the plugin export covers 10 frames but the video decodes to 12; the export "
          "does not belong to this canonical video", file=sys.stderr, flush=True)
    sys.exit(1)
release = pathlib.Path(settings.get("release", ""))
deadline = time.monotonic() + 60
while settings.get("release") and not release.exists() and time.monotonic() < deadline:
    time.sleep(0.05)
print("[refine] done in 1.0 s", flush=True)
run_json("passed")
print("highlights:", flush=True)
"""


@pytest.fixture
def fake(unicode_tmp_path: Path) -> dict[str, Path]:
    folder = unicode_tmp_path / "fake"
    folder.mkdir()
    script = folder / "fake_highlights.py"
    write_text(script, FAKE_COMMAND)
    return {"script": script, "control": folder / "control.json", "folder": folder}


def _control(fake: Mapping[str, Path], mode: str, *, release: Path | None = None) -> None:
    settings: dict[str, Any] = {"mode": mode}
    if release is not None:
        settings["release"] = str(release)
    write_json(fake["control"], settings)


def _manager(runs: Path, fake: Mapping[str, Path], ceiling: Path) -> JobManager:
    return JobManager(
        runs,
        command=[sys.executable, str(fake["script"])],
        environ={**os.environ, "GBF_WEB_FAKE_CONTROL": str(fake["control"])},
        repo_root=ceiling / "not-the-repo",
        git_ceiling=ceiling,
    )


# --------------------------------------------------------------------------- the server


class Running:
    """A WebServer serving in a thread, with a small HTTP client."""

    def __init__(self, server: WebServer) -> None:
        self.server = server
        self.logs: list[str] = []
        server.log = self.logs.append
        self.thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.thread.start()

    @property
    def host(self) -> str:
        return f"127.0.0.1:{self.server.port}"

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        token: bool = True,
        host: str | None = None,
    ) -> tuple[int, http.client.HTTPMessage, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=30)
        try:
            conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", host if host is not None else self.host)
            if token:
                conn.putheader(server_mod.TOKEN_HEADER, self.server.token)
            for name, value in (headers or {}).items():
                conn.putheader(name, value)
            if body is not None:
                conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(body)
            response = conn.getresponse()
            return response.status, response.headers, response.read()
        finally:
            conn.close()

    def get_json(self, path: str, **kwargs: Any) -> Any:
        status, _, body = self.request("GET", path, **kwargs)
        assert status == 200, body
        return json.loads(body)

    def post_job(
        self, payload: Any, *, headers: Mapping[str, str] | None = None
    ) -> tuple[int, Any]:
        status, _, body = self.request(
            "POST",
            "/api/jobs",
            headers={"Content-Type": "application/json", **(headers or {})},
            body=json.dumps(payload).encode("utf-8"),
        )
        return status, json.loads(body)

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server.jobs.close(grace_s=0)
        self.thread.join(timeout=10)


@pytest.fixture
def web(runs_root: Path, fake: dict[str, Path], unicode_tmp_path: Path) -> Iterator[Running]:
    manager = _manager(runs_root, fake, unicode_tmp_path)
    running = Running(WebServer(runs_root, jobs=manager))
    yield running
    running.stop()


def _job_request(runs: Path, media: Path, out_name: str, **changes: Any) -> dict[str, Any]:
    inputs = runs.parent / "inputs"
    (inputs / "sngs").mkdir(parents=True, exist_ok=True)
    write_text(inputs / "ball_raw.jsonl", "")
    payload: dict[str, Any] = {
        "video": str(media),
        "sngs_dir": str(inputs / "sngs"),
        "ball": str(inputs / "ball_raw.jsonl"),
        "out_name": out_name,
    }
    payload.update(changes)
    return payload


def _wait_for(web: Running, job_id: str, *, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = web.get_json(f"/api/jobs/{job_id}")
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still running after {timeout} s")


def _late(body: bytes, delay: float = 0.2) -> Iterator[bytes]:
    """A request body sent ``delay`` seconds after the headers (http.client sends the body
    of a request as a second write; a busy client may send it later still)."""
    time.sleep(delay)
    yield body


def _answer(sock: socket.socket, method: str = "GET") -> http.client.HTTPResponse:
    """The next answer on the raw connection ``sock`` (headers read, body not yet)."""
    response = http.client.HTTPResponse(sock, method=method)
    response.begin()
    return response


# --------------------------------------------------------------------------- listing and detail


def test_run_list_has_the_runs_newest_first_and_nothing_else(web: Running, runs_root: Path) -> None:
    _write_run(runs_root / "older-run", created="2025-12-31T00:00:00Z")
    data = web.get_json("/api/runs")
    assert data["runs_dir"] == str(runs_root.resolve())
    assert [run["name"] for run in data["runs"]] == [RUN, "older-run"]
    run = data["runs"][0]
    assert run["status"] == "passed"
    assert run["created_at"] == "2026-01-02T03:04:05Z"
    assert run["video"] == "synthetic clip.mp4"
    assert (run["frames"], run["fps"]) == (30, FPS)
    assert run["highlight_count"] == 2
    assert run["restrictions"] == ["no_share", "research_nc"]
    assert run["banner_zh"] == WATERMARK


def test_run_detail_shows_highlights_on_the_original_frames(web: Running) -> None:
    detail = web.get_json(f"/api/runs/{RUN}")
    assert detail["fps"] == FPS
    assert detail["frame_count"] == 30
    assert detail["restrictions"] == ["no_share", "research_nc"]
    assert detail["banner_zh"] == WATERMARK
    assert detail["run"]["notes_zh"] == ["合成数据的说明。"]
    assert detail["run"]["team_names"] == {"left": "左队", "right": "右队"}
    assert detail["report"] == "report.html"
    assert detail["full_clip"] == {"path": "clips/full.mp4", "mapping": "manifest"}
    files = {item["path"] for item in detail["files"]}
    assert {"run.json", "report.html", "clips/full.mp4", "keyframes/h02.png"} <= files
    assert "pose.parquet" not in files
    assert ".hidden.json" not in files
    assert detail["highlights_final"]["frame_count"] == 30
    # H1, H2 ... in peak order, as in the report
    first, second = detail["highlights"]
    assert (first["id"], first["name"], second["id"], second["name"]) == ("h01", "H1", "h02", "H2")
    assert second["type_zh"] == "贴身争抢"
    assert second["tier_zh"] == "主要高光"
    assert (second["peak_clock"], second["start_clock"]) == ("00:00.80", "00:00.40")
    assert second["key_frame_rule_zh"] == "球在两人脚下时距离最近的一帧"
    assert (second["keyframe"], second["clip"]) == ("keyframes/h02.png", "clips/h02.mp4")
    assert first["keyframe"] is None
    assert first["clip"] is None
    # the full clip runs 1:1 with the source: frame f plays at the middle of its frame
    assert second["peak_seek_s"] == pytest.approx(20.5 / 25)
    assert [(m["frame"], m["kind_zh"], m["seek_s"]) for m in second["moments"]] == [
        (18, "触球", pytest.approx(18.5 / 25)),
        (21, "身体贴近", pytest.approx(21.5 / 25)),
    ]
    assert second["moments_source_zh"] == "2D 姿态和球"
    assert "检测到左队球员触球 2 次" in second["touches_zh"][0]
    assert "未验证" in second["touches_zh"][0]


def test_every_number_keeps_its_tier_and_badges(web: Running) -> None:
    detail = web.get_json(f"/api/runs/{RUN}")
    rows = {row["metric"]: row for row in detail["highlights"][1]["measurements"]}
    assert set(rows) == {
        "peak-speed",
        "body-orientation",
        "image-trunk-lean",
        "trunk-lean-at-contact",
    }
    assert detail["highlights"][1]["blocked_count"] == 1  # the C-tier record is never shown

    def texts(metric: str) -> list[str]:
        return [badge["text"] for badge in rows[metric]["badges"]]

    speed = rows["peak-speed"]
    assert (speed["value_text"], speed["unit"], speed["tier_zh"]) == ("27.0", "km/h", "A 档")
    assert texts("peak-speed") == ["未验证"]
    assert not speed["lift3d"]
    assert rows["body-orientation"]["tier_zh"] == "A 候选（3D）"
    assert texts("body-orientation") == ["未验证", "3D 参考"]
    # the report's own 3D badge, once, in the page's styled kind
    assert rows["body-orientation"]["badges"][-1]["kind"] == "lift3d"
    assert texts("image-trunk-lean") == ["未验证", "仅画面平面"]
    assert rows["image-trunk-lean"]["sign_note"] is not None
    assert texts("trunk-lean-at-contact") == ["不可用", "3D 参考"]
    assert rows["trunk-lean-at-contact"]["value_text"] == "—"
    assert speed["seek_s"] == pytest.approx(12.5 / 25)
    # a frame the full clip does not show cannot be played from
    other = detail["highlights"][0]["measurements"][0]
    assert other["frame"] == 999
    assert other["seek_s"] is None


def test_label_comparison_uses_the_page_names(web: Running, runs_root: Path) -> None:
    labels = web.get_json(f"/api/runs/{RUN}")["labels"]
    assert labels["counts"]["n_matched"] == 1
    assert labels["sentences_zh"] == ["标注 M1 → 检出 H2，贴身争抢 一致。"]
    # Where the labels come from, in the label file's own words (gbf.label_eval.source_zh).
    assert labels["source_zh"] == "标注来源（照录标注文件）：标注者“示例”；盲法“未说明”。"
    assert labels["labels_source"] == {"labelled_by": "示例", "blinding": "未说明"}
    assert labels["rows"][0]["highlight"] == "H2"
    assert labels["rows"][0]["label_type_zh"] == "贴身争抢"
    assert labels["unmatched"] == ["H1"]
    # a label_eval.json the run itself did not make (no labels input) is not shown
    _write_run(runs_root / "no-labels-run", labels=False)
    detail = web.get_json("/api/runs/no-labels-run")
    assert detail["labels"] is None
    assert detail["label_eval"] is None


def test_unknown_or_refused_run_names(web: Running) -> None:
    assert web.request("GET", "/api/runs/not-a-run")[0] == 404
    assert web.request("GET", "/api/runs/missing")[0] == 404
    assert web.request("GET", "/api/runs/.hidden-run")[0] == 403
    assert web.request("GET", "/api/runs/%2e%2e")[0] == 403


# --------------------------------------------------------------------------- security


def test_api_calls_need_the_token(web: Running) -> None:
    for path in ("/api/session", "/api/runs", f"/api/runs/{RUN}", "/api/jobs"):
        assert web.request("GET", path, token=False)[0] == 403
        status, _, body = web.request("GET", path, token=False, headers={"X-GBF-Token": "wrong"})
        assert status == 403
        assert "访问令牌" in json.loads(body)["error_zh"]
    # the cookie opens a run's files, never the API
    cookie = f"{web.server.cookie_name}={web.server.token}"
    assert web.request("GET", "/api/runs", token=False, headers={"Cookie": cookie})[0] == 403
    assert web.request("GET", "/api/runs")[0] == 200


def test_run_files_need_the_token_header_or_the_session_cookie(web: Running) -> None:
    path = f"/runs/{RUN}/report.html"
    assert web.request("GET", path, token=False)[0] == 403
    assert web.request("GET", path)[0] == 200
    status, headers, _ = web.request("GET", "/api/session")
    assert status == 200
    cookie = headers["Set-Cookie"]
    assert f"{web.server.cookie_name}={web.server.token}" in cookie
    for attribute in ("Path=/runs/", "HttpOnly", "SameSite=Strict"):
        assert attribute in cookie
    name_value = cookie.split(";", 1)[0]
    assert web.request("GET", path, token=False, headers={"Cookie": name_value})[0] == 200
    wrong = f"{web.server.cookie_name}=wrong"
    assert web.request("GET", path, token=False, headers={"Cookie": wrong})[0] == 403


@pytest.mark.parametrize(
    "host",
    [
        "evil.example:{port}",
        "127.0.0.1:{other}",
        "127.0.0.1",
        "localhost.evil.example:{port}",
        "[::1]:{port}",
        "",
    ],
)
def test_foreign_host_is_refused(web: Running, host: str) -> None:
    value = host.format(port=web.server.port, other=web.server.port + 1)
    for path in ("/", "/api/runs", f"/runs/{RUN}/report.html"):
        assert web.request("GET", path, host=value)[0] == 403
    assert web.logs
    assert "Host" in web.logs[-1]


def test_localhost_and_loopback_hosts_are_accepted(web: Running) -> None:
    for host in (f"127.0.0.1:{web.server.port}", f"localhost:{web.server.port}"):
        assert web.request("GET", "/api/runs", host=host)[0] == 200
        assert web.request("GET", "/", host=host, token=False)[0] == 200


def test_foreign_origin_and_cross_site_requests_are_refused(web: Running, media: Path) -> None:
    for origin in ("http://evil.example", "null", f"http://127.0.0.1:{web.server.port + 1}"):
        headers = {"Origin": origin}
        assert web.request("GET", "/api/runs", headers=headers)[0] == 403
        assert web.request("GET", f"/runs/{RUN}/report.html", headers=headers)[0] == 403
        status, _ = web.post_job(_job_request(web.server.runs_dir, media, "x1"), headers=headers)
        assert status == 403
    for site in ("cross-site", "same-site"):
        headers = {"Sec-Fetch-Site": site}
        assert web.request("GET", "/api/runs", headers=headers)[0] == 403
        assert web.request("GET", f"/runs/{RUN}/report.html", headers=headers)[0] == 403
    for origin in (f"http://127.0.0.1:{web.server.port}", f"http://localhost:{web.server.port}"):
        headers = {"Origin": origin, "Sec-Fetch-Site": "same-origin"}
        assert web.request("GET", "/api/runs", headers=headers)[0] == 200
    assert web.request("GET", "/api/runs", headers={"Sec-Fetch-Site": "none"})[0] == 200


@pytest.mark.parametrize(
    "path",
    [
        "/runs/demo-run/../../outside/secret.json",
        "/runs/demo-run/%2e%2e/%2e%2e/outside/secret.json",
        "/runs/demo-run/%2E%2E%2F%2E%2E%2Foutside%2Fsecret.json",
        "/runs/demo-run/..%5c..%5coutside%5csecret.json",
        "/runs/../outside/secret.json",
        "/runs/%2e%2e/outside/secret.json",
        "/runs/demo-run//outside/secret.json",
        "/runs/demo-run/%2Foutside%2Fsecret.json",
        "/runs/demo-run/C:%5Coutside%5Csecret.json",
        "/runs/demo-run/C:/outside/secret.json",
        "/runs/demo-run/%00secret.json",
        "/runs/demo-run/run.json%00.png",
        "/runs/demo-run/.hidden.json",
        "/runs/demo-run/run.json.",
        "/runs/demo-run/NUL.json",
        "/runs/demo-run/%ff.json",
        "/runs/elsewhere-run/run.json",
        "//outside/secret.json",
        "http://127.0.0.1/runs/demo-run/run.json",
    ],
)
def test_path_traversal_is_refused(web: Running, path: str) -> None:
    status, _, body = web.request("GET", path)
    assert status in (400, 403, 404), (status, path)
    assert SECRET.encode() not in body


def test_absolute_path_after_the_run_is_refused(web: Running, unicode_tmp_path: Path) -> None:
    secret = (unicode_tmp_path / "outside" / "secret.json").resolve()
    quoted = "/".join(quote(part, safe="") for part in secret.as_posix().split("/"))
    status, _, body = web.request("GET", f"/runs/{RUN}/{quoted}")
    assert status in (403, 404)
    assert SECRET.encode() not in body


def test_links_that_leave_the_run_are_refused(web: Running, unicode_tmp_path: Path) -> None:
    outside = unicode_tmp_path / "outside"
    inner = web.server.runs_dir / RUN / "linked"
    _link_dir(inner, outside)
    run_link = web.server.runs_dir / "linked-run"
    _link_dir(run_link, outside / "elsewhere-run")
    try:
        status, _, body = web.request("GET", f"/runs/{RUN}/linked/secret.json")
        assert status == 403
        assert SECRET.encode() not in body
        # a run folder that is a link to a folder outside the runs folder is not a run
        assert web.request("GET", "/runs/linked-run/run.json")[0] == 403
        assert web.request("GET", "/api/runs/linked-run")[0] == 403
        names = [run["name"] for run in web.get_json("/api/runs")["runs"]]
        assert "linked-run" not in names
        files = {item["path"] for item in web.get_json(f"/api/runs/{RUN}")["files"]}
        assert not any(path.startswith("linked/") for path in files)
    finally:
        _unlink_dir(inner)
        _unlink_dir(run_link)


def test_names_that_resolve_to_hidden_names_are_refused(web: Running) -> None:
    # A name that passes the segment rules but stands for a hidden one: a link inside the
    # runs folder or the run, and where the volume makes them a Windows 8.3 short name
    # (HIDDEN~1 for .hidden-run, HIDDEN~1.PNG for .hidden.png).
    root = web.server.runs_dir
    run = root / RUN
    write_text(run / ".hidden-dir" / "notes.json", json.dumps({"marker": SECRET}) + "\n")
    (run / ".hidden.png").write_bytes(SECRET.encode())
    (run / "keyframes").rename(run / ".hidden-frames")
    links = [
        (root / "alias-run", root / ".hidden-run"),
        (run / "alias-dir", run / ".hidden-dir"),
        (run / "keyframes", run / ".hidden-frames"),
    ]
    for link, target in links:
        _link_dir(link, target)
    try:
        urls = [
            "/api/runs/alias-run",
            "/runs/alias-run/run.json",
            f"/runs/{RUN}/alias-dir/notes.json",
            f"/runs/{RUN}/keyframes/h02.png",
        ]
        for long_path, url in (
            (root / ".hidden-run", "/api/runs/{}"),
            (root / ".hidden-run", "/runs/{}/run.json"),
            (run / ".hidden.png", f"/runs/{RUN}/{{}}"),
        ):
            short = _short_name(long_path)
            if short is not None:
                urls.append(url.format(short))
        for url in urls:
            status, _, body = web.request("GET", url)
            assert status == 403, (url, status)
            assert SECRET.encode() not in body
        assert "alias-run" not in [item["name"] for item in web.get_json("/api/runs")["runs"]]
        detail = web.get_json(f"/api/runs/{RUN}")
        files = {item["path"] for item in detail["files"]}
        assert not any(path.startswith(("alias-dir/", "keyframes/", ".hidden")) for path in files)
        # the documents and media the page names follow the same rule
        assert detail["highlights"][1]["keyframe"] is None
    finally:
        for link, _ in links:
            _unlink_dir(link)


def test_unknown_suffixes_and_directories_are_refused(web: Running) -> None:
    assert web.request("GET", f"/runs/{RUN}/pose.parquet")[0] == 403
    assert web.request("GET", f"/runs/{RUN}/clips")[0] == 403
    assert web.request("GET", f"/runs/{RUN}/clips/")[0] == 403
    assert web.request("GET", f"/runs/{RUN}/")[0] == 403
    assert web.request("GET", f"/runs/{RUN}")[0] == 404
    assert web.request("GET", f"/runs/{RUN}/missing.json")[0] == 404
    status, headers, _ = web.request("GET", f"/runs/{RUN}/run.json")
    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    status, headers, _ = web.request("GET", f"/runs/{RUN}/measurements.jsonl")
    assert status == 200
    assert headers["Content-Type"].startswith("text/plain")
    status, headers, _ = web.request("GET", f"/runs/{RUN}/keyframes/h02.png")
    assert status == 200
    assert headers["Content-Type"] == "image/png"


def test_unknown_suffixes_are_refused_before_the_file_system_is_asked(
    runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = runs_root.resolve()
    real_resolve = Path.resolve

    def guarded(self: Path, *args: Any, **kwargs: Any) -> Path:
        if self.suffix in (".parquet", ".log"):
            raise AssertionError(f"resolved {self.name}, whose suffix is never served")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded)
    for name in ("pose.parquet", "missing.log"):
        with pytest.raises(RunPathError) as caught:
            resolve_artifact(root, RUN, [name])
        assert caught.value.status == 403
    assert resolve_artifact(root, RUN, ["run.json"]).name == "run.json"


def test_range_requests_return_partial_content(web: Running, runs_root: Path) -> None:
    data = (runs_root / RUN / "clips" / "full.mp4").read_bytes()
    size = len(data)
    path = f"/runs/{RUN}/clips/full.mp4"
    status, headers, body = web.request("GET", path)
    assert (status, body) == (200, data)
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Type"] == "video/mp4"
    status, headers, body = web.request("GET", path, headers={"Range": "bytes=10-19"})
    assert (status, body) == (206, data[10:20])
    assert headers["Content-Range"] == f"bytes 10-19/{size}"
    assert headers["Content-Length"] == "10"
    status, headers, body = web.request("GET", path, headers={"Range": "bytes=-5"})
    assert (status, body) == (206, data[-5:])
    status, headers, body = web.request("GET", path, headers={"Range": f"bytes={size - 3}-"})
    assert (status, body) == (206, data[-3:])
    status, headers, body = web.request("GET", path, headers={"Range": f"bytes=5-{size + 99}"})
    assert (status, body) == (206, data[5:])
    status, headers, body = web.request("GET", path, headers={"Range": f"bytes={size}-"})
    assert (status, body) == (416, b"")
    assert headers["Content-Range"] == f"bytes */{size}"
    for ignored in ("bytes=9-2", "bytes=0-1,4-5", "items=0-1", "bytes=0-" + "9" * 5000):
        status, _, body = web.request("GET", path, headers={"Range": ignored})
        assert (status, body) == (200, data)
    status, headers, body = web.request("HEAD", path, headers={"Range": "bytes=0-99"})
    assert (status, body, headers["Content-Length"]) == (206, b"", "100")


def test_parse_range() -> None:
    assert parse_range(None, 10) is None
    assert parse_range("bytes=0-", 10) == (0, 9)
    assert parse_range("bytes=2-4", 10) == (2, 4)
    assert parse_range("bytes=-3", 10) == (7, 9)
    assert parse_range("bytes=-30", 10) == (0, 9)
    assert parse_range("bytes=10-", 10) == (-1, -1)
    assert parse_range("bytes=-0", 10) == (-1, -1)
    assert parse_range("bytes=0-", 0) == (-1, -1)
    assert parse_range("bytes=4-2", 10) is None
    assert parse_range("bytes=1-2,4-5", 10) is None
    assert parse_range("bytes=-", 10) is None
    # numbers of more than 18 digits are not read (int() refuses more than 4300 digits)
    assert parse_range("bytes=0-" + "9" * 18, 10) == (0, 9)
    for absurd in ("9" * 19 + "-", "0-" + "9" * 5000, "-" + "9" * 5000, "9" * 5000 + "-"):
        assert parse_range("bytes=" + absurd, 10) is None


def test_static_page_headers_and_methods(web: Running) -> None:
    status, headers, body = web.request("GET", "/", token=False)
    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert b"/static/app.js" in body
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    for name, value in SECURITY_HEADERS.items():
        assert headers[name] == value
    status, headers, _ = web.request("GET", "/static/app.js", token=False)
    assert status == 200
    assert headers["Content-Type"] == "text/javascript; charset=utf-8"
    assert web.request("GET", "/static/missing.js", token=False)[0] == 404
    assert web.request("GET", "/static/../server.py", token=False)[0] == 404
    status, headers, _ = web.request("OPTIONS", "/api/jobs")
    assert status == 405
    assert web.request("POST", "/", token=False)[0] == 405
    assert web.request("DELETE", f"/runs/{RUN}/run.json")[0] == 405
    # a run's files never run script; nothing ever allows another origin
    status, headers, _ = web.request("GET", f"/runs/{RUN}/report.html")
    assert "script-src" not in headers["Content-Security-Policy"]
    for path in ("/", "/api/runs", f"/runs/{RUN}/report.html"):
        _, headers, _ = web.request("GET", path)
        assert not any(name.lower().startswith("access-control-") for name in headers)


def test_every_answer_carries_the_security_headers(web: Running) -> None:
    clip = f"/runs/{RUN}/clips/full.mp4"
    answers = {
        "page": web.request("GET", "/", token=False),
        "style": web.request("GET", "/static/style.css", token=False),
        "api": web.request("GET", "/api/runs"),
        "no token": web.request("GET", "/api/runs", token=False),
        "run file": web.request("GET", f"/runs/{RUN}/report.html"),
        "range": web.request("GET", clip, headers={"Range": "bytes=0-9"}),
        "past the end": web.request("GET", clip, headers={"Range": "bytes=99999999-"}),
        "suffix": web.request("GET", f"/runs/{RUN}/pose.parquet"),
        "unknown": web.request("GET", "/nowhere"),
        "method": web.request("DELETE", f"/runs/{RUN}/run.json"),
        "http.server's own answer": web.request("TRACE", "/"),
    }
    statuses = {what: answer[0] for what, answer in answers.items()}
    assert list(statuses.values()) == [200, 200, 200, 403, 200, 206, 416, 403, 404, 405, 501]
    for what, (_, headers, _) in answers.items():
        for name, value in SECURITY_HEADERS.items():
            assert headers[name] == value, (what, name)
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"], what


def test_server_listens_on_loopback_only(web: Running) -> None:
    assert web.server.server_address[0] == "127.0.0.1"
    assert web.server.url == f"http://127.0.0.1:{web.server.port}/#t={web.server.token}"
    options = {action.dest for action in server_mod._parser()._actions}
    assert options == {"help", "runs_dir", "port", "no_open"}
    other = WebServer(web.server.runs_dir, jobs=web.server.jobs)
    try:
        assert other.token != web.server.token
        assert len(other.token) >= 40
    finally:
        other.server_close()


# --------------------------------------------------------------------------- jobs


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"video": None}, "视频文件：必填"),
        ({"video": "relative/clip.mp4"}, "完整路径"),
        ({"video": "\\\\server\\share\\clip.mp4"}, "网络路径"),
        ({"sngs_dir": "//server/share/sngs"}, "网络路径"),
        ({"video": "C:/clips/a\x07b.mp4"}, "控制字符"),
        ({"ball": "{missing}"}, "找不到这个文件"),
        ({"sngs_dir": "{video}"}, "找不到这个文件夹"),
        ({"out_name": "../escape"}, "输出名"),
        ({"out_name": ".hidden"}, "输出名"),
        ({"out_name": "a b"}, "输出名"),
        ({"out_name": "NUL"}, "设备名"),
        ({"out_name": "x" * 81}, "输出名"),
        ({"allow_licenses": ["GPL 3"]}, "SPDX"),
        ({"allow_licenses": "GPL-3.0-only"}, "列表"),
        ({"team_names": {"left": "a,b"}}, "逗号"),
        ({"team_names": {"middle": "a"}}, "left 和 right"),
        ({"surprise": 1}, "不认识的字段"),
    ],
)
def test_job_requests_are_checked(
    web: Running, media: Path, changes: dict[str, Any], expected: str
) -> None:
    payload = _job_request(web.server.runs_dir, media, "new-run")
    for key, value in changes.items():
        if value == "{missing}":
            value = str(web.server.runs_dir.parent / "missing.jsonl")
        elif value == "{video}":
            value = str(media)
        payload[key] = value
    status, data = web.post_job(payload)
    assert status == 400
    assert any(expected in problem for problem in data["problems_zh"]), data
    assert not (web.server.runs_dir / jobs_mod.JOBS_DIR).exists()


def test_network_paths_are_refused_before_the_file_system_is_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_access(self: Path) -> bool:
        raise AssertionError(f"the file system was asked about {self}")

    monkeypatch.setattr(Path, "is_file", no_access)
    monkeypatch.setattr(Path, "is_dir", no_access)
    with pytest.raises(JobRequestError) as caught:
        JobRequest.from_json(
            {
                "video": "\\\\attacker\\share\\a.mp4",
                "sngs_dir": "//attacker/share",
                "ball": '"\\\\?\\C:\\ball.jsonl"',
                "out_name": "x",
            }
        )
    assert sum("网络路径" in p for p in caught.value.problems) == 3


@pytest.mark.parametrize(
    "text",
    [
        "/\\127.0.0.1\\share\\clip.mp4",
        "\\/127.0.0.1/share/clip.mp4",
        "//127.0.0.1\\share/clip.mp4",
        "/\\?\\C:\\clip.mp4",
        "\\/./pipe/clip",
        "\\??\\UNC\\127.0.0.1\\share\\clip.mp4",
        "/??/UNC/127.0.0.1/share/clip.mp4",
    ],
)
def test_network_paths_with_mixed_separators_are_refused_before_the_file_system_is_asked(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    # Windows reads "/" and "\" alike: each of these names another machine or a device.
    # (A loopback host, so that even a check that went through would stay on this machine.)
    asked = _record_file_system(monkeypatch)
    with pytest.raises(JobRequestError) as caught:
        JobRequest.from_json({"video": text, "sngs_dir": text, "ball": text, "out_name": "x"})
    assert asked == []
    assert sum("网络路径" in p for p in caught.value.problems) == 3


@pytest.mark.parametrize("control", ["\x00", "\t", "\n", "\x1f", "\x7f"])
def test_job_paths_with_control_characters_are_refused(
    monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    asked = _record_file_system(monkeypatch)
    text = f"C:/clips/a{control}b.mp4"
    with pytest.raises(JobRequestError) as caught:
        JobRequest.from_json({"video": text, "sngs_dir": text, "ball": text, "out_name": "x"})
    assert asked == []
    assert sum("控制字符" in p for p in caught.value.problems) == 3


def test_post_bodies_must_be_json(web: Running, media: Path) -> None:
    body = json.dumps(_job_request(web.server.runs_dir, media, "x1")).encode()
    for content_type in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
        status, _, _ = web.request(
            "POST", "/api/jobs", headers={"Content-Type": content_type}, body=body
        )
        assert status == 415
    status, _, _ = web.request(
        "POST",
        "/api/jobs",
        headers={"Content-Type": "application/json; charset=latin-1"},
        body=body,
    )
    assert status == 415
    status, _, _ = web.request(
        "POST", "/api/jobs", headers={"Content-Type": "application/json"}, body=b"{not json"
    )
    assert status == 400
    status, data = web.post_job([1, 2])
    assert status == 400
    assert data["problems_zh"] == ["请求必须是一个 JSON 对象。"]
    status, _, _ = web.request(
        "POST",
        "/api/jobs",
        headers={"Content-Type": "application/json"},
        body=b"{" + b" " * server_mod.MAX_BODY_BYTES + b"}",
    )
    assert status == 413
    assert web.request("POST", "/api/jobs", token=False, body=body)[0] == 403


def test_deeply_nested_json_is_a_bad_request(web: Running) -> None:
    body = b'{"video": ' + b"[" * 30000 + b"]" * 30000 + b"}"
    assert len(body) <= server_mod.MAX_BODY_BYTES
    status, _, answer = web.request(
        "POST", "/api/jobs", headers={"Content-Type": "application/json"}, body=body
    )
    assert status == 400
    assert "JSON" in json.loads(answer)["error_zh"]


def test_refused_posts_get_their_answer_on_one_client_connection(web: Running) -> None:
    # The server reads a refused request's body before answering: a connection closed with
    # the body unread is reset, and the client loses the answer (Windows: WinError 10053).
    # Each body here comes 0.2 s after its headers, so without that read every answer is
    # lost; the client reconnects whenever an answer says Connection: close.
    body = json.dumps({"video": "C:/" + "v" * 20000, "out_name": "x1"}).encode()
    token = {server_mod.TOKEN_HEADER: web.server.token}
    json_type = {"Content-Type": "application/json"}
    refused = [
        ("/api/jobs", json_type, 403),  # no token
        ("/api/jobs", {**token, **json_type, "Origin": "http://evil.example"}, 403),
        ("/api/jobs", {**token, "Content-Type": "text/plain"}, 415),
        ("/api/nowhere", {**token, **json_type}, 404),
        (f"/runs/{RUN}/run.json", {**token, **json_type}, 405),
        ("/", json_type, 405),
    ]
    conn = http.client.HTTPConnection("127.0.0.1", web.server.port, timeout=30)
    try:
        for _ in range(2):
            for path, headers, expected in refused:
                conn.putrequest("POST", path, skip_host=True, skip_accept_encoding=True)
                conn.putheader("Host", web.host)
                for name, value in headers.items():
                    conn.putheader(name, value)
                conn.putheader("Content-Length", str(len(body)))
                conn.endheaders(_late(body))
                response = conn.getresponse()
                assert (path, response.status) == (path, expected)
                assert response.getheader("Connection") == "close"
                assert response.read()
    finally:
        conn.close()
    assert not (web.server.runs_dir / jobs_mod.JOBS_DIR).exists()


def test_a_refused_request_ends_its_connection(web: Running) -> None:
    def request(extra: str) -> bytes:
        return f"GET /api/runs HTTP/1.1\r\nHost: {web.host}\r\n{extra}\r\n".encode()

    with socket.create_connection(("127.0.0.1", web.server.port), timeout=10) as sock:
        sock.sendall(request(f"{server_mod.TOKEN_HEADER}: {web.server.token}\r\n"))
        accepted = _answer(sock)
        assert (accepted.status, accepted.getheader("Connection")) == (200, None)
        accepted.read()
        sock.sendall(request(""))  # the same connection, without the token
        refused = _answer(sock)
        assert (refused.status, refused.getheader("Connection")) == (403, "close")
        refused.read()
        sock.settimeout(5)
        assert sock.recv(1) == b""  # closed by the server


def test_a_refused_request_after_an_accepted_post_still_gets_its_answer(
    web: Running, fake: dict[str, Path], media: Path
) -> None:
    # The body of the accepted POST was read; the next request on the same keep-alive
    # connection is refused, and its body (sent 0.2 s after its headers) must be read and
    # dropped too before the connection is closed, or the client loses the answer. The
    # server forgot that only when it did not start each request as unread.
    _control(fake, "ok")
    payload = json.dumps(_job_request(web.server.runs_dir, media, "keep-alive-run")).encode()
    refused = json.dumps({"video": "C:/" + "v" * 20000, "out_name": "x2"}).encode()
    job: dict[str, Any] = {}
    conn = http.client.HTTPConnection("127.0.0.1", web.server.port, timeout=30)
    try:
        for content_type, body, expected in (
            ("application/json", payload, 202),
            ("text/plain", refused, 415),
        ):
            conn.putrequest("POST", "/api/jobs", skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", web.host)
            conn.putheader(server_mod.TOKEN_HEADER, web.server.token)
            conn.putheader("Content-Type", content_type)
            conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(_late(body) if expected == 415 else body)
            response = conn.getresponse()
            answer = json.loads(response.read())
            assert response.status == expected, answer
            if expected == 202:
                assert response.getheader("Connection") is None  # kept alive
                job = answer
            else:
                assert response.getheader("Connection") == "close"
                assert "JSON" in answer["error_zh"]
    finally:
        conn.close()
    assert _wait_for(web, job["id"])["status"] == "succeeded"


def test_a_refused_request_announcing_a_large_body_is_answered_at_once(web: Running) -> None:
    # Only the headers are sent. A body announced above MAX_BODY_BYTES is left unread (the
    # connection ends anyway): waiting for it would hold the answer until the client gave
    # up or the server's 60 s timeout.
    head = [
        "POST /api/jobs HTTP/1.1",
        f"Host: {web.host}",
        f"{server_mod.TOKEN_HEADER}: {web.server.token}",
        "Content-Type: application/json",
        f"Content-Length: {server_mod.MAX_BODY_BYTES + 1}",
    ]
    with socket.create_connection(("127.0.0.1", web.server.port), timeout=5) as sock:
        started = time.monotonic()
        sock.sendall(("\r\n".join(head) + "\r\n\r\n").encode())
        answer = _answer(sock, "POST")
        assert (answer.status, answer.getheader("Connection")) == (413, "close")
        assert "太大" in json.loads(answer.read())["error_zh"]
        assert time.monotonic() - started < 5
    assert not (web.server.runs_dir / jobs_mod.JOBS_DIR).exists()


def test_oversized_json_with_a_late_body_keeps_its_error_response(web: Running) -> None:
    # Reject the size from its headers, but allow the caller to receive that answer even
    # when http.client writes the oversized body after the server has already replied.
    body = b"{" + b" " * server_mod.MAX_BODY_BYTES + b"}"
    conn = http.client.HTTPConnection("127.0.0.1", web.server.port, timeout=5)
    try:
        conn.putrequest("POST", "/api/jobs")
        conn.putheader(server_mod.TOKEN_HEADER, web.server.token)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(_late(body))
        response = conn.getresponse()
        assert (response.status, response.getheader("Connection")) == (413, "close")
        assert "太大" in json.loads(response.read())["error_zh"]
    finally:
        conn.close()
    assert not (web.server.runs_dir / jobs_mod.JOBS_DIR).exists()


@pytest.mark.parametrize("length", [None, 10])
def test_chunked_bodies_are_refused_without_being_read(web: Running, length: int | None) -> None:
    # Only the headers are sent: the server must answer at once, neither waiting for a
    # chunked body nor reading Content-Length bytes of it as the body.
    head = [
        "POST /api/jobs HTTP/1.1",
        f"Host: {web.host}",
        f"{server_mod.TOKEN_HEADER}: {web.server.token}",
        "Content-Type: application/json",
        "Transfer-Encoding: chunked",
    ]
    if length is not None:
        head.append(f"Content-Length: {length}")
    with socket.create_connection(("127.0.0.1", web.server.port), timeout=5) as sock:
        sock.sendall(("\r\n".join(head) + "\r\n\r\n").encode())
        answer = _answer(sock, "POST")
        assert (answer.status, answer.getheader("Connection")) == (411, "close")
        assert "长度" in json.loads(answer.read())["error_zh"]
        assert sock.recv(1) == b""
    assert not (web.server.runs_dir / jobs_mod.JOBS_DIR).exists()


def test_one_job_at_a_time_with_progress_and_result(
    web: Running, fake: dict[str, Path], media: Path
) -> None:
    release = fake["folder"] / "release"
    _control(fake, "ok", release=release)
    payload = _job_request(
        web.server.runs_dir,
        media,
        "new-run",
        team_names={"left": "左队", "right": "右队"},
        allow_licenses=["GPL-3.0-only"],
        audio_source=f'"{media}"',
    )
    status, job = web.post_job(payload)
    assert status == 202, job
    assert job["status"] == "running"
    assert job["out_name"] == "new-run"
    # a second analysis while one runs is refused, with a clear message
    status, refused = web.post_job(_job_request(web.server.runs_dir, media, "other-run"))
    assert status == 409
    assert "同一时间只能运行一个分析" in refused["problems_zh"][0]
    assert web.get_json("/api/jobs")["current"] == job["id"]
    # progress: the stages so far, from the log
    deadline = time.monotonic() + 60
    while True:
        running = web.get_json(f"/api/jobs/{job['id']}")
        states = {stage["name"]: stage["state"] for stage in running["stages"]}
        if states["refine"] == "running" or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    assert (states["import"], states["shots"], states["refine"]) == ("done", "skipped", "running")
    assert states["report"] == "pending"
    assert running["current_stage"] == "refine"
    assert "清理轨迹" in running["message_zh"]
    assert "[import] done in 0.1 s" in running["log_tail"]
    release.write_text("go", encoding="utf-8")
    done = _wait_for(web, job["id"])
    assert (done["status"], done["exit_code"]) == ("succeeded", 0)
    assert done["message_zh"] == "分析完成。"
    assert done["run_ready"]
    assert done["log_file"] == f".gbf-web/jobs/{job['id']}.log"
    argv = json.loads((fake["folder"] / "argv.json").read_text(encoding="utf-8"))
    assert argv == [
        f"--video={media}",
        f"--audio-source={media}",
        f"--sngs={web.server.runs_dir.parent / 'inputs' / 'sngs'}",
        f"--ball={web.server.runs_dir.parent / 'inputs' / 'ball_raw.jsonl'}",
        f"--out={web.server.runs_dir / 'new-run'}",
        "--team-names=left=左队,right=右队",
        "--allow-license=GPL-3.0-only",
    ]
    # the new run is listed; its folder now exists, so the same name is refused
    assert "new-run" in [run["name"] for run in web.get_json("/api/runs")["runs"]]
    status, again = web.post_job(_job_request(web.server.runs_dir, media, "new-run"))
    assert status == 409
    assert "已经存在" in again["problems_zh"][0]
    # one job is over: the next may start
    _control(fake, "ok")
    status, second = web.post_job(_job_request(web.server.runs_dir, media, "second-run"))
    assert status == 202
    assert _wait_for(web, second["id"])["status"] == "succeeded"


def test_exit_codes_become_chinese_messages(
    web: Running, fake: dict[str, Path], media: Path
) -> None:
    _control(fake, "refused")
    status, job = web.post_job(_job_request(web.server.runs_dir, media, "refused-run"))
    assert status == 202
    refused = _wait_for(web, job["id"])
    assert (refused["status"], refused["exit_code"]) == ("refused", 2)
    assert refused["message_zh"].startswith("输入被拒绝，分析没有开始")
    assert len(refused["problems_zh"]) == 1
    assert "GPL-3.0-only" in refused["problems_zh"][0]
    assert "接受的许可证" in refused["problems_zh"][0]
    assert not refused["run_ready"]

    _control(fake, "failed")
    status, job = web.post_job(_job_request(web.server.runs_dir, media, "failed-run"))
    failed = _wait_for(web, job["id"])
    assert (failed["status"], failed["exit_code"]) == ("failed", 1)
    assert "清理轨迹、重拟合标定" in failed["message_zh"]
    states = {stage["name"]: stage["state"] for stage in failed["stages"]}
    assert (states["import"], states["refine"], states["moments"]) == ("done", "failed", "pending")
    assert failed["problems_zh"] == [
        "插件导出覆盖 10 帧，视频却解码出 12 帧：这份导出不属于这个视频（帧号必须一一对应）。"
    ]
    assert failed["run_ready"]  # a failed run still writes run.json; the page can show it
    listed = {run["name"]: run["status"] for run in web.get_json("/api/runs")["runs"]}
    assert listed["failed-run"] == "failed"
    assert "refused-run" not in listed


def test_jobs_are_remembered_across_restarts(
    runs_root: Path, fake: dict[str, Path], media: Path, unicode_tmp_path: Path
) -> None:
    _control(fake, "ok")
    first = _manager(runs_root, fake, unicode_tmp_path)
    request = JobRequest.from_json(_job_request(runs_root, media, "remembered"))
    job = first.start(request)
    job.waiter.join(timeout=60)
    first.close(grace_s=0)
    later = _manager(runs_root, fake, unicode_tmp_path)
    rows = later.recent()
    assert [(row["id"], row["status"], row["out_name"]) for row in rows] == [
        (job.id, "succeeded", "remembered")
    ]
    assert rows[0]["request"]["video"] == str(media)
    snapshot = later.get(job.id).snapshot(runs_root)
    assert snapshot["status"] == "succeeded"
    assert snapshot["run_ready"]
    assert later.get("not-an-id") is None
    assert later.current() is None


def test_closing_the_server_stops_a_running_job(
    runs_root: Path, fake: dict[str, Path], media: Path, unicode_tmp_path: Path
) -> None:
    release = fake["folder"] / "release"
    _control(fake, "ok", release=release)
    manager = _manager(runs_root, fake, unicode_tmp_path)
    job = manager.start(JobRequest.from_json(_job_request(runs_root, media, "stopped")))
    assert manager.current() is job
    deadline = time.monotonic() + 60
    while job.snapshot(runs_root)["current_stage"] != "refine" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert job.running
    manager.close(grace_s=0)
    assert job.exit_code is not None
    assert job.status == "interrupted"
    assert "中断" in job.snapshot(runs_root)["message_zh"]
    # the analysis itself has stopped (on Windows sys.executable may be a venv launcher
    # whose child is the real interpreter): released now, it would finish within 0.1 s
    release.write_text("go", encoding="utf-8")
    time.sleep(1.5)
    assert not (runs_root / "stopped" / "run.json").exists()
    with pytest.raises(JobRefused) as caught:
        manager.start(JobRequest.from_json(_job_request(runs_root, media, "stopped")))
    assert caught.value.status == 409  # its folder may hold a partial result


def test_output_inside_a_git_work_tree_is_refused_before_starting(
    runs_root: Path, fake: dict[str, Path], media: Path, unicode_tmp_path: Path
) -> None:
    (runs_root / ".git").mkdir()
    manager = _manager(runs_root, fake, unicode_tmp_path)
    with pytest.raises(JobRefused) as caught:
        manager.start(JobRequest.from_json(_job_request(runs_root, media, "in-git")))
    assert caught.value.status == 400
    assert "git 仓库" in caught.value.problems[0]
    assert not (runs_root / "in-git").exists()


def test_job_arguments_and_environment(media: Path, unicode_tmp_path: Path) -> None:
    sngs = unicode_tmp_path / "sngs"
    sngs.mkdir()
    request = JobRequest.from_json(
        {
            "video": f"  {media}  ",
            "sngs_dir": str(sngs),
            "ball": str(media),
            "labels": "",
            "team_names": {"left": " 左队 ", "right": ""},
            "allow_licenses": ["NOASSERTION", "GPL-3.0-only", "GPL-3.0-only"],
            "out_name": "run_1.a-b",
        }
    )
    assert request.audio_source == media
    assert request.labels is None
    assert request.team_names == {"left": "左队"}
    assert request.allow_licenses == ("GPL-3.0-only", "NOASSERTION")
    out = unicode_tmp_path / "runs" / "run_1.a-b"
    assert request.arguments(out)[-3:] == [
        "--team-names=left=左队",
        "--allow-license=GPL-3.0-only",
        "--allow-license=NOASSERTION",
    ]
    assert jobs_mod.highlights_command() == [sys.executable, "-P", "-m", "gbf", "highlights"]
    env = jobs_mod.child_environment({"PYTHONPATH": "elsewhere"})
    source = Path(gbf.__file__).resolve().parent.parent
    added = [] if source.name.lower() == "site-packages" else [str(source)]
    assert env["PYTHONPATH"].split(os.pathsep) == [*added, "elsewhere"]
    assert (env["PYTHONUTF8"], env["PYTHONIOENCODING"]) == ("1", "utf-8")


@pytest.mark.parametrize(
    ("code", "stopped", "status"),
    [
        (None, False, "running"),
        (0, False, "succeeded"),
        (2, False, "refused"),
        (1, False, "failed"),
        (3, False, "failed"),
        (1, True, "interrupted"),
        (jobs_mod.STATUS_CONTROL_C_EXIT, False, "interrupted"),
        (-2, False, "interrupted"),
        (0, True, "succeeded"),
    ],
)
def test_classify_exit_codes(code: int | None, stopped: bool, status: str) -> None:
    assert jobs_mod.classify(code, stopped=stopped) == status


def test_translate_problem() -> None:
    translate = jobs_mod.translate_problem
    assert translate("--video: not a file: D:/x.mp4") == "视频文件不存在：D:/x.mp4"
    assert translate("--ball-raw: not a file: D:/b.jsonl") == "球检测文件不存在：D:/b.jsonl"
    assert translate("--sngs: not a directory: D:/s") == "插件导出文件夹不存在：D:/s"
    licence = translate(
        "ball input (ball-receipt.json): license 'NOASSERTION' is not satisfied by the allowed "
        "licenses (not allowed: NOASSERTION); accept it explicitly with --allow-license <id>"
    )
    assert licence.startswith("球检测的输入（ball-receipt.json）")
    assert "NOASSERTION" in licence
    assert "接受的许可证" in licence
    assert "git 仓库（D:/repo）" in translate(
        "--out D:/repo/runs/a is inside the git work tree D:/repo; run outputs must stay "
        "outside every repository"
    )
    assert "恒定帧率" in translate("the video is not constant-frame-rate; normalise it first")
    missing = translate(r"[Errno 2] No such file or directory: 'D:\\s\\plugin-receipt.json'")
    assert missing.startswith("找不到文件：D:\\s\\plugin-receipt.json（")
    unquoted = translate("[Errno 2] No such file or directory: 'cut off")
    assert unquoted.startswith("找不到文件：'cut off（")
    # Receipts the pipeline refuses before it writes anything (exit 2).
    assert (
        translate(
            "--sngs: plugin-receipt.json is missing: the plugin export needs its receipt "
            "(licence, restrictions and the sha256 of its files)"
        )
        == "插件导出缺少 plugin-receipt.json：没有它不知道许可证和数据限制，所以不运行。"
    )
    assert (
        translate(
            "plugin-receipt.json cannot be read as JSON: Expecting value: line 1 column 1 (char 0)"
        )
        == "plugin-receipt.json 读不成 JSON：不知道许可证和数据限制，所以不运行。"
    )
    assert translate("plugin-receipt.json declares no restrictions list").startswith(
        "plugin-receipt.json 没有声明数据限制"
    )
    assert translate("something new happened") == "something new happened"


@pytest.mark.parametrize("tag_repr", ["'made_up'", '"don\'t_share"', "None", "3"])
def test_unknown_restriction_tag_has_an_actionable_chinese_refusal(tag_repr: str) -> None:
    text = jobs_mod.translate_problem(
        f"plugin-receipt.json: 0: {tag_repr} is not one of ['nda', 'no_share']"
    )
    assert "plugin-receipt.json" in text
    assert tag_repr.strip("'\"") in text
    assert "限制标签" in text
    assert "不运行" in text


def test_unrelated_receipt_enum_problem_is_not_called_a_restriction() -> None:
    text = "plugin-receipt.json: 0: 'other' is not one of ['MIT', 'GPL-3.0-only']"
    assert jobs_mod.translate_problem(text) == text


@pytest.mark.parametrize("track", [None, 3])
@pytest.mark.parametrize("gap", [None, 0.4])
def test_web_touch_sentence_matches_report_wording(track: int | None, gap: float | None) -> None:
    from gbf import report
    from gbf.web import runs

    touch = {"label_zh": "左队球员", "count": 2, "track_id": track, "mean_interval_s": gap}
    sentence = runs._touch_sentence(touch)
    assert sentence == report._touch_sentence(touch)
    assert sentence.count("画面平面估计，未验证") == 1


# --------------------------------------------------------------------------- pieces


@pytest.mark.parametrize(
    "segment",
    [
        "",
        ".",
        "..",
        ".git",
        "a/b",
        "a\\b",
        "C:",
        "a:b",
        "a*",
        "a?",
        "a\x00",
        "a\x1f",
        "a.",
        "a ",
        "CON",
        "nul.json",
        "Com1.png",
        "lpt9",
    ],
)
def test_segment_problem_refuses(segment: str) -> None:
    assert segment_problem(segment) is not None


@pytest.mark.parametrize(
    "segment", ["run.json", "h01-card.png", "非ASCII 路径-ü", "con1.json", "a.b"]
)
def test_segment_problem_accepts(segment: str) -> None:
    assert segment_problem(segment) is None


def test_full_clip_seeker() -> None:
    manifest = {
        "output": {"fps": "25/1"},
        "segments": [
            {"kind": "normal", "src_start": 100, "src_end": 199, "out_start": 0, "repeat": 1},
            {"kind": "slow", "src_start": 150, "src_end": 160, "out_start": 100, "repeat": 4},
        ],
    }
    mapping, seek = full_clip_seeker(manifest, Fraction(25))
    assert mapping == "manifest"
    assert seek(100) == pytest.approx(0.5 / 25)
    assert seek(150) == pytest.approx(50.5 / 25)
    assert seek(99) is None
    assert seek(200) is None
    mapping, seek = full_clip_seeker(None, Fraction(30000, 1001))
    assert mapping == "frame_over_fps"
    assert seek(300) == pytest.approx(300.5 * 1001 / 30000)
    assert full_clip_seeker(None, None)[0] == "none"


def test_measurement_rows_follow_the_report() -> None:
    def seek(frame: int) -> float:
        return frame / 25

    blocked = measurement_row(_record("h", "foot-speed", "C", 3.0, "m/s"), seek)
    assert blocked is None
    row = measurement_row(
        _record("h", "nearest-opponent-image-distance", "image_plane", 0.5, "box_h", plane="image"),
        seek,
    )
    assert row is not None
    assert (row["value_text"], row["unit"]) == ("0.50", "个身高")
    assert [b["kind"] for b in row["badges"]] == ["unverified", "image_plane"]
    lifted = measurement_row(_record("h", "some-new-3d-metric", "A_candidate", 9.0, "deg"), seek)
    assert lifted is not None
    assert lifted["lift3d"]
    assert lifted["badges"][-1] == {"kind": "lift3d", "text": "3D 参考"}


def test_lift3d_metrics_are_the_registry_a_candidate_tier() -> None:
    import tomllib

    with open(repo_root() / "registries" / "metrics.toml", "rb") as handle:
        registry = {entry["id"]: entry["tier"] for entry in tomllib.load(handle)["metrics"]}
    candidates = {
        metric
        for metric, spec in METRICS.items()
        if spec.explicit_tier is None and registry[metric] == "A_candidate"
    }
    assert candidates == LIFT3D_METRICS


def test_runs_dir_default_and_problems(unicode_tmp_path: Path) -> None:
    data_root = unicode_tmp_path / "data root"
    (data_root / "runs").mkdir(parents=True)
    found, problems = runs_dir_from(None, {"DATA_ROOT": str(data_root)})
    assert (found, problems) == ((data_root / "runs").resolve(), [])
    found, problems = runs_dir_from(None, {})
    assert found is None
    assert "--runs-dir" in problems[0]
    found, problems = runs_dir_from(unicode_tmp_path / "missing", {})
    assert found is None
    assert "不存在" in problems[0]
    found, problems = runs_dir_from(Path("//server/share/runs"), {})
    assert found is None
    assert "网络路径" in problems[0]


@pytest.mark.parametrize(
    "text",
    [
        "\\\\127.0.0.1\\share\\runs",
        "//127.0.0.1/share/runs",
        "/\\127.0.0.1\\share\\runs",
        "\\/127.0.0.1/share/runs",
        # the NT object-manager prefix: \??\UNC\host\share is a share, \??\C: a device path
        "\\??\\UNC\\127.0.0.1\\share\\runs",
        "/??/UNC/127.0.0.1/share/runs",
        "\\??\\C:\\runs",
    ],
)
def test_a_network_runs_dir_is_refused_before_the_file_system_is_asked(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    asked = _record_file_system(monkeypatch)
    found, problems = runs_dir_from(Path(text), {})
    assert asked == []
    assert found is None
    assert "网络路径" in problems[0]


def test_command_is_registered_and_refuses_a_missing_runs_dir(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.COMMANDS["web"][0] == "gbf.web.server"
    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert cli.main(["web", "--no-open"]) == 2
    assert "--runs-dir" in capsys.readouterr().err
    assert cli.main(["web", "--port", "70000"]) == 2


def test_stage_names_cover_the_pipeline() -> None:
    assert set(jobs_mod.STAGE_NAMES_ZH) == set(STAGES)


def test_automatic_time_includes_predictions_without_replacing_core_time(runs_root: Path) -> None:
    from gbf.web.runs import run_detail

    before = run_detail(runs_root, RUN)
    assert before["automatic"] is None
    write_json(runs_root / RUN / "automatic.json", {"wall_s": 185.5, "conversion": {"fps": 25}})
    after = run_detail(runs_root, RUN)
    assert after["automatic"] == {"wall_s": 185.5, "converted": True}
    assert after["run"]["wall_s"] == before["run"]["wall_s"]


# --------------------------------------------------------------------------- the page itself


def _web_files() -> list[str]:
    root = repo_root()
    package = root / "src" / "gbf" / "web"
    return sorted(
        path.relative_to(root).as_posix()
        for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def test_the_page_passes_the_gate() -> None:
    root = repo_root()
    files = _web_files()
    assert {"src/gbf/web/static/index.html", "src/gbf/web/static/app.js"} <= set(files)
    workspace = Workspace(root, files=files)
    for check in (
        check_eol,
        check_credentials,
        check_abs_paths,
        check_media,
        check_restricted_data,
        check_forbidden_imports,
    ):
        result = check(root, workspace)
        assert result.problems == [], [problem.format() for problem in result.problems]


def test_the_gate_would_catch_a_banner_or_tags_in_the_page(unicode_tmp_path: Path) -> None:
    # the red side of the check above: the same page with the banner, or a tag list, fails
    page = (server_mod.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    write_text(
        unicode_tmp_path / "with-banner.html", page.replace("</main>", WATERMARK + "</main>")
    )
    tags = "<dl><dt>Restrictions:</dt><dd>no_share</dd></dl>"
    write_text(unicode_tmp_path / "with-tags.html", page.replace("</main>", tags + "</main>"))
    workspace = Workspace(unicode_tmp_path, files=["with-banner.html", "with-tags.html"])
    rules = {(p.path, p.rule) for p in check_restricted_data(unicode_tmp_path, workspace).problems}
    assert ("with-banner.html", "private-banner") in rules
    assert any(path == "with-tags.html" for path, _ in rules)


def test_every_table_of_the_page_names_its_columns_once() -> None:
    script = (server_mod.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    tables = [json.loads(text) for text in re.findall(r"const headers = (\[[^\]]*\]);", script)]
    assert len(tables) == 4  # runs, labels, key moments, measurements
    for names in tables:
        assert len(set(names)) == len(names), names
    assert "类型一致" in tables[1]


def test_the_docked_player_folds_and_leaves_room_at_the_end_of_the_page() -> None:
    # No browser runs in this suite: these check the pieces the docked player relies on
    # (its behaviour was checked once in a browser, on synthetic data). It is at most 420 px
    # wide, folds to its title bar, and while it is open the page ends with room for it, so
    # the last rows can be scrolled clear of it; a new view starts at the top of the page.
    static = server_mod.STATIC_DIR
    page = (static / "index.html").read_text(encoding="utf-8")
    style = (static / "style.css").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")
    assert re.search(r'<button [^>]*id="dock-fold"[^>]*aria-expanded="true"', page)
    dock = re.search(r"\n\.dock \{([^}]*)\}", style)
    assert dock is not None
    width = re.search(r"width: min\((\d+)px", dock[1])
    assert width is not None
    assert int(width[1]) <= 420
    assert re.search(r"\n\.dock\.folded video \{[^}]*display: none", style)
    main = re.search(r"\n\.main \{([^}]*)\}", style)
    assert main is not None
    assert "var(--dock-space" in main[1]
    assert '"--dock-space"' in script
    set_view = re.search(r"function setView\([^)]*\) \{([^}]*)\}", script)
    assert set_view is not None
    assert "scrollTo(0, 0)" in set_view[1]


def test_the_key_frame_rule_can_wrap_on_a_phone() -> None:
    # At 375 px the run page scrolled sideways: the card line's span.num, which never
    # wraps, held the key frame's rule text (refined_by) after the frame and its clock. Only
    # the frame and its clock stay on one line now (checked once in a browser at 375 px, on
    # a synthetic run with refined_by on every highlight: 484 px wide before, 375 px after).
    style = (server_mod.STATIC_DIR / "style.css").read_text(encoding="utf-8")
    script = (server_mod.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert re.search(r"\n\.num \{[^}]*white-space: nowrap", style)
    spans = []
    for match in re.finditer(r'el\(\s*"span",\s*\{ class: "num" \}', script):
        depth, start = 0, match.start() + len("el")
        end = start
        for end in range(start, len(script)):
            depth += {"(": 1, ")": -1}.get(script[end], 0)
            if depth == 0:
                break
        spans.append(script[start : end + 1])
    assert any("card.peak_frame" in span for span in spans)
    assert "key_frame_rule_zh" in script
    assert not any("key_frame_rule_zh" in span for span in spans)


def test_the_page_loads_nothing_from_outside() -> None:
    tags = set(load_schema("restrictions")["$defs"]["tag"]["enum"])
    banner = WATERMARK.split()[0]  # the banner the gate refuses in a page
    for name in server_mod.STATIC_FILES:
        text = (server_mod.STATIC_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"https?://|//[A-Za-z0-9-]+\.[A-Za-z]", text), name
        assert "@import" not in text, name
        assert "url(" not in text, name
        assert banner not in text, name
        assert not any(re.search(rf"\b{re.escape(tag)}\b", text) for tag in tags), name
