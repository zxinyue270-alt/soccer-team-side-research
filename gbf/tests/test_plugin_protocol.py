"""Tests for the plugin protocol (gbf.plugins): job.json, run_plugin, accept_outputs, the
licence allowlist, the calib.v1 Arrow contract and the stand-in calibration plugin.

accept_outputs is tested red-green: one clean, in-process "plugin run" is accepted, and each
fault injected into a copy of it (tampered output, missing output, GPL licence without an
explicit allow, schema-invalid receipt, ...) must be rejected with that specific problem.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gbf import contracts
from gbf.geometry import scale_intrinsics
from gbf.ingest import iter_frames, pixel_sha256
from gbf.plugins import (
    CALIB_SCHEMA,
    PERMISSIVE_LICENSES,
    CalibFormatError,
    CalibFrame,
    JobError,
    JobInput,
    OutputsRejectedError,
    PluginRunError,
    accept_outputs,
    build_job,
    create_job,
    effective_allowlist,
    frame_from_camera,
    job_sha256,
    license_problems,
    plugin_environment,
    read_calib,
    run_plugin,
    write_calib,
    write_job,
)
from gbf.plugins.licenses import LicenseExpressionError, checked_allowlist
from gbf.synth.render import sequence_cameras
from gbf.synth.video import write_frames_mp4
from gbf.util.hashing import canonical_sha256, sha256_file
from gbf.util.io import read_json, write_json

STUB = Path(__file__).parent / "plugin_stubs" / "stub_calib.py"
FIRST, COUNT = 10, 3  # the job's input holds canonical frames 10..12
START, END = 10, 12
CALIB_OUTPUT = {"name": "calib", "relpath": "out/calib.parquet", "format": "calib.v1"}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


FRAME_HASHES = {index: _digest(f"pixels of frame {index}") for index in range(FIRST, FIRST + COUNT)}


def make_job(
    parent: Path,
    *,
    allow: tuple[str, ...] = (),
    restrictions: tuple[str, ...] = (),
    expected_frames: tuple[int, ...] = (START, END),
) -> Path:
    """A calibration job over canonical frames 10..12; returns the path of job.json."""
    sources = parent / "sources"
    sources.mkdir(exist_ok=True)
    video = sources / "window.mkv"
    video.write_bytes(b"stand-in bytes for a window file")
    cameras = sources / "cameras.json"
    cameras.write_bytes(b'{"format": "gbf.synth.cameras.v1", "frames": []}\n')
    return create_job(
        parent / "plugins",
        task="calibration",
        inputs=[
            JobInput("video", video, "in/video.mkv", restrictions),
            JobInput("recorded_cameras", cameras, "in/cameras.json"),
        ],
        frames={
            "canonical_id": _digest("source video"),
            "fps": "25/1",
            "input": "video",
            "input_first_frame": FIRST,
            "input_frame_count": COUNT,
            "start": START,
            "end": END,
            "width": 1280,
            "height": 720,
            "pixel_sha256": [
                {"frame_index": index, "sha256": FRAME_HASHES[index]} for index in expected_frames
            ],
        },
        outputs=[CALIB_OUTPUT],
        config={"recorded_cameras": "recorded_cameras"},
        allow_licenses=allow,
    )


def reference_cameras() -> dict[int, Any]:
    """Distinct cameras at 1920x1080 for canonical frames 10..12."""
    cameras = sequence_cameras(1920, 1080, 15, 11)
    return {index: cameras[index].camera for index in range(START, END + 1)}


def simulate_plugin(job_path: Path, *, license: str = "MIT", frames: range | None = None) -> dict:
    """Do in-process what a well-behaved plugin does: write calib.parquet and the receipt."""
    job_dir = job_path.parent
    job = read_json(job_path)
    cameras = reference_cameras()
    indices = frames if frames is not None else range(START, END + 1)
    write_calib(
        job_dir / CALIB_OUTPUT["relpath"],
        [frame_from_camera(index, cameras.get(index, cameras[START])) for index in indices],
    )
    target = job_dir / CALIB_OUTPUT["relpath"]
    receipt = {
        "schema_version": 1,
        "job_sha256": sha256_file(job_path),
        "status": "ok",
        "plugin": {
            "name": "simulated-calib",
            "version": "1.0",
            "license": license,
            "commit": "0" * 40,
            "weights": [{"name": "w.pt", "sha256": _digest("weights")}],
        },
        "outputs": [
            {**CALIB_OUTPUT, "sha256": sha256_file(target), "bytes": target.stat().st_size}
        ],
        "decoded_frames": [
            {"frame_index": index, "sha256": FRAME_HASHES[index]}
            for index in range(FIRST, FIRST + COUNT)
        ],
        "environment": {"python": "3.12"},
        "restrictions": list(job["restrictions"]),
    }
    write_json(job_dir / "plugin-receipt.json", receipt)
    return receipt


def edit_receipt(job_dir: Path, change) -> None:
    receipt = read_json(job_dir / "plugin-receipt.json")
    change(receipt)
    write_json(job_dir / "plugin-receipt.json", receipt)


def rejection(job_dir: Path, **kwargs: Any) -> list[str]:
    with pytest.raises(OutputsRejectedError) as caught:
        accept_outputs(job_dir, **kwargs)
    return caught.value.problems


@pytest.fixture
def finished_job(unicode_tmp_path: Path) -> Path:
    """A job directory after a clean simulated plugin run; returns the job directory."""
    job_path = make_job(unicode_tmp_path)
    simulate_plugin(job_path)
    return job_path.parent


# --- accept_outputs: green ----------------------------------------------------------------
def test_clean_run_is_accepted(finished_job: Path) -> None:
    accepted = accept_outputs(finished_job)
    assert set(accepted.outputs) == {"calib"}
    assert accepted.outputs["calib"] == finished_job / "out" / "calib.parquet"
    assert accepted.job_sha256 == sha256_file(finished_job / "job.json")
    assert accepted.plugin["license"] == "MIT"
    frames = read_calib(accepted.outputs["calib"])
    assert [frame.frame_index for frame in frames] == [10, 11, 12]
    refs = accepted.output_refs(finished_job.parent)
    assert refs[0]["relpath"] == f"{finished_job.name}/out/calib.parquet"
    assert refs[0]["sha256"] == sha256_file(accepted.outputs["calib"])


def test_job_directory_is_named_by_the_job_hash(finished_job: Path) -> None:
    job = read_json(finished_job / "job.json")
    assert finished_job.name == f"calibration-{job_sha256(job)[:16]}"
    assert sha256_file(finished_job / "job.json") == job_sha256(job)
    assert (finished_job / "job.json").read_bytes().count(b"\r\n") == 0
    assert job["allowed_licenses"] == sorted(PERMISSIVE_LICENSES, key=str.casefold)
    assert job["config_sha256"] == canonical_sha256(job["config"])


# --- accept_outputs: red (one fault each) -------------------------------------------------
def test_tampered_output_is_rejected(finished_job: Path) -> None:
    target = finished_job / "out" / "calib.parquet"
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF  # same size, different bytes
    target.write_bytes(bytes(data))
    problems = rejection(finished_job)
    assert len(problems) == 1, problems
    assert "output 'calib'" in problems[0]
    assert "sha256 mismatch" in problems[0]


def test_missing_output_is_rejected(finished_job: Path) -> None:
    (finished_job / "out" / "calib.parquet").unlink()
    problems = rejection(finished_job)
    assert problems == ["output 'calib' (out/calib.parquet): missing from the job directory"]


def test_output_not_listed_in_the_receipt_is_rejected(finished_job: Path) -> None:
    edit_receipt(finished_job, lambda r: r.update(outputs=[]))
    problems = rejection(finished_job)
    assert problems == [
        "output 'calib' (out/calib.parquet): requested by the job but not listed in the receipt"
    ]


def test_gpl_plugin_without_allow_is_rejected(finished_job: Path) -> None:
    edit_receipt(finished_job, lambda r: r["plugin"].update(license="GPL-3.0-only"))
    problems = rejection(finished_job)
    assert len(problems) == 1, problems
    assert "plugin licence" in problems[0]
    assert "not allowed: GPL-3.0-only" in problems[0]
    assert "allow_licenses" in problems[0]


def test_gpl_plugin_with_explicit_allow_is_accepted(unicode_tmp_path: Path) -> None:
    job_path = make_job(unicode_tmp_path, allow=("GPL-3.0-only",))
    simulate_plugin(job_path, license="GPL-3.0-only")
    assert "GPL-3.0-only" in read_json(job_path)["allowed_licenses"]
    accepted = accept_outputs(job_path.parent, allow_licenses=["GPL-3.0-only"])
    assert accepted.plugin["license"] == "GPL-3.0-only"
    # The user must say it again at acceptance time: the job alone is not enough.
    problems = rejection(job_path.parent)
    assert len(problems) == 1
    assert "not allowed: GPL-3.0-only" in problems[0]


def test_allow_at_acceptance_without_the_job_allowing_it_is_rejected(finished_job: Path) -> None:
    edit_receipt(finished_job, lambda r: r["plugin"].update(license="GPL-3.0-only"))
    problems = rejection(finished_job, allow_licenses=["GPL-3.0-only"])
    assert len(problems) == 1, problems
    assert "job's allowed_licenses do not cover it" in problems[0]


def test_schema_invalid_receipt_is_rejected(finished_job: Path) -> None:
    edit_receipt(finished_job, lambda r: r.pop("environment"))
    problems = rejection(finished_job)
    assert problems == ["plugin-receipt.json: <root>: 'environment' is a required property"]


def test_receipt_with_an_absolute_output_path_is_rejected(finished_job: Path) -> None:
    absolute = str((finished_job / "out" / "calib.parquet").resolve())
    edit_receipt(finished_job, lambda r: r["outputs"][0].update(relpath=absolute))
    problems = rejection(finished_job)
    assert problems
    assert all(p.startswith("plugin-receipt.json: outputs/0/relpath") for p in problems)


def test_every_problem_is_reported_at_once(finished_job: Path) -> None:
    target = finished_job / "out" / "calib.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    (finished_job / "out" / "extra.log").write_text("debug\n", encoding="utf-8")

    def change(receipt: dict) -> None:
        receipt["plugin"]["license"] = "AGPL-3.0-only"
        receipt["restrictions"] = []
        receipt["decoded_frames"] = receipt["decoded_frames"][1:]

    edit_receipt(finished_job, change)
    job = read_json(finished_job / "job.json")
    assert job["restrictions"] == []  # nothing to drop in this job
    problems = rejection(finished_job)
    text = "\n".join(problems)
    assert "bytes on disk" in text
    assert "sha256 mismatch" in text
    assert "not allowed: AGPL-3.0-only" in text
    assert "out/extra.log: not an output the job requested" in text
    assert "frame 10: the plugin reported no pixel hash" in text
    assert len(problems) == 5, problems


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda r: r.update(job_sha256="0" * 64), "written for another job"),
        (lambda r: r.update(status="failed", messages=["model crashed"]), "status 'failed'"),
        (
            lambda r: r["decoded_frames"][2].update(sha256=_digest("other pixels")),
            "frame 12: the plugin decoded different pixels",
        ),
        (
            lambda r: r["decoded_frames"].append({"frame_index": 13, "sha256": _digest("x")}),
            "frames [13] are outside the input (10..12)",
        ),
        (
            lambda r: r["outputs"].append({**r["outputs"][0], "name": "bonus"}),
            "output 'bonus': listed in the receipt but not requested",
        ),
        (lambda r: r["outputs"][0].update(format="calib.v0"), "format 'calib.v0'"),
    ],
)
def test_receipt_faults_are_named(finished_job: Path, change, expected: str) -> None:
    edit_receipt(finished_job, change)
    problems = rejection(finished_job)
    assert len(problems) == 1, problems
    assert expected in problems[0]


def test_missing_receipt_is_rejected(finished_job: Path) -> None:
    (finished_job / "plugin-receipt.json").unlink()
    assert rejection(finished_job) == ["plugin-receipt.json: missing"]


def test_edited_job_is_rejected(finished_job: Path) -> None:
    job = read_json(finished_job / "job.json")
    job["frames"]["start"] = 11
    write_json(finished_job / "job.json", job)
    problems = rejection(finished_job)
    assert any("written for another job" in p for p in problems)
    assert any("rows for frame(s) [10] outside the job's frames 11..12" in p for p in problems)


def test_changed_input_is_rejected(finished_job: Path) -> None:
    (finished_job / "in" / "cameras.json").write_bytes(b"{}\n")
    problems = rejection(finished_job)
    assert problems == ["input 'recorded_cameras' (in/cameras.json): changed while the plugin ran"]


def test_dropped_restriction_is_rejected(unicode_tmp_path: Path) -> None:
    job_path = make_job(unicode_tmp_path, restrictions=("nda",))
    simulate_plugin(job_path)
    assert read_json(job_path)["restrictions"] == ["nda"]
    accept_outputs(job_path.parent)
    edit_receipt(job_path.parent, lambda r: r.update(restrictions=[]))
    problems = rejection(job_path.parent)
    assert problems == ["plugin-receipt.json: drops restriction(s) ['nda'] carried by the inputs"]


def test_calib_rows_must_cover_the_job_frames(unicode_tmp_path: Path) -> None:
    job_path = make_job(unicode_tmp_path)
    simulate_plugin(job_path, frames=range(0, 3))  # file-local numbers instead of canonical
    problems = rejection(job_path.parent)
    assert len(problems) == 2, problems
    assert "no row for canonical frame(s) [10, 11, 12]" in problems[0]
    assert "rows for frame(s) [0, 1, 2] outside the job's frames 10..12" in problems[1]


def test_invalid_calib_content_is_rejected(finished_job: Path) -> None:
    target = finished_job / "out" / "calib.parquet"
    with pq.ParquetFile(target) as handle:
        table = handle.read()
    target.unlink()
    broken = table.set_column(
        table.schema.get_field_index("calib_status"),
        pa.field("calib_status", pa.string(), nullable=False),
        pa.array(["fresh", "stale", "fresh"], pa.string()),
    )
    pq.write_table(broken, target)
    edit_receipt(
        finished_job,
        lambda r: r["outputs"][0].update(sha256=sha256_file(target), bytes=target.stat().st_size),
    )
    problems = rejection(finished_job)
    assert len(problems) == 1, problems
    assert "frame 11: calib_status 'stale' is not one of fresh, held, none" in problems[0]


# --- jobs ---------------------------------------------------------------------------------
def test_same_job_twice_is_refused(unicode_tmp_path: Path) -> None:
    """G17: an existing job directory is an error, and the earlier run stays untouched."""
    job_path = make_job(unicode_tmp_path)
    simulate_plugin(job_path)
    before = {
        path.relative_to(job_path.parent).as_posix(): path.read_bytes()
        for path in job_path.parent.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        make_job(unicode_tmp_path)
    after = {
        path.relative_to(job_path.parent).as_posix(): path.read_bytes()
        for path in job_path.parent.rglob("*")
        if path.is_file()
    }
    assert after == before
    accept_outputs(job_path.parent)


def test_write_job_reports_every_problem(unicode_tmp_path: Path) -> None:
    job_dir = unicode_tmp_path / "manual"
    (job_dir / "in").mkdir(parents=True)
    (job_dir / "in" / "a.bin").write_bytes(b"abc")
    entry = {
        "name": "video",
        "relpath": "in/a.bin",
        "sha256": hashlib.sha256(b"abd").hexdigest(),
        "bytes": 3,
        "restrictions": ["nda"],
    }
    job = build_job(
        task="calibration",
        inputs=[entry, {**entry, "relpath": "in/missing.bin", "restrictions": []}],
        frames={
            "canonical_id": _digest("v"),
            "fps": "25/1",
            "input": "frames_input",
            "input_first_frame": 5,
            "input_frame_count": 2,
            "start": 4,
            "end": 6,
            "width": 64,
            "height": 48,
            "pixel_sha256": [{"frame_index": 9, "sha256": _digest("p")}],
        },
        outputs=[CALIB_OUTPUT],
        config={"k": 1},
    )
    job["config_sha256"] = canonical_sha256({"k": 2})
    job["restrictions"] = []
    with pytest.raises(JobError) as caught:
        write_job(job_dir, job)
    text = "\n".join(caught.value.problems)
    for expected in (
        "config_sha256 is not the sha256",
        "input name 'video' is used more than once",
        "frames.input 'frames_input' is not one of the inputs",
        "frames 4..6 must be a non-empty range inside the input's canonical frames 5..6",
        "frames.pixel_sha256 names frames outside the input: [9]",
        "missing ['nda']",
        "input 'video' (in/a.bin) does not match its sha256",
        "input 'video' (in/missing.bin) is not a file in the job directory",
    ):
        assert expected in text, expected
    assert len(caught.value.problems) == 8
    assert not (job_dir / "job.json").exists()


def test_write_job_schema_problems(unicode_tmp_path: Path) -> None:
    job = build_job(
        task="detection",
        inputs=[
            {
                "name": "video",
                "relpath": "../video.mkv",
                "sha256": _digest("v"),
                "bytes": 1,
                "restrictions": [],
            }
        ],
        frames={},
        outputs=[{**CALIB_OUTPUT, "relpath": "calib.parquet"}],
        config={},
    )
    with pytest.raises(JobError) as caught:
        write_job(unicode_tmp_path, job)
    text = "\n".join(caught.value.problems)
    assert "job: task: 'detection' is not one of ['calibration']" in text
    assert "job: inputs/0/relpath" in text
    assert "job: outputs/0/relpath" in text
    assert "job: frames: 'canonical_id' is a required property" in text


def test_existing_job_file_is_never_rewritten(finished_job: Path) -> None:
    job = read_json(finished_job / "job.json")
    with pytest.raises(FileExistsError):
        write_job(finished_job, job)


def test_bad_allowlist_entries_are_refused(unicode_tmp_path: Path) -> None:
    with pytest.raises(LicenseExpressionError, match="MIT OR GPL-3.0-only"):
        make_job(unicode_tmp_path, allow=("MIT OR GPL-3.0-only",))
    with pytest.raises(LicenseExpressionError):
        checked_allowlist("GPL-3.0-only")


# --- licences -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("expression", "ok"),
    [
        ("MIT", True),
        ("mit", True),
        ("Apache-2.0 OR GPL-3.0-only", True),
        ("(MIT OR BSD-3-Clause) AND Zlib", True),
        ("Apache-2.0 WITH LLVM-exception", True),
        ("MIT AND GPL-3.0-only", False),
        ("GPL-3.0-only WITH Classpath-exception-2.0", False),
        ("GPL-3.0", False),
        ("GPL-2.0+", False),
        ("LicenseRef-Proprietary", False),
        ("NONE", False),
        ("NOASSERTION", False),
    ],
)
def test_permissive_allowlist(expression: str, ok: bool) -> None:
    assert (license_problems(expression, PERMISSIVE_LICENSES) == []) is ok


@pytest.mark.parametrize("expression", ["", "MIT OR", "MIT or Apache-2.0", "(MIT", "MIT)", "AND"])
def test_malformed_licence_expressions_are_problems(expression: str) -> None:
    problems = license_problems(expression, PERMISSIVE_LICENSES)
    assert len(problems) == 1
    assert "license expression" in problems[0]


def test_explicit_allow_extends_the_allowlist() -> None:
    allowed = effective_allowlist(["GPL-3.0-only", "NOASSERTION", "mit"])
    assert license_problems("GPL-3.0-only WITH Classpath-exception-2.0", allowed) == []
    assert license_problems("NOASSERTION", allowed) == []
    assert allowed.count("MIT") == 1
    assert "mit" not in allowed
    assert set(PERMISSIVE_LICENSES) <= set(allowed)


# --- calib.v1 -----------------------------------------------------------------------------
def calib_frames() -> list[CalibFrame]:
    cameras = reference_cameras()
    fresh = frame_from_camera(10, cameras[10], rep_err=0.75)
    held = frame_from_camera(11, cameras[10], calib_status="held", calib_age=1)
    none = CalibFrame(12, 1920, 1080, None, None, None, "none", 0, None)
    return [fresh, held, none]


def test_calib_round_trip_and_rescale(unicode_tmp_path: Path) -> None:
    path = unicode_tmp_path / "calib.parquet"
    write_calib(path, calib_frames())
    frames = read_calib(path)
    assert [(f.frame_index, f.calib_status, f.calib_age, f.rep_err) for f in frames] == [
        (10, "fresh", 0, 0.75),
        (11, "held", 1, None),
        (12, "none", 0, None),
    ]
    camera = reference_cameras()[10]
    assert np.array_equal(frames[0].K, camera.K)
    assert np.array_equal(frames[1].R, camera.R)
    assert frames[2].camera() is None
    small = frames[0].camera(1280, 720)
    assert small is not None
    assert small.size == (1280, 720)
    expected = scale_intrinsics(camera.K, (1920, 1080), (1280, 720))
    np.testing.assert_allclose(small.K, expected, rtol=1e-15, atol=0)
    with pq.ParquetFile(path) as handle:
        assert handle.schema_arrow.equals(CALIB_SCHEMA)
    with pytest.raises(FileExistsError):
        write_calib(path, calib_frames())


def test_calib_schema_problems_are_listed(unicode_tmp_path: Path) -> None:
    path = unicode_tmp_path / "wrong-schema.parquet"
    table = pa.table(
        {
            "frame_index": pa.array([0], pa.int64()),
            "width": pa.array([1920], pa.int64()),
            "K": pa.array([[1.0] * 9], pa.list_(pa.float64(), 9)),
            "colour": pa.array(["red"]),
        }
    )
    pq.write_table(table, path)
    with pytest.raises(CalibFormatError) as caught:
        read_calib(path)
    text = "\n".join(caught.value.problems)
    assert "missing column(s): height, R, C, calib_status, calib_age, rep_err" in text
    assert "unexpected column(s): colour" in text
    assert "column width: type int64, expected int32" in text


def test_calib_value_problems_are_listed(unicode_tmp_path: Path) -> None:
    good = calib_frames()
    bad_K = np.array(good[0].K)
    bad_K[2, 2] = 2.0
    rows = [
        {**good[0].to_row(), "frame_index": 5, "calib_age": 2},
        {**good[1].to_row(), "frame_index": 4, "calib_age": 0},
        {**good[2].to_row(), "frame_index": 6, "K": [1.0] * 9},
        {**good[0].to_row(), "frame_index": 7, "R": None, "rep_err": -1.0},
        {**good[0].to_row(), "frame_index": 8, "K": [float(v) for v in bad_K.ravel()]},
        {**good[0].to_row(), "frame_index": 9, "calib_status": "stale"},
    ]
    path = unicode_tmp_path / "bad-values.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=CALIB_SCHEMA), path)
    with pytest.raises(CalibFormatError) as caught:
        read_calib(path)
    text = "\n".join(caught.value.problems)
    for expected in (
        "frame_index must be strictly increasing",
        "frame 5: calib_age must be 0 for a fresh calibration, got 2",
        "frame 4: calib_age must be >= 1 for a held calibration, got 0",
        "frame 6: K, R and C must be null when calib_status is 'none'",
        "frame 7: R must not be null for status 'fresh'",
        "frame 7: rep_err must be a finite number >= 0 or null, got -1.0",
        "frame 8: K[2,2] = 2; K must be normalised to K[2,2] = 1",
        "frame 9: calib_status 'stale' is not one of fresh, held, none",
    ):
        assert expected in text, expected


def test_non_parquet_file_is_a_format_error(unicode_tmp_path: Path) -> None:
    path = unicode_tmp_path / "calib.parquet"
    path.write_bytes(b"not parquet")
    with pytest.raises(CalibFormatError, match="cannot read as parquet"):
        read_calib(path)


# --- running plugins ----------------------------------------------------------------------
def test_plugin_environment_drops_secrets_and_strangers() -> None:
    base = {
        "PATH": "p",
        "SystemRoot": "s",
        "GBF_DATA_ROOT": "d",
        "GBF_HF_TOKEN": "t1",
        "HF_TOKEN": "t2",
        "OPENAI_API_KEY": "t3",
        "HTTPS_PROXY": "x",
        "USERNAME": "someone",
        "PYTHONPATH": "guard",
    }
    env = plugin_environment(base)
    assert env == {
        "PATH": "p",
        "SystemRoot": "s",
        "GBF_DATA_ROOT": "d",
        "PYTHONPATH": "guard",
        "PYTHONUTF8": "1",
        "PYTHONHASHSEED": "0",
    }


def test_run_plugin_argument_problems(unicode_tmp_path: Path) -> None:
    other = unicode_tmp_path / "other.json"
    other.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PluginRunError) as caught:
        run_plugin("python stub.py", other, 0)
    assert len(caught.value.problems) == 3
    with pytest.raises(PluginRunError) as caught:
        run_plugin([sys.executable, ""], unicode_tmp_path / "job.json", float("nan"))
    text = "\n".join(caught.value.problems)
    assert "command[1] must be a non-empty string" in text
    assert "job file does not exist" in text
    assert "timeout_s must be finite and > 0" in text


def test_run_plugin_failures(finished_job: Path) -> None:
    job = finished_job / "job.json"
    with pytest.raises(PluginRunError) as caught:
        run_plugin(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('model weights missing'); sys.exit(3)",
            ],
            job,
            60,
        )
    assert caught.value.problems == ["exited with code 3"]
    assert "model weights missing" in caught.value.stderr_tail
    with pytest.raises(PluginRunError) as caught:
        run_plugin([sys.executable, "-c", "import time; time.sleep(30)"], job, 0.5)
    assert caught.value.problems == ["timed out after 0.5 s and was stopped"]
    with pytest.raises(PluginRunError) as caught:
        run_plugin([str(finished_job / "no such program.exe")], job, 5)
    assert caught.value.problems[0].startswith("cannot start")


def test_run_plugin_passes_a_clean_environment(finished_job: Path, monkeypatch) -> None:
    monkeypatch.setenv("GBF_TEST_SECRET_API_KEY", "should-not-pass")
    monkeypatch.setenv("SOME_UNRELATED_VARIABLE", "should-not-pass")
    script = (
        "import os, sys, json; sys.stdout.write(json.dumps("
        "{'cwd': os.getcwd(), 'arg': sys.argv[1], 'env': sorted(os.environ)}))"
    )
    run = run_plugin([sys.executable, "-c", script], finished_job / "job.json", 60)
    seen = json.loads(run.stdout_tail)
    assert Path(seen["cwd"]).resolve() == finished_job.resolve()
    assert Path(seen["arg"]).resolve() == (finished_job / "job.json").resolve()
    upper = {name.upper() for name in seen["env"]}
    assert "GBF_TEST_SECRET_API_KEY" not in upper
    assert "SOME_UNRELATED_VARIABLE" not in upper
    assert {"PATH", "PYTHONUTF8"} <= upper


# --- the stand-in plugin ------------------------------------------------------------------
def test_stub_imports_only_the_standard_library_and_pyarrow() -> None:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".")[0])
    assert modules, "no imports found"
    assert "gbf" not in modules
    assert modules <= set(sys.stdlib_module_names) | {"pyarrow", "__future__"}, modules


def test_stub_schema_matches_the_core_schema() -> None:
    spec = importlib.util.spec_from_file_location("stub_calib_for_test", STUB)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CALIB_SCHEMA.equals(CALIB_SCHEMA)
    assert [field.nullable for field in module.CALIB_SCHEMA] == [
        field.nullable for field in CALIB_SCHEMA
    ]


def test_stub_round_trip_at_a_window_offset(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    """The real subprocess path: a 3-frame window holding canonical frames 7..9."""
    assert ffmpeg_bin
    width, height = 64, 48
    frames = [np.full((height, width, 3), 40 * (k + 1), dtype=np.uint8) for k in range(3)]
    video = unicode_tmp_path / "window 7-9.mp4"
    write_frames_mp4(frames, video)
    decoded = [frame for _, frame in iter_frames(video)]
    cameras = sequence_cameras(width, height, 12, 8)
    record = {
        "format": "gbf.synth.cameras.v1",
        "frames": [item.to_dict() for item in cameras],
    }
    write_json(unicode_tmp_path / "cameras.json", record)
    job_path = create_job(
        unicode_tmp_path / "plugins",
        task="calibration",
        inputs=[
            JobInput("video", video, "in/video.mp4"),
            JobInput("recorded_cameras", unicode_tmp_path / "cameras.json", "in/cameras.json"),
        ],
        frames={
            "canonical_id": _digest("source"),
            "fps": "25/1",
            "input": "video",
            "input_first_frame": 7,
            "input_frame_count": 3,
            "start": 7,
            "end": 9,
            "width": width,
            "height": height,
            "pixel_sha256": [
                {"frame_index": 7 + k, "sha256": pixel_sha256(frame)}
                for k, frame in enumerate(decoded)
            ],
        },
        outputs=[CALIB_OUTPUT],
        config={"recorded_cameras": "recorded_cameras"},
    )
    run = run_plugin([sys.executable, str(STUB)], job_path, 120)
    assert run.returncode == 0
    accepted = accept_outputs(job_path.parent)
    receipt = read_json(job_path.parent / "plugin-receipt.json")
    contracts.validate(receipt, "plugin-receipt")
    assert [item["frame_index"] for item in receipt["decoded_frames"]] == [7, 8, 9]
    rows = read_calib(accepted.outputs["calib"])
    assert [row.frame_index for row in rows] == [7, 8, 9]
    assert {(row.width, row.height) for row in rows} == {(1920, 1080)}
    for row in rows:
        truth = cameras[row.frame_index].camera
        camera = row.camera(width, height)
        assert camera is not None
        np.testing.assert_allclose(camera.K, truth.K, rtol=1e-12, atol=0)
        assert np.array_equal(camera.R, truth.R)
        assert np.array_equal(camera.C, truth.C)


def test_stub_refuses_a_job_that_does_not_allow_mit(unicode_tmp_path: Path) -> None:
    job_path = make_job(unicode_tmp_path)
    job = read_json(job_path)
    job_path.unlink()
    job["allowed_licenses"] = ["Apache-2.0"]
    write_job(job_path.parent, job)
    with pytest.raises(PluginRunError) as caught:
        run_plugin([sys.executable, str(STUB)], job_path, 60)
    assert caught.value.problems == ["exited with code 3"]
    assert "does not allow this plugin's licence MIT" in caught.value.stderr_tail
    assert not (job_path.parent / "plugin-receipt.json").exists()
