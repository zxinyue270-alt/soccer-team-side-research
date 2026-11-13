"""The command line: output format, exit codes, the JSON report, redaction, failures of
the gate itself, and ``python -m gbf gate`` dispatch."""

from __future__ import annotations

import io
import os
import re
import string
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

from gbf import contracts
from gbf.gate import cli
from gbf.gate.model import CheckResult
from gbf.gate.patterns import REDACTED
from gbf.gate.static import STATIC_CHECK_NAMES
from gbf.util.io import read_json, write_text

RunGate = Callable[..., tuple[int, str]]
NoLeak = Callable[..., None]

CHECK_NAMES = ("files", *STATIC_CHECK_NAMES, "ruff", "pytest")
PROBLEM_LINE = re.compile(
    r"(?P<check>" + "|".join(CHECK_NAMES) + r") (?P<path>.+?)(?::(?P<line>\d+))? "
    r"(?P<rule>[A-Za-z0-9_.-]+)(?: \((?P<detail>.*)\))?"
)


def _inject_problems(root: Path, make_token: Callable[..., str]) -> str:
    """CRLF in one file and a home path in another; returns the home-path user name."""
    (root / "docs").mkdir()
    (root / "docs" / "crlf.txt").write_bytes(b"a\r\n")
    name = make_token("gbfuser", 5, string.ascii_lowercase)
    write_text(root / "docs" / "where.md", "x\n" + "/".join(["", "home", name, "a.mp4"]) + "\n")
    return name


def test_every_output_line_is_a_status_line_or_a_problem_line(
    gate_repo: Path, run_gate: RunGate, make_token: Callable[..., str], no_leak: NoLeak
) -> None:
    name = _inject_problems(gate_repo, make_token)
    code, output = run_gate(gate_repo)
    assert code == 1
    problems = [line for line in output.splitlines() if not line.startswith("[gate] ")]
    assert problems == [
        "eol docs/crlf.txt:1 crlf (1 CRLF line endings)",
        "abs_paths docs/where.md:2 user-home-path",
    ]
    for line in problems:
        assert PROBLEM_LINE.fullmatch(line), line
    assert "[gate] eol FAILED (1)" in output
    assert output.splitlines()[-1].startswith("[gate] FAILED: 2 problems in 2 of 10 checks")
    no_leak(output, gate_repo, name)


def test_json_report_is_a_valid_gate_receipt(
    gate_repo: Path,
    run_gate: RunGate,
    unicode_tmp_path: Path,
    make_token: Callable[..., str],
    no_leak: NoLeak,
) -> None:
    name = _inject_problems(gate_repo, make_token)
    report_path = unicode_tmp_path / "reports" / "gate.json"
    code, output = run_gate(gate_repo, "--json", str(report_path))
    assert code == 1
    assert "[gate] JSON report written (gate.json)" in output
    raw = report_path.read_bytes()
    assert b"\r\n" not in raw
    report = read_json(report_path)
    assert contracts.problems(report, "receipt") == []
    assert (report["kind"], report["status"], report["producer"]) == (
        "gate",
        "failed",
        "gbf.gate.cli",
    )
    assert report["code"]["git_commit"] == "unknown"  # the test repository has no commit
    checks = {check["name"]: check for check in report["result"]["checks"]}
    assert list(checks) == ["files", *STATIC_CHECK_NAMES]
    assert checks["eol"]["items"] == ["eol docs/crlf.txt:1 crlf (1 CRLF line endings)"]
    assert checks["eol"]["status"] == "failed"
    assert checks["credentials"] == {**checks["credentials"], "status": "passed", "problems": 0}
    assert [ref["relpath"] for ref in report["inputs"]] == ["pyproject.toml", "uv.lock"]
    assert report["result"]["mode"] == "static"
    # The limitations name both kinds of local path that abs_paths looks for.
    assert (
        "user-home paths and absolute paths through a gbf-data or private folder"
        in (report["limitations"][1])
    )
    no_leak(raw.decode("utf-8"), gate_repo, name)


def test_python_m_gbf_gate_dispatches_to_the_cli(gate_repo: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "gbf", "gate", "--root", str(gate_repo)],
        capture_output=True,
        env={**os.environ, "PYTHONUTF8": "1"},
        check=False,
    )
    output = completed.stdout.decode("utf-8", "replace")
    assert completed.returncode == 0, output
    assert "[gate] PASSED: 10 checks" in output


def test_a_root_that_is_not_a_work_tree_top_fails_cleanly(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    code, output = run_gate(gate_repo / "src")
    assert code == 1
    assert "files - not-a-git-work-tree" in output
    assert "[gate] FAILED: 1 problems in 1 of 1 checks" in output
    assert "Traceback" not in output
    no_leak(output, gate_repo)


def test_a_counts_base_that_looks_like_an_option_is_a_usage_error(gate_repo: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["--root", str(gate_repo), "--counts-base=--output=x"])
    assert caught.value.code == 2


def test_printed_lines_pass_through_redaction(
    gate_repo: Path, run_gate: RunGate, make_token: Callable[..., str]
) -> None:
    token = make_token("s" + "k-", 40)
    (gate_repo / "docs").mkdir()
    (gate_repo / "docs" / f"{token}.txt").write_bytes(b"a\r\n")
    code, output = run_gate(gate_repo)
    assert code == 1
    assert token not in output
    assert f"eol docs/{REDACTED}.txt:1 crlf (1 CRLF line endings)" in output


def test_a_crashing_check_fails_without_printing_its_message(
    gate_repo: Path,
    run_gate: RunGate,
    monkeypatch: pytest.MonkeyPatch,
    make_token: Callable[..., str],
    no_leak: NoLeak,
) -> None:
    secret = make_token("crash-message-", 12)

    def explode(root: Path, workspace: object = None) -> CheckResult:
        raise RuntimeError(secret)

    monkeypatch.setattr(cli, "STATIC_CHECKS", (("eol", explode),))
    code, output = run_gate(gate_repo)
    assert code == 1
    assert "eol - internal-error (RuntimeError)" in output
    no_leak(output, gate_repo, secret)


def test_wall_time_over_budget_warns_without_failing(
    gate_repo: Path, run_gate: RunGate, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "WALL_TIME_BUDGET_S", 0.0)
    code, output = run_gate(gate_repo)
    assert code == 0
    assert "[gate] warning: wall time" in output
    assert "budget (G1)" in output


def test_output_survives_a_console_that_cannot_encode_file_names(
    gate_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="ascii", newline="\n")
    monkeypatch.setattr(sys, "stdout", stream)
    (gate_repo / "docs").mkdir()
    (gate_repo / "docs" / "帧.txt").write_bytes(b"a\r\n")
    code = cli.main(["--root", str(gate_repo)])
    stream.flush()
    output = buffer.getvalue().decode("ascii")
    escaped = chr(92) + "u5e27"
    assert code == 1
    assert f"eol docs/{escaped}.txt:1 crlf (1 CRLF line endings)" in output.splitlines()


def test_a_crashing_full_step_fails_without_a_traceback(
    gate_repo: Path,
    run_gate: RunGate,
    unicode_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_token: Callable[..., str],
    no_leak: NoLeak,
) -> None:
    secret = make_token("step-message-", 12)

    def explode(*args: object, **kwargs: object) -> CheckResult:
        raise RuntimeError(secret)

    monkeypatch.setattr(cli, "run_ruff_check", explode)
    monkeypatch.setattr(cli, "run_pytest_check", explode)
    monkeypatch.setattr(tempfile, "tempdir", str(unicode_tmp_path))
    code, output = run_gate(gate_repo, "--full")
    assert code == 1
    assert "ruff - internal-error (RuntimeError)" in output
    assert "pytest - internal-error (RuntimeError)" in output
    assert "Traceback" not in output
    no_leak(output, gate_repo, secret)
