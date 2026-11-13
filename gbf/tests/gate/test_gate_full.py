"""The --full steps: sanitised environment, JUnit and summary parsing, minimum test
counts, and real ruff / pytest subprocess runs on tiny temporary repositories."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

from gbf.gate.full import (
    MinCounts,
    evaluate_counts,
    is_valid_revision,
    load_baseline_counts,
    outcome_problems,
    parse_junit,
    parse_min_counts,
    parse_summary,
    run_pytest_check,
    run_ruff_check,
    sanitized_env,
    suite_counts,
)
from gbf.gate.model import CheckResult
from gbf.util.io import write_json, write_text

RunGate = Callable[..., tuple[int, str]]
ADR = "docs/adr/0009-split-suite.md"


def lines(result: CheckResult) -> list[str]:
    return [p.format() for p in result.sorted_problems()]


# --- environment -----------------------------------------------------------------------
def test_sanitized_env_keeps_only_what_python_git_and_ffmpeg_need() -> None:
    kept = {
        "PATH": "p",
        "SystemRoot": "s",
        "TEMP": "t",
        "HOME": "h",
        "LANG": "C.UTF-8",
        "GBF_DATA_ROOT": "d",
    }
    dropped = {
        name: "x"
        for name in (
            "GBF_HF_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "HF_TOKEN",
            "HF_HOME",
            "KAGGLE_KEY",
            "KAGGLE_USERNAME",
            "AWS_SECRET_ACCESS_KEY",
            "ARK_API_KEY",
            "DOUBAO_API_KEY",
            "DEEPSEEK_API_KEY",
            "ELEVENLABS_API_KEY",
            "MOSS_API_KEY",
            "DASHSCOPE_API_KEY",
            "QWEN_API_KEY",
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "GIT_DIR",
            "HTTPS_PROXY",
            "PYTEST_ADDOPTS",
            "PYTHONWARNINGS",
            "VIRTUAL_ENV",
        )
    }
    base = {**kept, **dropped, "PYTHONHASHSEED": "random", "PYTHONPATH": "inherited"}
    env = sanitized_env(base, python_path=[Path("site")])
    assert env == {
        **kept,
        "PYTHONUTF8": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": os.pathsep.join(["site", "inherited"]),
    }


def test_sanitized_env_defaults_to_this_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GBF_GATE_PROBE", "kept")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dropped")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = sanitized_env()
    assert env["GBF_GATE_PROBE"] == "kept"
    assert "ELEVENLABS_API_KEY" not in env
    assert "PYTHONPATH" not in env
    assert env["PYTHONUTF8"] == "1"


# --- JUnit and summary -----------------------------------------------------------------
def _junit() -> bytes:
    windows_file = chr(92).join(["tests", "gate", "test_b.py"])
    cases = [
        '<testcase classname="tests.test_a" name="test_ok" file="tests/test_a.py" line="1"/>',
        '<testcase classname="tests.test_a" name="test_fail" file="tests/test_a.py" line="4">'
        '<failure message="assert 0">tb</failure></testcase>',
        '<testcase classname="tests.test_a" name="test_err" file="tests/test_a.py" line="7">'
        '<error message="failed on setup with &quot;E&quot;">tb</error></testcase>',
        '<testcase classname="tests.test_a" name="test_skip" file="tests/test_a.py" line="9">'
        '<skipped type="pytest.skip" message="why">x</skipped></testcase>',
        '<testcase classname="tests.test_a" name="test_xf" file="tests/test_a.py" line="11">'
        '<skipped type="pytest.xfail" message="known"/></testcase>',
        '<testcase classname="tests.test_a" name="test_xp" file="tests/test_a.py" line="13">'
        '<failure message="[XPASS(strict)] r">x</failure></testcase>',
        f'<testcase classname="tests.gate.test_b.TestK" name="test_p[a/b]" file="{windows_file}"'
        ' line="20"/>',
        f'<testcase classname="tests.gate.test_b.TestK" name="test_p[c]" file="{windows_file}"'
        ' line="20"/>',
        '<testcase classname="" name="tests.test_broken" file="tests/test_broken.py">'
        '<error message="collection failure">SyntaxError</error></testcase>',
    ]
    body = "".join(cases)
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
        f"{body}</testsuite></testsuites>"
    ).encode()


def test_parse_junit_classifies_every_outcome() -> None:
    cases = parse_junit(_junit())
    assert [(c.file, c.line, c.function, c.outcomes) for c in cases] == [
        ("tests/test_a.py", 2, "test_ok", ()),
        ("tests/test_a.py", 5, "test_fail", ("failed",)),
        ("tests/test_a.py", 8, "test_err", ("error",)),
        ("tests/test_a.py", 10, "test_skip", ("skipped",)),
        ("tests/test_a.py", 12, "test_xf", ("xfail",)),
        ("tests/test_a.py", 14, "test_xp", ("xpass",)),
        ("tests/gate/test_b.py", 21, "test_p", ()),
        ("tests/gate/test_b.py", 21, "test_p", ()),
        ("tests/test_broken.py", None, "tests", ("collection-error",)),
    ]
    assert suite_counts(cases, "tests") == {"gate/test_b.py": 2, "test_a.py": 6}
    problems = sorted(p.format() for p in outcome_problems(cases))
    assert problems == [
        "pytest tests/test_a.py:10 skipped (test_skip)",
        "pytest tests/test_a.py:12 xfail (test_xf)",
        "pytest tests/test_a.py:14 xpass (test_xp)",
        "pytest tests/test_a.py:5 failed (test_fail)",
        "pytest tests/test_a.py:8 error (test_err)",
        "pytest tests/test_broken.py collection-error",
    ]


def test_parse_summary_reads_the_final_counts_line() -> None:
    output = (
        "collected 8 items / 1 deselected / 7 selected\n"
        "tests/test_a.py .sxX\n"
        "== 1 failed, 3 passed, 1 skipped, 1 deselected, 1 xfailed, 1 xpassed, "
        "2 warnings, 1 error in 0.16s ==\n"
    )
    assert parse_summary(output) == {
        "failed": 1,
        "passed": 3,
        "skipped": 1,
        "deselected": 1,
        "xfailed": 1,
        "xpassed": 1,
        "warning": 2,
        "error": 1,
    }
    assert parse_summary("no tests ran in 0.01s\n") == {}


# --- minimum counts --------------------------------------------------------------------
def test_parse_min_counts_lists_every_problem() -> None:
    data = {
        "schema_version": 0,
        "extra": 1,
        "suites": {"test_a.py": -1, "test_b.py": 3, "../escape": 2},
        "exemptions": [
            {"suite": "test_a.py", "from": 5, "to": 3, "reason": " ", "adr": "README.md"},
            "not an object",
            {"suite": "test_b.py", "from": 4, "to": 3, "reason": "split", "adr": ADR},
        ],
    }
    counts, issues = parse_min_counts(data)
    assert counts.suites == {"test_b.py": 3}
    assert [(e.suite, e.from_count, e.to_count) for e in counts.exemptions] == [("test_b.py", 4, 3)]
    assert issues == [
        "schema_version must be 1",
        "1 unknown top-level keys",
        "suites/test_a.py must be a non-negative integer",
        "a suite name is not a relative .py path",
        "exemptions/0 needs a valid reason, adr (a docs/adr/<file>.md path)",
        "exemptions/1 must be an object",
    ]


def test_counts_must_reach_minimums_and_may_only_go_up() -> None:
    adr = ADR
    observed = {"test_a.py": 4, "test_b.py": 7, "test_new.py": 2}
    current, issues = parse_min_counts(
        {
            "schema_version": 1,
            "suites": {"test_a.py": 5, "test_b.py": 6, "test_c.py": 1},
            "exemptions": [
                {"suite": "test_b.py", "from": 9, "to": 6, "reason": "split", "adr": adr}
            ],
        }
    )
    assert issues == []
    baseline = MinCounts(suites={"test_a.py": 5, "test_b.py": 9, "test_c.py": 2, "test_d.py": 3})
    problems, hints = evaluate_counts(observed, current, baseline, frozenset({adr}))
    lowered = "pytest gate/min_counts.json minimum-lowered"
    assert sorted(p.format() for p in problems) == [
        f"{lowered} (tests/test_c.py: 2 -> 1 without an exemption)",
        f"{lowered} (tests/test_d.py: 3 -> 0 without an exemption)",
        "pytest tests/test_a.py below-min-count (4 tests, minimum 5)",
        "pytest tests/test_c.py below-min-count (0 tests, minimum 1)",
    ]
    assert hints == [
        "tests/test_b.py has 7 tests; its minimum (6) in gate/min_counts.json can be raised",
        "tests/test_new.py has 2 tests and no minimum in gate/min_counts.json",
    ]
    problems, _ = evaluate_counts(observed, current, baseline, frozenset())
    assert "pytest gate/min_counts.json exemption-adr-missing (" + adr + ")" in [
        p.format() for p in problems
    ]


def test_baseline_minimums_come_from_the_committed_file(
    gate_repo: Path, git: Callable[..., str]
) -> None:
    assert load_baseline_counts(gate_repo, "HEAD") is None  # no commit yet
    write_json(gate_repo / "gate" / "min_counts.json", {"schema_version": 1, "suites": {"a.py": 4}})
    git(gate_repo, "add", "-A")
    git(gate_repo, "commit", "-q", "-m", "baseline")
    write_json(gate_repo / "gate" / "min_counts.json", {"schema_version": 1, "suites": {"a.py": 1}})
    baseline = load_baseline_counts(gate_repo, "HEAD")
    assert baseline is not None
    assert baseline.suites == {"a.py": 4}
    with pytest.raises(ValueError, match="revision"):
        load_baseline_counts(gate_repo, "--output=x")


@pytest.mark.parametrize(
    ("revision", "valid"),
    [
        ("HEAD", True),
        ("origin/main", True),
        ("HEAD~1", True),
        ("v1.2.0", True),
        ("main@{1}", True),
        ("-x", False),
        ("--output=x", False),
        ("a b", False),
        ("", False),
    ],
)
def test_counts_base_revision_names(revision: str, valid: bool) -> None:
    assert is_valid_revision(revision) is valid


# --- real subprocess runs --------------------------------------------------------------
def test_ruff_step_reports_lint_and_format_problems_by_path_and_line(gate_repo: Path) -> None:
    write_text(gate_repo / "src" / "demo" / "unused.py", "import os\n")
    write_text(gate_repo / "src" / "demo" / "ugly.py", "x  =  1\n")
    result = run_ruff_check(gate_repo)
    assert lines(result) == ["ruff src/demo/ugly.py:1 format", "ruff src/demo/unused.py:1 F401"]


MIXED_TESTS = """\
import pytest


@pytest.fixture
def broken():
    raise RuntimeError("fixture-canary")


def test_fails():
    assert "failure-canary" == "other"


def test_setup_error(broken):
    pass


@pytest.mark.xfail(reason="known-canary")
def test_xfail():
    raise ValueError


@pytest.mark.xfail(reason="strict by default")
def test_xpass_strict():
    pass


@pytest.mark.xfail(reason="lenient-canary", strict=False)
def test_xpass_lenient():
    pass


def test_deselected():
    pass
"""


def test_pytest_step_fails_every_outcome_that_is_not_a_pass(
    gate_repo: Path, unicode_tmp_path: Path, no_leak: Callable[..., None]
) -> None:
    pyproject = (gate_repo / "pyproject.toml").read_text(encoding="utf-8")
    write_text(
        gate_repo / "pyproject.toml",
        pyproject.replace(
            'testpaths = ["tests"]',
            'testpaths = ["tests"]\naddopts = "--deselect tests/test_mixed.py::test_deselected"',
        ),
    )
    write_text(gate_repo / "tests" / "test_mixed.py", MIXED_TESTS)
    write_json(
        gate_repo / "gate" / "min_counts.json",
        {"schema_version": 1, "suites": {"test_demo.py": 1, "test_mixed.py": 9}},
    )

    result = run_pytest_check(gate_repo, scratch=unicode_tmp_path / "scratch", counts_base=None)

    def line(name: str) -> int:
        """1-based line pytest reports: the first decorator, else the def line."""
        source = MIXED_TESTS.splitlines()
        index = next(i for i, text in enumerate(source) if text.startswith(f"def {name}("))
        while index > 0 and source[index - 1].startswith("@"):
            index -= 1
        return index + 1

    path = "pytest tests/test_mixed.py"
    assert lines(result) == [
        "pytest - deselected (1 tests)",
        f"{path} below-min-count (5 tests, minimum 9)",
        f"{path} xpass (xfail marked strict=False: test_xpass_lenient)",
        f"{path}:{line('test_fails')} failed (test_fails)",
        f"{path}:{line('test_setup_error')} error (test_setup_error)",
        f"{path}:{line('test_xfail')} xfail (test_xfail)",
        f"{path}:{line('test_xpass_strict')} xpass (test_xpass_strict)",
    ]
    printed = "\n".join(lines(result) + result.notes + result.hints)
    no_leak(printed, gate_repo, "fixture-canary", "failure-canary", "known-canary", "lenient-")


ENVIRONMENT_TESTS = """\
import os
import socket
import subprocess
import sys

PROBE = "from gbf.testing import offline; print(offline.is_installed())"


def test_secrets_and_pytest_options_are_not_inherited():
    for name in ("OPENAI_API_KEY", "HF_TOKEN", "PYTEST_ADDOPTS", "GIT_CONFIG_GLOBAL"):
        assert name not in os.environ
    assert os.environ["GBF_GATE_PROBE"] == "kept"


def test_python_flags_are_forced():
    assert os.environ["PYTHONUTF8"] == "1"
    assert os.environ["PYTHONHASHSEED"] == "0"
    assert sys.flags.utf8_mode == 1


def test_this_process_is_offline():
    assert socket.getaddrinfo.__name__ == "guarded_getaddrinfo"
    assert socket.socket.connect.__name__ == "guarded_connect"


def test_spawned_python_processes_are_offline():
    done = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0
    assert done.stdout.strip() == "True"
"""


def test_full_gate_runs_ruff_and_offline_tests_in_a_clean_environment(
    gate_repo: Path,
    unicode_tmp_path: Path,
    run_gate: RunGate,
    monkeypatch: pytest.MonkeyPatch,
    make_token: Callable[..., str],
    no_leak: Callable[..., None],
) -> None:
    scratch = unicode_tmp_path / "system temp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    secret = make_token("", 24)
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("GBF_GATE_PROBE", "kept")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    write_text(gate_repo / "tests" / "test_environment.py", ENVIRONMENT_TESTS)
    write_json(
        gate_repo / "gate" / "min_counts.json",
        {"schema_version": 1, "suites": {"test_demo.py": 1, "test_environment.py": 4}},
    )

    code, output = run_gate(gate_repo, "--full")

    assert code == 0, output
    assert "[gate] ruff ok" in output
    assert "[gate] pytest ok" in output
    assert "[gate] note pytest: 5 tests in 2 suites" in output
    assert "[gate] note pytest: no gate/min_counts.json at HEAD" in output
    assert "[gate] PASSED: 12 checks" in output
    assert list(scratch.iterdir()) == []  # the gate removed its scratch directory
    no_leak(output, gate_repo, secret)
