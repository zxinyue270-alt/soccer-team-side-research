"""The ``--full`` steps: ruff and the test suite (gate G1).

ruff: ``ruff check`` and ``ruff format --check`` on src/ and tests/, read as JSON, so the
gate prints ``ruff <path>:<line> <code>`` and never ruff's text (which can quote code).

pytest runs in a subprocess with a sanitised environment (:func:`sanitized_env`) and a
generated ``sitecustomize`` first on PYTHONPATH that installs
:func:`gbf.testing.offline.install` in every Python process of the run, including
processes spawned by tests. xfail is forced strict. The JUnit XML report is parsed and
every failure, error, skip, xfail, xpass and collection problem fails the gate, as does
any deselected test. Test counts per suite (the test file path relative to tests/) must
reach the minimums in ``gate/min_counts.json``; see :func:`evaluate_counts`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gbf
from gbf.gate.model import NO_PATH, CheckResult, GateError, Problem
from gbf.gate.repo import run_git
from gbf.util.io import write_text

MIN_COUNTS_FILE = "gate/min_counts.json"
PYTEST_TIMEOUT_S = 30 * 60.0
RUFF_TIMEOUT_S = 5 * 60.0
OFFLINE_GUARD_EXIT = 86

#: Variables a test run keeps (compared upper case): what Python, git and ffmpeg need.
KEEP_ENV = frozenset(
    {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "USERNAME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_ARCHITEW6432",
        "OS",
        "LD_LIBRARY_PATH",
        "FONTCONFIG_FILE",
        "FONTCONFIG_PATH",
    }
)
KEEP_PREFIXES = ("GBF_",)
FORCED_ENV = {"PYTHONUTF8": "1", "PYTHONHASHSEED": "0"}
# Dropped even when kept by name or prefix (e.g. GBF_HF_TOKEN).
_SECRET_ENV_NAME = re.compile(
    r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|PRIVATE_KEY)$", re.IGNORECASE
)

_SUMMARY_COUNT = re.compile(
    r"(\d+) (failed|passed|skipped|deselected|xfailed|xpassed|errors?|warnings?|rerun)\b"
)
_XPASS_LINE = re.compile(r"^XPASS(?:\(strict\))?\s+(\S+)", re.MULTILINE)
_FUNCTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ADR_PATH = re.compile(r"docs/adr/[A-Za-z0-9._-]+\.md")
_SUITE = re.compile(r"[A-Za-z0-9_./-]+\.py")
_REVISION = re.compile(r"[A-Za-z0-9_./~^@{}-]{1,200}")

_SITECUSTOMIZE_LINES = (
    '"""Written by the gbf gate: every Python process of the gated test run is offline."""',
    "",
    "import os",
    "import sys",
    "",
    "try:",
    "    try:",
    "        from gbf.testing import offline",
    "    except ImportError:",
    "        sys.path.append({gbf_parent})",
    "        from gbf.testing import offline",
    "    offline.install()",
    "except BaseException as error:",
    "    sys.stderr.write('gbf gate: cannot install the offline guard: '",
    "                     + type(error).__name__ + chr(10))",
    "    sys.stderr.flush()",
    "    os._exit({exit_code})",
    "",
)


# --- environment -----------------------------------------------------------------------
def sanitized_env(
    base: Mapping[str, str] | None = None, *, python_path: Sequence[Path] = ()
) -> dict[str, str]:
    """Environment for gated subprocesses.

    Keeps KEEP_ENV and GBF_* variables (minus any whose name ends like a secret), forces
    PYTHONUTF8=1 and PYTHONHASHSEED=0, and sets PYTHONPATH to ``python_path`` followed by
    the entries of the base PYTHONPATH. Everything else (API keys, HF_*, KAGGLE_*, AWS_*,
    GIT_*, PYTEST_ADDOPTS, other PYTHON* variables, proxies, ...) is dropped.
    """
    source = dict(os.environ if base is None else base)
    env: dict[str, str] = {}
    inherited_path = ""
    for name, value in source.items():
        upper = name.upper()
        if upper == "PYTHONPATH":
            inherited_path = value
            continue
        if _SECRET_ENV_NAME.search(upper):
            continue
        if upper in KEEP_ENV or upper.startswith(KEEP_PREFIXES):
            env[name] = value
    env.update(FORCED_ENV)
    entries = [str(p) for p in python_path] + [p for p in inherited_path.split(os.pathsep) if p]
    if entries:
        env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def write_sitecustomize(directory: Path) -> Path:
    """Write the offline-guard ``sitecustomize.py`` into ``directory`` and return it."""
    gbf_parent = str(Path(gbf.__file__).resolve().parent.parent)
    text = "\n".join(_SITECUSTOMIZE_LINES).format(
        gbf_parent=repr(gbf_parent), exit_code=OFFLINE_GUARD_EXIT
    )
    path = directory / "sitecustomize.py"
    write_text(path, text)
    return path


def _relative(root: Path, filename: str) -> str:
    try:
        return Path(filename).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return NO_PATH


# --- ruff ------------------------------------------------------------------------------
def _ruff_json(
    root: Path, args: Sequence[str], env: Mapping[str, str], python: str
) -> tuple[list[dict[str, Any]] | None, int]:
    completed = subprocess.run(
        [python, "-m", "ruff", *args],
        cwd=root,
        env=dict(env),
        capture_output=True,
        timeout=RUFF_TIMEOUT_S,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, completed.returncode
    if not isinstance(payload, list):
        return None, completed.returncode
    return [item for item in payload if isinstance(item, dict)], completed.returncode


def run_ruff_check(
    root: Path, *, base_env: Mapping[str, str] | None = None, python: str | None = None
) -> CheckResult:
    """``ruff check`` and ``ruff format --check`` on src/ and tests/ (as present)."""
    root = Path(root)
    result = CheckResult("ruff")
    targets = [name for name in ("src", "tests") if (root / name).is_dir()]
    if not targets:
        result.add(NO_PATH, "nothing-to-check")
        return result
    env = sanitized_env(base_env)
    interpreter = python or sys.executable
    steps = (
        ("check", ["check", "--no-cache", "--output-format", "json", *targets]),
        ("format", ["format", "--check", "--no-cache", "--output-format", "json", *targets]),
    )
    for step, args in steps:
        try:
            diagnostics, returncode = _ruff_json(root, args, env, interpreter)
        except subprocess.TimeoutExpired:
            result.add(NO_PATH, f"{step}-timeout", detail=f"over {RUFF_TIMEOUT_S:.0f} s")
            continue
        except OSError as error:
            result.add(NO_PATH, "unavailable", detail=type(error).__name__)
            continue
        if diagnostics is None:
            result.add(NO_PATH, f"{step}-failed", detail=f"exit {returncode}, no JSON report")
            continue
        for item in diagnostics:
            location = item.get("location") if isinstance(item.get("location"), dict) else {}
            row = location.get("row")
            code = item.get("code") or "syntax-error"
            rule = "format" if code == "unformatted" else str(code)
            result.add(
                _relative(root, str(item.get("filename", ""))),
                rule,
                line=row if isinstance(row, int) else None,
            )
        if returncode not in (0, 1) and not diagnostics:
            result.add(NO_PATH, f"{step}-failed", detail=f"exit {returncode}")
    return result


# --- JUnit -----------------------------------------------------------------------------
@dataclass(frozen=True)
class JUnitCase:
    """One ``<testcase>`` of a JUnit report.

    ``file`` is POSIX relative to the repository root, ``line`` is 1-based (pytest writes
    0-based lines), ``function`` is the test function name without parameters, and
    ``outcomes`` lists failed / error / skipped / xfail / xpass / collection-error /
    collection-skipped, or is empty for a pass.
    """

    file: str
    line: int | None
    function: str
    outcomes: tuple[str, ...]

    @property
    def is_collection_entry(self) -> bool:
        return any(o.startswith("collection-") for o in self.outcomes)


def _classify(child: ET.Element) -> str | None:
    message = child.get("message", "")
    if child.tag == "failure":
        return "xpass" if message.startswith("[XPASS(strict)]") else "failed"
    if child.tag == "error":
        return "collection-error" if message == "collection failure" else "error"
    if child.tag == "skipped":
        if child.get("type") == "pytest.xfail":
            return "xfail"
        if message == "collection skipped":
            return "collection-skipped"
        if message == "xfail-marked test passes unexpectedly":
            return "xpass"
        return "skipped"
    return None


def parse_junit(xml_bytes: bytes) -> list[JUnitCase]:
    """Test cases of a pytest JUnit XML report (junit_family xunit1, file and line attrs)."""
    root = ET.fromstring(xml_bytes)
    cases: list[JUnitCase] = []
    for element in root.iter("testcase"):
        file = element.get("file", "").replace("\\", "/")
        if not file:
            classname = element.get("classname", "")
            file = classname.replace(".", "/") + ".py" if classname else NO_PATH
        raw_line = element.get("line")
        line = int(raw_line) + 1 if raw_line is not None and raw_line.isdigit() else None
        name_match = _FUNCTION_NAME.match(element.get("name", ""))
        outcomes = tuple(o for o in (_classify(child) for child in element) if o is not None)
        cases.append(JUnitCase(file, line, name_match.group(0) if name_match else "?", outcomes))
    return cases


def parse_summary(output: str) -> dict[str, int]:
    """Counts from pytest's final summary line, e.g. ``{'passed': 3, 'xpassed': 1}``."""
    for line in reversed(output.splitlines()):
        found = _SUMMARY_COUNT.findall(line)
        if found and " in " in line:
            counts: Counter[str] = Counter()
            for number, word in found:
                counts[word.rstrip("s") if word in {"errors", "warnings"} else word] += int(number)
            return dict(counts)
    return {}


def suite_of(file: str, test_dir: str) -> str:
    prefix = test_dir.rstrip("/") + "/"
    return file[len(prefix) :] if file.startswith(prefix) else file


def suite_counts(cases: Sequence[JUnitCase], test_dir: str) -> dict[str, int]:
    """Test cases per suite (test file relative to ``test_dir``), collection entries excluded."""
    counts: Counter[str] = Counter()
    for case in cases:
        if not case.is_collection_entry:
            counts[suite_of(case.file, test_dir)] += 1
    return dict(sorted(counts.items()))


def outcome_problems(cases: Sequence[JUnitCase]) -> list[Problem]:
    """One problem per (file, line, outcome); the detail names the test function(s).

    Collection entries carry no line and no function name."""
    grouped: dict[tuple[str, int | None, str], list[str]] = {}
    for case in cases:
        for outcome in case.outcomes:
            if case.is_collection_entry:
                grouped.setdefault((case.file, None, outcome), [])
            else:
                grouped.setdefault((case.file, case.line, outcome), []).append(case.function)
    problems = []
    for (file, line, outcome), functions in grouped.items():
        names = sorted(set(functions))
        detail = ", ".join(names)
        if len(functions) > len(names):
            detail += f"; {len(functions)} cases"
        problems.append(Problem("pytest", file, outcome, line, detail))
    return problems


def lenient_xpasses(output: str) -> list[tuple[str, str]]:
    """(file, test function) of every XPASS line in pytest's short summary (``-rX``).

    These are tests whose xfail marker sets strict=False explicitly; they pass in the
    JUnit report, so the terminal summary is the only place that names them."""
    found: set[tuple[str, str]] = set()
    for node in _XPASS_LINE.findall(output):
        file, _, rest = node.partition("::")
        name = _FUNCTION_NAME.match(rest.rsplit("::", 1)[-1])
        found.add((file.replace("\\", "/"), name.group(0) if name else "?"))
    return sorted(found)


# --- minimum counts --------------------------------------------------------------------
@dataclass(frozen=True)
class Exemption:
    suite: str
    from_count: int
    to_count: int
    reason: str
    adr: str


@dataclass
class MinCounts:
    """Contents of gate/min_counts.json.

    Format::

        {"schema_version": 1,
         "suites": {"<suite>": <minimum>, ...},
         "exemptions": [{"suite": ..., "from": <old>, "to": <new>,
                         "reason": "...", "adr": "docs/adr/<file>.md"}]}

    A suite is a test file path relative to tests/. Minimums may only go up: lowering a
    minimum (or removing a suite) relative to the baseline revision needs an exemption
    whose from/to match the change exactly, with a reason and an existing ADR file.
    """

    suites: dict[str, int] = field(default_factory=dict)
    exemptions: list[Exemption] = field(default_factory=list)


def parse_min_counts(data: Any) -> tuple[MinCounts, list[str]]:
    """Parse min_counts data; return the usable part and a list of every problem found."""
    issues: list[str] = []
    counts = MinCounts()
    if not isinstance(data, dict):
        return counts, ["top level must be an object"]
    if data.get("schema_version") != 1:
        issues.append("schema_version must be 1")
    unknown = set(data) - {"schema_version", "description", "suites", "exemptions"}
    if unknown:
        issues.append(f"{len(unknown)} unknown top-level keys")
    suites = data.get("suites")
    if not isinstance(suites, dict):
        issues.append("suites must be an object of suite -> minimum")
    else:
        for suite, minimum in suites.items():
            if not _SUITE.fullmatch(suite):
                issues.append("a suite name is not a relative .py path")
            elif isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
                issues.append(f"suites/{suite} must be a non-negative integer")
            else:
                counts.suites[suite] = minimum
    exemptions = data.get("exemptions", [])
    if not isinstance(exemptions, list):
        issues.append("exemptions must be a list")
        exemptions = []
    for index, item in enumerate(exemptions):
        where = f"exemptions/{index}"
        if not isinstance(item, dict):
            issues.append(f"{where} must be an object")
            continue
        suite, old, new = item.get("suite"), item.get("from"), item.get("to")
        reason, adr = item.get("reason"), item.get("adr")
        bad = []
        if not isinstance(suite, str) or not _SUITE.fullmatch(suite):
            bad.append("suite")
        for key, value in (("from", old), ("to", new)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                bad.append(key)
        if not isinstance(reason, str) or not reason.strip():
            bad.append("reason")
        if not isinstance(adr, str) or not _ADR_PATH.fullmatch(adr):
            bad.append("adr (a docs/adr/<file>.md path)")
        if bad:
            issues.append(f"{where} needs a valid {', '.join(bad)}")
            continue
        counts.exemptions.append(Exemption(suite, old, new, reason.strip(), adr))
    return counts, issues


def evaluate_counts(
    observed: Mapping[str, int],
    current: MinCounts,
    baseline: MinCounts | None,
    repo_files: frozenset[str],
    *,
    test_dir: str = "tests",
) -> tuple[list[Problem], list[str]]:
    """Problems and hints for the observed test counts per suite.

    - every suite listed in ``current`` must reach its minimum;
    - every minimum in ``baseline`` that ``current`` lowers (or drops) needs a matching
      exemption whose ADR file is a candidate file of the repository;
    - a hint is given for every suite whose count is above its minimum or unlisted.
    """
    problems: list[Problem] = []
    hints: list[str] = []
    prefix = test_dir.rstrip("/") + "/"
    for suite, minimum in sorted(current.suites.items()):
        count = observed.get(suite, 0)
        if count < minimum:
            problems.append(
                Problem(
                    "pytest",
                    prefix + suite,
                    "below-min-count",
                    detail=f"{count} tests, minimum {minimum}",
                )
            )
        elif count > minimum:
            hints.append(
                f"{prefix}{suite} has {count} tests; its minimum ({minimum}) in "
                f"{MIN_COUNTS_FILE} can be raised"
            )
    for suite in sorted(set(observed) - set(current.suites)):
        hints.append(
            f"{prefix}{suite} has {observed[suite]} tests and no minimum in {MIN_COUNTS_FILE}"
        )
    for exemption in current.exemptions:
        if exemption.adr not in repo_files:
            problems.append(
                Problem("pytest", MIN_COUNTS_FILE, "exemption-adr-missing", detail=exemption.adr)
            )
    if baseline is not None:
        for suite, old in sorted(baseline.suites.items()):
            new = current.suites.get(suite, 0)
            if new >= old:
                continue
            justified = any(
                e.suite == suite and e.from_count == old and e.to_count == new
                for e in current.exemptions
            )
            if not justified:
                problems.append(
                    Problem(
                        "pytest",
                        MIN_COUNTS_FILE,
                        "minimum-lowered",
                        detail=f"{prefix}{suite}: {old} -> {new} without an exemption",
                    )
                )
    return problems, hints


def load_min_counts(root: Path) -> tuple[MinCounts | None, list[str]]:
    """Current gate/min_counts.json from the work tree; (None, problems) if unusable."""
    path = root / MIN_COUNTS_FILE
    if not path.is_file():
        return None, ["missing"]
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, [f"unreadable ({type(error).__name__})"]
    return parse_min_counts(data)


def is_valid_revision(revision: str) -> bool:
    """A git revision name that cannot be mistaken for an option or a path expression."""
    return bool(_REVISION.fullmatch(revision)) and not revision.startswith("-")


def load_baseline_counts(root: Path, revision: str) -> MinCounts | None:
    """gate/min_counts.json at ``revision`` (None if git has no such revision or file)."""
    if not is_valid_revision(revision):
        raise ValueError("invalid git revision for the min_counts baseline")
    try:
        raw = run_git(root, "show", f"{revision}:{MIN_COUNTS_FILE}")
        data = json.loads(raw.decode("utf-8"))
    except (GateError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    counts, _ = parse_min_counts(data)
    return counts


# --- pytest ----------------------------------------------------------------------------
def pytest_command(root: Path, scratch: Path, test_dir: str, python: str) -> tuple[list[str], Path]:
    junit = scratch / "junit.xml"
    command = [
        python,
        "-m",
        "pytest",
        test_dir,
        "--rootdir",
        str(root),
        "-p",
        "no:cacheprovider",
        "--junitxml",
        str(junit),
        "-o",
        "junit_family=xunit1",
        "-o",
        "xfail_strict=true",
        "-rX",
        "--basetemp",
        str(scratch / "basetemp"),
    ]
    if (root / "pyproject.toml").is_file():
        command += ["-c", str(root / "pyproject.toml")]
    return command, junit


def run_pytest_check(
    root: Path,
    *,
    scratch: Path,
    repo_files: frozenset[str] | None = None,
    base_env: Mapping[str, str] | None = None,
    test_dir: str = "tests",
    counts_base: str | None = "HEAD",
    timeout: float = PYTEST_TIMEOUT_S,
    python: str | None = None,
) -> CheckResult:
    """Run the test suite offline in a subprocess and judge it (see the module docstring).

    ``scratch`` is an existing empty directory for the sitecustomize, the JUnit report and
    pytest's basetemp. ``repo_files`` (candidate files) is needed to verify exemption ADR
    files; ``counts_base`` is the git revision whose minimums may not be lowered (None
    skips that comparison; a name that looks like an option raises ValueError). The
    result's ``seconds`` is left for the caller to set.
    """
    root = Path(root)
    result = CheckResult("pytest")
    site_dir = scratch / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    write_sitecustomize(site_dir)
    command, junit = pytest_command(root, scratch, test_dir, python or sys.executable)
    env = sanitized_env(base_env, python_path=[site_dir])
    try:
        completed = subprocess.run(
            command, cwd=root, env=env, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        result.add(NO_PATH, "timeout", detail=f"over {timeout:.0f} s")
        return result
    except OSError as error:
        result.add(NO_PATH, "unavailable", detail=type(error).__name__)
        return result
    output = completed.stdout.decode("utf-8", "replace")
    if completed.returncode == OFFLINE_GUARD_EXIT:
        result.add(NO_PATH, "offline-guard-failed")
        return result
    try:
        cases = parse_junit(junit.read_bytes())
    except (OSError, ET.ParseError) as error:
        result.add(NO_PATH, "junit-unreadable", detail=type(error).__name__)
        cases = []
    for problem in outcome_problems(cases):
        result.add(problem.path, problem.rule, line=problem.line, detail=problem.detail)
    summary = parse_summary(output)
    lenient = lenient_xpasses(output)
    for file, function in lenient:
        result.add(file, "xpass", detail=f"xfail marked strict=False: {function}")
    if summary.get("xpassed", 0) > len(lenient):
        result.add(NO_PATH, "xpass", detail=f"{summary['xpassed']} tests")
    if summary.get("deselected", 0):
        result.add(NO_PATH, "deselected", detail=f"{summary['deselected']} tests")
    counts = suite_counts(cases, test_dir)
    result.scanned = sum(counts.values())
    if not counts:
        result.add(NO_PATH, "no-tests")
    if completed.returncode not in (0, 1) or (completed.returncode == 1 and result.passed):
        result.add(NO_PATH, "exit-code", detail=str(completed.returncode))
    current, issues = load_min_counts(root)
    for issue in issues:
        if issue == "missing":
            result.add(MIN_COUNTS_FILE, "missing")
        else:
            result.add(MIN_COUNTS_FILE, "invalid", detail=issue)
    if current is not None:
        baseline = load_baseline_counts(root, counts_base) if counts_base else None
        if counts_base and baseline is None:
            result.notes.append(
                f"no {MIN_COUNTS_FILE} at {counts_base}: the only-up rule was not compared"
            )
        files = repo_files if repo_files is not None else frozenset()
        problems, hints = evaluate_counts(counts, current, baseline, files, test_dir=test_dir)
        for problem in problems:
            result.add(problem.path, problem.rule, line=problem.line, detail=problem.detail)
        result.hints.extend(hints)
    result.notes.append(f"{result.scanned} tests in {len(counts)} suites")
    return result
