"""``python -m gbf gate [--full] [--json PATH] [--root DIR] [--counts-base REV]``.

Runs the offline quality gate. Without ``--full`` only the fast static checks run;
``--full`` adds ruff and the test suite. The exit code is 0 only if every check passes,
1 if any fails, 2 for usage errors.

Output: one status line per check (``[gate] <check> ok|FAILED``), then every problem as
``<check> <relative path>[:line] <rule id>`` with an optional ``(detail)``, then notes,
hints, warnings and a summary with the wall time in seconds (a warning above
WALL_TIME_BUDGET_S, gate G1). Nothing printed contains the matched text of a secret or
a local path, and every line also passes through :func:`gbf.gate.patterns.redact`.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from gbf.gate.full import is_valid_revision, run_pytest_check, run_ruff_check
from gbf.gate.model import NO_PATH, CheckResult
from gbf.gate.patterns import redact
from gbf.gate.report import build_report
from gbf.gate.static import STATIC_CHECKS, open_workspace, run_check
from gbf.util.io import write_json
from gbf.util.paths import RepoRootNotFoundError, repo_root

WALL_TIME_BUDGET_S = 10 * 60.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gbf gate",
        description="Offline quality gate: static checks, plus ruff and tests with --full.",
    )
    parser.add_argument(
        "--full", action="store_true", help="also run ruff and the test suite (offline)"
    )
    parser.add_argument(
        "--json", metavar="PATH", type=Path, help="write a JSON report (a receipt of kind gate)"
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        type=Path,
        help="repository root (default: the gbf checkout around the current directory)",
    )
    parser.add_argument(
        "--counts-base",
        metavar="REV",
        default="HEAD",
        help="git revision whose test-count minimums may not be lowered without an "
        "exemption (default: HEAD; CI passes the target branch)",
    )
    return parser


def default_root() -> Path:
    """The gbf checkout containing the current directory, else the one holding this code."""
    try:
        return repo_root(Path.cwd())
    except RepoRootNotFoundError:
        return repo_root()


def _emit(line: str) -> None:
    print(redact(line), flush=True)


def _tolerate_unencodable_output() -> None:
    """Escape characters (in file names) that a redirected console encoding cannot show."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(errors="backslashreplace")
        except (ValueError, OSError):
            pass


def _guarded(name: str, step: Callable[[], CheckResult]) -> CheckResult:
    """Run one --full step; an exception fails it as ``internal-error`` (type only)."""
    start = time.perf_counter()
    try:
        result = step()
    except Exception as error:  # a crashing step is a failing step, never a traceback
        result = CheckResult(name)
        result.add(NO_PATH, "internal-error", detail=type(error).__name__)
    result.seconds = time.perf_counter() - start
    return result


def _report_result(result: CheckResult) -> None:
    status = "ok" if result.passed else f"FAILED ({len(result.problems)})"
    _emit(f"[gate] {result.name} {status} [{result.seconds:.2f} s]")
    for problem in result.sorted_problems():
        _emit(problem.format())


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not is_valid_revision(args.counts_base):
        parser.error("--counts-base must be a plain git revision name")
    _tolerate_unencodable_output()
    started_at = datetime.now(UTC)
    start = time.monotonic()
    try:
        root = Path(args.root) if args.root is not None else default_root()
    except RepoRootNotFoundError:
        _emit("[gate] no gbf checkout found; pass --root")
        return 2
    mode = "full" if args.full else "static"

    workspace, files_result = open_workspace(root)
    results: list[CheckResult] = [files_result]
    if workspace is None:
        _report_result(files_result)
    else:
        _emit(f"[gate] {mode} run over {len(workspace.files)} candidate files")
        for name, check in STATIC_CHECKS:
            result = run_check(name, check, root, workspace)
            results.append(result)
            _report_result(result)
        if args.full:
            _emit("[gate] ruff check and ruff format --check on src and tests")
            ruff = _guarded("ruff", lambda: run_ruff_check(root))
            results.append(ruff)
            _report_result(ruff)
            _emit("[gate] pytest in an offline subprocess with a sanitised environment")
            with tempfile.TemporaryDirectory(
                prefix="gbf-gate-", ignore_cleanup_errors=True
            ) as scratch:
                tests = _guarded(
                    "pytest",
                    lambda: run_pytest_check(
                        root,
                        scratch=Path(scratch),
                        repo_files=workspace.file_set,
                        counts_base=args.counts_base,
                    ),
                )
            results.append(tests)
            _report_result(tests)

    for result in results:
        for note in result.notes:
            _emit(f"[gate] note {result.name}: {note}")
        for hint in result.hints:
            _emit(f"[gate] hint {result.name}: {hint}")
    wall = time.monotonic() - start
    warnings = [w for r in results for w in r.warnings]
    if wall > WALL_TIME_BUDGET_S:
        warnings.append(
            f"wall time {wall / 60:.1f} min exceeds the {WALL_TIME_BUDGET_S / 60:.0f} min "
            "budget (G1)"
        )
    for warning in warnings:
        _emit(f"[gate] warning: {warning}")

    failed = [r for r in results if not r.passed]
    problem_count = sum(len(r.problems) for r in failed)
    if failed:
        _emit(
            f"[gate] FAILED: {problem_count} problems in {len(failed)} of {len(results)} "
            f"checks; wall time {wall:.1f} s"
        )
    else:
        _emit(f"[gate] PASSED: {len(results)} checks; wall time {wall:.1f} s")

    if args.json is not None:
        report = build_report(
            results,
            mode=mode,
            root=root,
            workspace=workspace,
            wall_seconds=wall,
            warnings=warnings,
            started_at=started_at,
        )
        try:
            write_json(args.json, report)
        except OSError as error:
            _emit(f"[gate] JSON report could not be written ({type(error).__name__})")
            return 1
        _emit(f"[gate] JSON report written ({Path(args.json).name})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
