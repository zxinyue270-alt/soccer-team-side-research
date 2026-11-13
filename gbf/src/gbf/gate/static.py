"""The fast static checks, run in a fixed order over one shared list of candidate files.

Each check is a function ``check(root, workspace=None) -> CheckResult`` that also works on
its own (tests point it at a temporary repository). A check that raises is reported as a
failing ``internal-error`` with the exception type only, never its message.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from gbf.gate.content import check_abs_paths, check_credentials, check_eol
from gbf.gate.datafiles import check_media, check_restricted_data
from gbf.gate.imports import check_deps, check_forbidden_imports
from gbf.gate.model import NO_PATH, CheckResult, GateError
from gbf.gate.registries import check_receipts, check_registries
from gbf.gate.repo import Workspace

CheckFunction = Callable[[Path, Workspace | None], CheckResult]

STATIC_CHECKS: tuple[tuple[str, CheckFunction], ...] = (
    ("eol", check_eol),
    ("credentials", check_credentials),
    ("abs_paths", check_abs_paths),
    ("media", check_media),
    ("restricted_data", check_restricted_data),
    ("forbidden_imports", check_forbidden_imports),
    ("deps", check_deps),
    ("registries", check_registries),
    ("receipts", check_receipts),
)
STATIC_CHECK_NAMES = tuple(name for name, _ in STATIC_CHECKS)


def open_workspace(root: Path) -> tuple[Workspace | None, CheckResult]:
    """List the candidate files once. The ``files`` result fails if git cannot list them."""
    result = CheckResult("files")
    start = time.perf_counter()
    try:
        workspace = Workspace(Path(root))
    except GateError as error:
        result.add(NO_PATH, error.rule)
        workspace = None
    else:
        result.scanned = len(workspace.files)
    result.seconds = time.perf_counter() - start
    return workspace, result


def run_check(name: str, check: CheckFunction, root: Path, workspace: Workspace) -> CheckResult:
    start = time.perf_counter()
    try:
        result = check(Path(root), workspace)
    except Exception as error:  # a crashing check is a failing check
        result = CheckResult(name)
        result.add(NO_PATH, "internal-error", detail=type(error).__name__)
    result.seconds = time.perf_counter() - start
    return result


def run_static(root: Path, only: Sequence[str] | None = None) -> list[CheckResult]:
    """Run the static checks (all, or the names in ``only``) and return their results.

    The first result is always ``files``; if it fails, no other check can run and the
    list holds only that result.
    """
    unknown = sorted(set(only or ()) - set(STATIC_CHECK_NAMES))
    if unknown:
        raise ValueError(f"unknown static checks: {', '.join(unknown)}")
    workspace, files_result = open_workspace(Path(root))
    results = [files_result]
    if workspace is None:
        return results
    for name, check in STATIC_CHECKS:
        if only is None or name in only:
            results.append(run_check(name, check, Path(root), workspace))
    return results
