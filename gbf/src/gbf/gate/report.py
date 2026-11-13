"""The ``--json`` report: a short per-check summary in a receipt envelope of kind ``gate``.

The report validates against ``urn:gbf:schema:receipt``. ``inputs`` lists the gate's
configuration files with their sha256; ``result.checks`` holds one entry per check with
its status, problem count, wall time in seconds and up to MAX_ITEMS problem lines.
Every string passes through :func:`gbf.gate.patterns.redact`.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import gbf
from gbf.gate.model import CheckResult, GateError
from gbf.gate.patterns import redact
from gbf.gate.repo import Workspace, run_git

MAX_ITEMS = 50
CONFIG_INPUTS = (
    "pyproject.toml",
    "uv.lock",
    "gate/min_counts.json",
    "gate/registries.json",
)
LIMITATIONS = (
    "Checks the working tree of the candidate files (tracked plus untracked, not ignored), "
    "not the staged index or history.",
    "Credential and path patterns cover known token formats, literal assignments, "
    "user-home paths and absolute paths through a gbf-data or private folder in text files; "
    "they cannot prove that no secret or private path is present.",
    "Dependencies are checked against a deny list only; license metadata (pip-licenses, "
    "REUSE, NOTICES) is not scanned before P5.",
    "Registry and receipt schemas are validated as they exist in the installed gbf package.",
)


def utc_timestamp(moment: datetime | None = None) -> str:
    """ISO 8601 UTC timestamp with a Z suffix, to the second."""
    value = (moment or datetime.now(UTC)).astimezone(UTC)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def code_state(root: Path) -> dict[str, Any]:
    """gbf version, HEAD commit (or "unknown") and whether the work tree is dirty."""
    commit = "unknown"
    dirty = True
    try:
        head = run_git(root, "rev-parse", "--verify", "--quiet", "HEAD").decode("ascii").strip()
        if len(head) == 40:
            commit = head
        dirty = bool(run_git(root, "status", "--porcelain").strip())
    except (GateError, UnicodeDecodeError):
        pass
    return {"gbf_version": gbf.__version__, "git_commit": commit, "git_dirty": dirty}


def config_inputs(workspace: Workspace | None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if workspace is None:
        return refs
    for rel in CONFIG_INPUTS:
        if rel not in workspace.file_set:
            continue
        try:
            digest = workspace.sha256(rel)
        except OSError:
            continue
        refs.append({"name": rel.rsplit("/", 1)[-1], "relpath": rel, "sha256": digest})
    return refs


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def build_report(
    results: Sequence[CheckResult],
    *,
    mode: str,
    root: Path,
    workspace: Workspace | None,
    wall_seconds: float,
    warnings: Sequence[str],
    started_at: datetime | None = None,
) -> dict[str, Any]:
    """The receipt dictionary for one gate run (see the module docstring)."""
    passed = all(r.passed for r in results)
    environment = {"python": platform.python_version(), "platform": sys.platform}
    for distribution in ("pytest", "ruff", "jsonschema"):
        version = _version(distribution)
        if version is not None:
            environment[distribution] = version
    checks = []
    for result in results:
        lines = [redact(p.format()) for p in result.sorted_problems()]
        checks.append(
            {
                "name": result.name,
                "status": "passed" if result.passed else "failed",
                "problems": len(lines),
                "items": lines[:MAX_ITEMS],
                "scanned": result.scanned,
                "seconds": round(result.seconds, 3),
                "notes": [redact(n) for n in result.notes],
            }
        )
    files = len(workspace.files) if workspace is not None else 0
    return {
        "schema_version": 1,
        "kind": "gate",
        "status": "passed" if passed else "failed",
        "created_at": utc_timestamp(started_at),
        "producer": "gbf.gate.cli",
        "code": code_state(root),
        "inputs": config_inputs(workspace),
        "scope": (
            f"gbf gate ({mode}) over the {files} candidate files of this checkout"
            + (", plus ruff and the offline test suite" if mode == "full" else "")
        ),
        "limitations": list(LIMITATIONS),
        "restrictions": [],
        "reviewer_type": "automated",
        "environment": environment,
        "result": {
            "mode": mode,
            "passed": passed,
            "wall_time_s": round(wall_seconds, 3),
            "candidate_files": files,
            "checks": checks,
            "warnings": [redact(w) for w in warnings],
        },
    }
