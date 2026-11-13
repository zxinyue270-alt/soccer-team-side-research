"""Receipt metadata shared by ``fetch`` and ``align``: UTC timestamps and code version."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gbf
from gbf.util import paths

_SHA1 = re.compile(r"^[0-9a-f]{40}$")


def utc_timestamp(now: datetime | None = None) -> str:
    """RFC 3339 UTC timestamp with second precision, e.g. ``2026-10-21T08:00:00Z``.

    ``now`` wins; otherwise ``SOURCE_DATE_EPOCH`` (integer seconds since the Unix
    epoch, the reproducible-builds convention) wins over the wall clock, so a run
    can be made byte-for-byte reproducible.
    """
    if now is None:
        epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if epoch is not None:
            if not epoch.isascii() or not epoch.isdigit():
                raise ValueError(f"SOURCE_DATE_EPOCH must be integer seconds, got {epoch!r}")
            now = datetime.fromtimestamp(int(epoch), tz=UTC)
        else:
            now = datetime.now(tz=UTC)
    if now.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return now.astimezone(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def is_within(child: str | Path, parent: str | Path) -> bool:
    """``child`` equals ``parent`` or lies below it (case-insensitive on Windows).

    Both paths are compared as given (made absolute, not resolved); callers
    resolve symlinks first when that matters.
    """
    try:
        return os.path.commonpath([_norm(child), _norm(parent)]) == _norm(parent)
    except ValueError:  # different drives
        return False


def same_path(a: str | Path, b: str | Path) -> bool:
    return _norm(a) == _norm(b)


def relpath_under(path: str | Path, root: str | Path | None) -> str | None:
    """POSIX path of ``path`` relative to ``root``, or ``None`` if it is not below ``root``.

    Both are resolved first. Receipts store only such relative paths.
    """
    if root is None:
        return None
    resolved, base = Path(path).resolve(), Path(root).resolve()
    if not is_within(resolved, base) or same_path(resolved, base):
        return None
    return Path(os.path.relpath(resolved, base)).as_posix()


def _git(root: Path, *args: str) -> str | None:
    env = dict(os.environ)
    # Never take optional locks: other processes may be writing the index.
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=False,
            env=env,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", "replace")


def code_version(root: str | Path | None = None) -> dict[str, Any]:
    """``{"gbf_version", "git_commit", "git_dirty"}`` for receipts.

    ``git_commit`` is the HEAD commit of the gbf checkout, or ``"unknown"`` when
    gbf does not run from a git checkout or git is unavailable. ``git_dirty`` is
    true when tracked files differ from HEAD or untracked, non-ignored files exist
    (new modules count); it is also true when the state cannot be determined.
    """
    try:
        base = Path(root) if root is not None else paths.repo_root()
    except paths.RepoRootNotFoundError:
        base = None
    commit = "unknown"
    dirty = True
    if base is not None:
        head = _git(base, "rev-parse", "HEAD")
        if head is not None and _SHA1.match(head.strip()):
            commit = head.strip()
            status = _git(base, "status", "--porcelain=v1")
            dirty = status is None or bool(status.strip())
    return {"gbf_version": gbf.__version__, "git_commit": commit, "git_dirty": dirty}


def environment() -> dict[str, str]:
    """Library versions that affect parsing, logged into every receipt."""
    import kloppy

    return {"python": platform.python_version(), "kloppy": str(kloppy.__version__)}
