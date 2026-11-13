"""Repository-relative paths."""

from __future__ import annotations

import tomllib
from pathlib import Path


class RepoRootNotFoundError(RuntimeError):
    pass


def repo_root(start: str | Path | None = None) -> Path:
    """Return the gbf repository root (the directory holding gbf's pyproject.toml)."""
    here = Path(start if start is not None else __file__).resolve()
    for candidate in [here, *here.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            with open(pyproject, "rb") as fh:
                data = tomllib.load(fh)
            if data.get("project", {}).get("name") == "gbf":
                return candidate
    raise RepoRootNotFoundError(f"no gbf pyproject.toml above {here}")


def to_repo_relative(path: str | Path, root: str | Path | None = None) -> str:
    """POSIX-style path relative to the repo root; receipts never store absolute paths."""
    base = Path(root) if root is not None else repo_root()
    return Path(path).resolve().relative_to(base.resolve()).as_posix()
