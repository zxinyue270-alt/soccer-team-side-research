"""The files a gate run looks at, and how they are read.

Candidate files are what ``git ls-files -z --cached --others --exclude-standard`` lists:
tracked files plus untracked files that are not ignored, i.e. everything that is in the
repository or that ``git add -A`` would add. Paths are POSIX strings relative to the
repository root, sorted. Content is read from the working tree as raw bytes; a symbolic
link is read as its target string, which is what git stores for it.

The root must be the top level of a git work tree. Pointing the gate at a plain
directory inside another repository would otherwise list nothing and pass silently.
"""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Any

from gbf.gate.model import CheckResult, GateError
from gbf.util.hashing import sha256_bytes, sha256_file

#: Text checks read at most this many bytes per file (16 MiB); bigger text files fail.
TEXT_SCAN_LIMIT = 16 * 1024 * 1024
#: A NUL byte in the first 8000 bytes marks a file as binary (git uses the same rule).
BINARY_SNIFF_BYTES = 8000
GIT_TIMEOUT_S = 120.0

# Environment variables that would point git at another repository or index.
_GIT_LOCATION_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
)

#: Media, data, model-weight and archive suffixes (lower case). Shared with the media check.
MEDIA_SUFFIXES = frozenset(
    {
        # images
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif",
        ".avif", ".exr", ".ppm", ".pgm", ".pbm", ".pnm", ".tga", ".ico",
        # video
        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv",
        ".m2ts", ".mts", ".3gp", ".ogv", ".y4m", ".h264", ".h265", ".hevc", ".ivf",
        # audio
        ".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a", ".opus", ".wma",
        # array and table data
        ".npy", ".npz", ".parquet", ".feather", ".arrow", ".h5", ".hdf5",
        # the pickle family
        ".pkl", ".pickle", ".pklz", ".joblib", ".dill",
        # model weights
        ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".gguf", ".ggml", ".tflite", ".pb",
        ".keras", ".bin", ".engine",
        # archives
        ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".zst", ".7z", ".rar", ".lz4", ".whl",
    }
)  # fmt: skip

# Suffixes whose content is never text-scanned (binary formats).
_BINARY_SUFFIXES = MEDIA_SUFFIXES | {
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".pdf",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pyd",
    ".class",
    ".jar",
    ".o",
    ".a",
    ".lib",
}

_TOML_LINE = re.compile(r"line (\d+)")


def git_env() -> dict[str, str]:
    """The current environment without variables that redirect git, and without prompts."""
    env = {k: v for k, v in os.environ.items() if k.upper() not in _GIT_LOCATION_VARS}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def run_git(root: Path, *args: str, timeout: float = GIT_TIMEOUT_S) -> bytes:
    """Run ``git <args>`` in ``root`` and return stdout bytes; raise GateError on failure."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            env=git_env(),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as error:
        raise GateError("git-unavailable", "git is not on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise GateError("git-timeout", f"git {args[0]} timed out after {timeout:.0f} s") from error
    except OSError as error:
        detail = f"git could not start ({type(error).__name__})"
        raise GateError("git-unavailable", detail) from error
    if completed.returncode != 0:
        raise GateError("git-failed", f"git {args[0]} exited with {completed.returncode}")
    return completed.stdout


def ensure_work_tree_root(root: Path) -> None:
    """Raise GateError unless ``root`` is the top level of a git work tree."""
    if not root.is_dir():
        raise GateError("root-missing", "the repository root is not a directory")
    try:
        output = run_git(root, "rev-parse", "--show-toplevel")
    except GateError as error:
        if error.rule == "git-failed":
            message = "the root is not inside a git work tree"
            raise GateError("not-a-git-work-tree", message) from error
        raise
    top = Path(output.decode("utf-8", "replace").strip())
    try:
        same = top.is_dir() and top.samefile(root)
    except OSError:
        same = False
    if not same:
        raise GateError("not-a-git-work-tree", "the root is not the top level of its git work tree")


def candidate_files(root: Path) -> tuple[str, ...]:
    """Tracked plus untracked-not-ignored files of the work tree at ``root`` (sorted)."""
    root = Path(root)
    ensure_work_tree_root(root)
    output = run_git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    names: set[str] = set()
    for raw in output.split(b"\0"):
        if not raw:
            continue
        try:
            names.add(raw.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise GateError("undecodable-path", "git listed a path that is not UTF-8") from error
    return tuple(sorted(names))


def suffixes(rel: str) -> list[str]:
    """All lower-case suffixes of the file name, e.g. ``['.pkl', '.gz']``."""
    return [s.lower() for s in PurePosixPath(rel).suffixes]


def suffix(rel: str) -> str:
    return PurePosixPath(rel).suffix.lower()


def basename(rel: str) -> str:
    return rel.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class TextFile:
    """A candidate file whose content is text (no NUL byte, below TEXT_SCAN_LIMIT)."""

    rel: str
    data: bytes

    @cached_property
    def text(self) -> str:
        return self.data.decode("utf-8", "replace")


class Workspace:
    """Candidate files of one repository root. Content is read on demand, not cached.

    ``files`` defaults to :func:`candidate_files`; tests may pass an explicit list.
    """

    def __init__(self, root: Path, files: Sequence[str] | None = None) -> None:
        self.root = Path(root)
        self.files: tuple[str, ...] = (
            tuple(sorted(set(files))) if files is not None else candidate_files(self.root)
        )
        self.file_set = frozenset(self.files)

    def abspath(self, rel: str) -> Path:
        return self.root.joinpath(*rel.split("/"))

    def size(self, rel: str) -> int | None:
        """Size in bytes as git would store it, or None if missing or a directory."""
        path = self.abspath(rel)
        if path.is_symlink():
            return len(os.readlink(path).encode("utf-8", "surrogateescape"))
        if path.is_file():
            return path.stat().st_size
        return None

    def read(self, rel: str) -> bytes | None:
        """Raw bytes as git would store them, or None if missing or a directory."""
        path = self.abspath(rel)
        if path.is_symlink():
            return os.readlink(path).encode("utf-8", "surrogateescape")
        if path.is_file():
            return path.read_bytes()
        return None

    def sha256(self, rel: str) -> str:
        """sha256 hex of the raw bytes git would store (a symbolic link hashes its target).

        Raises OSError if the file cannot be read, FileNotFoundError if it is missing."""
        path = self.abspath(rel)
        if path.is_symlink():
            return sha256_bytes(os.readlink(path).encode("utf-8", "surrogateescape"))
        return sha256_file(path)

    def head(self, rel: str, limit: int) -> bytes | None:
        """The first ``limit`` bytes (or the link target), or None if missing."""
        path = self.abspath(rel)
        if path.is_symlink():
            return os.readlink(path).encode("utf-8", "surrogateescape")[:limit]
        if path.is_file():
            with open(path, "rb") as fh:
                return fh.read(limit)
        return None

    def text_files(self, result: CheckResult) -> Iterator[TextFile]:
        """Yield every text candidate file; add a problem to ``result`` for any that
        cannot be scanned (unreadable, or larger than TEXT_SCAN_LIMIT)."""
        for rel in self.files:
            if suffix(rel) in _BINARY_SUFFIXES:
                continue
            try:
                head = self.head(rel, BINARY_SNIFF_BYTES)
                if head is None or b"\0" in head:
                    continue
                size = self.size(rel)
                if size is None:
                    continue
                if size > TEXT_SCAN_LIMIT:
                    result.add(rel, "too-large-to-scan", detail=f"{size} bytes")
                    continue
                data = self.read(rel)
            except OSError as error:
                result.add(rel, "unreadable", detail=type(error).__name__)
                continue
            if data is None:
                continue
            result.scanned += 1
            yield TextFile(rel, data)


def line_of(text: str, index: int) -> int:
    """1-based line number of character ``index`` in ``text``."""
    return text.count("\n", 0, index) + 1


def toml_error_line(error: tomllib.TOMLDecodeError) -> int | None:
    """Line number from a TOML parse error (Python 3.11/3.12 put it only in the message)."""
    lineno = getattr(error, "lineno", None)
    if isinstance(lineno, int):
        return lineno
    match = _TOML_LINE.search(str(error))
    return int(match.group(1)) if match else None


def to_json_compatible(value: Any) -> Any:
    """TOML data as JSON data: dates and times become ISO 8601 strings."""
    if isinstance(value, dict):
        return {str(k): to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_compatible(v) for v in value]
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    return value
