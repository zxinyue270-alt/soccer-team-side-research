"""Locating and running ffmpeg and ffprobe.

Rules (CONTRIBUTING.md): ffmpeg and ffprobe are external tools found on ``PATH``; arguments
are always passed as a list (never ``shell=True``); media paths are passed as absolute
``file:`` URLs so that names with spaces, non-ASCII characters, a leading ``-`` or a
``protocol:`` look-alike are never misread; and any error output from a tool is an error.
"""

from __future__ import annotations

import collections
import platform
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from typing import IO

import numpy as np

import gbf
from gbf.ingest.errors import FFmpegError, ToolNotFoundError

#: How much of a tool's error output is kept for error messages (characters, from the end).
STDERR_TAIL_CHARS = 8000

#: Log level for every ffmpeg/ffprobe call: only real errors are printed, so any output
#: on stderr means something went wrong.
LOGLEVEL = "error"


def _which(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise ToolNotFoundError(
            f"{name} is required on PATH (see registries/tools.toml); "
            "it is an external tool and is never bundled"
        )
    return found


def ffmpeg_path() -> str:
    """Absolute path of the ``ffmpeg`` executable on ``PATH``."""
    return _which("ffmpeg")


def ffprobe_path() -> str:
    """Absolute path of the ``ffprobe`` executable on ``PATH``."""
    return _which("ffprobe")


@lru_cache(maxsize=8)
def _version_of(binary: str) -> str:
    proc = subprocess.run(
        [binary, "-version"], capture_output=True, stdin=subprocess.DEVNULL, check=False
    )
    text = decode_output(proc.stdout)
    first_line = text.splitlines()[0] if text else ""
    parts = first_line.split()
    if proc.returncode != 0 or len(parts) < 3 or parts[1] != "version":
        raise FFmpegError(f"{Path(binary).name} -version", proc.returncode, text[-500:])
    return parts[2]


def ffmpeg_version() -> str:
    """Version string of the ffmpeg on PATH, e.g. ``8.1.1-essentials_build-www.gyan.dev``."""
    return _version_of(ffmpeg_path())


def ffprobe_version() -> str:
    """Version string of the ffprobe on PATH."""
    return _version_of(ffprobe_path())


def tool_versions() -> dict[str, str]:
    """Versions of every tool and library whose behaviour shapes ingest outputs."""
    return {
        "ffmpeg": ffmpeg_version(),
        "ffprobe": ffprobe_version(),
        "gbf": gbf.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def media_arg(path: str | Path) -> str:
    """The argument ffmpeg/ffprobe receive for a local file: ``file:`` + absolute path."""
    return "file:" + str(Path(path).resolve())


def decode_output(data: bytes) -> str:
    """Tool output as text (UTF-8; undecodable bytes are replaced, never dropped silently)."""
    return data.decode("utf-8", errors="replace")


def tail(text: str, limit: int = STDERR_TAIL_CHARS) -> str:
    """The last ``limit`` characters of ``text``, marked when something was cut off."""
    if len(text) <= limit:
        return text
    return "[... earlier output cut ...]\n" + text[-limit:]


def run_tool(
    args: list[str], *, what: str, input_bytes: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run a tool to completion and return its result.

    Raises :class:`FFmpegError` (with the tool's stderr) when the exit code is not zero or
    when the tool printed anything at the ``error`` log level.
    """
    if input_bytes is None:
        proc = subprocess.run(args, capture_output=True, stdin=subprocess.DEVNULL, check=False)
    else:
        proc = subprocess.run(args, capture_output=True, input=input_bytes, check=False)
    stderr = decode_output(proc.stderr)
    if proc.returncode != 0 or stderr.strip():
        raise FFmpegError(what, proc.returncode, tail(stderr))
    return proc


class StderrCollector:
    """Drain a process's stderr on a background thread.

    Without this, a tool that writes a lot of error output while the caller is blocked on
    its stdout pipe would fill the stderr pipe buffer and both sides would wait forever.
    Only the last ``limit`` bytes are kept.
    """

    def __init__(self, stream: IO[bytes], limit: int = 64 * 1024):
        self._stream = stream
        self._limit = limit
        self._chunks: collections.deque[bytes] = collections.deque()
        self._size = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="gbf-ffmpeg-stderr", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        reader = getattr(self._stream, "read1", self._stream.read)
        try:
            while True:
                chunk = reader(4096)
                if not chunk:
                    break
                with self._lock:
                    self._chunks.append(chunk)
                    self._size += len(chunk)
                    while self._size > self._limit and len(self._chunks) > 1:
                        self._size -= len(self._chunks.popleft())
        except (OSError, ValueError):
            # The pipe was closed under us while the process was being stopped.
            pass
        finally:
            self._stream.close()

    def text(self, timeout: float = 10.0) -> str:
        """Everything collected so far, after waiting up to ``timeout`` s for end of output."""
        self._thread.join(timeout)
        with self._lock:
            data = b"".join(self._chunks)
        return tail(decode_output(data))
