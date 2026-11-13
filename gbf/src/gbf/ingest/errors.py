"""Exceptions raised by :mod:`gbf.ingest`.

Every exception that results from validation lists all problems it found, not just the
first one, so a single run tells the user everything that has to be fixed.
"""

from __future__ import annotations

from collections.abc import Sequence


class IngestError(RuntimeError):
    """Base class for every ingest failure."""


class ToolNotFoundError(IngestError):
    """ffmpeg or ffprobe is not on PATH."""


class FFmpegError(IngestError):
    """An ffmpeg or ffprobe process failed or reported errors.

    ``stderr`` holds the tail of the tool's error output, which is also part of the message.
    """

    def __init__(self, what: str, returncode: int | None, stderr: str):
        self.what = what
        self.returncode = returncode
        self.stderr = stderr
        if returncode is None:
            status = "was stopped"
        elif returncode == 0:
            status = "exited with code 0 but reported errors"
        else:
            status = f"exited with code {returncode}"
        detail = stderr.strip() or "(no error output)"
        super().__init__(f"{what}: {status}\n--- ffmpeg stderr ---\n{detail}")


class _ProblemList(IngestError):
    def __init__(self, summary: str, problems: Sequence[str]):
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{summary}: {len(self.problems)} problem(s)\n  - {joined}")


class ProbeError(_ProblemList):
    """A file cannot be probed as a video (no video stream, unusable metadata, ...)."""


class NormalizationError(_ProblemList):
    """The normalised output does not match the expected constant-frame-rate video."""


class FrameRangeError(_ProblemList, ValueError):
    """Requested frame indices are invalid or lie beyond the end of the video."""


class WindowError(_ProblemList):
    """A window request is invalid, or the written window is not bit-identical."""


class RecordError(_ProblemList, ValueError):
    """A ``video.json`` record violates an invariant the schema cannot express."""
