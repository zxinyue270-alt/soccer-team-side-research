"""Exceptions of the plugin protocol. Each one lists every problem found, not just the first."""

from __future__ import annotations

from collections.abc import Sequence


class PluginProtocolError(Exception):
    """Base class. ``where`` names the job or file, ``problems`` lists every problem."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


class JobError(PluginProtocolError, ValueError):
    """A job description is invalid or does not match the files in its job directory."""


class PluginRunError(PluginProtocolError, RuntimeError):
    """A plugin could not be started, timed out, or exited with a non-zero code.

    ``stderr_tail`` holds the end of the plugin's error output (also in the message).
    """

    def __init__(self, where: str, problems: Sequence[str], stderr_tail: str = "") -> None:
        self.stderr_tail = stderr_tail
        listed = list(problems)
        if stderr_tail.strip():
            listed.append("plugin stderr (tail):\n" + stderr_tail.rstrip())
        super().__init__(where, listed)
        self.problems = list(problems)


class OutputsRejectedError(PluginProtocolError, ValueError):
    """accept_outputs refused a plugin's outputs; ``problems`` says why, all at once."""


class CalibFormatError(PluginProtocolError, ValueError):
    """A calib.parquet file does not follow the calib.v1 Arrow schema or its value rules."""
