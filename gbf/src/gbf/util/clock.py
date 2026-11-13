"""The clock text of a canonical frame: one formatter for every title, summary and table.

A frame's time is ``frame / fps`` seconds, written ``mm:ss.cc``: minutes, seconds and
hundredths of a second (not frames). The hundredths are floored, so a time never runs ahead
of its frame, and the arithmetic is exact (rational), so the same frame gets the same text
everywhere at every rate: frame 2 at 30000/1001 is 0.0667 s, "00:00.06". At 25/1 every
frame is exact (0.04 s). The frame number printed next to a time stays the exact reference.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from gbf.ingest.rates import parse_rate


class ClockError(ValueError):
    """An invalid frame or rate. ``problems`` lists every problem found."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("format_clock: " + "; ".join(self.problems))


def clock_rate(fps: str | Fraction | int) -> Fraction:
    """``fps`` as an exact rate: a string such as ``"30000/1001"``, a Fraction or a positive
    int (a float is refused: 29.97 is not 30000/1001). Raises ClockError."""
    if isinstance(fps, bool):
        raise ClockError([f"fps must be a rate such as '25/1', got {fps!r}"])
    if isinstance(fps, int | np.integer):
        if int(fps) > 0:
            return Fraction(int(fps))
        raise ClockError([f"fps must be positive, got {fps!r}"])
    try:
        return parse_rate(fps)
    except ValueError as exc:
        raise ClockError([f"fps: {exc}"]) from exc


def format_clock(frame: int, fps: str | Fraction | int) -> str:
    """``mm:ss.cc`` of ``frame / fps`` (hundredths floored, exact rationals).

    ``frame`` is a canonical frame index (an int >= 0); ``fps`` as :func:`clock_rate` takes
    it. Raises ClockError listing every problem.
    """
    problems: list[str] = []
    rate: Fraction | None = None
    try:
        rate = clock_rate(fps)
    except ClockError as exc:
        problems.extend(exc.problems)
    if isinstance(frame, bool) or not isinstance(frame, int | np.integer) or frame < 0:
        problems.append(f"frame must be an integer >= 0, got {frame!r}")
    if problems or rate is None:
        raise ClockError(problems)
    hundredths = math.floor(Fraction(int(frame)) * 100 / rate)
    minutes, rest = divmod(hundredths, 6000)
    seconds, cents = divmod(rest, 100)
    return f"{minutes:02d}:{seconds:02d}.{cents:02d}"
