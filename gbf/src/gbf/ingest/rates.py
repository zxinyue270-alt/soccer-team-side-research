"""Exact frame rates and time bases.

Rates are rational numbers written as ``"numerator/denominator"`` (``"25/1"``,
``"30000/1001"``). Decimal spellings such as ``"29.97"`` are rejected on purpose: 29.97 is
not 30000/1001, and the difference adds up to a frame every 33 seconds.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

_RATIONAL = re.compile(r"^(?P<num>[0-9]+)(?:/(?P<den>[0-9]+))?$")


def parse_rate(text: str | Fraction) -> Fraction:
    """Parse a positive rational such as ``"25/1"``, ``"30000/1001"`` or ``"25"``.

    Raises ``ValueError`` for decimals, zero or negative values, zero denominators and
    anything that is not a string or a ``Fraction``.
    """
    if isinstance(text, Fraction):
        if text <= 0:
            raise ValueError(f"rate must be positive, got {text}")
        return text
    if not isinstance(text, str):
        raise ValueError(f"rate must be a string like '25/1', got {type(text).__name__}")
    match = _RATIONAL.match(text.strip())
    if match is None:
        raise ValueError(
            f"rate must be written as 'numerator/denominator' (e.g. '25/1' or '30000/1001'), "
            f"got {text!r}"
        )
    num = int(match["num"])
    den = int(match["den"]) if match["den"] is not None else 1
    if den == 0 or num == 0:
        raise ValueError(f"rate must have a positive numerator and denominator, got {text!r}")
    return Fraction(num, den)


def parse_optional_rate(text: str | None) -> Fraction | None:
    """Like :func:`parse_rate`, but ffprobe's ``"0/0"`` (unknown) and ``None`` give ``None``."""
    if text is None:
        return None
    try:
        return parse_rate(text)
    except ValueError:
        return None


def format_rate(rate: Fraction) -> str:
    """Canonical text of a rate: ``"num/den"`` in lowest terms (``Fraction(25)`` -> ``"25/1"``)."""
    rate = Fraction(rate)
    return f"{rate.numerator}/{rate.denominator}"


def round_half_away(value: Fraction) -> int:
    """Round to the nearest integer, halves away from zero.

    This is FFmpeg's ``AV_ROUND_NEAR_INF``, which the ``fps`` filter uses by default. Python's
    built-in ``round`` rounds halves to even and would disagree with FFmpeg on exact halves.
    """
    value = Fraction(value)
    if value >= 0:
        return math.floor(value + Fraction(1, 2))
    return -math.floor(-value + Fraction(1, 2))
