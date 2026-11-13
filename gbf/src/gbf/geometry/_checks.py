"""Input checks shared by the gbf.geometry modules.

Every public geometry function collects all the problems it finds and raises one
``GeometryInputError``, so a caller sees the whole list at once. The helpers here append a
message to ``problems`` and return None for an unusable value; callers raise before they use
any returned value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class GeometryInputError(ValueError):
    """Malformed input to a geometry function. ``problems`` lists every violation found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


def real_array(name: str, value: Any, problems: list[str]) -> np.ndarray | None:
    """A new float64 copy of ``value``; None (with a problem) unless it holds real numbers."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        problems.append(f"{name}: cannot be read as a numeric array ({exc})")
        return None
    if array.dtype.kind not in "iuf":
        problems.append(f"{name}: expected real numbers, got dtype {array.dtype}")
        return None
    return np.array(array, dtype=np.float64)


def fixed_shape(
    name: str, value: Any, shape: tuple[int, ...], problems: list[str], *, finite: bool = True
) -> np.ndarray | None:
    """Float64 copy of ``value`` with exactly ``shape`` (and only finite entries if asked)."""
    array = real_array(name, value, problems)
    if array is None:
        return None
    if array.shape != shape:
        problems.append(f"{name}: expected shape {shape}, got {array.shape}")
        return None
    if finite:
        bad = int(np.count_nonzero(~np.isfinite(array)))
        if bad:
            problems.append(f"{name}: {bad} non-finite value(s)")
            return None
    return array


def trailing_dim(name: str, value: Any, size: int, problems: list[str]) -> np.ndarray | None:
    """Float64 copy of ``value`` with shape (..., size). Non-finite entries are allowed."""
    array = real_array(name, value, problems)
    if array is None:
        return None
    if array.ndim == 0 or array.shape[-1] != size:
        problems.append(f"{name}: expected shape (..., {size}), got {array.shape}")
        return None
    return array


def is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def positive_int(name: str, value: Any, problems: list[str]) -> int | None:
    """``value`` as a Python int if it is an integer (not a bool, not a float) above zero."""
    if is_bool(value) or not isinstance(value, (int, np.integer)):
        problems.append(f"{name}: expected a positive integer, got {value!r}")
        return None
    if value <= 0:
        problems.append(f"{name}: must be > 0, got {int(value)}")
        return None
    return int(value)


def image_size(name: str, value: Any, problems: list[str]) -> tuple[int, int] | None:
    """``value`` as (width, height) in pixels, both positive integers."""
    try:
        items = list(value)
    except TypeError:
        problems.append(f"{name}: expected (width, height), got {value!r}")
        return None
    if len(items) != 2:
        problems.append(f"{name}: expected (width, height), got {len(items)} value(s)")
        return None
    width = positive_int(f"{name} width", items[0], problems)
    height = positive_int(f"{name} height", items[1], problems)
    if width is None or height is None:
        return None
    return width, height


def finite_float(name: str, value: Any, problems: list[str]) -> float | None:
    """``value`` as a finite Python float (bools are rejected)."""
    if is_bool(value) or not isinstance(value, (int, float, np.integer, np.floating)):
        problems.append(f"{name}: expected a real number, got {value!r}")
        return None
    number = float(value)
    if not np.isfinite(number):
        problems.append(f"{name}: must be finite, got {number}")
        return None
    return number
