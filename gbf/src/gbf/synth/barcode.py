"""Frame-number barcode: large flat black and white squares that survive lossy coding.

Layout (``barcode_layout``): two rows of COLUMNS square cells near the top-left corner of
the frame, inside a mid-grey quiet zone half a cell wide.

- Row 0, left to right: START_GUARD, then DATA_BITS bits of the frame number (most
  significant bit first), then END_GUARD. A 1 bit is a white cell, a 0 bit a black cell.
- Row 1 is the complement of row 0, so every column holds one black and one white cell.

The cell side is ``8 * max(1, round(height / 360))`` pixels (16 px at 720p, 24 px at 1080p),
and the code starts one cell from the top-left corner, so cells sit on the 8-pixel grid of
H.264 transform blocks. The decoder averages the luma (BT.601 weights, 0-255) over the
central half of each cell and calls a cell white above LUMA_THRESHOLD. It raises
BarcodeError, listing every problem, when a cell is too close to the threshold, a column is
not complementary, or a guard does not match: a frame that carries no barcode, or one that
was cropped, scaled or mirrored, cannot pass as a frame number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DATA_BITS = 20
"""Bits of frame number: 0 .. 2**20 - 1, about 11.6 hours at 25 fps."""
START_GUARD: tuple[int, ...] = (1, 0, 1, 1)
END_GUARD: tuple[int, ...] = (1, 1, 0, 1)
COLUMNS = len(START_GUARD) + DATA_BITS + len(END_GUARD)
ROWS = 2
MAX_VALUE = 2**DATA_BITS - 1

BLACK = 0
WHITE = 255
QUIET = 128
LUMA_THRESHOLD = 128.0
"""A cell whose mean luma is above this is a 1 (white)."""
MIN_MARGIN = 48.0
"""A cell's mean luma must differ from LUMA_THRESHOLD by at least this, or it is ambiguous."""

_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])


class BarcodeError(ValueError):
    """A barcode cannot be drawn or read. ``problems`` lists every problem found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


@dataclass(frozen=True)
class BarcodeLayout:
    """Where the code sits in a frame: cell side and top-left corner of cell (0, 0), pixels."""

    cell: int
    x0: int
    y0: int

    @property
    def width(self) -> int:
        """Width of the cells in pixels (quiet zone excluded)."""
        return COLUMNS * self.cell

    @property
    def height(self) -> int:
        """Height of the cells in pixels (quiet zone excluded)."""
        return ROWS * self.cell

    def cell_box(self, row: int, column: int) -> tuple[int, int, int, int]:
        """(top, bottom, left, right) pixel bounds of one cell, bottom/right exclusive."""
        top = self.y0 + row * self.cell
        left = self.x0 + column * self.cell
        return top, top + self.cell, left, left + self.cell

    def outer_box(self) -> tuple[int, int, int, int]:
        """(top, bottom, left, right) of the code including its quiet zone, exclusive."""
        margin = self.cell // 2
        return (
            self.y0 - margin,
            self.y0 + self.height + margin,
            self.x0 - margin,
            self.x0 + self.width + margin,
        )


def barcode_layout(width: int, height: int) -> BarcodeLayout:
    """The layout for frames of ``width`` x ``height`` pixels; raises BarcodeError if the
    code does not fit."""
    problems = []
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int | np.integer) or value <= 0:
            problems.append(f"{name} must be a positive integer, got {value!r}")
    if problems:
        raise BarcodeError("barcode_layout", problems)
    cell = 8 * max(1, round(int(height) / 360))
    layout = BarcodeLayout(cell=cell, x0=cell, y0=cell)
    _, bottom, _, right = layout.outer_box()
    if right > width or bottom > height:
        raise BarcodeError(
            "barcode_layout",
            [
                f"a {COLUMNS}x{ROWS}-cell code with {cell}-pixel cells needs a frame of at "
                f"least {right}x{bottom} pixels, got {width}x{height}"
            ],
        )
    return layout


def _checked_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise BarcodeError("encode_barcode", [f"value must be an integer, got {value!r}"])
    if not 0 <= int(value) <= MAX_VALUE:
        raise BarcodeError(
            "encode_barcode", [f"value must be in 0..{MAX_VALUE} ({DATA_BITS} bits), got {value}"]
        )
    return int(value)


def encode_barcode(value: int) -> np.ndarray:
    """Cell bits (ROWS, COLUMNS) uint8 for ``value``: guards and data in row 0, complement in
    row 1. Raises BarcodeError for a value that is not an integer in 0..MAX_VALUE."""
    number = _checked_value(value)
    data = [(number >> shift) & 1 for shift in range(DATA_BITS - 1, -1, -1)]
    row = np.array([*START_GUARD, *data, *END_GUARD], dtype=np.uint8)
    return np.stack([row, 1 - row])


def _checked_frame(frame: object, where: str) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise BarcodeError(where, [f"frame must be a numpy array, got {type(frame).__name__}"])
    problems = []
    if frame.dtype != np.uint8:
        problems.append(f"frame dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        problems.append(f"frame must have shape (height, width, 3), got {frame.shape}")
    if problems:
        raise BarcodeError(where, problems)
    return frame


def draw_barcode(frame: np.ndarray, value: int) -> BarcodeLayout:
    """Draw the code for ``value`` into ``frame`` ((H, W, 3) uint8 RGB) in place.

    Returns the layout used. Raises BarcodeError for a bad frame or value, or when the code
    does not fit.
    """
    image = _checked_frame(frame, "draw_barcode")
    bits = encode_barcode(value)
    layout = barcode_layout(image.shape[1], image.shape[0])
    top, bottom, left, right = layout.outer_box()
    image[top:bottom, left:right] = QUIET
    for row in range(ROWS):
        for column in range(COLUMNS):
            y0, y1, x0, x1 = layout.cell_box(row, column)
            image[y0:y1, x0:x1] = WHITE if bits[row, column] else BLACK
    return layout


def cell_lumas(frame: np.ndarray) -> np.ndarray:
    """Mean luma (0-255, BT.601 weights) of the central half of every cell, (ROWS, COLUMNS)."""
    image = _checked_frame(frame, "cell_lumas")
    layout = barcode_layout(image.shape[1], image.shape[0])
    inset = layout.cell // 4
    lumas = np.empty((ROWS, COLUMNS))
    for row in range(ROWS):
        for column in range(COLUMNS):
            y0, y1, x0, x1 = layout.cell_box(row, column)
            patch = image[y0 + inset : y1 - inset, x0 + inset : x1 - inset].astype(np.float64)
            lumas[row, column] = float((patch @ _LUMA_WEIGHTS).mean())
    return lumas


def _bit_string(bits: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in bits)


def decode_barcode(frame: np.ndarray) -> int:
    """Read the frame number drawn by draw_barcode from an RGB frame (H, W, 3) uint8.

    Raises BarcodeError listing every ambiguous cell, every column whose two cells are not
    complementary, and any guard that does not match.
    """
    lumas = cell_lumas(frame)
    bits = (lumas > LUMA_THRESHOLD).astype(np.uint8)
    problems = []
    for row, column in zip(*np.nonzero(np.abs(lumas - LUMA_THRESHOLD) < MIN_MARGIN), strict=True):
        problems.append(
            f"cell (row {row}, column {column}) has mean luma {lumas[row, column]:.1f}, "
            f"within {MIN_MARGIN:g} of the threshold {LUMA_THRESHOLD:g}"
        )
    same = np.nonzero(bits[0] == bits[1])[0]
    if same.size:
        problems.append(
            f"{same.size} column(s) are not complementary: {', '.join(str(c) for c in same)}"
        )
    start = bits[0, : len(START_GUARD)]
    end = bits[0, COLUMNS - len(END_GUARD) :]
    if tuple(start) != START_GUARD:
        problems.append(
            f"start guard reads {_bit_string(start)}, expected {_bit_string(np.array(START_GUARD))}"
        )
    if tuple(end) != END_GUARD:
        problems.append(
            f"end guard reads {_bit_string(end)}, expected {_bit_string(np.array(END_GUARD))}"
        )
    if problems:
        raise BarcodeError("decode_barcode", problems)
    value = 0
    for bit in bits[0, len(START_GUARD) : len(START_GUARD) + DATA_BITS]:
        value = (value << 1) | int(bit)
    return value
