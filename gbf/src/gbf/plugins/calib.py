"""``calib.parquet`` (format ``calib.v1``): per-frame camera calibration from a plugin.

One row per canonical frame (plan 3.3 step 2, gate G6):

=============  ===============================  ===========================================
column         Arrow type                       meaning
=============  ===============================  ===========================================
frame_index    int64, not null                  canonical frame index (ADR 0003)
width          int32, not null                  pixel width of the frames K refers to
height         int32, not null                  pixel height of the frames K refers to
K              fixed_size_list<float64>[9]      intrinsics, row-major, pixels of width x
                                                height; null exactly when calib_status is
                                                "none"
R              fixed_size_list<float64>[9]      world-to-camera rotation, row-major; null
                                                exactly when calib_status is "none"
C              fixed_size_list<float64>[3]      camera centre, SoccerNet world frame, metres;
                                                null exactly when calib_status is "none"
calib_status   string, not null                 "fresh" (computed on this frame), "held"
                                                (carried over from an earlier frame) or "none"
calib_age      int32, not null                  frames since the calibration in use was
                                                computed: 0 when fresh, >= 1 when held, 0 for
                                                none
rep_err        float64, nullable                reprojection error in pixels of width x
                                                height; null when not computed (always null
                                                for none)
=============  ===============================  ===========================================

Conventions are gbf.geometry's: P = K R [I | -C], z down, pixel centres at integers. A
plugin may report K for a frame size other than the video's (sn-gamestate always reports
1920x1080); CalibFrame.camera() rescales it to the actual frames with
gbf.geometry.scale_intrinsics, so callers never use a K at the wrong resolution (gate G7).
Rows are sorted by frame_index without duplicates. read_calib checks all of this and lists
every problem at once.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gbf.geometry import Camera, GeometryInputError
from gbf.plugins.errors import CalibFormatError

CALIB_FORMAT = "calib.v1"
CALIB_STATUSES: tuple[str, ...] = ("fresh", "held", "none")
CALIB_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
        pa.field("K", pa.list_(pa.float64(), 9), nullable=True),
        pa.field("R", pa.list_(pa.float64(), 9), nullable=True),
        pa.field("C", pa.list_(pa.float64(), 3), nullable=True),
        pa.field("calib_status", pa.string(), nullable=False),
        pa.field("calib_age", pa.int32(), nullable=False),
        pa.field("rep_err", pa.float64(), nullable=True),
    ]
)
"""The calib.v1 Arrow schema. Field-level nullability flags are not enforced on read (some
writers mark every field nullable); actual nulls are."""
_NOT_NULL = ("frame_index", "width", "height", "calib_status", "calib_age")
_MATRIX_SIZES = {"K": 9, "R": 9, "C": 3}


@dataclass(frozen=True)
class CalibFrame:
    """One row of calib.parquet. K and R are (3, 3), C is (3,), all read-only float64 arrays;
    the three are None when calib_status is "none"."""

    frame_index: int
    width: int
    height: int
    K: np.ndarray | None
    R: np.ndarray | None
    C: np.ndarray | None
    calib_status: str
    calib_age: int
    rep_err: float | None

    def camera(self, width: int | None = None, height: int | None = None) -> Camera | None:
        """The camera for frames of ``width`` x ``height`` pixels (default: the reported
        size), with K rescaled by gbf.geometry.scale_intrinsics; None for status "none"."""
        if self.K is None or self.R is None or self.C is None:
            return None
        reported = Camera(K=self.K, R=self.R, C=self.C, width=self.width, height=self.height)
        if width is None and height is None:
            return reported
        target_width = self.width if width is None else width
        target_height = self.height if height is None else height
        return reported.resized(target_width, target_height)

    def to_row(self) -> dict[str, Any]:
        """The row as plain Python values in CALIB_SCHEMA order."""

        def flat(array: np.ndarray | None) -> list[float] | None:
            return None if array is None else [float(v) for v in np.asarray(array).ravel()]

        return {
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "K": flat(self.K),
            "R": flat(self.R),
            "C": flat(self.C),
            "calib_status": self.calib_status,
            "calib_age": self.calib_age,
            "rep_err": self.rep_err,
        }


def schema_problems(schema: pa.Schema) -> list[str]:
    """Differences between ``schema`` and CALIB_SCHEMA (names, order and types)."""
    problems = []
    names = schema.names
    expected = CALIB_SCHEMA.names
    missing = [name for name in expected if name not in names]
    extra = [name for name in names if name not in expected]
    if missing:
        problems.append(f"missing column(s): {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected column(s): {', '.join(extra)}")
    if not missing and not extra and names != expected:
        problems.append(f"columns must be in the order {', '.join(expected)}")
    for field in CALIB_SCHEMA:
        if field.name not in names:
            continue
        actual = schema.field(field.name).type
        if not actual.equals(field.type):
            problems.append(f"column {field.name}: type {actual}, expected {field.type}")
    return problems


def _row_problems(row: dict[str, Any], where: str) -> list[str]:
    problems = []
    status = row["calib_status"]
    if status not in CALIB_STATUSES:
        problems.append(
            f"{where}: calib_status {status!r} is not one of {', '.join(CALIB_STATUSES)}"
        )
    if row["width"] is not None and row["width"] < 1:
        problems.append(f"{where}: width must be >= 1, got {row['width']}")
    if row["height"] is not None and row["height"] < 1:
        problems.append(f"{where}: height must be >= 1, got {row['height']}")
    age = row["calib_age"]
    if age is not None:
        if status == "fresh" and age != 0:
            problems.append(f"{where}: calib_age must be 0 for a fresh calibration, got {age}")
        elif status == "held" and age < 1:
            problems.append(f"{where}: calib_age must be >= 1 for a held calibration, got {age}")
        elif status == "none" and age != 0:
            problems.append(f"{where}: calib_age must be 0 when there is no calibration, got {age}")
    matrices_null = [row[name] is None for name in _MATRIX_SIZES]
    if status == "none":
        if not all(matrices_null):
            problems.append(f"{where}: K, R and C must be null when calib_status is 'none'")
        if row["rep_err"] is not None:
            problems.append(f"{where}: rep_err must be null when calib_status is 'none'")
    elif status in CALIB_STATUSES:
        nulls = [name for name, null in zip(_MATRIX_SIZES, matrices_null, strict=True) if null]
        if nulls:
            problems.append(f"{where}: {', '.join(nulls)} must not be null for status {status!r}")
        else:
            values = {name: row[name] for name in _MATRIX_SIZES}
            if any(v is None for name in values for v in values[name]):
                problems.append(f"{where}: K, R and C must not contain null values")
            elif row["width"] is not None and row["height"] is not None:
                try:
                    Camera(
                        K=np.reshape(values["K"], (3, 3)),
                        R=np.reshape(values["R"], (3, 3)),
                        C=np.asarray(values["C"]),
                        width=int(row["width"]),
                        height=int(row["height"]),
                    )
                except GeometryInputError as exc:
                    problems.extend(f"{where}: {problem}" for problem in exc.problems)
    rep_err = row["rep_err"]
    if rep_err is not None and not (np.isfinite(rep_err) and rep_err >= 0.0):
        problems.append(f"{where}: rep_err must be a finite number >= 0 or null, got {rep_err}")
    return problems


def table_problems(table: pa.Table) -> list[str]:
    """Every problem with ``table`` as calib.v1: schema, nulls, order and per-row values."""
    problems = schema_problems(table.schema)
    if problems:
        return problems
    for name in _NOT_NULL:
        nulls = table.column(name).null_count
        if nulls:
            problems.append(f"column {name}: {nulls} null value(s)")
    rows = table.to_pylist()
    indices = [row["frame_index"] for row in rows]
    if any(index is not None and index < 0 for index in indices):
        problems.append("frame_index must be >= 0")
    known = [index for index in indices if index is not None]
    if known != sorted(set(known)):
        problems.append("frame_index must be strictly increasing (sorted, no duplicates)")
    for row in rows:
        problems.extend(_row_problems(row, f"frame {row['frame_index']}"))
    return problems


def _frame(row: dict[str, Any]) -> CalibFrame:
    def matrix(values: list[float] | None, shape: tuple[int, ...]) -> np.ndarray | None:
        if values is None:
            return None
        array = np.array(values, dtype=np.float64).reshape(shape)
        array.setflags(write=False)
        return array

    return CalibFrame(
        frame_index=int(row["frame_index"]),
        width=int(row["width"]),
        height=int(row["height"]),
        K=matrix(row["K"], (3, 3)),
        R=matrix(row["R"], (3, 3)),
        C=matrix(row["C"], (3,)),
        calib_status=str(row["calib_status"]),
        calib_age=int(row["calib_age"]),
        rep_err=None if row["rep_err"] is None else float(row["rep_err"]),
    )


def read_calib(path: str | Path) -> list[CalibFrame]:
    """Read and check a calib.parquet file; raises CalibFormatError listing every problem."""
    source = Path(path)
    try:
        # ParquetFile, not pq.read_table: the latter imports pyarrow.dataset (over 1 s).
        with pq.ParquetFile(source) as handle:
            table = handle.read()
    except (OSError, pa.ArrowException) as exc:
        raise CalibFormatError(source.name, [f"cannot read as parquet: {exc}"]) from exc
    problems = table_problems(table)
    if problems:
        raise CalibFormatError(source.name, problems)
    return [_frame(row) for row in table.to_pylist()]


def calib_table(frames: Iterable[CalibFrame]) -> pa.Table:
    """A calib.v1 table of ``frames``; raises CalibFormatError if they break its rules."""
    rows = [frame.to_row() for frame in frames]
    table = pa.Table.from_pylist(rows, schema=CALIB_SCHEMA)
    problems = table_problems(table)
    if problems:
        raise CalibFormatError("calib table", problems)
    return table


def write_calib(path: str | Path, frames: Sequence[CalibFrame]) -> None:
    """Write a checked calib.parquet (refuses to overwrite an existing file)."""
    destination = Path(path)
    table = calib_table(frames)
    if destination.exists():
        raise FileExistsError(f"{destination} exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)


def frame_from_camera(
    frame_index: int,
    camera: Camera,
    *,
    calib_status: str = "fresh",
    calib_age: int = 0,
    rep_err: float | None = None,
) -> CalibFrame:
    """A CalibFrame holding ``camera`` (its K, R, C and frame size)."""
    return CalibFrame(
        frame_index=frame_index,
        width=camera.width,
        height=camera.height,
        K=camera.K,
        R=camera.R,
        C=camera.C,
        calib_status=calib_status,
        calib_age=calib_age,
        rep_err=rep_err,
    )
