"""Import of the plugin-sngs export: ``tracks.parquet``, ``calib.parquet`` and ``import.json``.

Input (``sngs_dir``, written by the GPL plugin; the core only reads it, see
docs/design/m1-highlights.md, "plugin-sngs export rows"):

- ``detections.jsonl``: one detection per line with ``frame`` (canonical CFR index),
  ``state_frame``, ``track_id`` (-1 unknown), ``box`` [x1, y1, x2, y2] px, ``kf_box``,
  ``conf``, ``role``, ``team`` ("left" | "right" | null, the side the team DEFENDS: x < 0 is
  left), ``team_cluster``, ``jersey`` and ``jersey_conf``.
- ``calib.jsonl``: one row per frame with ``frame``, ``calib_status`` (fresh | held | none),
  ``calib_age`` and, when calibrated, ``K`` (9, row-major), ``R`` (9, world to camera),
  ``C`` (3, metres), ``radial_distortion``, ``width``, ``height``, ``rep_err_px`` and
  ``rep_err_n_keypoints``.
- ``plugin-receipt.json``: ``outputs`` (name, sha256, optional bytes and relpath),
  ``restrictions`` and optional facts such as ``video_frames``.

Every output listed in the receipt is re-hashed (raw bytes) before anything is accepted, and
every row is validated; all problems are collected and raised together as
:class:`SngsImportError`. Nothing is written unless the whole export is valid.

Outputs (``out_dir``):

- ``calib.parquet``: one row per canonical frame ``0 .. n_frames - 1`` with the calib.v1
  columns of ``gbf.plugins.calib`` (frame_index, width, height, K, R, C, calib_status,
  calib_age, rep_err) plus ``rep_err_n_keypoints`` (int32) and ``pitch_usable`` (bool).
  Frames the export does not list get calib_status "none" (counted in import.json).
- ``tracks.parquet``: one row per detection, the contract columns of the design document
  (see :data:`TRACKS_SCHEMA`).
- ``import.json``: input and output hashes, the receipt's restrictions, settings and counts.

Both parquet files carry the restrictions in their schema metadata (key
``gbf.restrictions``, a JSON list), because every artifact carries the union of its inputs'
restrictions (CONTRIBUTING.md).

Conventions:

- Pitch-usable frame (gate G6): ``calib_status == "fresh"`` and ``rep_err_px`` not null and
  ``rep_err_px <= rep_err_max_px`` (default 8 px, an empirical starting value), and the
  calibration is a valid pinhole camera (positive focal lengths, upper-triangular K) without
  lens distortion. Only these frames get pitch positions.
- Foot point = bottom centre of the detection ``box`` (not ``kf_box``): ``foot_u = (x1 +
  x2) / 2``, ``foot_v = y2``, in the pixel grid of the export. Box and K come from the same
  plugin and are used together unchanged; NBJW puts the principal point at (W/2, H/2), which
  suggests the plugin counts pixel edges from 0 (centres at +0.5) while gbf.geometry puts
  pixel centres at integers. Using both unchanged keeps the rays exact either way; only
  ``foot_u``/``foot_v`` may sit half a pixel off gbf's convention, far below detection noise.
- ``pitch_x``/``pitch_y``: where the foot pixel's ray meets the grass (gbf.geometry
  ``pixel_rays`` + ``intersect_ground``), SoccerNet world frame in metres (origin at the
  centre spot, far touchline at y = -34, z down). NaN unless the frame is pitch-usable and
  the ray meets the grass in front of the camera.
- ``in_pitch``: the ground point lies inside the pitch (105 x 68 m) extended by
  :data:`IN_PITCH_MARGIN_M` on every side; False when there is no ground point.

Data rules that are normalised instead of rejected (each is counted in import.json):
``team`` given as the string "nan" (a pandas NaN written as text) is stored as null; rows
with role "ball" are dropped (the ball comes from the external ball detector, not from this table).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import gbf
from gbf.contracts import def_problems
from gbf.geometry import (
    ROTATION_TOL,
    Camera,
    GeometryInputError,
    intersect_ground,
    pixel_rays,
)
from gbf.util.hashing import sha256_bytes, sha256_file
from gbf.util.io import write_json

DETECTIONS_NAME = "detections.jsonl"
CALIB_JSONL_NAME = "calib.jsonl"
RECEIPT_NAME = "plugin-receipt.json"
TRACKS_NAME = "tracks.parquet"
CALIB_NAME = "calib.parquet"
IMPORT_JSON_NAME = "import.json"

DEFAULT_REP_ERR_MAX_PX = 8.0
"""Largest ``rep_err_px`` of a pitch-usable frame, pixels: the empirical starting value of
docs/design/m1-highlights.md, not a validated one."""

IN_PITCH_MARGIN_M = 3.0
"""``in_pitch`` accepts ground points up to this many metres outside the pitch lines."""

PITCH_HALF_LENGTH_M = 52.5
PITCH_HALF_WIDTH_M = 34.0

ROLES: tuple[str, ...] = ("player", "goalkeeper", "referee", "other", "unknown")
"""Roles kept in tracks.parquet."""

DROPPED_ROLES: tuple[str, ...] = ("ball",)
"""Roles the sn-gamestate detector emits that are not people; such rows are dropped."""

TEAMS: tuple[str, ...] = ("left", "right")
TEAM_NULL_STRINGS: tuple[str, ...] = ("nan",)
"""Text the export writes instead of null for "no team" (pandas NaN turned into a string)."""

CALIB_STATUSES: tuple[str, ...] = ("fresh", "held", "none")

MAX_LISTED_PROBLEMS = 200
"""SngsImportError messages list at most this many problems; ``problems`` keeps them all."""

_RESTRICTIONS_KEY = b"gbf.restrictions"
_PRODUCER_KEY = b"gbf.producer"
_PRODUCER = "gbf.sngs_import"

TRACKS_SCHEMA = pa.schema(
    [
        pa.field("frame", pa.int64(), nullable=False),
        pa.field("track_id", pa.int64(), nullable=False),
        pa.field("x1", pa.float64(), nullable=False),
        pa.field("y1", pa.float64(), nullable=False),
        pa.field("x2", pa.float64(), nullable=False),
        pa.field("y2", pa.float64(), nullable=False),
        pa.field("conf", pa.float32(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("team", pa.string(), nullable=True),
        pa.field("jersey", pa.string(), nullable=True),
        pa.field("jersey_conf", pa.float64(), nullable=True),
        pa.field("foot_u", pa.float64(), nullable=False),
        pa.field("foot_v", pa.float64(), nullable=False),
        pa.field("calib_status", pa.string(), nullable=False),
        pa.field("pitch_usable", pa.bool_(), nullable=False),
        pa.field("pitch_x", pa.float64(), nullable=False),
        pa.field("pitch_y", pa.float64(), nullable=False),
        pa.field("in_pitch", pa.bool_(), nullable=False),
    ]
)
"""tracks.parquet: one row per detection, sorted by (frame, track_id). Box and foot
coordinates are pixels of the export's grid; pitch_x/pitch_y are metres (NaN, not null,
unless pitch-usable); team/jersey/jersey_conf are null when unknown."""

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
        pa.field("rep_err_n_keypoints", pa.int32(), nullable=False),
        pa.field("pitch_usable", pa.bool_(), nullable=False),
    ]
)
"""calib.parquet: the calib.v1 columns (gbf.plugins.calib) plus rep_err_n_keypoints and
pitch_usable, one row per canonical frame sorted by frame_index. K and R are row-major,
P = K R [I | -C]; K, R and C are null exactly when calib_status is "none". Values are the
plugin's own, including calibrations that are not valid cameras (they are never usable)."""


class SngsImportError(ValueError):
    """The plugin-sngs export (or a file derived from it) is invalid.

    ``problems`` lists every problem found; the message shows the first
    :data:`MAX_LISTED_PROBLEMS` of them.
    """

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        shown = self.problems[:MAX_LISTED_PROBLEMS]
        hidden = len(self.problems) - len(shown)
        joined = "\n  - ".join(shown)
        tail = f"\n  ... and {hidden} more problem(s)" if hidden else ""
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}{tail}")


@dataclass(frozen=True)
class ImportSummary:
    """What :func:`import_sngs` wrote.

    Attributes:
        out_dir: the output directory.
        tracks_path, calib_path, import_json_path: the three files written.
        n_frames: canonical frame count (the receipt's ``video_frames``, else the largest
            frame index in the export plus one).
        n_detection_rows: lines in detections.jsonl.
        n_track_rows: rows written to tracks.parquet (detections minus dropped roles).
        n_calib_rows: rows written to calib.parquet (always ``n_frames``).
        n_pitch_usable_frames: frames whose calibration is pitch-usable.
        restrictions: restriction tags copied from the receipt, sorted.
        counts: the ``counts`` object of import.json.
    """

    out_dir: Path
    tracks_path: Path
    calib_path: Path
    import_json_path: Path
    n_frames: int
    n_detection_rows: int
    n_track_rows: int
    n_calib_rows: int
    n_pitch_usable_frames: int
    restrictions: tuple[str, ...]
    counts: Mapping[str, Any]


# --- small value checks -----------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return _is_real(value) and math.isfinite(value)


def _real_list(
    where: str, name: str, value: Any, size: int, problems: list[str]
) -> list[float] | None:
    """``value`` as ``size`` finite floats, or None after recording the problem."""
    if not isinstance(value, list) or len(value) != size:
        got = f"{len(value)} item(s)" if isinstance(value, list) else repr(value)
        problems.append(f"{where}: {name} must be a list of {size} numbers, got {got}")
        return None
    if not all(_finite(item) for item in value):
        problems.append(f"{where}: {name} must hold {size} finite numbers, got {value!r}")
        return None
    return [float(item) for item in value]


def _frame_index(
    where: str, name: str, value: Any, n_frames: int | None, problems: list[str]
) -> int | None:
    if not _is_int(value):
        problems.append(f"{where}: {name} must be an integer, got {value!r}")
        return None
    if value < 0:
        problems.append(f"{where}: {name} must be >= 0, got {value}")
        return None
    if n_frames is not None and value >= n_frames:
        problems.append(
            f"{where}: {name} {value} is outside the video (frames 0..{n_frames - 1}"
            " according to the receipt's video_frames)"
        )
        return None
    return int(value)


# --- receipt and hashes -----------------------------------------------------------------


@dataclass(frozen=True)
class _Listed:
    """One output listed in the receipt."""

    where: str
    path: Path
    sha256: str
    size: Any  # the receipt's "bytes" value, None when absent


@dataclass(frozen=True)
class _Receipt:
    raw: dict[str, Any]
    raw_bytes: bytes
    restrictions: tuple[str, ...]
    video_frames: int | None
    listed: tuple[_Listed, ...]


def _output_path(
    sngs_dir: Path, entry: Mapping[str, Any], where: str, problems: list[str]
) -> Path | None:
    """Path of one receipt output: ``relpath`` (relative POSIX) if given, else ``name``."""
    relpath = entry.get("relpath", entry.get("name"))
    if not isinstance(relpath, str) or not relpath:
        problems.append(f"{where}: needs a non-empty 'name' (or 'relpath') string")
        return None
    posix = PurePosixPath(relpath)
    if "\\" in relpath or posix.is_absolute() or ".." in posix.parts or ":" in relpath:
        problems.append(f"{where}: {relpath!r} must be a relative POSIX path inside sngs_dir")
        return None
    return sngs_dir.joinpath(*posix.parts)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _read_receipt(sngs_dir: Path, problems: list[str]) -> _Receipt | None:
    """Parse plugin-receipt.json: restrictions, video_frames and the listed outputs."""
    path = sngs_dir / RECEIPT_NAME
    if not path.is_file():
        problems.append(f"{RECEIPT_NAME}: missing in sngs_dir")
        return None
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        problems.append(f"{RECEIPT_NAME}: not valid UTF-8 JSON ({exc})")
        return None
    if not isinstance(raw, dict):
        problems.append(f"{RECEIPT_NAME}: must be a JSON object")
        return None

    restrictions: tuple[str, ...] = ()
    if "restrictions" not in raw:
        problems.append(
            f"{RECEIPT_NAME}: 'restrictions' is missing; every plugin output must declare its"
            " restrictions (an empty list when there are none)"
        )
    else:
        found = def_problems(raw["restrictions"], "restrictions", "restrictions")
        problems.extend(f"{RECEIPT_NAME}: restrictions {problem}" for problem in found)
        if not found:
            restrictions = tuple(sorted(raw["restrictions"]))

    video_frames = raw.get("video_frames")
    if video_frames is not None and not (_is_int(video_frames) and video_frames > 0):
        problems.append(f"{RECEIPT_NAME}: video_frames must be a positive integer")
        video_frames = None

    listed: list[_Listed] = []
    outputs = raw.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        problems.append(f"{RECEIPT_NAME}: 'outputs' must be a non-empty list")
        outputs = []
    names: set[str] = set()
    for position, entry in enumerate(outputs):
        where = f"{RECEIPT_NAME}: outputs[{position}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: must be an object")
            continue
        target = _output_path(sngs_dir, entry, where, problems)
        expected = entry.get("sha256")
        if not _is_sha256(expected):
            problems.append(f"{where}: sha256 must be 64 lower-case hex digits")
            continue
        if target is None:
            continue
        if target.name in names:
            problems.append(f"{where}: {target.name} is listed more than once")
            continue
        names.add(target.name)
        listed.append(_Listed(where, target, expected, entry.get("bytes")))
    for name in (DETECTIONS_NAME, CALIB_JSONL_NAME):
        if outputs and name not in names:
            problems.append(
                f"{RECEIPT_NAME}: {name} is not listed in 'outputs', so its sha256 cannot be"
                " verified"
            )
    return _Receipt(raw, raw_bytes, restrictions, video_frames, tuple(listed))


def _listed_path(receipt: _Receipt | None, sngs_dir: Path, name: str) -> Path:
    """Where the receipt says the file called ``name`` is (default: ``sngs_dir / name``)."""
    if receipt is not None:
        for item in receipt.listed:
            if item.path.name == name:
                return item.path
    return sngs_dir / name


def _verify_outputs(receipt: _Receipt, parsed: Mapping[Path, bytes], problems: list[str]) -> None:
    """Re-hash every listed output. Files parsed here are hashed from the very bytes that
    were parsed, so a file cannot change between verification and use."""
    for item in receipt.listed:
        data = parsed.get(item.path)
        if data is None:
            if not item.path.is_file():
                problems.append(f"{item.where}: {item.path.name} is listed but missing")
                continue
            actual, actual_size = sha256_file(item.path), item.path.stat().st_size
        else:
            actual, actual_size = sha256_bytes(data), len(data)
        if actual != item.sha256:
            problems.append(
                f"{item.where}: sha256 mismatch for {item.path.name}: receipt {item.sha256},"
                f" file {actual}"
            )
        if item.size is not None and (not _is_int(item.size) or item.size != actual_size):
            problems.append(
                f"{item.where}: size mismatch for {item.path.name}: receipt {item.size!r} bytes,"
                f" file {actual_size} bytes"
            )


def _read_jsonl(path: Path, problems: list[str]) -> tuple[bytes | None, list[tuple[int, Any]]]:
    """Raw bytes (None when the file is missing) and ``(line number, parsed value)`` pairs
    of a JSON-lines file."""
    if not path.is_file():
        problems.append(f"{path.name}: missing in sngs_dir")
        return None, []
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append(f"{path.name}: not valid UTF-8 ({exc})")
        return data, []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    rows: list[tuple[int, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            problems.append(f"{path.name} line {number}: empty line")
            continue
        try:
            rows.append((number, json.loads(line)))
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name} line {number}: invalid JSON ({exc.msg})")
    return data, rows


# --- detections ---------------------------------------------------------------------------


@dataclass
class _Detections:
    frame: list[int]
    track_id: list[int]
    box: list[list[float]]
    conf: list[float]
    role: list[str]
    team: list[str | None]
    jersey: list[str | None]
    jersey_conf: list[float | None]
    dropped_roles: Counter[str]
    team_null_strings: int


_DETECTION_KEYS = ("frame", "track_id", "box", "conf", "role", "team", "jersey", "jersey_conf")


def _parse_detections(
    rows: list[tuple[int, Any]], n_frames: int | None, problems: list[str]
) -> _Detections:
    out = _Detections([], [], [], [], [], [], [], [], Counter(), 0)
    seen: dict[tuple[int, int], int] = {}
    for number, row in rows:
        where = f"{DETECTIONS_NAME} line {number}"
        if not isinstance(row, dict):
            problems.append(f"{where}: must be a JSON object")
            continue
        missing = [key for key in _DETECTION_KEYS if key not in row]
        if missing:
            problems.append(f"{where}: missing key(s) {', '.join(missing)}")
            continue
        found = len(problems)
        frame = _frame_index(where, "frame", row["frame"], n_frames, problems)
        if "state_frame" in row and row["state_frame"] is not None:
            if not _is_int(row["state_frame"]):
                problems.append(f"{where}: state_frame must be an integer or null")
        track_id = row["track_id"]
        if not _is_int(track_id) or track_id < -1:
            problems.append(f"{where}: track_id must be an integer >= -1, got {track_id!r}")
        box = _real_list(where, "box", row["box"], 4, problems)
        if box is not None and not (box[2] > box[0] and box[3] > box[1]):
            problems.append(f"{where}: box {box!r} must have x2 > x1 and y2 > y1")
        if row.get("kf_box") is not None:
            _real_list(where, "kf_box", row["kf_box"], 4, problems)
        conf = row["conf"]
        if not (_finite(conf) and 0.0 <= conf <= 1.0):
            problems.append(f"{where}: conf must be a number in [0, 1], got {conf!r}")
        role = row["role"]
        if role not in ROLES and role not in DROPPED_ROLES:
            problems.append(
                f"{where}: role {role!r} is not one of {', '.join(ROLES + DROPPED_ROLES)}"
            )
        team = row["team"]
        normalised_team = team
        if team in TEAM_NULL_STRINGS:
            normalised_team = None
        elif team is not None and team not in TEAMS:
            problems.append(f"{where}: team must be 'left', 'right' or null, got {team!r}")
        if row.get("team_cluster") is not None and not _finite(row["team_cluster"]):
            problems.append(f"{where}: team_cluster must be a number or null")
        jersey = row["jersey"]
        if jersey is not None and not isinstance(jersey, str):
            problems.append(f"{where}: jersey must be a string or null, got {jersey!r}")
        jersey_conf = row["jersey_conf"]
        if jersey_conf is not None and not _finite(jersey_conf):
            problems.append(f"{where}: jersey_conf must be a number or null, got {jersey_conf!r}")
        if len(problems) > found or frame is None or box is None:
            continue
        if role in DROPPED_ROLES:
            out.dropped_roles[role] += 1
            continue
        if track_id >= 0:
            key = (frame, int(track_id))
            if key in seen:
                problems.append(
                    f"{where}: track {track_id} appears twice in frame {frame}"
                    f" (also line {seen[key]})"
                )
                continue
            seen[key] = number
        if team in TEAM_NULL_STRINGS:
            out.team_null_strings += 1
        out.frame.append(frame)
        out.track_id.append(int(track_id))
        out.box.append(box)
        out.conf.append(float(conf))
        out.role.append(role)
        out.team.append(normalised_team)
        out.jersey.append(jersey)
        out.jersey_conf.append(None if jersey_conf is None else float(jersey_conf))
    return out


# --- calibration --------------------------------------------------------------------------


@dataclass
class _CalibRow:
    frame: int
    status: str
    age: int
    K: list[float] | None
    R: list[float] | None
    C: list[float] | None
    width: int | None
    height: int | None
    rep_err: float | None
    rep_err_n_keypoints: int
    distortion_free: bool


def _rotation_problems(R: list[float]) -> list[str]:
    matrix = np.array(R, dtype=np.float64).reshape(3, 3)
    found = []
    error = float(np.max(np.abs(matrix @ matrix.T - np.eye(3))))
    if error > ROTATION_TOL:
        found.append(f"R is not orthonormal (max |R R^T - I| = {error:.3g} > {ROTATION_TOL:g})")
    det = float(np.linalg.det(matrix))
    if abs(det - 1.0) > 2.0 * ROTATION_TOL:
        found.append(f"R has det {det:.6g}, must be +1")
    return found


def _parse_calib_row(
    number: int, row: Any, n_frames: int | None, problems: list[str]
) -> _CalibRow | None:
    where = f"{CALIB_JSONL_NAME} line {number}"
    if not isinstance(row, dict):
        problems.append(f"{where}: must be a JSON object")
        return None
    missing = [key for key in ("frame", "calib_status", "calib_age") if key not in row]
    if missing:
        problems.append(f"{where}: missing key(s) {', '.join(missing)}")
        return None
    found = len(problems)
    frame = _frame_index(where, "frame", row["frame"], n_frames, problems)
    status = row["calib_status"]
    if status not in CALIB_STATUSES:
        problems.append(
            f"{where}: calib_status {status!r} is not one of {', '.join(CALIB_STATUSES)}"
        )
        return None
    age = row["calib_age"]
    if status == "none" and age is None:
        age = 0
    if not _is_int(age) or age < 0:
        problems.append(f"{where}: calib_age must be an integer >= 0, got {age!r}")
    elif status == "fresh" and age != 0:
        problems.append(f"{where}: calib_age must be 0 for a fresh calibration, got {age}")
    elif status == "held" and age < 1:
        problems.append(f"{where}: calib_age must be >= 1 for a held calibration, got {age}")
    for key in ("n_keypoints", "n_line_classes"):
        value = row.get(key)
        if value is not None and not (_is_int(value) and value >= 0):
            problems.append(f"{where}: {key} must be an integer >= 0 or null, got {value!r}")

    K = R = C = None
    width = height = None
    rep_err: float | None = None
    rep_err_n = 0
    distortion_free = True
    if status != "none":
        needed = ("K", "R", "C", "width", "height", "rep_err_px", "rep_err_n_keypoints")
        absent = [key for key in needed if key not in row]
        if absent:
            problems.append(f"{where}: a {status} calibration needs key(s) {', '.join(absent)}")
            return None
        K = _real_list(where, "K", row["K"], 9, problems)
        R = _real_list(where, "R", row["R"], 9, problems)
        C = _real_list(where, "C", row["C"], 3, problems)
        if R is not None:
            problems.extend(f"{where}: {problem}" for problem in _rotation_problems(R))
        for key in ("width", "height"):
            value = row[key]
            if not (_is_int(value) and value > 0):
                problems.append(f"{where}: {key} must be a positive integer, got {value!r}")
        width, height = row["width"], row["height"]
        value = row["rep_err_px"]
        if value is not None and not (_finite(value) and value >= 0.0):
            problems.append(f"{where}: rep_err_px must be a number >= 0 or null, got {value!r}")
        rep_err = None if value is None else float(value)
        count = row["rep_err_n_keypoints"]
        if count is not None and not (_is_int(count) and count >= 0):
            problems.append(
                f"{where}: rep_err_n_keypoints must be an integer >= 0 or null, got {count!r}"
            )
        rep_err_n = int(count) if _is_int(count) else 0
        distortion = row.get("radial_distortion")
        if distortion is not None:
            if not isinstance(distortion, list) or not all(_finite(d) for d in distortion):
                problems.append(f"{where}: radial_distortion must be a list of numbers or null")
            else:
                distortion_free = all(float(d) == 0.0 for d in distortion)
    if len(problems) > found or frame is None:
        return None
    return _CalibRow(
        frame=frame,
        status=status,
        age=int(age),
        K=K,
        R=R,
        C=C,
        width=None if width is None else int(width),
        height=None if height is None else int(height),
        rep_err=rep_err,
        rep_err_n_keypoints=rep_err_n,
        distortion_free=distortion_free,
    )


def _camera(row: _CalibRow) -> tuple[Camera | None, list[str]]:
    """The row's camera, or None with the reasons it is not a valid pinhole camera."""
    if row.K is None or row.R is None or row.C is None or row.width is None:
        return None, ["no calibration"]
    try:
        camera = Camera(
            K=np.reshape(row.K, (3, 3)),
            R=np.reshape(row.R, (3, 3)),
            C=np.asarray(row.C),
            width=row.width,
            height=row.height,
        )
    except GeometryInputError as exc:
        return None, list(exc.problems)
    return camera, []


# --- the import ---------------------------------------------------------------------------


def import_sngs(
    sngs_dir: str | Path,
    out_dir: str | Path,
    *,
    rep_err_max_px: float = DEFAULT_REP_ERR_MAX_PX,
    overwrite: bool = False,
) -> ImportSummary:
    """Validate the plugin-sngs export in ``sngs_dir`` and write the three outputs.

    Args:
        sngs_dir: directory with detections.jsonl, calib.jsonl and plugin-receipt.json.
        out_dir: output directory (created if needed); must differ from ``sngs_dir``.
        rep_err_max_px: largest rep_err_px, in pixels, of a pitch-usable fresh frame.
        overwrite: replace existing outputs instead of refusing (FileExistsError).

    Raises:
        SngsImportError: listing every problem: missing files, sha256 or size mismatches
            against the receipt, unknown restriction tags, malformed rows (types, frame
            range, statuses, matrix sizes, R not a rotation), duplicate rows.
        FileExistsError: an output exists and ``overwrite`` is False.
        ValueError: invalid arguments.

    Calibrations whose K is not a valid pinhole intrinsics matrix (for example a negative
    focal length) or that carry lens distortion are not errors: they are stored unchanged,
    never pitch-usable, and listed in import.json under ``counts.invalid_camera_frames``.
    """
    source = Path(sngs_dir)
    target = Path(out_dir)
    if not (_finite(rep_err_max_px) and rep_err_max_px > 0.0):
        raise ValueError(f"rep_err_max_px must be a finite number > 0, got {rep_err_max_px!r}")
    if not isinstance(overwrite, bool):
        raise ValueError(f"overwrite must be a bool, got {overwrite!r}")
    if not source.is_dir():
        raise SngsImportError(str(source.name), ["sngs_dir is not a directory"])
    if target.exists() and target.resolve() == source.resolve():
        raise ValueError("out_dir must differ from sngs_dir (the plugin's outputs are read-only)")
    outputs = [target / TRACKS_NAME, target / CALIB_NAME, target / IMPORT_JSON_NAME]
    existing = [path.name for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{target}: output(s) exist: {', '.join(existing)}")

    problems: list[str] = []
    receipt = _read_receipt(source, problems)
    n_frames_hint = None if receipt is None else receipt.video_frames
    det_path = _listed_path(receipt, source, DETECTIONS_NAME)
    cal_path = _listed_path(receipt, source, CALIB_JSONL_NAME)
    det_bytes, det_rows = _read_jsonl(det_path, problems)
    cal_bytes, cal_rows = _read_jsonl(cal_path, problems)
    if receipt is not None:
        parsed_bytes = {
            path: data
            for path, data in ((det_path, det_bytes), (cal_path, cal_bytes))
            if data is not None
        }
        _verify_outputs(receipt, parsed_bytes, problems)
    detections = _parse_detections(det_rows, n_frames_hint, problems)
    calib: dict[int, _CalibRow] = {}
    for number, row in cal_rows:
        parsed = _parse_calib_row(number, row, n_frames_hint, problems)
        if parsed is None:
            continue
        if parsed.frame in calib:
            problems.append(f"{CALIB_JSONL_NAME} line {number}: frame {parsed.frame} repeated")
            continue
        calib[parsed.frame] = parsed
    if receipt is not None:
        _check_receipt_counts(receipt.raw, len(det_rows), len(cal_rows), problems)
    sizes = Counter((row.width, row.height) for row in calib.values() if row.width is not None)
    if len(sizes) > 1:
        listed = ", ".join(f"{w}x{h} ({n} rows)" for (w, h), n in sorted(sizes.items()))
        problems.append(f"{CALIB_JSONL_NAME}: calibrations refer to several frame sizes: {listed}")
    if receipt is not None and receipt.video_frames is not None:
        n_frames = receipt.video_frames
    else:
        last = max([*detections.frame, *calib.keys()], default=-1)
        n_frames = last + 1
    if n_frames < 1:
        problems.append("the export holds no frames")
    if not sizes and n_frames >= 1:
        problems.append(f"{CALIB_JSONL_NAME}: no calibrated row gives the frame size")
    if problems or receipt is None or det_bytes is None or cal_bytes is None:
        raise SngsImportError(str(source.name), problems)
    (frame_width, frame_height) = next(iter(sizes))

    calib_columns, usable, cameras, calib_counts = _calib_table_columns(
        calib, n_frames, frame_width, frame_height, float(rep_err_max_px)
    )
    tracks_columns, track_counts = _tracks_columns(detections, calib, usable, cameras)

    metadata = {
        _PRODUCER_KEY: _PRODUCER.encode("utf-8"),
        _RESTRICTIONS_KEY: json.dumps(list(receipt.restrictions)).encode("utf-8"),
    }
    calib_table = pa.Table.from_pydict(calib_columns, schema=CALIB_SCHEMA.with_metadata(metadata))
    tracks_table = pa.Table.from_pydict(
        tracks_columns, schema=TRACKS_SCHEMA.with_metadata(metadata)
    )
    # Write under temporary names and move into place at the end, so an interrupted run
    # never leaves a complete-looking but partial set of outputs.
    target.mkdir(parents=True, exist_ok=True)
    staged = {path: path.with_name(path.name + ".partial") for path in outputs}
    pq.write_table(calib_table, staged[target / CALIB_NAME])
    pq.write_table(tracks_table, staged[target / TRACKS_NAME])

    counts: dict[str, Any] = {
        "detection_rows": len(det_rows),
        "track_rows": tracks_table.num_rows,
        "calib_rows_in_export": len(calib),
        "calib_rows_written": calib_table.num_rows,
        "frames": n_frames,
        **calib_counts,
        **track_counts,
        "dropped_rows_by_role": dict(sorted(detections.dropped_roles.items())),
        "team_nan_strings_stored_as_null": detections.team_null_strings,
    }
    record = {
        "schema_version": 1,
        "producer": _PRODUCER,
        "gbf_version": gbf.__version__,
        "inputs": [
            {"name": DETECTIONS_NAME, "sha256": sha256_bytes(det_bytes), "bytes": len(det_bytes)},
            {"name": CALIB_JSONL_NAME, "sha256": sha256_bytes(cal_bytes), "bytes": len(cal_bytes)},
            {
                "name": RECEIPT_NAME,
                "sha256": sha256_bytes(receipt.raw_bytes),
                "bytes": len(receipt.raw_bytes),
            },
        ],
        "plugin": {
            key: receipt.raw[key]
            for key in ("plugin", "version", "license", "frame_alignment", "missing_video_frames")
            if key in receipt.raw and _plain_json(receipt.raw[key])
        },
        "restrictions": list(receipt.restrictions),
        "settings": {
            "rep_err_max_px": float(rep_err_max_px),
            "in_pitch_margin_m": IN_PITCH_MARGIN_M,
            "foot_point": "bottom centre of box",
            "pitch_usable": "calib_status fresh, rep_err_px <= rep_err_max_px, valid pinhole"
            " camera without lens distortion",
        },
        "frame_size": [frame_width, frame_height],
        "counts": counts,
        "outputs": [
            {
                "name": path.name,
                "sha256": sha256_file(staged[path]),
                "bytes": staged[path].stat().st_size,
            }
            for path in (target / TRACKS_NAME, target / CALIB_NAME)
        ],
        "limitations": [
            "rep_err_px is the plugin's self-consistency, not a measured position error",
            "pitch positions assume the foot point (bottom centre of the box) is on the grass",
            "position and speed errors are unmeasured",
        ],
    }
    write_json(staged[target / IMPORT_JSON_NAME], record)
    for final, temporary in staged.items():
        temporary.replace(final)
    return ImportSummary(
        out_dir=target,
        tracks_path=target / TRACKS_NAME,
        calib_path=target / CALIB_NAME,
        import_json_path=target / IMPORT_JSON_NAME,
        n_frames=n_frames,
        n_detection_rows=len(det_rows),
        n_track_rows=tracks_table.num_rows,
        n_calib_rows=calib_table.num_rows,
        n_pitch_usable_frames=int(sum(usable.values())),
        restrictions=receipt.restrictions,
        counts=counts,
    )


def _plain_json(value: Any) -> bool:
    """True when ``value`` serialises as strict JSON (no NaN or Infinity)."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _check_receipt_counts(
    raw: Mapping[str, Any], n_detections: int, n_calib: int, problems: list[str]
) -> None:
    """Cross-check the optional row counts some receipts state."""
    for key, actual, what in (
        ("detections", n_detections, DETECTIONS_NAME),
        ("frames_with_calibration_rows", n_calib, CALIB_JSONL_NAME),
    ):
        stated = raw.get(key)
        if stated is not None and stated != actual:
            problems.append(f"{RECEIPT_NAME}: {key} = {stated!r} but {what} has {actual} row(s)")


def _calib_table_columns(
    calib: Mapping[int, _CalibRow],
    n_frames: int,
    frame_width: int,
    frame_height: int,
    rep_err_max_px: float,
) -> tuple[dict[str, list[Any]], dict[int, bool], dict[int, Camera], dict[str, Any]]:
    columns: dict[str, list[Any]] = {name: [] for name in CALIB_SCHEMA.names}
    usable: dict[int, bool] = {}
    cameras: dict[int, Camera] = {}
    reasons: Counter[str] = Counter()
    invalid: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    added = 0
    for frame in range(n_frames):
        row = calib.get(frame)
        if row is None:
            added += 1
            row = _CalibRow(frame, "none", 0, None, None, None, None, None, None, 0, True)
        status_counts[row.status] += 1
        ok = False
        if row.status != "none":
            camera, camera_problems = _camera(row)
            if camera is None:
                invalid.append({"frame": frame, "problems": camera_problems})
                reasons["invalid_camera"] += 1
            elif not row.distortion_free:
                reasons["lens_distortion"] += 1
            elif row.status != "fresh":
                reasons["held"] += 1
            elif row.rep_err is None:
                reasons["fresh_rep_err_null"] += 1
            elif row.rep_err > rep_err_max_px:
                reasons["fresh_rep_err_above_max"] += 1
            else:
                ok = True
                cameras[frame] = camera
        else:
            reasons["none"] += 1
        usable[frame] = ok
        columns["frame_index"].append(frame)
        columns["width"].append(row.width if row.width is not None else frame_width)
        columns["height"].append(row.height if row.height is not None else frame_height)
        columns["K"].append(row.K)
        columns["R"].append(row.R)
        columns["C"].append(row.C)
        columns["calib_status"].append(row.status)
        columns["calib_age"].append(row.age)
        columns["rep_err"].append(row.rep_err)
        columns["rep_err_n_keypoints"].append(row.rep_err_n_keypoints)
        columns["pitch_usable"].append(ok)
    counts = {
        "calib_rows_added_as_none": added,
        "calib_status_counts": dict(sorted(status_counts.items())),
        "pitch_usable_frames": int(sum(usable.values())),
        "not_pitch_usable_by_reason": dict(sorted(reasons.items())),
        "invalid_camera_frames": invalid,
    }
    return columns, usable, cameras, counts


def _tracks_columns(
    detections: _Detections,
    calib: Mapping[int, _CalibRow],
    usable: Mapping[int, bool],
    cameras: Mapping[int, Camera],
) -> tuple[dict[str, Any], dict[str, Any]]:
    n = len(detections.frame)
    frame = np.array(detections.frame, dtype=np.int64)
    track_id = np.array(detections.track_id, dtype=np.int64)
    order = np.lexsort((track_id, frame))
    frame = frame[order]
    track_id = track_id[order]
    boxes = np.array(detections.box, dtype=np.float64).reshape(n, 4)[order]
    foot_u = 0.5 * (boxes[:, 0] + boxes[:, 2])
    foot_v = boxes[:, 3].copy()
    pitch_x = np.full(n, np.nan)
    pitch_y = np.full(n, np.nan)
    hit = np.zeros(n, dtype=bool)
    status = [calib[f].status if f in calib else "none" for f in frame.tolist()]
    row_usable = np.array([usable.get(f, False) for f in frame.tolist()], dtype=bool)
    if n:
        starts = np.flatnonzero(np.r_[True, frame[1:] != frame[:-1]])
        stops = np.r_[starts[1:], n]
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
            camera = cameras.get(int(frame[start]))
            if camera is None:
                continue
            uv = np.column_stack([foot_u[start:stop], foot_v[start:stop]])
            points, valid = intersect_ground(camera.C, pixel_rays(camera, uv))
            pitch_x[start:stop] = points[:, 0]
            pitch_y[start:stop] = points[:, 1]
            hit[start:stop] = valid
    in_pitch = (
        hit
        & (np.abs(pitch_x) <= PITCH_HALF_LENGTH_M + IN_PITCH_MARGIN_M)
        & (np.abs(pitch_y) <= PITCH_HALF_WIDTH_M + IN_PITCH_MARGIN_M)
    )
    pick = order.tolist()
    columns = {
        "frame": frame,
        "track_id": track_id,
        "x1": boxes[:, 0],
        "y1": boxes[:, 1],
        "x2": boxes[:, 2],
        "y2": boxes[:, 3],
        "conf": np.array(detections.conf, dtype=np.float32)[order],
        "role": [detections.role[i] for i in pick],
        "team": [detections.team[i] for i in pick],
        "jersey": [detections.jersey[i] for i in pick],
        "jersey_conf": [detections.jersey_conf[i] for i in pick],
        "foot_u": foot_u,
        "foot_v": foot_v,
        "calib_status": status,
        "pitch_usable": row_usable,
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "in_pitch": in_pitch,
    }
    counts = {
        "track_rows_unknown_track_id": int(np.count_nonzero(track_id < 0)),
        "track_rows_on_frames_without_calib_row": int(
            sum(1 for f in frame.tolist() if f not in calib)
        ),
        "track_rows_pitch_usable": int(np.count_nonzero(row_usable)),
        "track_rows_ray_missed_ground": int(np.count_nonzero(row_usable & ~hit)),
        "track_rows_in_pitch": int(np.count_nonzero(in_pitch)),
        "frames_with_detections": int(np.unique(frame).size),
    }
    return columns, counts


# --- readers ------------------------------------------------------------------------------


def _read_checked(path: str | Path, default_name: str, schema: pa.Schema) -> pd.DataFrame:
    source = Path(path)
    if source.is_dir():
        source = source / default_name
    try:
        table = pq.read_table(source)
    except (OSError, pa.ArrowException) as exc:
        raise SngsImportError(source.name, [f"cannot read as parquet: {exc}"]) from exc
    problems = []
    for field in schema:
        if field.name not in table.schema.names:
            problems.append(f"missing column {field.name}")
            continue
        actual = table.schema.field(field.name).type
        if not actual.equals(field.type):
            problems.append(f"column {field.name}: type {actual}, expected {field.type}")
        elif not field.nullable and table.column(field.name).null_count:
            problems.append(f"column {field.name}: {table.column(field.name).null_count} null(s)")
    if problems:
        raise SngsImportError(source.name, problems)
    frame = table.to_pandas()
    metadata = table.schema.metadata or {}
    raw = metadata.get(_RESTRICTIONS_KEY)
    frame.attrs["restrictions"] = [] if raw is None else list(json.loads(raw.decode("utf-8")))
    return frame


def load_tracks(path: str | Path) -> pd.DataFrame:
    """Read tracks.parquet (a file, or the directory holding it) as a DataFrame.

    Columns and dtypes follow :data:`TRACKS_SCHEMA`: frame/track_id int64, boxes and foot
    points float64 pixels, conf float32, role/team/jersey/calib_status strings (pandas'
    default string dtype; a null team or jersey reads as missing, test it with ``isna()``),
    jersey_conf float64 (NaN where null), pitch_usable/in_pitch bool, pitch_x/pitch_y
    float64 metres (NaN unless pitch-usable). ``attrs["restrictions"]`` holds the
    restriction tags stored in the file. Raises SngsImportError listing every schema problem.
    """
    return _read_checked(path, TRACKS_NAME, TRACKS_SCHEMA)


def load_calib(path: str | Path) -> pd.DataFrame:
    """Read calib.parquet (a file, or the directory holding it) as a DataFrame.

    One row per canonical frame; K and R hold flat 9-element float64 arrays (row-major) and
    C a 3-element array, or None when calib_status is "none". width/height/calib_age/
    rep_err_n_keypoints are int32, rep_err float64 (NaN where null). Use
    :func:`camera_for_frame` instead of rebuilding cameras by hand. ``attrs["restrictions"]``
    holds the stored restriction tags. Raises SngsImportError listing every schema problem.
    """
    return _read_checked(path, CALIB_NAME, CALIB_SCHEMA)


def _camera_from_row(row: Mapping[str, Any]) -> Camera | None:
    K, R, C = row["K"], row["R"], row["C"]
    if K is None or R is None or C is None:
        return None
    try:
        return Camera(
            K=np.reshape(np.asarray(K, dtype=np.float64), (3, 3)),
            R=np.reshape(np.asarray(R, dtype=np.float64), (3, 3)),
            C=np.asarray(C, dtype=np.float64),
            width=int(row["width"]),
            height=int(row["height"]),
        )
    except (GeometryInputError, ValueError):
        return None


def _check_calib_frame(calib_df: Any) -> None:
    if not isinstance(calib_df, pd.DataFrame):
        raise SngsImportError("calib_df", [f"expected a DataFrame, got {type(calib_df).__name__}"])
    missing = [name for name in CALIB_SCHEMA.names if name not in calib_df.columns]
    if missing:
        raise SngsImportError("calib_df", [f"missing column(s): {', '.join(missing)}"])


def camera_for_frame(
    calib_df: pd.DataFrame, frame: int, *, usable_only: bool = True
) -> Camera | None:
    """The gbf.geometry Camera of canonical ``frame``, or None.

    With ``usable_only`` (the default) only pitch-usable frames have a camera (gate G6), so
    a caller cannot project through a held or poorly fitted calibration by accident. With
    ``usable_only=False`` any calibrated frame (fresh or held) whose values form a valid
    camera is returned, e.g. for drawing. None when the frame has no row, its status is
    "none", or its values are not a valid camera. K is in the export's pixel grid
    (calib width x height); rescale with ``Camera.resized`` for other frame sizes.
    """
    _check_calib_frame(calib_df)
    if not _is_int(frame) and not isinstance(frame, np.integer):
        raise SngsImportError("camera_for_frame", [f"frame must be an integer, got {frame!r}"])
    if not isinstance(usable_only, bool):
        raise SngsImportError("camera_for_frame", ["usable_only must be a bool"])
    matches = calib_df.loc[calib_df["frame_index"].to_numpy() == int(frame)]
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise SngsImportError("camera_for_frame", [f"frame {int(frame)} has {len(matches)} rows"])
    row = matches.iloc[0]
    if usable_only and not bool(row["pitch_usable"]):
        return None
    return _camera_from_row(row)


def cameras_by_frame(calib_df: pd.DataFrame, *, usable_only: bool = True) -> dict[int, Camera]:
    """``{frame_index: Camera}`` for every frame :func:`camera_for_frame` would return.

    Builds each camera once; use it instead of calling camera_for_frame in a loop.
    """
    _check_calib_frame(calib_df)
    frames = calib_df["frame_index"].to_numpy()
    if np.unique(frames).size != frames.size:
        raise SngsImportError("cameras_by_frame", ["frame_index has duplicate values"])
    result: dict[int, Camera] = {}
    for row in calib_df.to_dict("records"):
        if usable_only and not bool(row["pitch_usable"]):
            continue
        camera = _camera_from_row(row)
        if camera is not None:
            result[int(row["frame_index"])] = camera
    return result
