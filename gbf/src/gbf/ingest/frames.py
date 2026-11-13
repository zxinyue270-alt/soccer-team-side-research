"""The project's single frame reader.

Every module reads video frames through this file and nowhere else (plan gate G5). Frames
come out of one ``ffmpeg`` process that decodes sequentially and writes raw ``rgb24`` to a
pipe; nothing here seeks, so a long-GOP video is never re-opened per frame and no frame can
be skipped or duplicated by an inexact seek (the tracklab ``VideoReader`` seeked to
``idx - 1`` and duplicated frames 0 and 1).

Frame index conventions:

- The index of a frame is its position in the decoded sequence, counted from 0. For files
  written by :func:`gbf.ingest.normalizing.normalize` or
  :func:`gbf.ingest.window.extract_window` this *is* the canonical CFR frame index.
- With ``fps="25/1"`` any file is first mapped onto the CFR grid on the fly, with the same
  filter chain that :func:`~gbf.ingest.normalizing.normalize` uses (:func:`cfr_filter`), so
  frame ``k`` is the source frame shown at ``k / fps`` seconds after the first frame. The
  pixels are the source's own decoded pixels; :func:`~gbf.ingest.normalizing.normalize`
  re-encodes them (lossy), so the two agree in timing, not bit for bit.
- Frames start at ``start`` by decoding and discarding, through a ``select`` filter with
  passthrough timing, so the first frame returned is exactly frame ``start``.

Arrays are ``(height, width, 3)`` ``uint8`` RGB, C-contiguous and writable. Display rotation
is applied the way ffmpeg applies it by default, so the size can be the stored size
transposed; :func:`gbf.ingest.probing.stream_geometry` reports it.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable, Iterator
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from typing import IO

import numpy as np

from gbf.ingest.errors import FFmpegError, FrameRangeError
from gbf.ingest.probing import (
    StreamGeometry,
    expected_cfr_frame_count,
    probe,
    stream_geometry,
)
from gbf.ingest.rates import format_rate, parse_rate
from gbf.ingest.tools import LOGLEVEL, StderrCollector, ffmpeg_path, media_arg

#: Longest ``select`` expression passed in one ffmpeg call; longer index lists are split
#: into several passes so the command line stays far below the Windows limit (32767).
MAX_SELECT_EXPRESSION = 8000

#: Most terms (single frames or runs) in one ``select`` expression: ffmpeg 8.1 refuses an
#: expression of 101 terms ("Cannot allocate memory") whatever its length.
MAX_SELECT_TERMS = 100

#: Header format of :func:`pixel_sha256`; other tools can recompute the hash from it.
PIXEL_HASH_HEADER = "gbf.rgb24 h={height} w={width} c=3\n"


def cfr_filter(fps: str | Fraction) -> str:
    """ffmpeg filter chain that maps a stream onto the canonical CFR grid at ``fps``.

    ``setpts=PTS-STARTPTS`` puts the first video frame at time 0, whatever the container's
    start time or audio offset is; ``fps`` then emits one frame per ``1/fps`` seconds, holding
    the latest source frame (timestamps rounded to the nearest grid point, halves away from
    zero) and ending at the rounded end of the last source frame. Rounding and end-of-stream
    handling are spelled out so a change of ffmpeg defaults cannot change frame counts.
    """
    return f"setpts=PTS-STARTPTS,fps=fps={format_rate(parse_rate(fps))}:round=near:eof_action=round"


def as_frame_index(value: object) -> int | None:
    """``value`` as a Python int when it is an integer (bools excluded), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        return None
    return int(value)


def _range_problems(start: object, stop: object) -> list[str]:
    problems = []
    first = as_frame_index(start)
    if first is None:
        problems.append(f"start must be an integer, got {start!r}")
    elif first < 0:
        problems.append(f"start must be >= 0, got {first}")
    if stop is not None:
        last = as_frame_index(stop)
        if last is None:
            problems.append(f"stop must be an integer or None, got {stop!r}")
        elif last < 0:
            problems.append(f"stop must be >= 0, got {last}")
        elif first is not None and last < first:
            problems.append(f"stop ({last}) must not be smaller than start ({first})")
    return problems


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")


def _read_into(stream: IO[bytes], buffer: bytearray) -> int:
    view = memoryview(buffer)
    total = 0
    while total < len(buffer):
        got = stream.readinto(view[total:])  # type: ignore[attr-defined]
        if not got:
            break
        total += got
    return total


def _stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - only if the OS refuses to kill
        pass


def _decode(
    path: Path,
    geometry: StreamGeometry,
    filters: list[str],
    max_frames: int | None,
    what: str,
) -> Iterator[np.ndarray]:
    """Yield decoded RGB frames from one ffmpeg process; see the module docstring."""
    args = [
        ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        LOGLEVEL,
        "-i",
        media_arg(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
    ]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    if max_frames is not None:
        args += ["-frames:v", str(max_frames)]
    args.append("pipe:1")

    width, height = geometry.frame_width, geometry.frame_height
    frame_bytes = width * height * 3
    proc = subprocess.Popen(
        args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout = proc.stdout
    if stdout is None or proc.stderr is None:  # pragma: no cover - Popen guarantees pipes
        _stop(proc)
        raise FFmpegError(what, None, "could not open pipes to ffmpeg")
    collector = StderrCollector(proc.stderr)
    finished = False
    try:
        while True:
            buffer = bytearray(frame_bytes)
            got = _read_into(stdout, buffer)
            if got == 0:
                break
            if got != frame_bytes:
                returncode = proc.wait()
                finished = True
                raise FFmpegError(
                    what,
                    returncode,
                    f"ffmpeg wrote a partial frame ({got} of {frame_bytes} bytes for "
                    f"{width}x{height} rgb24)\n{collector.text()}",
                )
            yield np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
        returncode = proc.wait()
        finished = True
        stderr = collector.text()
        if returncode != 0 or stderr.strip():
            raise FFmpegError(what, returncode, stderr)
    finally:
        if not finished:
            # The consumer stopped early (or an exception passed through): stop ffmpeg now
            # instead of letting it decode the rest of the file into a pipe nobody reads.
            _stop(proc)
        stdout.close()
        collector.text(timeout=10)


def iter_frames(
    path: str | Path,
    start: int = 0,
    stop: int | None = None,
    *,
    fps: str | Fraction | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, frame)`` for frames ``start <= index < stop`` in order.

    ``stop=None`` reads to the end. ``fps`` maps the file onto the CFR grid first (see the
    module docstring); leave it ``None`` for files that are already canonical CFR.
    Arguments are checked immediately. While iterating, :class:`FrameRangeError` is raised
    when ``stop`` (or ``start``) lies beyond the last frame, and :class:`FFmpegError`, with
    ffmpeg's stderr, when decoding fails or ffmpeg reports any error. Stopping early, for
    example with ``break``, ends the ffmpeg process.
    """
    path = Path(path)
    problems = _range_problems(start, stop)
    rate = None
    if fps is not None:
        try:
            rate = parse_rate(fps)
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise FrameRangeError(f"iter_frames({path.name})", problems)
    _require_file(path)
    start = int(start)
    stop = None if stop is None else int(stop)
    if stop == start:
        return iter(())  # an empty range, like range(start, start)
    geometry = stream_geometry(path)
    filters = [cfr_filter(rate)] if rate is not None else []
    if start > 0:
        filters.append(f"select='gte(n,{start})'")
    max_frames = None if stop is None else stop - start
    what = f"decode {path.name} frames {start}..{'end' if stop is None else stop - 1}"
    return _iter_indexed(path, geometry, filters, start, stop, max_frames, what)


def _iter_indexed(
    path: Path,
    geometry: StreamGeometry,
    filters: list[str],
    start: int,
    stop: int | None,
    max_frames: int | None,
    what: str,
) -> Iterator[tuple[int, np.ndarray]]:
    index = start
    with closing(_decode(path, geometry, filters, max_frames, what)) as frames:
        for frame in frames:
            yield index, frame
            index += 1
    if stop is not None and index < stop:
        raise FrameRangeError(
            what,
            [f"the video ends before frame {stop - 1}: frames {index}..{stop - 1} do not exist"],
        )
    if stop is None and start > 0 and index == start:
        raise FrameRangeError(what, [f"start {start} lies beyond the last frame"])


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    first = previous = indices[0]
    for value in indices[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((first, previous))
        first = previous = value
    runs.append((first, previous))
    return runs


def _select_passes(indices: list[int]) -> list[tuple[str, list[int]]]:
    """Split sorted unique indices into ``(select expression, indices)`` passes."""
    passes: list[tuple[str, list[int]]] = []
    terms: list[str] = []
    members: list[int] = []
    length = 0
    for first, last in _runs(indices):
        term = f"eq(n,{first})" if first == last else f"between(n,{first},{last})"
        if terms and (
            length + len(term) + 1 > MAX_SELECT_EXPRESSION or len(terms) >= MAX_SELECT_TERMS
        ):
            passes.append(("+".join(terms), members))
            terms, members, length = [], [], 0
        terms.append(term)
        members.extend(range(first, last + 1))
        length += len(term) + 1
    passes.append(("+".join(terms), members))
    return passes


def read_frames(
    path: str | Path,
    indices: Iterable[int],
    *,
    fps: str | Fraction | None = None,
) -> dict[int, np.ndarray]:
    """Return ``{index: frame}`` for the given frame indices (duplicates are read once).

    Decodes sequentially and keeps only the requested frames (``select`` filter), so a few
    frames from a long video cost one decode pass up to the largest index. Raises
    :class:`FrameRangeError` listing every invalid or missing index.
    """
    path = Path(path)
    problems = []
    wanted: set[int] = set()
    for value in indices:
        index = as_frame_index(value)
        if index is None:
            problems.append(f"frame index must be an integer, got {value!r}")
        elif index < 0:
            problems.append(f"frame index must be >= 0, got {index}")
        else:
            wanted.add(index)
    rate = None
    if fps is not None:
        try:
            rate = parse_rate(fps)
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise FrameRangeError(f"read_frames({path.name})", problems)
    _require_file(path)
    if not wanted:
        return {}
    geometry = stream_geometry(path)
    ordered = sorted(wanted)
    result: dict[int, np.ndarray] = {}
    for expression, members in _select_passes(ordered):
        filters = [cfr_filter(rate)] if rate is not None else []
        filters.append(f"select='{expression}'")
        what = f"read {len(members)} frame(s) of {path.name}"
        with closing(_decode(path, geometry, filters, len(members), what)) as frames:
            decoded = list(frames)
        for index, frame in zip(members, decoded, strict=False):
            result[index] = frame
    missing = [index for index in ordered if index not in result]
    if missing:
        shown = ", ".join(str(index) for index in missing[:20])
        more = f" and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FrameRangeError(
            f"read_frames({path.name})",
            [f"frames beyond the end of the video: {shown}{more}"],
        )
    return result


def frame_count(path: str | Path, *, fps: str | Fraction | None = None) -> int:
    """Number of frames :func:`iter_frames` yields for ``path`` (with the same ``fps``).

    Without ``fps`` this is ffprobe's decoded frame count (``-count_frames``); with ``fps``
    it is the CFR count of the decoded timeline, see
    :func:`gbf.ingest.probing.expected_cfr_frame_count`. Both decode the whole stream.
    """
    facts = probe(path)
    if fps is None:
        return facts.nb_frames_decoded
    return expected_cfr_frame_count(facts, fps)


def pixel_sha256(frame: np.ndarray) -> str:
    """sha256 of an RGB frame: a short header with its shape, then its raw bytes.

    The header is :data:`PIXEL_HASH_HEADER` (ASCII), so identical bytes in a different shape
    never hash alike. Raises ``ValueError`` listing every problem when ``frame`` is not an
    ``(height, width, 3)`` ``uint8`` array.
    """
    problems = []
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"frame must be a numpy array, got {type(frame).__name__}")
    if frame.dtype != np.uint8:
        problems.append(f"dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        problems.append(f"shape must be (height, width, 3), got {frame.shape}")
    elif frame.shape[0] < 1 or frame.shape[1] < 1:
        problems.append(f"frame must not be empty, got shape {frame.shape}")
    if problems:
        raise ValueError("pixel_sha256: " + "; ".join(problems))
    height, width = int(frame.shape[0]), int(frame.shape[1])
    digest = hashlib.sha256(PIXEL_HASH_HEADER.format(height=height, width=width).encode("ascii"))
    digest.update(np.ascontiguousarray(frame).data)
    return digest.hexdigest()
