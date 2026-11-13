"""Lossless event windows cut at canonical frame numbers (plan 3.3 step 1, 3.4).

A window file holds exactly the canonical CFR frames ``start..end`` (inclusive) of a source,
so decoded window frame ``k`` is bit-identical to canonical source frame ``start + k``.

Frames are decoded once by :func:`gbf.ingest.frames.iter_frames` (with ``fps``, so a source
that is not CFR yet is mapped onto the grid exactly as normalisation would map it), hashed,
and piped as raw RGB into FFV1 (lossless, every frame a keyframe) in Matroska. The window
stores RGB, not YUV, so decoding it never involves a colour-matrix conversion: its frames
come back bit-identical on any machine and any ffmpeg version, which is what makes the
recorded pixel hashes portable. Before the file is moved into place every decoded window
frame is compared with the frame that was written; the first and last frame hashes are
recorded.
"""

from __future__ import annotations

import os
import re
import subprocess
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from gbf.ingest.errors import FFmpegError, WindowError
from gbf.ingest.frames import as_frame_index, iter_frames, pixel_sha256
from gbf.ingest.probing import keyframe_flags, probe, stream_geometry
from gbf.ingest.rates import format_rate, parse_rate
from gbf.ingest.tools import LOGLEVEL, StderrCollector, ffmpeg_path, media_arg
from gbf.util.hashing import sha256_file

CODEC = "ffv1"
#: FFV1 stores 8-bit RGB as bgr0; decoding it to rgb24 is a byte shuffle, never a matrix.
PIX_FMT = "bgr0"
CONTAINER = "matroska"
OUTPUT_SUFFIX = ".mkv"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class WindowRecord:
    """Facts about one written window. ``file_name`` is for messages; it is not serialised."""

    window_id: str
    start_frame: int
    end_frame: int
    n_frames: int
    fps: str
    first_pixel_sha256: str
    last_pixel_sha256: str
    file_sha256: str
    file_bytes: int
    codec: str
    pix_fmt: str
    container: str
    source_sha256: str
    file_name: str

    def to_dict(self) -> dict[str, Any]:
        """JSON form used in ``video.json`` (schema ``video``, ``$defs/window``)."""
        return {
            "window_id": self.window_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "first_pixel_sha256": self.first_pixel_sha256,
            "last_pixel_sha256": self.last_pixel_sha256,
            "file_sha256": self.file_sha256,
            "file_bytes": self.file_bytes,
            "codec": self.codec,
            "pix_fmt": self.pix_fmt,
            "container": self.container,
            "source_sha256": self.source_sha256,
        }


def window_id(start_frame: int, end_frame: int) -> str:
    """Window id from its canonical frame range, e.g. ``w000474-000550``.

    Unique within one video; ``(canonical id, window id)`` is unique across videos. Track ids
    inside a window are prefixed with it (plan 3.4).
    """
    return f"w{start_frame:06d}-{end_frame:06d}"


def _encoder_args(tmp: Path, width: int, height: int, rate: Fraction) -> list[str]:
    return [
        ffmpeg_path(),
        "-hide_banner",
        "-nostats",
        "-loglevel",
        LOGLEVEL,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        format_rate(rate),
        "-i",
        "pipe:0",
        "-map_metadata",
        "-1",
        "-c:v",
        CODEC,
        "-level",
        "3",
        "-coder",
        "range_tab",
        "-context",
        "1",
        "-g",
        "1",
        "-slicecrc",
        "1",
        "-pix_fmt",
        PIX_FMT,
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-f",
        CONTAINER,
        media_arg(tmp),
    ]


def _request_problems(
    source: Path, start: object, end: object, dst: Path, overwrite: bool
) -> list[str]:
    problems = []
    first, last = as_frame_index(start), as_frame_index(end)
    if first is None:
        problems.append(f"start_frame must be an integer, got {start!r}")
    elif first < 0:
        problems.append(f"start_frame must be >= 0, got {first}")
    if last is None:
        problems.append(f"end_frame_inclusive must be an integer, got {end!r}")
    elif first is not None and last < first:
        problems.append(f"end_frame_inclusive ({last}) must be >= start_frame ({first})")
    if not source.is_file():
        problems.append(f"source is not a file: {source}")
    if dst.suffix.lower() != OUTPUT_SUFFIX:
        problems.append(
            f"destination must end in {OUTPUT_SUFFIX} (FFV1 in Matroska), got {dst.name!r}"
        )
    if dst.exists() and not overwrite:
        problems.append(f"destination exists: {dst} (pass overwrite=True to replace it)")
    if source.is_file() and dst.exists() and dst.resolve() == source.resolve():
        problems.append("destination is the source file")
    return problems


def _write(source: Path, start: int, end: int, tmp: Path, rate: Fraction, what: str) -> list[str]:
    """Encode canonical frames ``start..end`` of ``source`` into ``tmp``; return their hashes."""
    geometry = stream_geometry(source)
    encoder = subprocess.Popen(
        _encoder_args(tmp, geometry.frame_width, geometry.frame_height, rate),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stdin = encoder.stdin
    if stdin is None or encoder.stderr is None:  # pragma: no cover - Popen guarantees pipes
        encoder.kill()
        raise FFmpegError(what, None, "could not open pipes to ffmpeg")
    collector = StderrCollector(encoder.stderr)
    hashes: list[str] = []
    write_error: OSError | None = None
    try:
        with closing(iter_frames(source, start, end + 1, fps=rate)) as frames:
            for _, frame in frames:
                hashes.append(pixel_sha256(frame))
                try:
                    stdin.write(frame.data)
                except OSError as exc:  # the encoder exited early; its stderr says why
                    write_error = exc
                    break
    except BaseException:
        encoder.kill()
        encoder.wait(timeout=60)
        collector.text(timeout=10)
        raise
    finally:
        try:
            stdin.close()
        except OSError:
            pass  # already broken; reported through write_error or the exit code
    returncode = encoder.wait()
    stderr = collector.text()
    if write_error is not None or returncode != 0 or stderr.strip():
        detail = (
            stderr if write_error is None else f"writing frames failed: {write_error}\n{stderr}"
        )
        raise FFmpegError(what, returncode, detail)
    return hashes


def extract_window(
    source: str | Path,
    start_frame: int,
    end_frame_inclusive: int,
    dst: str | Path,
    fps: str | Fraction = "25/1",
    *,
    overwrite: bool = False,
    source_sha256: str | None = None,
) -> WindowRecord:
    """Write canonical frames ``start_frame..end_frame_inclusive`` of ``source`` to ``dst``.

    ``dst`` must end in ``.mkv``. ``fps`` is the canonical rate; a source already normalised
    at that rate is read frame for frame. ``source_sha256`` (sha256 of the source file bytes)
    saves hashing a large source once per window. Raises :class:`WindowError` listing every
    problem with the request or with the written file (frame count, all-intra, bit-exactness
    of every frame, including the first and the last),
    :class:`gbf.ingest.errors.FrameRangeError` when the range runs past the end of the video,
    and :class:`gbf.ingest.errors.FFmpegError` when ffmpeg fails.
    """
    source, dst = Path(source), Path(dst)
    what = f"window {start_frame}..{end_frame_inclusive} of {source.name}"
    problems = _request_problems(source, start_frame, end_frame_inclusive, dst, overwrite)
    if source_sha256 is not None and not _SHA256.match(source_sha256):
        problems.append(f"source_sha256 must be 64 lowercase hex digits, got {source_sha256!r}")
    rate = Fraction(1)
    try:
        rate = parse_rate(fps)
    except ValueError as exc:
        problems.append(str(exc))
    if problems:
        raise WindowError(what, problems)
    start, end = int(start_frame), int(end_frame_inclusive)
    n_frames = end - start + 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    try:
        written = _write(source, start, end, tmp, rate, what)
        facts = probe(tmp)
        problems = []
        if facts.nb_frames_decoded != n_frames:
            problems.append(
                f"window decodes to {facts.nb_frames_decoded} frames, expected {n_frames}"
            )
        if not facts.is_cfr_at(rate):
            problems.append(
                f"window is not CFR at {format_rate(rate)}: r_frame_rate={facts.r_frame_rate}, "
                f"avg_frame_rate={facts.avg_frame_rate}"
            )
        if facts.codec_name != CODEC:
            problems.append(f"window codec is {facts.codec_name}, expected {CODEC}")
        keyframes = keyframe_flags(tmp)
        if len(keyframes) != n_frames or not all(keyframes):
            problems.append(
                f"window is not all-intra: {sum(keyframes)} keyframes among "
                f"{len(keyframes)} packets"
            )
        decoded = [pixel_sha256(frame) for _, frame in iter_frames(tmp)]
        if not decoded or decoded[0] != written[0]:
            problems.append(f"first frame (canonical {start}) is not bit-identical to the source")
        if not decoded or decoded[-1] != written[-1]:
            problems.append(f"last frame (canonical {end}) is not bit-identical to the source")
        differing = [
            start + k for k, (a, b) in enumerate(zip(written, decoded, strict=False)) if a != b
        ]
        if differing:
            shown = ", ".join(str(index) for index in differing[:20])
            problems.append(f"{len(differing)} frame(s) differ from the source: {shown}")
        if problems:
            raise WindowError(what, problems)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return WindowRecord(
        window_id=window_id(start, end),
        start_frame=start,
        end_frame=end,
        n_frames=n_frames,
        fps=format_rate(rate),
        first_pixel_sha256=written[0],
        last_pixel_sha256=written[-1],
        file_sha256=facts.sha256,
        file_bytes=facts.file_bytes,
        codec=facts.codec_name,
        pix_fmt=facts.pix_fmt or PIX_FMT,
        container=CONTAINER,
        source_sha256=source_sha256 if source_sha256 is not None else sha256_file(source),
        file_name=dst.name,
    )
