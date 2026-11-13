"""Write rendered frames to an H.264 CFR mp4 and verify it (synthetic sources for L0).

The frames go to ffmpeg as raw rgb24 through a pipe (one process, no temporary images):

    ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -framerate R -i pipe:0 -map_metadata -1
           -c:v libx264 -preset medium -crf CRF -pix_fmt yuv420p -threads 8
           -fflags +bitexact -flags:v +bitexact -f mp4 DST

Raw input at a fixed frame rate gives exactly one output frame per input frame, every
frame lasting 1/R seconds, so the file is CFR at R with no duplicated or dropped frame. The
encoder settings are fixed (x264's output depends on its thread count), so the same frames
give the same bytes with the same ffmpeg build. The result is checked with
gbf.ingest.probe -- decoded frame count, CFR at R, frame size, codec -- before it is moved
into place; a failure raises SynthVideoError and leaves no file behind.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from gbf.ingest import FFmpegError, VideoProbe, format_rate, parse_rate, probe
from gbf.ingest.tools import LOGLEVEL, StderrCollector, ffmpeg_path, media_arg

CODEC = "libx264"
PRESET = "medium"
DEFAULT_CRF = 23
"""x264's own default quality: lossy enough to test barcodes and lines the way broadcast
compression would."""
PIX_FMT = "yuv420p"
X264_THREADS = 8


class SynthVideoError(ValueError):
    """Frames cannot be written, or the written file is not what was asked for.
    ``problems`` lists every problem found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


def _frame_problems(frames: Sequence[np.ndarray]) -> list[str]:
    problems: list[str] = []
    if len(frames) == 0:
        return ["no frames given"]
    first = frames[0]
    if not isinstance(first, np.ndarray) or first.ndim != 3 or first.shape[-1] != 3:
        shape = getattr(first, "shape", None)
        return [f"frame 0 must be an (height, width, 3) array, got shape {shape}"]
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        problems.append(f"{PIX_FMT} needs an even width and height, got {width}x{height}")
    for index, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray):
            problems.append(f"frame {index} is a {type(frame).__name__}, not a numpy array")
            continue
        if frame.dtype != np.uint8:
            problems.append(f"frame {index} has dtype {frame.dtype}, expected uint8")
        if frame.shape != first.shape:
            problems.append(f"frame {index} has shape {frame.shape}, frame 0 has {first.shape}")
    return problems


def encoder_args(width: int, height: int, rate: Fraction, crf: int, destination: Path) -> list[str]:
    """The ffmpeg command line (a list, never a shell string) that writes ``destination``."""
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
        "-preset",
        PRESET,
        "-crf",
        str(crf),
        "-pix_fmt",
        PIX_FMT,
        "-threads",
        str(X264_THREADS),
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-f",
        "mp4",
        media_arg(destination),
    ]


def _encode(frames: Sequence[np.ndarray], args: list[str], what: str) -> None:
    proc = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    stdin = proc.stdin
    if stdin is None or proc.stderr is None:  # pragma: no cover - Popen guarantees pipes
        proc.kill()
        raise FFmpegError(what, None, "could not open pipes to ffmpeg")
    collector = StderrCollector(proc.stderr)
    write_error: OSError | None = None
    try:
        for frame in frames:
            try:
                stdin.write(np.ascontiguousarray(frame).data)
            except OSError as exc:  # ffmpeg exited early; its stderr says why
                write_error = exc
                break
    except BaseException:
        proc.kill()
        proc.wait(timeout=60)
        collector.text(timeout=10)
        raise
    finally:
        try:
            stdin.close()
        except OSError:
            pass  # already broken; reported through write_error or the exit code
    returncode = proc.wait()
    stderr = collector.text()
    if write_error is not None or returncode != 0 or stderr.strip():
        detail = (
            stderr if write_error is None else f"writing frames failed: {write_error}\n{stderr}"
        )
        raise FFmpegError(what, returncode, detail)


def write_frames_mp4(
    frames: Sequence[np.ndarray],
    dst: str | Path,
    *,
    fps: str | Fraction = "25/1",
    crf: int = DEFAULT_CRF,
    overwrite: bool = False,
) -> VideoProbe:
    """Encode RGB frames ((H, W, 3) uint8, all the same size, H and W even) to ``dst``.

    ``dst`` must end in ``.mp4``; ``fps`` is an exact rational rate; ``crf`` is x264's
    quality (0-51, lower is better). Returns the probe of the written file. Raises
    SynthVideoError listing every problem with the arguments or with the written file,
    FileExistsError when ``dst`` exists and ``overwrite`` is False, and
    gbf.ingest.FFmpegError when ffmpeg fails.
    """
    destination = Path(dst)
    what = f"write {len(frames)} frame(s) to {destination.name}"
    problems = _frame_problems(frames)
    if destination.suffix.lower() != ".mp4":
        problems.append(f"destination must end in .mp4, got {destination.name!r}")
    if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
        problems.append(f"crf must be an integer in 0..51, got {crf!r}")
    rate = Fraction(25)
    try:
        rate = parse_rate(fps)
    except ValueError as exc:
        problems.append(str(exc))
    if problems:
        raise SynthVideoError(what, problems)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass overwrite=True to replace it")
    height, width = frames[0].shape[:2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".partial")
    try:
        _encode(frames, encoder_args(width, height, rate, crf, tmp), what)
        facts = probe(tmp)
        problems = []
        if facts.nb_frames_decoded != len(frames):
            problems.append(f"decodes to {facts.nb_frames_decoded} frames, wrote {len(frames)}")
        if not facts.is_cfr_at(rate):
            problems.append(
                f"not CFR at {format_rate(rate)}: r_frame_rate={facts.r_frame_rate}, "
                f"avg_frame_rate={facts.avg_frame_rate}, timestamps_cfr={facts.timestamps_cfr}"
            )
        if (facts.frame_width, facts.frame_height) != (width, height):
            problems.append(
                f"frames are {facts.frame_width}x{facts.frame_height}, wrote {width}x{height}"
            )
        if facts.codec_name != "h264" or facts.pix_fmt != PIX_FMT:
            problems.append(
                f"stream is {facts.codec_name}/{facts.pix_fmt}, expected h264/{PIX_FMT}"
            )
        if problems:
            raise SynthVideoError(what, problems)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)
    return replace(facts, file_name=destination.name)
