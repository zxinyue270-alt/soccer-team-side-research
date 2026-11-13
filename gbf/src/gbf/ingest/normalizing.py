"""Normalise a source video to constant frame rate (plan 3.3 step 1, gate G5).

One recipe, applied to every input:

    ffmpeg -i SRC -map 0:v:0 -an -sn -dn -map_metadata -1 -map_chapters -1
           -vf setpts=PTS-STARTPTS,fps=fps=R:round=near:eof_action=round
           -fps_mode cfr -r R -c:v libx264 -preset medium -crf 12 -pix_fmt yuv420p
           -threads 8 [-g 1] -fflags +bitexact -flags:v +bitexact -f mp4 DST

Why the ``fps`` filter in front of ``-fps_mode cfr``: ffmpeg's CFR output mode on its own
tolerates about one frame of drift before it duplicates or drops, and it reads per-frame
durations that MP4 derives from decode order. Measured with ffmpeg 8.1.1: one second of
30 fps video became 27 frames at 25 fps instead of 25, and one of two 25 fps test clips with
three missing frames became 31 frames instead of 30, with the content shifted by two frames
after the gap (the other clip came out right). The filter
chain (see :func:`gbf.ingest.frames.cfr_filter`) only uses presentation timestamps, so the
output count is the exact value :func:`gbf.ingest.probing.expected_cfr_frame_count` predicts,
and every gap is filled with the frame that was on screen.

The output is verified with ffprobe before it is moved into place; any mismatch raises
:class:`gbf.ingest.errors.NormalizationError` and leaves no output behind. x264's thread
count is fixed (x264 output depends on it), so identical inputs give identical bytes on the
same machine and the same ffmpeg build. Audio, subtitles, data streams, chapters and
container metadata (which can include a phone's GPS position) are dropped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from gbf.ingest.errors import NormalizationError
from gbf.ingest.frames import cfr_filter
from gbf.ingest.probing import VideoProbe, expected_cfr_frame_count, keyframe_flags, probe
from gbf.ingest.rates import format_rate, parse_rate
from gbf.ingest.tools import LOGLEVEL, ffmpeg_path, media_arg, run_tool
from gbf.util.hashing import sha256_file

#: Encoder settings of the normalised file. Changing any of them changes every normalised
#: file and therefore every downstream hash; record the change in an ADR.
CODEC = "libx264"
PRESET = "medium"
CRF = 12
PIX_FMT = "yuv420p"
X264_THREADS = 8
OUTPUT_SUFFIX = ".mp4"


@dataclass(frozen=True)
class NormalizeResult:
    """What :func:`normalize` did: input facts, verified output facts and the settings."""

    source: VideoProbe
    output: VideoProbe
    expected_frames: int
    settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "output": self.output.to_dict(),
            "expected_frames": self.expected_frames,
            "settings": dict(self.settings),
        }


def normalization_settings(fps: str | Fraction = "25/1", all_intra: bool = False) -> dict[str, Any]:
    """The recipe :func:`normalize` applies, as recorded in ``video.json``."""
    rate = parse_rate(fps)
    return {
        "filter": cfr_filter(rate),
        "fps_mode": "cfr",
        "rate": format_rate(rate),
        "codec": CODEC,
        "preset": PRESET,
        "crf": CRF,
        "pix_fmt": PIX_FMT,
        "threads": X264_THREADS,
        "all_intra": bool(all_intra),
    }


def _ffmpeg_args(src: Path, tmp: Path, rate: Fraction, all_intra: bool) -> list[str]:
    args = [
        ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        LOGLEVEL,
        "-y",
        "-i",
        media_arg(src),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-vf",
        cfr_filter(rate),
        "-fps_mode",
        "cfr",
        "-r",
        format_rate(rate),
        "-c:v",
        CODEC,
        "-preset",
        PRESET,
        "-crf",
        str(CRF),
        "-pix_fmt",
        PIX_FMT,
        "-threads",
        str(X264_THREADS),
    ]
    if all_intra:
        args += ["-g", "1"]
    args += ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-f", "mp4", media_arg(tmp)]
    return args


def _output_problems(
    output: VideoProbe,
    source: VideoProbe,
    expected: int,
    rate: Fraction,
    all_intra: bool,
    keyframes: list[bool] | None,
) -> list[str]:
    problems = []
    if output.nb_frames_decoded != expected:
        problems.append(
            f"decoded frame count is {output.nb_frames_decoded}, expected {expected} "
            f"(CFR count of the source timeline at {format_rate(rate)} fps)"
        )
    if output.nb_frames_container is not None and output.nb_frames_container != expected:
        problems.append(
            f"container frame count is {output.nb_frames_container}, expected {expected}"
        )
    if not output.is_cfr_at(rate):
        problems.append(
            f"output is not CFR at {format_rate(rate)}: r_frame_rate={output.r_frame_rate}, "
            f"avg_frame_rate={output.avg_frame_rate}, timestamps_cfr={output.timestamps_cfr}"
        )
    if output.first_pts != 0:
        problems.append(f"first frame timestamp is {output.first_pts}, expected 0")
    if output.codec_name != "h264" or output.pix_fmt != PIX_FMT:
        problems.append(f"output is {output.codec_name}/{output.pix_fmt}, expected h264/{PIX_FMT}")
    if output.rotation_deg != 0:
        problems.append(f"output still carries a display rotation of {output.rotation_deg} deg")
    source_size = (source.frame_width, source.frame_height)
    if (output.frame_width, output.frame_height) != source_size:
        problems.append(
            f"output frames are {output.frame_width}x{output.frame_height}, "
            f"source frames are {source_size[0]}x{source_size[1]}"
        )
    if all_intra and keyframes is not None:
        not_key = sum(1 for flag in keyframes if not flag)
        if not_key or len(keyframes) != expected:
            problems.append(
                f"all-intra output has {not_key} non-keyframe packet(s) "
                f"among {len(keyframes)} packets"
            )
    return problems


def normalize(
    src: str | Path,
    dst: str | Path,
    fps: str | Fraction = "25/1",
    all_intra: bool = False,
    *,
    overwrite: bool = False,
    source_probe: VideoProbe | None = None,
) -> NormalizeResult:
    """Write a CFR H.264 copy of ``src`` at ``fps`` to ``dst`` (``.mp4``) and verify it.

    ``all_intra=True`` makes every frame a keyframe (``-g 1``), for event windows that are
    accessed randomly. ``source_probe`` skips probing ``src`` again; it must describe the
    same bytes. Raises ``FileExistsError`` when ``dst`` exists (unless ``overwrite``),
    :class:`NormalizationError` listing every check the output failed, and
    :class:`gbf.ingest.errors.FFmpegError` when ffmpeg fails or reports an error.
    """
    src, dst = Path(src), Path(dst)
    rate = parse_rate(fps)
    what = f"normalize {src.name} -> {dst.name}"
    problems = []
    if not src.is_file():
        problems.append(f"source is not a file: {src}")
    if dst.suffix.lower() != OUTPUT_SUFFIX:
        problems.append(f"destination must end in {OUTPUT_SUFFIX}, got {dst.name!r}")
    if src.is_file() and dst.exists() and dst.resolve() == src.resolve():
        problems.append("destination is the source file")
    if problems:
        raise NormalizationError(what, problems)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} exists; pass overwrite=True to replace it")

    if source_probe is None:
        source = probe(src)
    else:
        source = source_probe
        if source.file_bytes != src.stat().st_size or source.sha256 != sha256_file(src):
            raise NormalizationError(what, ["source_probe does not describe the bytes of src"])
    expected = expected_cfr_frame_count(source, rate)
    if expected < 1:
        raise NormalizationError(what, [f"the source timeline holds no frame at {rate} fps"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    try:
        run_tool(_ffmpeg_args(src, tmp, rate, all_intra), what=what)
        output = probe(tmp)
        keyframes = keyframe_flags(tmp) if all_intra else None
        problems = _output_problems(output, source, expected, rate, all_intra, keyframes)
        if problems:
            raise NormalizationError(what, problems)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return NormalizeResult(
        source=source,
        output=replace(output, file_name=dst.name),
        expected_frames=expected,
        settings=normalization_settings(rate, all_intra),
    )
