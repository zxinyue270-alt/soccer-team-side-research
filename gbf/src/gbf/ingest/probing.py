"""Probing videos with ffprobe.

:func:`probe` decodes the whole first video stream (``-count_frames``) so the frame count it
reports is the number of frames a decoder really produces, not the number a container
header claims. It also reads every frame's best-effort timestamp so that the length of the
decoded timeline, and from it the expected constant-frame-rate (CFR) frame count, is exact.

Units and conventions:

- ``*_pts`` values are integers in units of the stream ``time_base`` (a rational, seconds per
  tick). ``first_pts`` is the presentation time of the first decoded frame; ``end_pts`` is
  the end of the last decoded frame (its timestamp plus its duration).
- Frame rates are rational strings ``"num/den"``; ffprobe's ``"0/0"`` (unknown) becomes ``None``.
- ``width`` and ``height`` are the stored (coded, cropped) frame size in pixels;
  ``frame_width`` and ``frame_height`` are the size of decoded frames after the display
  rotation that ffmpeg applies by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from gbf.ingest.errors import ProbeError
from gbf.ingest.rates import format_rate, parse_optional_rate, parse_rate, round_half_away
from gbf.ingest.tools import ffprobe_path, media_arg, run_tool
from gbf.util.hashing import sha256_file

_STREAM_ENTRIES = (
    "stream=index,codec_type,codec_name,width,height,pix_fmt,field_order,sample_aspect_ratio,"
    "r_frame_rate,avg_frame_rate,time_base,nb_frames,nb_read_frames,duration"
    ":stream_disposition=attached_pic:stream_side_data"
)


@dataclass(frozen=True)
class StreamGeometry:
    """Size and orientation of the first video stream (no decoding needed)."""

    stream_index: int
    width: int
    height: int
    rotation_deg: int

    @property
    def frame_width(self) -> int:
        """Width in pixels of decoded frames after display rotation."""
        return self.height if _swaps_axes(self.rotation_deg) else self.width

    @property
    def frame_height(self) -> int:
        """Height in pixels of decoded frames after display rotation."""
        return self.width if _swaps_axes(self.rotation_deg) else self.height


@dataclass(frozen=True)
class VideoProbe:
    """Facts about one video file and its first video stream.

    ``file_name`` is kept for messages only; :meth:`to_dict` never includes a name or a path.
    """

    file_name: str
    sha256: str
    file_bytes: int
    container: str
    stream_index: int
    codec_name: str
    pix_fmt: str | None
    field_order: str | None
    width: int
    height: int
    rotation_deg: int
    sample_aspect_ratio: str | None
    r_frame_rate: str | None
    avg_frame_rate: str | None
    time_base: str
    nb_frames_container: int | None
    nb_frames_decoded: int
    duration_s: float | None
    first_pts: int | None
    end_pts: int | None
    timestamps_cfr: bool
    ffprobe_version: str

    @property
    def frame_width(self) -> int:
        """Width in pixels of decoded frames after display rotation."""
        return self.height if _swaps_axes(self.rotation_deg) else self.width

    @property
    def frame_height(self) -> int:
        """Height in pixels of decoded frames after display rotation."""
        return self.width if _swaps_axes(self.rotation_deg) else self.height

    def is_cfr_at(self, fps: str | Fraction) -> bool:
        """True when the stream is constant-frame-rate at exactly ``fps``.

        Requires ``r_frame_rate == avg_frame_rate == fps`` and every decoded frame lasting the
        same number of ticks, which is what :func:`gbf.ingest.normalizing.normalize` produces.
        """
        rate = parse_rate(fps)
        return (
            self.timestamps_cfr
            and parse_optional_rate(self.r_frame_rate) == rate
            and parse_optional_rate(self.avg_frame_rate) == rate
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON form used in ``video.json`` (schema ``video``, ``$defs/probe``)."""
        return {
            "sha256": self.sha256,
            "bytes": self.file_bytes,
            "container": self.container,
            "stream_index": self.stream_index,
            "codec_name": self.codec_name,
            "pix_fmt": self.pix_fmt,
            "field_order": self.field_order,
            "width": self.width,
            "height": self.height,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "rotation_deg": self.rotation_deg,
            "sample_aspect_ratio": self.sample_aspect_ratio,
            "r_frame_rate": self.r_frame_rate,
            "avg_frame_rate": self.avg_frame_rate,
            "time_base": self.time_base,
            "nb_frames_container": self.nb_frames_container,
            "nb_frames_decoded": self.nb_frames_decoded,
            "duration_s": self.duration_s,
            "first_pts": self.first_pts,
            "end_pts": self.end_pts,
            "timestamps_cfr": self.timestamps_cfr,
            "ffprobe_version": self.ffprobe_version,
        }


def _swaps_axes(rotation_deg: int) -> bool:
    return rotation_deg % 180 == 90


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")


def _run_json(args: list[str], what: str) -> dict[str, Any]:
    proc = run_tool(args, what=what)
    try:
        data = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(what, [f"ffprobe did not return valid JSON: {exc}"]) from exc
    if not isinstance(data, dict):
        raise ProbeError(what, ["ffprobe did not return a JSON object"])
    return data


def _optional_int(value: Any) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rotation(stream: dict[str, Any]) -> int:
    for entry in stream.get("side_data_list") or []:
        if isinstance(entry, dict) and "rotation" in entry:
            value = _optional_float(entry["rotation"])
            if value is not None:
                return int(round(value))
    return 0


def _video_stream(data: dict[str, Any], what: str) -> dict[str, Any]:
    streams = data.get("streams") or []
    if not streams or streams[0].get("codec_type") != "video":
        raise ProbeError(what, ["no video stream"])
    stream = streams[0]
    problems = []
    if (stream.get("disposition") or {}).get("attached_pic") == 1:
        problems.append(
            "the first video stream is an attached picture (cover art); remux without it"
        )
    for key in ("width", "height"):
        value = _optional_int(stream.get(key))
        if value is None or value <= 0:
            problems.append(f"video stream has no valid {key} ({stream.get(key)!r})")
    if problems:
        raise ProbeError(what, problems)
    return stream


def stream_geometry(path: str | Path) -> StreamGeometry:
    """Size and display rotation of the first video stream, without decoding.

    Raises :class:`ProbeError` when there is no video stream or when the display rotation is
    not a multiple of 90 degrees (ffmpeg would rotate the picture by an arbitrary angle).
    """
    path = Path(path)
    _require_file(path)
    what = f"probe geometry of {path.name}"
    args = [
        ffprobe_path(),
        "-v",
        "error",
        "-hide_banner",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=index,codec_type,width,height:stream_disposition=attached_pic:stream_side_data",
        "-of",
        "json=compact=1",
        media_arg(path),
    ]
    stream = _video_stream(_run_json(args, what), what)
    rotation = _rotation(stream)
    if rotation % 90 != 0:
        raise ProbeError(
            what, [f"display rotation of {rotation} degrees is not a multiple of 90 degrees"]
        )
    return StreamGeometry(
        stream_index=int(stream.get("index", 0)),
        width=int(stream["width"]),
        height=int(stream["height"]),
        rotation_deg=rotation,
    )


def _last_frame_ticks(
    pts: list[int], last_duration: int | None, rate: Fraction | None, time_base: Fraction
) -> int:
    """Duration of the last frame in ticks, chosen the way ffmpeg's decoder estimates it.

    Mirrors ``video_duration_estimate`` in FFmpeg's ``fftools/ffmpeg_dec.c`` for containers
    with timestamps: the frame's own duration unless it is the made-up 1-tick value, then the
    gap to the previous frame, then one frame at the nominal rate.
    """
    ts_diff = pts[-1] - pts[-2] if len(pts) > 1 else None
    made_up = last_duration == 1 and ts_diff is not None and ts_diff > 2
    if last_duration is not None and last_duration > 0 and not made_up:
        return last_duration
    if ts_diff is not None and ts_diff > 0:
        return ts_diff
    if rate is not None:
        ticks = round_half_away(1 / (rate * time_base))
        if ticks > 0:
            return ticks
    return 1


def _timeline(
    frames: list[Any], rate: Fraction | None, time_base: Fraction
) -> tuple[int | None, int | None, bool]:
    """(first_pts, end_pts, timestamps_cfr); the timeline is unknown (None) when any frame
    lacks a timestamp or timestamps do not strictly increase."""
    pts: list[int] = []
    for frame in frames:
        value = (
            _optional_int(frame.get("best_effort_timestamp")) if isinstance(frame, dict) else None
        )
        if value is None:
            return None, None, False
        pts.append(value)
    if not pts:
        return None, None, False
    deltas = {b - a for a, b in zip(pts, pts[1:], strict=False)}
    if any(delta <= 0 for delta in deltas):
        return None, None, False
    last = frames[-1]
    last_ticks = _last_frame_ticks(pts, _optional_int(last.get("duration")), rate, time_base)
    cfr = len(deltas) <= 1 and (not deltas or deltas == {last_ticks})
    return pts[0], pts[-1] + last_ticks, cfr


def probe(path: str | Path) -> VideoProbe:
    """Probe ``path`` with ffprobe and hash its bytes.

    Decodes the entire first video stream, so this takes about as long as a full decode.
    Raises :class:`ProbeError` when there is no usable video stream, :class:`FFmpegError` when
    ffprobe fails or reports errors, and ``FileNotFoundError`` when ``path`` is not a file.
    """
    path = Path(path)
    _require_file(path)
    what = f"probe {path.name}"
    args = [
        ffprobe_path(),
        "-v",
        "error",
        "-hide_banner",
        "-show_program_version",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        _STREAM_ENTRIES + ":format=format_name,duration:frame=best_effort_timestamp,duration",
        "-of",
        "json=compact=1",
        media_arg(path),
    ]
    data = _run_json(args, what)
    stream = _video_stream(data, what)
    problems: list[str] = []

    try:
        time_base = parse_rate(stream.get("time_base", ""))
    except ValueError as exc:
        problems.append(f"unusable time_base: {exc}")
        time_base = Fraction(1)
    decoded = _optional_int(stream.get("nb_read_frames"))
    frames = data.get("frames") or []
    if decoded is None:
        problems.append("ffprobe did not report a decoded frame count (nb_read_frames)")
    elif decoded != len(frames):
        problems.append(f"ffprobe counted {decoded} frames but listed {len(frames)}")
    version = (data.get("program_version") or {}).get("version")
    if not version:
        problems.append("ffprobe did not report its version")
    if problems:
        raise ProbeError(what, problems)

    r_rate = parse_optional_rate(stream.get("r_frame_rate"))
    avg_rate = parse_optional_rate(stream.get("avg_frame_rate"))
    first_pts, end_pts, cfr = _timeline(frames, r_rate or avg_rate, time_base)
    duration = _optional_float(stream.get("duration"))
    if duration is None:
        # Matroska stores no per-stream duration; fall back to the container's.
        duration = _optional_float((data.get("format") or {}).get("duration"))
    sar = stream.get("sample_aspect_ratio")
    return VideoProbe(
        file_name=path.name,
        sha256=sha256_file(path),
        file_bytes=path.stat().st_size,
        container=str((data.get("format") or {}).get("format_name") or "unknown"),
        stream_index=int(stream.get("index", 0)),
        codec_name=str(stream.get("codec_name") or "unknown"),
        pix_fmt=stream.get("pix_fmt") or None,
        field_order=stream.get("field_order") or None,
        width=int(stream["width"]),
        height=int(stream["height"]),
        rotation_deg=_rotation(stream),
        sample_aspect_ratio=sar if isinstance(sar, str) and sar not in {"0:1", "N/A"} else None,
        r_frame_rate=format_rate(r_rate) if r_rate is not None else None,
        avg_frame_rate=format_rate(avg_rate) if avg_rate is not None else None,
        time_base=format_rate(time_base),
        nb_frames_container=_optional_int(stream.get("nb_frames")),
        nb_frames_decoded=int(decoded) if decoded is not None else 0,
        duration_s=duration,
        first_pts=first_pts,
        end_pts=end_pts,
        timestamps_cfr=cfr,
        ffprobe_version=str(version),
    )


def expected_cfr_frame_count(source: VideoProbe, fps: str | Fraction) -> int:
    """Number of frames a CFR conversion of ``source`` at ``fps`` must produce.

    ``round_half_away((end_pts - first_pts) * time_base * fps)``: the decoded timeline,
    measured from the first frame to the end of the last frame, cut into frames of
    ``1/fps`` seconds. This is exactly what the ``setpts=PTS-STARTPTS,fps=...`` chain used by
    :func:`gbf.ingest.normalizing.normalize` and :func:`gbf.ingest.frames.iter_frames` emits.
    """
    rate = parse_rate(fps)
    if source.first_pts is None or source.end_pts is None:
        reason = (
            "the video has no decoded frames"
            if source.nb_frames_decoded == 0
            else "some frames have no timestamp or the timestamps do not increase"
        )
        raise ProbeError(
            f"expected CFR frame count of {source.file_name}",
            [f"the decoded timeline is unknown: {reason}"],
        )
    span = Fraction(source.end_pts - source.first_pts) * parse_rate(source.time_base)
    return round_half_away(span * rate)


def keyframe_flags(path: str | Path) -> list[bool]:
    """For every packet of the first video stream, whether it is a keyframe (no decoding)."""
    path = Path(path)
    _require_file(path)
    args = [
        ffprobe_path(),
        "-v",
        "error",
        "-hide_banner",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=flags",
        "-of",
        "csv=p=0",
        media_arg(path),
    ]
    proc = run_tool(args, what=f"list packets of {path.name}")
    lines = [line.strip() for line in proc.stdout.decode("utf-8").splitlines() if line.strip()]
    return [line.startswith("K") for line in lines]
