"""Tests for gbf.ingest: probing, CFR normalisation, the single frame reader, cuts, windows,
video.json and the CLI.

Every video is generated here, 128x72 and 20-30 frames. Each frame carries its own index as
an 8-bit barcode of 16x16 black/white blocks in the top row (large flat blocks survive lossy
compression), over a flat background whose colour marks the shot and a moving square.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Iterable
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import gbf.ingest.frames as frames_module
import gbf.ingest.normalizing as normalize_module
from gbf import contracts
from gbf.__main__ import main as gbf_main
from gbf.ingest import (
    DEFAULT_CUT_THRESHOLD,
    FFmpegError,
    FrameRangeError,
    NormalizationError,
    ProbeError,
    RecordError,
    WindowError,
    build_video_record,
    cut_detection_settings,
    cut_scores,
    detect_cuts,
    expected_cfr_frame_count,
    extract_window,
    frame_count,
    iter_frames,
    keyframe_flags,
    normalize,
    pixel_sha256,
    probe,
    read_frames,
    stream_geometry,
    tool_versions,
    write_video_json,
)
from gbf.ingest.cli import main as ingest_main
from gbf.ingest.rates import format_rate, parse_rate, round_half_away
from gbf.util.hashing import sha256_file
from gbf.util.io import read_json

W, H = 128, 72
BITS = 8
X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
SHOT_COLOURS = [(40, 140, 50), (200, 60, 40)]
DROPPED = (10, 11, 20)  # frames removed from the 30-frame clip to make it VFR


# --------------------------------------------------------------------------- media helpers


def make_frame(index: int, shot: int = 0) -> np.ndarray:
    frame = np.empty((H, W, 3), dtype=np.uint8)
    frame[:] = SHOT_COLOURS[shot]
    x = (index * 3) % (W - 16)
    frame[40:56, x : x + 16] = (230, 230, 40)
    for bit in range(BITS):
        frame[0:16, bit * 16 : (bit + 1) * 16] = 255 if (index >> bit) & 1 else 0
    return frame


def code_of(frame: np.ndarray) -> int:
    value = 0
    for bit in range(BITS):
        if frame[4:12, bit * 16 + 4 : bit * 16 + 12].mean() > 127.5:
            value |= 1 << bit
    return value


def run_ffmpeg(ffmpeg_bin: str, *args: str, data: bytes | None = None) -> None:
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-nostats", "-loglevel", "error", "-y", *args],
        input=data,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr.decode("utf-8", "replace"))


def encode(
    ffmpeg_bin: str,
    dst: Path,
    frames: Iterable[np.ndarray],
    *,
    rate: str = "25",
    codec: list[str] = X264,
) -> Path:
    data = b"".join(np.ascontiguousarray(frame).tobytes() for frame in frames)
    run_ffmpeg(
        ffmpeg_bin,
        *("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", rate),
        *("-i", "pipe:0", *codec, str(dst)),
        data=data,
    )
    return dst


def make_clip(
    ffmpeg_bin: str, dst: Path, n: int = 20, shot: Callable[[int], int] = lambda i: 0
) -> Path:
    return encode(ffmpeg_bin, dst, [make_frame(i, shot(i)) for i in range(n)])


def make_vfr(ffmpeg_bin: str, folder: Path) -> Path:
    """30 frame slots at 25 fps with frames 10, 11 and 20 missing (27 frames, 1.2 s)."""
    full = make_clip(ffmpeg_bin, folder / "完整 片段.mp4", n=30)
    vfr = folder / "可变 帧率.mp4"
    drop = "+".join(f"eq(n,{index})" for index in DROPPED)
    run_ffmpeg(
        ffmpeg_bin,
        *("-i", str(full), "-vf", f"select='not({drop})'", "-fps_mode", "passthrough"),
        *X264,
        str(vfr),
    )
    return vfr


def codes(path: Path, *args: int, **kwargs: object) -> list[int]:
    return [code_of(frame) for _, frame in iter_frames(path, *args, **kwargs)]  # type: ignore[arg-type]


def no_local_paths(text: str, *roots: Path) -> list[str]:
    found = [str(root) for root in roots if str(root) in text or root.as_posix() in text]
    found += re.findall(r"[A-Za-z]:[\\/]", text)
    if "\\" in text:
        found.append("backslash")
    return found


@pytest.fixture(autouse=True)
def _tools(ffmpeg_bin: str, ffprobe_bin: str) -> None:
    """Every test here needs ffmpeg and ffprobe; fail loudly (never skip) without them."""


# --------------------------------------------------------------------------- rates


def test_rates_are_exact_and_round_like_ffmpeg() -> None:
    assert parse_rate("25/1") == 25
    assert parse_rate("30000/1001") == Fraction(30000, 1001)
    assert parse_rate("25") == 25
    assert format_rate(Fraction(50, 2)) == "25/1"
    for bad in ["29.97", "0/1", "25/0", "-25/1", "abc", ""]:
        with pytest.raises(ValueError, match="rate"):
            parse_rate(bad)
    # FFmpeg's fps filter rounds halves away from zero; Python's round() goes to even.
    assert round_half_away(Fraction(61, 2)) == 31
    assert round(Fraction(61, 2)) == 30
    assert round_half_away(Fraction(59, 2)) == 30
    assert round_half_away(Fraction(-5, 2)) == -3
    assert round_half_away(Fraction(30, 1)) == 30


# --------------------------------------------------------------------------- probe


def test_probe_reports_exact_facts(
    unicode_tmp_path: Path, ffmpeg_bin: str, ffprobe_bin: str
) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "源 视频.mp4", n=20)
    facts = probe(clip)
    assert (facts.width, facts.height) == (W, H)
    assert (facts.frame_width, facts.frame_height) == (W, H)
    assert facts.r_frame_rate == facts.avg_frame_rate == "25/1"
    assert facts.nb_frames_container == 20
    assert facts.nb_frames_decoded == 20
    assert facts.codec_name == "h264"
    assert facts.pix_fmt == "yuv420p"
    assert facts.field_order == "progressive"
    assert facts.duration_s == pytest.approx(0.8)
    assert facts.sha256 == hashlib.sha256(clip.read_bytes()).hexdigest()
    assert facts.file_bytes == clip.stat().st_size
    assert facts.rotation_deg == 0
    assert facts.time_base == "1/12800"
    assert (facts.first_pts, facts.end_pts) == (0, 20 * 512)
    assert facts.timestamps_cfr
    assert facts.is_cfr_at("25/1")
    assert not facts.is_cfr_at("50/1")
    version_line = subprocess.run(
        [ffprobe_bin, "-version"], capture_output=True, check=True
    ).stdout.decode("utf-8")
    assert facts.ffprobe_version == version_line.split()[2]
    serialised = json.dumps(facts.to_dict(), ensure_ascii=False)
    assert "源 视频" not in serialised
    assert no_local_paths(serialised, unicode_tmp_path) == []

    # Matroska stores no frame count and no stream duration: decoded count still exact.
    mkv = unicode_tmp_path / "源 视频.mkv"
    run_ffmpeg(ffmpeg_bin, "-i", str(clip), "-c", "copy", str(mkv))
    mkv_facts = probe(mkv)
    assert mkv_facts.nb_frames_container is None
    assert mkv_facts.nb_frames_decoded == 20
    assert mkv_facts.duration_s == pytest.approx(0.8)
    assert mkv_facts.container.startswith("matroska")


def test_probe_raises_without_video_stream(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    audio = unicode_tmp_path / "只有 音频.m4a"
    run_ffmpeg(ffmpeg_bin, "-f", "lavfi", "-i", "sine=frequency=440:duration=0.2", str(audio))
    with pytest.raises(ProbeError, match="no video stream"):
        probe(audio)
    with pytest.raises(ProbeError, match="no video stream"):
        stream_geometry(audio)

    not_media = unicode_tmp_path / "不是 视频.mp4"
    not_media.write_bytes(b"this is not a video\n" * 10)
    with pytest.raises(FFmpegError) as caught:
        probe(not_media)
    assert "Invalid data" in str(caught.value)  # ffprobe's own message is surfaced
    with pytest.raises(FileNotFoundError):
        probe(unicode_tmp_path / "missing.mp4")


def test_probe_reads_display_rotation(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=5)
    odd = unicode_tmp_path / "rotated 45.mp4"
    run_ffmpeg(ffmpeg_bin, "-display_rotation", "45", "-i", str(clip), "-c", "copy", str(odd))
    assert probe(odd).rotation_deg == 45
    with pytest.raises(ProbeError, match="not a multiple of 90"):
        stream_geometry(odd)


# --------------------------------------------------------------------------- normalize


def test_normalize_vfr_source_gives_expected_cfr_count(
    unicode_tmp_path: Path, ffmpeg_bin: str
) -> None:
    vfr = make_vfr(ffmpeg_bin, unicode_tmp_path)
    source = probe(vfr)
    assert source.nb_frames_decoded == 27
    assert not source.timestamps_cfr
    assert source.avg_frame_rate != "25/1"
    assert expected_cfr_frame_count(source, "25/1") == 30

    # The lesson: passing the timestamps through keeps 27 frames for 1.2 s of video.
    naive = unicode_tmp_path / "直接 转码.mp4"
    run_ffmpeg(ffmpeg_bin, "-i", str(vfr), "-fps_mode", "passthrough", *X264, str(naive))
    assert probe(naive).nb_frames_decoded == 27

    out = unicode_tmp_path / "输出" / "规范 化.mp4"
    result = normalize(vfr, out)
    assert result.expected_frames == 30
    assert result.output.nb_frames_decoded == 30
    assert result.output.nb_frames_container == 30
    assert result.output.r_frame_rate == result.output.avg_frame_rate == "25/1"
    assert result.output.timestamps_cfr
    assert result.output.first_pts == 0
    assert result.output.sha256 == sha256_file(out)
    assert result.source.sha256 == source.sha256
    assert not out.with_name(out.name + ".partial").exists()
    # Every gap holds the frame that was on screen: canonical frame k = source at k/25 s.
    assert codes(out) == [*range(10), 9, 9, *range(12, 20), 19, *range(21, 30)]
    # The reader maps the raw source onto the same grid on the fly.
    assert codes(vfr, fps="25/1") == codes(out)
    assert codes(vfr) == [i for i in range(30) if i not in DROPPED]


def test_normalize_resamples_other_rates_exactly(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    clip30 = encode(
        ffmpeg_bin, unicode_tmp_path / "三十 帧.mp4", [make_frame(i) for i in range(30)], rate="30"
    )
    assert probe(clip30).r_frame_rate == "30/1"
    out = unicode_tmp_path / "25.mp4"
    result = normalize(clip30, out, fps="25/1")
    # 1.0 s at 25 fps. ffmpeg's -fps_mode cfr alone gave 27 frames for 30 such frames
    # (ffmpeg 8.1.1, drift tolerance); the fps filter gives the exact 25.
    assert result.output.nb_frames_decoded == result.expected_frames == 25
    found = codes(out)
    assert found[0] == 0
    assert found[-1] == 29
    assert all(b > a for a, b in zip(found, found[1:], strict=False))


def test_normalize_rounds_exact_halves_away_from_zero(
    unicode_tmp_path: Path, ffmpeg_bin: str
) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=30)
    late = unicode_tmp_path / "最后 一帧 晚到.mkv"
    run_ffmpeg(
        ffmpeg_bin,
        *("-i", str(clip), "-vf", "setpts='(N/25+0.02*eq(N,29))/TB'", "-fps_mode", "passthrough"),
        *("-enc_time_base", "1/1000", *X264, str(late)),
    )
    facts = probe(late)
    assert (facts.time_base, facts.first_pts, facts.end_pts) == ("1/1000", 0, 1220)
    # 1.22 s * 25 = 30.5 frames: FFmpeg rounds to 31, Python's round() would say 30.
    assert expected_cfr_frame_count(facts, "25/1") == 31
    assert frame_count(late, fps="25/1") == 31
    assert codes(late, fps="25/1") == [*range(29), 28, 29]
    result = normalize(late, unicode_tmp_path / "out.mp4")
    assert result.output.nb_frames_decoded == 31


def test_normalize_starts_at_the_first_video_frame(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=30)
    delayed = unicode_tmp_path / "视频 晚开始.mp4"
    run_ffmpeg(
        ffmpeg_bin,
        *("-itsoffset", "0.2", "-i", str(clip)),
        *("-f", "lavfi", "-i", "sine=frequency=440:duration=1.5"),
        *("-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(delayed)),
    )
    facts = probe(delayed)
    assert Fraction(facts.first_pts or 0) * Fraction(facts.time_base) == Fraction(1, 5)
    out = unicode_tmp_path / "out.mp4"
    result = normalize(delayed, out)
    # Without setpts=PTS-STARTPTS ffmpeg pads the 0.2 s audio lead with 5 copies of frame 0.
    assert result.output.nb_frames_decoded == 30
    assert codes(out)[:3] == [0, 1, 2]


def test_normalize_all_intra_and_determinism(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=20)
    intra_a = normalize(clip, unicode_tmp_path / "a.mp4", all_intra=True)
    intra_b = normalize(clip, unicode_tmp_path / "b.mp4", all_intra=True)
    assert keyframe_flags(unicode_tmp_path / "a.mp4") == [True] * 20
    assert intra_a.output.sha256 == intra_b.output.sha256
    assert intra_a.settings["all_intra"] is True
    long_gop = normalize(clip, unicode_tmp_path / "c.mp4")
    assert normalize(clip, unicode_tmp_path / "d.mp4").output.sha256 == long_gop.output.sha256
    flags = keyframe_flags(unicode_tmp_path / "c.mp4")
    assert len(flags) == 20
    assert flags[0]
    assert not all(flags)
    assert long_gop.settings["filter"] == (
        "setpts=PTS-STARTPTS,fps=fps=25/1:round=near:eof_action=round"
    )


def test_normalize_rejects_bad_requests(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=5)
    with pytest.raises(NormalizationError) as caught:
        normalize(unicode_tmp_path / "missing.mp4", unicode_tmp_path / "out.avi")
    assert len(caught.value.problems) == 2  # missing source and wrong suffix, both reported
    existing = unicode_tmp_path / "exists.mp4"
    existing.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        normalize(clip, existing)
    assert existing.read_bytes() == b"keep me"
    with pytest.raises(ValueError, match="rate"):
        normalize(clip, unicode_tmp_path / "out.mp4", fps="29.97")
    other = make_clip(ffmpeg_bin, unicode_tmp_path / "other.mp4", n=6)
    with pytest.raises(NormalizationError, match="does not describe the bytes of src"):
        normalize(clip, unicode_tmp_path / "out.mp4", source_probe=probe(other))
    assert not (unicode_tmp_path / "out.mp4").exists()


def test_normalize_raises_and_cleans_up_when_count_differs(
    unicode_tmp_path: Path, ffmpeg_bin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=20)
    real = normalize_module.expected_cfr_frame_count
    monkeypatch.setattr(
        normalize_module, "expected_cfr_frame_count", lambda facts, fps: real(facts, fps) + 1
    )
    out = unicode_tmp_path / "out.mp4"
    with pytest.raises(NormalizationError, match="decoded frame count is 20, expected 21"):
        normalize(clip, out)
    assert list(unicode_tmp_path.iterdir()) == [clip]  # neither output nor partial file left


# --------------------------------------------------------------------------- frame reader


def test_iter_frames_from_nonzero_start(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=20)
    full = list(iter_frames(clip))
    assert [index for index, _ in full] == list(range(20))
    assert [code_of(frame) for _, frame in full] == list(range(20))  # last frame included
    first = full[0][1]
    assert first.shape == (H, W, 3)
    assert first.dtype == np.uint8
    assert first.flags.c_contiguous
    assert first.flags.writeable

    part = list(iter_frames(clip, 5, 9))
    assert [index for index, _ in part] == [5, 6, 7, 8]
    assert [code_of(frame) for _, frame in part] == [5, 6, 7, 8]  # no duplicated first frame
    for index, frame in part:
        assert np.array_equal(frame, full[index][1])  # bit-identical to a full decode
    assert [index for index, _ in iter_frames(clip, 7)] == list(range(7, 20))
    assert [code_of(frame) for _, frame in iter_frames(clip, 19)] == [19]
    assert list(iter_frames(clip, 4, 4)) == []
    # A canonical CFR file is its own CFR grid: mapping it changes nothing.
    mapped = list(iter_frames(clip, 3, 8, fps="25/1"))
    assert all(np.array_equal(frame, full[index][1]) for index, frame in mapped)


def test_iter_frames_range_errors(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=20)
    with pytest.raises(FrameRangeError) as caught:
        iter_frames(clip, -1, 2.5)  # type: ignore[arg-type]
    assert len(caught.value.problems) == 2  # raised at the call, before any decoding
    with pytest.raises(FrameRangeError, match="must not be smaller"):
        iter_frames(clip, 5, 3)
    with pytest.raises(FrameRangeError, match="beyond the last frame"):
        list(iter_frames(clip, 25))
    seen: list[int] = []

    def consume() -> None:
        for index, _ in iter_frames(clip, 15, 25):
            seen.append(index)

    with pytest.raises(FrameRangeError, match="frames 20..24 do not exist"):
        consume()
    assert seen == [15, 16, 17, 18, 19]  # the existing frames came first, then the error


def test_iter_frames_stops_ffmpeg_when_consumer_stops_early(
    unicode_tmp_path: Path, ffmpeg_bin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=30)
    started: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        proc = real_popen(*args, **kwargs)  # type: ignore[call-overload]
        started.append(proc)
        return proc

    monkeypatch.setattr(frames_module.subprocess, "Popen", recording_popen)
    frames = iter_frames(clip)
    taken = [next(frames)[0] for _ in range(3)]
    assert taken == [0, 1, 2]
    decoders = [proc for proc in started if "rawvideo" in proc.args]
    # 30 frames do not fit in the pipe, so ffmpeg is still running, blocked on writing.
    assert len(decoders) == 1
    assert decoders[0].poll() is None
    frames.close()  # what a for-loop's break does once the generator is dropped
    assert decoders[0].poll() is not None  # stopped, not left decoding into a dead pipe
    clip.unlink()  # on Windows this fails while another process holds the file open


def test_reader_surfaces_ffmpeg_stderr(
    unicode_tmp_path: Path, ffmpeg_bin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=10)
    monkeypatch.setattr(frames_module, "cfr_filter", lambda fps: "nosuchfilter")
    with pytest.raises(FFmpegError) as caught:
        list(iter_frames(clip, fps="25/1"))
    assert "nosuchfilter" in str(caught.value)
    assert caught.value.returncode != 0


def test_reader_treats_decode_errors_as_errors(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    clip = encode(
        ffmpeg_bin,
        unicode_tmp_path / "clip.mkv",
        [make_frame(i) for i in range(60)],
        codec=[*X264, "-g", "10"],
    )
    data = bytearray(clip.read_bytes())
    # Flip bytes in three clusters past the first ~20 frames, which ffprobe decodes while
    # probing stream parameters; the reader's own decode then has to report the damage.
    for fraction in (0.55, 0.7, 0.85):
        position = int(len(data) * fraction)
        for offset in range(position, position + 64):
            data[offset] ^= 0x5A
    broken = unicode_tmp_path / "损坏.mkv"
    broken.write_bytes(bytes(data))
    stream_geometry(broken)  # the header and the first frames are intact
    with pytest.raises(FFmpegError) as caught:
        list(iter_frames(broken))
    message = str(caught.value)
    assert "ffmpeg stderr" in message
    assert message.split("--- ffmpeg stderr ---", 1)[1].strip()  # ffmpeg's own words follow


def test_read_frames_arbitrary_indices(
    unicode_tmp_path: Path, ffmpeg_bin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=20)
    full = {index: frame for index, frame in iter_frames(clip)}
    got = read_frames(clip, [17, 3, 3, 11, 0, 19])
    assert list(got) == [0, 3, 11, 17, 19]
    for index, frame in got.items():
        assert code_of(frame) == index
        assert np.array_equal(frame, full[index])
    assert read_frames(clip, []) == {}

    # Index lists too long for one select expression are split into several passes.
    monkeypatch.setattr(frames_module, "MAX_SELECT_EXPRESSION", 20)
    split = read_frames(clip, [1, 4, 5, 6, 9, 13, 18])
    assert {index: code_of(frame) for index, frame in split.items()} == {
        i: i for i in [1, 4, 5, 6, 9, 13, 18]
    }

    # ffmpeg refuses select expressions of more than 100 terms, however short.
    monkeypatch.undo()
    clip_long = make_clip(ffmpeg_bin, unicode_tmp_path / "long.mp4", n=240)
    every_other = list(range(0, 240, 2))  # 120 separate frames, 120 terms
    many = read_frames(clip_long, every_other)
    assert list(many) == every_other
    assert all(code_of(frame) == index for index, frame in many.items())

    with pytest.raises(FrameRangeError, match="beyond the end of the video: 20, 25"):
        read_frames(clip, [5, 20, 25])
    with pytest.raises(FrameRangeError) as caught:
        read_frames(clip, [-1, 2.0, True])  # type: ignore[list-item]
    assert len(caught.value.problems) == 3


def test_read_frames_on_the_cfr_grid(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    vfr = make_vfr(ffmpeg_bin, unicode_tmp_path)
    got = read_frames(vfr, [9, 10, 11, 12, 20, 29], fps="25/1")
    assert {index: code_of(frame) for index, frame in got.items()} == {
        9: 9,
        10: 9,
        11: 9,
        12: 12,
        20: 19,
        29: 29,
    }


def test_frame_count_matches_reader_and_probe(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=20)
    assert frame_count(clip) == probe(clip).nb_frames_decoded == len(list(iter_frames(clip))) == 20
    vfr = make_vfr(ffmpeg_bin, unicode_tmp_path)
    assert frame_count(vfr) == len(list(iter_frames(vfr))) == 27
    assert frame_count(vfr, fps="25/1") == len(list(iter_frames(vfr, fps="25/1"))) == 30


def test_pixel_sha256_format_and_validation() -> None:
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    expected = hashlib.sha256(b"gbf.rgb24 h=2 w=3 c=3\n" + frame.tobytes()).hexdigest()
    assert pixel_sha256(frame) == expected
    assert pixel_sha256(frame.reshape(3, 2, 3)) != expected  # same bytes, other shape
    flipped = frame[:, ::-1]  # a non-contiguous view hashes as its pixels
    assert pixel_sha256(flipped) == pixel_sha256(np.ascontiguousarray(flipped))
    assert pixel_sha256(flipped) != expected
    with pytest.raises(ValueError, match="dtype must be uint8.*shape must be"):
        pixel_sha256(np.zeros((2, 3), dtype=np.float32))


def test_rotated_source_is_read_upright(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "clip.mp4", n=10)
    rotated = unicode_tmp_path / "竖 屏.mp4"
    run_ffmpeg(ffmpeg_bin, "-display_rotation", "90", "-i", str(clip), "-c", "copy", str(rotated))
    facts = probe(rotated)
    assert (facts.width, facts.height, facts.rotation_deg) == (W, H, 90)
    assert (facts.frame_width, facts.frame_height) == (H, W)
    stored = [frame for _, frame in iter_frames(clip)]
    upright = [frame for _, frame in iter_frames(rotated)]
    assert upright[0].shape == (W, H, 3)  # same byte count as (H, W, 3): must not be garbled
    assert all(np.array_equal(a, np.rot90(b)) for a, b in zip(upright, stored, strict=True))
    result = normalize(rotated, unicode_tmp_path / "upright.mp4")
    assert (result.output.width, result.output.height, result.output.rotation_deg) == (H, W, 0)


# --------------------------------------------------------------------------- cuts


def test_detect_cuts_finds_the_known_cut_and_nothing_else(
    unicode_tmp_path: Path, ffmpeg_bin: str
) -> None:
    clip = make_clip(
        ffmpeg_bin, unicode_tmp_path / "two shots.mp4", n=30, shot=lambda i: int(i >= 15)
    )
    assert detect_cuts(iter_frames(clip)) == [15]
    assert detect_cuts(iter_frames(clip, 10)) == [15]  # indices from iter_frames are kept
    bare = [frame for _, frame in iter_frames(clip)]
    assert detect_cuts(bare) == [15]
    assert detect_cuts(bare[15:]) == []
    scores = dict(cut_scores(bare))
    assert sorted(scores) == list(range(1, 30))
    assert scores[15] > 1.0  # the background colour changes everywhere
    assert max(score for index, score in scores.items() if index != 15) < DEFAULT_CUT_THRESHOLD
    assert detect_cuts(bare, threshold=scores[15] + 0.01) == []


def test_detect_cuts_rejects_bad_input() -> None:
    frame = make_frame(0)
    with pytest.raises(ValueError, match="consecutive"):
        detect_cuts([(0, frame), (2, frame)])
    with pytest.raises(ValueError, match="mixes"):
        detect_cuts([(0, frame), frame])
    for bad in (0, 2, float("nan"), "0.5"):
        with pytest.raises(ValueError, match="threshold"):
            detect_cuts([frame, frame], threshold=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dtype must be uint8"):
        detect_cuts([frame.astype(np.float32)])


# --------------------------------------------------------------------------- windows


def test_window_at_nonzero_offset_is_bit_identical(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    source = make_clip(ffmpeg_bin, unicode_tmp_path / "源.mp4", n=30)
    canonical = [frame for _, frame in iter_frames(source)]
    dst = unicode_tmp_path / "窗口" / "射门 1.mkv"
    record = extract_window(source, 7, 16, dst)
    assert record.window_id == "w000007-000016"
    assert (record.start_frame, record.end_frame, record.n_frames) == (7, 16, 10)
    assert (record.codec, record.container, record.fps) == ("ffv1", "matroska", "25/1")
    assert record.first_pixel_sha256 == pixel_sha256(canonical[7])
    assert record.last_pixel_sha256 == pixel_sha256(canonical[16])
    assert record.file_sha256 == sha256_file(dst)
    assert record.file_bytes == dst.stat().st_size
    assert record.source_sha256 == sha256_file(source)
    window = [frame for _, frame in iter_frames(dst)]
    assert [code_of(frame) for frame in window] == list(range(7, 17))  # window k = source 7+k
    for k, frame in enumerate(window):
        assert np.array_equal(frame, canonical[7 + k]), k
    assert keyframe_flags(dst) == [True] * 10
    again = extract_window(source, 7, 16, unicode_tmp_path / "again.mkv")
    assert again.file_sha256 == record.file_sha256  # deterministic bytes

    single = extract_window(source, 29, 29, unicode_tmp_path / "last.mkv")
    assert single.n_frames == 1
    assert single.first_pixel_sha256 == pixel_sha256(canonical[29])
    with pytest.raises(FrameRangeError, match="frames 30..30 do not exist"):
        extract_window(source, 25, 30, unicode_tmp_path / "too long.mkv")
    assert not (unicode_tmp_path / "too long.mkv").exists()
    assert not (unicode_tmp_path / "too long.mkv.partial").exists()


def test_window_from_vfr_source_uses_the_cfr_grid(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    vfr = make_vfr(ffmpeg_bin, unicode_tmp_path)
    record = extract_window(vfr, 8, 13, unicode_tmp_path / "w.mkv")
    expected = [frame for _, frame in iter_frames(vfr, 8, 14, fps="25/1")]
    window = [frame for _, frame in iter_frames(unicode_tmp_path / "w.mkv")]
    assert [code_of(frame) for frame in window] == [8, 9, 9, 9, 12, 13]
    assert all(np.array_equal(a, b) for a, b in zip(window, expected, strict=True))
    assert record.first_pixel_sha256 == pixel_sha256(expected[0])
    assert record.last_pixel_sha256 == pixel_sha256(expected[-1])


def test_window_rejects_bad_requests(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    source = make_clip(ffmpeg_bin, unicode_tmp_path / "source.mp4", n=10)
    existing = unicode_tmp_path / "exists.mkv"
    existing.write_bytes(b"keep me")
    with pytest.raises(WindowError) as caught:
        extract_window(source, -1, -3, unicode_tmp_path / "w.mp4", fps="29.97")
    assert len(caught.value.problems) == 4  # start, end, suffix and fps, all at once
    with pytest.raises(WindowError, match="destination exists"):
        extract_window(source, 0, 3, existing)
    assert existing.read_bytes() == b"keep me"
    with pytest.raises(WindowError, match="source_sha256"):
        extract_window(source, 0, 3, unicode_tmp_path / "w.mkv", source_sha256="abc")


# --------------------------------------------------------------------------- video.json


def test_video_json_validates_and_has_no_absolute_paths(
    unicode_tmp_path: Path, ffmpeg_bin: str
) -> None:
    vfr = make_vfr(ffmpeg_bin, unicode_tmp_path)
    source = probe(vfr)
    normalized = normalize(vfr, unicode_tmp_path / "规范.mp4", source_probe=source)
    window = extract_window(
        unicode_tmp_path / "规范.mp4",
        5,
        14,
        unicode_tmp_path / "w.mkv",
        source_sha256=normalized.output.sha256,
    )
    cuts = detect_cuts(iter_frames(unicode_tmp_path / "规范.mp4"))
    assert cuts == []
    out = unicode_tmp_path / "记录" / "video.json"
    record = write_video_json(
        out,
        source,
        normalized=normalized,
        cuts=[12],
        cut_detection=cut_detection_settings(),
        windows=[window],
        window_relpaths={window.window_id: "windows/w.mkv"},
        restrictions=["no_share", "nda"],
        source_relpath="private/clips/clip.mp4",
    )
    data = read_json(out)
    assert data == record
    contracts.validate(data, "video")
    assert data["canonical_id"] == source.sha256 == data["source"]["sha256"]
    assert data["normalized"]["sha256"] == sha256_file(unicode_tmp_path / "规范.mp4")
    assert (data["fps"], data["frame_count"], data["width"], data["height"]) == ("25/1", 30, W, H)
    assert data["normalization"]["expected_frames"] == 30
    assert data["windows"][0]["relpath"] == "windows/w.mkv"
    assert set(data["tools"]) >= {"ffmpeg", "ffprobe", "gbf"}
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    text = raw.decode("utf-8")
    assert no_local_paths(text, unicode_tmp_path, unicode_tmp_path.parent) == []
    assert "规范.mp4" not in text
    assert "可变 帧率" not in text
    write_video_json(
        unicode_tmp_path / "again.json",
        source,
        normalized=normalized,
        cuts=[12],
        cut_detection=cut_detection_settings(),
        windows=[window],
        window_relpaths={window.window_id: "windows/w.mkv"},
        restrictions=["no_share", "nda"],
        source_relpath="private/clips/clip.mp4",
    )
    assert (unicode_tmp_path / "again.json").read_bytes() == raw  # deterministic

    broken = dict(data, fps="29.97")
    del broken["tools"]
    assert len(contracts.problems(broken, "video")) == 2


def test_video_record_rejects_broken_invariants(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    vfr = make_vfr(ffmpeg_bin, unicode_tmp_path)
    source = probe(vfr)
    with pytest.raises(RecordError, match="not CFR at 25/1"):
        build_video_record(source)  # a VFR source must be normalised first
    normalized = normalize(vfr, unicode_tmp_path / "n.mp4", source_probe=source)
    window = extract_window(unicode_tmp_path / "n.mp4", 25, 29, unicode_tmp_path / "w.mkv")
    with pytest.raises(RecordError) as caught:
        build_video_record(
            source,
            normalized=normalized,
            cuts=[0, 30, 7, 7],
            windows=[window, window],
            source_relpath="C:/data/clip.mp4",
            tools={"ffmpeg": "x", "ffprobe": "y", "gbf": "z"},
        )
    found = "\n".join(caught.value.problems)
    assert "cuts/0: 0 is outside 1..29" in found
    assert "cuts/1: 30 is outside 1..29" in found
    assert "strictly ascending" in found
    assert "duplicate window id" in found
    assert "looks like a local path" in found
    # A clean CFR source can be its own canonical file.
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "cfr.mp4", n=12)
    plain = build_video_record(probe(clip), tools=tool_versions())
    assert plain["frame_count"] == 12
    assert "normalized" not in plain


# --------------------------------------------------------------------------- CLI


def test_cli_probe_normalize_window(
    unicode_tmp_path: Path, ffmpeg_bin: str, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = make_clip(ffmpeg_bin, unicode_tmp_path / "源 片.mp4", n=12)
    assert gbf_main(["ingest", "probe", str(clip)]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed)["nb_frames_decoded"] == 12
    assert no_local_paths(printed, unicode_tmp_path) == []
    assert "源 片" not in printed

    out = unicode_tmp_path / "n.mp4"
    assert ingest_main(["normalize", str(clip), str(out), "--all-intra"]) == 0
    assert json.loads(capsys.readouterr().out)["output"]["nb_frames_decoded"] == 12
    assert keyframe_flags(out) == [True] * 12

    assert ingest_main(["window", str(out), "2", "5", str(unicode_tmp_path / "w.mkv")]) == 0
    assert json.loads(capsys.readouterr().out)["n_frames"] == 4

    assert ingest_main(["window", str(out), "5", "2", str(unicode_tmp_path / "x.mkv")]) == 1
    assert "end_frame_inclusive (2) must be >= start_frame (5)" in capsys.readouterr().err
    assert ingest_main(["normalize", str(clip), str(out)]) == 1  # exists, not overwritten
    assert "exists" in capsys.readouterr().err
    assert ingest_main([]) == 2
    capsys.readouterr()
