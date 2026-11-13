"""Audio excitement per video frame, a highlight cue (M1, docs/design/m1-highlights.md).

The original broadcast audio carries the commentator's and the crowd's reaction, one of the
strongest cues that something worth watching happened. :func:`audio_features` decodes the
first audio stream of a file with ffmpeg into a pipe (no temporary files) and returns one
row per video frame. :func:`features_from_samples` does the same for samples already in
memory, which is what the tests use.

Timeline
    Audio time 0 is the first decoded sample of the first audio stream: its timestamps are
    rebased with ``asetpts=PTS-STARTPTS``, and gaps of more than ``GAP_FILL_MIN_S`` seconds
    inside the stream are filled with silence (``aresample=async=1``), so sample ``i`` lies at
    ``i / sample_rate`` seconds. Video frame ``k`` (canonical CFR index) is matched with audio
    time ``t_k = k / fps + offset_s``. ``offset_s`` is the audio time of video frame 0: 0 when
    the video and the audio start together (the usual case for one file whose streams both
    start at 0), positive when the audio starts earlier than the video, negative when later.

Window
    Frame ``k`` is described by ``L = round(WINDOW_FRAME_DURATIONS * sample_rate / fps)``
    samples centred on ``t_k`` (two frame durations: 1280 samples at 16 kHz and 25 fps); the
    first one is sample ``floor(t_k * sample_rate - L / 2 + 0.5)``. A frame has audio
    (``has_audio``) when its centre sample lies inside the decoded audio; its window is then
    shifted, if needed, to lie wholly inside the audio (zero-padded only when the whole audio
    is shorter than ``L``). Frames whose centre lies before the first sample or at or after
    the end of the audio get NaN in every feature and ``has_audio = False``.

Columns (one row per frame ``0 .. n_frames - 1``)
    ``frame`` (int64); ``rms_dbfs`` (float64): ``20 * log10(rms)`` of the window with full
    scale = 1.0 (a full-scale sine reads -3.01 dBFS), floored at ``RMS_FLOOR_DBFS`` for digital
    silence; ``flux`` (float64): spectral flux, the sum over frequency bins of the half-wave
    rectified increase of the magnitude spectrum from frame ``k - 1`` to frame ``k`` (periodic
    Hann window, magnitudes divided by the window sum so a full-scale sine peaks at 0.5),
    0 on a frame with audio whose previous frame has none; ``excitement_z`` (float64):
    ``z(rms_dbfs) + z(flux)``, averaged over a centred window of ``2 h + 1`` frames with
    ``h = round(EXCITEMENT_SMOOTHING_S * fps / 2)`` (13 frames, 0.52 s, at 25 fps; frames
    without audio are left out of the average), then z-scored again; ``has_audio`` (bool).
    Every z-score uses the mean and the population standard deviation over the frames with
    audio of this clip; a constant input gives 0.

Mono is the arithmetic mean of all channels after ffmpeg resampled them to ``sample_rate``
with ``aresample`` (libswresample defaults). The features are a selection cue, not a
measurement: they have no error receipt and are never shown as numbers of the match.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gbf.ingest.rates import format_rate, parse_rate
from gbf.ingest.tools import LOGLEVEL, decode_output, ffmpeg_path, ffprobe_path, media_arg, run_tool

#: Default analysis sample rate in Hz (speech and crowd energy sit well below 8 kHz).
DEFAULT_SAMPLE_RATE = 16000
#: Analysis window length in frame durations, centred on the frame time.
WINDOW_FRAME_DURATIONS = 2
#: Floor of ``rms_dbfs`` (dB relative to full scale); digital silence reads this value.
RMS_FLOOR_DBFS = -120.0
#: Length in seconds of the centred moving average applied to ``z(rms) + z(flux)``.
EXCITEMENT_SMOOTHING_S = 0.5
#: Smallest timestamp gap (seconds) that ``aresample=async=1`` fills with silence.
GAP_FILL_MIN_S = 0.1
#: Frames processed per block (bounds memory to about BLOCK_FRAMES * window * 8 bytes).
BLOCK_FRAMES = 2048
#: Lowest and highest accepted analysis sample rates in Hz.
MIN_SAMPLE_RATE = 1000
MAX_SAMPLE_RATE = 192000
#: Columns of the frame table, in order (the audio.parquet contract plus ``has_audio``).
AUDIO_COLUMNS = ("frame", "rms_dbfs", "flux", "excitement_z", "has_audio")


class AudioError(RuntimeError):
    """Base class for every failure in :mod:`gbf.audio`."""


class AudioInputError(AudioError, ValueError):
    """Invalid arguments. ``problems`` lists every problem found."""

    def __init__(self, what: str, problems: Sequence[str]):
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{what}: {len(self.problems)} problem(s)\n  - {joined}")


class NoAudioStreamError(AudioError):
    """The source file has no audio stream (pass the original file that carries the sound)."""


@dataclass(frozen=True)
class AudioStreamInfo:
    """Facts about the first audio stream of a file, as ffprobe reports them.

    ``start_time_s`` and ``duration_s`` are in seconds on the container timeline (``None``
    when unknown); ``sample_rate`` is the stream's own rate in Hz.
    """

    stream_index: int
    codec_name: str
    sample_rate: int
    channels: int
    start_time_s: float | None
    duration_s: float | None


def audio_feature_settings(sample_rate: int = DEFAULT_SAMPLE_RATE) -> dict[str, Any]:
    """Method and parameters of :func:`audio_features`, for receipts and evidence records."""
    return {
        "method": "rms_dbfs+spectral_flux",
        "sample_rate": int(sample_rate),
        "window_frame_durations": WINDOW_FRAME_DURATIONS,
        "window": "hann_periodic",
        "downmix": "mean_of_channels",
        "rms_floor_dbfs": RMS_FLOOR_DBFS,
        "excitement_smoothing_s": EXCITEMENT_SMOOTHING_S,
        "gap_fill_min_s": GAP_FILL_MIN_S,
        "timeline": "first decoded audio sample = 0; frame k at k / fps + offset_s",
        "validated": False,
    }


# --------------------------------------------------------------------------- arguments


def _rate_problems(fps: object) -> tuple[Fraction | None, list[str]]:
    if isinstance(fps, bool):
        return None, [f"fps must be a rate such as '25/1', got {fps!r}"]
    if isinstance(fps, int | np.integer):
        if int(fps) <= 0:
            return None, [f"fps must be positive, got {fps!r}"]
        return Fraction(int(fps)), []
    if isinstance(fps, float | np.floating):
        value = float(fps)
        if not math.isfinite(value) or value <= 0 or value != int(value):
            return None, [
                f"fps must be a positive rational such as '30000/1001'; a float is only "
                f"accepted when it is a whole number, got {fps!r}"
            ]
        return Fraction(int(value)), []
    try:
        return parse_rate(fps), []  # type: ignore[arg-type]
    except ValueError as exc:
        return None, [str(exc)]


def _argument_problems(
    fps: object, n_frames: object, offset_s: object, sample_rate: object
) -> tuple[Fraction | None, list[str]]:
    rate, problems = _rate_problems(fps)
    if isinstance(n_frames, bool) or not isinstance(n_frames, int | np.integer):
        problems.append(f"n_frames must be an integer, got {n_frames!r}")
    elif int(n_frames) < 0:
        problems.append(f"n_frames must be >= 0, got {n_frames}")
    if isinstance(offset_s, bool) or not isinstance(
        offset_s, int | float | np.integer | np.floating
    ):
        problems.append(f"offset_s must be a number of seconds, got {offset_s!r}")
    elif not math.isfinite(float(offset_s)):
        problems.append(f"offset_s must be finite, got {offset_s!r}")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int | np.integer):
        problems.append(f"sample_rate must be an integer number of Hz, got {sample_rate!r}")
    elif not MIN_SAMPLE_RATE <= int(sample_rate) <= MAX_SAMPLE_RATE:
        problems.append(
            f"sample_rate must lie in [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}] Hz, got {sample_rate}"
        )
    elif rate is not None and window_length(int(sample_rate), rate) < 4:
        problems.append(
            f"sample_rate {sample_rate} Hz gives fewer than 4 samples per window at fps "
            f"{format_rate(rate)}"
        )
    return rate, problems


def window_length(sample_rate: int, fps: Fraction) -> int:
    """Window length ``L`` in samples: ``WINDOW_FRAME_DURATIONS`` frame durations, rounded."""
    exact = Fraction(WINDOW_FRAME_DURATIONS * sample_rate) / Fraction(fps)
    return int(math.floor(exact + Fraction(1, 2)))


def frame_windows(
    n_frames: int, fps: Fraction, sample_rate: int, offset_s: float = 0.0
) -> tuple[np.ndarray, np.ndarray, int]:
    """Centre sample (float64), first sample (int64) and length ``L`` of every frame window.

    Sample indices are on the audio timeline described in the module docstring; they may be
    negative or lie beyond the end of the audio.
    """
    length = window_length(sample_rate, fps)
    hop = float(Fraction(sample_rate) / Fraction(fps))
    centres = np.arange(n_frames, dtype=np.float64) * hop + float(offset_s) * sample_rate
    starts = np.floor(centres - length / 2.0 + 0.5).astype(np.int64)
    return centres, starts, length


# --------------------------------------------------------------------------- features


def _hann(length: int) -> np.ndarray:
    n = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)


def zscore(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """z-score of ``values[mask]`` (population std); NaN outside ``mask``; 0 if constant."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    chosen = values[mask].astype(np.float64)
    if chosen.size == 0:
        return out
    mean = float(chosen.mean())
    std = float(chosen.std())
    out[mask] = 0.0 if std == 0.0 else (chosen - mean) / std
    return out


def centred_mean(values: np.ndarray, mask: np.ndarray, half_width: int) -> np.ndarray:
    """Mean of ``values`` over ``[i - half_width, i + half_width]`` restricted to ``mask``.

    Frames outside ``mask`` are left out of every average and get NaN themselves.
    """
    n = values.shape[0]
    filled = np.where(mask, values, 0.0).astype(np.float64)
    sums = np.concatenate(([0.0], np.cumsum(filled)))
    counts = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    index = np.arange(n)
    lo = np.clip(index - half_width, 0, n)
    hi = np.clip(index + half_width + 1, 0, n)
    total = sums[hi] - sums[lo]
    count = counts[hi] - counts[lo]
    out = np.full(n, np.nan, dtype=np.float64)
    ok = mask & (count > 0)
    out[ok] = total[ok] / count[ok]
    return out


def _excitement(
    rms_dbfs: np.ndarray, flux: np.ndarray, has: np.ndarray, fps: Fraction
) -> np.ndarray:
    half_width = int(math.floor(Fraction(EXCITEMENT_SMOOTHING_S) * fps / 2 + Fraction(1, 2)))
    combined = zscore(rms_dbfs, has) + zscore(flux, has)
    smoothed = centred_mean(np.nan_to_num(combined), has, half_width)
    return zscore(smoothed, has)


def features_from_samples(
    samples: np.ndarray,
    sample_rate: int,
    fps: str | Fraction | int,
    n_frames: int,
    *,
    offset_s: float = 0.0,
    start_sample: int = 0,
) -> pd.DataFrame:
    """Per-frame audio features of mono ``samples`` (see the module docstring).

    ``samples`` is a 1-D float array (full scale 1.0) holding audio-timeline samples
    ``start_sample .. start_sample + len(samples) - 1``; that span is the audio that exists.
    Raises :class:`AudioInputError` listing every invalid argument.
    """
    rate, problems = _argument_problems(fps, n_frames, offset_s, sample_rate)
    if not isinstance(samples, np.ndarray):
        problems.append(f"samples must be a numpy array, got {type(samples).__name__}")
    else:
        if samples.ndim != 1:
            problems.append(f"samples must be 1-D (mono), got shape {samples.shape}")
        if not np.issubdtype(samples.dtype, np.floating):
            problems.append(f"samples must be floating point, got {samples.dtype}")
        elif samples.ndim == 1 and not np.all(np.isfinite(samples)):
            problems.append("samples contain NaN or infinite values")
    if isinstance(start_sample, bool) or not isinstance(start_sample, int | np.integer):
        problems.append(f"start_sample must be an integer, got {start_sample!r}")
    if problems or rate is None:
        raise AudioInputError("features_from_samples", problems)
    n = int(n_frames)
    sr = int(sample_rate)
    first = int(start_sample)
    centres, starts, length = frame_windows(n, rate, sr, float(offset_s))
    avail_lo, avail_hi = first, first + int(samples.shape[0])
    has = (centres >= avail_lo) & (centres < avail_hi)
    rms_dbfs = np.full(n, np.nan, dtype=np.float64)
    flux = np.full(n, np.nan, dtype=np.float64)
    if np.any(has):
        _fill_features(samples, first, starts, length, has, rms_dbfs, flux)
    excitement = _excitement(rms_dbfs, flux, has, rate)
    return pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "rms_dbfs": rms_dbfs,
            "flux": flux,
            "excitement_z": excitement,
            "has_audio": has.astype(bool),
        },
        columns=list(AUDIO_COLUMNS),
    )


def _fill_features(
    samples: np.ndarray,
    first: int,
    starts: np.ndarray,
    length: int,
    has: np.ndarray,
    rms_dbfs: np.ndarray,
    flux: np.ndarray,
) -> None:
    """Write RMS (dBFS) and spectral flux of every frame with audio, block by block."""
    avail = int(samples.shape[0])
    avail_lo, avail_hi = first, first + avail
    upper = max(avail_lo, avail_hi - length)
    window = _hann(length)
    norm = float(window.sum())
    floor_rms = 10.0 ** (RMS_FLOOR_DBFS / 20.0)
    offsets = np.arange(length, dtype=np.int64)
    frames_with_audio = np.flatnonzero(has)
    previous_frame = -2
    previous_spectrum: np.ndarray | None = None
    for block_start in range(0, frames_with_audio.size, BLOCK_FRAMES):
        block = frames_with_audio[block_start : block_start + BLOCK_FRAMES]
        begin = np.clip(starts[block], avail_lo, upper) - avail_lo
        index = begin[:, None] + offsets[None, :]
        inside = index < avail
        chunk = np.zeros(index.shape, dtype=np.float64)
        chunk[inside] = samples[index[inside]]
        rms = np.sqrt(np.mean(chunk * chunk, axis=1))
        rms_dbfs[block] = 20.0 * np.log10(np.maximum(rms, floor_rms))
        spectra = np.abs(np.fft.rfft(chunk * window[None, :], axis=1)) / norm
        for row, frame in enumerate(block):
            spectrum = spectra[row]
            if previous_spectrum is not None and previous_frame == frame - 1:
                flux[frame] = float(np.maximum(spectrum - previous_spectrum, 0.0).sum())
            else:
                flux[frame] = 0.0
            previous_spectrum = spectrum
            previous_frame = int(frame)


# --------------------------------------------------------------------------- decoding


def probe_audio(source: str | Path) -> AudioStreamInfo:
    """Describe the first audio stream of ``source`` with ffprobe.

    Raises ``FileNotFoundError`` for a missing file, :class:`NoAudioStreamError` when the file
    has no audio stream, and :class:`gbf.ingest.FFmpegError` when ffprobe fails.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    args = [
        ffprobe_path(),
        "-v",
        LOGLEVEL,
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index,codec_name,sample_rate,channels,start_time,duration",
        "-of",
        "json",
        media_arg(path),
    ]
    proc = run_tool(args, what=f"probe the audio of {path.name}")
    data = json.loads(decode_output(proc.stdout) or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise NoAudioStreamError(
            f"{path.name} has no audio stream; pass the original file that carries the sound"
        )
    stream = streams[0]
    problems = []
    channels = _as_int(stream.get("channels"))
    rate = _as_int(stream.get("sample_rate"))
    if channels is None or channels < 1:
        problems.append(f"channel count {stream.get('channels')!r} is not a positive integer")
    if rate is None or rate < 1:
        problems.append(f"sample rate {stream.get('sample_rate')!r} is not a positive integer")
    if problems:
        raise AudioInputError(f"probe the audio of {path.name}", problems)
    return AudioStreamInfo(
        stream_index=int(stream.get("index", 0)),
        codec_name=str(stream.get("codec_name", "")),
        sample_rate=int(rate),  # type: ignore[arg-type]
        channels=int(channels),  # type: ignore[arg-type]
        start_time_s=_as_float(stream.get("start_time")),
        duration_s=_as_float(stream.get("duration")),
    )


def _as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def decode_mono(
    source: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    start_sample: int = 0,
    stop_sample: int | None = None,
    channels: int | None = None,
) -> np.ndarray:
    """Decode audio-timeline samples ``start_sample .. stop_sample - 1`` as mono float32.

    One ffmpeg process writes raw little-endian float32 into a pipe; the result may be
    shorter than requested when the audio ends first. ``channels`` is the stream's channel
    count (probed when ``None``). Raises :class:`gbf.ingest.FFmpegError` with ffmpeg's
    stderr when decoding fails or ffmpeg reports any error.
    """
    path = Path(source)
    problems = []
    start_ok = not isinstance(start_sample, bool) and isinstance(start_sample, int | np.integer)
    if not start_ok:
        problems.append(f"start_sample must be an integer, got {start_sample!r}")
    elif start_sample < 0:
        problems.append(f"start_sample must be >= 0, got {start_sample}")
    if stop_sample is not None:
        if isinstance(stop_sample, bool) or not isinstance(stop_sample, int | np.integer):
            problems.append(f"stop_sample must be an integer or None, got {stop_sample!r}")
        elif start_ok and stop_sample <= start_sample:
            problems.append(
                f"stop_sample ({stop_sample}) must be larger than start_sample ({start_sample})"
            )
    if channels is not None and (
        isinstance(channels, bool) or not isinstance(channels, int | np.integer) or channels < 1
    ):
        problems.append(f"channels must be a positive integer or None, got {channels!r}")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int | np.integer):
        problems.append(f"sample_rate must be an integer number of Hz, got {sample_rate!r}")
    elif not MIN_SAMPLE_RATE <= int(sample_rate) <= MAX_SAMPLE_RATE:
        problems.append(
            f"sample_rate must lie in [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}] Hz, got {sample_rate}"
        )
    if problems:
        raise AudioInputError(f"decode_mono({path.name})", problems)
    if channels is None:
        channels = probe_audio(path).channels
    filters = [
        "asetpts=PTS-STARTPTS",
        f"aresample={int(sample_rate)}:async=1:min_hard_comp={GAP_FILL_MIN_S}:first_pts=0",
        "aformat=sample_fmts=flt",
    ]
    trim = []
    if start_sample > 0:
        trim.append(f"start_sample={int(start_sample)}")
    if stop_sample is not None:
        trim.append(f"end_sample={int(stop_sample)}")
    if trim:
        filters.append("atrim=" + ":".join(trim))
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
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        ",".join(filters),
        "-ac",
        str(int(channels)),
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = run_tool(args, what=f"decode the audio of {path.name}")
    data = proc.stdout
    frame_bytes = 4 * int(channels)
    if len(data) % frame_bytes:
        raise AudioError(
            f"decode the audio of {path.name}: ffmpeg wrote {len(data)} bytes, not a whole "
            f"number of {channels}-channel float32 samples"
        )
    interleaved = np.frombuffer(data, dtype="<f4").reshape(-1, int(channels))
    if int(channels) == 1:
        return interleaved[:, 0].astype(np.float32)
    return interleaved.mean(axis=1, dtype=np.float64).astype(np.float32)


def audio_features(
    source: str | Path,
    fps: str | Fraction | int,
    n_frames: int,
    *,
    offset_s: float = 0.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> pd.DataFrame:
    """Per-frame audio features of the first audio stream of ``source``.

    ``fps`` is the video's exact frame rate (``"25/1"``, a ``Fraction`` or an int) and
    ``n_frames`` the number of video frames; the result has one row per frame with the
    columns :data:`AUDIO_COLUMNS` (see the module docstring for every definition and unit).
    Only the samples the frames need are decoded. Raises :class:`AudioInputError` listing
    every invalid argument, ``FileNotFoundError``, :class:`NoAudioStreamError`, and
    :class:`gbf.ingest.FFmpegError` when ffprobe or ffmpeg fails.
    """
    rate, problems = _argument_problems(fps, n_frames, offset_s, sample_rate)
    if problems or rate is None:
        raise AudioInputError("audio_features", problems)
    path = Path(source)
    info = probe_audio(path)
    n = int(n_frames)
    sr = int(sample_rate)
    _, starts, length = frame_windows(n, rate, sr, float(offset_s))
    if n == 0 or int(starts[-1]) + length <= 0:
        # No frame window reaches the audio: nothing to decode.
        return features_from_samples(
            np.zeros(0, dtype=np.float32), sr, rate, n, offset_s=float(offset_s), start_sample=0
        )
    first = max(0, int(starts[0]))
    stop = max(first + 1, int(starts[-1]) + length)
    samples = decode_mono(
        path, sample_rate=sr, start_sample=first, stop_sample=stop, channels=info.channels
    )
    return features_from_samples(samples, sr, rate, n, offset_s=float(offset_s), start_sample=first)
