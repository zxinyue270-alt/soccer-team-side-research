"""Tests for gbf.audio: per-frame RMS, spectral flux and excitement.

Every signal is synthetic: tones and seeded noise built here, written as WAV (and once as
AAC in MP4 through ffmpeg) into a temporary directory whose path has non-ASCII characters.
"""

from __future__ import annotations

import subprocess
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbf.audio import (
    AUDIO_COLUMNS,
    RMS_FLOOR_DBFS,
    AudioInputError,
    NoAudioStreamError,
    audio_feature_settings,
    audio_features,
    decode_mono,
    features_from_samples,
    frame_windows,
    probe_audio,
    window_length,
)
from gbf.util.io import write_image_png

SR = 16000
FPS = "25/1"


def noise(seconds: float, amplitude: float, seed: int, rate: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amplitude * rng.uniform(-1.0, 1.0, int(round(seconds * rate)))


def tone(seconds: float, frequency: float, amplitude: float, rate: int = SR) -> np.ndarray:
    t = np.arange(int(round(seconds * rate))) / rate
    return amplitude * np.sin(2 * np.pi * frequency * t)


#: rms_dbfs of the quiet part (0.01 peak sine, -43 dBFS) and of the burst (0.5 peak, -9 dBFS).
QUIET_DBFS = 20 * np.log10(0.01 / np.sqrt(2))
LOUD_DBFS = 20 * np.log10(0.5 / np.sqrt(2))


def burst_signal(rate: int = SR) -> np.ndarray:
    """4 s of a quiet 300 Hz tone with a loud 1 kHz burst from 2.0 s to 2.4 s.

    The burst covers video frames 50..59 at 25 fps. Tones below 8 kHz survive resampling to
    16 kHz unchanged, so decoded files can be compared with the in-memory reference.
    """
    signal = tone(4.0, 300.0, 0.01, rate)
    start = int(2.0 * rate)
    loud = tone(0.4, 1000.0, 0.5, rate)
    signal[start : start + loud.size] = loud
    return signal


def write_wav(path: Path, channels: list[np.ndarray], rate: int) -> None:
    data = np.stack(channels, axis=1)
    pcm = np.round(np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(len(channels))
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- windows


def test_windows_are_two_frames_long_and_centred_on_the_frame_time() -> None:
    assert window_length(SR, Fraction(25)) == 1280
    centres, starts, length = frame_windows(4, Fraction(25), SR, 0.0)
    assert length == 1280
    assert centres.tolist() == [0.0, 640.0, 1280.0, 1920.0]
    assert starts.tolist() == [-640, 0, 640, 1280]
    centres, starts, _ = frame_windows(2, Fraction(25), SR, 0.01)
    assert centres.tolist() == [160.0, 800.0]
    assert starts.tolist() == [-480, 160]
    # NTSC rate: 16000 * 1001 / 30000 = 533.87 samples per frame, two of them rounded.
    assert window_length(SR, Fraction(30000, 1001)) == 1068


# --------------------------------------------------------------------------- features


def test_rms_of_a_full_scale_sine_is_minus_3_dbfs_and_silence_hits_the_floor() -> None:
    t = np.arange(2 * SR) / SR
    sine = np.sin(2 * np.pi * 1000.0 * t)
    table = features_from_samples(sine, SR, FPS, 50)
    inner = table["rms_dbfs"].to_numpy()[2:-2]
    assert np.allclose(inner, -3.0103, atol=0.01)
    silent = features_from_samples(np.zeros(2 * SR), SR, FPS, 50)
    assert np.all(silent["rms_dbfs"] == RMS_FLOOR_DBFS)
    assert np.all(silent["flux"] == 0.0)
    assert np.all(silent["excitement_z"] == 0.0)


def test_columns_and_dtypes_follow_the_contract() -> None:
    table = features_from_samples(burst_signal(), SR, FPS, 100)
    assert tuple(table.columns) == AUDIO_COLUMNS
    assert table["frame"].dtype == np.int64
    assert table["has_audio"].dtype == bool
    for name in ("rms_dbfs", "flux", "excitement_z"):
        assert table[name].dtype == np.float64
    assert table["frame"].tolist() == list(range(100))


def test_a_loud_burst_is_found_on_its_frames() -> None:
    table = features_from_samples(burst_signal(), SR, FPS, 100)
    rms = table["rms_dbfs"].to_numpy()
    # Frames 51..58 see only the burst; frames up to 48 and from 61 on see none of it.
    assert np.all(rms[51:59] > LOUD_DBFS - 0.5)
    assert np.all(rms[:49] < QUIET_DBFS + 0.5)
    assert np.all(rms[61:] < QUIET_DBFS + 0.5)
    # Frame 50 is the first window that contains the onset: the flux peaks there or next.
    assert int(np.argmax(table["flux"].to_numpy())) in (50, 51)
    excitement = table["excitement_z"].to_numpy()
    assert 50 <= int(np.argmax(excitement)) <= 59
    assert excitement[54] > 2.0


def test_excitement_is_z_scored_over_the_frames_with_audio() -> None:
    table = features_from_samples(burst_signal(), SR, FPS, 120)
    has = table["has_audio"].to_numpy()
    assert has[:100].all()
    assert not has[100:].any()
    z = table["excitement_z"].to_numpy()
    assert abs(float(z[has].mean())) < 1e-9
    assert abs(float(z[has].std()) - 1.0) < 1e-9
    assert np.isnan(z[~has]).all()


def test_frames_after_the_end_of_the_audio_get_nan() -> None:
    samples = noise(1.0, 0.1, seed=3)
    table = features_from_samples(samples, SR, FPS, 40)
    has = table["has_audio"].to_numpy()
    # Frame k is centred on sample 640 k; the audio holds samples 0..15999.
    assert has[:25].all()
    assert not has[25:].any()
    for name in ("rms_dbfs", "flux", "excitement_z"):
        values = table[name].to_numpy()
        assert np.isfinite(values[:25]).all()
        assert np.isnan(values[25:]).all()


def test_offset_moves_the_audio_timeline() -> None:
    signal = burst_signal()
    # Video frame k plays audio time k / 25 + 0.4 s, so the burst at 2.0 s is frame 40.
    later = features_from_samples(signal, SR, FPS, 60, offset_s=0.4)
    rms = later["rms_dbfs"].to_numpy()
    assert np.all(rms[41:49] > LOUD_DBFS - 0.5)
    assert np.all(rms[:39] < QUIET_DBFS + 0.5)
    # A negative offset puts the first frames before the first sample: no audio there.
    earlier = features_from_samples(signal, SR, FPS, 20, offset_s=-0.2)
    has = earlier["has_audio"].to_numpy()
    assert not has[:5].any()
    assert has[5:].all()


def test_start_sample_places_a_partial_buffer_on_the_timeline() -> None:
    signal = burst_signal()
    full = features_from_samples(signal, SR, FPS, 90)
    first = 30 * 640 - 640  # the first sample of frame 30's window, frame 29's centre
    part = features_from_samples(signal[first:], SR, FPS, 90, start_sample=first)
    has = part["has_audio"].to_numpy()
    assert not has[:29].any()
    assert has[29:].all()
    # Frame 29's window is shifted into the buffer; from frame 30 on nothing differs.
    np.testing.assert_allclose(part["rms_dbfs"].to_numpy()[30:], full["rms_dbfs"].to_numpy()[30:])
    assert part["flux"].to_numpy()[29] == 0.0


def test_invalid_arguments_report_every_problem() -> None:
    with pytest.raises(AudioInputError) as caught:
        features_from_samples(
            np.full((10, 2), np.nan), 5, "29.97", -1, offset_s=float("inf"), start_sample=1.5
        )
    problems = "\n".join(caught.value.problems)
    for fragment in ("29.97", "n_frames", "offset_s", "sample_rate", "1-D", "start_sample"):
        assert fragment in problems
    with pytest.raises(AudioInputError):
        features_from_samples(np.array([0.0, np.nan]), SR, FPS, 1)


def test_settings_describe_the_method() -> None:
    settings = audio_feature_settings()
    assert settings["sample_rate"] == SR
    assert settings["window_frame_durations"] == 2
    assert settings["validated"] is False


# --------------------------------------------------------------------------- decoding


def test_audio_features_decodes_a_wav_through_ffmpeg(unicode_tmp_path: Path) -> None:
    rate = 48000
    path = unicode_tmp_path / "clip ü.wav"
    left = burst_signal(rate)[: 3 * rate]
    write_wav(path, [left, left], rate)
    info = probe_audio(path)
    assert (info.channels, info.sample_rate) == (2, rate)
    table = audio_features(path, FPS, 90)
    reference = features_from_samples(burst_signal()[: 3 * SR], SR, FPS, 90)
    has = table["has_audio"].to_numpy()
    assert has[:75].all()
    assert not has[75:].any()
    rms = table["rms_dbfs"].to_numpy()
    assert np.all(rms[51:59] > LOUD_DBFS - 0.5)
    assert np.all(rms[:49] < QUIET_DBFS + 0.5)
    # Resampling 48 kHz -> 16 kHz only touches the edges of the burst.
    np.testing.assert_allclose(rms[52:58], reference["rms_dbfs"].to_numpy()[52:58], atol=0.2)


def test_mono_is_the_mean_of_the_channels(unicode_tmp_path: Path) -> None:
    rate = 48000
    left = tone(2.0, 440.0, 0.5, rate)
    path = unicode_tmp_path / "left only.wav"
    write_wav(path, [left, np.zeros_like(left)], rate)
    samples = decode_mono(path)
    assert samples.dtype == np.float32
    assert abs(samples.size - 2 * SR) <= 2
    # (0.5 + 0) / 2 = 0.25 peak; ffmpeg's own downmix would give 0.354.
    assert abs(float(np.abs(samples[4000:28000]).max()) - 0.25) < 0.005
    table = audio_features(path, FPS, 40)
    expected = 20 * np.log10(0.25 / np.sqrt(2))
    assert np.allclose(table["rms_dbfs"].to_numpy()[3:-3], expected, atol=0.05)


def test_an_offset_decodes_only_the_needed_samples(unicode_tmp_path: Path) -> None:
    rate = 48000
    path = unicode_tmp_path / "offset.wav"
    write_wav(path, [burst_signal(rate)], rate)
    table = audio_features(path, FPS, 40, offset_s=1.2)
    rms = table["rms_dbfs"].to_numpy()
    # Burst at 2.0 s of audio = frame (2.0 - 1.2) * 25 = 20 of the video.
    assert np.all(rms[21:29] > LOUD_DBFS - 0.5)
    assert np.all(rms[:19] < QUIET_DBFS + 0.5)
    assert table["has_audio"].all()
    part = decode_mono(path, start_sample=SR, stop_sample=SR + 800)
    assert part.size == 800


def test_aac_in_mp4_keeps_the_alignment(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    rate = 48000
    wav = unicode_tmp_path / "source.wav"
    write_wav(wav, [burst_signal(rate)[: 3 * rate]] * 2, rate)
    mp4 = unicode_tmp_path / "源 clip.mp4"
    subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(wav),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(mp4),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    table = audio_features(mp4, FPS, 90)
    rms = table["rms_dbfs"].to_numpy()
    # Frames 51..59 hold only the burst; 50 and 60 hold half of it (3 dB less). AAC's
    # encoder delay is undone by the MP4 edit list, so the burst stays on its frames.
    loud = np.flatnonzero(rms > LOUD_DBFS - 2.0)
    assert int(loud[0]) == 51
    assert int(loud[-1]) == 59
    assert loud.size == 9


def test_a_file_without_audio_is_refused(unicode_tmp_path: Path) -> None:
    image = unicode_tmp_path / "frame.png"
    write_image_png(image, np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(NoAudioStreamError):
        audio_features(image, FPS, 10)
    with pytest.raises(FileNotFoundError):
        audio_features(unicode_tmp_path / "missing.wav", FPS, 10)
    with pytest.raises(AudioInputError):
        audio_features(image, "25/1", -3)


def test_no_frame_reaches_the_audio(unicode_tmp_path: Path) -> None:
    rate = 48000
    path = unicode_tmp_path / "short.wav"
    write_wav(path, [noise(1.0, 0.1, seed=4, rate=rate)], rate)
    before = audio_features(path, FPS, 10, offset_s=-5.0)
    assert not before["has_audio"].any()
    after = audio_features(path, FPS, 10, offset_s=5.0)
    assert not after["has_audio"].any()
    assert isinstance(after, pd.DataFrame)
    assert len(after) == 10
    empty = audio_features(path, FPS, 0)
    assert len(empty) == 0
    assert tuple(empty.columns) == AUDIO_COLUMNS
