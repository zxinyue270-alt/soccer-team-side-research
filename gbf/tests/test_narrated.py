"""Narrated-video acceptance: provenance, missing IDs, measured speech and real media."""

from __future__ import annotations

import io
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbf import narrated, narration_model
from gbf.ingest import probe
from gbf.ingest.tools import ffprobe_path, media_arg
from gbf.narration_model import validate_answer
from gbf.render import plan_highlight_edit
from gbf.synth.video import write_frames_mp4
from gbf.util.hashing import sha256_file
from gbf.util.io import read_json, write_json
from gbf.web.runs import run_detail


def highlight(first=0, key=25, last=49, **changes):
    return {
        "id": "h01",
        "start_frame": first,
        "peak_frame": key,
        "end_frame": last,
        "type": "duel_tackle",
        "score": 0.5,
        "track_ids": [7],
        **changes,
    }


def fixture_run(root: Path, highlights=None, *, textured=False):
    root.mkdir()
    video = root / "source.mp4"
    frames = [np.full((72, 128, 3), (i * 4, 80, 100), dtype=np.uint8) for i in range(50)]
    if textured:
        background = np.random.default_rng(17).integers(0, 256, (72, 128, 3), dtype=np.uint8)
        frames = [background.copy() for _ in range(50)]
    write_frames_mp4(frames, video, fps="25/1")
    facts = probe(video)
    run = root / "runs" / "sample"
    (run / "refine").mkdir(parents=True)
    write_json(
        run / "run.json",
        {
            "status": "passed",
            "restrictions": [],
            "result": {
                "video": {
                    "sha256": facts.sha256,
                    "frames": 50,
                    "fps": "25/1",
                    "width": 128,
                    "height": 72,
                }
            },
        },
    )
    write_json(
        run / "highlights_final.json",
        {
            "fps": "25/1",
            "frame_count": 50,
            "highlights": [highlight()] if highlights is None else highlights,
        },
    )
    write_json(run / "cuts.json", {"cuts": []})
    pd.DataFrame(
        [
            {"frame": i, "track_id": 7, "x1": 10 + i / 2, "y1": 12, "x2": 35 + i / 2, "y2": 65}
            for i in range(50)
        ]
    ).to_parquet(run / "refine/tracks.parquet")
    pd.DataFrame([{"start_frame": 0, "end_frame": 49}]).to_parquet(run / "shots.parquet")
    pd.DataFrame(
        [
            {
                "frame": i,
                "u": 40,
                "v": 50,
                "radius_px": 1,
                "status": "detected",
                "candidate_count": 1,
            }
            for i in range(50)
        ]
    ).to_parquet(run / "ball_track.parquet")
    return video, run


def test_selection_prefers_scores_suppresses_duplicate_keeps_distinct_moments():
    doc = {
        "frame_count": 250,
        "highlights": [
            highlight(0, 25, 49),
            highlight(5, 26, 50, id="duplicate", score=0.2),
            highlight(48, 70, 100, id="next", score=0.8),
            highlight(150, 175, 200, id="third", score=0.6),
        ],
    }
    assert [h["id"] for h in narrated.select_highlights(doc)] == ["h01", "next", "third"]


@pytest.mark.parametrize(
    "changes", [{"peak_frame": True}, {"end_frame": 251}, {"score": float("nan")}]
)
def test_bad_candidates_are_not_silently_rendered(changes):
    with pytest.raises(ValueError, match="Candidate"):
        narrated.select_highlights({"frame_count": 250, "highlights": [highlight(**changes)]})


def test_local_identity_stops_at_missing_and_duplicate_frames():
    rows = [
        {"frame": f, "track_id": tid, "x1": 5, "y1": 4, "x2": 15, "y2": 25}
        for f in range(8)
        for tid in (1, 2)
        if not (f == 2 and tid == 2)
    ]
    rows.append(dict(rows[-1], frame=6))
    boxes = narrated.continuous_boxes(pd.DataFrame(rows), [1, 2], 0, 7, 4)
    assert set(boxes) == {3, 4, 5}
    assert narrated.continuous_boxes(pd.DataFrame(rows), [1, 2], 0, 7, 2) == {}


def test_fractional_rate_schedule_never_cuts_or_overlaps_speech():
    fps = Fraction(30000, 1001)
    plan = plan_highlight_edit(0, 59, 30, fps)
    lengths = [AUDIO_RATE * 5 + 127, AUDIO_RATE * 12 + 1]
    chunks, spans = narrated.schedule(plan, lengths, fps)
    total = sum(len(c["frames"]) * c["repeat"] for c in chunks)
    assert spans[0][1] <= spans[1][0]
    assert spans[1][1] < total
    assert any(c["kind"] == "overview_hold" for c in chunks)
    for (start, end), n in zip(spans, lengths, strict=True):
        assert Fraction(end - start, 1) / fps >= Fraction(n, AUDIO_RATE)


AUDIO_RATE = narrated.AUDIO_RATE


@pytest.mark.parametrize(
    "line",
    [
        {"text": "前方球员正在移动。", "frames": [99], "kind": "observation"},
        {"text": "两人相距3米。", "frames": [1], "kind": "observation"},
        {"text": "这是设计好的跑位。", "frames": [1], "kind": "hypothesis"},
    ],
)
def test_model_contract_rejects_unsupported_frames_metrics_and_certain_intent(line):
    good = {"text": "慢放回看，球员可能在拉开空间。", "frames": [1], "kind": "hypothesis"}
    with pytest.raises(ValueError, match="Invalid|Unverified|Tactical"):
        validate_answer({"lines": [line, good]}, {1})


def test_source_mismatch_fails_before_creating_output(tmp_path):
    video, run = fixture_run(tmp_path / "case")
    doc = read_json(run / "run.json")
    doc["result"]["video"]["sha256"] = "0" * 64
    write_json(run / "run.json", doc)
    with pytest.raises(ValueError, match="SHA"):
        narrated.create(video, run, run / "commentary")
    assert not (run / "commentary").exists()


def test_no_candidate_creates_honest_empty_result(tmp_path, capsys):
    video, run = fixture_run(tmp_path / "empty", [])
    result = narrated.create(video, run, run / "commentary")
    assert result["status"] == "no_highlights"
    assert result["video"] is None
    assert not list((run / "commentary").glob("*.mp4"))
    assert run_detail(run.parent, run.name)["narrated"]["status"] == "no_highlights"
    assert "[commentary] skipped" in capsys.readouterr().out


def test_context_movie_keeps_lead_in_follow_through_and_other_participant(tmp_path, monkeypatch):
    monkeypatch.delenv("GBF_DEEPSEEK_API_KEY", raising=False)
    video, run = fixture_run(
        tmp_path / "context",
        [highlight(20, 25, 30, type="sprint", track_ids=[7, 8], key_focus_track=7)],
    )
    path = run / "refine/tracks.parquet"
    tracks = pd.read_parquet(path)
    tracks["y1"], tracks["y2"] = 20, 60
    other = pd.DataFrame([dict(frame=i, track_id=8, x1=65, y1=20, x2=75, y2=60) for i in range(50)])
    pd.concat([tracks, other], ignore_index=True).to_parquet(path)
    ball_path = run / "ball_track.parquet"
    ball = pd.read_parquet(ball_path)
    ball["u"] = 75
    ball.to_parquet(ball_path)
    monkeypatch.setattr(
        narrated, "speech", lambda *_: ([np.ones(24000, dtype=np.int16)] * 2, "test")
    )
    result = narrated.create(video, run, run / "commentary")
    clip = result["clips"][0]
    assert (clip["start_frame"], clip["end_frame"]) == (0, 49)
    assert clip["action_boundaries_verified"] is False
    assert clip["target_ids"] == [7, 8]
    mapping = read_json(run / "commentary/source-mapping.json")["frames"]
    normal = [row for row in mapping if row["mode"] == "normal"]
    assert [row["source_frame"] for row in normal] == list(range(50))
    assert all(row["crop"] is None for row in normal)
    freeze = next(row for row in mapping if row["mode"] == "freeze")
    assert freeze["crop"] is not None
    assert freeze["crop"][2] >= 76
    assert freeze["focus_track_ids"] == [7, 8]
    assert str(ball_path) in result["inputs_sha256"]


def test_movie_uses_complete_gate_evidence_and_keeps_off_ball_people(tmp_path, monkeypatch):
    monkeypatch.delenv("GBF_DEEPSEEK_API_KEY", raising=False)
    video, run = fixture_run(tmp_path / "gated")
    tracks_path = run / "refine/tracks.parquet"
    tracks = pd.read_parquet(tracks_path)
    tracks["y1"], tracks["y2"] = 25, 50
    far = pd.DataFrame(
        [dict(frame=f, track_id=99, x1=112, y1=20, x2=124, y2=45) for f in range(50)]
    )
    pd.concat([tracks, far], ignore_index=True).to_parquet(tracks_path)
    ball_path = run / "ball_track.parquet"
    ball = pd.read_parquet(ball_path)
    ball["candidate_count"] = 2
    ball.to_parquet(ball_path)
    evidence = run / "ball_candidates.parquet"
    pd.DataFrame(
        [
            dict(frame=f, cand_index=i, u=u, v=50, diameter_px=2, reason=reason)
            for f in range(50)
            for i, u, reason in [(0, 40, "ok"), (1, 120, "size")]
        ]
    ).to_parquet(evidence)
    monkeypatch.setattr(
        narrated, "speech", lambda *_: ([np.ones(24000, dtype=np.int16)] * 2, "test")
    )
    result = narrated.create(video, run, run / "commentary")
    assert result["status"] == "ready"
    assert str(evidence) in result["inputs_sha256"]
    plan = read_json(run / "commentary/part-1/focus-plan.json")["frames"]
    assert all({p["track_id"] for p in row["scene_people"]} == {7, 99} for row in plan)
    mapping = read_json(run / "commentary/source-mapping.json")["frames"]
    replay = [r for r in mapping if r["mode"] in {"slow", "freeze"}]
    assert all(r["crop"] is not None for r in replay)
    assert len({tuple(r["crop"]) for r in replay}) == 1


def test_movie_emphasizes_release_and_renders_estimated_tail_without_changing_model_pixels(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("GBF_DEEPSEEK_API_KEY", raising=False)
    video, run = fixture_run(
        tmp_path / "release",
        [
            highlight(
                10,
                20,
                35,
                type="sprint",
                key_moments=[dict(kind="ball_away", frame=28, track_ids=[7])],
            )
        ],
        textured=True,
    )
    ball = pd.read_parquet(run / "ball_track.parquet")
    ball["u"] = 40 + ball.frame / 2
    ball.to_parquet(run / "ball_track.parquet")
    poses = pd.read_parquet(run / "refine/tracks.parquet")
    poses["keypoints"] = [
        np.tile([(r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2, 0.9], 26).tolist()
        for _, r in poses.iterrows()
    ]
    poses["model_id"] = "test-halpe26"
    poses.to_parquet(run / "pose.parquet")
    monkeypatch.setattr(
        narrated, "speech", lambda *_: ([np.ones(24000, dtype=np.int16)] * 2, "test")
    )
    result = narrated.create(video, run, run / "commentary")
    clip = result["clips"][0]
    assert clip["key_frame"] == 28
    assert 28 in clip["model_sample_frames"]
    mapping = read_json(run / "commentary/source-mapping.json")["frames"]
    assert {r["source_frame"] for r in mapping if r["mode"] == "freeze"} == {28}
    assert 49 in {r["source_frame"] for r in mapping if r["mode"] == "slow"}
    assert any(len(r["ball_trail"]["points"]) > 1 for r in mapping if r["mode"] == "slow")
    assert clip["explanation_source"] == "local_replay_guidance"
    assert str(run / "pose.parquet") in result["inputs_sha256"]
    assert next(r for r in mapping if r["mode"] == "freeze")["player_effects"]["actors"][0]["edges"]
    raw = np.full((72, 128, 3), 50, dtype=np.uint8)
    before = raw.copy()
    plain = narrated.compose(raw, [], "slow", "test", "", 28, Fraction(25), False)
    trailed = narrated.compose(
        raw,
        [],
        "slow",
        "test",
        "",
        28,
        Fraction(25),
        False,
        trail=next(r["ball_trail"] for r in mapping if r["source_frame"] == 28),
    )
    np.testing.assert_array_equal(raw, before)
    assert np.any(np.asarray(plain) != np.asarray(trailed))


@pytest.mark.parametrize("replay_available", [True, False])
def test_replay_guidance_matches_available_crop(tmp_path, monkeypatch, replay_available):
    monkeypatch.delenv("GBF_DEEPSEEK_API_KEY", raising=False)
    video, run = fixture_run(tmp_path / "available", [highlight(type="sprint")])
    if not replay_available:
        # A missing peak box no longer suppresses other independently observed frames.
        path = run / "ball_track.parquet"
        ball = pd.read_parquet(path)
        ball["status"] = "missing"
        ball.to_parquet(path)
    monkeypatch.setattr(
        narrated, "speech", lambda *_: ([np.ones(24000, dtype=np.int16)] * 2, "test-voice")
    )
    result = narrated.create(video, run, run / "commentary")
    cues = read_json(run / "commentary/subtitles.json")["cues"]
    mapping = read_json(run / "commentary/source-mapping.json")["frames"]
    assert result["status"] == "ready"
    assert ("局部" in cues[1]["text"]) == replay_available
    assert "标记球员" not in cues[0]["text"]
    assert any(row["crop"] is not None for row in mapping) == replay_available


def test_real_mp4_with_speech_longer_than_original_has_full_audio_and_timed_subtitles(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("GBF_DEEPSEEK_API_KEY", raising=False)
    video, run = fixture_run(tmp_path / "media")
    lengths = [AUDIO_RATE * 3 + 150, AUDIO_RATE * 5 + 200]

    def sine_speech(lines, folder):
        assert len(lines) == 2
        return [
            np.round(3500 * np.sin(np.arange(n) * 2 * math.pi * 440 / AUDIO_RATE)).astype("<i2")
            for n in lengths
        ], "synthetic-test-voice"

    monkeypatch.setattr(narrated, "speech", sine_speech)
    result = narrated.create(video, run, run / "commentary")
    assert result["status"] == "ready"
    facts = probe(run / "commentary/highlights.mp4")
    assert facts.nb_frames_decoded == result["frames"]
    cues = read_json(run / "commentary/subtitles.json")["cues"]
    assert cues[-1]["end_s"] < result["duration_s"]
    assert cues[0]["end_s"] <= cues[1]["start_s"]
    assert abs(cues[1]["end_s"] - cues[1]["start_s"] - lengths[1] / AUDIO_RATE) < 1e-5
    mapping = read_json(run / "commentary/source-mapping.json")["frames"]
    assert len(mapping) == facts.nb_frames_decoded
    assert [x["source_frame"] for x in mapping[:50]] == list(range(50))
    assert set(x["mode"] for x in mapping) >= {"normal", "overview_hold", "slow", "freeze"}
    media = json.loads(
        subprocess.check_output(
            [
                ffprobe_path(),
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                media_arg(run / "commentary/highlights.mp4"),
            ]
        )
    )
    audio = next(s for s in media["streams"] if s["codec_type"] == "audio")
    assert abs(float(audio["duration"]) - result["duration_s"]) < 0.1
    assert run_detail(run.parent, run.name)["narrated"]["video"] == "commentary/highlights.mp4"
    write_json(run / "commentary/semantic-review.json", {"message_zh": "探索复核发现事实错误。"})
    assert run_detail(run.parent, run.name)["narrated"]["review_zh"] == "探索复核发现事实错误。"
    with pytest.raises(FileExistsError):
        narrated.create(video, run, run / "commentary")


def test_speech_failure_never_publishes_ready_movie(tmp_path, monkeypatch):
    monkeypatch.delenv("GBF_DEEPSEEK_API_KEY", raising=False)
    video, run = fixture_run(tmp_path / "broken")

    def broken(*args):
        raise RuntimeError("speech failed")

    monkeypatch.setattr(narrated, "speech", broken)
    with pytest.raises(RuntimeError, match="speech failed"):
        narrated.create(video, run, run / "commentary")
    manifest = read_json(run / "commentary/manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["video"] is None
    assert run_detail(run.parent, run.name)["narrated"]["video"] is None


def test_model_request_matches_reviewed_bytes_and_redacts_echoed_credential(tmp_path, monkeypatch):
    key = "unit-test-credential-123456"
    monkeypatch.setenv("GBF_DEEPSEEK_API_KEY", key)
    answer = {
        "lines": [
            {"text": "两名球员沿着边线移动。", "frames": [3], "kind": "observation"},
            {"text": "慢放回看，防守球员可能在收窄空间。", "frames": [3], "kind": "hypothesis"},
        ]
    }
    calls = []

    def response(request, timeout):
        calls.append(request)
        assert request.data == (tmp_path / "model-request.json").read_bytes()
        assert request.get_header("Authorization") == "Bearer " + key
        preflight = read_json(tmp_path / "model-preflight.json")
        assert preflight["payload_sha256"] == sha256_file(tmp_path / "model-request.json")
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": json.dumps(answer)}}
                    ],
                    "provider_debug": key,
                }
            ).encode()
        )

    monkeypatch.setattr(narration_model.urllib.request, "urlopen", response)
    result = narration_model.explain(
        [{"frame": 3, "evidence": {"frame": 3}, "full": b"test-image"}], "sprint", tmp_path
    )
    assert result["source"] == "deepseek-flash"
    assert result["accuracy_certified"] is False
    assert len(calls) == 1
    assert all(key not in p.read_text("utf-8") for p in tmp_path.glob("*.json"))
    assert "[REDACTED]" in (tmp_path / "model-response.json").read_text("utf-8")


@pytest.mark.parametrize("raw", [b'{"choices": []}', b"not json", b'{"choices": [{}]}'])
def test_invalid_provider_response_falls_back_without_retry(tmp_path, monkeypatch, raw):
    monkeypatch.setenv("GBF_DEEPSEEK_API_KEY", "unit-test-credential")
    calls = []

    def response(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(raw)

    monkeypatch.setattr(narration_model.urllib.request, "urlopen", response)
    result = narration_model.explain([], "sprint", tmp_path)
    assert len(calls) == 1
    assert result["source"] == "local_replay_guidance"
    assert result["tactical_explanation_generated"] is False


def test_original_audio_only_appears_in_selected_normal_speed_spans(tmp_path):
    import wave

    from gbf.ingest.tools import ffmpeg_path

    silent, source, voice = (tmp_path / name for name in ("picture.mp4", "audio.wav", "voice.wav"))
    write_frames_mp4([np.zeros((32, 64, 3), dtype=np.uint8)] * 50, silent, fps="25/1")
    for path, audible in ((source, True), (voice, False)):
        samples = np.arange(AUDIO_RATE * 2)
        values = (
            np.round(12000 * np.sin(samples * 2 * math.pi * 440 / AUDIO_RATE)).astype("<i2")
            if audible
            else np.zeros(len(samples), dtype="<i2")
        )
        with wave.open(str(path), "wb") as out:
            out.setparams((1, 2, AUDIO_RATE, 0, "NONE", "not compressed"))
            out.writeframes(values.tobytes())
    target = tmp_path / "mixed.mp4"
    narrated.mix_audio(
        silent,
        voice,
        source,
        [
            {"source_start_s": 0.4, "duration_s": 0.5, "output_start_s": 0.2},
            {"source_start_s": 1.1, "duration_s": 0.5, "output_start_s": 1.0},
        ],
        target,
        [],
    )
    raw = subprocess.check_output(
        [
            ffmpeg_path(),
            "-v",
            "error",
            "-i",
            media_arg(target),
            "-map",
            "0:a",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_RATE),
            "-",
        ]
    )
    audio = np.frombuffer(raw, dtype="<f4")

    def rms(start, end):
        return float(np.sqrt(np.mean(audio[int(start * AUDIO_RATE) : int(end * AUDIO_RATE)] ** 2)))

    assert 0.02 < rms(0.3, 0.6) < 0.04
    assert 0.02 < rms(1.1, 1.4) < 0.04
    assert rms(0.8, 0.9) < 0.001
    assert rms(1.7, 1.9) < 0.001
