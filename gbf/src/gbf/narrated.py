"""One narrated MP4 from verified canonical video and an existing analysis run."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import wave
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from gbf.action_context import POLICY, associated_ids, context_window, plan_focus, stabilize_replay
from gbf.camera_motion import BackgroundMotion
from gbf.fonts import find_cjk_fonts, load_font
from gbf.ingest import iter_frames, pixel_sha256, probe
from gbf.ingest.tools import ffmpeg_path, ffprobe_path, media_arg
from gbf.narration_model import TITLES, explain
from gbf.pipeline import WATERMARK, WATERMARK_TAGS
from gbf.player_effects import draw_player_effects, plan_player_effects
from gbf.render import VideoWriter
from gbf.replay_effects import draw_ball_trail, plan_ball_trails, plan_emphasis
from gbf.util.hashing import sha256_file
from gbf.util.io import read_json, write_json

SIZE = (1280, 720)
AUDIO_RATE = 24000
MANIFEST = "manifest.json"

SPEAK = r"""
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
$items=Get-Content -LiteralPath $env:GBF_SPEECH_PLAN -Encoding UTF8 -Raw | ConvertFrom-Json
$s=New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $voices=@($s.GetInstalledVoices() | Where-Object {
    $_.Enabled -and $_.VoiceInfo.Culture.Name -eq 'zh-CN'
  })
  if ($voices.Count -eq 0) { throw 'No installed Chinese speech voice' }
  $s.SelectVoice($voices[0].VoiceInfo.Name)
  $s.Rate=1
  $bits=[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen
  $channel=[System.Speech.AudioFormat.AudioChannel]::Mono
  $format=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(24000,$bits,$channel)
  foreach ($item in $items) {
    $s.SetOutputToWaveFile([string]$item.path,$format)
    $s.Speak([string]$item.text)
    $s.SetOutputToNull()
  }
  [Console]::OutputEncoding=New-Object System.Text.UTF8Encoding
  Write-Output $s.Voice.Name
} finally { $s.Dispose() }
"""


def select_highlights(document: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    count = document["frame_count"]
    valid = []
    for h in document["highlights"]:
        values = [h.get(k) for k in ("start_frame", "peak_frame", "end_frame")]
        if any(type(v) is not int for v in values):
            raise ValueError("Candidate has invalid frame indices")
        if not 0 <= values[0] <= values[1] <= values[2] < count:
            raise ValueError("Candidate lies outside the source video")
        if not isinstance(h.get("id"), str):
            raise ValueError("Candidate has no identifier")
        score = h.get("score", 0)
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Candidate has a nonfinite score")
        valid.append(h)
    selected: list[dict[str, Any]] = []
    for h in sorted(valid, key=lambda x: (-x.get("score", 0), x["peak_frame"], x["id"])):
        # Suppress substantial duplication, keeping adjacent distinct moments.
        duplicate = False
        for old in selected:
            overlap = max(
                0,
                min(h["end_frame"], old["end_frame"])
                - max(h["start_frame"], old["start_frame"])
                + 1,
            )
            shorter = (
                min(h["end_frame"] - h["start_frame"], old["end_frame"] - old["start_frame"]) + 1
            )
            if overlap / shorter > 0.5:
                duplicate = True
        if not duplicate:
            selected.append(h)
        if len(selected) == limit:
            break
    return sorted(selected, key=lambda x: (x["peak_frame"], x["id"]))


def targets(h: dict[str, Any]) -> list[int]:
    pair = h.get("key_pair") if h.get("type") == "duel_tackle" else None
    if isinstance(pair, list) and len(pair) == 2 and pair[0] != pair[1]:
        return pair
    focus = h.get("key_focus_track")
    if focus is not None:
        return [int(focus)]
    return [int(x) for x in h.get("track_ids", [])[:1]]


def continuous_boxes(
    tracks: pd.DataFrame, ids: list[int], first: int, last: int, key: int
) -> dict[int, list[list[float]]]:
    """Only the uninterrupted, unambiguous component containing the key frame."""
    usable: dict[int, list[list[float]]] = {}
    if not ids:
        return usable
    filtered = tracks[tracks.track_id.isin(ids) & tracks.frame.between(first, last)]
    for frame, group in filtered.groupby("frame"):
        if len(group) != len(ids) or set(group.track_id) != set(ids):
            continue
        boxes = []
        for tid in ids:
            row = group[group.track_id == tid].iloc[0]
            box = [float(row[c]) for c in ("x1", "y1", "x2", "y2")]
            if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                break
            boxes.append(box)
        if len(boxes) == len(ids):
            usable[int(frame)] = boxes
    if key not in usable:
        return {}
    lo = hi = key
    while lo - 1 in usable:
        lo -= 1
    while hi + 1 in usable:
        hi += 1
    return {f: usable[f] for f in range(lo, hi + 1)}


def crop_box(boxes: list[list[float]], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    width, height = size
    pad = max(b[3] - b[1] for b in boxes) * 0.5
    left = max(0, math.floor(min(b[0] for b in boxes) - pad))
    top = max(0, math.floor(min(b[1] for b in boxes) - pad))
    right = min(width, math.ceil(max(b[2] for b in boxes) + pad))
    bottom = min(height, math.ceil(max(b[3] for b in boxes) + pad))
    if right <= left or bottom <= top:
        return None
    # Bound enlargement: at least 1/4 source width and height.
    rw, rh = max(right - left, width / 4), max(bottom - top, height / 4)
    left = max(0, min(width - rw, (left + right - rw) / 2))
    top = max(0, min(height - rh, (top + bottom - rh) / 2))
    return (
        math.floor(left),
        math.floor(top),
        min(width, math.ceil(left + rw)),
        min(height, math.ceil(top + rh)),
    )


def jpeg(image: Image.Image, size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    ImageOps.contain(image, size, Image.Resampling.LANCZOS).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def speech(lines: list[str], folder: Path) -> tuple[list[np.ndarray], str]:
    if sys.platform != "win32":
        raise RuntimeError("当前配音需要Windows已安装的中文语音。")
    folder.mkdir()
    items = [{"text": t, "path": str((folder / f"{i}.wav").resolve())} for i, t in enumerate(lines)]
    plan = folder / "plan.json"
    write_json(plan, items)
    env = {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}
    env["GBF_SPEECH_PLAN"] = str(plan.resolve())
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", SPEAK],
        env=env,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise RuntimeError("本机中文配音失败，请检查Windows中文语音是否可用。")
    arrays = []
    for item in items:
        with wave.open(item["path"], "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (
                1,
                2,
                AUDIO_RATE,
            ):
                raise ValueError("Unexpected speech WAV format")
            data = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()
        if not len(data) or not np.any(data):
            raise ValueError("Speech synthesis produced empty or silent audio")
        arrays.append(data)
    return arrays, result.stdout.decode("utf-8", errors="replace").strip()


def schedule(plan, lengths: list[int], fps: Fraction) -> tuple[list[dict], list[tuple[int, int]]]:
    """Place narration by measured sample counts; extend holds instead of cutting speech."""
    chunks = [
        {"kind": s.kind, "frames": list(s.source_frames), "repeat": s.repeat} for s in plan.segments
    ]
    normal = len(chunks[0]["frames"])
    voice_frames = [math.ceil(Fraction(n, AUDIO_RATE) * fps) for n in lengths]
    gap = math.ceil(float(fps) * 0.2)
    pad = max(0, voice_frames[0] + 2 * gap - normal)
    if pad:
        chunks.insert(
            1, {"kind": "overview_hold", "frames": [chunks[0]["frames"][-1]], "repeat": pad}
        )
    replay_start = normal + pad
    current = sum(len(x["frames"]) * x["repeat"] for x in chunks)
    needed = replay_start + gap + voice_frames[1] + gap
    if needed > current:
        freeze = next(x for x in chunks if x["kind"] == "freeze")
        freeze["repeat"] += needed - current
    spans = [
        (gap, gap + voice_frames[0]),
        (replay_start + gap, replay_start + gap + voice_frames[1]),
    ]
    return chunks, spans


@lru_cache(maxsize=12)
def font(size: int):
    return load_font(find_cjk_fonts()[0], size)


def compose(
    raw: np.ndarray,
    boxes: list[list[float]],
    mode: str,
    title: str,
    subtitle: str,
    source_frame: int,
    fps: Fraction,
    ai: bool,
    watermark: str | None = None,
    *,
    focus: dict | None = None,
    trail: dict | None = None,
    actors: dict | None = None,
) -> Image.Image:
    source = Image.fromarray(raw).copy()
    if actors is not None:
        source = draw_player_effects(source, actors, skeleton=mode in {"slow", "freeze"})
    if trail:
        source = draw_ball_trail(source, trail)
    marks = ImageDraw.Draw(source)
    for i, box in enumerate(boxes if actors is None else []):
        marks.rectangle(box, outline="#64e4cf", width=3)
        marks.text((box[0], max(0, box[1] - 25)), chr(65 + i), font=font(23), fill="#64e4cf")
    canvas = Image.new("RGB", SIZE, "#0b1320")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 18), title, font=font(27), fill="#e9eef6")
    label = (
        "原速全景"
        if mode in {"normal", "overview_hold"}
        else "定格观察"
        if mode == "freeze"
        else "慢放回看"
    )
    draw.text((950, 22), label, font=font(21), fill="#64e4cf")
    if watermark:
        draw.text((590, 25), watermark, font=font(18), fill="#f2ca80")
    region = focus["crop"] if focus is not None else crop_box(boxes, source.size)
    if mode in {"normal", "overview_hold"} or region is None:
        view = ImageOps.contain(source, (1220, 540), Image.Resampling.LANCZOS)
        canvas.paste(view, ((1280 - view.width) // 2, 76 + (540 - view.height) // 2))
        if region is None and mode not in {"normal", "overview_hold"}:
            draw.text((30, 586), "局部对象暂不可用，保留全景", font=font(19), fill="#f2ca80")
    else:
        full = ImageOps.contain(source, (726, 490), Image.Resampling.LANCZOS)
        origin = (25, 96 + (490 - full.height) // 2)
        canvas.paste(full, origin)
        scale = full.width / source.width
        draw.rectangle(
            tuple(v * scale + origin[i % 2] for i, v in enumerate(region)),
            outline="#64e4cf",
            width=2,
        )
        detail = ImageOps.contain(source.crop(region), (480, 490), Image.Resampling.LANCZOS)
        canvas.paste(detail, (775 + (480 - detail.width) // 2, 96 + (490 - detail.height) // 2))
        draw.text((30, 64), "全景", font=font(18), fill="#aab9cf")
        draw.text((780, 64), "局部回放", font=font(18), fill="#aab9cf")
    draw.text(
        (30, 622), f"原片 {source_frame / float(fps):05.2f} 秒", font=font(16), fill="#93a4bc"
    )
    draw.text((910, 622), "AI解说 · 待核对" if ai else "回放引导", font=font(16), fill="#93a4bc")
    if trail and trail.get("points"):
        draw.text((1000, 590), "球路跟踪估计", font=font(18), fill="#ffdc82")
    if actors and any(a["edges"] for a in actors["actors"]) and mode in {"slow", "freeze"}:
        draw.text((650, 622), "动作骨架估计", font=font(16), fill="#64e4cf")
    if subtitle:
        lines = [subtitle[i : i + 30] for i in range(0, len(subtitle), 30)]
        for i, line in enumerate(lines):
            width = draw.textlength(line, font=font(26))
            draw.text(((1280 - width) / 2, 651 + i * 30), line, font=font(26), fill="white")
    return canvas


def srt_time(seconds: float) -> str:
    millis = round(seconds * 1000)
    return (
        f"{millis // 3600000:02}:{millis // 60000 % 60:02}:"
        f"{millis // 1000 % 60:02},{millis % 1000:03}"
    )


def mix_audio(
    silent: Path,
    voice: Path,
    original: Path,
    spans: list[dict],
    target: Path,
    restrictions: list[str],
) -> None:
    info = json.loads(
        subprocess.check_output(
            [ffprobe_path(), "-v", "error", "-show_streams", "-of", "json", media_arg(original)]
        )
    )
    has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    command = [
        ffmpeg_path(),
        "-v",
        "error",
        "-nostdin",
        "-i",
        media_arg(silent),
        "-i",
        media_arg(voice),
    ]
    filters = ["[1:a]aresample=48000,volume=1[voice]"]
    labels = ["[voice]"]
    if has_audio:
        command += ["-i", media_arg(original)]
        for i, span in enumerate(spans):
            filters.append(
                f"[2:a]atrim=start={span['source_start_s']}:duration={span['duration_s']},"
                "asetpts=PTS-STARTPTS,volume=0.12,"
                f"adelay={round(span['output_start_s'] * 1000)}:all=1[a{i}]"
            )
            labels.append(f"[a{i}]")
    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=first:normalize=0,alimiter=limit=0.95:level=false[a]"
    )
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-metadata",
        "comment=" + json.dumps({"restrictions": restrictions}),
        media_arg(target),
    ]
    result = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL, check=False)
    if result.returncode or result.stderr.strip():
        raise RuntimeError(
            "成片音频合成失败：" + result.stderr.decode("utf-8", errors="replace")[-500:]
        )


def create(
    video: Path,
    run: Path,
    output: Path,
    *,
    prediction: Path | None = None,
    audio_source: Path | None = None,
    ball_candidates: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("成片输出目录已存在，不会覆盖。")
    print("[commentary] running", flush=True)
    source = probe(video)
    run_doc = read_json(run / "run.json")
    grid = run_doc.get("result", {}).get("video", {})
    if source.sha256 != grid.get("sha256"):
        raise ValueError("视频与分析结果的来源SHA不匹配。")
    fps = Fraction(grid["fps"])
    if source.nb_frames_decoded != grid["frames"] or not source.is_cfr_at(fps):
        raise ValueError("视频与分析结果的帧率/帧数不匹配。")
    if grid["frames"] / float(fps) > 60.001:
        raise ValueError("当前成片模式仅支持60秒以内的视频。")
    prediction = prediction or run / "highlights_final.json"
    document = read_json(prediction)
    if Fraction(document["fps"]) != fps or document["frame_count"] != grid["frames"]:
        raise ValueError("候选片段与视频的帧网格不匹配。")
    selected = select_highlights(document)
    output.mkdir(parents=True)
    restrictions = sorted(
        set(run_doc.get("restrictions", [])) | set(document.get("restrictions", []))
    )
    result: dict[str, Any] = {
        "schema": "gbf.narrated/1",
        "status": "no_highlights",
        "video": None,
        "message_zh": "没有找到可剪辑的精彩候选，未生成虚构的高光。",
        "source_sha256": source.sha256,
        "audio_source_sha256": sha256_file(audio_source or video),
        "fps": str(fps),
        "clips": [],
        "restrictions": restrictions,
        "accuracy_certified": False,
        "inputs_sha256": {str(p): sha256_file(p) for p in [video, run / "run.json", prediction]},
    }
    if not selected:
        write_json(output / MANIFEST, result)
        print("[commentary] skipped", flush=True)
        return result
    try:
        tracks_path, shots_path = run / "refine/tracks.parquet", run / "shots.parquet"
        tracks, shots = pd.read_parquet(tracks_path), pd.read_parquet(shots_path)
        ball_path = run / "ball_track.parquet"
        ball = pd.read_parquet(ball_path) if ball_path.exists() else pd.DataFrame()
        if ball_path.exists():
            result["inputs_sha256"][str(ball_path)] = sha256_file(ball_path)
        poses_path = run / "pose.parquet"
        poses = pd.read_parquet(poses_path) if poses_path.exists() else pd.DataFrame()
        if poses_path.exists():
            result["inputs_sha256"][str(poses_path)] = sha256_file(poses_path)
        candidates_path = (
            ball_candidates if ball_candidates is not None else run / "ball_candidates.parquet"
        )
        candidates = pd.read_parquet(candidates_path) if candidates_path.exists() else None
        if ball_candidates is not None and candidates is None:
            raise FileNotFoundError(candidates_path)
        if candidates is not None:
            result["inputs_sha256"][str(candidates_path)] = sha256_file(candidates_path)
        result["inputs_sha256"].update(
            {str(p): sha256_file(p) for p in [tracks_path, shots_path, run / "cuts.json"]}
        )
        cuts = read_json(run / "cuts.json")["cuts"]
        all_cues, mapping, voice_parts, original_spans = [], [], [], []
        output_cursor = 0
        silent = output / "picture.mp4"
        with VideoWriter(
            silent, *SIZE, fps, preset="veryfast", restrictions=restrictions
        ) as writer:
            for index, h in enumerate(selected):
                folder = output / f"part-{index + 1}"
                folder.mkdir()
                matching = shots[
                    (shots.start_frame <= h["peak_frame"]) & (shots.end_frame >= h["peak_frame"])
                ]
                if len(matching) != 1:
                    raise ValueError("关键时刻不属于唯一镜头。")
                shot = matching.iloc[0]
                context = context_window(h, shot, grid["frames"], fps, cuts)
                first, last = context["start_frame"], context["end_frame"]
                emphasis, plan = plan_emphasis(h, first, last, fps, cuts=cuts)
                key = emphasis["key_frame"]
                ids = associated_ids(h)
                focus = plan_focus(
                    tracks,
                    ball,
                    ids,
                    first,
                    last,
                    (grid["width"], grid["height"]),
                    candidates=candidates,
                )
                player_effects = plan_player_effects(
                    h, tracks, poses, first, last, key, (grid["width"], grid["height"]), fps
                )
                replay_frames = {f for s in plan.segments[1:] for f in s.source_frames}
                stabilize_replay(focus, replay_frames, fps, (grid["width"], grid["height"]))
                write_json(
                    folder / "focus-plan.json",
                    {
                        "policy": POLICY,
                        "context": context,
                        "frames": [{"frame": f, **entry} for f, entry in focus.items()],
                    },
                )
                sample_frames = set(
                    np.linspace(first, last, min(24, last - first + 1), dtype=int).tolist()
                )
                if emphasis["event"]:
                    sample_frames.update(
                        key + round(offset * fps)
                        for offset in (-0.16, 0, 0.16, 0.5, 1, 2)
                        if first <= key + round(offset * fps) <= last
                    )
                stored, samples = {}, []
                camera = BackgroundMotion()
                camera_motion = {}
                ball_rows = (
                    {int(f): g.iloc[0] for f, g in ball.groupby("frame") if len(g) == 1}
                    if "frame" in ball
                    else {}
                )
                # Decode only once for model samples, then stream again for render.
                for frame, raw in iter_frames(video, first, last + 1, fps=fps):
                    ball_row = ball_rows.get(frame)
                    camera_motion[frame] = camera.update(
                        frame,
                        raw,
                        [p["bbox_xyxy"] for p in focus[frame]["scene_people"]],
                        (ball_row.u, ball_row.v, ball_row.radius_px)
                        if ball_row is not None
                        else None,
                        cut=frame in cuts,
                    )
                    if frame in replay_frames or frame == last:
                        stored[frame] = raw
                    if frame in sample_frames:
                        image = Image.fromarray(raw)
                        region = focus[frame]["crop"]
                        samples.append(
                            {
                                "frame": frame,
                                "full": jpeg(image, (1280, 720)),
                                "detail": jpeg(image.crop(region), (640, 480)) if region else None,
                                "evidence": {
                                    "frame": frame,
                                    "t": round((frame - first) / float(fps), 3),
                                    "boxes": focus[frame]["boxes"],
                                    "focus_track_ids": focus[frame]["track_ids"],
                                    "ball": focus[frame]["ball"],
                                    "focus_reason": focus[frame]["reason"],
                                    "scene_people": focus[frame]["scene_people"],
                                    "local_available": region is not None,
                                    "replay_event_estimate": emphasis["event"],
                                },
                            }
                        )
                trails = plan_ball_trails(
                    ball,
                    first,
                    last,
                    fps,
                    (grid["width"], grid["height"]),
                    cuts=cuts,
                    candidates=candidates,
                    camera_motion=camera_motion,
                )
                write_json(
                    folder / "effects-plan.json",
                    {
                        "emphasis": emphasis,
                        "trail": {
                            "history_s": 0.28,
                            "coordinates": "current_frame_image_pixels",
                            "camera_alignment": "causal_background_similarity",
                            "interpolated_points": False,
                            "identity_verified": False,
                        },
                        "frames": [
                            {"frame": f, **e, "player_effects": player_effects[f]}
                            for f, e in trails.items()
                        ],
                    },
                )
                explanation = explain(
                    samples,
                    h["type"],
                    folder,
                    local_available=any(
                        focus[frame]["crop"] is not None for frame in replay_frames
                    ),
                )
                lines = explanation["lines"]
                waves, voice_name = speech(lines, folder / "speech")
                chunks, spans = schedule(plan, [len(w) for w in waves], fps)
                local_cues = []
                for text, wave_data, (begin, _) in zip(lines, waves, spans, strict=True):
                    start_sample = round(Fraction(output_cursor + begin, 1) / fps * AUDIO_RATE)
                    voice_parts.append((start_sample, wave_data))
                    cue = {
                        "text": text,
                        "start_frame": output_cursor + begin,
                        "start_s": start_sample / AUDIO_RATE,
                        "end_s": (start_sample + len(wave_data)) / AUDIO_RATE,
                        "source": explanation["source"],
                        "part": index + 1,
                    }
                    all_cues.append(cue)
                    local_cues.append(cue)
                normal_end = output_cursor + len(chunks[0]["frames"])
                original_spans.append(
                    {
                        "source_start_s": first / float(fps),
                        "duration_s": (last - first + 1) / float(fps),
                        "output_start_s": output_cursor / float(fps),
                    }
                )
                first_output = output_cursor
                title = "关键出球回放" if emphasis["event"] else TITLES.get(h["type"], "比赛回放")
                for chunk in chunks:
                    cached_key, cached_image = None, None
                    if chunk["kind"] == "normal":
                        iterator = iter_frames(video, first, last + 1, fps=fps)
                    else:
                        iterator = ((f, stored[f]) for f in chunk["frames"])
                    for frame, raw in iterator:
                        frame_focus = focus[frame]
                        boxes = frame_focus["boxes"]
                        shown_crop = (
                            frame_focus["crop"] if chunk["kind"] in {"slow", "freeze"} else None
                        )
                        source_hash = pixel_sha256(raw)
                        shown_trail = trails[frame] if chunk["kind"] != "overview_hold" else None
                        actor_plan = player_effects[frame]
                        show_actors = chunk["kind"] in {"slow", "freeze"} or (
                            chunk["kind"] == "normal" and actor_plan["near_key"]
                        )
                        shown_actors = actor_plan if show_actors else {"actors": []}
                        for _ in range(chunk["repeat"]):
                            time_s = output_cursor / float(fps)
                            subtitle = next(
                                (
                                    c["text"]
                                    for c in local_cues
                                    if c["start_s"] <= time_s < c["end_s"]
                                ),
                                "",
                            )
                            image_key = (frame, subtitle)
                            if image_key != cached_key:
                                cached_image = compose(
                                    raw,
                                    boxes,
                                    chunk["kind"],
                                    f"回合 {index + 1}  /  {title}",
                                    subtitle,
                                    frame,
                                    fps,
                                    explanation["source"] != "local_replay_guidance",
                                    WATERMARK if WATERMARK_TAGS & set(restrictions) else None,
                                    focus=frame_focus,
                                    trail=shown_trail,
                                    actors=shown_actors,
                                )
                                cached_key = image_key
                            image = cached_image
                            writer.write(np.asarray(image))
                            if (
                                frame == key
                                and chunk["kind"] == "freeze"
                                and not (folder / "poster.jpg").exists()
                            ):
                                image.save(folder / "poster.jpg", quality=90)
                            mapping.append(
                                {
                                    "output_frame": output_cursor,
                                    "source_frame": frame,
                                    "source_rgb_sha256": source_hash,
                                    "part": index + 1,
                                    "mode": chunk["kind"],
                                    "crop": shown_crop,
                                    "local_available": shown_crop is not None,
                                    "planned_crop": frame_focus["crop"],
                                    "focus_track_ids": frame_focus["track_ids"]
                                    if shown_crop
                                    else [],
                                    "focus_reason": frame_focus["reason"],
                                    "replay_component": frame_focus["replay_component"],
                                    "ball_trail": shown_trail,
                                    "player_effects": shown_actors,
                                    "skeleton_shown": chunk["kind"] in {"slow", "freeze"}
                                    and any(a["edges"] for a in shown_actors["actors"]),
                                }
                            )
                            output_cursor += 1
                result["clips"].append(
                    {
                        "highlight_id": h["id"],
                        "start_frame": first,
                        "key_frame": key,
                        "end_frame": last,
                        "output_start_frame": first_output,
                        "output_end_frame": output_cursor - 1,
                        "normal_end_frame": normal_end,
                        "target_ids": ids,
                        "local_source_frames": sum(
                            focus[f]["crop"] is not None for f in replay_frames
                        ),
                        "planned_context_crop_source_frames": sum(
                            e["crop"] is not None for e in focus.values()
                        ),
                        "focus_policy": POLICY,
                        "emphasis": emphasis,
                        "effect_track_ids": player_effects[key]["selected_track_ids"],
                        "model_sample_frames": sorted(sample_frames),
                        **context,
                        "explanation_source": explanation["source"],
                        "lines": lines,
                        "voice": voice_name,
                    }
                )
                print(f"[commentary] part {index + 1}/{len(selected)} rendered", flush=True)
        n_samples = math.ceil(Fraction(output_cursor, 1) / fps * AUDIO_RATE)
        mixed = np.zeros(n_samples, dtype="<i2")
        for start, data in voice_parts:
            if start + len(data) > n_samples:
                raise ValueError("旁白长于画面，拒绝截断。")
            mixed[start : start + len(data)] = data
        with wave.open(str(output / "narration.wav"), "wb") as stream:
            stream.setparams((1, 2, AUDIO_RATE, 0, "NONE", "not compressed"))
            stream.writeframes(mixed.tobytes())
        srt = (
            "\n\n".join(
                f"{i}\n{srt_time(c['start_s'])} --> {srt_time(c['end_s'])}\n{c['text']}"
                for i, c in enumerate(all_cues, 1)
            )
            + "\n"
        )
        (output / "subtitles.srt").write_text(srt, encoding="utf-8")
        write_json(output / "subtitles.json", {"cues": all_cues})
        write_json(output / "source-mapping.json", {"fps": str(fps), "frames": mapping})
        temporary = output / "highlights.pending.mp4"
        mix_audio(
            silent,
            output / "narration.wav",
            audio_source or video,
            original_spans,
            temporary,
            restrictions,
        )
        facts = probe(temporary)
        if facts.nb_frames_decoded != output_cursor or not facts.is_cfr_at(fps):
            raise ValueError("成片帧数与计划不匹配。")
        subprocess.run(
            [
                ffmpeg_path(),
                "-v",
                "error",
                "-nostdin",
                "-i",
                media_arg(temporary),
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        final = output / "highlights.mp4"
        temporary.rename(final)
        result.update(
            status="ready",
            video=final.name,
            message_zh="精彩解说视频已生成。",
            frames=output_cursor,
            duration_s=output_cursor / float(fps),
            subtitles="subtitles.srt",
            explanation_sources=sorted({c["source"] for c in all_cues}),
            files_sha256={
                p.name: sha256_file(p)
                for p in [
                    final,
                    output / "subtitles.srt",
                    output / "source-mapping.json",
                    output / "narration.wav",
                ]
            },
        )
        write_json(output / MANIFEST, result)
        print("[commentary] done", flush=True)
        return result
    except Exception as exc:
        result.update(
            status="failed",
            video=None,
            message_zh="成片生成失败，已有分析结果保留。",
            error_type=type(exc).__name__,
        )
        write_json(output / MANIFEST, result)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render one narrated highlight video")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prediction", type=Path)
    args = parser.parse_args(argv)
    try:
        result = create(args.video, args.run, args.out, prediction=args.prediction)
        print(result["message_zh"], flush=True)
        return 0
    except Exception as exc:
        print(f"highlights: failed: 成片生成失败（{type(exc).__name__}）", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
