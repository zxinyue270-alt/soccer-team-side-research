"""Video-only orchestration; third-party predictors run in an isolated process.

The trusted local configuration selects the executable. A browser can supply only a
video and a result name, never a command. No plugin modules are imported here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from gbf.ingest import cfr_filter, iter_frames, probe
from gbf.ingest.tools import ffmpeg_path, media_arg, run_tool
from gbf.util.hashing import sha256_file
from gbf.util.io import read_json, write_json

AUTO_STAGES = {
    "prepare": "读取视频与准备画面",
    "calib_keypoints": "识别球场关键点",
    "calib_lines": "识别球场线条",
    "tracking": "识别球员、追踪与分队",
    "export": "核对帧号与导出场上位置",
    "detect_ball": "识别足球",
}
MAX_UPLOAD_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class RuntimeConfig:
    path: Path
    command: tuple[str, ...]
    models_dir: Path
    allow_licenses: tuple[str, ...]
    max_duration_s: int = 60

    @classmethod
    def load(cls, path: Path) -> RuntimeConfig:
        from gbf.web.jobs import network_path

        if network_path(str(path)) or not path.is_absolute():
            raise ValueError("自动识别配置必须是本机完整路径。")
        raw = read_json(path)
        command = raw.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item and "\x00" not in item for item in command)
        ):
            raise ValueError("自动识别配置缺少有效的 command 列表。")
        executable = Path(command[0])
        models = Path(raw.get("models_dir", ""))
        if (
            network_path(str(executable))
            or not executable.is_absolute()
            or not executable.is_file()
        ):
            raise ValueError("自动识别解释器不存在，请检查本机运行环境。")
        if network_path(str(models)) or not models.is_absolute() or not models.is_dir():
            raise ValueError("姿态模型目录不存在，请检查本机运行环境。")
        required_files = raw.get("required_files", [])
        if not isinstance(required_files, list) or not all(
            isinstance(item, str) for item in required_files
        ):
            raise ValueError("自动识别配置中的 required_files 列表无效。")
        for item in required_files:
            required = Path(item)
            if network_path(item) or not required.is_absolute() or not required.is_file():
                raise ValueError(f"自动识别缺少本机运行文件：{required.name}")
        from gbf.plugins.licenses import checked_allowlist

        licenses = raw.get("allow_licenses", [])
        if not isinstance(licenses, list) or not all(isinstance(x, str) for x in licenses):
            raise ValueError("自动识别配置中的许可证列表无效。")
        checked_allowlist(licenses)
        limit = raw.get("max_duration_s", 60)
        if type(limit) is not int or not 1 <= limit <= 60:
            raise ValueError("短片 CPU 模式的时长上限必须为 1～60 秒。")
        return cls(path, tuple(command), models, tuple(licenses), limit)


def runtime_status(environ: Mapping[str, str]) -> dict[str, Any]:
    text = environ.get("GBF_AUTO_CONFIG", "")
    try:
        if not text:
            raise ValueError("尚未配置自动识别运行环境；仍可在高级选项中导入已有结果。")
        config = RuntimeConfig.load(Path(text))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return {"available": False, "message_zh": str(exc)}
    return {
        "available": True,
        "max_duration_s": config.max_duration_s,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "message_zh": "本机 CPU 短片模式：30 秒视频可能需要数小时；不上传至外部服务。",
    }


def clean_environment() -> dict[str, str]:
    env = dict(os.environ)
    for name in ("__PYVENV_LAUNCHER__", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    return env


def prepare_video(source: Path, work: Path, max_duration_s: int) -> dict[str, Any]:
    """Normalize the image grid if necessary, export canonical frames, and hash them."""
    print("[prepare] running", flush=True)
    facts = probe(source)
    duration = facts.duration_s
    if duration is None or duration <= 0 or duration > max_duration_s + 0.001:
        raise ValueError(f"当前 CPU 模式只支持 {max_duration_s} 秒以内的视频，请先截取短片。")
    count_estimate = int(duration * 25 + 1)
    required = count_estimate * 1920 * 1080 * 3 + 512 * 1024**2
    if shutil.disk_usage(work).free < required:
        raise ValueError("本机磁盘空间不足，无法安全保存逐帧画面和分析结果。")
    conversion = None
    video = source
    if not (
        facts.is_cfr_at("25/1")
        and (facts.frame_width, facts.frame_height) == (1920, 1080)
        and facts.sample_aspect_ratio in (None, "1:1")
        and facts.rotation_deg == 0
    ):
        video = work / "normalized.mp4"
        filters = (
            cfr_filter("25/1")
            + ",scale=1920:1080:force_original_aspect_ratio=decrease:force_divisible_by=2"
            + ",pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
        run_tool(
            [
                ffmpeg_path(),
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-n",
                "-i",
                media_arg(source),
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
                filters,
                "-fps_mode",
                "cfr",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "12",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "2",
                media_arg(video),
            ],
            what="automatic video normalization",
        )
        converted = probe(video)
        if not converted.is_cfr_at("25/1") or (converted.frame_width, converted.frame_height) != (
            1920,
            1080,
        ):
            raise ValueError("视频转换后的帧率或尺寸核验失败。")
        conversion = {"filter": filters, "output": converted.to_dict()}
    frames = work / "canonical"
    frames.mkdir()
    manifest = []
    for index, rgb in iter_frames(video):
        path = frames / f"{index:06d}.png"
        Image.fromarray(rgb).save(path)
        import hashlib

        with Image.open(path) as saved:
            if saved.tobytes() != rgb.tobytes():
                raise ValueError(f"第 {index} 帧的无损导出核验失败。")
        manifest.append(
            {
                "index": index,
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "png_sha256": sha256_file(path),
            }
        )
        if index % 100 == 0:
            print(f"准备画面 {index + 1} 帧", flush=True)
    if not manifest or len(manifest) > max_duration_s * 25:
        raise ValueError("视频帧数为空或超过短片模式上限。")
    record = {
        "schema": "gbf.automatic-input/1",
        "source": str(source),
        "source_probe": facts.to_dict(),
        "video": str(video),
        "frames_dir": str(frames),
        "frames": len(manifest),
        "fps": "25/1",
        "width": 1920,
        "height": 1080,
        "conversion": conversion,
        "frame_hashes": manifest,
        "ground_truth_used": False,
    }
    write_json(work / "input.json", record)
    print("[prepare] done", flush=True)
    return record


def run(config: RuntimeConfig, source: Path, out: Path, options: dict[str, Any]) -> int:
    from gbf.pipeline import check_out_dir
    from gbf.pipeline import main as highlights_main

    if out.exists() or out.is_symlink():
        raise ValueError("输出目录已经存在，不会覆盖。")
    problems = check_out_dir(out)
    if problems:
        raise ValueError("; ".join(problems))
    work = out.parent / ".gbf-web" / "work" / out.name
    work.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = prepare_video(source, work, config.max_duration_s)
    plugin = work / "predictions"
    command = [*config.command, "--input", str(work / "input.json"), "--output", str(plugin)]
    write_json(
        work / "prediction-command.json",
        {"argv": command, "config_sha256": sha256_file(config.path)},
    )
    code = subprocess.call(command, env=clean_environment(), stdin=subprocess.DEVNULL)
    if code:
        print(f"highlights: failed: 自动识别失败（退出码 {code}），详情见上面的日志。", flush=True)
        return 1
    for name in (
        "detections.jsonl",
        "calib.jsonl",
        "plugin-receipt.json",
        "ball_raw.jsonl",
        "ball-receipt.json",
        "frame-verification.json",
    ):
        if not (plugin / name).is_file():
            raise ValueError(f"自动识别没有生成必需的结果：{name}")
    frame_check = read_json(plugin / "frame-verification.json")
    if frame_check.get("passed") is not True or frame_check.get("frames") != record["frames"]:
        raise ValueError("自动识别的逐帧输入核验失败，拒绝生成报告。")
    args = [
        f"--video={record['video']}",
        f"--audio-source={source}",
        f"--sngs={plugin}",
        f"--ball={plugin / 'ball_raw.jsonl'}",
        f"--out={out}",
        f"--models-dir={config.models_dir}",
    ]
    licenses = sorted(set(config.allow_licenses) | set(options.get("allow_licenses") or ()))
    args.extend(f"--allow-license={item}" for item in licenses)
    teams = options.get("team_names")
    if teams:
        args.append("--team-names=" + ",".join(f"{side}={name}" for side, name in teams.items()))
    code = highlights_main(args)
    if out.is_dir():
        write_json(
            out / "automatic.json",
            {
                "schema": "gbf.automatic-run/1",
                "source_probe": record["source_probe"],
                "input_manifest": str(work / "input.json"),
                "conversion": record["conversion"],
                "frame_verification": frame_check,
                "config_sha256": sha256_file(config.path),
                "wall_s": time.monotonic() - started,
                "exit_code": code,
                "accuracy_evaluated": False,
            },
        )
    return code


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Local video-only analysis")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    try:
        return run(RuntimeConfig.load(args.config), args.video, args.out, json.loads(args.options))
    except Exception as exc:
        print(f"highlights: failed: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
