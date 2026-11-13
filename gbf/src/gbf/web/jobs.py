"""Analyses started from the local web page (``POST /api/jobs``, docs/design/web.md).

:class:`JobRequest` checks what the page sends before anything runs: every input path must
be a complete local path (no network or device path, which would make Windows contact
another machine before the check even starts) to an existing file or folder of the right
kind; the output name is a new folder name of ``[A-Za-z0-9._-]`` in the runs folder;
team names and licence identifiers are checked as the pipeline would check them. Every
problem found is listed, in Chinese.

:class:`JobManager` runs ``python -P -m gbf highlights ...`` (:func:`highlights_command`,
the interpreter of the server; tests pass a fake command) as a subprocess, one job at a
time: a second request while one runs is refused, and so is an output folder that already
exists (nothing is overwritten). The command's output goes to
``<runs dir>/.gbf-web/jobs/<id>.log`` and the job's record, with the request and the exit
code, to ``<id>.json`` beside it (a hidden folder, never listed as a run). The records let
a later start of the server show earlier jobs and fill the form with the last request.

Progress and results come from the log, read incrementally: the stage lines of
gbf.pipeline (``[stage] running ...``, ``done in``, ``skipped``, ``kept``), the problem
lists of a refused or failed run (``<where>: N problem(s)`` followed by ``  - ...`` lines)
and the last :data:`LOG_TAIL_LINES` lines. Exit codes are those of gbf.pipeline.main: 0
done, 2 input refused (nothing ran), 1 a stage failed; an interrupt (Ctrl+C, or the
server stopping the job) is reported as such. :func:`translate_problem` gives the known
problems in Chinese and leaves the others as they are.
"""

from __future__ import annotations

import ast
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import gbf
from gbf.pipeline import STAGES, check_out_dir
from gbf.plugins.licenses import LicenseExpressionError, checked_allowlist
from gbf.util.io import read_json, write_json
from gbf.web.automatic import AUTO_STAGES, runtime_status
from gbf.web.processes import WindowsProcessTree
from gbf.web.runs import RUN_JSON, segment_problem

JOBS_DIR = Path(".gbf-web", "jobs")
"""Where job logs and records go, relative to the runs folder."""

OUT_NAME = re.compile(r"[A-Za-z0-9._-]{1,80}")
JOB_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
LOG_TAIL_LINES = 40
MAX_PATH_CHARS = 1000
MAX_TEAM_NAME_CHARS = 40
RECENT_JOBS = 20
READ_CHUNK = 1 << 20

STATUS_CONTROL_C_EXIT = 0xC000013A
"""Exit code of a Windows process ended by Ctrl+C (Python's KeyboardInterrupt)."""

STAGE_NAMES_ZH: Mapping[str, str] = MappingProxyType(
    {
        "import": "导入插件导出",
        "shots": "切镜和镜头类型",
        "refine": "清理轨迹、重拟合标定",
        "kinematics": "球场位置和速度",
        "ball": "球的轨迹和触球",
        "audio": "现场声音",
        "highlights": "找出高光",
        "pose": "2D 姿态",
        "lift3d": "3D 参考骨架",
        "moments": "关键时刻",
        "measure": "测量记录",
        "render": "渲染片段",
        "report": "生成报告",
    }
)
"""Chinese names of the gbf.pipeline stages, for the progress view."""

FIELD_NAMES_ZH: Mapping[str, str] = MappingProxyType(
    {
        "video": "视频文件",
        "audio_source": "原声文件",
        "sngs_dir": "插件导出文件夹",
        "ball": "球检测文件",
        "labels": "标注文件",
        "team_names": "球队名",
        "allow_licenses": "接受的许可证",
        "out_name": "输出名",
        "mode": "分析方式",
    }
)
"""The fields of a job request and their names on the page."""

_STAGE_LINE = re.compile(r"^\[(?P<stage>[a-z0-9_]+)\] (?P<what>running|done|skipped|kept)\b")
_PROBLEM_HEADER = re.compile(r"^(?:highlights: failed: )?[\w.-]+: \d+ problem\(s\)$")
_PROBLEM_ITEM = re.compile(r"^ {2}- (?P<text>.*)$")
_FAILED = re.compile(r"^highlights: failed: (?P<text>.*)$")
_FLAG_ZH = {
    "video": "视频文件",
    "audio-source": "原声文件",
    "ball-raw": "球检测文件",
    "labels": "标注文件",
}
_LICENCE_INPUT_ZH = {"plugin": "插件导出", "keypoints": "球场关键点", "ball": "球检测"}


def _licence_text(match: re.Match[str]) -> str:
    ids = match["ids"]
    what = _LICENCE_INPUT_ZH.get(match["what"], match["what"])
    return (
        f"{what}的输入（{match['file']}）声明的许可证 {match['expr']} 不在默认接受的宽松"
        f"许可证里（未接受：{ids}）。确认可以接受后，把 {ids} 填进“接受的许可证”，"
        "再开始分析。"
    )


_PROBLEM_RULES: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], str]], ...] = (
    (
        re.compile(r"^--(?P<flag>video|audio-source|ball-raw|labels): not a file: (?P<path>.*)$"),
        lambda m: f"{_FLAG_ZH[m['flag']]}不存在：{m['path']}",
    ),
    (
        re.compile(r"^--sngs: not a directory: (?P<path>.*)$"),
        lambda m: f"插件导出文件夹不存在：{m['path']}",
    ),
    (
        re.compile(r"^--sngs: (?P<file>\S+) is missing: "),
        lambda m: f"插件导出缺少 {m['file']}：没有它不知道许可证和数据限制，所以不运行。",
    ),
    (
        re.compile(r"^(?P<file>\S+) cannot be read as JSON: "),
        lambda m: f"{m['file']} 读不成 JSON：不知道许可证和数据限制，所以不运行。",
    ),
    (
        re.compile(
            r"^(?P<what>\w+) input \((?P<file>[^)]*)\): license (?P<expr>.+) is not satisfied "
            r"by the allowed licenses \(not allowed: (?P<ids>[^)]*)\); accept it explicitly "
            r"with --allow-license <id>$"
        ),
        _licence_text,
    ),
    (
        re.compile(r"^--allow-license: (?P<detail>.*)$"),
        lambda m: f"“接受的许可证”写法有误：{m['detail']}",
    ),
    (
        re.compile(r"^--out .* is inside the gbf checkout .*; run outputs never go into"),
        lambda m: (
            "输出文件夹在 gbf 代码仓库里：分析结果从不放进仓库，请换一个仓库外的运行目录"
            "（--runs-dir）。"
        ),
    ),
    (
        re.compile(r"^--out .* is inside the git work tree (?P<repo>.*); run outputs must stay"),
        lambda m: (
            f"输出文件夹在 git 仓库（{m['repo']}）里：分析结果必须放在所有仓库之外，"
            "请换一个仓库外的运行目录（--runs-dir）。"
        ),
    ),
    (
        re.compile(
            r"^the plugin export covers (?P<a>\d+) frames but the video decodes to (?P<b>\d+)"
        ),
        lambda m: (
            f"插件导出覆盖 {m['a']} 帧，视频却解码出 {m['b']} 帧：这份导出不属于这个视频"
            "（帧号必须一一对应）。"
        ),
    ),
    (
        re.compile(r"^the video is not constant-frame-rate"),
        lambda m: "视频不是恒定帧率：请先用 python -m gbf ingest 规范化，帧号才可靠。",
    ),
    (
        re.compile(r"^the video has no usable frame rate"),
        lambda m: "读不出视频的帧率。",
    ),
    (
        re.compile(r"^(?P<file>.+) declares no restrictions list$"),
        lambda m: (
            f"{m['file']} 没有声明数据限制（restrictions 列表），无法确定结果的使用范围，"
            "所以不运行。"
        ),
    ),
    (
        re.compile(
            r"^(?P<file>[^:]+receipt\.json): \d+: (?P<tag>.+?) is not one of "
            r"\[.*'nda'.*'no_share'.*\]$"
        ),
        lambda m: (
            f"{m['file']} 含有不认识的限制标签“{_unquoted(m['tag'])}”：无法确定结果的使用范围，"
            "所以不运行。请检查输入回执的 restrictions 列表。"
        ),
    ),
    (
        re.compile(r"^input missing: (?P<name>.*)$"),
        lambda m: f"缺少输入文件：{m['name']}",
    ),
    (
        re.compile(r"\[Errno 2\] No such file or directory: (?P<path>.+)$"),
        lambda m: (
            f"找不到文件：{_unquoted(m['path'])}（输入不完整，例如插件导出缺少 "
            "plugin-receipt.json）"
        ),
    ),
)


def _unquoted(text: str) -> str:
    """A path as Python's OSError message quotes it ('C:\\\\x'), without the quoting."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(value, str):
            return value
    return text


def translate_problem(text: str) -> str:
    """A pipeline problem in Chinese when it is a known one, else the text as it is."""
    for pattern, render in _PROBLEM_RULES:
        match = pattern.search(text)
        if match is not None:
            return render(match)
    return text


def _utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- the request


class JobRequestError(ValueError):
    """A job request the page may not start; ``problems`` lists every problem, in Chinese."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _path_text(value: Any) -> str | None:
    """The text of a path field: stripped, without the quotes Windows' "copy as path" adds;
    None when empty."""
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return text or None


def network_path(text: str) -> bool:
    """True for a network or device path: two separators at the start, in any mix of ``\\``
    and ``/``, which Windows reads alike (``\\\\host\\share``, ``//host/share``,
    ``/\\host\\share``, ``\\\\?\\C:\\...``, ``\\\\.\\pipe\\...``), or the NT object-manager
    prefix ``\\??\\`` (``\\??\\UNC\\host\\share``; ``/??/`` too, which pathlib turns into it).
    Asking the file system anything about one already contacts that machine or opens that
    device."""
    separators = text.replace("/", "\\")
    return separators.startswith("\\\\") or separators.startswith("\\??\\")


def _local_path(name: str, value: Any, kind: str, problems: list[str]) -> Path | None:
    """The checked path of field ``name`` (``kind`` "file" or "dir"), or None (absent, or a
    problem was added). Network and device paths (:func:`network_path`) are refused before
    the file system is asked anything, so that no check reaches another machine through
    them; a mapped network drive, or a link on a local disk to a share, is not detected."""
    label = FIELD_NAMES_ZH[name]
    if value is None:
        return None
    if not isinstance(value, str):
        problems.append(f"{label}：应当是文字（一个路径）。")
        return None
    text = _path_text(value)
    if text is None:
        return None
    if len(text) > MAX_PATH_CHARS:
        problems.append(f"{label}：路径太长。")
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        problems.append(f"{label}：路径里有控制字符。")
        return None
    if network_path(text):
        problems.append(
            f"{label}：不能用网络路径或设备路径（如 \\\\、// 或 \\??\\ 开头），只能用本机路径。"
        )
        return None
    path = Path(text)
    if not path.is_absolute():
        problems.append(f"{label}：要写完整路径（Windows 上从盘符开始），现在是：{text}")
        return None
    if kind == "dir" and not path.is_dir():
        problems.append(f"{label}：找不到这个文件夹：{text}")
        return None
    if kind == "file" and not path.is_file():
        problems.append(f"{label}：找不到这个文件：{text}")
        return None
    return path


def _team_names(value: Any, problems: list[str]) -> dict[str, str] | None:
    label = FIELD_NAMES_ZH["team_names"]
    if value is None:
        return None
    if not isinstance(value, Mapping):
        problems.append(f'{label}：格式应为 {{"left": 左侧球队, "right": 右侧球队}}。')
        return None
    extra = sorted(str(key) for key in value if key not in ("left", "right"))
    if extra:
        problems.append(f"{label}：只有 left 和 right 两侧，多了：{', '.join(extra)}")
    names: dict[str, str] = {}
    for side in ("left", "right"):
        raw = value.get(side)
        if raw is None:
            continue
        if not isinstance(raw, str):
            problems.append(f"{label}（{side}）：应当是文字。")
            continue
        text = raw.strip()
        if not text:
            continue
        if len(text) > MAX_TEAM_NAME_CHARS:
            problems.append(f"{label}（{side}）：最长 {MAX_TEAM_NAME_CHARS} 个字。")
        elif "," in text or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
            problems.append(f"{label}（{side}）：不能含英文逗号或控制字符。")
        else:
            names[side] = text
    return names or None


def _licences(value: Any, problems: list[str]) -> tuple[str, ...]:
    label = FIELD_NAMES_ZH["allow_licenses"]
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        problems.append(f"{label}：应当是许可证标识的列表。")
        return ()
    items = [item.strip() for item in value if item.strip()]
    bad = [item for item in items if not _single_licence(item)]
    if bad:
        problems.append(
            f"{label}：每一项必须是一个 SPDX 许可证标识（例如 GPL-3.0-only），不能有空格或 "
            f"AND、OR 这样的组合；这些不是：{'、'.join(bad)}"
        )
        return ()
    return tuple(checked_allowlist(items))


def _single_licence(item: str) -> bool:
    try:
        checked_allowlist([item])
    except LicenseExpressionError:
        return False
    return True


def _out_name(value: Any, problems: list[str]) -> str | None:
    label = FIELD_NAMES_ZH["out_name"]
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{label}：必填。")
        return None
    name = value.strip()
    if not OUT_NAME.fullmatch(name):
        problems.append(f"{label}：只能用英文字母、数字和 . _ -，最多 80 个字符。")
        return None
    if segment_problem(name) is not None:
        problems.append(f"{label}：不能以 . 开头或结尾，也不能是 NUL、CON 这样的 Windows 设备名。")
        return None
    return name


@dataclass(frozen=True)
class JobRequest:
    """A checked request for one ``python -m gbf highlights`` run (see :meth:`from_json`).

    Attributes:
        video, audio_source, sngs_dir, ball: the inputs (``audio_source`` defaults to the
            video itself).
        out_name: the new run folder's name inside the runs folder.
        labels: optional labels to compare the highlights with.
        team_names: optional {"left": name, "right": name}.
        allow_licenses: licence identifiers the owner accepts (``--allow-license``).
    """

    video: Path
    audio_source: Path
    sngs_dir: Path | None
    ball: Path | None
    out_name: str
    labels: Path | None = None
    team_names: Mapping[str, str] | None = None
    allow_licenses: tuple[str, ...] = ()
    mode: str = "manual"

    @classmethod
    def from_json(cls, data: Any) -> JobRequest:
        """Check a request as the page sends it: a JSON object with ``video``, ``sngs_dir``,
        ``ball`` and ``out_name`` (required), ``audio_source``, ``labels``, ``team_names``
        and ``allow_licenses`` (optional). Raises JobRequestError listing every problem."""
        if not isinstance(data, Mapping):
            raise JobRequestError(["请求必须是一个 JSON 对象。"])
        problems: list[str] = []
        mode = data.get("mode", "manual")
        if mode not in ("manual", "auto"):
            problems.append("分析方式：只能是 auto 或 manual。")
        automatic = mode == "auto"
        unknown = sorted(str(key) for key in data if key not in FIELD_NAMES_ZH)
        if unknown:
            problems.append(f"不认识的字段：{', '.join(unknown)}")
        paths: dict[str, Path | None] = {}
        for name, kind, required in (
            ("video", "file", True),
            ("audio_source", "file", False),
            ("sngs_dir", "dir", not automatic),
            ("ball", "file", not automatic),
            ("labels", "file", False),
        ):
            value = data.get(name)
            empty = value is None or (isinstance(value, str) and _path_text(value) is None)
            if empty and required:
                problems.append(f"{FIELD_NAMES_ZH[name]}：必填。")
                paths[name] = None
            else:
                paths[name] = _local_path(name, value, kind, problems)
        team_names = _team_names(data.get("team_names"), problems)
        licences = _licences(data.get("allow_licenses"), problems)
        out_name = _out_name(data.get("out_name"), problems)
        if automatic and any(paths.get(key) is not None for key in ("sngs_dir", "ball", "labels")):
            problems.append("自动模式不接受已有预测或标注，请在高级选项中选择手动导入。")
        if problems:
            raise JobRequestError(problems)
        video, sngs, ball = paths["video"], paths["sngs_dir"], paths["ball"]
        if video is None or out_name is None:  # pragma: no cover
            raise JobRequestError(["请求不完整。"])
        return cls(
            video=video,
            audio_source=paths["audio_source"] or video,
            sngs_dir=sngs,
            ball=ball,
            out_name=out_name,
            labels=paths["labels"],
            team_names=team_names,
            allow_licenses=licences,
            mode=mode,
        )

    def arguments(self, out_dir: Path) -> list[str]:
        """The arguments of ``python -m gbf highlights`` for this request (``--name=value``,
        so that no value is ever read as an option)."""
        args = [
            f"--video={self.video}",
            f"--audio-source={self.audio_source}",
            f"--sngs={self.sngs_dir}",
            f"--ball={self.ball}",
            f"--out={out_dir}",
        ]
        if self.labels is not None:
            args.append(f"--labels={self.labels}")
        if self.team_names:
            joined = ",".join(f"{side}={name}" for side, name in sorted(self.team_names.items()))
            args.append(f"--team-names={joined}")
        args.extend(f"--allow-license={licence}" for licence in self.allow_licenses)
        return args

    def as_json(self) -> dict[str, Any]:
        """The request as the page sends it (also kept in the job record)."""
        result = {
            "video": str(self.video),
            "audio_source": str(self.audio_source),
            "sngs_dir": str(self.sngs_dir) if self.sngs_dir else None,
            "ball": str(self.ball) if self.ball else None,
            "out_name": self.out_name,
            "labels": None if self.labels is None else str(self.labels),
            "team_names": dict(self.team_names) if self.team_names else None,
            "allow_licenses": list(self.allow_licenses),
        }
        if self.mode == "auto":
            result["mode"] = self.mode
        return result


# --------------------------------------------------------------------------- jobs


class JobRefused(Exception):
    """A job that cannot start: ``status`` is the HTTP status (409 another job runs or the
    output folder exists, 400 the output folder is not allowed, 500 it could not start) and
    ``problems`` the Chinese reasons."""

    def __init__(self, status: int, problems: Sequence[str]) -> None:
        self.status = status
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def highlights_command() -> list[str]:
    """``python -P -m gbf highlights`` with the server's interpreter (``-P``: the working
    directory never shadows a module)."""
    return [sys.executable, "-P", "-m", "gbf", "highlights"]


def child_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment of a job: ``environ`` with UTF-8 output, unbuffered, and the source
    folder of this gbf on PYTHONPATH (unless it is an installed site-packages), so that the
    job runs the same code as the server."""
    env = dict(environ)
    for name in ("__PYVENV_LAUNCHER__", "PYTHONHOME"):
        env.pop(name, None)
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    source = Path(gbf.__file__).resolve().parent.parent
    if source.name.lower() != "site-packages":
        parts = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
        env["PYTHONPATH"] = os.pathsep.join([str(source), *parts])
    return env


def classify(exit_code: int | None, *, stopped: bool = False) -> str:
    """The status of a job from its exit code: running (None), succeeded (0), refused (2),
    interrupted (Ctrl+C, a signal, or stopped by the server) or failed (1, anything else)."""
    if stopped and exit_code not in (None, 0):
        return "interrupted"
    if exit_code is None:
        return "running"
    if exit_code == 0:
        return "succeeded"
    if exit_code == 2:
        return "refused"
    if exit_code < 0 or exit_code in (STATUS_CONTROL_C_EXIT, 130):
        return "interrupted"
    return "failed"


@dataclass
class Job:
    """One analysis: its request, files, process and what its log said so far."""

    id: str
    request: JobRequest
    log_path: Path
    record_path: Path
    argv: tuple[str, ...]
    created_at: str
    process: subprocess.Popen[bytes] | None = None
    exit_code: int | None = None
    finished_at: str | None = None
    stopped: bool = False
    known_status: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    waiter: threading.Thread | None = field(default=None, repr=False)
    process_tree: WindowsProcessTree | None = field(default=None, repr=False)
    _offset: int = 0
    _partial: bytes = b""
    _stages: dict[str, str] = field(default_factory=dict)
    _tail: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_TAIL_LINES))
    _problems: list[str] = field(default_factory=list)
    _in_list: bool = False
    _failure: str | None = None

    @property
    def out_name(self) -> str:
        return self.request.out_name

    @property
    def status(self) -> str:
        """running, succeeded, refused, failed, interrupted, or unknown (a job of an earlier
        server whose end was never recorded)."""
        if self.exit_code is None and self.process is None:
            return self.known_status or "unknown"
        return classify(self.exit_code, stopped=self.stopped)

    @property
    def running(self) -> bool:
        return self.process is not None and self.exit_code is None

    @property
    def stage_names(self) -> tuple[str, ...]:
        return (*AUTO_STAGES, *STAGES) if self.request.mode == "auto" else tuple(STAGES)

    # ---------------------------------------------------------------- the log

    def _line(self, line: str) -> None:
        self._tail.append(line)
        stage = _STAGE_LINE.match(line)
        if stage is not None:
            self._stages[stage["stage"]] = stage["what"]
        if _PROBLEM_HEADER.match(line):
            self._in_list = True
            return
        item = _PROBLEM_ITEM.match(line)
        if self._in_list and item is not None:
            self._problems.append(item["text"])
            return
        self._in_list = False
        failed = _FAILED.match(line)
        if failed is not None:
            self._failure = failed["text"]

    def read_log(self, *, final: bool = False) -> None:
        """Read what the log gained since the last call (the caller holds ``lock``)."""
        try:
            with open(self.log_path, "rb") as handle:
                handle.seek(self._offset)
                while chunk := handle.read(READ_CHUNK):
                    self._offset += len(chunk)
                    lines = (self._partial + chunk).split(b"\n")
                    self._partial = lines.pop()
                    for raw in lines:
                        self._line(raw.decode("utf-8", "replace").rstrip("\r"))
        except OSError:
            return
        if final and self._partial:
            self._line(self._partial.decode("utf-8", "replace").rstrip("\r"))
            self._partial = b""

    def _stage_states(self, status: str) -> list[dict[str, str]]:
        states = []
        for name in self.stage_names:
            seen = self._stages.get(name)
            state = {"done": "done", "skipped": "skipped", "kept": "kept"}.get(seen or "")
            if seen == "running":
                state = "running" if status == "running" else "failed"
            states.append(
                {
                    "name": name,
                    "name_zh": AUTO_STAGES.get(name, STAGE_NAMES_ZH.get(name, name)),
                    "state": state or "pending",
                }
            )
        return states

    def _explain(self, status: str, current: str | None) -> tuple[str, list[str]]:
        problems = [translate_problem(p) for p in self._problems]
        if not problems and self._failure and not _PROBLEM_HEADER.match(self._failure):
            problems = [translate_problem(self._failure)]
        stage_zh = AUTO_STAGES.get(current or "", STAGE_NAMES_ZH.get(current or "", current or ""))
        if status == "running":
            return (f"正在分析：{stage_zh}……" if current else "正在启动分析……"), []
        if status == "succeeded":
            return "分析完成。", []
        if status == "refused":
            return "输入被拒绝，分析没有开始。原因：", problems
        if status == "failed":
            where = f"在“{stage_zh}”这一步" if current else ""
            return f"分析{where}失败了（退出码 {self.exit_code}）。原因：", problems
        if status == "interrupted":
            return (
                f"分析已中断，未完成的文件保留在 {self.out_name} 对应的工作目录中。"
                "重新分析请使用新的输出名。"
            ), []
        return "这个分析在上次服务停止前还没有结束，结果不明。", []

    def snapshot(self, runs_dir: Path) -> dict[str, Any]:
        """The job as ``GET /api/jobs/<id>`` reports it."""
        with self.lock:
            self.read_log(final=not self.running)
            status = self.status
            current = next(
                (
                    name
                    for name in reversed(self.stage_names)
                    if self._stages.get(name) == "running"
                ),
                None,
            )
            message, problems = self._explain(status, current)
            return {
                "id": self.id,
                "out_name": self.out_name,
                "status": status,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "exit_code": self.exit_code,
                "message_zh": message,
                "problems_zh": problems,
                "current_stage": current,
                "stages": self._stage_states(status),
                "log_tail": list(self._tail),
                "log_file": (JOBS_DIR / self.log_path.name).as_posix(),
                "run_ready": (runs_dir / self.out_name / RUN_JSON).is_file(),
                "request": self.request.as_json(),
            }

    def summary(self) -> dict[str, Any]:
        """A short row for the job list."""
        return {
            "id": self.id,
            "out_name": self.out_name,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "request": self.request.as_json(),
        }

    def record(self) -> dict[str, Any]:
        return {
            "schema": "gbf.web-job/1",
            **self.summary(),
            "argv": list(self.argv),
            "log": self.log_path.name,
        }


class JobManager:
    """Starts and follows analyses for one runs folder; at most one runs at a time.

    ``command`` is the program and leading arguments (default :func:`highlights_command`);
    the request's arguments follow it. ``environ`` is the base environment of a job
    (default os.environ, see :func:`child_environment`). ``repo_root`` and ``git_ceiling``
    go to gbf.pipeline.check_out_dir, which refuses output folders inside a git work tree
    before a job starts (tests point them at their own folders).
    """

    def __init__(
        self,
        runs_dir: Path,
        command: Sequence[str] | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        repo_root: Path | None = None,
        git_ceiling: Path | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir).resolve()
        self.jobs_dir = self.runs_dir / JOBS_DIR
        self.command = list(command) if command is not None else highlights_command()
        self.environ = dict(os.environ if environ is None else environ)
        self.repo_root = repo_root
        self.git_ceiling = git_ceiling
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._current: Job | None = None

    def start(self, request: JobRequest) -> Job:
        """Start ``request`` unless a job is running or its output folder exists (JobRefused)."""
        out_dir = self.runs_dir / request.out_name
        with self._lock:
            current = self._current
            if current is not None and current.running:
                raise JobRefused(
                    409,
                    [
                        f"已经有一个分析在运行（输出 {current.out_name}）。同一时间只能运行"
                        "一个分析，请等它结束后再开始新的。"
                    ],
                )
            work = self.runs_dir / ".gbf-web" / "work" / request.out_name
            if out_dir.exists() or out_dir.is_symlink() or work.exists():
                raise JobRefused(
                    409,
                    [
                        f"输出文件夹 {request.out_name} 已经存在：不会覆盖已有的结果，"
                        "请换一个输出名。"
                    ],
                )
            problems = check_out_dir(
                out_dir, repo_root=self.repo_root, git_ceiling=self.git_ceiling
            )
            if problems:
                raise JobRefused(400, [translate_problem(problem) for problem in problems])
            if request.mode == "auto":
                status = runtime_status(self.environ)
                if not status["available"]:
                    raise JobRefused(400, [status["message_zh"]])
            job = self._launch(request, out_dir)
            self._jobs[job.id] = job
            self._current = job
        return job

    def _launch(self, request: JobRequest, out_dir: Path) -> Job:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        job_id = f"{stamp}-{secrets.token_hex(3)}"
        if request.mode == "auto":
            argv = (
                sys.executable,
                "-P",
                "-m",
                "gbf.web.automatic",
                f"--video={request.video}",
                f"--out={out_dir}",
                f"--config={self.environ['GBF_AUTO_CONFIG']}",
                "--options="
                + json.dumps(
                    {"team_names": request.team_names, "allow_licenses": request.allow_licenses}
                ),
            )
        else:
            argv = (*self.command, *request.arguments(out_dir))
        job = Job(
            id=job_id,
            request=request,
            log_path=self.jobs_dir / f"{job_id}.log",
            record_path=self.jobs_dir / f"{job_id}.json",
            argv=argv,
            created_at=_utc_now(),
        )
        try:
            with open(job.log_path, "wb") as log:
                job.process = subprocess.Popen(
                    list(argv),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=self.runs_dir,
                    env=child_environment(self.environ),
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP | 0x4)
                    if os.name == "nt"
                    else 0,  # CREATE_SUSPENDED: assign before children exist
                    start_new_session=os.name != "nt",
                )
                if os.name == "nt":
                    job.process_tree = WindowsProcessTree(job.process)
                    job.process_tree.resume(job.process)
        except OSError as exc:
            if job.process_tree is not None:
                job.process_tree.close()
            if job.process is not None and job.process.poll() is None:
                job.process.kill()
                job.process.wait(timeout=30)
            raise JobRefused(500, [f"无法启动分析：{exc}"]) from exc
        self._write_record(job)
        job.waiter = threading.Thread(target=self._wait, args=(job,), daemon=True)
        job.waiter.start()
        return job

    def _wait(self, job: Job) -> None:
        if job.process is None:  # pragma: no cover - set before the thread starts
            return
        code = job.process.wait()
        with job.lock:
            if job.process_tree is not None:
                job.process_tree.close()
            elif os.name != "nt":
                import signal

                try:
                    os.killpg(job.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            job.exit_code = code
            job.finished_at = _utc_now()
        self._write_record(job)

    def _write_record(self, job: Job) -> None:
        partial = job.record_path.with_name(job.record_path.name + ".partial")
        try:
            write_json(partial, job.record())
            os.replace(partial, job.record_path)
        except OSError:
            partial.unlink(missing_ok=True)

    def _load(self, job_id: str) -> Job | None:
        """A job of an earlier server, from its record (None if there is none). The request
        is taken as it was recorded (it was checked when the job started; an input may have
        moved since), so the job's log and result can still be seen."""
        record_path = self.jobs_dir / f"{job_id}.json"
        try:
            record = read_json(record_path)
            raw = record["request"]
            request = JobRequest(
                video=Path(raw["video"]),
                audio_source=Path(raw["audio_source"]),
                sngs_dir=Path(raw["sngs_dir"]) if raw.get("sngs_dir") else None,
                ball=Path(raw["ball"]) if raw.get("ball") else None,
                out_name=str(raw["out_name"]),
                labels=Path(raw["labels"]) if raw.get("labels") else None,
                team_names=raw.get("team_names") or None,
                allow_licenses=tuple(raw.get("allow_licenses") or ()),
                mode=raw.get("mode", "manual"),
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
        return self._job_from_record(job_id, record, request, record_path)

    def _job_from_record(
        self, job_id: str, record: Mapping[str, Any], request: JobRequest, record_path: Path
    ) -> Job:
        code = record.get("exit_code")
        status = record.get("status")
        return Job(
            id=job_id,
            request=request,
            log_path=record_path.with_suffix(".log"),
            record_path=record_path,
            argv=tuple(str(a) for a in record.get("argv") or ()),
            created_at=str(record.get("created_at") or ""),
            exit_code=code if isinstance(code, int) and not isinstance(code, bool) else None,
            finished_at=record.get("finished_at")
            if isinstance(record.get("finished_at"), str)
            else None,
            stopped=status == "interrupted",
            known_status=None
            if status == "running"
            else status
            if isinstance(status, str)
            else None,
        )

    def get(self, job_id: str) -> Job | None:
        """The job ``job_id`` of this server or of an earlier one (None when unknown)."""
        if not JOB_ID.fullmatch(job_id):
            return None
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            job = self._load(job_id)
            if job is not None:
                with self._lock:
                    job = self._jobs.setdefault(job_id, job)
        return job

    def recent(self, limit: int = RECENT_JOBS) -> list[dict[str, Any]]:
        """The newest jobs (this server's and earlier ones'), newest first."""
        ids: set[str] = set()
        try:
            ids.update(
                path.stem for path in self.jobs_dir.glob("*.json") if JOB_ID.fullmatch(path.stem)
            )
        except OSError:
            pass
        with self._lock:
            ids.update(self._jobs)
        rows = []
        for job_id in sorted(ids, reverse=True)[:limit]:
            job = self.get(job_id)
            if job is not None:
                rows.append(job.summary())
        return rows

    def current(self) -> Job | None:
        with self._lock:
            return self._current if self._current is not None and self._current.running else None

    def cancel(self, job_id: str) -> Job | None:
        """Stop only a process tree started by this manager; preserve its logs and files."""
        job = self.get(job_id)
        if job is None or job.process is None or job.process.poll() is not None:
            return job
        with job.lock:
            job.stopped = True
            if job.process_tree is not None:
                job.process_tree.terminate()
        if job.process_tree is not None:
            pass  # the owned Windows job already ended the entire tree
        elif os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(job.process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if result.returncode and job.process.poll() is None:
                raise OSError("未能结束分析进程树，请查看本机进程。")
        else:
            import signal

            try:
                os.killpg(job.process.pid, signal.SIGTERM)
                job.process.wait(timeout=5)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(job.process.pid, signal.SIGKILL)
        job.process.wait(timeout=30)
        if job.waiter is not None:
            job.waiter.join(timeout=30)
        return job

    def close(self, grace_s: float = 10.0) -> None:
        """Stop every job this manager started that still runs: wait up to ``grace_s``
        seconds (Ctrl+C in the server's console reaches a job too, which then records its
        own failure), then terminate it; return once their records are written."""
        with self._lock:
            started = [job for job in self._jobs.values() if job.process is not None]
        deadline = time.monotonic() + max(grace_s, 0.0)
        for job in started:
            if job.process is None or job.process.poll() is not None:
                continue
            job.stopped = True
            try:
                job.process.wait(timeout=max(deadline - time.monotonic(), 0.0))
            except subprocess.TimeoutExpired:
                self.cancel(job.id)
        for job in started:
            if job.waiter is not None:
                job.waiter.join(timeout=30)


__all__ = [
    "FIELD_NAMES_ZH",
    "JOBS_DIR",
    "STAGE_NAMES_ZH",
    "Job",
    "JobManager",
    "JobRefused",
    "JobRequest",
    "JobRequestError",
    "child_environment",
    "classify",
    "highlights_command",
    "network_path",
    "translate_problem",
]
