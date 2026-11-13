"""``python -m gbf highlights``: from a broadcast clip to analysed highlight moments (M1).

The pipeline finds the highlight moments of a clip and analyses them on the ORIGINAL
frames (docs/design/m1-highlights.md); 3D is only a supporting layer. Each stage writes its
artifacts under ``--out`` and a stage receipt ``receipts/<stage>.json`` (schema
``urn:gbf:schema:receipt``, kind "run"): the sha256 of every input, the parameters, the
code version, the wall time and the sha256 of every output.

========== ======================================================= ===========================
stage      what it does                                            artifacts
========== ======================================================= ===========================
import     validate the plugin-sngs export (gbf.sngs_import)       import/tracks.parquet,
                                                                   import/calib.parquet,
                                                                   import/import.json
shots      hard cuts of the video and the shot grammar             cuts.json, shots.parquet
kinematics smoothed positions, speeds and segment peaks            kinematics.parquet,
                                                                   segments.parquet
ball       ball track on the original frames, strike events and    ball_track.parquet,
           contact events (gbf.ball, DEFAULT and CONTACT params)   ball_events.parquet,
                                                                   ball_contacts.parquet
audio      excitement of the original sound (gbf.audio)            audio.parquet
highlights cues per frame and the highlight moments                signals.parquet,
                                                                   highlights.json
pose       2D pose of the involved players inside the highlight    pose.parquet
           windows, only the frames needed (gbf.pose)
lift3d     calibrated height-prior skeleton on pitch frames        lift3d.parquet
measure    evidence records (gbf.measure)                          measurements.jsonl
render     per highlight: the window at normal speed with the      clips/<id>.mp4 (+ manifest),
           original audio, slow motion around the key frame and a  keyframes/<id>.png,
           freeze card; the whole clip annotated (gbf.render)      clips/full.mp4 (+ manifest)
report     report.html with the score timeline (gbf.report)        report.html, timeline*.png
========== ======================================================= ===========================

Re-running
    A stage is skipped when its receipt's key -- the sha256 of the stage name, the sha256 of
    every input file, the stage's parameters and a fingerprint of the code: every .py file
    of the gbf package this code runs from (gbf.gate and gbf.web left out, they never run
    in a stage), so any code change re-runs every stage -- equals the current key, and
    every output the receipt lists still has its recorded sha256. Measure also hashes
    registries/metrics.toml, where its tiers come from. ``--force`` runs the selected
    stages anyway. Stages left out of ``--stages`` are never run: their outputs must exist.

Conventions and rules
    Frames are canonical CFR indices of ``--video`` (ADR 0003): the video must be CFR, and
    its decoded frame count must equal the plugin export's ``video_frames``. ``--audio-source``
    is the file that carries the original sound (the canonical re-encode may have none);
    its audio time 0 is matched with canonical frame 0 (``audio_offset_s`` moves it).
    Pitch frames, the only frames whose metres feed measurements, are the ``ev_pitch``
    frames of signals.parquet (fresh calibration with rep_err <= 8 px inside a wide shot).
    Every artifact carries the restrictions of the run: the plugin's tags plus
    ``--restrictions`` (default ``no_share``, because a broadcast clip is private material,
    ADR 0002); renders get the watermark 私有素材 · 禁止分享 when no_share or nda is among
    them. The output directory must lie outside every git work tree, the gbf checkout
    included. Receipts and run.json hold file names and run-relative paths only.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import gbf
from gbf import audio as audio_mod
from gbf import ball as ball_mod
from gbf import calib_refine as calib_refine_mod
from gbf import contracts
from gbf import enhance as enhance_mod
from gbf import fonts as fonts_mod
from gbf import highlights as highlights_mod
from gbf import kinematics as kinematics_mod
from gbf import label_eval as label_eval_mod
from gbf import lift3d as lift_mod
from gbf import measure as measure_mod
from gbf import moments as moments_mod
from gbf import pose as pose_mod
from gbf import render as render_mod
from gbf import report as report_mod
from gbf import role_check as role_check_mod
from gbf import shots_grammar as shots_mod
from gbf import side_check as side_check_mod
from gbf import sngs_import as import_mod
from gbf import team_check as team_check_mod
from gbf import track_tools as track_tools_mod
from gbf.ingest import cuts as cuts_mod
from gbf.ingest import format_rate, iter_frames, parse_rate, probe
from gbf.ingest.tools import ffmpeg_version
from gbf.plugins import licenses as licenses_mod
from gbf.util import paths
from gbf.util.hashing import canonical_sha256, sha256_file
from gbf.util.io import read_json, write_json

PRODUCER = "gbf.pipeline"

STAGES: tuple[str, ...] = (
    "import",
    "shots",
    "refine",
    "kinematics",
    "ball",
    "audio",
    "highlights",
    "pose",
    "lift3d",
    "moments",
    "measure",
    "render",
    "report",
)
"""Stage names in run order."""

TRACKS = "refine/tracks.parquet"
"""Track table every stage after ``refine`` reads (ids split at cuts, refitted positions)."""
CALIB = "refine/calib.parquet"
"""Calibration every stage after ``refine`` reads."""
FINAL = "highlights_final.json"
"""Highlights after key moments and ranking; measure, render and report read this one."""

POSE_MODELS: Mapping[str, str | None] = {
    "x": "rtmpose-x-halpe26-384x288",
    "m": "rtmpose-m-halpe26-256x192",
    "none": None,
}
"""``--pose-model`` choices: the pinned gbf.models ids, or no pose stage output."""

DEFAULT_RESTRICTIONS: tuple[str, ...] = ("no_share",)
"""Tags every run carries on top of its inputs' tags: a broadcast clip is private material
(ADR 0002). ``--restrictions`` adds to them; it never removes one."""

POSE_MODEL_RESTRICTIONS: tuple[str, ...] = ("research_nc",)
"""Tags of every run whose pose comes from the pinned RTMPose weights (registries/models.toml,
rtmpose-halpe26: the licences of their training data are not verified)."""

BALL_RECEIPT_NAME = "ball-receipt.json"
"""Optional receipt next to the ball file: the detector's licence expression ("license") and
the tags of its output ("restrictions"). Without one the ball input adds no_share."""

CARD_MAX_ROWS = 6
"""Rows on a highlight's freeze-frame data card: its key moments first, then measurements."""

CARD_MOMENT_ROWS = 4
"""At most this many of the card's rows are key moments."""

WATERMARK = "私有素材 · 禁止分享"
WATERMARK_TAGS = frozenset({"no_share", "nda"})

KINEMATICS_PARAMS: Mapping[str, Any] = {
    "sg_window_frames": 11,
    "sg_poly": 2,
    "max_gap_frames": 3,
    "min_segment_frames": 25,
    "max_speed_mps": 12.0,
    "max_common_step_m": kinematics_mod.DEFAULT_MAX_COMMON_STEP_M,
    "min_common_players": 4,
    "outlier_radius_frames": 5,
    "outlier_threshold_m": kinematics_mod.DEFAULT_OUTLIER_THRESHOLD_M,
    "max_box_height_change": kinematics_mod.DEFAULT_MAX_BOX_HEIGHT_CHANGE,
    "max_box_width_change": kinematics_mod.DEFAULT_MAX_BOX_WIDTH_CHANGE,
    "jump_threshold_m": kinematics_mod.DEFAULT_JUMP_THRESHOLD_M,
    "jump_factor": kinematics_mod.DEFAULT_JUMP_FACTOR,
    "jump_window_frames": kinematics_mod.DEFAULT_JUMP_WINDOW_FRAMES,
    "max_acceleration_mps2": kinematics_mod.DEFAULT_MAX_ACCELERATION_MPS2,
    "speed_chord_s": kinematics_mod.DEFAULT_SPEED_CHORD_S,
}
"""kinematics_with_rejections settings (the arguments of compute_kinematics and
rejected_positions), stated explicitly (CONTRIBUTING: library defaults change)."""

POSE_RANGE_MERGE_GAP = 25
"""Requested pose frames closer than this are decoded in one pass (frames in between are
decoded and dropped, which is cheaper than restarting ffmpeg)."""

FULL_CLIP_ID = "full"
"""File stem of the whole annotated clip under clips/."""

RESTRICTIONS_KEY = b"gbf.restrictions"
PRODUCER_KEY = b"gbf.producer"

SOURCE_LEFT_OUT: tuple[str, ...] = ("gate", "web")
"""Subpackages of gbf outside the source fingerprint: the quality gate and the local web
page, which no stage runs."""

_SOURCE_DIR = Path(__file__).resolve().parent
"""The gbf package this code runs from, whose source every stage key fingerprints."""

METRICS_REGISTRY_INPUT = "registries/metrics.toml"
"""Input name of the metric registry in the measure stage's receipt (gbf.measure reads the
tiers from it)."""


class PipelineError(RuntimeError):
    """The pipeline cannot start or a stage failed. ``problems`` lists every reason."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{where}: {len(self.problems)} problem(s)\n  - {joined}")


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class PipelineConfig:
    """Inputs and settings of one run.

    Attributes:
        video: the canonical CFR video (frames are its decode indices).
        audio_source: the file that carries the original audio (may equal ``video``).
        sngs_dir: the plugin-sngs export (detections.jsonl, calib.jsonl,
            plugin-receipt.json: readable JSON with a restrictions list, else the input is
            refused).
        ball_raw: ball_raw.jsonl of the external ball detector (a last line without a
            newline, from a file still being written, is dropped and reported).
        out_dir: run directory; refused inside any git work tree.
        team_names: {"left": name, "right": name} from provider or user metadata (never
            inferred from colours); default 左侧球队 / 右侧球队. Neither is used when
            gbf.side_check cannot confirm the plugin's left/right names: the texts
            then say 甲队 / 乙队.
        max_highlights: most highlights to keep.
        min_separation_s: smallest time between two highlight peaks, seconds.
        pose_model: "x" (rtmpose-x), "m" (rtmpose-m) or "none".
        models_dir: where gbf.models keeps the weights (default gbf.models' own default).
        stages: the stages that may run; others must already have their outputs.
        force: run the selected stages even when their inputs did not change.
        restrictions: tags every artifact carries on top of its inputs' tags (default
            no_share).
        allow_licenses: licence identifiers the user accepts for inputs whose receipts
            declare a licence outside the permissive allowlist (ADR 0001 section 4;
            NONE and NOASSERTION never pass implicitly).
        full_clip: also render the whole clip with the overlay.
        audio_offset_s: audio time of canonical frame 0 in the audio source (seconds).
        rep_err_max_px: pitch-usable calibration limit for the import, pixels.
        refine_calibration: refit the calibration from the plugin's pitch keypoints when
            the export has them (gbf.calib_refine); False keeps the plugin's per-frame fits.
        labels: optional labels of the clip (gbf.label_eval); the report then compares
            the highlights with them.
        pose_estimator: a ready gbf.pose.PoseEstimator (tests pass one with a fake runner);
            None builds the pinned model named by ``pose_model``.
        repo_root, git_ceiling: for the work-tree check: the gbf checkout to refuse (default
            the one this code runs from) and a directory where the search for an enclosing
            ``.git`` stops (tests keep their files inside the checkout's ignored .local/).
        log: progress callback (default: print to stdout).
    """

    video: Path
    audio_source: Path
    sngs_dir: Path
    ball_raw: Path
    out_dir: Path
    team_names: Mapping[str, str] | None = None
    max_highlights: int = 6
    min_separation_s: float = 4.0
    pose_model: str = "x"
    models_dir: Path | None = None
    stages: tuple[str, ...] = STAGES
    force: bool = False
    restrictions: tuple[str, ...] = DEFAULT_RESTRICTIONS
    allow_licenses: tuple[str, ...] = ()
    full_clip: bool = True
    audio_offset_s: float = 0.0
    rep_err_max_px: float = import_mod.DEFAULT_REP_ERR_MAX_PX
    refine_calibration: bool = True
    labels: Path | None = None
    pose_estimator: Any = None
    repo_root: Path | None = None
    git_ceiling: Path | None = None
    log: Callable[[str], None] | None = None


@dataclass(frozen=True)
class StageResult:
    """What happened to one stage: "ran", "skipped" (up to date) or "kept" (not selected,
    outputs reused); its wall time in seconds and its run-relative receipt and outputs."""

    name: str
    status: str
    wall_s: float
    receipt: str
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class RunResult:
    """A finished run: the run directory, every stage and the highlights document."""

    out_dir: Path
    stages: tuple[StageResult, ...]
    highlights: Mapping[str, Any]
    run_json: Path


# --------------------------------------------------------------------------- small helpers


def _norm(path: Path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _is_within(child: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath([_norm(child), _norm(parent)]) == _norm(parent)
    except ValueError:  # different drives
        return False


def check_out_dir(
    out_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    git_ceiling: str | Path | None = None,
) -> list[str]:
    """Problems with writing the run into ``out_dir`` (empty when it is allowed).

    Refused: the gbf checkout (``repo_root``, default the checkout this code runs from) and
    any directory below a ``.git`` entry, searching upwards until ``git_ceiling`` (exclusive).
    """
    resolved = Path(os.path.abspath(out_dir)).resolve()
    problems: list[str] = []
    repos: list[Path] = []
    if repo_root is not None:
        repos.append(Path(repo_root).resolve())
    else:
        try:
            repos.append(paths.repo_root())
        except paths.RepoRootNotFoundError:
            pass
    for repo in repos:
        if _is_within(resolved, repo):
            problems.append(
                f"--out {resolved} is inside the gbf checkout {repo}; run outputs never go "
                "into the repository (ADR 0002)"
            )
    ceiling = Path(git_ceiling).resolve() if git_ceiling is not None else None
    for candidate in [resolved, *resolved.parents]:
        if ceiling is not None and (
            _norm(candidate) == _norm(ceiling) or not _is_within(candidate, ceiling)
        ):
            break
        if (candidate / ".git").exists():
            if not any(_norm(candidate) == _norm(repo) for repo in repos):
                problems.append(
                    f"--out {resolved} is inside the git work tree {candidate}; run outputs "
                    "must stay outside every repository"
                )
            break
    return problems


def _git(root: Path, *args: str) -> str | None:
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"  # never take optional locks: others may use the index
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=False,
            env=env,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.decode("utf-8", "replace") if done.returncode == 0 else None


def code_version() -> dict[str, Any]:
    """{"gbf_version", "git_commit" ("unknown" outside a checkout), "git_dirty"}."""
    commit, dirty = "unknown", True
    try:
        root: Path | None = paths.repo_root()
    except paths.RepoRootNotFoundError:
        root = None
    if root is not None:
        head = _git(root, "rev-parse", "HEAD")
        if head is not None and re.fullmatch(r"[0-9a-f]{40}", head.strip()):
            commit = head.strip()
            status = _git(root, "status", "--porcelain=v1")
            dirty = status is None or bool(status.strip())
    return {"gbf_version": gbf.__version__, "git_commit": commit, "git_dirty": dirty}


def _utc_now() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None and epoch.isascii() and epoch.isdigit():
        moment = datetime.fromtimestamp(int(epoch), tz=UTC)
    else:
        moment = datetime.now(tz=UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_fingerprint(package: Path) -> str:
    """sha256 over every .py file of the gbf package at ``package``, by its path inside the
    package and the sha256 of its bytes, :data:`SOURCE_LEFT_OUT` left out. Every stage key
    holds it, so an edit anywhere in the code the pipeline runs (a helper, a constant, a
    module no stage names) re-runs every stage instead of keeping outputs of older code."""
    files = {}
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if relative.parts[0] not in SOURCE_LEFT_OUT:
            files[relative.as_posix()] = sha256_file(path)
    return canonical_sha256(files)


def _fingerprint(function: Callable[..., Any], parts: Sequence[Any]) -> str:
    """sha256 over a stage function's source and the code it names: whole source files for
    modules, the source of single functions or classes otherwise. The run's source
    fingerprint (:func:`_source_fingerprint`) covers every other module and helper."""
    digest = hashlib.sha256()
    digest.update(inspect.getsource(function).encode("utf-8"))
    for part in parts:
        if isinstance(part, ModuleType):
            source = getattr(part, "__file__", None)
            if source is None:
                continue
            digest.update(part.__name__.encode("utf-8"))
            digest.update(Path(source).read_bytes())
        else:
            digest.update(f"{part.__module__}.{part.__qualname__}".encode())
            digest.update(inspect.getsource(part).encode("utf-8"))
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Plain JSON (no numpy scalars, NaN as None) for receipts and parameters."""
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Fraction):
        return format_rate(value)
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=str) if isinstance(value, set | frozenset) else items
    return str(value)


def write_parquet(frame: pd.DataFrame, path: Path, restrictions: Sequence[str]) -> None:
    """Write ``frame`` (no index) with the run's restrictions in the schema metadata.

    The file is written under a temporary name and moved into place.
    """
    table = pa.Table.from_pandas(frame, preserve_index=False)
    _write_table(table, path, restrictions)


def _write_table(table: pa.Table, path: Path, restrictions: Sequence[str]) -> None:
    metadata = dict(table.schema.metadata or {})
    metadata[RESTRICTIONS_KEY] = json.dumps(sorted(set(restrictions))).encode("utf-8")
    metadata.setdefault(PRODUCER_KEY, PRODUCER.encode("utf-8"))  # keep another producer's
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    pq.write_table(table.replace_schema_metadata(metadata), partial)
    os.replace(partial, path)


def read_restrictions(path: str | Path) -> list[str]:
    """Restriction tags stored in a parquet file's metadata by :func:`write_parquet`."""
    metadata = pq.read_schema(Path(path)).metadata or {}
    raw = metadata.get(RESTRICTIONS_KEY)
    return [] if raw is None else list(json.loads(raw.decode("utf-8")))


def _json_write(path: Path, payload: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    write_json(partial, payload)
    os.replace(partial, path)


# --------------------------------------------------------------------------- the run


@dataclass
class _VideoFacts:
    fps: Fraction
    n_frames: int
    width: int
    height: int
    sha256: str
    file_bytes: int


@dataclass
class _Stage:
    name: str
    function: Callable[[_Run], list[str]]
    modules: tuple[Any, ...]
    inputs: Callable[[_Run], list[tuple[str, Path]]]
    parameters: Callable[[_Run], dict[str, Any]]
    scope: str
    limitations: tuple[str, ...]


class _Run:
    """State shared by the stages of one run."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.dir = Path(config.out_dir)
        self.log = config.log or (lambda text: print(text, flush=True))
        self._hashes: dict[str, tuple[int, int, str]] = {}
        self.code = code_version()
        self.source = _source_fingerprint(_SOURCE_DIR)
        self.results: list[StageResult] = []
        self.notes: list[str] = []
        self.video = Path(config.video)
        self.audio = Path(config.audio_source)
        self._has_audio_stream: bool | None = None
        self.sngs = Path(config.sngs_dir)
        self.ball_raw = Path(config.ball_raw)
        self.facts: _VideoFacts | None = None
        self.restrictions: tuple[str, ...] = ()
        self.restriction_sources: dict[str, list[str]] = {}
        self.result_extras: dict[str, Any] = {}

    # ------------------------------------------------------------------ paths and hashes
    def path(self, relative: str) -> Path:
        return self.dir.joinpath(*relative.split("/"))

    @property
    def has_audio_stream(self) -> bool:
        """Whether the supplied audio file has a stream; malformed files still fail."""
        if self._has_audio_stream is None:
            try:
                audio_mod.probe_audio(self.audio)
            except audio_mod.NoAudioStreamError:
                self._has_audio_stream = False
            else:
                self._has_audio_stream = True
        return self._has_audio_stream

    def sha256(self, path: Path) -> str:
        stat = path.stat()
        key = _norm(path)
        cached = self._hashes.get(key)
        if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime_ns):
            return cached[2]
        digest = sha256_file(path)
        self._hashes[key] = (stat.st_size, stat.st_mtime_ns, digest)
        return digest

    def file_ref(self, name: str, path: Path, *, internal: bool) -> dict[str, Any]:
        ref: dict[str, Any] = {
            "name": name,
            "sha256": self.sha256(path),
            "bytes": path.stat().st_size,
        }
        if internal:
            ref["relpath"] = path.relative_to(self.dir).as_posix()
        return ref

    @property
    def rate(self) -> Fraction:
        if self.facts is None:  # pragma: no cover - set before any stage runs
            raise PipelineError("run", ["video facts are not known yet"])
        return self.facts.fps

    @property
    def n_frames(self) -> int:
        if self.facts is None:  # pragma: no cover - set before any stage runs
            raise PipelineError("run", ["video facts are not known yet"])
        return self.facts.n_frames

    # ------------------------------------------------------------------ table readers
    def import_tracks(self) -> pd.DataFrame:
        return import_mod.load_tracks(self.path("import/tracks.parquet"))

    def import_calib(self) -> pd.DataFrame:
        return import_mod.load_calib(self.path("import/calib.parquet"))

    def tracks(self) -> pd.DataFrame:
        """The cleaned track table of the ``refine`` stage."""
        return pd.read_parquet(self.path(TRACKS))

    def calib(self) -> pd.DataFrame:
        """The calibration of the ``refine`` stage (refitted where keypoints allowed)."""
        return pd.read_parquet(self.path(CALIB))

    def table(self, relative: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(relative))

    def highlights(self) -> dict[str, Any]:
        return dict(read_json(self.path("highlights.json")))

    def final_highlights(self) -> dict[str, Any]:
        return dict(read_json(self.path(FINAL)))

    def cuts(self) -> list[int]:
        return [int(c) for c in read_json(self.path("cuts.json"))["cuts"]]

    def pitch_frames(self) -> frozenset[int]:
        signals = pd.read_parquet(self.path("signals.parquet"), columns=["frame", "ev_pitch"])
        flags = signals["ev_pitch"].to_numpy(dtype=bool)
        return frozenset(int(f) for f in signals["frame"].to_numpy()[flags])

    def ball_events_all(self) -> pd.DataFrame:
        """Strike-profile and contact-profile ball events with a ``profile`` column."""
        strike = self.table("ball_events.parquet").assign(profile="strike")
        contact = self.table("ball_contacts.parquet").assign(profile="contact")
        parts = [part for part in (strike, contact) if len(part)]
        if not parts:
            return strike.iloc[0:0]
        combined = pd.concat(parts, ignore_index=True)
        return combined.sort_values(["frame", "profile"], kind="stable").reset_index(drop=True)

    def write(self, frame: pd.DataFrame, relative: str) -> str:
        write_parquet(frame, self.path(relative), self.restrictions)
        return relative


# --------------------------------------------------------------------------- stage bodies


def _stage_import(run: _Run) -> list[str]:
    target = run.path("import")
    summary = import_mod.import_sngs(
        run.sngs, target, rep_err_max_px=run.config.rep_err_max_px, overwrite=True
    )
    if summary.n_frames != run.n_frames:
        raise PipelineError(
            "import",
            [
                f"the plugin export covers {summary.n_frames} frames but the video decodes to "
                f"{run.n_frames}; the export does not belong to this canonical video"
            ],
        )
    # The tables are per-item derivatives of the video too: they carry the run's tags
    # (the plugin's plus --restrictions), not only the plugin's.
    if set(summary.restrictions) != set(run.restrictions):
        for name in ("import/tracks.parquet", "import/calib.parquet"):
            _write_table(pq.read_table(run.path(name)), run.path(name), run.restrictions)
        report = read_json(run.path("import/import.json"))
        report["plugin_restrictions"] = report.get("restrictions")
        report["restrictions"] = list(run.restrictions)
        _json_write(run.path("import/import.json"), report)
    run.result_extras["import"] = {
        "frames": summary.n_frames,
        "track_rows": summary.n_track_rows,
        "pitch_usable_frames": summary.n_pitch_usable_frames,
        "plugin_restrictions": list(summary.restrictions),
    }
    return ["import/tracks.parquet", "import/calib.parquet", "import/import.json"]


def _stage_shots(run: _Run) -> list[str]:
    cuts = shots_mod.detect_cut_frames(run.video)
    shots = shots_mod.build_shots(cuts, run.n_frames, run.import_tracks(), run.import_calib())
    _json_write(
        run.path("cuts.json"),
        {
            "schema_version": 1,
            "producer": PRODUCER,
            "cuts": [int(c) for c in cuts],
            "settings": cuts_mod.cut_detection_settings(),
            "frame_count": run.n_frames,
            "fps": format_rate(run.rate),
            "restrictions": list(run.restrictions),
        },
    )
    run.write(shots, "shots.parquet")
    run.result_extras["shots"] = {
        "cuts": [int(c) for c in cuts],
        "shots": [
            [int(row.start_frame), int(row.end_frame), str(row.shot_type)]
            for row in shots.itertuples(index=False)
        ],
    }
    return ["cuts.json", "shots.parquet"]


def _stage_refine(run: _Run) -> list[str]:
    shots = run.table("shots.parquet")
    tracks, counts = enhance_mod.clean_tracks(run.import_tracks(), shots)
    calib = run.import_calib()
    report: dict[str, Any] | None = None
    if run.config.refine_calibration:
        settings = calib_refine_mod.RefineSettings(rep_err_max_px=run.config.rep_err_max_px)
        tracks, calib, report = enhance_mod.refine_from_plugin(
            run.sngs, tracks, calib, shots, settings
        )
    tracks, teams = _check_team_labels(run, tracks)
    tracks, roles = _check_role_labels(run, tracks)
    sides = _check_sides(run, tracks)
    run.write(tracks, TRACKS)
    run.write(calib, CALIB)
    summary = {
        "schema_version": 1,
        "producer": PRODUCER,
        "tracks": counts,
        "teams": teams,
        "roles": roles,
        "sides": sides,
        "calibration": report
        or {
            "refined": False,
            "reason": "disabled"
            if not run.config.refine_calibration
            else "the plugin export has no keypoints.jsonl",
        },
        "restrictions": list(run.restrictions),
    }
    _json_write(run.path("refine/refine.json"), summary)
    run.result_extras["refine"] = {
        "tracks": counts,
        "teams": teams["counts"],
        "roles": {key: roles[key] for key in ("verdict", "crowded_share", "rows_cleared")},
        "sides": {
            key: sides[key]
            for key in ("verdict", "attack_direction_known", "difference_m", "interval_m")
        },
        "calibration": (report or {}).get("counts"),
        "holdout": (report or {}).get("holdout"),
        "ground_jitter": (report or {}).get("ground_jitter"),
    }
    if report is not None:
        run.notes.append(_refine_note(report, shots))
    return [TRACKS, CALIB, "refine/refine.json"]


def _refine_note(report: Mapping[str, Any], shots: pd.DataFrame) -> str:
    """The report note on the calibration refit (gbf.enhance.refine_from_plugin's report)."""
    spans = {int(s.shot_id): (int(s.start_frame), int(s.end_frame)) for s in shots.itertuples()}

    def camera(group: Mapping[str, Any]) -> str:
        frames = "、".join(f"{spans[s][0]}–{spans[s][1]}" for s in group["shots"])
        return f"第 {frames} 帧的机位"

    def why_fell_back(group: Mapping[str, Any]) -> str:
        # gbf.calib_refine reports a solver result only for a group it refitted on all points
        if group.get("converged") is False:
            return (
                f"{camera(group)}重拟合用满 {group['solver_nfev']} 次求解仍未收敛，"
                "整组改用插件的单帧标定。"
            )
        held = group["holdout"]
        return (
            f"{camera(group)}重拟合后留出关键点误差 P90 为 {held['refined']['p90_px']:.1f} 像素，"
            f"超过单帧拟合（{held['free_centre']['p90_px']:.1f} 像素）的 "
            f"{report['settings']['holdout_fallback_ratio']:g} 倍，整组改用插件的单帧标定。"
        )

    fell_back = "".join(why_fell_back(g) for g in report["groups"] if g.get("fallback"))
    if not report["counts"].get("refined_frames"):
        return (
            "插件导出了球场关键点，但没有一帧按机位重拟合成功：标定沿用插件的单帧标定。" + fell_back
        )
    jitter = report["ground_jitter"]
    rejected = [camera(g) for g in report["groups"] if g.get("rejected")]
    unconverged = "".join(
        f"{camera(g)}重拟合用满 {g['solver_nfev']} 次求解仍未收敛：这组标定可能还没到最好，"
        "换机器或线程数，结果也可能不同。"
        for g in report["groups"]
        if g.get("converged") is False and not g.get("fallback")
    )
    return (
        "标定按同一机位固定相机中心重拟合（插件导出的球场关键点；每帧旋转和焦距平滑）："
        f"可用帧 {report['counts']['pitch_usable_before']} → "
        f"{report['counts']['pitch_usable_after']}；球员位置逐帧抖动（二阶差分，各自可用帧上）"
        f"{_jitter_p90(jitter['plugin']['p90_m'], jitter['refined']['p90_m'])}。"
        + unconverged
        + (
            f"{'、'.join(rejected)}的留出关键点误差 P90 超过 "
            f"{report['settings']['holdout_p90_max_px']:g} 像素，整组不给米制数值。"
            if rejected
            else ""
        )
        + fell_back
        + "仍只是自洽检验。"
    )


def _jitter_p90(before: float | None, after: float | None) -> str:
    """'P90 a → b 米' for the ground jitter before and after the refit; a side without usable
    frames (P90 None, gbf.calib_refine.ground_jitter) says so."""
    if before is not None and after is not None:
        return f"P90 {before:.2f} → {after:.2f} 米"
    sides = ["没有可用帧" if value is None else f"{value:.2f} 米" for value in (before, after)]
    return f"P90 {sides[0]} → {sides[1]}"


def _check_team_labels(run: _Run, tracks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check the plugin's team labels against shirt colours (gbf.team_check, one streaming
    pass over the video) and clear the labels that contradict them and those of split-off
    pieces (:func:`_clear_split_pieces`); the report note (:func:`_team_note`) says what was
    checked, cleared and kept."""
    facts = run.facts
    size = (facts.width, facts.height) if facts is not None else None
    plan = team_check_mod.plan_samples(tracks, frame_size=size)
    with closing(iter_frames(run.video)) as frames:
        colours, samples = team_check_mod.measure_shirts(plan, frames)
    tracks, report = team_check_mod.check_teams(tracks, colours, samples=samples)
    tracks, report = _clear_split_pieces(tracks, report)
    run.notes.append(_team_note(report))
    return tracks, report


def _teams_judged(report: Mapping[str, Any]) -> bool:
    """Both teams got a shirt-colour prototype, so the check judged the tracks."""
    return all(report["prototypes"].get(team) is not None for team in ("left", "right"))


def _clear_split_pieces(
    tracks: pd.DataFrame, report: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Once both teams have a shirt colour, a piece of a track split off at a cut (an id of
    gbf.track_tools.ID_STRIDE or more) whose shirt could not be checked loses its team label
    too: it was inherited from the player who had the id before the cut.

    Returns a copy of ``tracks`` without those labels, and ``report`` updated in place: such
    a piece's entry gets ``cleared`` "split_piece" (gbf.team_check marks the contradicted
    ones "inconsistent"), ``cleared_inherited_tracks`` lists the pieces, and the counts give
    ``rows_cleared_split_pieces`` and, in ``rows_cleared``, every player row whose label the
    refine stage cleared."""
    judged = _teams_judged(report)
    pieces = [
        entry
        for entry in report["tracks"]
        if judged
        and entry["verdict"] == "unchecked"
        and int(entry["track_id"]) >= track_tools_mod.ID_STRIDE
    ]
    rows_cleared = 0
    if pieces:
        tracks = tracks.copy()
        missing = (
            tracks["team"].dtype.na_value if hasattr(tracks["team"].dtype, "na_value") else None
        )
        player = (tracks["role"] == "player").to_numpy(dtype=bool, na_value=False)
        ids = tracks["track_id"].to_numpy(dtype=np.int64)
        for entry in pieces:
            team = (tracks["team"] == entry["team"]).to_numpy(dtype=bool, na_value=False)
            rows = player & team & (ids == int(entry["track_id"]))
            rows_cleared += int(rows.sum())
            tracks.loc[rows, "team"] = missing
            entry["cleared"] = "split_piece"
    report["cleared_inherited_tracks"] = sorted({int(entry["track_id"]) for entry in pieces})
    counts = report["counts"]
    counts["rows_cleared"] = int(counts.get("rows_cleared", 0)) + rows_cleared
    counts["rows_cleared_split_pieces"] = rows_cleared
    return tracks, report


def _team_note(report: Mapping[str, Any]) -> str:
    """The report note on the team check: that it could not run, or how many player tracks
    it checked and found contradicted, how many it could not check (too few shirt samples:
    they keep the plugin's label) and how many split-off pieces lost theirs."""
    if not _teams_judged(report):
        return "球衣颜色核对没能进行（至少一队没有足够的球衣样本）：球队标签沿用插件的，未经核对。"
    entries = report["tracks"]
    checked = sum(entry["verdict"] in ("consistent", "inconsistent") for entry in entries)
    contradicted = sum(entry["verdict"] == "inconsistent" for entry in entries)
    kept = sum(entry["verdict"] == "unchecked" and not entry.get("cleared") for entry in entries)
    pieces = sum(entry.get("cleared") == "split_piece" for entry in entries)
    parts = [
        f"{checked} 条轨迹球衣样本足够，"
        + (
            f"其中 {contradicted} 条的球队标签与球衣颜色矛盾，改为未知球队（不翻转）"
            if contradicted
            else "球队标签都没有与球衣颜色矛盾"
        )
    ]
    if kept:
        parts.append(f"{kept} 条轨迹球衣样本太少、未能核对，沿用插件的球队标签")
    if pieces:
        parts.append(f"{pieces} 条切镜后拆出的轨迹无法核对球衣，也不标球队")
    return "球队标签按球衣颜色核对：" + "；".join(parts) + "。"


def _check_role_labels(run: _Run, tracks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check the plugin's role labels against the one referee the laws allow on the pitch
    (gbf.role_check); when they are implausible every team label is cleared and the report
    note (:func:`_role_note`) says why."""
    tracks, report = role_check_mod.check_roles(tracks)
    if report["verdict"] == "implausible":
        run.notes.append(_role_note(report))
    return tracks, report


def _role_note(report: Mapping[str, Any]) -> str:
    """The report note on implausible role labels: how often two or more people labelled
    referee stood inside the pitch at once, and what that cost."""
    return (
        f"角色标签不合理：{report['frames']} 帧里有 {report['crowded_frames']} 帧"
        f"（{report['crowded_share']:.0%}）同时有两人以上被插件标成裁判、站在场内，"
        "而规则只允许一名裁判在场内。插件只给标成球员的人分队，所以分队也不可信："
        "球队标签全部清空，依赖球队的数值（最近对手等）不再计算。角色沿用插件的，未经核对。"
    )


def _check_sides(run: _Run, tracks: pd.DataFrame) -> dict[str, Any]:
    """Check the plugin's left/right team names against where the teams stand
    (gbf.side_check); when they are not borne out, the direction each team attacks counts
    as unknown for the later stages and the report note (:func:`_sides_note`) says why."""
    report = side_check_mod.check_sides(tracks)
    if not report["attack_direction_known"]:
        run.notes.append(_sides_note(report))
    return report


def _sides_note(report: Mapping[str, Any]) -> str:
    """The report note when the left/right names are not borne out: the numbers the check
    went by and what an unknown attack direction costs."""
    if report["verdict"] == "not_judged":
        tracks = report["tracks"]
        finding = (
            f"有一队可用于核对的球员轨迹少于 {report['params']['min_tracks']} 条"
            f"（左 {tracks['left']} 条、右 {tracks['right']} 条），无法核对。"
        )
    else:
        low, high = report["interval_m"]
        where = (
            "区间跨过 0"
            if report["verdict"] == "uncertain"
            else "区间整个在 0 以下，与插件的左右相反"
        )
        finding = (
            "按 gbf 的标定，标为右侧的球队平均位置减去标为左侧的为 "
            f"{report['difference_m']:+.1f} 米（{report['params']['level']:.0%} 区间 "
            f"{low:+.1f} 至 {high:+.1f} 米），{where}。"
        )
    return (
        "左右分队未能核实：插件只按两队全部检测的平均位置定左右。"
        + finding
        + "所以不知道哪队进攻哪个球门：标题、说明和信息卡里两队只称「甲队」「乙队」"
        "（不用配置的队名），禁区压迫和射门方向两种线索不计，禁区进攻人数和相对进攻球门的"
        "身体朝向不给值；最近对手距离等只需分清两队的数值照常。"
    )


NEUTRAL_TEAM_NAMES_ZH: Mapping[str, str] = MappingProxyType({"left": "甲队", "right": "乙队"})
"""Team names when the direction each team attacks is unknown: they name neither side."""


def _attack_direction_known(run: _Run) -> bool:
    """Whether gbf.side_check bore out the plugin's left/right names (``sides`` of
    ``refine/refine.json``; True for a refine stage from before the check)."""
    sides = read_json(run.path("refine/refine.json")).get("sides")
    return True if sides is None else bool(sides["attack_direction_known"])


def _stage_kinematics(run: _Run) -> list[str]:
    # one run of the outlier tests gives the tables and every rejected position by test,
    # also those outside every segment (segments.parquet n_outliers counts only the ones
    # inside a segment's span)
    kinematics, segments, rejected = kinematics_mod.kinematics_with_rejections(
        run.tracks(), run.table("shots.parquet"), run.rate, **KINEMATICS_PARAMS
    )
    run.write(kinematics, "kinematics.parquet")
    run.write(segments, "segments.parquet")
    valid = kinematics[kinematics["valid"].to_numpy(dtype=bool)]
    by_test = rejected["test"].value_counts()
    run.result_extras["kinematics"] = {
        "rows": int(len(kinematics)),
        "valid_rows": int(len(valid)),
        "segments": int(len(segments)),
        "max_valid_speed_mps": float(valid["speed"].max()) if len(valid) else None,
        "rejected_positions": int(len(rejected)),
        "rejected_by_test": {
            test: int(by_test.get(test, 0)) for test in kinematics_mod.REJECTION_TESTS
        },
    }
    return ["kinematics.parquet", "segments.parquet"]


def _ball_file_facts(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    lines = data.split(b"\n")
    tail = lines[-1].strip()
    complete = [line for line in lines[:-1] if line.strip()]
    return {
        "lines": len(complete),
        "incomplete_tail_dropped": bool(tail),
    }


def _stage_ball(run: _Run) -> list[str]:
    facts = _ball_file_facts(run.ball_raw)
    candidates = ball_mod.load_candidates(run.ball_raw, allow_incomplete_tail=True)
    tracks, calib, shots = run.tracks(), run.calib(), run.table("shots.parquet")
    track = ball_mod.track_ball(candidates, tracks, calib, shots, run.rate, run.n_frames)
    strikes = ball_mod.ball_events(track, tracks, run.rate, shots=shots)
    contacts = ball_mod.ball_events(
        track, tracks, run.rate, shots=shots, params=ball_mod.CONTACT_PARAMS
    )
    run.write(track, "ball_track.parquet")
    run.write(strikes, "ball_events.parquet")
    run.write(contacts, "ball_contacts.parquet")
    frames = sorted(candidates)
    missing = run.n_frames - sum(1 for f in frames if 0 <= f < run.n_frames)
    counts = track["status"].value_counts().to_dict()
    run.result_extras["ball"] = {
        **facts,
        "frames_with_a_line": len(frames),
        "frames_without_a_line": int(missing),
        "status_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "strike_events": int(len(strikes)),
        "contact_events": int(len(contacts)),
    }
    if missing or facts["incomplete_tail_dropped"]:
        run.notes.append(
            f"球检测文件不完整：{missing} 帧没有记录"
            + ("，最后一行未写完已丢弃" if facts["incomplete_tail_dropped"] else "")
            + "；这些帧按“没有候选”处理。"
        )
    return ["ball_track.parquet", "ball_events.parquet", "ball_contacts.parquet"]


def _stage_audio(run: _Run) -> list[str]:
    if run.has_audio_stream:
        table = audio_mod.audio_features(
            run.audio, run.rate, run.n_frames, offset_s=run.config.audio_offset_s
        )
    else:
        table = audio_mod.features_from_samples(
            np.empty(0, dtype=np.float32),
            audio_mod.DEFAULT_SAMPLE_RATE,
            run.rate,
            run.n_frames,
            offset_s=run.config.audio_offset_s,
        )
    run.write(table, "audio.parquet")
    with_audio = int(table["has_audio"].sum())
    run.result_extras["audio"] = {
        "frames_with_audio": with_audio,
        "frames": run.n_frames,
        "has_stream": run.has_audio_stream,
    }
    if not run.has_audio_stream:
        run.notes.append("输入文件无音轨；声音线索不可用，本次分析仅使用画面，输出视频无音轨。")
    elif with_audio < run.n_frames:
        run.notes.append(
            f"原声只覆盖 {with_audio}/{run.n_frames} 帧；其余帧的声音线索为 0，渲染时补静音。"
        )
    return ["audio.parquet"]


def _configured_team_names(run: _Run) -> dict[str, str] | None:
    names = run.config.team_names
    return None if names is None else {str(k): str(v) for k, v in names.items()}


def _team_names(run: _Run) -> dict[str, str] | None:
    """The team names the texts use: the configured ones, or the neutral ones when the attack
    direction is unknown, since the configured names are given per side."""
    if not _attack_direction_known(run):
        return dict(NEUTRAL_TEAM_NAMES_ZH)
    return _configured_team_names(run)


def _stage_highlights(run: _Run) -> list[str]:
    tracks, shots = run.tracks(), run.table("shots.parquet")
    events = run.table("ball_events.parquet")
    signals = highlights_mod.compute_signals(
        tracks,
        run.table("kinematics.parquet"),
        run.table("ball_track.parquet"),
        events,
        shots,
        run.table("audio.parquet"),
        run.rate,
        attack_known=_attack_direction_known(run),
    )
    document = highlights_mod.detect_highlights(
        signals,
        shots,
        tracks,
        events,
        run.rate,
        min_separation_s=run.config.min_separation_s,
        max_highlights=run.config.max_highlights,
        team_names=_team_names(run),
    )
    document = moments_mod.merge_adjacent_highlights(document, run.cuts(), run.rate)
    document = moments_mod.add_duel_partners(document, tracks)
    document["restrictions"] = list(run.restrictions)
    run.write(signals, "signals.parquet")
    _json_write(run.path("highlights.json"), document)
    run.result_extras["highlights"] = [
        {
            "id": h["id"],
            "type": h["type"],
            "start_frame": h["start_frame"],
            "peak_frame": h["peak_frame"],
            "end_frame": h["end_frame"],
            "score": h["score"],
        }
        for h in document["highlights"]
    ]
    return ["signals.parquet", "highlights.json"]


def _pose_ranges(frames: Sequence[int], gap: int) -> list[tuple[int, int]]:
    """Inclusive frame ranges covering ``frames`` (sorted), merging gaps up to ``gap``."""
    ranges: list[tuple[int, int]] = []
    for frame in sorted(set(int(f) for f in frames)):
        if ranges and frame - ranges[-1][1] <= gap:
            ranges[-1] = (ranges[-1][0], frame)
        else:
            ranges.append((frame, frame))
    return ranges


def _pose_estimator(run: _Run) -> Any:
    if run.config.pose_estimator is not None:
        return run.config.pose_estimator
    model = POSE_MODELS[run.config.pose_model]
    return pose_mod.PoseEstimator.from_pinned(model, run.config.models_dir)


def _stage_pose(run: _Run) -> list[str]:
    document = run.highlights()
    requests = measure_mod.pose_requests(
        document["highlights"], run.tracks(), run.ball_events_all(), measure_mod.DEFAULT_PARAMS
    )
    if run.config.pose_model == "none" and run.config.pose_estimator is None:
        pose = pose_mod.pose_table([])
        run.notes.append(
            "本次运行没有做姿态估计（--pose-model none）：姿态和三维相关的量都不可用。"
        )
    else:
        estimator = _pose_estimator(run)
        parts = []
        ranges = _pose_ranges(requests["frame"].tolist(), POSE_RANGE_MERGE_GAP)
        for first, last in ranges:
            started = time.perf_counter()
            subset = requests[(requests["frame"] >= first) & (requests["frame"] <= last)]
            frames = iter_frames(run.video, first, last + 1)
            try:
                parts.append(estimator.estimate_frames(frames, subset))
            finally:
                close = getattr(frames, "close", None)
                if close is not None:
                    close()
            run.log(
                f"  pose frames {first}-{last}: {len(subset)} boxes in "
                f"{time.perf_counter() - started:.1f} s"
            )
        pose = pose_mod.pose_table([row for part in parts for row in part.to_dict("records")])
    table = pa.Table.from_pandas(
        pose[list(pose_mod.POSE_COLUMNS)], schema=pose_mod.POSE_SCHEMA, preserve_index=False
    )
    _write_table(table.replace_schema_metadata(None), run.path("pose.parquet"), run.restrictions)
    run.result_extras["pose"] = {
        "requested_boxes": int(len(requests)),
        "rows": int(len(pose)),
        "frames": int(pose["frame"].nunique()) if len(pose) else 0,
        "rows_below_display_score": int(
            (pose["mean_score"] < measure_mod.DEFAULT_PARAMS.pose_min_mean_score).sum()
        )
        if len(pose)
        else 0,
    }
    return ["pose.parquet"]


def _stage_lift3d(run: _Run) -> list[str]:
    pose = pose_mod.read_pose_parquet(run.path("pose.parquet"))
    pitch = run.pitch_frames()
    cameras = {
        frame: camera
        for frame, camera in import_mod.cameras_by_frame(run.calib()).items()
        if frame in pitch
    }
    lift = lift_mod.lift_pose_table(pose, cameras)
    table = pa.Table.from_pandas(
        lift[list(lift_mod.LIFT3D_COLUMNS)], schema=lift_mod.LIFT3D_SCHEMA, preserve_index=False
    )
    _write_table(table.replace_schema_metadata(None), run.path("lift3d.parquet"), run.restrictions)
    lifted = lift[lift["method"] == measure_mod.LIFTED] if len(lift) else lift
    run.result_extras["lift3d"] = {
        "rows": int(len(lift)),
        "lifted": int(len(lifted)),
        "reasons": {str(k): int(v) for k, v in lift["reason"].value_counts().items()}
        if len(lift)
        else {},
    }
    return ["lift3d.parquet"]


def _stage_moments(run: _Run) -> list[str]:
    pose = pose_mod.read_pose_parquet(run.path("pose.parquet"))
    pose = pose[pose["mean_score"] >= measure_mod.DEFAULT_PARAMS.pose_min_mean_score]
    document = moments_mod.add_key_moments(
        run.highlights(),
        pose,
        run.rate,
        ball_track=run.table("ball_track.parquet"),
        tracks=run.tracks(),
        team_names=_team_names(run),
        touches=run.ball_events_all(),
        shots=run.table("shots.parquet"),
    )
    document = moments_mod.rank_highlights(document)
    document["restrictions"] = list(run.restrictions)
    _json_write(run.path(FINAL), document)
    outputs = [FINAL]
    if run.config.labels is not None:
        labels = read_json(run.config.labels)
        result = label_eval_mod.evaluate(labels, document)
        result["summary_zh"] = label_eval_mod.summary_zh(result, float(run.rate))
        result["source_zh"] = label_eval_mod.source_zh(labels)
        result["restrictions"] = list(run.restrictions)
        _json_write(run.path("label_eval.json"), result)
        outputs.append("label_eval.json")
        run.result_extras["label_eval"] = {
            k: result[k]
            for k in ("n_labels", "n_matched", "n_peak_within_tolerance", "n_type_agrees")
        }
    run.result_extras["moments"] = [
        {
            "id": h["id"],
            "type": h["type"],
            "peak_frame": h["peak_frame"],
            "tier": h.get("tier"),
            "key_moments": [(m["frame"], m["kind"]) for m in h.get("key_moments", [])],
        }
        for h in document["highlights"]
    ]
    return outputs


_MEASURE_INPUTS = (
    TRACKS,
    CALIB,
    "refine/refine.json",
    "shots.parquet",
    "kinematics.parquet",
    "ball_events.parquet",
    "ball_contacts.parquet",
    "pose.parquet",
    "lift3d.parquet",
    "signals.parquet",
    FINAL,
)


def _measure_inputs(run: _Run) -> list[tuple[str, Path]]:
    """The run's tables the records are computed from, and the metric registry their tiers
    come from (outside the run directory: its receipt entry has no path)."""
    return [
        *_internal(*_MEASURE_INPUTS)(run),
        (METRICS_REGISTRY_INPUT, measure_mod.default_registry_path()),
    ]


def _measure_context(run: _Run) -> measure_mod.MeasureContext:
    facts = run.facts
    if facts is None:  # pragma: no cover - set before any stage runs
        raise PipelineError("measure", ["video facts are not known"])
    return measure_mod.MeasureContext(
        fps=run.rate,
        restrictions=run.restrictions,
        video_sha256=facts.sha256,
        inputs=tuple((name, run.sha256(run.path(name))) for name in _MEASURE_INPUTS),
        code=run.code,
        registry=measure_mod.load_metric_registry(),
        attack_direction_known=_attack_direction_known(run),
    )


def _team_verdicts(run: _Run) -> dict[tuple[int, str], str]:
    """Team-check verdicts keyed by the refined track and its claimed team."""
    report = read_json(run.path("refine/refine.json"))
    return {
        (int(row["track_id"]), str(row["team"])): str(row["verdict"])
        for row in report.get("teams", {}).get("tracks", [])
    }


def _stage_measure(run: _Run) -> list[str]:
    tables = measure_mod.MeasureTables(
        tracks=run.tracks(),
        shots=run.table("shots.parquet"),
        kinematics=run.table("kinematics.parquet"),
        ball_events=run.ball_events_all(),
        pose=pose_mod.read_pose_parquet(run.path("pose.parquet")),
        lift=pd.read_parquet(run.path("lift3d.parquet")),
        pitch_frames=run.pitch_frames(),
        team_verdicts=_team_verdicts(run),
        frame_size=(run.facts.width, run.facts.height) if run.facts is not None else None,
        calib=run.calib(),
    )
    records = measure_mod.measure_highlights(
        run.final_highlights(), tables, _measure_context(run), measure_mod.DEFAULT_PARAMS
    )
    measure_mod.write_measurements(run.path(measure_mod.MEASUREMENTS_NAME), records)
    ok = sum(1 for record in records if record["status"] == "ok")
    run.result_extras["measure"] = {
        "records": len(records),
        "ok": ok,
        "unavailable": len(records) - ok,
        "by_metric": dict(
            sorted(
                pd.Series([r["metric_id"] for r in records], dtype=object)
                .value_counts()
                .to_dict()
                .items()
            )
        )
        if records
        else {},
    }
    return [measure_mod.MEASUREMENTS_NAME]


def _style(run: _Run) -> render_mod.Style:
    watermark = WATERMARK if WATERMARK_TAGS & set(run.restrictions) else None
    return render_mod.Style(watermark=watermark).with_fonts()


def _render_team_names(run: _Run) -> tuple[tuple[str, str], ...]:
    names = _team_names(run)
    if not names:
        return render_mod.TEAM_NAMES_ZH
    defaults = dict(render_mod.TEAM_NAMES_ZH)
    return tuple((team, names.get(team, defaults[team])) for team in ("left", "right"))


DISPLAY_UNITS_ZH: Mapping[str, str] = {"box_h": "个身高", "bh/s": "个身高/秒", "count": "人"}
"""Chinese display units for gbf.measure units that gbf.report does not translate yet."""


def display_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of an evidence record for display only: units in Chinese. The stored record
    (measurements.jsonl, its evidenceId) is never changed."""
    shown = dict(record)
    unit = record.get("unit")
    if isinstance(unit, str) and unit in DISPLAY_UNITS_ZH:
        shown["unit"] = DISPLAY_UNITS_ZH[unit]
    return shown


def _card_rows(
    records: Sequence[Mapping[str, Any]], highlight: Mapping[str, Any], focus: int | None
) -> tuple[render_mod.CardRow, ...]:
    """The freeze card of a highlight (gbf.measure.card_records). A row names the player
    whenever the value belongs to someone other than the focus player of the scene, and the
    frame whenever it was measured on another frame than the key frame, both before the
    label (gbf.report.card_rows; gbf.render cuts a label that does not fit at its end), so
    a number is never read as the focus player's or as one of the frozen frame; a
    frame-valued row (the contact frame) does not repeat its frame in the label."""
    rows = []
    key = int(highlight["peak_frame"])
    for record in measure_mod.card_records(records, highlight, prefer_track=focus, key_frame=key):
        shown = display_record(record)
        if record.get("unit") == "frame":
            shown["frame"] = None
        rows.append(report_mod.metric_row(shown))
    return report_mod.card_rows(rows, key_frame=key, name_tracks=True, focus_track=focus)


def _card(
    records: Sequence[Mapping[str, Any]],
    highlight: Mapping[str, Any],
    scene: Any,
    shot_of: Callable[[int], int],
) -> tuple[tuple[Any, ...], int]:
    """The freeze card of a highlight and how many rows it leaves out: its key moments first
    (at most CARD_MOMENT_ROWS), then measurements of the key frame's shot about the player
    the key frame is about (the overlay's spotlight, render.HighlightScene.roles_at), at
    most CARD_MAX_ROWS rows in all. A value measured in another shot (another camera, or
    seconds away) never sits on the frozen frame; the report lists every record. The count
    is every key-moment row and every shown metric of the shot (one row each, as
    gbf.measure.card_records picks it) beyond the rows kept: the card's footer adds them to
    the rows the frame has no room for (另有 N 项见报告)."""
    key = int(highlight["peak_frame"])
    focus, _pair, involved = scene.roles_at(key)
    key_shot = shot_of(key)

    def here(record: Mapping[str, Any]) -> bool:
        frame = record.get("frame")
        if frame is not None:
            return shot_of(int(frame)) == key_shot
        track = record.get("track_id")
        return track is None or track in involved

    nearby = [record for record in records if here(record)]
    moments = _moment_rows(highlight)
    values = _card_rows(nearby, highlight, focus)
    rows = (moments[:CARD_MOMENT_ROWS] + tuple(values))[:CARD_MAX_ROWS]
    every = measure_mod.card_records(
        nearby, highlight, limit=len(nearby), prefer_track=focus, key_frame=key
    )
    shown = sum(not report_mod.metric_row(display_record(r)).blocked for r in every)
    return rows, len(moments) + shown - len(rows)


class _FullClipAnnotator:
    """Overlay of the whole clip: each frame inside a highlight window shows that
    highlight's overlay (the nearest peak wins where windows overlap); every other frame
    shows players, ball and minimap with a neutral banner."""

    def __init__(self, scenes: Sequence[Any], base: Any, rate: Fraction) -> None:
        self.scenes = list(scenes)
        self.base = base
        self.rate = rate

    def __call__(self, frame: int, kind: str) -> render_mod.FrameAnnotations:
        inside = [
            scene
            for scene in self.scenes
            if int(scene.highlight["start_frame"]) <= frame <= int(scene.highlight["end_frame"])
        ]
        if inside:
            scene = min(
                inside,
                key=lambda s: (abs(int(s.highlight["peak_frame"]) - frame), str(s.highlight["id"])),
            )
            return scene.annotate(frame, kind)
        annotations = self.base.annotate(frame, kind)
        plane = "球场坐标" if self.base.is_pitch_frame(frame) else "画面平面"
        banner = render_mod.Banner(
            "全片逐帧标注（不在高光窗口内）",
            render_mod.format_clock(frame, self.rate),
            f"第 {frame} 帧 · {plane}",
        )
        return replace(annotations, banner=banner)


def _facing_gates() -> render_mod.FacingGates:
    """gbf.measure's per-frame display gates of body-orientation, which the focus player's
    lifted facing passes before the overlay draws it back on a frame: the same rows the data
    card's value may use (render.FacingGates)."""
    params = measure_mod.DEFAULT_PARAMS
    entry = measure_mod.load_metric_registry().get("body-orientation")
    return render_mod.FacingGates(
        min_conf=params.facing_min_conf,
        max_sigma_deg=params.facing_max_sigma_deg,
        min_box_px=float(entry.min_bbox_px) if entry is not None else 0.0,
    )


_RENDER_LIFT_COLUMNS = (
    "frame",
    "track_id",
    "method",
    "facing_deg",
    "facing_conf",
    "facing_sigma_deg",
)
"""The lift3d.parquet columns the overlay reads (the facing it draws back on the frames)."""


def _stage_render(run: _Run) -> list[str]:
    old_receipt = run.path("receipts/render.json")
    if old_receipt.is_file():
        for entry in read_json(old_receipt).get("outputs", []):
            stale = run.path(str(entry.get("relpath", "")))
            if entry.get("relpath") and stale.is_file():
                stale.unlink()
    document = run.final_highlights()
    records = measure_mod.read_measurements(run.path(measure_mod.MEASUREMENTS_NAME))
    pose = pose_mod.read_pose_parquet(run.path("pose.parquet"))
    pose = pose[pose["mean_score"] >= measure_mod.DEFAULT_PARAMS.pose_min_mean_score]
    # The 3D lift reaches the overlay only as the focus player's facing, drawn back on the
    # original frames where it passes gbf.measure's gates; every other mark is 2D.
    tables = render_mod.SceneTables(
        tracks=run.tracks(),
        calib=run.calib(),
        ball=run.table("ball_track.parquet"),
        kinematics=run.table("kinematics.parquet"),
        pose=pose,
        lift=pd.read_parquet(run.path("lift3d.parquet"), columns=list(_RENDER_LIFT_COLUMNS)),
        team_verdicts=_team_verdicts(run),
    )
    gates = _facing_gates()
    cuts = run.cuts()
    pitch = sorted(run.pitch_frames())
    shots = run.table("shots.parquet")

    def shot_of(frame: int) -> int:
        return int(shots_mod.shot_ids(shots, [frame])[0])

    # A second, fainter watermark at the bottom: a frame cut off at the top still says it.
    style = replace(_style(run), watermark_repeat=True)
    # --audio-offset-s counts from the first audio sample, as the audio cue does; the clips
    # play the same alignment (render's offset counts on ffmpeg's timeline).
    audio_source = run.audio if run.has_audio_stream else None
    offset = (
        render_mod.first_sample_offset(audio_source, run.config.audio_offset_s)
        if audio_source is not None
        else 0.0
    )
    sha = run.facts.sha256 if run.facts is not None else None
    team_names = _render_team_names(run)
    outputs: list[str] = []
    scenes = []
    rendered: dict[str, Any] = {}
    withheld: dict[str, set[int]] = {}
    for record in records:
        if (
            record.get("metric_id") == "peak-speed"
            and record.get("status") == "unavailable"
            and record.get("reasonCode")
            in {
                "speed_coverage_split",
                "speed_coverage_rejected",
                "speed_coverage_short",
                "speed_coverage_implausible",
                "speed_peak_hidden",
            }
            and record.get("track_id") is not None
        ):
            withheld.setdefault(str(record["highlight_id"]), set()).add(int(record["track_id"]))
    for highlight in document["highlights"]:
        started = time.perf_counter()
        scene = render_mod.HighlightScene(
            tables,
            highlight,
            run.rate,
            cuts=cuts,
            pitch_frames=pitch,
            team_names=team_names,
            facing_gates=gates,
            frame_size=(run.facts.width, run.facts.height) if run.facts is not None else None,
            peak_speed_withheld=sorted(withheld.get(str(highlight["id"]), ())),
        )
        scenes.append(scene)
        rows, omitted = _card(records, highlight, scene, shot_of)
        result = render_mod.render_highlight(
            run.video,
            run.dir,
            scene,
            card_rows=rows,
            card_omitted=omitted,
            style=style,
            audio_source=audio_source,
            audio_offset_s=offset,
            source_sha256=sha,
            restrictions=run.restrictions,
            overwrite=True,
        )
        outputs += [result.clip, result.manifest, result.keyframe, result.keyframe_card]
        rendered[str(highlight["id"])] = {
            "frames": result.edit.video.frames,
            "segments": result.edit.plan.table(),
            "seconds": round(time.perf_counter() - started, 1),
        }
        run.log(
            f"  render {highlight['id']}: {result.edit.video.frames} frames in "
            f"{time.perf_counter() - started:.1f} s"
        )
    if run.config.full_clip:
        started = time.perf_counter()
        base = render_mod.HighlightScene(
            tables,
            {
                "id": FULL_CLIP_ID,
                "type": "full",
                "start_frame": 0,
                "peak_frame": 0,
                "end_frame": run.n_frames - 1,
                "track_ids": [],
                "title_zh": "全片逐帧标注",
            },
            run.rate,
            cuts=cuts,
            pitch_frames=pitch,
            team_names=team_names,
            facing_gates=gates,
            frame_size=(run.facts.width, run.facts.height) if run.facts is not None else None,
            peak_speed_withheld=sorted(set().union(*withheld.values())),
        )
        plan = render_mod.plan_highlight_edit(
            0, run.n_frames - 1, 0, run.rate, slow_factor=1, freeze_s=0
        )
        clip = run.path(f"clips/{FULL_CLIP_ID}.mp4")
        edit = render_mod.render_edit(
            run.video,
            clip,
            plan,
            annotate=_FullClipAnnotator(scenes, base, run.rate),
            style=style,
            audio_source=audio_source,
            audio_offset_s=offset,
            source_sha256=sha,
            restrictions=run.restrictions,
            overwrite=True,
            part_badges=False,
        )
        outputs += [f"clips/{FULL_CLIP_ID}.mp4", f"clips/{edit.manifest_path.name}"]
        rendered[FULL_CLIP_ID] = {
            "frames": edit.video.frames,
            "seconds": round(time.perf_counter() - started, 1),
        }
        run.log(
            f"  render full clip: {edit.video.frames} frames in "
            f"{time.perf_counter() - started:.1f} s"
        )
    run.result_extras["render"] = rendered
    return outputs


def _moment_label(moment: Mapping[str, Any]) -> str:
    """Short card label of a key moment: its players and the name of what was measured
    (gbf.report.MOMENT_KINDS_ZH), so the card does not cut the sentence the report shows in
    full; the badges say it is an image-plane estimate."""
    kind = str(moment.get("kind"))
    name = report_mod.MOMENT_KINDS_ZH.get(kind, kind)
    ids = [int(t) for t in moment.get("track_ids") or []]
    return f"轨迹{'、'.join(str(t) for t in ids)}·{name}" if ids else name


def _touch_label(touch: Mapping[str, Any]) -> str:
    """Card label of one player's touch count, named by the player's track like the key
    moments above it (轨迹54·检测到的触球), so that touches of another player than the one in
    focus are never read as the focus player's."""
    track = touch.get("track_id")
    if isinstance(track, int | np.integer) and not isinstance(track, bool):
        return f"轨迹{int(track)}·检测到的触球"
    return f"{touch.get('label_zh') or '球员'}·检测到的触球"


def _moment_rows(highlight: Mapping[str, Any]) -> tuple[Any, ...]:
    """Data-card rows for the key moments of a highlight (not the single touches) and one
    row per player's touch count, each naming its players' tracks; image-plane estimates,
    so badged as such."""
    rows = []
    for moment in highlight.get("key_moments") or []:
        if moment.get("kind") == "touch":
            continue
        rows.append(
            render_mod.CardRow(
                _moment_label(moment),
                str(moment["time"]),
                f"第 {int(moment['frame'])} 帧",
                (render_mod.Badge.of("未验证"), render_mod.Badge.of("仅画面平面")),
            )
        )
    for touch in highlight.get("touches") or []:
        rows.append(
            render_mod.CardRow(
                _touch_label(touch),
                str(int(touch["count"])),
                "次",
                (render_mod.Badge.of("未验证"), render_mod.Badge.of("仅画面平面")),
            )
        )
    return tuple(rows)


def _calibration_summary(run: _Run) -> dict[str, Any]:
    """The plugin's per-frame calibration (fresh frames, their self-consistency) and, when
    the refine stage refitted it, the refit: frames usable before and after, the held-out
    keypoint error of the fit and the frames of camera groups rejected for their held-out
    tail. ``pitch_usable`` is what the analysis used."""
    calib = run.import_calib()
    fresh = calib[calib["calib_status"] == "fresh"]
    errors = fresh["rep_err"].dropna()
    refit = read_json(run.path("refine/refine.json")).get("calibration") or {}
    counts = refit.get("counts") or {}
    held = (refit.get("holdout") or {}).get("refined") or {}
    return {
        "frames": int(len(calib)),
        "fresh": int(len(fresh)),
        "pitch_usable": int(run.calib()["pitch_usable"].sum()),
        "rep_err_median_px": float(errors.median()) if len(errors) else None,
        "rep_err_p90_px": float(errors.quantile(0.9)) if len(errors) else None,
        "threshold_px": float(run.config.rep_err_max_px),
        "refit": (
            {
                "pitch_usable_before": counts.get("pitch_usable_before"),
                "refined_frames": counts.get("refined_frames"),
                "rejected_frames": counts.get("rejected_group_frames", 0),
                "holdout_median_px": held.get("median_px"),
                "holdout_p90_px": held.get("p90_px"),
                "holdout_p90_max_px": (refit.get("settings") or {}).get("holdout_p90_max_px"),
            }
            if counts
            else None
        ),
    }


_REPORT_ARTIFACTS = (
    "import/tracks.parquet",
    "import/calib.parquet",
    TRACKS,
    CALIB,
    "refine/refine.json",
    "shots.parquet",
    "kinematics.parquet",
    "segments.parquet",
    "ball_track.parquet",
    "ball_events.parquet",
    "ball_contacts.parquet",
    "audio.parquet",
    "signals.parquet",
    "highlights.json",
    FINAL,
    "pose.parquet",
    "lift3d.parquet",
    "measurements.jsonl",
    "label_eval.json",
)


def _tool_versions(run: _Run) -> list[tuple[str, str]]:
    tools = [
        ("python", platform.python_version()),
        ("ffmpeg", ffmpeg_version()),
        ("numpy", np.__version__),
        ("pandas", pd.__version__),
        ("pyarrow", pa.__version__),
    ]
    receipt = run.path("receipts/pose.json")
    if receipt.is_file():
        params = read_json(receipt).get("result", {}).get("parameters", {})
        if params.get("model_id"):
            tools.append(("pose model", str(params["model_id"])))
    return tools


def _stage_report(run: _Run) -> list[str]:
    document = run.final_highlights()
    records = measure_mod.read_measurements(run.path(measure_mod.MEASUREMENTS_NAME))
    images: dict[str, list[Any]] = {}
    clips: dict[str, list[Any]] = {}
    full = run.path(f"clips/{FULL_CLIP_ID}.mp4")
    for highlight in document["highlights"]:
        ident = str(highlight["id"])
        keyframe = run.path(f"keyframes/{ident}.png")
        card = run.path(f"keyframes/{ident}-card.png")
        if keyframe.is_file():
            images[ident] = [f"keyframes/{ident}.png"]
            if card.is_file():
                images[ident].append(
                    report_mod.ImageLink(f"keyframes/{ident}-card.png", "定格数据卡")
                )
        links = []
        if run.path(f"clips/{ident}.mp4").is_file():
            links.append(report_mod.ClipLink("原速 + 慢放 + 定格（含原声）", f"clips/{ident}.mp4"))
        if full.is_file():
            links.append(report_mod.ClipLink("全片逐帧标注（含原声）", f"clips/{FULL_CLIP_ID}.mp4"))
        if links:
            clips[ident] = links
    signals = pd.read_parquet(run.path("signals.parquet"), columns=["frame", "score"])
    score = signals.sort_values("frame")["score"].to_numpy(dtype=np.float64)
    shots = run.table("shots.parquet")
    facts = run.facts
    if facts is None:  # pragma: no cover - set before any stage runs
        raise PipelineError("report", ["video facts are not known"])
    artifacts = tuple(
        report_mod.ArtifactRef(name, run.sha256(run.path(name)), run.path(name).stat().st_size)
        for name in _REPORT_ARTIFACTS
        if run.path(name).is_file()
        # a label comparison left by an earlier run with --labels is not this run's
        and (name != "label_eval.json" or run.config.labels is not None)
    )
    notes = list(dict.fromkeys(run.notes))
    notes.append(
        "触球帧来自球的像面速度突变与脚部关键点（或框底）的距离，以帧为单位，未经人工确认，"
        "按估计值（B 档）显示。"
    )
    notes.append("高光窗口与关键帧由线索自动选取（评分阈值等均为未验证的起始值），仅用于挑选片段。")
    evaluation: tuple[str, ...] = ()
    source: str | None = None
    if run.config.labels is not None and run.path("label_eval.json").is_file():
        # The sentences are written from the comparison's numbers here, so that the page
        # words them as this code does even when label_eval.json comes from an earlier run;
        # where the labels come from is read from the label file itself.
        compared = read_json(run.path("label_eval.json"))
        evaluation = tuple(label_eval_mod.summary_zh(compared, float(run.rate)))
        source = label_eval_mod.source_zh(read_json(Path(run.config.labels)))
    meta = report_mod.ReportMeta(
        clip_name=run.video.name,
        fps=format_rate(run.rate),
        frame_count=run.n_frames,
        width=facts.width,
        height=facts.height,
        source_sha256=facts.sha256,
        cuts=tuple(run.cuts()),
        shots=tuple(
            (int(row.start_frame), int(row.end_frame), str(row.shot_type))
            for row in shots.itertuples(index=False)
        ),
        score=tuple(float(v) for v in score),
        calibration=_calibration_summary(run),
        audio_source=run.audio.name if run.has_audio_stream else None,
        restrictions=tuple(run.restrictions),
        artifacts=artifacts,
        tools=tuple(_tool_versions(run)),
        notes_zh=tuple(notes),
        evaluation_zh=evaluation,
        evaluation_source_zh=source,
        watermark=WATERMARK if WATERMARK_TAGS & set(run.restrictions) else None,
    )
    shown = [display_record(record) for record in records]
    report_mod.write_report(run.dir, document, shown, images, clips, meta)
    return ["report.html", "timeline.png", "timeline-dark.png"]


# --------------------------------------------------------------------------- stage table


def _inputs_import(run: _Run) -> list[tuple[str, Path]]:
    return [
        (f"sngs/{name}", run.sngs / name)
        for name in (
            import_mod.DETECTIONS_NAME,
            import_mod.CALIB_JSONL_NAME,
            import_mod.RECEIPT_NAME,
        )
    ]


def _internal(*names: str) -> Callable[[_Run], list[tuple[str, Path]]]:
    def inputs(run: _Run) -> list[tuple[str, Path]]:
        return [(name, run.path(name)) for name in names]

    return inputs


def _with_video(*names: str) -> Callable[[_Run], list[tuple[str, Path]]]:
    def inputs(run: _Run) -> list[tuple[str, Path]]:
        return [("video", run.video), *((name, run.path(name)) for name in names)]

    return inputs


def _params_import(run: _Run) -> dict[str, Any]:
    return {
        "rep_err_max_px": run.config.rep_err_max_px,
        "in_pitch_margin_m": import_mod.IN_PITCH_MARGIN_M,
    }


def _params_shots(run: _Run) -> dict[str, Any]:
    return {
        "cuts": cuts_mod.cut_detection_settings(),
        "close_min_box_h_px": shots_mod.CLOSE_MIN_BOX_H_PX,
        "close_uncalibrated_min_box_h_px": shots_mod.CLOSE_UNCALIBRATED_MIN_BOX_H_PX,
        "close_max_fresh_share": shots_mod.CLOSE_MAX_FRESH_SHARE,
        "wide_max_box_h_px": shots_mod.WIDE_MAX_BOX_H_PX,
        "wide_min_fresh_share": shots_mod.WIDE_MIN_FRESH_SHARE,
        "n_frames": run.n_frames,
    }


def _params_ball(run: _Run) -> dict[str, Any]:
    return {
        "strike": asdict(ball_mod.DEFAULT_PARAMS),
        "contact": asdict(ball_mod.CONTACT_PARAMS),
        "allow_incomplete_tail": True,
        "n_frames": run.n_frames,
    }


def _params_audio(run: _Run) -> dict[str, Any]:
    return {
        "features": audio_mod.audio_feature_settings(),
        "offset_s": run.config.audio_offset_s,
        "n_frames": run.n_frames,
    }


def _params_highlights(run: _Run) -> dict[str, Any]:
    return {
        "max_highlights": run.config.max_highlights,
        "min_separation_s": run.config.min_separation_s,
        "team_names": _team_names(run),
        "cue_weights": dict(highlights_mod.CUE_WEIGHTS),
        "min_peak_score": highlights_mod.MIN_PEAK_SCORE,
        "attack_direction": {"left": 1, "right": -1},
        "attack_known": _attack_direction_known(run),
    }


def _pose_request_params() -> dict[str, Any]:
    """The gbf.measure parameters that decide which boxes get a pose (and nothing else, so
    a change of a measurement threshold never re-runs the expensive pose stage)."""
    params = measure_mod.DEFAULT_PARAMS
    return {
        "contact_candidates": params.contact_candidates,
        "contact_candidate_max_bh": params.contact_candidate_max_bh,
    }


def _params_pose(run: _Run) -> dict[str, Any]:
    estimator = run.config.pose_estimator
    if estimator is not None:
        return {
            "model_id": str(getattr(estimator, "model_id", "injected")),
            "input_size": list(getattr(estimator, "input_size", ())),
            "input_channels": str(getattr(estimator, "input_channels", "rgb")),
            "injected_estimator": True,
            "requests": _pose_request_params(),
        }
    model = POSE_MODELS[run.config.pose_model]
    if model is None:
        return {"model_id": None}
    from gbf import models as models_mod

    pin = models_mod.resolve_pin(model)
    return {
        "model_id": pin.model_id,
        "onnx_sha256": pin.onnx_sha256,
        "input_size": list(pin.input_size),
        "input_channels": pose_mod.DEFAULT_INPUT_CHANNELS,
        "backend": "onnxruntime",
        "device": "cpu",
        "requests": _pose_request_params(),
        "range_merge_gap_frames": POSE_RANGE_MERGE_GAP,
    }


def _params_lift(run: _Run) -> dict[str, Any]:
    return {
        "score_threshold": lift_mod.DEFAULT_SCORE_THRESHOLD,
        "cameras": "pitch-usable calibrations on signals ev_pitch frames only",
    }


def _params_measure(run: _Run) -> dict[str, Any]:
    return {"params": measure_mod.DEFAULT_PARAMS.as_dict(), "code": run.code}


def _params_render(run: _Run) -> dict[str, Any]:
    return {
        "full_clip": run.config.full_clip,
        "team_names": _team_names(run),
        "watermark": WATERMARK if WATERMARK_TAGS & set(run.restrictions) else None,
        "audio_source": run.audio.name,
        "audio_offset_s": run.config.audio_offset_s,
        "edit": {"slow_factor": 4, "slow_before_s": 0.6, "slow_after_s": 0.4, "freeze_s": 2.0},
        "card_rows": CARD_MAX_ROWS,
        "card_moment_rows": CARD_MOMENT_ROWS,
        "facing_arrow": {
            "length_m": render_mod.FACING_ARROW_LENGTH_M,
            "gates": asdict(_facing_gates()),
        },
    }


def _params_report(run: _Run) -> dict[str, Any]:
    # The notes collected from the stages (incomplete ball file, partial audio) are part of
    # the report, so a change of them re-runs the report.
    return {"rep_err_max_px": run.config.rep_err_max_px, "notes_zh": list(dict.fromkeys(run.notes))}


def _render_inputs(run: _Run) -> list[tuple[str, Path]]:
    names = [
        TRACKS,
        CALIB,
        "refine/refine.json",
        "ball_track.parquet",
        "kinematics.parquet",
        "pose.parquet",
        "lift3d.parquet",
        "signals.parquet",
        FINAL,
        "cuts.json",
        "shots.parquet",
        "measurements.jsonl",
    ]
    return [("video", run.video), ("audio_source", run.audio)] + [
        (name, run.path(name)) for name in names
    ]


def _report_inputs(run: _Run) -> list[tuple[str, Path]]:
    names = [
        FINAL,
        "measurements.jsonl",
        "signals.parquet",
        "shots.parquet",
        "cuts.json",
        "import/calib.parquet",
        CALIB,
        "refine/refine.json",
    ]
    if run.config.labels is not None:
        names.append("label_eval.json")
    found = [(name, run.path(name)) for name in names]
    if run.config.labels is not None:
        # the report quotes where the labels come from out of the label file itself
        found.append(("labels", Path(run.config.labels)))
    receipt = run.path("receipts/render.json")
    if receipt.is_file():
        found.append(("receipts/render.json", receipt))
    return found


def _refine_inputs(run: _Run) -> list[tuple[str, Path]]:
    # the team check reads shirt colours from the video
    found = _with_video("import/tracks.parquet", "import/calib.parquet", "shots.parquet")(run)
    for name in (enhance_mod.KEYPOINTS_NAME, enhance_mod.KEYPOINTS_RECEIPT_NAME):
        if run.config.refine_calibration and (run.sngs / name).is_file():
            found.append((f"sngs/{name}", run.sngs / name))
    return found


def _params_refine(run: _Run) -> dict[str, Any]:
    return {
        "refine_calibration": bool(run.config.refine_calibration),
        "settings": asdict(
            calib_refine_mod.RefineSettings(rep_err_max_px=run.config.rep_err_max_px)
        ),
        "jersey": {"min_mean_conf": 0.8, "min_reads": 5, "min_agreement": 0.7},
        "id_stride": track_tools_mod.ID_STRIDE,
        "team_check": team_check_mod.DEFAULT_PARAMS.as_dict(),
        "role_check": role_check_mod.DEFAULT_PARAMS.as_dict(),
        "side_check": side_check_mod.DEFAULT_PARAMS.as_dict(),
    }


def _moments_inputs(run: _Run) -> list[tuple[str, Path]]:
    found = _internal(
        "highlights.json",
        "pose.parquet",
        "ball_track.parquet",
        "ball_events.parquet",
        "ball_contacts.parquet",
        "shots.parquet",
        TRACKS,
        "refine/refine.json",
    )(run)
    if run.config.labels is not None:
        found.append(("labels", Path(run.config.labels)))
    return found


def _params_moments(run: _Run) -> dict[str, Any]:
    return {
        "team_names": _team_names(run),
        "min_score": moments_mod.MIN_SCORE,
        "fall": [
            moments_mod.UPRIGHT_DEG,
            moments_mod.FALL_DOWN_DEG,
            moments_mod.FALL_MAX_S,
            moments_mod.FALL_HIP_RATIO,
        ],
        "contact": [
            moments_mod.CONTACT_BODY_H,
            moments_mod.CONTACT_GAP_FRAMES,
            moments_mod.MIN_BODY_PX,
        ],
        "rank": [moments_mod.MAIN_SHARE, moments_mod.MAIN_MAX],
        "pose_min_mean_score": measure_mod.DEFAULT_PARAMS.pose_min_mean_score,
        "touches": measure_mod.DEFAULT_PARAMS.as_dict(),
        "release": [
            moments_mod.RELEASE_NEAR_BODY_H,
            moments_mod.RELEASE_AWAY_BODY_H,
            moments_mod.RELEASE_WITHIN_S,
            moments_mod.RELEASE_MIN_SPEED_BH_S,
        ],
        "gesture": [
            moments_mod.RAISE_SHOULDER_W,
            moments_mod.ARM_STRAIGHT_DEG,
            moments_mod.GESTURE_MIN_S,
            moments_mod.GESTURE_GAP_S,
            moments_mod.GESTURE_MIN_SHOULDER_PX,
        ],
        "reaction_lead_s": moments_mod.REACTION_LEAD_S,
        "shot_names": [
            dict(highlights_mod.SHOT_TYPE_NAMES_ZH),
            highlights_mod.SHOT_UPPER_BODY_MIN_BOX_H_PX,
        ],
    }


def _stage_table() -> dict[str, _Stage]:
    common = ("Values are computed automatically; their errors are unmeasured (ADR 0004).",)
    stages = [
        _Stage(
            "import",
            _stage_import,
            (import_mod,),
            _inputs_import,
            _params_import,
            "The plugin-sngs export was re-hashed against its receipt, validated and "
            "converted to tracks.parquet and calib.parquet.",
            ("rep_err_px is the plugin's self-consistency, not a measured error.", *common),
        ),
        _Stage(
            "shots",
            _stage_shots,
            (shots_mod, cuts_mod),
            _with_video("import/tracks.parquet", "import/calib.parquet"),
            _params_shots,
            "Hard cuts detected on the canonical frames and the shot table built from them.",
            ("The cut threshold is an empirical starting value, not validated.", *common),
        ),
        _Stage(
            "refine",
            _stage_refine,
            (
                enhance_mod,
                calib_refine_mod,
                track_tools_mod,
                team_check_mod,
                _check_team_labels,
                role_check_mod,
                _check_role_labels,
                side_check_mod,
                _check_sides,
                _sides_note,
            ),
            _refine_inputs,
            _params_refine,
            "Track ids split at cuts, unconfident jersey numbers dropped, and (with the "
            "plugin's pitch keypoints) the calibration refitted with a fixed camera centre "
            "per camera and smooth rotation and zoom.",
            ("The refit is checked on held-out keypoints only: self-consistency.", *common),
        ),
        _Stage(
            "kinematics",
            _stage_kinematics,
            (kinematics_mod,),
            _internal(TRACKS, "shots.parquet"),
            lambda run: dict(KINEMATICS_PARAMS),
            "Smoothed pitch positions and speeds on pitch-usable frames.",
            ("Speeds are trajectory-level estimates from a per-frame calibration.", *common),
        ),
        _Stage(
            "ball",
            _stage_ball,
            (ball_mod,),
            lambda run: [
                ("ball_raw.jsonl", run.ball_raw),
                *_internal(TRACKS, CALIB, "shots.parquet")(run),
            ],
            _params_ball,
            "Ball track on the original frames and ball events (strike and contact profiles).",
            ("Ball pitch positions assume a grounded ball.", *common),
        ),
        _Stage(
            "audio",
            _stage_audio,
            (audio_mod,),
            lambda run: [("audio_source", run.audio)],
            _params_audio,
            "Loudness, spectral flux and excitement of the original audio per frame.",
            ("A selection cue only; never shown as a number of the match.",),
        ),
        _Stage(
            "highlights",
            _stage_highlights,
            (highlights_mod, moments_mod, _attack_direction_known, _team_names),
            _internal(
                TRACKS,
                "refine/refine.json",
                "kinematics.parquet",
                "ball_track.parquet",
                "ball_events.parquet",
                "shots.parquet",
                "cuts.json",
                "audio.parquet",
            ),
            _params_highlights,
            "Per-frame highlight cues and the highlight moments of the clip.",
            ("Weights and thresholds are unvalidated starting values.",),
        ),
        _Stage(
            "pose",
            _stage_pose,
            (
                pose_mod,
                measure_mod.pose_requests,
                measure_mod.contact_candidates,
            ),
            _with_video(
                "highlights.json",
                TRACKS,
                "ball_events.parquet",
                "ball_contacts.parquet",
            ),
            _params_pose,
            "2D Halpe-26 pose of the involved players inside the highlight windows.",
            (
                "RTMPose weights: training-data licences unverified; experimental, private use.",
                *common,
            ),
        ),
        _Stage(
            "lift3d",
            _stage_lift3d,
            (lift_mod,),
            _internal("pose.parquet", CALIB, "signals.parquet"),
            _params_lift,
            "Calibrated height-prior skeletons on pitch frames; image-plane angles elsewhere.",
            ("The height-prior skeleton is an approximation, not a measured 3D pose.", *common),
        ),
        _Stage(
            "moments",
            _stage_moments,
            (
                moments_mod,
                label_eval_mod,
                measure_mod,
                highlights_mod.format_time,
                highlights_mod.shot_name_zh,
            ),
            _moments_inputs,
            _params_moments,
            "Frame-precise key moments (contact, touches, ball release, falls) from 2D pose "
            "and the ball on the original frames, and the main/candidate ranking.",
            ("Image-plane estimates; thresholds checked on one development clip only.",),
        ),
        _Stage(
            "measure",
            _stage_measure,
            (
                measure_mod,
                track_tools_mod,
                _measure_context,
                _attack_direction_known,
                _team_verdicts,
            ),
            _measure_inputs,
            _params_measure,
            "Evidence records of every highlight (urn:gbf:schema:measurement).",
            ("No error was measured: every record says error.kind unmeasured.", *common),
        ),
        _Stage(
            "render",
            _stage_render,
            (
                render_mod,
                fonts_mod,
                measure_mod,
                report_mod,
                _card,
                _card_rows,
                _moment_rows,
                display_record,
                _FullClipAnnotator,
                _style,
                _render_team_names,
                _facing_gates,
                track_tools_mod,
                _team_verdicts,
            ),
            _render_inputs,
            _params_render,
            "Annotated highlight edits and the annotated full clip on the original frames.",
            ("Overlays show unverified estimates; renders contain private broadcast footage.",),
        ),
        _Stage(
            "report",
            _stage_report,
            (report_mod, label_eval_mod, _calibration_summary, _tool_versions, display_record),
            _report_inputs,
            _params_report,
            "The Chinese HTML report of the run.",
            ("Private material: the report links broadcast footage.",),
        ),
    ]
    return {stage.name: stage for stage in stages}


# --------------------------------------------------------------------------- receipts


def _stage_key(run: _Run, stage: _Stage, inputs: list[dict[str, Any]], params: Any) -> str:
    # The run's tags are part of the key: a re-run with other tags (another pose model,
    # --restrictions) writes every artifact again with them, never keeps old ones. So is the
    # whole package source: a stage's outputs never outlive an edit of the code it runs.
    return canonical_sha256(
        {
            "stage": stage.name,
            "inputs": [[ref["name"], ref["sha256"]] for ref in inputs],
            "parameters": params,
            "code": _fingerprint(stage.function, stage.modules),
            "source": run.source,
            "restrictions": sorted(set(run.restrictions)),
        }
    )


def _outputs_intact(run: _Run, receipt: Mapping[str, Any]) -> bool:
    for entry in receipt.get("outputs", []):
        relpath = entry.get("relpath")
        if not relpath:
            return False
        path = run.path(str(relpath))
        if not path.is_file() or run.sha256(path) != entry.get("sha256"):
            return False
    return True


def _read_receipt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_receipt(
    run: _Run,
    stage: _Stage,
    key: str,
    inputs: list[dict[str, Any]],
    params: Any,
    outputs: list[str],
    wall_s: float,
    notes: Sequence[str] = (),
) -> None:
    receipt = {
        "schema_version": 1,
        "kind": "run",
        "status": "passed",
        "created_at": _utc_now(),
        "producer": f"{PRODUCER}.{stage.name}",
        "code": dict(run.code),
        "inputs": inputs,
        "outputs": [
            run.file_ref(name, run.path(name), internal=True) for name in dict.fromkeys(outputs)
        ],
        "scope": stage.scope,
        "limitations": list(stage.limitations),
        "restrictions": list(run.restrictions),
        "environment": {"python": platform.python_version(), "gbf": gbf.__version__},
        "result": {
            "stage": stage.name,
            "key": key,
            "source_sha256": run.source,
            "parameters": params,
            "wall_s": round(wall_s, 3),
            "facts": _jsonable(run.result_extras.get(stage.name, {})),
            "notes_zh": list(notes),
        },
    }
    contracts.validate(receipt, "receipt")
    path = run.path(f"receipts/{stage.name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    _json_write(path, receipt)


def _restore(run: _Run, stage: _Stage, receipt: Mapping[str, Any]) -> None:
    """Facts and report notes of a stage that is not run again, from its receipt."""
    result = receipt.get("result", {})
    run.result_extras[stage.name] = result.get("facts", {})
    run.notes.extend(str(note) for note in result.get("notes_zh", []))


def _run_stage(run: _Run, stage: _Stage) -> StageResult:
    selected = stage.name in run.config.stages
    receipt_path = run.path(f"receipts/{stage.name}.json")
    relative_receipt = f"receipts/{stage.name}.json"
    old = _read_receipt(receipt_path)
    if not selected:
        if old is None or not _outputs_intact(run, old):
            raise PipelineError(
                stage.name,
                [
                    f"stage {stage.name} is not in --stages and its outputs are missing or "
                    "changed; add it to --stages"
                ],
            )
        outputs = tuple(str(entry["relpath"]) for entry in old.get("outputs", []))
        _restore(run, stage, old)
        run.log(f"[{stage.name}] kept (not selected)")
        return StageResult(stage.name, "kept", 0.0, relative_receipt, outputs)
    missing_inputs = [name for name, path in stage.inputs(run) if not path.is_file()]
    if missing_inputs:
        raise PipelineError(stage.name, [f"input missing: {name}" for name in missing_inputs])
    inputs = [
        run.file_ref(name, path, internal=_is_within(path, run.dir))
        for name, path in stage.inputs(run)
    ]
    params = _jsonable(stage.parameters(run))
    key = _stage_key(run, stage, inputs, params)
    if (
        not run.config.force
        and old is not None
        and old.get("result", {}).get("key") == key
        and _outputs_intact(run, old)
    ):
        outputs = tuple(str(entry["relpath"]) for entry in old.get("outputs", []))
        _restore(run, stage, old)
        run.log(f"[{stage.name}] skipped (inputs, parameters and code unchanged)")
        return StageResult(stage.name, "skipped", 0.0, relative_receipt, outputs)
    run.log(f"[{stage.name}] running ...")
    started = time.perf_counter()
    notes_before = len(run.notes)
    outputs_list = stage.function(run)
    wall = time.perf_counter() - started
    notes = run.notes[notes_before:]
    _write_receipt(run, stage, key, inputs, params, outputs_list, wall, notes)
    run.log(f"[{stage.name}] done in {wall:.1f} s")
    return StageResult(stage.name, "ran", wall, relative_receipt, tuple(outputs_list))


# --------------------------------------------------------------------------- entry points


def _check_config(config: PipelineConfig) -> list[str]:
    problems: list[str] = []
    for name in ("video", "audio_source", "ball_raw"):
        value = getattr(config, name)
        if not Path(value).is_file():
            problems.append(f"--{name.replace('_', '-')}: not a file: {value}")
    if not Path(config.sngs_dir).is_dir():
        problems.append(f"--sngs: not a directory: {config.sngs_dir}")
    else:
        problems.extend(_missing_receipts(config))
    problems.extend(_unusable_receipts(config))
    unknown = [stage for stage in config.stages if stage not in STAGES]
    if unknown:
        problems.append(f"--stages: unknown stage(s) {', '.join(unknown)}; choose from {STAGES}")
    if config.pose_model not in POSE_MODELS:
        problems.append(f"--pose-model must be one of {sorted(POSE_MODELS)}")
    if (
        isinstance(config.max_highlights, bool)
        or not isinstance(config.max_highlights, int)
        or config.max_highlights < 1
    ):
        problems.append(f"--max-highlights must be an integer >= 1, got {config.max_highlights!r}")
    tags = sorted(set(config.restrictions))
    problems.extend(
        f"--restrictions {problem}"
        for problem in contracts.def_problems(tags, "restrictions", "restrictions")
    )
    problems.extend(_licence_problems(config))
    if config.team_names is not None:
        extra = sorted(set(config.team_names) - {"left", "right"})
        if extra:
            problems.append(f"--team-names: unknown side(s) {', '.join(extra)}; use left/right")
    problems.extend(
        check_out_dir(config.out_dir, repo_root=config.repo_root, git_ceiling=config.git_ceiling)
    )
    return problems


def _plugin_restrictions(sngs_dir: Path) -> list[str]:
    receipt = sngs_dir / import_mod.RECEIPT_NAME
    data = read_json(receipt)
    tags = data.get("restrictions") if isinstance(data, dict) else None
    if not isinstance(tags, list):
        raise PipelineError("restrictions", [f"{receipt.name} declares no restrictions list"])
    found = contracts.def_problems(tags, "restrictions", "restrictions")
    if found:
        raise PipelineError("restrictions", [f"{receipt.name}: {p}" for p in found])
    return [str(tag) for tag in tags]


def _input_receipts(config: PipelineConfig) -> list[tuple[str, Path]]:
    """(input, receipt path) of every input that comes with a receipt: the plugin export,
    its pitch keypoints when the refit uses them, and the ball file's optional receipt."""
    sngs = Path(config.sngs_dir)
    found = [("plugin", sngs / import_mod.RECEIPT_NAME)]
    if config.refine_calibration and (sngs / enhance_mod.KEYPOINTS_NAME).is_file():
        found.append(("keypoints", sngs / enhance_mod.KEYPOINTS_RECEIPT_NAME))
    ball_receipt = Path(config.ball_raw).parent / BALL_RECEIPT_NAME
    if ball_receipt.is_file():
        found.append(("ball", ball_receipt))
    return found


def _missing_receipts(config: PipelineConfig) -> list[str]:
    """Inputs that come without the receipt they need: the plugin export without
    plugin-receipt.json, and its pitch keypoints, when the refit would use them, without
    theirs. The run's restrictions and licences are read from these receipts, so such an
    input is refused before anything is written."""
    problems = []
    for what, receipt in _input_receipts(config):
        if receipt.is_file():
            continue
        if what == "plugin":
            problems.append(
                f"--sngs: {receipt.name} is missing: the plugin export needs its receipt "
                "(licence, restrictions and the sha256 of its files)"
            )
        else:
            problems.append(
                f"--sngs: {receipt.name} is missing: {enhance_mod.KEYPOINTS_NAME} needs its "
                "receipt (licence and restrictions), or run with --no-calib-refine"
            )
    return problems


def _unusable_receipts(config: PipelineConfig) -> list[str]:
    """Receipts that are there but unusable: not readable as JSON, or without a valid
    restrictions list. The run's restrictions are read from them once the run folder exists,
    so such an input is refused before anything is written, as a missing receipt is."""
    problems: list[str] = []
    for _what, receipt in _input_receipts(config):
        if not receipt.is_file():
            continue  # refused by _missing_receipts, or optional (the ball file's)
        try:
            data = read_json(receipt)
        except (OSError, ValueError, RecursionError) as exc:
            detail = exc.strerror if isinstance(exc, OSError) else str(exc)
            problems.append(f"{receipt.name} cannot be read as JSON: {detail}")
            continue
        try:
            _declared_tags(data, receipt.name)
        except PipelineError as exc:
            problems.extend(exc.problems)
    return problems


def _licence_problems(config: PipelineConfig) -> list[str]:
    """Inputs whose declared licence is outside the permissive allowlist and was not named
    with --allow-license (a receipt without a licence counts as NOASSERTION)."""
    try:
        allowed = licenses_mod.effective_allowlist(config.allow_licenses)
    except licenses_mod.LicenseExpressionError as exc:
        return [f"--allow-license: {exc}"]
    problems = []
    for what, receipt in _input_receipts(config):
        if not receipt.is_file():
            continue  # refused by _missing_receipts
        try:
            data = read_json(receipt)
        except (OSError, ValueError):
            continue
        expression = data.get("license") if isinstance(data, dict) else None
        expression = expression if isinstance(expression, str) and expression else "NOASSERTION"
        for problem in licenses_mod.license_problems(expression, allowed):
            problems.append(
                f"{what} input ({receipt.name}): {problem}; accept it explicitly with "
                "--allow-license <id>"
            )
    return problems


def _declared_tags(data: Any, where: str) -> list[str]:
    tags = data.get("restrictions") if isinstance(data, dict) else None
    if not isinstance(tags, list):
        raise PipelineError("restrictions", [f"{where} declares no restrictions list"])
    found = contracts.def_problems(tags, "restrictions", "restrictions")
    if found:
        raise PipelineError("restrictions", [f"{where}: {p}" for p in found])
    return [str(tag) for tag in tags]


def _run_restrictions(run: _Run) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    """The run's tags: the union of every input's tags and ``config.restrictions`` (ADR
    0002 rule 1), and where each input's tags came from. An input without declared tags
    (a ball file without a receipt, labels without a "restrictions" list) adds no_share;
    pose from the pinned weights adds POSE_MODEL_RESTRICTIONS."""
    sources: dict[str, list[str]] = {"plugin": _plugin_restrictions(run.sngs)}
    for what, receipt in _input_receipts(run.config)[1:]:
        sources[what] = _declared_tags(read_json(receipt), receipt.name)
    if "ball" not in sources:
        sources["ball"] = ["no_share"]
    if run.config.labels is not None:
        labels = read_json(Path(run.config.labels))
        declared = labels.get("restrictions") if isinstance(labels, dict) else None
        sources["labels"] = (
            _declared_tags(labels, Path(run.config.labels).name)
            if declared is not None
            else ["no_share"]
        )
    if run.config.pose_estimator is None and POSE_MODELS.get(run.config.pose_model):
        sources["pose_model"] = list(POSE_MODEL_RESTRICTIONS)
    sources["config"] = list(run.config.restrictions)
    tags = sorted({tag for found in sources.values() for tag in found})
    return tuple(tags), sources


def _video_facts(run: _Run) -> _VideoFacts:
    facts = probe(run.video)
    problems = []
    rate = None
    try:
        rate = parse_rate(str(facts.r_frame_rate))
    except ValueError:
        problems.append(f"the video has no usable frame rate ({facts.r_frame_rate!r})")
    if rate is not None and not facts.is_cfr_at(rate):
        problems.append(
            "the video is not constant-frame-rate; normalise it first (python -m gbf ingest) "
            "so that frame indices are canonical"
        )
    if problems or rate is None:
        raise PipelineError("video", problems)
    return _VideoFacts(
        fps=rate,
        n_frames=int(facts.nb_frames_decoded),
        width=int(facts.frame_width),
        height=int(facts.frame_height),
        sha256=facts.sha256,
        file_bytes=int(facts.file_bytes),
    )


def _write_run_json(run: _Run, status: str, error: list[str] | None, wall_s: float) -> Path:
    facts = run.facts
    document = {
        "schema_version": 1,
        "kind": "run",
        "status": status,
        "created_at": _utc_now(),
        "producer": PRODUCER,
        "code": dict(run.code),
        "inputs": [
            run.file_ref(name, path, internal=False)
            for name, path in (
                ("video", run.video),
                ("audio_source", run.audio),
                ("ball_raw.jsonl", run.ball_raw),
                ("ball-receipt.json", run.ball_raw.parent / BALL_RECEIPT_NAME),
                *((f"sngs/{n}", run.sngs / n) for n in _SNGS_FILES),
                ("sngs/" + enhance_mod.KEYPOINTS_NAME, run.sngs / enhance_mod.KEYPOINTS_NAME),
                (
                    "sngs/" + enhance_mod.KEYPOINTS_RECEIPT_NAME,
                    run.sngs / enhance_mod.KEYPOINTS_RECEIPT_NAME,
                ),
                *((("labels", Path(run.config.labels)),) if run.config.labels else ()),
            )
            if path.is_file()
        ],
        "outputs": [
            run.file_ref(f"receipts/{result.name}.json", run.path(result.receipt), internal=True)
            for result in run.results
            if run.path(result.receipt).is_file()
        ],
        "scope": "One run of the M1 highlight pipeline: which stages ran or were reused, their "
        "receipts and wall times, and the restrictions of every artifact.",
        "limitations": [
            "Every number in the run is an automatic estimate whose error is unmeasured.",
            "Highlight thresholds and cue weights are unvalidated starting values.",
        ],
        "restrictions": list(run.restrictions),
        "environment": {"python": platform.python_version(), "gbf": gbf.__version__},
        "result": _jsonable(
            {
                "video": None
                if facts is None
                else {
                    "file_name": run.video.name,
                    "sha256": facts.sha256,
                    "fps": format_rate(facts.fps),
                    "frames": facts.n_frames,
                    "width": facts.width,
                    "height": facts.height,
                },
                "audio_source": run.audio.name,
                "source_sha256": run.source,
                "restriction_sources": run.restriction_sources,
                "allow_licenses": list(run.config.allow_licenses),
                "parameters": {
                    "stages": list(run.config.stages),
                    "force": run.config.force,
                    "max_highlights": run.config.max_highlights,
                    "min_separation_s": run.config.min_separation_s,
                    "pose_model": run.config.pose_model,
                    "team_names": _configured_team_names(run),
                    "full_clip": run.config.full_clip,
                    "audio_offset_s": run.config.audio_offset_s,
                    "rep_err_max_px": run.config.rep_err_max_px,
                },
                "stages": [
                    {
                        "name": result.name,
                        "status": result.status,
                        "wall_s": round(result.wall_s, 3),
                        "receipt": result.receipt,
                    }
                    for result in run.results
                ],
                "wall_s": round(wall_s, 3),
                "notes_zh": list(dict.fromkeys(run.notes)),
                "error": error,
                "facts": run.result_extras,
            }
        ),
    }
    contracts.validate(document, "receipt")
    path = run.path("run.json")
    _json_write(path, document)
    return path


_SNGS_FILES = (import_mod.DETECTIONS_NAME, import_mod.CALIB_JSONL_NAME, import_mod.RECEIPT_NAME)


def run_pipeline(config: PipelineConfig) -> RunResult:
    """Run (or bring up to date) every stage of the pipeline; see the module docstring.

    Raises PipelineError listing every problem with the configuration (nothing is written
    then), and PipelineError or the stage's own exception when a stage fails; run.json is
    written in both cases once the run directory exists.
    """
    problems = _check_config(config)
    if problems:
        raise PipelineError("highlights", problems)
    run = _Run(config)
    run.dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        run.restrictions, run.restriction_sources = _run_restrictions(run)
        run.log(f"video: probing {run.video.name} ...")
        probe_started = time.perf_counter()
        run.facts = _video_facts(run)
        run.result_extras["probe"] = {"wall_s": round(time.perf_counter() - probe_started, 3)}
        run.log(
            f"video: {run.facts.n_frames} frames, {format_rate(run.facts.fps)} fps, "
            f"{run.facts.width}x{run.facts.height}; restrictions {', '.join(run.restrictions)}"
        )
        table = _stage_table()
        for name in STAGES:
            run.results.append(_run_stage(run, table[name]))
    except BaseException as exc:
        error = list(getattr(exc, "problems", [])) or [f"{type(exc).__name__}: {exc}"]
        try:
            _write_run_json(run, "failed", error, time.perf_counter() - started)
        except Exception as secondary:  # noqa: BLE001 - never hide the stage's own error
            run.log(f"run.json could not be written: {secondary}")
        raise
    run_json = _write_run_json(run, "passed", None, time.perf_counter() - started)
    if run.path(FINAL).is_file():
        highlights = run.final_highlights()
    elif run.path("highlights.json").is_file():
        highlights = run.highlights()
    else:
        highlights = {}
    return RunResult(run.dir, tuple(run.results), highlights, run_json)


def _parse_team_names(text: str) -> dict[str, str]:
    names: dict[str, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        side, sep, name = part.partition("=")
        if not sep or not name.strip():
            raise argparse.ArgumentTypeError(f"expected left=NAME,right=NAME, got {text!r}")
        names[side.strip()] = name.strip()
    return names


def _parse_stages(text: str) -> tuple[str, ...]:
    if text.strip() == "all":
        return STAGES
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gbf highlights",
        description="Find the highlight moments of a broadcast clip and analyse them on the "
        "original frames (docs/design/m1-highlights.md).",
    )
    parser.add_argument("--video", required=True, type=Path, help="canonical CFR video")
    parser.add_argument(
        "--audio-source", required=True, type=Path, help="file that carries the original audio"
    )
    parser.add_argument("--sngs", required=True, type=Path, help="plugin-sngs export directory")
    parser.add_argument("--ball", required=True, type=Path, help="ball_raw.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="run directory (outside git)")
    parser.add_argument(
        "--team-names",
        type=_parse_team_names,
        default=None,
        help="left=NAME,right=NAME from provider or user metadata",
    )
    parser.add_argument("--max-highlights", type=int, default=6)
    parser.add_argument("--min-separation-s", type=float, default=4.0)
    parser.add_argument("--pose-model", choices=sorted(POSE_MODELS), default="x")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument(
        "--stages",
        type=_parse_stages,
        default=STAGES,
        help=f"comma-separated stages to (re)run, or 'all' (default): {','.join(STAGES)}",
    )
    parser.add_argument("--force", action="store_true", help="re-run the selected stages")
    parser.add_argument(
        "--restrictions",
        default="",
        help="extra restriction tags for every artifact, added to the inputs' tags and the "
        "default no_share",
    )
    parser.add_argument(
        "--allow-license",
        action="append",
        default=[],
        metavar="ID",
        help="accept inputs under this licence (SPDX id, repeatable or comma-separated), "
        "for example GPL-3.0-only for the plugin-sngs export",
    )
    parser.add_argument("--no-full-clip", action="store_true", help="skip the full clip")
    parser.add_argument("--audio-offset-s", type=float, default=0.0)
    parser.add_argument(
        "--no-calib-refine",
        action="store_true",
        help="keep the plugin's per-frame calibration (skip the fixed-centre refit)",
    )
    parser.add_argument(
        "--labels", type=Path, default=None, help="labels to compare the highlights with"
    )
    return parser


def main(argv: list[str]) -> int:
    """``python -m gbf highlights ...``; returns 0 on success, 2 for refused input, 1 when a
    stage fails."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    config = PipelineConfig(
        video=args.video,
        audio_source=args.audio_source,
        sngs_dir=args.sngs,
        ball_raw=args.ball,
        out_dir=args.out,
        team_names=args.team_names,
        max_highlights=args.max_highlights,
        min_separation_s=args.min_separation_s,
        pose_model=args.pose_model,
        models_dir=args.models_dir,
        stages=args.stages,
        force=args.force,
        restrictions=tuple(
            sorted(
                set(DEFAULT_RESTRICTIONS)
                | {t.strip() for t in args.restrictions.split(",") if t.strip()}
            )
        ),
        allow_licenses=tuple(
            sorted({i.strip() for item in args.allow_license for i in item.split(",") if i.strip()})
        ),
        full_clip=not args.no_full_clip,
        audio_offset_s=args.audio_offset_s,
        refine_calibration=not args.no_calib_refine,
        labels=args.labels,
    )
    problems = _check_config(config)
    if problems:
        print(str(PipelineError("highlights", problems)), file=sys.stderr)
        return 2
    try:
        result = run_pipeline(config)
    except Exception as exc:  # noqa: BLE001 - report every stage failure, then exit 1
        print(f"highlights: failed: {exc}", file=sys.stderr)
        return 1
    print(summary_text(result))
    return 0


def summary_text(result: RunResult) -> str:
    """Plain-text summary of a run: the highlights and every stage's wall time."""
    lines = ["highlights:"]
    fps = result.highlights.get("fps", "25/1") if result.highlights else "25/1"
    for item in (result.highlights or {}).get("highlights", []):
        start, peak, end = (
            highlights_mod.format_time(item[key], fps)
            for key in ("start_frame", "peak_frame", "end_frame")
        )
        lines.append(
            f"  {item['id']} {item['type']:<16} {start}-{end} peak {peak} "
            f"(frame {item['peak_frame']}) score {item['score']}"
        )
    lines.append("stages:")
    for stage in result.stages:
        lines.append(f"  {stage.name:<11} {stage.status:<8} {stage.wall_s:8.1f} s")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_RESTRICTIONS",
    "FULL_CLIP_ID",
    "KINEMATICS_PARAMS",
    "POSE_MODELS",
    "STAGES",
    "PipelineConfig",
    "PipelineError",
    "RunResult",
    "StageResult",
    "check_out_dir",
    "code_version",
    "display_record",
    "main",
    "read_restrictions",
    "run_pipeline",
    "summary_text",
    "write_parquet",
]
