"""Core side of the plugin protocol (plan 3.1; ADR 0001 sections 3 and 4; gates G5, G17).

A plugin is a separate program with its own licence and environment. The core never imports
it; the two exchange files in one job directory (every relpath is relative to it):

    <task>-<first 16 hex digits of the job sha256>/
        job.json              written once by the core (urn:gbf:schema:job)
        in/...                inputs, copied in by the core
        out/...               the plugin's outputs, exactly at the requested relpaths
        plugin-receipt.json   written by the plugin (urn:gbf:schema:plugin-receipt)

1. create_job copies the inputs in, hashes them and writes job.json through write_job,
   which checks the job against its schema and against the files (sizes, sha256, frame
   ranges, config hash). A job directory that already exists is an error (G17).
2. run_plugin starts the plugin: argv = command + [absolute path of job.json], working
   directory = the job directory, a whitelisted environment (plugin_environment), a timeout.
3. accept_outputs reads plugin-receipt.json and accepts the outputs only if the receipt
   follows its schema and belongs to this job.json, the plugin reports status "ok", its
   licence is permissive or explicitly allowed, every requested output exists with the
   listed size and sha256 and passes its format check (calib.v1: gbf.plugins.calib), no
   other file sits in out/, the plugin decoded the canonical frames the job names (pixel
   hashes, G5), its restrictions include the job's, and the inputs are unchanged. It lists
   every problem at once.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gbf import contracts
from gbf.plugins.calib import CALIB_FORMAT, read_calib
from gbf.plugins.errors import CalibFormatError, JobError, OutputsRejectedError, PluginRunError
from gbf.plugins.licenses import (
    LicenseExpressionError,
    effective_allowlist,
    license_problems,
    parse_license_expression,
)
from gbf.util.hashing import canonical_sha256, sha256_bytes, sha256_file
from gbf.util.io import write_text

JOB_FILE = "job.json"
RECEIPT_FILE = "plugin-receipt.json"
INPUT_DIR = "in"
OUTPUT_DIR = "out"
JOB_SCHEMA = "job"
RECEIPT_SCHEMA = "plugin-receipt"
SCHEMA_VERSION = 1
TAIL_CHARS = 4000
"""How much of a plugin's stdout and stderr is kept (characters, from the end)."""

PLUGIN_ENV_KEEP = frozenset(
    {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_ARCHITEW6432",
        "OS",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "SOURCE_DATE_EPOCH",
        # Kept so that the gate's offline guard (a sitecustomize on PYTHONPATH) also reaches
        # plugin processes started during a gated test run.
        "PYTHONPATH",
    }
)
"""Environment variables a plugin inherits (compared upper case), plus GBF_* ones. Anything
else -- API keys, tokens, proxies, the user name -- is not passed on."""
PLUGIN_ENV_FORCED = {"PYTHONUTF8": "1", "PYTHONHASHSEED": "0"}
_SECRET_ENV_NAME = re.compile(
    r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|PRIVATE_KEY)$", re.IGNORECASE
)


# --- jobs ------------------------------------------------------------------------------
@dataclass(frozen=True)
class JobInput:
    """A file to hand to a plugin: copied to ``relpath`` (under in/) of the job directory."""

    name: str
    source: Path
    relpath: str
    restrictions: tuple[str, ...] = ()


def job_text(job: Mapping[str, Any]) -> str:
    """The exact text of job.json: UTF-8 JSON, sorted keys, two-space indent, final LF."""
    return json.dumps(job, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def job_sha256(job: Mapping[str, Any]) -> str:
    """sha256 of the job.json bytes for ``job`` (what a plugin's receipt must name)."""
    return sha256_bytes(job_text(job).encode("utf-8"))


def job_dir_name(job: Mapping[str, Any]) -> str:
    """Directory name for a job: ``<task>-<first 16 hex digits of job_sha256>``."""
    return f"{job['task']}-{job_sha256(job)[:16]}"


def build_job(
    *,
    task: str,
    inputs: Sequence[Mapping[str, Any]],
    frames: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    allow_licenses: Iterable[str] = (),
) -> dict[str, Any]:
    """Assemble a job dict: config_sha256 from ``config``, allowed_licenses = permissive
    allowlist plus ``allow_licenses``, restrictions = union of the inputs' restrictions.

    ``inputs`` are entries with name, relpath, sha256, bytes and restrictions (create_job
    computes them). The result is not validated here; write_job validates it.
    """
    config_dict = dict(config)
    restrictions = sorted({tag for item in inputs for tag in item.get("restrictions", ())})
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "inputs": [dict(item) for item in inputs],
        "frames": dict(frames),
        "outputs": [dict(item) for item in outputs],
        "config": config_dict,
        "config_sha256": canonical_sha256(config_dict),
        "allowed_licenses": effective_allowlist(allow_licenses),
        "restrictions": restrictions,
    }


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        (repeated if value in seen else seen).add(value)
    return sorted(repeated)


def job_problems(job_dir: str | Path, job: Mapping[str, Any]) -> list[str]:
    """Every problem with ``job`` for the job directory ``job_dir``: its schema, then (if the
    schema holds) the config hash, unique names and paths, the frame range, the pixel-hash
    frames, the restrictions, and every input file's presence, size and sha256."""
    directory = Path(job_dir)
    problems = [f"job: {problem}" for problem in contracts.problems(job, JOB_SCHEMA)]
    if problems:
        return problems
    try:
        if canonical_sha256(job["config"]) != job["config_sha256"]:
            problems.append("config_sha256 is not the sha256 of the canonical JSON of config")
        job_text(job)
    except (TypeError, ValueError) as exc:
        problems.append(f"the job is not plain JSON (no NaN, Infinity or other objects): {exc}")
    input_names = [item["name"] for item in job["inputs"]]
    output_names = [item["name"] for item in job["outputs"]]
    for kind, names in (("input", input_names), ("output", output_names)):
        for name in _duplicates(names):
            problems.append(f"{kind} name {name!r} is used more than once")
    relpaths = [item["relpath"] for item in (*job["inputs"], *job["outputs"])]
    for relpath in _duplicates(path.casefold() for path in relpaths):
        problems.append(f"relpath {relpath!r} is used more than once (case-insensitively)")
    frames = job["frames"]
    if frames["input"] not in input_names:
        problems.append(f"frames.input {frames['input']!r} is not one of the inputs")
    first = frames["input_first_frame"]
    last = first + frames["input_frame_count"] - 1
    if not first <= frames["start"] <= frames["end"] <= last:
        problems.append(
            f"frames {frames['start']}..{frames['end']} must be a non-empty range inside the "
            f"input's canonical frames {first}..{last}"
        )
    hashed = [item["frame_index"] for item in frames["pixel_sha256"]]
    outside = sorted({index for index in hashed if not first <= index <= last})
    if outside:
        problems.append(f"frames.pixel_sha256 names frames outside the input: {outside}")
    if len(hashed) != len(set(hashed)):
        problems.append("frames.pixel_sha256 names a frame more than once")
    carried = {tag for item in job["inputs"] for tag in item["restrictions"]}
    missing = sorted(carried - set(job["restrictions"]))
    if missing:
        problems.append(f"restrictions must include every input's restrictions; missing {missing}")
    for item in job["inputs"]:
        path = directory / item["relpath"]
        label = f"input {item['name']!r} ({item['relpath']})"
        if not path.is_file():
            problems.append(f"{label} is not a file in the job directory")
            continue
        if not _inside(path, directory):
            problems.append(f"{label} resolves outside the job directory")
            continue
        size = path.stat().st_size
        if size != item["bytes"]:
            problems.append(f"{label} has {size} bytes, the job says {item['bytes']}")
        elif sha256_file(path) != item["sha256"]:
            problems.append(f"{label} does not match its sha256")
    return problems


def write_job(job_dir: str | Path, job: Mapping[str, Any]) -> Path:
    """Validate ``job`` against job.schema.json and the files in ``job_dir``; write job.json.

    Returns the path of job.json. Raises FileExistsError if job.json, an output or the
    receipt already exists (a job is never rewritten), and JobError listing every problem.
    """
    directory = Path(job_dir)
    path = directory / JOB_FILE
    if path.exists():
        raise FileExistsError(f"{path} exists; a job is never rewritten")
    problems = job_problems(directory, job)
    if not problems:
        existing = [
            item["relpath"] for item in job["outputs"] if (directory / item["relpath"]).exists()
        ]
        if (directory / RECEIPT_FILE).exists():
            existing.append(RECEIPT_FILE)
        if existing:
            raise FileExistsError(f"{directory}: outputs already exist: {', '.join(existing)}")
    if problems:
        raise JobError(f"job in {directory.name}", problems)
    write_text(path, job_text(job))
    return path


def create_job(
    parent: str | Path,
    *,
    task: str,
    inputs: Sequence[JobInput],
    frames: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    allow_licenses: Iterable[str] = (),
) -> Path:
    """Create ``parent/<job_dir_name>``, copy the inputs in and write job.json.

    Returns the path of job.json. Raises FileExistsError when the job directory already
    exists (G17: outputs are never mixed with an earlier run), JobError listing every
    problem with the job, and LicenseExpressionError for a malformed ``allow_licenses``.
    On any failure the new directory is removed again.
    """
    entries = []
    problems = []
    for item in inputs:
        source = Path(item.source)
        if not source.is_file():
            problems.append(f"input {item.name!r}: source is not a file: {source}")
            continue
        entries.append(
            {
                "name": item.name,
                "relpath": item.relpath,
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "restrictions": sorted(set(item.restrictions)),
            }
        )
    if problems:
        raise JobError("create_job", problems)
    job = build_job(
        task=task,
        inputs=entries,
        frames=frames,
        outputs=outputs,
        config=config,
        allow_licenses=allow_licenses,
    )
    schema_problems = contracts.problems(job, JOB_SCHEMA)
    if schema_problems:
        raise JobError("create_job", [f"job: {problem}" for problem in schema_problems])
    directory = Path(parent) / job_dir_name(job)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        for item, entry in zip(inputs, entries, strict=True):
            target = directory / entry["relpath"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item.source, target)
        return write_job(directory, job)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


# --- running ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PluginRun:
    """A finished plugin run: exit code 0, wall time in seconds, output tails."""

    returncode: int
    seconds: float
    stdout_tail: str
    stderr_tail: str


def plugin_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a plugin gets: PLUGIN_ENV_KEEP and GBF_* variables from ``base``
    (default os.environ) except names that look like secrets, plus PLUGIN_ENV_FORCED."""
    source = os.environ if base is None else base
    env: dict[str, str] = {}
    for name, value in source.items():
        upper = name.upper()
        if _SECRET_ENV_NAME.search(upper):
            continue
        if upper in PLUGIN_ENV_KEEP or upper.startswith("GBF_"):
            env[name] = value
    env.update(PLUGIN_ENV_FORCED)
    return env


def _tail(data: bytes | str | None) -> str:
    if data is None:
        return ""
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    return text if len(text) <= TAIL_CHARS else "[... cut ...]\n" + text[-TAIL_CHARS:]


def run_plugin(
    command: Sequence[str | Path],
    job_path: str | Path,
    timeout_s: float,
    *,
    extra_env: Mapping[str, str] | None = None,
) -> PluginRun:
    """Run ``command + [job.json path]`` in the job directory and wait at most ``timeout_s``.

    ``command`` is a list (never a shell string), e.g. ``[sys.executable, "plugin.py"]``.
    The plugin gets plugin_environment() plus ``extra_env``. Raises PluginRunError (with
    the tail of the plugin's stderr) when the arguments are invalid, the program cannot be
    started, it runs longer than ``timeout_s`` seconds (it is then killed; programs it
    started itself may survive on Windows), or it exits with a non-zero code.
    """
    problems = []
    argv: list[str] = []
    if isinstance(command, str | bytes) or not isinstance(command, Sequence) or not command:
        problems.append("command must be a non-empty list of strings or paths, not a shell string")
    else:
        for position, part in enumerate(command):
            if isinstance(part, Path) or (isinstance(part, str) and part):
                argv.append(str(part))
            else:
                problems.append(
                    f"command[{position}] must be a non-empty string or a path, got {part!r}"
                )
    job = Path(job_path)
    if job.name != JOB_FILE:
        problems.append(f"job_path must point to a {JOB_FILE}, got {job.name!r}")
    elif not job.is_file():
        problems.append(f"job file does not exist: {job}")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float):
        problems.append(f"timeout_s must be a number of seconds, got {timeout_s!r}")
    elif not (math.isfinite(timeout_s) and timeout_s > 0):
        problems.append(f"timeout_s must be finite and > 0, got {timeout_s!r}")
    where = f"plugin {' '.join(Path(part).name for part in argv) or '?'} on {job.parent.name}"
    if problems:
        raise PluginRunError(where, problems)
    env = plugin_environment()
    env.update(extra_env or {})
    argv.append(str(job.resolve()))
    start = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=job.parent,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PluginRunError(
            where, [f"timed out after {timeout_s:g} s and was stopped"], _tail(exc.stderr)
        ) from exc
    except OSError as exc:
        raise PluginRunError(where, [f"cannot start {argv[0]!r}: {exc}"]) from exc
    seconds = time.monotonic() - start
    stderr = _tail(completed.stderr)
    if completed.returncode != 0:
        raise PluginRunError(where, [f"exited with code {completed.returncode}"], stderr)
    return PluginRun(
        returncode=0, seconds=seconds, stdout_tail=_tail(completed.stdout), stderr_tail=stderr
    )


# --- accepting -------------------------------------------------------------------------
@dataclass(frozen=True)
class AcceptedOutputs:
    """Outputs that passed accept_outputs. ``outputs`` maps output names to their files."""

    job_dir: Path
    job: dict[str, Any]
    job_sha256: str
    receipt: dict[str, Any]
    outputs: dict[str, Path]

    @property
    def plugin(self) -> dict[str, Any]:
        """The receipt's plugin block: name, version, license, commit, weights."""
        return dict(self.receipt["plugin"])

    @property
    def restrictions(self) -> list[str]:
        """Tags every artifact derived from these outputs carries (ADR 0002): the union of
        the job's (its inputs') restrictions and those the plugin added."""
        return sorted(set(self.job["restrictions"]) | set(self.receipt["restrictions"]))

    def output_refs(self, base: str | Path) -> list[dict[str, Any]]:
        """Receipt file_refs (name, relpath relative to ``base``, sha256, bytes) of the
        outputs, in the job's order."""
        refs = []
        listed = {item["name"]: item for item in self.receipt["outputs"]}
        for spec in self.job["outputs"]:
            path = self.outputs[spec["name"]]
            refs.append(
                {
                    "name": spec["name"],
                    "relpath": path.resolve().relative_to(Path(base).resolve()).as_posix(),
                    "sha256": listed[spec["name"]]["sha256"],
                    "bytes": listed[spec["name"]]["bytes"],
                }
            )
        return refs


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON number {name}")


def _read_json(path: Path, label: str, problems: list[str]) -> tuple[Any, bytes]:
    """(parsed document, raw bytes), or (None, b"") after recording why it is unusable."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        problems.append(f"{label}: missing")
        return None, b""
    except OSError as exc:
        problems.append(f"{label}: unreadable ({type(exc).__name__})")
        return None, b""
    try:
        document = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        problems.append(f"{label}: not valid UTF-8 JSON ({exc})")
        return None, raw
    return document, raw


def _licence_problems(licence: str, allowed: list[str], job_allowed: list[str]) -> list[str]:
    try:
        parse_license_expression(licence)
    except LicenseExpressionError as exc:
        return [f"plugin licence: {exc}"]
    found = license_problems(licence, allowed)
    if found:
        return [
            f"plugin licence: {found[0]}; a licence outside the permissive allowlist is "
            "accepted only when named explicitly in allow_licenses (--allow-license)"
        ]
    found = license_problems(licence, job_allowed)
    if found:
        return [f"plugin licence: the job's allowed_licenses do not cover it: {found[0]}"]
    return []


def _content_problems(fmt: str, path: Path, job: Mapping[str, Any]) -> list[str]:
    if fmt != CALIB_FORMAT:
        return [f"no checker for format {fmt!r}"]
    try:
        frames = read_calib(path)
    except CalibFormatError as exc:
        return list(exc.problems)
    expected = list(range(job["frames"]["start"], job["frames"]["end"] + 1))
    got = [frame.frame_index for frame in frames]
    problems = []
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    if missing:
        problems.append(f"no row for canonical frame(s) {missing}")
    if extra:
        problems.append(
            f"rows for frame(s) {extra} outside the job's frames {expected[0]}..{expected[-1]}"
        )
    return problems


def _output_problems(
    directory: Path, job: Mapping[str, Any], receipt: Mapping[str, Any]
) -> tuple[list[str], dict[str, Path]]:
    problems: list[str] = []
    verified: dict[str, Path] = {}
    listed: dict[str, Mapping[str, Any]] = {}
    for item in receipt["outputs"]:
        if item["name"] in listed:
            problems.append(f"output {item['name']!r} is listed more than once in the receipt")
        listed.setdefault(item["name"], item)
    requested = {item["name"]: item for item in job["outputs"]}
    for name, spec in requested.items():
        label = f"output {name!r} ({spec['relpath']})"
        item = listed.get(name)
        if item is None:
            problems.append(f"{label}: requested by the job but not listed in the receipt")
            continue
        if item["relpath"] != spec["relpath"]:
            problems.append(f"{label}: the receipt puts it at {item['relpath']!r}")
            continue
        if item["format"] != spec["format"]:
            problems.append(
                f"{label}: format {item['format']!r}, the job asked for {spec['format']!r}"
            )
        path = directory / spec["relpath"]
        if not path.is_file():
            problems.append(f"{label}: missing from the job directory")
            continue
        if not _inside(path, directory):
            problems.append(f"{label}: resolves outside the job directory")
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != item["bytes"]:
            problems.append(f"{label}: {size} bytes on disk, the receipt says {item['bytes']}")
        if digest != item["sha256"]:
            problems.append(
                f"{label}: sha256 mismatch: the file hashes to {digest}, "
                f"the receipt says {item['sha256']}"
            )
            continue
        content = _content_problems(spec["format"], path, job)
        problems.extend(f"{label}: {problem}" for problem in content)
        if not content and size == item["bytes"]:
            verified[name] = path
    for name in sorted(set(listed) - set(requested)):
        problems.append(f"output {name!r}: listed in the receipt but not requested by the job")
    out_dir = directory / OUTPUT_DIR
    known = {item["relpath"] for item in job["outputs"]}
    if out_dir.is_dir():
        for path in sorted(out_dir.rglob("*")):
            relpath = path.relative_to(directory).as_posix()
            if not path.is_dir() and relpath not in known:
                problems.append(f"{relpath}: not an output the job requested")
    return problems, verified


def _frame_problems(job: Mapping[str, Any], receipt: Mapping[str, Any]) -> list[str]:
    frames = job["frames"]
    first = frames["input_first_frame"]
    last = first + frames["input_frame_count"] - 1
    reported: dict[int, str] = {}
    problems = []
    for item in receipt["decoded_frames"]:
        index, digest = item["frame_index"], item["sha256"]
        if reported.get(index, digest) != digest:
            problems.append(f"decoded_frames: frame {index} is reported with two different hashes")
        reported[index] = digest
    outside = sorted(index for index in reported if not first <= index <= last)
    if outside:
        problems.append(f"decoded_frames: frames {outside} are outside the input ({first}..{last})")
    for item in frames["pixel_sha256"]:
        index, expected = item["frame_index"], item["sha256"]
        got = reported.get(index)
        if got is None:
            problems.append(
                f"frame {index}: the plugin reported no pixel hash, so nothing shows that it "
                f"decoded canonical frame {index}"
            )
        elif got != expected:
            problems.append(
                f"frame {index}: the plugin decoded different pixels (sha256 {got}, expected "
                f"{expected}); it does not see canonical frame {index}"
            )
    return problems


def _input_problems(directory: Path, job: Mapping[str, Any]) -> list[str]:
    problems = []
    for item in job["inputs"]:
        path = directory / item["relpath"]
        label = f"input {item['name']!r} ({item['relpath']})"
        if not path.is_file():
            problems.append(f"{label}: disappeared while the plugin ran")
        elif path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            problems.append(f"{label}: changed while the plugin ran")
    return problems


def accept_outputs(job_dir: str | Path, allow_licenses: Iterable[str] = ()) -> AcceptedOutputs:
    """Verify a finished job and return its outputs; see the module docstring for the checks.

    ``allow_licenses`` names non-permissive SPDX identifiers the user accepts explicitly;
    the plugin's licence must also be covered by the job's allowed_licenses. Raises
    OutputsRejectedError listing every problem found, and LicenseExpressionError for a
    malformed ``allow_licenses``.
    """
    directory = Path(job_dir)
    allowed = effective_allowlist(allow_licenses)
    problems: list[str] = []
    job, job_bytes = _read_json(directory / JOB_FILE, JOB_FILE, problems)
    job_ok = False
    if job is not None:
        found = contracts.problems(job, JOB_SCHEMA)
        problems.extend(f"{JOB_FILE}: {problem}" for problem in found)
        job_ok = not found
    receipt, _ = _read_json(directory / RECEIPT_FILE, RECEIPT_FILE, problems)
    receipt_ok = False
    if receipt is not None:
        found = contracts.problems(receipt, RECEIPT_SCHEMA)
        problems.extend(f"{RECEIPT_FILE}: {problem}" for problem in found)
        receipt_ok = not found
    verified: dict[str, Path] = {}
    if job_ok and receipt_ok:
        digest = sha256_bytes(job_bytes)
        if receipt["job_sha256"] != digest:
            problems.append(
                f"{RECEIPT_FILE}: written for another job (job_sha256 {receipt['job_sha256']}, "
                f"this job.json hashes to {digest})"
            )
        if receipt["status"] != "ok":
            notes = "; ".join(receipt.get("messages", [])) or "no messages"
            problems.append(
                f"{RECEIPT_FILE}: the plugin reported status {receipt['status']!r} ({notes})"
            )
        problems.extend(
            _licence_problems(receipt["plugin"]["license"], allowed, job["allowed_licenses"])
        )
        output_problems, verified = _output_problems(directory, job, receipt)
        problems.extend(output_problems)
        problems.extend(_frame_problems(job, receipt))
        dropped = sorted(set(job["restrictions"]) - set(receipt["restrictions"]))
        if dropped:
            problems.append(f"{RECEIPT_FILE}: drops restriction(s) {dropped} carried by the inputs")
        problems.extend(_input_problems(directory, job))
    if problems:
        raise OutputsRejectedError(f"job {directory.name}", problems)
    return AcceptedOutputs(
        job_dir=directory,
        job=job,
        job_sha256=sha256_bytes(job_bytes),
        receipt=receipt,
        outputs=verified,
    )
