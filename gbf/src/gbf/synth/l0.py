"""``python -m gbf l0 --out DIR [--resolution 720p|1080p|both]``: the L0 synthetic smoke run.

Plan P0 deliverable and gate G5. For each resolution, in ``DIR/<resolution>/``:

1. render 5 frames (gbf.synth.render) with a hard cut at frame 3 and write them to an H.264
   CFR 25/1 mp4 (``l0.mp4``), and the true cameras to ``cameras.json``;
2. ``video``: ffprobe sees 5 frames, CFR 25/1, the rendered size;
3. ``barcodes``: every frame read through gbf.ingest's single frame reader carries its own
   number, [0, 1, 2, 3, 4];
4. ``cuts``: gbf.ingest.detect_cuts finds [3];
5. ``window``: frames 2..4 (a non-zero offset) cut losslessly with gbf.ingest.extract_window:
   window frame k carries barcode 2 + k and is bit-identical to frame 2 + k of the mp4;
   ``video.json`` is written with the cut and the window;
6. ``plugin_round_trip``: the calibration stand-in (a recorded-output replay that reports K
   at 1920x1080 whatever the frame size) runs on the window through the plugin protocol;
   once its outputs are accepted, K, R and C read back and rescaled to the frame size equal
   the truth (rtol 1e-12, atol 0) and every frame is fresh;
7. ``landmarks``: every pitch landmark the round-trip cameras see lands on a line pixel of
   the decoded frame (all channels >= LINE_MIN_CHANNEL). As a control, the same check with
   the plugin's K used unscaled is recorded too: it must miss when the sizes differ.

The run writes ``DIR/l0-receipt.json`` (urn:gbf:schema:receipt, kind "l0"; relpaths are
relative to DIR, no absolute path appears anywhere), prints a JSON summary and returns 0
when every check passed, 1 otherwise, 2 for usage errors.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

import gbf
from gbf import contracts
from gbf.geometry import Camera, project
from gbf.ingest import (
    DEFAULT_CUT_THRESHOLD,
    cut_detection_settings,
    cut_scores,
    extract_window,
    iter_frames,
    pixel_sha256,
    tool_versions,
    window_id,
    write_video_json,
)
from gbf.plugins import (
    JobInput,
    accept_outputs,
    create_job,
    read_calib,
    run_plugin,
)
from gbf.synth.barcode import BarcodeError, decode_barcode
from gbf.synth.pitch import landmarks
from gbf.synth.render import FPS, RenderedSequence, render_sequence, visible_landmarks
from gbf.synth.video import write_frames_mp4
from gbf.util.hashing import sha256_file
from gbf.util.io import write_json
from gbf.util.paths import RepoRootNotFoundError, repo_root

RESOLUTIONS: dict[str, tuple[int, int]] = {"720p": (1280, 720), "1080p": (1920, 1080)}
N_FRAMES = 5
CUT_AT = 3
WINDOW = (2, 4)
"""Canonical frames of the window, inclusive: a non-zero offset on purpose (plan P0)."""
RTOL = 1e-12
"""Relative tolerance (atol 0) for K, R, C after the plugin round trip."""
LINE_MIN_CHANNEL = 170
"""A decoded pixel is a line pixel when all three channels reach this. Measured at x264
CRF 23: landmark pixels decode to 197 or more; every other scene colour has a channel of
112 or less."""
MIN_LANDMARKS_PER_FRAME = 4
"""Fewer visible landmarks than this make the landmark check fail (it would prove little)."""
PLUGIN_TIMEOUT_S = 120.0
"""Seconds the calibration plugin may run before it is stopped."""
CHECK_NAMES = ("video", "barcodes", "cuts", "window", "plugin_round_trip", "landmarks")
"""Every check of one resolution, in the order they run; a run passes only if all ran."""
RECEIPT_NAME = "l0-receipt.json"
STUB_RELPATH = "tests/plugin_stubs/stub_calib.py"
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_ABSOLUTE = re.compile(
    r"(?:file:)?(?:[A-Za-z]:[\\/]|\\\\)[^\s'\"]*|(?<![\w.:/])/[^\s'\"/]+(?:/[^\s'\"]*)+"
)


class L0Error(RuntimeError):
    """The L0 run cannot start (bad output directory, missing plugin)."""


# --- receipt helpers --------------------------------------------------------------------
def _git(root: Path, *args: str) -> str | None:
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"  # never take optional locks: others may write the index
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, env=env, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.decode("utf-8", "replace") if done.returncode == 0 else None


def code_version() -> dict[str, Any]:
    """gbf version, HEAD commit ("unknown" outside a git checkout) and whether the work
    tree is dirty (also true when that cannot be determined)."""
    commit, dirty = "unknown", True
    try:
        root: Path | None = repo_root()
    except RepoRootNotFoundError:
        root = None
    if root is not None:
        head = _git(root, "rev-parse", "HEAD")
        if head is not None and _SHA1.match(head.strip()):
            commit = head.strip()
            status = _git(root, "status", "--porcelain")
            dirty = status is None or bool(status.strip())
    return {"gbf_version": gbf.__version__, "git_commit": commit, "git_dirty": dirty}


def utc_timestamp() -> str:
    """RFC 3339 UTC time to the second; SOURCE_DATE_EPOCH (integer seconds) wins if set."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None and epoch.isascii() and epoch.isdigit():
        moment = datetime.fromtimestamp(int(epoch), tz=UTC)
    else:
        moment = datetime.now(tz=UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def scrub(text: str, roots: Sequence[Path] = ()) -> str:
    """``text`` with ``roots`` replaced by ``<root>`` and any other absolute path by
    ``<path>``, so no message in a receipt names a local directory."""
    for root in roots:
        for form in {str(root), str(root.resolve()), root.resolve().as_posix()}:
            text = text.replace(form, "<root>")
    return _ABSOLUTE.sub("<path>", text)


def _scrub_all(value: Any, roots: Sequence[Path]) -> Any:
    if isinstance(value, str):
        return scrub(value, roots)
    if isinstance(value, dict):
        return {key: _scrub_all(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_all(item, roots) for item in value]
    return value


def default_plugin_command() -> list[str]:
    """[this Python, the stand-in plugin script of the source checkout]."""
    try:
        stub = repo_root() / STUB_RELPATH
    except RepoRootNotFoundError as exc:
        raise L0Error(
            f"the stand-in plugin ships with the source checkout ({STUB_RELPATH})"
        ) from exc
    if not stub.is_file():
        raise L0Error(f"stand-in plugin not found: {STUB_RELPATH}")
    return [sys.executable, str(stub)]


def is_line_pixel(pixel: np.ndarray) -> bool:
    """True for a decoded RGB pixel whose three channels all reach LINE_MIN_CHANNEL."""
    return bool(np.min(pixel) >= LINE_MIN_CHANNEL)


def landmark_hits(camera: Camera, frame: np.ndarray) -> dict[str, bool]:
    """For every landmark ``camera`` sees: does the pixel holding its projection show a
    line? ``camera`` must be for the frame's size."""
    hits = {}
    for name, (u, v) in visible_landmarks(camera).items():
        column, row = int(np.floor(u + 0.5)), int(np.floor(v + 0.5))
        hits[name] = is_line_pixel(frame[row, column])
    return hits


def control_hits(camera: Camera, frame: np.ndarray, names: Sequence[str]) -> int:
    """How many landmarks in ``names`` land on a line pixel of ``frame`` when projected
    with ``camera`` as is, even if it was calibrated for another frame size."""
    if not names:
        return 0
    named = landmarks()
    uv, _, valid = project(camera, np.array([named[name] for name in names]))
    height, width = frame.shape[:2]
    count = 0
    for (u, v), ok in zip(uv, valid, strict=True):
        if not ok:
            continue
        column, row = int(np.floor(u + 0.5)), int(np.floor(v + 0.5))
        if 0 <= column < width and 0 <= row < height and is_line_pixel(frame[row, column]):
            count += 1
    return count


# --- checks -----------------------------------------------------------------------------
def _check(name: str, ok: bool, detail: dict[str, Any], problems: Sequence[str] = ()) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail, "problems": list(problems)}


def _failure(name: str, exc: BaseException) -> dict:
    listed = getattr(exc, "problems", None)
    problems = [str(p) for p in listed] if isinstance(listed, list) and listed else [str(exc)]
    return _check(name, False, {"error": type(exc).__name__}, problems)


def _file_ref(name: str, path: Path, base: Path) -> dict[str, Any]:
    return {
        "name": name,
        "relpath": path.resolve().relative_to(base.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _run_resolution(
    label: str,
    out_dir: Path,
    seed: int,
    plugin_command: Sequence[str],
    timeout_s: float,
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    width, height = RESOLUTIONS[label]
    work = out_dir / label
    work.mkdir(parents=True, exist_ok=False)
    checks: list[dict[str, Any]] = []
    result: dict[str, Any] = {"resolution": label, "width": width, "height": height}

    def attempt(name: str, step: Callable[[], dict]) -> dict | None:
        try:
            check = step()
        except Exception as exc:  # any failure is recorded in the receipt, never hidden
            checks.append(_failure(name, exc))
            return None
        checks.append(check)
        return check

    sequence: RenderedSequence = render_sequence(width, height, N_FRAMES, cut_at=CUT_AT, seed=seed)
    video_path = work / "l0.mp4"
    cameras_path = work / "cameras.json"
    facts = write_frames_mp4(sequence.frames, video_path, fps=FPS)  # probed after writing
    write_json(cameras_path, sequence.truth_record())
    outputs += [
        _file_ref(f"{label}/video", video_path, out_dir),
        _file_ref(f"{label}/cameras", cameras_path, out_dir),
    ]
    result["scene"] = {
        "seed": seed,
        "people": len(sequence.scene.persons),
        "ball": sequence.scene.ball is not None,
        "dropped": sequence.scene.dropped,
        "shots": [item.shot_id for item in sequence.truth],
    }

    def video_check() -> dict:
        problems = []
        if facts.nb_frames_decoded != N_FRAMES:
            problems.append(f"decodes to {facts.nb_frames_decoded} frames, expected {N_FRAMES}")
        if not facts.is_cfr_at(FPS):
            problems.append(f"not CFR at {FPS}")
        if (facts.frame_width, facts.frame_height) != (width, height):
            problems.append(f"frames are {facts.frame_width}x{facts.frame_height}")
        detail = {
            "codec": facts.codec_name,
            "pix_fmt": facts.pix_fmt,
            "nb_frames_decoded": facts.nb_frames_decoded,
            "r_frame_rate": facts.r_frame_rate,
            "avg_frame_rate": facts.avg_frame_rate,
            "timestamps_cfr": facts.timestamps_cfr,
            "frame_size": [facts.frame_width, facts.frame_height],
            "sha256": facts.sha256,
        }
        return _check("video", not problems, detail, problems)

    attempt("video", video_check)
    decoded = [frame for _, frame in iter_frames(video_path)]

    def barcode_check() -> dict:
        numbers: list[int | None] = []
        problems = []
        for index, frame in enumerate(decoded):
            try:
                numbers.append(decode_barcode(frame))
            except BarcodeError as exc:
                numbers.append(None)
                problems.extend(f"frame {index}: {problem}" for problem in exc.problems)
        expected = list(range(N_FRAMES))
        if numbers != expected:
            problems.append(f"barcodes read {numbers}, expected {expected}")
        return _check(
            "barcodes", not problems, {"expected": expected, "decoded": numbers}, problems
        )

    attempt("barcodes", barcode_check)

    def cut_check() -> dict:
        scores = cut_scores(enumerate(decoded))
        detected = [index for index, score in scores if score > DEFAULT_CUT_THRESHOLD]
        detail = {
            "expected": sequence.cuts,
            "detected": detected,
            "threshold": DEFAULT_CUT_THRESHOLD,
            "scores": [[index, round(score, 6)] for index, score in scores],
        }
        ok = detected == sequence.cuts
        return _check("cuts", ok, detail, [] if ok else [f"detected {detected}"])

    attempt("cuts", cut_check)

    start, end = WINDOW
    window_path = work / "windows" / f"{window_id(start, end)}.mkv"
    window_record = extract_window(
        video_path, start, end, window_path, fps=FPS, source_sha256=facts.sha256
    )
    outputs.append(_file_ref(f"{label}/window", window_path, out_dir))

    def window_check() -> dict:
        frames = [frame for _, frame in iter_frames(window_path)]
        problems = []
        numbers = []
        for k, frame in enumerate(frames):
            try:
                numbers.append(decode_barcode(frame))
            except BarcodeError as exc:
                numbers.append(None)
                problems.extend(f"window frame {k}: {problem}" for problem in exc.problems)
        expected = list(range(start, end + 1))
        if numbers != expected:
            problems.append(f"window barcodes read {numbers}, expected {expected}")
        identical = [
            k < len(frames) and np.array_equal(frames[k], decoded[start + k])
            for k in range(end - start + 1)
        ]
        if len(frames) != end - start + 1:
            problems.append(f"window holds {len(frames)} frames, expected {end - start + 1}")
        if not all(identical):
            problems.append(f"bit-identical per window frame: {identical}")
        detail = {
            "start_frame": start,
            "end_frame": end,
            "window_id": window_record.window_id,
            "barcodes": numbers,
            "bit_identical": identical,
            "pixel_sha256": [pixel_sha256(frame) for frame in frames],
        }
        return _check("window", not problems, detail, problems)

    attempt("window", window_check)

    # No relpaths in video.json: they are defined relative to the repository root or
    # DATA_ROOT, and an L0 run lives in an arbitrary --out directory. The receipt lists the
    # files with paths relative to itself.
    record_path = work / "video.json"
    write_video_json(
        record_path,
        facts,
        fps=FPS,
        cuts=sequence.cuts,
        cut_detection=cut_detection_settings(),
        windows=[window_record],
    )
    outputs.append(_file_ref(f"{label}/video_record", record_path, out_dir))

    def plugin_check() -> dict:
        frames_spec = {
            "canonical_id": facts.sha256,
            "fps": FPS,
            "input": "video",
            "input_first_frame": start,
            "input_frame_count": end - start + 1,
            "start": start,
            "end": end,
            "width": width,
            "height": height,
            "pixel_sha256": [
                {"frame_index": start + k, "sha256": pixel_sha256(decoded[start + k])}
                for k in range(end - start + 1)
            ],
        }
        job_path = create_job(
            work / "plugins",
            task="calibration",
            inputs=[
                JobInput("video", window_path, "in/video.mkv"),
                JobInput("recorded_cameras", cameras_path, "in/cameras.json"),
            ],
            frames=frames_spec,
            outputs=[{"name": "calib", "relpath": "out/calib.parquet", "format": "calib.v1"}],
            config={"recorded_cameras": "recorded_cameras"},
        )
        run_plugin(plugin_command, job_path, timeout_s)
        accepted = accept_outputs(job_path.parent)
        outputs.extend(
            {**ref, "name": f"{label}/{ref['name']}"} for ref in accepted.output_refs(out_dir)
        )
        outputs.append(_file_ref(f"{label}/job", job_path, out_dir))
        outputs.append(
            _file_ref(f"{label}/plugin_receipt", job_path.parent / "plugin-receipt.json", out_dir)
        )
        calib = read_calib(accepted.outputs["calib"])
        truth = {item.frame_index: item.camera for item in sequence.truth}
        problems = []
        per_frame = []
        round_trip: dict[int, Camera] = {}
        for row in calib:
            camera = row.camera(width, height)
            expected = truth[row.frame_index]
            if camera is None:
                problems.append(f"frame {row.frame_index}: no calibration")
                continue
            round_trip[row.frame_index] = camera
            close = {
                name: bool(
                    np.allclose(getattr(camera, name), getattr(expected, name), rtol=RTOL, atol=0.0)
                )
                for name in ("K", "R", "C")
            }
            max_abs = {
                name: float(np.max(np.abs(getattr(camera, name) - getattr(expected, name))))
                for name in ("K", "R", "C")
            }
            if not all(close.values()) or row.calib_status != "fresh":
                problems.append(
                    f"frame {row.frame_index}: close={close}, calib_status={row.calib_status}"
                )
            per_frame.append(
                {
                    "frame_index": row.frame_index,
                    "calib_status": row.calib_status,
                    "reported_size": [row.width, row.height],
                    "close": close,
                    "max_abs_diff": max_abs,
                }
            )
        if sorted(round_trip) != list(range(start, end + 1)):
            problems.append(f"calibrated frames {sorted(round_trip)}, expected {start}..{end}")
        result["round_trip_cameras"] = round_trip
        result["reported"] = {row.frame_index: row for row in calib}
        detail = {
            "plugin": {key: accepted.plugin[key] for key in ("name", "version", "license")},
            "job_sha256": accepted.job_sha256,
            "frames": per_frame,
            "rtol": RTOL,
            "atol": 0.0,
            "outputs_verified": sorted(accepted.outputs),
        }
        return _check("plugin_round_trip", not problems, detail, problems)

    attempt("plugin_round_trip", plugin_check)

    def landmark_check() -> dict:
        cameras: dict[int, Camera] = result.get("round_trip_cameras", {})
        if not cameras:
            return _check("landmarks", False, {}, ["no round-trip cameras to project with"])
        problems = []
        per_frame = []
        for frame_index, camera in sorted(cameras.items()):
            hits = landmark_hits(camera, decoded[frame_index])
            misses = sorted(name for name, hit in hits.items() if not hit)
            reported = result["reported"][frame_index].camera()
            control = control_hits(reported, decoded[frame_index], sorted(hits))
            if misses:
                problems.append(f"frame {frame_index}: landmarks off the lines: {misses}")
            if len(hits) < MIN_LANDMARKS_PER_FRAME:
                problems.append(
                    f"frame {frame_index}: only {len(hits)} landmark(s) visible, "
                    f"need {MIN_LANDMARKS_PER_FRAME}"
                )
            per_frame.append(
                {
                    "frame_index": frame_index,
                    "checked": len(hits),
                    "on_line": len(hits) - len(misses),
                    "names": sorted(hits),
                    "control_unscaled_on_line": control,
                }
            )
        detail = {"line_min_channel": LINE_MIN_CHANNEL, "frames": per_frame}
        return _check("landmarks", not problems, detail, problems)

    attempt("landmarks", landmark_check)
    result.pop("round_trip_cameras", None)
    result.pop("reported", None)
    result["checks"] = checks
    names = [check["name"] for check in checks]
    result["passed"] = names == list(CHECK_NAMES) and all(check["ok"] for check in checks)
    return result


SCOPE = (
    "L0 synthetic smoke run (plan P0, gate G5): on the machine and tool versions in "
    "'environment', 5 rendered frames with a hard cut at frame 3 survive H.264 CFR 25/1 "
    "encoding, the single frame reader returns them in order (barcodes 0..4), the cut is "
    "detected at 3, a lossless window at a non-zero offset (2..4) is bit-identical to the "
    "canonical frames, and camera parameters survive a plugin-protocol round trip "
    "(rescaled to the frame size) unchanged and reproject pitch landmarks onto drawn lines."
)
LIMITATIONS = [
    "Synthetic frames only: flat colours and billboard players; nothing here measures "
    "perception, calibration or tracking accuracy on real broadcast video.",
    "The calibration plugin is a recorded-output replay of the true cameras; the round trip "
    "certifies the protocol and the rescaling, not any calibration method.",
    "Pixel hashes depend on the ffmpeg and x264 builds; another environment must re-run L0 "
    "instead of comparing hashes (cross-environment pixel tolerance is in prereg/gates.v1.json).",
    "Cut detection is checked on one synthetic hard cut with the unvalidated default threshold.",
]


def run_l0(
    out_dir: str | Path,
    resolutions: Sequence[str] = ("720p",),
    *,
    seed: int = 0,
    plugin_command: Sequence[str] | None = None,
    timeout_s: float = PLUGIN_TIMEOUT_S,
) -> dict[str, Any]:
    """Run L0 for each resolution into ``out_dir`` and write ``out_dir/l0-receipt.json``.

    Returns the receipt. ``plugin_command`` defaults to the stand-in plugin of the source
    checkout run by this Python. Raises L0Error when ``out_dir`` already holds an L0 run or
    a resolution is unknown; failed checks do not raise, they make the receipt "failed".
    """
    root = Path(out_dir)
    unknown = [label for label in resolutions if label not in RESOLUTIONS]
    if isinstance(resolutions, str) or unknown or not resolutions:
        raise L0Error(f"resolutions must be among {sorted(RESOLUTIONS)}, got {resolutions!r}")
    if len(set(resolutions)) != len(resolutions):
        raise L0Error(f"each resolution may be run once, got {list(resolutions)}")
    clashes = [name for name in (RECEIPT_NAME, *resolutions) if (root / name).exists()]
    if clashes:
        raise L0Error(f"{root} already holds {', '.join(clashes)}; use a new --out directory")
    command = list(plugin_command) if plugin_command is not None else default_plugin_command()
    root.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    runs = []
    for label in resolutions:
        try:
            runs.append(_run_resolution(label, root, seed, command, timeout_s, outputs))
        except Exception as exc:  # a stage that everything else depends on failed
            runs.append({"resolution": label, "checks": [_failure("setup", exc)], "passed": False})
    passed = all(run["passed"] for run in runs)
    environment = dict(tool_versions())
    environment["pyarrow"] = pa.__version__
    environment["platform"] = platform.system()
    plugin_inputs = []
    script = Path(command[-1])
    if script.is_file():  # the plugin's script (the stand-in by default), by content
        plugin_inputs.append({"name": f"plugin/{script.name}", "sha256": sha256_file(script)})
    receipt = {
        "schema_version": 1,
        "kind": "l0",
        "status": "passed" if passed else "failed",
        "created_at": utc_timestamp(),
        "producer": "gbf.synth.l0",
        "code": code_version(),
        "inputs": plugin_inputs,
        "outputs": outputs,
        "seeds": [seed],
        "scope": SCOPE,
        "limitations": LIMITATIONS,
        "restrictions": [],
        "reviewer_type": "automated",
        "environment": environment,
        "result": {
            "relpaths_relative_to": "the directory holding this receipt",
            "n_frames": N_FRAMES,
            "cut_at": CUT_AT,
            "fps": FPS,
            "window": list(WINDOW),
            "runs": runs,
        },
    }
    scrub_roots = [root]
    try:
        scrub_roots.append(repo_root())
    except RepoRootNotFoundError:
        pass
    receipt = _scrub_all(receipt, scrub_roots)
    contracts.validate(receipt, "receipt")
    write_json(root / RECEIPT_NAME, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gbf l0",
        description="Run the L0 synthetic smoke pipeline and write l0-receipt.json.",
    )
    parser.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    parser.add_argument(
        "--resolution",
        choices=["720p", "1080p", "both"],
        default="720p",
        help="frame size(s) to run (default 720p)",
    )
    parser.add_argument("--seed", type=int, default=0, help="scene seed (default 0)")
    parser.add_argument(
        "--plugin",
        type=Path,
        default=None,
        help="calibration plugin script run with this Python (default: the stand-in)",
    )
    return parser


def main(argv: list[str]) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits on --help (0) and on usage errors (2)
        return exc.code if isinstance(exc.code, int) else 2
    labels = list(RESOLUTIONS) if args.resolution == "both" else [args.resolution]
    command = None if args.plugin is None else [sys.executable, str(args.plugin)]
    try:
        receipt = run_l0(args.out, labels, seed=args.seed, plugin_command=command)
    except (L0Error, OSError, ValueError) as exc:
        print(f"error: {scrub(str(exc), [Path(args.out)])}", file=sys.stderr)
        return 1
    summary = {
        "status": receipt["status"],
        "receipt": RECEIPT_NAME,
        "checks": {
            run["resolution"]: {check["name"]: check["ok"] for check in run["checks"]}
            for run in receipt["result"]["runs"]
        },
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if receipt["status"] == "passed" else 1
