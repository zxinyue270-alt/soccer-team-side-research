"""Stand-in for an external calibration plugin: recorded-output replay (plan P0, L0 smoke).

Usage: python stub_calib.py PATH/TO/job.json

It behaves like a third-party plugin: it imports only the standard library and pyarrow,
never gbf, and talks to the core only through files (urn:gbf:schema:job in,
urn:gbf:schema:plugin-receipt and calib.parquet out). Steps:

1. Read job.json, check every input's size and sha256, and refuse to run unless the job's
   allowed_licenses include this plugin's licence (MIT).
2. Decode the job's input video with ffmpeg (raw rgb24) and hash every frame the way the
   core does (header "gbf.rgb24 h=<H> w=<W> c=3" and a newline, then the raw bytes), so the
   core can check which canonical frames the plugin saw.
3. Replay the recorded cameras (the input named by config["recorded_cameras"], a
   gbf.synth.cameras.v1 document) for canonical frames start..end, reporting K at
   1920x1080 whatever the input size -- as sn-gamestate does -- so the core must rescale.
4. Write calib.parquet (calib.v1, every frame fresh, rep_err null) and plugin-receipt.json.

Exit codes: 0 done, 2 usage, 3 licence not allowed by the job, 4 bad job or inputs,
5 ffmpeg failed.

The Arrow columns are built from raw buffers (stdlib ``array``) rather than with
``pa.array``/``Table.from_pylist``: in pyarrow 26 those probe for pandas and import it,
which would add over a second to every run of this stub.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from array import array
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

NAME = "stub-calib"
VERSION = "0.1.0"
LICENSE = "MIT"
REPORT_SIZE = (1920, 1080)
PIXEL_HASH_HEADER = "gbf.rgb24 h={height} w={width} c=3\n"
# The calib.v1 Arrow schema (gbf.plugins.calib.CALIB_SCHEMA); a test keeps the two equal.
CALIB_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
        pa.field("K", pa.list_(pa.float64(), 9), nullable=True),
        pa.field("R", pa.list_(pa.float64(), 9), nullable=True),
        pa.field("C", pa.list_(pa.float64(), 3), nullable=True),
        pa.field("calib_status", pa.string(), nullable=False),
        pa.field("calib_age", pa.int32(), nullable=False),
        pa.field("rep_err", pa.float64(), nullable=True),
    ]
)


class StubError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rescale_intrinsics(
    K: list[list[float]], src: tuple[int, int], dst: tuple[int, int]
) -> list[list[float]]:
    """K for frames resized from src to dst (width, height): u' = (u + 0.5) * s - 0.5."""
    sx = dst[0] / src[0]
    sy = dst[1] / src[1]
    A = [[sx, 0.0, 0.5 * (sx - 1.0)], [0.0, sy, 0.5 * (sy - 1.0)], [0.0, 0.0, 1.0]]
    return [[sum(A[i][k] * K[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _buffer(code: str, values: list) -> pa.Buffer:
    """Native-order bytes of ``values`` as array typecode ``code`` (i: int32, q: int64,
    d: float64), which is the layout pyarrow expects in from_buffers."""
    data = array(code, values)
    expected = {"i": 4, "q": 8, "d": 8}[code]
    if data.itemsize != expected:
        raise StubError(4, f"array typecode {code!r} has {data.itemsize} bytes, not {expected}")
    return pa.py_buffer(data.tobytes())


def calib_table(rows: list[dict]) -> pa.Table:
    """The calib.v1 table of ``rows`` (every frame fresh, rep_err null), built from buffers."""
    n = len(rows)
    columns = [
        pa.Array.from_buffers(
            pa.int64(), n, [None, _buffer("q", [r["frame_index"] for r in rows])]
        ),
        pa.Array.from_buffers(pa.int32(), n, [None, _buffer("i", [r["width"] for r in rows])]),
        pa.Array.from_buffers(pa.int32(), n, [None, _buffer("i", [r["height"] for r in rows])]),
    ]
    for name, size in (("K", 9), ("R", 9), ("C", 3)):
        flat = [float(value) for r in rows for value in r[name]]
        child = pa.Array.from_buffers(pa.float64(), n * size, [None, _buffer("d", flat)])
        columns.append(pa.FixedSizeListArray.from_arrays(child, size))
    encoded = [r["calib_status"].encode("utf-8") for r in rows]
    offsets = [0]
    for item in encoded:
        offsets.append(offsets[-1] + len(item))
    columns.append(
        pa.Array.from_buffers(
            pa.string(), n, [None, _buffer("i", offsets), pa.py_buffer(b"".join(encoded))]
        )
    )
    columns.append(
        pa.Array.from_buffers(pa.int32(), n, [None, _buffer("i", [r["calib_age"] for r in rows])])
    )
    all_null = pa.py_buffer(bytes((n + 7) // 8))
    columns.append(pa.Array.from_buffers(pa.float64(), n, [all_null, _buffer("d", [0.0] * n)]))
    table = pa.Table.from_arrays(columns, schema=CALIB_SCHEMA)
    table.validate(full=True)
    return table


def ffmpeg_version(ffmpeg: str) -> str:
    done = subprocess.run([ffmpeg, "-version"], capture_output=True, check=False)
    words = done.stdout.decode("utf-8", "replace").split()
    return words[2] if done.returncode == 0 and len(words) > 2 else "unknown"


def decode_hashes(ffmpeg: str, video: Path, width: int, height: int) -> list[str]:
    """Pixel hashes of every decoded frame of ``video``, in decode order."""
    done = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            "file:" + str(video.resolve()),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    errors = done.stderr.decode("utf-8", "replace").strip()
    if done.returncode != 0 or errors:
        raise StubError(5, f"ffmpeg could not decode the input video: {errors}")
    size = width * height * 3
    if len(done.stdout) % size:
        raise StubError(5, f"decoded {len(done.stdout)} bytes, not whole {width}x{height} frames")
    header = PIXEL_HASH_HEADER.format(height=height, width=width).encode("ascii")
    return [
        hashlib.sha256(header + done.stdout[start : start + size]).hexdigest()
        for start in range(0, len(done.stdout), size)
    ]


def run(job_path: Path) -> None:
    job_dir = job_path.parent
    job_bytes = job_path.read_bytes()
    job = json.loads(job_bytes.decode("utf-8"))
    if job.get("schema_version") != 1 or job.get("task") != "calibration":
        raise StubError(4, "not a calibration job of schema_version 1")
    if LICENSE not in job["allowed_licenses"]:
        raise StubError(3, f"the job does not allow this plugin's licence {LICENSE}")
    inputs = {}
    for item in job["inputs"]:
        path = job_dir / item["relpath"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise StubError(4, f"input {item['name']} is missing or has the wrong size")
        if sha256_file(path) != item["sha256"]:
            raise StubError(4, f"input {item['name']} does not match its sha256")
        inputs[item["name"]] = path
    frames = job["frames"]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise StubError(5, "ffmpeg is not on PATH")
    hashes = decode_hashes(ffmpeg, inputs[frames["input"]], frames["width"], frames["height"])
    expected_count = frames["input_frame_count"]
    if len(hashes) != expected_count:
        raise StubError(
            4, f"the input decodes to {len(hashes)} frames, the job says {expected_count}"
        )
    first = frames["input_first_frame"]
    recorded = json.loads(inputs[job["config"]["recorded_cameras"]].read_bytes().decode("utf-8"))
    if recorded.get("format") != "gbf.synth.cameras.v1":
        raise StubError(4, "the recorded cameras are not a gbf.synth.cameras.v1 document")
    by_index = {item["frame_index"]: item for item in recorded["frames"]}
    rows = []
    for index in range(frames["start"], frames["end"] + 1):
        camera = by_index.get(index)
        if camera is None:
            raise StubError(4, f"no recorded camera for canonical frame {index}")
        K = rescale_intrinsics(camera["K"], (camera["width"], camera["height"]), REPORT_SIZE)
        rows.append(
            {
                "frame_index": index,
                "width": REPORT_SIZE[0],
                "height": REPORT_SIZE[1],
                "K": [value for row in K for value in row],
                "R": [value for row in camera["R"] for value in row],
                "C": list(camera["C"]),
                "calib_status": "fresh",
                "calib_age": 0,
            }
        )
    outputs = []
    for spec in job["outputs"]:
        if spec["format"] != "calib.v1":
            raise StubError(4, f"cannot write format {spec['format']}")
        target = job_dir / spec["relpath"]
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(calib_table(rows), target)
        outputs.append(
            {
                "name": spec["name"],
                "relpath": spec["relpath"],
                "format": spec["format"],
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )
    receipt = {
        "schema_version": 1,
        "job_sha256": hashlib.sha256(job_bytes).hexdigest(),
        "status": "ok",
        "plugin": {
            "name": NAME,
            "version": VERSION,
            "license": LICENSE,
            "commit": None,
            "weights": [],
        },
        "outputs": outputs,
        "decoded_frames": [
            {"frame_index": first + k, "sha256": digest} for k, digest in enumerate(hashes)
        ],
        "environment": {
            "python": platform.python_version(),
            "pyarrow": pa.__version__,
            "ffmpeg": ffmpeg_version(ffmpeg),
        },
        "restrictions": list(job["restrictions"]),
        "messages": [
            "recorded-output replay: cameras are the recorded ones, K reported at "
            f"{REPORT_SIZE[0]}x{REPORT_SIZE[1]} for {frames['width']}x{frames['height']} frames"
        ],
    }
    text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with open(job_dir / "plugin-receipt.json", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: stub_calib.py PATH/TO/job.json", file=sys.stderr)
        return 2
    try:
        run(Path(argv[0]))
    except StubError as exc:
        print(f"stub-calib: {exc}", file=sys.stderr)
        return exc.code
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"stub-calib: bad job: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
