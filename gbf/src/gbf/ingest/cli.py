"""``python -m gbf ingest probe|normalize|window ...``.

Every command prints one JSON object to stdout (no paths or file names in it) and returns 0;
on failure it prints ``error: ...`` with every problem found to stderr and returns 1. Usage
errors return 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gbf.contracts import ContractError
from gbf.ingest.errors import IngestError
from gbf.ingest.normalizing import normalize
from gbf.ingest.probing import probe
from gbf.ingest.window import extract_window


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gbf ingest",
        description="Probe and normalize videos; cut lossless windows at canonical frames.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    probe_cmd = commands.add_parser(
        "probe", help="ffprobe facts with the exact decoded frame count and sha256."
    )
    probe_cmd.add_argument("video", type=Path)

    norm = commands.add_parser("normalize", help="Write a verified CFR H.264 .mp4 copy.")
    norm.add_argument("src", type=Path)
    norm.add_argument("dst", type=Path, help="output path ending in .mp4")
    norm.add_argument("--fps", default="25/1", help="canonical rate as num/den (default 25/1)")
    norm.add_argument("--all-intra", action="store_true", help="every frame a keyframe (-g 1)")
    norm.add_argument("--overwrite", action="store_true", help="replace an existing output")

    win = commands.add_parser(
        "window", help="Write canonical frames START..END (inclusive) losslessly to an .mkv."
    )
    win.add_argument("source", type=Path)
    win.add_argument("start", type=int, help="first canonical frame")
    win.add_argument("end", type=int, help="last canonical frame (inclusive)")
    win.add_argument("dst", type=Path, help="output path ending in .mkv")
    win.add_argument("--fps", default="25/1", help="canonical rate as num/den (default 25/1)")
    win.add_argument("--overwrite", action="store_true", help="replace an existing output")
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "probe":
        return probe(args.video).to_dict()
    if args.command == "normalize":
        result = normalize(
            args.src, args.dst, fps=args.fps, all_intra=args.all_intra, overwrite=args.overwrite
        )
        return result.to_dict()
    record = extract_window(
        args.source, args.start, args.end, args.dst, fps=args.fps, overwrite=args.overwrite
    )
    return record.to_dict()


def main(argv: list[str]) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits on --help (0) and on usage errors (2)
        return exc.code if isinstance(exc.code, int) else 2
    try:
        result = _run(args)
    except (IngestError, ContractError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return 0
