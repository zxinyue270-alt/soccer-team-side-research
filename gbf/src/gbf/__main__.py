"""Command-line entry point: ``python -m gbf <command> [args]``.

Each command lives in its own module and exposes ``main(argv: list[str]) -> int``.
Modules are imported lazily so that one broken command never blocks the others.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    "gate": ("gbf.gate.cli", "Run the offline quality gate (use --full for everything)."),
    "fetch": ("gbf.events.fetch", "Download pinned open data; requires explicit terms acceptance."),
    "align": ("gbf.events.align", "Align provider events to SoccerNet half videos."),
    "ingest": ("gbf.ingest.cli", "Probe and normalize videos; write video.json."),
    "l0": ("gbf.synth.l0", "Run the L0 synthetic smoke pipeline."),
    "highlights": ("gbf.pipeline", "Find highlight moments; analyse them on the original video."),
    "models": ("gbf.models", "Fetch or verify pinned model weights (kept outside the repository)."),
    "web": ("gbf.web.server", "Local web page (127.0.0.1 only) to run analyses and view results."),
    "eval-gsr": (
        "gbf.evaluate.gsr",
        "Deviation of a run from GSR annotations (tracking, positions, speeds).",
    ),
    "eval-gsr-dev": (
        "gbf.evaluate.gsr_dev",
        "Deviation of runs on the gates.v1 GSR dev clips from their annotations.",
    ),
}


def _usage() -> str:
    lines = ["usage: python -m gbf <command> [args]", "", "commands:"]
    width = max(len(name) for name in COMMANDS)
    for name, (_, summary) in COMMANDS.items():
        lines.append(f"  {name.ljust(width)}  {summary}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return 0 if args else 2
    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        print(f"unknown command: {name}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[name][0])
    return int(module.main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
