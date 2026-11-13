"""``video.json``: the ingest record of one source video (schema ``urn:gbf:schema:video``).

The record names the canonical frame index (plan 3.4): its id is the sha256 of the source
file bytes, its frames are the decode order of the canonical CFR file (the normalised file,
or the source itself when it already is CFR at ``fps``). It lists cuts and windows by
canonical frame number and carries the tool versions and the restriction tags. It holds no
absolute path, file name or timestamp, so identical inputs produce identical bytes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from gbf import contracts
from gbf.ingest.errors import RecordError
from gbf.ingest.normalizing import NormalizeResult
from gbf.ingest.probing import VideoProbe
from gbf.ingest.rates import format_rate, parse_rate
from gbf.ingest.tools import tool_versions
from gbf.ingest.window import WindowRecord
from gbf.util.io import write_json

SCHEMA_NAME = "video"
SCHEMA_VERSION = 1

# Strings that look like local paths: drive letters, UNC or POSIX roots, backslashes.
_PATH_LIKE = re.compile(r"(^[A-Za-z]:)|(^/)|(\\)|(^~)")


def _path_like_strings(value: Any, location: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        if _PATH_LIKE.search(value):
            found.append(f"{location or '<root>'}: looks like a local path: {value!r}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_path_like_strings(item, f"{location}/{key}" if location else str(key)))
    elif isinstance(value, list | tuple):
        for position, item in enumerate(value):
            found.extend(_path_like_strings(item, f"{location}/{position}"))
    return found


def _probe_dict(facts: VideoProbe, relpath: str | None) -> dict[str, Any]:
    data = facts.to_dict()
    if relpath is not None:
        data["relpath"] = relpath
    return data


def build_video_record(
    source: VideoProbe,
    *,
    normalized: NormalizeResult | None = None,
    fps: str | Fraction = "25/1",
    cuts: Iterable[int] = (),
    cut_detection: Mapping[str, Any] | None = None,
    windows: Sequence[WindowRecord] = (),
    window_relpaths: Mapping[str, str] | None = None,
    restrictions: Iterable[str] = (),
    tools: Mapping[str, str] | None = None,
    source_relpath: str | None = None,
    normalized_relpath: str | None = None,
) -> dict[str, Any]:
    """Assemble and validate a ``video.json`` record.

    ``normalized`` is the result of :func:`gbf.ingest.normalizing.normalize` for ``source``;
    without it the source must already be CFR at ``fps``. ``cuts`` are canonical frame
    indices (e.g. from :func:`gbf.ingest.cuts.detect_cuts`); ``cut_detection`` describes how
    they were found (:func:`gbf.ingest.cuts.cut_detection_settings`). ``window_relpaths``
    maps window ids to relpaths. ``tools`` defaults to :func:`gbf.ingest.tools.tool_versions`.

    Raises :class:`RecordError` listing every broken invariant (frame ranges, rates, ids,
    path-like strings) and :class:`gbf.contracts.ContractError` listing every schema problem.
    """
    rate = parse_rate(fps)
    problems: list[str] = []
    canonical = normalized.output if normalized is not None else source
    if normalized is not None and normalized.source.sha256 != source.sha256:
        problems.append("normalized.source is not the given source (sha256 differs)")
    if not canonical.is_cfr_at(rate):
        which = "normalized output" if normalized is not None else "source"
        problems.append(
            f"the {which} is not CFR at {format_rate(rate)} "
            f"(r_frame_rate={canonical.r_frame_rate}, avg_frame_rate={canonical.avg_frame_rate}, "
            f"timestamps_cfr={canonical.timestamps_cfr}); normalise it first"
        )
    if normalized is not None and normalized.settings.get("rate") != format_rate(rate):
        problems.append(
            f"normalized at {normalized.settings.get('rate')}, but fps is {format_rate(rate)}"
        )
    frame_count = canonical.nb_frames_decoded

    cut_list = list(cuts)
    for position, cut in enumerate(cut_list):
        if isinstance(cut, bool) or not isinstance(cut, int):
            problems.append(f"cuts/{position}: not an integer: {cut!r}")
        elif not 1 <= cut < frame_count:
            problems.append(f"cuts/{position}: {cut} is outside 1..{frame_count - 1}")
    if all(isinstance(c, int) for c in cut_list) and cut_list != sorted(set(cut_list)):
        problems.append("cuts must be strictly ascending without duplicates")

    allowed_sources = {source.sha256, canonical.sha256}
    relpaths = dict(window_relpaths or {})
    seen_ids: set[str] = set()
    window_dicts = []
    for position, window in enumerate(windows):
        where = f"windows/{position} ({window.window_id})"
        if window.window_id in seen_ids:
            problems.append(f"{where}: duplicate window id")
        seen_ids.add(window.window_id)
        if not 0 <= window.start_frame <= window.end_frame < frame_count:
            problems.append(
                f"{where}: frames {window.start_frame}..{window.end_frame} are outside "
                f"0..{frame_count - 1}"
            )
        if window.n_frames != window.end_frame - window.start_frame + 1:
            problems.append(f"{where}: n_frames {window.n_frames} does not match its range")
        if window.fps != format_rate(rate):
            problems.append(f"{where}: cut at {window.fps} fps, record is {format_rate(rate)}")
        if window.source_sha256 not in allowed_sources:
            problems.append(f"{where}: read from a file that is neither the source nor canonical")
        data = window.to_dict()
        if window.window_id in relpaths:
            data["relpath"] = relpaths.pop(window.window_id)
        window_dicts.append(data)
    for unknown in sorted(relpaths):
        problems.append(f"window_relpaths: no window with id {unknown!r}")

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "canonical_id": source.sha256,
        "source": _probe_dict(source, source_relpath),
        "fps": format_rate(rate),
        "frame_count": frame_count,
        "width": canonical.frame_width,
        "height": canonical.frame_height,
        "cuts": cut_list,
        "windows": window_dicts,
        "tools": dict(tools) if tools is not None else tool_versions(),
        "restrictions": list(restrictions),
    }
    if normalized is not None:
        record["normalized"] = _probe_dict(normalized.output, normalized_relpath)
        record["normalization"] = {
            **normalized.settings,
            "expected_frames": normalized.expected_frames,
        }
    elif normalized_relpath is not None:
        problems.append("normalized_relpath given without a normalized result")
    if cut_detection is not None:
        record["cut_detection"] = dict(cut_detection)

    problems.extend(_path_like_strings(record))
    if problems:
        raise RecordError("video.json record", problems)
    contracts.validate(record, SCHEMA_NAME)
    return record


def write_video_json(path: str | Path, source: VideoProbe, **kwargs: Any) -> dict[str, Any]:
    """Build the record with :func:`build_video_record` and write it (UTF-8, LF, sorted keys).

    Returns the record. Nothing is written when validation fails.
    """
    record = build_video_record(source, **kwargs)
    write_json(path, record)
    return record
