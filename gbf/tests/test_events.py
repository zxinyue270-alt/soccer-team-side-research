"""Tests for gbf.events: pins and fetch, provider parsing, alignment, frame mapping,
main-camera share, attack-direction checks and the align.json receipt.

Every provider-shaped document here is synthetic and built at run time: team names,
players, ids and times are made up, and nothing is derived from StatsBomb or
SoccerNet files. Downloads go through a fake opener; the session-wide offline
guard (tests/conftest.py) blocks any real network access.
"""

from __future__ import annotations

import io
import json
import random
import re
from dataclasses import dataclass
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from gbf import __main__ as gbf_main
from gbf import contracts
from gbf.events import _meta, align, direction, fetch, pins, providers
from gbf.events._meta import code_version, relpath_under, utc_timestamp
from gbf.util import hashing, paths

# ------------------------------------------------------------------ synthetic builders

HOME = {"id": 9101, "name": "Synthetic Home"}
AWAY = {"id": 9102, "name": "Synthetic Away"}
SHOT_OUTCOMES = {"Saved": 100, "Off T": 98, "Goal": 97, "Blocked": 96}
FIXED_CODE = {"gbf_version": "0.0.1", "git_commit": "0" * 40, "git_dirty": False}
FIXED_TIME = "2026-10-21T00:00:00Z"
FIXED_ENV = {"python": "3.12", "kloppy": "test"}


def _event_id(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def _ts(ms: int) -> str:
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _tactics(team: dict[str, Any]) -> dict[str, Any]:
    return {
        "formation": 442,
        "lineup": [
            {
                "player": {"id": team["id"] * 100 + k, "name": f"{team['name']} {k}"},
                "position": {"id": k, "name": "Goalkeeper" if k == 1 else "Outfield"},
                "jersey_number": k,
            }
            for k in range(1, 12)
        ],
    }


def statsbomb_bytes(
    shots: list[tuple[int, int, dict[str, Any], str]],
    kickoffs: list[tuple[int, int, dict[str, Any]]],
) -> tuple[bytes, bytes]:
    """Synthetic StatsBomb-like events and lineups JSON.

    ``shots``: (period, ms, team, outcome name); ``kickoffs``: (period, ms, team).
    Periods 1 and 2 always exist (Half Start at 0, Half End at 47:00).
    """
    counter = iter(range(1, 10_000))
    events: list[dict[str, Any]] = []

    def add(type_id: int, type_name: str, team: dict, period: int, ms: int, **extra: Any) -> None:
        number = next(counter)
        event = {
            "id": _event_id(number),
            "index": number,
            "period": period,
            "timestamp": _ts(ms),
            "minute": ms // 60_000,
            "second": (ms // 1000) % 60,
            "type": {"id": type_id, "name": type_name},
            "possession": 1,
            "possession_team": team,
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": team,
        }
        event.update(extra)
        events.append(event)

    add(35, "Starting XI", HOME, 1, 0, tactics=_tactics(HOME))
    add(35, "Starting XI", AWAY, 1, 0, tactics=_tactics(AWAY))
    for period in (1, 2):
        add(18, "Half Start", HOME, period, 0)
        add(18, "Half Start", AWAY, period, 0)
        timeline: list[tuple[int, str, Any]] = [
            (ms, "kickoff", team) for p, ms, team in kickoffs if p == period
        ]
        timeline += [(ms, outcome, team) for p, ms, team, outcome in shots if p == period]
        for ms, what, team in sorted(timeline, key=lambda item: item[0]):
            if what == "kickoff":
                add(
                    30,
                    "Pass",
                    team,
                    period,
                    ms,
                    player={"id": team["id"] * 100 + 10, "name": "kicker"},
                    location=[60.0, 40.0],
                    play_pattern={"id": 9, "name": "From Kick Off"},
                    **{
                        "pass": {
                            "type": {"id": 65, "name": "Kick Off"},
                            "end_location": [50.0, 40.0],
                            "recipient": {"id": team["id"] * 100 + 9, "name": "receiver"},
                        }
                    },
                )
            else:
                add(
                    16,
                    "Shot",
                    team,
                    period,
                    ms,
                    player={"id": team["id"] * 100 + 9, "name": "shooter"},
                    location=[100.0, 40.0],
                    shot={
                        "outcome": {"id": SHOT_OUTCOMES[what], "name": what},
                        "end_location": [120.0, 40.0, 1.0],
                        "type": {"id": 87, "name": "Open Play"},
                        "body_part": {"id": 40, "name": "Right Foot"},
                    },
                )
        add(34, "Half End", HOME, period, 47 * 60_000)
        add(34, "Half End", AWAY, period, 47 * 60_000)
    lineups = [
        {
            "team_id": team["id"],
            "team_name": team["name"],
            "lineup": [
                {
                    "player_id": team["id"] * 100 + k,
                    "player_name": f"{team['name']} {k}",
                    "player_nickname": None,
                    "jersey_number": k,
                    "country": {"id": 1, "name": "Nowhere"},
                }
                for k in range(1, 12)
            ],
        }
        for team in (HOME, AWAY)
    ]
    return json.dumps(events).encode("utf-8"), json.dumps(lineups).encode("utf-8")


def labels_doc(entries: list[tuple[int, int, str, str]]) -> dict[str, Any]:
    """Synthetic Labels-v2-like document from (half, position ms, label, team)."""
    annotations = []
    for half, position, label, team in entries:
        minutes, seconds = divmod(position // 1000, 60)
        annotations.append(
            {
                "gameTime": f"{half} - {minutes:02d}:{seconds:02d}",
                "label": label,
                "position": str(position),
                "team": team,
                "visibility": "visible",
            }
        )
    return {"UrlLocal": "synthetic/", "annotations": annotations}


ON, OFF = "Shots on target", "Shots off target"


@dataclass(frozen=True)
class Scenario:
    events: bytes
    lineups: bytes
    labels: bytes


def scenario() -> Scenario:
    """Half 1: offset 1.920 s with a boundary pair (-150 ms), a 151 ms pair and a planted
    1.580 s outlier. Half 2: four pairs, even count, offset 2.2925 s. Goal labels that
    would move the medians if they were used, a period-2 shot 0.1 s from a half-1 label
    (cross-period trap), one unlabelled shot, and kick-offs at 0 ms and 37 ms.
    """
    shots = [
        (1, 121_770, HOME, "Saved"),  # label 120 000, diff 1770, residual -150 (ok)
        (1, 301_915, AWAY, "Off T"),  # label 300 000, diff 1915, residual -5
        (1, 601_920, HOME, "Goal"),  # label 600 000, diff 1920, residual 0
        (1, 902_071, HOME, "Saved"),  # label 900 000, diff 2071, residual +151
        (1, 1_503_500, AWAY, "Off T"),  # label 1 500 000, diff 3500: planted outlier
        (2, 62_280, AWAY, "Saved"),  # label 60 000, diff 2280, residual -12.5
        (2, 300_100, HOME, "Blocked"),  # 0.1 s from half-1 label 300 000: other period
        (2, 422_290, HOME, "Off T"),  # label 420 000, diff 2290, residual -2.5
        (2, 1_202_300, AWAY, "Goal"),  # label 1 200 000, diff 2300, residual +7.5
        (2, 1_800_000, HOME, "Blocked"),  # no label near it
        (2, 2_402_295, HOME, "Saved"),  # label 2 400 000, diff 2295, residual +2.5
    ]
    kickoffs = [(1, 1_900, HOME), (1, 660_000, AWAY), (2, 2_300, AWAY)]
    events, lineups = statsbomb_bytes(shots, kickoffs)
    labels = labels_doc(
        [
            (1, 0, "Kick-off", "home"),
            (1, 120_000, ON, "home"),
            (1, 300_000, OFF, "away"),
            (1, 600_000, ON, "home"),
            (1, 602_500, "Goal", "home"),
            (1, 658_000, "Kick-off", "away"),
            (1, 900_000, ON, "away"),  # the StatsBomb shot is by home: team_match false
            (1, 1_500_000, OFF, "away"),
            (1, 1_504_000, "Goal", "away"),
            (2, 37, "Kick-off", "away"),
            (2, 60_000, ON, "away"),
            (2, 420_000, OFF, "home"),
            (2, 1_200_000, ON, "away"),
            (2, 1_203_000, "Goal", "away"),
            (2, 2_400_000, ON, "home"),
        ]
    )
    return Scenario(events, lineups, json.dumps(labels).encode("utf-8"))


def _shot(number: int, period: int, ms: int, side: str = "home") -> providers.StatsBombEvent:
    return providers.StatsBombEvent(
        event_id=_event_id(number),
        period=period,
        timestamp=_ts(ms),
        timestamp_ms=ms,
        team="Synthetic Home" if side == "home" else "Synthetic Away",
        team_side=side,
        outcome="Saved",
    )


def _label(index: int, half: int, ms: int, label: str = ON) -> providers.Label:
    minutes, seconds = divmod(ms // 1000, 60)
    return providers.Label(
        index=index,
        half=half,
        position_ms=ms,
        game_time=f"{half} - {minutes:02d}:{seconds:02d}",
        label=label,
        team="home",
        visibility="visible",
    )


def _write_inputs(root: Path, data: Scenario) -> tuple[Path, Path, Path]:
    raw = root / "raw" / "比赛 1"
    raw.mkdir(parents=True)
    paths_out = (raw / "events.json", raw / "lineups.json", raw / "Labels-v2.json")
    for path, payload in zip(paths_out, (data.events, data.lineups, data.labels), strict=True):
        path.write_bytes(payload)
    return paths_out


def _pin_entry(name: str, role: str, payload: bytes, revision: str) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "host": "hf-dataset" if role.startswith("soccernet") else "github",
        "repository": "example/synthetic",
        "revision": revision,
        "path": f"games/game 1/{name}.json",
        "git_blob_sha1": hashing.git_blob_sha1(payload),
        "bytes": len(payload),
        "restrictions": ["no_share"]
        if role.startswith("soccernet")
        else ["no_share", "non_commercial", "statsbomb_logo"],
        **({} if role.startswith("soccernet") else {"terms": "statsbomb"}),
    }


def _synthetic_pin_set(sources: list[dict[str, Any]]) -> pins.PinSet:
    terms = pins.load_pins().terms["statsbomb"]
    return pins.parse_pins(
        {
            "schema_version": 1,
            "terms": {
                "statsbomb": {
                    "title": terms.title,
                    "agreement_url": terms.agreement_url,
                    "agreement_git_blob_sha1": terms.agreement_git_blob_sha1,
                    "clauses": list(terms.clauses),
                }
            },
            "sources": sources,
        }
    )


NO_PINS = pins.PinSet(pins=(), terms={})


def _run_scenario(
    root: Path, *, pin_set: pins.PinSet = NO_PINS, out_name: str = "align.json", **kwargs: Any
) -> dict[str, Any]:
    data = scenario()
    events, lineups, labels = _write_inputs(root / "data", data)
    return align.run(
        events=events,
        lineups=lineups,
        labels=labels,
        out=root / "runs" / out_name,
        data_root=root / "data",
        pin_set=pin_set,
        code=FIXED_CODE,
        created_at=FIXED_TIME,
        **kwargs,
    )


# --------------------------------------------------------------------- StatsBomb parsing


def test_parse_statsbomb_timestamp_ms() -> None:
    assert providers.parse_statsbomb_timestamp_ms("00:12:34.567") == 754_567
    assert providers.parse_statsbomb_timestamp_ms("01:02:03.004") == 3_723_004
    for bad in ["00:12:34", "0:12:34.567", "00:60:00.000", "00:00:60.000", "00:00:00.5", None]:
        with pytest.raises(ValueError, match="not a StatsBomb timestamp|below 60"):
            providers.parse_statsbomb_timestamp_ms(bad)  # type: ignore[arg-type]
    assert providers.timedelta_to_ms(timedelta(seconds=754, microseconds=567_000)) == 754_567
    with pytest.raises(ValueError, match="whole number of milliseconds"):
        providers.timedelta_to_ms(timedelta(microseconds=1500))


def test_kloppy_loader_keeps_period_relative_timestamps() -> None:
    data = scenario()
    match = providers.load_statsbomb(data.events, data.lineups)
    assert match.periods == (1, 2)
    first_period_two = next(shot for shot in match.shots if shot.period == 2)
    # 00:01:02.280 in period 2 stays 62 280 ms; adding the cumulative period start
    # (47:00 = 2 820 000 ms here) would be the classic mistake.
    assert first_period_two.timestamp == "00:01:02.280"
    assert first_period_two.timestamp_ms == 62_280
    assert [(s.period, s.timestamp_ms) for s in match.shots][:2] == [(1, 121_770), (1, 301_915)]
    assert len(match.shots) == 11
    assert [(k.period, k.timestamp_ms) for k in match.kickoffs] == [
        (1, 1_900),
        (1, 660_000),
        (2, 2_300),
    ]
    sides = {shot.timestamp_ms: shot.team_side for shot in match.shots}
    assert sides[121_770] == "home"
    assert sides[301_915] == "away"
    assert {shot.outcome for shot in match.shots} == {"Saved", "Off T", "Goal", "Blocked"}
    assert match.loader.startswith("kloppy ")


def test_loader_rejects_kloppy_timestamps_that_disagree_with_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kloppy.infra.serializers.event.statsbomb import specification

    original = specification.parse_str_ts
    monkeypatch.setattr(
        specification, "parse_str_ts", lambda text: original(text) + timedelta(minutes=45)
    )
    data = scenario()
    with pytest.raises(providers.ProviderDataError) as caught:
        providers.load_statsbomb(data.events, data.lineups)
    # every shot and kick-off is reported, not just the first one
    assert len(caught.value.problems) == 11 + 3
    assert "not period-relative" in caught.value.problems[0]


def test_loader_turns_kloppy_failures_into_provider_errors() -> None:
    _, lineups = statsbomb_bytes([], [])
    with pytest.raises(providers.ProviderDataError, match="kloppy cannot parse"):
        providers.load_statsbomb(b"[]", lineups)


# ---------------------------------------------------------------------- label parsing


def test_labels_v2_parser_reads_positions_and_reports_every_problem() -> None:
    labels = providers.parse_labels_v2(
        labels_doc([(1, 754_567, ON, "home"), (2, 37, "Kick-off", "away")])
    )
    assert [(lab.half, lab.position_ms, lab.label, lab.index) for lab in labels] == [
        (1, 754_567, ON, 0),
        (2, 37, "Kick-off", 1),
    ]
    assert labels[0].game_time == "1 - 12:34"
    bad = {
        "annotations": [
            {"gameTime": "3 - 00:01", "label": ON, "position": "1000"},
            {"gameTime": "1 - 00:01", "label": ON, "position": "12.5"},
            {"gameTime": "1 - 00:01", "label": ON, "position": -5},
            {"gameTime": "1 - 00:01", "position": "1000"},
            {"gameTime": "1 - 00:61", "label": ON, "position": "1000"},
            "not an object",
        ]
    }
    with pytest.raises(providers.ProviderDataError) as caught:
        providers.parse_labels_v2(bad)
    assert len(caught.value.problems) == 6
    with pytest.raises(providers.ProviderDataError, match="'annotations' list"):
        providers.parse_labels_v2({"no": "annotations"})
    with pytest.raises(providers.ProviderDataError, match="not UTF-8 JSON"):
        providers.parse_labels_v2(b"\xff not json")
    raw = json.dumps(labels_doc([(1, 5_000, OFF, "home")])).encode("utf-8")
    assert providers.parse_labels_v2(raw)[0].position_ms == 5_000


def test_game_time_disagreement_is_reported() -> None:
    labels = (
        _label(0, 1, 754_000),
        providers.Label(1, 1, 754, "1 - 12:34", ON, "home", "visible"),  # seconds, not ms
    )
    found = providers.game_time_disagreements(labels)
    assert len(found) == 1
    assert "annotation 1" in found[0]


# ---------------------------------------------------------------------- alignment maths


def test_offsets_residuals_and_planted_outlier() -> None:
    data = scenario()
    match = providers.load_statsbomb(data.events, data.lineups)
    labels = providers.parse_labels_v2(data.labels)
    result = align.align_shots(match.shots, labels)

    first, second = result.halves
    assert first.offset_ms == 1920
    assert first.offset_s == Fraction(48, 25)
    residuals = [(pair.label.position_ms, pair.residual_ms, pair.status) for pair in first.pairs]
    assert residuals == [
        (120_000, -150, "ok"),  # exactly at the 0.15 s tolerance: ok
        (300_000, -5, "ok"),
        (600_000, 0, "ok"),
        (900_000, 151, "needs_review"),  # 1 ms beyond the tolerance
        (1_500_000, 1580, "needs_review"),  # planted outlier; the median ignores it
    ]
    # Even count: mean of the two middle diffs (2290 and 2295 ms), kept exact.
    assert second.offset_ms == Fraction(4585, 2)
    assert [pair.residual_ms for pair in second.pairs] == [
        Fraction(-25, 2),
        Fraction(-5, 2),
        Fraction(15, 2),
        Fraction(5, 2),
    ]
    assert all(pair.status == "ok" for pair in second.pairs)
    assert [pair.diff_ms for pair in first.pairs] == [1770, 1915, 1920, 2071, 3500]


def test_goal_labels_are_excluded() -> None:
    shots = [_shot(1, 1, 61_000), _shot(2, 1, 182_100), _shot(3, 1, 303_200)]
    labels = [
        _label(0, 1, 59_000),  # diff 2000
        _label(1, 1, 180_000),  # diff 2100
        _label(2, 1, 301_000),  # diff 2200
        # Goal labels sit after the shot. Used as shot labels they would add diffs of
        # -1000 and -900 ms, and the median of the five diffs would be 2000, not 2100.
        _label(3, 1, 62_000, "Goal"),
        _label(4, 1, 304_100, "Goal"),
    ]
    result = align.align_shots(shots, labels)
    half = result.half(1)
    assert half.offset_ms == 2100
    assert {pair.label.label for pair in half.pairs} == {ON}
    assert len(half.pairs) == 3
    assert dict(result.excluded_label_counts) == {"Goal": 2}


def test_nearest_match_stays_in_period_and_breaks_ties_deterministically() -> None:
    shots = [
        _shot(1, 2, 100_100),  # 0.1 s away but in period 2
        _shot(2, 1, 102_000),  # 2.0 s away, period 1: the right one
        _shot(3, 1, 110_000),
        _shot(4, 1, 200_000),  # tie for the label at 201 000: 1 s either side
        _shot(5, 1, 202_000),
    ]
    labels = [_label(0, 1, 100_000), _label(1, 1, 201_000)]
    result = align.align_shots(shots, labels)
    pairs = result.half(1).pairs
    assert pairs[0].shot.event_id == _event_id(2)
    assert pairs[1].shot.event_id == _event_id(4)  # tie: the earlier timestamp wins
    assert result.half(2).offset_ms is None  # shots alone do not align a half
    # sorted by (period, time, id), whatever the input order
    assert [shot.event_id for shot in result.unmatched_shots] == [
        _event_id(3),
        _event_id(5),
        _event_id(1),
    ]
    again = align.align_shots(list(reversed(shots)), list(reversed(labels)))
    assert again == result


def test_shared_matches_are_flagged_and_half_without_shots_is_unaligned() -> None:
    shots = [_shot(1, 1, 50_000)]
    labels = [_label(0, 1, 48_000), _label(1, 1, 48_500), _label(2, 2, 10_000)]
    result = align.align_shots(shots, labels)
    assert [pair.shared_match for pair in result.half(1).pairs] == [True, True]
    assert result.half(1).offset_ms == 1750  # diffs 2000 and 1500 ms
    assert result.half(2).aligned is False
    assert [label.index for label in result.half(2).unmatched_labels] == [2]
    with pytest.raises(ValueError, match="positive"):
        align.align_shots(shots, labels, tolerance_s=0)


def test_median_is_exact() -> None:
    assert align.median_ms([3, 1, 2]) == 2
    assert align.median_ms([1, 2]) == Fraction(3, 2)
    with pytest.raises(ValueError, match="empty"):
        align.median_ms([])


# --------------------------------------------------------------------- kick-off check


def test_kickoff_check_uses_earliest_kickoffs_and_flags_gross_errors() -> None:
    data = scenario()
    match = providers.load_statsbomb(data.events, data.lineups)
    labels = providers.parse_labels_v2(data.labels)
    result = align.align_shots(match.shots, labels)
    checks = align.check_kickoffs(match.kickoffs, labels, result)
    first, second = checks
    # half 1: 1.900 - 1.920 - 0.000 = -0.020 s; the 11:00 kick-off after the goal is ignored
    assert (first.status, first.deviation_ms, first.label.position_ms) == ("ok", -20, 0)
    assert first.kickoff.timestamp_ms == 1_900
    # half 2: (2.300 - 2.2925) - 0.037 = -0.0295 s
    assert (second.status, second.deviation_ms) == ("ok", Fraction(-59, 2))

    late = [providers.StatsBombEvent(_event_id(99), 2, "00:00:09.000", 9_000, "x", "away", None)]
    checks = align.check_kickoffs(late, labels, result, tolerance_s="3")
    assert checks[1].status == "needs_review"  # 9.000 - 2.2925 - 0.037 = 6.67 s > 3 s
    assert checks[0].status == "missing"
    assert checks[0].missing == ("statsbomb_kickoff",)
    no_labels = [label for label in labels if label.label != "Kick-off"]
    checks = align.check_kickoffs(match.kickoffs, no_labels, result)
    assert [c.missing for c in checks] == [("label",), ("label",)]


# ---------------------------------------------------------------------- frame mapping


def test_frame_for_time_edges() -> None:
    fps = "25/1"
    assert align.frame_for_time("1.92", offset_s="1.92", fps=fps) == 0
    assert align.frame_for_time("1.96", offset_s="1.92", fps=fps) == 1  # boundary: later frame
    assert align.frame_for_time("1.9599", offset_s="1.92", fps=fps) == 0
    # Floats are read as their decimal: 0.12 - 0.08 is exactly 0.04 s = frame 1,
    # where binary floating point would give 0.0399999... and frame 0.
    assert align.frame_for_time(0.12, offset_s=0.08, fps=25) == 1
    ntsc = "30000/1001"
    assert align.frame_for_time(Fraction(1001, 30000), offset_s=0, fps=ntsc) == 1
    assert align.frame_for_time(Fraction(1000, 30000), offset_s=0, fps=ntsc) == 0
    assert align.frame_for_time(align.frame_start_s(0, ntsc), offset_s=0, fps=ntsc) == 0
    # Frame starts where binary floats land just below the boundary (e.g. 29/25 s * 25
    # = 28.999999999999996 in floats): exact arithmetic must still give the frame.
    assert align.frame_for_time("3.08", offset_s="1.92", fps=fps) == 29
    for frame in (1, 2, 15, 25, 30, 1499, 67_499):
        start = align.frame_start_s(frame, ntsc)
        just_before = start - Fraction(1, 10**9)
        assert align.frame_for_time(start, offset_s=0, fps=ntsc) == frame
        assert align.frame_for_time(just_before, offset_s=0, fps=ntsc) == frame - 1
    assert align.frame_start_s(3, "25/1") == Fraction(3, 25)
    assert align.frame_for_time("41.92", offset_s="1.92", fps=fps, n_frames=1001) == 1000
    with pytest.raises(align.FrameRangeError, match="past the last frame"):
        align.frame_for_time("41.96", offset_s="1.92", fps=fps, n_frames=1001)
    with pytest.raises(align.FrameRangeError, match="before the first frame"):
        align.frame_for_time("1.919", offset_s="1.92", fps=fps)
    with pytest.raises(TypeError):
        align.frame_for_time(1, offset_s=0, fps=29.97)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        align.parse_fps("0/1")
    with pytest.raises(ValueError, match="finite"):
        align.to_fraction(float("nan"))


# ------------------------------------------------------------------- main-camera share


def _cameras_doc() -> dict[str, Any]:
    def change(half: int, ms: int, label: str, replay: str) -> dict[str, Any]:
        minutes, seconds = divmod(ms // 1000, 60)
        return {
            "gameTime": f"{half} - {minutes:02d}:{seconds:02d}",
            "label": label,
            "position": str(ms),
            "replay": replay,
            "change_type": "abrupt",
        }

    return {
        "annotations": [
            change(1, 0, "Main camera center", "real-time"),
            change(1, 10_000, "Close-up player or field referee", "real-time"),
            change(1, 20_000, "Main camera center", "replay"),
            change(1, 30_000, "Main camera left", "real-time"),
            change(2, 5_000, "Main camera right", "real-time"),
        ]
    }


def test_main_camera_share_counts_live_main_segments() -> None:
    times = [(1, 5), (1, "10.0"), (1, 25.5), (1, 40), (2, 1), (2, Fraction(6))]
    share = align.main_camera_share(_cameras_doc(), times)
    assert [event.camera for event in share.events] == [
        "Main camera center",
        "Close-up player or field referee",  # 10.0 s is the start of this segment
        "Main camera center",
        "Main camera left",
        None,  # before the half's first annotation
        "Main camera right",
    ]
    assert [event.is_main for event in share.events] == [True, False, False, True, False, True]
    assert (share.n_events, share.n_main, share.n_unknown) == (6, 3, 1)
    assert share.share == Fraction(1, 2)
    with_replays = align.main_camera_share(_cameras_doc(), times, include_replays=True)
    assert with_replays.n_main == 4
    centre_only = align.main_camera_share(_cameras_doc(), times, main_labels={"Main camera center"})
    assert centre_only.n_main == 1
    assert centre_only.main_labels == ("Main camera center",)
    raw = json.dumps(_cameras_doc()).encode("utf-8")
    assert align.main_camera_share(raw, times).n_main == 3
    assert align.main_camera_share(_cameras_doc(), []).share is None
    with pytest.raises(ValueError, match="half must be one of"):
        align.main_camera_share(_cameras_doc(), [(3, 1.0)])
    with pytest.raises(ValueError, match="negative video time"):
        align.main_camera_share(_cameras_doc(), [(1, -0.5)])
    # StatsBomb shot times reach the video through the offsets: 12.000 - 1.920 = 10.080 s
    # (close-up), 22.000 - 1.920 = 20.080 s (replay), 7.500 - 2.2925 = 5.2075 s (main).
    shots = [_shot(1, 1, 12_000), _shot(2, 1, 22_000), _shot(3, 2, 7_500)]
    offsets = {1: Fraction("1.92"), 2: Fraction("2.2925")}
    video_times = align.statsbomb_video_times(shots, offsets)
    assert video_times == [(1, Fraction("10.08")), (1, Fraction("20.08")), (2, Fraction("5.2075"))]
    mapped = align.main_camera_share(_cameras_doc(), video_times)
    assert [event.is_main for event in mapped.events] == [False, False, True]
    with pytest.raises(ValueError, match="no offset for period 2"):
        align.statsbomb_video_times(shots, {1: Fraction("1.92")})
    broken = _cameras_doc()
    broken["annotations"][0]["replay"] = "maybe"
    broken["annotations"][1]["position"] = "soon"
    with pytest.raises(providers.ProviderDataError) as caught:
        align.main_camera_share(broken, times)
    assert len(caught.value.problems) == 2


# ------------------------------------------------------------------ attack direction


def test_side_counts_and_exact_share_threshold() -> None:
    counts = direction.shots_in_attacking_half(
        [30.0, 40.5, -10.0, 0.0, 52.0], attacks_positive_x=True
    )
    assert (counts.n_total, counts.n_on_side, counts.n_on_line) == (5, 3, 1)
    assert counts.meets(Fraction(3, 5))
    assert not counts.meets(Fraction(61, 100))
    flipped = direction.shots_in_attacking_half([30.0, -40.0], attacks_positive_x=False)
    assert flipped.n_on_side == 1
    yards = direction.shots_in_attacking_half([100.0, 59.0], attacks_positive_x=True, halfway_x=60)
    assert yards.n_on_side == 1
    assert not direction.SideCount(0, 0, 0).meets(Fraction(0))
    with pytest.raises(ValueError, match="non-finite") as caught:
        direction.shots_in_attacking_half([1.0, float("nan")], attacks_positive_x=True)
    assert str(caught.value).endswith("[1]")


def test_keeper_side_equals_nearer_goal_by_distance() -> None:
    rng = random.Random(20261021)
    half_length = 52.5
    for _ in range(500):
        x, y = rng.uniform(-half_length, half_length), rng.uniform(-34.0, 34.0)
        for defends_positive in (True, False):
            own = half_length if defends_positive else -half_length
            nearer_own = (x - own) ** 2 + y**2 < (x + own) ** 2 + y**2
            counted = direction.keeper_near_own_goal([x], defends_positive_x=defends_positive)
            assert (counted.n_on_side == 1) == nearer_own, (x, y, defends_positive)


def test_direction_consistency_problems() -> None:
    good = {("A", 1): True, ("B", 1): False, ("A", 2): False, ("B", 2): True}
    assert direction.direction_problems(good) == []
    same_way = {**good, ("B", 1): True}
    no_swap = {**good, ("A", 2): True, ("B", 2): False}
    found = direction.direction_problems(same_way)
    assert any("half 1: both teams attack +x" in p for p in found)
    assert any("B: attacks the same way" in p for p in found)
    found = direction.direction_problems(no_swap)
    assert sorted(p.split(":")[0] for p in found) == ["A", "B"]


def test_check_attack_directions_passes_and_fails() -> None:
    observations = [
        direction.TeamHalf("A", 1, True, [40.0, 35.0, 20.0, 45.0, -5.0], [50.0, 48.0, 51.0]),
        direction.TeamHalf("B", 1, False, [-40.0, -30.0, -20.0, -45.0], [-50.0, -49.0, -51.0]),
        direction.TeamHalf("A", 2, False, [-40.0, -35.0, -20.0, -45.0], [-50.0, -48.0, -51.0]),
        direction.TeamHalf("B", 2, True, [40.0, 30.0, 20.0, 45.0], [50.0, 49.0, 51.0]),
    ]
    thresholds = {"min_shot_share": Fraction(4, 5), "min_keeper_share": Fraction(1), "min_count": 3}
    report = direction.check_attack_directions(observations, **thresholds)
    assert report.passed, report.problems
    assert report.checks[0].shots.n_on_side == 4  # 4 of 5 meets 4/5 exactly

    # A direction hypothesis that is wrong for team A in half 2 must fail all three ways.
    a2 = observations[2]
    flipped = direction.TeamHalf("A", 2, True, a2.shot_x, a2.opposing_keeper_x)
    wrong = [observations[0], observations[1], flipped, observations[3]]
    report = direction.check_attack_directions(wrong, **thresholds)
    assert not report.passed
    joined = "\n".join(report.problems)
    assert "half 2: both teams attack +x" in joined
    assert "A half 2: only 0/4 shots" in joined
    assert "A half 2: only 0/3 keeper positions" in joined

    sparse = [direction.TeamHalf("A", 1, True, [40.0], [])]
    report = direction.check_attack_directions(sparse, **thresholds)
    assert not report.passed
    assert not report.problems
    assert len(report.not_judged) == 2
    float_threshold: Any = 0.8
    with pytest.raises(ValueError, match="Fraction"):
        direction.check_attack_directions(
            observations, min_shot_share=float_threshold, min_keeper_share=Fraction(1), min_count=3
        )
    with pytest.raises(ValueError, match="duplicate"):
        direction.check_attack_directions([observations[0], observations[0]], **thresholds)


# ---------------------------------------------------------------- receipt and schema


def test_align_receipt_is_valid_exact_and_free_of_local_paths(unicode_tmp_path: Path) -> None:
    receipt = _run_scenario(unicode_tmp_path)
    assert contracts.problems(receipt, "align") == []
    result = receipt["result"]
    assert [half["offset_s"] for half in result["halves"]] == [1.92, 2.2925]
    assert [(h["n_pairs"], h["n_ok"], h["n_needs_review"]) for h in result["halves"]] == [
        (5, 3, 2),
        (4, 4, 0),
    ]
    assert result["summary"] == {
        "n_pairs": 9,
        "n_ok": 7,
        "n_needs_review": 2,
        "n_unmatched_labels": 0,
        "n_unmatched_statsbomb_shots": 2,
        "n_excluded_labels": 3,
    }
    assert result["excluded_label_counts"] == {"Goal": 3}
    outlier = result["pairs"][4]
    assert (outlier["diff_s"], outlier["residual_s"], outlier["status"]) == (
        3.5,
        1.58,
        "needs_review",
    )
    assert outlier["statsbomb_timestamp"] == "00:25:03.500"
    assert result["pairs"][3]["team_match"] is False
    assert result["pairs"][0]["team_match"] is True
    assert [s["statsbomb_timestamp"] for s in result["unmatched_statsbomb_shots"]] == [
        "00:05:00.100",
        "00:30:00.000",
    ]
    assert [check["status"] for check in result["kickoff_checks"]] == ["ok", "ok"]
    assert receipt["restrictions"] == ["no_share", "non_commercial", "statsbomb_logo"]
    # synthetic inputs match no pin: provenance unverified, so the receipt is partial
    assert receipt["status"] == "partial"
    assert [entry["source_revision"] for entry in receipt["inputs"]] == ["unpinned"] * 3
    assert receipt["inputs"][0]["relpath"] == "raw/比赛 1/events.json"
    data = scenario()
    assert receipt["inputs"][2]["sha256"] == hashing.sha256_bytes(data.labels)
    assert receipt["inputs"][2]["git_blob_sha1"] == hashing.git_blob_sha1(data.labels)

    text = (unicode_tmp_path / "runs" / "align.json").read_text(encoding="utf-8")
    assert str(unicode_tmp_path) not in text
    assert unicode_tmp_path.as_posix() not in text
    assert re.search(r"[A-Za-z]:/", text) is None
    assert re.search("[A-Za-z]:" + re.escape(chr(92)), text) is None
    assert b"\r\n" not in (unicode_tmp_path / "runs" / "align.json").read_bytes()


def test_align_receipt_passes_when_inputs_match_pins(unicode_tmp_path: Path) -> None:
    data = scenario()
    revision = "1" * 40
    pin_set = _synthetic_pin_set(
        [
            _pin_entry("synthetic-events", "statsbomb_events", data.events, revision),
            _pin_entry("synthetic-lineups", "statsbomb_lineups", data.lineups, revision),
            _pin_entry("synthetic-labels", "soccernet_labels_v2", data.labels, "2" * 40),
        ]
    )
    receipt = _run_scenario(unicode_tmp_path, pin_set=pin_set)
    assert receipt["status"] == "passed"
    assert [e["source_revision"] for e in receipt["inputs"]] == [revision, revision, "2" * 40]
    assert [s["pin"] for s in receipt["result"]["sources"]] == [
        "synthetic-events",
        "synthetic-lineups",
        "synthetic-labels",
    ]
    assert not any("matches no pin" in w for w in receipt["result"]["warnings"])


def test_align_receipt_is_deterministic_and_not_overwritten(unicode_tmp_path: Path) -> None:
    first = _run_scenario(unicode_tmp_path / "a")
    second = _run_scenario(unicode_tmp_path / "b")
    assert first == second
    bytes_a = (unicode_tmp_path / "a" / "runs" / "align.json").read_bytes()
    bytes_b = (unicode_tmp_path / "b" / "runs" / "align.json").read_bytes()
    assert bytes_a == bytes_b
    data_root = unicode_tmp_path / "a" / "data"
    events, lineups, labels = (
        data_root / "raw" / "比赛 1" / n for n in ("events.json", "lineups.json", "Labels-v2.json")
    )
    with pytest.raises(FileExistsError):
        align.run(
            events=events,
            lineups=lineups,
            labels=labels,
            out=unicode_tmp_path / "a" / "runs" / "align.json",
            code=FIXED_CODE,
            created_at=FIXED_TIME,
        )


def test_schema_rejects_broken_receipts_listing_every_problem(unicode_tmp_path: Path) -> None:
    receipt = _run_scenario(unicode_tmp_path)
    broken = json.loads(json.dumps(receipt))
    broken["restrictions"] = ["no_share", "non_commercial"]
    broken["inputs"][0]["relpath"] = "C:/data/events.json"
    broken["result"]["method"]["name"] = "something-else"
    broken["result"]["pairs"][0]["status"] = "maybe"
    broken["result"]["pairs"][1]["player_name"] = "not allowed here"
    broken["result"]["halves"][1]["offset_s"] = None
    found = contracts.problems(broken, "align")
    locations = {problem.split(":")[0] for problem in found}
    assert {
        "restrictions",
        "inputs/0/relpath",
        "result/method/name",
        "result/pairs/0/status",
        "result/pairs/1",
        "result/halves/1/offset_s",
    } <= locations
    with pytest.raises(contracts.ContractError) as caught:
        contracts.validate(broken, "align")
    assert len(caught.value.problems) == len(found) >= 6


def test_offsets_round_trip_exactly_through_the_receipt(unicode_tmp_path: Path) -> None:
    receipt = _run_scenario(unicode_tmp_path)
    offsets = align.offsets_from_receipt(receipt)
    assert offsets == {1: Fraction(48, 25), 2: Fraction(4585, 2000)}
    # a StatsBomb shot at 00:25:03.500 in half 1 lands on frame floor((1503.5 - 1.92) * 25)
    assert align.frame_for_time("1503.5", offset_s=offsets[1], fps="25/1") == 37_539


def test_expectations_never_change_the_method(unicode_tmp_path: Path) -> None:
    met = _run_scenario(
        unicode_tmp_path,
        out_name="met.json",
        expectations=align.Expectations({1: Fraction("1.91"), 2: Fraction("2.29")}, min_ok=7),
    )
    assert met["result"]["expectations"]["ok"] is True
    assert met["status"] == "partial"  # unchanged by expectations that are met
    root = unicode_tmp_path / "second"
    missed = _run_scenario(root, expectations=align.Expectations({1: Fraction("1.85")}, min_ok=8))
    expectation = missed["result"]["expectations"]
    assert expectation["ok"] is False
    assert missed["status"] == "failed"
    assert expectation["offsets"][0] == {
        "half": 1,
        "expected_s": 1.85,
        "actual_s": 1.92,
        "tolerance_s": 0.04,
        "ok": False,
    }
    assert expectation["min_ok"] == {"expected": 8, "actual": 7, "ok": False}
    assert missed["result"]["halves"] == met["result"]["halves"]


def test_align_cli_writes_receipt_and_reports_status(
    unicode_tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events, lineups, labels = _write_inputs(unicode_tmp_path / "data", scenario())
    monkeypatch.setenv("DATA_ROOT", str(unicode_tmp_path / "data"))
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1792540800")
    out = unicode_tmp_path / "runs" / "align-synthetic" / "align.json"
    argv = ["align", "--events", str(events), "--lineups", str(lineups), "--labels", str(labels)]
    argv += ["--out", str(out), "--expect-offset", "1=1.92", "--expect-offset", "2=2.2925"]
    assert gbf_main.main(argv) == 1  # partial: synthetic inputs match no pin
    printed = capsys.readouterr().out
    assert "half 1: offset +1.9200 s, 3/5 pairs within 0.15 s" in printed
    assert "expectations: met" in printed
    assert "status: partial" in printed
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["created_at"] == "2026-10-21T00:00:00Z"
    assert contracts.problems(receipt, "align") == []
    assert receipt["code"]["git_commit"] == "unknown" or re.fullmatch(
        "[0-9a-f]{40}", receipt["code"]["git_commit"]
    )
    assert gbf_main.main(argv) == 2  # refuses to overwrite without --force
    assert "already exists" in capsys.readouterr().err
    assert gbf_main.main([*argv, "--force", "--expect-min-ok", "8"]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "failed"
    bad_labels = unicode_tmp_path / "data" / "bad.json"
    bad_labels.write_bytes(b'{"annotations": [{"gameTime": "9 - 00:00"}]}')
    argv_bad = [*argv[:6], str(bad_labels), "--out", str(unicode_tmp_path / "x.json")]
    assert gbf_main.main(argv_bad) == 2
    assert not (unicode_tmp_path / "x.json").exists()


# ------------------------------------------------------------------------ pins, fetch


def test_packaged_pins_are_valid_and_name_their_revision() -> None:
    pin_set = pins.load_pins()
    assert [pin.role for pin in pin_set.pins] == [
        "statsbomb_events",
        "statsbomb_lineups",
        "soccernet_labels_v2",
        "soccernet_labels_cameras",
    ]
    for pin in pin_set.pins:
        assert pin.revision in pin.url
        assert pin.url.startswith("https://")
        assert pin.local_relpath.startswith(f"{pin.host}/{pin.repository}/{pin.revision[:12]}/")
        assert " " not in pin.url
        if pin.role.startswith("statsbomb"):
            assert pin.terms == "statsbomb"
            assert pin.url.startswith("https://raw.githubusercontent.com/hudl/open-data/")
            assert set(pin.restrictions) == {"no_share", "non_commercial", "statsbomb_logo"}
        else:
            assert pin.terms is None
            assert "no_share" in pin.restrictions
            assert "Manchester%20City%206%20-%201%20Newcastle%20Utd" in pin.url
    terms = pin_set.terms["statsbomb"]
    assert pin_set.pins[0].revision in terms.agreement_url
    assert [clause.split()[0] for clause in terms.clauses] == ["1.2.1", "1.2.2", "1.4", "2.2"]


def test_parse_pins_reports_every_problem() -> None:
    with pytest.raises(pins.PinsError) as caught:
        pins.parse_pins(
            {
                "schema_version": 2,
                "sources": [
                    {
                        "name": "Bad Name",
                        "role": "video",
                        "host": "ftp",
                        "repository": "no-slash",
                        "revision": "abc",
                        "path": "../escape.json",
                        "git_blob_sha1": "XYZ",
                        "bytes": 0,
                        "restrictions": ["made_up"],
                        "terms": "unknown",
                        "extra": 1,
                    }
                ],
            }
        )
    assert len(caught.value.problems) >= 12


class FakeResponse:
    def __init__(self, data: bytes, status: int = 200):
        self._buffer = io.BytesIO(data)
        self.status = status
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    """Serves fixed bytes per URL and records every request; never touches a socket."""

    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.requests: list[Any] = []
        self.responses: list[FakeResponse] = []

    def open(self, request: Any, timeout: float | None = None) -> FakeResponse:
        self.requests.append(request)
        response = FakeResponse(self.payloads[request.full_url])
        self.responses.append(response)
        return response


class ExplodingOpener:
    def open(self, request: Any, timeout: float | None = None) -> Any:
        raise AssertionError(f"no download expected, got {request.full_url}")


def test_fetch_verifies_git_blob_through_fake_opener(
    unicode_tmp_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.dumps(labels_doc([(1, 1_000, ON, "home")])).encode("utf-8")
    pin_set = _synthetic_pin_set(
        [_pin_entry("synthetic-labels", "soccernet_labels_v2", payload, "3" * 40)]
    )
    pin = pin_set.pins[0]
    opener = FakeOpener({pin.url: payload})
    monkeypatch.setattr(fetch, "_build_opener", lambda: opener)
    root = unicode_tmp_path / "data" / "raw"
    kwargs = {
        "root": root,
        "pin_set": pin_set,
        "repo_root": unicode_tmp_path / "repo",
        "git_ceiling": tmp_path,
    }

    results = fetch.fetch([], **kwargs)
    target = root / Path(pin.local_relpath)
    assert target.read_bytes() == payload
    assert results[0].status == "downloaded"
    assert results[0].sha256 == hashing.sha256_bytes(payload)
    assert opener.requests[0].get_header("User-agent").startswith("gbf-fetch/")
    assert opener.responses[0].closed
    record = json.loads(target.with_name(target.name + ".fetch.json").read_text(encoding="utf-8"))
    assert record["sha256"] == hashing.sha256_bytes(payload)
    assert record["git_blob_sha1"] == hashing.git_blob_sha1(payload)
    assert record["terms"] is None
    assert contracts.def_problems(record, "align", "fetch_record") == []

    # Already present and verified: no network at all.
    monkeypatch.setattr(fetch, "_build_opener", lambda: ExplodingOpener())
    assert fetch.fetch([], **kwargs)[0].status == "present"

    # A tampered download (same size, one byte changed) is rejected and never written.
    tampered = payload.replace(b"home", b"hime")
    assert len(tampered) == len(payload)
    other_root = unicode_tmp_path / "other" / "raw"
    monkeypatch.setattr(fetch, "_build_opener", lambda: FakeOpener({pin.url: tampered}))
    with pytest.raises(fetch.VerificationError) as caught:
        fetch.fetch([], **{**kwargs, "root": other_root})
    assert caught.value.problems == [
        f"synthetic-labels: git blob {hashing.git_blob_sha1(tampered)} does not match "
        f"pinned {pin.git_blob_sha1}"
    ]
    assert not other_root.exists()

    # An oversized body is cut off at size + 1 and fails on size and blob.
    monkeypatch.setattr(fetch, "_build_opener", lambda: FakeOpener({pin.url: payload + b" " * 5}))
    with pytest.raises(fetch.VerificationError) as caught:
        fetch.fetch([], **{**kwargs, "root": other_root})
    assert (
        caught.value.problems[0]
        == f"synthetic-labels: {len(payload) + 1} bytes, pinned {len(payload)}"
    )
    assert len(caught.value.problems) == 2

    # A local copy that differs from its pin is not overwritten without force.
    target.write_bytes(tampered)
    monkeypatch.setattr(fetch, "_build_opener", lambda: FakeOpener({pin.url: payload}))
    with pytest.raises(fetch.VerificationError, match="without --force"):
        fetch.fetch([], **kwargs)
    assert target.read_bytes() == tampered
    assert fetch.fetch([], force=True, **kwargs)[0].status == "downloaded"
    assert target.read_bytes() == payload
    assert not list(target.parent.glob("*.part"))


def test_fetch_refuses_statsbomb_without_terms(
    unicode_tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(fetch, "_build_opener", lambda: ExplodingOpener())
    dest = unicode_tmp_path / "raw"
    assert gbf_main.main(["fetch", "--dest", str(dest), "statsbomb-events-3754079"]) == 2
    err = capsys.readouterr().err
    terms = pins.load_pins().terms["statsbomb"]
    assert terms.agreement_url in err
    for clause in ("1.2.1", "1.2.2", "1.4", "2.2"):
        assert f"  {clause} " in err
    assert "--accept-statsbomb-terms" in err
    assert "inside" not in err  # the terms gate comes before the destination check
    assert not dest.exists()
    with pytest.raises(fetch.TermsNotAcceptedError):
        fetch.fetch(["statsbomb-lineups-3754079"], root=dest, accepted_terms=[])


def test_fetch_refuses_destinations_inside_a_repository(
    unicode_tmp_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(fetch, "_build_opener", lambda: ExplodingOpener())
    inside = paths.repo_root() / ".local" / "fetch-must-refuse-this"
    argv = ["fetch", "--dest", str(inside), "--accept-statsbomb-terms", "statsbomb-events-3754079"]
    assert gbf_main.main(argv) == 2
    assert "inside the gbf repository" in capsys.readouterr().err
    assert not inside.exists()
    # DATA_ROOT inside the repository is refused the same way (labels need no terms).
    monkeypatch.setenv("DATA_ROOT", str(paths.repo_root() / ".local" / "data-root-refused"))
    assert gbf_main.main(["fetch", "soccernet-labels-v2-3754079"]) == 2
    assert "inside the gbf repository" in capsys.readouterr().err

    other_repo = unicode_tmp_path / "other repo"
    (other_repo / ".git").mkdir(parents=True)
    with pytest.raises(fetch.DestinationError, match="inside the git work tree"):
        fetch.check_destination(
            other_repo / "data" / "raw", repo_root=unicode_tmp_path / "repo", git_ceiling=tmp_path
        )
    fake_repo = unicode_tmp_path / "repo"
    with pytest.raises(fetch.DestinationError, match="inside the gbf repository"):
        fetch.check_destination(fake_repo / "raw", repo_root=fake_repo, git_ceiling=tmp_path)
    allowed = fetch.check_destination(
        unicode_tmp_path / "data" / "raw", repo_root=fake_repo, git_ceiling=tmp_path
    )
    assert allowed.name == "raw"
    assert not allowed.exists()


def test_fetch_cli_lists_pins_and_needs_a_root(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(fetch, "_build_opener", lambda: ExplodingOpener())
    assert gbf_main.main(["fetch", "--list"]) == 0
    listing = capsys.readouterr().out
    for pin in pins.load_pins().pins:
        assert pin.name in listing
        assert pin.git_blob_sha1 in listing
    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert gbf_main.main(["fetch", "soccernet-labels-v2-3754079"]) == 2
    assert "DATA_ROOT" in capsys.readouterr().err
    assert gbf_main.main(["fetch", "--dest", "x", "no-such-pin"]) == 2
    assert "unknown pin 'no-such-pin'" in capsys.readouterr().err


# ------------------------------------------------------------------------ receipt meta


def test_code_version_and_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    version = code_version()
    assert re.fullmatch("[0-9a-f]{40}", version["git_commit"])
    assert isinstance(version["git_dirty"], bool)
    monkeypatch.setattr(_meta, "_git", lambda root, *args: None)  # git missing or failing
    assert code_version() == {"gbf_version": "0.0.1", "git_commit": "unknown", "git_dirty": True}
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    assert utc_timestamp() == "1970-01-01T00:00:00Z"
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "soon")
    with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
        utc_timestamp()


def test_relpath_under(unicode_tmp_path: Path) -> None:
    root = unicode_tmp_path / "data"
    assert relpath_under(root / "raw" / "a b.json", root) == "raw/a b.json"
    assert relpath_under(unicode_tmp_path / "elsewhere.json", root) is None
    assert relpath_under(root, root) is None
    assert relpath_under(root / "x.json", None) is None
