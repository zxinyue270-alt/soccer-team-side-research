"""Provider files: StatsBomb events (through kloppy) and SoccerNet labels.

Time conventions (all integers are milliseconds):

- StatsBomb ``timestamp`` is ``HH:MM:SS.fff`` and restarts at ``00:00:00.000`` in
  every period. kloppy 3.19 keeps exactly this value in ``Event.timestamp``
  ("relative to the period kick-off"); only ``Period.start_timestamp`` is
  cumulative over the match, so it is never added here. Every timestamp taken from
  kloppy is cross-checked against an integer parse of the raw string.
- SoccerNet ``position`` is milliseconds from the first frame of the half video,
  and ``gameTime`` is ``"<half> - MM:SS"`` of the same instant.

StatsBomb data is parsed from bytes that the caller has already read (and hashed),
so the hashed bytes and the parsed bytes are the same object.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

_SB_TIMESTAMP = re.compile(r"^([0-9]{2}):([0-9]{2}):([0-9]{2})[.]([0-9]{3})$")
_GAME_TIME = re.compile(r"^([12]) - ([0-9]{1,3}):([0-9]{2})$")


class ProviderDataError(ValueError):
    """A provider file is not what the parser expects. ``problems`` lists every issue."""

    def __init__(self, source: str, problems: list[str]):
        self.source = source
        self.problems = problems
        joined = "\n  - ".join(problems)
        super().__init__(f"{source}: {len(problems)} problem(s)\n  - {joined}")


# --------------------------------------------------------------------------- StatsBomb


def parse_statsbomb_timestamp_ms(text: str) -> int:
    """``"HH:MM:SS.fff"`` (period-relative) to integer milliseconds.

    Raises ``ValueError`` for any other shape or for minutes/seconds >= 60.
    """
    match = _SB_TIMESTAMP.match(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(f"not a StatsBomb timestamp (HH:MM:SS.fff): {text!r}")
    hours, minutes, seconds, millis = (int(group) for group in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"minutes and seconds must be below 60: {text!r}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def timedelta_to_ms(value: timedelta) -> int:
    """Exact integer milliseconds of a timedelta; raises if it has sub-millisecond parts."""
    micros = (value.days * 86_400 + value.seconds) * 1_000_000 + value.microseconds
    if micros % 1000:
        raise ValueError(f"timedelta {value!r} is not a whole number of milliseconds")
    return micros // 1000


@dataclass(frozen=True)
class StatsBombEvent:
    """One StatsBomb event used for alignment.

    ``timestamp`` is the raw period-relative string; ``timestamp_ms`` the same
    instant in milliseconds. ``team_side`` is ``home`` or ``away`` as kloppy
    resolves it from the lineups. ``outcome`` is the raw shot outcome name
    (``None`` for kick-offs).
    """

    event_id: str
    period: int
    timestamp: str
    timestamp_ms: int
    team: str
    team_side: str
    outcome: str | None


@dataclass(frozen=True)
class StatsBombMatch:
    """Shots and kick-off passes of one match, each sorted by (period, time, id)."""

    shots: tuple[StatsBombEvent, ...]
    kickoffs: tuple[StatsBombEvent, ...]
    periods: tuple[int, ...]
    loader: str


def _sort_key(event: StatsBombEvent) -> tuple[int, int, str]:
    return (event.period, event.timestamp_ms, event.event_id)


def load_statsbomb(events: bytes, lineups: bytes) -> StatsBombMatch:
    """Load shots and kick-off passes from StatsBomb event and lineup JSON bytes.

    Parsing goes through ``kloppy.statsbomb.load`` (no network: the inputs are
    bytes). For every event kept, kloppy's ``timestamp`` must equal the integer
    parse of the raw ``timestamp`` string and ``period`` must equal the raw
    ``period``; any disagreement raises :class:`ProviderDataError` listing all of
    them.

    Kick-offs are passes that kloppy tags with ``SetPieceType.KICK_OFF``
    (StatsBomb pass type "Kick Off"), including those after goals.
    """
    import kloppy
    from kloppy import statsbomb
    from kloppy.domain import EventType, Ground, SetPieceQualifier, SetPieceType

    try:
        dataset = statsbomb.load(event_data=events, lineup_data=lineups, coordinates="statsbomb")
    except Exception as error:  # kloppy raises many types (KeyError, DeserializationError, ...)
        raise ProviderDataError(
            "StatsBomb events", [f"kloppy cannot parse the files: {type(error).__name__}: {error}"]
        ) from error
    problems: list[str] = []
    shots: list[StatsBombEvent] = []
    kickoffs: list[StatsBombEvent] = []
    for event in dataset.events:
        is_shot = event.event_type == EventType.SHOT
        is_kickoff = (
            event.event_type == EventType.PASS
            and event.get_qualifier_value(SetPieceQualifier) == SetPieceType.KICK_OFF
        )
        if not (is_shot or is_kickoff):
            continue
        raw = event.raw_event
        where = f"event {event.event_id}"
        try:
            raw_ms = parse_statsbomb_timestamp_ms(raw.get("timestamp"))
        except ValueError as error:
            problems.append(f"{where}: {error}")
            continue
        try:
            kloppy_ms = timedelta_to_ms(event.timestamp)
        except ValueError as error:
            problems.append(f"{where}: kloppy timestamp: {error}")
            continue
        if kloppy_ms != raw_ms:
            problems.append(
                f"{where}: kloppy timestamp {kloppy_ms} ms != raw {raw['timestamp']!r} "
                f"({raw_ms} ms); kloppy is not period-relative here"
            )
        if event.period.id != raw.get("period"):
            problems.append(f"{where}: kloppy period {event.period.id} != raw {raw.get('period')}")
        side = "home" if event.team.ground == Ground.HOME else "away"
        outcome = None
        if is_shot:
            outcome = (raw.get("shot", {}).get("outcome") or {}).get("name")
        record = StatsBombEvent(
            event_id=str(event.event_id),
            period=int(event.period.id),
            timestamp=raw["timestamp"],
            timestamp_ms=raw_ms,
            team=str(event.team.name),
            team_side=side,
            outcome=outcome,
        )
        (shots if is_shot else kickoffs).append(record)
    if problems:
        raise ProviderDataError("StatsBomb events", problems)
    periods = tuple(sorted({int(p.id) for p in dataset.metadata.periods}))
    return StatsBombMatch(
        shots=tuple(sorted(shots, key=_sort_key)),
        kickoffs=tuple(sorted(kickoffs, key=_sort_key)),
        periods=periods,
        loader=f"kloppy {kloppy.__version__} statsbomb.load from local bytes",
    )


# --------------------------------------------------------------------------- SoccerNet


@dataclass(frozen=True)
class Label:
    """One Labels-v2 annotation. ``index`` is its position in the annotations array."""

    index: int
    half: int
    position_ms: int
    game_time: str
    label: str
    team: str | None
    visibility: str | None


@dataclass(frozen=True)
class CameraChange:
    """One Labels-cameras annotation: from ``position_ms`` on, ``label`` is on air.

    ``replay`` is ``True`` for replay segments, ``False`` for real-time ones and
    ``None`` when the annotation does not say.
    """

    index: int
    half: int
    position_ms: int
    game_time: str
    label: str
    replay: bool | None
    change_type: str | None


def _position_ms(value: Any) -> int | None:
    """SoccerNet stores ``position`` as a decimal string of milliseconds."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    return None


def _annotation_basics(
    index: int, raw: Any, problems: list[str]
) -> tuple[int, int, str, str] | None:
    """Half, position (ms), gameTime and label of one annotation; problems are appended."""
    where = f"annotations[{index}]"
    if not isinstance(raw, Mapping):
        problems.append(f"{where} must be an object")
        return None
    ok = True
    game_time = raw.get("gameTime")
    match = _GAME_TIME.match(game_time) if isinstance(game_time, str) else None
    if match is None:
        problems.append(f"{where}.gameTime must look like '1 - 12:34': {game_time!r}")
        ok = False
    elif int(match.group(3)) >= 60:
        problems.append(f"{where}.gameTime seconds must be below 60: {game_time!r}")
        ok = False
    position = _position_ms(raw.get("position"))
    if position is None:
        problems.append(
            f"{where}.position must be non-negative integer milliseconds: {raw.get('position')!r}"
        )
        ok = False
    label = raw.get("label")
    if not isinstance(label, str) or not label:
        problems.append(f"{where}.label must be a non-empty string: {label!r}")
        ok = False
    if not ok or match is None or position is None:
        return None
    return int(match.group(1)), position, game_time, label


def _annotations(source: str, obj: Any) -> Sequence[Any]:
    if isinstance(obj, (bytes, bytearray)):
        try:
            obj = json.loads(bytes(obj).decode("utf-8"))
        except ValueError as error:  # UnicodeDecodeError and JSONDecodeError
            raise ProviderDataError(source, [f"not UTF-8 JSON: {error}"]) from error
    if not isinstance(obj, Mapping) or not isinstance(obj.get("annotations"), list):
        raise ProviderDataError(source, ["top level must be an object with an 'annotations' list"])
    return obj["annotations"]


def parse_labels_v2(obj: Mapping[str, Any] | bytes) -> tuple[Label, ...]:
    """Parse a ``Labels-v2.json`` document (parsed JSON or raw bytes).

    Labels are returned in file order. Raises :class:`ProviderDataError` listing
    every malformed annotation.
    """
    source = "Labels-v2"
    problems: list[str] = []
    labels = []
    for index, raw in enumerate(_annotations(source, obj)):
        basics = _annotation_basics(index, raw, problems)
        if basics is None:
            continue
        half, position, game_time, label = basics
        team = raw.get("team")
        visibility = raw.get("visibility")
        labels.append(
            Label(
                index=index,
                half=half,
                position_ms=position,
                game_time=game_time,
                label=label,
                team=team if isinstance(team, str) else None,
                visibility=visibility if isinstance(visibility, str) else None,
            )
        )
    if problems:
        raise ProviderDataError(source, problems)
    return tuple(labels)


def parse_labels_cameras(obj: Mapping[str, Any] | bytes) -> tuple[CameraChange, ...]:
    """Parse a ``Labels-cameras.json`` document (parsed JSON or raw bytes).

    ``replay`` is read from the annotation's ``replay`` field: ``"replay"`` means a
    replay, ``"real-time"`` live play; anything else is a problem. Raises
    :class:`ProviderDataError` listing every malformed annotation.
    """
    source = "Labels-cameras"
    problems: list[str] = []
    changes = []
    for index, raw in enumerate(_annotations(source, obj)):
        basics = _annotation_basics(index, raw, problems)
        if basics is None:
            continue
        half, position, game_time, label = basics
        replay_raw = raw.get("replay")
        replay: bool | None
        if replay_raw is None:
            replay = None
        elif replay_raw in ("replay", "real-time"):
            replay = replay_raw == "replay"
        else:
            problems.append(
                f"annotations[{index}].replay must be 'replay' or 'real-time': {replay_raw!r}"
            )
            continue
        change_type = raw.get("change_type")
        changes.append(
            CameraChange(
                index=index,
                half=half,
                position_ms=position,
                game_time=game_time,
                label=label,
                replay=replay,
                change_type=change_type if isinstance(change_type, str) else None,
            )
        )
    if problems:
        raise ProviderDataError(source, problems)
    return tuple(changes)


def game_time_disagreements(labels: Sequence[Label | CameraChange]) -> list[str]:
    """Annotations whose ``gameTime`` is more than one second away from ``position``.

    A disagreement usually means the position is not in milliseconds; it is
    reported as a warning, never silently fixed.
    """
    found = []
    for item in labels:
        match = _GAME_TIME.match(item.game_time)
        if match is None:  # pragma: no cover - parsers only build valid game times
            continue
        game_s = int(match.group(2)) * 60 + int(match.group(3))
        if abs(item.position_ms // 1000 - game_s) > 1:
            found.append(
                f"annotation {item.index}: gameTime {item.game_time!r} disagrees with "
                f"position {item.position_ms} ms"
            )
    return found
