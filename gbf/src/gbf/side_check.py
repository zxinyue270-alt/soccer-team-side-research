"""Check of the plugin's left/right team names against where the two teams stand.

The tracking plugin names its two team clusters by one comparison over the whole clip: the
cluster whose detections have the larger mean pitch x is "right" (the team whose goal is on the
right, at x > 0). gbf takes the names as given and its convention follows them: team "left"
defends x < 0 and attacks +x (gbf.highlights, gbf.measure). When both teams stand in one half,
the two means can lie a metre apart and the names follow chance; named the wrong way round, a
highlight title names the wrong team, the attack cues point at the wrong goal and the attackers
in the box are counted in the other box (dev clips of the gates.v1 evaluation, 2026-10-31).

:func:`check_sides` recomputes the comparison from gbf's own rows (outfield players with a
team label and a pitch-usable position: the difference of the right team's mean x and the left
team's, every row weighing alike as in the plugin) and its uncertainty by resampling whole
tracks with replacement within each team (rows of a track are not independent). The names are
borne out (``consistent``) when the whole ``level`` percentile interval lies above 0;
``uncertain`` when it straddles 0; ``contradicted`` when it lies below 0 (gbf's calibration
puts the teams the other way round); ``not_judged`` when a team has fewer than ``min_tracks``
tracks; ``no_teams`` when no row has a team label (gbf.role_check clears them all when the
roles are implausible). The direction each team attacks counts as known for ``consistent`` and
``no_teams`` only; the names are never swapped.

It is pure: the caller passes a track table with gbf.sngs_import's columns (pitch positions in
metres, NaN unless pitch-usable) and stores the report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

COLUMNS: tuple[str, ...] = ("track_id", "role", "team", "pitch_usable", "pitch_x")
"""Columns of the track table the check reads."""

VERDICTS: tuple[str, ...] = ("consistent", "uncertain", "contradicted", "not_judged", "no_teams")
KNOWN_VERDICTS: frozenset[str] = frozenset({"consistent", "no_teams"})
"""Verdicts under which the direction each team attacks counts as known."""


class SideCheckError(ValueError):
    """Invalid input to gbf.side_check; ``problems`` lists everything found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        super().__init__(f"{where}: " + "; ".join(self.problems))


@dataclass(frozen=True)
class SideCheckParams:
    """Settings of the side check (conventional values, as in the dev evaluation's bootstrap).

    Attributes:
        n_boot: resamples of the tracks.
        seed: seed of ``numpy.random.default_rng``.
        level: central share of the resampled differences forming the interval.
        min_tracks: a team with fewer tracks gives no verdict (one track cannot be resampled).
    """

    n_boot: int = 2000
    seed: int = 0
    level: float = 0.95
    min_tracks: int = 2

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name, least in (("n_boot", 1), ("min_tracks", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                problems.append(f"{name} must be an int >= {least}, got {value!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            problems.append(f"seed must be an int, got {self.seed!r}")
        level = self.level
        if isinstance(level, bool) or not isinstance(level, int | float) or not 0 < level < 1:
            problems.append(f"level must be a number in (0, 1), got {level!r}")
        if problems:
            raise SideCheckError("SideCheckParams", problems)

    def as_dict(self) -> dict[str, Any]:
        """JSON form for reports and receipts."""
        return asdict(self)


DEFAULT_PARAMS = SideCheckParams()


def _per_track(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per track of ``rows``: the sum of its x and its number of rows."""
    grouped = rows.groupby("track_id", sort=True)["pitch_x"]
    return grouped.sum().to_numpy(dtype=np.float64), grouped.size().to_numpy(dtype=np.float64)


def _difference(sums: dict[str, np.ndarray], counts: dict[str, np.ndarray]) -> float:
    """The right team's mean x minus the left team's, every row weighing alike."""
    right = sums["right"].sum() / counts["right"].sum()
    return float(right - sums["left"].sum() / counts["left"].sum())


def check_sides(tracks: pd.DataFrame, params: SideCheckParams | None = None) -> dict[str, Any]:
    """Check the left/right team names against where the teams stand: a JSON-ready report with
    ``params``, ``verdict`` (:data:`VERDICTS`), ``attack_direction_known``, ``difference_m``
    (right team's mean x minus the left team's, rounded to 3 places; None without both teams),
    ``interval_m`` ([low, high] at ``level``, rounded; None without a verdict from it), and
    ``tracks`` and ``rows`` counted per team."""
    params = DEFAULT_PARAMS if params is None else params
    absent = [name for name in COLUMNS if name not in tracks.columns]
    if absent:
        raise SideCheckError("check_sides", [f"tracks lacks the columns {absent}"])
    x = tracks["pitch_x"].to_numpy(dtype=np.float64)
    keep = (
        (tracks["role"] == "player").to_numpy(dtype=bool, na_value=False)
        & tracks["pitch_usable"].to_numpy(dtype=bool)
        & np.isfinite(x)
    )
    team = tracks["team"]
    sums, counts = {}, {}
    for name in ("left", "right"):
        rows = tracks[keep & (team == name).to_numpy(dtype=bool, na_value=False)]
        sums[name], counts[name] = _per_track(rows)
    report: dict[str, Any] = {
        "params": params.as_dict(),
        "verdict": None,
        "attack_direction_known": None,
        "difference_m": None,
        "interval_m": None,
        "tracks": {name: int(len(counts[name])) for name in ("left", "right")},
        "rows": {name: int(counts[name].sum()) for name in ("left", "right")},
    }
    if not tracks["team"].isin(["left", "right"]).any():
        verdict = "no_teams"
    elif min(len(counts["left"]), len(counts["right"])) < params.min_tracks:
        verdict = "not_judged"
        if len(counts["left"]) and len(counts["right"]):
            report["difference_m"] = round(_difference(sums, counts), 3)
    else:
        rng = np.random.default_rng(params.seed)
        means = {}
        for name in ("left", "right"):
            k = len(counts[name])
            pick = rng.integers(0, k, size=(params.n_boot, k))
            means[name] = sums[name][pick].sum(axis=1) / counts[name][pick].sum(axis=1)
        resampled = means["right"] - means["left"]
        tail = (1.0 - params.level) / 2.0
        low, high = (float(v) for v in np.quantile(resampled, [tail, 1.0 - tail]))
        verdict = "consistent" if low > 0 else "contradicted" if high < 0 else "uncertain"
        report["difference_m"] = round(_difference(sums, counts), 3)
        report["interval_m"] = [round(low, 3), round(high, 3)]
    report["verdict"] = verdict
    report["attack_direction_known"] = verdict in KNOWN_VERDICTS
    return report
