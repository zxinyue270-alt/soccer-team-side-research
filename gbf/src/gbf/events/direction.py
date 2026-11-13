"""Attack-direction checks for tracking output (pure functions, no I/O).

Coordinates are pitch metres along the length axis x, origin at the centre spot,
as in the SoccerNet world frame; ``pitch_zup = (x, -y, -z)`` keeps x, so the checks
hold in either frame. For other frames pass ``halfway_x`` (60 for StatsBomb's
120 x 80 yards). The goals sit on the x axis at equal distance on both sides of
the halfway line.

A team *attacks +x* in a half when it shoots at the goal on the +x side; teams swap
ends at half time. The checks:

- ``shots_in_attacking_half``: a team's shots should mostly be taken in the half it
  attacks (x beyond the halfway line on its attacking side).
- ``keeper_near_own_goal``: at a shot, the defending goalkeeper (e.g. in a freeze
  frame) should be nearer the goal he defends than the other goal. With the goals
  at ``halfway_x -/+ L/2`` on the x axis, ``|p - own| < |p - other|`` holds exactly
  when p lies on the defended goal's side of the halfway line, whatever y is, so
  only x is needed.
- ``direction_problems``: in each half the two teams attack opposite ways, and each
  team swaps at half time.

Shares are compared exactly (``n_on_side / n_total >= min_share`` with fractions),
and thresholds have no defaults: they belong to the pre-registered gates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction


@dataclass(frozen=True)
class SideCount:
    """How many positions lie on the expected side of the halfway line.

    ``n_on_line`` positions sit exactly on the line and count for neither side.
    """

    n_total: int
    n_on_side: int
    n_on_line: int

    def meets(self, min_share: Fraction) -> bool:
        """``n_on_side / n_total >= min_share``, exactly; false when there is no data."""
        if self.n_total == 0:
            return False
        share = Fraction(min_share)
        return self.n_on_side * share.denominator >= share.numerator * self.n_total


def _finite_xs(values: Iterable[float], what: str) -> list[float]:
    xs = list(values)
    bad = [
        index
        for index, x in enumerate(xs)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
    ]
    if bad:
        raise ValueError(f"{what}: non-finite or non-numeric x at indices {bad}")
    return [float(x) for x in xs]


def _count_side(xs: list[float], positive_side: bool, halfway_x: float) -> SideCount:
    on_line = sum(1 for x in xs if x == halfway_x)
    if positive_side:
        on_side = sum(1 for x in xs if x > halfway_x)
    else:
        on_side = sum(1 for x in xs if x < halfway_x)
    return SideCount(n_total=len(xs), n_on_side=on_side, n_on_line=on_line)


def shots_in_attacking_half(
    shot_x: Iterable[float], *, attacks_positive_x: bool, halfway_x: float = 0.0
) -> SideCount:
    """Count a team's shots (x of the shooter or the ball, metres) in its attacking half."""
    return _count_side(_finite_xs(shot_x, "shot_x"), attacks_positive_x, halfway_x)


def keeper_near_own_goal(
    keeper_x: Iterable[float], *, defends_positive_x: bool, halfway_x: float = 0.0
) -> SideCount:
    """Count goalkeeper positions nearer the goal he defends (+x goal if ``defends_positive_x``)."""
    return _count_side(_finite_xs(keeper_x, "keeper_x"), defends_positive_x, halfway_x)


def direction_problems(directions: Mapping[tuple[str, int], bool]) -> list[str]:
    """Consistency of ``{(team, half): attacks_positive_x}`` for halves 1 and 2.

    Problems: two teams attacking the same way in one half, more than two teams in
    a half, or a team that does not swap ends between halves 1 and 2.
    """
    problems = []
    by_half: dict[int, dict[str, bool]] = {}
    for (team, half), positive in sorted(directions.items()):
        by_half.setdefault(half, {})[team] = positive
    for half, teams in sorted(by_half.items()):
        if len(teams) > 2:
            problems.append(f"half {half}: more than two teams: {sorted(teams)}")
        elif len(teams) == 2 and len(set(teams.values())) == 1:
            way = "+x" if next(iter(teams.values())) else "-x"
            problems.append(f"half {half}: both teams attack {way}")
    teams_seen = {team for team, _ in directions}
    for team in sorted(teams_seen):
        first, second = directions.get((team, 1)), directions.get((team, 2))
        if first is not None and second is not None and first == second:
            problems.append(f"{team}: attacks the same way in both halves (no swap at half time)")
    return problems


@dataclass(frozen=True)
class TeamHalf:
    """Observations of one team in one half, from tracking output.

    ``shot_x``: x of each of the team's shots. ``opposing_keeper_x``: x of the
    defending goalkeeper at each of those shots (he defends the goal the team
    attacks). ``attacks_positive_x`` is the direction under test.
    """

    team: str
    half: int
    attacks_positive_x: bool
    shot_x: Sequence[float] = field(default_factory=tuple)
    opposing_keeper_x: Sequence[float] = field(default_factory=tuple)


@dataclass(frozen=True)
class TeamHalfCheck:
    team: str
    half: int
    attacks_positive_x: bool
    shots: SideCount
    keepers: SideCount
    shots_ok: bool | None
    keepers_ok: bool | None


@dataclass(frozen=True)
class DirectionReport:
    """Outcome of :func:`check_attack_directions`.

    ``passed`` requires no problem and every share judged (``None`` means fewer
    than ``min_count`` observations, which is reported in ``not_judged``).
    """

    checks: tuple[TeamHalfCheck, ...]
    problems: tuple[str, ...]
    not_judged: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.problems and not self.not_judged


def check_attack_directions(
    observations: Sequence[TeamHalf],
    *,
    min_shot_share: Fraction,
    min_keeper_share: Fraction,
    min_count: int,
    halfway_x: float = 0.0,
) -> DirectionReport:
    """Assert the attack directions of every team-half against tracking output.

    A share is judged only with at least ``min_count`` observations; otherwise it is
    listed in ``not_judged``. Thresholds are exact fractions, e.g. ``Fraction(4, 5)``.
    """
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValueError(f"min_count must be a positive integer: {min_count!r}")
    for name, share in (("min_shot_share", min_shot_share), ("min_keeper_share", min_keeper_share)):
        if not isinstance(share, Fraction) or not 0 <= share <= 1:
            raise ValueError(f"{name} must be a Fraction in [0, 1]: {share!r}")
    keys = [(item.team, item.half) for item in observations]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate team-half observations: {duplicates}")
    problems = direction_problems(
        {(item.team, item.half): item.attacks_positive_x for item in observations}
    )
    not_judged = []
    checks = []
    for item in observations:
        shots = shots_in_attacking_half(
            item.shot_x, attacks_positive_x=item.attacks_positive_x, halfway_x=halfway_x
        )
        keepers = keeper_near_own_goal(
            item.opposing_keeper_x, defends_positive_x=item.attacks_positive_x, halfway_x=halfway_x
        )
        way = "+x" if item.attacks_positive_x else "-x"
        shots_ok = keepers_ok = None
        if shots.n_total >= min_count:
            shots_ok = shots.meets(min_shot_share)
            if not shots_ok:
                problems.append(
                    f"{item.team} half {item.half}: only {shots.n_on_side}/{shots.n_total} "
                    f"shots in the attacking half for direction {way}"
                )
        else:
            not_judged.append(f"{item.team} half {item.half}: {shots.n_total} shot(s)")
        if keepers.n_total >= min_count:
            keepers_ok = keepers.meets(min_keeper_share)
            if not keepers_ok:
                problems.append(
                    f"{item.team} half {item.half}: only {keepers.n_on_side}/{keepers.n_total} "
                    f"keeper positions nearer the goal attacked in direction {way}"
                )
        else:
            not_judged.append(f"{item.team} half {item.half}: {keepers.n_total} keeper position(s)")
        checks.append(
            TeamHalfCheck(
                item.team, item.half, item.attacks_positive_x, shots, keepers, shots_ok, keepers_ok
            )
        )
    return DirectionReport(tuple(checks), tuple(problems), tuple(not_judged))
