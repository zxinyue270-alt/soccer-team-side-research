"""Check of the plugin's role labels against the one referee the laws allow on the pitch.

The tracking plugin labels each person player, goalkeeper or referee from re-identification
features, and its role classifier goes by the look of the kit: when one team's kit looks like a
referee's, it can label that whole team referee and the referee, dressed otherwise, player (two
dev clips of one match, a team in yellow shirts and black shorts). The plugin forms its teams
from the people it labels player only, so with such roles its team labels are wrong too: one
team has none and the referee plays for the other, and a nearest-opponent value skips the real
nearest opponent.

The laws allow one referee on the field of play; the assistant referees run along the
touchlines and the fourth official stays off the pitch. :func:`check_roles` counts, on every
frame with a pitch-usable row, the rows labelled referee whose pitch point is usable and lies at
least ``inner_margin_m`` inside the touchlines and the goal lines. When more than
``max_crowded_share`` of those frames hold two or more of them, the labels are implausible:
every team label is cleared (the table's own missing value), so nothing that needs the teams is
computed from them. Roles are left as they are: the count says that some are wrong, not which
ones or who the referee is. With fewer than ``min_frames`` such frames the check gives no
verdict and changes nothing.

It is pure: the caller passes a track table with gbf.sngs_import's columns (pitch positions in
metres, NaN unless pitch-usable) and stores the results.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from gbf.sngs_import import PITCH_HALF_LENGTH_M, PITCH_HALF_WIDTH_M

COLUMNS: tuple[str, ...] = ("frame", "role", "team", "pitch_usable", "pitch_x", "pitch_y")
"""Columns of the track table the check reads."""

VERDICTS: tuple[str, ...] = ("plausible", "implausible", "not_judged")


class RoleCheckError(ValueError):
    """Invalid input to gbf.role_check; ``problems`` lists everything found."""

    def __init__(self, where: str, problems: Sequence[str]) -> None:
        self.where = where
        self.problems = list(problems)
        super().__init__(f"{where}: " + "; ".join(self.problems))


def _number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float | np.integer | np.floating)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class RoleCheckParams:
    """Thresholds of the role check.

    Attributes:
        inner_margin_m: a person labelled referee counts when their pitch point lies at least
            this far inside the touchlines and the goal lines: the assistant referees run
            along the touchlines, and the calibration can put a point a metre off.
        max_crowded_share: the labels are implausible when more than this share of the frames
            with a pitch-usable row hold two or more counted referees.
        min_frames: with fewer frames with a pitch-usable row the check gives no verdict.
    """

    inner_margin_m: float = 2.0
    max_crowded_share: float = 0.5
    min_frames: int = 25

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not _number(self.inner_margin_m) or self.inner_margin_m < 0:
            problems.append(
                f"inner_margin_m must be a finite number >= 0, got {self.inner_margin_m!r}"
            )
        if not _number(self.max_crowded_share) or not 0 <= self.max_crowded_share <= 1:
            problems.append(
                f"max_crowded_share must be a number in [0, 1], got {self.max_crowded_share!r}"
            )
        if (
            isinstance(self.min_frames, bool)
            or not isinstance(self.min_frames, int)
            or self.min_frames < 1
        ):
            problems.append(f"min_frames must be an int >= 1, got {self.min_frames!r}")
        if problems:
            raise RoleCheckError("RoleCheckParams", problems)

    def as_dict(self) -> dict[str, Any]:
        """JSON form for reports and receipts."""
        return asdict(self)


DEFAULT_PARAMS = RoleCheckParams()


def _missing_value(column: pd.Series) -> Any:
    """The missing value ``column`` already uses: its dtype's, else that of a missing entry."""
    na_value = getattr(column.dtype, "na_value", None)
    if na_value is not None:
        return na_value
    missing = column[column.isna()]
    return missing.iloc[0] if len(missing) else None


def check_roles(
    tracks: pd.DataFrame, params: RoleCheckParams | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check the role labels against the one referee on the pitch: ``(tracks, report)``.

    Returns ``tracks`` itself when the labels are plausible or get no verdict, else a copy in
    which every team label is missing; and a JSON-ready report: ``params``, ``verdict``
    ("plausible", "implausible" or "not_judged"), ``frames`` (frames with a pitch-usable
    row), ``crowded_frames`` (those with two or more counted referees), ``crowded_share``
    (their share, rounded to 3 places; None without frames), ``max_referees_inside`` and
    ``rows_cleared`` (rows whose team label was cleared).
    """
    params = DEFAULT_PARAMS if params is None else params
    absent = [name for name in COLUMNS if name not in tracks.columns]
    if absent:
        raise RoleCheckError("check_roles", [f"tracks lacks the columns {absent}"])
    frame = tracks["frame"].to_numpy(dtype=np.int64)
    x = tracks["pitch_x"].to_numpy(dtype=np.float64)
    y = tracks["pitch_y"].to_numpy(dtype=np.float64)
    usable = tracks["pitch_usable"].to_numpy(dtype=bool) & np.isfinite(x) & np.isfinite(y)
    referee = (tracks["role"] == "referee").to_numpy(dtype=bool, na_value=False)
    margin = params.inner_margin_m
    inside = (
        usable
        & referee
        & (np.abs(x) <= PITCH_HALF_LENGTH_M - margin)
        & (np.abs(y) <= PITCH_HALF_WIDTH_M - margin)
    )
    frames = np.unique(frame[usable]).size
    _, per_frame = np.unique(frame[inside], return_counts=True)
    crowded = int((per_frame >= 2).sum())
    share = crowded / frames if frames else None
    if frames < params.min_frames or share is None:
        verdict = "not_judged"
    elif share > params.max_crowded_share:
        verdict = "implausible"
    else:
        verdict = "plausible"
    out = tracks
    labelled = tracks["team"].notna().to_numpy(dtype=bool)
    if verdict == "implausible" and labelled.any():
        out = tracks.copy()
        out.loc[labelled, "team"] = _missing_value(out["team"])
    report = {
        "params": params.as_dict(),
        "verdict": verdict,
        "frames": int(frames),
        "crowded_frames": crowded,
        "crowded_share": None if share is None else round(share, 3),
        "max_referees_inside": int(per_frame.max()) if per_frame.size else 0,
        "rows_cleared": int(labelled.sum()) if verdict == "implausible" else 0,
    }
    return out, report
