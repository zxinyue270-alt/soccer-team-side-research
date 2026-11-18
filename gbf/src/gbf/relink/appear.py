"""Appearance prototypes of the relink step: one mean embedding per sub-piece of a clip.

A sub-piece is a stretch of rows that the relink step treats as one person (a piece of the
track table, or a part of one after a relink cut). This module describes every sub-piece by
the mean of its unit appearance embeddings and prepares what the linker needs to compare
two of them:

- the prototype: the mean over the sub-piece's clean rows (``clean`` of the inputs: a box
  away from the frame border, tall enough, overlapping no other box, with an embedding);
  with fewer than ``min_clean`` clean rows, the mean over all its rows that have an
  embedding (the sub-piece is then called low);
- the team means: per team, the mean embedding over the prototype rows of the sub-pieces
  labelled player of that team that are not low, used from ``team_mean_min_rows`` rows on.
  Players in the same kit share most of the embedding; what is left after subtracting the
  team mean tells them apart. A sub-piece without a team label takes the team whose mean
  is nearer (the appearance team);
- the whitening matrix ``T`` (within-class covariance normalisation): the covariance of
  the prototype rows around their own sub-piece's mean, shrunk by ``wccn_lambda`` times
  its mean eigenvalue and inverted to the power one half, so that directions in which one
  person varies from frame to frame count less.

The linker's similarity of two candidates i and j is then the cosine of
``unit(T @ (msum[i] / nrow[i] - centre[i]))`` and the same for j; ``msum`` and ``nrow`` are
kept apart so that the prototypes of joined sub-pieces can be pooled.

This is a transliteration of the appearance part of the method ``fuse_r1`` of the
re-identification lab (reid_lab, outside the repository), where the parameter values were
chosen. The linker's decisions are thresholds on these floats, so the arithmetic keeps the
lab's operations, operands and orders: sub-pieces in ascending id, rows of a sub-piece in
ascending frame, the same concatenations before every mean and matrix product. Nothing
here reads an annotation, a file or another gbf module, and nothing is cached: the result
is a function of the arguments.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AppearParams:
    """Parameters of :func:`build` (the lab's values)."""

    #: clean rows a sub-piece needs for a prototype of its clean rows only
    min_clean: int = 5
    #: prototype rows a team needs for a team mean
    team_mean_min_rows: int = 50
    #: shrinkage of the within-piece covariance: share of its mean eigenvalue that is added
    wccn_lambda: float = 0.3
    #: prototype rows a sub-piece needs to enter the within-piece covariance
    wccn_min_rows: int = 10

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name, least in (("min_clean", 0), ("team_mean_min_rows", 1), ("wccn_min_rows", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                problems.append(f"{name} must be an integer >= {least}, got {value!r}")
        lam = self.wccn_lambda
        if isinstance(lam, bool) or not isinstance(lam, int | float) or not 0 < lam < float("inf"):
            problems.append(f"wccn_lambda must be a positive number, got {lam!r}")
        if problems:
            raise ValueError("AppearParams: " + "; ".join(problems))

    def as_dict(self) -> dict[str, Any]:
        """JSON form for reports and receipts."""
        return asdict(self)


DEFAULT_PARAMS = AppearParams()


@dataclass(frozen=True, eq=False)
class Appearance:
    """The link candidates of one clip: the sub-pieces that are not referees and have a
    prototype, in ascending sub-piece id. n candidates, D embedding dimensions.

    Attributes:
        ids: (n,) int64 sub-piece ids, ascending.
        rows: (n,) int64 boxes of each sub-piece (all its rows, with or without embedding).
        first, last: (n,) int64 first and last frame of each.
        frames: the frames of each, as frozensets.
        team: (n,) object, the appearance team "left" or "right": the sub-piece's label,
            else the team with the nearer mean; "" when neither exists.
        msum: (n, D) float64, the prototype times the rows it was averaged over.
        nrow: (n,) float64, the rows the prototype was averaged over.
        centre: (n, D) float64, the mean of the appearance team; the global centre when
            that team has no mean.
        T: (D, D) float64 whitening matrix; the identity when no sub-piece is long enough.

    Equality is identity; compare the arrays explicitly.
    """

    ids: np.ndarray
    rows: np.ndarray
    first: np.ndarray
    last: np.ndarray
    frames: tuple[frozenset[int], ...]
    team: np.ndarray
    msum: np.ndarray
    nrow: np.ndarray
    centre: np.ndarray
    T: np.ndarray


def unit(v: np.ndarray) -> np.ndarray:
    """``v`` divided by its length along the last axis (a zero vector stays zero)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _majority(labels: Iterable[Any]) -> str | None:
    """The most frequent string among ``labels``, the smallest string among equally
    frequent ones (what ``pandas.Series.mode().iloc[0]`` gives); None without a string."""
    counts = Counter(v for v in labels if isinstance(v, str))
    if not counts:
        return None
    top = max(counts.values())
    return min(label for label, count in counts.items() if count == top)


def _stack(vectors: list[np.ndarray], dim: int) -> np.ndarray:
    return np.stack(vectors) if vectors else np.zeros((0, dim))


def _empty(dim: int) -> Appearance:
    return Appearance(
        ids=np.zeros(0, np.int64),
        rows=np.zeros(0, np.int64),
        first=np.zeros(0, np.int64),
        last=np.zeros(0, np.int64),
        frames=(),
        team=np.array([], dtype=object),
        msum=np.zeros((0, dim)),
        nrow=np.zeros(0),
        centre=np.zeros((0, dim)),
        T=np.eye(dim),
    )


def build(inp: Any, sub: np.ndarray, params: AppearParams = DEFAULT_PARAMS) -> Appearance:
    """The appearance candidates of one single-shot clip.

    Args:
        inp: the clip's rows, read by attribute (gbf.relink.inputs.ClipInputs or any object
            with these arrays of n rows): ``frame`` (int), ``emb`` ((n, D) float64 unit rows,
            an all-NaN row = no embedding), ``clean`` (bool), ``role`` and ``team`` (object:
            str or None).
        sub: (n,) int sub-piece id of every row; a negative id = the row belongs to none.
        params: see :class:`AppearParams`.

    Without any embedding the result has no candidate and ``T`` is the identity. The arrays
    of the result are new ones (none is a view of an input) and are not write-protected: a
    caller that pools prototypes works on copies.

    Raises:
        ValueError: when the arrays do not describe the same n rows.
    """
    frame = np.asarray(inp.frame)
    emb = np.asarray(inp.emb, dtype=np.float64)
    clean = np.asarray(inp.clean, dtype=bool)
    role = np.asarray(inp.role, dtype=object)
    team = np.asarray(inp.team, dtype=object)
    sub = np.asarray(sub)
    problems: list[str] = []
    n = frame.size
    named = (("frame", frame), ("sub", sub), ("clean", clean), ("role", role), ("team", team))
    for name, array in named:
        if array.shape != (n,):
            problems.append(f"{name} has shape {array.shape}, expected ({n},)")
    if emb.ndim != 2 or emb.shape[0] != n:
        problems.append(f"emb has shape {emb.shape}, expected ({n}, D)")
    for name, array in (("frame", frame), ("sub", sub)):
        if not np.issubdtype(array.dtype, np.integer):
            problems.append(f"{name} has dtype {array.dtype}, expected integers")
    if problems:
        raise ValueError("gbf.relink.appear.build: " + "; ".join(problems))

    dim = emb.shape[1]
    has = ~np.isnan(emb[:, 0])
    if not has.any():
        return _empty(dim)
    E = np.where(has[:, None], emb, 0.0)

    # sub-pieces in ascending id, the rows of each in ascending frame
    order = np.lexsort((frame, sub))
    sub_sorted = sub[order]
    pieces = [int(p) for p in np.unique(sub_sorted[sub_sorted >= 0])]
    rows_of: dict[int, np.ndarray] = {}
    for pid in pieces:
        lo = np.searchsorted(sub_sorted, pid, "left")
        hi = np.searchsorted(sub_sorted, pid, "right")
        rows_of[pid] = order[lo:hi]
    role_of = {pid: _majority(role[rows]) for pid, rows in rows_of.items()}
    team_of = {pid: _majority(team[rows]) for pid, rows in rows_of.items()}

    # prototypes: the mean over the clean rows, over all rows with an embedding when low
    mean: dict[int, np.ndarray] = {}
    low: dict[int, bool] = {}
    used: dict[int, np.ndarray] = {}
    for pid, rows in rows_of.items():
        r = rows[clean[rows] & has[rows]]
        low[pid] = len(r) < params.min_clean
        if low[pid]:
            r = rows[has[rows]]
        if len(r):
            mean[pid] = E[r].mean(axis=0)
            used[pid] = r

    # team means from the sub-pieces labelled player of the team, and the global centre
    team_mean: dict[str, np.ndarray | None] = {}
    for name in ("left", "right"):
        parts = [
            used[pid]
            for pid in pieces
            if role_of[pid] == "player" and team_of[pid] == name and pid in used and not low[pid]
        ]
        rows = np.concatenate(parts) if parts else np.zeros(0, int)
        team_mean[name] = E[rows].mean(axis=0) if len(rows) >= params.team_mean_min_rows else None
    known = [v for v in team_mean.values() if v is not None]
    glob = np.mean(known, axis=0) if known else E[has].mean(axis=0)

    # appearance team: the label, else the team with the nearer mean; its mean is the centre
    ateam: dict[int, str | None] = {}
    centre: dict[int, np.ndarray] = {}
    for pid, m in mean.items():
        side = team_of[pid]
        if side not in ("left", "right"):
            near = [(float(unit(m) @ unit(v)), k) for k, v in team_mean.items() if v is not None]
            side = max(near)[1] if near else None
        ateam[pid] = side
        own = team_mean.get(side) if side is not None else None
        centre[pid] = own if own is not None else glob

    # whitening from the within-piece covariance: rows around their own sub-piece's mean
    within = [
        E[r] - mean[pid]
        for pid, r in used.items()
        if not low[pid] and len(r) >= params.wccn_min_rows
    ]
    if within:
        D = np.concatenate(within)
        Sw = D.T @ D / len(D)
        w, V = np.linalg.eigh(Sw)
        T = (V / np.sqrt(np.clip(w, 0, None) + params.wccn_lambda * w.mean())) @ V.T
    else:
        T = np.eye(dim)

    ids = [pid for pid in pieces if role_of[pid] != "referee" and pid in mean]
    return Appearance(
        ids=np.array(ids, np.int64),
        rows=np.array([len(rows_of[pid]) for pid in ids], np.int64),
        first=np.array([frame[rows_of[pid][0]] for pid in ids], np.int64),
        last=np.array([frame[rows_of[pid][-1]] for pid in ids], np.int64),
        frames=tuple(frozenset(frame[rows_of[pid]].tolist()) for pid in ids),
        team=np.array([ateam[pid] or "" for pid in ids], dtype=object),
        msum=_stack([mean[pid] * len(used[pid]) for pid in ids], dim),
        nrow=np.array([len(used[pid]) for pid in ids], float),
        centre=_stack([centre[pid] for pid in ids], dim),
        T=T,
    )
