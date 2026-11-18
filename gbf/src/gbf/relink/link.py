"""Relink, link step: the pair bonus and the core agglomeration of solid sub-pieces.

Port of the frozen method fuse_r1 of the re-identification lab (reid_lab, outside the
repository): the bonus of its pair model, the plugin's own joins, and stage 1 of its two-stage
linker. Stage 2 (short pieces attached as leaves, for the recall mode) starts from the state
that stage 1 returns and lives in a module of its own; nothing here imports it.

A candidate is a sub-piece of the track table (a gbf piece, or a part of one after the cuts of
gbf.relink.cuts) that is not a referee and has an appearance prototype. A candidate with at
least ``n_solid`` boxes is SOLID, the others are SHORT. Stage 1 works on the solid candidates
only: a short one is neither a candidate nor a rival here, and the position evidence handed in
was computed without it.

Similarity of two groups G and H of solid candidates (a single candidate is a group of one):

    S(G, H) = cosine of their pooled, team-centred, whitened appearance prototypes
            + the largest pair bonus at a junction of G and H

A junction is a pair (p in G, q in H) that would be neighbours in time in the merged chain. The
pair bonus of a time-ordered pair comes from the position model (gbf.relink.space: the doubly
normalised assignment score ``sink``) and from "the plugin tracked both as one id" (``plug``),
weighted by the lab's logistic pair model and divided by its weight of the cosine, so that it is
expressed in cosine units:

    bonus = (w_sink_lo * sink_lo + w_plug * plug) / w_cos        kept when > 0
    sink_lo = logit(clip(sink, sink_floor, sink_clip)) - logit(sink_floor)

Rule: two groups may merge when they are of the same appearance team and share no frame.
Repeatedly merge the most similar allowed pair (G, H) that passes all of

- the appearance cosine alone is at least ``floor`` and S is at least ``t_abs``;
- H is G's most similar allowed group and G is H's (mutual best);
- S exceeds by ``margin`` both ``t0`` (the "nobody" rival) and G's best S to an allowed group
  seen together with H (such a group is certainly not H's person), and the same for H;

then pool the two prototypes, pool the shared-frame counts, refresh the junction bonuses of the
new group and continue. Links are the consecutive members, in time order, of every group.

The arithmetic follows the lab's expression by expression (operand shapes and orders included)
because every decision is a threshold on floats that went through BLAS: ``X @ X.T`` over all
candidates before masking, one matrix-vector product per prototype, ``X @ X[g]`` after a merge,
first-index ``argmax`` on ties. The one departure is the rival of :func:`decide`, computed
column by column instead of through an n x n x n array: a maximum over the same numbers.

Everything here is pure: no state between calls, no file, no annotation. ``app`` is read by
attribute name (the ``Appearance`` of gbf.relink.appear): ``ids``, ``rows``, ``first``,
``last``, ``frames``, ``team``, ``msum``, ``nrow``, ``centre``, ``T``; it is not changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LinkParams:
    """The frozen numbers of the linker (the lab's "all13" fit; see the parameter table)."""

    n_solid: int = 5
    """Boxes: a sub-piece with at least this many is solid."""
    floor: float = 0.3
    """No merge below this appearance cosine."""
    t0: float = 0.5
    """The "nobody" rival: every margin is taken against at least this similarity."""
    t_abs: float = 0.5
    """Least fused similarity of a merge."""
    margin: float = 0.6
    """The strict margin: how far S must exceed the best rival and ``t0``."""
    w_cos: float = 7.1758068639395765
    """Pair model weight of the whitened cosine (the unit of the bonus)."""
    w_sink_lo: float = 0.64737768782866
    """Pair model weight of the position score's log odds above its floor."""
    w_plug: float = -0.10943515159871339
    """Pair model weight of "consecutive pieces of one plugin id"."""
    sink_floor: float = 1e-3
    """Lower clip of the position score before the log odds."""
    sink_clip: float = 0.98
    """Upper clip of the position score before the log odds."""

    def as_dict(self) -> dict[str, Any]:
        """JSON form for reports and receipts."""
        return asdict(self)


DEFAULT_PARAMS = LinkParams()


# --------------------------------------------------------------------------- pair bonus


def unit(v: np.ndarray) -> np.ndarray:
    """``v`` scaled to unit length along its last axis (a zero vector stays zero)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def candidate_pairs(app: Any, n_solid: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(ids of a, ids of b) of every ordered candidate pair: a ends before b starts and both
    have the same appearance team; with ``n_solid`` > 0 both also have at least that many
    boxes. Pairs are a-major, then b, both in the order of ``app.ids``."""
    ids = np.asarray(app.ids)
    first = np.asarray(app.first)
    last = np.asarray(app.last)
    team = np.asarray(app.team, dtype=object)
    ok = (last[:, None] < first[None, :]) & (team[:, None] == team[None, :])
    if n_solid > 0:
        solid = np.asarray(app.rows) >= n_solid
        ok &= solid[:, None] & solid[None, :]
    ii, jj = np.nonzero(ok)
    return ids[ii], ids[jj]


def plugin_joins(
    piece: np.ndarray, parent: np.ndarray, plugin: np.ndarray, first: np.ndarray, last: np.ndarray
) -> set[tuple[int, int]]:
    """{(a, b)}: consecutive (sub-)pieces of one plugin id that gbf's re-entry split separated.

    The arrays describe the pieces of one run of the position model: the piece id, the gbf
    piece it came from (``parent``; two parts of one gbf piece were separated by a relink cut,
    not by gbf, and are no join), the plugin's track id, and the first and last frame. Pieces
    are ordered by (plugin id, first frame); a is directly followed by b under one plugin id,
    their parents differ and a ends before b starts.
    """
    piece = np.asarray(piece)
    parent = np.asarray(parent)
    plugin = np.asarray(plugin)
    first = np.asarray(first)
    last = np.asarray(last)
    order = np.lexsort((first, plugin))
    out: set[tuple[int, int]] = set()
    for i, j in zip(order[:-1], order[1:], strict=True):
        if plugin[i] == plugin[j] and parent[i] != parent[j] and last[i] < first[j]:
            out.add((int(piece[i]), int(piece[j])))
    return out


def pair_bonus(
    a: np.ndarray,
    b: np.ndarray,
    sink: np.ndarray,
    plug: np.ndarray,
    params: LinkParams = DEFAULT_PARAMS,
    *,
    use_plug: bool = True,
) -> dict[tuple[int, int], float]:
    """{(a, b): bonus in cosine units} for the pairs with a positive bonus, in input order.

    ``a``, ``b``, ``sink`` and ``plug`` are parallel: the ids of a time-ordered pair, the
    position model's assignment score (0.0 where it does not admit the pair) and 1.0 where the
    pair is a plugin join. ``use_plug=False`` gives the position-only bonus of the leaves.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    sink = np.asarray(sink, dtype=np.float64)
    plug = np.asarray(plug, dtype=np.float64)
    if not (len(a) == len(b) == len(sink) == len(plug)):
        raise ValueError(
            "pair_bonus: a, b, sink and plug must be parallel, got lengths "
            f"{len(a)}, {len(b)}, {len(sink)}, {len(plug)}"
        )
    if not len(a):
        return {}
    c = np.clip(sink, params.sink_floor, params.sink_clip)
    lo = np.log(c / (1 - c))
    # >= 0, and 0 when sink <= sink_floor
    sink_lo = (lo - lo.min()) * 0 + (lo + np.log((1 - params.sink_floor) / params.sink_floor))
    if use_plug:
        weights = np.asarray([params.w_sink_lo, params.w_plug]) / params.w_cos
        v = np.column_stack([sink_lo, plug]) @ weights
    else:
        weights = np.asarray([params.w_sink_lo]) / params.w_cos
        v = np.column_stack([sink_lo]) @ weights
    keep = v > 0
    return {(int(x), int(y)): float(z) for x, y, z in zip(a[keep], b[keep], v[keep], strict=True)}


def bonus_matrix(
    bonus: Mapping[tuple[int, int], float] | None, pos: Mapping[int, int], n: int
) -> np.ndarray:
    """(n, n) piece-level bonus, [earlier, later]; ``pos`` maps a candidate id to its index.
    Pairs with an id that is no candidate, and values that are not positive, are left out."""
    B = np.zeros((n, n))
    if bonus:
        for (a, b), v in bonus.items():
            i, j = pos.get(int(a)), pos.get(int(b))
            if i is not None and j is not None and v > 0:
                B[i, j] = v
    return B


def junction_bonus(mg: Sequence[int], mh: Sequence[int], first: np.ndarray, B: np.ndarray) -> float:
    """Largest bonus over the junctions of two groups.

    ``mg`` and ``mh`` are the member indices of the groups, ``first`` the first frame per
    index, ``B`` the piece-level bonus [earlier, later]. The members of both are put in time
    order; a junction is a pair of neighbours that belong to different groups.
    """
    seq = sorted([(first[i], 0, i) for i in mg] + [(first[i], 1, i) for i in mh])
    best = 0.0
    for (_, ga, ia), (_, gb, ib) in zip(seq[:-1], seq[1:], strict=True):
        if ga != gb:
            v = B[ia, ib]
            if v > best:
                best = v
    return best


# --------------------------------------------------------------------------- the rule


def _rival(Sc: np.ndarray, together: np.ndarray) -> np.ndarray:
    """rival[g, h] = the largest Sc[g, x] over the x with ``together[h, x]``, -inf without one.

    The lab writes ``(Sc[:, None, :] + Om[None, :, :]).max(axis=2)`` with Om = 0.0 where
    ``together`` and -inf elsewhere. This is the same maximum over the same numbers (the 0.0
    is added here too, so that even the sign of a zero agrees), one column at a time.
    """
    n = len(Sc)
    rival = np.full((n, n), -np.inf)
    for h in range(n):
        cols = np.flatnonzero(together[h])
        if len(cols):
            rival[:, h] = (Sc[:, cols] + 0.0).max(axis=1)
    return rival


def decide(
    A: np.ndarray,
    Bg: np.ndarray,
    N: np.ndarray,
    can: np.ndarray,
    alive: np.ndarray,
    t_abs: float,
    margin: float,
    floor: float | None,
    t0: float | None,
) -> tuple[tuple[int, int] | None, np.ndarray, np.ndarray]:
    """One step of the rule: (the index pair (g, h) to merge or None, S, the margins).

    ``A`` is the appearance cosine of the groups (-inf where there is none), ``Bg`` their
    junction bonus, ``N`` their shared-frame counts, ``can`` the allowed pairs and ``alive``
    the groups that still exist. ``margins[g, h]`` is S[g, h] minus g's best allowed S to a
    living group seen together with h, taken against at least ``t0``; with ``t0`` None a side
    without such a rival passes (NaN). ``floor`` None switches the appearance floor off.
    """
    n = len(A)
    S = A + Bg
    seen = N >= 1
    Sc = np.where(can, S, -np.inf)
    rival = _rival(Sc, seen & alive[None, :])
    with np.errstate(invalid="ignore"):
        if t0 is None:
            mg = np.where(np.isfinite(rival), S - rival, np.nan)
        else:
            mg = S - np.maximum(rival, t0)  # rival is -inf where there is none
    mh = mg.T
    ok = can & (S >= t_abs)
    if floor is not None:
        ok &= A >= floor
    ok &= (np.isnan(mg) | (mg >= margin)) & (np.isnan(mh) | (mh >= margin))
    best = Sc.argmax(axis=1)
    isbest = np.zeros((n, n), bool)
    isbest[np.arange(n), best] = True
    ok &= isbest & isbest.T
    if not ok.any():
        return None, S, mg
    Sok = np.where(ok, S, -np.inf)
    g, h = np.unravel_index(int(Sok.argmax()), Sok.shape)
    return (int(g), int(h)), S, mg


# --------------------------------------------------------------------------- stage 1


@dataclass
class CoreState:
    """The linker's state after stage 1: what stage 2 (the leaves) starts from.

    n candidates in the order of ``ids``; an index below is a position in that order.
    """

    ids: np.ndarray
    """(n,) sub-piece ids."""
    pos: dict[int, int]
    """Index of every id."""
    solid: np.ndarray
    """(n,) bool: at least ``n_solid`` boxes (the others are the short candidates)."""
    alive: np.ndarray
    """(n,) bool: representative of a stage-1 group (a solid candidate not merged away)."""
    members: list[list[int]]
    """Per index, the solid members of its group in time order (read at the living ones)."""
    N: np.ndarray
    """(n, n) int: shared-frame counts of the groups, pooled after every merge."""
    frames: list[frozenset[int]]
    """Per index, the frames of its group."""
    X: np.ndarray
    """(n, D) unit prototypes, team-centred and whitened; pooled at the representatives."""
    first: np.ndarray
    """(n,) first frame of every candidate."""
    last: np.ndarray
    """(n,) last frame of every candidate."""
    same_team: np.ndarray
    """(n, n) bool: equal appearance teams."""


@dataclass(frozen=True)
class CoreResult:
    """Stage 1: ``links`` are (earlier sub-piece id, later sub-piece id), group by group in
    the order of the representatives, consecutive members in time order."""

    links: tuple[tuple[int, int], ...]
    state: CoreState


def _app_problems(app: Any, n: int) -> list[str]:
    problems: list[str] = []
    for name in ("rows", "first", "last", "frames", "team", "msum", "nrow", "centre"):
        if len(getattr(app, name)) != n:
            problems.append(f"{name} has {len(getattr(app, name))} entries for {n} candidates")
    if problems or n == 0:
        return problems
    msum, centre, T = np.asarray(app.msum), np.asarray(app.centre), np.asarray(app.T)
    if msum.ndim != 2 or centre.shape != msum.shape:
        problems.append(f"msum {msum.shape} and centre {centre.shape} must be equal (n, D) arrays")
    elif T.shape != (msum.shape[1], msum.shape[1]):
        problems.append(f"T {T.shape} must be (D, D) with D = {msum.shape[1]}")
    return problems


def link_core(
    app: Any,
    bonus: Mapping[tuple[int, int], float] | None,
    params: LinkParams = DEFAULT_PARAMS,
    *,
    margin: float | None = None,
    log: list[dict] | None = None,
) -> CoreResult:
    """Stage 1: the links between solid candidates, and the state for stage 2.

    ``bonus`` is {(a, b): value > 0} for time-ordered pairs of SOLID sub-pieces, in cosine
    units (:func:`pair_bonus` on the position scores of the solid sub-pieces); None or empty
    = appearance only. ``margin`` None means ``params.margin``. With fewer than two candidates
    there is nothing to link. ``log``, when given, receives one entry per merge: its
    similarity, appearance cosine, bonus, the ids of both groups and the two margins.
    """
    use_margin = params.margin if margin is None else margin
    ids = np.asarray(app.ids)
    n = len(ids)
    problems = _app_problems(app, n)
    if problems:
        raise ValueError("link_core: " + "; ".join(problems))
    pos = {int(p): i for i, p in enumerate(ids)}
    solid = np.asarray(app.rows) >= params.n_solid
    nrow = np.array(app.nrow, dtype=np.float64)
    msum = np.array(app.msum, dtype=np.float64)
    team = np.asarray(app.team, dtype=object)
    centre = np.asarray(app.centre, dtype=np.float64)
    T = np.ascontiguousarray(app.T, dtype=np.float64)  # row-major, as the lab's operand
    first = np.asarray(app.first)
    last = np.asarray(app.last)
    frames = [frozenset(f) for f in app.frames]
    members = [[i] for i in range(n)]  # solid members of a group, in time order
    N = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            if first[i] <= last[j] and first[j] <= last[i]:
                N[i, j] = N[j, i] = len(frames[i] & frames[j])
    same_team = team[:, None] == team[None, :]
    alive = solid.copy()
    if n < 2:
        state = CoreState(
            ids=ids,
            pos=pos,
            solid=solid,
            alive=alive,
            members=members,
            N=N,
            frames=frames,
            X=np.zeros(msum.shape if msum.ndim == 2 else (n, 0)),
            first=first,
            last=last,
            same_team=same_team,
        )
        return CoreResult((), state)

    def proto(i: int) -> np.ndarray:
        v = msum[i] / nrow[i]
        v = v - centre[i]
        return unit(T @ v)

    X = np.stack([proto(i) for i in range(n)])

    B = bonus_matrix(bonus, pos, n)
    Bg = np.maximum(B, B.T)  # group-level bonus (single pieces: the pair's own)
    A = X @ X.T  # appearance similarity of groups
    np.fill_diagonal(A, -np.inf)
    A[~alive, :] = -np.inf
    A[:, ~alive] = -np.inf
    while True:
        can = same_team & (N <= 0) & alive[:, None] & alive[None, :]
        np.fill_diagonal(can, False)
        pick, S, mg = decide(
            A, Bg, N, can, alive, params.t_abs, use_margin, params.floor, params.t0
        )
        if pick is None:
            break
        g, h = pick
        if log is not None:
            log.append(
                {
                    "stage": 1,
                    "s": float(S[g, h]),
                    "app": float(A[g, h]),
                    "bonus": float(Bg[g, h]),
                    "g": [int(ids[i]) for i in members[g]],
                    "h": [int(ids[i]) for i in members[h]],
                    "rival_g": float(mg[g, h]),
                    "rival_h": float(mg[h, g]),
                }
            )
        msum[g] += msum[h]
        nrow[g] += nrow[h]
        frames[g] = frames[g] | frames[h]
        members[g] = sorted(members[g] + members[h], key=lambda i: first[i])
        alive[h] = False
        N[g] += N[h]
        N[:, g] += N[:, h]
        N[g, g] = 0
        X[g] = proto(g)
        v = X @ X[g]
        v[~alive] = -np.inf
        A[g, :] = v
        A[:, g] = v
        A[g, g] = -np.inf
        A[h, :] = -np.inf
        A[:, h] = -np.inf
        if bonus:
            for x in np.nonzero(alive)[0]:
                if x != g:
                    Bg[g, x] = Bg[x, g] = junction_bonus(members[g], members[x], first, B)
        Bg[h, :] = 0.0
        Bg[:, h] = 0.0
    links: list[tuple[int, int]] = []
    for i in np.nonzero(alive)[0]:
        m = members[i]
        links += [(int(ids[m[k]]), int(ids[m[k + 1]])) for k in range(len(m) - 1)]
    state = CoreState(
        ids=ids,
        pos=pos,
        solid=solid,
        alive=alive,
        members=members,
        N=N,
        frames=frames,
        X=X,
        first=first,
        last=last,
        same_team=same_team,
    )
    return CoreResult(tuple(links), state)
