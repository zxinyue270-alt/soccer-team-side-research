"""Relink, recall mode: attach the short sub-pieces ("leaves") to the stage-1 groups.

Port of stage 2 of the two-stage linker of the frozen method fuse_r1 of the re-identification
lab (reid_lab, outside the repository). Stage 1 is gbf.relink.link, which this module builds
on; the recall mode runs stage 1 with the recall margin and then this step.

Every SHORT candidate (fewer than ``n_solid`` boxes) may be attached to at most one stage-1
group (a single solid candidate is a group of one). A leaf never joins another short piece,
never connects two groups, and does not change the group's appearance prototype. The
similarity of a leaf f and a group G is

    S(f, G) = cosine of f's prototype and G's pooled prototype
            + the largest leaf bonus at a junction of f with G's SOLID members in time order

and the rule is the one of stage 1: same appearance team, no shared frame (leaves already
attached to G count), appearance cosine at least ``floor``, S at least ``t_abs``, f is G's best
free leaf and G is f's best group, and S exceeds by the margin both ``t0`` ("nobody") and the
best rival on either side (for f: its best S to a group seen together with G; for G: its best
S to a leaf seen together with f). The most similar passing pair is attached first.

The leaf bonus is position evidence only (gbf.relink.link.pair_bonus with ``use_plug=False``)
from the position model run on ALL sub-pieces; the caller builds it. The link written for a
leaf joins it to its anchor: of the group's solid members just before and just after the leaf
in time, the one with the larger leaf bonus, then the one nearer in frames.

Why two stages: when short pieces took part in the agglomeration, a piece of one to four boxes
could join a group on thin evidence and then, as its last member, pass a strong position bonus
on to the next piece, joining two long pieces of different people through it. Here two solid
pieces are joined on their own evidence only.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from gbf.relink import link

RECALL_MARGIN = 0.2
"""The margin of the recall mode, for stage 1 and for the leaves (recall = leaves on)."""


def attach_leaves(
    state: link.CoreState,
    bonus_leaf: Mapping[tuple[int, int], float] | None,
    params: link.LinkParams = link.DEFAULT_PARAMS,
    *,
    margin: float = RECALL_MARGIN,
    log: list[dict] | None = None,
) -> tuple[tuple[int, int], ...]:
    """The leaf links (earlier sub-piece id, later sub-piece id), in the order they were made.

    ``state`` is the state of :func:`gbf.relink.link.link_core` (not changed here);
    ``bonus_leaf`` is {(a, b): value > 0} for time-ordered pairs of sub-pieces, in cosine
    units, None or empty = appearance only. ``margin`` is the margin stage 1 ran with.
    ``log``, when given, receives one entry per attached leaf, like stage 1's.
    """
    ids = state.ids
    n = len(ids)
    if n < 2:
        return ()
    solid = state.solid
    alive = state.alive
    members = state.members
    first = state.first
    last = state.last
    same_team = state.same_team
    X = state.X
    N = state.N.copy()
    frames = list(state.frames)

    leaf_links: list[tuple[int, int]] = []
    isleaf = ~solid
    if isleaf.any() and alive.any():
        groups = alive.copy()  # representatives of the stage-1 groups
        alive2 = groups | isleaf
        B2 = link.bonus_matrix(bonus_leaf, state.pos, n)
        A2 = X @ X.T
        np.fill_diagonal(A2, -np.inf)
        A2[~alive2, :] = -np.inf
        A2[:, ~alive2] = -np.inf
        Bg2 = np.zeros((n, n))
        if bonus_leaf:
            for f in np.nonzero(isleaf)[0]:
                for g in np.nonzero(groups)[0]:
                    Bg2[f, g] = Bg2[g, f] = link.junction_bonus(members[g], [f], first, B2)
        cross = isleaf[:, None] != isleaf[None, :]
        while True:
            can = same_team & (N <= 0) & alive2[:, None] & alive2[None, :] & cross
            pick, S, mg = link.decide(
                A2, Bg2, N, can, alive2, params.t_abs, margin, params.floor, params.t0
            )
            if pick is None:
                break
            g, f = pick if isleaf[pick[1]] else (pick[1], pick[0])
            # anchor: the solid member next to the leaf in time with the larger junction bonus
            mem = members[g]
            prev = [i for i in mem if first[i] < first[f]]
            nxt = [i for i in mem if first[i] > first[f]]
            cand = []
            if prev:
                cand.append((B2[prev[-1], f], -abs(int(first[f]) - int(last[prev[-1]])), prev[-1]))
            if nxt:
                cand.append((B2[f, nxt[0]], -abs(int(first[nxt[0]]) - int(last[f])), nxt[0]))
            if not cand:
                raise ValueError(
                    f"attach_leaves: leaf {int(ids[f])} starts on the first frame of every "
                    f"member of the group of {int(ids[g])}, although they share no frame"
                )
            anchor = max(cand)[2]
            a, b = (anchor, f) if first[anchor] < first[f] else (f, anchor)
            leaf_links.append((int(ids[a]), int(ids[b])))
            if log is not None:
                log.append(
                    {
                        "stage": 2,
                        "s": float(S[g, f]),
                        "app": float(A2[g, f]),
                        "bonus": float(Bg2[g, f]),
                        "g": [int(ids[i]) for i in mem],
                        "h": [int(ids[f])],
                        "rival_g": float(mg[g, f]),
                        "rival_h": float(mg[f, g]),
                    }
                )
            alive2[f] = False
            frames[g] = frames[g] | frames[f]
            N[g] += N[f]
            N[:, g] += N[:, f]
            N[g, g] = 0
            A2[f, :] = -np.inf
            A2[:, f] = -np.inf
    return tuple(leaf_links)
