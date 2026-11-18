"""Tests for gbf.relink.leaves (stage 2 of the relink linker: short pieces as leaves).

The scenes are the synthetic ones of test_relink_link: stretches of frames with unit prototypes
built from the cosines a case needs. A piece of three frames has three boxes and is short; the
others are solid. Stage 1 runs first, with the same margin, as in the recall mode.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import pytest

from gbf.relink import leaves, link
from test_relink_link import make_app, piece, scene

Bonus = Mapping[tuple[int, int], float] | None


def run(
    app: SimpleNamespace,
    bonus_leaf: Bonus = None,
    *,
    bonus_core: Bonus = None,
    margin: float = leaves.RECALL_MARGIN,
    log: list[dict] | None = None,
) -> tuple[link.CoreResult, tuple[tuple[int, int], ...]]:
    core = link.link_core(app, bonus_core, margin=margin)
    return core, leaves.attach_leaves(core.state, bonus_leaf, margin=margin, log=log)


def test_the_recall_margin() -> None:
    assert leaves.RECALL_MARGIN == pytest.approx(0.2)
    # S = 0.75 is 0.25 above the "nobody" rival: enough for 0.2, not for 0.3
    app = scene([piece(1, 0, 19), piece(2, 30, 32)], {(1, 2): 0.75})
    state = link.link_core(app, None, margin=leaves.RECALL_MARGIN).state
    assert leaves.attach_leaves(state, None) == ((1, 2),)
    assert leaves.attach_leaves(state, None, margin=0.3) == ()
    assert leaves.attach_leaves(state, {(1, 2): 0.1}, margin=0.3) == ((1, 2),)


def test_a_short_piece_joins_the_group_it_looks_like() -> None:
    # 1 and 2 are one person, 3 another one seen together with 1; 4 has three boxes
    pieces = [piece(1, 0, 19), piece(2, 40, 59), piece(3, 0, 19), piece(4, 25, 27)]
    app = scene(pieces, {(1, 2): 0.9, (1, 4): 0.85, (2, 4): 0.85})
    log: list[dict] = []
    core, leaf = run(app, log=log)
    assert core.links == ((1, 2),)
    # the link goes to the nearer solid member of the group, earlier piece first
    assert leaf == ((1, 4),)
    assert len(log) == 1
    entry = log[0]
    assert entry["stage"] == 2
    assert entry["g"] == [1, 2]
    assert entry["h"] == [4]
    assert entry["bonus"] == 0.0
    # cosine with the pooled prototype of the group: 1.7 / (2 * sqrt(0.95))
    assert entry["app"] == pytest.approx(0.85 / np.sqrt(0.95), abs=1e-9)
    assert entry["s"] == pytest.approx(entry["app"])
    # in the strict mode there are no leaves: stage 1 alone leaves the short piece out
    assert all(4 not in pair for pair in core.links)


def test_a_leaf_never_joins_another_short_piece() -> None:
    pieces = [piece(1, 0, 19), piece(2, 30, 32), piece(3, 50, 52)]
    app = scene(pieces, {(2, 3): 0.95})
    for margin in (leaves.RECALL_MARGIN, 0.0):
        assert run(app, margin=margin)[1] == ()
        assert run(app, {(2, 3): 0.9}, margin=margin)[1] == ()


def test_a_leaf_joins_one_group_and_never_connects_two() -> None:
    pieces = [piece(1, 0, 19), piece(2, 30, 32), piece(3, 50, 69)]
    nearer_first = scene(pieces, {(1, 2): 0.8, (2, 3): 0.75, (1, 3): 0.4})
    core, leaf = run(nearer_first)
    assert core.links == ()
    assert leaf == ((1, 2),)
    nearer_last = scene(pieces, {(1, 2): 0.75, (2, 3): 0.8, (1, 3): 0.4})
    core, leaf = run(nearer_last)
    assert core.links == ()
    assert leaf == ((2, 3),)
    # even with no margin at all the leaf is used once
    core, leaf = run(nearer_first, margin=0.0)
    assert [pair for pair in core.links + leaf if 2 in pair] == [(1, 2)]


def test_two_groups_seen_together_block_a_leaf_that_looks_like_both() -> None:
    pieces = [piece(1, 0, 19), piece(2, 0, 19), piece(3, 30, 32)]
    app = scene(pieces, {(1, 3): 0.85, (2, 3): 0.8, (1, 2): 0.6})
    core, leaf = run(app)
    assert core.links == ()
    assert leaf == ()
    # position evidence for one of them settles it: S = 1.35 against 0.8
    assert run(app, {(1, 3): 0.5})[1] == ((1, 3),)


def _two_leaves(second: tuple[int, int]) -> SimpleNamespace:
    pieces = [piece(1, 0, 19), piece(2, 30, 32), piece(3, *second)]
    return scene(pieces, {(1, 2): 0.9, (1, 3): 0.85, (2, 3): 0.8})


def test_a_second_leaf_that_shares_a_frame_with_the_first_stays_out() -> None:
    bonus = {(1, 2): 0.5}  # S(1, 2) = 1.4, S(1, 3) = 0.85
    together = _two_leaves((31, 33))
    assert run(together, bonus)[1] == ((1, 2),)
    assert run(together, bonus, margin=0.0)[1] == ((1, 2),)
    # two short pieces seen together that both look like the group, and nothing to tell
    # them apart: neither is attached
    assert run(together)[1] == ()
    # a second leaf at another time joins the same group after the first
    apart = _two_leaves((40, 42))
    assert run(apart, bonus)[1] == ((1, 2), (1, 3))


def test_an_attached_leaf_does_not_change_the_group_or_the_state() -> None:
    app = _two_leaves((40, 42))
    core = link.link_core(app, None, margin=leaves.RECALL_MARGIN)
    state = core.state
    before = copy.deepcopy(state)
    log: list[dict] = []
    leaf = leaves.attach_leaves(state, {(1, 2): 0.5}, log=log)
    assert leaf == ((1, 2), (1, 3))
    # the second leaf is compared with the group's own prototype, not one pooled with the
    # first leaf (that cosine would be 0.853), and the group lists solid members only
    assert log[1]["app"] == pytest.approx(0.85, abs=1e-9)
    assert log[1]["app"] == pytest.approx(float(state.X[0] @ state.X[2]), abs=1e-12)
    assert log[1]["g"] == [1]
    assert log[1]["h"] == [3]
    assert np.array_equal(state.N, before.N)
    assert np.array_equal(state.alive, before.alive)
    assert np.array_equal(state.X, before.X)
    assert state.frames == before.frames
    assert state.members == before.members
    # so the same state gives the same leaves again
    assert leaves.attach_leaves(state, {(1, 2): 0.5}) == leaf


@pytest.mark.parametrize(
    ("leaf_piece", "cosine", "attached"),
    [
        (piece(2, 30, 32), 0.85, True),
        (piece(2, 30, 32, team="right"), 0.85, False),
        (piece(2, 18, 20), 0.85, False),  # frames 18 and 19 are the group's too
        (piece(2, 30, 32), 0.2, False),  # S = 1.1 passes, the appearance floor does not
        (piece(2, 30, 33), 0.85, True),  # four boxes are still short
        (piece(2, 30, 34), 0.85, False),  # five boxes are solid: no leaf (and no core link)
    ],
)
def test_team_shared_frames_floor_and_size_rule_a_leaf_out(
    leaf_piece: dict, cosine: float, attached: bool
) -> None:
    app = scene([piece(1, 0, 19), leaf_piece], {(1, 2): cosine})
    core = link.link_core(app, None, margin=1.5)  # too strict for any stage-1 merge
    assert core.links == ()
    leaf = leaves.attach_leaves(core.state, {(1, 2): 0.9})
    assert leaf == (((1, 2),) if attached else ())


GROUP = (piece(1, 0, 19), piece(2, 60, 79))
LATE_GROUP = (piece(1, 20, 39), piece(2, 60, 79))


@pytest.mark.parametrize(
    ("group", "leaf_span", "bonus", "expected"),
    [
        (GROUP, (30, 32), {(1, 3): 0.3, (3, 2): 0.1}, (1, 3)),  # the larger bonus
        (GROUP, (30, 32), {(1, 3): 0.1, (3, 2): 0.3}, (3, 2)),
        (GROUP, (30, 32), None, (1, 3)),  # no bonus: 11 frames after 1, 28 before 2
        (GROUP, (50, 52), None, (3, 2)),  # 31 frames after 1, 8 before 2
        (GROUP, (50, 52), {(1, 3): 0.2}, (1, 3)),  # the bonus counts before the distance
        (LATE_GROUP, (5, 7), None, (3, 1)),  # before the whole group
        (GROUP, (90, 92), None, (2, 3)),  # after the whole group
    ],
)
def test_the_anchor_of_a_leaf(
    group: tuple[dict, dict], leaf_span: tuple[int, int], bonus: Bonus, expected: tuple[int, int]
) -> None:
    pieces = [*group, piece(3, *leaf_span)]
    app = scene(pieces, {(1, 2): 0.9, (1, 3): 0.85, (2, 3): 0.85})
    core, leaf = run(app, bonus)
    assert core.links == ((1, 2),)
    assert leaf == (expected,)


def test_the_leaf_bonus_is_read_at_the_junctions_with_solid_members() -> None:
    # the leaf lies between 1 and 2; its bonus with 2's later partner 4 is no junction
    pieces = [piece(1, 0, 19), piece(2, 60, 79), piece(3, 30, 32), piece(4, 100, 119)]
    cosines = {(1, 2): 0.95, (2, 4): 0.9, (1, 4): 0.85, (1, 3): 0.6, (2, 3): 0.6, (3, 4): 0.6}
    app = scene(pieces, cosines)
    log: list[dict] = []
    core, leaf = run(app, {(3, 4): 0.9}, log=log)
    assert core.links == ((1, 2), (2, 4))
    # cosine 0.62 with the group and no bonus at (1, 3) or (3, 2): 0.12 above "nobody"
    assert leaf == ()
    log = []
    core, leaf = run(app, {(3, 4): 0.9, (3, 2): 0.3}, log=log)
    assert leaf == ((3, 2),)
    assert log[0]["bonus"] == pytest.approx(0.3)
    assert log[0]["g"] == [1, 2, 4]


def test_nothing_to_attach() -> None:
    solid_only = scene([piece(1, 0, 19), piece(2, 40, 59)], {(1, 2): 0.9})
    assert run(solid_only, {(1, 2): 0.9})[1] == ()
    short_only = scene([piece(1, 0, 2), piece(2, 40, 42)], {(1, 2): 0.9})
    assert run(short_only, {(1, 2): 0.9})[1] == ()
    assert run(short_only, {(1, 2): 0.9}, bonus_core={(1, 2): 0.9}, margin=0.0)[1] == ()
    single = scene([piece(1, 0, 2)], {})
    assert run(single)[1] == ()
    assert run(make_app([], np.zeros((0, 4))))[1] == ()
