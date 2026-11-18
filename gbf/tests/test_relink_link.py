"""Tests for gbf.relink.link (the pair bonus and stage 1 of the relink linker).

Every scene is synthetic. A candidate is a stretch of frames with a unit prototype; the
prototypes of a scene are built from the cosines the case needs (the Cholesky factor of their
Gram matrix), so every similarity in a case is known and sits well away from the thresholds.
One case draws noisy embeddings from a seeded generator instead.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import pytest

from gbf.relink import link

STRICT = link.DEFAULT_PARAMS.margin
LOOSE = 0.2


def piece(
    pid: int, first: int, last: int, team: str = "left", rows: int | None = None
) -> dict[str, object]:
    """A candidate seen on every frame from ``first`` to ``last``; ``rows`` = its boxes."""
    return {
        "id": pid,
        "first": first,
        "last": last,
        "team": team,
        "rows": last - first + 1 if rows is None else rows,
    }


def unit_rows(pieces: Sequence[Mapping], cosines: Mapping[tuple[int, int], float]) -> np.ndarray:
    """Unit prototypes, one row per piece, with the given cosine per pair of ids (0 for the
    pairs that are not named)."""
    index = {p["id"]: i for i, p in enumerate(pieces)}
    gram = np.eye(len(pieces))
    for (a, b), value in cosines.items():
        gram[index[a], index[b]] = gram[index[b], index[a]] = value
    return np.linalg.cholesky(gram)


def make_app(pieces: Sequence[Mapping], vectors: np.ndarray) -> SimpleNamespace:
    """The appearance object the linker reads: the prototype of a piece is ``vectors[i]``
    (identity whitening, zero centre, one prototype row per box)."""
    n, dim = vectors.shape
    rows = np.array([p["rows"] for p in pieces], np.int64)
    return SimpleNamespace(
        ids=np.array([p["id"] for p in pieces], np.int64),
        rows=rows,
        first=np.array([p["first"] for p in pieces], np.int64),
        last=np.array([p["last"] for p in pieces], np.int64),
        frames=tuple(frozenset(range(p["first"], p["last"] + 1)) for p in pieces),
        team=np.array([p["team"] for p in pieces], dtype=object),
        msum=vectors * rows[:, None],
        nrow=rows.astype(np.float64),
        centre=np.zeros((n, dim)),
        T=np.eye(dim),
    )


def scene(pieces: Sequence[Mapping], cosines: Mapping[tuple[int, int], float]) -> SimpleNamespace:
    return make_app(pieces, unit_rows(pieces, cosines))


def links_of(app, bonus=None, margin=None, params=link.DEFAULT_PARAMS, log=None):
    return link.link_core(app, bonus, params, margin=margin, log=log).links


# --------------------------------------------------------------------------- parameters


def test_default_params_are_the_frozen_numbers() -> None:
    params = link.DEFAULT_PARAMS
    assert params == link.LinkParams()
    assert params.as_dict() == {
        "n_solid": 5,
        "floor": 0.3,
        "t0": 0.5,
        "t_abs": 0.5,
        "margin": 0.6,
        "w_cos": 7.1758068639395765,
        "w_sink_lo": 0.64737768782866,
        "w_plug": -0.10943515159871339,
        "sink_floor": 0.001,
        "sink_clip": 0.98,
    }
    assert json.loads(json.dumps(params.as_dict())) == params.as_dict()


def test_unit_scales_along_the_last_axis_and_keeps_a_zero_vector() -> None:
    out = link.unit(np.array([[3.0, 4.0], [0.0, 0.0], [0.0, -2.0]]))
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0], [0.0, -1.0]])
    assert np.allclose(link.unit(np.array([0.0, 5.0])), [0.0, 1.0])


# --------------------------------------------------------------------------- pair bonus


def test_candidate_pairs_are_time_ordered_same_team_and_a_major() -> None:
    pieces = [
        piece(1, 0, 9),
        piece(2, 20, 29),
        piece(3, 5, 14),
        piece(4, 40, 49, team="right"),
        piece(5, 30, 32),
        piece(6, 60, 69, team=""),
        piece(7, 80, 89, team=""),
    ]
    app = make_app(pieces, np.eye(len(pieces)))
    a, b = link.candidate_pairs(app)
    assert list(zip(a.tolist(), b.tolist(), strict=True)) == [
        (1, 2),
        (1, 5),
        (2, 5),
        (3, 2),
        (3, 5),
        (6, 7),
    ]
    a, b = link.candidate_pairs(app, 5)
    assert list(zip(a.tolist(), b.tolist(), strict=True)) == [(1, 2), (3, 2), (6, 7)]
    empty = make_app([], np.zeros((0, 3)))
    a, b = link.candidate_pairs(empty)
    assert len(a) == 0
    assert len(b) == 0


def test_plugin_joins_are_consecutive_pieces_of_one_plugin_id_split_by_gbf() -> None:
    stride = 10**7
    # plugin id 7: three gbf pieces, the second one cut by relink at frame 25; plugin id 8:
    # two pieces that overlap in time
    piece_id = np.array([7, 100007, 100007 + stride, 200007, 8, 100008])
    plugin = np.array([7, 7, 7, 7, 8, 8])
    first = np.array([0, 20, 25, 40, 0, 5])
    last = np.array([9, 24, 29, 49, 9, 30])
    expected = {(7, 100007), (100007 + stride, 200007)}
    assert link.plugin_joins(piece_id, piece_id % stride, plugin, first, last) == expected
    order = np.random.default_rng(3).permutation(len(piece_id))
    shuffled = link.plugin_joins(
        piece_id[order], (piece_id % stride)[order], plugin[order], first[order], last[order]
    )
    assert shuffled == expected
    none = np.array([], np.int64)
    assert link.plugin_joins(none, none, none, none, none) == set()


def test_pair_bonus_is_the_position_log_odds_in_cosine_units() -> None:
    params = link.DEFAULT_PARAMS
    a = np.array([1, 2, 3, 4, 5, 6])
    b = a + 10
    sink = np.array([0.0, 0.0005, 0.001, 0.5, 0.98, 0.999])
    bonus = link.pair_bonus(a, b, sink, np.zeros(6))
    # nothing at or below the floor of the position score
    for key in ((1, 11), (2, 12), (3, 13)):
        assert bonus.get(key, 0.0) < 1e-9
    assert [key for key in bonus if bonus[key] > 1e-9] == [(4, 14), (5, 15), (6, 16)]
    scale = params.w_sink_lo / params.w_cos
    assert bonus[(4, 14)] == pytest.approx(np.log(999.0) * scale, rel=1e-9)
    # clipped at sink_clip: the largest bonus there is
    assert bonus[(5, 15)] == pytest.approx((np.log(49.0) + np.log(999.0)) * scale, rel=1e-9)
    assert bonus[(6, 16)] == pytest.approx(bonus[(5, 15)], rel=1e-12)
    assert 0.9 < bonus[(5, 15)] < 1.0
    assert all(isinstance(value, float) for value in bonus.values())


def test_pair_bonus_plugin_weight_is_negative_and_can_be_left_out() -> None:
    params = link.DEFAULT_PARAMS
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    sink = np.array([0.5, 0.00101, 0.0])
    plain = link.pair_bonus(a, b, sink, np.zeros(3))
    joined = link.pair_bonus(a, b, sink, np.ones(3))
    assert joined[(1, 4)] == pytest.approx(plain[(1, 4)] + params.w_plug / params.w_cos, rel=1e-9)
    assert joined[(1, 4)] < plain[(1, 4)]
    # a weak position score with a plugin join falls below zero and is dropped
    assert 0.0 < plain[(2, 5)] < 0.01
    assert list(joined) == [(1, 4)]
    without = link.pair_bonus(a, b, sink, np.ones(3), use_plug=False)
    assert without[(1, 4)] == pytest.approx(plain[(1, 4)], rel=1e-12)
    assert without[(2, 5)] == pytest.approx(plain[(2, 5)], rel=1e-9)


def test_pair_bonus_follows_its_params_and_checks_its_inputs() -> None:
    a, b = np.array([1]), np.array([2])
    double = link.LinkParams(w_sink_lo=2 * link.DEFAULT_PARAMS.w_sink_lo)
    base = link.pair_bonus(a, b, np.array([0.3]), np.zeros(1))[(1, 2)]
    assert link.pair_bonus(a, b, np.array([0.3]), np.zeros(1), double)[(1, 2)] == pytest.approx(
        2 * base, rel=1e-9
    )
    high_floor = link.LinkParams(sink_floor=0.4)
    assert link.pair_bonus(a, b, np.array([0.3]), np.zeros(1), high_floor).get((1, 2), 0.0) < 1e-9
    none = np.array([], np.int64)
    assert link.pair_bonus(none, none, np.array([]), np.array([])) == {}
    with pytest.raises(ValueError, match="parallel"):
        link.pair_bonus(a, b, np.array([0.3, 0.4]), np.zeros(1))


def test_bonus_matrix_keeps_positive_values_of_known_pairs() -> None:
    pos = {10: 0, 20: 1, 30: 2}
    bonus = {(10, 20): 0.4, (20, 30): 0.0, (30, 10): -0.2, (10, 99): 0.7, (30, 20): 0.1}
    B = link.bonus_matrix(bonus, pos, 3)
    expected = np.zeros((3, 3))
    expected[0, 1] = 0.4
    expected[2, 1] = 0.1
    assert np.array_equal(B, expected)
    assert np.array_equal(link.bonus_matrix(None, pos, 3), np.zeros((3, 3)))
    assert np.array_equal(link.bonus_matrix({}, pos, 3), np.zeros((3, 3)))


def test_junction_bonus_counts_only_time_adjacent_members_of_different_groups() -> None:
    first = np.array([0, 20, 40, 60])
    B = np.zeros((4, 4))  # [earlier, later]
    B[0, 1], B[0, 2], B[1, 2], B[2, 3], B[1, 3] = 0.2, 0.9, 0.3, 0.4, 0.8
    assert link.junction_bonus([0], [1], first, B) == pytest.approx(0.2)
    # 0 and 2 are not neighbours once 1 stands between them
    assert link.junction_bonus([0, 1], [2], first, B) == pytest.approx(0.3)
    assert link.junction_bonus([0, 2], [1], first, B) == pytest.approx(0.3)
    assert link.junction_bonus([1], [0, 2], first, B) == pytest.approx(0.3)
    # chain 0 | 1 2 | 3: the junctions are (0, 1) and (2, 3), not (1, 3)
    assert link.junction_bonus([0, 3], [1, 2], first, B) == pytest.approx(0.4)
    assert link.junction_bonus([0], [3], first, B) == 0.0
    # the order of the two groups does not matter
    assert link.junction_bonus([2], [0, 1], first, B) == pytest.approx(0.3)


# --------------------------------------------------------------------------- the rule


def _decide_dense(A, Bg, N, can, alive, t_abs, margin, floor, t0):
    """The rule with the rival written through the n x n x n array."""
    n = len(A)
    S = A + Bg
    seen = N >= 1
    Sc = np.where(can, S, -np.inf)
    Om = np.where(seen & alive[None, :], 0.0, -np.inf)
    rival = (Sc[:, None, :] + Om[None, :, :]).max(axis=2)
    with np.errstate(invalid="ignore"):
        if t0 is None:
            mg = np.where(np.isfinite(rival), S - rival, np.nan)
        else:
            mg = S - np.maximum(rival, t0)
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


def _random_step(rng: np.random.Generator, n: int):
    alive = rng.random(n) < 0.85
    A = rng.uniform(-0.2, 1.0, (n, n))
    A = (A + A.T) / 2
    np.fill_diagonal(A, -np.inf)
    A[~alive, :] = -np.inf
    A[:, ~alive] = -np.inf
    Bg = np.where(rng.random((n, n)) < 0.3, rng.uniform(0.0, 0.9, (n, n)), 0.0)
    Bg = np.maximum(Bg, Bg.T)
    N = np.where(rng.random((n, n)) < 0.25, rng.integers(1, 40, (n, n)), 0)
    N = np.maximum(N, N.T)
    np.fill_diagonal(N, 0)
    team = rng.integers(0, 2, n)
    can = (team[:, None] == team[None, :]) & (N <= 0) & alive[:, None] & alive[None, :]
    np.fill_diagonal(can, False)
    return A, Bg, N, can, alive


def test_the_column_wise_rival_equals_the_dense_formula_on_random_inputs() -> None:
    rng = np.random.default_rng(20261115)
    picked = 0
    cases = 0
    for n in (2, 3, 7, 30):
        for _ in range(40):
            A, Bg, N, can, alive = _random_step(rng, n)
            for t0, floor in ((0.5, 0.3), (None, None), (0.5, None), (None, 0.3)):
                margin = float(rng.choice([0.0, 0.05, 0.2, 0.6]))
                got = link.decide(A, Bg, N, can, alive, 0.5, margin, floor, t0)
                want = _decide_dense(A, Bg, N, can, alive, 0.5, margin, floor, t0)
                assert got[0] == want[0]
                assert np.array_equal(got[1], want[1])
                assert np.array_equal(got[2], want[2], equal_nan=True)
                picked += got[0] is not None
                cases += 1
    assert 0 < picked < cases


def test_decide_picks_the_most_similar_passing_pair_or_nothing() -> None:
    inf = np.inf
    A = np.array(
        [
            [-inf, 0.90, 0.10, 0.10],
            [0.90, -inf, 0.10, 0.10],
            [0.10, 0.10, -inf, 0.95],
            [0.10, 0.10, 0.95, -inf],
        ]
    )
    zero = np.zeros((4, 4))
    N = np.zeros((4, 4), int)
    alive = np.ones(4, bool)
    can = ~np.eye(4, dtype=bool)
    pick, S, margins = link.decide(A, zero, N, can, alive, 0.5, 0.2, 0.3, 0.5)
    assert pick == (2, 3)
    assert S[0, 1] == pytest.approx(0.90)
    assert margins[2, 3] == pytest.approx(0.45)
    pick, _, _ = link.decide(A, zero, N, can, alive, 0.5, 0.42, 0.3, 0.5)
    assert pick == (2, 3)
    pick, _, _ = link.decide(A, zero, N, can, alive, 0.5, 0.6, 0.3, 0.5)
    assert pick is None
    nothing = np.zeros((4, 4), bool)
    assert link.decide(A, zero, N, nothing, alive, 0.5, 0.0, 0.3, 0.5)[0] is None


# --------------------------------------------------------------------------- stage 1


def test_two_alike_pieces_merge_at_the_strict_margin_only_with_position_evidence() -> None:
    app = scene([piece(1, 0, 19), piece(2, 40, 59)], {(1, 2): 0.9})
    assert STRICT == pytest.approx(0.6)
    assert links_of(app, {(1, 2): 0.5}) == ((1, 2),)
    assert links_of(app, None) == ()
    assert links_of(app, {}) == ()
    assert links_of(app, {(1, 2): 0.1}) == ()  # 1.0 is short of t0 + margin = 1.1
    # one number sets how sure a merge must be
    assert links_of(app, None, margin=LOOSE) == ((1, 2),)
    assert links_of(app, {(1, 2): 0.5}, margin=1.0) == ()


def test_margin_none_means_the_margin_of_the_params() -> None:
    app = scene([piece(1, 0, 19), piece(2, 40, 59)], {(1, 2): 0.9})
    loose = link.LinkParams(margin=LOOSE)
    assert links_of(app, None, params=loose) == ((1, 2),)
    assert links_of(app, None, margin=STRICT, params=loose) == ()


@pytest.mark.parametrize(
    ("second", "cosine", "merges"),
    [
        (piece(2, 40, 59), 0.9, True),
        (piece(2, 15, 34), 0.9, False),  # frames 15-19 are shared
        (piece(2, 19, 38), 0.9, False),  # one shared frame is enough
        (piece(2, 40, 59, team="right"), 0.9, False),
        (piece(2, 40, 59, team=""), 0.9, False),
        (piece(2, 40, 59), 0.2, False),  # S = 1.15 passes, the appearance floor does not
        (piece(2, 40, 59), 0.4, True),
    ],
)
def test_shared_frames_other_team_and_low_cosine_never_merge(
    second: dict, cosine: float, merges: bool
) -> None:
    app = scene([piece(1, 0, 19), second], {(1, 2): cosine})
    for margin in (STRICT, LOOSE, 0.0):
        assert links_of(app, {(1, 2): 0.95}, margin=margin) == (((1, 2),) if merges else ())


def test_pieces_without_a_team_merge_with_each_other() -> None:
    app = scene([piece(1, 0, 19, team=""), piece(2, 40, 59, team="")], {(1, 2): 0.9})
    assert links_of(app, {(1, 2): 0.5}) == ((1, 2),)


def test_the_fused_similarity_must_reach_t_abs() -> None:
    params = link.LinkParams(t0=0.0, t_abs=0.5)
    weak = scene([piece(1, 0, 19), piece(2, 40, 59)], {(1, 2): 0.4})
    fair = scene([piece(1, 0, 19), piece(2, 40, 59)], {(1, 2): 0.6})
    assert links_of(weak, None, margin=0.1, params=params) == ()
    assert links_of(weak, {(1, 2): 0.2}, margin=0.1, params=params) == ((1, 2),)
    assert links_of(fair, None, margin=0.1, params=params) == ((1, 2),)


def _rival_scene(rival_rows: int | None = None) -> SimpleNamespace:
    # 2 and 3 are seen together after 1 has gone; both look like 1
    pieces = [piece(1, 0, 19), piece(2, 40, 59), piece(3, 40, 59, rows=rival_rows)]
    return scene(pieces, {(1, 2): 0.9, (1, 3): 0.8, (2, 3): 0.7})


def test_a_co_visible_rival_within_the_margin_blocks() -> None:
    app = _rival_scene()
    both = {(1, 2): 0.7, (1, 3): 0.5}  # S(1, 2) = 1.6, S(1, 3) = 1.3: 0.3 apart
    assert links_of(app, both) == ()
    # a rival that is far enough behind does not block: S(1, 3) = 0.8
    assert links_of(app, {(1, 2): 0.7}) == ((1, 2),)
    # with a small margin 1 takes 2; 3 shares frames with the new group and stays out
    log: list[dict] = []
    assert links_of(app, both, margin=0.05, log=log) == ((1, 2),)
    assert len(log) == 1
    assert min(log[0]["rival_g"], log[0]["rival_h"]) == pytest.approx(0.3)
    assert max(log[0]["rival_g"], log[0]["rival_h"]) == pytest.approx(1.1)
    # without 2, the same evidence joins 1 and 3
    pair = scene([piece(1, 0, 19), piece(3, 40, 59)], {(1, 3): 0.8})
    assert links_of(pair, {(1, 3): 0.5}) == ((1, 3),)


def test_a_merge_needs_the_mutual_best() -> None:
    # 2's best is 3, which a co-visible rival (4) blocks; 1's best is 2
    pieces = [piece(1, 0, 19), piece(2, 40, 59), piece(3, 80, 99), piece(4, 80, 99)]
    cosines = {(1, 2): 0.8, (2, 3): 0.9, (2, 4): 0.85, (1, 3): 0.6, (1, 4): 0.6, (3, 4): 0.7}
    app = scene(pieces, cosines)
    # S(1, 2) = 1.3 passes every threshold on its own, but 2 prefers 3 (1.4) and 4 (1.35)
    assert links_of(app, {(1, 2): 0.5, (2, 3): 0.5, (2, 4): 0.5}) == ()
    # when 2's later options are weaker than 1, the pair is mutual and merges
    assert links_of(app, {(1, 2): 0.5, (2, 3): 0.1, (2, 4): 0.1}) == ((1, 2),)


def test_short_pieces_take_no_part_in_stage_1() -> None:
    # 3 has three boxes: neither a candidate nor a rival
    app = _rival_scene(rival_rows=3)
    both = {(1, 2): 0.5, (1, 3): 0.9}
    result = link.link_core(app, both)
    assert result.links == ((1, 2),)
    assert result.state.solid.tolist() == [True, True, False]
    assert result.state.alive.tolist() == [True, False, False]
    # the same piece with five boxes is solid and blocks, as in the rival case
    assert links_of(_rival_scene(rival_rows=5), {(1, 2): 0.5, (1, 3): 0.5}) == ()
    # a short piece never merges, however good its evidence
    lone = scene([piece(1, 0, 19), piece(2, 40, 43)], {(1, 2): 0.95})
    assert links_of(lone, {(1, 2): 0.9}, margin=0.0) == ()
    # n_solid is a parameter
    assert links_of(lone, {(1, 2): 0.9}, params=link.LinkParams(n_solid=4)) == ((1, 2),)


def test_a_chain_gives_consecutive_links_in_time_order() -> None:
    # ids are not in time order: 3 is seen first, then 1, then 2
    pieces = [piece(1, 40, 59), piece(2, 80, 99), piece(3, 0, 19)]
    app = scene(pieces, {(1, 2): 0.9, (1, 3): 0.9, (2, 3): 0.9})
    bonus = {(3, 1): 0.6, (1, 2): 0.5, (3, 2): 0.9}
    log: list[dict] = []
    result = link.link_core(app, bonus, log=log)
    # (3, 2) has the largest S and merges first; 1 then joins between them, and the links
    # follow the chain 3, 1, 2
    assert result.links == ((3, 1), (1, 2))
    assert [entry["stage"] for entry in log] == [1, 1]
    assert sorted(log[0]["g"] + log[0]["h"]) == [2, 3]
    assert log[0]["s"] == pytest.approx(1.8)
    assert log[0]["app"] == pytest.approx(0.9)
    assert log[0]["bonus"] == pytest.approx(0.9)
    assert log[0]["rival_g"] == pytest.approx(1.3)
    assert log[0]["rival_h"] == pytest.approx(1.3)
    # the junction bonus of the group {3, 2} with 1 is the larger of (3, 1) and (1, 2)
    assert log[1]["bonus"] == pytest.approx(0.6)
    assert sorted(log[1]["g"] + log[1]["h"]) == [1, 2, 3]
    state = result.state
    rep = int(np.flatnonzero(state.alive)[0])
    assert [int(state.ids[i]) for i in state.members[rep]] == [3, 1, 2]
    assert state.frames[rep] == frozenset(range(0, 20)) | frozenset(range(40, 60)) | frozenset(
        range(80, 100)
    )


def test_position_speaks_only_at_the_junctions_of_a_chain() -> None:
    # 1, 2 and 3 follow each other; 3 has a bonus with the first only
    pieces = [piece(1, 0, 19), piece(2, 40, 59), piece(3, 80, 99)]
    app = scene(pieces, {(1, 2): 0.95, (1, 3): 0.6, (2, 3): 0.6})
    log: list[dict] = []
    # 1 and 2 merge first (S = 1.45); the group's junction with 3 is (2, 3), which has no
    # bonus, so the bonus of (1, 3) no longer counts: 3 stays out
    assert links_of(app, {(1, 2): 0.5, (1, 3): 0.45}, margin=LOOSE, log=log) == ((1, 2),)
    assert len(log) == 1
    # alone, 1 and 3 would have merged on it
    pair = scene([piece(1, 0, 19), piece(3, 80, 99)], {(1, 3): 0.6})
    assert links_of(pair, {(1, 3): 0.45}, margin=LOOSE) == ((1, 3),)


def test_prototypes_are_centred_whitened_and_pooled_by_rows() -> None:
    rng = np.random.default_rng(7)
    dim = 6
    pieces = [piece(1, 0, 9), piece(2, 20, 49), piece(3, 60, 69)]
    means = rng.normal(size=(3, dim))
    means[1] = means[0] + 0.01 * rng.normal(size=dim)
    means[2] = -means[0]
    app = make_app(pieces, means)
    app.centre = np.tile(rng.normal(size=dim) * 0.1, (3, 1))
    root = rng.normal(size=(dim, dim))
    app.T = root @ root.T + dim * np.eye(dim)
    before = {name: np.array(getattr(app, name)) for name in ("msum", "nrow", "centre", "T")}
    frames_before = tuple(app.frames)

    result = link.link_core(app, None, margin=LOOSE)
    assert result.links == ((1, 2),)
    X = result.state.X

    def proto(mean: np.ndarray, centre: np.ndarray) -> np.ndarray:
        v = app.T @ (mean - centre)
        return v / np.linalg.norm(v)

    pooled = (10 * means[0] + 30 * means[1]) / 40
    assert np.allclose(X[0], proto(pooled, app.centre[0]), atol=1e-12)
    assert np.allclose(X[1], proto(means[1], app.centre[1]), atol=1e-12)
    assert np.allclose(X[2], proto(means[2], app.centre[2]), atol=1e-12)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
    # the inputs are not changed
    for name, value in before.items():
        assert np.array_equal(getattr(app, name), value)
    assert tuple(app.frames) == frames_before
    assert app.frames[0] == frozenset(range(0, 10))
    # the memory layout of the whitening matrix is not part of the input
    app.T = np.asfortranarray(app.T)
    again = link.link_core(app, None, margin=LOOSE)
    assert again.links == result.links
    assert np.allclose(again.state.X, X, atol=1e-12)


def test_fewer_than_two_candidates_give_no_links() -> None:
    one = scene([piece(1, 0, 19)], {})
    result = link.link_core(one, {(1, 2): 0.9})
    assert result.links == ()
    assert result.state.ids.tolist() == [1]
    empty = make_app([], np.zeros((0, 4)))
    result = link.link_core(empty, None)
    assert result.links == ()
    assert len(result.state.ids) == 0


def test_bonus_for_unknown_ids_is_ignored_and_bad_inputs_are_named() -> None:
    app = scene([piece(1, 0, 19), piece(2, 40, 59)], {(1, 2): 0.9})
    assert links_of(app, {(1, 77): 0.9, (88, 2): 0.9}) == ()
    assert links_of(app, {(1, 77): 0.9, (1, 2): 0.5}) == ((1, 2),)
    app.nrow = app.nrow[:1]
    app.T = np.eye(5)
    with pytest.raises(ValueError, match="nrow has 1 entries for 2 candidates"):
        link.link_core(app, None)
    app = scene([piece(1, 0, 19), piece(2, 40, 59)], {(1, 2): 0.9})
    app.T = np.eye(5)
    with pytest.raises(ValueError, match="T .* must be"):
        link.link_core(app, None)


def _noisy_scene(seed: int) -> tuple[SimpleNamespace, dict[tuple[int, int], float]]:
    """Six people of two teams, each seen twice with a gap; prototypes are means of noisy unit
    embeddings around a random unit vector per person, centred on the team mean."""
    rng = np.random.default_rng(seed)
    dim = 48
    people = link.unit(rng.normal(size=(6, dim)))
    pieces = []
    msum = []
    for person in range(6):
        team = "left" if person < 3 else "right"
        for visit, (first, last) in enumerate(((0, 29), (60, 89))):
            rows = last - first + 1
            emb = link.unit(people[person] + 0.05 * rng.normal(size=(rows, dim)))
            pieces.append(piece(10 * (person + 1) + visit, first, last, team=team))
            msum.append(emb.sum(axis=0))
    app = make_app(pieces, np.zeros((len(pieces), dim)))
    app.msum = np.array(msum)
    means = app.msum / app.nrow[:, None]
    team = np.array([p["team"] for p in pieces])
    app.centre = np.stack([means[team == t].mean(axis=0) for t in team])
    returns = {(10 * (p + 1), 10 * (p + 1) + 1): 0.6 for p in range(6)}
    return app, returns


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noisy_returns_are_joined_person_by_person(seed: int) -> None:
    app, returns = _noisy_scene(seed)
    expected = set(returns)
    strict = links_of(app, returns)
    assert set(strict) == expected
    assert len(strict) == 6
    assert links_of(app, None) == ()
    assert set(links_of(app, None, margin=LOOSE)) == expected
    # a wrong hint of the position model does not join two different people
    wrong = {(10, 21): 0.6, (40, 51): 0.6}
    assert links_of(app, wrong) == ()
    # the same call gives the same answer
    assert links_of(app, returns) == strict
