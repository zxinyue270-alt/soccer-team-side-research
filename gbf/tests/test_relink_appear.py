"""Tests for gbf.relink.appear (appearance prototypes of the relink step).

Every clip is synthetic and seeded: a team has a kit direction, a person is a unit vector
near the kit of his team, and a row's embedding is the person plus small noise, scaled to
unit length. Numbers are compared with tolerances and decisions are kept far from their
thresholds: the last bits of a mean or a matrix product differ between machines.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from gbf.relink import appear
from gbf.relink.appear import Appearance, AppearParams

DIM = 16
KIT = {"left": np.eye(DIM)[0], "right": np.eye(DIM)[1]}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class _Clip:
    """The rows of a synthetic clip, added one sub-piece (or one part of it) at a time."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.frame: list[int] = []
        self.sub: list[int] = []
        self.emb: list[np.ndarray] = []
        self.clean: list[bool] = []
        self.role: list[str | None] = []
        self.team: list[str | None] = []

    def person(self, team: str, trait: int | None = None) -> np.ndarray:
        """A person of ``team``: a unit vector near the team's kit direction. With ``trait``
        (0 to 13) the person also stands out along an axis of his own, so that two team mates
        with different traits are a known distance apart."""
        if trait is None:
            return _unit(KIT[team] + 0.1 * self.rng.normal(size=DIM))
        own = 0.3 * np.eye(DIM)[2 + trait]
        return _unit(KIT[team] + own + 0.02 * self.rng.normal(size=DIM))

    def add(
        self,
        sub: int,
        frames: Iterable[int],
        person: np.ndarray,
        *,
        role: str | None | Sequence[str | None] = "player",
        team: str | None | Sequence[str | None] = None,
        clean: bool | Sequence[bool] = True,
        embedded: bool = True,
        noise: float = 0.03,
    ) -> np.ndarray:
        """One row per frame for sub-piece ``sub``; returns the rows' embeddings (NaN rows
        when ``embedded`` is false). ``role``, ``team`` and ``clean`` may be given per row."""
        frames = list(frames)
        n = len(frames)
        rows = _unit(person + noise * self.rng.normal(size=(n, DIM)))
        if not embedded:
            rows = np.full((n, DIM), np.nan)
        self.frame += frames
        self.sub += [sub] * n
        self.emb += list(rows)
        self.clean += [clean and embedded] * n if isinstance(clean, bool) else list(clean)
        self.role += [role] * n if role is None or isinstance(role, str) else list(role)
        self.team += [team] * n if team is None or isinstance(team, str) else list(team)
        return rows

    def inputs(self, order: np.ndarray | None = None) -> tuple[SimpleNamespace, np.ndarray]:
        """(inp, sub) as gbf.relink.appear.build reads them; ``order`` permutes or selects
        the rows."""
        n = len(self.frame)
        order = np.arange(n) if order is None else order
        inp = SimpleNamespace(
            frame=np.array(self.frame, np.int64)[order],
            emb=np.array(self.emb, np.float64).reshape(n, DIM)[order],
            clean=np.array(self.clean, bool)[order],
            role=np.array(self.role, dtype=object)[order],
            team=np.array(self.team, dtype=object)[order],
        )
        return inp, np.array(self.sub, np.int64)[order]


def _by_id(out: Appearance) -> dict[int, int]:
    return {int(p): i for i, p in enumerate(out.ids)}


def _two_teams(clip: _Clip, *, left: int = 3, right: int = 3, rows: int = 20) -> dict[str, Any]:
    """``left`` + ``right`` labelled players of ``rows`` clean rows each: sub-pieces 0.. are
    the left team, 100.. the right team. Returns the embeddings per team."""
    out: dict[str, Any] = {"left": [], "right": []}
    for team, first_id, count in (("left", 0, left), ("right", 100, right)):
        for k in range(count):
            person = clip.person(team)
            out[team].append(clip.add(first_id + k, range(rows), person, team=team))
    return out


# ------------------------------------------------------------------------- small parts


def test_unit_scales_the_last_axis_and_keeps_zero_vectors() -> None:
    v = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, -2.0]])
    np.testing.assert_allclose(appear.unit(v), [[0.6, 0.8], [0.0, 0.0], [0.0, -1.0]], atol=1e-15)
    np.testing.assert_allclose(appear.unit(np.array([0.0, 5.0, 12.0])), [0.0, 5 / 13, 12 / 13])


def test_params_carry_the_lab_values_and_are_frozen() -> None:
    assert appear.DEFAULT_PARAMS == AppearParams()
    assert appear.DEFAULT_PARAMS.as_dict() == {
        "min_clean": 5,
        "team_mean_min_rows": 50,
        "wccn_lambda": 0.3,
        "wccn_min_rows": 10,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        appear.DEFAULT_PARAMS.min_clean = 3  # type: ignore[misc]
    assert AppearParams(**appear.DEFAULT_PARAMS.as_dict()) == appear.DEFAULT_PARAMS


def test_params_name_every_impossible_value() -> None:
    bad = {"min_clean": -1, "team_mean_min_rows": 0, "wccn_lambda": 0.0, "wccn_min_rows": 2.5}
    with pytest.raises(ValueError, match="AppearParams") as info:
        AppearParams(**bad)
    for name in bad:
        assert name in str(info.value)
    loosest = AppearParams(min_clean=0, team_mean_min_rows=1, wccn_lambda=1, wccn_min_rows=1)
    assert loosest.min_clean == 0


# -------------------------------------------------------------------------- prototypes


def test_prototype_is_the_mean_of_the_clean_rows() -> None:
    clip = _Clip(seed=1)
    person, other = clip.person("left"), clip.person("right")
    good = clip.add(7, range(10, 16), person, team="left")
    clip.add(7, range(16, 19), other, team="left", clean=False)  # e.g. overlapped by a rival
    out = appear.build(*clip.inputs())

    assert out.ids.tolist() == [7]
    assert out.rows.tolist() == [9]  # all its boxes, clean or not
    assert (out.first.tolist(), out.last.tolist()) == ([10], [18])
    assert out.frames == (frozenset(range(10, 19)),)
    assert out.nrow.tolist() == [6.0]
    np.testing.assert_allclose(out.msum[0] / out.nrow[0], good.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.msum[0], good.sum(axis=0), rtol=0, atol=1e-12)


def test_low_piece_falls_back_to_all_rows_with_an_embedding() -> None:
    clip = _Clip(seed=2)
    person, other = clip.person("left"), clip.person("right")
    a = clip.add(3, range(0, 4), person, team="left")  # 4 clean rows: one short of min_clean
    b = clip.add(3, range(4, 8), other, team="left", clean=False)
    clip.add(3, range(8, 10), person, team="left", embedded=False)
    inp, sub = clip.inputs()

    out = appear.build(inp, sub)
    assert out.rows.tolist() == [10]
    assert out.nrow.tolist() == [8.0]  # the rows without an embedding never count
    everything = np.concatenate([a, b]).mean(axis=0)
    np.testing.assert_allclose(out.msum[0] / out.nrow[0], everything, rtol=0, atol=1e-12)

    out = appear.build(inp, sub, AppearParams(min_clean=4))  # 4 clean rows are enough now
    assert out.nrow.tolist() == [4.0]
    np.testing.assert_allclose(out.msum[0] / out.nrow[0], a.mean(axis=0), rtol=0, atol=1e-12)
    assert float(np.abs(a.mean(axis=0) - everything).max()) > 0.1  # the two answers differ


def test_sums_and_row_counts_pool_to_the_mean_of_both_pieces() -> None:
    clip = _Clip(seed=3)
    person = clip.person("left")
    a = clip.add(1, range(0, 12), person, team="left")
    b = clip.add(2, range(30, 37), person, team="left")
    out = appear.build(*clip.inputs())
    pooled = (out.msum[0] + out.msum[1]) / (out.nrow[0] + out.nrow[1])
    np.testing.assert_allclose(pooled, np.concatenate([a, b]).mean(axis=0), rtol=0, atol=1e-12)


# ----------------------------------------------------------- team means and the centre


def test_team_mean_needs_fifty_rows_else_the_global_centre() -> None:
    clip = _Clip(seed=4)
    rows = _two_teams(clip, left=3, right=2)  # 60 rows of left players, 40 of right players
    keeper = clip.add(50, range(20), clip.person("right"), role="goalkeeper", team="left")
    short = clip.add(51, range(3), clip.person("right"), team="left")  # low: 3 rows
    inp, sub = clip.inputs()
    left_mean = np.concatenate(rows["left"]).mean(axis=0)
    right_mean = np.concatenate(rows["right"]).mean(axis=0)
    assert float(np.abs(left_mean - right_mean).max()) > 0.5

    # left has a mean (goalkeepers and low pieces stay out of it); right has too few rows, so
    # its pieces are centred on the global centre: the mean of the known team means
    out = appear.build(inp, sub)
    at = _by_id(out)
    assert out.team[at[0]] == "left"
    assert out.team[at[100]] == "right"
    assert out.team[at[50]] == "left"
    for i in range(len(out.ids)):
        np.testing.assert_allclose(out.centre[i], left_mean, rtol=0, atol=1e-12)

    # from 40 rows on, both teams have a mean and every piece is centred on its own team's
    out = appear.build(inp, sub, AppearParams(team_mean_min_rows=40))
    for i, team in enumerate(out.team):
        expected = left_mean if team == "left" else right_mean
        np.testing.assert_allclose(out.centre[i], expected, rtol=0, atol=1e-12)

    # 60 rows are not enough for 61: no team mean, the centre is the mean of every embedding
    out = appear.build(inp, sub, AppearParams(team_mean_min_rows=61))
    everything = np.concatenate([*rows["left"], *rows["right"], keeper, short]).mean(axis=0)
    assert out.team[_by_id(out)[100]] == "right"  # a label stays without a team mean
    for i in range(len(out.ids)):
        np.testing.assert_allclose(out.centre[i], everything, rtol=0, atol=1e-12)


def test_global_centre_is_the_mean_of_the_two_team_means() -> None:
    clip = _Clip(seed=5)
    rows = _two_teams(clip, left=4, right=3)  # 80 and 60 rows: unequal on purpose
    clip.add(200, range(20), clip.person("left"), team=None, role="other")
    clip.add(201, range(20), clip.person("left"), team="away")  # not a side: nearer team
    inp, sub = clip.inputs()
    left_mean = np.concatenate(rows["left"]).mean(axis=0)
    right_mean = np.concatenate(rows["right"]).mean(axis=0)

    out = appear.build(inp, sub)
    at = _by_id(out)
    assert out.team[at[200]] == "left"
    assert out.team[at[201]] == "left"
    np.testing.assert_allclose(out.centre[at[200]], left_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.centre[at[100]], right_mean, rtol=0, atol=1e-12)


def test_unlabelled_piece_takes_the_nearer_team() -> None:
    clip = _Clip(seed=6)
    rows = _two_teams(clip)
    clip.add(200, range(30, 45), clip.person("right"), team=None)
    clip.add(201, range(30, 45), clip.person("left"), team=None)
    swapped = clip.add(202, range(30, 45), clip.person("right"), team="left")  # the label wins
    clip.add(203, range(30, 32), clip.person("right"), team=None, clean=False)  # low, 2 rows
    inp, sub = clip.inputs()
    # a team mean is made of the labelled players only: 202 is one of the left team's
    right_mean = np.concatenate(rows["right"]).mean(axis=0)
    left_mean = np.concatenate([*rows["left"], swapped]).mean(axis=0)

    out = appear.build(inp, sub)
    at = _by_id(out)
    assert [out.team[at[p]] for p in (200, 201, 202, 203)] == ["right", "left", "left", "right"]
    np.testing.assert_allclose(out.centre[at[200]], right_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.centre[at[201]], left_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.centre[at[202]], left_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.centre[at[203]], right_mean, rtol=0, atol=1e-12)


def test_unlabelled_piece_with_one_or_no_team_mean() -> None:
    clip = _Clip(seed=7)
    _two_teams(clip, left=3, right=2)  # only left reaches 50 rows
    clip.add(200, range(30, 45), clip.person("right"), team=None)
    inp, sub = clip.inputs()

    out = appear.build(inp, sub)
    assert out.team[_by_id(out)[200]] == "left"  # the only team with a mean

    out = appear.build(inp, sub, AppearParams(team_mean_min_rows=1000))
    at = _by_id(out)
    assert out.team[at[200]] == ""  # no team mean at all: no appearance team
    assert out.team[at[0]] == "left"
    np.testing.assert_allclose(out.centre[at[200]], out.centre[at[0]], rtol=0, atol=0)


# --------------------------------------------------------------------------- whitening


def _whitening_clip() -> tuple[_Clip, list[np.ndarray]]:
    clip = _Clip(seed=8)
    eligible = []
    for k in range(4):
        eligible.append(clip.add(k, range(15), clip.person("left"), team="left"))
    # 11 clean rows enter, its 3 other rows do not
    person = clip.person("right")
    eligible.append(clip.add(4, range(11), person, team="right"))
    clip.add(4, range(11, 14), clip.person("left"), team="right", clean=False, noise=0.3)
    # 9 clean rows: one short of wccn_min_rows; noisy, so that it would change T
    clip.add(5, range(9), clip.person("right"), team="right", noise=0.3)
    # 12 rows with an embedding but only 2 clean ones: low, never enters
    clip.add(
        6, range(12), clip.person("right"), team="right", noise=0.3, clean=[True] * 2 + [False] * 10
    )
    return clip, eligible


def test_whitening_equals_the_formula() -> None:
    clip, eligible = _whitening_clip()
    inp, sub = clip.inputs()
    out = appear.build(inp, sub)

    D = np.concatenate([rows - rows.mean(axis=0) for rows in eligible])
    Sw = D.T @ D / len(D)
    w, V = np.linalg.eigh(Sw)
    expected = (V / np.sqrt(np.clip(w, 0, None) + 0.3 * w.mean())) @ V.T
    np.testing.assert_allclose(out.T, expected, rtol=1e-9, atol=1e-9)

    # the same without an eigen-decomposition: T is the inverse square root of the
    # covariance shrunk by 0.3 times its mean eigenvalue
    shrunk = Sw + 0.3 * np.trace(Sw) / DIM * np.eye(DIM)
    np.testing.assert_allclose(out.T, out.T.T, rtol=0, atol=1e-9)
    np.testing.assert_allclose(out.T @ shrunk @ out.T, np.eye(DIM), rtol=0, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(out.T) > 0)

    # another shrinkage, another matrix
    lam = 1.0
    out = appear.build(inp, sub, AppearParams(wccn_lambda=lam))
    shrunk = Sw + lam * np.trace(Sw) / DIM * np.eye(DIM)
    np.testing.assert_allclose(out.T @ shrunk @ out.T, np.eye(DIM), rtol=0, atol=1e-9)


def test_whitening_leaves_out_short_and_low_pieces() -> None:
    clip, _ = _whitening_clip()
    inp, sub = clip.inputs()
    base = appear.build(inp, sub).T
    with_nine = appear.build(inp, sub, AppearParams(wccn_min_rows=9)).T  # piece 5 enters
    assert float(np.abs(with_nine - base).max()) > 0.05 * float(np.abs(base).max())
    # the low piece 6 has 12 prototype rows and stays out all the same: T is the one of the
    # clip without it
    keep = np.flatnonzero(sub != 6)
    without = appear.build(*clip.inputs(keep)).T
    np.testing.assert_allclose(without, base, rtol=1e-12, atol=1e-12)
    # and the row order of the table does not matter
    again = appear.build(*clip.inputs(clip.rng.permutation(len(sub)))).T
    np.testing.assert_allclose(again, base, rtol=1e-9, atol=1e-9)


def test_whitening_is_the_identity_without_a_long_piece() -> None:
    clip = _Clip(seed=9)
    for k in range(8):
        clip.add(k, range(9), clip.person("left"), team="left")  # 9 rows each: none enters
    out = appear.build(*clip.inputs())
    assert len(out.ids) == 8
    np.testing.assert_array_equal(out.T, np.eye(DIM))


def test_centring_and_whitening_tell_team_mates_apart() -> None:
    clip = _Clip(seed=10)
    people = {team: [clip.person(team, trait=k) for k in range(4)] for team in KIT}
    sub = 0
    owner = {}
    for team, persons in people.items():
        for k, person in enumerate(persons):
            for start in (0, 100):  # two pieces per person
                clip.add(sub, range(start, start + 20), person, team=team)
                owner[sub] = (team, k)
                sub += 1
    out = appear.build(*clip.inputs())
    proto = out.msum / out.nrow[:, None]
    raw = appear.unit(proto)
    white = appear.unit((proto - out.centre) @ out.T.T)  # what the linker compares
    same, mates, raw_mates = [], [], []
    for i, a in enumerate(out.ids.tolist()):
        for j, b in enumerate(out.ids.tolist()):
            if i < j and owner[a] == owner[b]:
                same.append(float(white[i] @ white[j]))
            elif i < j and owner[a][0] == owner[b][0]:
                mates.append(float(white[i] @ white[j]))
                raw_mates.append(float(raw[i] @ raw[j]))
    assert len(same) == 8
    assert len(mates) == 48
    assert min(raw_mates) > 0.8  # the kit dominates the raw cosine of two team mates
    assert min(same) > 0.8
    assert max(mates) < 0.2


# -------------------------------------------------------------------------- candidates


def test_referees_and_pieces_without_embedding_are_not_candidates() -> None:
    clip = _Clip(seed=11)
    _two_teams(clip)
    person = clip.person("left")
    clip.add(300, range(40, 60), person, role="referee")
    clip.add(301, range(40, 60), person, team="left", embedded=False)
    clip.add(302, range(40, 60), person, role="goalkeeper", team="left")
    clip.add(303, range(40, 60), person, role=None)
    clip.add(304, range(40, 60), person, role="other")
    clip.add(305, range(40, 48), person, role=["referee"] * 5 + ["player"] * 3)
    clip.add(306, range(40, 48), person, role=["referee"] * 3 + ["player"] * 5)
    clip.add(307, range(40, 48), person, role=[None] * 7 + ["referee"])  # None is no label
    clip.add(-1, range(40, 60), person, team="left")  # rows of no piece
    out = appear.build(*clip.inputs())
    assert out.ids.tolist() == [0, 1, 2, 100, 101, 102, 302, 303, 304, 306]
    assert out.ids.dtype == np.int64
    assert out.msum.shape == (10, DIM)
    assert out.centre.shape == (10, DIM)
    assert out.team.dtype == object


def test_referees_do_not_enter_a_team_mean_but_enter_the_fallback_centre() -> None:
    clip = _Clip(seed=12)
    rows = _two_teams(clip, left=3, right=3)
    referee = clip.add(300, range(20), clip.person("right"), role="referee", team="left")
    inp, sub = clip.inputs()
    out = appear.build(inp, sub)
    left_mean = np.concatenate(rows["left"]).mean(axis=0)
    np.testing.assert_allclose(out.centre[0], left_mean, rtol=0, atol=1e-12)
    out = appear.build(inp, sub, AppearParams(team_mean_min_rows=1000))
    everything = np.concatenate([*rows["left"], *rows["right"], referee]).mean(axis=0)
    np.testing.assert_allclose(out.centre[0], everything, rtol=0, atol=1e-12)


def test_label_ties_go_to_the_smaller_string() -> None:
    clip = _Clip(seed=13)
    rows = _two_teams(clip)
    person = clip.person("right")
    tie = ["right"] * 4 + ["left"] * 4
    tied = clip.add(200, range(8), person, team=tie)  # "left" < "right"
    more = clip.add(201, range(9), person, team=["right"] * 5 + ["left"] * 4)  # the majority
    one = clip.add(202, range(8), person, team=[None] * 7 + ["right"])  # None is no label
    both = clip.add(203, range(8), person, role=["referee"] * 4 + ["player"] * 4, team="right")
    clip.add(204, range(8), person, role=["referee"] * 4 + ["goalkeeper"] * 4, team="right")
    clip.add(205, range(8), person, role=["staff"] * 4 + ["referee"] * 4, team="right")
    out = appear.build(*clip.inputs())
    at = _by_id(out)
    assert out.team[at[200]] == "left"
    assert out.team[at[201]] == "right"
    assert out.team[at[202]] == "right"
    assert 203 in at  # "player" < "referee": a player, so a candidate
    assert 204 in at  # "goalkeeper" < "referee"
    assert 205 not in at  # "referee" < "staff": a referee
    # the team means show the same decisions: 200 is a left player; 201, 202 and the tied
    # player 203 are right players; the goalkeeper 204 and the referee 205 are in neither
    left_mean = np.concatenate([*rows["left"], tied]).mean(axis=0)
    right_mean = np.concatenate([*rows["right"], more, one, both]).mean(axis=0)
    np.testing.assert_allclose(out.centre[at[0]], left_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.centre[at[100]], right_mean, rtol=0, atol=1e-12)


def test_candidates_ascend_by_id_whatever_the_row_order() -> None:
    clip = _Clip(seed=14)
    late, early = clip.person("left"), clip.person("left")
    clip.add(10_000_004, range(50, 70), late, team="left")  # a sub-piece after a cut
    clip.add(12, range(20, 40), early, team="left")
    clip.add(4, range(0, 50), late, team="left")
    clip.add(100, range(0, 60), clip.person("right"), team="right")
    out = appear.build(*clip.inputs())
    assert out.ids.tolist() == [4, 12, 100, 10_000_004]
    assert out.first.tolist() == [0, 20, 0, 50]
    assert out.last.tolist() == [49, 39, 59, 69]
    assert out.rows.tolist() == [50, 20, 60, 20]
    assert [len(f) for f in out.frames] == [50, 20, 60, 20]

    order = clip.rng.permutation(len(clip.frame))
    mixed = appear.build(*clip.inputs(order))
    for name in ("ids", "rows", "first", "last"):
        np.testing.assert_array_equal(getattr(mixed, name), getattr(out, name))
    assert mixed.frames == out.frames
    assert mixed.team.tolist() == out.team.tolist() == ["left", "left", "right", "left"]
    # rows of a piece are taken in frame order, so the prototypes do not move
    np.testing.assert_allclose(mixed.msum, out.msum, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(mixed.nrow, out.nrow)
    np.testing.assert_allclose(mixed.centre, out.centre, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mixed.T, out.T, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------- empty and refusals


def _assert_empty(out: Appearance) -> None:
    assert out.ids.shape == out.rows.shape == out.first.shape == out.last.shape == (0,)
    assert out.ids.dtype == out.rows.dtype == out.first.dtype == out.last.dtype == np.int64
    assert out.frames == ()
    assert out.team.shape == (0,)
    assert out.msum.shape == out.centre.shape == (0, DIM)
    assert out.nrow.shape == (0,)


def test_empty_input(recwarn: pytest.WarningsRecorder) -> None:
    out = appear.build(*_Clip(seed=15).inputs())  # no row at all
    _assert_empty(out)
    np.testing.assert_array_equal(out.T, np.eye(DIM))

    clip = _Clip(seed=15)
    clip.add(1, range(20), clip.person("left"), team="left", embedded=False)
    clip.add(2, range(20), clip.person("left"), team="left", embedded=False)
    out = appear.build(*clip.inputs())  # rows, but no embedding
    _assert_empty(out)
    np.testing.assert_array_equal(out.T, np.eye(DIM))
    assert len(recwarn) == 0  # no mean of nothing


def test_rows_of_no_piece_give_no_candidate(recwarn: pytest.WarningsRecorder) -> None:
    clip = _Clip(seed=16)
    clip.add(-1, range(20), clip.person("left"), team="left")
    clip.add(-1, range(20), clip.person("right"), team="right")
    out = appear.build(*clip.inputs())
    _assert_empty(out)
    np.testing.assert_array_equal(out.T, np.eye(DIM))
    assert len(recwarn) == 0


def test_only_referees_give_no_candidate_but_a_whitening() -> None:
    clip = _Clip(seed=17)
    clip.add(1, range(20), clip.person("left"), role="referee")
    clip.add(2, range(20), clip.person("right"), role="referee")
    out = appear.build(*clip.inputs())
    _assert_empty(out)
    assert not np.array_equal(out.T, np.eye(DIM))


def test_build_names_every_mismatched_array() -> None:
    clip = _Clip(seed=18)
    clip.add(1, range(6), clip.person("left"), team="left")
    inp, sub = clip.inputs()
    inp.clean = inp.clean[:-1]
    inp.emb = inp.emb[:, 0]
    with pytest.raises(ValueError, match=r"clean has shape \(5,\).*emb has shape \(6,\)") as info:
        appear.build(inp, sub[:4].astype(float))
    assert "sub has shape (4,)" in str(info.value)
    assert "sub has dtype float64" in str(info.value)


def test_result_is_frozen() -> None:
    clip = _Clip(seed=19)
    clip.add(1, range(6), clip.person("left"), team="left")
    inp, sub = clip.inputs()
    out = appear.build(inp, sub)
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.ids = np.zeros(0, np.int64)  # type: ignore[misc]
    for name in ("ids", "rows", "first", "last", "msum", "nrow", "centre", "T"):  # new arrays
        assert not any(np.shares_memory(getattr(out, name), a) for a in (inp.emb, inp.frame, sub))
    names = [f.name for f in dataclasses.fields(out)]
    assert names == [
        "ids",
        "rows",
        "first",
        "last",
        "frames",
        "team",
        "msum",
        "nrow",
        "centre",
        "T",
    ]
