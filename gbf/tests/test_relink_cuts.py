"""Tests for gbf.relink.cuts: cutting a track where the tracked person changes.

All data is synthetic and seeded. People are orthonormal prototype vectors (two different
people have cosine 0, two rows of one person about 0.99), every row is a prototype plus a
little noise, renormalised; a person walks 0.1 m per frame along x from the origin its stretch
is given. The input object carries the attributes the module reads (one entry per row).

The checks are decisions and numbers far from any threshold: the last bits of a matrix product
differ between machines.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gbf.relink import cuts as rc

DIM = 32
FIELDS = ("frame", "piece", "emb", "clean", "pitch_xy", "pitch_usable", "in_pitch")


def _people(seed: int, count: int) -> np.ndarray:
    """``count`` people: orthonormal rows."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(DIM, DIM)))
    return q[:count]


def _stretch(
    rng: np.random.Generator,
    piece: int,
    frames: Any,
    person: np.ndarray | None,
    *,
    clean: Any = True,
    origin: tuple[float, float] = (0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Rows of ``piece`` at ``frames`` showing ``person`` (None: rows without an embedding)."""
    frames = np.asarray(list(frames), dtype=np.int64)
    n = len(frames)
    if person is None:
        emb = np.full((n, DIM), np.nan)
    else:
        emb = person + 0.02 * rng.normal(size=(n, DIM))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return {
        "frame": frames,
        "piece": np.full(n, piece, dtype=np.int64),
        "emb": emb,
        "clean": np.broadcast_to(np.asarray(clean, dtype=bool), (n,)) & (person is not None),
        "pitch_xy": np.asarray(origin, dtype=np.float64) + np.outer(frames, (0.1, 0.0)),
        "pitch_usable": np.ones(n, dtype=bool),
        "in_pitch": np.ones(n, dtype=bool),
    }


def _scene(*stretches: dict[str, np.ndarray]) -> SimpleNamespace:
    return SimpleNamespace(**{k: np.concatenate([s[k] for s in stretches]) for k in FIELDS})


def _reordered(inp: SimpleNamespace, order: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(**{k: getattr(inp, k)[order] for k in FIELDS})


def _gap_scene(seed: int = 1, *, embeddings: bool = True) -> SimpleNamespace:
    """Piece 1 shows one person up to frame 39 and another, 20 m away, from frame 50 on;
    piece 2 shows one person over the same frames."""
    rng = np.random.default_rng(seed)
    a, b, c = _people(seed, 3) if embeddings else (None, None, None)
    return _scene(
        _stretch(rng, 1, range(0, 40), a),
        _stretch(rng, 1, range(50, 90), b, origin=(20.0, 10.0)),
        _stretch(rng, 2, range(0, 40), c, origin=(-30.0, -5.0)),
        _stretch(rng, 2, range(50, 90), c, origin=(-30.0, -5.0)),
    )


# --------------------------------------------------------------------------- cuts at a gap


def test_a_switch_at_a_gap_is_cut_at_the_first_frame_after_it() -> None:
    inp = _gap_scene()
    assert rc.detect_cuts(inp) == ((1, 50),)  # piece 2 has the same gap and one person

    table = rc.cut_table(inp)
    assert list(table.columns) == [
        "piece", "f0", "f1", "gap", "cos", "alt_d", "alt_r", "d_m", "logit", "prob",
    ]  # fmt: skip
    row = table.iloc[0]
    assert (row["piece"], row["f0"], row["f1"], row["gap"]) == (1, 39, 50, 10)
    assert abs(row["cos"]) < 0.1
    # medians of frames 37-39 at (3.8, 0) and of frames 50-52 at (25.1, 10)
    assert row["d_m"] == pytest.approx(np.hypot(21.3, 10.0), abs=1e-9)
    assert row["prob"] > 0.999
    assert row["prob"] == pytest.approx(1.0 / (1.0 + np.exp(-row["logit"])))
    # piece 2 is seen right before and right after the gap: no better origin, no continuation
    assert np.isnan(row["alt_d"])
    assert np.isnan(row["alt_r"])


def test_rows_without_a_track_are_ignored() -> None:
    rng = np.random.default_rng(5)
    stranger = _people(5, 4)[3]
    inp = _gap_scene()
    extra = _stretch(rng, -1, range(30, 60), stranger, origin=(5.0, 5.0))
    with_untracked = _scene(*({k: getattr(inp, k) for k in FIELDS}, extra))
    assert rc.detect_cuts(with_untracked) == ((1, 50),)


def test_without_embeddings_nothing_is_cut() -> None:
    inp = _gap_scene(embeddings=False)
    assert rc.detect_cuts(inp) == ()
    table = rc.cut_table(inp)
    assert len(table) == 0
    assert list(table.columns)[:3] == ["piece", "f0", "f1"]
    assert str(table["piece"].dtype) == "int64"
    assert str(table["prob"].dtype) == "float64"

    empty = SimpleNamespace(**{k: getattr(inp, k)[:0] for k in FIELDS})
    assert rc.detect_cuts(empty) == ()


def test_a_gap_without_embeddings_stays_below_the_cut_level() -> None:
    # Without embeddings the appearance features of a gap read "the same person" (cosine 1,
    # both margins at their clip), so only the gap and the distance on the pitch speak for a
    # cut. They must not reach the level alone: docs/design/m1-highlights.md and the report
    # note say that a piece without embeddings is not cut.
    params = rc.DEFAULT_PARAMS
    level = math.log(params.p_cut / (1.0 - params.p_cut))
    far = math.hypot(105.0 + 6.0, 68.0 + 6.0)  # corner to corner of the pitch and its margin
    candidates = pd.DataFrame(
        {
            "cos": [np.nan] * 3,
            "alt_d": [np.nan] * 3,
            "alt_r": [np.nan] * 3,
            "gap": [250, 50_000, 50_000],  # ten seconds, and over half an hour at 25 fps
            "d_m": [far, far, np.nan],
        }
    )
    logits = rc._logit(candidates, params)
    assert (logits < level).all()
    assert logits[0] < logits[1]  # a longer gap does count for more
    assert logits[2] < logits[1]  # and so does a known distance


def test_candidates_between_the_same_window_rows_are_one_decision_at_the_largest_gap() -> None:
    rng = np.random.default_rng(2)
    a, b = _people(2, 2)
    inp = _scene(
        _stretch(rng, 4, range(0, 40), a),
        _stretch(rng, 4, [42], None),  # a row nobody can be described by, between two gaps
        _stretch(rng, 4, range(50, 90), b, origin=(20.0, 0.0)),
    )
    # the boundaries 39 -> 42 (2 frames missing) and 42 -> 50 (7 missing) separate the same rows
    assert rc.detect_cuts(inp) == ((4, 50),)


# --------------------------------------------------------------------------- cuts inside a stretch


def test_a_switch_without_a_gap_is_cut_where_the_look_changes() -> None:
    rng = np.random.default_rng(3)
    a, b, c = _people(3, 3)
    inp = _scene(
        _stretch(rng, 7, range(0, 40), a),
        _stretch(rng, 7, range(40, 80), b),
        _stretch(rng, 8, range(0, 80), c, origin=(0.0, 20.0)),
    )
    assert rc.detect_cuts(inp) == ((7, 40),)
    table = rc.cut_table(inp)
    assert table["gap"].tolist() == [0]
    assert table["f0"].tolist() == [39]


def test_a_switch_is_placed_inside_its_plateau() -> None:
    """Rows 36-43 are not clean, so the boundaries 35 ... 43 separate the same clean rows; the
    person changes after row 37."""
    rng = np.random.default_rng(4)
    a, b = _people(4, 2)
    clean_a = np.arange(0, 38) < 36
    clean_b = np.arange(38, 80) > 43
    with_look = _scene(
        _stretch(rng, 7, range(0, 38), a, clean=clean_a),
        _stretch(rng, 7, range(38, 80), b, clean=clean_b),
    )
    assert rc.detect_cuts(with_look) == ((7, 38),)  # neither end of the plateau nor its middle

    # the same rows without any embedding inside the plateau: nothing says where, so the cut
    # falls at the plateau's first boundary
    blind = _scene(
        _stretch(rng, 7, range(0, 36), a),
        _stretch(rng, 7, range(36, 44), None),
        _stretch(rng, 7, range(44, 80), b),
    )
    assert rc.detect_cuts(blind) == ((7, 36),)


def test_better_origins_and_continuations_come_from_other_pieces() -> None:
    rng = np.random.default_rng(6)
    a, b = _people(6, 2)
    swap = [
        _stretch(rng, 1, range(0, 40), a),
        _stretch(rng, 1, range(40, 80), b),
    ]
    # piece 2 showed the second person until frame 30, piece 3 shows the first from frame 50 on
    inp = _scene(
        *swap,
        _stretch(rng, 2, range(0, 31), b, origin=(0.0, 20.0)),
        _stretch(rng, 3, range(50, 90), a, origin=(0.0, 40.0)),
    )
    table = rc.cut_table(inp)
    assert list(zip(table["piece"], table["f1"], strict=True)) == [(1, 40)]
    assert table["alt_d"][0] > 0.95  # the after side looks like the end of piece 2
    assert table["alt_r"][0] > 0.95  # the before side looks like the start of piece 3
    assert abs(table["cos"][0]) < 0.1

    # a piece still seen right after the cut frame is no origin, one seen right before no
    # continuation
    guarded = _scene(
        *swap,
        _stretch(rng, 2, range(0, 44), b, origin=(0.0, 20.0)),
        _stretch(rng, 3, range(36, 90), a, origin=(0.0, 40.0)),
    )
    table = rc.cut_table(guarded)
    assert list(zip(table["piece"], table["f1"], strict=True)) == [(1, 40)]
    assert np.isnan(table["alt_d"][0])
    assert np.isnan(table["alt_r"][0])


# --------------------------------------------------------------------------- suppression, threshold


def _three_people_scene() -> SimpleNamespace:
    """One piece: a person up to frame 39, another for five rows (frames 50-54), a third from
    frame 65 on."""
    rng = np.random.default_rng(7)
    a, b, c = _people(7, 3)
    return _scene(
        _stretch(rng, 3, range(0, 40), a),
        _stretch(rng, 3, range(50, 55), b, origin=(20.0, 0.0)),
        _stretch(rng, 3, range(65, 105), c, origin=(40.0, 0.0)),
    )


def test_two_candidates_closer_than_nms_rows_give_one_cut() -> None:
    inp = _three_people_scene()
    cuts = rc.detect_cuts(inp)
    assert len(cuts) == 1
    assert cuts[0] in {(3, 50), (3, 65)}
    # the same piece with a suppression distance below the five rows between them
    assert rc.detect_cuts(inp, replace(rc.DEFAULT_PARAMS, nms_rows=2)) == ((3, 50), (3, 65))


def test_the_probability_level_decides() -> None:
    inp = _gap_scene()
    table = rc.cut_table(inp, replace(rc.DEFAULT_PARAMS, p_cut=0.001))
    assert list(zip(table["piece"], table["f1"], strict=True)) == [(1, 50), (2, 50)]
    same_person = table.iloc[1]
    assert 0.001 < same_person["prob"] < 0.05
    assert same_person["cos"] > 0.98
    assert rc.detect_cuts(inp, replace(rc.DEFAULT_PARAMS, p_cut=0.05)) == ((1, 50),)
    assert rc.detect_cuts(inp, replace(rc.DEFAULT_PARAMS, p_cut=1 - 1e-12)) == ()


def test_a_candidate_exactly_at_the_threshold_is_cut() -> None:
    """Weights of zero make every logit the intercept; at p_cut 0.5 the threshold is 0."""
    inp = _gap_scene()
    flat = replace(rc.DEFAULT_PARAMS, w=(0.0, 0.0, 0.0, 0.0), p_cut=0.5)
    at = rc.cut_table(inp, replace(flat, b=0.0))
    assert list(zip(at["piece"], at["f1"], strict=True)) == [(1, 50), (2, 50)]
    assert at["logit"].tolist() == [0.0, 0.0]
    assert at["prob"].tolist() == [0.5, 0.5]
    assert rc.detect_cuts(inp, replace(flat, b=-1e-9)) == ()


# --------------------------------------------------------------------------- order, ids, inputs


def test_the_row_order_of_the_input_does_not_matter() -> None:
    rng = np.random.default_rng(8)
    a, b, c, d = _people(8, 4)
    inp = _scene(
        _stretch(rng, 1, range(0, 40), a),
        _stretch(rng, 1, range(50, 90), b, origin=(20.0, 10.0)),
        _stretch(rng, 5, range(0, 45), c, origin=(0.0, 20.0)),
        _stretch(rng, 5, range(45, 90), d, origin=(0.0, 20.0)),
        _stretch(rng, 2, range(0, 31), b, origin=(0.0, 30.0)),
    )
    expected = rc.cut_table(inp)
    assert list(zip(expected["piece"], expected["f1"], strict=True)) == [(1, 50), (5, 45)]
    for seed in (0, 1, 2):
        order = np.random.default_rng(seed).permutation(len(inp.frame))
        table = rc.cut_table(_reordered(inp, order))
        for name in ("piece", "f0", "f1", "gap"):
            assert table[name].tolist() == expected[name].tolist()
        for name in ("cos", "alt_d", "alt_r", "d_m", "logit", "prob"):
            np.testing.assert_allclose(table[name], expected[name], rtol=1e-9, atol=1e-12)
        assert rc.detect_cuts(_reordered(inp, order)) == ((1, 50), (5, 45))


def test_norm_cuts_sorts_and_drops_repeats() -> None:
    cuts = [[7, 30], (2, 40), [7, np.int64(10)], (np.int64(2), 40)]
    assert rc.norm_cuts(cuts) == ((2, 40), (7, 10), (7, 30))
    assert all(type(v) is int for cut in rc.norm_cuts(cuts) for v in cut)
    assert rc.norm_cuts(None) == ()
    assert rc.norm_cuts([]) == ()
    assert rc.norm_cuts(np.array([[3, 5], [1, 9]])) == ((1, 9), (3, 5))


def test_sub_ids_number_the_parts_of_a_piece_by_frame() -> None:
    assert rc.SUB_STRIDE == 10**7
    piece = np.array([5] * 10 + [-1] * 3 + [6] * 4, dtype=np.int64)
    frame = np.array(list(range(10)) + [2, 5, 8] + [3, 4, 5, 6], dtype=np.int64)
    cuts = [(5, 7), (5, 3)]  # listed out of order: the cut at frame 3 is the piece's first
    expected = np.array([5] * 3 + [10_000_005] * 4 + [20_000_005] * 3 + [-1] * 3 + [6] * 4)
    out = rc.sub_ids(piece, frame, cuts)
    assert out.dtype == np.int64
    assert out.tolist() == expected.tolist()
    assert piece.tolist() == [5] * 10 + [-1] * 3 + [6] * 4  # the input is not written to

    order = np.random.default_rng(0).permutation(len(piece))
    assert rc.sub_ids(piece[order], frame[order], cuts).tolist() == expected[order].tolist()
    assert rc.sub_ids(piece, frame, ()).tolist() == piece.tolist()
    assert rc.sub_ids(piece, frame, [(6, 0)]).tolist() == [5] * 10 + [-1] * 3 + [10_000_006] * 4

    with pytest.raises(ValueError, match="piece id"):
        rc.sub_ids(piece, frame, [(-1, 5)])
    with pytest.raises(ValueError, match="piece id"):
        rc.sub_ids(piece, frame, [(10**7, 5)])
    with pytest.raises(ValueError, match="alike"):
        rc.sub_ids(piece, frame[:-1], cuts)


def test_sub_ids_of_the_detected_cuts() -> None:
    inp = _three_people_scene()
    cuts = rc.detect_cuts(inp, replace(rc.DEFAULT_PARAMS, nms_rows=2))
    ids, rows = np.unique(rc.sub_ids(inp.piece, inp.frame, cuts), return_counts=True)
    assert ids.tolist() == [3, 10_000_003, 20_000_003]
    assert rows.tolist() == [40, 5, 40]


def test_params_have_a_json_form_and_refuse_nonsense() -> None:
    params = rc.DEFAULT_PARAMS
    assert params == rc.CutParams()
    stored = json.loads(json.dumps(params.as_dict()))
    assert stored["p_cut"] == 0.7
    assert stored["k_win"] == 20
    assert len(stored["w"]) == 4
    assert rc.CutParams(**stored) == params

    with pytest.raises(ValueError, match="p_cut") as caught:
        rc.CutParams(p_cut=1.0, nms_rows=-1, sd=(1.0, 0.0, 1.0, 1.0), mu=(1.0, 2.0))
    message = str(caught.value)
    assert "nms_rows" in message
    assert "sd must not hold a zero" in message
    assert "mu must be a tuple of 4" in message


def test_bad_inputs_are_named() -> None:
    inp = _gap_scene()
    twice = _scene(
        {k: getattr(inp, k) for k in FIELDS},
        {k: getattr(inp, k)[:1] for k in FIELDS},  # a second row of piece 1 in frame 0
    )
    with pytest.raises(ValueError, match="piece 1 has two rows in frame 0"):
        rc.detect_cuts(twice)

    short = SimpleNamespace(**{k: getattr(inp, k) for k in FIELDS})
    short.clean = inp.clean[:-1]
    short.pitch_xy = inp.pitch_xy[:, :1]
    with pytest.raises(ValueError, match="clean must have shape") as caught:
        rc.cut_table(short)
    assert "pitch_xy must have shape" in str(caught.value)
