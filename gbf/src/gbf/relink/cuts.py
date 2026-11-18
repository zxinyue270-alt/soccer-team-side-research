"""Relink, step 1: cut a track where the tracked person changes.

"Cut" here means cutting a TRACK. Inside one gbf piece the plugin's track sometimes moves onto
another person without leaving the picture (a tracker switch); this module finds those places and
returns them as cuts ``(piece, frame)``, ``frame`` being the first frame of the new sub-piece. It
has nothing to do with the camera cuts of gbf.ingest.cuts.

It is a port of the switch detector of the re-identification lab (reid_lab, outside the
repository) with that lab's frozen model: given the same rows, embeddings and clean flags it
gives the lab's cuts. Nothing here reads an annotation.

How it works. The rows of a piece are taken in frame order; a boundary lies between two
consecutive rows at frames f0 < f1, with gap = f1 - f0 - 1 missing frames. Candidate boundaries:

- "gap": every boundary with a gap of at least ``gap_min`` frames;
- "app": inside a gap-free stretch, a local minimum (below ``cos_cand``) of the cosine between
  the mean embeddings of the ``k_win`` usable rows before and after the boundary. Boundaries
  between the same two usable rows share that value (a plateau); the candidate is placed where
  the rows of the plateau stop looking like the rows before and start looking like the rows
  after.

Features of a candidate inside the stretch of its piece that is still in one part:

- cos: cosine between the mean embeddings of the ``k_win`` clean rows before and after it (with
  fewer than ``m_min`` clean rows on a side: its rows with an embedding);
- mm: the larger of cos - alt_d and cos - alt_r, each clipped to +-``margin_clip``. alt_d is the
  best cosine between the after side and the last rows of any other piece that was seen in the
  ``alt_frames`` before f1 and is not seen within ``alt_guard`` frames after it (a better origin
  of the after side); alt_r is the best cosine between the before side and the first rows of
  any other piece that appears in the ``alt_frames`` after f0 and is not seen within
  ``alt_guard`` frames before it (a better continuation of the before side);
- lgap: log(1 + gap);
- ld: log(1 + metres on the pitch between the last position before and the first after).

A logistic regression on these four gives the probability that the two sides are different
people. Each piece is segmented greedily: its most probable candidate is cut when the
probability is at least ``p_cut``, the features of the remaining candidates are recomputed
inside the two parts (a window never reaches across an earlier cut), and so on. Candidates
between the same two window rows are one decision, placed at the largest gap, and no cut is made
within ``nms_rows`` rows of an earlier cut. The alternatives (alt_d, alt_r) always come from the
whole other pieces, so the cuts of a piece do not depend on the cuts of another.

The input is any object with these attributes, one entry per row of the track table of one
single-shot clip (gbf.relink.inputs.ClipInputs has them): ``frame`` (int), ``piece`` (int, the
gbf piece id, negative = no track), ``emb`` (rows x D float64 unit embeddings, an all-NaN row =
no embedding), ``clean`` (bool: a row fit to describe a person by), ``pitch_xy`` (rows x 2,
metres, NaN where there is none), ``pitch_usable`` and ``in_pitch`` (bool).

The numbers of :class:`CutParams` are the lab's: the candidate rule and the windows were set by
judgement, the model was fitted with annotations on 13 SN-GSR clips at 1280x720 and 25 fps
(PRTReid embeddings). None is validated on other footage, and the constants counted in frames
or rows are not rescaled at another frame rate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

SUB_STRIDE = 10**7
"""Id stride of the sub-pieces after a cut: the rows of a piece from its j-th cut frame on
(cuts by frame, j from 0) get the id ``piece + SUB_STRIDE * (j + 1)``."""

_TABLE_DTYPES: dict[str, type] = {
    "piece": np.int64,
    "f0": np.int64,
    "f1": np.int64,
    "gap": np.int64,
    "cos": np.float64,
    "alt_d": np.float64,
    "alt_r": np.float64,
    "d_m": np.float64,
    "logit": np.float64,
    "prob": np.float64,
}
"""Columns of :func:`cut_table`."""


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class CutParams:
    """Settings of the switch detector (the lab's frozen values; none is validated elsewhere).

    Attributes:
        k_win: rows per side of the before/after descriptor.
        m_min: rows needed for a clean-row descriptor (else the side's rows with an embedding).
        gap_min: frames; a gap this long is always a candidate.
        min_side_rows: rows an appearance candidate keeps on each side.
        cos_cand: appearance minima below this are candidates inside a gap-free stretch.
        alt_frames: frames; other origins / continuations are looked for this far away (10 s at
            25 fps).
        alt_guard: frames; ... among the pieces not seen this close on the other side.
        nms_rows: rows; no second cut this close to a cut.
        margin_clip: clip of the two cosine margins.
        pitch_rows: placed rows per side for the pitch distance across the boundary.
        pitch_within: frames from the boundary these rows may lie.
        p_cut: a candidate is cut when P(the two sides are different people) is at least this.
        mu, sd, w, b: the logistic model on (cos, mm, lgap, ld): means and standard deviations
            of the inputs, weights and intercept.
    """

    k_win: int = 20
    m_min: int = 3
    gap_min: int = 2
    min_side_rows: int = 5
    cos_cand: float = 0.9
    alt_frames: int = 250
    alt_guard: int = 5
    nms_rows: int = 10
    margin_clip: float = 0.3
    pitch_rows: int = 3
    pitch_within: int = 12
    p_cut: float = 0.7
    mu: tuple[float, float, float, float] = (
        0.9444822950244501,
        0.11702873894030795,
        1.7169157719019472,
        0.7297202806447308,
    )
    sd: tuple[float, float, float, float] = (
        0.065096638607436,
        0.12347787727581538,
        0.7641615906398944,
        0.5140510070715412,
    )
    w: tuple[float, float, float, float] = (
        -1.3605814098310482,
        -0.6699718339482522,
        0.32168728401658697,
        0.3046642825081575,
    )
    b: float = -3.3411804792796573

    def __post_init__(self) -> None:
        for name in ("mu", "sd", "w"):  # as_dict() stores them as lists
            if isinstance(getattr(self, name), list):
                object.__setattr__(self, name, tuple(getattr(self, name)))
        problems: list[str] = []
        for name, least in (
            ("k_win", 1),
            ("m_min", 1),
            ("gap_min", 0),
            ("min_side_rows", 1),
            ("alt_frames", 0),
            ("alt_guard", 0),
            ("nms_rows", 0),  # a negative value would cut the same boundary for ever
            ("pitch_rows", 1),
            ("pitch_within", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                problems.append(f"{name} must be an int >= {least}, got {value!r}")
        for name in ("cos_cand", "margin_clip", "b"):
            if not _number(getattr(self, name)):
                problems.append(f"{name} must be a finite number, got {getattr(self, name)!r}")
        if _number(self.margin_clip) and self.margin_clip < 0:
            problems.append(f"margin_clip must not be negative, got {self.margin_clip!r}")
        if not _number(self.p_cut) or not 0 < self.p_cut < 1:
            problems.append(f"p_cut must be a number in (0, 1), got {self.p_cut!r}")
        for name in ("mu", "sd", "w"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != 4 or not all(map(_number, value)):
                problems.append(f"{name} must be a tuple of 4 finite numbers, got {value!r}")
            elif name == "sd" and 0 in value:
                problems.append(f"sd must not hold a zero, got {value!r}")
        if problems:
            raise ValueError("CutParams: " + "; ".join(problems))

    def as_dict(self) -> dict[str, Any]:
        """JSON form for reports and receipts."""
        return {name: list(v) if isinstance(v, tuple) else v for name, v in asdict(self).items()}


DEFAULT_PARAMS = CutParams()


# --------------------------------------------------------------------------- ids


def norm_cuts(cuts: Iterable[Iterable[int]] | None) -> tuple[tuple[int, int], ...]:
    """Cuts as a sorted tuple of unique ``(piece, frame)`` pairs of ints."""
    return tuple(sorted({(int(p), int(f)) for p, f in (() if cuts is None else cuts)}))


def sub_ids(
    piece: np.ndarray, frame: np.ndarray, cuts: Iterable[Iterable[int]] | None
) -> np.ndarray:
    """The sub-piece id of every row after ``cuts``.

    The rows of a piece before its first cut keep the piece id; its rows from the j-th cut
    frame on (cuts of the piece sorted by frame, j from 0) get ``piece + SUB_STRIDE * (j + 1)``.
    Rows of uncut pieces and rows without a track (negative id) are unchanged. Returns int64.
    """
    piece = np.asarray(piece)
    frame = np.asarray(frame)
    if piece.ndim != 1 or piece.shape != frame.shape:
        raise ValueError(
            f"sub_ids: piece and frame must be 1-D and alike, got {piece.shape} and {frame.shape}"
        )
    by: dict[int, list[int]] = {}
    for p, f in norm_cuts(cuts):
        by.setdefault(p, []).append(f)
    bad = sorted(p for p in by if not 0 <= p < SUB_STRIDE)
    if bad:
        raise ValueError(
            f"sub_ids: a cut needs a piece id in [0, {SUB_STRIDE}), got {bad[:5]} "
            f"({len(bad)} in all)"
        )
    out = piece.astype(np.int64)
    for p, frames in by.items():
        for j, f in enumerate(sorted(frames)):
            out[(piece == p) & (frame >= f)] = p + SUB_STRIDE * (j + 1)
    return out


# --------------------------------------------------------------------------- pieces


class _Piece:
    """Rows of one piece in frame order."""

    __slots__ = ("pid", "rows", "frame", "E", "has", "clean", "xy", "placed")

    def __init__(self, pid: int, rows: np.ndarray, table: _Rows) -> None:
        self.pid = int(pid)
        self.rows = rows
        self.frame = table.frame[rows]
        self.E = np.nan_to_num(table.emb[rows])
        self.has = ~np.isnan(table.emb[rows, 0])
        self.clean = table.clean[rows] & self.has
        self.xy = table.xy[rows]
        self.placed = table.usable[rows] & table.in_pitch[rows] & ~np.isnan(self.xy[:, 0])

    def __len__(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class _Rows:
    """The arrays this module reads from its input, checked."""

    frame: np.ndarray
    piece: np.ndarray
    emb: np.ndarray
    clean: np.ndarray
    xy: np.ndarray
    usable: np.ndarray
    in_pitch: np.ndarray


def _rows_of(inp: Any) -> _Rows:
    frame = np.asarray(inp.frame)
    piece = np.asarray(inp.piece)
    # C order and float64 are what the lab holds; both calls change nothing then. The layout
    # matters: a matrix product further down takes rows of this array.
    emb = np.ascontiguousarray(inp.emb, dtype=np.float64)
    xy = np.asarray(inp.pitch_xy, dtype=np.float64)
    flags = {
        name: np.asarray(getattr(inp, name), dtype=bool)
        for name in ("clean", "pitch_usable", "in_pitch")
    }
    problems: list[str] = []
    n = len(frame)
    for name, array in (("frame", frame), ("piece", piece), *flags.items()):
        if array.shape != (n,):
            problems.append(f"{name} must have shape ({n},), got {array.shape}")
    for name, array in (("frame", frame), ("piece", piece)):
        if not np.issubdtype(array.dtype, np.integer):
            problems.append(f"{name} must hold integers, got {array.dtype}")
    if emb.ndim != 2 or len(emb) != n or emb.shape[1] < 1:
        problems.append(f"emb must have shape ({n}, D) with D >= 1, got {emb.shape}")
    if xy.shape != (n, 2):
        problems.append(f"pitch_xy must have shape ({n}, 2), got {xy.shape}")
    if problems:
        raise ValueError("relink cuts: " + "; ".join(problems))
    return _Rows(frame, piece, emb, flags["clean"], xy, flags["pitch_usable"], flags["in_pitch"])


def _piece_table(table: _Rows) -> dict[int, _Piece]:
    """{piece id: its rows in frame order}, ids >= 0 in ascending order."""
    keep = np.flatnonzero(table.piece >= 0)
    order = np.lexsort((table.frame[keep], table.piece[keep]))
    idx = keep[order]
    ids = table.piece[idx]
    out: dict[int, _Piece] = {}
    if len(ids) == 0:
        return out
    twice = np.flatnonzero((np.diff(ids) == 0) & (np.diff(table.frame[idx]) == 0))
    if len(twice):
        first = idx[twice[0]]
        raise ValueError(
            f"relink cuts: piece {int(table.piece[first])} has two rows in frame "
            f"{int(table.frame[first])} ({len(twice)} such rows in all)"
        )
    starts = np.r_[0, np.where(np.diff(ids) != 0)[0] + 1, len(ids)]
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        out[int(ids[a])] = _Piece(ids[a], idx[a:b], table)
    return out


def _boundary_cos(p: _Piece, params: CutParams) -> np.ndarray:
    """Before/after cosine at every boundary of the piece (NaN where a side has fewer than
    ``m_min`` rows), with clean rows; where that is undefined, with all rows that have an
    embedding."""
    k, m_min = params.k_win, params.m_min
    n = len(p)
    out = np.full(max(n - 1, 0), np.nan)
    for mask in (p.clean, p.has):
        idx = np.where(mask)[0]
        if len(idx) < 2 * m_min:
            continue
        C = np.vstack([np.zeros((1, p.E.shape[1])), np.cumsum(p.E[idx], axis=0)])
        cnt = np.cumsum(mask)[:-1]  # usable rows with index <= i
        b0 = np.maximum(0, cnt - k)
        a1 = np.minimum(len(idx), cnt + k)
        ok = (cnt - b0 >= m_min) & (a1 - cnt >= m_min) & np.isnan(out)
        mb = C[cnt[ok]] - C[b0[ok]]
        ma = C[a1[ok]] - C[cnt[ok]]
        with np.errstate(invalid="ignore", divide="ignore"):  # two windows that cancel out
            out[ok] = (mb * ma).sum(axis=1) / (
                np.linalg.norm(mb, axis=1) * np.linalg.norm(ma, axis=1)
            )
    return out


def _pitch_side(p: _Piece, i: int, side: int, params: CutParams) -> np.ndarray | None:
    """Median pitch position of up to ``pitch_rows`` placed rows next to boundary i on one
    side, no further than ``pitch_within`` frames from the boundary row; None without one."""
    n, within = params.pitch_rows, params.pitch_within
    if side < 0:
        idx = np.where(p.placed[: i + 1] & (p.frame[: i + 1] >= p.frame[i] - within))[0][-n:]
    else:
        near = p.placed[i + 1 :] & (p.frame[i + 1 :] <= p.frame[i + 1] + within)
        idx = np.where(near)[0][:n] + i + 1
    if not len(idx):
        return None
    return np.median(p.xy[idx], axis=0)


class _Context:
    """Lookups of one call: the pieces and, per piece, the cumulative embedding sums of its
    clean rows and of its rows with an embedding."""

    def __init__(self, pieces: dict[int, _Piece], params: CutParams) -> None:
        self.P = pieces
        self.params = params
        self.cum: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for pid, p in pieces.items():
            ent = []
            for mask in (p.clean, p.has):
                idx = np.where(mask)[0]
                C = np.vstack([np.zeros((1, p.E.shape[1])), np.cumsum(p.E[idx], axis=0)])
                ent.append((idx, C))
            self.cum[pid] = ent

    def desc(
        self, pid: int, lo: int, hi: int, take: str, small: bool = False
    ) -> tuple[np.ndarray | None, int, int]:
        """Mean embedding of the ``k_win`` usable rows of rows[lo:hi] of a piece nearest its
        ``take`` end ("last" / "first"): clean rows when there are at least ``m_min``, else
        rows with an embedding (at least ``m_min``, or with ``small`` at least one).

        Returns (unit vector or None, rows used, row index of the nearest used row or -1).
        """
        k, m_min = self.params.k_win, self.params.m_min
        for which, (idx, C) in enumerate((*self.cum[pid], self.cum[pid][1])):
            if which == 2 and not small:
                break
            a, b = np.searchsorted(idx, lo), np.searchsorted(idx, hi)
            if b - a >= (m_min if which < 2 else 1):
                if take == "last":
                    a = max(a, b - k)
                    near = int(idx[b - 1])
                else:
                    b = min(b, a + k)
                    near = int(idx[a])
                v = C[b] - C[a]
                return v / np.linalg.norm(v), int(b - a), near
        return None, 0, -1


# --------------------------------------------------------------------------- candidates


def _split_in_plateau(p: _Piece, lo: int, hi: int, mb: np.ndarray, ma: np.ndarray) -> int:
    """Boundary in [lo, hi] (boundary j lies between rows j and j + 1) where the rows switch
    from looking like the before mean to looking like the after mean."""
    if hi <= lo:
        return lo
    rows = np.arange(lo + 1, hi + 1)  # rows strictly inside the plateau
    s = np.where(p.has[rows], p.E[rows] @ (mb - ma), 0.0)  # > 0: looks like before
    cum = np.r_[0.0, np.cumsum(s)]
    gain = cum - (s.sum() - cum)  # boundary lo + m keeps the first m plateau rows before
    return lo + int(np.argmax(gain))


def _boundaries(c: _Context, p: _Piece) -> dict[int, str]:
    """{boundary index: kind} of the piece's candidate boundaries."""
    params = c.params
    k_win, m_min, side_rows = params.k_win, params.m_min, params.min_side_rows
    n = len(p)
    if n < 2:
        return {}
    gaps = np.diff(p.frame) - 1
    cand = {int(i): "gap" for i in np.where(gaps >= params.gap_min)[0]}
    if n < 2 * side_rows:
        return cand
    cs = _boundary_cos(p, params)
    allowed = np.zeros(n - 1, bool)
    allowed[side_rows - 1 : n - side_rows] = True
    low = np.where(~np.isnan(cs) & (cs < params.cos_cand) & allowed)[0]
    if len(low):
        order = low[np.argsort(cs[low], kind="stable")]
        taken: list[int] = []
        usable = p.clean if p.clean.sum() >= 2 * m_min else p.has
        ui = np.where(usable)[0]
        for i in order:
            if any(abs(i - j) <= k_win for j in taken):
                continue
            taken.append(int(i))
            # plateau: the boundaries between the same two usable rows share the value
            k = np.searchsorted(ui, i, side="right")  # usable rows <= i
            lo = int(ui[k - 1]) if k > 0 else 0
            hi = int(ui[k] - 1) if k < len(ui) else n - 2
            lo, hi = max(lo, side_rows - 1), min(hi, n - side_rows - 1)
            if hi < lo:
                continue
            near = range(lo - k_win // 2, hi + k_win // 2 + 1)
            if any(j in near for j, kind in cand.items() if kind == "gap"):
                continue  # a gap candidate covers it
            mb = c.desc(p.pid, 0, lo + 1, "last")[0]
            ma = c.desc(p.pid, hi + 1, n, "first")[0]
            if mb is not None and ma is not None:
                j = _split_in_plateau(p, lo, hi, mb, ma)
            else:
                j = (lo + hi) // 2
            cand[int(j)] = "app"
    return cand


def _static(c: _Context, p: _Piece, i: int, kind: str) -> dict[str, Any]:
    """Everything about a candidate that does not depend on the other cuts of its piece."""
    params = c.params
    alt_frames, alt_guard = params.alt_frames, params.alt_guard
    f0, f1 = int(p.frame[i]), int(p.frame[i + 1])
    gap = f1 - f0 - 1
    xb = _pitch_side(p, i, -1, params)
    xa = _pitch_side(p, i, +1, params)
    d = float(np.hypot(*(xa - xb))) if xb is not None and xa is not None else np.nan
    # alternative origins of the after side / continuations of the before side
    D, R = [], []
    for q in c.P.values():
        if q.pid == p.pid:
            continue
        j1 = np.searchsorted(q.frame, f1)  # rows of q before f1
        seen_after = j1 < len(q) and q.frame[j1] <= f1 + alt_guard
        if j1 > 0 and q.frame[j1 - 1] >= f1 - alt_frames and not seen_after:
            v = c.desc(q.pid, 0, j1, "last")[0]
            if v is not None:
                D.append(v)
        j0 = np.searchsorted(q.frame, f0, side="right")  # rows of q up to f0
        seen_before = j0 > 0 and q.frame[j0 - 1] >= f0 - alt_guard
        if j0 < len(q) and q.frame[j0] <= f0 + alt_frames and not seen_before:
            v = c.desc(q.pid, j0, len(q), "first")[0]
            if v is not None:
                R.append(v)
    dim = p.E.shape[1]
    return {
        "piece": p.pid,
        "i": i,
        "f0": f0,
        "f1": f1,
        "gap": gap,
        "kind": kind,
        "d_m": d,
        "_D": np.array(D).reshape(-1, dim),
        "_R": np.array(R).reshape(-1, dim),
    }


def _static_candidates(c: _Context) -> dict[int, list[dict[str, Any]]]:
    """{piece id: its candidates in row order} for the pieces that have one."""
    out = {}
    for p in c.P.values():
        b = _boundaries(c, p)
        if b:
            out[p.pid] = [_static(c, p, i, kind) for i, kind in sorted(b.items())]
    return out


def _dynamic(c: _Context, p: _Piece, s: dict[str, Any], lo: int, hi: int) -> dict[str, Any]:
    """Features of a candidate inside the stretch rows[lo:hi] of its piece."""
    i = s["i"]
    B, nb, near_b = c.desc(p.pid, lo, i + 1, "last", small=True)
    A, na, near_a = c.desc(p.pid, i + 1, hi, "first", small=True)
    cos = float(B @ A) if B is not None and A is not None else np.nan
    alt_d, alt_r = np.nan, np.nan
    if A is not None and len(s["_D"]):
        alt_d = float((s["_D"] @ A).max())
    if B is not None and len(s["_R"]):
        alt_r = float((s["_R"] @ B).max())
    out = {k: v for k, v in s.items() if not k.startswith("_")}
    out.update(
        {
            "cos": cos,
            "nb": nb,
            "na": na,
            "near_b": near_b,
            "near_a": near_a,
            "alt_d": alt_d,
            "alt_r": alt_r,
        }
    )
    return out


# --------------------------------------------------------------------------- model


def _design(cand: pd.DataFrame, params: CutParams) -> np.ndarray:
    """Model inputs of candidates, one row each: cos (1 when undefined); mm = the larger of the
    two margins cos - alt_d and cos - alt_r, each clipped to +-``margin_clip``, a missing
    alternative counting as +``margin_clip``; lgap = log(1 + gap); ld = log(1 + pitch distance
    across the boundary) (0 when unknown)."""
    clip = params.margin_clip
    cos = cand["cos"].astype(float)
    md = (cos - cand["alt_d"].astype(float)).clip(-clip, clip).fillna(clip)
    mr = (cos - cand["alt_r"].astype(float)).clip(-clip, clip).fillna(clip)
    columns = [
        cos.fillna(1.0).to_numpy(float),
        np.maximum(md, mr).to_numpy(float),
        np.log1p(cand["gap"].astype(float)).to_numpy(float),
        np.log1p(cand["d_m"].astype(float).fillna(0.0)).to_numpy(float),
    ]
    # The lab takes this matrix out of a pandas frame, which hands out the transpose of a
    # (columns, rows) array. The matrix product of _logit picks its BLAS routine by that
    # memory layout and the routines differ in the last bits, so the layout is kept.
    return np.array(columns).T


def _logit(cand: pd.DataFrame, params: CutParams) -> np.ndarray:
    """Logit of "the two sides are different people" for candidates (rows with features)."""
    X = _design(cand, params)
    return ((X - np.array(params.mu)) / np.array(params.sd)) @ np.array(params.w) + params.b


# --------------------------------------------------------------------------- segmentation


def _segment_piece(
    c: _Context, p: _Piece, statics: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Greedy binary segmentation of one piece: the chosen candidates (their features at the
    moment they were cut, "logit" and the stretch [lo, hi) they split)."""
    nms_rows = c.params.nms_rows
    bounds = [0, len(p)]  # row indices where stretches start (sorted)
    cut_rows: list[int] = []
    chosen: list[dict[str, Any]] = []
    while True:
        rows = []
        for s in statics:
            i = s["i"]
            if any(abs(i - j) <= nms_rows for j in cut_rows):
                continue
            k = int(np.searchsorted(bounds, i, side="right"))
            row = _dynamic(c, p, s, bounds[k - 1], bounds[k])
            row["lo"], row["hi"] = bounds[k - 1], bounds[k]
            rows.append(row)
        if not rows:
            break
        df = pd.DataFrame(rows)
        df["logit"] = _logit(df, c.params)
        # candidates with the same before/after windows are one decision, at the largest gap
        key = ["lo", "near_b", "near_a"]
        df["_best"] = df.groupby(key)["logit"].transform("max")
        rep = df.sort_values(["gap", "i"], ascending=[False, True], kind="stable")
        rep = rep.drop_duplicates(key)
        rep = rep.sort_values(["_best", "gap"], ascending=False, kind="stable")
        top = rep.iloc[0]
        if not (top["_best"] >= threshold):
            break
        d = top.to_dict()
        d["logit"] = float(d.pop("_best"))
        chosen.append(d)
        cut_rows.append(int(top["i"]))
        bounds = sorted(bounds + [int(top["i"]) + 1])
    return chosen


def cut_table(inp: Any, params: CutParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """The cuts of a clip with their features, sorted by (piece, f1).

    Columns: ``piece``; ``f0`` (the last frame before the cut) and ``f1`` (the first frame of
    the new sub-piece); ``gap`` (missing frames between them); ``cos``, ``alt_d``, ``alt_r``
    (NaN where undefined) and ``d_m`` (metres on the pitch across the cut, NaN when unknown) at
    the moment of the cut; ``logit`` and ``prob`` of "the two sides are different people".
    """
    c = _Context(_piece_table(_rows_of(inp)), params)
    threshold = float(np.log(params.p_cut / (1 - params.p_cut)))
    rows: list[dict[str, Any]] = []
    for pid, statics in _static_candidates(c).items():
        rows += _segment_piece(c, c.P[pid], statics, threshold)
    df = pd.DataFrame(rows)
    if not len(df):
        return pd.DataFrame({name: np.zeros(0, dtype) for name, dtype in _TABLE_DTYPES.items()})
    df["prob"] = 1.0 / (1.0 + np.exp(-df["logit"]))
    df = df.sort_values(["piece", "f1"]).reset_index(drop=True)
    return df[list(_TABLE_DTYPES)].astype(_TABLE_DTYPES)


def detect_cuts(inp: Any, params: CutParams = DEFAULT_PARAMS) -> tuple[tuple[int, int], ...]:
    """The cuts of a clip: sorted unique ``(piece, first frame of the new sub-piece)``."""
    table = cut_table(inp, params)
    return norm_cuts(zip(table["piece"].tolist(), table["f1"].tolist(), strict=True))
