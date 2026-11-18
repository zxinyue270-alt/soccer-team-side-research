"""Relink: cut tracks where the tracker changed person and join the pieces of one person.

The top level of the relink step, an optional part of the refine stage (``--relink``; off by
default). It is a port of the frozen method ``fuse_r1`` of the re-identification lab
(reid_lab, outside the repository), with a product safety policy that isolates conflicting
unclean observations. Its results need not match the unguarded lab method. For one shot
(:func:`relink_clip`):

1. cuts (gbf.relink.cuts): inside a piece of the track table, find the places where the
   plugin's track moved onto another person, and cut there;
2. isolate unclean conflicts corroborated by a different shirt colour (gbf.relink.safety),
   then appearance (gbf.relink.appear): one prototype per sub-piece, centred and whitened;
3. tiers: a sub-piece with at least ``n_solid`` boxes is solid, the others are short;
4. position evidence (gbf.relink.space) on the solid sub-pieces only;
5. the pair bonus and the core links between solid sub-pieces (gbf.relink.link);
6. in the recall mode, short sub-pieces attached as leaves (gbf.relink.leaves), with position
   evidence computed on all sub-pieces.

Modes: ``strict`` (few links, almost all right on the development clips) and ``recall`` (a
lower margin, and leaves); ``off`` is the pipeline's default and runs nothing of this package.

Identifiers. Inside the package a piece is a track id of the table handed in (below
``gbf.relink.cuts.SUB_STRIDE``), and the rows of a piece from its j-th cut on (j from 0) form
the sub-piece ``piece + SUB_STRIDE * (j + 1)``; cuts are ``(piece, first frame of the new
sub-piece)`` and links ``(earlier sub-piece, later sub-piece)``. In the track table
(:func:`apply_ids`) every row gets the id of its identity, which is the sub-piece id of the
identity's earliest member, and the old id moves to a new last column ``piece_id``.
Isolated observations receive separate fresh ids above every sub-piece id, one per row;
these ids mean unverified observations, not newly discovered people.

Shots. The method assumes one continuous camera, and pieces never span a camera cut, so
:func:`relink_tracks` runs the method once per shot, on the shot's rows with frames counted
from the shot's first frame, and no link joins two shots.

The method's numbers are the lab's (see docs/design/m1-highlights.md): fitted
or set on 13 SN-GSR clips at 1280x720 and 25 fps with PRTReid embeddings, validated on no other
footage. Constants counted in frames are not rescaled at another frame rate. Nothing here
reads an annotation or a file.
The isolation policy reuses k_win, m_min and cos_cand; that new use has not been validated.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gbf.relink import appear, cuts, inputs, leaves, link, safety, shirts, space

MODES: tuple[str, ...] = ("off", "strict", "recall")
"""Values of ``--relink``; "off" never reaches this package."""

FITTED_FPS = 25.0
"""Frame rate of the clips the parameters were fitted on. Reported next to a run's own frame
rate; no arithmetic uses it."""

METHOD = "fuse_r1 (frozen) with observed-colour row isolation (product safety policy)"
"""What the ``relink`` section of refine.json calls the method."""

SKIP_REASONS: tuple[str, ...] = (
    "no_embeddings",
    "no_identity",
    "track_ids_out_of_range",
    "no_frame_size",
)
"""Why a mode that is on did not run (``reason`` of the refine.json section)."""


class RelinkError(ValueError):
    """The relink inputs are unusable, or a result does not fit its table."""


@dataclass(frozen=True)
class RelinkParams:
    """The parameters of every step (the lab's frozen values)."""

    cuts: cuts.CutParams = cuts.DEFAULT_PARAMS
    appear: appear.AppearParams = appear.DEFAULT_PARAMS
    space: space.SpaceParams = space.DEFAULT_PARAMS
    link: link.LinkParams = link.DEFAULT_PARAMS

    def as_dict(self) -> dict[str, Any]:
        """JSON form for reports and receipts."""
        return {
            "cuts": self.cuts.as_dict(),
            "appear": self.appear.as_dict(),
            "space": self.space.as_dict(),
            "link": self.link.as_dict(),
        }


DEFAULT_PARAMS = RelinkParams()


@dataclass(frozen=True)
class ClipLinks:
    """The method's result for one shot: ``cuts`` are (piece, frame counted from the shot's
    first frame), the links (earlier sub-piece id, later sub-piece id) in the order made.
    isolated_rows are positions in the input arrays, excluded from the link evidence.
    """

    cuts: tuple[tuple[int, int], ...]
    core_links: tuple[tuple[int, int], ...]
    leaf_links: tuple[tuple[int, int], ...]
    isolated_rows: tuple[int, ...] = ()


@dataclass(frozen=True, eq=False)
class RelinkResult:
    """What :func:`relink_tracks` found for a track table.

    Attributes:
        mode: the mode it ran in.
        cuts: (piece, canonical frame) of every cut, sorted.
        core_links, leaf_links: (earlier sub-piece id, later sub-piece id), shot by shot in
            the order made.
        sub_id: (rows,) int64 sub-piece id of every row of the table; rows without a track
            keep their id. Isolated observations get unique fresh ids.
        track_id: (rows,) int64 final id of every row (the id of its identity).
        refused: links not applied because their two identities share a frame (none
            expected: the linker never proposes one).
        report: the ``relink`` section of refine.json.
    """

    mode: str
    cuts: tuple[tuple[int, int], ...]
    core_links: tuple[tuple[int, int], ...]
    leaf_links: tuple[tuple[int, int], ...]
    sub_id: np.ndarray
    track_id: np.ndarray
    refused: tuple[tuple[int, int], ...]
    report: dict[str, Any]

    @property
    def links(self) -> tuple[tuple[int, int], ...]:
        """Core links, then leaf links."""
        return self.core_links + self.leaf_links


def setting(mode: str, params: RelinkParams = DEFAULT_PARAMS) -> dict[str, Any]:
    """``{"margin", "leaves"}`` of a mode that is on: the linker's margin and whether short
    sub-pieces are attached as leaves."""
    if mode == "strict":
        return {"margin": params.link.margin, "leaves": False}
    if mode == "recall":
        return {"margin": leaves.RECALL_MARGIN, "leaves": True}
    on = ", ".join(m for m in MODES if m != "off")
    raise RelinkError(f"relink: mode must be one of {on}, got {mode!r}")


def skipped(mode: str, reason: str) -> dict[str, Any]:
    """The refine.json section of a mode that is on but did not run."""
    if reason not in SKIP_REASONS:
        raise RelinkError(f"relink: unknown skip reason {reason!r}")
    return {"mode": mode, "applied": False, "reason": reason}


# --------------------------------------------------------------------------- one shot


def _bonus(
    app: appear.Appearance, pos: space.PositionScores, lp: link.LinkParams, *, core: bool
) -> dict[tuple[int, int], float]:
    """The pair bonus of the candidate pairs: the position score of each pair (0.0 where the
    position model does not admit it) and, for the core, whether the plugin tracked both
    under one id."""
    a, b = link.candidate_pairs(app, lp.n_solid if core else 0)
    sink_of = {(int(x), int(y)): float(s) for x, y, s in zip(pos.a, pos.b, pos.sink, strict=True)}
    sink = np.array(
        [sink_of.get((int(x), int(y)), 0.0) for x, y in zip(a, b, strict=True)], dtype=float
    )
    plug = np.zeros(len(a))
    if core:
        e = pos.pieces
        joins = link.plugin_joins(e.piece, e.piece % cuts.SUB_STRIDE, e.plugin, e.first, e.last)
        plug = np.array(
            [float((int(x), int(y)) in joins) for x, y in zip(a, b, strict=True)], dtype=float
        )
    return link.pair_bonus(a, b, sink, plug, lp, use_plug=core)


def relink_clip(
    inp: inputs.ClipInputs,
    mode: str,
    params: RelinkParams = DEFAULT_PARAMS,
    log: list[dict] | None = None,
) -> ClipLinks:
    """The cuts and links of one single-shot clip.

    ``inp`` holds the clip's rows (gbf.relink.inputs.build_clip); ``mode`` is "strict" or
    "recall". ``log``, when given, receives one entry per merge of the linker (its
    similarity, appearance cosine, bonus, the two groups and the two rival margins).
    """
    chosen = setting(mode, params)
    margin = chosen["margin"]
    cut_list = cuts.detect_cuts(inp, params.cuts)
    sub = cuts.sub_ids(inp.piece, inp.frame, cut_list)
    isolated = tuple(int(i) for i in safety.conflicting_rows(inp, sub, params.cuts))
    sub[list(isolated)] = -1  # neither appearance nor spatial evidence may use these rows
    app = appear.build(inp, sub, params.appear)
    if len(app.ids) < 2:
        return ClipLinks(cut_list, (), (), isolated)
    ids, n = np.unique(sub[sub >= 0], return_counts=True)
    solid = ids[n >= params.link.n_solid]
    pos = space.position_scores(inp, np.where(np.isin(sub, solid), sub, -1), params.space)
    core = link.link_core(
        app, _bonus(app, pos, params.link, core=True), params.link, margin=margin, log=log
    )
    leaf: tuple[tuple[int, int], ...] = ()
    if chosen["leaves"]:
        pos_all = space.position_scores(inp, sub, params.space)
        leaf = leaves.attach_leaves(
            core.state,
            _bonus(app, pos_all, params.link, core=False),
            params.link,
            margin=margin,
            log=log,
        )
    return ClipLinks(cut_list, core.links, leaf, isolated)


# --------------------------------------------------------------------------- a track table


def _input_problems(
    tracks: pd.DataFrame,
    calib: pd.DataFrame,
    embeddings: np.ndarray,
    shots: pd.DataFrame,
    fps: float,
    frame_size: tuple[int, int],
) -> list[str]:
    problems = [
        f"tracks: column {name} is missing"
        for name in inputs.TRACK_COLUMNS
        if name not in tracks.columns
    ]
    problems += [
        f"calib: column {name} is missing"
        for name in inputs.CALIB_COLUMNS
        if name not in calib.columns
    ]
    problems += [
        f"shots: column {name} is missing"
        for name in ("start_frame", "end_frame")
        if name not in shots.columns
    ]
    shape = np.shape(embeddings)
    if len(shape) != 2 or shape[0] != len(tracks) or shape[1] < 1:
        problems.append(
            f"embeddings: expected one row per track row ({len(tracks)}, D >= 1), got shape {shape}"
        )
    if "track_id" in tracks.columns and len(tracks):
        largest = int(tracks["track_id"].max())
        if largest >= cuts.SUB_STRIDE:
            problems.append(
                f"tracks: track ids must be below {cuts.SUB_STRIDE}, got {largest} "
                "(ids from that value on name the sub-pieces after a cut)"
            )
    if not (
        isinstance(fps, numbers.Real)
        and not isinstance(fps, bool)
        and math.isfinite(fps)
        and fps > 0
    ):
        problems.append(f"fps must be a positive number, got {fps!r}")
    if len(tuple(frame_size)) != 2 or not all(float(v) > 0 for v in frame_size):
        problems.append(f"frame_size must be two positive numbers, got {frame_size!r}")
    return problems


def _shot_spans(shots: pd.DataFrame) -> list[tuple[int, int]]:
    """(start, end) of every shot in frame order; RelinkError when two shots overlap."""
    spans = sorted(
        (int(s), int(e))
        for s, e in zip(shots["start_frame"].tolist(), shots["end_frame"].tolist(), strict=True)
    )
    for (_s0, e0), (s1, _e1) in zip(spans[:-1], spans[1:], strict=True):
        if s1 <= e0:
            raise RelinkError(f"relink: shots overlap at frame {s1}")
    return spans


def _check_pieces(frame: np.ndarray, piece: np.ndarray, spans: list[tuple[int, int]]) -> None:
    """RelinkError when a piece has two rows in one frame, rows in two shots, or rows in a
    shot and rows in no shot. A piece that lies wholly outside every shot is not an error:
    no shot holds it, so it is left as it is."""
    tracked = piece >= 0
    keys = np.stack([piece[tracked], frame[tracked]], axis=1)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    problems: list[str] = []
    if (counts > 1).any():
        first = unique[counts > 1][0]
        problems.append(
            f"{int((counts > 1).sum())} (track id, frame) pair(s) have more than one row, "
            f"first: track {int(first[0])} in frame {int(first[1])}"
        )
    if spans:
        starts = np.array([s for s, _e in spans], dtype=np.int64)
        ends = np.array([e for _s, e in spans], dtype=np.int64)
        shot = np.searchsorted(starts, frame, side="right") - 1
        inside = (shot >= 0) & (frame <= ends[np.clip(shot, 0, None)])
        shot = np.where(inside, shot, -1)
        ids, where = np.unique(piece[tracked], return_inverse=True)
        low = np.full(len(ids), len(spans), dtype=np.int64)
        high = np.full(len(ids), -2, dtype=np.int64)
        np.minimum.at(low, where, shot[tracked])
        np.maximum.at(high, where, shot[tracked])
        split = ids[low != high]
        if len(split):
            problems.append(
                f"{len(split)} track id(s) have rows in more than one shot (or beside the "
                f"shots), first: {int(split[0])}; relink needs ids split at the camera cuts"
            )
    if problems:
        raise RelinkError("relink: " + "; ".join(problems))


def _final_ids(
    sub: np.ndarray, frame: np.ndarray, links: tuple[tuple[int, int], ...]
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], list[dict[str, Any]]]:
    """(final id per row, refused links, identities of several sub-pieces).

    An identity is a set of sub-pieces joined by links; its id is the sub-piece id of its
    earliest member (smallest first frame, then smallest id). Links are applied in order; one
    whose two identities share a frame is refused.
    """
    tracked = np.flatnonzero(sub >= 0)
    ids, where = np.unique(sub[tracked], return_inverse=True)
    index = {int(p): i for i, p in enumerate(ids)}
    frames: list[set[int]] = [set() for _ in ids]
    for i, f in zip(where.tolist(), frame[tracked].tolist(), strict=True):
        frames[i].add(f)
    first = [min(f) for f in frames]
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    refused: list[tuple[int, int]] = []
    for a, b in links:
        if a not in index or b not in index:
            raise RelinkError(f"relink: the link ({a}, {b}) names a sub-piece without rows")
        ra, rb = find(index[a]), find(index[b])
        if ra == rb:
            continue
        if frames[ra] & frames[rb]:
            refused.append((int(a), int(b)))
            continue
        parent[rb] = ra
        frames[ra] |= frames[rb]
    members: dict[int, list[int]] = {}
    for i in range(len(ids)):
        members.setdefault(find(i), []).append(i)
    name = np.zeros(len(ids), dtype=np.int64)
    identities: list[dict[str, Any]] = []
    for group in members.values():
        group.sort(key=lambda i: (first[i], int(ids[i])))
        name[group] = ids[group[0]]
        if len(group) > 1:
            identities.append(
                {"track_id": int(ids[group[0]]), "pieces": [int(ids[i]) for i in group]}
            )
    identities.sort(key=lambda item: item["track_id"])
    final = sub.astype(np.int64)
    final[tracked] = name[where]
    return final, tuple(refused), identities


def _duplicates(frame: np.ndarray, track_id: np.ndarray) -> int:
    """Number of (frame, id >= 0) pairs that more than one row has."""
    tracked = track_id >= 0
    keys = np.stack([track_id[tracked], frame[tracked]], axis=1)
    _unique, counts = np.unique(keys, axis=0, return_counts=True)
    return int((counts > 1).sum())


def relink_tracks(
    tracks: pd.DataFrame,
    calib: pd.DataFrame,
    embeddings: np.ndarray,
    shots: pd.DataFrame,
    *,
    fps: float,
    frame_size: tuple[int, int],
    mode: str,
    params: RelinkParams = DEFAULT_PARAMS,
    shirt_lab: np.ndarray | None = None,
) -> RelinkResult:
    """Run the relink method on a track table, shot by shot.

    Args:
        tracks: the refine stage's track table before relink (the columns of
            gbf.relink.inputs.TRACK_COLUMNS; ``track_id`` holds the piece ids), any row order.
        calib: the calibration table (``frame_index``, ``K``, ``R``, ``C``).
        embeddings: (rows of ``tracks``, D) the plugin's appearance embeddings as stored
            (float32, not normalised), row i for row i of ``tracks``; a NaN row = none.
        shots: the shot table (``start_frame``, ``end_frame`` inclusive).
        fps: frames per second of the video: any real number above zero (an int, a float,
            a Fraction, a numpy number), not a bool.
        frame_size: (width, height), in pixels, of the frames the boxes and the calibration
            refer to (the calibration table's ``width`` and ``height``). The pixel constants
            scale with this height.
        mode: "strict" or "recall".
        params: see :class:`RelinkParams`.
        shirt_lab: optional (rows of tracks, 3) CIELAB measurements from canonical RGB,
            e.g. measure_shirt_colours. Missing rows are NaN. None disables row isolation,
            not cuts or links: a corrupted ReID descriptor alone is not a person swap.

    A shot with fewer than two frames or fewer than two pieces is left as it is, and so is a
    track that lies wholly outside every shot (no shot holds its rows, so nothing cuts or
    joins it; it keeps its id). The table is not changed: :func:`apply_ids` writes the result
    into a copy.

    Rows without an embedding (``counts.rows_without_embedding``: a value that is not finite,
    or a row of length zero) take no part in the appearance of their piece, and a piece none
    of whose rows has one is no candidate for a link.

    Raises RelinkError naming every problem with the inputs: a mode that is not on, a missing
    column, embeddings of another length, a track id of ``gbf.relink.cuts.SUB_STRIDE`` or
    more, a track with two rows in one frame, with rows in two shots, or with rows in a shot
    and rows in no shot.
    """
    chosen = setting(mode, params)
    problems = _input_problems(tracks, calib, embeddings, shots, fps, frame_size)
    if problems:
        raise RelinkError("relink: " + "; ".join(problems))
    try:
        shirt_lab = shirts.checked_colours(shirt_lab, len(tracks))
    except (ValueError, TypeError) as exc:
        raise RelinkError(str(exc)) from exc
    embeddings = np.asarray(embeddings)
    frame = tracks["frame"].to_numpy(dtype=np.int64)
    piece = tracks["track_id"].to_numpy(dtype=np.int64)
    spans = _shot_spans(shots)
    _check_pieces(frame, piece, spans)

    all_cuts: list[tuple[int, int]] = []
    core: list[tuple[int, int]] = []
    leaf: list[tuple[int, int]] = []
    merges: list[dict[str, Any]] = []
    shots_run = 0
    isolated = np.zeros(len(tracks), dtype=bool)
    for start, end in spans:
        try:
            inp, _rows = inputs.build_clip(
                tracks,
                calib,
                embeddings,
                start_frame=start,
                end_frame=end,
                fps=float(fps),
                frame_size=frame_size,
                shirt_lab=shirt_lab,
            )
        except ValueError as exc:
            raise RelinkError(str(exc)) from exc
        if inp.n_frames < 2 or len(np.unique(inp.piece[inp.piece >= 0])) < 2:
            isolated[_rows[safety.conflicting_rows(inp, inp.piece, params.cuts)]] = True
            continue
        shots_run += 1
        log: list[dict[str, Any]] = []
        clip = relink_clip(inp, mode, params, log)
        isolated[_rows[list(clip.isolated_rows)]] = True
        all_cuts += [(p, f + start) for p, f in clip.cuts]
        core += clip.core_links
        leaf += clip.leaf_links
        merges += log

    cut_list = cuts.norm_cuts(all_cuts)
    sub = cuts.sub_ids(piece, frame, cut_list)
    # One fresh id per unverified observation: never infer that two ambiguous boxes are
    # the same person. Deterministic under input row permutations, unique across shots.
    isolated_rows = np.flatnonzero(isolated)
    isolated_rows = isolated_rows[np.lexsort((piece[isolated_rows], frame[isolated_rows]))]
    first_isolated = int(sub.max()) + 1 if len(sub) else 0
    if first_isolated + len(isolated_rows) > np.iinfo(np.int64).max:
        raise RelinkError("relink: no int64 ids available for isolated observations")
    sub[isolated_rows] = first_isolated + np.arange(len(isolated_rows), dtype=np.int64)
    links = tuple(core) + tuple(leaf)
    final, refused, identities = _final_ids(sub, frame, links)
    twice = _duplicates(frame, final)
    if twice:
        raise RelinkError(f"relink: {twice} (frame, track id) pair(s) repeat after the renaming")

    width, height = (float(v) for v in frame_size)
    sub_ids, sub_rows = np.unique(sub[sub >= 0], return_counts=True)
    solid = int((sub_rows >= params.link.n_solid).sum())
    report: dict[str, Any] = {
        "mode": mode,
        "applied": True,
        "reason": None,
        "method": METHOD,
        "setting": {"margin": float(chosen["margin"]), "leaves": bool(chosen["leaves"])},
        "params": params.as_dict(),
        "fps": float(fps),
        "fitted_fps": FITTED_FPS,
        "frame_size": [int(round(width)), int(round(height))],
        "px_scale": height / inputs.REFERENCE_HEIGHT_PX,
        "counts": {
            "shots": len(spans),
            "shots_relinked": shots_run,
            "pieces": int(len(np.unique(piece[piece >= 0]))),
            "cuts": len(cut_list),
            "sub_pieces": int(len(sub_ids)),
            "solid": solid,
            "short": int(len(sub_ids)) - solid,
            "core_links": len(core),
            "leaf_links": len(leaf),
            "identities": int(len(np.unique(final[final >= 0]))),
            "rows_renamed": int((final != piece).sum()),
            "rows_without_embedding": int((~inputs.has_embedding(embeddings)).sum()),
            "links_refused_same_frame": len(refused),
        },
        "cuts": [[int(p), int(f)] for p, f in cut_list],
        "links": [[int(a), int(b)] for a, b in links],
        "leaf_links": [[int(a), int(b)] for a, b in leaf],
        "identities": identities,
        "merges": merges,
        "isolation": {
            "policy": "unclean ReID conflict corroborated by a different shirt colour",
            "version": 3,
            "colour_evidence_supplied": shirt_lab is not None,
            "colour_margin_delta_e": shirts.COLOUR_MARGIN,
            "colour_max_alternative_delta_e": shirts.MAX_COLOUR_DISTANCE,
            "colour_measurement": shirts.MEASUREMENT_PARAMS.as_dict(),
            "own_colour_reference": "every measured sample in each supported clean-side window",
            "rows": len(isolated_rows),
            "cos_min": params.cuts.cos_cand,
            "window_rows": params.cuts.k_win,
            "min_clean_rows": params.cuts.m_min,
            "observations": [
                {"frame": int(frame[i]), "piece_id": int(piece[i]), "track_id": int(sub[i])}
                for i in isolated_rows
            ],
            "limitation": "Isolated ids are unverified single observations, not new people. "
            "Colour cannot detect same-kit swaps and missing colour leaves rows unchanged. "
            "A contaminated own-side sample can veto isolation of a real swap. "
            "This safety policy has not been validated on independent footage.",
        },
    }
    return RelinkResult(
        mode=mode,
        cuts=cut_list,
        core_links=tuple(core),
        leaf_links=tuple(leaf),
        sub_id=sub,
        track_id=final,
        refused=refused,
        report=report,
    )


def apply_ids(tracks: pd.DataFrame, result: RelinkResult | None) -> pd.DataFrame:
    """A copy of ``tracks`` with the relink ids: ``track_id`` rewritten to the identities'
    ids and a new last column ``piece_id`` (int64) holding the old ``track_id``.

    Rows, their order and every other column stay as they are. ``result`` None (a mode that
    is on but did not run) adds ``piece_id`` only, so the table's columns depend on the mode
    alone. Raises RelinkError when the result does not belong to the table (another length,
    another id for a row without a track) or when two rows of one frame would get one id.
    """
    problems: list[str] = []
    for name in ("frame", "track_id"):
        if name not in tracks.columns:
            problems.append(f"column {name} is missing")
    if "piece_id" in tracks.columns:
        problems.append("the table already has a piece_id column")
    if problems:
        raise RelinkError("relink ids: " + "; ".join(problems))
    old = tracks["track_id"].to_numpy(dtype=np.int64)
    out = tracks.copy()
    if result is not None:
        new = np.asarray(result.track_id)
        if new.shape != old.shape or not np.issubdtype(new.dtype, np.integer):
            raise RelinkError(
                f"relink ids: expected {len(old)} integer ids, got shape {new.shape} of {new.dtype}"
            )
        new = new.astype(np.int64)
        untracked = old < 0
        if not np.array_equal(new[untracked], old[untracked]) or (new[~untracked] < 0).any():
            raise RelinkError("relink ids: the result does not belong to this table")
        twice = _duplicates(tracks["frame"].to_numpy(dtype=np.int64), new)
        if twice:
            raise RelinkError(f"relink ids: {twice} (frame, track id) pair(s) would repeat")
        out["track_id"] = new
    out["piece_id"] = old
    return out
