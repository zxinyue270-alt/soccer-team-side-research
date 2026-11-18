"""Position evidence for relink: could a later piece of a track be an earlier one's person?

A port of the label-free position model of the re-identification lab (reid_lab, outside the
repository; its modules m_space/geom.py, features.py and the apply part of linker.py). Given
the same inputs it gives the lab's scores; the constants and the fitted weights are the lab's
frozen ones (docs/design/m1-highlights.md, "Tuned parameters").

A piece is a set of rows with one id (``ids``, -1 = not a piece; the caller decides which
pieces take part). For every ordered pair (a, b) of pieces with a's last frame before b's
first, the module measures, without appearance and without the plugin's identity:

- where a ended and where b started on the pitch (robust line fits over the last / first
  ``end_window`` frames, :func:`piece_ends`);
- how far apart these are once the movement of the visible players is taken out: a's end
  point is carried forward to b's first frame by the local velocity field of the players
  around it, b's start point is carried backwards to a's last frame, and both keep their own
  velocity relative to the collective flow for a short while (``r_adv_cv``, metres);
- whether a hidden path within ``max_speed`` connects the two points in the gap (``reach``);
- for how long the carried points lay inside the picture during the gap: a person who stayed
  in the picture would have been tracked (the in-picture seconds);
- whether the pieces end / start at a border of the frame, and their team and role labels.

:func:`position_scores` turns the admissible pairs (within reach, same role, teams not known
to differ, both points known) into a conditional-logit evidence z per pair and lets the
candidates compete: P(a | b) among the earlier pieces of b with a "nobody" alternative,
P(b | a) among the later pieces of a, and the doubly normalised assignment (Sinkhorn with a
"nobody" slack on both sides), which is the score the linker reads.

World frame: metres, origin at the centre spot, z down (gbf.geometry). Cameras are the
projection matrices P = K R [I | -C] of the frames. Pixel constants are given at 720 lines and
scale with the frame height; constants counted in frames are the lab's at 25 fps and are not
rescaled. Everything here is a pure function of its arguments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SpaceParams:
    """Constants of the position evidence (the lab's values) and its fitted weights.

    ``w`` belongs to the pair evidence (lu, ls, tu, ee, ex, fa, ba), ``g_in`` and ``g_out`` to
    the "nobody" context (one, edge, time), ``sigma`` to sigma(dt) = s0 + s1 * dt ** s2.
    """

    edge_px_720: float = 10.0
    """px at 720 lines: a box this close to the frame border touches it."""
    end_window: int = 25
    """Frames at a piece's end / start used for its position and velocity."""
    max_speed_fit: float = 10.0
    """m/s: cap of a fitted end / start velocity."""
    fit_min_points: int = 6
    """Points (and inliers) a line fit needs."""
    fit_min_span: int = 6
    """Frames the points of a line fit must span."""
    fit_nearest: int = 5
    """Rows of the median fallback, and nearest placed rows used outside the window."""
    fit_outlier_floor_m: float = 1.0
    """m: least outlier threshold of the refit."""
    fit_outlier_scale: float = 2.5
    """Outlier threshold of the refit in robust sigmas (1.4826 x the median residual)."""
    body_height_m: float = 1.8
    """m: head height of the in-picture test."""
    flow_half: int = 6
    """Frames: half chord of a row velocity."""
    flow_min_players: int = 3
    """Player velocities a frame needs for a collective flow."""
    flow_max_speed: float = 12.0
    """m/s: faster chords are dropped."""
    adv_kernel_m: float = 15.0
    """m: Gaussian kernel of the local velocity field."""
    adv_shrink: float = 0.5
    """Pseudo-weight of the collective flow in the local velocity field."""
    cv_horizon_s: float = 1.0
    """s: a piece's own relative velocity is extrapolated this long at most."""
    view_margin_px_720: float = 20.0
    """px at 720 lines: feet and head this far inside the frame = in the picture."""
    grid_m: float = 1.0
    """m: cell of the reach programme."""
    grid_x: tuple[float, float] = (-62.0, 62.0)
    """m: extent of the grid along the touchlines."""
    grid_y: tuple[float, float] = (-44.0, 44.0)
    """m: extent of the grid across the pitch."""
    step: int = 10
    """Frames per step of the reach programme."""
    max_speed: float = 8.0
    """m/s: speed bound of the hidden path."""
    start_radius: float = 3.0
    """m: slack around both end points of the hidden path."""
    inview_cap_s: float = 3.0
    """s: cap of the in-picture seconds."""
    time_cap_s: float = 30.0
    """s: cap of the time context."""
    sinkhorn_iters: int = 200
    """Most Sinkhorn sweeps."""
    sinkhorn_tol: float = 1e-6
    """The sweeps stop when the largest change is below this."""
    sigma: tuple[float, float, float] = (
        0.3277373957752331,
        1.5025066206921236,
        0.800034644312043,
    )
    """Typical residual of true partners after dt seconds: s0 + s1 * dt ** s2 (fitted)."""
    w: tuple[float, float, float, float, float, float, float] = (
        -1.4404843400874152,
        -0.5572009493363808,
        -1.2556011387607238,
        0.21317124065580034,
        -0.3546536674430176,
        -0.5033163671123451,
        -0.36252916475801683,
    )
    """Weights of the pair evidence lu, ls, tu, ee, ex, fa, ba (fitted)."""
    g_in: tuple[float, float, float] = (
        -3.2633513520891686,
        -0.4235423949588176,
        -0.10996270863227203,
    )
    """Weights of "nobody came before b": one, edge, time (fitted)."""
    g_out: tuple[float, float, float] = (
        -0.9511335274908453,
        0.7228607672659471,
        -1.1918239495110512,
    )
    """Weights of "nobody comes after a": one, edge, time (fitted)."""

    def __post_init__(self) -> None:
        sizes = {"grid_x": 2, "grid_y": 2, "sigma": 3, "w": 7, "g_in": 3, "g_out": 3}
        problems: list[str] = []
        for name, size in sizes.items():
            value = getattr(self, name)
            if isinstance(value, list):  # as_dict() stores tuples as lists
                value = tuple(value)
                object.__setattr__(self, name, value)
            if not isinstance(value, tuple) or len(value) != size:
                problems.append(f"{name}: expected {size} numbers, got {value!r}")
        for name in ("end_window", "fit_nearest", "flow_half", "step"):
            if getattr(self, name) < 1:
                problems.append(f"{name}: must be at least 1, got {getattr(self, name)!r}")
        for name in ("grid_m", "max_speed", "adv_kernel_m"):
            if not getattr(self, name) > 0:
                problems.append(f"{name}: must be positive, got {getattr(self, name)!r}")
        if problems:
            raise ValueError("SpaceParams: " + "; ".join(problems))

    def as_dict(self) -> dict[str, Any]:
        """JSON form for reports and receipts."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


DEFAULT_PARAMS = SpaceParams()


@dataclass(frozen=True)
class PieceEnds:
    """Start and end state of the k pieces of one run (ids >= 0), in ascending id."""

    piece: np.ndarray
    """(k,) int64 ids."""
    plugin: np.ndarray
    """(k,) int64: source id of the piece's first row."""
    first: np.ndarray
    """(k,) int64 first frame."""
    last: np.ndarray
    """(k,) int64 last frame."""
    team: np.ndarray
    """(k,) object: the most frequent team label of its rows, or None."""
    role: np.ndarray
    """(k,) object: the most frequent role label of its rows, or None."""
    end_xy: np.ndarray
    """(k, 2) float64 metres: ground position at the last frame, NaN when unknown."""
    start_xy: np.ndarray
    """(k, 2) float64 metres: ground position at the first frame, NaN when unknown."""
    end_v: np.ndarray
    """(k, 2) float64 m/s: fitted velocity at the end (0 when too few rows)."""
    start_v: np.ndarray
    """(k, 2) float64 m/s: fitted velocity at the start."""
    end_edge: np.ndarray
    """(k,) bool: the last box touches a border of the frame."""
    start_edge: np.ndarray
    """(k,) bool: the first box touches a border of the frame."""


@dataclass(frozen=True)
class PairTable:
    """Every ordered pair of pieces with a's last frame before b's first: q rows, a-major
    then b, both in ascending id."""

    a: np.ndarray
    """(q,) int64: the earlier piece."""
    b: np.ndarray
    """(q,) int64: the later piece."""
    dt: np.ndarray
    """(q,) float64 seconds from a's last to b's first frame."""
    known: np.ndarray
    """(q,) bool: a has an end point and b a start point."""
    r_adv_cv: np.ndarray
    """(q,) float64 metres: residual after local advection and own velocity; NaN when not
    known."""
    reach: np.ndarray
    """(q,) bool: a hidden path within the speed bound connects the two points."""
    fwd_adv_inview_s: np.ndarray
    """(q,) float64 seconds of the gap a's carried end point lies inside the picture."""
    back_adv_inview_s: np.ndarray
    """(q,) float64 seconds of the gap b's start point, carried back, lies inside it."""
    a_end_edge: np.ndarray
    """(q,) bool: a's last box touches a border of the frame."""
    b_start_edge: np.ndarray
    """(q,) bool: b's first box touches a border of the frame."""
    same_team: np.ndarray
    """(q,) int64: 1 same team label, 0 different, -1 when a label is missing."""
    same_role: np.ndarray
    """(q,) int64: 1 when both pieces have the same role label, else 0."""
    a_last: np.ndarray
    """(q,) int64: a's last frame."""
    b_first: np.ndarray
    """(q,) int64: b's first frame."""


@dataclass(frozen=True)
class PositionScores:
    """The position model on the m admissible pairs, a-major then b, both ascending."""

    a: np.ndarray
    """(m,) int64: the earlier piece."""
    b: np.ndarray
    """(m,) int64: the later piece."""
    z: np.ndarray
    """(m,) float64 log evidence of the pair."""
    p_in: np.ndarray
    """(m,) float64 P(a | b) among b's earlier candidates and "nobody"."""
    p_out: np.ndarray
    """(m,) float64 P(b | a) among a's later candidates and "nobody"."""
    sink: np.ndarray
    """(m,) float64 doubly normalised assignment score."""
    pieces: PieceEnds
    """The pieces of the run (all of them, not only those in a pair)."""


# --------------------------------------------------------------------------- cameras


def project(P: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P (..., 3, 4), xyz (..., 3) -> pixels uv (..., 2) and depth (...,)."""
    h = np.concatenate([xyz, np.ones(xyz.shape[:-1] + (1,))], axis=-1)
    q = np.einsum("...ij,...j->...i", P, h)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[..., :2] / q[..., 2:3]
    return uv, q[..., 2]


def view_margin(
    P: np.ndarray, xy: np.ndarray, width: float, height: float, body: float
) -> np.ndarray:
    """Signed distance in px of a person standing on ground point(s) xy (..., 2) from the
    border of the frame of camera(s) P (..., 3, 4): > 0 inside (the smaller of the feet's and
    the head's distance to the nearest border; the head is ``body`` metres up), < 0 outside,
    -inf behind the camera or without a camera."""
    z0 = np.zeros(xy.shape[:-1] + (1,))
    feet, d1 = project(P, np.concatenate([xy, z0], axis=-1))
    head, d2 = project(P, np.concatenate([xy, z0 - body], axis=-1))
    out = np.full(np.broadcast_shapes(xy.shape[:-1], P.shape[:-2]), np.inf)
    for uv in (feet, head):
        m = np.minimum(
            np.minimum(uv[..., 0], width - 1 - uv[..., 0]),
            np.minimum(uv[..., 1], height - 1 - uv[..., 1]),
        )
        out = np.minimum(out, m)
    good = (d1 > 0) & (d2 > 0) & np.isfinite(out)
    return np.where(good, out, -np.inf)


# --------------------------------------------------------------------------- pieces


def _checked_ids(inp: Any, ids: np.ndarray) -> np.ndarray:
    """``ids`` as int64 after a check of the shapes; ValueError names every problem."""
    out = np.asarray(ids)
    frame = np.asarray(inp.frame)
    problems: list[str] = []
    if out.ndim != 1 or out.shape != frame.shape:
        problems.append(f"ids: expected one id per row {frame.shape}, got shape {out.shape}")
    elif not np.issubdtype(out.dtype, np.integer):
        problems.append(f"ids: expected integers, got {out.dtype}")
    for name in ("source", "box", "role", "team", "pitch_xy", "pitch_usable"):
        if len(getattr(inp, name)) != len(frame):
            problems.append(f"{name}: expected {len(frame)} rows, got {len(getattr(inp, name))}")
    n = int(inp.n_frames)
    if frame.size and (int(frame.min()) < 0 or int(frame.max()) >= n):
        problems.append(f"frame: values must lie in 0..{n - 1}")
    if not float(inp.fps) > 0:
        problems.append(f"fps: must be positive, got {inp.fps!r}")
    if np.shape(inp.cam_P) != (n, 3, 4):
        problems.append(f"cam_P: expected shape {(n, 3, 4)}, got {np.shape(inp.cam_P)}")
    if problems:
        raise ValueError("relink position evidence: " + "; ".join(problems))
    return out.astype(np.int64, copy=False)


def _rows_by_piece(ids: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """(ids >= 0 in ascending order, the row positions of each in ascending frame)."""
    tracked = np.flatnonzero(ids >= 0)
    if not len(tracked):
        return np.zeros(0, np.int64), []
    order = tracked[np.lexsort((frame[tracked], ids[tracked]))]
    sorted_ids = ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    stops = np.r_[starts[1:], len(order)]
    return sorted_ids[starts], [order[a:b] for a, b in zip(starts, stops, strict=True)]


def _majority(labels: np.ndarray) -> str | None:
    """The most frequent label (ties: the smallest string); None when no row has one."""
    counts: dict[str, int] = {}
    for value in labels:
        if isinstance(value, str):
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    top = max(counts.values())
    return min(label for label, count in counts.items() if count == top)


def _side(
    x1: float, y1: float, x2: float, y2: float, width: float, height: float, edge: float
) -> str:
    """Which borders of the frame a box touches: letters of L, R, T, B (empty = none)."""
    s = ""
    if x1 <= edge:
        s += "L"
    if x2 >= width - 1 - edge:
        s += "R"
    if y1 <= edge:
        s += "T"
    if y2 >= height - 1 - edge:
        s += "B"
    return s


def _fit(
    frames: np.ndarray, xy: np.ndarray, at: float, fps: float, params: SpaceParams
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Robust position at frame ``at`` and velocity (m/s) from the points (frames, xy): a
    line through them, fitted once more without its outliers; the median of the nearest
    points and no velocity when they are too few or span too few frames."""
    n = len(frames)
    if n == 0:
        return (np.nan, np.nan), (0.0, 0.0)
    if n < params.fit_min_points or frames.max() - frames.min() < params.fit_min_span:
        k = min(n, params.fit_nearest)
        order = np.argsort(np.abs(frames - at))[:k]
        med = np.median(xy[order], axis=0)
        return (float(med[0]), float(med[1])), (0.0, 0.0)
    t = (frames - at) / fps
    A = np.column_stack([np.ones(n), t])
    keep = np.ones(n, bool)
    coef = None
    for _ in range(2):  # least squares, then once more without the outliers
        coef, *_ = np.linalg.lstsq(A[keep], xy[keep], rcond=None)
        res = np.hypot(*(xy - A @ coef).T)
        thr = max(
            params.fit_outlier_floor_m, params.fit_outlier_scale * np.median(res[keep]) * 1.4826
        )
        new = res <= thr
        if new.sum() < params.fit_min_points:
            break
        keep = new
    pos, vel = coef[0], coef[1]
    sp = float(np.hypot(*vel))
    if sp > params.max_speed_fit:
        vel = vel * params.max_speed_fit / sp
    return (float(pos[0]), float(pos[1])), (float(vel[0]), float(vel[1]))


def piece_ends(inp: Any, ids: np.ndarray, params: SpaceParams = DEFAULT_PARAMS) -> PieceEnds:
    """Start and end state on the pitch of every piece of ``ids`` (ids >= 0).

    The ground position at the last (first) frame comes from a robust line fit over the
    placed rows (pitch_usable, finite position) of the last (first) ``end_window`` frames.
    Rows whose box is cut by the bottom border are left out when at least two others exist
    (their feet are not seen); without a placed row in the window the nearest placed rows
    are used. The velocity is the fitted one, 0 when the rows are too few for a line.
    """
    ids = _checked_ids(inp, ids)
    frame = np.asarray(inp.frame)
    pitch_xy = np.asarray(inp.pitch_xy, float)
    boxes = np.asarray(inp.box, float)
    usable = np.asarray(inp.pitch_usable, bool)
    teams, roles = np.asarray(inp.team, object), np.asarray(inp.role, object)
    fps = float(inp.fps)
    width, height = float(inp.width), float(inp.height)
    edge = params.edge_px_720 * float(inp.px_scale)
    pieces, rows_of = _rows_by_piece(ids, frame)
    k = len(pieces)
    plugin = np.zeros(k, np.int64)
    first = np.zeros(k, np.int64)
    last = np.zeros(k, np.int64)
    team = np.full(k, None, dtype=object)
    role = np.full(k, None, dtype=object)
    state = {name: np.full((k, 2), np.nan) for name in ("end_xy", "start_xy", "end_v", "start_v")}
    end_edge = np.zeros(k, bool)
    start_edge = np.zeros(k, bool)
    for i, rows in enumerate(rows_of):
        f = frame[rows]
        xy = pitch_xy[rows]
        box = boxes[rows]
        placed = usable[rows] & np.isfinite(xy).all(axis=1)
        bottom = box[:, 3] >= height - 1 - edge
        first[i], last[i] = int(f[0]), int(f[-1])
        plugin[i] = int(inp.source[rows[0]])
        team[i] = _majority(teams[rows])
        role[i] = _majority(roles[rows])
        end_edge[i] = _side(*box[-1], width, height, edge) != ""
        start_edge[i] = _side(*box[0], width, height, edge) != ""
        windows = (
            ("end", int(last[i]), f >= last[i] - params.end_window + 1),
            ("start", int(first[i]), f <= first[i] + params.end_window - 1),
        )
        for name, at, sel in windows:
            use = sel & placed & ~bottom
            if use.sum() < 2:
                use = sel & placed
            if use.sum() == 0 and placed.any():  # nothing placed in the window
                idx = np.flatnonzero(placed)
                near = idx[np.argsort(np.abs(f[idx] - at))[: params.fit_nearest]]
                use = np.zeros(len(f), bool)
                use[near] = True
            pos, vel = _fit(f[use], xy[use], float(at), fps, params)
            state[f"{name}_xy"][i] = pos
            state[f"{name}_v"][i] = vel
    return PieceEnds(
        piece=pieces.astype(np.int64),
        plugin=plugin,
        first=first,
        last=last,
        team=team,
        role=role,
        end_xy=state["end_xy"],
        start_xy=state["start_xy"],
        end_v=state["end_v"],
        start_v=state["start_v"],
        end_edge=end_edge,
        start_edge=start_edge,
    )


# --------------------------------------------------------------------------- flow


def _player_velocities(
    inp: Any, ids: np.ndarray, params: SpaceParams
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per frame: positions (k, 2) and velocities (k, 2) of the tracked players with a chord
    of ``flow_half`` frames to both sides, in ascending piece id."""
    n = int(inp.n_frames)
    fps = float(inp.fps)
    frame = np.asarray(inp.frame)
    pitch_xy = np.asarray(inp.pitch_xy, float)
    player = np.fromiter((r == "player" for r in inp.role), bool, len(frame))
    rows_ok = (ids >= 0) & np.asarray(inp.pitch_usable, bool) & ~np.isnan(pitch_xy[:, 0]) & player
    _, rows_of = _rows_by_piece(np.where(rows_ok, ids, -1), frame)
    pos: list[list[np.ndarray]] = [[] for _ in range(n)]
    vel: list[list[np.ndarray]] = [[] for _ in range(n)]
    for rows in rows_of:
        f = frame[rows]
        if len(f) < 2 * params.flow_half + 1:
            continue
        xy = pitch_xy[rows]
        at = {int(ff): i for i, ff in enumerate(f)}
        for k, ff in enumerate(f):
            i, j = at.get(int(ff) - params.flow_half), at.get(int(ff) + params.flow_half)
            if i is None or j is None:
                continue
            v = (xy[j] - xy[i]) / (2 * params.flow_half / fps)
            if np.hypot(*v) < params.flow_max_speed:
                pos[int(ff)].append(xy[k])
                vel[int(ff)].append(v)
    return (
        [np.array(p).reshape(-1, 2) for p in pos],
        [np.array(v).reshape(-1, 2) for v in vel],
    )


def _flow(vel: list[np.ndarray], fps: float, params: SpaceParams) -> np.ndarray:
    """S (n_frames, 2): cumulative collective displacement in metres. S[t2] - S[t1] is how
    far the visible players moved together between the frames (per-frame median of the row
    velocities, 0 on frames with fewer than ``flow_min_players`` of them)."""
    F = np.zeros((len(vel), 2))
    for f, v in enumerate(vel):
        if len(v) >= params.flow_min_players:
            F[f] = np.median(v, axis=0)
    return np.cumsum(F, axis=0) / fps


def _local_velocity(
    x: np.ndarray, pos: np.ndarray, vel: np.ndarray, glob: np.ndarray, params: SpaceParams
) -> np.ndarray:
    """Velocity field at the points x (m, 2): kernel mean of the player velocities around
    each point, shrunk to the collective flow ``glob`` of the frame."""
    if len(pos) == 0:
        return np.tile(glob, (len(x), 1))
    d2 = ((x[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2)
    k = np.exp(-d2 / (2.0 * params.adv_kernel_m**2))
    return (k @ vel + params.adv_shrink * glob) / (k.sum(axis=1, keepdims=True) + params.adv_shrink)


def _advected(
    ends: PieceEnds,
    pos: list[np.ndarray],
    vel: list[np.ndarray],
    S: np.ndarray,
    fps: float,
    params: SpaceParams,
) -> tuple[np.ndarray, np.ndarray]:
    """(fwd, back), each (pieces, n_frames, 2): the end point of every piece carried forward
    by the local velocity field (NaN before its last frame) and its start point carried
    backwards (NaN after its first frame)."""
    n = len(S)
    k = len(ends.piece)
    G = np.vstack([S[:1], np.diff(S, axis=0)]) * fps  # collective velocity per frame
    last, first = ends.last, ends.first
    F = np.full((k, n, 2), np.nan)
    B = np.full((k, n, 2), np.nan)
    for i in range(k):
        F[i, last[i]] = ends.end_xy[i]
        B[i, first[i]] = ends.start_xy[i]
    ok_e = np.isfinite(ends.end_xy).all(axis=1)
    ok_s = np.isfinite(ends.start_xy).all(axis=1)
    for t in range(1, n):  # forward: x(t) = x(t-1) + v(x(t-1), t-1) / fps
        act = np.flatnonzero((last < t) & ok_e)
        if len(act):
            prev = F[act, t - 1]
            F[act, t] = prev + _local_velocity(prev, pos[t - 1], vel[t - 1], G[t - 1], params) / fps
    for t in range(n - 2, -1, -1):  # backward: x(t) = x(t+1) - v(x(t+1), t+1) / fps
        act = np.flatnonzero((first > t) & ok_s)
        if len(act):
            nxt = B[act, t + 1]
            B[act, t] = nxt - _local_velocity(nxt, pos[t + 1], vel[t + 1], G[t + 1], params) / fps
    return F, B


def _inview_counts(paths: np.ndarray, inp: Any, params: SpaceParams) -> np.ndarray:
    """(pieces, n_frames) int64: per path, the cumulative number of frames on which a person
    standing on it is wholly inside the picture."""
    P = np.asarray(inp.cam_P, float)
    width, height = float(inp.width), float(inp.height)
    threshold = params.view_margin_px_720 * float(inp.px_scale)
    out = np.zeros(paths.shape[:2], np.int64)
    for i, path in enumerate(paths):
        ok = np.isfinite(path).all(axis=1)
        m = np.zeros(len(path), bool)
        if ok.any():
            margin = view_margin(P[ok], path[ok], width, height, params.body_height_m)
            m[ok] = margin >= threshold
        out[i] = np.cumsum(m)
    return out


# --------------------------------------------------------------------------- reach


def _grid(params: SpaceParams) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(params.grid_x[0], params.grid_x[1] + 1e-9, params.grid_m)
    ys = np.arange(params.grid_y[0], params.grid_y[1] + 1e-9, params.grid_m)
    return xs, ys


def _disc(radius_cells: float) -> np.ndarray:
    k = int(np.floor(radius_cells))
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    return (xx * xx + yy * yy) <= radius_cells * radius_cells + 1e-9


def _cell(x: float, y: float, xs: np.ndarray, ys: np.ndarray, grid_m: float) -> tuple[int, int]:
    i = int(np.clip(np.rint((y - ys[0]) / grid_m), 0, len(ys) - 1))
    j = int(np.clip(np.rint((x - xs[0]) / grid_m), 0, len(xs) - 1))
    return i, j


def _local_any(arr: np.ndarray, i: int, j: int, footprint: np.ndarray) -> bool:
    """Is a cell of ``arr`` set within the footprint centred on (i, j)?"""
    k = footprint.shape[0] // 2
    i0, i1 = max(0, i - k), min(arr.shape[0], i + k + 1)
    j0, j1 = max(0, j - k), min(arr.shape[1], j + k + 1)
    sub = arr[i0:i1, j0:j1]
    fp = footprint[(i0 - i + k) : (i1 - i + k), (j0 - j + k) : (j1 - j + k)]
    return bool(sub[fp].any())


def _reach(ends: PieceEnds, n_frames: int, fps: float, params: SpaceParams) -> np.ndarray:
    """(pieces, pieces) bool: [a, b] is True when a.last < b.first, both points are known and
    a path within ``max_speed`` (with ``start_radius`` of slack at both ends) leads from a's
    end point to b's start point in the gap.

    The lab's programme over a grid of ``grid_m`` cells in steps of ``step`` frames, reduced
    to reachability: its in-picture cost only adds finite seconds to cells that are reachable
    anyway. The cells within the slack of a's end point are reachable in the step of a's last
    frame; every step spreads them by the distance covered in one step; b is reached when a
    reachable cell lies within the slack of its start cell at the step of its first frame
    (pieces starting in a's own step are tested with those of the next one).
    """
    k = len(ends.piece)
    out = np.zeros((k, k), bool)
    xs, ys = _grid(params)
    n_steps = int(np.ceil(n_frames / params.step))
    step_s = params.step / fps
    move = _disc(params.max_speed * step_s / params.grid_m)
    slack = _disc(params.start_radius / params.grid_m)
    ok_e = np.isfinite(ends.end_xy).all(axis=1)
    ok_s = np.isfinite(ends.start_xy).all(axis=1)
    cells = [
        _cell(x, y, xs, ys, params.grid_m) if ok else None
        for (x, y), ok in zip(ends.start_xy, ok_s, strict=True)
    ]
    for a in np.flatnonzero(ok_e):
        by_step: dict[int, list[int]] = {}
        for b in np.flatnonzero((ends.first > ends.last[a]) & ok_s):
            by_step.setdefault(int(ends.first[b]) // params.step, []).append(int(b))
        if not by_step:
            continue
        seen = np.zeros((len(ys), len(xs)), bool)
        seen[_cell(*ends.end_xy[a], xs, ys, params.grid_m)] = True
        seen = ndimage.binary_dilation(seen, structure=slack)
        s = int(ends.last[a]) // params.step  # a is seen in the step of its last frame
        last_step = max(by_step)
        first_round = True
        while s <= last_step:
            seen = ndimage.binary_dilation(seen, structure=move)  # one step further
            targets = by_step.get(s + 1, []) + (by_step.get(s, []) if first_round else [])
            for b in targets:
                out[a, b] = _local_any(seen, *cells[b], slack)
            first_round = False
            s += 1
            if s >= n_steps:
                break
            if seen.all():  # every cell is reachable from here on: so is every later piece
                for step, later in by_step.items():
                    if step > s:
                        out[a, later] = True
                break
    return out


# --------------------------------------------------------------------------- pair table


def _codes(labels: np.ndarray) -> np.ndarray:
    """An integer per label, equal for equal labels; -1 for a missing one."""
    names = sorted({v for v in labels if isinstance(v, str)})
    index = {name: i for i, name in enumerate(names)}
    return np.array([index[v] if isinstance(v, str) else -1 for v in labels], np.int64)


def _empty_pairs() -> PairTable:
    i, x, flag = np.zeros(0, np.int64), np.zeros(0), np.zeros(0, bool)
    return PairTable(
        a=i,
        b=i.copy(),
        dt=x,
        known=flag,
        r_adv_cv=x.copy(),
        reach=flag.copy(),
        fwd_adv_inview_s=x.copy(),
        back_adv_inview_s=x.copy(),
        a_end_edge=flag.copy(),
        b_start_edge=flag.copy(),
        same_team=i.copy(),
        same_role=i.copy(),
        a_last=i.copy(),
        b_first=i.copy(),
    )


def pair_table(
    inp: Any,
    ids: np.ndarray,
    params: SpaceParams = DEFAULT_PARAMS,
    ends: PieceEnds | None = None,
) -> PairTable:
    """The position features of every ordered pair of pieces (a, b) with a's last frame
    before b's first. ``ends``: the result of :func:`piece_ends` for the same arguments, when
    the caller has it."""
    ids = _checked_ids(inp, ids)
    if ends is None:
        ends = piece_ends(inp, ids, params)
    ia, ib = np.nonzero(ends.last[:, None] < ends.first[None, :])
    if not len(ia):
        return _empty_pairs()
    n = int(inp.n_frames)
    fps = float(inp.fps)
    pos, vel = _player_velocities(inp, ids, params)
    S = _flow(vel, fps, params)
    a_last, b_first = ends.last[ia], ends.first[ib]
    dt = (b_first - a_last) / fps
    # own velocity relative to the flow, kept for a short horizon, from both sides
    Fv = np.gradient(S, axis=0) * fps
    h = np.minimum(dt, params.cv_horizon_s) / 2.0
    va = ends.end_v[ia] - Fv[a_last]
    vb = ends.start_v[ib] - Fv[b_first]
    # local advection: end point of a carried forward to b.first, start of b carried back
    AF, AB = _advected(ends, pos, vel, S, fps, params)
    ef = AF[ia, b_first] - ends.start_xy[ib]
    eb = AB[ib, a_last] - ends.end_xy[ia]
    mid = 0.5 * (-ef + eb)  # mean of the two residual vectors, as b - a
    cva = mid - (va + vb) * h[:, None]
    r_adv_cv = np.hypot(cva[:, 0], cva[:, 1])
    known = np.isfinite(ends.end_xy[ia]).all(axis=1) & np.isfinite(ends.start_xy[ib]).all(axis=1)
    reach = _reach(ends, n, fps, params)[ia, ib]
    fwd, back = _inview_counts(AF, inp, params), _inview_counts(AB, inp, params)
    team, role = _codes(ends.team), _codes(ends.role)
    team_known = (team[ia] >= 0) & (team[ib] >= 0)
    return PairTable(
        a=ends.piece[ia],
        b=ends.piece[ib],
        dt=dt,
        known=known,
        r_adv_cv=r_adv_cv,
        reach=reach,
        fwd_adv_inview_s=(fwd[ia, b_first - 1] - fwd[ia, a_last]) / fps,
        back_adv_inview_s=(back[ib, b_first - 1] - back[ib, a_last]) / fps,
        a_end_edge=ends.end_edge[ia],
        b_start_edge=ends.start_edge[ib],
        same_team=np.where(team_known, (team[ia] == team[ib]).astype(np.int64), -1),
        same_role=((role[ia] >= 0) & (role[ia] == role[ib])).astype(np.int64),
        a_last=a_last,
        b_first=b_first,
    )


# --------------------------------------------------------------------------- the model


def _sigma(dt: np.ndarray, s: tuple[float, float, float]) -> np.ndarray:
    return s[0] + s[1] * np.power(np.maximum(dt, 0.0), s[2])


def _groups(keys: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    uniq, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    return len(uniq), inv, first


def _lse(score: np.ndarray, grp: np.ndarray, const: np.ndarray) -> np.ndarray:
    """log(exp(const_g) + sum over the rows of group g of exp(score)) per group."""
    m = const.copy()
    np.maximum.at(m, grp, score)
    tot = np.exp(const - m)
    np.add.at(tot, grp, np.exp(score - m[grp]))
    return m + np.log(tot)


def _scores(
    z: np.ndarray,
    gb: np.ndarray,
    nb: int,
    cb: np.ndarray,
    ga: np.ndarray,
    na: int,
    ca: np.ndarray,
    params: SpaceParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cb / ca: "nobody" log-weights per later / earlier piece. Returns p_in, p_out and the
    doubly normalised score."""
    p_in = np.exp(z - _lse(z, gb, cb)[gb])
    p_out = np.exp(z - _lse(z, ga, ca)[ga])
    # Sinkhorn with slack: M_ab = u_a K_ab v_b, u_a (e^ca + sum_b K v) = 1,
    # v_b (e^cb + sum_a K u) = 1
    lu, lv = np.zeros(na), np.zeros(nb)
    for _ in range(params.sinkhorn_iters):
        lu = -_lse(z + lv[gb], ga, ca)
        lv_new = -_lse(z + lu[ga], gb, cb)
        if np.max(np.abs(lv_new - lv)) < params.sinkhorn_tol:
            lv = lv_new
            break
        lv = lv_new
    return p_in, p_out, np.exp(z + lu[ga] + lv[gb])


def position_scores(
    inp: Any, ids: np.ndarray, params: SpaceParams = DEFAULT_PARAMS
) -> PositionScores:
    """The position model of one clip on the pieces of ``ids``.

    ``inp`` is read by attribute: frame, source, box, role, team, pitch_xy, pitch_usable,
    cam_P, n_frames, fps, width, height, px_scale (gbf.relink.inputs.ClipInputs). ``ids`` is
    the id of every row for this run of the model, -1 for a row that belongs to no piece.

    A pair is admissible when b is within reach of a, both have the same role label, their
    team labels do not differ and both points are known. Pair evidence, z = w . x:
    lu = log(1 + (r_adv_cv / sigma(dt))^2), ls = log sigma(dt), tu = a team label is missing,
    ee = a ends at a border of the frame and b starts at one, ex = exactly one of the two,
    fa / ba = the capped in-picture seconds of the carried points. "Nobody" context of a
    piece: 1, it starts (ends) at a border, log(1 + seconds since the clip's start (to its
    end), capped).
    """
    ends = piece_ends(inp, ids, params)
    table = pair_table(inp, ids, params, ends)
    ok = table.reach & (table.same_role == 1) & (table.same_team != 0) & table.known
    a, b = table.a[ok], table.b[ok]
    if not len(a):
        none = np.zeros(0)
        return PositionScores(a, b, none, none.copy(), none.copy(), none.copy(), ends)
    fps = float(inp.fps)
    sg = _sigma(table.dt[ok], params.sigma)
    u = table.r_adv_cv[ok] / sg
    ea, eb = table.a_end_edge[ok], table.b_start_edge[ok]
    design = np.column_stack(
        [
            np.log1p(u * u),
            np.log(sg),
            (table.same_team[ok] == -1).astype(float),
            (ea & eb).astype(float),
            (ea ^ eb).astype(float),
            np.minimum(table.fwd_adv_inview_s[ok], params.inview_cap_s),
            np.minimum(table.back_adv_inview_s[ok], params.inview_cap_s),
        ]
    )
    z = design @ np.asarray(params.w)
    nb, gb, fb = _groups(b)
    na, ga, fa = _groups(a)
    one = np.ones(len(a))
    since_start = table.b_first[ok].astype(float) / fps
    to_end = (float(inp.n_frames) - 1 - table.a_last[ok].astype(float)).clip(min=0) / fps
    ctx_in = np.column_stack(
        [one, eb.astype(float), np.log1p(np.minimum(since_start, params.time_cap_s))]
    )
    ctx_out = np.column_stack(
        [one, ea.astype(float), np.log1p(np.minimum(to_end, params.time_cap_s))]
    )
    cb = (ctx_in @ np.asarray(params.g_in))[fb]
    ca = (ctx_out @ np.asarray(params.g_out))[fa]
    p_in, p_out, sink = _scores(z, gb, nb, cb, ga, na, ca, params)
    return PositionScores(a=a, b=b, z=z, p_in=p_in, p_out=p_out, sink=sink, pieces=ends)
