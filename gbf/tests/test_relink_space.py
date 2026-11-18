"""Tests for gbf.relink.space (position evidence of the relink step).

Every clip is synthetic: people stand or move on the grass in simple ways, and a piece is a
stretch of rows of one person under one id. Where a case needs the picture, the boxes are
projected with gbf.geometry's synthetic main broadcast camera at 1280x720, which sees about
20 m to each side of the centre spot. Numbers are compared with tolerances and decisions are
kept far from their thresholds: the last bits of a fit or a matrix product differ between
machines.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from gbf.geometry import synthetic_broadcast_camera
from gbf.relink import space
from gbf.relink.space import PairTable, PositionScores, SpaceParams

FPS = 25.0
W, H = 1280, 720
P_CAMERA = synthetic_broadcast_camera(W, H).projection_matrix()
MID_BOX = (620.0, 320.0, 650.0, 400.0)
"""A box in the middle of the frame, for the cases that do not need the picture."""

Point = tuple[float, float]
Box = tuple[float, float, float, float]


def _projected_box(x: float, y: float) -> Box:
    """The box of a person 1.8 m tall standing on (x, y), not clipped to the frame."""
    q = P_CAMERA @ np.array([[x, y, 0.0, 1.0], [x, y, -1.8, 1.0]]).T
    (u, v), (_, top) = (q[:2] / q[2]).T
    half = 0.2 * (v - top)
    return (float(u - half), float(top), float(u + half), float(v))


class _Clip:
    """The rows of a synthetic clip, added one piece (or one part of it) at a time."""

    def __init__(self, n_frames: int, *, camera: bool = False) -> None:
        self.n_frames = n_frames
        self.camera = camera
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        pid: int,
        frames: Iterable[int],
        where: Point | Callable[[int], Point],
        *,
        source: int | None = None,
        role: str | None = "player",
        team: str | None | Callable[[int], str | None] = "left",
        usable: bool | Callable[[int], bool] = True,
        box: Box | Callable[[int], Box] | None = None,
    ) -> None:
        """One row per frame for piece ``pid`` standing on ``where`` (a point, or a function
        of the frame). Without ``box`` the box is the middle one, or with a camera the
        projected one, clipped to the frame; a frame on which the person's feet are left or
        right of the frame then gets no row."""
        for frame in frames:
            x, y = where(frame) if callable(where) else where
            if box is not None:
                x1, y1, x2, y2 = box(frame) if callable(box) else box
            elif not self.camera:
                x1, y1, x2, y2 = MID_BOX
            else:
                x1, y1, x2, y2 = _projected_box(x, y)
                if not 0.0 <= (x1 + x2) / 2 <= W - 1:
                    continue
                x1, x2 = max(x1, 0.0), min(x2, W - 1.0)
                y1, y2 = max(y1, 0.0), min(y2, H - 1.0)
            self.rows.append(
                {
                    "frame": frame,
                    "id": pid,
                    "source": pid if source is None else source,
                    "role": role,
                    "team": team(frame) if callable(team) else team,
                    "xy": (x, y),
                    "usable": usable(frame) if callable(usable) else usable,
                    "box": (x1, y1, x2, y2),
                }
            )

    def inputs(
        self,
        *,
        order: np.ndarray | None = None,
        scale: float = 1.0,
        blind: Iterable[int] = (),
    ) -> tuple[SimpleNamespace, np.ndarray]:
        """(inp, ids) as gbf.relink.space reads them. ``order`` permutes the rows; ``scale``
        is the frame height over 720 (boxes and the cameras' pixels are scaled with it);
        the frames in ``blind`` have no camera."""
        n = len(self.rows)
        order = np.arange(n) if order is None else order
        cam_P = np.full((self.n_frames, 3, 4), np.nan)
        if self.camera:
            cam_P[:] = np.diag([scale, scale, 1.0]) @ P_CAMERA
            cam_P[list(blind)] = np.nan
        usable = np.array([r["usable"] for r in self.rows], bool)
        xy = np.array([r["xy"] for r in self.rows], np.float64).reshape(n, 2)
        inp = SimpleNamespace(
            frame=np.array([r["frame"] for r in self.rows], np.int64)[order],
            source=np.array([r["source"] for r in self.rows], np.int64)[order],
            box=(np.array([r["box"] for r in self.rows], np.float64).reshape(n, 4) * scale)[order],
            role=np.array([r["role"] for r in self.rows], dtype=object)[order],
            team=np.array([r["team"] for r in self.rows], dtype=object)[order],
            pitch_xy=np.where(usable[:, None], xy, np.nan)[order],
            pitch_usable=usable[order],
            cam_P=cam_P,
            n_frames=self.n_frames,
            fps=FPS,
            width=W * scale,
            height=H * scale,
            px_scale=scale,
        )
        return inp, np.array([r["id"] for r in self.rows], np.int64)[order]


def _index(ids: np.ndarray) -> dict[int, int]:
    return {int(p): i for i, p in enumerate(ids)}


def _pairs(out: PairTable | PositionScores) -> dict[tuple[int, int], int]:
    return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(out.a, out.b, strict=True))}


def _best_earlier(scores: PositionScores, later: Iterable[int]) -> dict[int, int]:
    """Per later piece: the earlier piece with the highest assignment score."""
    return {b: int(scores.a[scores.b == b][np.argmax(scores.sink[scores.b == b])]) for b in later}


def _evidence(table: PairTable, row: int, n_frames: int) -> tuple[float, float, float]:
    """(z, c_in, c_out) of one row of the pair table, written out from the model: the pair
    evidence w . x and the "nobody" log-weights of the later and of the earlier piece."""
    p = space.DEFAULT_PARAMS
    s0, s1, s2 = p.sigma
    sg = s0 + s1 * float(table.dt[row]) ** s2
    ea, eb = bool(table.a_end_edge[row]), bool(table.b_start_edge[row])
    x = [
        math.log1p((float(table.r_adv_cv[row]) / sg) ** 2),
        math.log(sg),
        float(table.same_team[row] == -1),
        float(ea and eb),
        float(ea != eb),
        min(float(table.fwd_adv_inview_s[row]), p.inview_cap_s),
        min(float(table.back_adv_inview_s[row]), p.inview_cap_s),
    ]
    since_start = min(float(table.b_first[row]) / FPS, p.time_cap_s)
    to_end = min((n_frames - 1 - float(table.a_last[row])) / FPS, p.time_cap_s)
    c_in = p.g_in[0] + p.g_in[1] * float(eb) + p.g_in[2] * math.log1p(since_start)
    c_out = p.g_out[0] + p.g_out[1] * float(ea) + p.g_out[2] * math.log1p(to_end)
    return sum(w * v for w, v in zip(p.w, x, strict=True)), c_in, c_out


# ------------------------------------------------------------------------- parameters


def test_params_carry_the_lab_values_and_are_frozen() -> None:
    p = space.DEFAULT_PARAMS
    assert p == SpaceParams()
    assert len(dataclasses.fields(SpaceParams)) == 30
    assert (p.edge_px_720, p.view_margin_px_720) == (10.0, 20.0)
    assert (p.end_window, p.flow_half, p.step, p.sinkhorn_iters) == (25, 6, 10, 200)
    assert (p.max_speed_fit, p.max_speed, p.start_radius, p.flow_max_speed) == (10, 8, 3, 12)
    assert (p.grid_x, p.grid_y, p.grid_m) == ((-62.0, 62.0), (-44.0, 44.0), 1.0)
    assert (len(p.sigma), len(p.w), len(p.g_in), len(p.g_out)) == (3, 7, 3, 3)
    assert p.w[0] == pytest.approx(-1.4404843400874152, rel=1e-15)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.step = 5  # type: ignore[misc]


def test_params_as_dict_is_json_and_round_trips() -> None:
    stored = json.loads(json.dumps(space.DEFAULT_PARAMS.as_dict()))
    assert stored["grid_x"] == [-62.0, 62.0]
    assert stored["w"] == list(space.DEFAULT_PARAMS.w)
    assert SpaceParams(**stored) == space.DEFAULT_PARAMS


def test_params_report_every_problem() -> None:
    with pytest.raises(ValueError, match="SpaceParams") as error:
        SpaceParams(w=(1.0, 2.0), step=0, grid_m=0.0)
    message = str(error.value)
    assert "w: expected 7 numbers" in message
    assert "step: must be at least 1" in message
    assert "grid_m: must be positive" in message


# ------------------------------------------------------------------------- end and start state


def test_standing_piece_ends_and_starts_where_it_stands() -> None:
    rng = np.random.default_rng(11)
    jitter = 0.02 * rng.normal(size=(40, 2))
    clip = _Clip(60)
    clip.add(7, range(10, 50), lambda f: (12.0 + jitter[f - 10, 0], -5.0 + jitter[f - 10, 1]))
    ends = space.piece_ends(*clip.inputs())
    assert ends.piece.tolist() == [7]
    assert (int(ends.first[0]), int(ends.last[0]), int(ends.plugin[0])) == (10, 49, 7)
    assert (ends.team[0], ends.role[0]) == ("left", "player")
    np.testing.assert_allclose(ends.end_xy[0], [12.0, -5.0], atol=0.05)
    np.testing.assert_allclose(ends.start_xy[0], [12.0, -5.0], atol=0.05)
    np.testing.assert_allclose(ends.end_v[0], [0.0, 0.0], atol=0.2)
    np.testing.assert_allclose(ends.start_v[0], [0.0, 0.0], atol=0.2)
    assert not ends.end_edge[0]
    assert not ends.start_edge[0]


def test_running_piece_has_its_velocity_and_the_position_of_its_last_frame() -> None:
    clip = _Clip(80)
    clip.add(3, range(5, 65), lambda f: (-20.0 + 4.0 * f / FPS, 8.0 - 2.0 * f / FPS))
    ends = space.piece_ends(*clip.inputs())
    np.testing.assert_allclose(ends.end_v[0], [4.0, -2.0], atol=1e-6)
    np.testing.assert_allclose(ends.start_v[0], [4.0, -2.0], atol=1e-6)
    np.testing.assert_allclose(
        ends.end_xy[0], [-20.0 + 4.0 * 64 / FPS, 8.0 - 2.0 * 64 / FPS], atol=1e-6
    )
    np.testing.assert_allclose(
        ends.start_xy[0], [-20.0 + 4.0 * 5 / FPS, 8.0 - 2.0 * 5 / FPS], atol=1e-6
    )


def test_fitted_velocity_is_capped() -> None:
    clip = _Clip(40)
    clip.add(1, range(40), lambda f: (-30.0 + 12.0 * f / FPS, -16.0 * f / FPS))  # 20 m/s
    ends = space.piece_ends(*clip.inputs())
    cap = space.DEFAULT_PARAMS.max_speed_fit
    np.testing.assert_allclose(ends.end_v[0], [0.6 * cap, -0.8 * cap], atol=1e-6)
    np.testing.assert_allclose(
        ends.end_xy[0], [-30.0 + 12.0 * 39 / FPS, -16.0 * 39 / FPS], atol=1e-6
    )


def test_few_rows_give_a_median_position_and_no_velocity() -> None:
    spots = {0: (0.0, 0.0), 1: (2.0, 1.0), 2: (4.0, 2.0), 3: (6.0, 3.0), 4: (30.0, 9.0)}
    clip = _Clip(10)
    clip.add(1, range(5), lambda f: spots[f])
    clip.add(2, [8], (-7.0, 7.0))
    ends = space.piece_ends(*clip.inputs())
    np.testing.assert_allclose(ends.end_xy, [[4.0, 2.0], [-7.0, 7.0]], atol=1e-12)
    np.testing.assert_allclose(ends.start_xy, [[4.0, 2.0], [-7.0, 7.0]], atol=1e-12)
    assert not ends.end_v.any()
    assert not ends.start_v.any()


def test_a_few_wild_rows_do_not_move_the_fit() -> None:
    wild = {30, 34, 38}
    clip = _Clip(50)
    clip.add(1, range(20, 45), lambda f: (9.0, 4.0 + (8.0 if f in wild else 0.0)))
    ends = space.piece_ends(*clip.inputs())
    np.testing.assert_allclose(ends.end_xy[0], [9.0, 4.0], atol=1e-6)
    np.testing.assert_allclose(ends.end_v[0], [0.0, 0.0], atol=1e-6)


def test_border_flags_follow_the_first_and_the_last_box() -> None:
    def touching(first: Box, last: Box) -> Callable[[int], Box]:
        return lambda f: first if f == 0 else last if f == 19 else MID_BOX

    clip = _Clip(20)
    boxes = {
        1: (MID_BOX, (1240.0, 300.0, 1269.5, 380.0)),  # ends at the right border
        2: ((9.5, 300.0, 40.0, 380.0), MID_BOX),  # starts at the left border
        3: ((600.0, 8.0, 630.0, 90.0), (600.0, 630.0, 640.0, 709.5)),  # top, then bottom
        4: ((11.0, 11.0, 40.0, 90.0), (1240.0, 630.0, 1268.0, 708.0)),  # near, not touching
    }
    for pid, (first, last) in boxes.items():
        clip.add(pid, range(20), (0.0, 0.0), box=touching(first, last))
    ends = space.piece_ends(*clip.inputs())
    assert ends.start_edge.tolist() == [False, True, True, False]
    assert ends.end_edge.tolist() == [True, False, True, False]


def test_border_distance_scales_with_the_frame_height() -> None:
    clip = _Clip(20)
    clip.add(1, range(20), (0.0, 0.0), box=(12.0, 300.0, 40.0, 380.0))
    inp, ids = clip.inputs()
    assert not space.piece_ends(inp, ids).start_edge[0]  # 12 px from the border of 720 lines
    inp.width, inp.height, inp.px_scale = 1920.0, 1080.0, 1.5
    assert space.piece_ends(inp, ids).start_edge[0]  # 12 px at 1080 lines are 8 px at 720


def test_rows_cut_by_the_bottom_border_are_left_out() -> None:
    cut = (600.0, 600.0, 640.0, 715.0)
    clip = _Clip(60)
    # the feet of the last ten rows are not seen: their ground points lie 5 m off
    clip.add(
        1,
        range(40),
        lambda f: (0.0, 20.0 if f < 30 else 25.0),
        box=lambda f: MID_BOX if f < 30 else cut,
    )
    # every row of the last second is cut: then they are all there is
    clip.add(2, range(40), (3.0, 25.0), box=cut)
    ends = space.piece_ends(*clip.inputs())
    np.testing.assert_allclose(ends.end_xy, [[0.0, 20.0], [3.0, 25.0]], atol=1e-6)


def test_without_a_placed_row_in_the_window_the_nearest_ones_are_used() -> None:
    clip = _Clip(80)
    clip.add(1, range(70), lambda f: (0.1 * f, 2.0), usable=lambda f: f < 30)
    clip.add(2, range(70), (5.0, 5.0), usable=False)
    ends = space.piece_ends(*clip.inputs())
    # piece 1: rows 25..29 are the placed ones nearest to its last frame; no line, no velocity
    np.testing.assert_allclose(ends.end_xy[0], [2.7, 2.0], atol=1e-9)
    np.testing.assert_allclose(ends.end_v[0], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(ends.start_xy[0], [0.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(ends.start_v[0], [0.1 * FPS, 0.0], atol=1e-6)
    assert np.isnan(ends.end_xy[1]).all()
    assert np.isnan(ends.start_xy[1]).all()
    assert not ends.end_v[1].any()


def test_pieces_come_in_ascending_id_with_majority_labels() -> None:
    clip = _Clip(30)
    clip.add(40, range(10, 20), (0.0, 0.0), source=4, team=lambda f: "right" if f < 16 else "left")
    clip.add(12, range(0, 6), (1.0, 1.0), source=9, team=lambda f: "right" if f < 3 else "left")
    clip.add(-1, range(0, 30), (2.0, 2.0), source=77, role="referee", team=None)
    clip.add(5, range(3, 9), (3.0, 3.0), source=5, role=None, team=None)
    inp, ids = clip.inputs(order=np.random.default_rng(5).permutation(len(clip.rows)))
    ends = space.piece_ends(inp, ids)
    assert ends.piece.tolist() == [5, 12, 40]
    assert ends.plugin.tolist() == [5, 9, 4]
    assert ends.first.tolist() == [3, 0, 10]
    assert ends.last.tolist() == [8, 5, 19]
    assert ends.team.tolist() == [None, "left", "right"]  # a tie goes to the smaller string
    assert ends.role.tolist() == [None, "player", "player"]


# ------------------------------------------------------------------------- the picture


def test_view_margin_is_positive_inside_the_picture_only() -> None:
    def margin(point: Point, P: np.ndarray = P_CAMERA) -> float:
        return float(space.view_margin(P, np.array(point), float(W), float(H), 1.8))

    feet_v = _projected_box(0.0, 0.0)[3]
    head_v = _projected_box(0.0, 0.0)[1]
    assert feet_v == pytest.approx((H - 1) / 2, abs=1e-6)  # the camera looks at the centre spot
    assert margin((0.0, 0.0)) == pytest.approx(head_v, abs=1e-6)  # the head is nearer a border
    assert 0.0 < margin((19.0, 0.0)) < margin((18.0, 0.0)) < margin((0.0, 0.0))  # the right border
    assert margin((30.0, 0.0)) < -100.0  # right of the picture
    assert margin((0.0, 200.0)) == -np.inf  # behind the camera
    assert margin((0.0, 0.0), np.full((3, 4), np.nan)) == -np.inf  # no camera
    many = space.view_margin(
        np.stack([P_CAMERA, P_CAMERA]), np.array([[0.0, 0.0], [30.0, 0.0]]), float(W), float(H), 1.8
    )
    assert many.shape == (2,)
    assert many[0] > 0 > many[1]


def test_in_picture_seconds_count_the_frames_of_the_gap_with_a_camera() -> None:
    clip = _Clip(100, camera=True)
    clip.add(1, range(0, 11), (0.0, 0.0))  # ends in the middle of the picture
    clip.add(2, range(0, 11), (40.0, 0.0), box=MID_BOX)  # ends far outside it
    clip.add(3, range(60, 80), (2.0, 1.0))
    table = space.pair_table(*clip.inputs())
    at = _pairs(table)
    assert sorted(at) == [(1, 3), (2, 3)]
    gap_s = (60 - 1 - 10) / FPS  # frames 11..59
    assert table.fwd_adv_inview_s[at[1, 3]] == pytest.approx(gap_s)
    assert table.back_adv_inview_s[at[1, 3]] == pytest.approx(gap_s)
    assert table.fwd_adv_inview_s[at[2, 3]] == 0.0  # nobody carried it into the picture
    assert table.back_adv_inview_s[at[2, 3]] == pytest.approx(gap_s)
    # frames without a camera cannot say that he would have been seen
    blind = space.pair_table(*clip.inputs(blind=range(30, 45)))
    assert blind.fwd_adv_inview_s[at[1, 3]] == pytest.approx(gap_s - 15 / FPS)
    no_camera = _Clip(100)
    no_camera.rows = clip.rows
    assert not space.pair_table(*no_camera.inputs()).fwd_adv_inview_s.any()


# ------------------------------------------------------------------------- flow


def _block(clip: _Clip, velocity: Point, frames: range, first_id: int = 10) -> None:
    """Six players who move together."""
    spots = [(-8.0, -6.0), (-3.0, 5.0), (0.0, -2.0), (4.0, 8.0), (7.0, -9.0), (10.0, 3.0)]
    for k, (x0, y0) in enumerate(spots):
        clip.add(
            first_id + k,
            frames,
            lambda f, x0=x0, y0=y0: (x0 + velocity[0] * f / FPS, y0 + velocity[1] * f / FPS),
            team="left" if k % 2 else "right",
        )


def test_flow_is_the_common_movement_of_the_players() -> None:
    velocity = (3.0, 0.5)
    clip = _Clip(80)
    _block(clip, velocity, range(80))
    clip.add(30, range(80), (20.0, 20.0), role="referee")  # not a player: no part of the flow
    clip.add(31, range(80), (-20.0, 20.0), usable=False)  # no ground point
    clip.add(32, range(0, 10), (0.0, 20.0))  # too short for a chord
    inp, ids = clip.inputs()
    positions, velocities = space._player_velocities(inp, ids, space.DEFAULT_PARAMS)
    half = space.DEFAULT_PARAMS.flow_half
    assert [len(v) for v in velocities[:half]] == [0] * half  # no chord at the clip's start
    assert len(velocities[40]) == len(positions[40]) == 6
    np.testing.assert_allclose(velocities[40], np.tile(velocity, (6, 1)), atol=1e-9)
    S = space._flow(velocities, FPS, space.DEFAULT_PARAMS)
    assert S.shape == (80, 2)
    np.testing.assert_allclose(S[60] - S[20], np.array(velocity) * 40 / FPS, atol=1e-9)
    # two players are not enough for a flow
    few, few_ids = clip.inputs()
    few_ids = np.where(np.isin(few_ids, [10, 11]), few_ids, -1)
    lonely = space._flow(
        space._player_velocities(few, few_ids, space.DEFAULT_PARAMS)[1], FPS, space.DEFAULT_PARAMS
    )
    assert not lonely.any()


def test_a_player_carried_with_the_block_has_no_residual() -> None:
    velocity = (3.0, 0.5)

    def moving(f: int) -> Point:
        return (-5.0 + velocity[0] * f / FPS, 12.0 + velocity[1] * f / FPS)

    clip = _Clip(80)
    _block(clip, velocity, range(80))
    clip.add(1, range(0, 30), moving)
    clip.add(2, range(50, 80), moving)  # the same player after a gap: he moved with the others
    clip.add(3, range(50, 80), moving(29))  # somebody who stands where piece 1 ended
    table = space.pair_table(*clip.inputs())
    at = _pairs(table)
    assert table.dt[at[1, 2]] == pytest.approx(21 / FPS)
    assert table.r_adv_cv[at[1, 2]] < 0.05
    # the block moved 2.6 m in the gap; half of that is explained by standing still
    assert table.r_adv_cv[at[1, 3]] == pytest.approx(
        0.5 * math.hypot(*velocity) * 21 / FPS, abs=0.1
    )


def test_a_runner_keeps_his_own_velocity_through_a_short_gap() -> None:
    def running(f: int) -> Point:
        return (-10.0 + 6.0 * f / FPS, 15.0)

    clip = _Clip(80)
    _block(clip, (0.0, 0.0), range(80))  # everybody else stands
    clip.add(1, range(0, 30), running)
    clip.add(2, range(45, 80), running)
    clip.add(3, range(45, 80), running(29))  # stands where the runner was last seen
    table = space.pair_table(*clip.inputs())
    at = _pairs(table)
    assert table.r_adv_cv[at[1, 2]] < 0.05
    # against the standing one only the runner's own velocity counts, for half the gap
    assert table.r_adv_cv[at[1, 3]] == pytest.approx(6.0 * (16 / FPS) / 2, abs=0.05)


# ------------------------------------------------------------------------- reach


def test_reach_agrees_with_a_distance_bound() -> None:
    rng = np.random.default_rng(7)
    p = space.DEFAULT_PARAMS
    clip = _Clip(800)
    spot: dict[int, tuple[int, int, Point]] = {}
    for pid in range(44):
        start = int(rng.integers(0, 150)) if pid < 40 else int(rng.integers(600, 790))
        point = (float(rng.uniform(-50.0, 50.0)), float(rng.uniform(-32.0, 32.0)))
        clip.add(pid, range(start, start + 3), point)
        spot[pid] = (start, start + 2, point)
    table = space.pair_table(*clip.inputs())
    assert table.known.all()
    per_step = p.max_speed * p.step / FPS  # metres of a hidden path per step of the programme
    inside = outside = 0
    for a, b, reach in zip(table.a.tolist(), table.b.tolist(), table.reach.tolist(), strict=True):
        steps = max(1, spot[b][0] // p.step - spot[a][1] // p.step)
        distance = math.dist(spot[a][2], spot[b][2])
        if distance <= 0.8 * per_step * steps + p.start_radius:  # clearly within the bound
            assert reach, (a, b, distance, steps)
            inside += 1
        elif distance >= per_step * steps + 2 * p.start_radius + 2.0:  # clearly beyond it
            assert not reach, (a, b, distance, steps)
            outside += 1
    assert inside >= 150
    assert outside >= 150
    late = table.b >= 40  # after a gap this long every point of the pitch is within reach
    assert late.sum() >= 160
    assert table.reach[late & (table.a < 40)].all()


def test_reach_needs_both_points() -> None:
    clip = _Clip(60)
    clip.add(1, range(0, 10), (0.0, 0.0))
    clip.add(2, range(0, 10), (1.0, 0.0), usable=False)
    clip.add(3, range(30, 40), (0.5, 0.5))
    clip.add(4, range(30, 40), (0.5, -0.5), usable=False)
    table = space.pair_table(*clip.inputs())
    at = _pairs(table)
    assert sorted(at) == [(1, 3), (1, 4), (2, 3), (2, 4)]
    assert table.known.tolist() == [True, False, False, False]
    assert table.reach.tolist() == [True, False, False, False]
    assert np.isfinite(table.r_adv_cv[at[1, 3]])
    assert np.isnan(table.r_adv_cv[[at[1, 4], at[2, 3], at[2, 4]]]).all()
    scores = space.position_scores(*clip.inputs())
    assert sorted(_pairs(scores)) == [(1, 3)]


# ------------------------------------------------------------------------- the model


def test_a_single_pair_has_the_closed_form_scores() -> None:
    at_border = (1245.0, 300.0, 1275.0, 380.0)
    clip = _Clip(100)
    clip.add(1, range(0, 30), (5.0, 3.0), box=lambda f: at_border if f == 29 else MID_BOX)
    clip.add(2, range(50, 70), (6.5, 3.5))
    inp, ids = clip.inputs()
    table = space.pair_table(inp, ids)
    assert (table.a.tolist(), table.b.tolist()) == ([1], [2])
    assert (int(table.a_last[0]), int(table.b_first[0])) == (29, 50)
    assert table.dt[0] == pytest.approx(21 / FPS)
    assert table.r_adv_cv[0] == pytest.approx(math.hypot(1.5, 0.5), abs=1e-9)
    assert (bool(table.a_end_edge[0]), bool(table.b_start_edge[0])) == (True, False)
    assert (int(table.same_team[0]), int(table.same_role[0])) == (1, 1)
    assert table.reach[0]
    assert table.known[0]
    again = space.pair_table(inp, ids, ends=space.piece_ends(inp, ids))  # the caller's ends
    for field in dataclasses.fields(PairTable):
        np.testing.assert_array_equal(getattr(again, field.name), getattr(table, field.name))

    scores = space.position_scores(inp, ids)
    z, c_in, c_out = _evidence(table, 0, clip.n_frames)
    assert scores.z[0] == pytest.approx(z, abs=1e-9)
    assert scores.p_in[0] == pytest.approx(math.exp(z) / (math.exp(c_in) + math.exp(z)), rel=1e-9)
    assert scores.p_out[0] == pytest.approx(math.exp(z) / (math.exp(c_out) + math.exp(z)), rel=1e-9)
    # one pair with slack on both sides: M = K (1 - M)^2 / (A B), K = e^z, A = e^c_out, B = e^c_in
    K, AB = math.exp(z), math.exp(c_in + c_out)
    expected = ((2 * K + AB) - math.sqrt((2 * K + AB) ** 2 - 4 * K * K)) / (2 * K)
    assert scores.sink[0] == pytest.approx(expected, rel=1e-5)
    assert 0.0 < scores.sink[0] < min(scores.p_in[0], 1.0)
    assert scores.pieces.piece.tolist() == [1, 2]


def test_sinkhorn_rows_and_columns_sum_to_one_with_their_slack() -> None:
    rng = np.random.default_rng(3)
    keep = rng.random((7, 6)) < 0.6
    keep[np.arange(6), np.arange(6)] = True  # every later piece has a candidate
    keep[6, 0] = True  # and every earlier one
    ia, ib = np.nonzero(keep)
    na, ga, _ = space._groups(ia)
    nb, gb, _ = space._groups(ib)
    assert (na, nb) == (7, 6)
    z = rng.normal(-2.0, 2.0, len(ia))
    ca, cb = rng.normal(-1.0, 1.0, na), rng.normal(-2.0, 1.0, nb)
    p_in, p_out, sink = space._scores(z, gb, nb, cb, ga, na, ca, space.DEFAULT_PARAMS)
    np.testing.assert_allclose(
        p_in, np.exp(z) / (np.exp(cb) + np.bincount(gb, np.exp(z), nb))[gb], rtol=1e-9
    )
    np.testing.assert_allclose(
        p_out, np.exp(z) / (np.exp(ca) + np.bincount(ga, np.exp(z), na))[ga], rtol=1e-9
    )
    slack_a = 1.0 - np.bincount(ga, sink, na)  # "nobody comes after a"
    slack_b = 1.0 - np.bincount(gb, sink, nb)  # "nobody came before b"
    assert (slack_a > 0).all()
    assert (slack_b > 0).all()
    # M_ab = u_a e^z v_b with u_a e^ca = slack_a and v_b e^cb = slack_b
    np.testing.assert_allclose(
        sink, slack_a[ga] * slack_b[gb] * np.exp(z - ca[ga] - cb[gb]), rtol=1e-4
    )


def _crowd() -> _Clip:
    """Three players who vanish around frame 20 and four who appear around frame 45."""
    clip = _Clip(90)
    clip.add(1, range(0, 20), (0.0, 0.0))
    clip.add(2, range(2, 22), (4.0, 1.0))
    clip.add(3, range(0, 18), (-3.0, 5.0), team="right")
    clip.add(11, range(45, 80), (0.8, 0.3))
    clip.add(12, range(47, 80), (4.5, 2.0))
    clip.add(13, range(44, 80), (-2.0, 4.0), team="right")
    clip.add(14, range(46, 80), (1.5, 2.5), team=None)
    return clip


def test_scores_of_a_scene_are_a_doubly_normalised_assignment() -> None:
    clip = _crowd()
    inp, ids = clip.inputs()
    table = space.pair_table(inp, ids)
    scores = space.position_scores(inp, ids)
    at = _pairs(table)
    # same team or a missing label, in the table's order
    assert list(_pairs(scores)) == [
        (1, 11),
        (1, 12),
        (1, 14),
        (2, 11),
        (2, 12),
        (2, 14),
        (3, 13),
        (3, 14),
    ]
    z, c_in, c_out = np.array(
        [_evidence(table, at[pair], clip.n_frames) for pair in _pairs(scores)]
    ).T
    np.testing.assert_allclose(scores.z, z, atol=1e-9)
    _, ga = np.unique(scores.a, return_inverse=True)
    _, gb = np.unique(scores.b, return_inverse=True)
    slack_a = 1.0 - np.bincount(ga, scores.sink)
    slack_b = 1.0 - np.bincount(gb, scores.sink)
    assert (slack_a > 0).all()
    assert (slack_b > 0).all()
    np.testing.assert_allclose(
        scores.sink, slack_a[ga] * slack_b[gb] * np.exp(z - c_out - c_in), rtol=1e-3
    )
    # P(a | b) over the earlier candidates of b leaves the share of "nobody"
    weight = np.bincount(gb, np.exp(z))
    nobody = np.zeros(len(weight))
    nobody[gb] = np.exp(c_in)
    np.testing.assert_allclose(np.bincount(gb, scores.p_in), weight / (nobody + weight), rtol=1e-9)
    # the nearest candidate is the likely one
    assert _best_earlier(scores, (11, 12, 13)) == {11: 1, 12: 2, 13: 3}
    assert scores.sink[_pairs(scores)[1, 11]] > 0.5 > 0.2 > scores.sink[_pairs(scores)[2, 11]]


def test_a_rival_takes_its_share() -> None:
    alone = _Clip(90)
    alone.add(1, range(0, 20), (0.0, 0.0))
    alone.add(11, range(45, 80), (0.8, 0.3))
    one = space.position_scores(*alone.inputs())
    rival = _Clip(90)
    rival.rows = list(alone.rows)
    rival.add(2, range(0, 20), (1.6, 0.6))  # as near to where 11 appears as piece 1
    two = space.position_scores(*rival.inputs())
    assert two.a.tolist() == [1, 2]
    assert two.z[0] == pytest.approx(one.z[0], abs=1e-9)  # the evidence of a pair is its own
    assert two.sink[0] == pytest.approx(two.sink[1], rel=1e-3)
    assert two.sink[0] < 0.6 * one.sink[0]
    assert two.sink.sum() < 1.0
    assert two.p_in.sum() < 1.0


def test_admissible_pairs_have_one_role_and_no_team_conflict() -> None:
    clip = _Clip(80)
    clip.add(1, range(0, 20), (0.0, 0.0))
    clip.add(9, range(0, 20), (0.0, 1.0), role=None)
    later = {
        2: {},
        3: {"team": "right"},
        4: {"role": "goalkeeper"},
        5: {"team": None},
        6: {"role": None},
        7: {"where": (50.0, 30.0)},
        8: {"usable": False},
    }
    for pid, how in later.items():
        clip.add(pid, range(40, 60), how.pop("where", (1.0, 0.0)), **how)
    inp, ids = clip.inputs()
    table = space.pair_table(inp, ids)
    at = _pairs(table)
    assert len(at) == 14
    assert [int(table.same_team[at[1, b]]) for b in (2, 3, 5)] == [1, 0, -1]
    assert [int(table.same_role[at[1, b]]) for b in (2, 4, 6)] == [1, 0, 0]
    assert int(table.same_role[at[9, 6]]) == 0  # two missing labels are not the same role
    assert not table.reach[at[1, 7]]
    assert not table.known[at[1, 8]]
    scores = space.position_scores(inp, ids)
    assert list(_pairs(scores)) == [(1, 2), (1, 5)]
    assert scores.pieces.piece.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # the same geometry without a team label: only the weight of "a label is missing"
    assert scores.z[1] - scores.z[0] == pytest.approx(space.DEFAULT_PARAMS.w[2], abs=1e-9)


def test_rows_of_no_piece_take_no_part() -> None:
    clip = _crowd()
    inp, ids = clip.inputs()
    masked = space.position_scores(inp, np.where(np.isin(ids, [2, 14]), -1, ids))
    without = _Clip(clip.n_frames)
    without.rows = [r for r in clip.rows if r["id"] not in (2, 14)]
    plain = space.position_scores(*without.inputs())
    assert list(_pairs(masked)) == list(_pairs(plain)) == [(1, 11), (1, 12), (3, 13)]
    assert masked.pieces.piece.tolist() == [1, 3, 11, 12, 13]
    for name in ("z", "p_in", "p_out", "sink"):
        np.testing.assert_allclose(getattr(masked, name), getattr(plain, name), rtol=1e-9)
    everybody = space.position_scores(inp, ids)
    assert masked.sink[0] > everybody.sink[0]  # fewer rivals for piece 11


def test_row_order_does_not_matter() -> None:
    clip = _crowd()
    plain = space.position_scores(*clip.inputs())
    order = np.random.default_rng(2).permutation(len(clip.rows))
    mixed = space.position_scores(*clip.inputs(order=order))
    assert list(_pairs(mixed)) == list(_pairs(plain))
    for name in ("z", "p_in", "p_out", "sink"):
        np.testing.assert_allclose(getattr(mixed, name), getattr(plain, name), rtol=1e-9)


# ------------------------------------------------------------------------- frame size


def _match(n_frames: int = 150) -> _Clip:
    """A scene in the picture: a block of players drifts, two runners leave the picture at
    its right and left border and come back, one player is lost for a while in the middle."""
    clip = _Clip(n_frames, camera=True)
    _block(clip, (0.6, 0.2), range(n_frames), first_id=20)
    clip.add(1, range(0, 70), lambda f: (6.0 + 6.0 * f / FPS, -4.0))  # out at the right
    clip.add(2, range(95, n_frames), lambda f: (23.0 - 5.0 * (f - 95) / FPS, -6.0))
    clip.add(3, range(0, 80), lambda f: (-8.0 - 5.0 * f / FPS, 6.0), team="right")  # out left
    clip.add(4, range(100, n_frames), lambda f: (-21.0 + 6.0 * (f - 100) / FPS, 4.0), team="right")
    clip.add(5, range(0, 41), (-3.0, 10.0))
    clip.add(6, range(70, n_frames), (-2.5, 10.5))
    clip.add(7, range(0, 51), (12.0, -20.0), role="referee", team=None)
    clip.add(8, range(80, n_frames), (11.0, -18.0), role="referee", team=None)
    clip.add(9, range(110, n_frames), (2.0, -12.0), team=None)
    clip.add(10, range(0, 70), lambda f: (9.0 + 5.0 * f / FPS, 2.0))  # out at the right too
    return clip


def test_the_same_scene_at_720_and_1080_lines() -> None:
    clip = _match()
    small = clip.inputs()
    large = clip.inputs(scale=1.5)
    assert (large[0].width, large[0].height, large[0].px_scale) == (1920.0, 1080.0, 1.5)
    t720, t1080 = space.pair_table(*small), space.pair_table(*large)
    s720, s1080 = space.position_scores(*small), space.position_scores(*large)
    # the scene has what the frame size could change: borders, and paths entering the picture
    ends = s720.pieces
    at = _index(ends.piece)
    assert ends.end_edge[[at[1], at[3]]].all()
    assert ends.start_edge[[at[2], at[4]]].all()
    assert not ends.end_edge[at[5]]
    assert not ends.start_edge[at[6]]
    gap = t720.b_first - 1 - t720.a_last
    for seconds in (t720.fwd_adv_inview_s, t720.back_adv_inview_s):
        partly = (seconds > 0) & (seconds * FPS < gap - 0.5)  # a path that crosses the border
        assert partly.sum() >= 3
    assert len(s720.a) >= 6

    for name in (
        "a",
        "b",
        "known",
        "reach",
        "a_end_edge",
        "b_start_edge",
        "same_team",
        "same_role",
        "a_last",
        "b_first",
    ):
        np.testing.assert_array_equal(getattr(t1080, name), getattr(t720, name), err_msg=name)
    for name in ("dt", "r_adv_cv", "fwd_adv_inview_s", "back_adv_inview_s"):
        np.testing.assert_allclose(
            getattr(t1080, name), getattr(t720, name), rtol=0, atol=1e-9, err_msg=name
        )
    for name in ("end_edge", "start_edge"):
        np.testing.assert_array_equal(getattr(s1080.pieces, name), getattr(ends, name))
    for name in ("end_xy", "start_xy", "end_v", "start_v"):
        np.testing.assert_allclose(
            getattr(s1080.pieces, name), getattr(ends, name), rtol=0, atol=1e-9
        )
    np.testing.assert_array_equal(s1080.a, s720.a)
    np.testing.assert_array_equal(s1080.b, s720.b)
    for name in ("z", "p_in", "p_out", "sink"):
        np.testing.assert_allclose(
            getattr(s1080, name), getattr(s720, name), rtol=0, atol=1e-9, err_msg=name
        )
    # the returns are found at both sizes
    assert _best_earlier(s1080, (2, 4, 6)) == {2: 1, 4: 3, 6: 5}


def test_pixel_thresholds_are_given_at_720_lines() -> None:
    # the 1080-line scene read with the 720-line threshold: a carried point then counts as
    # seen when it is 20 instead of 30 px inside the frame
    clip = _match()
    inp, ids = clip.inputs(scale=1.5)
    right = space.pair_table(inp, ids)
    inp.px_scale = 1.0
    wrong = space.pair_table(inp, ids)
    assert wrong.fwd_adv_inview_s.sum() > right.fwd_adv_inview_s.sum()
    assert (wrong.fwd_adv_inview_s >= right.fwd_adv_inview_s).all()


# ------------------------------------------------------------------------- nothing to do


def _assert_no_scores(scores: PositionScores, pieces: list[int]) -> None:
    assert scores.pieces.piece.tolist() == pieces
    assert scores.a.dtype == scores.b.dtype == np.int64
    for name in ("a", "b", "z", "p_in", "p_out", "sink"):
        assert getattr(scores, name).shape == (0,), name


def test_no_piece_gives_empty_results(recwarn: pytest.WarningsRecorder) -> None:
    empty = _Clip(50)
    inp, ids = empty.inputs()
    assert len(ids) == 0
    table = space.pair_table(inp, ids)
    for field in dataclasses.fields(PairTable):
        assert getattr(table, field.name).shape == (0,), field.name
    _assert_no_scores(space.position_scores(inp, ids), [])
    ends = space.piece_ends(inp, ids)
    assert ends.end_xy.shape == ends.start_v.shape == (0, 2)

    untracked = _Clip(50)
    untracked.add(-1, range(50), (0.0, 0.0))
    _assert_no_scores(space.position_scores(*untracked.inputs()), [])
    assert len(recwarn) == 0


def test_no_pair_gives_empty_results(recwarn: pytest.WarningsRecorder) -> None:
    alone = _Clip(50)
    alone.add(4, range(0, 50), (0.0, 0.0))
    _assert_no_scores(space.position_scores(*alone.inputs()), [4])

    together = _Clip(50)  # pieces that share a frame are no pair
    together.add(1, range(0, 30), (0.0, 0.0))
    together.add(2, range(30, 50), (1.0, 0.0))
    together.add(3, range(10, 40), (2.0, 0.0))
    table = space.pair_table(*together.inputs())
    assert (table.a.tolist(), table.b.tolist()) == ([1], [2])
    overlapping = _Clip(50)
    overlapping.rows = [r for r in together.rows if r["id"] != 2]
    assert space.pair_table(*overlapping.inputs()).a.shape == (0,)
    _assert_no_scores(space.position_scores(*overlapping.inputs()), [1, 3])
    one_frame = _Clip(1)
    one_frame.add(1, [0], (0.0, 0.0))
    one_frame.add(2, [0], (1.0, 0.0))
    assert space.pair_table(*one_frame.inputs()).a.shape == (0,)
    _assert_no_scores(space.position_scores(*one_frame.inputs()), [1, 2])
    assert len(recwarn) == 0


def test_no_admissible_pair_and_no_camera_give_no_warning(recwarn: pytest.WarningsRecorder) -> None:
    lost = _Clip(60, camera=True)  # no ground point anywhere
    lost.add(1, range(0, 20), (0.0, 0.0), usable=False)
    lost.add(2, range(30, 50), (1.0, 0.0), usable=False)
    table = space.pair_table(*lost.inputs())
    assert (table.a.tolist(), table.b.tolist()) == ([1], [2])
    assert not table.known[0]
    assert not table.reach[0]
    assert np.isnan(table.r_adv_cv[0])
    _assert_no_scores(space.position_scores(*lost.inputs()), [1, 2])

    apart = _Clip(60)  # another role, and no camera on any frame
    apart.add(1, range(0, 20), (0.0, 0.0))
    apart.add(2, range(30, 50), (1.0, 0.0), role="goalkeeper")
    _assert_no_scores(space.position_scores(*apart.inputs()), [1, 2])

    blind = _crowd()  # positions but no camera: the scores exist, nothing counts as seen
    scores = space.position_scores(*blind.inputs())
    assert len(scores.a) == 8
    assert np.isfinite(scores.sink).all()
    assert len(recwarn) == 0


def test_malformed_input_is_refused() -> None:
    clip = _crowd()
    inp, ids = clip.inputs()
    with pytest.raises(ValueError, match="ids: expected one id per row"):
        space.position_scores(inp, ids[:-1])
    with pytest.raises(ValueError, match="ids: expected integers"):
        space.piece_ends(inp, ids.astype(float))
    short = SimpleNamespace(**{**vars(inp), "box": inp.box[:-2], "team": inp.team[:-1]})
    with pytest.raises(ValueError, match=r"box: expected \d+ rows, got \d+; team: expected"):
        space.piece_ends(short, ids)
    inp.n_frames = 60
    with pytest.raises(
        ValueError, match=r"frame: values must lie in 0\.\.59.*cam_P: expected shape"
    ):
        space.pair_table(inp, ids)
