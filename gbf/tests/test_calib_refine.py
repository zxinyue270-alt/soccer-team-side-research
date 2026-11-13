"""Tests for gbf.calib_refine on synthetic broadcast footage with a known camera."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from gbf import calib_refine as cr
from gbf.geometry import Camera, look_at, project
from gbf.synth.pitch import landmarks, pitch_markings

W, H = 1920, 1080
TRUE_C = np.array([0.25, 70.3, -17.4])


def _pitch_points() -> np.ndarray:
    points = [p for p in landmarks().values()]
    for marking in pitch_markings(arc_step_deg=10.0):
        points.extend(marking.points)
    unique = np.unique(np.round(np.asarray(points), 3), axis=0)
    return unique[unique[:, 2] == 0.0]


PITCH_POINTS = _pitch_points()


def _camera(C: np.ndarray, target: np.ndarray, focal: float) -> Camera:
    K = np.array([[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]])
    return Camera(K=K, R=look_at(C, target), C=C, width=W, height=H)


def _targets(n: int, start_x: float, end_x: float) -> list[np.ndarray]:
    xs = start_x + (end_x - start_x) * (1 - np.cos(np.linspace(0, math.pi, n))) / 2
    return [np.array([x, -4.0, 0.0]) for x in xs]


def _keypoints(
    camera: Camera, rng: np.random.Generator, n_points: int = 14, noise_px: float = 0.7
) -> np.ndarray:
    uv, depth, valid = project(camera, PITCH_POINTS)
    inside = (
        valid & (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    noise = rng.normal(0.0, noise_px, size=(int(inside.sum()), 2))
    pix = np.round(uv[inside] / 4.0) * 4.0 + noise
    order = rng.permutation(len(pix))[:n_points]
    world = PITCH_POINTS[inside][order]
    return np.column_stack([pix[order], world, np.full(len(order), 0.9)])


def _scene(
    seed: int = 0,
    shots: tuple[tuple[int, float, float, np.ndarray], ...] | None = None,
    jitter_m: float = 4.0,
    n_points: int = 14,
    noise_px: float = 0.7,
):
    """Frames of one or more wide shots, true cameras, plugin-like jittery calibrations."""
    rng = np.random.default_rng(seed)
    shots = shots or ((60, -12.0, 12.0, TRUE_C),)
    true_cams, rows, kps, shot_rows = {}, [], {}, []
    frame = 0
    for shot_id, (n, x0, x1, centre) in enumerate(shots):
        shot_rows.append(
            {
                "shot_id": shot_id,
                "start_frame": frame,
                "end_frame": frame + n - 1,
                "shot_type": "wide",
                "fresh_share": 1.0,
                "median_box_h": 100.0,
            }
        )
        focals = 3200 + 200 * np.sin(np.linspace(0, 2, n))
        for target, focal in zip(_targets(n, x0, x1), focals, strict=True):
            cam = _camera(centre, target, focal)
            true_cams[frame] = cam
            kps[frame] = _keypoints(cam, rng, n_points, noise_px)
            # Plugin-like fit: slide along the viewing ray and zoom to compensate.
            ray = (centre - target) / np.linalg.norm(centre - target)
            moved = centre + rng.normal(0.0, jitter_m) * ray
            plugin = _camera(
                moved,
                target,
                focal * np.linalg.norm(moved - target) / np.linalg.norm(centre - target),
            )
            uv, _, _ = project(plugin, kps[frame][:, 2:5])
            rep = float(np.median(np.hypot(*(uv - kps[frame][:, :2]).T)))
            rows.append(
                {
                    "frame_index": frame,
                    "width": W,
                    "height": H,
                    "K": plugin.K.ravel().copy(),
                    "R": plugin.R.ravel().copy(),
                    "C": plugin.C.copy(),
                    "calib_status": "fresh",
                    "calib_age": 0,
                    "rep_err": rep,
                    "rep_err_n_keypoints": len(kps[frame]),
                    "pitch_usable": rep <= 8.0,
                }
            )
            frame += 1
    return pd.DataFrame(rows), kps, pd.DataFrame(shot_rows), true_cams


def _players(true_cams: dict[int, Camera], seed: int = 1) -> tuple[pd.DataFrame, dict]:
    """Players walking in straight lines; foot pixels from the true camera."""
    rng = np.random.default_rng(seed)
    starts = rng.uniform([-20, -25], [20, 20], size=(8, 2))
    velocity = rng.uniform(-3, 3, size=(8, 2)) / 25.0
    rows, truth = [], {}
    for frame, cam in true_cams.items():
        for k in range(8):
            p = starts[k] + velocity[k] * frame
            uv, depth, valid = project(cam, np.array([[p[0], p[1], 0.0]]))
            if not valid[0] or not (0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H):
                continue
            rows.append(
                {
                    "frame": frame,
                    "track_id": k,
                    "foot_u": uv[0, 0],
                    "foot_v": uv[0, 1],
                    "calib_status": "fresh",
                    "pitch_usable": True,
                    "pitch_x": np.nan,
                    "pitch_y": np.nan,
                    "in_pitch": True,
                }
            )
            truth[(frame, k)] = p
    return pd.DataFrame(rows), truth


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra @ Rb.T).as_rotvec())))


def test_refine_recovers_centre_and_removes_ground_jitter() -> None:
    calib, kps, shots, true_cams = _scene()
    result = cr.refine_calibration(calib, kps, shots)
    out = result.calib
    assert out["refined"].all()
    centre = np.asarray(out["C"].iloc[0])
    assert np.linalg.norm(centre - TRUE_C) < 1.0
    tracks, truth = _players(true_cams)
    before = cr.reproject_tracks(tracks, calib)
    after = cr.reproject_tracks(tracks, out)
    assert cr.ground_jitter(after)["p90_m"] < 0.5 * cr.ground_jitter(before)["p90_m"]
    errors = [
        np.hypot(*(row[["pitch_x", "pitch_y"]].to_numpy(float) - truth[(row.frame, row.track_id)]))
        for _, row in after.iterrows()
    ]
    assert np.median(errors) < 0.3
    holdout = result.report["holdout"]
    assert holdout["refined"]["median_px"] < holdout["free_centre"]["median_px"]


def test_cut_is_not_smoothed_across() -> None:
    shots = ((40, -15.0, -5.0, TRUE_C), (40, 20.0, 25.0, TRUE_C))
    calib, kps, shots_df, true_cams = _scene(seed=3, shots=shots)
    out = cr.refine_calibration(
        calib, kps, shots_df, cr.RefineSettings(holdout_fraction=0.0)
    ).calib.set_index("frame_index")
    for frame in (39, 40, 41):
        refined_R = np.asarray(out.loc[frame, "R"]).reshape(3, 3)
        assert _angle_deg(refined_R, true_cams[frame].R) < 0.2, frame


def test_smoothing_prior_reduces_rotation_noise() -> None:
    calib, kps, shots, true_cams = _scene(
        seed=9, shots=((60, -8.0, 8.0, TRUE_C),), jitter_m=0.0, n_points=5, noise_px=3.0
    )

    def median_error(settings: cr.RefineSettings) -> float:
        out = cr.refine_calibration(calib, kps, shots, settings).calib
        return float(
            np.median(
                [
                    _angle_deg(np.asarray(r).reshape(3, 3), true_cams[f].R)
                    for f, r in zip(out["frame_index"], out["R"], strict=True)
                ]
            )
        )

    smooth = median_error(cr.RefineSettings(holdout_fraction=0.0))
    rough = median_error(
        cr.RefineSettings(holdout_fraction=0.0, sigma_rot_acc=1000.0, sigma_logf_acc=1000.0)
    )
    assert smooth < 0.6 * rough


def test_camera_groups_split_distinct_centres_and_skip_close_shots() -> None:
    other = TRUE_C + np.array([0.0, 25.0, -5.0])
    shots = ((30, -10.0, 0.0, TRUE_C), (30, 0.0, 10.0, other), (30, 5.0, 15.0, TRUE_C))
    calib, kps, shots_df, _ = _scene(seed=4, shots=shots, jitter_m=1.0)
    shots_df.loc[len(shots_df)] = {
        "shot_id": 3,
        "start_frame": 90,
        "end_frame": 99,
        "shot_type": "close",
        "fresh_share": 0.0,
        "median_box_h": 600.0,
    }
    groups = cr.camera_groups(shots_df, calib)
    assert groups == [[0, 2], [1]]


def test_short_gaps_are_bridged_and_long_gaps_are_not_usable() -> None:
    calib, kps, shots, _ = _scene(seed=5, shots=((80, -12.0, 12.0, TRUE_C),))
    for frame in list(range(20, 26)) + list(range(40, 70)):
        kps[frame] = kps[frame][:2]
    out = cr.refine_calibration(
        calib, kps, shots, cr.RefineSettings(holdout_fraction=0.0)
    ).calib.set_index("frame_index")
    assert out.loc[20:25, "refined"].all()
    assert out.loc[20:25, "pitch_usable"].all()
    assert not out.loc[40:69, "refined"].any()
    assert not out.loc[40:69, "pitch_usable"].any()
    assert out.loc[70:79, "refined"].all()


def test_frames_outside_groups_keep_plugin_status() -> None:
    calib, kps, shots, _ = _scene(seed=6, shots=((40, -5.0, 5.0, TRUE_C),))
    shots.loc[0, "shot_type"] = "medium"
    out = cr.refine_calibration(calib, kps, shots, cr.RefineSettings(holdout_fraction=0.0)).calib
    assert not out["refined"].any()
    assert out["pitch_usable"].tolist() == calib["pitch_usable"].tolist()
    assert out["calib_status"].tolist() == calib["calib_status"].tolist()


@pytest.mark.parametrize(
    ("held", "reason"),
    [
        (False, "0 frame(s) to fit, at least 3 needed"),
        (True, "none of the 15 frames to fit has a fresh plugin calibration"),
    ],
)
def test_group_that_cannot_be_fitted_keeps_the_plugin_calibration(held: bool, reason: str) -> None:
    """The second camera cannot be fitted: too few keypoints on every frame, or keypoints
    only on frames without a fresh plugin fit to start from. Its frames keep the plugin's
    calibration and usability, the report says why, and the first camera is refined."""
    other = TRUE_C + np.array([0.0, 25.0, -5.0])
    shots = ((30, -10.0, 0.0, TRUE_C), (30, 0.0, 10.0, other))
    calib, kps, shots_df, _ = _scene(seed=4, shots=shots, jitter_m=1.0)
    for frame in range(30, 45 if held else 60):
        kps[frame] = kps[frame][:2]
    if held:
        calib.loc[calib["frame_index"] >= 45, "calib_status"] = "held"
    before = calib.set_index("frame_index")
    assert before.loc[30:59, "pitch_usable"].all()
    result = cr.refine_calibration(calib, kps, shots_df, cr.RefineSettings(holdout_fraction=0.0))
    out = result.calib.set_index("frame_index")
    assert out.loc[0:29, "refined"].all()
    assert out.loc[0:29, "pitch_usable"].all()
    assert not out.loc[30:59, "refined"].any()
    for column in ("pitch_usable", "calib_status", "rep_err"):
        assert out.loc[30:59, column].tolist() == before.loc[30:59, column].tolist(), column
    for column in ("K", "R", "C"):
        assert all(
            np.array_equal(a, b)
            for a, b in zip(out.loc[30:59, column], before.loc[30:59, column], strict=True)
        ), column
    first, second = result.report["groups"]
    assert first["shots"] == [0]
    assert "not_built" not in first
    assert second["shots"] == [1]
    assert second["frames"] == 0
    assert second["not_built"].startswith(reason)
    assert second["not_built"].endswith("frames keep the plugin calibration")
    counts = result.report["counts"]
    assert counts["refined_frames"] == 30
    assert counts["pitch_usable_after"] == 60


def test_group_whose_refit_is_far_worse_than_per_frame_fits_keeps_the_plugin_calibration() -> None:
    """A missed cut joins two cameras 30 m apart along the touchline into one wide shot, so
    one fixed centre cannot fit it (rotation and zoom cannot make up for the parallax): the
    refit's held-out P90 is many times that of per-frame fits with a free centre (the
    plugin's model). That group keeps the plugin's calibration and usability, as a group that
    cannot be fitted does, its report entry says why, and it stays out of the held-out
    summary of the refits that were used; the first camera is refined as before."""
    shots = (
        (30, -10.0, 0.0, TRUE_C),
        (20, -5.0, 5.0, TRUE_C + np.array([25.0, 0.0, -3.0])),
        (20, 0.0, 10.0, TRUE_C + np.array([55.0, 0.0, -6.0])),
    )
    calib, kps, shots_df, _ = _scene(seed=4, shots=shots, jitter_m=1.0)
    shots_df = shots_df.iloc[[0, 1]].assign(end_frame=[29, 69])  # the cut at frame 50 was missed
    before = calib.set_index("frame_index")
    assert before["pitch_usable"].all()
    result = cr.refine_calibration(calib, kps, shots_df)
    out = result.calib.set_index("frame_index")
    first, joined = result.report["groups"]
    assert out.loc[0:29, "refined"].all()
    assert out.loc[0:29, "pitch_usable"].all()
    held = joined["holdout"]
    assert held["refined"]["p90_px"] > 2.0 * held["free_centre"]["p90_px"]
    assert joined["fallback"].endswith("frames keep the plugin calibration")
    assert "rejected" not in joined
    assert not out.loc[30:69, "refined"].any()
    for column in ("pitch_usable", "calib_status", "rep_err"):
        assert out.loc[30:69, column].tolist() == before.loc[30:69, column].tolist(), column
    for column in ("K", "R", "C"):
        assert all(
            np.array_equal(a, b)
            for a, b in zip(out.loc[30:69, column], before.loc[30:69, column], strict=True)
        ), column
    assert result.report["holdout"]["refined"] == first["holdout"]["refined"]
    counts = result.report["counts"]
    assert counts["refined_frames"] == 30
    assert counts["fallback_group_frames"] == 40
    assert counts["rejected_group_frames"] == 0
    assert counts["pitch_usable_after"] == 70


def test_refinement_is_deterministic() -> None:
    calib, kps, shots, _ = _scene(seed=7, shots=((16, -5.0, 5.0, TRUE_C),))
    a = cr.refine_calibration(calib, kps, shots)
    b = cr.refine_calibration(calib, kps, shots)
    assert a.report == b.report
    assert all(np.array_equal(x, y) for x, y in zip(a.calib["R"], b.calib["R"], strict=True))


def test_the_report_says_whether_each_refit_converged() -> None:
    """A refit the evaluation budget stopped (solver status 0) has not converged: it can stop
    anywhere, and where depends on floating-point detail (on the dev clip SNGS-023 the same
    400 evaluations ended elsewhere with another BLAS thread count). The group entry says
    whether the solver met its tolerances."""
    calib, kps, shots, _ = _scene(seed=7, shots=((16, -5.0, 5.0, TRUE_C),))
    stopped = cr.refine_calibration(
        calib, kps, shots, cr.RefineSettings(max_nfev=1, holdout_fraction=0.0)
    ).report["groups"][0]
    assert (stopped["solver_status"], stopped["solver_nfev"], stopped["converged"]) == (0, 1, False)
    done = cr.refine_calibration(calib, kps, shots).report["groups"][0]
    assert done["solver_status"] > 0
    assert done["converged"] is True


def test_the_report_says_whether_the_held_out_fit_converged() -> None:
    """The held-out check refits each group on part of its keypoints, and that fit decides
    whether the group falls back or is rejected; its entry says whether the solver met its
    tolerances, as the full refit's does (on the dev clip SNGS-058, where the check decided a
    fallback, the report did not say)."""
    calib, kps, shots, _ = _scene(seed=7, shots=((16, -5.0, 5.0, TRUE_C),))
    report = cr.refine_calibration(calib, kps, shots, cr.RefineSettings(max_nfev=1)).report
    held = report["groups"][0]["holdout"]
    assert (held["solver_status"], held["solver_nfev"], held["converged"]) == (0, 1, False)
    done = cr.refine_calibration(calib, kps, shots).report["groups"][0]["holdout"]
    assert done["solver_status"] > 0
    assert done["converged"] is True


def test_group_whose_refit_did_not_converge_keeps_the_plugin_calibration() -> None:
    """A refit the evaluation budget stopped is wherever the solver happened to be, not the
    fit the method defines (on the dev clip SNGS-058 one left 90 of the plugin's 659 usable
    frames usable). Its group keeps the plugin's calibration and usability, as a group whose
    refit is far worse than per-frame fits does, and its report entry says why."""
    calib, kps, shots, _ = _scene(seed=7, shots=((16, -5.0, 5.0, TRUE_C),))
    before = calib.set_index("frame_index")
    result = cr.refine_calibration(
        calib, kps, shots, cr.RefineSettings(max_nfev=1, holdout_fraction=0.0)
    )
    out = result.calib.set_index("frame_index")
    entry = result.report["groups"][0]
    assert (entry["solver_status"], entry["converged"]) == (0, False)
    assert entry["fallback"].endswith("frames keep the plugin calibration")
    assert not out["refined"].any()
    for column in ("pitch_usable", "calib_status", "rep_err"):
        assert out[column].tolist() == before[column].tolist(), column
    for column in ("K", "R", "C"):
        assert all(
            np.array_equal(a, b) for a, b in zip(out[column], before[column], strict=True)
        ), column
    counts = result.report["counts"]
    assert (counts["refined_frames"], counts["fallback_group_frames"]) == (0, 16)
    assert counts["pitch_usable_after"] == counts["pitch_usable_before"]


def test_load_keypoints_round_trip_and_problems(unicode_tmp_path: Path) -> None:
    good = unicode_tmp_path / "关键点.jsonl"
    good.write_text(
        json.dumps({"frame": 0, "points": [[10.0, 20.0, 1.0, 2.0, 0.0, 0.8, 7]]})
        + "\n"
        + json.dumps({"frame": 2, "points": []})
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    loaded = cr.load_keypoints(good)
    assert sorted(loaded) == [0, 2]
    assert loaded[0].shape == (1, 6)
    assert loaded[2].shape == (0, 6)
    bad = unicode_tmp_path / "bad.jsonl"
    bad.write_text(
        "not json\n"
        + json.dumps({"frame": -1, "points": []})
        + "\n"
        + json.dumps({"frame": 3, "points": [[1, 2, 3]]})
        + "\n"
        + json.dumps({"frame": 4, "points": [[1, 2, 3, 4, 5, 1.5]]})
        + "\n"
        + json.dumps({"frame": 5, "points": []})
        + "\n"
        + json.dumps({"frame": 5, "points": []})
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(cr.CalibRefineError) as caught:
        cr.load_keypoints(bad)
    assert len(caught.value.problems) == 5


def test_load_keypoints_checks_each_point_before_reshaping(unicode_tmp_path: Path) -> None:
    """Points of the wrong length must be reported as such, never regrouped: six points of
    five numbers would read as five points of six, two of three as one point of six."""
    good = [10.0, 20.0, 1.0, 2.0, 0.0, 0.8]
    rows = [
        {"frame": 0, "points": [good[:5]] * 6},
        {"frame": 1, "points": [good[:3], good[3:]]},
        {"frame": 2, "points": [[good[:2], good[2:4], good[4:]]]},
        {"frame": 3, "points": [good, [*good[:5], "0.8"]]},
        {"frame": 4, "points": [[*good, 7], 5.0]},
        {"frame": 5, "points": [[10**400, *good[1:]]]},
    ]
    path = unicode_tmp_path / "关键点.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    with pytest.raises(cr.CalibRefineError) as caught:
        cr.load_keypoints(path)
    problems = caught.value.problems
    expected = (
        "line 1: 6 of 6 point(s) malformed (point 0 has 5 value(s)); every point needs 6 numbers",
        "line 2: 2 of 2 point(s) malformed (point 0 has 3 value(s))",
        "line 3: 1 of 1 point(s) malformed (point 0 has 3 value(s))",
        "line 4: 1 of 2 point(s) malformed (point 1 has a value that is not a number)",
        "line 5: 1 of 2 point(s) malformed (point 1 is a float, not a list)",
        "line 6: a point holds a number too large for a float",
    )
    assert len(problems) == len(expected)
    for problem, start in zip(problems, expected, strict=True):
        assert problem.startswith(start), problem


def test_bad_inputs_and_settings_list_every_problem() -> None:
    settings = cr.RefineSettings(
        sigma_px=0.0, min_points=2, holdout_fraction=0.7, holdout_fallback_ratio=0.5
    )
    with pytest.raises(cr.CalibRefineError) as caught:
        cr.refine_calibration(pd.DataFrame({"frame_index": [0]}), [], pd.DataFrame(), settings)
    text = " ".join(caught.value.problems)
    for expected in (
        "sigma_px",
        "min_points",
        "holdout_fraction",
        "holdout_fallback_ratio must be a number >= 1",
        "calib_df misses",
        "shots_df misses",
        "keypoints must be",
    ):
        assert expected in text


def test_reproject_tracks_respects_unusable_frames() -> None:
    calib, kps, shots, true_cams = _scene(seed=8, shots=((20, -3.0, 3.0, TRUE_C),))
    calib.loc[calib["frame_index"] == 5, "pitch_usable"] = False
    tracks, _ = _players(true_cams)
    out = cr.reproject_tracks(tracks, calib)
    five = out[out["frame"] == 5]
    assert not five["pitch_usable"].any()
    assert five["pitch_x"].isna().all()
    assert not five["in_pitch"].any()
    assert out.loc[out["frame"] != 5, "pitch_x"].notna().all()


def test_group_with_a_heavy_held_out_tail_is_not_pitch_usable() -> None:
    calib, kps, shots, _ = _scene(seed=5)
    rng = np.random.default_rng(9)
    for frame, points in kps.items():  # a quarter of the keypoints are wrong matches
        bad = rng.random(len(points)) < 0.25
        points[bad, :2] += rng.choice([-1.0, 1.0], size=(int(bad.sum()), 2)) * 150.0
        kps[frame] = points
    result = cr.refine_calibration(calib, kps, shots)
    group = result.report["groups"][0]
    assert group["holdout"]["refined"]["p90_px"] > 25.0
    assert "rejected" in group
    assert result.calib["refined"].all(), "the frames are still refined"
    assert not result.calib["pitch_usable"].any()
    assert result.report["counts"]["rejected_group_frames"] == len(calib)
    clean = cr.refine_calibration(*_scene(seed=5)[:3])
    assert "rejected" not in clean.report["groups"][0]
    assert clean.calib["pitch_usable"].all()
