"""Tests for gbf.sngs_import: plugin-sngs export to tracks.parquet, calib.parquet, import.json.

Every export here is synthetic: a known camera (gbf.geometry.synthetic_broadcast_camera)
projects known pitch positions into detection boxes whose bottom centre is the foot point,
so the import must give those positions back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gbf.geometry import Camera, project, synthetic_broadcast_camera
from gbf.sngs_import import (
    CALIB_SCHEMA,
    TRACKS_SCHEMA,
    SngsImportError,
    camera_for_frame,
    cameras_by_frame,
    import_sngs,
    load_calib,
    load_tracks,
)
from gbf.util.hashing import sha256_bytes

RESTRICTIONS = ["no_share", "hl3"]


def _camera() -> Camera:
    return synthetic_broadcast_camera(1920, 1080)


def _box(camera: Camera, x: float, y: float, height_px: float = 100.0) -> list[float]:
    """Detection box whose bottom centre is the projection of pitch point (x, y)."""
    uv, _, valid = project(camera, [x, y, 0.0])
    if not bool(valid):
        raise ValueError("point behind the camera")
    u, v = float(uv[0]), float(uv[1])
    half_width = 0.2 * height_px
    return [u - half_width, v - height_px, u + half_width, v]


def _det(
    frame: int,
    track_id: int,
    box: list[float],
    *,
    role: str = "player",
    team: str | None = "left",
    jersey: str | None = None,
    jersey_conf: float | None = 0.0,
    conf: float = 0.9,
) -> dict[str, Any]:
    return {
        "frame": frame,
        "state_frame": frame + 1,
        "track_id": track_id,
        "box": box,
        "kf_box": box,
        "conf": conf,
        "role": role,
        "team": team,
        "team_cluster": 0.0 if team == "left" else 1.0,
        "jersey": jersey,
        "jersey_conf": jersey_conf,
    }


def _calib(
    frame: int,
    camera: Camera,
    *,
    status: str = "fresh",
    age: int | None = None,
    rep_err: float | None = 2.0,
    K: list[float] | None = None,
    R: list[float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame": frame,
        "state_frame": frame + 1,
        "calib_status": status,
        "calib_age": (0 if status == "fresh" else 1) if age is None else age,
        "n_keypoints": 12,
        "n_line_classes": 5,
    }
    if status != "none":
        row.update(
            {
                "K": camera.K.ravel().tolist() if K is None else K,
                "R": camera.R.ravel().tolist() if R is None else R,
                "C": camera.C.tolist(),
                "radial_distortion": [0.0] * 6,
                "width": camera.width,
                "height": camera.height,
                "rep_err_px": rep_err,
                "rep_err_n_keypoints": 0 if rep_err is None else 10,
            }
        )
    return row


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")


def _write_export(
    folder: Path,
    detections: list[dict[str, Any]],
    calib: list[dict[str, Any]],
    *,
    video_frames: int | None = None,
    receipt: dict[str, Any] | None = None,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    files = {"detections.jsonl": _jsonl(detections), "calib.jsonl": _jsonl(calib)}
    for name, data in files.items():
        (folder / name).write_bytes(data)
    record: dict[str, Any] = {
        "plugin": "plugin-sngs-export",
        "version": "0.1.0",
        "license": "GPL-3.0-only",
        "outputs": [
            {"name": name, "sha256": sha256_bytes(data), "bytes": len(data)}
            for name, data in files.items()
        ],
        "restrictions": RESTRICTIONS,
    }
    if video_frames is not None:
        record["video_frames"] = video_frames
    record.update(receipt or {})
    (folder / "plugin-receipt.json").write_bytes(json.dumps(record).encode("utf-8"))
    return folder


TRUTH = {  # track_id -> (x0, y0, vx, vy, team): metres, metres per frame
    1: (-20.0, 5.0, 0.2, 0.0, "left"),
    2: (10.0, -15.0, -0.1, 0.05, "right"),
    3: (40.0, 20.0, 0.0, -0.2, "right"),
}


def _truth(track_id: int, frame: int) -> tuple[float, float]:
    x0, y0, vx, vy, _ = TRUTH[track_id]
    return x0 + vx * frame, y0 + vy * frame


def _simple_export(folder: Path, n_frames: int = 5, **kwargs: Any) -> Path:
    camera = _camera()
    detections = [
        _det(frame, track, _box(camera, *_truth(track, frame)), team=TRUTH[track][4])
        for frame in range(n_frames)
        for track in TRUTH
    ]
    calib = [_calib(frame, camera) for frame in range(n_frames)]
    return _write_export(folder, detections, calib, video_frames=n_frames, **kwargs)


def test_import_recovers_pitch_positions(unicode_tmp_path: Path) -> None:
    source = _simple_export(unicode_tmp_path / "export")
    summary = import_sngs(source, unicode_tmp_path / "out")

    tracks = load_tracks(summary.tracks_path)
    assert len(tracks) == 15
    assert summary.n_track_rows == 15
    assert summary.n_frames == 5
    assert summary.n_pitch_usable_frames == 5
    assert tracks["pitch_usable"].all()
    assert tracks["in_pitch"].all()
    assert (tracks["calib_status"] == "fresh").all()
    keys = list(zip(tracks["frame"], tracks["track_id"], strict=True))
    expected = np.array([_truth(t, f) for f, t in keys])
    np.testing.assert_allclose(tracks[["pitch_x", "pitch_y"]].to_numpy(), expected, atol=1e-6)
    # Foot point = bottom centre of the box.
    np.testing.assert_allclose(tracks["foot_u"], 0.5 * (tracks["x1"] + tracks["x2"]))
    np.testing.assert_array_equal(tracks["foot_v"], tracks["y2"])
    assert keys == sorted(keys)  # sorted by (frame, track_id)
    assert tracks.attrs["restrictions"] == sorted(RESTRICTIONS)


def test_output_schemas_and_import_json(unicode_tmp_path: Path) -> None:
    source = _simple_export(unicode_tmp_path / "export")
    summary = import_sngs(source, unicode_tmp_path / "out")

    tracks_schema = pq.read_schema(summary.tracks_path)
    calib_schema = pq.read_schema(summary.calib_path)
    for schema, expected in ((tracks_schema, TRACKS_SCHEMA), (calib_schema, CALIB_SCHEMA)):
        assert schema.names == expected.names
        for field in expected:
            assert schema.field(field.name).type.equals(field.type), field.name
    assert calib_schema.field("K").type == pa.list_(pa.float64(), 9)
    assert calib_schema.field("C").type == pa.list_(pa.float64(), 3)
    assert calib_schema.field("calib_age").type == pa.int32()
    assert json.loads(tracks_schema.metadata[b"gbf.restrictions"]) == sorted(RESTRICTIONS)

    record = json.loads(summary.import_json_path.read_bytes())
    assert record["restrictions"] == sorted(RESTRICTIONS)
    assert record["counts"]["track_rows"] == 15
    assert record["counts"]["pitch_usable_frames"] == 5
    names = {item["name"]: item for item in record["inputs"]}
    data = (source / "detections.jsonl").read_bytes()
    assert names["detections.jsonl"]["sha256"] == sha256_bytes(data)
    outputs = {item["name"]: item for item in record["outputs"]}
    assert outputs["tracks.parquet"]["sha256"] == sha256_bytes(summary.tracks_path.read_bytes())
    assert b"\r\n" not in summary.import_json_path.read_bytes()
    text = summary.import_json_path.read_text(encoding="utf-8")
    assert str(unicode_tmp_path) not in text  # no local paths in the record

    calib = load_calib(summary.calib_path)
    assert list(calib["frame_index"]) == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(np.stack(calib["K"].to_list()), np.tile(_camera().K.ravel(), (5, 1)))


def test_outputs_are_deterministic(unicode_tmp_path: Path) -> None:
    source = _simple_export(unicode_tmp_path / "export")
    first = import_sngs(source, unicode_tmp_path / "a")
    second = import_sngs(source, unicode_tmp_path / "b")
    for name in ("tracks.parquet", "calib.parquet", "import.json"):
        assert (first.out_dir / name).read_bytes() == (second.out_dir / name).read_bytes()
    assert sorted(p.name for p in first.out_dir.iterdir()) == [
        "calib.parquet",
        "import.json",
        "tracks.parquet",
    ]


def test_missing_export_file_is_reported(unicode_tmp_path: Path) -> None:
    source = _simple_export(unicode_tmp_path / "export")
    (source / "calib.jsonl").unlink()
    with pytest.raises(SngsImportError) as caught:
        import_sngs(source, unicode_tmp_path / "out")
    text = "\n".join(caught.value.problems)
    assert "calib.jsonl: missing in sngs_dir" in text
    assert "sha256 mismatch" not in text


def test_held_and_poorly_fitted_frames_have_no_pitch_position(unicode_tmp_path: Path) -> None:
    camera = _camera()
    calib = [
        _calib(0, camera, rep_err=2.0),
        _calib(1, camera, status="held", age=1, rep_err=2.0),
        _calib(2, camera, rep_err=9.5),
        _calib(3, camera, rep_err=None),
        _calib(4, camera, rep_err=8.0),  # the limit itself is usable
    ]
    detections = [_det(frame, 1, _box(camera, 5.0, 5.0)) for frame in range(5)]
    source = _write_export(unicode_tmp_path / "export", detections, calib, video_frames=5)
    summary = import_sngs(source, unicode_tmp_path / "out")

    tracks = load_tracks(summary.tracks_path).set_index("frame")
    assert tracks["pitch_usable"].to_dict() == {0: True, 1: False, 2: False, 3: False, 4: True}
    assert tracks.loc[[1, 2, 3], ["pitch_x", "pitch_y"]].isna().all().all()
    assert not tracks.loc[[1, 2, 3], "in_pitch"].any()
    np.testing.assert_allclose(tracks.loc[[0, 4], "pitch_x"], 5.0, atol=1e-6)
    assert tracks.loc[1, "calib_status"] == "held"
    reasons = summary.counts["not_pitch_usable_by_reason"]
    assert reasons == {"fresh_rep_err_above_max": 1, "fresh_rep_err_null": 1, "held": 1}

    loose = import_sngs(source, unicode_tmp_path / "loose", rep_err_max_px=10.0)
    assert load_tracks(loose.tracks_path).set_index("frame").loc[2, "pitch_usable"]


def test_tampered_export_is_rejected_and_nothing_written(unicode_tmp_path: Path) -> None:
    source = _simple_export(unicode_tmp_path / "export")
    path = source / "detections.jsonl"
    data = bytearray(path.read_bytes())
    data[data.index(b'"conf": 0.9') + 10] = ord("8")  # 0.9 -> 0.8: same size, other bytes
    path.write_bytes(bytes(data))
    calib_path = source / "calib.jsonl"
    calib_path.write_bytes(calib_path.read_bytes() + b"\n")

    with pytest.raises(SngsImportError) as caught:
        import_sngs(source, unicode_tmp_path / "out")
    problems = caught.value.problems
    assert any("sha256 mismatch for detections.jsonl" in p for p in problems)
    assert any("sha256 mismatch for calib.jsonl" in p for p in problems)
    assert any("size mismatch for calib.jsonl" in p for p in problems)
    assert not any("size mismatch for detections.jsonl" in p for p in problems)
    assert not (unicode_tmp_path / "out").exists()


def test_every_row_problem_is_listed(unicode_tmp_path: Path) -> None:
    camera = _camera()
    good = _box(camera, 0.0, 0.0)
    scaled_r = (camera.R * 1.01).ravel().tolist()
    detections = [
        _det(0, 1, good),
        _det(0, 1, good),  # duplicate (frame, track_id)
        _det(1, 2, good, role="coach"),
        _det(1, 3, good, team="middle"),
        _det(9, 4, good),  # beyond video_frames
        _det(1, 5, [10.0, 20.0, 5.0, 40.0]),  # x2 < x1
        _det(1, 6, good, conf=1.5),
    ]
    calib = [
        _calib(0, camera),
        _calib(0, camera),  # repeated frame
        _calib(1, camera, R=scaled_r),
        _calib(2, camera, K=[1.0] * 8),
        _calib(3, camera, status="stale"),
        _calib(4, camera, status="held", age=0),
    ]
    source = _write_export(unicode_tmp_path / "export", detections, calib, video_frames=5)
    with pytest.raises(SngsImportError) as caught:
        import_sngs(source, unicode_tmp_path / "out")
    text = "\n".join(caught.value.problems)
    for expected in (
        "track 1 appears twice in frame 0",
        "role 'coach' is not one of",
        "team must be 'left', 'right' or null, got 'middle'",
        "frame 9 is outside the video",
        "must have x2 > x1 and y2 > y1",
        "conf must be a number in [0, 1]",
        "R is not orthonormal",
        "K must be a list of 9 numbers",
        "frame 0 repeated",
        "calib_status 'stale' is not one of",
        "calib_age must be >= 1 for a held calibration",
    ):
        assert expected in text, expected
    assert len(caught.value.problems) >= 11


def test_receipt_problems(unicode_tmp_path: Path) -> None:
    unknown = _simple_export(unicode_tmp_path / "a", receipt={"restrictions": ["top_secret"]})
    with pytest.raises(SngsImportError, match="restrictions"):
        import_sngs(unknown, unicode_tmp_path / "out_a")

    missing = _simple_export(unicode_tmp_path / "b")
    record = json.loads((missing / "plugin-receipt.json").read_bytes())
    del record["restrictions"]
    record["outputs"] = [o for o in record["outputs"] if o["name"] != "calib.jsonl"]
    (missing / "plugin-receipt.json").write_bytes(json.dumps(record).encode("utf-8"))
    with pytest.raises(SngsImportError) as caught:
        import_sngs(missing, unicode_tmp_path / "out_b")
    text = "\n".join(caught.value.problems)
    assert "'restrictions' is missing" in text
    assert "calib.jsonl is not listed in 'outputs'" in text

    escape = _simple_export(unicode_tmp_path / "c")
    record = json.loads((escape / "plugin-receipt.json").read_bytes())
    record["outputs"].append({"name": "x", "relpath": "../x.jsonl", "sha256": "0" * 64})
    record["detections"] = 99  # a stated count that disagrees with the file
    (escape / "plugin-receipt.json").write_bytes(json.dumps(record).encode("utf-8"))
    with pytest.raises(SngsImportError) as caught:
        import_sngs(escape, unicode_tmp_path / "out_c")
    text = "\n".join(caught.value.problems)
    assert "must be a relative POSIX path inside sngs_dir" in text
    assert "detections = 99 but detections.jsonl has 15 row(s)" in text


def test_receipt_relpaths_locate_the_files(unicode_tmp_path: Path) -> None:
    flat = _simple_export(unicode_tmp_path / "flat")
    job = unicode_tmp_path / "job"
    (job / "out").mkdir(parents=True)
    record = json.loads((flat / "plugin-receipt.json").read_bytes())
    for output in record["outputs"]:
        (job / "out" / output["name"]).write_bytes((flat / output["name"]).read_bytes())
        output["relpath"] = "out/" + output["name"]
    (job / "plugin-receipt.json").write_bytes(json.dumps(record).encode("utf-8"))
    summary = import_sngs(job, unicode_tmp_path / "result")
    assert summary.n_track_rows == 15


def test_ball_rows_dropped_and_nan_team_stored_as_null(unicode_tmp_path: Path) -> None:
    camera = _camera()
    detections = [
        _det(0, 1, _box(camera, 0.0, 0.0)),
        _det(0, 2, _box(camera, 3.0, 0.0, height_px=8.0), role="ball", team="nan"),
        _det(0, 3, _box(camera, 6.0, 0.0), role="referee", team="nan", jersey_conf=None),
        _det(0, 4, _box(camera, 9.0, 0.0), role="goalkeeper", team="right", jersey="1"),
    ]
    source = _write_export(unicode_tmp_path / "e", detections, [_calib(0, camera)])
    summary = import_sngs(source, unicode_tmp_path / "out")
    tracks = load_tracks(summary.tracks_path).set_index("track_id")
    assert sorted(tracks.index) == [1, 3, 4]
    assert tracks["team"].isna().to_dict() == {1: False, 3: True, 4: False}
    assert np.isnan(tracks.loc[3, "jersey_conf"])
    assert tracks.loc[4, "jersey"] == "1"
    assert summary.counts["dropped_rows_by_role"] == {"ball": 1}
    assert summary.counts["team_nan_strings_stored_as_null"] == 1
    assert summary.n_detection_rows == 4


def test_in_pitch_uses_a_three_metre_margin(unicode_tmp_path: Path) -> None:
    camera = synthetic_broadcast_camera(1920, 1080, target=(50.0, 0.0, 0.0))
    points = {1: (54.0, 0.0), 2: (56.5, 0.0), 3: (50.0, 36.5), 4: (50.0, 37.5)}
    detections = [_det(0, t, _box(camera, x, y)) for t, (x, y) in points.items()]
    source = _write_export(unicode_tmp_path / "e", detections, [_calib(0, camera)])
    tracks = load_tracks(import_sngs(source, unicode_tmp_path / "out").tracks_path)
    tracks = tracks.set_index("track_id")
    assert tracks["in_pitch"].to_dict() == {1: True, 2: False, 3: True, 4: False}
    np.testing.assert_allclose(tracks.loc[2, "pitch_x"], 56.5, atol=1e-6)


def test_feet_above_the_horizon_have_no_ground_point(unicode_tmp_path: Path) -> None:
    camera = _camera()
    # 1140 px above the image centre: about 5 degrees above the horizon of this camera.
    box = [900.0, -700.0, 940.0, -600.0]
    source = _write_export(unicode_tmp_path / "e", [_det(0, 1, box)], [_calib(0, camera)])
    summary = import_sngs(source, unicode_tmp_path / "out")
    row = load_tracks(summary.tracks_path).iloc[0]
    assert row["pitch_usable"]
    assert np.isnan(row["pitch_x"])
    assert not row["in_pitch"]
    assert summary.counts["track_rows_ray_missed_ground"] == 1


def test_invalid_camera_is_kept_but_never_usable(unicode_tmp_path: Path) -> None:
    camera = _camera()
    bad_k = camera.K.ravel().tolist()
    bad_k[0] = bad_k[4] = -0.03  # a diverged fit, as seen on real exports
    calib = [_calib(0, camera), _calib(1, camera, K=bad_k)]
    detections = [_det(frame, 1, _box(camera, 0.0, 0.0)) for frame in range(2)]
    source = _write_export(unicode_tmp_path / "e", detections, calib)
    summary = import_sngs(source, unicode_tmp_path / "out")
    calib_df = load_calib(summary.calib_path)
    assert calib_df["pitch_usable"].tolist() == [True, False]
    assert calib_df["K"].iloc[1][0] == -0.03
    assert [item["frame"] for item in summary.counts["invalid_camera_frames"]] == [1]
    assert camera_for_frame(calib_df, 1, usable_only=False) is None


def test_frames_missing_from_the_export_get_status_none(unicode_tmp_path: Path) -> None:
    camera = _camera()
    detections = [_det(frame, 1, _box(camera, 0.0, 0.0)) for frame in (0, 3)]
    calib = [_calib(frame, camera) for frame in range(3)]
    source = _write_export(unicode_tmp_path / "e", detections, calib, video_frames=5)
    summary = import_sngs(source, unicode_tmp_path / "out")
    calib_df = load_calib(summary.calib_path)
    assert calib_df["calib_status"].tolist() == ["fresh", "fresh", "fresh", "none", "none"]
    assert calib_df["K"].iloc[3] is None
    assert calib_df["width"].tolist() == [1920] * 5
    assert summary.counts["calib_rows_added_as_none"] == 2
    tracks = load_tracks(summary.tracks_path).set_index("frame")
    assert tracks.loc[3, "calib_status"] == "none"
    assert not tracks.loc[3, "pitch_usable"]


def test_camera_for_frame(unicode_tmp_path: Path) -> None:
    camera = _camera()
    calib = [_calib(0, camera), _calib(1, camera, status="held", age=1)]
    source = _write_export(
        unicode_tmp_path / "e", [_det(0, 1, _box(camera, 0.0, 0.0))], calib, video_frames=3
    )
    calib_df = load_calib(import_sngs(source, unicode_tmp_path / "out").calib_path)

    usable = camera_for_frame(calib_df, 0)
    assert isinstance(usable, Camera)
    np.testing.assert_allclose(usable.K, camera.K)
    np.testing.assert_allclose(usable.R, camera.R)
    np.testing.assert_allclose(usable.C, camera.C)
    assert usable.size == (1920, 1080)
    assert camera_for_frame(calib_df, 1) is None  # held: not pitch-usable
    held = camera_for_frame(calib_df, 1, usable_only=False)
    assert isinstance(held, Camera)
    assert camera_for_frame(calib_df, 2, usable_only=False) is None  # status none
    assert camera_for_frame(calib_df, 7) is None  # no row
    assert sorted(cameras_by_frame(calib_df)) == [0]
    assert sorted(cameras_by_frame(calib_df, usable_only=False)) == [0, 1]
    with pytest.raises(SngsImportError):
        camera_for_frame(calib_df.drop(columns="pitch_usable"), 0)


def test_refuses_to_overwrite_and_to_write_into_the_export(unicode_tmp_path: Path) -> None:
    source = _simple_export(unicode_tmp_path / "export")
    out = unicode_tmp_path / "out"
    import_sngs(source, out)
    with pytest.raises(FileExistsError):
        import_sngs(source, out)
    import_sngs(source, out, overwrite=True)
    with pytest.raises(ValueError, match="out_dir must differ"):
        import_sngs(source, source)
    with pytest.raises(ValueError, match="rep_err_max_px"):
        import_sngs(source, unicode_tmp_path / "other", rep_err_max_px=0.0)


def test_loaders_reject_foreign_files(unicode_tmp_path: Path) -> None:
    path = unicode_tmp_path / "tracks.parquet"
    pq.write_table(pa.table({"frame": pa.array([0], pa.int64())}), path)
    with pytest.raises(SngsImportError) as caught:
        load_tracks(path)
    assert any("missing column track_id" in p for p in caught.value.problems)
    with pytest.raises(SngsImportError, match="cannot read as parquet"):
        load_calib(unicode_tmp_path / "absent.parquet")
