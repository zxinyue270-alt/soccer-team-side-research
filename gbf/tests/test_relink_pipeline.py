"""The relink step inside gbf.pipeline: off changes nothing, on renames tracks, and the cases
in which a mode that is on does not run.

The clip is the synthetic one of tests/test_pipeline.py (100 frames of 960x540, a wide shot
of six tracked people until the cut at frame 60, then a close-up). Two variants of its plugin
export are written here: with an ``embeddings.npz`` (one made-up appearance vector per
detection line: the person's own direction plus the team's and a little noise), and with a
return: track 2 loses its player for ten frames and the tracker gives him the new id 22.
Nothing here comes from a real clip.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gbf import pipeline as pipeline_module
from gbf import relink
from gbf.pipeline import PipelineConfig, PipelineError, main, read_restrictions, run_pipeline
from gbf.sngs_import import SngsImportError, load_embeddings
from gbf.util.hashing import sha256_bytes
from gbf.util.io import read_json
from test_pipeline import (
    N_FRAMES,
    PLAYERS,
    PLUGIN_TAGS,
    _calib,
    _camera,
    _config,
    _detections,
    _jsonl,
    make_inputs,
)
from test_relink_api import FPS, Scene, make_scene

THREE = ("import", "shots", "refine")
DIM = 16
RETURNING, RETURNED = 2, 22
GAP = (25, 34)  # frames without a box of the returning player
TABLES = ("import/tracks.parquet", "import/calib.parquet")
REFINED = ("refine/tracks.parquet", "refine/calib.parquet", "refine/refine.json")


def _return_detections() -> list[dict[str, Any]]:
    """The clip's detections with the return: no box of track 2's player in the gap, and the
    id 22 for him afterwards."""
    rows = []
    for row in _detections(_camera()):
        if row["track_id"] == RETURNING and GAP[0] <= row["frame"] <= GAP[1]:
            continue
        if row["track_id"] == RETURNING and row["frame"] > GAP[1]:
            row = {**row, "track_id": RETURNED}
        rows.append(row)
    return rows


def _embeddings(detections: list[dict[str, Any]], seed: int = 0) -> np.ndarray:
    """One float32 row per detection line, not normalised: the person's direction (track 22
    is track 2's player) plus the team's, with a little noise."""
    rng = np.random.default_rng(seed)
    teams = {team: rng.normal(size=DIM) for team in ("left", "right")}
    people = {}
    for track in sorted(
        {RETURNED if t == RETURNING else t for t in (r["track_id"] for r in detections)}
    ):
        team = PLAYERS[RETURNING if track == RETURNED else track][0] if track != 31 else "left"
        own = rng.normal(size=DIM)
        vector = teams[team] / np.linalg.norm(teams[team]) + 0.8 * own / np.linalg.norm(own)
        people[track] = vector / np.linalg.norm(vector)
    rows = []
    for row in detections:
        person = RETURNED if row["track_id"] == RETURNING else row["track_id"]
        rows.append(2.5 * (people[person] + 0.05 * rng.normal(size=DIM)))
    return np.asarray(rows, dtype=np.float32)


def _npz(detections: list[dict[str, Any]], emb: np.ndarray, **changes: np.ndarray) -> bytes:
    arrays = {
        "frame": np.array([row["frame"] for row in detections], dtype=np.int32),
        "track": np.array([row["track_id"] for row in detections], dtype=np.int64),
        "box": np.array([row["box"] for row in detections], dtype=np.float64),
        "emb": emb,
    }
    arrays.update(changes)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def write_export(
    folder: Path,
    detections: list[dict[str, Any]],
    npz: bytes | None = None,
    *,
    listed: bool = True,
    written: bool = True,
    tags: list[str] | None = None,
    stated: dict[str, Any] | None = None,
    calib: list[dict[str, Any]] | None = None,
) -> Path:
    """A plugin export of the synthetic clip: the two row files and the receipt, and an
    ``embeddings.npz`` when ``npz`` is given (``listed`` in the receipt, ``written`` to disk).
    ``tags`` are the receipt's restrictions (default: the clip's), ``stated`` its
    ``embeddings`` object and ``calib`` other calibration rows than the clip's."""
    folder.mkdir(parents=True)
    files = {
        "detections.jsonl": _jsonl(detections),
        "calib.jsonl": _jsonl(_calib(_camera()) if calib is None else calib),
    }
    if npz is not None:
        files["embeddings.npz"] = npz
    for name, data in files.items():
        if name != "embeddings.npz" or written:
            (folder / name).write_bytes(data)
    receipt = {
        "plugin": "plugin-sngs-export",
        "version": "0.1.0",
        "license": "GPL-3.0-only",
        "outputs": [
            {"name": name, "sha256": sha256_bytes(data), "bytes": len(data)}
            for name, data in files.items()
            if name != "embeddings.npz" or listed
        ],
        "restrictions": PLUGIN_TAGS if tags is None else tags,
        "video_frames": N_FRAMES,
    }
    if stated is not None:
        receipt["embeddings"] = stated
    (folder / "plugin-receipt.json").write_bytes(json.dumps(receipt).encode("utf-8"))
    return folder


def run_three(inputs: dict[str, Path], out: Path, base: Path, **changes: Any) -> dict[str, str]:
    """Run import, shots and refine (the run then stops at the first stage that was not asked
    for and has no outputs) and return each stage's status."""
    config = _config(inputs, out, base, stages=THREE, **changes)
    with pytest.raises(PipelineError, match="not in --stages") as caught:
        run_pipeline(config)
    assert caught.value.where == "kinematics"
    stages = read_json(out / "run.json")["result"]["stages"]
    return {stage["name"]: stage["status"] for stage in stages}


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The synthetic clip with its plain export and the variants of it."""
    base = tmp_path_factory.mktemp("relink") / "非ASCII 路径-ü"
    base.mkdir()
    inputs = make_inputs(base / "inputs")
    plain = _detections(_camera())
    back = _return_detections()
    exports = {
        "plain": inputs["sngs"],
        "embeddings": write_export(base / "sngs emb", plain, _npz(plain, _embeddings(plain))),
        "return": write_export(base / "sngs return", back, _npz(back, _embeddings(back))),
        "return plain": write_export(base / "sngs return plain", back),
    }
    return {"base": base, "inputs": inputs, "exports": exports, "return": back}


def _with(clip: dict[str, Any], export: str) -> dict[str, Path]:
    return {**clip["inputs"], "sngs": clip["exports"][export]}


@pytest.fixture(scope="module")
def off_run(clip: dict[str, Any]) -> Path:
    """Import, shots and refine of the export with the return, relink off."""
    out = clip["base"] / "run off"
    assert run_three(_with(clip, "return"), out, clip["base"]) == dict.fromkeys(THREE, "ran")
    return out


@pytest.fixture(scope="module")
def strict_run(clip: dict[str, Any]) -> Path:
    """The same with ``relink="strict"``."""
    out = clip["base"] / "run strict"
    statuses = run_three(_with(clip, "return"), out, clip["base"], relink="strict")
    assert statuses == dict.fromkeys(THREE, "ran")
    return out


# --------------------------------------------------------------------------- off


def test_off_is_the_default_and_embeddings_in_the_export_change_no_artifact(
    clip: dict[str, Any],
) -> None:
    base = clip["base"]
    assert PipelineConfig.__dataclass_fields__["relink"].default == "off"
    plain, carrying = base / "off plain", base / "off carrying"
    assert run_three(clip["inputs"], plain, base) == dict.fromkeys(THREE, "ran")
    assert run_three(_with(clip, "embeddings"), carrying, base) == dict.fromkeys(THREE, "ran")
    for out in (plain, carrying):
        assert not (out / "import" / "embeddings.parquet").exists()
        assert "piece_id" not in pq.read_schema(out / "refine" / "tracks.parquet").names
        assert "relink" not in read_json(out / "refine" / "refine.json")
        receipt = read_json(out / "receipts" / "refine.json")["result"]
        assert "relink" not in receipt["facts"]
        assert receipt["parameters"]["relink"] == {"mode": "off"}
        assert not any("relink" in note for note in receipt["notes_zh"])
        limitations = read_json(out / "receipts" / "refine.json")["limitations"]
        assert not any("relink" in line for line in limitations)
        assert "relink" not in read_json(out / "run.json")["result"]["restriction_sources"]
        imported = read_json(out / "receipts" / "import.json")
        assert imported["result"]["parameters"]["embeddings"] is False
        assert [entry["name"] for entry in imported["inputs"]] == [
            "sngs/detections.jsonl",
            "sngs/calib.jsonl",
            "sngs/plugin-receipt.json",
        ]
        assert sorted(p.name for p in (out / "import").iterdir()) == [
            "calib.parquet",
            "import.json",
            "tracks.parquet",
        ]
    # the embeddings were not read: the export's tables and every refine file keep their bytes
    for name in (*TABLES, *REFINED):
        assert (plain / name).read_bytes() == (carrying / name).read_bytes(), name
    # import.json records the receipt, which lists one more file: nothing else differs
    records = [read_json(out / "import" / "import.json") for out in (plain, carrying)]
    for record in records:
        assert [entry["name"] for entry in record["inputs"]][2] == "plugin-receipt.json"
        del record["inputs"][2]
    assert records[0] == records[1]


# --------------------------------------------------------------------------- on


def test_strict_renames_the_returned_track_and_keeps_the_old_ids_in_piece_id(
    off_run: Path, strict_run: Path
) -> None:
    before = pd.read_parquet(off_run / "refine" / "tracks.parquet")
    after = pd.read_parquet(strict_run / "refine" / "tracks.parquet")
    assert list(after.columns) == [*before.columns, "piece_id"]
    assert after["piece_id"].tolist() == before["track_id"].tolist()
    for name in before.columns:
        if name != "track_id":
            pd.testing.assert_series_equal(after[name], before[name])
    returned = (before["track_id"] == RETURNED).to_numpy()
    assert returned.sum() == 59 - GAP[1]
    assert (after.loc[returned, "track_id"] == RETURNING).all()
    assert after.loc[~returned, "track_id"].tolist() == before.loc[~returned, "track_id"].tolist()
    assert not after.duplicated(["frame", "track_id"]).any()
    assert read_restrictions(strict_run / "refine" / "tracks.parquet") == read_restrictions(
        off_run / "refine" / "tracks.parquet"
    )
    # the calibration is not touched
    assert (strict_run / REFINED[1]).read_bytes() == (off_run / REFINED[1]).read_bytes()


def test_refine_json_and_the_run_facts_report_the_relink(off_run: Path, strict_run: Path) -> None:
    plain = read_json(off_run / "refine" / "refine.json")
    report = read_json(strict_run / "refine" / "refine.json")
    section = report.pop("relink")
    assert report == plain  # every other report describes the ids from before
    assert (section["mode"], section["applied"], section["reason"]) == ("strict", True, None)
    assert section["setting"] == {"margin": 0.6, "leaves": False}
    assert section["params"] == relink.DEFAULT_PARAMS.as_dict()
    assert (section["fps"], section["fitted_fps"]) == (25.0, 25.0)
    assert section["frame_size"] == [960, 540]
    assert section["px_scale"] == 0.75
    assert section["embeddings"] == {"model": None, "dim": DIM}  # the receipt names no model
    assert (section["cuts"], section["links"]) == ([], [[RETURNING, RETURNED]])
    assert section["identities"] == [{"track_id": RETURNING, "pieces": [RETURNING, RETURNED]}]
    counts = section["counts"]
    assert (counts["shots"], counts["shots_relinked"]) == (2, 1)  # the close-up has one track
    assert (counts["pieces"], counts["sub_pieces"], counts["identities"]) == (8, 8, 7)
    assert (counts["core_links"], counts["leaf_links"], counts["rows_renamed"]) == (1, 0, 25)
    assert counts["rows_without_embedding"] == 0
    assert counts["links_refused_same_frame"] == 0

    receipt = read_json(strict_run / "receipts" / "refine.json")
    assert receipt["result"]["facts"]["relink"] == {
        "mode": "strict",
        "applied": True,
        "reason": None,
        "counts": counts,
    }
    parameters = receipt["result"]["parameters"]["relink"]
    assert parameters == {
        "mode": "strict",
        "setting": {"margin": 0.6, "leaves": False},
        "params": relink.DEFAULT_PARAMS.as_dict(),
    }
    assert "import/embeddings.parquet" in [entry["name"] for entry in receipt["inputs"]]
    # the limitation of the step is the receipt's only while a mode is on
    assert sum("relink parameters" in line for line in receipt["limitations"]) == 1
    assert (
        receipt["limitations"][0]
        == read_json(off_run / "receipts" / "refine.json")["limitations"][0]
    )
    notes = [note for note in receipt["result"]["notes_zh"] if "--relink strict" in note]
    assert len(notes) == 1
    for part in ("piece_id", "1 对轨迹片段", "0 条轨迹", "13 段 SN-GSR", "没有在其他素材上验证"):
        assert part in notes[0]
    assert "帧/秒不同" not in notes[0]  # the clip runs at the fitted 25 fps
    assert "没有外观特征" not in notes[0]  # every row has an embedding
    # the run reads the embeddings: it says itself that it is hl3 (the export says so too)
    run = read_json(strict_run / "run.json")
    assert run["result"]["restriction_sources"]["relink"] == ["hl3"]
    assert run["result"]["restriction_sources"]["plugin"] == PLUGIN_TAGS
    assert notes[0] in read_json(strict_run / "run.json")["result"]["notes_zh"]


def test_the_import_writes_the_embeddings_in_the_order_of_the_track_rows(
    clip: dict[str, Any], strict_run: Path
) -> None:
    path = strict_run / "import" / "embeddings.parquet"
    frame, track_id, emb = load_embeddings(path)
    tracks = pd.read_parquet(strict_run / "import" / "tracks.parquet")
    assert frame.tolist() == tracks["frame"].tolist()
    assert track_id.tolist() == tracks["track_id"].tolist()
    stored = _embeddings(clip["return"])
    line = {(row["frame"], row["track_id"]): i for i, row in enumerate(clip["return"])}
    rows = [line[key] for key in zip(frame.tolist(), track_id.tolist(), strict=True)]
    np.testing.assert_array_equal(emb, stored[rows])
    assert emb.dtype == np.float32
    assert emb.shape == (len(tracks), DIM)
    # the file carries the run's tags and is an output of the import stage
    assert read_restrictions(path) == read_restrictions(strict_run / "import" / "tracks.parquet")
    receipt = read_json(strict_run / "receipts" / "import.json")
    assert "import/embeddings.parquet" in [entry["relpath"] for entry in receipt["outputs"]]
    assert "sngs/embeddings.npz" in [entry["name"] for entry in receipt["inputs"]]
    assert receipt["result"]["parameters"]["embeddings"] is True
    assert receipt["result"]["facts"]["embedding_rows"] == len(tracks)
    record = read_json(strict_run / "import" / "import.json")
    assert record["counts"]["embedding_rows"] == len(tracks)
    assert record["counts"]["embedding_dim"] == DIM
    assert record["counts"]["track_rows_without_embedding"] == 0


def test_a_second_run_is_cached_and_switching_the_mode_off_restores_the_plain_tables(
    clip: dict[str, Any], off_run: Path, strict_run: Path
) -> None:
    base, inputs = clip["base"], _with(clip, "return")
    assert run_three(inputs, strict_run, base, relink="strict") == dict.fromkeys(THREE, "skipped")
    # another mode is another parameter of the import and of the refine stage; the tables
    # the shots stage reads keep their bytes, so it is not run again
    out = base / "run switched"
    assert run_three(inputs, out, base, relink="strict") == dict.fromkeys(THREE, "ran")
    assert (out / "import" / "embeddings.parquet").is_file()
    statuses = run_three(inputs, out, base)
    assert statuses == {"import": "ran", "shots": "skipped", "refine": "ran"}
    assert not (out / "import" / "embeddings.parquet").exists()  # no stale embeddings
    for name in (*TABLES, *REFINED):
        assert (out / name).read_bytes() == (off_run / name).read_bytes(), name
    assert read_json(out / "import" / "import.json") == read_json(
        off_run / "import" / "import.json"
    )


def test_the_later_stages_run_on_the_renamed_table(clip: dict[str, Any]) -> None:
    base = clip["base"]
    out = base / "run whole"
    result = run_pipeline(_config(_with(clip, "return"), out, base, relink="strict"))
    assert {stage.status for stage in result.stages} == {"ran"}
    tracks = pd.read_parquet(out / "refine" / "tracks.parquet")
    assert RETURNED not in set(tracks["track_id"])
    assert RETURNED in set(tracks["piece_id"])
    kinematics = pd.read_parquet(out / "kinematics.parquet")
    assert RETURNED not in set(kinematics["track_id"])
    after_gap = kinematics[(kinematics["track_id"] == RETURNING) & (kinematics["frame"] > GAP[1])]
    assert len(after_gap) > 0  # the joined track goes on after the gap
    assert result.highlights["highlights"]
    assert (out / "report.html").is_file()
    run = read_json(out / "run.json")
    assert run["result"]["parameters"]["relink"] == "strict"
    assert run["result"]["facts"]["refine"]["relink"]["counts"]["core_links"] == 1
    assert "sngs/embeddings.npz" in [entry["name"] for entry in run["inputs"]]
    # the team check judged the two pieces; the measure and render stages ask per identity
    verdicts = pipeline_module._team_verdicts(SimpleNamespace(path=lambda name: out / name))
    pieces = {
        (row["track_id"], row["team"]): row["verdict"]
        for row in read_json(out / "refine" / "refine.json")["teams"]["tracks"]
    }
    assert (RETURNED, "left") in pieces
    assert (RETURNED, "left") not in verdicts
    of_identity = [pieces[(piece, "left")] for piece in (RETURNING, RETURNED)]
    weakest = next((verdict for verdict in of_identity if verdict != "consistent"), "consistent")
    assert verdicts[(RETURNING, "left")] == weakest
    # a cached second run restores the note and the facts from the receipts
    again = run_pipeline(_config(_with(clip, "return"), out, base, relink="strict"))
    assert {stage.status for stage in again.stages} == {"skipped"}
    notes = read_json(out / "run.json")["result"]["notes_zh"]
    assert sum("--relink strict" in note for note in notes) == 1


# --------------------------------------------------------------------------- tags, frame size


def test_a_run_that_reads_the_embeddings_carries_hl3_whatever_the_receipt_says(
    clip: dict[str, Any],
) -> None:
    base, back = clip["base"], clip["return"]
    stated = {
        "rows": len(back),
        "dim": DIM,
        "dtype": "float32",
        "normalised": False,
        "model": "a made-up model",
    }
    # the export lists embeddings, but its receipt carries no tag at all
    npz = _npz(back, _embeddings(back))
    untagged = write_export(base / "sngs untagged", back, npz, tags=[], stated=stated)
    inputs = {**clip["inputs"], "sngs": untagged}
    out = base / "run untagged"
    assert run_three(inputs, out, base, relink="strict") == dict.fromkeys(THREE, "ran")
    run = read_json(out / "run.json")
    assert run["result"]["restriction_sources"]["plugin"] == []
    assert run["result"]["restriction_sources"]["relink"] == ["hl3"]
    assert pipeline_module.RELINK_RESTRICTIONS == ("hl3",)
    assert run["restrictions"] == ["hl3", "no_share"]  # no_share: the ball file has no receipt
    for name in (*TABLES, "import/embeddings.parquet", REFINED[0], REFINED[1]):
        assert read_restrictions(out / name) == ["hl3", "no_share"], name
    report = read_json(out / "refine" / "refine.json")
    assert report["restrictions"] == ["hl3", "no_share"]
    assert report["relink"]["applied"] is True
    assert report["relink"]["links"] == [[RETURNING, RETURNED]]
    # the model and the dimension the receipt states, through the import
    named = {"model": "a made-up model", "dim": DIM}
    assert report["relink"]["embeddings"] == named
    record = read_json(out / "import" / "import.json")
    assert record["embeddings"] == named
    assert (record["restrictions"], record["plugin_restrictions"]) == (["hl3", "no_share"], [])
    for stage in ("import", "refine"):
        assert read_json(out / "receipts" / f"{stage}.json")["restrictions"] == ["hl3", "no_share"]

    # off: the embeddings are not read and nothing of the run is hl3
    off = base / "run untagged off"
    assert run_three(inputs, off, base) == dict.fromkeys(THREE, "ran")
    run = read_json(off / "run.json")
    assert "relink" not in run["result"]["restriction_sources"]
    assert run["restrictions"] == ["no_share"]
    assert read_restrictions(off / REFINED[0]) == ["no_share"]
    # on, but the run must not link identities: the embeddings are not read either
    out = base / "run untagged no identity"
    run_three(inputs, out, base, relink="strict", restrictions=("no_identity",))
    run = read_json(out / "run.json")
    assert "relink" not in run["result"]["restriction_sources"]
    assert run["restrictions"] == ["no_identity", "no_share"]
    assert read_json(out / "refine" / "refine.json")["relink"]["reason"] == "no_identity"
    # on, but the export has no embeddings: there is nothing of that model in the run
    plain = write_export(base / "sngs untagged plain", back, tags=[])
    out = base / "run untagged plain"
    run_three({**clip["inputs"], "sngs": plain}, out, base, relink="strict")
    run = read_json(out / "run.json")
    assert "relink" not in run["result"]["restriction_sources"]
    assert run["restrictions"] == ["no_share"]
    assert read_json(out / "refine" / "refine.json")["relink"]["reason"] == "no_embeddings"


def test_the_frame_size_is_the_calibration_tables_not_the_video_files(
    clip: dict[str, Any], strict_run: Path, unicode_tmp_path: Path
) -> None:
    # the synthetic video is 960x540. This export says that the plugin worked on frames of
    # 1280x720: its boxes and cameras are in that grid, so the pixel constants are not scaled
    base, back = clip["base"], clip["return"]
    rows = [
        {**row, "width": 1280, "height": 720} if "width" in row else row
        for row in _calib(_camera())
    ]
    export = write_export(base / "sngs 720", back, _npz(back, _embeddings(back)), calib=rows)
    out = base / "run 720"
    statuses = run_three({**clip["inputs"], "sngs": export}, out, base, relink="strict")
    assert statuses == dict.fromkeys(THREE, "ran")
    video = read_json(out / "run.json")["result"]["video"]
    assert (video["width"], video["height"]) == (960, 540)
    section = read_json(out / "refine" / "refine.json")["relink"]
    assert section["applied"] is True
    assert (section["frame_size"], section["px_scale"]) == ([1280, 720], 1.0)
    assert read_json(out / "import" / "import.json")["frame_size"] == [1280, 720]
    same_size = read_json(strict_run / "refine" / "refine.json")["relink"]
    assert (same_size["frame_size"], same_size["px_scale"]) == ([960, 540], 0.75)

    # the step alone: the size of the video is not asked for at all
    scene = make_scene(scale=1.5)
    assert scene.size == (1920, 1080)
    calib = _sized(scene.calib, scene.size)
    for number, facts in enumerate((SimpleNamespace(width=1280, height=720), None)):
        run = _fake_run(unicode_tmp_path / str(number), scene, facts=facts)
        _tracks, section = pipeline_module._relink(run, scene.tracks, calib, scene.shots)
        assert (section["frame_size"], section["px_scale"]) == ([1920, 1080], 1.5)
        assert section["links"] == [[5, 9]]
    size = pipeline_module._calib_frame_size
    assert size(calib) == (1920, 1080)
    assert size(calib.iloc[:0]) is None
    assert size(scene.calib) is None  # no width and height columns
    two_sizes = pd.concat([calib, _sized(scene.calib.iloc[:1], (1280, 720))], ignore_index=True)
    assert size(two_sizes) is None
    assert size(_sized(scene.calib, (0, 720))) is None


# --------------------------------------------------------------------------- skipped


def test_a_mode_that_is_on_is_skipped_without_embeddings_and_says_so(
    clip: dict[str, Any], off_run: Path
) -> None:
    base = clip["base"]
    out = base / "run no embeddings"
    statuses = run_three(_with(clip, "return plain"), out, base, relink="strict")
    assert statuses == dict.fromkeys(THREE, "ran")
    assert not (out / "import" / "embeddings.parquet").exists()
    report = read_json(out / "refine" / "refine.json")
    assert report["relink"] == {"mode": "strict", "applied": False, "reason": "no_embeddings"}
    tracks = pd.read_parquet(out / "refine" / "tracks.parquet")
    plain = pd.read_parquet(off_run / "refine" / "tracks.parquet")
    assert list(tracks.columns) == [*plain.columns, "piece_id"]  # the columns follow the mode
    assert tracks["piece_id"].tolist() == tracks["track_id"].tolist()
    pd.testing.assert_frame_equal(tracks.drop(columns="piece_id"), plain)
    receipt = read_json(out / "receipts" / "refine.json")["result"]
    assert receipt["facts"]["relink"] == {
        "mode": "strict",
        "applied": False,
        "reason": "no_embeddings",
        "counts": None,
    }
    notes = [note for note in receipt["notes_zh"] if "--relink strict" in note]
    assert len(notes) == 1
    assert "没有运行" in notes[0]
    assert "embeddings.npz" in notes[0]


def test_embeddings_without_a_single_vector_skip_the_step_and_say_so(
    clip: dict[str, Any], off_run: Path
) -> None:
    base, back = clip["base"], clip["return"]
    nothing = np.full((len(back), DIM), np.nan, dtype=np.float32)
    nothing[::2] = 0.0  # a vector of length zero is no embedding either
    export = write_export(base / "sngs empty emb", back, _npz(back, nothing))
    out = base / "run empty emb"
    statuses = run_three({**clip["inputs"], "sngs": export}, out, base, relink="strict")
    assert statuses == dict.fromkeys(THREE, "ran")
    assert (out / "import" / "embeddings.parquet").is_file()  # the file was there and was read
    tracks = pd.read_parquet(out / "refine" / "tracks.parquet")
    assert read_json(out / "refine" / "refine.json")["relink"] == {
        "mode": "strict",
        "applied": False,
        "reason": "no_embeddings",
        "rows_without_embedding": len(tracks),
    }
    assert tracks["piece_id"].tolist() == tracks["track_id"].tolist()
    plain = pd.read_parquet(off_run / "refine" / "tracks.parquet")
    pd.testing.assert_frame_equal(tracks.drop(columns="piece_id"), plain)
    receipt = read_json(out / "receipts" / "refine.json")["result"]
    assert receipt["facts"]["relink"] == {
        "mode": "strict",
        "applied": False,
        "reason": "no_embeddings",
        "counts": None,
    }
    notes = [note for note in receipt["notes_zh"] if "--relink strict" in note]
    assert len(notes) == 1
    assert "没有运行" in notes[0]
    assert f"{len(tracks)} 行检测没有一行带外观特征" in notes[0]


def test_rows_without_an_embedding_are_counted_and_the_note_says_what_that_means(
    unicode_tmp_path: Path,
) -> None:
    scene = make_scene()
    emb = scene.emb.copy()
    returned = (scene.tracks["track_id"] == 9).to_numpy()
    emb[returned] = np.nan
    emb[np.flatnonzero(~returned)[:2]] = 0.0
    partly = Scene(scene.tracks, scene.calib, emb, scene.shots, scene.size)
    run = _fake_run(unicode_tmp_path / "partly", partly)
    calib = _sized(scene.calib, scene.size)
    tracks, section = pipeline_module._relink(run, scene.tracks, calib, scene.shots)
    assert section["applied"] is True
    assert section["links"] == []  # the piece of the return has no embedding: not joined
    assert section["counts"]["rows_without_embedding"] == int(returned.sum()) + 2
    assert tracks["track_id"].tolist() == scene.tracks["track_id"].tolist()
    assert len(run.notes) == 1
    assert f"{int(returned.sum()) + 2} 行检测没有外观特征" in run.notes[0]
    assert "不会被接上" in run.notes[0]
    assert "也不会被切开" in run.notes[0]
    # with an embedding in every row the note has no such sentence
    run = _fake_run(unicode_tmp_path / "all", scene)
    _tracks, section = pipeline_module._relink(run, scene.tracks, calib, scene.shots)
    assert section["counts"]["rows_without_embedding"] == 0
    assert "没有外观特征" not in run.notes[0]


def test_a_run_that_must_not_link_identities_reads_no_embedding(clip: dict[str, Any]) -> None:
    base = clip["base"]
    out = base / "run no identity"
    tags = ("no_share", "no_identity")
    statuses = run_three(_with(clip, "return"), out, base, relink="strict", restrictions=tags)
    assert statuses == dict.fromkeys(THREE, "ran")
    assert not (out / "import" / "embeddings.parquet").exists()
    imported = read_json(out / "receipts" / "import.json")
    assert imported["result"]["parameters"]["embeddings"] is False
    assert "sngs/embeddings.npz" not in [entry["name"] for entry in imported["inputs"]]
    report = read_json(out / "refine" / "refine.json")
    assert report["relink"] == {"mode": "strict", "applied": False, "reason": "no_identity"}
    tracks = pd.read_parquet(out / "refine" / "tracks.parquet")
    assert tracks["piece_id"].tolist() == tracks["track_id"].tolist()
    assert RETURNED in set(tracks["track_id"])
    notes = read_json(out / "receipts" / "refine.json")["result"]["notes_zh"]
    assert any("no_identity" in note and "没有运行" in note for note in notes)


def _sized(calib: pd.DataFrame, size: tuple[int, int]) -> pd.DataFrame:
    """A scene's calibration with the frame size columns of the refine stage's table."""
    return calib.assign(width=np.int32(size[0]), height=np.int32(size[1]))


def _fake_run(folder: Path, scene: Scene | None, **changes: Any) -> SimpleNamespace:
    """What gbf.pipeline._relink reads of a run, with the scene's embeddings as the import
    stage would have written them (``scene`` None: no embeddings file). ``facts`` is the
    video's size, which the step does not ask for."""
    if scene is not None:
        (folder / "import").mkdir(parents=True, exist_ok=True)
        dim = scene.emb.shape[1]
        table = pa.table(
            {
                "frame": pa.array(scene.tracks["frame"].to_numpy(), pa.int64()),
                "track_id": pa.array(scene.tracks["source_track_id"].to_numpy(), pa.int64()),
                "emb": pa.FixedSizeListArray.from_arrays(
                    pa.array(scene.emb.reshape(-1), pa.float32()), dim
                ),
            }
        )
        pq.write_table(table, folder / "import" / "embeddings.parquet")
    settings: dict[str, Any] = {
        "config": SimpleNamespace(relink="strict"),
        "facts": SimpleNamespace(width=1280, height=720),
        "restrictions": ("no_share",),
        "rate": FPS,
        "notes": [],
        "path": lambda name: folder.joinpath(*name.split("/")),
    }
    settings.update(changes)
    return SimpleNamespace(**settings)


def test_the_refine_step_checks_the_embeddings_against_the_track_rows(
    unicode_tmp_path: Path,
) -> None:
    scene = make_scene()
    run = _fake_run(unicode_tmp_path / "good", scene)
    calib = _sized(scene.calib, scene.size)
    tracks, section = pipeline_module._relink(run, scene.tracks, calib, scene.shots)
    assert section["applied"] is True
    assert section["links"] == [[5, 9]]
    assert section["embeddings"] == {"model": None, "dim": scene.emb.shape[1]}
    assert tracks["piece_id"].tolist() == scene.tracks["track_id"].tolist()
    assert set(tracks.loc[tracks["piece_id"] == 9, "track_id"]) == {5}
    assert len(run.notes) == 1

    # rows of another order: never a silent misalignment
    turned = scene.tracks.iloc[::-1].reset_index(drop=True)
    with pytest.raises(PipelineError, match="frame differs") as caught:
        pipeline_module._relink(run, turned, calib, scene.shots)
    assert caught.value.where == "refine"
    other_ids = scene.tracks.assign(source_track_id=scene.tracks["source_track_id"] + 1)
    with pytest.raises(PipelineError, match="track_id differs from the table's source_track_id"):
        pipeline_module._relink(run, other_ids, calib, scene.shots)
    with pytest.raises(PipelineError, match="rows but the track table has"):
        pipeline_module._relink(run, scene.tracks.iloc[:-1], calib, scene.shots)
    assert len(run.notes) == 1  # a failed step leaves no note


def test_the_refine_step_is_skipped_for_ids_it_cannot_cut_and_without_a_frame_size(
    unicode_tmp_path: Path,
) -> None:
    scene = make_scene()
    big = scene.tracks.copy()
    big.loc[big["track_id"] == 14, "track_id"] = relink.cuts.SUB_STRIDE + 14
    sized = _sized(scene.calib, scene.size)
    two_sizes = pd.concat([sized, _sized(scene.calib.iloc[:1], (960, 540))], ignore_index=True)
    empty = Scene(scene.tracks, scene.calib, np.zeros_like(scene.emb), scene.shots, scene.size)
    rows = {"rows_without_embedding": len(scene.tracks)}
    cases = [
        (_fake_run(unicode_tmp_path / "a", scene), big, sized, "track_ids_out_of_range", {}),
        # the calibration table gives no frame size, or more than one
        (_fake_run(unicode_tmp_path / "b", scene), scene.tracks, scene.calib, "no_frame_size", {}),
        (_fake_run(unicode_tmp_path / "b2", scene), scene.tracks, two_sizes, "no_frame_size", {}),
        (_fake_run(unicode_tmp_path / "c", None), scene.tracks, sized, "no_embeddings", {}),
        # the file is there, but no row of it holds an embedding
        (_fake_run(unicode_tmp_path / "c2", empty), scene.tracks, sized, "no_embeddings", rows),
        (
            _fake_run(unicode_tmp_path / "d", scene, restrictions=("no_identity", "no_share")),
            scene.tracks,
            sized,
            "no_identity",
            {},
        ),
    ]
    for run, table, calib, reason, more in cases:
        tracks, section = pipeline_module._relink(run, table, calib, scene.shots)
        assert section == {"mode": "strict", "applied": False, "reason": reason, **more}
        assert tracks["piece_id"].tolist() == table["track_id"].tolist()
        assert tracks["track_id"].tolist() == table["track_id"].tolist()
        assert len(run.notes) == 1
        assert "没有运行" in run.notes[0]
        assert "轨迹编号没有改变" in run.notes[0]
    # a table without rows has nothing to skip for: the step runs and finds nothing
    none = Scene(scene.tracks.iloc[:0], scene.calib, scene.emb[:0], scene.shots, scene.size)
    run = _fake_run(unicode_tmp_path / "e", none)
    _tracks, section = pipeline_module._relink(run, none.tracks, sized, scene.shots)
    assert (section["applied"], section["cuts"], section["links"]) == (True, [], [])


# --------------------------------------------------------------------------- import errors


def test_embeddings_that_do_not_describe_the_export_stop_the_import_only_when_they_are_read(
    clip: dict[str, Any],
) -> None:
    base = clip["base"]
    plain = _detections(_camera())
    emb = _embeddings(plain)
    frames = np.array([row["frame"] for row in plain], dtype=np.int32)
    frames[7] += 1
    exports = {
        "one row short": write_export(base / "bad rows", plain, _npz(plain[:-1], emb[:-1])),
        "another frame": write_export(base / "bad frame", plain, _npz(plain, emb, frame=frames)),
    }
    expected = {"one row short": "one row per detection line", "another frame": "does not describe"}
    for name, export in exports.items():
        inputs = {**clip["inputs"], "sngs": export}
        with pytest.raises(SngsImportError, match=expected[name]):
            run_pipeline(_config(inputs, base / f"run {name}", base, stages=THREE, relink="strict"))
        assert not (base / f"run {name}" / "import").exists()
    # off never opens the file: its hash is right, and that is all the import asks
    inputs = {**clip["inputs"], "sngs": exports["one row short"]}
    assert run_three(inputs, base / "run short off", base) == dict.fromkeys(THREE, "ran")
    # a listed file that is missing fails the receipt check in every mode, as before
    missing = write_export(base / "bad missing", plain, _npz(plain, emb), written=False)
    for mode in ("strict", "off"):
        with pytest.raises(SngsImportError, match="embeddings.npz is listed but missing"):
            run_pipeline(
                _config(
                    {**clip["inputs"], "sngs": missing},
                    base / f"run missing {mode}",
                    base,
                    stages=THREE,
                    relink=mode,
                )
            )
    # a file that is there but not listed is not read: nothing vouches for it
    unlisted = write_export(base / "unlisted", plain, _npz(plain, emb), listed=False)
    out = base / "run unlisted"
    run_three({**clip["inputs"], "sngs": unlisted}, out, base, relink="strict")
    assert read_json(out / "refine" / "refine.json")["relink"]["reason"] == "no_embeddings"


# --------------------------------------------------------------------------- CLI and config


def test_the_cli_takes_the_modes_of_gbf_relink_and_refuses_another(
    clip: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = clip["inputs"]
    common = [
        *("--video", str(inputs["video"]), "--audio-source", str(inputs["audio"])),
        *("--sngs", str(inputs["sngs"]), "--ball", str(inputs["ball"])),
        *("--out", str(clip["base"] / "cli run")),
    ]
    parser = pipeline_module._parser()
    assert parser.parse_args(common).relink == "off"
    for mode in relink.MODES:
        assert parser.parse_args([*common, "--relink", mode]).relink == mode
    assert "strict" in relink.MODES
    assert main([*common, "--relink", "loose"]) == 2
    assert "--relink" in capsys.readouterr().err
    assert not (clip["base"] / "cli run").exists()
    config = _config(inputs, clip["base"] / "cli run", clip["base"], relink="loose")
    problems = pipeline_module._check_config(config)
    assert problems == [f"--relink must be one of {', '.join(relink.MODES)}, got 'loose'"]
    with pytest.raises(PipelineError, match="--relink must be one of"):
        run_pipeline(config)
    assert not (clip["base"] / "cli run").exists()


def test_the_refine_stage_key_covers_every_relink_module() -> None:
    package = Path(pipeline_module.__file__).resolve().parent
    table = pipeline_module._stage_table()
    sources = pipeline_module._stage_sources(package, table["refine"])
    modules = sorted(f"relink/{path.name}" for path in (package / "relink").glob("*.py"))
    assert "relink/__init__.py" in modules
    assert "relink/api.py" in modules
    assert [name for name in modules if name not in sources] == []
    assert pipeline_module.relink_mod in table["refine"].modules
    assert pipeline_module._relink in table["refine"].modules
    assert pipeline_module._relink_note in table["refine"].modules
    # no other stage is rebuilt when a relink module changes
    for name, stage in table.items():
        if name != "refine":
            found = pipeline_module._stage_sources(package, stage)
            assert not any(key.startswith("relink/") for key in found), name


# --------------------------------------------------------------------------- notes, verdicts


def test_the_note_says_what_was_cut_and_joined_and_when_the_frame_rate_differs() -> None:
    section = {
        "mode": "strict",
        "applied": True,
        "reason": None,
        "fps": 50.0,
        "fitted_fps": 25.0,
        "cuts": [[7, 40], [7, 90], [12, 300]],
        "counts": {"cuts": 3, "core_links": 5, "leaf_links": 2, "links_refused_same_frame": 1},
    }
    note = pipeline_module._relink_note(section)
    for part in (
        "--relink strict",
        "2 条轨迹",
        "共 3 处",
        "7 对轨迹片段",
        "其中 2 对",
        "piece_id",
        "50 帧/秒",
        "25 帧/秒不同",
        "1 对片段在同一帧",
    ):
        assert part in note
    same_rate = pipeline_module._relink_note({**section, "fps": 25.1})
    assert "帧/秒不同" not in same_rate
    assert "没有外观特征" not in note
    isolated = pipeline_module._relink_note({**section, "isolation": {"rows": 2}})
    for part in ("2 次不清晰检测", "身份未确认", "不代表新增球员", "独立素材"):
        assert part in isolated
    counts = {**section["counts"], "rows_without_embedding": 12}
    without = pipeline_module._relink_note({**section, "counts": counts})
    assert without.startswith(note)
    for part in ("12 行检测没有外观特征", "不会被接上", "也不会被切开"):
        assert part in without[len(note) :]
    counts["rows_without_embedding"] = 0
    assert pipeline_module._relink_note({**section, "counts": counts}) == note
    no_vector = {**relink.skipped("strict", "no_embeddings"), "rows_without_embedding": 40}
    assert "40 行检测没有一行带外观特征" in pipeline_module._relink_note(no_vector)
    reasons = set()
    for reason in relink.api.SKIP_REASONS:
        skipped = pipeline_module._relink_note(relink.skipped("strict", reason))
        assert "没有运行" in skipped
        reasons.add(skipped)
    assert len(reasons) == len(relink.api.SKIP_REASONS)


def test_the_team_verdict_of_an_identity_is_the_weakest_of_its_pieces(
    unicode_tmp_path: Path,
) -> None:
    (unicode_tmp_path / "refine").mkdir()
    verdicts = [
        (2, "left", "consistent"),
        (22, "left", "unchecked"),
        (5, "left", "consistent"),
        (9, "left", "consistent"),
        (7, "right", "inconsistent"),
        (31, "right", "consistent"),
        (40, "right", "inconsistent"),
        (41, "right", "unchecked"),
    ]
    report = {
        "teams": {
            "tracks": [
                {"track_id": track, "team": team, "verdict": verdict}
                for track, team, verdict in verdicts
            ]
        }
    }
    rows = [
        # (final id, piece, team of the rows)
        (2, 2, "left"),
        (2, 22, "left"),
        (5, 5, "left"),
        (5, 9, "left"),
        (5, 9, "left"),
        (7, 7, None),  # the team check cleared the contradicted piece's label
        (7, 31, "right"),
        (40, 41, "right"),
        (40, 40, "right"),
        (60, 60, "left"),  # a piece the team check did not list
        (-1, -1, "left"),
    ]
    tracks = pd.DataFrame(rows, columns=["track_id", "piece_id", "team"])
    tracks["team"] = tracks["team"].astype("string")
    tracks.to_parquet(unicode_tmp_path / "refine" / "tracks.parquet")
    run = SimpleNamespace(path=lambda name: unicode_tmp_path / name)

    path = unicode_tmp_path / "refine" / "refine.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    # without a relink section the verdicts are the team check's own, by its ids
    assert pipeline_module._team_verdicts(run) == {
        (track, team): verdict for track, team, verdict in verdicts
    }
    path.write_text(json.dumps({**report, "relink": {"mode": "strict"}}), encoding="utf-8")
    assert pipeline_module._team_verdicts(run) == {
        (2, "left"): "unchecked",
        (5, "left"): "consistent",
        (7, "right"): "consistent",
        (40, "right"): "inconsistent",  # the first other verdict in piece order
        (60, "left"): "unchecked",
    }


# --------------------------------------------------------------------------- recall


def test_recall_runs_with_its_own_setting_and_joins_the_return(
    clip: dict[str, Any], off_run: Path
) -> None:
    base = clip["base"]
    out = base / "run recall"
    statuses = run_three(_with(clip, "return"), out, base, relink="recall")
    assert statuses == dict.fromkeys(THREE, "ran")
    section = read_json(out / "refine" / "refine.json")["relink"]
    assert (section["mode"], section["applied"], section["reason"]) == ("recall", True, None)
    assert section["setting"] == {"margin": 0.2, "leaves": True}
    assert section["links"] == [[RETURNING, RETURNED]]
    assert section["leaf_links"] == []  # every piece of this clip is long enough to be solid
    assert section["embeddings"] == {"model": None, "dim": DIM}
    # like strict, the run reads the embeddings: its own tag and the stage's limitation
    assert read_json(out / "run.json")["result"]["restriction_sources"]["relink"] == ["hl3"]
    limitations = read_json(out / "receipts" / "refine.json")["limitations"]
    assert sum("relink parameters" in line for line in limitations) == 1
    receipt = read_json(out / "receipts" / "refine.json")["result"]
    assert receipt["parameters"]["relink"]["mode"] == "recall"
    assert receipt["parameters"]["relink"]["setting"] == {"margin": 0.2, "leaves": True}
    assert sum("--relink recall" in note for note in receipt["notes_zh"]) == 1
    tracks = pd.read_parquet(out / "refine" / "tracks.parquet")
    plain = pd.read_parquet(off_run / "refine" / "tracks.parquet")
    assert tracks["piece_id"].tolist() == plain["track_id"].tolist()
    assert RETURNED not in set(tracks["track_id"])
    # the mode is a parameter of the refine stage: strict's result is not reused for recall
    statuses = run_three(_with(clip, "return"), out, base, relink="strict")
    assert statuses == {"import": "skipped", "shots": "skipped", "refine": "ran"}
    assert read_json(out / "refine" / "refine.json")["relink"]["mode"] == "strict"
