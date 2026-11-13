"""L0 synthetic smoke test (plan P0 deliverable, gate G5) and the gbf.synth building blocks.

test_l0_smoke runs ``python -m gbf l0`` at 720p and at 1080p inside a non-ASCII path with a
space, then re-checks every acceptance criterion from the files it wrote, through gbf.ingest
and gbf.plugins rather than through the run's own receipt: 5 frames CFR 25/1 of the right
size, barcodes [0..4] through the single frame reader, the cut at [3], a window at a
non-zero offset (2..4) whose frame k carries barcode 2 + k and is bit-identical to frame
2 + k, the calibration round trip (K, R, C equal the truth after rescaling, rtol 1e-12),
landmarks on drawn lines, and a schema-valid receipt without absolute paths.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from gbf import contracts
from gbf.gate.patterns import find_local_paths
from gbf.geometry import (
    REFERENCE_SIZE,
    Camera,
    look_at,
    pixel_to_ground,
    project,
    scale_intrinsics,
    synthetic_broadcast_camera,
)
from gbf.ingest import detect_cuts, iter_frames, pixel_sha256, probe
from gbf.plugins import accept_outputs, read_calib
from gbf.synth import l0
from gbf.synth.barcode import (
    COLUMNS,
    END_GUARD,
    MAX_VALUE,
    START_GUARD,
    BarcodeError,
    barcode_layout,
    decode_barcode,
    draw_barcode,
    encode_barcode,
)
from gbf.synth.pitch import goal_frames, landmarks, pitch_markings, spots
from gbf.synth.render import (
    GRASS_DARK,
    GRASS_LIGHT,
    LABEL_GRASS,
    LABEL_LINE,
    LABEL_SKY,
    LABEL_STAND,
    NEAR_CLIP_M,
    SynthError,
    ground_points,
    line_mask,
    line_radius_px,
    render_sequence,
    sequence_cameras,
    visible_landmarks,
)
from gbf.synth.video import SynthVideoError, write_frames_mp4
from gbf.util.hashing import sha256_file
from gbf.util.io import read_json

SIZES = {"720p": (1280, 720), "1080p": (1920, 1080)}


def all_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for key, item in value.items() for s in (key, *all_strings(item))]
    if isinstance(value, list):
        return [s for item in value for s in all_strings(item)]
    return []


# --- the end-to-end smoke run ------------------------------------------------------------
@pytest.mark.parametrize("label", ["720p", "1080p"])
def test_l0_smoke(
    label: str, unicode_tmp_path: Path, ffmpeg_bin: str, ffprobe_bin: str, capsys
) -> None:
    assert ffmpeg_bin
    assert ffprobe_bin
    width, height = SIZES[label]
    out = unicode_tmp_path / "l0 run"
    code = l0.main(["--out", str(out), "--resolution", label])
    printed = capsys.readouterr()
    assert code == 0, printed.err + printed.out
    assert json.loads(printed.out)["status"] == "passed"
    work = out / label

    # The video: 5 frames, CFR 25/1, the rendered size.
    facts = probe(work / "l0.mp4")
    assert facts.nb_frames_decoded == 5
    assert facts.is_cfr_at("25/1")
    assert facts.r_frame_rate == facts.avg_frame_rate == "25/1"
    assert (facts.frame_width, facts.frame_height) == (width, height)
    assert facts.codec_name == "h264"

    # Every frame through the single frame reader carries its own number.
    frames = [frame for _, frame in iter_frames(work / "l0.mp4")]
    assert [decode_barcode(frame) for frame in frames] == [0, 1, 2, 3, 4]
    assert detect_cuts(frames) == [3]

    # The window at a non-zero offset: frame k is canonical frame 2 + k, bit for bit.
    window = [frame for _, frame in iter_frames(work / "windows" / "w000002-000004.mkv")]
    assert [decode_barcode(frame) for frame in window] == [2, 3, 4]
    assert all(np.array_equal(frame, frames[2 + k]) for k, frame in enumerate(window))
    record = read_json(work / "video.json")
    contracts.validate(record, "video")
    assert record["cuts"] == [3]
    assert record["frame_count"] == 5
    assert record["windows"][0]["start_frame"] == 2
    assert record["windows"][0]["end_frame"] == 4
    assert record["windows"][0]["first_pixel_sha256"] == pixel_sha256(frames[2])
    assert record["windows"][0]["last_pixel_sha256"] == pixel_sha256(frames[4])

    # The plugin round trip: the core re-verifies the receipt's hashes, then the cameras.
    (job_dir,) = list((work / "plugins").iterdir())
    accepted = accept_outputs(job_dir)
    plugin_receipt = read_json(job_dir / "plugin-receipt.json")
    contracts.validate(plugin_receipt, "plugin-receipt")
    listed = plugin_receipt["outputs"][0]
    assert listed["sha256"] == sha256_file(job_dir / listed["relpath"])
    assert [item["frame_index"] for item in plugin_receipt["decoded_frames"]] == [2, 3, 4]
    assert [item["sha256"] for item in plugin_receipt["decoded_frames"]] == [
        pixel_sha256(frame) for frame in frames[2:5]
    ]
    rows = read_calib(accepted.outputs["calib"])
    truth = {item.frame_index: item.camera for item in sequence_cameras(width, height, 5, 3)}
    recorded = read_json(work / "cameras.json")["frames"]
    assert [item["frame_index"] for item in recorded] == [0, 1, 2, 3, 4]
    assert [row.frame_index for row in rows] == [2, 3, 4]
    for row in rows:
        assert (row.width, row.height) == REFERENCE_SIZE  # reported at 1920x1080 regardless
        assert (row.calib_status, row.calib_age, row.rep_err) == ("fresh", 0, None)
        camera = row.camera(width, height)
        assert camera is not None
        expected = truth[row.frame_index]
        np.testing.assert_allclose(camera.K, expected.K, rtol=1e-12, atol=0)
        np.testing.assert_allclose(camera.R, expected.R, rtol=1e-12, atol=0)
        np.testing.assert_allclose(camera.C, expected.C, rtol=1e-12, atol=0)
        np.testing.assert_allclose(np.array(recorded[row.frame_index]["K"]), expected.K, rtol=0)

        # Landmarks projected with the round-trip camera land on drawn lines.
        visible = visible_landmarks(camera)
        assert len(visible) >= 4
        if row.frame_index == 2:
            assert "centre_spot" in visible  # the main camera looks at the centre circle
        unscaled = row.camera()
        assert unscaled is not None
        unscaled_hits = 0
        for name, (u, v) in visible.items():
            pixel = frames[row.frame_index][int(np.floor(v + 0.5)), int(np.floor(u + 0.5))]
            assert pixel.min() >= l0.LINE_MIN_CHANNEL, (row.frame_index, name, pixel)
            (u2, v2), _, ok = project(unscaled, landmarks()[name])
            i, j = int(np.floor(u2 + 0.5)), int(np.floor(v2 + 0.5))
            if ok and 0 <= i < width and 0 <= j < height:
                unscaled_hits += int(frames[row.frame_index][j, i].min() >= l0.LINE_MIN_CHANNEL)
        if (width, height) != REFERENCE_SIZE:
            # Control: forgetting to rescale K (gate G7) would make this check fail.
            assert unscaled_hits < len(visible) / 2, (row.frame_index, unscaled_hits)

    # The L0 receipt: schema-valid, passed, and free of absolute paths.
    receipt = read_json(out / "l0-receipt.json")
    contracts.validate(receipt, "receipt")
    assert (receipt["kind"], receipt["status"], receipt["producer"]) == (
        "l0",
        "passed",
        "gbf.synth.l0",
    )
    (run,) = receipt["result"]["runs"]
    assert [check["name"] for check in run["checks"]] == list(l0.CHECK_NAMES)
    assert all(check["ok"] for check in run["checks"])
    for ref in receipt["outputs"]:
        assert sha256_file(out / ref["relpath"]) == ref["sha256"]
    text = (out / "l0-receipt.json").read_text(encoding="utf-8")
    assert find_local_paths(text) == []
    roots = {str(unicode_tmp_path), unicode_tmp_path.as_posix(), str(Path.home())}
    for value in all_strings(receipt):
        assert not re.match(r"^[A-Za-z]:|^/|^~", value), value
        assert "\\" not in value, value
        assert not any(root in value for root in roots), value


def test_l0_cli_usage_errors(unicode_tmp_path: Path, capsys) -> None:
    assert l0.main([]) == 2
    assert l0.main(["--out", str(unicode_tmp_path), "--resolution", "4k"]) == 2
    (unicode_tmp_path / "taken" / "720p").mkdir(parents=True)
    assert l0.main(["--out", str(unicode_tmp_path / "taken")]) == 1
    err = capsys.readouterr().err
    assert "already holds 720p" in err
    assert str(unicode_tmp_path) not in err  # error messages are scrubbed as well


def test_scrub_removes_local_paths(unicode_tmp_path: Path) -> None:
    # Home paths are assembled at run time so that this file never matches the gate's
    # user-home-path rule itself.
    backslash, users, home = chr(92), "Us" + "ers", "ho" + "me"
    windows_home = backslash.join(["C:", users, "someone", "video.mp4"])
    posix_home = "/".join(["", home, "someone", "clip.mp4"])
    text = (
        f"cannot open {unicode_tmp_path / 'x.mkv'} or file:D:/data/a b.mp4 or "
        f"{windows_home} or {posix_home}; frames 25/1 in/video.mkv"
    )
    cleaned = l0.scrub(text, [unicode_tmp_path])
    assert "<root>" in cleaned
    assert "<path>" in cleaned
    assert "someone" not in cleaned
    assert "D:/" not in cleaned
    assert "25/1" in cleaned
    assert "in/video.mkv" in cleaned


# --- barcode -----------------------------------------------------------------------------
@pytest.mark.parametrize(("width", "height"), [(320, 180), (1280, 720), (1920, 1080)])
def test_barcode_round_trip(width: int, height: int) -> None:
    for value in (0, 1, 2, 5, 1023, 0x5A5A5, MAX_VALUE):
        frame = np.full((height, width, 3), 90, dtype=np.uint8)
        draw_barcode(frame, value)
        assert decode_barcode(frame) == value
    layout = barcode_layout(width, height)
    assert layout.cell % 8 == 0
    assert layout.x0 % 8 == 0
    assert layout.y0 % 8 == 0


def test_barcode_bits() -> None:
    bits = encode_barcode(0b1010_0000_0000_0000_0011)
    assert bits.shape == (2, COLUMNS)
    assert tuple(bits[0, : len(START_GUARD)]) == START_GUARD
    assert tuple(bits[0, -len(END_GUARD) :]) == END_GUARD
    data = bits[0, len(START_GUARD) : -len(END_GUARD)]
    assert "".join(map(str, data)) == "10100000000000000011"
    assert np.array_equal(bits[1], 1 - bits[0])


@pytest.mark.parametrize("value", [-1, MAX_VALUE + 1, True, 1.0, "7"])
def test_barcode_rejects_bad_values(value: object) -> None:
    with pytest.raises(BarcodeError) as caught:
        encode_barcode(value)  # type: ignore[arg-type]
    assert repr(value) in caught.value.problems[0] or str(value) in caught.value.problems[0]


def _flip_column(frame: np.ndarray, column: int) -> None:
    """Swap the colours of both cells of one column (keeps them complementary)."""
    layout = barcode_layout(frame.shape[1], frame.shape[0])
    top0, bottom0, left, right = layout.cell_box(0, column)
    top1, bottom1, _, _ = layout.cell_box(1, column)
    upper = frame[top0:bottom0, left:right].copy()
    frame[top0:bottom0, left:right] = frame[top1:bottom1, left:right]
    frame[top1:bottom1, left:right] = upper


def test_decoder_raises_on_guard_mismatch() -> None:
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    draw_barcode(frame, 3)
    start_broken = frame.copy()
    _flip_column(start_broken, 1)
    with pytest.raises(BarcodeError) as caught:
        decode_barcode(start_broken)
    assert caught.value.problems == ["start guard reads 1111, expected 1011"]
    end_broken = frame.copy()
    _flip_column(end_broken, COLUMNS - 1)
    with pytest.raises(BarcodeError) as caught:
        decode_barcode(end_broken)
    assert caught.value.problems == ["end guard reads 1100, expected 1101"]


def test_decoder_raises_on_broken_cells_and_on_frames_without_a_code() -> None:
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    draw_barcode(frame, 12345)
    layout = barcode_layout(1280, 720)
    top, bottom, left, right = layout.cell_box(1, 10)
    frame[top:bottom, left:right] = 255 - frame[top:bottom, left:right]  # breaks the pair
    top, bottom, left, right = layout.cell_box(0, 12)
    frame[top:bottom, left:right] = 128  # ambiguous grey
    with pytest.raises(BarcodeError) as caught:
        decode_barcode(frame)
    text = "\n".join(caught.value.problems)
    assert "cell (row 0, column 12) has mean luma 128.0" in text
    assert "not complementary: 10" in text
    plain = np.zeros((720, 1280, 3), dtype=np.uint8)
    plain[...] = (48, 112, 48)
    with pytest.raises(BarcodeError) as caught:
        decode_barcode(plain)
    assert len(caught.value.problems) >= 3
    with pytest.raises(BarcodeError, match="needs a frame of at least"):
        decode_barcode(np.zeros((40, 200, 3), dtype=np.uint8))


def test_barcode_survives_harsh_h264_at_720p(unicode_tmp_path: Path, ffmpeg_bin: str) -> None:
    """Large flat cells survive x264 at CRF 35 over a noisy background."""
    assert ffmpeg_bin
    rng = np.random.default_rng(7)
    values = [0, 0x5A5A5, MAX_VALUE]
    frames = []
    for value in values:
        frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
        draw_barcode(frame, value)
        frames.append(frame)
    path = unicode_tmp_path / "noisy codes.mp4"
    write_frames_mp4(frames, path, crf=35)
    decoded = [frame for _, frame in iter_frames(path)]
    assert any(not np.array_equal(a, b) for a, b in zip(decoded, frames, strict=True))  # lossy
    assert [decode_barcode(frame) for frame in decoded] == values


# --- pitch -------------------------------------------------------------------------------
def _distance_to_polyline(point: np.ndarray, points: np.ndarray) -> float:
    a, b = points[:-1], points[1:]
    d = b - a
    t = np.clip(np.einsum("ij,ij->i", point - a, d) / np.einsum("ij,ij->i", d, d), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * d - point, axis=1)))


def test_pitch_landmarks_lie_on_the_markings() -> None:
    markings = (*pitch_markings(), *goal_frames())
    marks = spots()
    for name, point in landmarks().items():
        if name in marks:
            continue
        nearest = min(_distance_to_polyline(point, m.points) for m in markings)
        assert nearest < 1e-9, (name, nearest)
    by_name = {marking.name: marking.points for marking in markings}
    circle = by_name["centre_circle"]
    np.testing.assert_allclose(np.linalg.norm(circle[:, :2], axis=1), 9.15, rtol=0, atol=1e-12)
    front = by_name["penalty_area_left_front"]
    assert np.isclose(np.linalg.norm(front[1] - front[0]), 40.32)
    arc = by_name["penalty_arc_left"]
    np.testing.assert_allclose(arc[[0, -1], 0], -36.0, atol=1e-9)  # it ends on the area line
    assert (arc[1:-1, 0] > -36.0).all()
    assert np.isclose(np.linalg.norm(marks["penalty_spot_right"] - [52.5, 0.0, 0.0]), 11.0)
    goal = by_name["goal_left"]
    assert np.isclose(goal[1, 2], -2.44)  # crossbar 2.44 m up: z = -2.44 in the z-down frame
    assert np.isclose(np.linalg.norm(goal[2] - goal[1]), 7.32)
    for marking in markings:
        inside = (np.abs(marking.points[:, 0]) <= 52.5 + 1e-12) & (
            np.abs(marking.points[:, 1]) <= 34.0 + 1e-12
        )
        assert inside.all(), marking.name


# --- renderer ----------------------------------------------------------------------------
def test_cameras_of_a_sequence() -> None:
    for width, height in SIZES.values():
        truth = sequence_cameras(width, height, 5, 3)
        assert [item.shot_id for item in truth] == [0, 0, 0, 1, 1]
        reference = synthetic_broadcast_camera(width, height)
        assert np.array_equal(truth[0].camera.K, reference.K)
        assert np.array_equal(truth[0].camera.R, reference.R)
        assert np.array_equal(truth[0].camera.C, reference.C)
        for a in range(5):
            for b in range(a + 1, 5):
                ca, cb = truth[a].camera, truth[b].camera
                same = all(np.array_equal(getattr(ca, n), getattr(cb, n)) for n in "KRC")
                assert not same, (a, b)
        assert np.linalg.norm(truth[3].camera.C - truth[2].camera.C) > 50.0
    small = sequence_cameras(1280, 720, 5, 3)
    large = sequence_cameras(1920, 1080, 5, 3)
    for s, b in zip(small, large, strict=True):
        assert np.array_equal(s.camera.R, b.camera.R)
        assert np.array_equal(s.camera.C, b.camera.C)
        expected = scale_intrinsics(b.camera.K, (1920, 1080), (1280, 720))
        np.testing.assert_allclose(s.camera.K, expected, rtol=1e-14, atol=1e-12)


def test_closed_form_ground_points_match_gbf_geometry() -> None:
    seen_ground = seen_sky = 0
    for item in sequence_cameras(640, 360, 5, 3):
        u, v = np.meshgrid(np.arange(0, 640, 9.5), np.arange(0, 360, 7.25))
        uv = np.stack([u, v], axis=-1)
        xy, valid = ground_points(item.camera, uv)
        points, expected_valid = pixel_to_ground(item.camera, uv)
        assert np.array_equal(valid, expected_valid)
        np.testing.assert_allclose(xy[valid], points[valid][:, :2], rtol=1e-9, atol=1e-9)
        assert np.isnan(xy[~valid]).all()
        seen_ground += int(valid.sum())
        seen_sky += int((~valid).sum())
    assert seen_ground > 1000
    assert seen_sky > 100


def test_painted_background_follows_the_ground_geometry() -> None:
    """Sky exactly where pixel rays miss the grass; grass (with its stripe colour) exactly
    where they meet the pitch or its 5 m apron; stands elsewhere."""
    sequence = render_sequence(640, 360, n_frames=5, cut_at=3, seed=0)
    u, v = np.meshgrid(np.arange(640.0), np.arange(360.0))
    for item, frame, labels in zip(sequence.truth, sequence.frames, sequence.labels, strict=True):
        xy, valid = ground_points(item.camera, np.stack([u, v], axis=-1))
        x = np.where(valid, xy[..., 0], 0.0)
        y = np.where(valid, xy[..., 1], 0.0)
        grass = valid & (np.abs(x) <= 57.5) & (np.abs(y) <= 39.0)
        expected = np.where(~valid, LABEL_SKY, np.where(grass, LABEL_GRASS, LABEL_STAND))
        background = np.isin(labels, (LABEL_SKY, LABEL_STAND, LABEL_GRASS))
        assert background.mean() > 0.9
        assert (labels[background] != expected[background]).sum() <= 5, item.frame_index
        light = np.floor((x + 52.5) / 5.25) % 2 == 0
        on_grass = background & (labels == LABEL_GRASS) & (expected == LABEL_GRASS)
        colour = np.where(light[..., None], GRASS_LIGHT, GRASS_DARK)
        wrong = (frame != colour).any(axis=-1) & on_grass
        assert wrong.sum() <= 5, item.frame_index
    assert (sequence.labels[3] == LABEL_SKY).any()  # the reverse angle shows sky
    assert not (sequence.labels[0] == LABEL_SKY).any()  # the main camera does not


def test_render_is_deterministic_and_seeded() -> None:
    first = render_sequence(480, 270, n_frames=3, cut_at=2, seed=0)
    again = render_sequence(480, 270, n_frames=3, cut_at=2, seed=0)
    other = render_sequence(480, 270, n_frames=3, cut_at=2, seed=1)
    hashes = [pixel_sha256(frame) for frame in first.frames]
    assert hashes == [pixel_sha256(frame) for frame in again.frames]
    assert hashes != [pixel_sha256(frame) for frame in other.frames]
    for a, b in zip(first.truth, other.truth, strict=True):
        assert np.array_equal(a.camera.K, b.camera.K)
        assert np.array_equal(a.camera.R, b.camera.R)
    assert len(first.scene.persons) == 23
    assert first.scene.dropped == 0
    assert not first.frames[0].flags.writeable
    record = first.truth_record()
    assert record["cuts"] == [2]
    assert [f["shot_id"] for f in record["frames"]] == [0, 0, 1]
    assert json.loads(json.dumps(record)) == record


def test_landmarks_fall_on_drawn_lines_in_every_frame() -> None:
    sequence = render_sequence(640, 360, n_frames=5, cut_at=3, seed=0)
    for item, labels in zip(sequence.truth, sequence.labels, strict=True):
        visible = visible_landmarks(item.camera)
        assert len(visible) >= 4, item.frame_index
        for name, (u, v) in visible.items():
            assert labels[int(np.floor(v + 0.5)), int(np.floor(u + 0.5))] == LABEL_LINE, name
    assert visible_landmarks(sequence.truth[0].camera).keys() >= {"centre_spot"}


def test_lines_are_clipped_behind_the_camera() -> None:
    """A camera standing on the pitch: lines behind it must neither show up mirrored nor
    break the lines in front of it. Checked against densely sampled marking points."""
    centre = np.array([0.0, 0.0, -2.0])
    K = scale_intrinsics(
        [[700.0, 0.0, 959.5], [0.0, 700.0, 539.5], [0.0, 0.0, 1.0]], (1920, 1080), (640, 360)
    )
    camera = Camera(K=K, R=look_at(centre, (30.0, 5.0, 0.0)), C=centre, width=640, height=360)
    mask = line_mask(camera)
    samples = [np.array(list(spots().values()))]  # marks are discs around these
    for marking in (*pitch_markings(), *goal_frames()):
        points = marking.points
        for a, b in zip(points[:-1], points[1:], strict=True):
            steps = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.01)) + 1)
            samples.append(a + np.linspace(0.0, 1.0, steps)[:, None] * (b - a))
    world = np.concatenate(samples)
    uv, depth, valid = project(camera, world)
    keep = valid & (depth >= NEAR_CLIP_M)
    assert (depth < 0).sum() > 10_000  # much of the pitch lies behind this camera
    pixels = np.floor(uv[keep] + 0.5).astype(int)
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < 640) & (pixels[:, 1] >= 0) & (pixels[:, 1] < 360)
    pixels = pixels[inside]
    assert len(pixels) > 1000
    # No gaps: every sample in front of the camera lies on a drawn line pixel.
    assert mask[pixels[:, 1], pixels[:, 0]].all()
    # No phantoms: every drawn pixel lies near a sample in front of the camera.
    near = np.zeros_like(mask)
    reach = int(np.ceil(line_radius_px(360))) + 2
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            if dx * dx + dy * dy <= (reach + 0.5) ** 2:
                x = np.clip(pixels[:, 0] + dx, 0, 639)
                y = np.clip(pixels[:, 1] + dy, 0, 359)
                near[y, x] = True
    assert not (mask & ~near).any()


def test_render_sequence_reports_every_bad_argument() -> None:
    with pytest.raises(SynthError) as caught:
        render_sequence(0, 720.0, n_frames=5, cut_at=5, seed=-1)  # type: ignore[arg-type]
    text = "\n".join(caught.value.problems)
    assert len(caught.value.problems) == 4
    assert "width must be a positive integer" in text
    assert "height must be" in text
    assert "cut_at must lie in 1..n_frames-1 = 1..4, got 5" in text
    assert "seed must be a non-negative integer" in text
    with pytest.raises(BarcodeError, match="needs a frame of at least"):
        render_sequence(200, 100)


def test_write_frames_mp4_reports_every_problem(unicode_tmp_path: Path) -> None:
    good = np.zeros((48, 64, 3), dtype=np.uint8)
    with pytest.raises(SynthVideoError) as caught:
        write_frames_mp4(
            [np.zeros((47, 63, 3), np.uint8), good, good.astype(np.float32)],
            unicode_tmp_path / "out.mkv",
            crf=99,
        )
    text = "\n".join(caught.value.problems)
    assert "even width and height, got 63x47" in text
    assert "frame 1 has shape (48, 64, 3), frame 0 has (47, 63, 3)" in text
    assert "frame 2 has dtype float32" in text
    assert "must end in .mp4" in text
    assert "crf must be an integer in 0..51" in text
    existing = unicode_tmp_path / "exists.mp4"
    existing.write_bytes(b"")
    with pytest.raises(FileExistsError):
        write_frames_mp4([good], existing)
    assert existing.read_bytes() == b""
