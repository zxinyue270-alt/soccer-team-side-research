"""Tests for gbf.fonts and gbf.render: CJK fonts, overlays on frames, H.264 writing, edits.

Everything here is synthetic: frames are drawn with numpy, videos come from gbf's own
synthetic writer, audio from ffmpeg's lavfi sources. Frame-exactness of edits is checked
with the frame-number barcodes of gbf.synth.barcode.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import ImageFont

from gbf import measure
from gbf import render as render_module
from gbf.fonts import (
    FALLBACK_ORDER,
    FontNotFoundError,
    find_cjk_font,
    find_cjk_fonts,
    load_font,
    missing_glyphs,
)
from gbf.geometry import (
    Camera,
    pitch_zup_to_soccernet,
    pixel_to_ground,
    project,
    synthetic_broadcast_camera,
)
from gbf.ingest import iter_frames, probe
from gbf.ingest.tools import media_arg
from gbf.lift3d import lift_height_prior, lift_row, lift_table
from gbf.measure import LIFTED, MeasureError
from gbf.pose import HALPE26_INDEX
from gbf.render import (
    FACING_ARROW_LENGTH_M,
    FACING_LABEL_ZH,
    FOCUS_BALL_MAX_M,
    HIGHLIGHT_TYPES_ZH,
    RING_RX_PER_BOX_WIDTH,
    RING_RY_PER_RX,
    TRACK_HOLD_MAX_GAP,
    Arrow,
    AudioSpan,
    Badge,
    BallMark,
    Banner,
    CardRow,
    FacingGates,
    FrameAnnotations,
    HighlightScene,
    InfoCard,
    LinkLine,
    Minimap,
    PlayerMark,
    RenderError,
    SceneTables,
    Silence,
    Skeleton,
    Style,
    TextLabel,
    VideoWriter,
    annotation_texts,
    camera_footprint,
    draw_annotations,
    format_clock,
    ground_circle_px,
    merge_annotations,
    minimap_geometry,
    plan_highlight_edit,
    render_edit,
    render_highlight,
    save_png,
    write_video,
)
from gbf.synth.barcode import decode_barcode, draw_barcode
from gbf.synth.video import write_frames_mp4
from gbf.track_tools import OFF_PITCH_MARGIN_M
from gbf.util.hashing import sha256_file
from gbf.util.io import read_image_rgb

GRASS = (60, 120, 60)


@pytest.fixture(scope="module")
def style() -> Style:
    """The default style with the system CJK fonts resolved once (a missing font fails)."""
    return Style().with_fonts()


def _frame(width: int = 640, height: int = 360, colour: tuple[int, int, int] = GRASS):
    frame = np.empty((height, width, 3), np.uint8)
    frame[:] = colour
    return frame


def _changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a != b).any(axis=2)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _record_drawn_texts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """From now on, every string gbf.render draws, in order (wraps its text primitive)."""
    drawn: list[str] = []
    draw_text = render_module._text

    def recording(c: object, xy: object, text: str, *args: object, **kwargs: object) -> None:
        drawn.append(text)
        draw_text(c, xy, text, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(render_module, "_text", recording)
    return drawn


# ---------------------------------------------------------------------------- fonts
def test_system_cjk_font_draws_real_glyphs(style: Style) -> None:
    info = style.font
    assert info is not None
    font = load_font(info, 32)
    assert missing_glyphs(font, "中文高光分析未验证球场") == []
    assert info.sha256 == sha256_file(info.path)
    manifest = info.manifest()
    assert manifest["file_name"] == info.path.name
    assert "/" not in json.dumps(manifest)
    assert "\\" not in json.dumps(manifest)


def test_missing_glyphs_detects_a_font_without_cjk() -> None:
    latin = ImageFont.load_default(size=24)
    assert isinstance(latin, ImageFont.FreeTypeFont)
    assert missing_glyphs(latin, "A中") == ["中"]


def test_explicit_font_path_must_work(unicode_tmp_path: Path) -> None:
    missing = unicode_tmp_path / "nope.ttf"
    with pytest.raises(FontNotFoundError) as excinfo:
        find_cjk_font(path=missing)
    assert "nope.ttf" in str(excinfo.value)
    broken = unicode_tmp_path / "broken.ttf"
    broken.write_bytes(b"not a font")
    with pytest.raises(FontNotFoundError, match="cannot be opened"):
        find_cjk_font(path=broken)


def test_environment_variable_overrides_and_never_falls_back(
    style: Style, unicode_tmp_path: Path
) -> None:
    assert style.font is not None
    chosen = find_cjk_font(environ={"GBF_CJK_FONT": str(style.font.path)})
    assert chosen.source == "environment"
    assert chosen.sha256 == style.font.sha256
    with pytest.raises(FontNotFoundError, match="GBF_CJK_FONT is set but unusable"):
        find_cjk_font(environ={"GBF_CJK_FONT": str(unicode_tmp_path / "gone.ttf")})


def test_fallback_order_skips_broken_and_missing_candidates(
    style: Style, unicode_tmp_path: Path
) -> None:
    assert style.font is not None
    first, second = FALLBACK_ORDER["win32"]["regular"][:2]
    (unicode_tmp_path / first.file_name).write_bytes(b"broken")
    shutil.copyfile(style.font.path, unicode_tmp_path / second.file_name)
    found = find_cjk_font(platform="win32", search_dirs=[unicode_tmp_path], environ={})
    assert found.path.name == second.file_name
    assert found.source == "system"
    empty = unicode_tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FontNotFoundError) as excinfo:
        find_cjk_font(platform="win32", search_dirs=[empty], environ={})
    assert len(excinfo.value.problems) == len(FALLBACK_ORDER["win32"]["regular"])


def test_load_font_rejects_bad_sizes(style: Style) -> None:
    assert style.font is not None
    for size in (0, -3, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="size_px"):
            load_font(style.font, size)


def test_find_cjk_fonts_regular_and_bold() -> None:
    regular, bold = find_cjk_fonts()
    assert regular.weight == "regular"
    assert bold is None or bold.weight == "bold"


# ---------------------------------------------------------------------------- drawing
def test_empty_annotations_return_an_identical_copy(style: Style) -> None:
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(90, 160, 3), dtype=np.uint8)
    for supersample in (1, 2, 3):
        out = draw_annotations(
            frame, FrameAnnotations(), Style(font=style.font, supersample=supersample)
        )
        assert out is not frame
        assert np.array_equal(out, frame)


def test_drawing_never_modifies_the_input(style: Style) -> None:
    frame = _frame()
    reference = frame.copy()
    annotations = FrameAnnotations(
        players=(PlayerMark((300, 120, 330, 220), team="left", label="#9 测试", emphasis="focus"),),
        banner=Banner("高光 1", "00:01.00", "副标题"),
        badges=("未验证",),
    )
    out = draw_annotations(frame, annotations, style)
    assert np.array_equal(frame, reference)
    assert out.dtype == np.uint8
    assert out.shape == frame.shape
    assert out.flags.writeable
    view = frame[:, ::-1]  # a non-contiguous input is accepted and left alone too
    draw_annotations(view, annotations, style)
    assert np.array_equal(frame, reference)


def test_player_ring_and_brackets_change_only_pixels_around_the_player(style: Style) -> None:
    frame = _frame()
    x1, y1, x2, y2 = 300.0, 120.0, 330.0, 220.0
    out = draw_annotations(
        frame, FrameAnnotations(players=(PlayerMark((x1, y1, x2, y2), team="right"),)), style
    )
    changed = _changed(out, frame)
    ui = 360 / 1080
    rx = max(RING_RX_PER_BOX_WIDTH * (x2 - x1), 10 * ui)
    ry = max(RING_RY_PER_RX * rx, 4 * ui)
    cx = (x1 + x2) / 2
    left, top, right, bottom = _bbox(changed)
    assert left >= math.floor(min(x1, cx - rx)) - 3
    assert right <= math.ceil(max(x2, cx + rx)) + 3
    assert top >= y1 - 4
    assert bottom <= math.ceil(y2 + ry) + 3
    assert changed[
        int(round(y2)) - 1 : int(round(y2)) + 2, int(round(cx + rx)) - 1 : int(round(cx + rx)) + 2
    ].any()
    assert changed[int(y1) - 3 : int(y1) + 2, int(x1) - 3 : int(x1) + 2].any()  # top-left bracket
    assert not changed[
        int(y1) + 30 : int(y1) + 60, int(x1) + 8 : int(x2) - 8
    ].any()  # inside the box


def test_focus_ring_uses_the_accent_and_team_colours(style: Style) -> None:
    frame = _frame(colour=(40, 40, 40))
    x1, y1, x2, y2 = 300.0, 100.0, 360.0, 260.0
    out = draw_annotations(
        frame,
        FrameAnnotations(players=(PlayerMark((x1, y1, x2, y2), team="left", emphasis="focus"),)),
        style,
    )
    rx = RING_RX_PER_BOX_WIDTH * (x2 - x1)
    u, v = int(round((x1 + x2) / 2 + rx)), int(round(y2))
    window = out[v - 2 : v + 3, u - 2 : u + 3].reshape(-1, 3).astype(int)
    accent = np.array(style.accent)
    assert (np.abs(window - accent).sum(axis=1) < 60).any()
    custom = Style(font=style.font, team_colours=(("left", (255, 0, 255)),))
    out = draw_annotations(
        frame, FrameAnnotations(players=(PlayerMark((x1, y1, x2, y2), team="left"),)), custom
    )
    window = out[v - 2 : v + 3, u - 2 : u + 3].reshape(-1, 3).astype(int)
    assert (np.abs(window - np.array([255, 0, 255])).sum(axis=1) < 60).any()


def test_labels_do_not_cover_each_other(style: Style) -> None:
    frame = _frame(colour=(200, 200, 200))
    box = (300.0, 150.0, 330.0, 250.0)
    marks = (PlayerMark(box, team="left", label="甲"), PlayerMark(box, team="right", label="乙"))
    out = draw_annotations(frame, FrameAnnotations(players=marks), style)
    one = draw_annotations(frame, FrameAnnotations(players=marks[:1]), style)
    rows_two = np.nonzero(_changed(out, frame)[:146, 280:350].any(axis=1))[0]
    rows_one = np.nonzero(_changed(one, frame)[:146, 280:350].any(axis=1))[0]
    height_one = rows_one.max() - rows_one.min() + 1
    height_two = rows_two.max() - rows_two.min() + 1
    assert height_two >= 2 * height_one, "the second label is stacked above the first"


def test_compact_badge_is_its_coloured_text_without_a_chip(style: Style) -> None:
    frame = _frame(1920, 1080, colour=(200, 200, 200))
    box = (900.0, 450.0, 990.0, 750.0)
    amber = np.array(style.badge_colour("unverified"))

    def label(badges: tuple[Badge, ...]) -> tuple[int, bool]:
        """Width of the label panel above the player, and whether amber is drawn in it."""
        mark = PlayerMark(box, team="right", label="轨迹 2 · 32.4 km/h", badges=badges)
        out = draw_annotations(frame, FrameAnnotations(players=(mark,)), style)[:440]
        left, _, right, _ = _bbox(_changed(out, frame[:440]))
        return right - left, bool((np.abs(out.astype(int) - amber).sum(axis=2) < 60).any())

    plain, plain_amber = label(())
    compact, compact_amber = label((Badge("未验证", "unverified", compact=True),))
    chip, _ = label((Badge("未验证", "unverified"),))
    assert plain < compact < chip, "the compact marker is narrower than the chip"
    assert compact_amber, "the marker is the badge text in the unverified colour"
    assert not plain_amber
    with pytest.raises(RenderError, match="compact"):
        Badge("未验证", "unverified", compact="yes")  # type: ignore[arg-type]


def test_semi_transparent_label_panel_and_cjk_text(style: Style) -> None:
    frame = _frame(colour=(200, 200, 200))
    out = draw_annotations(
        frame,
        FrameAnnotations(texts=(TextLabel((320, 180), "中文标签", anchor="mm", size=40),)),
        style,
    )
    changed = _changed(out, frame)
    left, top, right, bottom = _bbox(changed)
    assert left < 320 < right
    assert top < 180 < bottom
    panel = out[top + 1, (left + right) // 2].astype(int)
    assert panel.sum() < 3 * 200 - 150, "the panel darkens the frame"
    assert panel.sum() > 3 * 10, "the panel is semi-transparent, not opaque black"
    assert (out[top:bottom, left:right].min(axis=2) > 240).any(), "white text is drawn"
    tofu = draw_annotations(
        frame,
        FrameAnnotations(texts=(TextLabel((320, 180), chr(0xFFFF) * 4, anchor="mm", size=40),)),
        style,
    )
    assert not np.array_equal(out, tofu), "CJK glyphs differ from the missing-glyph box"


def test_skeleton_limbs_are_coloured_by_side_and_low_scores_hidden(style: Style) -> None:
    frame = _frame(colour=(30, 30, 30))
    kp = np.full((26, 3), np.nan)
    kp[18] = (320, 60, 0.9)  # neck
    kp[19] = (320, 180, 0.9)  # hip
    kp[5], kp[7], kp[9] = (250, 70, 0.9), (250, 140, 0.9), (250, 210, 0.1)  # left arm
    kp[6], kp[8], kp[10] = (390, 70, 0.9), (390, 140, 0.9), (390, 210, 0.9)  # right arm
    out = draw_annotations(frame, FrameAnnotations(skeletons=(Skeleton(kp, min_score=0.3),)), style)
    left_px = out[105, 250].astype(int)
    right_px = out[105, 390].astype(int)
    assert np.abs(left_px - np.array(style.skeleton_left)).sum() < 60
    assert np.abs(right_px - np.array(style.skeleton_right)).sum() < 60
    assert np.array_equal(out[190, 248:253], frame[190, 248:253]), "a 0.1-score wrist is hidden"
    assert not np.array_equal(out[190, 388:393], frame[190, 388:393])


def test_skeleton_skips_keypoints_outside_the_frame(style: Style) -> None:
    frame = _frame(colour=(30, 30, 30))
    kp = np.full((26, 3), np.nan)
    kp[17], kp[18] = (320, 120, 0.9), (320, 200, 0.9)  # head, neck
    kp[19] = (500, -150, 0.9)  # pelvis above the frame: the model's guess
    kp[6], kp[8] = (380, 210, 0.9), (380, 280, 0.9)  # right shoulder, elbow
    kp[10] = (660, 290, 0.9)  # right wrist beyond the right edge
    out = draw_annotations(frame, FrameAnnotations(skeletons=(Skeleton(kp),)), style)
    changed = _changed(out, frame)
    assert changed[160, 318:323].any(), "head to neck is drawn"
    assert changed[245, 378:383].any(), "the upper arm is drawn"
    assert not changed[:110, 330:].any(), "no line from the neck to the top edge"
    assert not changed[270:310, 400:].any(), "no forearm to the right edge"


def test_skeleton_of_a_person_cut_by_the_bottom_edge_has_no_lower_body(style: Style) -> None:
    frame = _frame(colour=(30, 30, 30))
    kp = np.full((26, 3), np.nan)
    kp[17], kp[18] = (320, 60, 0.9), (320, 120, 0.9)  # head, neck
    kp[5], kp[7] = (280, 130, 0.9), (270, 200, 0.9)  # left shoulder, elbow
    kp[19], kp[11], kp[13] = (320, 250, 0.9), (300, 250, 0.9), (300, 320, 0.9)  # pelvis, hip, knee
    kp[20] = (250, 70, 0.9)  # a "toe" found next to the head
    cut = (240.0, 40.0, 400.0, 357.0)  # ends 2.5 px above the bottom edge of 360 rows
    out = draw_annotations(frame, FrameAnnotations(skeletons=(Skeleton(kp, box=cut),)), style)
    changed = _changed(out, frame)
    assert changed[165, 270:281].any(), "the arm is drawn"
    assert changed[90, 318:323].any(), "head to neck is drawn"
    assert not changed[150:340, 290:330].any(), "no trunk, hip or knee"
    assert not changed[60:80, 240:260].any(), "no stray toe by the head"
    whole = (240.0, 40.0, 400.0, 330.0)  # 30 px above the edge: the legs are in the picture
    out = draw_annotations(frame, FrameAnnotations(skeletons=(Skeleton(kp, box=whole),)), style)
    assert _changed(out, frame)[260:310, 295:305].any(), "the leg is drawn"
    # A player on the ground lies flat: limbs are never dropped for their angle.
    flat = np.full((26, 3), np.nan)
    flat[17], flat[18], flat[19] = (150, 300, 0.9), (200, 300, 0.9), (330, 305, 0.9)
    flat[11], flat[13], flat[15] = (330, 300, 0.9), (420, 300, 0.9), (500, 305, 0.9)
    lying = Skeleton(flat, box=(130.0, 280.0, 520.0, 320.0))
    changed = _changed(draw_annotations(frame, FrameAnnotations(skeletons=(lying,)), style), frame)
    assert changed[296:309, 260].any(), "the trunk is drawn"
    assert changed[294:309, 460].any(), "the leg is drawn"
    with pytest.raises(RenderError, match="box"):
        Skeleton(kp, box=(10.0, 10.0, 5.0, 5.0))


def test_no_ground_ellipse_for_close_ups_or_boxes_at_the_bottom_edge(style: Style) -> None:
    frame = _frame()
    tall = PlayerMark((200.0, 20.0, 330.0, 220.0))  # taller than half of the 360 rows
    cut = PlayerMark((400.0, 250.0, 450.0, 359.0))  # ends at the bottom edge
    ordinary = PlayerMark((500.0, 200.0, 530.0, 300.0))
    out = draw_annotations(frame, FrameAnnotations(players=(tall, cut, ordinary)), style)
    changed = _changed(out, frame)
    assert not changed[228:242, 240:290].any(), "no ellipse under the close-up box"
    assert not changed[346:357, 412:438].any(), "no ellipse at the bottom edge"
    assert changed[303:306, 510:521].any(), "the ordinary player keeps the ellipse"
    assert changed[18:30, 196:212].any(), "the close-up keeps its corner brackets"
    ring = tuple((265.0 + 40.0 * math.cos(a), 230.0 + 8.0 * math.sin(a)) for a in range(0, 7))
    calibrated = replace(tall, ring=ring)
    out = draw_annotations(frame, FrameAnnotations(players=(calibrated,)), style)
    assert _changed(out, frame)[225:241, 225:306].any(), "a calibrated ring is still drawn"


def test_ball_trail_fades_towards_the_past(style: Style) -> None:
    frame = _frame(colour=(20, 20, 20))
    trail = tuple((100.0 + 20 * i, 200.0) for i in range(12))
    out = draw_annotations(
        frame, FrameAnnotations(ball=BallMark((360.0, 200.0), 6.0, trail=trail)), style
    )
    old = int(out[200, 112].astype(int).sum() - frame[200, 112].astype(int).sum())
    new = int(out[200, 320].astype(int).sum() - frame[200, 320].astype(int).sum())
    assert new > old + 100
    assert _changed(out, frame)[190:211, 345:376].any(), "a ring is drawn around the ball"


def test_minimap_places_team_dots_in_soccernet_metres(style: Style) -> None:
    frame = _frame(1280, 720, colour=(0, 0, 0))
    minimap = Minimap(players=((0.0, -30.0, "left"), (40.0, 30.0, "right")), ball=(-20.0, 0.0))
    out = draw_annotations(frame, FrameAnnotations(minimap=minimap), style)
    geometry = minimap_geometry(1280, 720, minimap, style)
    top_u, top_v = geometry.pixel(0.0, -30.0)
    low_u, low_v = geometry.pixel(40.0, 30.0)
    assert top_v < low_v, "the far touchline (y = -34) is at the top"
    assert low_u > top_u
    for (u, v), team in (((top_u, top_v), "left"), ((low_u, low_v), "right")):
        pixel = out[int(round(v)), int(round(u))].astype(int)
        assert np.abs(pixel - np.array(style.team_colour(team))).sum() < 45
    ball_u, ball_v = geometry.pixel(-20.0, 0.0)
    assert out[int(round(ball_v)), int(round(ball_u))].min() > 230
    x0, y0, width, height = geometry.panel
    assert x0 + width <= 1280
    assert y0 + height <= 720
    assert x0 > 640
    assert y0 > 360


def test_minimap_draws_no_ball_far_off_the_pitch(style: Style) -> None:
    frame = _frame(1280, 720, colour=(0, 0, 0))
    for ball, drawn in (
        ((10.0, -34.8), True),
        ((10.0, -35.5), False),
        ((53.4, 0.0), True),
        ((54.0, 5.0), False),
    ):
        minimap = Minimap(ball=ball)
        out = draw_annotations(frame, FrameAnnotations(minimap=minimap), style)
        u, v = minimap_geometry(1280, 720, minimap, style).pixel(*ball)
        assert (out[int(round(v)), int(round(u))].min() > 230) == drawn, ball


def test_banner_badges_card_and_watermark_stay_in_their_areas(style: Style) -> None:
    frame = _frame(1280, 720)
    annotations = FrameAnnotations(
        banner=Banner("高光 2 · 示例争抢", "00:04.00", "帧 100–164"),
        badges=(Badge("未验证"), "仅画面平面"),
        card=InfoCard("定格数据", (CardRow("峰值速度", "27.4", "km/h", ("未验证",)),)),
    )
    out = draw_annotations(
        frame, annotations, Style(font=style.font, font_bold=style.font_bold, watermark="私有素材")
    )
    changed = _changed(out, frame)
    assert changed[500:700, 40:500].any(), "lower-third banner"
    assert changed[:40, 500:780].any(), "watermark at the top centre"
    assert changed[150:600, 800:1240].any(), "card on the right"
    assert not changed[100:400, 100:500].any(), "nothing in the upper left"


@pytest.mark.parametrize("corner", ["br", "tr"])
@pytest.mark.parametrize("size", [(1920, 1080), (1280, 600)], ids=["1080p", "600p"])
def test_card_on_the_minimap_side_stays_clear_of_it(
    style: Style, monkeypatch: pytest.MonkeyPatch, corner: str, size: tuple[int, int]
) -> None:
    width, height = size
    framed = Style(font=style.font, font_bold=style.font_bold, minimap_corner=corner)  # type: ignore[arg-type]
    frame = _frame(width, height)
    rows = tuple(CardRow(f"指标 {i}", f"{i}.0", "m", ("未验证",)) for i in range(6))
    card = InfoCard("关键帧 · 第 1 帧", rows, footer="数值为自动估计", side="right")
    minimap = Minimap(players=((0.0, 0.0, "left"),), title="俯视图")
    annotations = FrameAnnotations(minimap=minimap, card=card)
    only_map = draw_annotations(frame, FrameAnnotations(minimap=minimap), framed)
    drawn = _record_drawn_texts(monkeypatch)
    both = draw_annotations(frame, annotations, framed)
    x0, y0, _, panel_h = minimap_geometry(width, height, minimap, framed).panel
    columns = slice(int(x0) - 150, int(x0) - 2)
    beside = (slice(int(y0) - 4, int(y0 + panel_h) + 4), columns)
    assert not (both[beside] != only_map[beside]).any(), "the card keeps clear of the minimap"
    if corner == "br":  # the card ends above the minimap ...
        card_rows = slice(int(0.28 * height), int(y0) - 20)
    else:  # ... or starts below it
        card_rows = slice(int(y0 + panel_h) + 20, int(0.9 * height))
    assert (both[card_rows, columns] != only_map[card_rows, columns]).any(), "the card is drawn"
    # The card leaves out its last row to fit, and the manifest texts are what is drawn.
    labels = [text for text in drawn if text.startswith("指标")]
    assert labels == [f"指标 {i}" for i in range(5)]
    footer = "数值为自动估计；另有 1 项见报告"
    assert footer in drawn
    listed = annotation_texts(annotations, (width, height), framed)
    assert [text for kind, text, _ in listed if kind == "card_row"] == [
        f"指标 {i} {i}.0 m" for i in range(5)
    ]
    assert [text for kind, text, _ in listed if kind == "card_footer"] == [footer]
    assert sum(kind == "card_row" for kind, _, _ in annotation_texts(annotations)) == 6, (
        "without a frame size every row is listed"
    )


def test_card_leaves_out_rows_below_the_title_safe_area(
    style: Style, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame(1920, 1080)
    rows = tuple(CardRow(f"指标 {i}", f"{i}.0", "m") for i in range(12))
    annotations = FrameAnnotations(card=InfoCard("关键帧", rows))
    drawn = _record_drawn_texts(monkeypatch)
    out = draw_annotations(frame, annotations, style)
    labels = [text for text in drawn if text.startswith("指标")]
    assert 6 <= len(labels) < 12
    footer = f"另有 {12 - len(labels)} 项见报告"
    assert footer in drawn
    listed = annotation_texts(annotations, (1920, 1080), style)
    assert [text for kind, text, _ in listed if kind == "card_row"] == [
        f"指标 {i} {i}.0 m" for i in range(len(labels))
    ]
    assert [text for kind, text, _ in listed if kind == "card_footer"] == [footer]
    assert not _changed(out, frame)[1080 - 50 :].any(), "nothing below the title-safe area"
    with pytest.raises(RenderError, match="frame_size"):
        annotation_texts(annotations, (1920, 0), style)


def test_card_degree_and_percent_follow_the_number_without_a_gap(style: Style) -> None:
    frame = _frame(1920, 1080, colour=(40, 40, 40))

    def draw(value: str, unit: str) -> np.ndarray:
        card = InfoCard("关键帧", (CardRow("角度", value, unit),), side="left")
        return draw_annotations(frame, FrameAnnotations(card=card), style)

    for value, unit in (("93", "°"), ("40", "%")):
        same = np.array_equal(draw(value, unit), draw(value + unit, ""))
        assert same, f"{value} and {unit} are drawn as the one number {value}{unit}"
    assert not np.array_equal(draw("27.4", "m"), draw("27.4m", "")), "other units keep a gap"
    card = InfoCard("卡", (CardRow("角度", "93", "°"), CardRow("速度", "27.4", "km/h")))
    texts = [text for kind, text, _ in annotation_texts(FrameAnnotations(card=card))]
    assert texts == ["卡", "角度 93°", "速度 27.4 km/h"]


def _offset_box(mask: np.ndarray, x0: int = 0, y0: int = 0) -> tuple[int, int, int, int]:
    """The bounding box of ``mask`` (a slice starting at x0, y0) in frame pixels."""
    left, top, right, bottom = _bbox(mask)
    return left + x0, top + y0, right + x0, bottom + y0


def _under(region: tuple[float, float, float, float], label: str, ident: str) -> PlayerMark:
    """A player on a 720p frame whose label point (6 px at 1080p above the top of its box)
    is the centre of ``region``."""
    u, v = (region[0] + region[2]) / 2, (region[1] + region[3]) / 2
    top = v + 6 * 720 / 1080
    return PlayerMark(
        (u - 10, top, u + 10, top + 60), team="right", label=label,
        badges=(Badge("未验证", "unverified", compact=True),), evidence_ids=(ident,),
    )  # fmt: skip


def test_labels_under_the_card_are_left_out(style: Style, monkeypatch: pytest.MonkeyPatch) -> None:
    # The freeze card's panel is translucent: a label under it would be legible through it
    # (a player's speed under the card's title, say). Such labels are not drawn, and the
    # manifest does not list them.
    frame = _frame(1280, 720)
    rows = tuple(CardRow(f"指标 {i}", f"{i}.0", "m", ("未验证",), (f"ev-c{i}",)) for i in range(4))
    card = InfoCard("关键帧", rows, side="right")
    region = _bbox(_changed(draw_annotations(frame, FrameAnnotations(card=card), style), frame))
    hidden = _under(region, "轨迹 2 · 18.6 km/h", "ev-2")
    shown = PlayerMark((300.0, 300.0, 330.0, 390.0), team="left", label="轨迹 1", emphasis="focus")
    centre = ((region[0] + region[2]) / 2, (region[1] + region[3]) / 2 + 40)
    line = LinkLine(
        (centre[0] - 60, centre[1]), (centre[0] + 60, centre[1]), "最近对手 · 4.2 m",
        badges=(Badge("未验证", "unverified"),), evidence_ids=("ev-l",),
    )  # fmt: skip
    annotations = FrameAnnotations(players=(hidden, shown), lines=(line,), card=card)
    drawn = _record_drawn_texts(monkeypatch)
    out = draw_annotations(frame, annotations, style)
    assert "轨迹 1" in drawn
    assert "轨迹 2 · 18.6 km/h" not in drawn, "the label under the card is left out"
    assert "最近对手 · 4.2 m" not in drawn
    unlabelled = FrameAnnotations(
        players=(replace(hidden, label=None, badges=()), shown),
        lines=(replace(line, label=None, badges=()),),
        card=card,
    )
    assert np.array_equal(out, draw_annotations(frame, unlabelled, style)), "nothing shows through"
    listed = annotation_texts(annotations, (1280, 720), style)
    assert ("player_label", "轨迹 1", ()) in listed
    assert not [row for row in listed if row[2] in (("ev-2",), ("ev-l",))], "nor its badge"
    # Without the card (the frames before the freeze) the labels are drawn and listed.
    plain = replace(annotations, card=None)
    listed = annotation_texts(plain, (1280, 720), style)
    assert ("player_label", "轨迹 2 · 18.6 km/h", ("ev-2",)) in listed
    assert ("badge", "未验证", ("ev-2",)) in listed
    assert ("line_label", "最近对手 · 4.2 m", ("ev-l",)) in listed
    drawn.clear()
    draw_annotations(frame, plain, style)
    assert {"轨迹 2 · 18.6 km/h", "最近对手 · 4.2 m"} <= set(drawn)


def test_labels_keep_clear_of_the_minimap_banner_badges_and_watermark(
    style: Style, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The minimap, the banner with its badge row and the watermark are drawn over the
    # labels, on every frame. A label whose point lies under one of them is left out, and
    # so is a free text under one; a label that only reaches into one moves clear of it.
    width, height = 1280, 720
    framed = Style(font=style.font, font_bold=style.font_bold, watermark="私有素材 · 禁止分享")
    frame = _frame(width, height)
    minimap = Minimap(title="俯视图")
    banner = Banner("高光 · 示例", "00:01.00", "副标题")
    badges = (Badge("未验证"), Badge("仅画面平面"))
    panels = FrameAnnotations(minimap=minimap, banner=banner, badges=badges)
    with_panels = draw_annotations(frame, panels, framed)
    only_banner = draw_annotations(frame, FrameAnnotations(banner=banner), framed)
    with_badges = draw_annotations(frame, FrameAnnotations(banner=banner, badges=badges), framed)
    mx, my, mw, mh = minimap_geometry(width, height, minimap, framed).panel
    regions = {
        "minimap": (mx, my, mx + mw, my + mh),
        "banner": _offset_box(_changed(only_banner, frame)[height // 2 :], 0, height // 2),
        "badges": _offset_box(_changed(with_badges, only_banner)[height // 2 :], 0, height // 2),
        "watermark": _offset_box(_changed(with_panels, frame)[: height // 6]),
    }
    hidden = tuple(
        _under(box, f"轨迹 {n} · 9.9 km/h", f"ev-{n}") for n, box in enumerate(regions.values())
    )
    beside = PlayerMark((mx - 30, my + 34, mx - 10, my + 94), team="left", label="轨迹 9")
    under_text = TextLabel((mx + mw / 2, my + mh / 2), "说明 · 3.1 m", evidence_ids=("ev-t",))
    free_text = TextLabel((320, 200), "说明 · 1.0 m", evidence_ids=("ev-f",))
    annotations = replace(panels, players=(*hidden, beside), texts=(under_text, free_text))
    drawn = _record_drawn_texts(monkeypatch)
    out = draw_annotations(frame, annotations, framed)
    assert "轨迹 9" in drawn
    assert "说明 · 1.0 m" in drawn
    assert not [text for text in drawn if "9.9 km/h" in text or text == "说明 · 3.1 m"]
    without = replace(
        annotations,
        players=tuple(replace(mark, label=None, badges=()) for mark in annotations.players),
        texts=(free_text,),
    )
    reference = draw_annotations(frame, without, framed)
    for name, (x0, y0, x1, y1) in regions.items():
        window = (slice(int(y0), int(y1) + 1), slice(int(x0), int(x1) + 1))
        assert np.array_equal(out[window], reference[window]), f"no text under the {name}"
    listed = annotation_texts(annotations, (width, height), framed)
    assert [row for row in listed if row[0] in ("player_label", "text")] == [
        ("player_label", "轨迹 9", ()), ("text", "说明 · 1.0 m", ("ev-f",)),
    ]  # fmt: skip
    assert not [row for row in listed if row[0] == "badge" and row[2]], "nor their badges"
    assert len(annotation_texts(annotations)) > len(listed), "without a frame size, everything"


def test_card_keeps_clear_of_the_banner_and_its_badges(
    style: Style, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A card on the banner's side ends above the banner's badge row, as it ends above the
    # minimap: the banner, drawn later, would otherwise cover rows that show through it.
    frame = _frame(1280, 720)
    rows = tuple(CardRow(f"指标 {i}", f"{i}.0", "m") for i in range(8))
    banner = Banner("高光 · 示例", "00:01.00", "副标题")
    badges = (Badge("未验证"), Badge("定格"))
    annotations = FrameAnnotations(
        banner=banner, badges=badges, card=InfoCard("关键帧", rows, side="left")
    )
    only_banner = draw_annotations(frame, replace(annotations, card=None), style)
    left, top, right, bottom = _bbox(_changed(only_banner, frame))
    drawn = _record_drawn_texts(monkeypatch)
    both = draw_annotations(frame, annotations, style)
    window = (slice(top - 4, bottom + 1), slice(left, right + 1))
    assert np.array_equal(both[window], only_banner[window]), "the card ends above the badges"
    assert (both[: top - 4, left:right] != only_banner[: top - 4, left:right]).any(), "a card"
    labels = [text for text in drawn if text.startswith("指标")]
    assert 2 <= len(labels) < 8
    footer = f"另有 {8 - len(labels)} 项见报告"
    assert footer in drawn
    listed = annotation_texts(annotations, (1280, 720), style)
    assert [text for kind, text, _ in listed if kind == "card_row"] == [
        f"指标 {i} {i}.0 m" for i in range(len(labels))
    ]
    assert [text for kind, text, _ in listed if kind == "card_footer"] == [footer]


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080)], ids=["720p", "1080p"])
def test_line_labels_stay_short_on_small_frames(style: Style, size: tuple[int, int]) -> None:
    # A line's label says what it links (争抢, 最近对手) before the distance; with its
    # 未验证 chip it stays narrower than a sixth of the frame, at 720p as at 1080p.
    width, height = size
    frame = _frame(width, height)
    u, v = width / 2, height / 2
    for text in ("最近对手 · 12.8 m", "争抢 · 1.6 m", "争抢"):
        line = LinkLine((u - 4, v), (u + 4, v), text, badges=(Badge("未验证", "unverified"),))
        out = draw_annotations(frame, FrameAnnotations(lines=(line,)), style)
        left, _, right, _ = _bbox(_changed(out, frame))
        assert right - left + 1 <= width / 6, text


def test_dim_with_spotlight_darkens_only_outside_it(style: Style) -> None:
    frame = _frame(colour=(200, 200, 200))
    out = draw_annotations(frame, FrameAnnotations(dim=0.5, spotlight=(320.0, 180.0, 80.0)), style)
    assert abs(int(out[10, 10, 0]) - 100) <= 2
    assert int(out[180, 320, 0]) >= 195


def test_arrow_and_link_line_draw_between_their_end_points(style: Style) -> None:
    frame = _frame()
    out = draw_annotations(
        frame,
        FrameAnnotations(
            arrows=(Arrow((100, 300), (300, 300)),),
            lines=(LinkLine((400, 100), (600, 100), "4.2 m", dashed=False),),
        ),
        style,
    )
    changed = _changed(out, frame)
    assert changed[298:303, 150].any()
    assert changed[290:311, 295:301].any()
    assert changed[98:103, 450].any()
    assert not changed[200, :].any()


def _record_text_spots(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, float, float]]:
    """From now on, every string gbf.render draws with its anchor point in frame pixels
    (the canvas point divided by the supersampling factor)."""
    spots: list[tuple[str, float, float]] = []
    draw_text = render_module._text

    def recording(c: Any, xy: tuple[float, float], text: str, *args: Any, **kwargs: Any) -> None:
        spots.append((text, xy[0] / c.s, xy[1] / c.s))
        draw_text(c, xy, text, *args, **kwargs)

    monkeypatch.setattr(render_module, "_text", recording)
    return spots


def test_arrow_label_sits_past_the_head_clear_of_the_player(
    style: Style, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An arrow on the grass from a player's feet: its label sits just past the head, where
    # it hides neither the arrow nor the player. An arrow that points up the image points
    # into the player, so its label goes below the arrow's start instead.
    frame = _frame()
    body = (300.0, 150.0, 330.0, 230.0)
    player = PlayerMark(body, team="left")
    chip = (Badge("未验证", "unverified"),)
    spots = _record_text_spots(monkeypatch)
    sideways = Arrow((315.0, 230.0), (380.0, 236.0), "朝向（3D 参考）", badges=chip)
    draw_annotations(frame, FrameAnnotations(players=(player,), arrows=(sideways,)), style)
    ((x, y),) = [(u, v) for text, u, v in spots if text == "朝向（3D 参考）"]
    assert x > 380.0, "past the head, clear of the player's box"
    assert 236.0 - 10 < y < 236.0 + 20, "beside the head, not above the player"
    spots.clear()
    upward = Arrow((315.0, 230.0), (318.0, 205.0), "朝向（3D 参考）", badges=chip)
    draw_annotations(frame, FrameAnnotations(players=(player,), arrows=(upward,)), style)
    ((x, y),) = [(u, v) for text, u, v in spots if text == "朝向（3D 参考）"]
    assert y > 230.0, "below the arrow's start, not over the player"


def test_an_arrow_label_keeps_off_players_where_it_can_and_off_labels_always(
    style: Style,
) -> None:
    # An ordinary label that finds no free spot overlaps what is there; an arrow's label
    # (strict) may cover a player when it must, but is left out rather than drawn over
    # another label.
    g = render_module._geometry(640, 360, style)
    everywhere = [(0.0, 0.0, float(g.canvas_w), float(g.canvas_h))]
    spot = (g, (300.0, 300.0), 250.0, 290.0, 100.0, 20.0, "centre")
    assert render_module._free_spot(*spot, everywhere, []) == (250.0, 290.0)
    assert render_module._free_spot(*spot, everywhere, [], strict=True) is None
    assert render_module._free_spot(*spot, [], [], everywhere, strict=True) == (250.0, 290.0)
    assert render_module._free_spot(*spot, [], [], [(240.0, 285.0, 360.0, 309.0)]) == (
        250.0,
        290.0 + 20.0 + g.k(3),
    ), "the next spot down, off the player"


def test_a_label_with_no_room_above_or_below_sits_beside(style: Style) -> None:
    # Every spot above and below the player's head lies on a player the label should not
    # hide: the label sits beside the head instead, right of the player's box.
    g = render_module._geometry(640, 360, style)
    gap = g.k(3)
    column = [(240.0, 100.0, 330.0, 330.0)]
    spot = (g, (300.0, 300.0), 250.0, 280.0, 100.0, 20.0, "above")
    assert render_module._free_spot(*spot, [], [], column, beside=(270.0, 330.0)) == (
        330.0 + 2 * gap,
        280.0,
    )
    # an arrow's label (strict) keeps to its own candidates
    assert render_module._free_spot(*spot, [], [], column, strict=True, beside=(270.0, 330.0)) == (
        250.0,
        280.0,
    )


def test_labels_find_room_off_the_involved_players_in_a_crowd(style: Style) -> None:
    # Six involved players stand close together, each with a long label: every label is
    # drawn clear of every one of their bodies and of the other labels (farther out where
    # nothing nearer is free), and each moved one points back to its player.
    g = render_module._geometry(1920, 1080, style)
    marks = []
    for k in range(6):
        x, y = 860.0 + (k % 3) * 70.0, 420.0 + (k // 3) * 120.0
        marks.append(
            PlayerMark(
                (x, y, x + 40.0, y + 110.0),
                team="left" if k % 2 else "right",
                label=f"轨迹 5{k} · 1{k}.8 km/h",
                emphasis="focus" if k == 0 else "normal",
                badges=(Badge("未验证", "unverified", compact=k > 0),),
            )
        )
    layout = render_module._layout(g, FrameAnnotations(players=tuple(marks)))
    bodies = [(*g.p(m.box[0], m.box[1]), *g.p(m.box[2], m.box[3])) for m in marks]
    labels = [label for label in layout.players if label is not None]
    boxes = [label.box for label in labels if label.box is not None]
    assert len(boxes) == len(marks), "every label is drawn"
    assert [box for box in boxes if render_module._overlaps(box, bodies)] == []
    assert not any(render_module._overlaps(a, boxes[i + 1 :]) for i, a in enumerate(boxes))
    for label, mark in zip(labels, marks, strict=True):
        x, y = render_module._clamp_spot(g, *_wanted_spot(g, label, mark))
        spot = label.spot
        assert spot is not None
        moved = math.hypot(spot[0] - x, spot[1] - y) >= label.height
        assert (label.leader is not None) is moved, "a moved label points back"


def _wanted_spot(g: Any, label: Any, mark: PlayerMark) -> tuple[float, float, float, float]:
    """(x0, y0, width, height) of a player label's wanted spot: centred above the head."""
    u, v = g.p((mark.box[0] + mark.box[2]) / 2.0, mark.box[1] - 6.0 * g.ui)
    return u - label.width / 2, v - label.height, label.width, label.height


def test_an_arrow_is_drawn_only_with_its_label(style: Style) -> None:
    # The label says what the arrow shows (the facing arrow: a 3D reference). An arrow whose
    # label is left out, its head under the minimap here, is left out with it; an arrow
    # without a label is drawn as before.
    width, height = 1280, 720
    frame = _frame(width, height)
    minimap = Minimap(title="俯视图")
    mx, my, _mw, mh = minimap_geometry(width, height, minimap, style).panel
    start, head = (mx - 60.0, my + mh / 2), (mx + 40.0, my + mh / 2)
    arrow = Arrow(
        start, head, "朝向（3D 参考）", badges=(Badge("未验证", "unverified"),),
        evidence_ids=("lift3d.parquet#frame=3&track_id=1",),
    )  # fmt: skip
    plain = draw_annotations(frame, FrameAnnotations(minimap=minimap), style)
    annotations = FrameAnnotations(arrows=(arrow,), minimap=minimap)
    assert np.array_equal(draw_annotations(frame, annotations, style), plain)
    listed = annotation_texts(annotations, (width, height), style)
    assert not [row for row in listed if row[2] == ("lift3d.parquet#frame=3&track_id=1",)]
    bare = replace(annotations, arrows=(replace(arrow, label=None, badges=()),))
    assert not np.array_equal(draw_annotations(frame, bare, style), plain)


def test_labels_keep_off_the_involved_players_and_moved_ones_point_back(style: Style) -> None:
    # The focus player's label would lie on the head of the involved player standing right
    # behind it: it keeps clear of that player's body, and a thin leader line runs from the
    # moved label back to its own player, so that it is never read as the other's. The body
    # of a context player does not move a label.
    frame = _frame(1920, 1080)
    focus = PlayerMark(
        (900.0, 500.0, 960.0, 700.0), team="left", label="左侧球队 · 10号", emphasis="focus"
    )
    behind = PlayerMark((905.0, 380.0, 965.0, 560.0), team="right")
    g = render_module._geometry(1920, 1080, style)
    label = render_module._layout(g, FrameAnnotations(players=(focus, behind))).players[0]
    assert label is not None
    assert label.box is not None
    body = tuple(v * g.s for v in behind.box)
    assert not render_module._overlaps(label.box, [body]), "clear of the involved player"
    assert label.leader == pytest.approx(g.p(930.0, 500.0 - 6.0 * g.ui))
    ignored = FrameAnnotations(players=(focus, replace(behind, emphasis="context")))
    in_place = render_module._layout(g, ignored).players[0]
    assert in_place is not None
    assert in_place.leader is None, "in place: no leader"
    # the leader runs down through the other player's box to the focus player's head
    drawn = draw_annotations(frame, FrameAnnotations(players=(focus, behind)), style)
    unlabelled = FrameAnnotations(players=(replace(focus, label=None), behind))
    plain = draw_annotations(frame, unlabelled, style)
    assert (drawn[440:470, 929:932] != plain[440:470, 929:932]).any()
    # A duel line's label at the pair's feet keeps off both players' legs too.
    pair = (focus, replace(behind, box=(965.0, 510.0, 1025.0, 705.0)))
    line = LinkLine((930.0, 700.0), (995.0, 705.0), "争抢 · 1.6 m", badges=(Badge("未验证"),))
    (line_label,) = render_module._layout(g, FrameAnnotations(players=pair, lines=(line,))).lines
    assert line_label is not None
    assert line_label.box is not None
    bodies = [tuple(v * g.s for v in mark.box) for mark in pair]
    assert not render_module._overlaps(line_label.box, bodies)


def test_an_arrow_label_moves_off_a_player_standing_past_the_head(
    style: Style, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End to end: the facing arrow's label sits past its head; a player standing there is
    # not covered while a spot clear of it is free (the players' boxes reach the label).
    frame = _frame(1920, 1080)
    other = PlayerMark((780.0, 400.0, 900.0, 560.0), team="right")
    arrow = Arrow(
        (600.0, 480.0), (700.0, 480.0), FACING_LABEL_ZH, badges=(Badge("未验证", "unverified"),)
    )
    spots = _record_text_spots(monkeypatch)
    draw_annotations(frame, FrameAnnotations(players=(other,), arrows=(arrow,)), style)
    ((_x, y),) = [(u, v) for text, u, v in spots if text == FACING_LABEL_ZH]
    assert y > 560.0, "below the player, not over it"


def test_an_arrow_label_never_lies_over_another_label_nor_does_its_badge_stay(
    style: Style,
) -> None:
    # End to end: every spot of the facing arrow's label is clamped onto the spot of another
    # label (its head lies below the frame's title-safe area): the arrow is left out with
    # its label rather than drawn over that label, and the frame's 未验证, which only the
    # arrow called for (arrow_badges), leaves the badge row with it.
    frame = _frame(1920, 1080)
    unverified = Badge("未验证", "unverified")
    arrow = Arrow(
        (960.0, 1000.0), (960.0, 1100.0), FACING_LABEL_ZH, badges=(unverified,),
        evidence_ids=("lift3d.parquet#frame=3&track_id=1",),
    )  # fmt: skip
    blocker = PlayerMark((940.0, 1059.0, 980.0, 1075.0), label="轨迹 7", emphasis="context")
    banner = Banner("高光 · 示例", "00:01.00", "副标题")
    blocked = FrameAnnotations(
        players=(blocker,), arrows=(arrow,), banner=banner, arrow_badges=(unverified,)
    )
    listed = annotation_texts(blocked, (1920, 1080), style)
    assert not [row for row in listed if row[0] == "arrow_label"], "left out, not drawn over"
    assert ("player_label", "轨迹 7", ()) in listed
    assert ("badge", "未验证", ()) not in listed
    without = replace(blocked, arrows=(), arrow_badges=())
    assert np.array_equal(
        draw_annotations(frame, blocked, style), draw_annotations(frame, without, style)
    ), "neither the arrow nor the badge is drawn"
    free = replace(blocked, players=(replace(blocker, label=None),))
    listed = annotation_texts(free, (1920, 1080), style)
    assert ("arrow_label", FACING_LABEL_ZH, arrow.evidence_ids) in listed
    assert [text for kind, text, ids in listed if kind == "badge" and not ids] == ["未验证"]


def test_drawing_is_deterministic(style: Style) -> None:
    frame = _frame()
    annotations = FrameAnnotations(
        players=(PlayerMark((300, 120, 330, 220), team="left", label="#9", emphasis="focus"),),
        ball=BallMark((350, 220), 5, trail=((300, 230), (320, 225))),
        banner=Banner("测试", "00:00.40"),
    )
    assert np.array_equal(
        draw_annotations(frame, annotations, style), draw_annotations(frame, annotations, style)
    )


def test_invalid_annotations_list_every_problem() -> None:
    with pytest.raises(RenderError) as excinfo:
        PlayerMark((10, 10, 5, 5), colour=(300, 0, 0), emphasis="loud")  # type: ignore[arg-type]
    assert len(excinfo.value.problems) == 3
    with pytest.raises(RenderError, match="shape"):
        Skeleton(np.zeros((17, 3)))
    with pytest.raises(RenderError, match="dim"):
        FrameAnnotations(dim=1.5)
    with pytest.raises(RenderError) as excinfo:
        Style(supersample=0, minimap_corner="middle", safe_margin=0.5)  # type: ignore[arg-type]
    assert len(excinfo.value.problems) == 3
    # The banner and its badges would cover the minimap's title and note in a shared corner.
    for corner, position in (("bl", "bottom_left"), ("tl", "top_left")):
        with pytest.raises(RenderError, match="share a corner"):
            Style(minimap_corner=corner, banner_position=position)  # type: ignore[arg-type]
    assert Style(minimap_corner="bl", banner_position="top_left").minimap_corner == "bl"
    assert Style(minimap_corner="tl").minimap_corner == "tl"


def test_invalid_frames_are_rejected(style: Style) -> None:
    with pytest.raises(RenderError, match="dtype"):
        draw_annotations(np.zeros((10, 10, 3), np.float32), FrameAnnotations(), style)
    with pytest.raises(RenderError, match="shape"):
        draw_annotations(np.zeros((10, 10), np.uint8), FrameAnnotations(), style)


def test_badges_texts_and_merge() -> None:
    assert Badge.of("未验证").kind == "unverified"
    assert Badge.of("仅画面平面").kind == "image_plane"
    assert Badge.of("3D 参考").kind == "reference_3d"
    assert Style().badge_colour("reference_3d") not in (
        Style().badge_colour(kind) for kind in ("unverified", "image_plane", "info")
    ), "a 3D reference has a dot colour of its own"
    assert Badge.of("不可用：无标定").kind == "unavailable"
    assert Badge.of("慢放 ×4").kind == "info"
    base = FrameAnnotations(
        players=(PlayerMark((0, 0, 10, 10), label="7.6 m/s", evidence_ids=("ev-1",)),), dim=0.2
    )
    extra = FrameAnnotations(
        card=InfoCard("卡", (CardRow("速度", "27.4", "km/h", (), ("ev-2",)),)), dim=0.5
    )
    merged = merge_annotations(base, extra)
    assert merged.dim == 0.5
    assert merged.card is extra.card
    assert merged.players == base.players
    texts = annotation_texts(merged)
    assert ("player_label", "7.6 m/s", ("ev-1",)) in texts
    assert ("card_row", "速度 27.4 km/h", ("ev-2",)) in texts


def test_save_png_round_trips(style: Style, unicode_tmp_path: Path) -> None:
    frame = draw_annotations(
        _frame(160, 90), FrameAnnotations(texts=(TextLabel((80, 45), "中"),)), style
    )
    path = unicode_tmp_path / "关键帧 1.png"
    digest = save_png(path, frame)
    assert digest == sha256_file(path)
    assert np.array_equal(read_image_rgb(path), frame)


# ---------------------------------------------------------------------------- geometry
def test_ground_circle_projects_a_metre_ring_on_the_pitch() -> None:
    camera = synthetic_broadcast_camera(1920, 1080)
    ring = ground_circle_px(camera, (5.0, 3.0), 1.0, n=36)
    assert ring.shape == (36, 2)
    assert np.isfinite(ring).all()
    back, valid = pixel_to_ground(camera, ring)
    assert valid.all()
    distances = np.hypot(back[:, 0] - 5.0, back[:, 1] - 3.0)
    assert np.allclose(distances, 1.0, atol=1e-6)
    centre, _, _ = project(camera, np.array([5.0, 3.0, 0.0]))
    assert abs(ring[:, 0].mean() - centre[0]) < 5
    assert abs(ring[:, 1].mean() - centre[1]) < 5


def test_camera_footprint_is_a_clipped_polygon_around_the_view() -> None:
    camera = synthetic_broadcast_camera(1920, 1080)
    polygon = camera_footprint(camera, limit=(60.0, 40.0))
    assert len(polygon) >= 4
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    assert max(map(abs, xs)) <= 60.0 + 1e-9
    assert max(map(abs, ys)) <= 40.0 + 1e-9
    centre_uv, _, _ = project(camera, np.array([0.0, 0.0, 0.0]))
    if 0 <= centre_uv[0] < camera.width and 0 <= centre_uv[1] < camera.height:
        inside = False  # even-odd ray casting
        for (xa, ya), (xb, yb) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
            if (ya > 0) != (yb > 0) and 0 < xa + (0 - ya) * (xb - xa) / (yb - ya):
                inside = not inside
        assert inside


def test_format_clock_uses_hundredths_and_exact_rates() -> None:
    assert format_clock(0) == "00:00.00"
    assert format_clock(476, "25/1") == "00:19.04"
    assert format_clock(1800, "25/1") == "01:12.00"
    assert format_clock(30, "30000/1001") == "00:01.00"
    assert format_clock(29, "30000/1001") == "00:00.96"
    with pytest.raises(RenderError):
        format_clock(-1)


# ---------------------------------------------------------------------------- video
def _frames(n: int, width: int = 64, height: int = 48) -> list[np.ndarray]:
    frames = []
    for k in range(n):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, :, 1] = 90
        frame[:, (3 * k) % width] = 255
        frames.append(frame)
    return frames


def _audio(path: Path, expression: str, seconds: float, ffmpeg_bin: str) -> Path:
    """A synthetic stereo AAC file from an aevalsrc expression of t (seconds)."""
    subprocess.run(
        [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"aevalsrc=exprs='{expression}':s=48000:d={seconds}",
            "-ac", "2", "-c:a", "aac", "-b:a", "192k", media_arg(path),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return path


def _pcm(path: Path, ffmpeg_bin: str) -> np.ndarray:
    proc = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-nostdin", "-i", media_arg(path), "-map", "0:a:0", "-ac", "1",
         "-ar", "48000", "-f", "f32le", "pipe:1"],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _rms(samples: np.ndarray, start_s: float, end_s: float) -> float:
    part = samples[int(start_s * 48000) : int(end_s * 48000)]
    return float(np.sqrt(np.mean(np.square(part.astype(np.float64)))))


def test_video_writer_writes_exactly_the_frames_given(unicode_tmp_path: Path) -> None:
    out = unicode_tmp_path / "片段 1.mp4"
    result = write_video(out, _frames(13), "25/1")
    facts = probe(out)
    assert result.frames == facts.nb_frames_decoded == 13
    assert facts.is_cfr_at("25/1")
    assert facts.codec_name == "h264"
    assert facts.pix_fmt == "yuv420p"
    assert result.audio is None
    assert result.sha256 == sha256_file(out)
    assert not (unicode_tmp_path / "片段 1.mp4.partial").exists()
    manifest = json.dumps(result.manifest(), ensure_ascii=False)
    assert str(unicode_tmp_path) not in manifest
    assert not re.search(r"[A-Za-z]:[/\\]", manifest)
    decoded = [frame for _, frame in iter_frames(out)]
    assert len(decoded) == 13
    assert np.abs(decoded[5].astype(int) - _frames(13)[5].astype(int)).mean() < 3


def test_video_writer_repeat_and_colour_tags(unicode_tmp_path: Path, ffprobe_bin: str) -> None:
    out = unicode_tmp_path / "repeat.mp4"
    frames = _frames(3)
    with VideoWriter(out, 64, 48, "25/1") as writer:
        writer.write(frames[0])
        writer.write(frames[1], repeat=4)
        writer.write(frames[2], repeat=2)
    assert writer.result is not None
    assert writer.result.frames == 7
    tags = subprocess.run(
        [ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=color_space,color_primaries,color_transfer,color_range", "-of", "json",
         media_arg(out)],
        check=True,
        capture_output=True,
    )  # fmt: skip
    stream = json.loads(tags.stdout)["streams"][0]
    assert stream == {
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
    }


def test_video_writer_muxes_a_time_range_of_the_audio(
    unicode_tmp_path: Path, ffmpeg_bin: str
) -> None:
    # a 1 kHz tone only between 1.0 s and 2.0 s of the source audio
    audio = _audio(
        unicode_tmp_path / "原声.m4a", "if(between(t,1,2),sin(2*PI*1000*t),0)", 3, ffmpeg_bin
    )
    out = unicode_tmp_path / "with audio.mp4"
    result = write_video(
        out, _frames(50), "25/1", audio=[AudioSpan(audio, Fraction(1), Fraction(1)), Silence(1)]
    )
    assert result.frames == 50
    assert result.audio is not None
    assert result.audio["codec_name"] == "aac"
    assert result.audio["sample_rate"] == 48000
    assert result.audio["channels"] == 2
    assert abs(result.audio["duration_s"] - 2.0) < 0.05
    pcm = _pcm(out, ffmpeg_bin)
    assert _rms(pcm, 0.1, 0.9) > 0.5, "the tone of 1.0-2.0 s plays first"
    assert _rms(pcm, 1.1, 1.9) < 0.01, "then the silence part"


def test_video_writer_rejects_bad_arguments_and_frames(unicode_tmp_path: Path) -> None:
    with pytest.raises(RenderError) as excinfo:
        VideoWriter(unicode_tmp_path / "x.mkv", 65, 0, "29.97", crf=60)
    assert len(excinfo.value.problems) == 5
    out = unicode_tmp_path / "bad.mp4"
    writer = VideoWriter(out, 64, 48)
    with pytest.raises(RenderError, match="shape"):
        writer.write(np.zeros((48, 32, 3), np.uint8))
    writer.abort()
    assert not out.exists()
    assert not (unicode_tmp_path / "bad.mp4.partial").exists()
    with pytest.raises(RenderError, match="no frames"):
        VideoWriter(out, 64, 48).close()
    write_video(out, _frames(2))
    with pytest.raises(FileExistsError):
        VideoWriter(out, 64, 48)


def test_video_writer_aborts_when_the_block_raises(unicode_tmp_path: Path) -> None:
    out = unicode_tmp_path / "aborted.mp4"

    def write_then_fail() -> None:
        with VideoWriter(out, 64, 48) as writer:
            writer.write(_frames(1)[0])
            raise KeyError("stop")

    with pytest.raises(KeyError):
        write_then_fail()
    assert not out.exists()
    assert not (unicode_tmp_path / "aborted.mp4.partial").exists()


def test_video_writer_enforces_max_bytes(unicode_tmp_path: Path) -> None:
    out = unicode_tmp_path / "big.mp4"
    with pytest.raises(RenderError, match="max_bytes"):
        write_video(out, _frames(5), max_bytes=100)
    assert not out.exists()


# ---------------------------------------------------------------------------- edits
def test_plan_highlight_edit_segments_and_frame_map() -> None:
    plan = plan_highlight_edit(
        5, 30, 20, "25/1", slow_factor=3, slow_before_s=0.2, slow_after_s=0.2, freeze_s=0.8
    )
    kinds = [segment.kind for segment in plan.segments]
    assert kinds == ["normal", "slow", "freeze", "slow"]
    assert plan.n_frames == 26 + 6 * 3 + 20 + 5 * 3
    mapping = plan.output_to_source()
    assert mapping[:26] == list(range(5, 31))
    assert mapping[26:32] == [15, 15, 15, 16, 16, 16]
    assert mapping[44:64] == [20] * 20
    assert mapping[-3:] == [25, 25, 25]
    table = plan.table()
    assert table[2] == {
        "kind": "freeze", "out_start": 44, "out_end": 63, "src_start": 20, "src_end": 20,
        "repeat": 20, "frames": 20, "seconds": 0.8,
    }  # fmt: skip
    plain = plan_highlight_edit(5, 30, 20, slow_factor=1, freeze_s=0)
    assert [s.kind for s in plain.segments] == ["normal"]
    assert plain.n_frames == 26
    late = plan_highlight_edit(
        0, 10, 10, slow_factor=2, slow_before_s=0.2, freeze_s=0.08, freeze_inside_slow=False
    )
    assert [s.kind for s in late.segments] == ["normal", "slow", "freeze"]
    assert late.output_to_source()[-3:] == [10, 10, 10]


def test_plan_highlight_edit_validation() -> None:
    with pytest.raises(RenderError) as excinfo:
        plan_highlight_edit(10, 20, 25, slow_factor=0, freeze_s=-1)
    assert len(excinfo.value.problems) == 2
    with pytest.raises(RenderError, match="window"):
        plan_highlight_edit(10, 20, 25)
    with pytest.raises(RenderError, match="cuts"):
        plan_highlight_edit(10, 20, 15, cuts=[-1, True])  # type: ignore[list-item]


def test_plan_highlight_edit_keeps_slow_motion_inside_the_key_shot() -> None:
    def slow(cuts: list[int]) -> list[tuple[int, ...]]:
        plan = plan_highlight_edit(
            0, 40, 20, slow_factor=2, slow_before_s=0.4, slow_after_s=0.4, freeze_s=0.2, cuts=cuts
        )
        assert plan.segments[0].source_frames == tuple(range(0, 41)), "normal speed crosses cuts"
        return [segment.source_frames for segment in plan.segments if segment.kind == "slow"]

    # 10 frames before and after the key frame 20; a cut is the first frame of a new shot
    assert slow([]) == [tuple(range(10, 21)), tuple(range(21, 31))]
    assert slow([17, 24]) == [tuple(range(17, 21)), tuple(range(21, 24))]
    assert slow([5, 35]) == [tuple(range(10, 21)), tuple(range(21, 31))], "cuts far away"
    assert slow([20, 21]) == [(20,)], "a one-frame shot has no slow motion after the key"


@pytest.fixture
def barcode_video(unicode_tmp_path: Path) -> Path:
    """50 frames, 320x180, 25 fps; frame k carries a barcode of k."""
    frames = []
    for k in range(50):
        frame = np.empty((180, 320, 3), np.uint8)
        frame[:] = GRASS
        draw_barcode(frame, k)
        frames.append(frame)
    path = unicode_tmp_path / "源 视频.mp4"
    write_frames_mp4(frames, path, fps="25/1", crf=12)
    return path


def _decoded_numbers(path: Path) -> list[int]:
    return [decode_barcode(frame) for _, frame in iter_frames(path)]


@pytest.mark.parametrize(
    ("slow_factor", "freeze_s", "expected"),
    [(1, 0.0, 26), (3, 0.4, 26 + 6 * 3 + 10 + 5 * 3)],
    ids=["normal-only", "slow-and-freeze"],
)
def test_render_edit_writes_the_planned_frames_exactly(
    barcode_video: Path,
    unicode_tmp_path: Path,
    style: Style,
    slow_factor: int,
    freeze_s: float,
    expected: int,
) -> None:
    plan = plan_highlight_edit(
        5,
        30,
        20,
        "25/1",
        slow_factor=slow_factor,
        slow_before_s=0.2,
        slow_after_s=0.2,
        freeze_s=freeze_s,
    )
    seen: list[tuple[int, str]] = []

    def annotate(frame: int, kind: str) -> FrameAnnotations:
        seen.append((frame, kind))
        return FrameAnnotations(banner=Banner(f"帧 {frame}", format_clock(frame)))

    out = unicode_tmp_path / f"edit {slow_factor}.mp4"
    result = render_edit(
        barcode_video,
        out,
        plan,
        annotate=annotate,
        freeze_layer=FrameAnnotations(
            card=InfoCard("定格", (CardRow("速度", "20", "km/h", ("未验证",), ("ev-9",)),))
        ),
        style=style,
        restrictions=("no_share", "hl3"),
    )
    assert plan.n_frames == expected == result.video.frames == probe(out).nb_frames_decoded
    assert _decoded_numbers(out) == plan.output_to_source()
    assert seen[:26] == [(k, "normal") for k in range(5, 31)]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["output"]["sha256"] == sha256_file(out)
    assert manifest["segments"] == plan.table()
    assert manifest["restrictions"] == ["hl3", "no_share"]
    assert manifest["slow_motion"] == {"method": "frame_repeat", "interpolated_frames": 0}
    text = result.manifest_path.read_text(encoding="utf-8")
    assert str(unicode_tmp_path) not in text
    assert not re.search(r"[A-Za-z]:[/\\]", text)
    rows = {(row["kind"], row["text"]): row for row in manifest["texts"]}
    if freeze_s:
        assert rows[("card_row", "速度 20 km/h")]["evidence_ids"] == ["ev-9"]
        assert rows[("card_row", "速度 20 km/h")]["frames"] == 10
        assert rows[("badge", f"慢放 ×{slow_factor}")]["frames"] == 33
    else:
        assert ("card_row", "速度 20 km/h") not in rows


def test_render_edit_audio_continues_after_the_window(
    barcode_video: Path, unicode_tmp_path: Path, style: Style, ffmpeg_bin: str
) -> None:
    # silence during the window (frames 0..24 = 0.0-1.0 s), a tone right after it
    audio = _audio(
        unicode_tmp_path / "tone after.m4a",
        "if(between(t,1,2.5),sin(2*PI*1000*t),0)",
        3,
        ffmpeg_bin,
    )
    plan = plan_highlight_edit(
        0, 24, 20, slow_factor=2, slow_before_s=0.2, slow_after_s=0.12, freeze_s=0.4
    )
    out = unicode_tmp_path / "with sound.mp4"
    result = render_edit(
        barcode_video, out, plan, style=style, audio_source=audio, part_badges=False
    )
    assert result.video.frames == plan.n_frames == 25 + 12 + 10 + 6
    assert result.video.audio is not None
    assert abs(result.video.audio["duration_s"] - plan.n_frames / 25) < 0.05
    pcm = _pcm(out, ffmpeg_bin)
    assert _rms(pcm, 0.05, 0.95) < 0.01, "the window's own audio is silent"
    replay = _rms(pcm, 1.1, 2.0)
    assert 0.25 < replay < 0.45, "the replay carries the following tone at -6 dB (rms 0.354)"
    silent = render_edit(
        barcode_video, unicode_tmp_path / "silent replay.mp4", plan, style=style,
        audio_source=audio, replay_audio="silence", part_badges=False,
    )  # fmt: skip
    pcm = _pcm(silent.video.path, ffmpeg_bin)
    assert _rms(pcm, 1.1, 2.0) < 0.01


def _reads(rows: tuple[dict, ...]) -> list[tuple[str, float, float]]:
    """(kind, start s, end s) of the audio parts of a render; silence has start 0."""
    out = []
    for row in rows:
        start = row.get("start_s", 0.0)
        out.append((row["kind"], round(start, 6), round(start + row["samples"] / 48000, 6)))
    return out


def test_render_edit_replays_the_window_audio_when_the_source_audio_ends(
    barcode_video: Path, unicode_tmp_path: Path, style: Style, ffmpeg_bin: str
) -> None:
    # The source audio is a tone that ends 0.8 s into the 1.0 s window: no audio follows it.
    audio = _audio(unicode_tmp_path / "ends early.m4a", "sin(2*PI*1000*t)", 0.8, ffmpeg_bin)
    plan = plan_highlight_edit(
        0, 24, 20, slow_factor=2, slow_before_s=0.2, slow_after_s=0.12, freeze_s=0.4
    )
    result = render_edit(
        barcode_video, unicode_tmp_path / "window replay.mp4", plan, style=style,
        audio_source=audio, part_badges=False,
    )  # fmt: skip
    # the window's last 0.2 s has no audio; the replay (0.48 + 0.4 + 0.24 s) plays the
    # window's own 0.8 s from its start and starts over when it runs out
    assert _reads(result.video.audio_parts) == [
        ("source", 0.0, 0.8), ("silence", 0.0, 0.2), ("source", 0.0, 0.48),
        ("source", 0.48, 0.8), ("source", 0.0, 0.08), ("source", 0.08, 0.32),
    ]  # fmt: skip
    assert sum(row["samples"] for row in result.video.audio_parts) == 48000 * 53 // 25
    pcm = _pcm(result.video.path, ffmpeg_bin)
    assert _rms(pcm, 0.05, 0.75) > 0.5
    assert _rms(pcm, 0.85, 0.95) < 0.01, "silence where the source has no audio"
    assert 0.25 < _rms(pcm, 1.1, 2.0) < 0.45, "the replay carries the window's tone at -6 dB"
    # Partly after the window: the first replay part still continues the broadcast audio.
    longer = _audio(unicode_tmp_path / "1.5 s.m4a", "sin(2*PI*1000*t)", 1.5, ffmpeg_bin)
    mixed = render_edit(
        barcode_video, unicode_tmp_path / "mixed replay.mp4", plan, style=style,
        audio_source=longer, part_badges=False,
    )  # fmt: skip
    assert _reads(mixed.video.audio_parts) == [
        ("source", 0.0, 1.0), ("source", 1.0, 1.48), ("source", 0.0, 0.4),
        ("source", 0.4, 0.64),
    ]  # fmt: skip
    # A window without any audio: everything is silence, and the manifest says so.
    late = plan_highlight_edit(
        15, 30, 25, slow_factor=2, slow_before_s=0.12, slow_after_s=0.08, freeze_s=0.12
    )
    short = _audio(unicode_tmp_path / "0.4 s.m4a", "sin(2*PI*1000*t)", 0.4, ffmpeg_bin)
    silent = render_edit(
        barcode_video, unicode_tmp_path / "no audio.mp4", late, style=style,
        audio_source=short, part_badges=False,
    )  # fmt: skip
    rows = json.loads(silent.manifest_path.read_text(encoding="utf-8"))["output"]["audio_parts"]
    assert {row["kind"] for row in rows} == {"silence"}
    assert sum(row["samples"] for row in rows) == 48000 * late.n_frames // 25
    assert _rms(_pcm(silent.video.path, ffmpeg_bin), 0.0, late.n_frames / 25) < 0.01


def test_render_edit_audio_offset_between_two_samples(
    barcode_video: Path, unicode_tmp_path: Path, style: Style, ffmpeg_bin: str
) -> None:
    # audio-file time = canonical time - 0.50001 s: the audio starts 0.50001 s into the
    # window, between two samples at 48 kHz (sample 24000.48)
    audio = _audio(unicode_tmp_path / "late start.m4a", "sin(2*PI*1000*t)", 2, ffmpeg_bin)
    plan = plan_highlight_edit(0, 24, 20, slow_factor=1, freeze_s=0)
    result = render_edit(
        barcode_video, unicode_tmp_path / "offset.mp4", plan, style=style,
        audio_source=audio, audio_offset_s=-0.50001, part_badges=False,
    )  # fmt: skip
    silence, source = result.video.audio_parts
    assert (silence["kind"], silence["samples"]) == ("silence", 24001), "rounded up"
    assert source["kind"] == "source"
    assert source["start_s"] == pytest.approx(24001 / 48000), "not before the first sample"
    assert source["samples"] == 48000 - 24001
    pcm = _pcm(result.video.path, ffmpeg_bin)
    assert _rms(pcm, 0.05, 0.45) < 0.01
    assert _rms(pcm, 0.6, 0.95) > 0.5


def test_render_edit_audio_offset_a_float_bit_off_a_fraction(
    barcode_video: Path, unicode_tmp_path: Path, style: Style, ffmpeg_bin: str
) -> None:
    # -(0.1 + 0.2) is -0.30000000000000004, a float's last bit from -0.3: the sample counts
    # and the filter graph take the offset as the same fraction, so the span that starts on
    # the audio's first sample does not start before the audio file.
    audio = _audio(unicode_tmp_path / "late start.m4a", "sin(2*PI*1000*t)", 2, ffmpeg_bin)
    plan = plan_highlight_edit(0, 24, 20, slow_factor=1, freeze_s=0)
    result = render_edit(
        barcode_video, unicode_tmp_path / "offset.mp4", plan, style=style,
        audio_source=audio, audio_offset_s=-(0.1 + 0.2), part_badges=False,
    )  # fmt: skip
    silence, source = result.video.audio_parts
    assert (silence["kind"], silence["samples"]) == ("silence", 14400)
    assert (source["kind"], source["start_s"], source["samples"]) == ("source", 0.3, 33600)
    pcm = _pcm(result.video.path, ffmpeg_bin)
    assert _rms(pcm, 0.05, 0.25) < 0.01
    assert _rms(pcm, 0.4, 0.95) > 0.5
    # Offsets of whole milliseconds as floats compute them: none starts before the file.
    for k in range(1, 1000):
        for offset in (-(k * 0.001), -(k / 1000), -(k * 0.1 / 100), -(0.001 * k + 1e-17)):
            parts = render_module._edit_audio(plan, audio, "continue", 0.0, offset)
            _graph, rows = render_module._audio_graph(parts, {audio: 1})
            assert sum(row["samples"] for row in rows) == 48000, offset
    # A span that does start before the audio file is still refused.
    with pytest.raises(RenderError, match="part 0 starts before the audio file"):
        render_module._audio_graph([AudioSpan(audio, 0, 0.5, offset_s=-0.001)], {audio: 1})


def test_render_edit_refuses_to_buffer_too_many_frames(
    barcode_video: Path, unicode_tmp_path: Path, style: Style
) -> None:
    plan = plan_highlight_edit(0, 40, 30, slow_factor=2, slow_before_s=1.0, slow_after_s=0.4)
    out = unicode_tmp_path / "too long.mp4"
    with pytest.raises(RenderError, match="max_buffer_frames"):
        render_edit(barcode_video, out, plan, style=style, max_buffer_frames=10)
    assert not out.exists()
    assert not out.with_name("too long.render.json").exists()


def test_render_edit_manifest_lists_the_card_rows_that_are_drawn(
    barcode_video: Path, unicode_tmp_path: Path, style: Style, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = tuple(CardRow(f"指标 {i}", f"{i}.0", "m", ("未验证",), (f"ev-{i}",)) for i in range(6))
    minimap = Minimap(players=((0.0, 0.0, "left"),), title="俯视图")
    drawn = _record_drawn_texts(monkeypatch)
    result = render_edit(
        barcode_video, unicode_tmp_path / "card.mp4",
        plan_highlight_edit(5, 10, 8, slow_factor=1, freeze_s=0.08),
        annotate=lambda frame, kind: FrameAnnotations(minimap=minimap),
        freeze_layer=FrameAnnotations(card=InfoCard("定格", rows, footer="数值为自动估计")),
        style=style, part_badges=False,
    )  # fmt: skip
    texts = result.manifest["texts"]
    listed = [row for row in texts if row["kind"] == "card_row"]
    drawn_rows = sorted({text for text in drawn if text.startswith("指标")})
    assert drawn_rows == [f"指标 {i}" for i in range(5)], "the minimap takes the last row's room"
    assert [row["text"] for row in listed] == [f"指标 {i} {i}.0 m" for i in range(5)]
    assert [row["evidence_ids"] for row in listed] == [[f"ev-{i}"] for i in range(5)]
    assert all(row["frames"] == 2 for row in listed), "on the two freeze frames"
    footer = "数值为自动估计；另有 1 项见报告"
    assert footer in drawn
    assert [row["text"] for row in texts if row["kind"] == "card_footer"] == [footer]


def test_render_edit_manifest_leaves_out_labels_under_the_card(
    barcode_video: Path, unicode_tmp_path: Path, style: Style
) -> None:
    # A player under the freeze card on every frame: its label is drawn and listed on the
    # frames without the card, never on the freeze frames.
    card = InfoCard("定格", (CardRow("速度", "18.6", "km/h", ("未验证",), ("ev-1",)),))
    frame = _frame(320, 180)
    region = _bbox(_changed(draw_annotations(frame, FrameAnnotations(card=card), style), frame))
    u, top = (region[0] + region[2]) / 2, region[1] + 10.0
    mark = PlayerMark(
        (u - 5, top, u + 5, top + 20), team="right", label="轨迹 2 · 18.6 km/h",
        badges=(Badge("未验证", "unverified", compact=True),), evidence_ids=("ev-2",),
    )  # fmt: skip
    plan = plan_highlight_edit(5, 10, 8, slow_factor=1, freeze_s=0.08)
    result = render_edit(
        barcode_video, unicode_tmp_path / "hidden.mp4", plan,
        annotate=lambda frame, kind: FrameAnnotations(players=(mark,)),
        freeze_layer=FrameAnnotations(card=card), style=style, part_badges=False,
    )  # fmt: skip
    frames = {
        (row["kind"], row["text"], tuple(row["evidence_ids"])): row["frames"]
        for row in result.manifest["texts"]
    }
    freeze = sum(row["frames"] for row in plan.table() if row["kind"] == "freeze")
    assert freeze == 2
    assert frames[("player_label", "轨迹 2 · 18.6 km/h", ("ev-2",))] == plan.n_frames - freeze
    assert frames[("badge", "未验证", ("ev-2",))] == plan.n_frames - freeze
    assert frames[("badge", "未验证", ("ev-1",))] == freeze, "the card row's own marker"


# ---------------------------------------------------------------------------- scenes
def _tracks(frames: range, rows: list[dict]) -> pd.DataFrame:
    """One row per frame and player: dicts with track_id, box, team, role, pitch."""
    records = []
    for frame in frames:
        for row in rows:
            x1, y1, x2, y2 = row["box"]
            pitch = row.get("pitch", (math.nan, math.nan))
            records.append(
                {
                    "frame": frame, "track_id": row["track_id"], "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2, "conf": 0.9, "role": row.get("role", "player"),
                    "team": row.get("team"), "jersey": row.get("jersey"),
                    "pitch_x": pitch[0], "pitch_y": pitch[1],
                }
            )  # fmt: skip
    return pd.DataFrame.from_records(records)


PLAYERS = [
    {"track_id": 1, "box": (100, 60, 130, 140), "team": "left", "jersey": "10",
     "pitch": (5.0, 3.0)},
    {"track_id": 2, "box": (150, 62, 180, 142), "team": "right", "pitch": (6.5, 3.5)},
    {"track_id": 3, "box": (250, 50, 270, 100), "team": "left", "pitch": (20.0, -10.0)},
    {"track_id": 4, "box": (40, 40, 60, 90), "team": None, "role": "referee"},
]  # fmt: skip


def _drop(tracks: pd.DataFrame, missing: dict[int, range]) -> pd.DataFrame:
    """``tracks`` without the rows of each track of ``missing`` on its frames there."""
    drop = np.zeros(len(tracks), dtype=bool)
    for track, frames in missing.items():
        drop |= (tracks["track_id"] == track) & tracks["frame"].isin(frames)
    return tracks[~drop].reset_index(drop=True)


def _duel(**changes: object) -> dict:
    highlight = {
        "id": "h1", "type": "duel_tackle", "start_frame": 5, "peak_frame": 20, "end_frame": 30,
        "score": 0.5, "track_ids": [1, 2], "title_zh": "00:00.80（第 20 帧）贴身争抢",
        "cues": {"duel": {"contribution": 0.1, "distance": 1.6, "unit": "m", "frame": 20,
                          "track_ids": [1, 2]}},
    }  # fmt: skip
    highlight.update(changes)
    return highlight


def test_scene_tables_report_missing_columns() -> None:
    with pytest.raises(RenderError) as excinfo:
        SceneTables(
            tracks=pd.DataFrame({"frame": [0]}),
            calib=pd.DataFrame({"K": [None]}),
            kinematics=pd.DataFrame({"frame": [0]}),
        )
    joined = "\n".join(excinfo.value.problems)
    assert "tracks misses column(s): track_id, x1, y1, x2, y2, role, team" in joined
    assert "calib misses column frame_index (or frame)" in joined
    assert "kinematics misses column(s): track_id, speed, valid" in joined


def test_scene_marks_focus_involved_and_context_players_on_image_frames() -> None:
    scene = HighlightScene(SceneTables(tracks=_tracks(range(0, 40), PLAYERS)), _duel(), "25/1")
    annotations = scene.annotate(20)
    by_track = {
        mark.box: mark.emphasis for mark in annotations.players
    }  # boxes are unique per player
    assert by_track == {
        (100.0, 60.0, 130.0, 140.0): "focus",
        (150.0, 62.0, 180.0, 142.0): "normal",
        (250.0, 50.0, 270.0, 100.0): "context",
    }, "the referee is not drawn"
    focus = next(mark for mark in annotations.players if mark.emphasis == "focus")
    assert focus.label == "左侧球队 · 10号"
    assert [line.label for line in annotations.lines] == ["争抢"], "no metres off the pitch"
    assert annotations.minimap is not None
    assert annotations.minimap.note is not None
    assert [badge.text for badge in annotations.badges] == ["仅画面平面"]
    assert annotations.banner == Banner("贴身争抢", "00:00.80", "贴身争抢 · 第 20 帧 · 画面平面")


def test_scene_texts_claim_no_more_than_the_evidence() -> None:
    # A ball_strike is an abrupt change of the ball's image velocity near a player; a duel,
    # two opponents close together near the ball; a reaction's shot need not be a close-up.
    assert HIGHLIGHT_TYPES_ZH["ball_strike"] == "触球（球速突变）"
    assert HIGHLIGHT_TYPES_ZH["closeup_reaction"] == "球员反应"
    assert HIGHLIGHT_TYPES_ZH["duel_tackle"] == "贴身争抢"
    claims = ("射门", "长传", "传球", "抢断", "特写")
    assert not [name for name in HIGHLIGHT_TYPES_ZH.values() if any(c in name for c in claims)]
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS))
    strike = HighlightScene(tables, _duel(type="ball_strike", cues={}), "25/1")
    annotations = strike.annotate(20)
    assert annotations.banner is not None
    assert annotations.banner.subtitle == "触球（球速突变） · 第 20 帧 · 画面平面"
    # Pitch frames are decided frame by frame: the note speaks of this frame, not the shot.
    assert annotations.minimap is not None
    assert annotations.minimap.note == "本帧没有可用标定，不显示球场位置"


def test_scene_shows_metres_and_speeds_only_on_pitch_frames() -> None:
    kinematics = pd.DataFrame(
        {"frame": [20, 20, 21, 30], "track_id": [1, 2, 1, 1], "speed": [7.0, 9.0, 6.0, 8.0],
         "valid": [True, False, True, True]}
    )  # fmt: skip
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS), kinematics=kinematics)
    scene = HighlightScene(tables, _duel(), "25/1", pitch_frames=range(10, 25))
    annotations = scene.annotate(20)
    focus = next(mark for mark in annotations.players if mark.emphasis == "focus")
    assert focus.label == "左侧球队 · 10号 · 25.2 km/h"
    assert focus.evidence_ids == ("kinematics.parquet#frame=20&track_id=1",)
    other = next(mark for mark in annotations.players if mark.emphasis == "normal")
    # The duel opponent is named with its team too; an invalid speed row is not shown.
    assert other.label == "右侧球队 · 轨迹 2"
    (line,) = annotations.lines
    assert line.label == f"争抢 · {math.hypot(1.5, 0.5):.1f} m"
    assert line.evidence_ids == ("refine/tracks.parquet#frame=20&track_id=1,2",)
    assert [badge.text for badge in line.badges] == ["未验证"]
    assert annotations.minimap is not None
    assert len(annotations.minimap.players) == 3
    assert [badge.text for badge in annotations.badges] == ["未验证"]
    assert scene.annotate(30).lines[0].label == "争抢", "frame 30 is not a pitch frame"
    late = next(mark for mark in scene.annotate(30).players if mark.emphasis == "focus")
    assert late.label == "左侧球队 · 10号", "no speed on a frame without pitch positions"
    assert late.evidence_ids == ()


def test_scene_rings_follow_the_calibrated_ground() -> None:
    camera = synthetic_broadcast_camera(1920, 1080)
    calib = pd.DataFrame(
        {"frame_index": [20], "K": [camera.K.reshape(-1).tolist()],
         "R": [camera.R.reshape(-1).tolist()], "C": [camera.C.tolist()], "width": [1920],
         "height": [1080], "pitch_usable": [True]}
    )  # fmt: skip
    tables = SceneTables(tracks=_tracks(range(18, 22), PLAYERS), calib=calib)
    scene = HighlightScene(tables, _duel(), "25/1")
    assert scene.is_pitch_frame(20)
    assert not scene.is_pitch_frame(21), "pitch frames default to pitch-usable calibration"
    focus = next(mark for mark in scene.annotate(20).players if mark.emphasis == "focus")
    assert focus.ring is not None
    expected = ground_circle_px(camera, (5.0, 3.0), 0.9, n=48)
    assert np.allclose(np.array(focus.ring), expected)
    assert scene.annotate(20).minimap is not None
    assert len(scene.annotate(20).minimap.view) >= 3
    plain = next(mark for mark in scene.annotate(21).players if mark.emphasis == "focus")
    assert plain.ring is None


def test_scene_ball_trail_never_crosses_a_cut() -> None:
    ball = pd.DataFrame(
        {"frame": list(range(0, 15)), "u": [10.0 * k for k in range(15)],
         "v": [100.0] * 15, "radius_px": [4.0] * 15,
         "status": ["detected"] * 12 + ["interpolated", "missing", "detected"]}
    )  # fmt: skip
    tables = SceneTables(tracks=_tracks(range(0, 15), PLAYERS[:2]), ball=ball)
    scene = HighlightScene(tables, _duel(), "25/1", cuts=[8], trail_frames=5)
    mark = scene.annotate(10).ball
    assert mark is not None
    assert mark.position == (100.0, 100.0)
    assert mark.trail == ((80.0, 100.0), (90.0, 100.0)), "frames 8 and 9 only: 7 is before the cut"
    assert scene.annotate(12).ball is not None
    assert scene.annotate(12).ball.interpolated
    assert scene.annotate(13).ball is not None
    assert scene.annotate(13).ball.position is None, "a missing ball keeps only its trail"


def test_scene_focus_and_roles_follow_the_highlight() -> None:
    attack = _duel(
        type="attack_in_box",
        track_ids=[2, 1],
        cues={"box_pressure": {"team": "left", "contribution": 0.2}},
    )
    scene = HighlightScene(SceneTables(tracks=_tracks(range(0, 40), PLAYERS)), attack, "25/1")
    assert scene.focus == 1, "the attacking team's player, not the first id"
    joined = _duel(
        cues={"cut_emphasis": {"contribution": 0.1, "cut_frame": 25,
                               "joined_shot_event": {"frame": 30, "track_ids": [3, 2]}}},
    )  # fmt: skip
    scene = HighlightScene(
        SceneTables(tracks=_tracks(range(0, 40), PLAYERS)), joined, "25/1", cuts=[25]
    )
    before = {mark.box[0]: mark.emphasis for mark in scene.annotate(20).players}
    after = {mark.box[0]: mark.emphasis for mark in scene.annotate(30).players}
    assert before == {100.0: "focus", 150.0: "normal", 250.0: "context"}
    assert after == {100.0: "context", 150.0: "normal", 250.0: "focus"}
    assert len(scene.annotate(30).lines) == 1


# Players of the shot after a cut at frame 25 (tracks do not survive the cut).
JOINED_PLAYERS = [
    {"track_id": 5, "box": (120, 70, 150, 150), "team": "left", "pitch": (5.0, 3.0)},
    {"track_id": 6, "box": (160, 72, 190, 152), "team": "right", "pitch": (6.0, 3.4)},
    {"track_id": 7, "box": (260, 50, 280, 100), "team": "left", "pitch": (20.0, -10.0)},
]  # fmt: skip


def _two_shots() -> SceneTables:
    tracks = pd.concat(
        [_tracks(range(0, 25), PLAYERS), _tracks(range(25, 40), JOINED_PLAYERS)],
        ignore_index=True,
    )
    return SceneTables(tracks=tracks)


def _joined_duel(partner: int | None = 6, **changes: object) -> dict:
    """A duel cue before the cut at 25 and the peak at 30 in the joined shot, where the
    joined event names a single player (5) and, by default, its duel partner (6)."""
    event: dict[str, object] = {"cue": "strike", "frame": 30, "track_ids": [5]}
    if partner is not None:
        event["partner_track_id"] = partner
    cues = {
        "duel": {"contribution": 0.1, "distance": 1.6, "unit": "m", "frame": 20,
                 "track_ids": [1, 2]},
        "cut_emphasis": {"contribution": 0.1, "cut_frame": 25, "joined_shot_event": event},
    }  # fmt: skip
    return _duel(peak_frame=30, end_frame=39, track_ids=[1, 2, 5, 6], cues=cues, **changes)


def test_scene_key_frame_roles_come_from_the_key_fields() -> None:
    highlight = _joined_duel(key_pair=[5, 6], key_focus_track=6)
    scene = HighlightScene(_two_shots(), highlight, "25/1", cuts=[25])
    annotations = scene.annotate(30)
    emphasis = {mark.box[0]: mark.emphasis for mark in annotations.players}
    assert emphasis == {120.0: "normal", 160.0: "focus", 260.0: "context"}, "both of the pair"
    labels = {mark.box[0]: mark.label for mark in annotations.players}
    assert (labels[120.0], labels[160.0]) == ("左侧球队 · 轨迹 5", "右侧球队 · 轨迹 6")
    (line,) = annotations.lines
    assert (line.start, line.end) == ((135.0, 150.0), (175.0, 152.0)), "the pair's feet"
    layer = scene.freeze_layer(())
    assert layer.spotlight == (175.0, 112.0, 90.0), "the spotlight is on the key focus"
    assert scene.roles_at(30) == (6, (5, 6), {5, 6})
    # The shot before the cut keeps the duel cue's players: the key fields name players of
    # the joined shot, so they do not become the highlight's own focus.
    assert scene.roles_at(20) == (1, (1, 2), {1, 2, 5, 6})
    assert scene.focus == 1
    # In a single-shot duel, the default focus is the player the key frame is about.
    single = HighlightScene(
        SceneTables(tracks=_tracks(range(0, 40), PLAYERS)),
        _duel(key_pair=[2, 1], key_focus_track=2),
        "25/1",
    )
    assert single.focus == 2
    assert single.roles_at(20) == (2, (2, 1), {1, 2})
    focus = next(mark for mark in single.annotate(20).players if mark.emphasis == "focus")
    assert focus.box == (150.0, 62.0, 180.0, 142.0)
    pair_only = HighlightScene(_two_shots(), _joined_duel(key_pair=[6, 5]), "25/1", cuts=[25])
    assert pair_only.roles_at(30) == (5, (6, 5), {5, 6}), "the joined focus is one of the pair"


def test_scene_joined_partner_makes_the_duel_pair() -> None:
    scene = HighlightScene(_two_shots(), _joined_duel(), "25/1", cuts=[25])
    annotations = scene.annotate(30)
    emphasis = {mark.box[0]: mark.emphasis for mark in annotations.players}
    assert emphasis == {120.0: "focus", 160.0: "normal", 260.0: "context"}
    assert [line.label for line in annotations.lines] == ["争抢"]
    assert scene.roles_at(30) == (5, (5, 6), {5, 6})


def test_scene_without_key_fields_keeps_its_roles() -> None:
    tables = _two_shots()
    scene = HighlightScene(tables, _joined_duel(partner=None), "25/1", cuts=[25])
    nulls = _joined_duel(partner=None, key_pair=None, key_focus_track=None)
    same = HighlightScene(tables, nulls, "25/1", cuts=[25])
    for frame in (20, 30):
        assert same.annotate(frame) == scene.annotate(frame), "null key fields are absent"
    emphasis = {mark.box[0]: mark.emphasis for mark in scene.annotate(30).players}
    assert emphasis == {120.0: "focus", 160.0: "context", 260.0: "context"}
    assert scene.annotate(30).lines == ()
    assert scene.focus == 1
    assert scene.roles_at(30) == (5, None, {5})
    assert scene.roles_at(20) == (1, (1, 2), {1, 2, 5, 6})


def test_scene_rejects_malformed_key_fields() -> None:
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS))
    for pair, focus in (([1], "2"), ([1, 1], -1), ([True, 2], 2.0)):
        with pytest.raises(RenderError) as excinfo:
            HighlightScene(tables, _duel(key_pair=pair, key_focus_track=focus), "25/1")
        assert len(excinfo.value.problems) == 2, (pair, focus)


def test_scene_rejects_bad_frames_and_focus_track() -> None:
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS))
    for peak in (None, "20", 20.5, -1, True):
        with pytest.raises(RenderError) as excinfo:
            HighlightScene(tables, _duel(peak_frame=peak), "25/1")
        assert excinfo.value.problems == [
            f"highlight peak_frame must be a non-negative int, got {peak!r}"
        ]
    with pytest.raises(RenderError) as excinfo:
        HighlightScene(tables, _duel(start_frame=None, end_frame="x"), "25/1", focus_track="1")
    assert len(excinfo.value.problems) == 3, "every problem is listed"
    assert HighlightScene(tables, _duel(peak_frame=np.int64(20)), "25/1").roles_at(20)[0] == 1


# Close-up reaction: a cut at 25 into a close shot of player 5 (_two_shots), peak at 30.
CLOSEUP = {
    "id": "h2", "type": "closeup_reaction", "start_frame": 5, "peak_frame": 30,
    "end_frame": 39, "score": 0.4, "track_ids": [1, 5],
    "cues": {"cut_emphasis": {"contribution": 0.1, "cut_frame": 25,
                              "joined_shot_event": {"cue": "subject", "frame": 32,
                                                    "track_ids": [5]}}},
}  # fmt: skip


def test_scene_null_key_focus_track_keeps_the_focus() -> None:
    # gbf.moments writes key_pair with a null key_focus_track for highlights that are not
    # duels: their key frame is still about the players gbf.highlights chose.
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS))
    sprint = _duel(
        type="sprint", track_ids=[3, 1, 2], cues={"speed": {"track_id": 3}},
        key_pair=[1, 2], key_focus_track=None,
    )  # fmt: skip
    scene = HighlightScene(tables, sprint, "25/1")
    assert scene.focus == 3
    assert scene.roles_at(20) == (3, (1, 2), {1, 2, 3})
    focus = next(mark for mark in scene.annotate(20).players if mark.emphasis == "focus")
    assert focus.box == (250.0, 50.0, 270.0, 100.0), "the sprinter"
    assert scene.freeze_layer(()).spotlight == (260.0, 75.0, 90.0)
    assert scene.annotate(20).lines == (), "no line of the key pair, which is no duel"
    # A close-up whose key pair are two other players of the close-up: the subject stays.
    closeup = HighlightScene(_two_shots(), {**CLOSEUP, "key_pair": [6, 7]}, "25/1", cuts=[25])
    assert closeup.roles_at(30) == (5, (6, 7), {5, 6, 7})
    assert closeup.freeze_layer(()).spotlight == (135.0, 110.0, 90.0)


def test_scene_key_fields_apply_only_to_players_of_the_key_shot() -> None:
    # A close-up whose key pair are players of the shot before the cut.
    scene = HighlightScene(_two_shots(), {**CLOSEUP, "key_pair": [1, 2]}, "25/1", cuts=[25])
    assert scene.roles_at(30) == (5, None, {5}), "the players of another shot take no role"
    emphasis = {mark.box[0]: mark.emphasis for mark in scene.annotate(30).players}
    assert emphasis == {120.0: "focus", 160.0: "context", 260.0: "context"}
    assert scene.freeze_layer(()).spotlight == (135.0, 110.0, 90.0)
    # Both key fields name players of the shot before the cut, the peak is after it.
    stale = _joined_duel(partner=None, key_pair=[1, 2], key_focus_track=1)
    scene = HighlightScene(_two_shots(), stale, "25/1", cuts=[25])
    assert scene.roles_at(30) == (5, None, {5})
    focus = next(mark for mark in scene.annotate(30).players if mark.emphasis == "focus")
    assert focus.box == (120.0, 70.0, 150.0, 150.0)
    layer = scene.freeze_layer(())
    assert (layer.spotlight, layer.dim) == ((135.0, 110.0, 90.0), 0.35)
    assert scene.roles_at(20) == (1, (1, 2), {1, 2, 5, 6}), "the shot before the cut as it was"


def test_scene_key_frame_focus_is_a_player_the_key_frame_shows() -> None:
    tracks = _tracks(range(0, 40), PLAYERS)

    def scene(missing: dict[int, range], **changes: object) -> HighlightScene:
        tables = SceneTables(tracks=_drop(tracks, missing))
        return HighlightScene(tables, _duel(key_pair=[2, 1], key_focus_track=2, **changes), "25/1")

    # The key focus is in the shot but off the key frame (13 frames, too long to fill in).
    away = scene({2: range(14, 27)})
    assert away.roles_at(20)[0] == 1, "the highlight's own focus, which the key frame shows"
    assert away.freeze_layer(()).spotlight == (115.0, 100.0, 90.0)
    # A gap the display fills in (7 frames) keeps it on the key frame.
    filled = scene({2: range(17, 24)})
    assert filled.roles_at(20)[0] == 2
    assert filled.freeze_layer(()).spotlight == (165.0, 102.0, 90.0)
    # Neither of the pair on the key frame: the next involved player that is.
    third = scene({1: range(14, 27), 2: range(14, 27)}, track_ids=[1, 2, 3])
    assert third.roles_at(20) == (3, (2, 1), {1, 2, 3})
    assert third.freeze_layer(()).spotlight == (260.0, 75.0, 90.0)
    # Nobody to show: the key focus stays (it is on the other frames of the shot).
    nobody = scene({1: range(14, 27), 2: range(14, 27)})
    assert nobody.roles_at(20)[0] == 2
    assert nobody.freeze_layer(()).spotlight is None


def test_scene_key_frame_stand_in_is_for_the_key_frame_only() -> None:
    # The duel pair (key focus 2) is off the key frame for 13 frames, too long to fill in;
    # the key frame shows a listed player (3). Only the key frame is about that player: the
    # other frames of the shot keep the key focus, as the duel's line keeps the pair.
    tracks = _tracks(range(0, 40), PLAYERS)
    pair_away = SceneTables(tracks=_drop(tracks, {1: range(14, 27), 2: range(14, 27)}))
    duel = _duel(track_ids=[1, 2, 3], key_pair=[2, 1], key_focus_track=2)
    scene = HighlightScene(pair_away, duel, "25/1")
    assert scene.roles_at(20) == (3, (2, 1), {1, 2, 3})
    assert scene.freeze_layer(()).spotlight == (260.0, 75.0, 90.0)
    assert scene.focus == 2, "the stand-in is not the highlight's focus"
    for frame in (0, 13, 19, 21, 27, 39):
        assert scene.roles_at(frame) == (2, (2, 1), {1, 2, 3}), frame
    annotations = scene.annotate(10)
    emphasis = {mark.box[0]: mark.emphasis for mark in annotations.players}
    assert emphasis == {100.0: "normal", 150.0: "focus", 250.0: "normal"}
    assert [line.label for line in annotations.lines] == ["争抢"]
    # A sprint without key fields whose sprinter (3) is off the key frame.
    sprint = _duel(type="sprint", track_ids=[3, 1, 2], cues={"speed": {"track_id": 3}})
    scene = HighlightScene(SceneTables(tracks=_drop(tracks, {3: range(14, 27)})), sprint, "25/1")
    assert scene.roles_at(20) == (1, None, {1, 2, 3}), "a listed player the key frame shows"
    assert scene.freeze_layer(()).spotlight == (115.0, 100.0, 90.0)
    assert scene.focus == 3
    for frame in (0, 13, 19, 21, 27, 39):
        assert scene.roles_at(frame) == (3, None, {1, 2, 3}), frame
    focus = next(mark for mark in scene.annotate(10).players if mark.emphasis == "focus")
    assert focus.box == (250.0, 50.0, 270.0, 100.0), "the sprinter"


def test_scene_stand_in_is_never_the_focus_of_another_shot() -> None:
    # The duel cue (1 and 2, frame 20) is before the cut at 25, the key frame (30) after it
    # in a shot without them and without a joined event; track_ids list players of both
    # shots. That shot takes a listed player it shows (5); the duel cue's shot keeps 1.
    duel = _duel(peak_frame=30, end_frame=39, track_ids=[1, 2, 5, 6])
    scene = HighlightScene(_two_shots(), duel, "25/1", cuts=[25])
    assert scene.focus == 1
    assert scene.roles_at(20) == (1, (1, 2), {1, 2, 5, 6})
    annotations = scene.annotate(20)
    focus = next(mark for mark in annotations.players if mark.emphasis == "focus")
    assert focus.box == (100.0, 60.0, 130.0, 140.0), "the duel's player at the duel"
    assert [line.label for line in annotations.lines] == ["争抢"]
    assert scene.roles_at(30) == scene.roles_at(26) == (5, (1, 2), {1, 2, 5, 6})
    assert scene.freeze_layer(()).spotlight == (135.0, 110.0, 90.0)


def test_scene_key_frame_candidates_in_order() -> None:
    # Player 1 (the duel cue's first, and the key focus where given) is off the key frame for
    # 13 frames; it shows the other of the key pair (2) and a listed player (3). With a key
    # focus the key frame is about the key pair, without one about the listed players.
    tables = SceneTables(tracks=_drop(_tracks(range(0, 40), PLAYERS), {1: range(14, 27)}))
    keyed = HighlightScene(
        tables, _duel(track_ids=[3, 1, 2], key_pair=[1, 2], key_focus_track=1), "25/1"
    )
    assert keyed.roles_at(20) == (2, (1, 2), {1, 2, 3})
    assert keyed.roles_at(10)[0] == 1
    null = HighlightScene(tables, _duel(track_ids=[3, 1, 2], key_pair=[1, 2]), "25/1")
    assert null.roles_at(20) == (3, (1, 2), {1, 2, 3})
    assert null.roles_at(10)[0] == 1
    # The shot's own focus (the sprinter, 3) comes right after the key focus.
    sprint = _duel(
        type="sprint", track_ids=[2, 3], cues={"speed": {"track_id": 3}},
        key_pair=[1, 2], key_focus_track=1,
    )  # fmt: skip
    assert HighlightScene(tables, sprint, "25/1").roles_at(20) == (3, (1, 2), {1, 2, 3})


def test_scene_shot_without_rows_of_its_focus_takes_an_involved_player_it_shows() -> None:
    # The key fields name players of the shot after the cut (no joined event): the duel's
    # focus is its key focus (6), which the duel cue's shot has no rows of.
    keyed = _duel(peak_frame=30, end_frame=39, key_pair=[5, 6], key_focus_track=6)
    scene = HighlightScene(_two_shots(), keyed, "25/1", cuts=[25])
    assert scene.focus == 6
    assert scene.roles_at(30) == (6, (5, 6), {1, 2, 5, 6})
    assert scene.roles_at(20) == (1, (1, 2), {1, 2}), "the type's focus, which the shot shows"
    focus = next(mark for mark in scene.annotate(20).players if mark.emphasis == "focus")
    assert focus.box == (100.0, 60.0, 130.0, 140.0)
    # An own shot after the joined shot (cuts at 25 and 40), with new track ids.
    later = [
        {"track_id": 8, "box": (300, 50, 320, 100), "team": "left"},
        {"track_id": 9, "box": (200, 60, 230, 140), "team": "right"},
    ]
    tracks = [_tracks(range(0, 25), PLAYERS), _tracks(range(25, 40), JOINED_PLAYERS)]
    tables = SceneTables(
        tracks=pd.concat([*tracks, _tracks(range(40, 60), later)], ignore_index=True)
    )
    longer = {**_joined_duel(), "end_frame": 59}
    listed = HighlightScene(tables, {**longer, "track_ids": [1, 2, 5, 6, 9]}, "25/1", cuts=[25, 40])
    assert listed.roles_at(45) == (9, (1, 2), {1, 2, 5, 6, 9}), "a listed player it shows"
    assert listed.roles_at(20)[0] == 1
    focus = next(mark for mark in listed.annotate(45).players if mark.emphasis == "focus")
    assert focus.box == (200.0, 60.0, 230.0, 140.0)
    unlisted = HighlightScene(tables, longer, "25/1", cuts=[25, 40])
    assert unlisted.roles_at(45)[0] == 1
    marks = unlisted.annotate(45).players
    assert {mark.emphasis for mark in marks} == {"context"}, "nobody involved is in the shot"


def test_scene_referee_takes_no_key_role() -> None:
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS))  # track 4 is the referee
    scene = HighlightScene(tables, _duel(key_pair=[1, 2], key_focus_track=4), "25/1")
    assert scene.focus == 1
    for frame in (5, 20, 35):
        assert scene.roles_at(frame) == (1, (1, 2), {1, 2}), frame
    pair = HighlightScene(tables, _duel(key_pair=[2, 4], key_focus_track=2), "25/1")
    assert pair.roles_at(20) == (2, (1, 2), {1, 2}), "the duel cue's pair stays"


def test_scene_focus_is_involved_in_every_shot() -> None:
    # A sprint whose sprinter (3) is not in track_ids, with the key frame after a cut: the
    # sprinter is involved in both shots, so its skeleton is drawn in both.
    keypoints = np.zeros((26, 3))
    keypoints[:, 0], keypoints[:, 1], keypoints[:, 2] = 260.0, np.linspace(50, 100, 26), 0.9
    pose = pd.DataFrame(
        {"frame": [10, 30], "track_id": [3, 3], "keypoints": [keypoints.reshape(-1).tolist()] * 2}
    )
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS), pose=pose)
    sprint = _duel(
        type="sprint", peak_frame=30, end_frame=39, track_ids=[1, 2],
        cues={"speed": {"track_id": 3}},
    )  # fmt: skip
    scene = HighlightScene(tables, sprint, "25/1", cuts=[25])
    assert scene.roles_at(10) == scene.roles_at(30) == (3, None, {1, 2, 3})
    for frame in (10, 30):
        (skeleton,) = scene.annotate(frame).skeletons
        assert skeleton.emphasis == "focus", frame


def test_scene_explicit_focus_track_wins_over_the_key_fields() -> None:
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS))
    scene = HighlightScene(tables, _duel(key_pair=[2, 1], key_focus_track=2), "25/1", focus_track=1)
    assert scene.focus == 1
    assert scene.roles_at(20) == (1, (2, 1), {1, 2})
    assert scene.freeze_layer(()).spotlight == (115.0, 100.0, 90.0)
    # focus_track is the focus of the highlight's own shots; a key frame in the joined
    # shot after a cut keeps the roles of that shot's players.
    joined = _joined_duel(key_pair=[5, 6], key_focus_track=6)
    scene = HighlightScene(_two_shots(), joined, "25/1", cuts=[25], focus_track=2)
    assert (scene.focus, scene.roles_at(20)) == (2, (2, (1, 2), {1, 2, 5, 6}))
    assert scene.roles_at(30) == (6, (5, 6), {5, 6})
    # It is never replaced, even in an own shot without rows of it.
    scene = HighlightScene(_two_shots(), _duel(end_frame=39), "25/1", cuts=[25], focus_track=5)
    assert scene.roles_at(20)[0] == scene.roles_at(30)[0] == 5


def test_scene_marks_every_speed_with_its_own_unverified_marker() -> None:
    kinematics = pd.DataFrame(
        {"frame": [20, 20], "track_id": [1, 2], "speed": [7.0, 9.0], "valid": [True, True]}
    )
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS), kinematics=kinematics)
    scene = HighlightScene(tables, _duel(), "25/1", pitch_frames=range(10, 25))
    annotations = scene.annotate(20)
    focus = next(mark for mark in annotations.players if mark.emphasis == "focus")
    other = next(mark for mark in annotations.players if mark.emphasis == "normal")
    assert other.label == "右侧球队 · 轨迹 2 · 32.4 km/h"
    texts = annotation_texts(annotations)
    assert ("badge", "未验证", ("kinematics.parquet#frame=20&track_id=2",)) in texts
    assert focus.badges == (Badge("未验证", "unverified"),)
    assert other.badges == (Badge("未验证", "unverified", compact=True),), "a compact marker"
    context = next(mark for mark in annotations.players if mark.emphasis == "context")
    assert context.label is None
    assert context.badges == ()


def test_scene_skeletons_and_freeze_layer() -> None:
    keypoints = np.zeros((26, 3))
    keypoints[:, 0], keypoints[:, 1], keypoints[:, 2] = 110.0, np.linspace(60, 140, 26), 0.9
    pose = pd.DataFrame(
        {"frame": [20, 20], "track_id": [1, 2],
         "keypoints": [keypoints.reshape(-1).tolist(), [1.0, 2.0, 3.0]]}
    )  # fmt: skip
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS), pose=pose)
    scene = HighlightScene(tables, _duel(), "25/1")
    (skeleton,) = scene.annotate(20).skeletons
    assert skeleton.emphasis == "focus"
    assert np.allclose(skeleton.keypoints, keypoints)
    layer = scene.freeze_layer((CardRow("速度", "25.2", "km/h", ("未验证",)),))
    assert layer.card is not None
    assert layer.card.title == "关键帧 · 第 20 帧"
    assert layer.spotlight == (115.0, 100.0, 90.0)
    assert layer.dim == 0.35
    assert layer.card.side == "right", "the card goes to the side away from the focus"
    narrow = HighlightScene(tables, _duel(), "25/1", frame_size=(200, 180))
    assert narrow.freeze_layer(()).card is not None
    assert narrow.freeze_layer(()).card.side == "left"


def test_scene_skeletons_carry_the_player_box() -> None:
    keypoints = np.zeros((26, 3))
    keypoints[:, 0], keypoints[:, 1], keypoints[:, 2] = 110.0, np.linspace(60, 140, 26), 0.9
    pose = pd.DataFrame(
        {"frame": [20], "track_id": [1], "keypoints": [keypoints.reshape(-1).tolist()]}
    )
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS), pose=pose)
    (skeleton,) = HighlightScene(tables, _duel(), "25/1").annotate(20).skeletons
    assert skeleton.box == (100.0, 60.0, 130.0, 140.0), "the drawing knows where the legs end"


def test_scene_fills_short_track_gaps_for_display_only() -> None:
    tracks = _tracks(range(0, 60), PLAYERS[:2])
    mover = tracks["track_id"] == 1
    shift = np.where(mover, 2.0 * tracks["frame"], 0.0)  # track 1 moves 2 px per frame
    tracks["x1"], tracks["x2"] = tracks["x1"] + shift, tracks["x2"] + shift
    # missing on 10-12 (3 frames), 20-27 (8 frames), 31-39 (9 frames) and 48-50 (a cut at 50)
    gaps = mover & tracks["frame"].isin([10, 11, 12, *range(20, 28), *range(31, 40), 48, 49, 50])
    tracks = tracks[~gaps].reset_index(drop=True)
    keypoints = np.zeros((26, 3))
    keypoints[:, 0], keypoints[:, 1], keypoints[:, 2] = 120.0, np.linspace(60, 140, 26), 0.9
    later = keypoints.copy()
    later[:, 0], later[:, 2] = 128.0, 0.5
    pose = pd.DataFrame(
        {"frame": [9, 13], "track_id": [1, 1],
         "keypoints": [keypoints.reshape(-1).tolist(), later.reshape(-1).tolist()]}
    )  # fmt: skip
    # valid speed rows on a measured frame (9) and on filled-in frames (11, 23), as
    # kinematics.parquet has them where it interpolates a position
    kinematics = pd.DataFrame(
        {"frame": [9, 11, 23], "track_id": [1, 1, 1], "speed": [7.0, 6.0, 6.5],
         "valid": [True, True, True]}
    )  # fmt: skip
    tables = SceneTables(tracks=tracks, pose=pose, kinematics=kinematics)
    before = tracks.copy()
    highlight = _duel(start_frame=0, peak_frame=5, end_frame=59)
    scene = HighlightScene(tables, highlight, "25/1", cuts=[50], pitch_frames=range(0, 60))

    def focus(frame: int) -> PlayerMark | None:
        return next((m for m in scene.annotate(frame).players if m.emphasis == "focus"), None)

    held, seen = focus(11), focus(9)
    assert held is not None, "a 3-frame gap is filled in"
    assert held.box == (122.0, 60.0, 152.0, 140.0), "the box is interpolated"
    assert seen is not None
    assert seen.label == "左侧球队 · 10号 · 25.2 km/h"
    assert held.label == "左侧球队 · 10号", "the label held, without a speed of its own"
    assert (held.evidence_ids, held.badges) == ((), ())
    (skeleton,) = scene.annotate(11).skeletons
    assert np.allclose(skeleton.keypoints[:, 0], 124.0)
    assert np.allclose(skeleton.keypoints[:, 2], 0.5), "the lower score of the two poses"
    assert len(scene.annotate(11).minimap.players) == 2
    (line,) = scene.annotate(11).lines
    measured = scene.annotate(9).lines[0]
    assert measured.evidence_ids == ("refine/tracks.parquet#frame=9&track_id=1,2",)
    assert line.label == measured.label + "（沿用）", "the held distance is explicit"
    assert line.label == f"争抢 · {math.hypot(1.5, 0.5):.1f} m（沿用）"
    assert line.evidence_ids == measured.evidence_ids, "with the evidence of where it was measured"
    longest = focus(23)
    assert longest is not None, "an 8-frame gap is filled in"
    assert longest.label == "左侧球队 · 10号"
    assert "km/h" not in "".join(text for _, text, _ in annotation_texts(scene.annotate(23)))
    assert focus(35) is None, "a 9-frame gap is not filled in"
    assert focus(49) is None, "never across a cut"
    pd.testing.assert_frame_equal(tables.tracks, before)


def test_scene_fills_pose_gaps_only_within_a_shot_and_up_to_eight_frames() -> None:
    keypoints = np.zeros((26, 3))
    keypoints[:, 0], keypoints[:, 1], keypoints[:, 2] = 115.0, np.linspace(60, 140, 26), 0.9
    # poses of track 1 on 3, 12 (8 frames after it), 22 (9 frames after that), 36, and 41
    # after the cut at 40 (4 frames between 36 and 41); its box is on every frame
    frames = [3, 12, 22, 36, 41]
    pose = pd.DataFrame(
        {"frame": frames, "track_id": [1] * len(frames),
         "keypoints": [keypoints.reshape(-1).tolist()] * len(frames)}
    )  # fmt: skip
    tables = SceneTables(tracks=_tracks(range(0, 50), PLAYERS[:2]), pose=pose)
    scene = HighlightScene(tables, _duel(end_frame=49), "25/1", cuts=[40])
    drawn = [frame for frame in range(0, 50) if scene.annotate(frame).skeletons]
    assert drawn == [*range(3, 13), 22, 36, 41], "an 8-frame gap only, never across the cut"


def _ball(rows: dict[int, tuple[float, float]], no_ground: tuple[int, ...] = ()) -> pd.DataFrame:
    frames = range(0, max(rows) + 1)
    return pd.DataFrame(
        {"frame": list(frames), "u": [100.0 + f for f in frames], "v": [50.0] * len(frames),
         "status": ["detected" if f in rows else "missing" for f in frames],
         "pitch_x": [rows[f][0] if f in rows else math.nan for f in frames],
         "pitch_y": [rows[f][1] if f in rows else math.nan for f in frames],
         "ground_assumption": [f in rows and f not in no_ground for f in frames]}
    )  # fmt: skip


def test_scene_minimap_ball_only_where_the_ground_assumption_holds() -> None:
    rows = {f: (0.4 * f, 5.0) for f in range(0, 10)}  # rolling at 10 m/s
    rows.update({f: (10.0 + 2.5 * (f - 12), 5.0) for f in range(12, 20)})  # 62.5 m/s
    rows.update({f: (30.0 + 0.4 * f, -34.5) for f in range(22, 26)})  # along the touchline
    rows.update({f: (30.0 + 0.4 * f, -35.5) for f in range(26, 30)})  # 1.5 m beyond it
    tables = SceneTables(tracks=_tracks(range(0, 30), PLAYERS[:2]), ball=_ball(rows, (5,)))
    scene = HighlightScene(tables, _duel(), "25/1", pitch_frames=range(0, 30))

    def ball(frame: int) -> tuple[float, float] | None:
        minimap = scene.annotate(frame).minimap
        assert minimap is not None
        return minimap.ball

    assert ball(3) == pytest.approx((1.2, 5.0))
    assert ball(5) is None, "the table says the ground assumption does not hold"
    assert ball(15) is None, "too fast for a ball on the grass: in the air"
    assert ball(23) == pytest.approx((39.2, -34.5)), "within the margin of the touchline"
    assert ball(27) is None, "1.5 m beyond the touchline"
    assert scene.annotate(15).ball is not None, "the ball on the frame itself is unchanged"


def test_scene_duel_line_only_around_the_duel() -> None:
    cue = {"contribution": 0.1, "distance": 1.6, "unit": "m", "frame": 60, "track_ids": [1, 2]}
    highlight = _duel(start_frame=0, peak_frame=60, end_frame=90, cues={"duel": cue})
    tables = SceneTables(tracks=_tracks(range(0, 100), PLAYERS))
    scene = HighlightScene(tables, highlight, "25/1", pitch_frames=range(0, 20))
    assert scene.annotate(30).lines == (), "no 争抢 line 1.2 s before the duel"
    assert [line.label for line in scene.annotate(35).lines] == ["争抢"]
    assert [line.label for line in scene.annotate(85).lines] == ["争抢"]
    assert scene.annotate(86).lines == ()
    (line,) = scene.annotate(10).lines  # a pitch frame: the nearest opponent, as anywhere
    assert line.label == f"最近对手 · {math.hypot(1.5, 0.5):.1f} m"


def test_scene_key_pair_line_around_the_key_frame() -> None:
    # Track 3 is an opponent of track 1, nearer to it (0.5 m) than track 2 (1.6 m).
    near = {"track_id": 3, "box": (250, 50, 270, 100), "team": "right", "pitch": (5.5, 3.2)}
    tables = SceneTables(tracks=_tracks(range(0, 100), [*PLAYERS[:2], near]))

    def keyed(key_pair: list[int], cue_frame: int, pitch: bool = False) -> HighlightScene:
        cue = {"contribution": 0.1, "frame": cue_frame, "track_ids": [1, 2]}
        highlight = _duel(
            start_frame=0, peak_frame=80, end_frame=99, cues={"duel": cue},
            key_pair=key_pair, key_focus_track=1,
        )  # fmt: skip
        return HighlightScene(tables, highlight, "25/1", pitch_frames=range(100) if pitch else ())

    def duel_frames(scene: HighlightScene) -> list[int]:
        return [f for f in range(100) if [ln.label for ln in scene.annotate(f).lines] == ["争抢"]]

    # The key pair is the duel cue's pair (in either order): 1 s around the cue's frame
    # and 1 s around the key frame.
    for key_pair in ([1, 2], [2, 1]):
        assert duel_frames(keyed(key_pair, 20)) == [*range(0, 46), *range(55, 100)], key_pair
    on_pitch = keyed([1, 2], 20, pitch=True)
    (line,) = on_pitch.annotate(80).lines
    assert line.label == f"争抢 · {math.hypot(1.5, 0.5):.1f} m", "the key pair, not the nearest"
    assert line.evidence_ids == ("refine/tracks.parquet#frame=80&track_id=1,2",)
    (between,) = on_pitch.annotate(50).lines
    assert between.evidence_ids == ("refine/tracks.parquet#frame=50&track_id=1,3",), (
        "nearest opponent"
    )
    # Another pair than the cue's: around the key frame only, as the cue is about the
    # cue's pair.
    assert duel_frames(keyed([1, 3], 80)) == list(range(55, 100))
    assert duel_frames(keyed([1, 3], 20)) == list(range(55, 100))
    # Track ids that run on across a cut (at 70): the key frame is a moment of the key pair
    # in its own shot only, not on the frames before the cut within 1 s of it.
    across = _duel(
        start_frame=0, peak_frame=80, end_frame=99, key_pair=[2, 1], key_focus_track=1,
        cues={"duel": {"contribution": 0.1, "frame": 20, "track_ids": [1, 2]}},
    )  # fmt: skip
    cut = HighlightScene(tables, across, "25/1", cuts=[70])
    assert duel_frames(cut) == [*range(0, 46), *range(70, 100)]


def test_scene_joined_pair_line_around_the_joined_event() -> None:
    # A 75-frame joined shot after the cut at 25, whose event (frame 40: player 5 and its
    # partner 6) is the moment of that pair: the 争抢 line within 1 s of it, in that shot.
    tracks = [_tracks(range(0, 25), PLAYERS), _tracks(range(25, 100), JOINED_PLAYERS)]
    tables = SceneTables(tracks=pd.concat(tracks, ignore_index=True))
    event = {"cue": "strike", "frame": 40, "track_ids": [5], "partner_track_id": 6}
    cues = {
        "duel": {"contribution": 0.1, "frame": 20, "track_ids": [1, 2]},
        "cut_emphasis": {"contribution": 0.1, "cut_frame": 25, "joined_shot_event": event},
    }  # fmt: skip
    highlight = _duel(start_frame=0, end_frame=99, track_ids=[1, 2, 5, 6], cues=cues)

    def duel_frames(scene: HighlightScene) -> list[int]:
        return [f for f in range(100) if [ln.label for ln in scene.annotate(f).lines] == ["争抢"]]

    scene = HighlightScene(tables, highlight, "25/1", cuts=[25])
    assert scene.roles_at(50) == (5, (5, 6), {5, 6})
    assert duel_frames(scene) == list(range(0, 66)), "the cue's pair, then the event's"
    # The same pair as the key pair, with the key frame (95) in that shot: both moments.
    keyed = {**highlight, "peak_frame": 95, "key_pair": [6, 5], "key_focus_track": 5}
    scene = HighlightScene(tables, keyed, "25/1", cuts=[25])
    assert duel_frames(scene) == [*range(0, 66), *range(70, 100)]
    # Track ids that run on across the cut (at 50), the event's pair being the duel cue's:
    # the event's frame (60) is a moment of the pair in the event's shot only.
    event = {"cue": "duel", "frame": 60, "track_ids": [2, 1]}
    cues = {
        "duel": {"contribution": 0.1, "frame": 10, "track_ids": [1, 2]},
        "cut_emphasis": {"contribution": 0.1, "cut_frame": 50, "joined_shot_event": event},
    }  # fmt: skip
    same_ids = SceneTables(tracks=_tracks(range(0, 100), PLAYERS))
    highlight = _duel(start_frame=0, peak_frame=10, end_frame=99, cues=cues)
    scene = HighlightScene(same_ids, highlight, "25/1", cuts=[50])
    assert duel_frames(scene) == [*range(0, 36), *range(50, 86)]


def test_scene_no_opponent_line_when_a_player_without_a_team_is_nearer() -> None:
    # gbf.team_check clears team labels that contradict the shirt colours: a player
    # without a label may be the nearest opponent.
    cleared = {"track_id": 5, "box": (132, 61, 146, 141), "team": None, "pitch": (5.5, 3.0)}
    cue = {"contribution": 0.1, "distance": 1.6, "unit": "m", "frame": 60, "track_ids": [1, 2]}
    highlight = _duel(start_frame=0, peak_frame=60, end_frame=90, cues={"duel": cue})

    def lines(player: dict) -> tuple:
        tables = SceneTables(tracks=_tracks(range(0, 100), [*PLAYERS, player]))
        scene = HighlightScene(tables, highlight, "25/1", pitch_frames=range(0, 20))
        return scene.annotate(10).lines

    assert lines(cleared) == (), "0.5 m from the focus, nearer than the labelled opponent"
    (line,) = lines({**cleared, "pitch": (30.0, 20.0)})
    assert line.label == f"最近对手 · {math.hypot(1.5, 0.5):.1f} m", "far away: unchanged"


@pytest.mark.parametrize("kind", ["attack_in_box", "sprint", "ball_strike", "closeup_reaction"])
def test_scene_only_a_duel_links_its_pair(kind: str) -> None:
    # gbf.moments gives key_pair to every highlight whose key moments used a pair, here the
    # duel cue's (1 and 2, 2 frames before the key frame); in a highlight that is no duel
    # the key frame, as every frame, links the focus player (1) to the nearest opponent,
    # the player the data card's 最近对手距离 is about: track 3, 0.5 m away, not 2 (1.6 m).
    near = {"track_id": 3, "box": (250, 50, 270, 100), "team": "right", "pitch": (5.5, 3.2)}
    tables = SceneTables(tracks=_tracks(range(0, 100), [*PLAYERS[:2], near]))
    cues = {
        "duel": {"contribution": 0.1, "frame": 78, "track_ids": [1, 2]},
        "box_pressure": {"team": "left", "contribution": 0.2},
        "speed": {"track_id": 1},
    }
    highlight = _duel(
        type=kind, start_frame=0, peak_frame=80, end_frame=99, cues=cues, key_pair=[1, 2],
        key_focus_track=None,
    )  # fmt: skip
    scene = HighlightScene(tables, highlight, "25/1", pitch_frames=range(0, 100))
    assert scene.roles_at(80) == (1, (1, 2), {1, 2})
    lines = {frame: scene.annotate(frame).lines for frame in range(0, 100)}
    linked = {f: [line.evidence_ids[0].rsplit("=", 1)[1] for line in lines[f]] for f in lines}
    assert [frame for frame in lines if linked[frame] != ["1,3"]] == [], "key frame included"
    assert {line.label for frame in lines for line in lines[frame]} == {
        f"最近对手 · {math.hypot(0.5, 0.2):.1f} m"
    }
    off_pitch = HighlightScene(tables, highlight, "25/1")
    assert [f for f in range(0, 100) if off_pitch.annotate(f).lines] == [], "no 争抢 line"
    # The same highlight as a duel links its pair around the key frame and the cue's frame.
    duel = HighlightScene(
        tables, {**highlight, "type": "duel_tackle"}, "25/1", pitch_frames=range(0, 100)
    )
    (line,) = duel.annotate(80).lines
    assert line.label == f"争抢 · {math.hypot(1.5, 0.5):.1f} m"
    assert line.evidence_ids == ("refine/tracks.parquet#frame=80&track_id=1,2",)


def test_scene_line_labels_say_what_they_measure() -> None:
    # Every line says what it links, on pitch frames before its distance: 争抢 for the duel
    # pair, 最近对手 for the focus player's nearest opponent. A distance held over a filled-in
    # frame keeps that form; without one to hold, the line still says what it links. Over a
    # frame where the focus player is filled in, its nearest-opponent line goes on to the
    # opponent that was nearest on the last frame it was observed, never chosen anew by an
    # interpolated position.
    tables = SceneTables(tracks=_drop(_tracks(range(0, 60), PLAYERS), {1: range(10, 13)}))
    highlight = _duel(start_frame=0, peak_frame=20, end_frame=59)  # the duel cue at 20
    metres = f"{math.hypot(1.5, 0.5):.1f} m"
    scene = HighlightScene(tables, highlight, "25/1", pitch_frames=range(0, 60))
    assert [line.label for line in scene.annotate(20).lines] == [f"争抢 · {metres}"]
    assert [line.label for line in scene.annotate(50).lines] == [f"最近对手 · {metres}"]
    (held,) = scene.annotate(11).lines
    assert (held.label, held.evidence_ids) == (
        f"争抢 · {metres}（沿用）",
        ("refine/tracks.parquet#frame=9&track_id=1,2",),
    ), "the distance measured on frame 9, held over the filled-in frame 11"
    late = HighlightScene(tables, highlight, "25/1", pitch_frames=range(10, 60))
    assert [(line.label, line.evidence_ids) for line in late.annotate(11).lines] == [("争抢", ())]
    cue = {"contribution": 0.1, "frame": 50, "track_ids": [1, 2]}  # frame 11 is far from it
    away = _duel(start_frame=0, peak_frame=20, end_frame=59, cues={"duel": cue})
    nearest = HighlightScene(tables, away, "25/1", pitch_frames=range(0, 60))
    assert [(line.label, line.evidence_ids) for line in nearest.annotate(11).lines] == [
        (f"最近对手 · {metres}（沿用）", ("refine/tracks.parquet#frame=9&track_id=1,2",))
    ], "the opponent nearest on frame 9, the last frame the focus player was observed on"
    late = HighlightScene(tables, away, "25/1", pitch_frames=range(10, 60))
    assert late.annotate(11).lines == (), "the focus was not observed since: nobody to link"
    assert [
        line.label for line in HighlightScene(tables, highlight, "25/1").annotate(20).lines
    ] == ["争抢"], "off the pitch frames"


def test_scene_nearest_opponent_line_leaves_out_goalkeepers() -> None:
    # The data card's 最近对手距离 (gbf.measure) is the distance to the nearest outfield
    # player of the other team; the line labelled 最近对手 links that player too, not the
    # other team's goalkeeper standing nearer.
    keeper = {"track_id": 6, "box": (132, 61, 146, 141), "team": "right", "role": "goalkeeper",
              "pitch": (5.2, 3.1)}  # fmt: skip
    cue = {"contribution": 0.1, "distance": 1.6, "unit": "m", "frame": 60, "track_ids": [1, 2]}
    highlight = _duel(start_frame=0, peak_frame=60, end_frame=90, cues={"duel": cue})
    tables = SceneTables(tracks=_tracks(range(0, 100), [*PLAYERS, keeper]))
    scene = HighlightScene(tables, highlight, "25/1", pitch_frames=range(0, 20))
    (line,) = scene.annotate(10).lines
    assert line.label == f"最近对手 · {math.hypot(1.5, 0.5):.1f} m"
    assert line.evidence_ids == ("refine/tracks.parquet#frame=10&track_id=1,2",)


def test_scene_line_never_links_a_box_on_the_focus_players_own_spot() -> None:
    # A box around two players that ends at the feet of the focus player (the detector merged
    # it with the opponent behind) shares its bottom edge and its ground position: no
    # opponent 0.0 m away. The line links the real nearest opponent at its own distance, as
    # gbf.measure's card does; a duel pair on one spot gets no distance (below the floor).
    merged = {"track_id": 9, "box": (101, 20, 129, 140), "team": "right", "pitch": (5.03, 3.0)}
    tables = SceneTables(tracks=_tracks(range(0, 40), [*PLAYERS, merged]))
    strike = _duel(type="ball_strike", cues={}, start_frame=0, peak_frame=20, end_frame=39)
    (line,) = HighlightScene(tables, strike, "25/1", pitch_frames=range(0, 40)).annotate(20).lines
    assert line.label == f"最近对手 · {math.hypot(1.5, 0.5):.1f} m"
    assert line.evidence_ids == ("refine/tracks.parquet#frame=20&track_id=1,2",)
    cue = {"contribution": 0.1, "frame": 20, "track_ids": [1, 9]}
    duel = _duel(start_frame=0, peak_frame=20, end_frame=39, cues={"duel": cue})
    (line,) = HighlightScene(tables, duel, "25/1", pitch_frames=range(0, 40)).annotate(20).lines
    assert (line.label, line.evidence_ids) == ("争抢", ()), "never a distance below the floor"


def test_scene_nearest_opponent_is_chosen_among_the_players_on_the_frame() -> None:
    # Track 5 runs past the focus player and is missing on frames 11-17: its filled-in
    # position (for display only) passes nearer than track 2, but measures nothing. The line
    # cannot claim a farther observed player is nearest while that player is drawn nearer.
    passing = [
        _tracks(range(0, 11), [{"track_id": 5, "box": (110, 64, 140, 144), "team": "right",
                                "pitch": (4.2, 3.8)}]),
        _tracks(range(18, 40), [{"track_id": 5, "box": (126, 64, 156, 144), "team": "right",
                                 "pitch": (5.8, 3.8)}]),
    ]  # fmt: skip
    tracks = pd.concat([_tracks(range(0, 40), PLAYERS), *passing], ignore_index=True)
    strike = _duel(type="ball_strike", cues={}, start_frame=0, peak_frame=30, end_frame=39)
    scene = HighlightScene(SceneTables(tracks=tracks), strike, "25/1", pitch_frames=range(0, 40))
    assert any(mark.box[0] == 118.0 for mark in scene.annotate(14).players), "drawn, filled in"
    assert not scene.annotate(14).lines, "filled position withholds; never selects or measures"
    missing_focus = _drop(tracks, {1: range(15, 16)})
    held = HighlightScene(
        SceneTables(tracks=missing_focus), strike, "25/1", pitch_frames=range(0, 40)
    )
    assert not held.annotate(15).lines, (
        "no held measurement when the last observation was ambiguous"
    )
    (line,) = scene.annotate(10).lines
    assert line.evidence_ids == ("refine/tracks.parquet#frame=10&track_id=1,5",), (
        "observed: nearest"
    )
    # A duel partner drawn at a filled-in position gets no distance: nothing measured there.
    duel = _duel(start_frame=0, peak_frame=20, end_frame=39)  # the pair 1 and 2, cue at 20
    gap = SceneTables(tracks=_drop(_tracks(range(0, 40), PLAYERS), {2: range(18, 21)}))
    (line,) = HighlightScene(gap, duel, "25/1", pitch_frames=range(0, 40)).annotate(19).lines
    assert (line.label, line.evidence_ids) == ("争抢", ())


@pytest.mark.parametrize("team", ["right", None])
def test_held_nearest_is_withheld_when_ambiguity_appears_now(team: str | None) -> None:
    # The last observed focus frame is clear. A filled opponent moves nearer only during
    # the focus gap, so checking the historical frame alone is insufficient.
    extra = {"track_id": 5, "box": (300, 64, 330, 144), "team": team}
    passing = pd.concat(
        [
            _tracks([10], [{**extra, "pitch": (8.0, 3.0)}]),
            _tracks([14], [{**extra, "pitch": (5.4, 3.0)}]),
        ]
    )
    tracks = pd.concat([_drop(_tracks(range(40), PLAYERS), {1: range(11, 14)}), passing])
    strike = _duel(type="ball_strike", cues={}, start_frame=0, end_frame=39)
    scene = HighlightScene(SceneTables(tracks=tracks), strike, pitch_frames=range(40))
    assert scene.annotate(10).lines[0].evidence_ids, "historical nearest is measured"
    assert scene.annotate(11).lines, "filled other player still farther"
    assert not scene.annotate(13).lines, "new current ambiguity withholds held distance"


def test_scene_boxes_without_a_track_are_context_only() -> None:
    # Boxes without a track (id -1) are not one person: never filled in between two of them,
    # never the nearest opponent (gbf.measure leaves them out too), no team colour, no label
    # and no minimap dot; at most a faint ring.
    nobody = {"track_id": -1, "box": (140, 61, 154, 141), "team": "right", "pitch": (5.6, 3.0)}
    tracks = pd.concat(
        [_tracks(range(0, 40), PLAYERS), _tracks([5, 9], [nobody])], ignore_index=True
    )
    strike = _duel(type="ball_strike", cues={}, start_frame=0, peak_frame=20, end_frame=39)
    scene = HighlightScene(SceneTables(tracks=tracks), strike, "25/1", pitch_frames=range(0, 40))
    annotations = scene.annotate(5)
    marks = [mark for mark in annotations.players if mark.box[0] == 140.0]
    assert [(m.emphasis, m.team, m.label) for m in marks] == [("context", None, None)]
    assert annotations.minimap is not None
    assert len(annotations.minimap.players) == 3
    (line,) = annotations.lines
    assert line.evidence_ids == ("refine/tracks.parquet#frame=5&track_id=1,2",)
    assert not [m for m in scene.annotate(7).players if m.box[0] == 140.0], "never filled in"


def test_scene_people_off_the_pitch_are_context_only() -> None:
    # A coach of the other team in the technical area, beyond the touchline by more than
    # the margin, listed first among the highlight's players: no team colour, no label, no
    # minimap dot, no line and never the focus; the nearest opponent is a player in play.
    beyond = 34.0 + OFF_PITCH_MARGIN_M + 1.0
    coach = {"track_id": 7, "box": (300, 150, 330, 230), "team": "right", "pitch": (6.0, beyond)}
    players = [
        {**PLAYERS[0], "pitch": (5.0, 33.6)}, {**PLAYERS[1], "pitch": (6.5, 30.0)}, PLAYERS[2],
        coach,
    ]  # fmt: skip
    strike = _duel(
        type="ball_strike", cues={}, track_ids=[7, 1, 2], start_frame=0, peak_frame=20,
        end_frame=39,
    )  # fmt: skip
    tables = SceneTables(tracks=_tracks(range(0, 40), players))
    scene = HighlightScene(tables, strike, "25/1", pitch_frames=range(0, 40))
    annotations = scene.annotate(10)
    (mark,) = [mark for mark in annotations.players if mark.box[0] == 300.0]
    assert (mark.emphasis, mark.team, mark.label) == ("context", None, None)
    assert scene.roles_at(10)[0] == 1, "the first of the highlight's players on the pitch"
    assert annotations.minimap is not None
    assert len(annotations.minimap.players) == 3
    (line,) = annotations.lines
    assert line.evidence_ids == ("refine/tracks.parquet#frame=10&track_id=1,2",)
    assert line.label == f"最近对手 · {math.hypot(1.5, 3.6):.1f} m"


@pytest.mark.parametrize("cut_off", [True, False])
def test_scene_a_person_the_frame_cuts_off_past_the_near_touchline_is_off_it(
    cut_off: bool,
) -> None:
    # The camera stands beyond the near touchline and the bottom of the frame lies just past
    # it. A coach in the technical area under the camera is cut off by the frame: the ground
    # point of his box, the frame edge's, is past the line by less than the margin, while
    # his feet, below the frame, are nearer the camera and so farther out. He is off the
    # pitch. The same ground point under a box the frame does not cut off is within the
    # margin (a player taking a throw-in): a player.
    camera = synthetic_broadcast_camera(1920, 1080, target=(0.0, 12.0, 0.0))
    ground, _valid = pixel_to_ground(camera, [[1200.0, 1079.5]])
    edge_x, edge_y = float(ground[0][0]), float(ground[0][1])
    assert 34.0 < edge_y < 34.0 + OFF_PITCH_MARGIN_M
    bottom = 1079.5 if cut_off else 1040.0
    frames = list(range(0, 10))
    rows = []
    for frame in frames:
        for track, team, (x, y) in ((1, "left", (3.0, 25.0)), (2, "right", (5.0, 27.0))):
            x1, y1, x2, y2 = _standing_box(camera, x, y)
            rows.append(
                {"frame": frame, "track_id": track, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                 "role": "player", "team": team, "pitch_x": x, "pitch_y": y}
            )  # fmt: skip
        rows.append(
            {"frame": frame, "track_id": 7, "x1": 1180.0, "y1": bottom - 80.0, "x2": 1220.0,
             "y2": bottom, "role": "player", "team": "right", "pitch_x": edge_x,
             "pitch_y": edge_y}
        )  # fmt: skip
    tables = SceneTables(tracks=pd.DataFrame(rows), calib=_calib(camera, frames))
    strike = _duel(
        type="ball_strike", cues={}, track_ids=[7, 1, 2], start_frame=0, peak_frame=5,
        end_frame=9,
    )  # fmt: skip
    scene = HighlightScene(tables, strike, "25/1", pitch_frames=frames)
    annotations = scene.annotate(5)
    (mark,) = [mark for mark in annotations.players if mark.box[0] == 1180.0]
    assert annotations.minimap is not None
    if cut_off:
        assert (mark.emphasis, mark.team, mark.label) == ("context", None, None)
        assert scene.roles_at(5)[0] == 1, "never the focus"
        assert len(annotations.minimap.players) == 2
    else:
        assert (mark.emphasis, mark.team) == ("focus", "right")
        assert len(annotations.minimap.players) == 3


def test_scene_merged_part_after_the_key_frame_has_no_focus() -> None:
    # A highlight merged from two parts: after the key frame's shot, a shot without the key
    # frame's players is the later part's, about other players. Nobody is marked as the
    # highlight's focus there; its players stay involved.
    tracks = pd.concat(
        [_tracks(range(0, 25), PLAYERS), _tracks(range(25, 40), JOINED_PLAYERS)],
        ignore_index=True,
    )
    parts = [{"id": "h1", "start_frame": 0, "end_frame": 24},
             {"id": "h2", "start_frame": 25, "end_frame": 39}]  # fmt: skip
    merged = _duel(
        start_frame=0, peak_frame=20, end_frame=39, track_ids=[1, 2, 5, 6], key_pair=[1, 2],
        key_focus_track=1, merged_parts=parts,
    )  # fmt: skip
    scene = HighlightScene(SceneTables(tracks=tracks), merged, "25/1", cuts=[25])
    assert scene.roles_at(20)[0] == 1
    focus, _pair, involved = scene.roles_at(30)
    assert focus is None
    assert {5, 6} <= involved
    assert not [mark for mark in scene.annotate(30).players if mark.emphasis == "focus"]
    single = {key: value for key, value in merged.items() if key != "merged_parts"}
    assert (
        HighlightScene(SceneTables(tracks=tracks), single, "25/1", cuts=[25]).roles_at(30)[0] == 5
    )


def test_scene_focus_before_the_cut_of_a_merged_highlight_stays_at_the_ball() -> None:
    # In a merged or extended highlight, seconds before the cut into the key frame's shot
    # the play can be about other players than the key frame's: there the focus is marked
    # only within FOCUS_BALL_MAX_M of the ball on the grass.
    tracks = pd.concat(
        [_tracks(range(0, 25), PLAYERS), _tracks(range(25, 40), JOINED_PLAYERS)],
        ignore_index=True,
    )
    away = 5.0 + FOCUS_BALL_MAX_M + 1.0
    rows = {f: (5.5, 3.0) for f in range(0, 10)}  # at the focus player's feet
    rows.update({f: (away, 3.0) for f in range(10, 40)})
    tables = SceneTables(tracks=tracks, ball=_ball(rows))
    event = {"cue": "strike", "frame": 30, "track_ids": [5], "partner_track_id": 6}
    extended = _duel(
        start_frame=0, peak_frame=30, end_frame=39, track_ids=[1, 2, 5, 6],
        cues={"duel": {"contribution": 0.1, "frame": 5, "track_ids": [1, 2]},
              "cut_emphasis": {"contribution": 0.1, "cut_frame": 25, "joined_shot_event": event}},
    )  # fmt: skip
    scene = HighlightScene(tables, extended, "25/1", cuts=[25], pitch_frames=range(0, 40))
    assert scene.roles_at(4)[0] == 1
    assert scene.roles_at(15)[0] is None, "the ball is elsewhere"
    assert [mark.emphasis for mark in scene.annotate(15).players if mark.box[0] == 100.0] == [
        "normal"
    ], "still involved"
    assert scene.roles_at(30)[0] == 5, "the key frame's shot is unchanged"


def test_scene_duel_line_only_while_the_ball_is_at_the_pair() -> None:
    # The 争抢 line stands for the duel cue: both players near the ball. Once the ball has
    # gone, the line is the focus player's nearest opponent, within the duel's second too.
    rows = {f: (5.5, 3.2) for f in range(0, 20)}
    rows.update({f: (15.0, 3.2) for f in range(20, 40)})
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS), ball=_ball(rows))
    duel = _duel(start_frame=0, peak_frame=20, end_frame=39)  # the cue at 20, +/-25 frames
    scene = HighlightScene(tables, duel, "25/1", pitch_frames=range(0, 40))
    metres = f"{math.hypot(1.5, 0.5):.1f} m"
    assert [line.label for line in scene.annotate(10).lines] == [f"争抢 · {metres}"]
    assert [line.label for line in scene.annotate(30).lines] == [f"最近对手 · {metres}"]
    plain = HighlightScene(SceneTables(tracks=tables.tracks), duel, "25/1")
    assert [line.label for line in plain.annotate(30).lines] == ["争抢"], "no ball track"


def test_scene_key_frame_of_a_ball_event_is_about_that_events_player() -> None:
    # The key frame was chosen at a ball event of track 2 (a touch, the ball leaving its
    # feet): without key_focus_track, the freeze frame's spotlight and the data card are
    # about that player, not the attacking team's first involved player.
    moments = [{"frame": 20, "kind": "touch", "track_ids": [2]},
               {"frame": 20, "kind": "ball_away", "track_ids": [2]}]  # fmt: skip
    attack = _duel(
        type="attack_in_box", track_ids=[2, 1], key_moments=moments, key_focus_track=None,
        cues={"box_pressure": {"team": "left", "contribution": 0.2}},
    )  # fmt: skip
    scene = HighlightScene(SceneTables(tracks=_tracks(range(0, 40), PLAYERS)), attack, "25/1")
    assert scene.roles_at(20)[0] == 2
    assert [scene.roles_at(f)[0] for f in range(5, 31)] == [2] * 26, "slow window"
    assert scene.roles_at(4)[0] == scene.roles_at(31)[0] == 1, "outside the window"
    cut = HighlightScene(scene.tables, attack, "25/1", cuts=[18, 23])
    assert cut.roles_at(17)[0] == cut.roles_at(23)[0] == 1, "event focus never crosses a cut"
    assert [cut.roles_at(f)[0] for f in range(18, 23)] == [2] * 5
    spotlight = scene.freeze_layer(()).spotlight
    assert spotlight is not None
    assert spotlight[0] == pytest.approx(165.0)
    keyed = {**attack, "key_focus_track": 1}
    assert (
        HighlightScene(SceneTables(tracks=_tracks(range(0, 40), PLAYERS)), keyed, "25/1").roles_at(
            20
        )[0]
        == 1
    ), "gbf.moments' key_focus_track wins"


def test_scene_speed_label_holds_its_last_valid_value_greyed() -> None:
    # Track 1's speed is valid on frames 20-22 and 35, not on 23-34: the label holds the
    # last valid speed, greyed (label_held), with the evidence of its frame, over at most
    # TRACK_HOLD_MAX_GAP frames, so slow motion does not blink. A player whose peak speed
    # gbf.measure withholds names the observed peak lower bound explicitly.
    frames = list(range(20, 36))
    valid = [f in (20, 21, 22, 35) for f in frames]
    kinematics = pd.DataFrame(
        {"frame": frames, "track_id": [1] * len(frames),
         "speed": [8.0 if f == 21 else 7.0 for f in frames],
         "valid": valid}
    )  # fmt: skip
    tables = SceneTables(tracks=_tracks(range(0, 40), PLAYERS), kinematics=kinematics)
    scene = HighlightScene(tables, _duel(), "25/1", pitch_frames=range(0, 40))

    def focus(frame: int, which: HighlightScene = scene) -> PlayerMark:
        return next(m for m in which.annotate(frame).players if m.emphasis == "focus")

    assert (focus(22).label, focus(22).label_held) == ("左侧球队 · 10号 · 25.2 km/h", None)
    held = focus(25)
    assert (held.label, held.label_held) == ("左侧球队 · 10号", " · 25.2 km/h")
    assert held.evidence_ids == ("kinematics.parquet#frame=22&track_id=1",)
    assert focus(22 + TRACK_HOLD_MAX_GAP).label_held is not None
    assert focus(23 + TRACK_HOLD_MAX_GAP).label_held is None, "a longer stretch: no speed"
    assert ("player_label", "左侧球队 · 10号 · 25.2 km/h", held.evidence_ids) in annotation_texts(
        scene.annotate(25)
    )
    bound = HighlightScene(
        tables, _duel(), "25/1", pitch_frames=range(0, 40), peak_speed_withheld=[1]
    )
    assert focus(22, bound).label == "左侧球队 · 10号 · 峰值≥28.8 km/h"
    assert focus(25, bound).label_held == " · 峰值≥28.8 km/h"
    assert focus(25, bound).evidence_ids == ("kinematics.parquet#frame=21&track_id=1",)


@pytest.mark.parametrize("unsafe", ["no_ground", "cut_off", "safe", "checked"])
def test_scene_unchecked_unsafe_tracks_are_neutral(unsafe: str) -> None:
    bystander = {
        "track_id": 7,
        "box": (300, 100, 330, 180 if unsafe == "cut_off" else 170),
        "team": "right",
        "pitch": (math.nan, math.nan) if unsafe in {"no_ground", "checked"} else (6.0, 3.0),
    }
    tracks = _tracks(range(0, 40), [*PLAYERS, bystander])
    tables = SceneTables(
        tracks=tracks,
        team_verdicts={(7, "right"): "consistent" if unsafe == "checked" else "unchecked"},
    )
    scene = HighlightScene(tables, _duel(), "25/1", frame_size=(640, 180))
    (mark,) = [m for m in scene.annotate(20).players if m.box[0] == 300.0]
    assert mark.team == ("right" if unsafe in {"safe", "checked"} else None)


@pytest.mark.parametrize("withheld", [False, True])
def test_speed_labels_reject_unsafe_chord_endpoints(withheld: bool) -> None:
    tracks = _tracks(range(40), PLAYERS)
    tracks.loc[(tracks.track_id == 1) & (tracks.frame == 20), "y2"] = 180.0
    kin = pd.DataFrame(
        {"frame": [14, 15], "track_id": [1, 1], "speed": [7.0, 9.0], "valid": [True, True]}
    )
    scene = HighlightScene(
        SceneTables(tracks=tracks, kinematics=kin, team_verdicts={(1, "left"): "unchecked"}),
        _duel(),
        frame_size=(640, 180),
        pitch_frames=range(40),
        peak_speed_withheld=[1] if withheld else [],
    )

    def subject(frame: int) -> PlayerMark:
        return next(p for p in scene.annotate(frame).players if p.box[0] == 100.0)

    assert "32.4" not in str(subject(15)), "safe centre with unsafe chord endpoint"
    assert "25.2" in (subject(15).label_held or "")
    assert subject(15).evidence_ids == ("kinematics.parquet#frame=14&track_id=1",)
    assert subject(16).evidence_ids == ("kinematics.parquet#frame=14&track_id=1",)
    if withheld:
        assert scene._peak_bounds[1] == (14, 7.0)


def test_explicit_unsafe_focus_has_no_spotlight_or_skeleton() -> None:
    bystander = {"track_id": 7, "box": (300, 100, 330, 180), "team": "right", "pitch": (6.0, 3.0)}
    points = np.array([[315.0, 140.0, 0.9]] * 26).reshape(-1).tolist()
    tables = SceneTables(
        tracks=_tracks(range(40), [bystander]),
        pose=pd.DataFrame({"frame": [20], "track_id": [7], "keypoints": [points]}),
        team_verdicts={(7, "right"): "unchecked"},
    )
    scene = HighlightScene(
        tables, _duel(track_ids=[7], key_focus_track=7), frame_size=(640, 180), focus_track=7
    )
    assert scene.roles_at(20)[0] is None
    assert not scene.annotate(20).skeletons
    assert scene.freeze_layer(()).spotlight is None


# ---------------------------------------------------------------------------- 3D facing
FACING_GATES = FacingGates(min_conf=0.5, max_sigma_deg=15.0, min_box_px=80.0)
"""Gates with the values of gbf.measure's defaults for body-orientation (gbf.pipeline passes
gbf.measure's own values; tests/test_pipeline.py checks that)."""

FOCUS_AT = (5.0, 3.0)
"""SoccerNet metres of the focus player of the facing scenes (84 px tall at 1080p)."""


def _calib(camera: Camera, frames: list[int]) -> pd.DataFrame:
    n = len(frames)
    return pd.DataFrame(
        {"frame_index": frames, "K": [camera.K.reshape(-1).tolist()] * n,
         "R": [camera.R.reshape(-1).tolist()] * n, "C": [camera.C.tolist()] * n,
         "width": [camera.width] * n, "height": [camera.height] * n,
         "pitch_usable": [True] * n}
    )  # fmt: skip


def _standing_box(camera: Camera, x: float, y: float) -> tuple[float, float, float, float]:
    """The box of a 1.80 m player standing at SoccerNet (x, y) on the camera's frame."""
    foot, _, _ = project(camera, np.array([x, y, 0.0]))
    head, _, _ = project(camera, np.array([x, y, -1.8]))
    half = 0.22 * (foot[1] - head[1])
    return (float(foot[0] - half), float(head[1]), float(foot[0] + half), float(foot[1]))


def _facing_scene(
    facings: Mapping[int, float],
    *,
    camera: Camera | None = None,
    gates: FacingGates = FACING_GATES,
    pitch_frames: list[int] | None = None,
    calibrated: list[int] | None = None,
    posed: list[int] | None = None,
    lean: float | None = None,
    **row: Any,
) -> tuple[HighlightScene, Camera]:
    """A highlight about player 1 (the focus) with a teammate, player 2, beside it; both
    have a box, a pose and a lift row on every frame of ``facings`` (player 1 facing that
    way, player 2 facing +x, both passing ``gates``). ``row`` overrides player 1's lift
    columns (facing_deg, method, facing_conf, facing_sigma_deg); every frame is a pitch
    frame with a camera unless ``pitch_frames`` or ``calibrated`` say otherwise, and
    player 1 has a pose on ``posed`` (default every frame): no keypoints, or with ``lean``
    its hips and shoulders, the trunk leaning that many degrees from upright in the picture
    (+ to the right)."""
    camera = camera or synthetic_broadcast_camera(1920, 1080)
    frames = sorted(facings)
    at = {1: FOCUS_AT, 2: (8.0, 3.5)}
    tracks, lift, pose = [], [], []
    for frame in frames:
        for track, (x, y) in at.items():
            x1, y1, x2, y2 = _standing_box(camera, x, y)
            tracks.append(
                {"frame": frame, "track_id": track, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                 "role": "player", "team": "left", "pitch_x": x, "pitch_y": y}
            )  # fmt: skip
            own = {"facing_deg": facings[frame], **row} if track == 1 else {"facing_deg": 0.0}
            lift.append(
                {"frame": frame, "track_id": track, "method": LIFTED, "facing_conf": 0.9,
                 "facing_sigma_deg": 5.0, **own}
            )  # fmt: skip
            keypoints = np.full((26, 3), math.nan)
            if track == 1 and lean is not None:
                hip_v, shoulder_v = y1 + 0.5 * (y2 - y1), y1 + 0.2 * (y2 - y1)
                shift = (hip_v - shoulder_v) * math.tan(math.radians(lean))
                centre = (x1 + x2) / 2
                for name, u, v in (
                    ("left_hip", centre + 4, hip_v), ("right_hip", centre - 4, hip_v),
                    ("left_shoulder", centre + 6 + shift, shoulder_v),
                    ("right_shoulder", centre - 6 + shift, shoulder_v),
                ):  # fmt: skip
                    keypoints[HALPE26_INDEX[name]] = (u, v, 0.9)
            if track == 2 or posed is None or frame in posed:
                pose.append(
                    {"frame": frame, "track_id": track, "keypoints": keypoints.reshape(-1).tolist()}
                )
    tables = SceneTables(
        tracks=pd.DataFrame(tracks),
        calib=_calib(camera, frames if calibrated is None else calibrated),
        pose=pd.DataFrame(pose),
        lift=pd.DataFrame(lift),
    )
    highlight = {
        "id": "h1", "type": "ball_strike", "start_frame": frames[0], "peak_frame": frames[0],
        "end_frame": frames[-1], "score": 0.5, "track_ids": [1, 2], "cues": {},
    }  # fmt: skip
    scene = HighlightScene(
        tables,
        highlight,
        "25/1",
        pitch_frames=frames if pitch_frames is None else pitch_frames,
        facing_gates=gates,
    )
    return scene, camera


@pytest.mark.parametrize(
    ("facing", "towards"),
    [(0.0, (1.0, 0.0)), (90.0, (0.0, -1.0)), (180.0, (-1.0, 0.0)), (-90.0, (0.0, 1.0))],
    ids=["+x-right", "far-touchline-up", "-x-left", "near-touchline-down"],
)
def test_scene_draws_the_3d_facing_back_on_the_frame(
    facing: float, towards: tuple[float, float]
) -> None:
    # gbf.lift3d's facing is in pitch_zup degrees from +x, counter-clockwise: +90 faces the
    # far touchline (SoccerNet y = -34), the top of the main camera's picture. The arrow runs
    # FACING_ARROW_LENGTH_M metres on the grass from the foot point, both ends projected
    # with the frame's camera, and says it is a 3D reference.
    scene, camera = _facing_scene({20: facing})
    annotations = scene.annotate(20)
    (arrow,) = annotations.arrows
    heading, reach = math.radians(facing), FACING_ARROW_LENGTH_M
    x, y = FOCUS_AT
    start, _, _ = project(camera, np.array([x, y, 0.0]))
    tip = [x + reach * math.cos(heading), y - reach * math.sin(heading), 0.0]
    end, _, _ = project(camera, np.array(tip))
    assert np.allclose(arrow.start, start)
    assert np.allclose(arrow.end, end)
    du, dv = end - start
    assert (du * towards[0] + dv * towards[1]) / math.hypot(du, dv) > 0.95
    assert (arrow.label, FACING_LABEL_ZH) == ("朝向（3D 参考）", "朝向（3D 参考）")
    assert arrow.dashed
    assert arrow.badges == (Badge("未验证", "unverified"),)
    assert arrow.evidence_ids == ("lift3d.parquet#frame=20&track_id=1",)
    # an estimate is shown: the frame's 未验证 joins the badge row with the arrow drawn
    assert (annotations.badges, annotations.arrow_badges) == ((), (Badge("未验证", "unverified"),))
    listed = annotation_texts(annotations, (1920, 1080))
    assert [text for kind, text, ids in listed if kind == "badge" and not ids] == ["未验证"]


def test_scene_draws_the_facing_on_pitch_frames_with_a_camera_only() -> None:
    scene, _ = _facing_scene(
        {20: 0.0, 21: 0.0, 22: 0.0}, pitch_frames=[20, 22], calibrated=[20, 21]
    )
    assert len(scene.annotate(20).arrows) == 1
    assert scene.annotate(21).arrows == (), "not a pitch frame"
    assert scene.annotate(22).arrows == (), "a pitch frame without a camera"
    off_pitch = scene.annotate(21)
    assert [badge.text for badge in off_pitch.badges] == ["仅画面平面"]


@pytest.mark.parametrize(
    ("row", "drawn"),
    [
        ({}, True),
        ({"facing_conf": 0.5}, True),  # facing_conf >= facing_min_conf
        ({"facing_conf": 0.49}, False),
        ({"facing_conf": math.nan}, False),
        ({"facing_sigma_deg": 14.99}, True),  # facing_sigma_deg < facing_max_sigma_deg
        ({"facing_sigma_deg": 15.0}, False),
        ({"facing_sigma_deg": math.nan}, False),
        ({"facing_deg": math.nan}, False),
        ({"method": "image_plane_only"}, False),
    ],
    ids=lambda value: str(value),
)
def test_scene_draws_a_facing_only_where_it_passes_the_display_gates(
    row: dict[str, Any], drawn: bool
) -> None:
    # The gates of gbf.measure's body-orientation, frame by frame: a lifted row, a finite
    # facing, facing_conf >= min_conf and a finite facing_sigma_deg < max_sigma_deg.
    assert render_module.FacingGates is FacingGates is measure.FacingGates, "gbf.measure's gate"
    assert LIFTED == "height_prior_calibrated"
    scene, _ = _facing_scene({20: 0.0}, **row)
    assert bool(scene.annotate(20).arrows) is drawn


def test_scene_draws_a_facing_only_on_a_box_of_the_minimum_height() -> None:
    camera = synthetic_broadcast_camera(1920, 1080)
    x1, y1, x2, y2 = _standing_box(camera, *FOCUS_AT)
    tall = FacingGates(0.5, 15.0, y2 - y1)
    taller = FacingGates(0.5, 15.0, y2 - y1 + 0.01)
    assert len(_facing_scene({20: 0.0}, gates=tall)[0].annotate(20).arrows) == 1
    assert _facing_scene({20: 0.0}, gates=taller)[0].annotate(20).arrows == ()


@pytest.mark.parametrize(
    ("lean", "facing", "drawn"),
    [
        (-30.0, 0.0, False),  # leans to the picture's left, faces right: reversed
        (-30.0, 180.0, True),  # faces the way it leans
        (30.0, 0.0, True),
        (30.0, 180.0, False),
        (-19.0, 0.0, True),  # upright enough: nothing to tell front from back
        (-30.0, 90.0, True),  # faces up the picture: neither side
        (None, 0.0, True),  # no trunk seen
    ],
    ids=lambda value: str(value),
)
def test_scene_leaves_out_a_facing_reversed_against_the_lean(
    lean: float | None, facing: float, drawn: bool
) -> None:
    # gbf.measure's per-frame gate of body-orientation, its front-back reversal check
    # included: a player whose trunk leans past upright to one side of the picture lunges,
    # falls or sprints that way; a lifted facing pointing to the other side is the
    # height-prior lift's reversal, left out of the card and of the frame alike.
    scene, _ = _facing_scene({20: facing}, lean=lean)
    assert bool(scene.annotate(20).arrows) is drawn


def test_scene_draws_only_the_focus_players_own_facing() -> None:
    # The teammate's facing passes the gates too, but only the focus player's is drawn; and
    # only on a frame where the focus player has a row and a pose of its own: a filled-in
    # box (TRACK_HOLD_MAX_GAP) or a filled-in pose measures nothing.
    scene, _ = _facing_scene({20: 0.0, 21: 180.0, 22: 0.0, 23: 0.0}, posed=[20, 21, 23])
    (arrow,) = scene.annotate(20).arrows
    assert arrow.evidence_ids == ("lift3d.parquet#frame=20&track_id=1",)
    assert scene.annotate(22).arrows == (), "the pose of frame 22 is filled in"
    tables = scene.tables
    tracks = tables.tracks[~((tables.tracks["track_id"] == 1) & (tables.tracks["frame"] == 21))]
    filled = HighlightScene(
        replace(tables, tracks=tracks.reset_index(drop=True)), scene.highlight, "25/1",
        pitch_frames=[20, 21, 22, 23], facing_gates=FACING_GATES,
    )  # fmt: skip
    assert any(m.emphasis == "focus" for m in filled.annotate(21).players), "drawn, filled in"
    assert filled.annotate(21).arrows == (), "but no facing on a filled-in box"
    assert len(filled.annotate(23).arrows) == 1


def test_scene_needs_the_facing_gates_with_a_lift_table() -> None:
    scene, _ = _facing_scene({20: 0.0})
    with pytest.raises(RenderError, match="facing_gates is required"):
        HighlightScene(scene.tables, scene.highlight, "25/1")
    with pytest.raises(RenderError, match="lift misses column"):
        SceneTables(tracks=scene.tables.tracks, lift=pd.DataFrame({"frame": [20]}))
    with pytest.raises(MeasureError) as excinfo:
        FacingGates(min_conf=-0.1, max_sigma_deg=math.nan, min_box_px="80")  # type: ignore[arg-type]
    assert len(excinfo.value.problems) == 3
    plain = HighlightScene(replace(scene.tables, lift=None), scene.highlight, "25/1")
    assert plain.annotate(20).arrows == (), "no lift table, no facing"


# (forward, left, height) in metres of the Halpe-26 keypoints of a standing 1.80 m player
# facing "forward" (the body template of tests/test_lift3d.py).
BODY = {
    "nose": (0.10, 0.0, 1.65), "left_eye": (0.08, 0.035, 1.65),
    "right_eye": (0.08, -0.035, 1.65), "left_ear": (0.0, 0.075, 1.65),
    "right_ear": (0.0, -0.075, 1.65), "left_shoulder": (0.0, 0.19, 1.45),
    "right_shoulder": (0.0, -0.19, 1.45), "left_elbow": (0.0, 0.22, 1.10),
    "right_elbow": (0.0, -0.22, 1.10), "left_wrist": (0.03, 0.23, 0.85),
    "right_wrist": (0.03, -0.23, 0.85), "left_hip": (0.0, 0.10, 0.95),
    "right_hip": (0.0, -0.10, 0.95), "left_knee": (0.02, 0.10, 0.50),
    "right_knee": (0.02, -0.10, 0.50), "left_ankle": (0.0, 0.10, 0.09),
    "right_ankle": (0.0, -0.10, 0.09), "head": (0.0, 0.0, 1.80), "neck": (0.0, 0.0, 1.55),
    "hip": (0.0, 0.0, 0.95), "left_big_toe": (0.17, 0.08, 0.03),
    "right_big_toe": (0.17, -0.08, 0.03), "left_small_toe": (0.15, 0.13, 0.03),
    "right_small_toe": (0.15, -0.13, 0.03), "left_heel": (-0.05, 0.10, 0.05),
    "right_heel": (-0.05, -0.10, 0.05),
}  # fmt: skip


def _seen_body(camera: Camera, x: float, y: float, facing: float) -> np.ndarray:
    """(26, 3) keypoints [u, v, score] of the template player standing at SoccerNet (x, y)
    facing ``facing`` (pitch_zup degrees), as a pose model would give them: the face is
    hidden when the player turns away from the camera."""
    heading = math.radians(facing)
    forward = np.array([math.cos(heading), math.sin(heading), 0.0])
    left = np.array([-math.sin(heading), math.cos(heading), 0.0])
    base = np.array([x, -y, 0.0])  # pitch_zup
    joints = np.zeros((26, 3))
    for name, (f, side, height) in BODY.items():
        joints[HALPE26_INDEX[name]] = base + f * forward + side * left + [0.0, 0.0, height]
    uv, _, valid = project(camera, pitch_zup_to_soccernet(joints))
    assert valid.all()
    keypoints = np.column_stack([uv, np.full(26, 0.9)])
    centre = pitch_zup_to_soccernet(camera.C)
    to_camera = math.degrees(math.atan2(centre[1] - base[1], centre[0] - base[0]))
    if abs((facing - to_camera + 180.0) % 360.0 - 180.0) > 110.0:
        keypoints[[HALPE26_INDEX[n] for n in ("nose", "left_eye", "right_eye")], 2] = 0.1
    return keypoints


@pytest.mark.parametrize("facing", [0.0, 90.0, 180.0, -90.0, -135.0])
def test_scene_arrow_points_where_the_lifted_body_faces(facing: float) -> None:
    # The whole way: a synthetic player seen by the main camera, lifted by gbf.lift3d, and
    # its facing drawn back on the frame. The arrow points the way the feet point in the
    # picture (the 2D keypoints from the heels to the toes).
    camera = synthetic_broadcast_camera(1920, 1080)
    keypoints = _seen_body(camera, *FOCUS_AT, facing)
    result = lift_height_prior(keypoints, camera)
    assert result.method == LIFTED
    assert abs((result.facing_deg - facing + 180.0) % 360.0 - 180.0) < 3.0
    u, v = keypoints[:, 0], keypoints[:, 1]
    tracks = pd.DataFrame(
        {"frame": [20], "track_id": [1], "x1": [u.min()], "y1": [v.min()], "x2": [u.max()],
         "y2": [v.max()], "role": ["player"], "team": ["left"], "pitch_x": [FOCUS_AT[0]],
         "pitch_y": [FOCUS_AT[1]]}
    )  # fmt: skip
    tables = SceneTables(
        tracks=tracks,
        calib=_calib(camera, [20]),
        pose=pd.DataFrame(
            {"frame": [20], "track_id": [1], "keypoints": [keypoints.reshape(-1).tolist()]}
        ),
        lift=lift_table([lift_row(20, 1, result)]),
    )
    highlight = {"id": "h1", "type": "sprint", "start_frame": 20, "peak_frame": 20,
                 "end_frame": 20, "track_ids": [1], "cues": {}}  # fmt: skip
    scene = HighlightScene(
        tables, highlight, "25/1", pitch_frames=[20], facing_gates=FacingGates(0.0, 180.0, 0.0)
    )
    (arrow,) = scene.annotate(20).arrows
    toes = keypoints[[HALPE26_INDEX[n] for n in ("left_big_toe", "right_big_toe")], :2]
    heels = keypoints[[HALPE26_INDEX[n] for n in ("left_heel", "right_heel")], :2]
    feet = toes.mean(axis=0) - heels.mean(axis=0)
    shaft = np.subtract(arrow.end, arrow.start)
    assert float(feet @ shaft) / (np.linalg.norm(feet) * np.linalg.norm(shaft)) > 0.95


def test_render_edit_manifest_lists_the_facing_arrow(
    barcode_video: Path, unicode_tmp_path: Path, style: Style
) -> None:
    # Every frame of the edit shows the focus player's facing (a 320x180 frame: the box
    # gate is lowered to its players' size); the manifest lists the label with its 未验证
    # badge and the lift3d.parquet row of each frame.
    scene, _ = _facing_scene(
        {frame: 0.0 for frame in range(50)},
        camera=synthetic_broadcast_camera(320, 180),
        gates=FacingGates(0.5, 15.0, 10.0),
    )
    plan = plan_highlight_edit(
        5, 30, 20, slow_factor=2, slow_before_s=0.08, slow_after_s=0.04, freeze_s=0.08
    )
    result = render_edit(
        barcode_video, unicode_tmp_path / "朝向 facing.mp4", plan, annotate=scene.annotate,
        style=style,
    )  # fmt: skip
    texts = {
        (row["kind"], row["text"], tuple(row["evidence_ids"])): row["frames"]
        for row in result.manifest["texts"]
    }
    own = ("lift3d.parquet#frame=20&track_id=1",)
    assert texts[("arrow_label", FACING_LABEL_ZH, own)] == texts[("badge", "未验证", own)]
    shown = {ids[0] for kind, _text, ids in texts if kind == "arrow_label"}
    assert shown == {f"lift3d.parquet#frame={f}&track_id=1" for f in plan.source_frames()}
    total = sum(n for (kind, _text, _ids), n in texts.items() if kind == "arrow_label")
    assert total == plan.n_frames, "on every output frame"


def test_scene_drives_render_edit(
    barcode_video: Path, unicode_tmp_path: Path, style: Style
) -> None:
    kinematics = pd.DataFrame(
        {"frame": list(range(0, 50)), "track_id": [1] * 50, "speed": [5.0] * 50,
         "valid": [True] * 50}
    )  # fmt: skip
    tables = SceneTables(tracks=_tracks(range(0, 50), PLAYERS), kinematics=kinematics)
    scene = HighlightScene(tables, _duel(), "25/1", pitch_frames=range(0, 50))
    plan = plan_highlight_edit(
        5, 30, 20, slow_factor=2, slow_before_s=0.12, slow_after_s=0.08, freeze_s=0.2
    )
    out = unicode_tmp_path / "scene edit.mp4"
    result = render_edit(
        barcode_video, out, plan, annotate=scene.annotate,
        freeze_layer=scene.freeze_layer(
            (CardRow("速度", "18.0", "km/h", ("未验证",), ("ev-1",)),), dim=0.0
        ),  # dimming would darken the barcode cells; dimming has its own test
        style=style,
    )  # fmt: skip
    assert result.video.frames == plan.n_frames == probe(out).nb_frames_decoded
    assert _decoded_numbers(out) == plan.output_to_source()
    evidence = {tuple(row["evidence_ids"]) for row in result.manifest["texts"]}
    assert ("kinematics.parquet#frame=20&track_id=1",) in evidence
    assert ("ev-1",) in evidence


def test_render_highlight_writes_clip_manifest_and_key_frames(
    barcode_video: Path, unicode_tmp_path: Path, style: Style
) -> None:
    tables = SceneTables(tracks=_tracks(range(0, 50), PLAYERS))
    scene = HighlightScene(tables, _duel(), "25/1")
    out = unicode_tmp_path / "run 输出"
    rendered = render_highlight(
        barcode_video, out, scene, card_rows=(CardRow("距离", "1.6", "m", ("未验证",)),),
        style=style, slow_factor=2, slow_before_s=0.08, slow_after_s=0.04, freeze_s=0.12,
    )  # fmt: skip
    assert (rendered.clip, rendered.manifest) == ("clips/h1.mp4", "clips/h1.render.json")
    for relative in (rendered.clip, rendered.manifest, rendered.keyframe, rendered.keyframe_card):
        assert (out / relative).is_file(), relative
    assert rendered.edit.video.frames == 26 + 3 * 2 + 3 + 1 * 2
    source = next(frame for index, frame in iter_frames(barcode_video) if index == 20)
    expected = draw_annotations(source, scene.annotate(20), style)
    assert np.array_equal(read_image_rgb(out / rendered.keyframe), expected)
    card = read_image_rgb(out / rendered.keyframe_card)
    assert not np.array_equal(card, expected), "the card variant carries the data card"
    with pytest.raises(FileExistsError):
        render_highlight(barcode_video, out, scene, style=style)
    with pytest.raises(RenderError, match="file name"):
        render_highlight(barcode_video, out, HighlightScene(tables, _duel(id="../x"), "25/1"))


def test_render_highlight_keeps_slow_motion_inside_the_peak_shot(
    barcode_video: Path, unicode_tmp_path: Path, style: Style
) -> None:
    tables = SceneTables(tracks=_tracks(range(0, 50), PLAYERS))
    scene = HighlightScene(tables, _duel(), "25/1", cuts=[18, 22])  # the peak 20 in 18..21
    rendered = render_highlight(
        barcode_video, unicode_tmp_path / "cut", scene, style=style,
        slow_factor=2, slow_before_s=0.2, slow_after_s=0.2, freeze_s=0.08,
    )  # fmt: skip
    segments = rendered.edit.manifest["segments"]
    slow = [(row["src_start"], row["src_end"]) for row in segments if row["kind"] == "slow"]
    assert slow == [(18, 20), (21, 21)], "5 frames each side, clipped to the shot"
    assert rendered.edit.video.frames == 26 + 3 * 2 + 2 + 1 * 2
