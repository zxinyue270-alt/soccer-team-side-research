"""Tests for gbf.report: the Chinese HTML report and its timeline chart (synthetic inputs)."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest
from PIL import Image, ImageDraw

from gbf import report as report_module
from gbf.fonts import FontInfo, find_cjk_font
from gbf.measure import METRICS, load_metric_registry, metric_label
from gbf.render import FACING_ARROW_LENGTH_M, Badge
from gbf.report import (
    HIGHLIGHT_TYPES_ZH,
    REFERENCE_3D_METRICS,
    TIMELINE_DARK_NAME,
    TIMELINE_NAME,
    ArtifactRef,
    ClipLink,
    ImageLink,
    ReportInputError,
    ReportMeta,
    card_rows,
    draw_timeline,
    format_number,
    metric_row,
    write_report,
)
from gbf.util.hashing import sha256_file
from gbf.util.io import write_image_png

N_FRAMES = 500


@pytest.fixture(scope="module")
def font() -> FontInfo:
    return find_cjk_font("regular")


def _meta(font: FontInfo, **changes: object) -> ReportMeta:
    fields: dict[str, object] = {
        "clip_name": "synthetic clip.mp4",
        "fps": "25/1",
        "frame_count": N_FRAMES,
        "width": 320,
        "height": 180,
        "source_sha256": "ab" * 32,
        "cuts": (120, 300),
        "score": tuple(0.5 + 0.4 * math.sin(i / 30) for i in range(N_FRAMES)),
        "calibration": {"frames": N_FRAMES, "pitch_usable": 400, "threshold_px": 8.0},
        "audio_source": "original.m4a",
        "restrictions": ("no_share",),
        "artifacts": (ArtifactRef("tracks.parquet", "cd" * 32, 1234),),
        "tools": (("ffmpeg", "test"),),
        "font": font,
    }
    fields.update(changes)
    return ReportMeta(**fields)  # type: ignore[arg-type]


def _highlights() -> list[dict]:
    return [
        {
            "id": "h2",
            "type": "duel_tackle",
            "start_frame": 300,
            "peak_frame": 340,
            "end_frame": 380,
            "score": 1.25,
            "cues": {"grammar": 1.0, "attackers_in_box": 2},
            "track_ids": [4, 9],
            "title_zh": "底线一对一",
            "summary_zh": "防守球员贴身逼抢。",
        },
        {
            "id": "h1",
            "type": "attack_in_box",
            "start_frame": 40,
            "peak_frame": 100,
            "end_frame": 119,
            "score": 2.5,
            "cues": {"audio_z": 2.37},
            "track_ids": [7],
            "title_zh": "<script>alert('x')</script> & 禁区",
            "summary_zh": "a < b & c > d",
        },
        {
            "id": "h3",
            "type": "sprint",
            "start_frame": 420,
            "peak_frame": 450,
            "end_frame": 499,
            "score": 0.9,
            "cues": {},
            "track_ids": [],
            "title_zh": "反击冲刺",
        },
    ]


def _records() -> list[dict]:
    return [
        {"evidenceId": "ev-speed", "highlightId": "h1", "metric": "peak-speed",
         "label_zh": "峰值速度", "value": 7.6, "unit": "m/s", "tier": "A",
         "error": {"kind": "unmeasured"}, "frame": 100, "trackId": 7},
        {"evidenceId": "ev-dist", "metric": "nearest-opponent", "label_zh": "最近对手距离",
         "value": 3.94, "unit": "m", "tier": "A", "error": {"kind": "unmeasured"}, "frame": 350},
        {"evidenceId": "ev-knee", "highlightId": "h2", "metric": "knee-angle",
         "label_zh": "膝角", "value": 152.3, "unit": "deg", "tier": "B", "plane": "image"},
        {"evidenceId": "ev-lean", "highlightId": "h2", "metric": "trunk-lean",
         "label_zh": "躯干前倾", "value": None, "status": "unavailable",
         "reasonCode": "no_calibration", "tier": "A_candidate"},
        {"evidenceId": "ev-blocked", "highlightId": "h2", "metric": "knee-angular-velocity",
         "label_zh": "膝角速度", "value": 4321.0, "unit": "deg/s", "tier": "C"},
        {"evidenceId": "ev-orphan", "metric": "orphan-metric", "label_zh": "孤立记录",
         "value": 1.5, "frame": 5},
    ]  # fmt: skip


@pytest.fixture
def out_dir(unicode_tmp_path: Path) -> Path:
    """A report directory with synthetic key frames and clips (clips are dummy bytes: the
    report only links them)."""
    out = unicode_tmp_path / "报告 run"
    (out / "keyframes").mkdir(parents=True)
    (out / "clips").mkdir()
    for name, colour in (("h1", (200, 40, 40)), ("h2", (40, 200, 40))):
        frame = np.zeros((90, 160, 3), np.uint8)
        frame[:] = colour
        write_image_png(out / "keyframes" / f"{name} 关键帧.png", frame)
    (out / "clips" / "h1 原速+慢放#1.mp4").write_bytes(b"synthetic clip bytes")
    return out


def _write(out_dir: Path, font: FontInfo, **meta_changes: object) -> Path:
    return write_report(
        out_dir,
        _highlights(),
        _records(),
        {"h1": "keyframes/h1 关键帧.png", "h2": [out_dir / "keyframes" / "h2 关键帧.png"]},
        {"h1": [ClipLink("原速 + 慢放 + 定格", "clips/h1 原速+慢放#1.mp4")]},
        _meta(font, **meta_changes),
    )


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: list[str] = []
        self.tags: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.add(tag)
        for name, value in attrs:
            if name in ("src", "href", "srcset") and value is not None:
                self.links.append(value)
            if name == "id" and value is not None:
                self.ids.append(value)


def _parse(html_text: str) -> _Links:
    parser = _Links()
    parser.feed(html_text)
    return parser


class _Text(HTMLParser):
    """The text a reader sees (no tags, attributes or style sheet)."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.in_style = self.in_style or tag == "style"

    def handle_endtag(self, tag: str) -> None:
        self.in_style = self.in_style and tag != "style"

    def handle_data(self, data: str) -> None:
        if not self.in_style:
            self.parts.append(data)


def _visible(html_text: str) -> str:
    parser = _Text()
    parser.feed(html_text)
    return "".join(parser.parts)


def _section(html_text: str, start: str, end: str) -> str:
    """Visible text between the elements with ids ``start`` and ``end``."""
    return _visible(html_text[html_text.index(f'id="{start}"') : html_text.index(f'id="{end}"')])


def test_report_has_every_highlight_in_peak_order(out_dir: Path, font: FontInfo) -> None:
    path = _write(out_dir, font)
    assert path == out_dir / "report.html"
    text = path.read_text(encoding="utf-8")
    parsed = _parse(text)
    cards = [ident for ident in parsed.ids if ident.startswith("hl-")]
    assert cards == ["hl-h1", "hl-h2", "hl-h3"], "ordered by peak frame"
    for title in ("底线一对一", "反击冲刺", "禁区"):
        assert title in text
    assert "00:01.60–00:04.76" in text, "h1 runs from frame 40 to 119 at 25 fps"
    assert "帧 40–119" in text
    assert "关键帧 100（00:04.00）" in text
    # One name for the frozen frame: the summaries' 关键帧, also in the table and the chart.
    assert "峰值帧" not in _visible(text)
    assert "<th>关键帧</th>" in text


def test_report_uses_only_relative_links_to_existing_files(out_dir: Path, font: FontInfo) -> None:
    text = _write(out_dir, font).read_text(encoding="utf-8")
    links = _parse(text).links
    assert links
    files = [link for link in links if not link.startswith("#")]
    assert {"timeline.png", "timeline-dark.png"} <= set(files)
    for link in files:
        assert not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", link), link
        assert not link.startswith("/")
        assert ".." not in link.split("/")
        assert (out_dir / unquote(link)).is_file(), link
    assert "clips/h1%20%E5%8E%9F%E9%80%9F%2B%E6%85%A2%E6%94%BE%231.mp4" in files
    assert str(out_dir) not in text
    assert str(out_dir.parent) not in text
    assert not re.search(r"[A-Za-z]:[/\\]", text)


def test_report_escapes_all_text(out_dir: Path, font: FontInfo) -> None:
    text = _write(out_dir, font).read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt; &amp; 禁区" in text
    assert "a &lt; b &amp; c &gt; d" in text
    assert "script" not in _parse(text).tags


def test_report_metrics_badges_units_and_attachment(out_dir: Path, font: FontInfo) -> None:
    text = _write(out_dir, font).read_text(encoding="utf-8")
    h1 = text[text.index('id="hl-h1"') : text.index('id="hl-h2"')]
    h2 = text[text.index('id="hl-h2"') : text.index('id="hl-h3"')]
    assert "峰值速度" in h1
    assert ">27.4<" in h1, "7.6 m/s is shown as 27.4 km/h"
    assert "km/h" in h1
    assert "未验证" in h1
    assert "最近对手距离" in h2, "attached by frame 350 inside h2's window"
    assert "仅画面平面" in h2
    assert "实验" in h2
    assert "不可用" in h2
    assert "本帧没有可用标定" in h2
    assert "膝角速度" not in text, "C-tier records are never shown"
    assert ">4321<" not in text
    assert "另有 1 条 C 档记录" in h2
    assert "孤立记录" in text[text.index('id="appendix"') :]


def test_report_honest_notes_and_private_banner(out_dir: Path, font: FontInfo) -> None:
    text = _write(out_dir, font).read_text(encoding="utf-8")
    notes = text[text.index('id="notes"') : text.index('id="appendix"')]
    assert "自洽" in notes
    assert "rep_err_px ≤ 8.0 px" in notes
    assert "误差都还没有对照真值测量" in notes
    # Hiding unverified cards in public builds is ADR 0004's rule, not something done yet.
    assert "ADR 0004 规定公开版本要自动隐藏未验证的卡片，但这条规则还没有实现" in notes
    assert "公开版本会自动隐藏" not in notes
    assert "禁止分享" in notes
    assert "私有素材 · 禁止分享" in text[: text.index('id="overview"')]
    public = _write(out_dir, font, restrictions=()).read_text(encoding="utf-8")
    assert "私有素材 · 禁止分享" not in public
    assert "核对画面和数据的来源权利" in public


def test_report_appendix_hashes_every_linked_file(out_dir: Path, font: FontInfo) -> None:
    text = _write(out_dir, font).read_text(encoding="utf-8")
    appendix = text[text.index('id="appendix"') :]
    for relative in (
        TIMELINE_NAME,
        TIMELINE_DARK_NAME,
        "keyframes/h1 关键帧.png",
        "clips/h1 原速+慢放#1.mp4",
    ):
        assert sha256_file(out_dir / relative) in appendix
    assert "ab" * 32 in appendix
    assert "cd" * 32 in appendix


def test_report_is_deterministic(out_dir: Path, font: FontInfo) -> None:
    first = _write(out_dir, font).read_bytes()
    timeline = (out_dir / TIMELINE_NAME).read_bytes()
    second = _write(out_dir, font).read_bytes()
    assert first == second
    assert timeline == (out_dir / TIMELINE_NAME).read_bytes()


def test_report_rejects_bad_links_and_lists_every_problem(out_dir: Path, font: FontInfo) -> None:
    outside = out_dir.parent / "outside.png"
    write_image_png(outside, np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ReportInputError) as excinfo:
        write_report(
            out_dir,
            _highlights(),
            [],
            {"h1": "../outside.png", "h2": outside, "h3": "keyframes/missing.png", "hx": "a.png"},
            {"h1": ["https://example.org/clip.mp4"], "h2": ["keyframes/h1 关键帧.png"]},
            _meta(font),
        )
    problems = excinfo.value.problems
    assert len(problems) == 6
    joined = "\n".join(problems)
    for needle in ("'..'", "outside the report directory", "does not exist", "unknown highlight",
                   "not a relative file path", "must end in"):  # fmt: skip
        assert needle in joined
    assert not (out_dir / "report.html").exists()


def test_report_rejects_inconsistent_highlights_and_meta(out_dir: Path, font: FontInfo) -> None:
    bad = _highlights()
    bad[0]["peak_frame"] = 999
    bad[1]["id"] = "h2"
    del bad[2]["score"]
    with pytest.raises(ReportInputError) as excinfo:
        write_report(
            out_dir,
            {"fps": "30/1", "frame_count": N_FRAMES, "highlights": bad},
            [{"value": 1}],
            {},
            {},
            _meta(font, score=(1.0, 2.0)),
        )
    joined = "\n".join(excinfo.value.problems)
    for needle in ("fps 30/1 differs", "start <= peak <= end", "used twice", "misses score",
                   "measurements[0]", "meta.score has 2 values"):  # fmt: skip
        assert needle in joined


def test_metric_row_rules() -> None:
    speed = metric_row({"metric": "peak-speed", "value": 7.6, "unit": "m/s", "tier": "A"})
    assert (speed.value_text, speed.unit, speed.badges) == (
        "27.4",
        "km/h",
        (("unverified", "未验证"),),
    )
    angle = metric_row(
        {"metric": "knee", "value": 152.34, "unit": "deg", "tier": "B", "space": "image_plane"}
    )
    assert angle.value_text == "152"
    assert angle.unit == "°"
    assert [kind for kind, _ in angle.badges] == ["unverified", "experimental", "image_plane"]
    measured = metric_row(
        {"metricId": "x", "value": 0.456, "error": {"kind": "measured", "receiptId": "r1"}}
    )
    assert measured.value_text == "0.46"
    assert measured.badges == (("measured", "已测量"),)
    unmeasured_claim = metric_row({"metric": "x", "value": 1.0, "error": {"kind": "measured"}})
    assert unmeasured_claim.badges == (("unverified", "未验证"),), "measured needs a receipt"
    official = metric_row({"metric": "xg", "value": 0.12, "claimType": "provider_model"})
    assert official.badges == (("official", "官方数据"),)
    literature = metric_row({"metric": "x", "value": 2.0, "error": {"kind": "literature"}})
    assert literature.badges == (("unverified", "未验证（仅文献参考）"),)
    missing = metric_row(
        {"metric": "lean", "value": 3.0, "status": "unavailable", "reason_code": "occluded"}
    )
    assert (missing.value_text, missing.unit, missing.reason) == ("—", "", "被遮挡")
    assert missing.badges == (("unavailable", "不可用"),)
    blocked = metric_row({"metric": "omega", "value": 900.0, "tier": "C"})
    assert blocked.blocked
    assert blocked.value_text == "—"
    flag = metric_row({"metric": "contact", "value": True})
    assert flag.value_text == "是"
    with pytest.raises(ReportInputError):
        metric_row({"value": 1.0})
    with pytest.raises(ReportInputError):
        metric_row(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_values_of_the_3d_lift_read_as_a_reference() -> None:
    # Facing and trunk lean come from gbf.lift3d's height-prior skeleton (A_candidate: 3D,
    # ADR 0004). On the card and in the report they carry 3D 参考 after their other badges,
    # whether or not a value could be given.
    facing = metric_row(
        {"evidenceId": "ev-o", "metric_id": "body-orientation", "label_zh": "朝向与进攻方向夹角",
         "value": 51.13, "unit": "deg", "decimals": 0, "tier": "A_candidate",
         "error": {"kind": "unmeasured"}, "frame": 13, "track_id": 1}
    )  # fmt: skip
    assert facing.badges == (("unverified", "未验证"), ("reference_3d", "3D 参考"))
    lean = metric_row(
        {"metric_id": "trunk-lean-at-contact", "value": None, "status": "unavailable",
         "reasonCode": "lean_uncertain", "tier": "A_candidate"}
    )  # fmt: skip
    assert lean.badges == (("unavailable", "不可用"), ("reference_3d", "3D 参考"))
    untiered = metric_row({"metric": "body-orientation", "value": 12.0, "unit": "deg"})
    assert untiered.badges[-1] == ("reference_3d", "3D 参考"), "known by its metric id too"
    speed = metric_row({"metric": "peak-speed", "value": 7.6, "unit": "m/s", "tier": "A"})
    assert ("reference_3d", "3D 参考") not in speed.badges, "the original frames' values"
    (card,) = card_rows([facing], key_frame=13, name_tracks=True, focus_track=1)
    assert (card.label, card.value, card.unit) == ("朝向与进攻方向夹角", "51", "°")
    assert card.badges == (Badge("未验证", "unverified"), Badge("3D 参考", "reference_3d"))


def test_the_3d_reference_metrics_are_gbf_measures_a_candidate_metrics() -> None:
    registry = load_metric_registry()
    lifted = {name for name in METRICS if name in registry and registry[name].tier == "A_candidate"}
    assert lifted == set(REFERENCE_3D_METRICS) == {"body-orientation", "trunk-lean-at-contact"}


def test_report_says_which_numbers_come_from_the_3d_lift(out_dir: Path, font: FontInfo) -> None:
    text = _write(out_dir, font).read_text(encoding="utf-8")
    overview = _section(text, "overview", "timeline")
    assert "数据来源" in overview
    assert "原始画面：2D 姿态、球、球员框和跟踪" in overview
    assert "3D 提升（只作参考）：身体朝向和躯干前倾" in overview
    assert "所有结论以原始画面为准" in overview
    notes = _section(text, "notes", "appendix")
    assert "3D 只作参考，所有结论以原始画面为准。" in notes
    assert f"脚下画一支 {FACING_ARROW_LENGTH_M:g} 米长的箭头，标“朝向（3D 参考）”" in notes
    assert "其余内容（2D 姿态、球、球员位置、速度、距离、触球帧和关键时刻）都来自原始画面" in notes
    h2 = text[text.index('id="hl-h2"') : text.index('id="hl-h3"')]
    assert '<span class="chip reference_3d">3D 参考</span>' in h2, "the A_candidate record"
    assert ".chip.reference_3d::before" in text


def test_format_number() -> None:
    assert format_number(27.36) == "27.4"
    assert format_number(152.3) == "152"
    assert format_number(0.456) == "0.46"
    assert format_number(-0.0001, 0) == "0"
    assert format_number(float("nan")) == "—"
    assert format_number(3.14159, 3) == "3.142"


def test_timeline_chart_marks_windows_peaks_and_score(font: FontInfo) -> None:
    meta = _meta(font)
    image = draw_timeline(
        meta, sorted(_highlights(), key=lambda h: h["peak_frame"]), size=(800, 240)
    )
    assert image.size == (800, 240)
    pixels = np.asarray(image).astype(int)
    orange = np.abs(pixels - np.array([235, 104, 52])).sum(axis=2) < 40
    blue = np.abs(pixels - np.array([42, 120, 214])).sum(axis=2) < 60
    assert orange.any()
    assert blue.any()
    left, right = 64, 800 - 20
    columns = np.nonzero(orange.any(axis=0))[0]

    def x_of(frame: float) -> float:
        return left + (right - left) * frame / (N_FRAMES - 1)

    for item in _highlights():
        assert np.any(np.abs(columns - x_of(item["peak_frame"])) <= 6), item["id"]
    window_tint = pixels[120, int(x_of(70))]
    outside = pixels[120, int(x_of(200))]
    assert window_tint[0] > outside[0] or window_tint[2] < outside[2], "window band tints the plot"


def test_timeline_without_score_still_draws_windows(out_dir: Path, font: FontInfo) -> None:
    _write(out_dir, font, score=None)
    image = Image.open(out_dir / TIMELINE_NAME)
    assert image.size == (1600, 420)
    dark = np.asarray(Image.open(out_dir / TIMELINE_DARK_NAME).convert("RGB")).astype(int)
    assert dark[5, 5].sum() < 3 * 60, "the dark variant has a dark surface"


def test_nested_cue_objects_show_contributions_and_evidence(out_dir: Path, font: FontInfo) -> None:
    highlights = _highlights()
    highlights[0]["cues"] = {
        "audio": {"contribution": 0.09, "excitement_z": 2.35, "frame": 330, "weight": 1.5},
        "speed": {"contribution": 0.05, "speed_mps": 7.57, "track_id": 9, "frame": 335},
        "box_pressure": {"contribution": 0.12, "attackers": 3, "frame": 340, "team": "left"},
        "duel": {"contribution": 0.0, "distance": 1.83, "unit": "m", "frame": 338},
        # Shown as +0.000 at three decimals: it did not contribute either.
        "ball_goal": {"contribution": 0.0004, "distance_m": 24.6, "frame": 336},
    }
    highlights[0]["refined_by"] = "duel_min_distance"
    write_report(out_dir, highlights, [], {}, {}, _meta(font))
    text = (out_dir / "report.html").read_text(encoding="utf-8")
    card = text[text.index('id="hl-h2"') : text.index('id="hl-h3"')]
    box = card.index("禁区内进攻人数 +0.120（3 人，帧 340）")
    audio = card.index("现场声音响度变化 +0.090（z = 2.35，帧 330）")
    speed = card.index("冲刺速度 +0.050（27.3 km/h，帧 335，轨迹 9）")
    assert box < audio < speed, "cues are sorted by their contribution"
    assert "未贡献：球附近双方最近距离、球距球门" in card
    assert "+0.000" not in card
    assert "球在两人脚下时距离最近的一帧" in card


def test_card_rows_name_other_frames_and_skip_blocked() -> None:
    rows = [metric_row(record) for record in _records()]
    cards = card_rows(rows, limit=3, key_frame=350)
    assert [card.label for card in cards] == ["第100帧·峰值速度", "最近对手距离", "膝角"]
    assert (cards[0].value, cards[0].unit) == ("27.4", "km/h")
    assert [badge.text for badge in cards[0].badges] == ["未验证"]
    assert cards[0].evidence_ids == ("ev-speed",)
    assert all(card.label != "膝角速度" for card in card_rows(rows, limit=10))
    count = metric_row({"metric": "attackers", "value": 3, "unit": "count"})
    assert (count.value_text, count.unit) == ("3", "个")


def test_card_rows_put_the_player_and_the_frame_before_the_label() -> None:
    # gbf.render cuts a long label at its end: the player and the frame come first, without
    # spaces, so that they leave the label as much room as they can.
    rows = [metric_row(record) for record in _records()]
    other = card_rows(rows, limit=3, key_frame=350, name_tracks=True, focus_track=9)
    assert [card.label for card in other] == [
        "轨迹7·第100帧·峰值速度",
        "最近对手距离",
        "膝角",
    ]
    focus = card_rows(rows, limit=1, key_frame=100, name_tracks=True, focus_track=7)
    assert [card.label for card in focus] == ["峰值速度"], "the focus player's own value"
    nobody = card_rows(rows, limit=1, key_frame=100, name_tracks=True)
    assert [card.label for card in nobody] == ["轨迹7·峰值速度"], "no focus: every player"
    assert [card.label for card in card_rows(rows, limit=1, key_frame=100)] == ["峰值速度"]


def test_image_links_carry_captions(out_dir: Path, font: FontInfo) -> None:
    write_report(
        out_dir,
        _highlights(),
        [],
        {"h1": ["keyframes/h1 关键帧.png", ImageLink("keyframes/h2 关键帧.png", "定格 <数据卡>")]},
        {},
        _meta(font),
    )
    text = (out_dir / "report.html").read_text(encoding="utf-8")
    card = text[text.index('id="hl-h1"') : text.index('id="hl-h2"')]
    assert "<figcaption>关键帧 100（00:04.00）· 原始画面上的标注</figcaption>" in card
    assert "<figcaption>定格 &lt;数据卡&gt;</figcaption>" in card
    assert card.index("h1%20") < card.index("h2%20"), "the key frame comes first"


def test_clip_render_manifests_are_linked_and_hashed(out_dir: Path, font: FontInfo) -> None:
    manifest = out_dir / "clips" / "h1 原速+慢放#1.render.json"
    manifest.write_bytes(b'{"schema": "gbf.render-manifest/1"}\n')
    text = _write(out_dir, font).read_text(encoding="utf-8")
    card = text[text.index('id="hl-h1"') : text.index('id="hl-h2"')]
    href = "clips/h1%20%E5%8E%9F%E9%80%9F%2B%E6%85%A2%E6%94%BE%231.render.json"
    assert f'<a href="{href}" title="逐帧来源与证据">渲染清单 · 原速 + 慢放 + 定格</a>' in card
    assert sha256_file(manifest) in text[text.index('id="appendix"') :]


def test_key_moments_tier_and_evaluation_sections(unicode_tmp_path: Path) -> None:
    from gbf import report as rp

    item = {
        "id": "h01",
        "type": "duel_tackle",
        "start_frame": 0,
        "peak_frame": 20,
        "end_frame": 40,
        "score": 0.4,
        "tier": "main",
        "refined_by": "pose_key_moment",
        "title_zh": "对抗",
        "summary_zh": "说明",
        "key_moments": [
            {"frame": 10, "kind": "touch", "text_zh": "甲触球"},
            {"frame": 14, "kind": "ball_away", "text_zh": "球离开甲的脚下"},
            {"frame": 20, "kind": "ball_release", "text_zh": "球离开两人脚下"},
            {"frame": 25, "kind": "on_ground", "text_zh": "乙躯干倾倒超过 45°"},
        ],
        "touches": [{"label_zh": "甲", "count": 3, "mean_interval_s": 0.4, "track_id": 5}],
    }
    meta = rp.ReportMeta(
        clip_name="c.mp4",
        fps="25/1",
        frame_count=50,
        width=64,
        height=36,
        source_sha256="0" * 64,
        evaluation_zh=("标注 M1 → 检出 h01：峰值差 -3 帧。",),
    )
    html = rp._card(1, item, meta, [], [], [])
    assert "<h4>关键时刻（逐帧，原始画面上的 2D 姿态和球）</h4>" in html
    assert "<td>球脱离争抢</td>" in html
    assert "<td>球离开脚下</td>" in html
    # on_ground marks the trunk passing 45 degrees; the body lands a few frames later.
    assert "<td>躯干过 45°</td>" in html
    assert "倒地" not in html
    # Who was detected touching the ball, with the track the card and the table name, and
    # one parenthesis for how touches are found.
    assert (
        "<p>检测到甲（轨迹 5）触球 3 次，平均间隔 0.40 秒"
        "（球速突变且脚在球边；轻触可能漏检；画面平面估计，未验证）。</p>"
    ) in html
    assert "）（" not in html
    assert "主要高光" in html
    assert "以原始画面上逐帧定出的关键瞬间为准" in html
    assert "球" not in rp.REFINED_BY_ZH["pose_key_moment"], "pose alone can set the key moment"
    assert 'href="#evaluation"' in rp._nav(meta)
    assert "峰值差 -3 帧" in rp._evaluation(meta)
    bare = rp.ReportMeta(
        clip_name="c.mp4",
        fps="25/1",
        frame_count=50,
        width=64,
        height=36,
        source_sha256="0" * 64,
    )
    assert 'href="#evaluation"' not in rp._nav(bare)
    assert rp._evaluation(bare) == ""


def test_key_moments_read_from_pose_alone_do_not_name_the_ball() -> None:
    from gbf import report as rp

    item = {
        "key_moments": [
            {"frame": 5, "kind": "contact_start", "text_zh": "甲与乙身体贴近（画面平面）"},
            {"frame": 28, "kind": "fall_onset", "text_zh": "乙躯干开始倾倒"},
            {"frame": 28, "kind": "contact_end", "text_zh": "身体贴近结束"},
            {"frame": 30, "kind": "on_ground", "text_zh": "乙躯干倾倒超过 45°（画面平面）"},
        ],
        "touches": [],
    }
    html = rp._key_moments(item, "25/1")
    assert "<h4>关键时刻（逐帧，原始画面上的 2D 姿态）</h4>" in html
    assert "球" not in html
    # Keypoints close in the image: the players came close, which is not a measured touch.
    assert "<td>身体贴近</td>" in html
    assert "<td>身体贴近结束</td>" in html
    assert "接触" not in "".join(rp.MOMENT_KINDS_ZH.values())
    # One touch is enough to name the ball.
    item["key_moments"].insert(1, {"frame": 10, "kind": "touch", "text_zh": "甲触球"})
    assert "原始画面上的 2D 姿态和球）" in rp._key_moments(item, "25/1")


REFIT = {
    "pitch_usable_before": 400,
    "refined_frames": 450,
    "rejected_frames": 30,
    "holdout_median_px": 3.5,
    "holdout_p90_px": 12.5,
    "holdout_p90_max_px": 25.0,
}


def _calibration(refit: object) -> dict[str, object]:
    return {
        "frames": N_FRAMES,
        "fresh": 420,
        "pitch_usable": 380,
        "rep_err_median_px": 2.5,
        "rep_err_p90_px": 6.0,
        "threshold_px": 8.0,
        "refit": refit,
    }


def test_overview_and_notes_describe_the_refitted_calibration(
    out_dir: Path, font: FontInfo
) -> None:
    text = _write(out_dir, font, calibration=_calibration(dict(REFIT))).read_text(encoding="utf-8")
    overview = _section(text, "overview", "timeline")
    assert (
        "标定（按机位重拟合后）球场可用 400 → 380/500 帧 · 重拟合 450 帧 · "
        "留出关键点误差中位 3.5 px · P90 12.5 px · "
        "拒用 30 帧（所在机位留出误差 P90 超过 25.0 px） · 单帧阈值 8.0 px"
    ) in overview
    # The plugin's own statistics stay, labelled as the plugin's.
    assert "插件逐帧标定（重拟合前）新算 420/500 帧 · 自洽误差中位 2.5 px · P90 6.0 px" in overview
    notes = _section(text, "notes", "appendix")
    assert "重拟合过的机位里，重拟合误差不超过 8.0 px 的帧" in notes
    assert "所在机位留出关键点误差的 P90 没有超过 25.0 px（否则整个机位不用）" in notes
    assert (
        "不在重拟合机位里的帧沿用插件的单帧标定，要求新算（fresh）且 rep_err_px ≤ 8.0 px" in notes
    )
    assert "只有新算（fresh）" not in notes
    plain = _write(out_dir, font, calibration=_calibration(None)).read_text(encoding="utf-8")
    overview = _section(plain, "overview", "timeline")
    assert (
        "标定（插件逐帧）球场可用 380/500 帧 · 新算 420/500 帧 · 自洽误差中位 2.5 px · "
        "P90 6.0 px · 阈值 8.0 px"
    ) in overview
    assert "重拟合" not in overview
    assert "只有新算（fresh）且 rep_err_px ≤ 8.0 px 的帧" in _section(plain, "notes", "appendix")
    with pytest.raises(ReportInputError) as excinfo:
        _write(out_dir, font, calibration=_calibration({"holdout_p90_px": "large"}))
    assert "holdout_p90_px" in "\n".join(excinfo.value.problems)


def test_report_text_uses_display_names_and_explains_formats(out_dir: Path, font: FontInfo) -> None:
    highlights = _highlights()
    # Ids that are not the ranks: peaks 340, 100 and 450 make them H2, H1 and H3.
    for item, ident in zip(highlights, ("h07", "h04", "h09"), strict=True):
        item["id"] = ident
    del highlights[2]["title_zh"]
    highlights[1]["cues"] = {
        "audio": {"contribution": 0.05, "excitement_z": 1.9, "frame": 99},
        "strike": {"contribution": 0.03, "event_frame": 101, "frame": 101},
    }
    records = [
        {"evidenceId": "ev-tilt", "highlightId": "h07", "metric_id": "image-trunk-lean",
         "label_zh": "躯干倾斜（画面）", "value": -12.0, "unit": "deg", "tier": "image_plane",
         "plane": "image", "frame": 340},
    ]  # fmt: skip
    evaluation = (
        "标注 M1（第 98 帧）→ 检出 h04（第 100 帧）：峰值差 +2 帧。",
        "标注 M2（closeup_reaction，第 452 帧）：没有单独成段，但在 h09 的窗口里。",
    )
    write_report(out_dir, highlights, records, {}, {}, _meta(font, evaluation_zh=evaluation))
    text = (out_dir / "report.html").read_text(encoding="utf-8")
    seen = _visible(text)
    for code in ("image_plane", "closeup_reaction", "h04", "h07", "h09"):
        assert code not in seen, code
    assert "检出 H1（第 100 帧）" in seen
    reaction = HIGHLIGHT_TYPES_ZH["closeup_reaction"]  # the names the cards and the video use
    assert f"（{reaction}，第 452 帧）：没有单独成段，但在 H3 的窗口里" in seen
    assert "冲刺 H3" in _section(text, "hl-h09", "evaluation"), "untitled card named by its H"
    assert "躯干倾斜（画面）：画面平面量" in _section(text, "overview", "timeline")
    card = _section(text, "hl-h07", "hl-h09")
    assert "画面平面量" in card
    assert "0° 为竖直，正值是上身向画面右侧倾斜，负值向左" in card
    first = _section(text, "hl-h04", "hl-h07")
    assert "现场声音响度变化 +0.050（z = 1.90，帧 99）" in first
    assert "触球事件 +0.030（触球帧 101）" in first, "the touch frame is named once"
    for claim in ("激动", "击球", "精确到帧", "mm:ss"):
        assert claim not in seen, claim
    assert "以帧为单位（未验证；本片一帧 40 ms）" in _section(text, "overview", "timeline")
    assert "触球帧以帧为单位（未验证；本片一帧 40 ms）" in _section(text, "notes", "appendix")
    timeline = _section(text, "timeline", "highlights")
    assert "时间（分:秒.百分之一秒）" in timeline
    assert "小数点后两位不是帧号" in timeline
    # The dots of the chart are per-frame scores; the table's score is the smoothed peak.
    assert "评分（平滑后的峰值）" in timeline
    assert "圆点画在关键帧的逐帧分数上" in timeline


def test_card_rows_say_which_sign_of_a_tilt_is_which() -> None:
    tilt = metric_row(
        {"metric_id": "image-trunk-lean", "label_zh": "躯干倾斜（画面）", "value": 7.0,
         "unit": "deg", "tier": "image_plane", "plane": "image"}
    )  # fmt: skip
    (row,) = card_rows([tilt])
    assert (row.value, row.unit) == ("7.0", "°（画面右倾为正）")
    knee = metric_row({"metric_id": "image-knee-angle-left", "value": 150.0, "unit": "deg"})
    assert card_rows([knee])[0].unit == "°"


def test_render_manifest_links_have_distinct_names(out_dir: Path, font: FontInfo) -> None:
    for name in ("h1 原速+慢放#1", "full", "other"):
        (out_dir / "clips" / f"{name}.mp4").write_bytes(b"synthetic clip bytes")
        (out_dir / "clips" / f"{name}.render.json").write_bytes(b"{}\n")
    clips = {
        "h1": [
            ClipLink("原速 + 慢放", "clips/h1 原速+慢放#1.mp4"),
            ClipLink("全片", "clips/full.mp4"),
        ],
        "h2": [ClipLink("片段", "clips/full.mp4"), ClipLink("片段", "clips/other.mp4")],
    }
    text = write_report(out_dir, _highlights(), [], {}, clips, _meta(font)).read_text(
        encoding="utf-8"
    )
    first = _section(text, "hl-h1", "hl-h2")
    assert "渲染清单 · 原速 + 慢放" in first
    assert "渲染清单 · 全片" in first
    second = _section(text, "hl-h2", "hl-h3")
    assert "渲染清单 · 片段（full.render.json）" in second
    assert "渲染清单 · 片段（other.render.json）" in second


def test_calibration_text_says_what_the_refit_did_and_no_p90_stands_alone(
    out_dir: Path, font: FontInfo
) -> None:
    text = _write(out_dir, font, calibration=_calibration(dict(REFIT))).read_text(encoding="utf-8")
    notes = _section(text, "notes", "appendix")
    # The held-out error comes from a second fit; the calibration in use is fitted on all.
    assert "每帧留出一部分关键点，另用其余关键点再拟合一次，量留出点的误差" in notes
    assert "实际使用的标定用全部关键点拟合" in notes
    assert "只用来量误差" not in notes
    # A refit report in which no frame was refitted: the plugin's calibration stayed.
    unchanged = {**REFIT, "pitch_usable_before": 380, "refined_frames": 0, "rejected_frames": 0}
    unchanged.update(holdout_median_px=None, holdout_p90_px=None)
    text = _write(out_dir, font, calibration=_calibration(unchanged)).read_text(encoding="utf-8")
    overview = _section(text, "overview", "timeline")
    notes = _section(text, "notes", "appendix")
    assert "插件导出了球场关键点，但没有一帧按机位重拟合，全部沿用插件的单帧标定" in notes
    assert "只有新算（fresh）且 rep_err_px ≤ 8.0 px 的帧" in notes
    assert "重拟合了" not in notes
    assert "留出" not in notes
    assert "标定（插件逐帧，没有一帧重拟合）球场可用 380/500 帧 · 新算 420/500 帧" in overview
    assert "重拟合后" not in overview
    # Partial numbers: a P90 without its median still names what it is the P90 of, and a
    # refit without held-out numbers claims no held-out check.
    partial = _calibration({"refined_frames": 450, "holdout_p90_px": 12.5})
    partial["rep_err_median_px"] = None
    overview = _section(
        _write(out_dir, font, calibration=partial).read_text(encoding="utf-8"),
        "overview",
        "timeline",
    )
    assert "留出关键点误差 P90 12.5 px" in overview
    assert "自洽误差 P90 6.0 px" in overview
    assert overview.count("P90") == overview.count("误差 P90") == 2
    bare = _calibration({"refined_frames": 450})
    notes = _section(
        _write(out_dir, font, calibration=bare).read_text(encoding="utf-8"), "notes", "appendix"
    )
    assert "全景镜头按机位重拟合了标定" in notes
    assert "留出" not in notes


def test_overview_names_the_peak_speed_as_the_card_does(out_dir: Path, font: FontInfo) -> None:
    seen = _visible(_write(out_dir, font).read_text(encoding="utf-8"))
    label = metric_label("peak-speed")  # the data card's label of the number
    assert f"高光窗口内最高的 0.4 秒平均速度（数据卡上的“{label}”）" in seen
    assert "片段内峰值速度" not in seen
    # Uncalibrated shots are named as the highlight text names them: 中景 or 特写.
    assert "无标定的中景和特写" in seen
    assert "近景" not in seen


def test_timeline_ticks_of_a_short_clip_have_distinct_labels(
    font: FontInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    drawn: list[str] = []
    draw_text = ImageDraw.ImageDraw.text

    def recording(self: ImageDraw.ImageDraw, xy: object, text: str, *a: object, **k: object):
        drawn.append(text)
        return draw_text(self, xy, text, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", recording)
    short = _meta(font, frame_count=100, cuts=(60,), score=tuple(0.5 for _ in range(100)))
    draw_timeline(short, [], size=(800, 240))
    ticks = [text for text in drawn if re.fullmatch(r"\d\d:\d\d(\.\d\d)?", text)]
    # Half-second steps over 4 s (the last frame is at 3.96 s), in hundredths.
    assert ticks == [f"00:0{i // 2}.{50 * (i % 2):02d}" for i in range(8)]
    caption = report_module._timeline_section(short, [])
    assert "横轴是时间（分:秒.百分之一秒）" in caption
    drawn.clear()
    draw_timeline(_meta(font), [], size=(800, 240))
    ticks = [text for text in drawn if re.fullmatch(r"\d\d:\d\d(\.\d\d)?", text)]
    assert ticks == ["00:00", "00:05", "00:10", "00:15"], "whole seconds on a 20 s clip"
    assert "横轴是时间（分:秒）" in report_module._timeline_section(_meta(font), [])


def test_report_sentences_must_be_strings(out_dir: Path, font: FontInfo) -> None:
    with pytest.raises(ReportInputError) as excinfo:
        _write(
            out_dir,
            font,
            evaluation_zh=("标注 M1 → 检出 h1。", 3),
            evaluation_source_zh=("标注者",),
            notes_zh="一句话",
        )
    joined = "\n".join(excinfo.value.problems)
    assert "meta.evaluation_zh[1] must be a string, got int 3" in joined
    assert "meta.evaluation_source_zh must be a string or None, got tuple" in joined
    assert "meta.notes_zh must be a sequence of strings, got str" in joined
    assert not (out_dir / "report.html").exists()


def test_the_evaluation_section_says_where_the_labels_come_from_in_their_own_words() -> None:
    # Who wrote the labels and how is the label file's to say (gbf.label_eval.source_zh):
    # the page quotes it and claims nothing more, neither in the title nor in the text.
    from gbf import report as rp

    source = "标注来源（照录标注文件）：标注者“a script, from thumbnails”；盲法“before any run”。"
    meta = rp.ReportMeta(
        clip_name="c.mp4",
        fps="25/1",
        frame_count=50,
        width=64,
        height=36,
        source_sha256="0" * 64,
        evaluation_zh=("标注 M1（第 20 帧）→ 检出 h01（第 22 帧）：关键帧差 +2 帧。",),
        evaluation_source_zh=source,
    )
    text = _visible(rp._evaluation(meta, {"h01": "H1"}))
    assert text.startswith("与标注对比" + source)
    assert "检出 H1（第 22 帧）" in text
    assert "下面只是检出结果和这份标注的对照，不是验证。" in text
    for claim in ("人工", "逐帧", "预先", "任何输出", "原视频"):
        assert claim not in text, claim
    assert '<a href="#evaluation">标注对比</a>' in rp._nav(meta)
    # A quotation keeps its words: highlight ids and type codes are renamed in the
    # comparison's sentences only.
    quoted = replace(meta, evaluation_source_zh="标注来源（照录标注文件）：说明“sprint of h01”。")
    assert "说明“sprint of h01”" in _visible(rp._evaluation(quoted, {"h01": "H1"}))
    unknown = _visible(rp._evaluation(replace(meta, evaluation_source_zh=None)))
    assert unknown.startswith("与标注对比没有给出标注的来源。")


@pytest.mark.parametrize(("mps", "decimals"), [(6.2069, 2), (0.25, None), (8.0139, 1)])
def test_a_stored_speed_reads_the_same_in_the_cue_list_and_the_table(
    mps: float, decimals: int | None
) -> None:
    # One function rounds a stored speed once, to km/h with one decimal, for the cue chips
    # and for the metrics table (and so the data card), whatever decimals a record gives.
    record: dict = {"metric": "peak-speed", "value": mps, "unit": "m/s", "tier": "A"}
    if decimals is not None:
        record["decimals"] = decimals
    row = metric_row(record)
    assert (row.value_text, row.unit) == (f"{mps * 3.6:.1f}", "km/h")
    chips, _idle = report_module._cue_chips(
        {"speed": {"contribution": 0.05, "speed_mps": mps, "track_id": 3, "frame": 7}}
    )
    assert f"冲刺速度 +0.050（{row.value_text} km/h，帧 7，轨迹 3）" in chips


def test_a_cue_names_its_frame_once_and_the_duel_cue_claims_no_contact() -> None:
    detail = report_module._cue_detail
    assert detail("strike", {"event_frame": 41, "frame": 41}) == "触球帧 41"
    assert detail("strike", {"event_frame": 41, "frame": 44}) == "触球帧 41，帧 44"
    assert detail("cut_emphasis", {"cut_frame": 60, "frame": 52}) == "切镜帧 60，帧 52"
    # The duel cue is the distance of the nearest opposing pair near the ball, close or not.
    assert report_module.CUE_ZH["duel"] == "球附近双方最近距离"
    assert detail("duel", {"distance": 2.5, "unit": "m", "frame": 52}) == "2.50 m，帧 52"
    assert not any("贴身" in label for label in report_module.CUE_ZH.values())


def test_a_speed_chip_names_the_player_it_was_read_on() -> None:
    # Two synthetic players share a frame; the chip must identify its own speed's track.
    detail = report_module._cue_detail
    cue = {"speed_mps": 7.5, "track_id": 3, "frame": 12}
    assert detail("speed", cue) == "27.0 km/h，帧 12，轨迹 3"
    assert detail("speed", {**cue, "track_id": None}) == "27.0 km/h，帧 12"
    assert detail("speed", {**cue, "track_id": -1}) == "27.0 km/h，帧 12"
    # Other cues name no track of their own.
    assert detail("audio", {"excitement_z": 2.0, "frame": 9, "track_id": 4}) == "z = 2.00，帧 9"


def test_the_image_trunk_angle_is_described_as_the_line_it_is_measured_on() -> None:
    # gbf.lift3d.image_trunk_lean_deg runs from the hip midpoint to the shoulder midpoint;
    # the sentence used to name the neck.
    short, sentence = report_module.METRIC_SIGNS_ZH["image-trunk-lean"]
    assert short == "画面右倾为正"
    assert sentence.startswith("髋部中点到两肩中点的连线与画面竖直方向的夹角")
    assert "颈" not in sentence


@pytest.mark.parametrize("watermark", [None, "私有素材 · 禁止分享"])
def test_exported_timelines_carry_restrictions_and_optional_visible_watermark(
    out_dir: Path, font: FontInfo, watermark: str | None
) -> None:
    _write(out_dir, font, watermark=watermark)
    for name, theme in (
        (TIMELINE_NAME, report_module.LIGHT_THEME),
        (TIMELINE_DARK_NAME, report_module.DARK_THEME),
    ):
        with Image.open(out_dir / name) as image:
            assert image.text["Comment"] == "gbf restrictions: no_share"
            chart = np.array(image.convert("RGB"))
        ordered = sorted(_highlights(), key=lambda item: (item["peak_frame"], item["id"]))
        plain = np.array(draw_timeline(_meta(font), ordered, theme=theme).convert("RGB"))
        if watermark:
            assert np.count_nonzero(chart[:60, -400:] != plain[:60, -400:]) > 100
            assert np.array_equal(chart[100:], plain[100:])
        else:
            assert np.array_equal(chart, plain)


@pytest.mark.parametrize(("fps", "seconds"), [("25/1", "0.4"), ("24/1", "0.42")])
def test_overview_names_the_chord_of_the_clip_rate_as_the_card_does(
    out_dir: Path, font: FontInfo, fps: str, seconds: str
) -> None:
    # At 24 fps the 0.4 s chord spans 10 frames, 0.42 s, and the data card says so.
    seen = _visible(_write(out_dir, font, fps=fps).read_text(encoding="utf-8"))
    label = metric_label("peak-speed", report_module.MEASURE_PARAMS, Fraction(fps))
    assert label == f"峰值速度（{seconds}秒平均）"
    assert f"高光窗口内最高的 {seconds} 秒平均速度（数据卡上的“{label}”）" in seen


def test_overview_and_notes_claim_no_more_than_is_measured(out_dir: Path, font: FontInfo) -> None:
    text = _write(out_dir, font).read_text(encoding="utf-8")
    overview = _section(text, "overview", "timeline")
    # The lean is given at the player's own touch when one was detected, else at the
    # player's reference frame: not only at a touch.
    assert "触球时躯干前倾" not in overview
    assert "取球员本人检测到触球的那一帧前后，没有就取该球员的参考帧，通常是关键帧" in overview
    # The ball's image-plane speed is measured; its true speed (km/h) is not.
    assert "球和脚的真实速度（km/h）" in overview
    # Attackers in the box count only with the ball at the box (gbf.highlights), and the
    # duel cue is the closest opposing pair near the ball, not a measured contact.
    assert "球在禁区内或紧挨禁区时禁区内的进攻人数、看得到球时球到球门的距离" in overview
    assert "球附近双方球员的最近距离" in overview
    assert "球速" not in overview
    notes = _section(text, "notes", "appendix")
    assert "C 档指标（角速度、球和脚的真实速度、触球时长等）不输出" in notes
    assert "球速" not in notes
    # The facing arrow is left out with its label when the label has no free spot.
    assert "会随门槛时有时无；箭头的标签没有空位" in notes
    assert "时，箭头也不画。" in notes


def test_the_3d_reference_names_are_public() -> None:
    assert {"REFERENCE_3D_METRICS", "REFERENCE_3D_ZH"} <= set(report_module.__all__)
    assert all(hasattr(report_module, name) for name in report_module.__all__)
