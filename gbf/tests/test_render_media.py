"""Tests for the media side of gbf.render: the banner's fit, the clips' sound, the data card's
footer and the restriction tags media carry in their own metadata.

- The banner and its time pill end left of the minimap; a long title goes on to a second
  line between clauses, one too long for two lines keeps only whole clauses; the render
  manifest lists the title as drawn.
- ``--audio-offset-s`` counts from the first audio sample, as gbf.audio does:
  :func:`gbf.render.first_sample_offset` turns it into render's offset, so a clip plays at
  every frame the sound the audio cue measured, also when the audio starts after the video.
- The card's footer counts the rows its caller left out with the rows the frame has no
  room for, and the pipeline's card counts every row it cuts.
- Clips and images carry their restrictions in their own metadata (an mp4 comment, a PNG
  text chunk); frames can carry a second, fainter watermark at the bottom, and charts get
  the frames' watermark in their top-right corner.
- The banner's clock is gbf.util.clock's, and the card names the speed chord measured at
  any rate.

Everything is synthetic: frames from numpy, sound from ffmpeg's lavfi sources.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from gbf import render as render_module
from gbf.audio import audio_features, decode_mono
from gbf.ingest import parse_rate
from gbf.ingest.tools import media_arg
from gbf.kinematics import chord_half_frames
from gbf.measure import metric_label
from gbf.pipeline import _card
from gbf.render import (
    Banner,
    CardRow,
    FrameAnnotations,
    HighlightScene,
    InfoCard,
    Minimap,
    RenderError,
    SceneTables,
    Style,
    TextLabel,
    annotation_texts,
    draw_annotations,
    first_sample_offset,
    format_clock,
    minimap_geometry,
    plan_highlight_edit,
    probe_audio,
    render_edit,
    render_highlight,
    restrictions_comment,
    save_png,
    stamp_watermark,
    write_video,
)
from gbf.synth.video import write_frames_mp4
from gbf.util.clock import format_clock as shared_clock
from gbf.util.hashing import sha256_file

GRASS = (60, 120, 60)
LONG_TITLE = (
    "甲队球员与乙队球员贴身对抗：甲队球员躯干开始倾倒，随后倾倒过 45°（画面平面，第 1234 帧）"
)
CLAUSES = (
    "甲队球员与乙队球员贴身对抗：",
    "甲队球员躯干开始倾倒，",
    "随后倾倒过 45°（画面平面，第 1234 帧），",
    "乙队球员躯干倾倒过 30°（画面平面），",
    "球脱离争抢，",
    "甲队球员第二次触球，",
    "乙队球员随后倒地，",
    "裁判暂停比赛，",
    "双方球员围拢争执",
)
WATERMARK = "私有素材 · 禁止分享"


@pytest.fixture(scope="module")
def style() -> Style:
    """The default style with the system CJK fonts resolved once (a missing font fails)."""
    return Style().with_fonts()


def _frame(width: int, height: int, colour: tuple[int, int, int] = GRASS) -> np.ndarray:
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


def _title_lines(drawn: list[str], others: tuple[str, ...]) -> list[str]:
    return [text for text in drawn if text not in others]


# ---------------------------------------------------------------------------- the banner
@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720)], ids=["1080p", "720p"])
@pytest.mark.parametrize(
    ("corner", "position"), [("br", "bottom_left"), ("tr", "top_left")], ids=["bottom", "top"]
)
def test_the_banner_and_its_time_pill_end_left_of_the_minimap(
    style: Style, size: tuple[int, int], corner: str, position: str
) -> None:
    # A title as long as a duel's with a fall used to push the time pill and the banner's
    # edge over the minimap's frame and pitch.
    width, height = size
    framed = replace(style, minimap_corner=corner, banner_position=position)  # type: ignore[arg-type]
    frame = _frame(width, height)
    minimap = Minimap(title="俯视图")
    banner = Banner(LONG_TITLE, "01:23.45", "贴身争抢 · 第 1234 帧 · 球场坐标")
    only_map = draw_annotations(frame, FrameAnnotations(minimap=minimap), framed)
    both = draw_annotations(frame, FrameAnnotations(minimap=minimap, banner=banner), framed)
    x0, y0, panel_w, panel_h = minimap_geometry(width, height, minimap, framed).panel
    rows = slice(int(y0) - 2, int(y0 + panel_h) + 3)
    assert np.array_equal(both[rows, int(x0) - 2 :], only_map[rows, int(x0) - 2 :]), (
        "nothing of the banner on or beside the minimap"
    )
    _left, _top, right, _bottom = _bbox(_changed(both, only_map))
    assert right < x0 - 15 * height / 1080, "16 px (1080p) of gap"
    accent = np.abs(both.astype(int) - np.array(framed.accent)).sum(axis=2) < 30
    pill = accent & _changed(both, only_map)
    pill[:, : int(0.05 * width) + int(30 * height / 1080)] = False  # not the accent stripe
    assert pill.any(), "the time pill is drawn"
    assert _bbox(pill)[2] < x0 - 15 * height / 1080


def test_a_long_banner_title_goes_on_to_a_second_line_whole(
    style: Style, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotations = FrameAnnotations(
        minimap=Minimap(title="俯视图"), banner=Banner(LONG_TITLE, "01:23.45", "副标题")
    )
    drawn = _record_drawn_texts(monkeypatch)
    draw_annotations(_frame(1920, 1080), annotations, style)
    lines = _title_lines(drawn, ("俯视图", "01:23.45", "副标题"))
    assert len(lines) == 2
    assert "".join(lines) == LONG_TITLE, "the whole title, nothing cut"
    assert lines[0].endswith("，"), "the line ends where a clause ends"
    listed = annotation_texts(annotations, (1920, 1080), style)
    assert ("banner_title", "\n".join(lines), ()) in listed, "the manifest lists what is drawn"
    assert ("banner_title", LONG_TITLE, ()) in annotation_texts(annotations), "as given"


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720)], ids=["1080p", "720p"])
def test_a_title_too_long_for_two_lines_loses_whole_clauses_only(
    style: Style, monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    title = "".join(CLAUSES)
    annotations = FrameAnnotations(
        minimap=Minimap(title="俯视图"), banner=Banner(title, "01:23.45", "副标题")
    )
    drawn = _record_drawn_texts(monkeypatch)
    draw_annotations(_frame(*size), annotations, style)
    lines = _title_lines(drawn, ("俯视图", "01:23.45", "副标题"))
    assert len(lines) == 2
    shown = "".join(lines)
    assert shown.endswith("…")
    kept = [n for n in range(1, len(CLAUSES)) if "".join(CLAUSES[:n])[:-1] == shown[:-1]]
    assert kept, f"the banner keeps whole clauses: {shown!r}"
    assert kept[0] >= 3, "as many as two lines hold"
    listed = annotation_texts(annotations, size, style)
    assert ("banner_title", "\n".join(lines), ()) in listed


def test_banner_lines_break_as_chinese_text_does() -> None:
    text = "甲队球员（画面平面，第 1234 帧）倾倒过 45°，球速 0.42 秒，00:19.04 Team Alpha"
    breaks = render_module._line_breaks(text)
    assert breaks
    for i in breaks:
        assert text[i] not in "，）°", f"a line never starts with {text[i]!r}"
        assert text[i - 1] != "（", "a line never ends with an opening bracket"
    for token in ("第 1234 帧", "45°", "0.42", "00:19.04", "Team"):
        start = text.index(token)
        assert not [i for i in breaks if start < i < start + len(token)], token
    assert text.index("Alpha") in breaks, "a space between Latin words is a break"
    comma = text.index("，球速") + 1
    assert render_module._clause_ends(text) == [comma, text.index("，00") + 1], (
        "the comma inside the brackets ends no clause"
    )


# ---------------------------------------------------------------------------- the sound
def _late_audio(path: Path, ffmpeg_bin: str) -> Path:
    """A Matroska file whose video starts at 0 s and whose PCM sound starts at 0.3 s: a quiet
    tone with a loud burst from 1.0 to 1.4 s after the first audio sample."""
    burst = "if(between(t,1.0,1.4),0.5*sin(2*PI*1000*t),0.01*sin(2*PI*300*t))"
    subprocess.run(
        [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", "color=c=green:s=64x48:r=25:d=3",
            "-itsoffset", "0.3", "-f", "lavfi", "-i", f"aevalsrc=exprs='{burst}':s=48000:d=2.5",
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", "-ac", "2", media_arg(path),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return path


def _rms(samples: np.ndarray, start_s: float, end_s: float) -> float:
    part = samples[int(start_s * 48000) : int(end_s * 48000)].astype(np.float64)
    return float(np.sqrt(np.mean(np.square(part))))


def test_first_sample_offset_plays_the_sound_the_audio_cue_measured(
    unicode_tmp_path: Path, ffmpeg_bin: str, style: Style
) -> None:
    source = _late_audio(unicode_tmp_path / "原声 late.mkv", ffmpeg_bin)
    facts = probe_audio(source)
    assert facts.audio_start_s == pytest.approx(0.3, abs=1e-3)
    assert facts.default_offset_s == pytest.approx(0.0, abs=1e-3)
    # gbf.audio matches frame 0 with the first audio sample: the burst is on frames 25-35.
    cue = audio_features(source, "25/1", 75)
    loud = cue.loc[cue["rms_dbfs"] > -20.0, "frame"].tolist()
    assert set(range(26, 35)) <= set(loud) <= set(range(24, 37))
    video = unicode_tmp_path / "画面.mp4"
    frames = [_frame(64, 48, (40, 90 + k, 40)) for k in range(75)]
    write_frames_mp4(frames, video, fps="25/1")
    plan = plan_highlight_edit(0, 74, 30, "25/1", slow_factor=1, freeze_s=0)
    out = unicode_tmp_path / "片段.mp4"
    for offset_s in (0.0, 0.5, -0.2):
        result = render_edit(
            video, out, plan, style=style, audio_source=source, part_badges=False,
            audio_offset_s=first_sample_offset(source, offset_s), overwrite=True,
        )  # fmt: skip
        sound = decode_mono(out, sample_rate=48000)
        start = 1.0 - offset_s  # where the cue puts the burst with this offset
        assert _rms(sound, start + 0.05, start + 0.35) > 0.2, offset_s
        assert _rms(sound, start - 0.4, start - 0.05) < 0.03, offset_s
        assert _rms(sound, start + 0.45, start + 0.8) < 0.03, offset_s
        assert result.manifest["audio"]["offset_s"] == pytest.approx(offset_s + 0.3)
        sources = [p for p in result.manifest["output"]["audio_parts"] if p["kind"] == "source"]
        assert [p["offset_s"] for p in sources] == [pytest.approx(offset_s + 0.3)]
    # render's own default follows the container (frame 0 = the video's start): 0.3 s later
    render_edit(
        video, out, plan, style=style, audio_source=source, part_badges=False, overwrite=True
    )
    sound = decode_mono(out, sample_rate=48000)
    assert _rms(sound, 1.35, 1.65) > 0.2
    assert _rms(sound, 1.0, 1.25) < 0.03
    with pytest.raises(RenderError, match="offset_s"):
        first_sample_offset(source, float("nan"))


# ---------------------------------------------------------------------------- the card footer
def test_the_card_footer_counts_the_rows_left_out_before_and_for_room(
    style: Style, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = tuple(CardRow(f"指标 {i}", f"{i}.0", "m", ("未验证",)) for i in range(6))
    footer = "数值为自动估计，误差未测量（未验证）"
    card = InfoCard("关键帧", rows, footer=footer, omitted=3)
    roomy = FrameAnnotations(card=card)
    listed = annotation_texts(roomy, (1920, 1080), style)
    assert [text for kind, text, _ in listed if kind == "card_footer"] == [
        f"{footer}；另有 3 项见报告"
    ]
    assert [text for kind, text, _ in annotation_texts(roomy) if kind == "card_footer"] == [
        f"{footer}；另有 3 项见报告"
    ]
    # beside the minimap on a low frame the card has no room for every row: both counted
    framed = replace(style, minimap_corner="br")
    tight = FrameAnnotations(card=card, minimap=Minimap(title="俯视图"))
    drawn = _record_drawn_texts(monkeypatch)
    draw_annotations(_frame(1280, 600), tight, framed)
    shown = [text for text in drawn if text.startswith("指标")]
    assert 1 <= len(shown) < 6
    expected = f"{footer}；另有 {3 + 6 - len(shown)} 项见报告"
    assert expected in drawn
    listed = annotation_texts(tight, (1280, 600), framed)
    assert [text for kind, text, _ in listed if kind == "card_footer"] == [expected]
    assert InfoCard("关键帧", rows).omitted == 0
    for bad in (-1, 1.5, True):
        with pytest.raises(RenderError, match="omitted"):
            InfoCard("关键帧", rows, omitted=bad)  # type: ignore[arg-type]


def test_a_card_footer_too_long_for_the_card_is_listed_as_drawn(
    style: Style, monkeypatch: pytest.MonkeyPatch
) -> None:
    footer = "数值为自动估计，误差未测量（未验证）。" * 3
    drawn = _record_drawn_texts(monkeypatch)

    def footer_of(card: InfoCard) -> str:
        drawn.clear()
        draw_annotations(_frame(1920, 1080), FrameAnnotations(card=card), style)
        listed = annotation_texts(FrameAnnotations(card=card), (1920, 1080), style)
        footers = [text for kind, text, _ in listed if kind == "card_footer"]
        assert len(footers) == 1
        assert footers[0] in drawn, "the manifest lists the footer as drawn"
        return footers[0]

    rows = (CardRow("距离", "1.6", "m"),)
    assert footer_of(InfoCard("关键帧", rows, footer=footer)).endswith("…")
    with_count = footer_of(InfoCard("关键帧", rows, footer=footer, omitted=2))
    assert with_count.startswith("另有 2 项见报告；数值为自动估计"), "the count is never cut off"
    assert with_count.endswith("…")


class _Scene:
    """The part of render.HighlightScene that pipeline._card reads."""

    def __init__(self, focus: int, involved: set[int]) -> None:
        self.focus, self.involved = focus, involved

    def roles_at(self, frame: int) -> tuple[int, None, set[int]]:
        return self.focus, None, self.involved


def _record(metric: str, track: int, frame: int, **changes: Any) -> dict[str, Any]:
    """A synthetic evidence record as gbf.measure writes it (only the fields the card uses)."""
    record = {
        "evidenceId": f"ev-{metric}-{track}-{frame}",
        "highlight_id": "h01",
        "metric_id": metric,
        "label_zh": metric_label(metric),
        "status": "ok",
        "value": 1.25,
        "unit": "m",
        "decimals": 2,
        "tier": "A",
        "plane": "pitch",
        "error": {"kind": "unmeasured"},
        "frame": frame,
        "track_id": track,
    }
    record.update(changes)
    return record


def test_the_pipeline_card_counts_every_row_it_cuts() -> None:
    # 6 key moments and a touch count, 7 shown metrics and a blocked C-tier one: the card
    # keeps 4 moments and 2 values, and leaves 3 + 5 rows to the report.
    moments = [
        {"kind": kind, "track_ids": [5, 9], "frame": 40 + n, "time": f"00:01.{60 + 4 * n}"}
        for n, kind in enumerate(
            ("contact_start", "fall_onset", "fall_down", "ball_away", "contact_end", "release")
        )
    ]
    highlight = {
        "id": "h01",
        "type": "duel_tackle",
        "peak_frame": 50,
        "key_moments": moments,
        "touches": [{"label_zh": "甲队球员", "count": 2, "track_id": 5}],
    }
    metrics = (
        "nearest-opponent-distance",
        "nearest-opponent-image-distance",
        "ball-contact-frame",
        "body-orientation",
        "trunk-lean-at-contact",
        "peak-speed",
        "ball-image-speed",
    )
    records = [_record(metric, 5, 49) for metric in metrics]
    records.append(_record("attackers-in-box", 5, 49, tier="C"))  # never shown anywhere
    records.append(_record("peak-speed", 5, 49, highlight_id="h02"))  # another highlight
    rows, omitted = _card(records, highlight, _Scene(5, {5, 9}), lambda frame: 0)
    assert len(rows) == 6
    assert [row.label.startswith("轨迹5、9·") for row in rows[:4]] == [True] * 4
    assert omitted == (7 - 4) + (7 - 2)
    # values of another shot are no rows of this card, so none of them is counted
    rows, omitted = _card(records, highlight, _Scene(5, {5, 9}), lambda frame: int(frame >= 50))
    assert len(rows) == 4
    assert omitted == 7 - 4


def test_render_highlight_counts_the_cut_rows_and_tags_its_files(
    unicode_tmp_path: Path, style: Style, ffprobe_bin: str
) -> None:
    frames = [_frame(320, 180, (60, 100 + k, 60)) for k in range(40)]
    video = unicode_tmp_path / "源 视频.mp4"
    write_frames_mp4(frames, video, fps="25/1")
    tracks = pd.DataFrame.from_records(
        [
            {"frame": f, "track_id": t, "x1": x, "y1": 60.0, "x2": x + 30, "y2": 140.0,
             "role": "player", "team": team}
            for f in range(40)
            for t, x, team in ((1, 100.0, "left"), (2, 150.0, "right"))
        ]
    )  # fmt: skip
    highlight = {
        "id": "h1", "type": "duel_tackle", "start_frame": 5, "peak_frame": 20, "end_frame": 30,
        "score": 0.5, "track_ids": [1, 2], "title_zh": "贴身争抢",
    }  # fmt: skip
    scene = HighlightScene(SceneTables(tracks=tracks), highlight, "25/1")
    framed = replace(style, watermark=WATERMARK, watermark_repeat=True)
    out = unicode_tmp_path / "run 输出"
    rendered = render_highlight(
        video, out, scene, card_rows=(CardRow("距离", "1.6", "m", ("未验证",)),),
        card_omitted=4, style=framed, restrictions=("no_share", "hl3"),
        slow_factor=2, slow_before_s=0.08, slow_after_s=0.04, freeze_s=0.12,
    )  # fmt: skip
    footers = [r["text"] for r in rendered.edit.manifest["texts"] if r["kind"] == "card_footer"]
    assert footers == ["数值为自动估计，误差未测量（未验证）；另有 4 项见报告"]
    comment = "gbf restrictions: hl3,no_share"
    for image in (rendered.keyframe, rendered.keyframe_card):
        with Image.open(out / image) as png:
            assert png.text == {"Comment": comment}, image
    assert rendered.edit.manifest["output"]["comment"] == comment
    tags = subprocess.run(
        [ffprobe_bin, "-v", "error", "-show_entries", "format_tags", "-of", "json",
         media_arg(out / rendered.clip)],
        check=True, capture_output=True,
    )  # fmt: skip
    assert json.loads(tags.stdout)["format"]["tags"]["comment"] == comment
    assert rendered.edit.manifest["style"]["watermark_repeat"] is True


# ---------------------------------------------------------------------------- media tags
def test_clips_and_images_carry_their_restrictions(
    unicode_tmp_path: Path, ffprobe_bin: str
) -> None:
    frames = [_frame(64, 48, (30, 60 + 10 * k, 30)) for k in range(3)]
    comment = "gbf restrictions: hl3,no_share"
    assert restrictions_comment(["no_share", "hl3", "no_share"]) == comment
    assert restrictions_comment([]) is None

    def tags(path: Path) -> dict[str, str]:
        proc = subprocess.run(
            [ffprobe_bin, "-v", "error", "-show_entries", "format_tags", "-of", "json",
             media_arg(path)],
            check=True, capture_output=True,
        )  # fmt: skip
        return json.loads(proc.stdout).get("format", {}).get("tags", {})

    tagged = write_video(unicode_tmp_path / "片段 1.mp4", frames, restrictions=("no_share", "hl3"))
    assert tagged.comment == tagged.manifest()["comment"] == comment
    assert tags(tagged.path)["comment"] == comment
    plain = write_video(unicode_tmp_path / "plain.mp4", frames)
    assert plain.comment is None
    assert "comment" not in tags(plain.path)
    png = unicode_tmp_path / "关键帧 1.png"
    digest = save_png(png, frames[0], restrictions=["no_share", "hl3"])
    with Image.open(png) as image:
        assert image.text == {"Comment": comment}
        assert np.array_equal(np.asarray(image.convert("RGB")), frames[0])
    assert digest == sha256_file(png)
    bare = unicode_tmp_path / "bare.png"
    save_png(bare, frames[0])
    with Image.open(bare) as image:
        assert image.text == {}
    with pytest.raises(RenderError) as caught:
        write_video(unicode_tmp_path / "bad.mp4", frames, restrictions=("no_share", 3, ""))
    assert "restrictions" in str(caught.value)
    with pytest.raises(RenderError, match="list of tags"):
        save_png(png, frames[0], restrictions="no_share")  # type: ignore[arg-type]


def test_a_second_fainter_watermark_sits_in_the_bottom_margin(style: Style) -> None:
    width, height = 1280, 720
    frame = _frame(width, height, (128, 128, 128))
    margin = int(0.05 * height)
    once = replace(style, watermark=WATERMARK)
    twice = replace(once, watermark_repeat=True)
    single = _changed(draw_annotations(frame, FrameAnnotations(), once), frame)
    assert single[:margin].any()
    assert not single[margin:].any()
    out = draw_annotations(frame, FrameAnnotations(), twice)
    changed = _changed(out, frame)
    assert np.array_equal(changed[:margin], single[:margin]), "the top mark is unchanged"
    assert changed[height - margin :].any(), "a second mark in the bottom margin"
    assert not changed[margin : height - margin].any(), "nothing between the margins"
    top = out[:margin][changed[:margin]]
    bottom = out[height - margin :][changed[height - margin :]]
    assert bottom.max() < top.max(), "the bottom one is fainter: lighter text"
    assert bottom.min() > top.min(), "and a lighter panel"
    # like every panel it hides no text: a free text under it is left out
    note = TextLabel((width / 2, height - 0.55 * margin), "说明 · 1.0 m", evidence_ids=("ev-t",))
    annotations = FrameAnnotations(texts=(note,))
    assert not [row for row in annotation_texts(annotations, (width, height), twice) if row[2]]
    assert annotation_texts(annotations, (width, height), once)[0][1] == "说明 · 1.0 m"


def test_charts_get_the_watermark_in_their_top_right_corner() -> None:
    chart = np.full((420, 1600, 3), 252, np.uint8)
    out = stamp_watermark(chart, WATERMARK)
    left, top, right, bottom = _bbox(_changed(out, chart))
    assert right >= 1600 - 14, "in the top-right corner"
    assert top <= 14
    assert left > 1600 - 280, "the frames' badge size"
    assert bottom < 50
    assert out[top:bottom, left:right].min() < 120, "a dark panel with light text"
    assert np.array_equal(out[:, :left], chart[:, :left])
    with pytest.raises(RenderError):
        stamp_watermark(chart, " ")


# ---------------------------------------------------------------------------- clock and chord
def test_the_render_clock_is_the_shared_clock() -> None:
    for fps in ("25/1", "30000/1001", "24000/1001", Fraction(50), 24):
        for frame in range(0, 3000, 7):
            assert format_clock(frame, fps) == shared_clock(frame, fps), (fps, frame)
    assert format_clock(476) == "00:19.04"
    assert format_clock(np.int64(2), "30000/1001") == "00:00.06", "floored, never 00:00.07"
    with pytest.raises(RenderError) as caught:
        format_clock(-1, "29.97")
    assert len(caught.value.problems) == 2, "every problem, the frame's and the rate's"
    for fps in (0, "0/1", 29.97, True):
        with pytest.raises(RenderError, match="fps"):
            format_clock(1, fps)  # type: ignore[arg-type]


@pytest.mark.parametrize("fps", ["25/1", "30000/1001", "24000/1001", "24/1", "50/1"])
@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720)], ids=["1080p", "720p"])
def test_the_card_names_the_speed_chord_measured_at_any_rate(
    fps: str, size: tuple[int, int], style: Style, monkeypatch: pytest.MonkeyPatch
) -> None:
    # gbf.measure labels peak-speed with the chord it measured (2 * chord_half_frames frames,
    # 0.42 s at 24 fps for 0.4 s); the card shows that label whole, the focus player's and
    # another player's on another frame alike.
    rate = parse_rate(fps)
    seconds = f"{float(2 * chord_half_frames(rate) / rate):.2f}".rstrip("0").rstrip(".")
    label = metric_label("peak-speed", rate=rate)
    assert label == f"峰值速度（{seconds}秒平均）"
    highlight = {"id": "h01", "type": "sprint", "peak_frame": 1200, "key_moments": []}
    records = [
        _record("peak-speed", 7, 1200, label_zh=label, unit="m/s", value=7.5),
        _record("nearest-opponent-distance", 7, 1200),
    ]
    other = [_record("peak-speed", 42, 1191, label_zh=label, unit="m/s", value=8.0)]
    for rows in (
        _card(records, highlight, _Scene(7, {7, 42}), lambda frame: 0)[0],
        _card(other, highlight, _Scene(7, {7, 42}), lambda frame: 0)[0],
    ):
        drawn = _record_drawn_texts(monkeypatch)
        draw_annotations(_frame(*size), FrameAnnotations(card=InfoCard("关键帧", rows)), style)
        speed = [row.label for row in rows if row.label.endswith(label)]
        assert speed, [row.label for row in rows]
        assert set(speed) <= set(drawn), f"cut on the card: {drawn}"
        monkeypatch.undo()
