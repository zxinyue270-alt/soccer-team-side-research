"""Tests for gbf.label_eval."""

from __future__ import annotations

import pytest

from gbf import label_eval as le

LABELS = {
    "id": "demo",
    "status": "development clip",
    "moments": [
        {
            "id": "M1",
            "type": "duel_tackle",
            "start": 100,
            "peak": 150,
            "end": 160,
            "peak_tolerance_frames": 6,
        },
        {"id": "M2", "type": "attack_in_box", "start": 300, "peak": 320, "end": 340},
        {"id": "M3", "type": "sprint", "start": 500, "peak": 510, "end": 520},
    ],
}


def _h(hid: str, kind: str, start: int, peak: int, end: int) -> dict:
    return {
        "id": hid,
        "type": kind,
        "start_frame": start,
        "peak_frame": peak,
        "end_frame": end,
        "score": 0.3,
    }


def test_matching_errors_and_unmatched() -> None:
    doc = {
        "highlights": [
            _h("h01", "duel_tackle", 90, 147, 170),
            _h("h02", "duel_tackle", 290, 345, 350),
            _h("h03", "closeup_reaction", 700, 720, 760),
        ]
    }
    result = le.evaluate(LABELS, doc)
    rows = {r["label"]: r for r in result["rows"]}
    assert rows["M1"]["peak_error_frames"] == -3
    assert rows["M1"]["peak_within_tolerance"]
    assert rows["M1"]["type_agrees"]
    assert rows["M2"]["peak_error_frames"] == 25
    assert not rows["M2"]["peak_within_tolerance"]  # default tolerance 12
    assert not rows["M2"]["type_agrees"]
    assert not rows["M3"]["matched"]
    assert [u["highlight"] for u in result["unmatched_highlights"]] == ["h03"]
    assert (result["n_matched"], result["n_peak_within_tolerance"], result["n_type_agrees"]) == (
        2,
        1,
        1,
    )
    text = "".join(le.summary_zh(result))
    assert "-3 帧" in text
    assert "未检出" in text


def test_each_highlight_is_used_once() -> None:
    labels = {
        "moments": [
            {"id": "A", "type": "duel_tackle", "start": 0, "peak": 10, "end": 20},
            {"id": "B", "type": "duel_tackle", "start": 15, "peak": 25, "end": 30},
        ]
    }
    result = le.evaluate(labels, {"highlights": [_h("h01", "duel_tackle", 0, 12, 30)]})
    assert [r["matched"] for r in result["rows"]] == [True, False]
    assert result["rows"][1]["covered_by"] == "h01"
    assert "在 h01 的窗口里" in "".join(le.summary_zh(result))


def test_bad_inputs_are_listed() -> None:
    with pytest.raises(le.LabelEvalError) as caught:
        le.evaluate(
            {"moments": [{"id": "X", "start": 5, "peak": 1, "end": 9}]},
            {"highlights": [{"id": "h01"}]},
        )
    text = " ".join(caught.value.problems)
    assert "missing 'type'" in text
    assert "start <= peak <= end" in text
    assert "highlight 0" in text


def test_sentences_call_them_labels_and_say_nothing_of_who_wrote_them() -> None:
    # Who wrote the labels, and how, is the label file's to say (source_zh): the sentences
    # never call them 人工标注 (hand labels by a person).
    doc = {"highlights": [_h("h01", "duel_tackle", 90, 147, 170)]}
    lines = le.summary_zh(le.evaluate(LABELS, doc))
    assert lines[0].startswith("标注 M1（第 150 帧）→ 检出 h01")
    assert lines[-1].startswith("合计：3 个标注中检出 1 个")
    assert all("人工" not in line for line in lines)


def test_the_source_sentence_quotes_the_label_files_own_fields() -> None:
    labels = {
        **LABELS,
        "labelled_by": "a script, from thumbnails every 10 frames",
        "blinding": "written before the detector's output was looked at",
        "labelled_on": "2026-01-02",
        "method": 42,  # not text: not quoted
        "status": "  development clip  ",
    }
    assert le.evaluate(labels, {"highlights": []})["labels_source"] == {
        "labelled_by": "a script, from thumbnails every 10 frames",
        "labelled_on": "2026-01-02",
        "blinding": "written before the detector's output was looked at",
        "status": "development clip",
    }
    assert le.source_zh(labels) == (
        "标注来源（照录标注文件）：标注者“a script, from thumbnails every 10 frames”；"
        "标注日期“2026-01-02”；盲法“written before the detector's output was looked at”；"
        "说明“development clip”。"
    )
    # Without any of the fields the sentence claims nothing about the labels.
    bare = le.source_zh(LABELS | {"status": ""})
    assert bare == "标注文件没有写标注者、方法、盲法等来源字段，标注的来源不明。"
    assert le.evaluate(LABELS | {"status": ""}, {"highlights": []})["labels_source"] == {}
