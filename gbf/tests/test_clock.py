"""Tests for gbf.util.clock: the one clock text of titles, summaries, cards and tables.

A frame's time is ``frame / fps`` with the hundredths floored, in exact rational arithmetic;
titles used to round it while the banner, the data card and the report's tables floored it,
so at 30000/1001 half the frames showed two different times.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from gbf.highlights import format_time
from gbf.measure import display_text, format_frame_time
from gbf.util.clock import ClockError, clock_rate, format_clock

RATES = ("25/1", "30000/1001", "24000/1001")


@pytest.mark.parametrize(
    ("fps", "frame", "text"),
    [
        ("25/1", 0, "00:00.00"),
        ("25/1", 1, "00:00.04"),
        ("25/1", 459, "00:18.36"),
        ("25/1", 1500, "01:00.00"),
        ("25/1", 1680, "01:07.20"),
        ("30000/1001", 1, "00:00.03"),
        ("30000/1001", 2, "00:00.06"),  # 0.0667 s: floored, never 00:00.07
        ("30000/1001", 29, "00:00.96"),
        ("30000/1001", 30, "00:01.00"),  # 1.001 s
        ("30000/1001", 1798, "00:59.99"),
        ("30000/1001", 1799, "01:00.02"),
        ("24000/1001", 1, "00:00.04"),
        ("24000/1001", 23, "00:00.95"),
        ("24000/1001", 24, "00:01.00"),
        ("24000/1001", 1438, "00:59.97"),
        ("24000/1001", 1439, "01:00.01"),
    ],
)
def test_the_clock_floors_hundredths_exactly(fps: str, frame: int, text: str) -> None:
    assert format_clock(frame, fps) == text
    assert format_clock(frame, Fraction(fps)) == text


@pytest.mark.parametrize("fps", RATES)
def test_titles_summaries_cards_and_tables_name_the_same_time(fps: str) -> None:
    # gbf.highlights (titles, summaries, key-moment times) and gbf.measure (the display
    # text of frame records on the cards and in the report's tables) write one clock.
    rate = clock_rate(fps)
    for frame in range(0, 3000):
        text = format_clock(frame, rate)
        assert format_time(frame, fps) == text, frame
        assert format_frame_time(frame, rate) == text, frame
    assert display_text(2, "frame", None, rate).endswith(f"（{format_clock(2, rate)}）")


def test_the_clock_lists_every_problem() -> None:
    with pytest.raises(ClockError) as caught:
        format_clock(-1, "29.97")
    assert len(caught.value.problems) == 2
    for frame in (True, 1.5, "3"):
        with pytest.raises(ClockError, match="frame"):
            format_clock(frame, "25/1")  # type: ignore[arg-type]
    for fps in (0, "0/1", 29.97, True):
        with pytest.raises(ClockError, match="fps"):
            format_clock(1, fps)  # type: ignore[arg-type]
    assert clock_rate(25) == Fraction(25)
