"""gbf.evaluate.pitch_lines: the ground lines of the SoccerNet pitch model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gbf.evaluate import pitch_lines

GROUND_LINE_NAMES = {
    "Side line top",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Middle line",
    "Circle central",
    "Circle left",
    "Circle right",
    *(
        f"{box} {side} {part}"
        for box in ("Big rect.", "Small rect.")
        for side in ("left", "right")
        for part in ("top", "bottom", "main")
    ),
}


def test_names_are_the_twenty_ground_lines() -> None:
    assert set(pitch_lines.ground_lines()) == GROUND_LINE_NAMES


def test_touchlines_and_boxes() -> None:
    lines = pitch_lines.ground_lines()
    top = lines["Side line top"]
    assert top[0] == pytest.approx([-52.5, -34.0])
    assert top[-1] == pytest.approx([52.5, -34.0])
    main = lines["Big rect. left main"]
    assert np.allclose(main[:, 0], -36.0)
    assert main[[0, -1], 1] == pytest.approx([-20.16, 20.16])
    small = lines["Small rect. right top"]
    assert small[0] == pytest.approx([52.5, -9.16])
    assert small[-1] == pytest.approx([47.0, -9.16])


def test_penalty_arcs_end_on_the_box_lines() -> None:
    lines = pitch_lines.ground_lines()
    half_chord = math.sqrt(9.15**2 - 5.5**2)
    left, right = lines["Circle left"], lines["Circle right"]
    assert left[[0, -1], 0] == pytest.approx([-36.0, -36.0])
    assert sorted(left[[0, -1], 1]) == pytest.approx([-half_chord, half_chord])
    assert right[[0, -1], 0] == pytest.approx([36.0, 36.0])
    assert left[:, 0].max() == pytest.approx(-41.5 + 9.15, abs=1e-3)


def test_samples_are_at_most_one_step_apart() -> None:
    for name, points in pitch_lines.ground_lines(step=0.1).items():
        gaps = np.hypot(*np.diff(points, axis=0).T)
        assert gaps.max() <= 0.1 + 1e-9, name
