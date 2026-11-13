"""Synthetic broadcast frames with known cameras, and the L0 smoke pipeline (plan P0).

- :mod:`gbf.synth.pitch` -- markings and landmarks of a 105 x 68 m pitch (SoccerNet frame).
- :mod:`gbf.synth.barcode` -- frame-number barcode: encoder, renderer and strict decoder.
- :mod:`gbf.synth.render` -- deterministic frames from known cameras, with a hard cut.
- :mod:`gbf.synth.video` -- frames to an H.264 CFR 25/1 mp4, verified with ffprobe.
- :mod:`gbf.synth.l0` -- ``python -m gbf l0``: the end-to-end L0 smoke run and its receipt.
"""

from gbf.synth.barcode import BarcodeError, decode_barcode, draw_barcode, encode_barcode
from gbf.synth.render import (
    FrameTruth,
    RenderedSequence,
    SynthError,
    render_sequence,
    visible_landmarks,
)
from gbf.synth.video import SynthVideoError, write_frames_mp4

__all__ = [
    "BarcodeError",
    "FrameTruth",
    "RenderedSequence",
    "SynthError",
    "SynthVideoError",
    "decode_barcode",
    "draw_barcode",
    "encode_barcode",
    "render_sequence",
    "visible_landmarks",
    "write_frames_mp4",
]
