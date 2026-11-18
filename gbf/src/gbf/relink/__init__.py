"""Relink: an optional step of the refine stage that cuts tracks where the tracker changed
person and joins the pieces of one person (``--relink``; off by default).

A port of the frozen method ``fuse_r1`` of the re-identification lab (reid_lab, outside the
repository). The modules, in the order the method uses them:

- :mod:`gbf.relink.inputs`: the rows of one shot in the form every module reads;
- :mod:`gbf.relink.cuts`: cuts inside a track where the tracked person changes;
- :mod:`gbf.relink.appear`: appearance prototypes of the sub-pieces;
- :mod:`gbf.relink.space`: position evidence for pairs of sub-pieces;
- :mod:`gbf.relink.link`: the pair bonus and the links between solid sub-pieces;
- :mod:`gbf.relink.leaves`: short sub-pieces attached to the linked groups (recall mode);
- :mod:`gbf.relink.api`: the method for one shot and for a track table, and the ids.
- :mod:`gbf.relink.safety`: product guard isolating conflicting unclean observations;
  this changes the frozen method's inputs to linking and is not independently validated.

The imports below are static on purpose: the pipeline's stage cache follows them
(gbf.pipeline hashes a stage's modules through their static imports).
"""

from __future__ import annotations

from gbf.relink import api, appear, cuts, inputs, leaves, link, shirts, space
from gbf.relink.api import (
    DEFAULT_PARAMS,
    MODES,
    ClipLinks,
    RelinkError,
    RelinkParams,
    RelinkResult,
    apply_ids,
    relink_clip,
    relink_tracks,
    setting,
    skipped,
)
from gbf.relink.inputs import ClipInputs
from gbf.relink.shirts import measure_shirt_colours

__all__ = [
    "DEFAULT_PARAMS",
    "MODES",
    "ClipInputs",
    "ClipLinks",
    "RelinkError",
    "RelinkParams",
    "RelinkResult",
    "api",
    "appear",
    "apply_ids",
    "cuts",
    "inputs",
    "leaves",
    "link",
    "measure_shirt_colours",
    "relink_clip",
    "relink_tracks",
    "setting",
    "shirts",
    "skipped",
    "space",
]
