"""Colour-corroborated row isolation around the frozen relink method.

An unclean detection is not necessarily a different person. Only isolate it when its
embedding disagrees with every supported nearby clean-side descriptor, and its measured
shirt colour is clearly closer to a different sub-piece's clean colour prototype than to
every measured sample in each supported own-side window. A single own-side median would
discard real shirt variation (print, sleeves, pose and lighting), so it is not sufficient
negative evidence. Missing colour evidence leaves the identity unchanged.
This is a product policy, not the lab's fitted method or a validated identity test.
"""

from __future__ import annotations

import numpy as np

from gbf.relink import cuts, inputs, shirts


def conflicting_rows(inp: inputs.ClipInputs, sub: np.ndarray, params: cuts.CutParams) -> np.ndarray:
    """Positions of unclean ReID conflicts corroborated by a different shirt colour.

    Work within a shot and an already cut sub-piece. Up to k_win clean rows on each side
    describe that side when at least m_min exist. A missing side makes no judgement; no
    usable side means no isolation. Clean rows, missing embeddings/colours and ambiguous
    colours are left alone. Another colour does not identify a different person; an
    isolated observation still receives only an unverified singleton id.
    """
    if inp.shirt_lab is None:
        return np.empty(0, dtype=np.int64)
    colours = inp.shirt_lab
    measured = np.isfinite(colours).all(axis=1)
    bad = np.zeros(len(sub), dtype=bool)
    has = inputs.has_embedding(inp.emb)
    prototypes = {}
    for pid in np.unique(sub[sub >= 0]):
        reference = (sub == pid) & inp.clean & measured
        if int(reference.sum()) >= params.m_min:
            prototypes[int(pid)] = np.median(colours[reference], axis=0)
    for pid in np.unique(sub[sub >= 0]):
        rows = np.flatnonzero(sub == pid)
        rows = rows[np.argsort(inp.frame[rows], kind="stable")]
        anchors = np.flatnonzero(inp.clean[rows] & has[rows])
        if len(anchors) < params.m_min:
            continue
        for j in np.flatnonzero(~inp.clean[rows] & has[rows] & measured[rows]):
            split = int(np.searchsorted(anchors, j))
            sides = (
                anchors[max(0, split - params.k_win) : split],
                anchors[split : split + params.k_win],
            )
            similarities = []
            for side in sides:
                if len(side) < params.m_min:
                    continue
                mean = inp.emb[rows[side]].mean(axis=0)
                length = float(np.linalg.norm(mean))
                if length > 0 and np.isfinite(length):
                    similarities.append(float(inp.emb[rows[j]] @ (mean / length)))
            if not similarities or max(similarities) >= params.cos_cand:
                continue
            own_distances = []
            for side in sides:
                supported = rows[side][measured[rows[side]]]
                if len(supported) >= params.m_min:
                    # Existing variation is evidence against a confident foreign-kit
                    # judgement. Do not erase minority colours with a median prototype.
                    own_distances.extend(
                        np.linalg.norm(colours[supported] - colours[rows[j]], axis=1).tolist()
                    )
            alternatives = [
                float(np.linalg.norm(colours[rows[j]] - value))
                for other, value in prototypes.items()
                if other != int(pid)
            ]
            if not own_distances or not alternatives:
                continue
            alternative = min(alternatives)
            if alternative <= shirts.MAX_COLOUR_DISTANCE and all(
                distance - alternative > shirts.COLOUR_MARGIN for distance in own_distances
            ):
                bad[rows[j]] = True
    return np.flatnonzero(bad)
