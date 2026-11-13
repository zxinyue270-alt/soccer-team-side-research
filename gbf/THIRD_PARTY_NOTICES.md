# Third-party notices

Status: M1, 2026-10-21. Not legal advice.

gbf's own code is MIT licensed ([LICENSE](LICENSE)). The MIT licence covers only original
gbf code. Third-party code, data, models, media, trademarks and broadcast footage keep
their own rights and terms.

## Material copied into this repository

The repository contains no vendored third-party code, no model weights, no datasets and
no third-party media, and no files have been ported from GameBuddy. Factual tables from
one file are reproduced: in `src/gbf/pose.py`, the Halpe-26 keypoint names and skeleton
edges (`HALPE26_KEYPOINTS`, `HALPE26_EDGES`), the left/right keypoint pairs
(`HALPE26_FLIP_PAIRS`, rtmlib's `swap` entries) and the edge colours
(`HALPE26_EDGE_COLORS_RGB`, from rtmlib's three colours `LEFT_RGB`, `RIGHT_RGB`,
`CENTRE_RGB`) follow rtmlib's `visualization/skeleton/halpe26.py` (rtmlib 0.0.16,
Apache-2.0 as its installed package metadata and LICENSE file state,
https://github.com/Tau-J/rtmlib) index for index. The keypoint colours
(`HALPE26_KEYPOINT_COLORS_RGB`) are gbf's own: the same three colours by body side, which
differ from rtmlib's for 10 of the 26 keypoints. When more material is copied, each item is
listed here with its origin, licence and location; files ported from GameBuddy
(imhaojie/BroadcastCode-2026, MIT) keep their "Copyright (c) 2026 GameBuddy contributors"
notice.

## Runtime dependencies

`uv sync` installs the Python dependencies pinned in `uv.lock`, each under its own
licence; this repository does not redistribute them. The direct dependencies, with the
licences declared in their installed package metadata for the versions locked on
2026-10-21:

| Package | Declared licence |
|---|---|
| numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| pyarrow | Apache-2.0 |
| jsonschema | MIT |
| pillow | MIT-CMU |
| kloppy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| scipy | BSD-3-Clause |

Development tools (dependency group `dev`, not needed at run time): pytest (MIT) and
ruff (MIT). The optional pose stage (dependency group `pose`, installed by default by
`uv sync` but not a dependency of the published package): onnxruntime (MIT) and rtmlib
(Apache-2.0); rtmlib pulls opencv-python and opencv-contrib-python (Apache-2.0), whose
Windows and Linux wheels bundle an LGPL-licensed FFmpeg library
(`opencv_videoio_ffmpeg*`). Transitive dependencies are listed in `uv.lock`. A complete notice generated
from the lock file and the registries (pip-licenses with an SPDX allowlist, plus a REUSE
file-level scan that also covers wheels with bundled native libraries, submodules and
copied code) arrives in P5.

## External tools

FFmpeg (`ffmpeg`, `ffprobe`), Git, uv and Python are installed by the user and are never
bundled with this repository or its releases; see
[`registries/tools.toml`](registries/tools.toml). Static FFmpeg builds that include x264
or x265 are GPL-licensed, which is one more reason they are never redistributed.

## Plugins, models and data

Plugins (plugin-sngs, plugin-sam3d, plugin-research-nc and others) live in their own
repositories under their own licences and carry their own notices
([ADR 0001](docs/adr/0001-license-architecture.md)). The licences and restrictions of the
models and datasets gbf may use are recorded in
[`registries/models.toml`](registries/models.toml) and
[`registries/data.toml`](registries/data.toml). Data fetched at run time (for example
StatsBomb Open Data and SoccerNet labels) is never redistributed by this repository
([ADR 0002](docs/adr/0002-data-and-media-policy.md)).

The RTMPose Halpe-26 ONNX weights (OpenMMLab mmpose, used through rtmlib, whose code is
Apache-2.0) are downloaded at run time by `python -m gbf models fetch`, verified against
pinned sha256 values and kept outside the repository; they are not distributed with gbf.
Their training-data licences are unverified, so pose results are experimental and for
private use only (see `registries/models.toml`, entry `rtmpose-halpe26`).
