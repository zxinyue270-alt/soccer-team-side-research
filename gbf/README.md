# gbf (working name)

gbf analyses technique in short windows around football events (a shot, a pass, a save)
in broadcast video, and attaches an evidence trail to every number it shows: where the
number comes from, whether and how its error was measured, how much of the event it
covers, and which data restrictions it carries.

It is a football take on GameBuddy, an NBA broadcast project (see
[Acknowledgements](#acknowledgements)). "gbf" is a working name.

**Status: M1, highlight moments on a development clip.** On top of the P0 contracts, gate,
registries and decision records, `python -m gbf highlights` finds the highlight moments of
a broadcast clip from a tracking plugin's export and analyses them on the original frames
(key moments from 2D pose, measurements with tiers, annotated clips, an HTML report). It has
been run on one private development clip only: nothing is validated and there are no
published results yet.

## What broadcast video can and cannot measure

Broadcast footage is monocular and usually 25 frames per second (40 ms per frame). In the
720p demo match, players in the main camera are estimated at roughly 57-97 px tall, and
most close-ups and replays have no camera calibration of their own. The bottleneck is
pixels and frame rate, not the 3D model, so gbf sorts every metric into a tier
([ADR 0004](docs/adr/0004-metric-tiers-and-certification.md),
[`registries/metrics.toml`](registries/metrics.toml)):

- **Official**: provider values such as single-shot xG, shot technique and pass end
  points, shown with a source badge. Their accuracy is the provider's.
- **A, trajectory level**: positions, peak speed within a clip, change of direction,
  run-up angle, nearest in-frame opponent, defenders between shot and goal. Measured in
  pitch metres on frames with a fresh calibration, always with coverage, and with an
  error reference that names its source.
- **A candidates, 3D**: body orientation when receiving, trunk lean at shot contact.
  Shown as "unverified" until their error is measured and they beat a 2D baseline on the
  same ground truth; if 3D does not win, the 2D method is used.
- **B, experimental**: goalkeeper stance and save type, joint-flexion snapshots, striking
  body part and similar. Bins and confidence only.
- **C, not measured by design**: joint angular velocities, foot and ball speed, ball
  contact point, duration and spin, centimetre-level support-foot distance, knee valgus,
  ground-contact time, injury scores, jump height, full-match distance, acceleration. At
  25 fps the knee can turn 34-69 degrees between two frames and an instep contact
  (about 9 ms) falls between frames; no model recovers that.

A number without a measurement receipt is labelled "unverified"; public builds are to hide
it (ADR 0004; not implemented yet, M1 only makes private reports). A missing value is shown
as unavailable with a reason, never as zero.

## Quickstart

Requirements: Git, [uv](https://github.com/astral-sh/uv), FFmpeg (`ffmpeg` and `ffprobe`
on `PATH`; you install it, gbf never bundles it) and Python 3.11 or 3.12 (uv can install
it). Clone into a path without spaces or non-ASCII characters. uv's cache and any Python
it installs stay inside the checkout under `.local/`, and every tool runs with the
virtualenv's interpreter rather than `uv run`.

Windows (PowerShell):

```powershell
git clone -c core.autocrlf=false <repository-url> gbf
cd gbf
$env:UV_CACHE_DIR = ".local/uv-cache"
$env:UV_PYTHON_INSTALL_DIR = ".local/python"
$env:PYTHONUTF8 = "1"
uv sync
.venv/Scripts/python.exe -m gbf gate --full
```

Linux:

```bash
git clone <repository-url> gbf
cd gbf
export UV_CACHE_DIR=.local/uv-cache UV_PYTHON_INSTALL_DIR=.local/python PYTHONUTF8=1
uv sync
.venv/bin/python -m gbf gate --full
```

`gate --full` is the same offline gate CI runs: static checks (line endings, credentials,
user-home paths and absolute paths through a folder named `gbf-data` or `private`, media,
restricted data, forbidden imports, dependencies, registries, receipts) plus ruff and the
test suite with the network blocked. Without `--full` only the fast static checks run. A
missing tool such as ffprobe is a failure, never a skip.

Commands (`python -m gbf --help`):

| Command | Purpose |
|---|---|
| `gate` | Run the offline quality gate (`--full` for everything) |
| `highlights` | Find and analyse the highlight moments of a clip (M1; see [docs/design/m1-highlights.md](docs/design/m1-highlights.md#running-it)). `--out` must lie outside every git work tree |
| `fetch` | Download pinned open data; requires explicit acceptance of its terms |
| `align` | Align provider events to SoccerNet half videos |
| `ingest` | Probe and normalise videos; write `video.json` |
| `l0` | Run the L0 synthetic smoke pipeline |
| `models` | Fetch or verify pinned model weights (kept outside the repository) |
| `web` | Local web page (127.0.0.1 only) to start `highlights` runs and look at their results ([docs/design/web.md](docs/design/web.md)) |

`python -m gbf web --runs-dir DIR` (default `$DATA_ROOT/runs`) prints an address with a
token drawn for this start and opens it in the browser. The page lists the runs, shows each
highlight with its key frame, clip, key moments and measurements (tiers and badges as in the
report; values from the 3D lift marked as a reference), plays the annotated full clip from
a listed frame, and starts new analyses one at a time. The server listens on 127.0.0.1
only and calls no online service, the page loads nothing from outside, and network paths
such as `\\server\share` are refused.

## Repository layout

| Path | Contents |
|---|---|
| `src/gbf/` | The MIT core package. It never imports plugin code |
| `src/gbf/contracts/schemas/` | JSON Schemas (`urn:gbf:schema:*`) for every artifact, receipt and registry |
| `src/gbf/util/`, `src/gbf/testing/` | Raw-byte hashing, LF-only text IO, repository paths; the network guard used by tests and the gate |
| `registries/` | Models, data rights, media, metrics and external tools (TOML, schema-checked) |
| `prereg/` | Gates fixed before the data they judge is used: frozen in the working tree; pre-registered once pushed ([prereg/README.md](prereg/README.md)) |
| `docs/adr/` | Architecture decision records 0001-0004 |
| `docs/data-access.md` | Applications, agreements, and the inventory of NDA derivatives |
| `tests/` | The offline test suite; media is generated at test time |
| `receipts/public/` | Receipts for public data (README result tables are generated from them; none yet) |

Run outputs, data and anything private live outside the repository under `DATA_ROOT`.
Contribution rules are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Data and licence policy

In short (the ADRs are the reference; none of this is legal advice):

- **MIT core, out-of-process plugins** ([ADR 0001](docs/adr/0001-license-architecture.md)).
  GPL, AGPL, Hippocratic-licensed and non-commercial components never enter the core. They
  run as separately installed plugins that exchange schema-checked files, and a plugin
  whose licence is not permissive runs only with an explicit `--allow-license`. SAM 3D
  Body (SAM License) runs in its own plugin; outputs published from it carry Meta's
  acknowledgement. One recorded exception: M1 runs the RTMPose 2D pose models in process
  (Apache-2.0 code, training data not verified licence by licence), so every output of a
  run that uses them carries the `research_nc` tag (ADR 0001, amendment of 2026-10-21).
- **No restricted data or broadcast pixels in git**
  ([ADR 0002](docs/adr/0002-data-and-media-policy.md)). Videos, frames, screenshots,
  StatsBomb JSON, SoccerNet label files and evidence derived from them are never
  committed; download scripts pin a revision and verify a hash. Every artifact carries
  restriction tags (the union of its inputs'), and publishing, upload and deploy tools
  act on them. SoccerNet NDA video is processed locally by default.
- **One frame index, LF files, no pickle**
  ([ADR 0003](docs/adr/0003-artifact-contracts-and-frame-index.md)). Frames are named by
  (source sha256, constant-frame-rate index); hashes are taken over raw bytes; artifacts
  are Parquet, npz or JSON with schemas; stored paths are relative.
- **Tiers and pre-registration**
  ([ADR 0004](docs/adr/0004-metric-tiers-and-certification.md)). Which dataset may
  certify which claim is fixed in advance, thresholds are integer counts, and gates are
  frozen by pushing them before any test-set run.
- The rights behind each dataset, model and tool are recorded in
  [`registries/data.toml`](registries/data.toml),
  [`registries/models.toml`](registries/models.toml) and
  [`registries/tools.toml`](registries/tools.toml); applications are tracked in
  [`docs/data-access.md`](docs/data-access.md).

## Licence

Code: MIT ([LICENSE](LICENSE)). Original documentation, and annotations made on public or
self-owned material: CC BY 4.0 (ADR 0001); per-file SPDX headers arrive with REUSE in P5.
Data, models, footage and external tools keep their own terms: see the registries and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

gbf is inspired by GameBuddy
([imhaojie/BroadcastCode-2026](https://github.com/imhaojie/BroadcastCode-2026), MIT), an
NBA broadcast project whose evidence discipline it adopts: models only pick ids while
programs copy the numbers, every receipt says what it does not certify, and generation,
delivery and content checks keep separate receipts. gbf is an independent project; it is
not affiliated with or endorsed by the GameBuddy authors, or by any league, broadcaster or
data provider named in this repository. Files ported from GameBuddy keep their
"Copyright (c) 2026 GameBuddy contributors" notice and are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
