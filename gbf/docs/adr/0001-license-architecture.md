# ADR 0001: Licence architecture: MIT core, out-of-process plugins

## Status

Accepted, 2026-10-21. Amended 2026-10-21 (RTMPose in the core, see Amendments).

> Licence statements in this record are engineering judgements, not legal advice. Have a
> lawyer who knows open-source licensing review them before any commercial use or public
> release.

## Context

The perception pipeline the project tested end to end, sn-gamestate, cannot be copied into
an MIT repository:

- sn-gamestate is GPL-3.0. It bundles NBJW (GPL-2.0-only), PnLCalib (GPLv2 text with no
  version statement, treated as GPL-2.0-only) and sn-calibration code that has no licence.
- Its default ReID models, PRTReid and BPBReID, are under the Hippocratic License 3.0: use
  restrictions, third-party enforcement rights, not an OSI licence, GPL-incompatible.
- tracklab is labelled MIT, but its bundled trackers derive from code that was GPL-3.0 at
  the time, and it requires ultralytics (AGPL-3.0; the vendor also claims trained weights
  and SaaS back ends).

GPL-3.0 and GPL-2.0-only are mutually incompatible, as are GPL-2.0-only and Apache-2.0.
Relicensing the whole project as GPL or AGPL would therefore still not make these pieces
combinable, and it would keep commercial users and contributors from permissive ecosystems
(PySport, Roboflow) away.

Every stage has a permissive replacement (RF-DETR N/S/M/L and roboflow/trackers under
Apache-2.0, TVCalib's optimisation code under MIT, rtmlib under Apache-2.0, kloppy under
BSD-3-Clause) except 3D body estimation: SAM 3D Body under the SAM License (commercial use
allowed, not OSI) is the only commercially usable monocular route found; SMPL-family
methods, NLF weights, GVHMR and Sapiens are non-commercial.

The GNU GPL FAQ (MereAggregation) treats programs that communicate through pipes, sockets
and command-line arguments as normally separate programs, unless they exchange complex
internal data structures. GameBuddy (MIT), the reference project, already isolates its
AGPL detector as a local experiment and keeps a third-party notice file.

## Decision

### 1. Core repository

- Code is MIT. Original documentation, and annotations made on public or self-owned
  material, are CC BY 4.0.
- The core depends only on allowlisted permissive packages and never imports plugin code
  (`tracklab`, `sn_gamestate`, `ultralytics`, `prtreid`, `torchreid`, `mmocr`,
  `sam_3d_body`, `nlf`, ...).
- The permissive perception stack of P5 (MIT/Apache) ships as an optional extra or a
  separate package with its own lock file and CI, so the core stays light.
- One exception is in force: M1's pose stage (amendment of 2026-10-21, below).

### 2. Plugins

Each plugin has its own repository, virtualenv and licence. The core starts it as a
subprocess and exchanges files only.

- **plugin-sngs** (GPL-3.0): transitional detection, ReID and roles, tracking, pitch
  keypoints, calibration and jersey numbers. GPL-3.0 patches touch only sn-gamestate's own
  wrappers (the calibration-status export lives in
  `sn_gamestate/calibration/nbjw_calib.py`); any change under
  `plugins/calibration/nbjw_calib/utils/` is GPL-2.0-only and lives in a separate patch
  file. tracklab stays pinned (1.3.24) and is patched at run time by one module.
- **plugin-sam3d**: an MIT glue layer that only imports SAM 3D Body, plus a SAM-License
  patch directory. Any change that lands in `sam_3d_body` sources (fp16, SDPA, body
  decoder only) is a derivative work and is distributed under the SAM License with the
  full agreement text.
- **plugin-research-nc**: not installed by default and never part of any distribution:
  NLF-L, the SMPL family, RTMW3D (training-data licences unverified), Field Converter
  weights, and anything fitted on WorldPose or FIFA data.
- **plugin-nbjw** (P5 fallback): the original NBJW repository run as a separate process
  (frames in, homography JSON out).

### 3. Plugin protocol

1. The core writes `job.json`: input hashes, the canonical frame range (ADR 0003) and the
   configuration hash.
2. The plugin runs as a subprocess.
3. It writes parquet/npz files that follow `src/gbf/contracts/schemas/`, plus
   `plugin-receipt.json` (SPDX licence, version, commit, weight sha256, environment).
4. The core validates schemas and hashes before accepting anything.

Output directories are named by the job hash; an existing directory is an error (G17).

### 4. Licence opt-in

`plugin.json` declares the plugin's licence expression. A plugin whose licence is not on
the core's permissive allowlist runs only when the user passes `--allow-license` naming
that licence explicitly, and the choice is written into the run receipt. `NONE` and
`NOASSERTION` never pass implicitly. The exact flag syntax belongs to the plugin runner.

### 5. Model registry and licence gate

Every model, weight file and third-party component is recorded in
[`registries/models.toml`](../../registries/models.toml) with code and weight licences,
restriction tags, status (`planned`, `experimental`, `research_only`, `blocked`,
`plugin_only`), `allowed_in_core`, `default_enabled` and its plugin. Gate G3 enforces the
banned-import and banned-dependency lists in P0; P5 adds pip-licenses with an SPDX
allowlist, REUSE file headers and generated notices (covering wheels with bundled native
libraries, FFmpeg builds, submodules and copied code). Fast SAM 3D Body, the `sam_3d_body`
patches and GEM-X's SAM extraction path are registered under the SAM License and never
enter the core.

### 6. SAM License (version of 2025-11-19)

SAM 3D Body's licence allows commercial use and distribution but is not an OSI licence.
The clauses that shape the architecture:

| Clause | Content | What gbf does |
|---|---|---|
| 1.b.i | SAM Materials and derivative works may be distributed only under this agreement, with a copy of it | Patches to `sam_3d_body` sources live in plugin-sam3d's SAM-License directory, shipped with the full text. Fast SAM 3D Body's `sam_3d_body/` (all 48 files), `tools/` and `demo.py` carry Meta copyright headers, so its top-level MIT label does not cover them |
| 1.b.ii | Publications must acknowledge SAM | The publish script attaches the acknowledgement and licence link to anything tagged `sam_license` (ADR 0002) |
| 1.b.iii | Use must comply with privacy, data-protection and trade-control law | Consent and de-identification rules for self-recorded footage (ADR 0002) |
| 1.b.iv | No reverse engineering | Inference configuration only (fp16, SDPA, body decoder only). Pruning, acceleration or analysis of model internals would need legal advice first |
| 1.b.v | Prohibited end uses | Adopted by reference: plugin-sam3d ships the full agreement and its README asks every user to check their use against clause 1.b.v. The research did not record the list's wording, so it is not paraphrased here; quote it verbatim when plugin-sam3d is created |
| 5.b | The licensee indemnifies Meta against third-party claims arising from use or distribution | Whether public or commercial builds enable 3D by default is an open maintainer decision (plan section 10, item 1) |
| 6 | After termination for breach, the materials must be deleted | Weights are listed in `registries/models.toml`, so every copy can be found |
| 8 | Meta may amend the agreement (within the spirit of the current version); continued use means acceptance | Pinning a version does not freeze the terms: a monthly review diffs the agreement text Meta publishes and records effective dates |

Access and related components:

- The Hugging Face weights need manual approval. The form asks for name, date of birth,
  country, affiliation and job title, geolocates the IP address, and requires accepting
  the licence and Meta's privacy policy. If access is refused, do not borrow someone
  else's account or use a proxy.
- Third-party mirrors are used only when they include the full SAM License text and the
  weights' sha256 matches the official files.
- Fast SAM 3D Body's pipeline also depends on Ultralytics (AGPL-3.0); it is not used until
  a separate AGPL assessment is done.
- GEM-X: code Apache-2.0; its model card says NVIDIA Open Model License (commercial use
  allowed) while the LICENSE file in the weights repository is Apache-2.0 text. Hugging
  Face removed the SAM 3D Body checkpoint it depends on on 2026-10-20 and a new release is
  announced for 2026-10-30; re-verify before any use. Because it calls SAM 3D Body, its
  errors are correlated with SAM 3D Body's, so it is a weak second opinion.

### 7. Publication and hosting

The demo site is static and serves precomputed results only; it runs no inference, which
keeps AGPL-3.0 section 13 out of play. The test for any component is whether its output
may be published: SAM 3D Body output may, with the acknowledgement; `research_nc` output
may not.

### 8. Name and attribution

The project is not called GameBuddy: the MIT licence grants no right to the name. The
README acknowledges imhaojie/BroadcastCode-2026; files ported from it keep
"Copyright (c) 2026 GameBuddy contributors" and are listed in `THIRD_PARTY_NOTICES.md`.

## Consequences

- Downstream users, including commercial ones, can use the core, and contributors from
  permissive ecosystems can work on it. sn-gamestate stays available through plugin-sngs
  as an accuracy reference.
- Detection, tracking, calibration and team assignment have to be rebuilt from permissive
  parts, and detectors used COCO-pretrained or retrained; the default pipeline may start
  out less accurate than sn-gamestate.
- The plugin interface and the licence checks are code to maintain. Plugin users carry
  the GPL, AGPL and HL3 obligations of what they install. GPL-2.0-only and GPL-3.0 patches
  must stay apart.
- 3D analysis depends on non-OSI terms that Meta can amend and on an indemnity clause;
  until the maintainer decides, 3D is not enabled by default in public or commercial
  builds.

## Alternatives considered

- **Whole repository under GPL-3.0** (copy sn-gamestate and tracklab code directly):
  rejected. It does not resolve the GPL-2.0-only, HL3, non-commercial or unlicensed parts,
  and it pushes away commercial users and permissive-ecosystem contributors.
- **Whole repository under AGPL-3.0**: rejected; compared with GPL-3.0 it only adds
  ultralytics.
- **MIT core plus a separate AGPL "full-feature" distribution** that bundles the YOLO
  plugin and runs a hosted demo: not adopted; a static demo of precomputed results avoids
  the need.

## Amendments

### 2026-10-21: RTMPose runs in the core under research_nc (exception to section 1)

M1's pose stage runs RTMPose Halpe-26 in process: `gbf.pose` imports rtmlib lazily, the
optional dependency group `pose` (rtmlib, onnxruntime) is installed by default
(`[tool.uv] default-groups` in `pyproject.toml`), and `python -m gbf highlights` runs the
stage unless `--pose-model none` is given. The code is permissive (rtmlib and mmpose
Apache-2.0, onnxruntime MIT), but rtmlib pulls opencv wheels that bundle an LGPL FFmpeg
library, and the weights' training data, mmpose's body7 collection, includes datasets
with research-only terms that were not checked one by one. Under section 1 such a model
would stay out of the core until its licences are settled (compare RTMW3D in
plugin-research-nc). On 2026-10-21 the maintainer asked for the plan's remaining open
items to follow the plan's recommendations; this amendment follows them and waits for the
maintainer's confirmation. The decision is to record what the code does and to contain it:

- [`registries/models.toml`](../../registries/models.toml), entry `rtmpose-halpe26`:
  `allowed_in_core = true`, `default_enabled = true`, `restrictions = ["research_nc"]`,
  status `experimental`, and `core_exception` pointing at this amendment. The registry
  schema accepts a core entry that carries restrictions or an open licence question only
  with `core_exception`. `rtmw-l` (same family; its training data includes body7) carries
  `research_nc` as well and stays out of the core.
- The `pose` group is not a dependency of the published package. The weights are fetched
  at run time by `python -m gbf models fetch`, checked against pinned sha256 values,
  stored outside the repository and never redistributed.
- Every output of a run that uses the model carries `research_nc`: the pipeline adds a
  copy of the model's registry restrictions (`POSE_MODEL_RESTRICTIONS`, kept equal to the
  registry by a test) to the run's restrictions, and the run's tags are part of every
  stage's cache key, so a re-run with other tags writes every artifact again. Nothing
  derived from it is published or used commercially (ADR 0002, section 1).
- The exception ends before P5: the pose stage moves to a model whose weights and
  training data are verified, or to one hosted in a plugin.

## References

- Plan sections 3.1, 3.6, 7 (G3, G12), 8.1-8.3 and appendix B (S1-S5, E3, E4).
- [`registries/models.toml`](../../registries/models.toml),
  [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md),
  [ADR 0002](0002-data-and-media-policy.md).
