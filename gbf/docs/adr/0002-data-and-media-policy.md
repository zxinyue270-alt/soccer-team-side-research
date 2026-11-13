# ADR 0002: Data and media policy: restriction tags, taint propagation, what never enters git

## Status

Accepted, 2026-10-21. Amended 2026-10-21 (M1's tuned parameters, see Amendments).

> Statements about data terms in this record are engineering judgements, not legal advice.
> Clause summaries are paraphrases; read the terms themselves before accepting them.

## Context

The data the project needs comes with terms that do not mix:

- **StatsBomb Open Data** (user agreement dated 2023-09-08): no providing the data to any
  third party (1.2.1); no commercial exploitation of the data or of analyses derived from
  it (1.2.2); published analyses must show the StatsBomb logo (1.4); users register before
  use (2.2).
- **SoccerNet videos** (NDA with KAUST): non-commercial research only; no sharing;
  disclosure only to representatives, advisers or subcontractors under equal
  confidentiality, or with written authorisation (3.c); obligations for 2 years.
- **WorldPose**: non-commercial academic use only; no training of commercial models; no
  redistribution, including converted model parameters; no identity recognition or
  linking.
- **SoccerNet-GSR**: the Hugging Face card says gpl-3.0, but the footage belongs to the
  league or broadcaster.
- **SoccerNet label files**: no licence declared.

Platform labels are not rights. The Roboflow football datasets show CC BY 4.0, applied by
the uploader, while their frames come from the DFL - Bundesliga Data Shootout competition,
whose rules restrict the data to competition use (7.A) and to participants (7.B).

Uploading is itself a legal act. Kaggle's terms (version of 2025-06-22) allow personal,
internal, non-commercial use only and not on behalf of third parties (section 3); the
uploader warrants having the necessary rights to everything uploaded, private datasets
included (section 6); uploaded content grants Kaggle a perpetual licence to operate the
service (section 9); deleted data takes about 2 months to be purged from storage and up to
6 months from encrypted backups. During the sn-gamestate tests a forwarded broadcast clip
was uploaded to a private cloud dataset before anyone checked section 6.

GameBuddy, the reference project, kept about 65 MB of broadcast screenshots in git and
published renders containing broadcast footage in its Releases.

## Decision

### 1. Restriction tags

Every artifact carries `restrictions`, a list of tags defined by
`urn:gbf:schema:restrictions`. Publishing, Releases uploads, demo-site deploys and
cloud-upload scripts act on the tags below; each tag gets a red/green test in the tool
that enforces it.

| Tag | Source | Action |
|---|---|---|
| `nda` | SoccerNet NDA videos and every per-item derivative (detections, tracks, skeletons, crops) | Refuse publishing and uploading. Process only on the maintainer's machine, or in an approved Kaggle session under the NDA rules below |
| `no_share` | The private development clip; StatsBomb raw data and evidence stores computed from it; SoccerNet label files; NDA case files `cases/<id>.json`; `identity_overrides.json` | Refuse |
| `research_nc` | Outputs of non-commercial models; parameters fitted on WorldPose or FIFA data or on StatsBomb weak labels | Refuse publishing; research reports only |
| `non_commercial` | Analyses derived from StatsBomb; compute products of Kaggle runs | Refuse in commercial channels |
| `statsbomb_logo` | Public analyses that use StatsBomb data | Require the StatsBomb logo automatically |
| `no_identity` | Every WorldPose or FIFA input | Plugins force ReID, jersey OCR and event-to-identity binding off; publishing refuses any identity field (G19) |
| `sam_license` | SAM 3D Body outputs and patches | Outputs may be published, but the publish script attaches the SAM acknowledgement (clause 1.b.ii) and the licence link; distributing patches requires the full agreement |
| `hl3` | PRTReid/BPBReID products | Not used in public videos by default; MIT colour team clustering instead |
| `agpl_network_source` | Plugins that contain AGPL components | Never run server side; static precomputed results may be published |
| `derived_aggregate` | Aggregate statistics: errors, confidence intervals, threshold tables, condition-to-confidence tables | Released only when the terms of every source dataset allow publishing aggregates (`aggregate_publication = "allowed"` in `registries/data.toml`, verified dataset by dataset) |

SoccerNet label files are listed under `no_share` because they declare no licence; the
download pins in `src/gbf/events/pins.toml` tag them the same way.

### 2. Taint propagation

1. **Per-item derivatives** (per-frame skeletons, per-event measurements) carry the union
   of the tags of all their inputs.
2. **Aggregate statistics** do not inherit per-item tags. They carry `derived_aggregate`
   and list their source datasets; the `derived_aggregate` action in the table above
   decides whether they may be released.
3. **Fitted parameters** (thresholds, correction tables, residual networks) record their
   data sources in `registries/metrics.toml` (`fitted_parameters`) and inherit those
   sources' restrictions. Commercial builds accept a fitted parameter only when every
   source allows commercial use: self-recorded footage with consent, IDSSE, Wyscout or
   synthetic data. Quoting a fitted value as a statistic in a report follows rule 2.

### 3. Rights sources

[`registries/data.toml`](../../registries/data.toml) records, for every dataset, where the
permission comes from (`rights_source`), the licence label, the tags its derivatives
inherit, the uses its terms allow, whether aggregates may be published, and `verified_url`
and `verified_on`. A licence label shown by a hosting platform is recorded but is never a
rights source by itself. Upload and publish scripts consult this registry, and the gate
rejects any use of a dataset marked `blocked` or `rejected` (Roboflow/DFL, FIFA Skeletal
Tracking Light). Applications and their dates are tracked in
[`docs/data-access.md`](../data-access.md).

### 4. Never committed

Gate G4 keeps the following out of git:

- any video, frame or screenshot;
- StatsBomb JSON and evidence stores computed from it;
- SoccerNet label files;
- raw plugin outputs;
- SMPL files and non-commercial weights;
- NDA case files `cases/<id>.json`;
- the manual labels of the private clip;
- `identity_overrides.json`;
- secrets (`.env`, `kaggle.json`, tokens).

These exist only through download scripts that pin a source revision and verify a hash,
or under `DATA_ROOT`, outside the repository, carrying their restrictions. A binary media
file may be tracked only if it is registered in
[`registries/media.toml`](../../registries/media.toml) with its sha256: synthetic or
self-owned images only, never broadcast pixels.

### 5. NDA video

- **Local by default.** The directory that holds NDA video must not be inside a synced
  folder (OneDrive or similar). Until SoccerNet answers in writing, only a few short
  windows are processed locally.
- **Kaggle only with the maintainer's approval, and only if SoccerNet does not object.**
  Then:
  - videos and frames live in a temporary directory that is not saved with the kernel
    version (not under `/kaggle/working`);
  - outputs are pixel-free parquet/npz files and logs; before finishing, the runner scans
    the output directory and fails on any image or video file extension;
  - every NDA job uses its own kernel name; once its output has been pulled and its hashes
    checked, the kernel is deleted (`kaggle kernels delete`). Batch-kernel outputs stay on
    Kaggle per version until deleted; deletion is permanent, but purging from storage
    takes about 2 months and encrypted backups up to 6 months;
  - the Hugging Face token lives in Kaggle Secrets, never in versioned source. Secrets can
    be attached only in the web editor, not by `kernels push`: attach it once by hand and
    verify that later pushes can still read it. Code never prints or writes it;
  - kernels and datasets are private;
  - outputs are fetched right after the run is confirmed successful: Kaggle CLI 2.2.4's
    `kernels output` ignores the version argument and always returns the latest version's
    files, even when that version failed.
- **Inventory.** Every NDA derivative (a `DATA_ROOT`-relative path, or a kernel name and
  version, with its creation date) is listed in the inventory of
  [`docs/data-access.md`](../data-access.md). It is counted before P5 and before the NDA
  expires.
- Renders that contain NDA broadcast footage are watched privately, watermarked, and never
  enter the repository, Releases, the README or issues.

### 6. Private development clip

The clip is processed on the maintainer's machine only and never uploaded again. Only its
sha256 and integer regression thresholds enter the repository, and aggregate statistics
measured on it as the provenance of tuned parameters (amendment of 2026-10-21); its frame
archive and manual labels stay outside. Whether the copy already uploaded to a private
cloud dataset, and earlier cloud run outputs, are deleted is the maintainer's decision
(plan section 10).

### 7. Compute platforms

Products of Kaggle runs carry `non_commercial` (Kaggle terms, section 3). If the project
goes commercial, compute moves to self-paid GPUs and affected models are retrained.

### 8. Receipt layering

- Receipts about public data are committed under `receipts/public/`; README result tables
  are generated from them.
- Receipts about private or NDA data stay outside the repository under `DATA_ROOT`, and
  every path in them is relative to `DATA_ROOT`. The schema definition
  `urn:gbf:schema:common#/$defs/relpath` rejects absolute paths, drive letters,
  backslashes and `..`.

### 9. Public media and people

- Public content uses only IDSSE data, self-recorded footage with written consent, and
  synthetic images. No club crests.
- Self-recorded footage needs itemised, revocable consent (filming, 3D reconstruction,
  public video, public skeletons and derived metrics) and a guardian's consent for minors.
  Blurring faces is not de-identification: body shape, gait, kit and venue can identify a
  person, so published skeletons count as personal data. Bystanders are avoided or
  blurred. The privacy law of the filming location applies; in mainland China, making
  personal information public needs separate consent (PIPL article 25), and whether gait
  counts as sensitive personal information has not been checked.

## Consequences

- Every artifact can be traced to the terms that govern it, and publishing becomes a
  mechanical check of tags rather than a matter of memory.
- The policy is conservative: work on NDA video and StatsBomb data cannot be shown
  publicly. Publishing a StatsBomb-based analysis (`statsbomb_logo`) needs a release step
  that keeps raw values and the evidence store private; it does not exist yet, so nothing
  derived from StatsBomb is published for now.
- Kaggle runs are non-commercial; a commercial decision means self-paid GPUs and
  retraining.
- Registries must be kept current as terms change; every fact carries `verified_on`.

## Open points

- SoccerNet's answers to three questions: cloud processing of NDA video, publication of
  derived skeletons and coordinates, private watermarked demos for named people
  ([`docs/data-access.md`](../data-access.md)).
- Whether WorldPose's terms allow publishing aggregates.
- Whether SkillCorner's MIT text, which speaks of "Software", covers its data.
- Whether the SoccerNet-Tracking download requires an NDA.

## Amendments

### 2026-10-21: M1's tuned parameters (section 2, rule 3; section 6)

Rule 3 covers parameters fitted to data, but the thresholds, weights and priors of M1,
set while the pipeline was developed on the private development clip, were recorded
nowhere. Two kinds are distinguished:

- **Fitted**: chosen from the clip's data by optimisation or as a measured statistic, for
  example the ratios of the ball detector's box to the projected ball and to the player
  box, measured on the clip (gbf.ball `ball_box_scale`, `ball_to_box_height`).
- **Hand-set**: starting values set by hand (from football facts, detector properties or
  a look at the clip's numbers) and checked by eye on the clip's output.

Both kinds:

- are recorded in [`docs/design/m1-highlights.md`](../design/m1-highlights.md), section
  "Tuned parameters", for every module of the M1 pipeline, with name, value, kind and
  provenance, and with their source: `private-dev-clip` in
  [`registries/data.toml`](../../registries/data.toml), restriction `no_share`.
  `tests/test_docs_params.py` fails when a listed value differs from the code, or when a
  numeric scalar constant of those modules, a numeric field of a listed dataclass, a field a
  listed object changes or an entry of a listed mapping is neither listed nor named, with
  its reason, as not tuned; tuple constants, function defaults and literals inside
  functions are checked only where a row lists them. Aggregate values only (medians, ratios, thresholds); no frames, labels, identities
  or descriptions of the clip;
- may not be presented as validated: reports, cards and documents call them starting
  values, and errors that depend on them stay unmeasured;
- are re-fitted, or confirmed, on public data (SoccerNet-GSR, `soccernet-gsr-2025`) before
  P5, with a receipt. A value that GSR cannot check (it has one main camera, no cuts and
  no ball labels) is named in that receipt as unconfirmed;
- may not be used in a commercial build while the private clip is their only source.

The same tables list dimensions of the pitch and the ball as football facts, so that they
are not taken for tuned values; they need no re-fit. Parameters fitted for a registered
metric stay in `registries/metrics.toml` (`fitted_parameters`).

This also extends section 6: besides the clip's sha256 and integer regression thresholds,
aggregate statistics measured on it enter the repository as the provenance of parameters,
in the design doc and in code docstrings.

## References

- Plan sections 3.2, 3.6, 5, 7 (G4, G17, G19), 8.4-8.7 and appendix B (R1-R4, F1-F6,
  K1-K4).
- [`registries/data.toml`](../../registries/data.toml),
  [`registries/media.toml`](../../registries/media.toml),
  [`registries/metrics.toml`](../../registries/metrics.toml),
  `src/gbf/contracts/schemas/restrictions.schema.json`.
