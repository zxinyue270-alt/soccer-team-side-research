# Offline product acceptance

Protocol: `prereg/product-acceptance-v1.proposed.md`. The maintainer approved its
implementation on 2026-11-13; new matches, annotators and the project remote are still absent.
The new `eval-product` command scores existing artifacts. It never starts inference,
downloads media, uploads data or changes a metric's certification status.

## Running

From the repository root, with a private acceptance directory selected locally:

```powershell
$acceptanceManifest = Join-Path $env:GBF_ACCEPTANCE_ROOT 'manifest.json'
& .\.venv\Scripts\python.exe -m gbf eval-product --preflight --manifest $acceptanceManifest --out (Join-Path $env:GBF_ACCEPTANCE_ROOT 'preflight-attempt-1')
& .\.venv\Scripts\python.exe -m gbf eval-product --manifest $acceptanceManifest --out (Join-Path $env:GBF_ACCEPTANCE_ROOT 'evaluation-attempt-1')
```

Use a new output directory for each attempt; evidence files are never overwritten.
Exit codes: 0 = a valid diagnostic or passing arithmetic result; 2 = malformed,
inconsistent or changed input; 3 = missing prerequisites; 4 = failed product criteria;
5 = insufficient independent sample size. Always read `data_kind` and `summary.status`:
synthetic/development results are `diagnostic_only`, including when every prediction matches.
No sample or missing blind labels produces `blocked`, not a zero-error result.

## Manifest v1

The manifest is private JSON with `schema_version: 1`, `data_kind`
(`synthetic`, `development`, `independent`), `core_commit` and a `clips` array.
An artifact reference is `{"path": "relative-to-manifest-or-absolute", "sha256": "64 hex digits"}`.
Every consumed artifact is verified against its hash, even optional coverage tables.
JSON duplicate keys and nonfinite constants are rejected.

Each clip has:

- `id`, a unique string; `match_id`, a canonical source namespace plus match ID;
- `stratum`, `event` or `uniform`; integer `frame_count`; rational-string `fps`;
- `video` and `labels`, artifact references;
- `outputs.status`: `passed` or `failed`. A failed run must have `failure_reason`;
  its events remain false negatives. A passed run references `highlights`, `verification`
  and `run`, the native final highlights, operational verification and run receipt;
- optional `outputs.tracks`, `outputs.lift`, `outputs.measurements`, references to native
  refined-track Parquet, lifted-pose Parquet and measurement JSONL files.

Final highlight `fps` and `frame_count`, the verification's `source_frames` and `status`,
the run's video hash, clean Git status and code commit must agree. All candidate highlights
count; a main-only comparison is reported separately. Events are limited to 128 per clip
as a computation guard, not a fitted product threshold; larger lists fail explicitly.

Independent data additionally requires:

- three matches, four 30-second clips each, two per stratum; no known exposed match;
- root `exposure_register_complete: true`, `exposed_matches`, `protocol` artifact reference,
  and a nonempty `runtime` array of pinned runner/model artifact references;
- per clip `local_use_rights_confirmed`, `selection_locked_before_predictions`,
  `selection_receipt` artifact and `frame_map` artifact; the frame map has
  `status: passed`, `canonical_video_sha256`, `frame_count`, and `fps`;
- `outputs.started_at`, an ISO timestamp with a timezone, after both label lock and push;
- a `freeze` object with `status: frozen`, `commit`, `remote_url`, `pushed_at`,
  `inputs_sha256` and `receipt`. The receipt is a hashed local JSON record of that same
  commit, URL, server push time and input fingerprint.

`input_fingerprint` canonicalises all manifest input fields except root `freeze` and
per-clip `outputs`. Thus output hashes can be recorded later without altering the selected
inputs; changing a label hash, selection, model hash or protocol changes the fingerprint.
The source commit tested before freezing is `core_commit`. A later freeze commit must
descend from it without changes to `src`, `tools` or `registries`; the working tree is clean
and checked out at that freeze commit when evaluating independent outputs.
That commit contains `prereg/product-acceptance-v1-freeze.json` with the matching
`inputs_sha256` and `core_commit`. Native run receipts then name the freeze commit.

The command verifies **local consistency of recorded evidence**. It cannot establish
that a named person was actually blind, that rights attestations are true, that a sample
selection receipt is complete, or that a recorded server time came from that server.
Those facts require the preparation review and independently obtained push receipt before
an independent batch. No live network verification is performed by this offline tool.

## Exhaustive labels and coverage

The label document states `frame_count`, `fps`, `exhaustive: true`, `events`, and optionally
`uncertain_intervals` (inclusive start/end frame pairs). An event is:

```json
{"id":"E1","start":25,"peak":40,"end":65,"types":["ball_strike"],"subjects":["subject-A"]}
```

Independent labels also require `locked_at`, two distinct `annotators` with `id` and
`kind: human`, `two_independent: true`, `no_model_outputs_seen: true`, two `originals`
artifact references and an `adjudication` artifact reference. Original annotations and
adjudication are retained privately. Labels must be locked before the pushed freeze.

For coverage, supply `targets` with every declared event subject's visibility on every
frame in peak +/-12, clipped to the video. Shared subject/frame entries are recorded once.
A target is `{subject_id, frame, visible, box}`; visible boxes are `[x1,y1,x2,y2]` in the
canonical video's pixel coordinates. Missing target frames are an error; absent `targets`
produces `not_evaluated_no_targets`, not a smaller denominator. Missing optional output
tables produce `not_evaluated_missing_artifacts`.

Visible subjects are assigned one-to-one to native track boxes per frame at IoU >=0.5,
maximising the number assigned and then box IoU. No detection remains in the denominator.
3D coverage counts a finite 78-coordinate output with method `height_prior_calibrated`
and reason `lifted`, with missing reasons and manual box-height strata. It is skeleton
output availability, not a claim that 3D technique measurements passed their display gates.
Accepted measurement availability is reported separately from raw skeleton availability.

Position availability uses native pitch-usable finite coordinates on the manual peak.
Speed and nearest-opponent availability use `status: ok` finite measurement values for
the associated track **on the exact manual peak frame**. A card at a different frame is
`not_reported_at_reference_frame`, not silently substituted. Nearest-opponent applicability
must be marked in the event's `metric_applicability`; absent/false applicability is counted
separately. True numerical errors and 3D accuracy remain not evaluated without references.

Temporal event matching is exact: maximum cardinality, then maximum sum of rational
inclusive-frame IoUs, then minimum absolute peak-frame difference, then lexical event and
prediction IDs. No floating epsilon or type preference affects the pairing. Unmatched
predictions that overlap uncertain intervals enter the conservative precision denominator.
Type-specific TP/FP/FN additionally count wrong-type pairings; multi-type reference labels
can contribute to more than one type's denominator, so those rows are not a partition.

## Verification

```powershell
& .\.venv\Scripts\python.exe -m pytest -q tests/test_evaluate_product.py
& .\.venv\Scripts\python.exe -m ruff check src tests tools
& .\.venv\Scripts\python.exe -m gbf gate --full
```

Synthetic tests cover exact global pairing against enumeration, denominator preservation,
ambiguous frames, failed runs, native Parquet/JSONL, provenance mismatches, explicit
blocking and immutable output evidence. Deliberate mutations are checked separately.
No synthetic result is a product accuracy measurement. The existing `gbf.label_eval`
report format and its historical results are preserved.
