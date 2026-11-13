# Evaluation protocol: gbf on the gates.v1 GSR dev clips (v1)

Written and committed before any prediction of the runs below was compared with the
annotations. The freeze is the local commit that adds this file together with
`prereg/eval-gsr-dev-v1-inputs.json`, `src/gbf/evaluate/gsr_dev.py`,
`src/gbf/evaluate/pitch_lines.py`, `tests/test_evaluate_gsr_dev.py`, `tests/test_pitch_lines.py`
and `tools/run_trackeval_dev.py`; none of them changes afterwards
except as a recorded deviation. The project has no remote, so, like gates.v1
(`prereg/README.md`, "Status of v1"), this is a local commit without a server-side time and
nothing here may be called pre-registered.

Results are worded as the deviation of gbf's output from the GSR annotation (the reference),
never as error (`gates.v1.json` → `gsr_evaluation.wording_on_screen`).

## 1. Question and scope

How far do gbf's runs on the six gates.v1 dev clips deviate from the clips' official SN-GSR-2025
annotations in:

- **T** person tracking in the image (detection and identity);
- **P** pitch positions of the people it tracks;
- **S** speeds of the reference people, per frame and as the peak of a track segment;
- **R** the reprojection of gbf's calibration against the annotated pitch lines;
- **N** the nearest-opponent distance gbf shows on screen?

Turn angles are not evaluable: gbf shows no change-of-direction quantity (the metric
`change-of-direction` of `registries/metrics.toml` is not implemented). Out of scope:
highlight detection, ball tracking, team and jersey attributes, the 3D lift, audio and GS-HOTA.

## 2. Relation to the pre-registration (`gates.v1.json`)

- **The clips.** Ordering the 58 sequences of the SN-GSR-2025 `valid.zip` (revision
  `ce84ee7d9acb2f9fe999ceed2b38e303ffa5efb9`) by `sha256("gbf-gsr-v1/" + name)` gives the dev
  clips SNGS-083, SNGS-023, SNGS-043, SNGS-082, SNGS-058 and SNGS-088 (ranks 1-6); the same
  computation ranks SNGS-021 49th, as found before. The clips come from three matches
  (`info.game_id` 5, 2 and 3). The upstream models may have seen SN-GSR data, so these are
  development clips, not a pristine test.
- **What may be looked at.** `prereg/README.md`: development data may be inspected once the
  gates that concern it are frozen; gates.v1 is frozen locally. This protocol still fixes
  every definition before predictions and annotations are compared.
- **What follows v1** (`gsr_evaluation`): per-frame Hungarian matching on box IoU accepted
  at IoU >= 0.5; position deviation between the predicted foot point (box bottom centre
  through the predicted calibration) and the reference pitch position; the same zero-phase
  smoother on both sides; peak-to-peak speed deviation per reference segment of at least 25
  frames with fresh predicted calibration; the noise floor of the reference speeds; the
  reprojection error on annotated pitch-line frames; the nearest-opponent distance compared
  as the on-screen quantity; medians and P90 with match-level bootstrap intervals (2000
  resamples, seed 0) and per-clip values; stratification by distance-to-camera terciles and
  depth direction.
- **Deviations from v1, stated before any result:**
  - D1. v1 fits the smoother on four train clips before the dev runs; that was not done.
    v1 states no objective for the fit, and the obvious one (the same smoother on both sides,
    chosen to bring the two speeds closest) favours ever heavier smoothing; the peak
    definition itself is open (`prereg/README.md` P1). The run's own smoother is used on both
    sides: Savitzky-Golay, window 11 frames, order 2, speed as the 0.4 s chord. The fit is
    left to v2 together with P1.
  - D3. The GSR files mark no key frames, so the noise floor is reported once for all frames.
  - D4. R is computed on every frame with pitch-line annotations (key frames are not
    marked), for the 20 ground lines only, with a pinhole projection without lens distortion.
  - D5. v1's 720p handling does not apply: all six clips are 1920 x 1080.
  - D6. Speed is the 0.4 s chord speed whose segment maximum gbf shows (P1 pending).
  - D8. Stratification: per clip as the per-clip evaluation reports it (P-primary), and
    pooled for P-primary; speeds are not stratified.
  - D9. "Fresh predicted calibration" is read as `pitch_usable` in `refine/calib.parquet`.
  - D10. S1 takes both peaks over F, the frames with a valid paired predicted speed; S1-all,
    the closer reading of v1, is reported per clip with the same prominence.
  - D11. N exists only where gbf shows it (at highlight moments); "the same segments" is read
    as the same frames gbf used for the shown value.

## 3. What was seen before this protocol was frozen

- Annotations, structure only: per clip 750 images of 1920 x 1080 numbered 000001-000750,
  no ignore regions, every image labelled, no `other` rows, `bbox_pitch_raw` present; images
  without a labelled camera 25, 57, 0, 0, 16 and 5; people tracks 24, 22, 24, 23, 25 and 23;
  game ids as in section 2. No annotation value was compared with a prediction.
- Video: the source JPEGs are 4:2:2 (`yuvj422p`), like SNGS-021's. Each clip's
  `<SEQ>-30s.mp4` is an H.264 CRF 18 encode at 25 fps with `pix_fmt yuvj420p`, as the SNGS-021
  video (750 decoded frames each; the command and checks are in each clip's
  `provenance.json`).
- Upstream run scripts: copies of the SNGS-021 scripts with recorded edits (video path by
  clip, drive-less because tracklab splits "<video>:<frame>" on ':'; the 5/50-frame runtime
  comparison only when its baseline exists), checked by a 5-frame run of SNGS-021 against the
  first five frames of its 750-frame run: ball candidates identical; calibration keypoints with
  identical positions, pitch coordinates and classes and confidences within 2.4e-7 (float32
  rounding; the check first demanded exact equality and was relaxed to 1e-6 after that run);
  K, R and C equal; detection boxes equal (track ids, roles and teams come from whole-run
  aggregation and differ in a 5-frame run, as expected). Two earlier attempts of that 5-frame
  run failed at tracking on the video path and were kept, renamed, as evidence.
- Run interruptions (`batch.log`): memory guards stopped three calibration steps when
  machine-wide available memory fell below their 700 MB floor while the step stayed far under
  its 6000 MB cap: SNGS-083's line calibration at frame 516 of 750 (537 MB available; the step
  held 1.77 GB), SNGS-043's keypoint fallback (407 MB; 2.42 GB) and SNGS-058's line calibration
  (622 MB; 1.74 GB). SNGS-082's keypoint calibration ended at frame 544 when the machine was
  shut down from the Start menu (exit code 0x40010004, Windows event 1074); its 544 cached
  records were first checked to parse with every field. Each clip was resumed (`resume`) from
  its verified canonical frames and per-frame calibration caches, each cache checked against
  its source image's hash on reading; after the first stop the batch did this by itself for
  guard stops caused by low machine memory (at most 3 times per clip). Partial outputs were
  kept, renamed. SNGS-023 and SNGS-088 ran without interruption.
- gbf failure and fix: on SNGS-058 the gbf run of commit 28ab357 stopped in its refine stage.
  The refit of the clip's only camera group failed gbf's held-out check (P90 7391.6 px against
  25 px; the solver used all 400 of its evaluations), so no frame kept a usable calibration
  (654 before the refit, 0 after), and the report note on the refit then formatted the missing
  post-refit jitter. Commit c2097d7 changes only that note (one test added); the refit is
  unchanged. Every clip's gbf pipeline was then run again with c2097d7 on the same upstream
  predictions, the earlier runs kept under a new name. For the four clips that had finished with
  28ab357 (SNGS-083, SNGS-023, SNGS-043 and SNGS-082) the files of the two runs are
  byte-identical except `run.json` and the stage receipts (they name the source they ran),
  `measurements.jsonl` (each record's `method.code.git_commit` and the `evidenceId` derived
  from it; SNGS-043 has no record) and the report and render manifests, which differ only in
  hexadecimal digests.
- Known before the freeze from the predictions alone: SNGS-058's run has no pitch-usable frame
  and no kinematics row, so it has no P-primary value and check 3 stops it; by section 6 it is
  reported with its check numbers and left out of everything pooled. No definition was changed
  because of it.
- Prediction-only counts per clip after the runs (rows and track ids of `refine/tracks.parquet`,
  its roles, `pitch_usable` frames of `refine/calib.parquet`, rows and valid rows of
  `kinematics.parquet`):

  | clip | rows | track ids | pitch-usable frames | kinematics rows | valid | roles |
  |---|---|---|---|---|---|---|
  | SNGS-083 | 9506 | 89 | 646 | 8721 | 6000 | goalkeeper, player, referee |
  | SNGS-023 | 12063 | 69 | 634 | 10563 | 5811 | goalkeeper, player, referee |
  | SNGS-043 | 7739 | 158 | 744 | 8826 | 3597 | goalkeeper, other, player, referee |
  | SNGS-082 | 11519 | 111 | 747 | 11976 | 8624 | goalkeeper, player, referee |
  | SNGS-058 | 8146 | 140 | 0 | 0 | 0 | goalkeeper, player, referee |
  | SNGS-088 | 11966 | 109 | 747 | 12495 | 9123 | goalkeeper, other, player, referee |
- The SNGS-021 supplementary evaluation (`prereg/eval-sngs021-v1.md`) and its results, which
  motivated P3, R and N, were known.
- The projection R uses was checked on prediction data only: on SNGS-021's run, projecting
  gbf's own pitch positions with its K, R and C as x ~ K R (X - C) lands on the box bottom
  middle for 200 random boxes (largest gap 0.0000 px).
- The evaluation code was tested on synthetic data only.

## 4. Data and versions

`prereg/eval-gsr-dev-v1-inputs.json`, frozen with this file, gives per clip the sha256 of
`Labels-GameState.json`, of `<SEQ>-30s.mp4` and of `refine/tracks.parquet`,
`import/tracks.parquet`, `refine/calib.parquet`, `kinematics.parquet`, `measurements.jsonl` and
`run.json` of the run `runs/gbf-750-openvino-safe`, with the counts of section 3. Every run was
made by gbf commit c2097d7 from a clean tree (each `run.json` says so). T uses sn-trackeval
0.4.0 in the sn-gamestate Python 3.9 environment.

## 5. Sets

- Reference people: object annotations with role `player`, `goalkeeper` or `referee`; the
  ball is not used. A clip with ignore regions would be refused (none has them).
- Predictions: every row of the evaluated tracks table.
- Frames: all 750 of every clip.

## 6. Per-clip measures

Every clip is evaluated by the frozen per-clip evaluation of `gbf.evaluate.gsr` (`evaluate`),
whose definitions are those of `prereg/eval-sngs021-v1.md`, sections 6-8: its checks 0-3 and
stop rules, matching M, P (primary, secondary 1 and 2, by role, terciles and depth direction),
S (S1, S1-all, S2, S3, S4) and the import comparison. A clip that stops is reported with its
check numbers and left out of everything pooled.

## 7. Dev additions

- **R.** For every annotated point of a ground line on a frame whose calibration is
  `pitch_usable`: the pixel distance from the point (normalised coordinates times width and
  height, no half-pixel correction) to the nearest sample (at most 0.05 m apart) of the same
  line projected with gbf's K, R and C as x ~ K R (X - C). Points of goal posts, crossbars,
  unknown lines, frames without usable calibration and lines behind the camera are counted.
- **N.** For every nearest-opponent value gbf shows: the reference person M pairs with gbf's
  track on the frames gbf used (none when it pairs with no single person there), and the
  median over those frames of the ground distance (`bbox_pitch`) from that person to the
  nearest reference outfield player (role `player`) of the other team (`attributes.team`);
  deviation = gbf's value - the reference value. Reported value by value; quantiles only from
  10 values on.
- **P3 jump windows** (`prereg/README.md`, P3): a jump frame has at least 5 reference people
  also present on the previous frame with finite raw positions and a median raw step above
  2 m; frames within 15 of a jump frame form the windows. Secondary only: P, S1 and S2 are also
  reported without the window frames (S1 keeps |F| >= 25), with each clip's jump frames and
  window share. The primary results are always the v1 ones.

## 8. Pooling and intervals

Over the clips that did not stop: P-primary |d|, |S1|, |S2| (each with and without the jump
windows) and R are pooled; for each, n, median, P90 and mean, every clip's own values, and the
95 % percentile intervals of the median and P90 from 2000 resamples of matches (`game_id`;
clips when a clip has none), each a fresh `numpy.random.default_rng(0)`, every value of a drawn
match counted once per draw. With three matches the intervals are wide and say little; the
per-clip values carry the information. T is computed by TrackEval per clip and combined over
the clips (`tools/run_trackeval_dev.py`), for refine (T-primary) and import (T-secondary),
with whether every number of the two is equal. Quantiles are `numpy.quantile` (linear).

## 9. Use of the results

- These numbers are the dev baseline: gbf changes are developed and judged on the dev clips
  against them; the v1 holdout is used only as gates.v1 prescribes, after P1 and P3 are
  settled in v2 and the project has a remote.
- No tier or registry status changes on dev results.

## 10. Outputs and deviations

Results go to `$DATA_ROOT/private/gsr-v1-dev/eval/v1/` (no_share): `results.json` (every
number above, inputs, code and tool versions, TrackEval outputs, a `deviations` list) and a
Chinese summary. A deviation found after the freeze is appended there; this file stays as
frozen. The result computed exactly as frozen is always reported; a deviation adds a number,
it never replaces one.
