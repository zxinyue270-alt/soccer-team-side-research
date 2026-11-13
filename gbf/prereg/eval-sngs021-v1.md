# Supplementary evaluation protocol: SNGS-021 tracking, pitch position and speed (v1)

Written and committed before any prediction of the run below was compared with the
annotations. The freeze is the commit that adds this file together with
`src/gbf/evaluate/gsr.py`, `tests/test_evaluate_gsr.py` and the TrackEval runner
`tools/run_trackeval.py` (section 4); none of them changes afterwards except as a recorded
deviation. The project has no remote yet, so, like gates.v1 (`prereg/README.md`, "Status of
v1"), this freeze is a local commit without a server-side time, and nothing here may be called
pre-registered.

Results are worded as the deviation of gbf's output from the GSR annotation (the reference),
never as error (`gates.v1.json` → `gsr_evaluation.wording_on_screen`).

## 1. Question and scope

How far does gbf's run on the SN-GSR-2025 validation sequence SNGS-021 deviate from the
sequence's official annotations, in:

- **T** person tracking in the image (detection and identity),
- **P** pitch positions of the people it tracks,
- **S** speeds of the reference people (players, goalkeeper and referees), per frame and as the
  peak of a track segment?

Out of scope here: highlight detection, ball tracking, team and jersey attributes, the 3D lift,
the nearest-opponent distance and turn angles (D7), audio, and GS-HOTA (the run had jersey OCR
switched off, so GS-HOTA would mostly measure the missing jersey numbers).

## 2. Relation to the pre-registration (`gates.v1.json`)

- **Not a v1 split clip.** SNGS-021 was chosen on 2026-10-22 as the lexicographically first
  sequence of the validation archive, before its frames, annotations or any model output were
  looked at; it was not chosen by v1's split rule. Under that rule (order the valid sequences
  by `sha256("gbf-gsr-v1/" + name)`), it ranks 49th of the 58 sequences listed in the
  archive's central directory: outside the dev clips (ranks 1-6) and outside the holdout
  fallback (ranks 7-18). Running gbf on it touched neither split.
- **Supplementary, certifies nothing.** One clip from outside the pre-registered split is not a
  pre-registered validation and changes no metric tier (ADR 0004). The upstream models may have
  seen SN-GSR data, so it is not a pristine held-out test either.
- **What follows v1** (`gsr_evaluation`): per-frame Hungarian matching on box IoU accepted at
  IoU >= 0.5; position deviation between the predicted foot point (box bottom centre through the
  predicted calibration) and the reference pitch position; the same zero-phase smoother for
  prediction and reference (its last stage only, see section 7 S); peak-to-peak speed deviation
  per reference track segment of at least 25 frames with fresh predicted calibration (speed and
  frames as in D6, D9 and D10); the noise floor of the reference speeds; medians and P90 with
  bootstrap intervals (2000 resamples, seed 0); stratification by distance-to-camera terciles
  and by the depth direction (P-primary only, D8).
- **Deviations from v1, stated before any result:**
  - D1. v1 fits the smoother's parameters on four train clips; that was never done. The run's
    own committed smoother is used for both sides: Savitzky-Golay, window 11 frames, order 2,
    speed as the 0.4 s chord (`gbf.pipeline.KINEMATICS_PARAMS`, `gbf.kinematics`).
  - D2. v1 bootstraps over clips (or matches); with one clip that is impossible. Intervals here
    resample reference tracks (a cluster bootstrap within the clip) and say nothing about
    variation between clips.
  - D3. v1 reports the noise floor separately for key frames and interpolated frames. The GSR
    file marks neither (its boxes look interpolated between key frames), so it is reported once
    for all frames and key frames are not inferred.
  - D4. The reprojection error on annotated pitch-line frames is left out by choice (the file has
    pitch lines on all 750 frames), to keep this supplementary evaluation small; P therefore
    cannot separate calibration deviation from box deviation.
  - D5. v1's 720p handling does not apply (the sequence is 1080p).
  - D6. Speed is the 0.4 s chord speed whose segment maximum gbf shows as the peak, not v1's
    "peak of the smoothed speed"; `prereg/README.md` (Pending for v2, P1) leaves that choice to
    v2, so no S result here cites v1 as certifying the shown value.
  - D7. v1's turn-angle and nearest-opponent comparison is not done.
  - D8. v1 stratifies its reporting by distance-to-camera terciles and depth direction; here
    only P-primary is stratified.
  - D9. v1's "fresh predicted calibration" is read as `pitch_usable` of the frame in
    `refine/calib.parquet` (a refitted frame whose own keypoints reproject within 8 px; on this
    run each has at least 9 keypoints, and 51 of them were "held", not fresh, in the plugin's
    calibration).
  - D10. S1 takes both peaks over F only, the frames where gbf shows a valid speed for the paired
    track; v1 restricts frames by calibration only. S1-all is the closer reading of v1 and is
    reported with the same prominence.

## 3. What was seen before this protocol was frozen

- Annotations: the file's structure and counts (roles, track ids, no ignore regions, a pitch
  position on every object row, objects per frame) and, at acquisition, a 50-frame contact sheet
  of the source frames. No annotation value was compared with a prediction.
- Predictions: the run was made on 2026-10-22 without the annotations (its scripts record
  `official_annotations_used_for_prediction: false` and `accuracy_evaluated: false`); its
  operational check (counts only), 7 source frames, 2 key frames and 10 frames of gbf's
  rendered full clip (gbf's own overlay, not the GSR annotation) were looked at; while writing
  this protocol, prediction-only counts were read (93 person track ids; roles player, goalkeeper
  and referee only; 749 of 750 frames with usable calibration).
- No script had compared predictions with the annotations (checked by reading every script of
  the run's execution folder that names the annotation file).
- Review before the freeze (2026-10-25), each side alone: annotation counts (23 people tracks,
  35 segments of 80-750 frames, gaps of 1 or at least 13 missing frames, 2 images without a
  labelled camera), |bbox_pitch - bbox_pitch_raw| (median 0.16 m, P90 1.10 m), frame-to-frame
  smoothness of `bbox_pitch` and of the image boxes, axis orientation; prediction counts
  (import and refine boxes and ids identical, plugin calibration fresh on 699 and held on 51
  frames, refine usable on 749 frames with at least 9 keypoints and one camera centre, one shot,
  no cuts, 89 kinematics tracks with 8571 valid rows).
- The evaluation code was tested on synthetic data only, and the TrackEval runner only on
  synthetic MOT files with hand-computed answers (a perfect track, one identity switch, one
  prediction on a distractor box), before the freeze.

## 4. Data and versions

- Reference: `Labels-GameState.json` of SNGS-021 (SN-GSR-2025 `valid.zip`, revision
  `ce84ee7d9acb2f9fe999ceed2b38e303ffa5efb9`), sha256
  `b0cc0802499a6e5e7b80ef8b76b27c4f9757c49feccb2d3f72971e5b3c0d4e7f`, held under
  `$DATA_ROOT/private/gsr-validation-SNGS-021/source/SNGS-021/` (no_share: never committed).
  Source image `N.jpg` is canonical frame `N - 1` (25 fps CFR).
- Prediction: the run `gbf-750-openvino-safe` (gbf `a5196a9`, clean; upstream tracking,
  calibration and ball detection by the CPU runner with jersey OCR off), files by sha256:
  - `import/tracks.parquet` `acce97910145ed18b0c2481cd562545cdf52efefb955f0bb1b358c9c5575dc0a`
  - `refine/tracks.parquet` `dae72d91a20fbf54d77436bfa9d9587c2176d7cd6139c80d08d43c86d162a82f`
  - `refine/calib.parquet` `8ccc112d7561dc214175930c20398d2057e2a1c52eb8f22052482273410cd1ff`
  - `kinematics.parquet` `2d2f25ca0b08ad233eab28d9a6836e6c64465b7eadb8ee0b171632bc249501a1`
  - `run.json` `587001e55127057afdd55fc8c30857bffcce254a777edd3aaa05e8c69c326211`
- `git diff a5196a9 522b7dc` touches only `src/gbf/web`, docs, `gate` and tests, so every stage
  that made these tables is unchanged in the current code. The run's input was
  `SNGS-021-30s.mp4` (sha256 `4c43ca06624d4e2db8f5972b4acb377dabfd6c1b718bfb2f71304bd0208eca11`),
  a CRF 18 H.264 re-encode of the 750 JPEGs (750 decoded frames, 25 fps CFR, first pts 0); its
  frame k is image k + 1 by construction (check 0 of section 8 verifies it).
- Evaluation code: `gbf.evaluate.gsr` at the freeze commit (`python -m gbf eval-gsr`). Tracking
  metrics: sn-trackeval 0.4.0 (MIT, TrackEval's `MotChallenge2DBox`), run out of process by
  `tools/run_trackeval.py` in the upstream runner's interpreter on files gbf writes; `eval-gsr
  --trackeval-python <that interpreter>` runs it and adds its summary, the runner's sha256 and
  the installed sn-trackeval and Python versions to `results.json`.

## 5. Sets

- Reference people: object annotations with role `player`, `goalkeeper` or `referee` (23 track
  ids). Ball rows are not used. The sequence has no `other` rows and no ignore regions; the
  evaluation refuses a file with ignore regions.
- Predictions: every row of the evaluated tracks table, all roles (player, goalkeeper, referee),
  no confidence threshold beyond the export's own.
- Frames: all 750.

## 6. Matching (M)

Per frame: IoU on continuous coordinates (area = w * h, no +1 pixel, as TrackEval computes it)
between reference boxes `bbox_image` [x, y, x + w, y + h] in ascending track id order and
predicted boxes [x1, y1, x2, y2] in ascending `track_id` order; IoU below 0.5 is set to 0;
`scipy.optimize.linear_sum_assignment(score, maximize=True)`; the assigned pairs with
IoU >= 0.5 form M, computed on the tracks table it serves (refine for P-primary and S, import
for P-secondary-1). T uses TrackEval's own matching.

## 7. Measures

Rows with a non-finite predicted or reference value are excluded from a measure and counted; a
statistic over an empty set is reported as null with n = 0. Quantiles everywhere (medians, P90,
tercile cut points, interval ends) are `numpy.quantile` with its default linear method.

### T. Tracking (image plane)

TrackEval `MotChallenge2DBox` (sn-trackeval 0.4.0 in the upstream runner's interpreter) with
BENCHMARK `GSR`, SPLIT_TO_EVAL `eval`, SEQ_INFO {`SNGS-021`: 750}, DO_PREPROC true, metrics
HOTA, CLEAR and Identity at THRESHOLD 0.5 (TrackEval adds Count). Reference rows:
`N, track_id, x, y, w, h, 1, 1, 1` with N = frame + 1 (rows of role `other`, of which SNGS-021
has none, would be written as class 8, a distractor TrackEval's preprocessing removes
predictions on); prediction rows:
`frame + 1, track_id, x1, y1, x2 - x1, y2 - y1, 1, -1, -1, -1`. Reported as fractions: HOTA,
DetA, AssA and LocA as means over TrackEval's 19 alpha thresholds (0.05-0.95); MOTA; MOTP (mean
IoU of CLEAR matches); IDF1; IDSW; Frag; Rcll = CLR_Re and Prcn = CLR_Pr (CLEAR at IoU 0.5);
GT_IDs and IDs.

- T-primary: `refine/tracks.parquet` (what gbf analyses).
- T-secondary: `import/tracks.parquet`, whose boxes and ids equal refine's (checked on the
  prediction side before the freeze); the evaluation reports whether every number it reports
  equals T-primary's.
- Cross-check: recall and precision of M (matched pairs over reference boxes and over
  predictions). They may differ slightly from CLEAR's Rcll and Prcn, which prefer continuing a
  previous frame's match.

### P. Pitch position

For every pair of M whose predicted row has a pitch position (`pitch_usable` true and finite
`pitch_x`, `pitch_y`), the deviation `d = p_pred - p_ref` in metres, with p_ref =
(`bbox_pitch.x_bottom_middle`, `bbox_pitch.y_bottom_middle`), the field the official GS-HOTA
evaluator reads; `bbox_pitch_raw` is never used. Both are in the SoccerNet world frame (origin
at the centre spot, x along the touchlines, far touchline at y = -34); no axis mapping is
applied. The plugin's boxes and intrinsics are used unchanged (principal point at (W/2, H/2));
the reference's pixel convention is unknown, and a difference of up to half a pixel is not
corrected, before or after the results.

- P-primary: `refine/tracks.parquet` positions (gbf's refit calibration). Frames 623 and 637
  (images with `has_labeled_camera: false`, whose rows have no `bbox_pitch_raw`) are included;
  P-primary is also reported without them.
- P-secondary-1: `import/tracks.parquet` positions (the upstream calibration), reported on its
  own pairs and, for the comparison with P-primary, on the pairs where both tables have a
  predicted position.
- P-secondary-2: smoothed positions behind the speeds: `kinematics.parquet` x, y of the paired
  predicted track against the reference track's positions smoothed by the reference smoother of
  S, on annotated (not filled) frames of reference segments of at least 25 frames where the
  paired predicted track has a `kinematics.parquet` row that is not interpolated.
- Reported: n, median, P90 and mean of |d|; coverage (pairs with a predicted position over all
  pairs of M); P-primary by reference role.
- Context, annotation-only: median and P90 of |bbox_pitch - bbox_pitch_raw| over the reference
  rows of the P-primary pairs where both exist.
- Distance-to-camera terciles (P-primary): c = (C[0], C[1]) from `refine/calib.parquet` at the
  pair's frame; distance = |p_ref - c|; cut points q1, q2 = quantiles 1/3 and 2/3 of the
  distances of all P-primary pairs; near: distance <= q1, middle: q1 < distance <= q2, far:
  distance > q2; median and P90 of |d| per tercile.
- Depth direction (P-primary): u = (p_ref - c) / |p_ref - c|; radial = d · u (positive:
  predicted farther from the camera than the reference); lateral = d · (-u_y, u_x) (positive:
  towards the image's right for a camera on the near side); for each, the signed median and the
  median and P90 of its absolute value.

### S. Speed

Reference speed. For each reference track: its annotated frames are split into segments
wherever more than 3 consecutive frames are missing (frame difference > 4); inside a segment
missing frames are filled by linear interpolation; a segment's length is last frame - first
frame + 1, filled frames included. Segments of at least 25 frames are smoothed, x and y
separately, by `scipy.signal.savgol_filter` (window 11, order 2, mode `interp`); shorter segments
get neither smoothed positions nor speeds. Speed at frame f = |p_s(f + 5) - p_s(f - 5)| / 0.4 s,
defined only where f - 5 and f + 5 are annotated (not filled) frames at least 5 positions from
both ends of the segment (its first frame is position 0). The reference gets no plausibility
cap, no derivative test, no outlier tests and no in-pitch filter: implausible reference speeds
are counted by S3 instead. A test pins this computation to `gbf.kinematics.compute_kinematics`
on clean synthetic tracks.

Limitation stated before any result: `bbox_pitch` is smooth from frame to frame (second
differences of a track's positions: median 0.004 m, against 0.14 m for `bbox_pitch_raw`) and the
boxes look interpolated between unmarked key frames, so the reference positions were smoothed
or interpolated upstream over an unknown span. gbf's smoother is applied on top of that; S
measures agreement with this annotation-derived speed, not with true speed, and reference peaks
between key frames may read lower than the true ones.

Predicted speed. `kinematics.parquet` `speed` where `valid`, of the predicted track that M pairs
with the reference track on that frame (same frame and `track_id`).

- **S1 (primary) peak deviation.** For each reference segment of at least 25 frames, F = its
  frames where the reference speed is defined, M pairs it with a predicted row whose speed in
  `kinematics.parquet` (same frame and `track_id`) is valid, and the frame is `pitch_usable` in
  `refine/calib.parquet`. A segment with |F| >= 25 is comparable: deviation = max over F of the
  predicted speed - max over F of the reference speed. Reported: the number of segments and of
  comparable ones; signed median; median and P90 of |deviation|; by reference role; per segment
  |F| over its frames with a defined reference speed; the share of comparable segments whose
  reference peak over all defined frames lies in F. Sensitivity, never replacing S1: S1 over the
  comparable segments where no reference speed on F exceeds 11 m/s and no reference
  acceleration on F exceeds 10 m/s^2.
- **S1-all (secondary, reported with S1):** for every reference segment of at least 25 frames
  with at least one valid paired predicted speed, the predicted peak over those frames minus the
  reference peak over all its defined frames; segments with none are counted.
- **S2 per-frame deviation (secondary).** predicted - reference speed on every frame of F of
  every comparable segment: signed median, median and P90 of |deviation|.
- **S3 noise floor.** Share of defined reference speeds above 11 m/s, and of reference
  accelerations above 10 m/s^2, the acceleration at f being |v(f + 5) - v(f - 5)| / 0.4 s on the
  reference chord velocities where both are defined.
- **S4 identity exposure.** For each comparable segment, the share of F where M pairs it with a
  predicted track other than the one it pairs with most often on F (ties: the lowest
  `track_id`); S1 is also reported for the segments whose F is paired with a single predicted
  track.

### Intervals

Intervals are given for the median and P90 of |d| (P-primary, P-secondary-1, P-secondary-2), of
|S1 deviation| and of |S2 deviation|, and for nothing else. For each of these separately: the
clusters are the reference track ids with at least one value of it, sorted ascending (n of
them); a new `numpy.random.default_rng(0)` draws 2000 resamples, each
`rng.integers(0, n, size=n)`; every value of a drawn track counts once per draw; the interval
is the 0.025 and 0.975 quantiles of the 2000 statistics. Tercile cut points are fixed from the
full sample. The intervals treat tracks as independent although all share each frame's
calibration and one camera centre, so they understate the uncertainty even within this clip
(D2).

## 8. Sanity checks and stop rules (before any other number is computed)

The evaluation first computes only check 0, the share of reference boxes matched under M, the
median IoU of M's pairs and the P-primary median |d|.

0. Frame alignment, independent of predictions and annotations: for decoded frames k = 2, 200,
   400, 600 and 747 of the run's input video, the mean absolute grey-level difference to source
   image k + 1 + o is smallest at o = 0 among o = -2..2.
1. The reference images are exactly 000001.jpg-000750.jpg (frame = N - 1), every predicted frame
   lies in 0-749, and both sides use 1920 x 1080.
2. Stop if fewer than 30 % of reference boxes are matched, or the median IoU of the matched
   pairs is below 0.6.
3. Stop if the P-primary median |d| exceeds 10 m.

If a check stops, nothing else is computed or written: `results.json` holds only these numbers
and the failed check. After a stop only these conventions may be changed, each verified on one
side alone: box layout (x, y, w, h versus x1, y1, x2, y2; corner versus centre), frame base and
the N.jpg = N - 1 mapping, image size, pitch-axis signs and origin. Any other change, and any
change made without a stop (code fixes included), is a deviation (section 10). The result
computed exactly as frozen stays the headline and the basis of section 9 unless the deviation is
one of these convention fixes.

## 9. Use of the results

- No gbf parameter is tuned on this clip. A change motivated by these results is developed and
  judged on the v1 dev clips only; the v1 holdout is used only as gates.v1 prescribes, never to
  choose between changes.
- No tier or registry status changes on the strength of one supplementary clip.
- Pre-declared display recommendations to the owner, read from the unrounded point estimate (not
  the interval) of the result computed as frozen: if the P-primary P90 of |d| is >= 1.0 m, pitch
  positions shown on screen should be rounded to 0.5 m instead of 0.1 m (distances between two
  people are not measured here and get no recommendation); if S1 has at least 10 comparable
  segments and its P90 of |deviation| is >= 1.0 m/s, the card's peak speed should be shown in
  whole km/h with 约; with fewer than 10 comparable segments no speed recommendation is made.
  These are recommendations, not automatic changes.

## 10. Outputs and deviations

Results go to `$DATA_ROOT/private/gsr-validation-SNGS-021/eval/v1/` (no_share): `results.json`
(every number above, the input hashes with the video's, the code and tool versions, the
TrackEval outputs and a
`deviations` list with reason and time for anything done differently from this text) and a
Chinese summary for the owner. A deviation found after the freeze is appended there; this file
stays as frozen. The result computed exactly as frozen is always reported; a deviation adds a
number, it never replaces one.
