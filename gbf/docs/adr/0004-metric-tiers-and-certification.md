# ADR 0004: Metric tiers and certification

## Status

Accepted, 2026-10-21.

## Context

- Broadcast footage is monocular and usually 25 fps (40 ms per frame). In the demo match
  (720p) main-camera players are estimated at roughly 57-97 px tall, and most close-ups and
  replays lack a camera calibration of their own. The bottleneck is pixels and frame rate,
  not the 3D model.
- No 3D technique metric has a measured error yet, and "3D is better than 2D" is itself
  unproven.
- GameBuddy, the reference project, deployed reinforcement-learning weights that had
  failed their pre-registered gate (one seed, 16 items per group), and its documentation
  did not say so.
- The sn-gamestate tests showed two traps: a threshold written as a rounded percentage
  (96.5% for 82/85 = 96.47%) fails on an exact reproduction and invites quietly moving
  the threshold; and data that was used to choose a method cannot also validate it.

## Decision

### 1. Tiers

[`registries/metrics.toml`](../../registries/metrics.toml) is the single source of each
metric's tier and prerequisites. Renderers, cards and narration read it.

| Tier | Display | Metrics |
|---|---|---|
| official | Shown directly with a source badge; value and error belong to the provider | Single-shot xG; shot technique, body part and outcome; pass end points |
| A | Trajectory level; always with coverage; the error reference names its source | Positions and trajectories; peak speed within a clip; change of direction; run-up angle and speed; nearest in-frame opponent; defenders between shot and goal; defensive line height and width |
| A_candidate | 3D; shown as "unverified" until P4 measures its error **and** it beats its 2D baseline | Body orientation when receiving or passing; trunk lean at shot contact |
| B | Experimental badge; bins and confidence only. General gate: box height >= 150 px (a candidate until the P1 benchmark), bins of at least 15 degrees, gate G12 | Goalkeeper stance, set position, save type and initiation; flexion snapshots; striking body part; passing technique; step frequency; header timing; pre-reception scan; shot pose clusters; first-touch outcome |
| C | Blocked | Angular velocities and kinetic sequencing; foot and ball speed; contact point, duration and spin; centimetre-level support-foot distance; knee valgus; hip rotation; ground-contact time; injury scores; jump height and flight time; full-match distance; acceleration; scanning-frequency rankings; goalkeeper centre-of-mass velocity; first-touch cushioning mechanics |

Why C exists: knee extension reaches 860-1720 degrees per second, 34-69 degrees within a
single frame; an instep contact lasts about 9 ms; the decisive valgus change happens within
40 ms of foot strike; one frame of error at take-off or landing is about 5 cm of jump
height.

### 2. The C-tier block

- C metrics are `blocked = true` in the registry. Any caption, card or narration that
  contains one fails the gate (G13).
- At 25 fps no time derivative of a joint is ever output.
- Interpolated or frame-blended images are detected first and excluded from measurement.
- When a replay's slow-motion rate is uncertain, no time quantity is computed.

### 3. Error status and the "unverified" display

Every evidence record carries `error.kind`, one of `measured`, `literature`, `inferred`
or `unmeasured`, with `value`, `ci95`, `dataset`, `conditions` and `receiptId`.

- **measured**: only from a receipt (`receiptId`) on held-out data, at the delivery
  resolution (720p), with the same method and a confidence interval (G10). Values certified
  on SoccerNet-GSR are worded "deviation relative to GSR annotations (reference)", never
  "error", because the box bottom is not the foot and that bias is shared by ground truth
  and prediction, which makes the number look smaller than the true error.
- **literature**: shown as context with its citation, never as our error.
- **inferred** (propagated from other errors) and **unmeasured**: the card says
  "unverified", and public builds hide unverified cards automatically.
- Official values carry the provider's badge (`claimType` `provider_model` or
  `official_fact`) and are never compared across providers.
- A missing value is written as `unavailable` with a `reasonCode`, never as zero; every
  event yields either a record or `unavailable`.
- At registry level, `error_status` is `provider` (official only), `unmeasured`, or
  `measured` together with `error_receipt`, the receipt that measured it.

### 4. 3D display gate (G12)

A 3D value reaches a card only after five checks: minimum box height, 2D keypoint
confidence, angle change within +/-1 frame, reprojection error relative to box height
(using an independent 2D detector), and the difference between two models. The two-model
difference is used only offline in P4 to calibrate a condition-to-confidence table; at run
time no non-commercial model is called. When research mode is not installed that check
reports `not_run`, the card shows at most "experimental", and it is never counted as
passed. Public builds use only the first four, self-consistency checks. GEM-X can serve as
a commercially usable second opinion, but it calls SAM 3D Body internally, so the two are
not independent.

### 5. The 2D-baseline rule (G20)

Every A_candidate metric has a 2D baseline (`two_d_baseline` in the registry) computed on
the same ground truth. The win criteria are written into gates v2 before the P3 test-set
run. If 3D does not win, the metric ships with the 2D method. Promotion from A_candidate
to A needs both the P4 measurement and the win.

### 6. Which dataset may certify which claim

| Source | May certify | May not |
|---|---|---|
| SoccerNet-GSR holdout clips (1080p and re-encoded 720p) | Trajectory-level deviation relative to GSR annotations: positions, peak speed (peak to peak), change of direction, nearest opponent | An absolute error; any joint-level or 3D metric |
| WorldPose, if approved | Absolute 3D pose accuracy under professional broadcast conditions, published as aggregates if its terms allow | Anything finer than its own 66-80 mm error against Vicon; 15-degree joint bins (its own noise is about 10 degrees, inferred); anything linked to identity |
| Self-recorded multi-camera reference, if filmed | Resolution and frame-rate degradation curves; joint-angle B metrics | Absolute error under professional broadcast conditions |
| SoccerNet-Tracking ball tracks plus two-annotator contact labels | Contact-frame (keyframe) accuracy against a pre-registered criterion, for example at least 90% within +/-1 frame (G14) | Anything else |
| StatsBomb labels (body part, technique, goalkeeper position) | Agreement rates as independent weak labels | Accuracy |
| Private development clip | Integer regression gates (for example 82 of 85 tracks) | Pre-registered validation |
| Published literature | Context for a card | Our error |

Until the keyframe module passes G14, metrics that need a contact frame use only manually
confirmed frames (the override records who confirmed it), or are shown only when every
frame of the uncertain window gives the same bin. If neither WorldPose nor self-recorded
footage becomes available, the 3D A tier stays "unverified", public builds do not show it,
and the README says why.

### 7. Pre-registration by data partition (G9)

- Partitions are fixed by a hash seed before use: by match when the clip-to-match mapping
  is known, otherwise by clip, and the receipt says which. Evaluation data is split into
  development, validation and holdout; the holdout is used once, for the release
  evaluation.
- Development data may be looked at freely once the gates that concern it are frozen.
  Thresholds that judge test data are frozen before the first test run.
- Gates live in `prereg/gates.v<N>.json` (status `proposed`, then `frozen`). Freezing means
  committing and pushing to the project remote before any input the gates judge is
  processed; the server-side push time is the pre-registration time, because local commit
  dates can be edited. A frozen file is never edited: changes go into the next version
  with a reason, and results cite the version they were computed under.
- Thresholds are integer counts (82/85), never rounded percentages.
- The same data never both tunes and certifies a claim; a threshold tuned on development
  data is a regression test, not a validation.
- README result tables are generated from receipts; without a receipt there is no number
  (G10, G11).

### 8. Reporting

Errors and deviations are stratified by distance to the camera and by depth direction,
never summarised by one global P90 alone. P90 comes with a clip-level (match-level when
possible) bootstrap 95% confidence interval, and per-clip values are listed.

## Consequences

- Early builds show many "unverified" cards, and public builds may show few or no 3D
  numbers. That is the intended outcome when nothing has been measured.
- Every A_candidate metric costs a 2D baseline and a shared ground truth.
- Certification depends on decisions and approvals outside the code: the WorldPose
  application, whether to film a self-recorded reference, and SoccerNet's answers.
- Integer thresholds and pushed, frozen gate files make it impossible to move a threshold
  quietly after seeing test results.

## References

- Plan sections 0, 1.1, 3.3 (evidence fields), 4, 5.2-5.4, 7 (G9-G14, G20).
- [`registries/metrics.toml`](../../registries/metrics.toml), `prereg/`,
  [ADR 0002](0002-data-and-media-policy.md) (fitted parameters and their restrictions),
  [ADR 0003](0003-artifact-contracts-and-frame-index.md).
