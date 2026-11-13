# Pre-registration

Thresholds and decision rules are written down before the data they judge is looked at.
GameBuddy's RL stage failed its pre-registered gate and was deployed anyway; this directory
exists so that cannot happen silently here (ADR 0004).

- `gates.v1.json` holds the rules for P1/P1b: the 3D coverage decision, box-height bins,
  team-clustering regression gates, the GSR evaluation split and metric definitions,
  frame-identity checks, and the P5 replacement thresholds.
- `status: "proposed"` means a maintainer has not signed off yet. Nothing listed under
  `blinded_until_frozen` may be computed while the file is in that state.
- Freezing means: set `status` to `"frozen"`, commit, and push to the project remote. The
  server-side push time is the pre-registration time.
- A frozen file is never edited. Changes go into the next version with a reason, and results
  keep citing the version they were computed under.
- Development data (private clip, GSR dev clips, P1 crops) may be inspected freely once the
  gates that concern it are frozen; before `gates.v1.json` freezes, the items listed under
  `blinded_until_frozen` stay unseen even when they come from development data, because
  v1 fixes the thresholds they would inform. Test data (GSR holdout, demo-match NDA windows,
  WorldPose and self-recorded holdout) is touched only after the relevant gates are frozen.

## Status of v1

`gates.v1.json` is marked `"status": "frozen"` in the working tree, but it is not
pre-registered yet. By the freeze protocol the pre-registration time is the server-side
time of the first push to the project remote, and the project has no remote yet (the
file's `signoff.remote_push` says so too). Until that push v1 is frozen locally only: no
result may be called pre-registered under v1, and everything computed before the push,
including every M1 run on the private development clip, was computed without a
pre-registration. The file itself, status field included, stays unedited.

## Errata

Errata correct facts that a frozen file states about the data. They change no threshold or
rule, and the frozen file itself stays unedited.

- **E1 (2026-10-21), `gates.v1.json` → `cut_detection.private_clip`.** The list of hard cuts
  of the private development clip was written from a coarse look at the sn-gamestate run and
  is wrong in two places: frame-by-frame inspection shows five hard cuts, not four, because
  one cut was missing and one listed cut is one frame late. The regression check on this
  clip is evaluated against the corrected list of five cuts. That list, with its frames and
  histogram scores, stays outside the repository with the clip's manual labels, in
  `gates-v1-e1-cuts.json` (tag `no_share`, sha256
  `0f1e035b540ddfe67dc6cfeb8c7d1ae3e51edb6978ad5aea79a46e6c87b6c10b`). The weakest of the
  five cuts scores about 0.02 above the 0.5 threshold, a known weak point of the threshold.

## Pending for v2

Rule changes go into the next version with a reason; v1 stays as frozen.

- **P1 (2026-10-21), `gsr_evaluation.peak_speed_error`.** v1 certifies the peak of the
  smoothed speed. Since M1's review rounds, the peak speed gbf shows (card, overlay, sprint
  cue) is the highest 0.4 s chord speed of the cleaned, smoothed positions
  (`gbf.kinematics`, `gbf.measure` peak-speed). v2 must say which of the two the gate
  certifies, and compute prediction and ground truth the same way, before any P4 holdout
  run. Until then no peak-speed result may cite v1 as certifying the shown value.
- **P2 (2026-10-21), `cut_detection.private_clip`.** v1 states the private clip's cut times
  in the tracked file, while ADR 0002 lets only the clip's sha256, integer regression
  thresholds and aggregate statistics into the repository. v2 keeps values specific to
  the clip (frames, times, per-frame scores) out of the tracked file: it states the gate as
  counts, for example all five cuts of the private list found within one frame and no
  other cut, and cites the sha256 of the private file that holds the list.
- **P3 (2026-10-26), `gsr_evaluation`: jumps in the reference positions.** The
  supplementary SNGS-021 evaluation (`prereg/eval-sngs021-v1.md`, computed after its freeze)
  found frames on which the annotation's raw pitch positions (`bbox_pitch_raw`) of all people
  jump together by 2 to 6 m from the previous frame, while the image boxes keep moving a few
  pixels per frame and gbf's positions stay continuous: 11 of the clip's 750 frames. The
  `soccernet-gsr-2025` note in `registries/data.toml` says how this can happen (reference
  positions are box bottoms projected with per-frame camera parameters fitted to annotated
  line points); the label file carries no camera parameters that would show which
  calibration is off. The smoothed `bbox_pitch` that the measures read spreads each jump over
  about 25 frames, which produced reference speeds up to 18 m/s and the tail of that clip's
  peak-speed deviation (P90 8.3 m/s over 32 segments, against 1.06 m/s over the 15 segments
  whose reference peak lies outside the jumps). v1 has no rule for this. v2 states one before
  any v2 run; it reads the reference only, never the predictions:
  - A jump frame is a frame f on which at least 5 reference people also appear on f - 1 and
    the median of their |`bbox_pitch_raw`(f) - `bbox_pitch_raw`(f - 1)| exceeds 2 m (50 m/s,
    beyond any running speed).
  - Frames f - 15 to f + 15 of every jump frame form the jump windows. Position measures drop
    the pairs in a window; speed measures drop window frames from F (S1 still needs
    |F| >= 25) and from the noise floor.
  - Every position and speed measure is reported with and without the exclusion, together
    with each clip's share of window frames. v2 names which of the two each gate reads, and
    whether a clip above a set share of window frames is left out of gate decisions.
  - A clip without `bbox_pitch_raw` cannot be checked and is reported as unchecked.

  The 2 m bound is physical; the 15 frames cover the spread seen on SNGS-021, so that clip
  cannot test the rule. Until v2, an evaluation under v1 may report the excluded variant only
  as a secondary result that its protocol declares before its data is compared, never in
  place of the v1 result.
- **P4 (2026-10-27), `gsr_evaluation`: reference positions that disagree with the
  reference's own pitch lines.** A post-hoc look at the gates.v1 dev evaluation
  (`prereg/eval-gsr-dev-v1.md`, after its freeze) found stretches where the annotation's raw
  pitch positions follow one camera and its annotated pitch lines another: a ground
  homography fitted to the people's box bottoms and `bbox_pitch_raw` places the annotated
  ground lines 15 to 21 px (median) from where they are drawn, against a few pixels elsewhere,
  while gbf's calibration and the plugin's fit those lines. On one dev clip about half of the
  frames are such, and there the reference puts people about 2.7 m farther from the camera than
  the lines imply; across the six dev clips the share runs from about 2 to 50 %. P3's jump
  frames sit at the entry and exit of such stretches and leave their inside unflagged. v1 has
  no rule for this. v2 states one before any v2 run; it reads the reference only, never the
  predictions:
  - A frame is checked when at least 6 reference people have `bbox_pitch_raw` and the frame
    has annotated ground lines. Fit the ground homography from the people's raw pitch
    positions to their box bottom middles, project each annotated ground line's pitch model
    through it, and take the median distance of the annotated line points to their projected
    line. The frame is inconsistent when that median exceeds X px.
  - Every position and speed measure is reported with and without the inconsistent frames,
    together with each clip's share of them and of unchecked frames; v2 names which of the two
    each gate reads. A frame that is both in a P3 window and inconsistent counts once.
  - X is set before any v2 run from a reason stated in advance (for example the annotation's
    click precision), not from results. The dev look used 8 px, chosen after seeing the dev
    results, so the dev clips cannot test X and 8 px is no more than a reference.
