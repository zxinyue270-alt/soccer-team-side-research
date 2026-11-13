# M1: highlight moments analysed on the original video

Status: in progress (2026-10-21). This milestone pulls a vertical slice forward from P2/P3:
find the highlight moments of a broadcast clip automatically, then analyse them on the
original frames. 3D is a supporting layer (a calibrated, height-prior skeleton), not the
deliverable: its values are a reference, badged 3D 参考, and drawn back on the original
frames. Honesty rules from ADR 0004 apply: every number has an evidence record, error is
"unmeasured" until a receipt says otherwise, and image-plane quantities are labelled as such.

## Inputs

- `video`: a CFR source video (canonical frame index = CFR decode index, ADR 0003).
- `audio_source`: the file that carries the original audio (may be the same file). If the
  input has no audio stream, pass the video itself: the pipeline marks audio features as
  missing, notes that analysis uses the images only, and renders without an audio stream.
  Missing or malformed input files remain errors.
- `sngs_dir`: plugin-sngs export (`detections.jsonl`, `calib.jsonl`, `plugin-receipt.json`),
  written by the GPL side; the core only reads these files.
- `ball_raw`: `ball_raw.jsonl` from Ultralytics YOLO11m with COCO-pretrained weights, run by
  a script outside this repository (no gbf plugin hosts it yet; `registries/models.toml`,
  `ultralytics-yolo11`: AGPL-3.0-only, tag `agpl_network_source`; the `ball-receipt.json`
  next to the file carries the licence and tags into the run), one line per frame:
  `{"frame": n, "cands": [[x1, y1, x2, y2, conf], ...]}` (COCO "sports ball" candidates).

### plugin-sngs export rows

`detections.jsonl`: `frame` (int, canonical), `state_frame`, `track_id` (int, -1 unknown),
`box` [x1, y1, x2, y2] px, `kf_box` (Kalman-smoothed box or null), `conf`, `role`
(player | goalkeeper | referee | other | unknown), `team` ("left" | "right" | null: the side
of the pitch the team defends, x < 0 is left), `team_cluster`, `jersey` (string or null),
`jersey_conf`.

`calib.jsonl`: `frame`, `state_frame`, `calib_status` (fresh | held | none), `calib_age`,
`n_keypoints`, `n_line_classes`, and when calibrated `K` (9, row-major), `R` (9, world to
camera), `C` (3, metres), `radial_distortion` (all zero for NBJW), `width`, `height`,
`rep_err_px` (median pixel distance between this frame's detected pitch keypoints with
p >= 0.5 and the projected model points, or null with fewer than 4 keypoints) and
`rep_err_n_keypoints`. `rep_err_px` is self-consistency, not ground truth.

Usable calibration ("pitch-usable") = `calib_status == "fresh"` AND `rep_err_px <= 8.0` (an
empirical starting value: on the development clip fresh frames have median 3.3 px, P90 4.6 px,
and a few outliers above 100 px). Only pitch-usable frames may feed pitch positions, speeds,
distances or 3D anchoring (gate G6); every other frame is image-plane only.

## Conventions

SoccerNet world frame (metres, origin at the centre spot, x along the touchlines, far
touchline at y = -34, z down) everywhere a pitch coordinate is stored, unless a column says
`_zup`. Pixel centres at integer coordinates. Times are canonical frame indices; seconds =
frame / fps. Speeds in m/s in storage, km/h only on screen.

## Artifacts (Parquet unless stated; written under the run directory, never in the repo)

| File | Producer | Columns |
|---|---|---|
| `tracks.parquet` | `gbf.sngs_import` | frame int64, track_id int64, x1 y1 x2 y2 float64, conf float32, role string, team string?, jersey string?, jersey_conf float64?, foot_u foot_v float64, calib_status string, pitch_usable bool, pitch_x pitch_y float64 (NaN unless pitch-usable), in_pitch bool |
| `calib.parquet` | `gbf.sngs_import` | one row per canonical frame (frames missing from the export get calib_status `none`): frame_index int64, width height int32, K R fixed_size_list<float64>[9] and C [3] (null when none), calib_status, calib_age int32, rep_err float64?, rep_err_n_keypoints int32, pitch_usable bool |
| `refine/tracks.parquet`, `refine/calib.parquet`, `refine/refine.json` | `gbf.enhance`, `gbf.team_check` | the tables above after clean-up and refit: track ids split at cuts (`source_track_id` keeps the plugin's id), jersey kept only when read confidently and consistently; with plugin keypoints (`keypoints.jsonl` + receipt) the calibration of wide-shot camera groups refitted with a fixed camera centre and smooth rotation and zoom (calib_status `refined`, extra columns refined, refit_group, refit_n_points, refit_rep_err) and the ground positions recomputed, except a group whose refit is far worse on held-out keypoints than per-frame fits or did not converge, which keeps the plugin calibration (`fallback` in `refine.json`); team labels checked against the shirt colours on the original frames: a player track whose shirt colour is nearer the other team's than its own by more than the margin loses its label, which is cleared, never flipped; once both teams have a colour, a piece split off at a cut whose shirt could not be checked loses its label too (it was inherited from the id's earlier player). Then the role labels are checked against the one referee the laws allow on the pitch (`gbf.role_check`): when more than half of the frames with a pitch-usable row hold two or more people labelled referee at least 2 m inside the touchlines and goal lines, every team label is cleared and a report note says why; roles are kept (the count says that some are wrong, not which). `refine.json` has the held-out reprojection error against a per-frame free-centre fit, each fitted camera group's solver status, evaluations and whether it converged, the ground jitter before and after, and the team check's report: each track's verdict (consistent, inconsistent, unchecked; a cleared split-off piece also `cleared` "split_piece"), `cleared_inherited_tracks`, and among the counts `rows_cleared` (every player row whose label the refine stage cleared) and `rows_cleared_split_pieces`; tracks with too few shirt samples keep the plugin's label, and the report note counts them; and the role check's report under `roles`: verdict (plausible, implausible, not_judged), frames, crowded frames and their share, the most referees inside at once and the rows whose team label it cleared; last, the plugin's left/right team names are checked against where the teams stand (`gbf.side_check`, report under `sides`: verdict consistent, uncertain, contradicted, not_judged or no_teams, `attack_direction_known`, the right team's mean pitch x minus the left team's over the outfield players' pitch-usable rows (`difference_m`) with its 95 % interval from resampling whole tracks (`interval_m`), and the tracks and rows per team): the plugin names its two clusters by that comparison alone, so when the interval does not lie wholly above 0 (or a team has fewer than two tracks) the direction each team attacks counts as unknown for the later stages and a report note says why (see "Attack direction" under the highlight cues); the labels are never swapped |
| `shots.parquet` | `gbf.shots_grammar` | shot_id int64, start_frame int64, end_frame int64 (inclusive), shot_type string (wide, medium, close), fresh_share float64, median_box_h float64 |
| `ball_track.parquet` | `gbf.ball` | frame int64, u v float64, radius_px float64, conf float32 (NaN unless detected), status string (detected, interpolated, missing), candidate_count int32, pitch_x pitch_y float64 (ball-centre plane z = -0.11, NaN unless pitch-usable), ground_assumption bool |
| `ball_events.parquet`, `ball_contacts.parquet` | `gbf.ball` | `BALL_EVENT_COLUMNS`: frame, kind (direction_change, speed_up, slow_down), magnitude (bh/s), u, v, nearest_track_id, nearest_team, nearest_dist_box_h, then shot_id, angle_deg, speed_before, speed_after, at_foot. Strike profile and contact profile (`CONTACT_PARAMS`); `ball_events` needs the keyword `shots` |
| `kinematics.parquet` | `gbf.kinematics` | frame int64, track_id int64, segment_id int64, x y float64 (smoothed), vx vy speed float64 (m/s: the chord of the smoothed positions centred on the frame over 0.4 s, 10 frames at 25 fps; NaN where the chord leaves the segment), valid bool (both chord ends measured and at least (window-1)/2 frames from the segment ends, no filter derivative above 12 m/s within (window-1)/2 frames of them, chord at most 12 m/s, segment at least 25 frames), interpolated bool |
| `segments.parquet` | `gbf.kinematics` | track_id, segment_id, shot_id, start_frame, end_frame, n_frames, n_interpolated, n_outliers (rejections between the segment's first and last frame only; `gbf.kinematics.rejected_positions` lists every rejection with its test), n_invalid int64, peak_speed float64 (highest valid speed, NaN when none), peak_frame int64 (-1 when none) |
| `audio.parquet` | `gbf.audio` | frame int64, rms_dbfs float64, flux float64, excitement_z float64, has_audio bool |
| `signals.parquet` | `gbf.highlights` | frame int64, shot_id int64, cue columns cut_emphasis audio speed box_pressure ball_goal duel strike (float64 in [0, 1]; audio = clip(excitement_z, 0, 4) / 4), score float64 (weighted mean, [0, 1]), evidence columns `ev_*` (`gbf.highlights.EVIDENCE_COLUMNS`) |
| `highlights.json` | `gbf.highlights` (+ merge in `gbf.moments`) | see below |
| `highlights_final.json` | `gbf.moments` | `highlights.json` after key moments and ranking, see below |
| `pose.parquet` | `gbf.pose` | frame int64, track_id int64, model_id string, x1 y1 x2 y2 float64, keypoints list<float64> (26 x [u, v, score], Halpe-26 order; u, v NaN where score <= 0; scores are rtmlib SimCC confidences, not probabilities), mean_score float64 |
| `lift3d.parquet` | `gbf.lift3d` | frame, track_id, method string (height_prior_calibrated, image_plane_only), joints3d list<float64> (26 x 3, pitch_zup metres, NaN when not lifted), stature_m (a fit scale, never shown), facing_deg, facing_conf, trunk_lean_deg (sagittal plane, + forward), image_lean_deg, reproj_err_px; extras reason (lifted, no_calibration, airborne, feet_low_confidence, torso_low_confidence, foot_ray_miss), facing_sigma_deg, trunk_tilt_deg, trunk_lean_alt_deg, trunk_lean_sigma_deg, image_knee_left_deg, image_knee_right_deg |
| `measurements.jsonl` | `gbf.measure` | one evidence record per line (canonical JSON), schema `urn:gbf:schema:measurement`: evidenceId (sha256 of the canonical JSON of the record without it), metric_id, label_zh, tier (registry tier, else explicit B or image_plane; never C), tier_source, claimType, status ok or unavailable (value null with reasonCode and reason_zh, never zero), value in storage units, unit (m/s, m, deg, count, frame, box_h, bh/s), display, frame, frames, highlight_id, track_id, track_ids, plane (pitch or image), quality, error {kind (unmeasured unless a receipt), value, ci95, dataset, conditions, receiptId}, coverage {window, frames_considered, frames_used, share}, method {id, summary, parameters, code}, restrictions, source {video_sha256, fps, inputs[{name, sha256}]} |
| `clips/<id>.mp4`, `clips/<id>.render.json` | `gbf.render` | H.264 yuv420p BT.709 CRF 18 with AAC 48 kHz stereo; the manifest (schema `gbf.render-manifest/1`) has the output sha256, the segment table from output to source frames, fonts by sha256, audio parts, every drawn text with its evidence ids or locators (`<table>.parquet#frame=<f>&track_id=<t>`; ground-distance lines cite `refine/tracks.parquet`), and the restrictions |
| `keyframes/<id>.png`, `keyframes/<id>-card.png`, `report.html`, `timeline.png`, `timeline-dark.png` | `gbf.render`, `gbf.report` | the key frame with the overlay, the same frame with the data card, the report |
| `label_eval.json` | `gbf.label_eval` | with `--labels`: the label file's moments matched to the highlights (peak error in frames, type agreement, window IoU), the comparison's sentences, and where the labels come from in the file's own words (`labels_source`, `source_zh`) |
| `cuts.json` | `gbf.pipeline` | cuts, cut-detection settings, frame_count, fps, restrictions |
| `clips/full.mp4`, `clips/full.render.json` | `gbf.render` via `gbf.pipeline` | the whole clip with the overlay and the original audio |
| `receipts/<stage>.json`, `run.json` | `gbf.pipeline` | `urn:gbf:schema:receipt` kind run: inputs and outputs by sha256, parameters, code, `source_sha256` (the fingerprint of the gbf source that every stage key holds), wall_s, facts, notes_zh; `run.json` lists each stage as ran, skipped or kept |

`highlights.json`: `{"schema_version": 1, "fps": "25/1", "frame_count": N, "highlights":
[{"id", "type", "start_frame", "peak_frame", "end_frame", "score", "cues": {...},
"track_ids": [...], "title_zh", "summary_zh", "score_peak_frame", "refined_by"}]}`. Types,
with their Chinese names (`gbf.highlights.TYPE_NAMES_ZH`): `duel_tackle` 贴身争抢,
`attack_in_box` 禁区进攻 (the ball in or at the penalty area, see "Ball at the box" below),
`ball_strike` 触球（球速突变） (an abrupt change of the ball's image velocity near a player: a
shot, a pass, a touch or a deflection alike), `sprint` 高速冲刺, `closeup_reaction` 球员反应
(what is left when no play cue is strong enough; its shot need not be a close-up).
`cues` maps each of the 7 cues to `{value, weight, contribution, frame, ...evidence}`;
`cues.cut_emphasis` also holds `extended_into_shot_id` and `joined_shot_event`, and the two
attack cues hold `ball_box_m` (how close the ball came to the box, `ev_ball_box_m` of their
frame). Numbers keep gbf.measure's four decimals and the text rounds them once, as the
measurement table does (a summary's speed reads the same as the table's for the same
frame). `refined_by` is one of `strike_event`, `duel_min_distance`, `speed_peak`,
`attack_peak`, `score_peak`, `pose_key_moment`. Times in `title_zh` and `summary_zh` are
`mm:ss.cc` (hundredths, floored, exact rationals) followed by the frame number; one formatter
(`gbf.util.clock.format_clock`) writes every time of the titles, summaries, key moments and
records, so a frame has one time everywhere at any rate. The frame the highlight is frozen on
(`peak_frame`) is its 关键帧 in the summaries, on the data card and in the report's tables.
Titles name players by team (甲队球员); summaries add the track id (甲队球员（轨迹 7）), so
two players of one team are told apart, and end with the players involved. A speed in a
summary is the speed on the frame it names (the 0.4 s chord around it), neither called a
peak nor said not to be one.
Shots are named by the size of their players (`shots.median_box_h`): 全景 for a wide shot,
中景 for a medium shot and for a close shot whose median person box is below
`SHOT_UPPER_BODY_MIN_BOX_H_PX` (players seen head to foot), 特写 for a close shot from that
height (upper bodies) or without any person box.
Highlights whose windows meet at a cut (the second starts on a cut frame at most
`MERGE_GAP_FRAMES` after the first ends, and the merged window lasts at most `MERGE_MAX_S`)
are merged (`merged_from`, `merged_parts`): shots that meet at a cut (切镜前后相接的镜头).
Nothing beyond that adjacency is checked, so the parts need not show one play. The merge
renumbers the highlights, so a part keeps the id it had before with the prefix `detect-`
(`detect-h05`), which never names another highlight.

`highlights_final.json` adds, per highlight: `key_moments` (frame, time, kind, track_ids,
text_zh and evidence, in frame order; kinds contact_start, touch, ball_release, ball_away,
fall_onset, on_ground and contact_end for a duel pair, hand_up, arm_top and hand_top for
close-up reactions), `touches`, `key_pair`, `key_focus_track`, `rank` and `tier` (`main` or
`candidate`), and `previous_peak` when the peak moved:

- `key_pair`: `[a, b]`, the duel pair whose moments are in `key_moments`, or null. Every
  pair candidate (the joined shot's duel pair, its subject with the partner found for it,
  the duel cue's pair) with a pose of both players on at least 5 frames gets a timeline; the
  first is kept unless a later one has more moments of `PAIR_RANK_KINDS` (ball_release,
  fall_onset, ball_away; touches do not count). A subject gets a partner only when its team
  is known (most of its rows carry one label) and the partner's rows carry the other one:
  two players are never paired without team labels that differ.
- The key-moment texts name players with their track id, as the summaries do; the players
  of `key_pair` and of the key moments join `track_ids` when missing, and the summary's list
  of the players involved (涉及球员) is written again from `track_ids`, since gbf.highlights
  wrote it before the merge, the partner and the key moments.
- The ball leaves the pair's feet fast when, after the last frame it is within
  `RELEASE_NEAR_BODY_H` body heights of an ankle or a big toe, its first detection more than
  `RELEASE_AWAY_BODY_H` away comes within `RELEASE_WITHIN_S` and its distance from the feet
  grew by at least `RELEASE_MIN_SPEED_BH_S` body heights per second on average between the
  two frames. That last frame is a `ball_release` when the two players were close together
  (a contact phase, below), else a `ball_away` of the player whose feet the ball left (a
  pass or a clearance, say).
- A contact phase is image-plane closeness of the two players' keypoints, not a measured
  touch (the text says 身体贴近（画面平面）). Contact phases and falls are read only where the pair's
  median body height (head to ankle) is at least `MIN_BODY_PX` (150 px); below it only
  touches and `ball_away` are listed.
- A fall is read from the trunk angle in the image: the line from the hip midpoint to the
  shoulder midpoint against the image vertical (`gbf.lift3d.image_trunk_lean_deg`), the same
  angle the report shows as 躯干倾斜（画面）, so the two never disagree about a frame (the
  report's angle used to run from the hip centre to the neck keypoint instead). `fall_onset`
  is its last upright frame (at most `UPRIGHT_DEG`), `on_ground`
  its first frame at `FALL_DOWN_DEG` or more; their evidence names the line
  (`trunk_line`), and every title that names the 45° says 画面平面.
- When the pair had a contact phase or one of them fell, and a ball release (else a fall
  onset) is found, that moment becomes `peak_frame` (`refined_by` `pose_key_moment`) and the
  type becomes `duel_tackle`. Otherwise the pair's moments are listed and change nothing.
- `touches`: per player of `key_pair`, the ball events that gbf.measure's criterion matches
  to a confident foot keypoint inside the highlight's final window: track_id, count, frames,
  mean_interval_s, label_zh. Given the same inputs, a count equals the number of that
  player's `foot_keypoint` `ball-contact-frame` records in the highlight, and the `touch`
  moments are the same touches. A soft touch that hardly changes the ball's velocity is
  missed, so a count is a lower bound.
- `key_focus_track`: the player the key frame is about: the subject of a close-up reaction
  whose peak moved to a gesture, the player of the moment that turned a highlight into a
  duel (the faller of `fall_onset`, the player whose feet the ball left at `ball_release`),
  or the first of `key_pair` in any other `duel_tackle`. It is null for every other
  highlight, and also when that player has no track row on the final `peak_frame` (track
  ids are split at cuts, so a joined shot's players may not exist in the key frame's shot).
- `previous_peak`: the peak frame, type, `refined_by`, title and window before the move.

In a `closeup_reaction` whose close-up shows one subject, the subject's raised hands are the
key moments: a wrist at least 0.25 shoulder widths above its shoulder for at least 0.3 s
(dips under 0.2 s do not split it), with a straight arm when the elbow is at least 140° for
0.3 s in all. Each gesture starts with a `hand_up`. The main gesture is the straight-arm one
raised highest, else the longest bent-arm one, and the frame its hand is highest becomes
`peak_frame`: an `arm_top` when the main gesture is a straight-arm one whose elbow is still
straight on that frame, else a `hand_top`. The window then starts at most `REACTION_LEAD_S`
before the cut into the close-up, and merged parts wholly before it are dropped. Gestures
are read only when the shoulders are at least 35 px apart. This rule was added after the
development clip's hand label for the close-up was compared with the output, so that
label no longer checks it independently.

## Highlight cues (per frame)

- Broadcast grammar: a cut from a wide shot to a close or medium shot marks the end of an
  action the director chose to emphasise; frames just before it and the shot after it get a
  bonus.
- Audio: loudness and spectral flux of the original track, z-scored over the clip.
- Play: peak player speed (valid chord speeds on pitch frames), attackers inside the penalty
  area, ball distance to goal, the distance between the nearest pair of opposing players
  near the ball, ball events (an abrupt change of the ball's image velocity near a player's
  feet: a shot, a pass, a touch or a deflection alike).
- Ball in sight: the two attack cues (attackers inside the penalty area, ball distance to
  goal) count only near a detected ball, a `ball_track` row with status `detected` within
  `ATTACK_BALL_SEEN_S` of the frame in the same shot. During a stoppage (a corner or a free
  kick being prepared, an injury) players stand in the box while the ball is out of the
  picture for seconds; that is not an attack.
- Ball at the box: the two attack cues count for the type `attack_in_box` only where the
  ball's ground position (a `ball_track` row detected or interpolated, on a pitch frame) lies
  in the penalty area the attacking side attacks, or within `BALL_AT_BOX_MARGIN_M` of it, at
  some frame within `ATTACK_BALL_SEEN_S` in the same shot (`ev_ball_box_m`). Runners in the
  box while the ball is still 20 m out are no attack in the box. Box pressure is meant to
  see an attack in the box near the ball, so the score counts it under the same condition
  (elsewhere the cue is 0; `ev_box_attackers` still counts the players in the box). Ball
  to goal counts in the score as it is: it measures the ball's own distance, and a ball in
  front of the box within 25 m of the goal is what it is meant to see; only the type asks
  for the ball at the box. The margin stands for the calibration error at the box, not for
  attacks from outside it (see the table of gbf.highlights below).
- Attack direction: when the refine stage's side check (`gbf.side_check`) does not bear
  out the plugin's left/right names, the goal each team attacks is unknown. The two attack
  cues are then 0 on every frame and their evidence is missing (`ev_box_attackers` 0,
  `ev_box_team`, `ev_ball_goal_m`, `ev_ball_box_m`), so no `attack_in_box` highlight is
  found; titles, summaries and cards call the teams 甲队 and 乙队 (configured names are
  given per side, so they are not used either); gbf.measure withholds attackers-in-box and
  body-orientation as `attack_direction_unknown`. Measures that only need to tell the
  teams apart (the nearest opponent, duels) do not change.
- Type: a play group competes when its strongest smoothed cue at the score peak reaches
  `PLAY_EVIDENCE_MIN`; the largest weighted contribution wins, and without a competing
  group the type is `closeup_reaction`. Box pressure leaves the attack group before the
  contest unless two attackers (`BOX_PRESSURE_ALONE_MIN`) were in the box near the peak, so
  a lone attacker never hides a duel or a run. An `attack_in_box` key frame is a frame where
  its own condition holds: a ball event elsewhere does not move it there, the strongest
  attack frame is used instead.
- Titles: a `closeup_reaction` outside a close or medium shot names sound or editing only
  when that cue's contribution reaches `BROADCAST_CUE_MIN`; otherwise it claims no cue
  (综合评分入选的瞬间).
Peaks are picked with a minimum separation, windows are snapped to shot boundaries, and every
highlight lists the cues that made it, so the choice can be audited.

## Analysis on the original frames

Inside each highlight, from the original frames: the involved players (ball carrier, nearest
opponents, fastest player), their 2D pose (RTMPose Halpe-26 through rtmlib; weights'
training-data licence not verified, so experimental and private only), the ball, ball contact
frames (ball velocity change near a foot keypoint, frame-accurate at 25 fps but not
sub-frame), key moments, and per frame, on calibrated frames: pitch positions (a box's foot
point through the frame's calibration), speed and the distance to the nearest opponent; on
frames without usable calibration (close-ups, low angles): image-plane angles and the
image-plane distance to the nearest opponent only.

From the 3D lift (gbf.lift3d's calibrated height-prior skeleton, on calibrated frames): the
facing and the trunk lean, and nothing else (the image-plane angles that `lift3d.parquet`
also holds are measured on the 2D keypoints). They are a reference, never the basis of a
conclusion: their values (body-orientation and trunk-lean-at-contact, the A_candidate tier)
carry the badge 3D 参考 after their status badges on the data card and in the report
(`gbf.report.metric_row`, which badges every A_candidate value and `REFERENCE_3D_METRICS`),
the report says in its overview (数据来源) and in its notes which numbers come from the
original frames and which from the 3D lift (3D 只作参考，所有结论以原始画面为准), and the
facing is drawn back on the original frames (below).

Every displayed number carries its tier and the "unverified" badge while its error is
unmeasured. On the freeze-frame data card, a value of another player than the one in focus,
or of another frame than the key frame, names that player and that frame before its label
(`轨迹42·第2391帧·最近对手距离（±2帧中位数）`): the card cuts a label that does not fit at its end, so they
stay visible. Key moments come first on the card, each as its players and the short name of
what was measured (`轨迹5、9·身体贴近`, from `gbf.report.MOMENT_KINDS_ZH`), badged 未验证 and 仅画面平面;
the report gives each one as a full sentence. The overlay's line says what it links: 争抢
between a duel pair (with the distance on pitch frames, 争抢 · 1.6 m) only in duel_tackle
highlights, within `gbf.render.DUEL_LINK_WINDOW_S` of a moment of the pair and, with a ball
track, only while the duel cue's rule holds for the pair on the frame (the ball within
`DUEL_BALL_RADIUS_M` of both players and the pair closer than `DUEL_FAR_M`, or the
image-plane counterparts); otherwise 最近对手 · 4.2 m to the focus player's nearest outfield
opponent, the player of the card's 最近对手距离 row (the line shows one frame's distance, the
card the median over the key frame +/-2). The overlay chooses that opponent as gbf.measure
does, among the players observed on the frame, and never takes a box on the focus player's
own spot (`gbf.track_tools.same_spot`: boxes overlapping more than `SAME_SPOT_IOU`, or foot
points closer than `MIN_PLAYER_GAP_M` on the grass, `SAME_SPOT_FOOT_BOX_H` mean box heights
in the image without ground positions: the player detected twice, or one box around the
player and the one behind that ends at the player's feet) or a person off the pitch
(`gbf.track_tools.off_pitch`: more than `OFF_PITCH_MARGIN_M` beyond a touchline or a goal
line, a coach or a steward). No distance below `MIN_PLAYER_GAP_M` is shown, nor one to a
partner drawn at a filled-in position (`gbf.render.TRACK_HOLD_MAX_GAP`): the line then
names only what it links. Over a frame on which the focus player is filled in, the line
holds the distance last measured to the opponent that was nearest then, visibly marked
（沿用） and citing that earlier frame. A drawn filled-in
opponent nearer than every observed opponent makes the nearest opponent unknown: its
position only withholds the line, never chooses a partner or measures a distance. No label or other
text is drawn where the card, the minimap, the banner or the watermark would cover it, and
the render manifest lists only what is drawn.

Who the overlay marks: a box without a track (id -1, not one person) is never filled in
across a gap and takes no role (not the focus, a duel pair's player, an opponent or the
player without a team label that leaves the nearest opponent unknown); it gets a context
ring at most. On pitch frames, a person off the pitch gets no team ring and no minimap dot
and is never the focus, linked or an opponent: a foot point more than `OFF_PITCH_MARGIN_M`
beyond a line, or a box the bottom of the frame cuts off whose ground point (the frame
edge's under the person) is beyond a line at all that the camera stands beyond too, since
the feet below the frame are nearer the camera and so farther out (a coach in the technical
area under the camera). gbf.measure applies the same camera and frame-edge rule. Both
stages read the team-check verdicts from `refine/refine.json`: an unchecked track without a
ground position, or whose box touches a frame edge, takes no role, team colour, label,
minimap dot or measurement. A checked close-up player may still use image-plane metrics.
In a merged highlight (`merged_parts`), a
shot after the key frame's that shows none of the key frame's players has no focus
spotlight; in a merged or extended highlight, on the shots before the key frame's, the
focus is marked only while within `gbf.render.FOCUS_BALL_MAX_M` of the ball on the grass
(the minimap's ball). A key frame at a ball moment (a touch, the ball leaving a player's
feet) puts the spotlight on that moment's player throughout the key slow-motion window,
clipped to its shot, unless the highlight has a `key_focus_track`. A player's speed label holds its last valid speed, greyed, over at most
`TRACK_HOLD_MAX_GAP` frames without one (a filled-in box shows none). For a player whose
peak-speed record is withheld, it instead names the observed peak lower bound
(峰值≥25.2 km/h), citing the highest valid observed speed in the highlight; an individual
frame speed is not called a lower bound. The unavailable card record and 未验证 marker stay.
Labels of players and lines
keep off the boxes of the focus player, the duel pair and the involved players wherever a
free place is found (above or below where they belong, then beside the player, then farther
out, the nearest first), the key frame and the freeze included, and never lie on another
label where a place off the labels exists; a label placed a step (its own height) or more
from where it belongs gets a thin leader line to its player or line.

The 3D lift reaches the overlay only as the focus player's facing, drawn back on the original
frame (the render stage reads `lift3d.parquet`): on a pitch frame, a dashed arrow on the
grass from the player's foot point, `gbf.render.FACING_ARROW_LENGTH_M` (2 m) along the
facing (pitch_zup degrees from +x, counter-clockwise, turned into SoccerNet metres), both
ends projected with the frame's camera like the ground rings, labelled 朝向（3D 参考） with
未验证 (evidence `lift3d.parquet#frame=<f>&track_id=<t>`). It is drawn only where that
frame's lifted row passes the per-frame gates of body-orientation, one function for the
card's value and the arrow (`gbf.measure.facing_frame_ok` with `gbf.measure.FacingGates`,
which the pipeline fills from `MeasureParams` and the registry: lifted, a finite facing,
facing_conf >= `facing_min_conf`, a finite facing_sigma_deg below `facing_max_sigma_deg`, a
box at least `min_bbox_px` tall, and no front-back reversal), the focus player has a row of
its own on the frame (not filled in) and the pose the facing was lifted from. The reversal
check (`gbf.measure.facing_reversed`): players lunge, fall and sprint the way they face,
and the height-prior lift, which follows the 2D pose's left/right labels, can reverse front
and back on exactly such frames with no more doubt in its facing_conf or facing_sigma_deg
than on a correct one. A frame is left out when its 2D trunk (hips to shoulders,
`gbf.lift3d.image_trunk_lean_deg`) leans more than `FACING_LEAN_DEG` from upright to one
side of the picture while the facing, placed in the picture at the player's foot point,
points to the other side by at least `FACING_SIDE_SHARE` of its direction there, or cannot
be placed. A trunk leaning less, or towards the facing, and a facing into or out of the
picture pass. The arrow places the facing with the frame's camera; the card with the
frame's homography from its players' ground positions to their foot points (the frame's
calibration links them, so one fits them to a pixel; without such a fit a leaning frame is
left out of the card's mean). The agreement of the facings around a reference frame
(`facing_min_resultant`) is a gate of the card's mean, not of one frame: the arrow shows
each frame's own facing, so it comes and goes with the gates, and turns from one frame to
the next where the lifted facing does. Its label sits just past the head (below the start
when the arrow points up the image, into the player), off the players' boxes where it can
be and never over another label; an arrow whose label cannot be drawn is left out with it,
and the frame lists the arrow's 未验证 badge only when the arrow is drawn.

Measurements (`gbf.measure.METRICS`). From the metric registry: peak-speed (A, m/s: the
highest valid kinematics speed in the window, so the overlay's speed labels and the sprint
cue, which show the same chord speeds, never exceed it; labelled with the chord's actual
duration, 峰值速度（0.4秒平均） at 25 fps; withheld as `speed_coverage_split`, `_rejected`, `_short`
or `_implausible`, after the commonest cause, when valid speeds cover less than half of the
frames the player was tracked on the pitch, the 10 frames at the start and end of the
player's track in a shot that no chord can cover left out; withheld as `speed_peak_hidden`
when the highest valid speed lies within half a chord of a stretch without valid speeds, or
the player crosses a gap in the track more than 5 % faster than it; the highest valid speed
stays in quality as a lower bound), nearest-opponent-distance (A, m: the median, over the
player's reference frame +/-2 in its shot (the key frame, or the joined shot's moment for a
player not on screen near the key frame), of the distance to the nearest outfield player of
the other team, never a box on the player's own spot (`gbf.track_tools.same_spot`) or a
person off the pitch (`gbf.track_tools.off_pitch`); goalkeeper reported separately; withheld
as `unknown_team_nearer` when on any of those frames an outfield player without a team label
(gbf.team_check clears labels that contradict the shirt colours) is nearer than every
labelled outfield opponent), body-orientation (A_candidate, deg: angle between the mean
lifted facing of the frames of the reference frame +/-2 that pass the per-frame gates
above, the reversal check included (quality counts the frames it leaves out,
`rows_reversed`), and the direction to the centre of the goal the player's team attacks;
labelled 朝向夹角（±2帧均值）), trunk-lean-at-contact (A_candidate, deg).
Signed sagittal trunk lean uses the same explicit front-back reversal check: reversing
the lifted forward direction reverses its sign. It retains its own box-height and lean
sigma gates, without requiring the facing confidence or facing sigma gates. Its quality
counts `rows_reversed`, and method parameters record `facing_lean_deg` and
`facing_side_share`. If reversal excludes every otherwise valid lean row, the value is
unavailable with `facing_uncertain`, rather than a median with an unreliable sign.
With explicit tiers: attackers-in-box (B, count, on the frame of the highlight's
box-pressure cue when its window has one, the frame its summary names, else on the key
frame), ball-contact-frame (B, frame, an estimate), ball-image-speed (image_plane, bh/s,
never km/h), nearest-opponent-image-distance (image_plane, box_h: between foot points over
the same frames, in mean box heights, never to a box on the player's own spot, withheld in
the same way; none in shots whose median person box exceeds
`image_feet_max_median_box_px`), image-trunk-lean, image-knee-angle-left
and image-knee-angle-right (image_plane, deg). The image-plane body angles and the image
distance to the nearest opponent are given only for a player whose reference frame has no
usable calibration; the ball's contact frames and its image speed after a contact are image
measurements, given on every frame (there is no ground model for a ball in flight).
A withheld peak-speed record's reason gives its own numbers: how many of the frames counted
carry a trusted speed and why the others do not (可信速度只覆盖 8/30 帧，其余帧多在太短的
连续轨迹段上), and the highest trusted speed with its frame as a lower bound, never "no
speed could be measured". attackers-in-box records where its team comes from
(`team_source`: the highlight's attack cue, else the first involved player's label
(`first_player`), else the side with more players in the box it attacks
(`largest_count`)). When outfield players are visible on its frame but none has a team
label (gbf.role_check clears every label of a run whose role labels are implausible), it is
withheld as `team_unknown`, with `outfield_players_without_team` in its quality, instead of
counting 0. When an outfield player without a team label is inside the box it counts for,
that player may be an attacker and the count only a lower bound: it is withheld as
`unknown_team_in_box`, with the team, `attackers_with_team` and
`players_without_team_in_box` in its quality, as nearest-opponent-distance is withheld as
`unknown_team_nearer`.
attackers-in-box and body-orientation need the goal a team attacks: when the side check
of the refine stage leaves it unknown (`MeasureContext.attack_direction_known` False),
both are withheld as `attack_direction_unknown` (进攻方向不确定：插件给两队定的左右未能
核实). Only `team_unknown` and the reasons a metric has before the direction matters (no
pose, an uncertain facing) come first.

Clips and key frames. The banner (title, time pill, subtitle) ends at least 16 px (1080p)
left of the minimap in its half of the frame. A title too long for its first line, beside
the time pill, goes on to a second line, after a clause where it can; a title too long for
two lines is set smaller (28 instead of 34 px), and only one too long for that loses its
last whole clauses, marked … (a first clause longer than two lines is cut inside;
`gbf.render.Banner`). The report keeps every title whole, and the render manifest lists the
banner as drawn (two lines joined by a line break). The data card's footer counts the rows it would show without its caps (另有 N 项见报告):
key-moment and touch-count rows beyond `CARD_MOMENT_ROWS`, and the key shot's metrics,
one per metric, beyond `CARD_MAX_ROWS`, plus the rows the frame has no room for. The
report lists every non-C measurement record; C-tier records are counted. The clips play the audio source as the audio cue reads it:
`--audio-offset-s` is the audio time of frame 0 counted from the first audio sample,
`gbf.render.first_sample_offset` turns it into the render's offset on ffmpeg's timeline (the
two differ when a file's audio and video start at different times), and the render manifests
record that offset. Container start offsets are not applied automatically: pass
`--audio-offset-s` explicitly when they should affect alignment. Every clip, key frame and timeline chart carries the run's restrictions in its own metadata as
`gbf restrictions: <tags>` (the mp4's comment, the PNG's tEXt chunk Comment); with no_share
or nda the frames carry the watermark 私有素材 · 禁止分享 at the top centre and a fainter copy at
the bottom centre, both in the margins outside the title-safe area, so a frame cut off at
the top still carries one. Exported light and dark timeline charts carry a top-right
watermark badge under the same restriction rule.

## Limitations

- Routine play without an outcome can be selected: a press that wins nothing, with no foul,
  fall or stoppage, still scores as a close duel near the ball. Nothing filters such moments
  out or marks them: the tier ranks by score only.
- A duel cue ends when the falling player's track is lost: a player going down in a duel
  often loses the detector's box, and from then on no pair distance is measured, so a duel
  that ends with a fall can be cut short or missed.
- A ball in the air is placed where its ray meets the ball-centre plane, farther from the
  camera than it is, so a cross or a lob can seem in the box or out of it;
  `BALL_AT_BOX_MARGIN_M` covers the calibration error only.
- Box pressure counts only with the ball's ground position at the box: when the ball is
  not placed (no detection with a ground position within `ATTACK_BALL_SEEN_S`) or a ball
  in the air is placed beyond the margin, attackers in the box add nothing to the score,
  so such an attack can rank lower or be missed.

## Running it

`python -m gbf highlights --video V --audio-source A --sngs DIR --ball JSONL --out DIR
[--team-names left=NAME,right=NAME] [--labels LABELS.json] [--no-calib-refine]
[--max-highlights N] [--min-separation-s S] [--pose-model x|m|none] [--models-dir D]
[--stages a,b,...|all] [--force] [--restrictions no_share,...] [--no-full-clip]
[--audio-offset-s S]`. `--out` must lie outside every git work tree. Stages: import, shots,
refine, kinematics, ball, audio, highlights, pose, lift3d, moments, measure, render, report;
a stage is skipped when its inputs (by sha256), parameters and code are unchanged. Code here
is every .py file of the gbf package the run executes (gbf.gate and gbf.web left out),
hashed into every stage's key, so any code change re-runs every stage, the expensive pose
stage included; measure also hashes `registries/metrics.toml`, where its tiers come from.
`--force` re-runs the selected stages anyway. Exit code 0 on success, 2 when the input is
refused (every problem listed, nothing written: among them a plugin export without
`plugin-receipt.json`, or with `keypoints.jsonl` but without its receipt when the refit
would use it, or with a receipt that is not readable JSON or gives no valid restrictions
list), 1 when a stage fails (`run.json` written with status failed). `python -m gbf web`
starts these runs from a local page and shows them ([web.md](web.md)).

With `--labels`, the report compares the highlights with the label file and says where the
labels come from only in the file's own words: it quotes the file's `labelled_by`, `method`,
`labelled_on`, `blinding` and `status` as written (`gbf.label_eval.source_zh`), says when
there are none, and calls them 标注, never 人工标注.

## Tuned parameters

M1's thresholds, weights and priors were chosen while it was developed on one clip: source
`private-dev-clip` ([`registries/data.toml`](../../registries/data.toml)), restriction
`no_share`. Following the amendment of 2026-10-21 to
[ADR 0002](../adr/0002-data-and-media-policy.md), the tables below list them for every
module of the M1 pipeline, with the value in the code, the kind and the provenance. Kinds:

- **fitted**: measured on the clip's data, or chosen by optimisation on it, as the code
  records;
- **hand-set**: a starting value set by hand (from football facts, detector properties or a
  look at the clip's numbers) and checked by eye on the clip's output;
- **football fact**: a dimension of the pitch or the ball, not chosen on the clip.

Provenance gives the reason the code states, if any; "not recorded" means that nothing in
the repository says how the value was chosen. Numbers measured on the clip appear only as
aggregates (medians, ratios, ranges, counts, thresholds); no frames, labels, identities or
descriptions of the clip. None of these values is validated: the fitted and hand-set ones
are re-fitted or confirmed on SoccerNet-GSR (`soccernet-gsr-2025`) before P5, and none may
be used in a commercial build while this clip is its only source.

Format, checked against the code by `tests/test_docs_params.py`: one table per module, under
a heading that starts with the module's name, with the columns Parameter, Value, Kind and
Provenance. Parameter holds names in backticks separated by commas: `NAME` (module
constant), `Class.field` (dataclass field default), `OBJECT.field` (attribute of a module
object), `NAME[key]` (entry of a module mapping) or `function(parameter)` (default of a
parameter); a name that starts with `gbf.` names its own module. A threshold written as a
literal inside a function has a row "literal in `function`: `code`", and the function must
contain that code. Value holds Python literals, one per name separated by `;`, or one that
every name of the row shares. Kind is one of fitted, hand-set and football fact. The test
also fails when a numeric scalar constant of an M1 module, a numeric field of a listed
dataclass, a field a listed object changes or an entry of a listed mapping is neither listed
here nor named, with its reason, in the test's `NOT_TUNED`.

Not listed, each named with its reason in `NOT_TUNED`: conventions and formats (the
1080-line reference height, record formats, keypoint counts, the id of a box without a
track, the id stride of tracks split at cuts, the codes of the kinematics rejection tests),
copies of listed values, numerical guards (`conf_floor`, focal-length bounds, the floor of
digital silence), gbf.label_eval's evaluation tolerance, the audio offset set per run,
limits of input checks, error reports and memory blocks, and output settings (encoder and
audio format, the ground ellipse under a player, the facing arrow's length, the data card's
layout, the pose stage's decode ranges). Not listed one by one either: the rule sets of
gbf.highlights
(`PLAY_GROUPS`, the order in which cue groups name a type; `EMPHASIS_TO_TYPES`, close and
medium shots joined by a cut; `PITCH_SHOT_TYPES`, wide shots), other literals, function
defaults and drawing defaults, for example the slow motion and freeze of the edit
(`gbf.render.plan_highlight_edit`), the colours and sizes of `gbf.render.Style`, the minimum
of 3 values of the kinematics box and median tests, and the bounds, weights and noise floors
inside gbf.lift3d's body fit and facing vote (`_fit_body`, `estimate_facing`). gbf.enhance,
gbf.pose and gbf.report have no tuned parameters of their own (the pose model is chosen with
`--pose-model`, and gbf.measure's `contact_candidates` and `contact_candidate_max_bh` choose
the boxes that get a pose), gbf.label_eval has only its evaluation tolerance, and
gbf.pipeline's own settings are listed with the module they are passed to.

### gbf.sngs_import (import of the plugin export)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `DEFAULT_REP_ERR_MAX_PX`, `gbf.pipeline.PipelineConfig.rep_err_max_px` | 8.0 | hand-set | an empirical starting value (docstring); the clip's fresh frames have median 3.3 px, P90 4.6 px (see Inputs); the refine stage passes it on to gbf.calib_refine |
| `IN_PITCH_MARGIN_M` | 3.0 | hand-set | a ground point farther outside the lines is a projection error or not a player (gbf.kinematics docstring); value not recorded |
| `PITCH_HALF_LENGTH_M`, `PITCH_HALF_WIDTH_M` | 52.5; 34.0 | football fact | SoccerNet's 105 x 68 m pitch |

### gbf.ingest.cuts (hard cuts of the shots stage)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `DEFAULT_CUT_THRESHOLD` | 0.5 | hand-set | docstring: an empirical starting value, to be validated in P1b; a cut between two shots of the same pitch can score only just above it |
| `BINS_PER_CHANNEL`, `SAMPLE_ROWS` | 8; 72 | hand-set | histogram settings; not recorded |

### gbf.shots_grammar (shot types)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `CLOSE_MIN_BOX_H_PX`, `CLOSE_UNCALIBRATED_MIN_BOX_H_PX`, `CLOSE_MAX_FRESH_SHARE` | 220.0; 150.0; 0.2 | hand-set | docstring: empirical starting values; not recorded |
| `WIDE_MAX_BOX_H_PX`, `WIDE_MIN_FRESH_SHARE` | 130.0; 0.5 | hand-set | docstring: empirical starting values; not recorded |

### gbf.track_tools (jersey numbers kept by the refine stage, through gbf.enhance; a box on a player's own spot and a person off the pitch, for gbf.measure and gbf.render)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `confident_jerseys(min_mean_conf)`, `confident_jerseys(min_reads)`, `confident_jerseys(min_agreement)` | 0.8; 5; 0.7 | hand-set | not recorded |
| `SAME_SPOT_IOU` | 0.75 | hand-set | docstring: two players overlap that much only when one stands right behind the other, and then the detector sees one of them; value not recorded |
| `MIN_PLAYER_GAP_M` | 0.3 | hand-set | docstring: no two players' foot points are closer on the grass, so a box whose foot point is nearer is the player again or a box around the player and the one behind; no distance below it is shown; value not recorded |
| `SAME_SPOT_FOOT_BOX_H` | 0.05 | hand-set | the image-plane test without ground positions, 9 cm at the players' own scale (docstring); value not recorded |
| `OFF_PITCH_MARGIN_M` | 0.5 | hand-set | docstring: the Laws of the Game end the technical area 1 m from the touchline, so a coach stands at least twice this far out, while a player taking a throw-in stands on or just behind the line |
| `PITCH_HALF_LENGTH_M`, `PITCH_HALF_WIDTH_M` | 52.5; 34.0 | football fact | SoccerNet's 105 x 68 m pitch |

### gbf.calib_refine (`RefineSettings`; the refine stage passes the run's `rep_err_max_px`)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `RefineSettings.min_score`, `RefineSettings.min_points`, `RefineSettings.max_gap_frames` | 0.4; 4; 12 | hand-set | not recorded |
| `RefineSettings.sigma_px` | 2.5 | hand-set | docstring: heatmap detectors quantise keypoints to their output stride; value not recorded |
| `RefineSettings.sigma_rot_acc`, `RefineSettings.sigma_logf_acc` | 0.0005; 0.001 | hand-set | not recorded: an earlier version of this page called these smoothing priors fitted by held-out keypoint error, but no receipt records such a search (`refine.json` gives the held-out error of each run) |
| `RefineSettings.same_camera_m`, `RefineSettings.min_group_frames` | 8.0; 10 | hand-set | not recorded |
| `RefineSettings.rep_err_max_px` | 8.0 | hand-set | gbf.sngs_import's `DEFAULT_REP_ERR_MAX_PX` |
| `RefineSettings.holdout_fraction`, `RefineSettings.seed` | 0.2; 0 | hand-set | share of every frame's keypoints held out for the evaluation fit, and its seed; the held-out P90 decides which camera groups `holdout_p90_max_px` rejects, so they change which frames are pitch-usable; not recorded |
| `RefineSettings.holdout_p90_max_px` | 25.0 | hand-set | docstring: the median can look fine while a tenth of the pitch is tens of pixels off; value not recorded |
| `RefineSettings.holdout_fallback_ratio` | 2.0 | hand-set | added 2026-10-27 after the gates.v1 dev evaluation, from held-out P90s only (`refine.json`, no annotation): the refit-to-per-frame ratio was 0.62–0.87 on the seven camera groups that refitted well (this clip's main camera, SNGS-021 and five dev clips), 0.97 on this clip's one group where both fits fail (it stays rejected) and about 250 on the dev clip SNGS-058, whose refit diverged; 2 lies well between |
| `RefineSettings.max_nfev` | 2000 | hand-set | raised 2026-10-27 from 400, a cap set from solver convergence counts alone (no accuracy figure was looked at): on the six dev clips the main-camera refits needed 112–848 evaluations, with or without the goal-post-top export fix; SNGS-023 had stopped unconverged at 400, and where it stopped depended on the BLAS thread count; SNGS-058's refit does not converge at any cap tried (400–4000) and falls back; a group that reaches the cap is reported as not converged (`converged` in `refine.json`, a sentence in the report note), and since 2026-10-31 it keeps the plugin calibration as a group that fails the held-out check does: where an unconverged solver stops is not the fit the method defines (with the plugin's keypoints decoded at the centre of their heatmap cells, SNGS-058's refit stopped at the cap with a held-out P90 of 12.7 px, below the per-frame fits' 14.0 px, but a per-frame median error of 13 px, which left 90 of the plugin's 659 usable frames usable) |
| `reproject_tracks(in_pitch_margin_m)` | 3.0 | hand-set | gbf.sngs_import's `IN_PITCH_MARGIN_M` |
| `_PITCH_HALF_LENGTH_M`, `_PITCH_HALF_WIDTH_M` | 52.5; 34.0 | football fact | SoccerNet's 105 x 68 m pitch |

### gbf.team_check (`TeamCheckParams`; the refine stage uses `DEFAULT_PARAMS`)

The class docstring calls every threshold an empirical starting value, not a validated one.

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `TeamCheckParams.samples_per_track`, `TeamCheckParams.max_frames` | 8; 200 | hand-set | not recorded |
| `TeamCheckParams.min_box_height_px`, `TeamCheckParams.min_aspect`, `TeamCheckParams.edge_px` | 50.0; 1.5; 2.0 | hand-set | docstring: smaller, squat or cut-off boxes put the torso crop on blur, the head or the background; values not recorded |
| `TeamCheckParams.torso_x`, `TeamCheckParams.torso_y` | (0.3, 0.7); (0.2, 0.45) | hand-set | docstring: the middle of the body, from below the shoulders to above the shorts; values not recorded |
| `TeamCheckParams.max_overlap`, `TeamCheckParams.min_shirt_share` | 0.1; 0.3 | hand-set | not recorded |
| `TeamCheckParams.grass_margin`, `TeamCheckParams.dark_l`, `TeamCheckParams.white_l`, `TeamCheckParams.white_chroma` | 10.0; 10.0; 85.0; 12.0 | hand-set | pixel tests for grass, deep shadow and white print; not recorded |
| `TeamCheckParams.min_samples`, `TeamCheckParams.min_team_tracks` | 3; 3 | hand-set | not recorded |
| `TeamCheckParams.margin` | 10.0 | hand-set | CIE76 delta E; not recorded |
| `MIN_CROP_PIXELS`, `MAX_CROP_PIXELS` | 24; 4096 | hand-set | not recorded |

### gbf.role_check (`RoleCheckParams`; the refine stage uses `DEFAULT_PARAMS`)

Added 2026-10-28 after the gates.v1 dev evaluation. The check rests on the laws: one referee on
the field of play, the assistant referees along the touchlines, the fourth official off the
pitch. The figures below are the aggregates looked at when the values were set, from the
plugin's labels and pitch positions on the six dev clips.

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `RoleCheckParams.inner_margin_m` | 2.0 | hand-set | the assistant referees run along the touchlines, and the calibration can put a point a metre off; in the dev annotations no two referees were 2 m or more inside the pitch at once, while counting up to the lines the assistants made two in up to a fifth of the frames |
| `RoleCheckParams.max_crowded_share` | 0.5 | hand-set | most frames; with the plugin's labels two or more referees stood 2 m or more inside on 0-10 % of the frames of four dev clips and on 85-98 % of those of the two clips of one match, where it labelled a team in a referee-like kit referee; every share from 0.1 to 0.85 gives the same verdicts |
| `RoleCheckParams.min_frames` | 25 | hand-set | one second at 25 fps; not tuned |

### gbf.side_check (`SideCheckParams`; the refine stage uses `DEFAULT_PARAMS`)

Added 2026-10-31 after the dev clips of the gates.v1 evaluation showed the plugin's names following chance when the two teams' mean x lay a metre or two apart (one clip named the wrong way round). The values are conventions, set before the check was run on any clip and not tuned.

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `SideCheckParams.n_boot` | 2000 | hand-set | convention: resamples, as in the frozen evaluation's bootstrap |
| `SideCheckParams.seed` | 0 | hand-set | convention: as in the frozen evaluation's bootstrap |
| `SideCheckParams.level` | 0.95 | hand-set | convention: the usual central interval |
| `SideCheckParams.min_tracks` | 2 | hand-set | convention: one track a team cannot be resampled |

### gbf.kinematics (box-size, trend and jump tests, speed chord)

Rows name the settings `gbf.pipeline.KINEMATICS_PARAMS` passes to
`kinematics_with_rejections` (the arguments of `compute_kinematics` and `rejected_positions`),
next to the module's `DEFAULT_*` constant where there is one.

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `gbf.pipeline.KINEMATICS_PARAMS[sg_window_frames]`, `gbf.pipeline.KINEMATICS_PARAMS[sg_poly]` | 11; 2 | hand-set | Savitzky-Golay window of 0.44 s at 25 fps; not recorded |
| `gbf.pipeline.KINEMATICS_PARAMS[max_gap_frames]`, `gbf.pipeline.KINEMATICS_PARAMS[min_segment_frames]` | 3; 25 | hand-set | not recorded |
| `gbf.pipeline.KINEMATICS_PARAMS[max_speed_mps]` | 12.0 | hand-set | docstring: above the fastest recorded match sprints, about 10.5-11 m/s |
| `DEFAULT_MAX_COMMON_STEP_M`, `gbf.pipeline.KINEMATICS_PARAMS[max_common_step_m]` | 0.5 | hand-set | docstring: a whole team moving at 12.5 m/s |
| `gbf.pipeline.KINEMATICS_PARAMS[min_common_players]`, `gbf.pipeline.KINEMATICS_PARAMS[outlier_radius_frames]` | 4; 5 | hand-set | not recorded |
| `DEFAULT_OUTLIER_THRESHOLD_M`, `gbf.pipeline.KINEMATICS_PARAMS[outlier_threshold_m]` | 1.5 | hand-set | docstring: a merged or occluded box moves the foot point by several metres, while a steady run stays far below; value not recorded |
| `DEFAULT_MAX_BOX_HEIGHT_CHANGE`, `gbf.pipeline.KINEMATICS_PARAMS[max_box_height_change]` | 0.2 | hand-set | box-size test; not recorded |
| `DEFAULT_MAX_BOX_WIDTH_CHANGE`, `gbf.pipeline.KINEMATICS_PARAMS[max_box_width_change]` | None | hand-set | off: arms and stride change a runner's box width by more than 20 % (docstring) |
| `DEFAULT_JUMP_THRESHOLD_M`, `gbf.pipeline.KINEMATICS_PARAMS[jump_threshold_m]` | 1.0 | hand-set | trend and jump test. Docstring: above the planted-foot steps once a trend is fitted, below an identity switch onto a player 1.5 m away |
| `DEFAULT_JUMP_FACTOR`, `gbf.pipeline.KINEMATICS_PARAMS[jump_factor]` | 5.0 | hand-set | docstring: a multiple of the other players' median, so that camera and calibration motion is no jump; value not recorded |
| `DEFAULT_JUMP_WINDOW_FRAMES`, `gbf.pipeline.KINEMATICS_PARAMS[jump_window_frames]` | 12 | hand-set | docstring: 0.48 s at 25 fps, about a sprinter's stride |
| `DEFAULT_MAX_ACCELERATION_MPS2`, `gbf.pipeline.KINEMATICS_PARAMS[max_acceleration_mps2]` | 8.0 | hand-set | acceleration test. Docstring: sprinters start and stop at up to about 6-7 m/s^2; a box that jumps 0.6 m onto a neighbour from one frame to the next, or slides 1.5 m onto one in 8 frames, needs more |
| `_MIN_JUMP_SIDE`, `_MIN_EXCURSION_SIDE`, `_EXCURSION_ROUNDS` | 3; 2; 3 | hand-set | not recorded |
| `_EXCURSION_FIT_CUT`, `_EXCURSION_REFITS` | 0.5; 2 | hand-set | excursion test: each block is tested again on side fits made twice more without the side positions farther than half of `jump_threshold_m` from their line (those continuing the block excepted), so that a second dip nearby does not hide it; a block only these fits find stands only when the test rejects the positions they left out as well (the other dip). Docstring: sprint positions held for 3 frames stay within 0.43 m of their line at 10.5 m/s; on 6972 synthetic sprints, accelerations, stops, turns and reversals with 0-20 cm of noise the test rejects the same positions as the first fit alone, and it rejects both dips of 3127 of 3150 clean pairs (the first fit alone 1987) |
| `_EXCURSION_CHUNK` | 20000 | hand-set | memory only: samples (whole tracks) whose blocks the excursion test fits at once; the result does not depend on it. Docstring: on 87272 synthetic samples `rejected_positions` peaked at 331 MB (traced) in one chunk and at 129 MB in chunks of 20000 |
| `_ACCELERATION_NOISE_SE` | 3.0 | hand-set | docstring: standard errors by which the acceleration test first reduces the displacement between the side trends, so that noise and the planted-foot steps are not taken for an acceleration; value not recorded |
| `DEFAULT_SPEED_CHORD_S`, `gbf.pipeline.KINEMATICS_PARAMS[speed_chord_s]` | 0.4 | hand-set | speed chord. Docstring: about a whole stride, so that the planted-foot steps average out; gbf.measure's `peak_speed_window_s` |

### gbf.ball (`BallParams`: `DEFAULT_PARAMS` for strikes, `CONTACT_PARAMS` for contacts)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `BallParams.aspect_min`, `BallParams.aspect_max` | 0.6; 1.6 | hand-set | not recorded |
| `BallParams.blur_min_conf`, `BallParams.blur_aspect_max` | 0.6; 3.0 | hand-set | motion-blur exception for elongated boxes: docstring, on the development clip a fast ball gave boxes about twice as long as wide at high confidence; the exact values are not recorded |
| `BallParams.blur_min_step`, `BallParams.blur_max_step` | 0.4; 2.0 | hand-set | blur_min_step not recorded; blur_max_step from the docstring's blur model: at the longest accepted blur a ball moves 4/3 to 2 long sides per frame |
| `BallParams.blur_axis_tol_deg`, `BallParams.blur_neighbour_min_conf`, `BallParams.blur_edge_px` | 30.0; 0.2; 2.0 | hand-set | not recorded |
| `BallParams.nearby_box_heights`, `BallParams.size_ratio_min`, `BallParams.size_ratio_max` | 3.0; 0.05; 0.3 | hand-set | a size gate around ball_to_box_height; not recorded |
| `BallParams.fallback_diameter_min_px`, `BallParams.fallback_diameter_max_px` | 4.0; 120.0 | hand-set | not recorded |
| `BallParams.pitch_half_length_m`, `BallParams.pitch_half_width_m` | 55.0; 36.5 | hand-set | the field lines (52.5 m, 34 m) plus a margin (docstring); not recorded |
| `BallParams.near_player_margin_bh` | 0.3 | hand-set | not recorded |
| `BallParams.static_radius_px`, `BallParams.static_min_s`, `BallParams.static_max_gap_s` | 3.0; 1.0; 0.12 | hand-set | not recorded |
| `BallParams.static_camera_shift_px`, `BallParams.static_min_moved_frames`, `BallParams.static_min_common_tracks` | 10.0; 3; 3 | hand-set | not recorded |
| `BallParams.parked_step_px` | 3.0 | hand-set | docstring: detector jitter (about 1 px) plus a slowly drifting camera |
| `BallParams.world_static_radius_m`, `BallParams.world_static_min_members` | 1.5; 6 | hand-set | radius: docstring, with the per-frame calibration a spare ball near the far touchline scatters over more than a metre within one shot; members: not recorded |
| `BallParams.off_field_margin_m`, `BallParams.off_field_point_margin_m`, `BallParams.off_field_carry_px`, `BallParams.off_field_carry_frames` | 1.0; 0.5; 4.0; 2 | hand-set | point margin: docstring, a ball in play has its centre at most 0.11 m beyond the line while a spare ball's points scatter by more than a metre; the others not recorded |
| `BallParams.size_sigma_log`, `BallParams.size_sigma_log_players` | 0.35; 0.7 | hand-set | not recorded |
| `BallParams.ball_box_scale` | 1.2 | fitted | detector box diameter over projected ball diameter, measured 1.17-1.21 on calibrated frames of the clip (docstring) |
| `BallParams.ball_to_box_height` | 0.155 | fitted | detector box diameter over player box height, measured on the clip: median 0.155, range 0.14-0.17 (docstring; the physical ratio is 0.12) |
| `BallParams.motion_ref_bh_per_s`, `BallParams.motion_weight` | 17.5; 0.5 | hand-set | 0.7 bh per frame at 25 fps, about 30 m/s (docstring); not recorded |
| `BallParams.max_link_speed_bh_per_s`, `BallParams.size_change_sigma_log`, `BallParams.max_link_gap_s` | 30.0; 0.15; 0.32 | hand-set | not recorded |
| `BallParams.isolated_gap_s`, `BallParams.isolated_min_conf` | 0.12; 0.6 | hand-set | not recorded |
| `BallParams.missing_cost` | 2.0 | hand-set | a candidate beats "missing" from conf exp(-2) = 0.135; on the clip 94 % of the tracked detections have conf >= 0.2 and the rejected clutter a median conf of 0.15 (docstring) |
| `BallParams.reacquire_cost` | 3.0 | hand-set | not recorded |
| `BallParams.max_interp_gap_s`, `BallParams.max_interp_speed_bh_per_s` | 0.24; 20.0 | hand-set | not recorded |
| `BallParams.event_window_s`, `BallParams.event_angle_deg`, `BallParams.event_speed_ratio` | 0.12; 35.0; 2.0 | hand-set | not recorded |
| `BallParams.event_speed_floor_bh_s`, `BallParams.event_min_change_bh_s`, `BallParams.event_max_speed_bh_s` | 1.5; 2.0; 30.0 | hand-set | not recorded |
| `BallParams.event_at_foot_bh` | 0.6 | hand-set | set between the clip's contacts, measured at 0.06-0.44 box heights, and its bounces and flight apexes, at 1.3 and more (docstring) |
| `BallParams.event_max_fit_rms_bh`, `BallParams.event_min_fit_rms_px` | 0.08; 2.0 | hand-set | not recorded |
| `CONTACT_PARAMS.event_speed_floor_bh_s`, `CONTACT_PARAMS.event_min_change_bh_s`, `CONTACT_PARAMS.event_angle_deg` | 0.3; 0.5; 45.0 | hand-set | docstring: close-shot ball speeds of 0.1-1 bh/s, which a touch changes by 0.5-1.1 bh/s (noise about 0.05 bh/s); 55 events on the clip, 44 of them at a foot |
| literal in `_segments`: `jump > 3.0 * p.size_change_sigma_log * math.sqrt(gap)` | 3.0 | hand-set | a diameter jump beyond 3 sigma starts a new event segment; not recorded |
| `BallParams.ball_diameter_m`, `BallParams.ball_radius_m` | 0.22; 0.11 | football fact | a size 5 ball |
| `FIELD_HALF_LENGTH_M`, `FIELD_HALF_WIDTH_M` | 52.5; 34.0 | football fact | SoccerNet's 105 x 68 m pitch |

### gbf.audio (excitement cue)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `DEFAULT_SAMPLE_RATE` | 16000 | hand-set | docstring: speech and crowd energy sit well below 8 kHz |
| `WINDOW_FRAME_DURATIONS`, `EXCITEMENT_SMOOTHING_S`, `GAP_FILL_MIN_S` | 2; 0.5; 0.1 | hand-set | not recorded |

### gbf.highlights (cues and detection)

The module docstring calls these starting values chosen from football facts (pitch
dimensions, sprint speeds, typical duel distances) rather than tuned to a clip. The clip is
still the only footage they were checked on, and the weights and evidence thresholds are
judgement, not facts.

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `CUE_WEIGHTS[cut_emphasis]`, `CUE_WEIGHTS[audio]`, `CUE_WEIGHTS[speed]`, `CUE_WEIGHTS[box_pressure]`, `CUE_WEIGHTS[ball_goal]`, `CUE_WEIGHTS[duel]`, `CUE_WEIGHTS[strike]` | 1.0; 1.5; 0.75; 1.0; 1.0; 1.0; 1.25 | hand-set | not recorded |
| `CUT_PRE_S`, `CUT_PRE_VALUE`, `CUT_POST_VALUE`, `FOLLOW_MAX_S` | 1.5; 1.0; 0.5; 10.0 | hand-set | not recorded |
| `AUDIO_Z_MAX`, `MIN_PITCH_FRESH_SHARE` | 4.0; 0.5 | hand-set | not recorded |
| `SPEED_LOW_MPS`, `SPEED_HIGH_MPS`, `SPEED_PLAUSIBLE_MAX_MPS` | 4.0; 9.0; 12.0 | hand-set | from sprint speeds (module docstring) |
| `BOX_FULL_ATTACKERS`, `BOX_PRESSURE_ALONE_MIN` | 3; 0.5 | hand-set | BOX_FULL_ATTACKERS not recorded; BOX_PRESSURE_ALONE_MIN is two attackers, since one player standing in the box while the ball is out of play is no attack (docstring) |
| `ATTACK_BALL_SEEN_S` | 1.0 | hand-set | docstring: during a stoppage players stand in the box while the ball is out of the picture for seconds; also the window in which the ball must be at the box for box pressure and for an attack_in_box; value not recorded |
| `BALL_AT_BOX_MARGIN_M` | 1.5 | hand-set | the calibration error at the box, not a tolerance for attacks from outside it: on the clip's pitch-usable frames with a penalty-area line in view, the 8 px reprojection error pitch-usable frames may have (`gbf.sngs_import.DEFAULT_REP_ERR_MAX_PX`) is 1.1 m (median), 1.3 m (P90) and at most 1.4 m on the ground along the most foreshortened direction at those lines (the frames' own reprojection errors: 0.5 m median, 1.1 m at most); rounded up |
| `BALL_GOAL_FAR_M` | 25.0 | hand-set | not recorded |
| `BALL_GOAL_NEAR_M` | 11.0 | hand-set | the distance at which the ball_goal cue reaches its full value, put at the penalty mark (11 m) |
| `DUEL_BALL_RADIUS_M`, `DUEL_NEAR_M`, `DUEL_FAR_M` | 4.0; 1.0; 4.0 | hand-set | from typical duel distances (module docstring) |
| `DUEL_BALL_RADIUS_H`, `DUEL_NEAR_H`, `DUEL_FAR_H` | 2.0; 0.5; 2.0 | hand-set | the image-plane counterparts, one box height standing for about 1.8 m (docstring) |
| `DUEL_IMAGE_MAX_BOX_H_PX`, `SHOT_UPPER_BODY_MIN_BOX_H_PX` | 540.0 | hand-set | half the frame: a taller median box is an upper body whose box bottom is the frame edge (docstring); also gbf.measure's `image_feet_max_median_box_px` |
| `STRIKE_HALF_WIDTH_S` | 0.6 | hand-set | not recorded |
| `STRIKE_FULL_MAGNITUDE_BH_S` | 10.0 | hand-set | docstring: about 18 m/s for a 1.8 m box, a firm pass or a shot |
| `STRIKE_FOOT_NEAR_H`, `STRIKE_FOOT_FAR_H`, `STRIKE_UNKNOWN_FOOT_WEIGHT` | 1.0; 2.0; 0.5 | hand-set | not recorded |
| `STRIKE_KIND_WEIGHTS[speed_up]`, `STRIKE_KIND_WEIGHTS[direction_change]`, `STRIKE_KIND_WEIGHTS[slow_down]` | 1.0; 1.0; 0.5 | hand-set | docstring: a slow-down is a reception or a block rather than a strike |
| `SMOOTH_SIGMA_S`, `MIN_PEAK_SCORE` | 0.5; 0.08 | hand-set | not recorded |
| `WINDOW_BEFORE_S`, `WINDOW_AFTER_S`, `SNAP_S` | 4.0; 2.0; 0.5 | hand-set | not recorded |
| `REFINE_S`, `REFINE_SMOOTH_S` | 1.0; 0.2 | hand-set | smoothing: comment, 5 frames at 25 fps so that a one-frame flicker cannot become the peak; `REFINE_S` not recorded |
| `PLAY_EVIDENCE_MIN`, `BROADCAST_CUE_MIN` | 0.2; 0.02 | hand-set | not recorded |
| literal in `_describe`: `contribution >= 0.005` | 0.005 | hand-set | smallest contribution of a cue named among a summary's reasons; not recorded |
| `JERSEY_MIN_CONF`, `MAX_TRACK_IDS` | 0.5; 6 | hand-set | not recorded |
| `detect_highlights(min_separation_s)`, `gbf.pipeline.PipelineConfig.min_separation_s` | 4.0 | hand-set | the default of `--min-separation-s`; not recorded |
| `detect_highlights(max_highlights)`, `gbf.pipeline.PipelineConfig.max_highlights` | 6 | hand-set | the default of `--max-highlights`; not recorded |
| `HALF_LENGTH_M`, `PENALTY_AREA_DEPTH_M`, `PENALTY_AREA_HALF_WIDTH_M` | 52.5; 16.5; 20.16 | football fact | SoccerNet's 105 m pitch; the penalty area is 16.5 m deep and 40.32 m wide |

### gbf.lift3d (height-prior skeleton, facing, trunk lean)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `DEFAULT_SCORE_THRESHOLD` | 0.3 | hand-set | not recorded |
| `PRIOR_STATURE_M`, `STATURE_RANGE_M` | 1.8; (1.55, 2.05) | hand-set | the stature the joint heights are written for, and the bounds of the fitted one; not recorded |
| `JOINT_HEIGHT_PRIOR_M` | (1.65, 1.65, 1.65, 1.65, 1.65, 1.45, 1.45, 1.1, 1.1, 0.85, 0.85, 0.95, 0.95, 0.5, 0.5, 0.09, 0.09, 1.8, 1.55, 0.95, 0.03, 0.03, 0.03, 0.03, 0.05, 0.05) | hand-set | joint heights of a standing 1.80 m player, in Halpe-26 order; source not recorded |
| `MAX_TRUNK_TILT_COMPONENT`, `MAX_FEET_SPREAD_M` | 0.7; 1.4 | hand-set | not recorded |
| `HIP_HEIGHT_SD_M`, `HIP_OFFSET_SD_M`, `POSTURE_HEIGHT_SD_M` | 0.05; 0.25; 0.05 | hand-set | not recorded |
| `MIRROR_COST_TOLERANCE`, `ROBUST_SCALE` | 0.05; 3.0 | hand-set | not recorded |
| `SHOULDER_WIDTH_M`, `HIP_WIDTH_M` | 0.38; 0.22 | hand-set | docstring: typical keypoint spacing of a 1.80 m player, plausibility weights only; source not recorded |
| `facing_sigma_deg(pixel_sigma)`, `trunk_lean_sigma_deg(pixel_sigma)` | 1.0 | hand-set | one pixel of keypoint error in u and in v; `lift_height_prior` relies on this default, and gbf.measure's `facing_max_sigma_deg` and `lean_max_sigma_deg` gate the resulting sigmas; not recorded |
| `FACE_VISIBLE_SCORE`, `FACE_HIDDEN_SCORE` | 0.6; 0.25 | hand-set | not recorded |
| `LABEL_CUE_WEIGHT`, `FACE_VISIBLE_WEIGHT`, `FACE_HIDDEN_WEIGHT` | 1.0; 1.5; 0.5 | hand-set | not recorded |

### gbf.moments (key moments, merging, ranking)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `MIN_SCORE` | 0.3 | hand-set | not recorded |
| `UPRIGHT_DEG`, `FALL_DOWN_DEG`, `FALL_MAX_S`, `FALL_HIP_RATIO` | 20.0; 45.0; 1.0; 0.8 | hand-set | not recorded |
| `CONTACT_BODY_H`, `CONTACT_GAP_FRAMES` | 0.12; 3 | hand-set | not recorded |
| `MIN_BODY_PX` | 150.0 | hand-set | comment: below it players overlap in the image without touching, and a crouch looks like a fall; value not recorded |
| `RELEASE_NEAR_BODY_H`, `RELEASE_AWAY_BODY_H`, `RELEASE_WITHIN_S`, `RELEASE_MIN_SPEED_BH_S` | 0.15; 0.3; 0.2; 2.5 | hand-set | not recorded |
| `RAISE_SHOULDER_W`, `ARM_STRAIGHT_DEG`, `GESTURE_MIN_S`, `GESTURE_GAP_S`, `GESTURE_MIN_SHOULDER_PX` | 0.25; 140.0; 0.3; 0.2; 35.0 | hand-set | shoulder width: comment, about a whole body of `MIN_BODY_PX` (a close-up or a near medium shot, never a wide shot); the others not recorded; see the close-up reaction rule above |
| `MERGE_GAP_FRAMES`, `MERGE_MAX_S` | 1; 15.0 | hand-set | not recorded |
| `REACTION_LEAD_S` | 2.0 | hand-set | not recorded |
| `PEAK_KINDS`, `PAIR_RANK_KINDS` | ("ball_release", "fall_onset"); ("ball_release", "fall_onset", "ball_away") | hand-set | rules: the moments that can move a peak, and the moments that rank duel pair candidates (touches do not) |
| literal in `_key_moments_of`: `len(both) < 5` | 5 | hand-set | a duel pair needs a pose of both players on at least 5 frames; not recorded |
| `PARTNER_MAX_BOX_H`, `PARTNER_MIN_SHARE` | 1.0; 0.5 | hand-set | not recorded |
| `MAIN_SHARE`, `MAIN_MAX` | 0.5; 3 | hand-set | not recorded |

### gbf.measure (`MeasureParams`; the pipeline uses `DEFAULT_PARAMS`)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `MeasureParams.key_half_window` | 2 | hand-set | frames on each side of a reference frame for medians (nearest opponent, facing, lean); docstring: the labels state it within the card's 16 characters |
| `MeasureParams.contact_radius_bh` | 0.15 | hand-set | chosen by eye on the clip's contact sheets: matches at 0.2-0.28 box heights showed the foot visibly away from the ball (docstring) |
| `MeasureParams.contact_fallback_bh` | 0.6 | hand-set | gbf.ball's `event_at_foot_bh` |
| `MeasureParams.contact_candidates`, `MeasureParams.contact_candidate_max_bh` | 3; 2.0 | hand-set | not recorded |
| `MeasureParams.keypoint_min_score`, `MeasureParams.pose_min_mean_score` | 0.3; 0.4 | hand-set | not recorded |
| `MeasureParams.facing_min_conf`, `MeasureParams.facing_max_sigma_deg`, `MeasureParams.facing_min_resultant`, `MeasureParams.lean_max_sigma_deg` | 0.5; 15.0; 0.7; 8.0 | hand-set | display gates of the A_candidate angles; not recorded |
| `MeasureParams.min_segment_frames`, `MeasureParams.max_gap_frames`, `MeasureParams.sg_window_frames` | 25; 3; 11 | hand-set | as the kinematics settings `min_segment_frames`, `max_gap_frames` and `sg_window_frames` |
| `MeasureParams.peak_speed_window_s` | 0.4 | hand-set | gbf.kinematics' `DEFAULT_SPEED_CHORD_S` |
| `MeasureParams.peak_speed_min_coverage` | 0.5 | hand-set | not recorded |
| `PEAK_SPEED_GAP_MARGIN` | 0.05 | hand-set | docstring: a stretch without valid speeds hides a faster peak when the player's mean speed across it exceeds the highest valid speed by more than this share; value not recorded |
| `FACING_LEAN_DEG`, `UPRIGHT_DEG` | 20.0 | hand-set | shared with gbf.moments' upright bound: a trunk leaning farther leans to that side (docstring) |
| `FACING_SIDE_SHARE` | 0.5 | hand-set | docstring: a facing 30 degrees or more off the picture's vertical points to one side of it; value not recorded |
| literal in `_fit_homography`: `float(residual.max()) > 1.0` | 1.0 | hand-set | the largest pixel residual of a frame's ground homography: the calibration links its players' ground positions to their foot points, so one homography fits them to rounding unless they come from different cameras (docstring); value not recorded |
| `MeasureParams.image_angle_min_box_px` | 150.0 | hand-set | the B-tier general gate (docstring) |
| `MeasureParams.image_feet_max_median_box_px` | 540.0 | hand-set | gbf.highlights' `DUEL_IMAGE_MAX_BOX_H_PX` |
| `HALF_LENGTH_M`, `PENALTY_AREA_DEPTH_M`, `PENALTY_AREA_HALF_WIDTH_M` | 52.5; 16.5; 20.16 | football fact | as in gbf.highlights |

### gbf.render (display rules)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `EDGE_TOUCH_PER_BOX_HEIGHT`, `gbf.track_tools.EDGE_TOUCH_PER_BOX_HEIGHT` | 0.03 | hand-set | shared cutoff tolerance: a detector's box around a person cut off by the frame stops a little short of the edge; value not recorded |
| `CLOSEUP_BOX_SHARE` | 0.5 | hand-set | not recorded |
| `Skeleton.min_score` | 0.3 | hand-set | joints scoring below it are not drawn; not recorded |
| `TRACK_HOLD_MAX_GAP` | 8 | hand-set | comment: a gap of a third of a second would blink on screen in 4x slow motion |
| `DUEL_LINK_WINDOW_S` | 1.0 | hand-set | not recorded |
| `FOCUS_BALL_MAX_M` | 5.0 | hand-set | comment: seconds before the cut of a merged or extended highlight the play can be about other players than the ones the key frame is about; value not recorded |
| `MINIMAP_BALL_MARGIN_M` | 1.0 | hand-set | not recorded |
| `BALL_GROUND_MAX_SPEED_MPS`, `BALL_SPEED_SPAN_FRAMES` | 30.0; 2 | hand-set | comment: a ball rolling on the grass is slower, and a ball in the air projected onto the grass races across the minimap; values not recorded |
| `REPLAY_LOOP_MIN_S` | 0.5 | hand-set | not recorded |
| `PITCH_LENGTH_M`, `PITCH_WIDTH_M` | 105.0; 68.0 | football fact | SoccerNet's pitch template |
