# ADR 0003: Artifact contracts and the canonical frame index

## Status

Accepted, 2026-10-21.

## Context

The sn-gamestate tests lost frames and trust in ways that no single model change would
have fixed:

- tracklab's `VideoReader` read frame `idx - 1`: frames 0 and 1 were duplicates, the last
  frame was never processed, and everything was 40 ms out of step with other models.
- The same 67-second clip decoded to 1,668 frames on one machine (ffmpeg without
  `-fps_mode cfr`) and 1,681 frames on Kaggle.
- Tracker state was written to disk only when a whole video finished, as a pandas pickle;
  one crash lost 23 minutes of compute.
- On Windows, Git and Python's text mode turn LF into CRLF unless told otherwise, so two
  copies of the "same" text file hash differently.
- GameBuddy's receipts had no shared schema and leaked absolute local paths and a user
  name.

Event data, broadcast video, plugins and receipts only line up if every artifact names
frames, files and paths the same way, on every machine.

## Decision

### 1. Canonical frame index

- A frame is identified by **(source sha256, CFR frame index)**: the sha256 of the raw
  bytes of the source video file (for SoccerNet, one half), and the frame's 0-based index
  after normalisation to a constant frame rate. Frame `k` is the source frame on screen
  `k / fps` seconds after the first decoded frame; `fps` is an exact rational (`25/1`,
  `30000/1001`).
- Ingest normalises to CFR (`-fps_mode cfr`). The frame count must equal what ffprobe
  reports for the canonical file, and one frame-reading function is used everywhere.
  `video.json` (`urn:gbf:schema:video`) records the source and canonical files, the
  rational frame rate, frame count, resolution, cuts and windows.
- Every stage artifact is keyed by the canonical frame index (plus `track_id` where
  relevant). Every result directory records the frame count, rational frame rate,
  resolution, video sha256, git commit and weight sha256s.
- **Windows.** An event window is a canonical frame range, inclusive at both ends. Frame
  `k` of a window file is canonical frame `start + k`, decoded bit-identical; the L0
  synthetic pipeline proves this with frame-number barcodes, including windows that start
  at a non-zero offset. `track_id`s carry the window id as a prefix. `job.json` hands
  plugins canonical frame ranges, never file-local frame numbers.
- **Clocks.** Video PTS, the match clock (period plus timestamp) and the output timeline
  (segment table) are converted through rational frame arithmetic; offsets and residuals
  are written into the records that use them.
- **Coordinates.** World coordinates follow SoccerNet: metres, origin at the centre spot,
  x in [-52.5, 52.5], y in [-34, 34], z pointing down; internally gbf uses `pitch_zup`.
  StatsBomb's 120 x 80 pitch is converted to metres with kloppy. The attack direction of
  each half is asserted automatically, not assumed.
- **Interpolated frames** (minterpolate, RIFE) exist only for slow-motion display and are
  marked in receipts. No time quantity is computed from a replay whose slow-motion rate
  is uncertain.
- **Proof before use.** Before a plugin run, the core checks that frame 0 as the plugin
  decodes it equals ffmpeg's frame 0 pixel for pixel. Across environments, 5 sampled
  frames are compared within the tolerance in `prereg/gates.v1.json` (frozen in the
  working tree; pre-registered once pushed, see `prereg/README.md`).

### 2. Text files and hashing

- Text files are UTF-8 with LF line endings in the repository and in the working tree
  (`.gitattributes`: `* text=auto eol=lf`; clone with `core.autocrlf=false` on Windows).
  Hash-registered fixtures and receipts are marked `-text`, so git never rewrites them.
- Code writes text through `gbf.util.io.write_text` / `write_json` (UTF-8, `newline="\n"`).
- Every hash is computed over raw bytes (`gbf.util.hashing`), never over decoded text.
  Downloads are verified by size and by the git blob id the host publishes for the file.
- Content-addressed ids (for example `evidenceId`) hash canonical JSON: sorted keys,
  compact separators, UTF-8, no NaN or Infinity.

### 3. File formats

- Tables are Parquet; arrays are `.npz`, loaded with `allow_pickle=False`; small
  documents are JSON.
- No pickle anywhere, including pandas pickles as long-term storage (ruff TID251 bans
  the module).
- Each stage writes its own files, so changing one stage reruns only that stage.

### 4. Contracts

- Every artifact the pipeline writes or accepts (plugin outputs, receipts, registries,
  ingest and alignment records) has a JSON Schema (draft 2020-12) in
  `src/gbf/contracts/schemas/<name>.schema.json` with `$id` `urn:gbf:schema:<name>`.
  Schemas reference each other through these URNs; shared pieces live in
  `urn:gbf:schema:common` and `urn:gbf:schema:restrictions`.
- Validation reports every problem at once: `gbf.contracts.ContractError` lists them all
  instead of stopping at the first.
- Receipts are written by scripts, never by hand. Each states what it certifies (`scope`)
  and what it does not (`limitations`).
- Registries are TOML and are validated as the dict that `tomllib` returns. TOML has no
  null and its date type is not a JSON string, so registries quote dates
  (`"YYYY-MM-DD"`) and use documented sentinels for inapplicable values.

### 5. Paths

- Artifacts and receipts store POSIX paths relative to the repository root or to
  `DATA_ROOT`. The `relpath` definition in `urn:gbf:schema:common` rejects absolute paths,
  drive letters, backslashes and `..` segments, so no receipt can leak a local path or
  user name.
- Code must work with spaces and non-ASCII characters in paths (tests run in such a
  directory); the checkout itself lives on an ASCII path. Subprocess arguments are
  passed as lists, never through `shell=True`. Image IO goes through Pillow or ffmpeg,
  never `cv2.imread` / `cv2.imwrite`, which fail silently on non-ASCII paths on Windows.

## Consequences

- Any two artifacts can be joined on (source sha256, frame index), whatever machine,
  plugin or window produced them, and an off-by-one shift shows up as a failed check
  instead of a silent 40 ms error.
- Normalising to CFR costs disk space and ingest time; window files are cut losslessly so
  that bit-identity can be checked.
- Every new artifact type needs a schema before it can be written or accepted, which
  slows the first version of each stage but keeps plugins honest about what they return.
- Hashes stay stable across Windows and Linux only as long as the LF rules hold; the
  gate re-hashes registered files on every run to catch drift.

## References

- Plan sections 3.1 (plugin protocol), 3.4, 5.7, 7 (G1, G2, G5, G16, G17).
- `src/gbf/util/hashing.py`, `src/gbf/util/io.py`, `src/gbf/contracts/`,
  `.gitattributes`, [ADR 0001](0001-license-architecture.md),
  [ADR 0002](0002-data-and-media-policy.md).
