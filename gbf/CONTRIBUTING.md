# Contributing

These rules come from real failures while testing the sn-gamestate baseline (see
`docs/adr/` and the project plan). Most of them are enforced by `python -m gbf gate --full`.

## Environment

- Python 3.11 or 3.12. On Windows the project keeps its toolchain inside the repo:
  `UV_CACHE_DIR=.local/uv-cache`, `UV_PYTHON_INSTALL_DIR=.local/python`, then `uv sync`.
- Run tools with the virtualenv interpreter (`.venv/Scripts/python.exe -m pytest` on
  Windows, `.venv/bin/python -m pytest` elsewhere). Do not rely on `uv run`.
- Set `PYTHONUTF8=1`. Keep the checkout on a path without spaces or non-ASCII characters.
- `ffmpeg` and `ffprobe` must be on `PATH` (recorded in `registries/tools.toml`). They are
  external tools; never bundle them.

## Code

- Text files are UTF-8 with LF line endings, in the repo and in the working tree
  (`.gitattributes`). Write files with `gbf.util.io.write_text` / `write_json`, or
  `open(..., newline="\n")`. Hash raw bytes (`gbf.util.hashing`), never decoded text.
- Paths: use `pathlib`; code must work with spaces and non-ASCII characters in paths.
  Pass subprocess arguments as lists; never use `shell=True`.
- Images: Pillow or ffmpeg. `cv2.imread` / `cv2.imwrite` are banned (they fail silently on
  non-ASCII paths on Windows).
- Frames are read only through `gbf.ingest` (`iter_frames`, `read_frames`, `frame_count`)
  and windows are cut only with `gbf.ingest.extract_window`. The canonical frame number is
  (source sha256, CFR index). `cv2.VideoCapture` is banned in project code.
- No `assert` for runtime validation in `src/` (it disappears under `python -O`);
  raise an explicit exception that names every problem found.
- No global warning filters, no pickle for artifacts.
- State key hyperparameters explicitly (library defaults change: scikit-learn's KMeans
  `n_init` did) and log library versions at startup of long runs.
- Seeds are explicit. Anything random is reported over several seeds with mean and worst/best.

## Tests

- No network: the test session blocks every non-loopback connection and DNS lookup.
- No skips: the gate treats a skipped or xfailed test as a failure. If a tool is missing,
  fail loudly.
- Use the `unicode_tmp_path` fixture for anything that touches the filesystem.
- Test files never contain literal user-home paths, absolute paths through a `gbf-data` or
  `private` folder, or credential-shaped strings; assemble them at run time (see
  `tests/gate/`), under a neutral root such as `D:/work` rather than your own machine's.
  The gate's rules, the minimum test counts and the exemption procedure are in
  `gate/README.md`.
- Generate media at test time. Never commit videos, frames or screenshots; a tracked image
  (e.g. a golden frame) must be registered in `registries/media.toml` with its sha256.
- A new patch or module runs on 5 frames first, then 50, then the full input.

## Branches and pushing

- Push only `main` (`git push origin main`), never `--all` or `--mirror`: merged or
  abandoned branches can still hold text that `main` has since removed.
- Delete merged `fix/*` and `integ/*` branches before a remote is added.

## Data and licenses

- Never commit videos, frames, StatsBomb JSON, SoccerNet label files or anything derived
  from them (ADR 0002). Download scripts pin a source revision and verify a hash.
- Never commit NDA case files, identity overrides, manual labels, raw plugin outputs, plugin
  job directories, or a JSON, JSON lines, HTML, SVG or XML file whose restrictions include
  `nda`, `no_share`, `research_nc` or `no_identity`; the gate rejects them (G4). It fails on
  anything below a directory named `cases`, `labels`, `label`, `manual_label`,
  `manual_labels*`, `runs`, `data` or `*-<16 hex digits>` (a plugin job); on file names
  that match `*identity_override*`, `*detections*` or `*ball_raw*` with any suffix,
  `calib*` or `keypoints*` as JSON lines, `manual_*` as JSON or JSON lines, and
  `*plugin-receipt*.json`, `*keypoints-receipt*.json` or `*ball-receipt*.json` (`-` and `_`
  alike), so copies such as `detections (1).jsonl` and `Copy of ball_raw.jsonl` fail too;
  on hand labels under any JSON name (JSON with a `moments` list whose items have the `id`,
  `type`, `start`, `peak` and `end` that `gbf.label_eval` reads); on a `restrictions` key
  in any letter case at the top of a JSON document, one level down, or in the first
  records of an array or a JSON lines file; on restriction statements and the 私有素材
  banner in HTML, SVG and XML pages; and on files it cannot scan (too large, undecodable,
  JSON that does not parse but mentions restrictions, a saved page nested too deeply to
  parse or with a charset that cannot be used). `.gitignore` hides the same names with the
  suffixes such material is written with (the `*identity_override*`, `*detections*` and
  `*ball_raw*` names as data, compressed copies such as `.jsonl.gz` and backups such as
  `.bak` or `~`) and re-includes nothing but
  `src/gbf/contracts/schemas/plugin-receipt.schema.json`, so what you exclude in
  `.git/info/exclude` stays excluded. It does not hide every copy that the gate rejects:
  archives (`.zip`, `.7z`), numbered or partial copies (`.jsonl.1`, `.part`), editor and
  lock files (`.swp`, `.~lock.*#`), spreadsheets and databases (`.xlsx`, `.sqlite`), hand
  labels under another name, and names in another letter case where git matches
  `.gitignore` case-sensitively stay visible to `git add -A`. The full rules, and what
  they do not cover, are in `src/gbf/gate/datafiles.py`.
- Keep ordinary files clear of those rules: never name a code package or a fixture
  directory `cases`, `labels`, `label`, `manual_labels`, `runs` or `data`. Code and docs
  (`*.py`, `*.pyi`, `*.md`, `*.rst`) may use the restricted names (`identity_overrides.py`,
  `detections.md`), and so may the JSON Schemas in `src/gbf/contracts/schemas/`
  (`plugin-receipt.schema.json`); a schema elsewhere needs a top-level `"$schema"` key.
  `.gitignore` hides every schema named like restricted material except
  `plugin-receipt.schema.json` itself (a later version of it is hidden too): add a negation
  of its exact path to `.gitignore`, next to that one, so that it can be committed without
  being ignored. Generate synthetic plugin outputs and hand labels at test time, and write
  a placeholder, not a blocked tag, when an HTML or SVG page explains a restriction
  statement.
- The core never imports plugin code (`tracklab`, `sn_gamestate`, `ultralytics`, `prtreid`,
  `torchreid`, `mmocr`, `sam_3d_body`, `nlf`, ...). Plugins run as subprocesses and exchange
  files that follow `src/gbf/contracts/schemas/` (ADR 0001). A model-registry entry that
  is blocked, research_only or plugin_only needs a row with its import and distribution
  names in `REGISTRY_BANS` (`src/gbf/gate/imports.py`), or the gate fails (G3). Besides
  import statements the gate counts calls of `importlib.import_module`, `__import__`,
  `pytest.importorskip`, `importlib.util.find_spec`, `pkgutil.get_loader`,
  `pkgutil.find_loader`, `runpy.run_module` and `pkgutil.resolve_name` with a literal name,
  also through an alias (imported, assigned or unpacked from a tuple) or with `name=`, in
  `.py`, `.pyi` and `.pyw` files under `src/` and `tests/`; names computed at run time are
  not checked, so a test that asserts a banned module is absent builds its name at run
  time.
- Do not copy code from repositories without a license, or from GPL/AGPL/HL3/non-commercial
  sources, into the core.
- Every artifact carries `restrictions`, the union of its inputs' restrictions.
