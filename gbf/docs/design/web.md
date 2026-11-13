# Local web page: `python -m gbf web`

Status: video-only workflow added 2026-10-22 (see [video-workflow.md](video-workflow.md)).
A page in the owner's browser, served by gbf on this
machine only, to start the M1 highlight analysis and look at its results. It adds no
analysis of its own: every number it shows comes from the run's files, with the display
rules of the report ([m1-highlights.md](m1-highlights.md), ADR 0004). The analysis is done
on the original frames; values computed from the 3D lift (`gbf.lift3d`) are labelled as a
reference and their results are drawn back onto those frames.

Code: `src/gbf/web/` (`server.py` the HTTP server and the command, `runs.py` what is shown
of a run and which files may be served, `jobs.py` analyses started from the page,
`static/` the page). Standard library only (`http.server.ThreadingHTTPServer`); the page is
plain HTML, CSS and JavaScript shipped in the package.

## Running it

```
python -m gbf web [--runs-dir DIR] [--port N] [--no-open]
```

- `--runs-dir`: the folder whose subfolders are runs (a run is a subfolder with a
  `run.json`). Default `$DATA_ROOT/runs` when `DATA_ROOT` is set (read as `gbf.models`
  reads it); otherwise the option is required. The folder must exist and must not be a
  network path (`\\server\share`, `//server/share`, or the two separators mixed, or the NT
  prefix `\??\` as in `\??\UNC\server\share`; a mapped network drive is not detected).
- `--port`: port on 127.0.0.1; 0 (the default) lets the system pick a free one.
- `--no-open`: print the address only; otherwise the default browser opens it.

The command prints the address, for example `http://127.0.0.1:53124/#t=<token>`, where the
token is drawn for this start. Ctrl+C stops the server and a running analysis with it.
Exit codes: 0 after Ctrl+C, 2 for refused arguments, 1 when the port cannot be bound.

## What the page shows

- **Runs**: name, creation time, status, video, highlight count and restriction tags of
  every run, newest first. Hidden folders, folders without `run.json` and links to folders
  outside the runs folder are not runs, and neither is a name that resolves to a hidden
  folder (a link, or a Windows 8.3 short name such as `HIDDEN~1` for `.hidden-run`).
- **A run**: the video facts (file, frames, fps, size, audio source, team names, wall
  time), the restriction tags and, when they include `no_share` or `nda`, the banner the
  renders carry (`gbf.pipeline.WATERMARK`); a failed run's error list; the run's notes;
  the timeline; the comparison with labels (only when the run itself used `--labels`,
  as in the report), headed 与标注对比, saying where the labels come from only in the label
  file's own words (`source_zh` of `label_eval.json`); one card per highlight; the list of
  files that can be opened; a link to the full `report.html`.
- **A highlight card**, named H1, H2, ... in peak order as in the report: title, type,
  main or candidate, score, window and key frame (frames and `mm:ss.hh` clocks, the
  hundredths of `gbf.render.format_clock`), the key frame image and the data card, the
  clip (`clips/<id>.mp4`: normal speed, slow motion, freeze-frame card, with sound), the
  summary, the key moments as a table, the touch counts, and the measurement records.
- **Measurements** are shaped by `gbf.report.metric_row` over
  `gbf.pipeline.display_record`, as in the report: value text, unit, tier and status
  badges (未验证 while the error is unmeasured, 仅画面平面 for image-plane values, 实验 for
  tier B, 不可用 with the reason for a missing value). Values from the 3D lift, the
  metrics in `gbf.web.runs.LIFT3D_METRICS` (the registry's `A_candidate` tier), also get
  **3D 参考**. C-tier records are never shown, only counted.
- **Playing a frame**: every key moment, every measurement with a frame and every key frame
  has a ▶ button that plays `clips/full.mp4`, the whole clip annotated frame by frame, from
  that frame in a player docked at the bottom right. The player is at most 400 px wide and
  folds to its title bar; while it is open the page ends with room for it, so the last rows
  can be scrolled clear of it. `full.mp4` runs 1:1 with the source, so frame f starts at
  f / fps; the time is read through the normal-speed segments of `clips/full.render.json`
  when it is there, and the player seeks to the middle of the frame, (f + 0.5) / fps, so
  that rounding never shows the frame before. A frame the clip does not hold has no button.

## Starting an analysis

The default form selects a video file (streamed only to this machine), or accepts an
absolute local video path. It submits `mode: "auto"`; tracking/calibration/ball predictions
run in an isolated local runtime selected by the server's `GBF_AUTO_CONFIG` JSON file.
The short CPU profile accepts at most 60 seconds and uploads at most 512 MiB. Missing
runtime configuration disables automatic submission with a reason; manual import remains
available under **高级选项**. Existing runs remain readable without the prediction runtime.

Automatic jobs add six real preparation/prediction stages ahead of the existing thirteen.
Progress is stage state and actual logs; no estimated percentage of model work is invented.
**停止分析** calls `POST /api/jobs/<id>/cancel` and keeps partial files. On Windows a
kill-on-close job object owns the process tree and caps its committed memory at 6000 MiB;
on POSIX the job runs in a new process group. Server exit or coordinator exit cleans up
Windows workers too. Restarted unfinished job records are marked unknown, never successful.

The trusted runtime config has `command` (a list beginning with an absolute executable),
`models_dir` (local pose weights), `allow_licenses` (SPDX identifiers) and `max_duration_s`
(1..60). The command receives `--input <manifest.json> --output <fresh-directory>`.
An optional `required_files` list names absolute local runtime/model files; missing files
disable automatic submission before a job is started.
The manifest contains canonical PNG paths and per-frame hashes. Its output includes
the normal plugin/ball receipts and `frame-verification.json` with `passed: true` and
the exact frame count. `automatic.json` in the result records source/conversion/provenance
and total elapsed time. Prediction implementation, dependencies and weights stay outside
the core package. No browser request can change this command.

The following describes the advanced **manual import** request (the original API).

The form sends `POST /api/jobs` with `video`, `sngs_dir`, `ball` and `out_name` (required),
`audio_source` (default: the video), `labels`, `team_names` (`{"left", "right"}`) and
`allow_licenses` (a list of SPDX identifiers). Before anything runs:

- every path must be a complete local path, without control characters, to an existing
  file or folder of the right kind; quotes from Windows' "copy as path" are removed;
- a path that starts with two separators, in any mix of `\` and `/` (`\\host\share`,
  `//host/share`, `/\host\share`, `\\?\C:\...`, `\\.\pipe\...`), or with the NT
  object-manager prefix `\??\` (`\??\UNC\host\share`, and `/??/`, which pathlib turns into
  it), is what Windows reads as a network or device path: it is refused before the file
  system is asked anything (a check of `\\host\share` would already contact that
  machine). A mapped network drive, or a link on a local disk that points to a share, is
  not detected;
- `out_name` is a new folder of `[A-Za-z0-9._-]` (at most 80 characters, no leading or
  trailing dot, no Windows device name) in the runs folder: an existing folder is refused,
  nothing is overwritten, and a folder inside a git work tree is refused as the pipeline
  would refuse it;
- team names may not contain a comma; licence identifiers must be single SPDX identifiers;
- unknown fields are refused; every problem is listed in Chinese.

One analysis runs at a time; a second request is refused (409) with a clear message. The
job is `python -P -m gbf highlights --video=... --out=<runs dir>/<out_name> ...` with the
server's interpreter and the same gbf source on `PYTHONPATH` (`-P`: the working directory
never shadows a module). Its output goes to `<runs dir>/.gbf-web/jobs/<id>.log` and a record
of the request and exit code to `<id>.json` beside it, so a later start of the server lists
earlier jobs and fills the form with the last request. The progress view polls
`GET /api/jobs/<id>`: the 13 stages with their state, read from the pipeline's
`[stage] running/done/skipped/kept` lines, and the last 40 lines of the log.

Exit codes are those of `gbf.pipeline.main`: 0 done; 2 input refused (nothing ran; for
example a licence outside the permissive allowlist, which names the identifier to accept
under 接受的许可证); 1 a stage failed (the message names the stage); Ctrl+C, a signal or the
server stopping the job is reported as interrupted. Known problems are translated
(`gbf.web.jobs.translate_problem`); others are shown as written. The stop button stops
one job; Ctrl+C in the server's console stops both the server and the analysis.

## Security

Any web page open in the same browser can send requests to 127.0.0.1, so the server
treats every request as possibly hostile:

- It listens on 127.0.0.1 only (no option to change that); on Windows the port is bound
  exclusively, so a second server cannot share it.
- A random token is drawn at every start (`secrets.token_urlsafe(32)`). The printed address
  carries it in the fragment (`#t=...`), which browsers never send to a server; the page's
  script reads it and sends it in the `X-GBF-Token` header with every API call. Images,
  videos and the report's relative links cannot carry a header, so `GET /api/session` (with
  the header) also sets an HttpOnly, `SameSite=Strict` session cookie for `/runs/`; a file
  of a run needs the header or that cookie, the API always the header. Tokens are compared
  in constant time. Cookies are scoped by host, not port: another server on 127.0.0.1 would
  receive the cookie for its own `/runs/` paths; the token only opens this server, and only
  while it runs.
- `Host` must be `127.0.0.1:<port>` or `localhost:<port>` (DNS rebinding). A request with
  any other `Origin` (`null` included) is refused, and so is a request to the API or to a
  run's files that the browser marks `Sec-Fetch-Site: cross-site` or `same-site`.
- JSON `POST` bodies must be `application/json` in UTF-8, at most 64 KiB, with a length;
  a body that is not valid JSON, or nests too deeply to parse, gets 400. The authenticated
  `/api/uploads` endpoint instead accepts `application/octet-stream` with one explicit
  length, at most 512 MiB, and streams it to a server-named local file. No CORS header is
  ever sent and `OPTIONS` is refused, so a script of another origin can neither read an
  answer nor send either accepted content type.
- A refused request ends its connection. Its body, when it has one `Content-Length` of at
  most 64 KiB and no `Transfer-Encoding`, is read and dropped before the answer: a
  connection closed with a body unread is reset, and the browser can lose the answer.
  A known oversized body gets the answer and a write-half close immediately; only then
  up to 128 KiB is discarded with a 0.5-second timeout to reduce Windows connection
  resets. Chunked bodies and bodies without a length are not read.
- Files are served only from inside a run folder: each decoded path segment is checked
  first (no empty, `.` or `..` segment, separator, colon, wildcard or control character,
  leading dot, trailing dot or space, or Windows device name), then the path is resolved
  and must stay inside the resolved run folder (a symbolic link or junction that leaves it
  is refused; so is a run folder that is itself a link to a folder elsewhere), and the
  names it resolves to must pass the same segment rules (a link, or a Windows 8.3 short
  name such as `HIDDEN~1.PNG` for `.hidden.png`, can stand for a hidden name). Only
  `.html`, `.png`, `.jpg`, `.mp4`, `.json`, `.jsonl`, `.css`, `.js` and `.svg` files are
  served, with their content types (`.jsonl` as plain text, so it opens in the browser
  instead of leaving a copy in the downloads folder), with Range support (206 Partial
  Content, 416 for a range past the end; a header the server does not read, such as several
  ranges or a number of more than 18 digits, is ignored and the whole file sent), and never
  a directory listing.
- Hard links are not detected: a hard link is a second name of the same file, so one inside
  a run folder is served like the run's own files even when the file's other name lies
  elsewhere. Path rules cannot tell the two apart; making one takes write access to the
  runs folder.
- Every answer carries `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
  `Cache-Control: no-store`, `Cross-Origin-Resource-Policy: same-origin`,
  `Cross-Origin-Opener-Policy: same-origin`, `X-Frame-Options: DENY` and a
  Content-Security-Policy: the page may load only its own script and style; a run's files
  may not run any script.
- Requests are checked before the file system is touched, and the names a run's path
  resolves to once more after; the reasons of refusals go to the console, never into an
  answer, and requests are otherwise not logged.

## Private material

Nothing leaves the machine through the page or the server: the page makes no external
request and loads no external font or script (the tests check the static files), the
server calls no online service and no language model, and it refuses network paths (a
mapped network drive, or a link to a share, set up by the owner is not detected).
Restriction tags and the private-material banner are rendered from API
data at run time: the static page contains neither, so the gate's page rules
(`src/gbf/gate/datafiles.py`) pass on it, and the job and run files the server writes stay
in the runs folder, outside the repository.

## What it does not do

- No access from another machine, no users or passwords, no HTTPS (loopback only).
- No editing, renaming or deleting of runs. Jobs can be cancelled; partial evidence remains.
- No hiding of unverified values: it is a private viewer. Public builds that hide them
  (ADR 0004) are not implemented.
- No analysis of its own and no language model.

## Where a language model could help later

A model could turn a highlight's measured facts into spoken-style commentary, or explain a
failed run's log in plain Chinese. It would only rephrase what the evidence records say:
programs copy the numbers, every number keeps its tier and its 未验证 or 3D 参考 badge, and
nothing is added that was not measured. This needs an API key from the owner, and it sends
data to a third party, which the run's restriction tags (`no_share`, `nda`) forbid for
broadcast material unless the owner decides otherwise per run. None of this is built.

## Tests

`tests/test_web.py` writes a synthetic runs folder (JSON documents, a tiny mp4 from
`gbf.synth.video`, a PNG), starts the server on a free port in a thread and checks the run
list and run page, the display rules, the token, foreign `Host`, `Origin` and
`Sec-Fetch-Site`, path traversal (`..`, `%2e%2e`, encoded separators, absolute paths, drive
letters, NUL, device names), links and junctions out of the run, names that resolve to
hidden ones (links, and 8.3 short names where the volume makes them), unknown suffixes and
directories, Range answers (absurd numbers included), the security headers of every kind
of answer (written out in the test), refused requests whose body comes late (each must
get its answer, on one client connection), the end of the connection after a refusal,
chunked bodies, deeply nested JSON, the request checks (network paths with mixed
separators and control characters, without asking the file system), one job at a time,
progress, exit codes and their messages (with a fake command in place of
`python -m gbf highlights`), stopping a job, distinct column names in the page's tables,
the pieces of the docked player, and that the static files pass the gate and load nothing
from outside. The page's own behaviour in a browser (the player folding, the room below
the page, a new view starting at the top) is not run by the suite; it was checked once in
a browser, on synthetic data.
