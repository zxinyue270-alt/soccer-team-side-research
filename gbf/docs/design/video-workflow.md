# Video-only local analysis

Status: implementation contract, 2026-10-22.

## Objective and acceptance

The owner selects a video in the browser, or supplies an absolute local path, and starts
one job. No tracking, calibration or ball file is required in automatic mode. The page
shows preparation, model prediction and the existing thirteen analysis stages; success
links to the normal run page. Cancellation stops the process tree, records interruption
and keeps partial evidence. Errors never become successful or invented predictions.

All processing and uploads stay on localhost. The first CPU profile is for short clips
(at most 60 seconds, uploads at most 512 MiB); a 30-second clip can take hours. Scores,
tracking and measurements remain unvalidated against ground truth. No live-speed claim.

## Architecture and boundaries

`web/automatic.py` prepares canonical input through `gbf.ingest`, exports lossless PNG
frames with hashes, and launches a configured external prediction command. The command
receives a JSON manifest and a fresh output directory, writes the existing plugin
contract and ball receipt, then exits. Core imports no GPL model code and unpickles no
state. Existing licence/restriction validation remains in force. The installed local
runtime and weights live outside the core repository and run offline in their original
interpreter. Runtime configuration is a local JSON file selected by `GBF_AUTO_CONFIG`.
It contains the fixed command, model directory, accepted SPDX identifiers and limit.
Browser requests cannot choose a command or change the configuration.

Frames use zero-based CFR indices and are read only by gbf.ingest. The local prediction
profile uses 1920x1080 images: other sizes are scaled with aspect ratio preserved and
padded, and VFR is normalized to 25 fps. Input/source hashes and conversion settings are
recorded. Audio comes from the original input; missing audio remains missing. New job
work directories and result names never overwrite earlier ones. No cross-video cache.

`POST /api/uploads` streams a length-bounded binary body into a server-named file outside
Git; it requires the same token/Host/Origin checks as all APIs. Partial uploads are
removed on transfer failure. JSON APIs keep their 64 KiB limit. `POST /api/jobs` accepts
`mode=auto` with video and out_name; `mode=manual` keeps the old import form. Only one job
may run. `POST /api/jobs/<id>/cancel` stops its process tree without stopping the server.
Server shutdown also stops the tree. Restarted servers report incomplete jobs honestly.

## Implementation order

1. Automatic input preparation, external command orchestration and provenance.
2. Job request/mode, real stage progress, cancellation and upload HTTP endpoint.
3. Video-first form, local upload progress, advanced manual form and error states.
4. Install the local isolated prediction adapter and update the desktop launcher.
5. Synthetic HTTP/process/normalization regression tests, full offline gate, browser
   smoke and a fresh short real-video run through the actual automatic entry point.

## Commands and verification

From the core checkout: `.venv/Scripts/python.exe -m pytest tests/test_web.py
tests/test_automatic.py -q`, `.venv/Scripts/python.exe -m gbf gate --full`.
Start with `.venv/Scripts/python.exe -m gbf web --runs-dir <external-runs-dir>`.
Code follows existing typed Python/dataclasses and plain JS/CSS; no new web framework.
Tests use synthetic data and fake process commands. Real media and runtime paths never
enter the repository. Test failure, invalid video, missing runtime, upload rejection,
busy/duplicate names, nested child cancellation and result navigation are acceptance
checks. Preserve the existing manual workflow and all current report display rules.
