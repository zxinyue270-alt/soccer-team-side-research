# Data access tracker

Applications and agreements gbf depends on, the questions sent to data owners, and the
inventory of NDA derivatives. The policy lives in the registries
([`registries/data.toml`](../registries/data.toml),
[`registries/models.toml`](../registries/models.toml)) and in
[ADR 0002](adr/0002-data-and-media-policy.md); this file holds the dated history.

The maintainer submits every application and accepts every agreement personally.
Contributors and automated agents never apply, sign, accept terms or log in on anyone's
behalf, and never use another person's account or a proxy if access is refused. Notes on
terms below are engineering judgements, not legal advice.

## Rules for recording

- Dates are ISO 8601 calendar dates (`YYYY-MM-DD`).
- Record the date of the event itself (submitted, signed, approved, refused, answered,
  deleted), taken from the provider's confirmation (e-mail, web page, receipt) when there
  is one, otherwise the calendar date on which the maintainer acted. Never estimate or
  backdate; write `unknown` if the date was not noted at the time.
- Never overwrite history: update the status cell, then append a dated line to the log
  saying what changed.
- Say where the evidence is kept in words ("confirmation e-mail", "Hugging Face access
  page"). Never paste credentials, tokens, personal data (date of birth, address), account
  names or ids into this file.
- An approved NDA gets an expiry line: signature date plus the obligation term (2 years
  according to the research record; confirm against the signed text). The NDA-derivatives
  inventory is counted before P5 and before that date.
- Dates that must prove an ordering, such as gates frozen before a test-set run, come from
  the git server's push time, not from this file (ADR 0004).
- When an answer or approval changes a registry fact (for example SoccerNet allows cloud
  processing), update `registries/data.toml` in the same commit and set its `verified_on`.

Status values: `not submitted`, `submitted`, `approved`, `refused`, `withdrawn`,
`not applying`.

## D1 applications and agreements

| # | Item | Where | What it involves | Status | Submitted | Decided | Tags once granted | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | SoccerNet_raw_HQ NDA | Hugging Face dataset `SoccerNet/SoccerNet_raw_HQ` | Electronic NDA (counterparty KAUST), approved manually and separately per repository. The access form's terms: research and non-commercial use only, the videos are copyrighted, do not share them with anyone | not submitted | | | `nda` | Needed for the demo match video (224p and 720p, 25 fps). Approval time unknown. On arrival: ffprobe resolution, frame rate and frame count, then `field_order` and idet |
| 2 | SAM 3D Body weights | Hugging Face model `facebook/sam-3d-body-dinov3` | Manual approval. The form asks for name, date of birth, country, affiliation and job title, geolocates the IP address, and requires accepting the SAM License and Meta's privacy policy | not submitted | | | `sam_license` | Until approved, research mode runs RTMW3D or NLF-L. Mirrors only with the full SAM License text and matching sha256 (ADR 0001) |
| 3 | WorldPose ground truth and camera parameters | ETH Zurich AIT academic application (WorldPose dataset page) | Apply with an institutional (university) e-mail address and state the intended use truthfully: anonymous pose-accuracy evaluation only | not submitted | | | `research_nc`, `no_identity` | Terms: non-commercial academic use; no commercial model training; no redistribution, including converted model parameters; no identity recognition or linking; ETH may end access at any time. The terms actually signed prevail over the CC BY-NC-SA shown on the challenge page |
| 4 | FIFA broadcast video for WorldPose | FIFA, as the WorldPose reply instructs | FIFA Content T&Cs are sent only after applying; read them in full before signing | not submitted (waits for item 3) | | | `research_nc`, `no_identity` | |
| 5 | StatsBomb registration and user agreement | StatsBomb (open data hosted at `hudl/open-data`) | Register before using the data (clause 2.2) and accept the StatsBomb Public Data User Agreement; `python -m gbf fetch` also asks for explicit acceptance | not submitted | | | `no_share`, `non_commercial`, `statsbomb_logo` | The agreement file is pinned in `src/gbf/events/pins.toml` |
| 6 | E-mail to SoccerNet | SoccerNet team | Three questions: (1) may NDA videos be processed on cloud compute such as Kaggle; (2) may skeletons and coordinates derived from them be published; (3) may watermarked private demos be shown to specific people | not submitted (e-mail not sent) | | | | Until answered: local processing only, nothing derived is published, no demos |
| 7 | SoccerNet-GAR (optional, phase 2) | Hugging Face dataset `OpenSportsLab/SoccerNet-GAR` | SoccerNet NDA: non-commercial research, no sharing | not submitted (maintainer decides whether to apply) | | | `nda` | |
| 8 | PFF FC 2022 World Cup data (optional, phase 2) | PFF / Gradient Sports request form | Form plus the PFF Terms of Use (not yet read) | not submitted (maintainer decides whether to apply) | | | set from the terms | Treated as not redistributable until the terms are read |
| 9 | FIFA Skeletal Tracking Light 2026 | Hugging Face dataset and Codabench challenge | | not applying | | | | Inputs only (no video, no ground truth); terms 9.a transfer entrants' intellectual property to FIFA; 3D ground truth comes through WorldPose instead. Reasons in `registries/data.toml` |

## Log

Append one line per change, newest last.

| Date | Item | Change | Evidence kept in |
|---|---|---|---|

## NDA-derivatives inventory

Every file or cloud output derived from NDA video: detections, tracks, skeletons, crops,
renders, and logs that contain per-item values. Local files are listed by their path
relative to `DATA_ROOT`; cloud outputs by kernel name (without the account prefix) and
version. Deletions are recorded with their date. The inventory is counted before P5 and
before the NDA expires (ADR 0002).

| ID | Derivative | Source (dataset, game, canonical frame range) | Location | Created | Tags | Last counted | Deleted | Notes |
|---|---|---|---|---|---|---|---|---|

No entries yet: no NDA video has been obtained.
