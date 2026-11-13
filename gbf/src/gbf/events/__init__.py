"""Provider events, SoccerNet labels and their alignment to the half videos.

- ``pins``: the pinned open-data sources in ``pins.toml`` and their validation.
- ``fetch``: ``python -m gbf fetch``, a downloader that verifies git blob ids and
  writes only under ``DATA_ROOT``, outside the repository.
- ``providers``: StatsBomb events loaded through kloppy from local bytes, and the
  SoccerNet ``Labels-v2.json`` / ``Labels-cameras.json`` parsers.
- ``align``: ``python -m gbf align``: per-half offsets, residuals, the kick-off
  check, the frame mapping, the main-camera share and the ``align.json`` receipt.
- ``direction``: attack-direction checks for tracking output.

Raw provider files and everything derived from them stay out of git (ADR 0002).
"""
