"""Local web page for gbf (``python -m gbf web``, docs/design/web.md).

- :mod:`gbf.web.server`: the HTTP server on 127.0.0.1, its security rules and the command.
- :mod:`gbf.web.runs`: what the page shows of a run folder, and which of its files may be
  served.
- :mod:`gbf.web.jobs`: analyses started from the page, one ``python -m gbf highlights`` at a
  time.
- ``static/``: the page itself (HTML, CSS, JavaScript; no external font, script or request).
"""
