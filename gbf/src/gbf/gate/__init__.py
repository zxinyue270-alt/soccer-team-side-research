"""The offline quality gate (plan section 7, G1-G4, G16, G17).

``python -m gbf gate`` runs the fast static checks over the candidate files (tracked plus
untracked-not-ignored): eol, credentials, abs_paths, media, restricted_data,
forbidden_imports, deps, registries and receipts. ``--full`` adds ruff and the test
suite in an offline subprocess with a sanitised environment. Every check is a function
of the repository root, so tests can run it against a temporary repository.

Problems print as ``<check> <relative path>[:line] <rule id>``; the gate never prints the
matched text of a secret or a local path.
"""

from gbf.gate.model import CheckResult, GateError, Problem
from gbf.gate.static import STATIC_CHECK_NAMES, STATIC_CHECKS, run_static

__all__ = [
    "STATIC_CHECKS",
    "STATIC_CHECK_NAMES",
    "CheckResult",
    "GateError",
    "Problem",
    "run_static",
]
