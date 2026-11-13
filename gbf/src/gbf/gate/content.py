"""Checks over the text content of candidate files: eol, credentials, abs_paths.

A text file is a candidate file without a binary suffix and without a NUL byte in its
first 8000 bytes (git's rule). Files larger than TEXT_SCAN_LIMIT fail as
``too-large-to-scan``; unreadable files fail as ``unreadable``. No file is exempt: the
hash-registered fixtures under tests/fixtures are byte-exact and must be LF as well.
"""

from __future__ import annotations

from pathlib import Path

from gbf.gate.model import CheckResult
from gbf.gate.patterns import find_credentials, find_local_paths
from gbf.gate.repo import Workspace, basename

_DOTENV_TEMPLATE_SUFFIXES = frozenset({"example", "sample", "template", "dist"})
_KEY_FILE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx"})
_KEY_FILE_NAMES = frozenset({"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"})
_CREDENTIAL_FILE_NAMES = frozenset({".netrc", "_netrc", ".pgpass"})


def sensitive_file_rule(rel: str) -> str | None:
    """Rule id for a file whose name alone marks it as a credential store, else None."""
    name = basename(rel).lower()
    if name == "kaggle.json":
        return "kaggle-json-file"
    if name == ".env" or (
        name.startswith(".env.") and name.rsplit(".", 1)[-1] not in _DOTENV_TEMPLATE_SUFFIXES
    ):
        return "dotenv-file"
    if name in _KEY_FILE_NAMES or Path(name).suffix in _KEY_FILE_SUFFIXES:
        return "private-key-file"
    if name in _CREDENTIAL_FILE_NAMES:
        return "credentials-file"
    return None


def check_eol(root: Path, workspace: Workspace | None = None) -> CheckResult:
    """Fail every text file that contains a CRLF byte pair; report its first CRLF line."""
    ws = workspace if workspace is not None else Workspace(root)
    result = CheckResult("eol")
    for text_file in ws.text_files(result):
        data = text_file.data
        index = data.find(b"\r\n")
        if index < 0:
            continue
        count = data.count(b"\r\n")
        line = data.count(b"\n", 0, index) + 1
        result.add(text_file.rel, "crlf", line=line, detail=f"{count} CRLF line endings")
    return result


def check_credentials(root: Path, workspace: Workspace | None = None) -> CheckResult:
    """Fail credential-store file names and credential-shaped content (see patterns)."""
    ws = workspace if workspace is not None else Workspace(root)
    result = CheckResult("credentials")
    for rel in ws.files:
        rule = sensitive_file_rule(rel)
        if rule is not None:
            result.add(rel, rule)
    for text_file in ws.text_files(result):
        for finding in find_credentials(text_file.text, text_file.rel):
            result.add(text_file.rel, finding.rule, line=finding.line)
    return result


def check_abs_paths(root: Path, workspace: Workspace | None = None) -> CheckResult:
    """Fail text files that contain a user-home path or an absolute path through a gbf-data
    or private folder (see patterns)."""
    ws = workspace if workspace is not None else Workspace(root)
    result = CheckResult("abs_paths")
    for text_file in ws.text_files(result):
        for finding in find_local_paths(text_file.text):
            result.add(text_file.rel, finding.rule, line=finding.line)
    return result
