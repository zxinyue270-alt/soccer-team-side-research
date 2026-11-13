"""Hashing helpers.

Every hash in this project is computed over raw bytes, never over decoded text,
so that line-ending conversion can never make two copies of a file disagree
silently (ADR 0003).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """Git's object id for a blob: sha1(b"blob <size>\\0" + data).

    Providers such as GitHub and Hugging Face publish this id for every file,
    which lets a download script verify a pinned file without trusting the
    transport.
    """
    header = b"blob " + str(len(data)).encode("ascii") + b"\x00"
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text used for content-addressed ids (e.g. evidenceId)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def canonical_sha256(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))
