"""Pinned open-data sources (``pins.toml``).

A pin names one file at one immutable revision of a public repository, together
with the file's git blob id (sha1 of ``b"blob <size>\\0" + content``) and its size.
The downloader verifies both before writing anything, and ``align`` uses the blob
id to recognise which pinned source a local file is.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from gbf import contracts

PINS_PATH = Path(__file__).with_name("pins.toml")

HOSTS = ("github", "hf-dataset")
ROLES = (
    "statsbomb_events",
    "statsbomb_lineups",
    "soccernet_labels_v2",
    "soccernet_labels_cameras",
)
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
# Characters that would make a source path escape its repository or its local
# mirror directory, or that Windows cannot store in a file name.
_BAD_PATH_CHARS = set('\\:*?"<>|')


class PinsError(ValueError):
    """``pins.toml`` is malformed. ``problems`` lists every violation found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        joined = "\n  - ".join(problems)
        super().__init__(f"pins.toml: {len(problems)} problem(s)\n  - {joined}")


@dataclass(frozen=True)
class Terms:
    """Terms a user must accept before a pinned source may be downloaded."""

    key: str
    title: str
    agreement_url: str
    agreement_git_blob_sha1: str
    clauses: tuple[str, ...]


@dataclass(frozen=True)
class Pin:
    """One pinned file. ``bytes`` is the exact size in bytes."""

    name: str
    role: str
    host: str
    repository: str
    revision: str
    path: str
    git_blob_sha1: str
    bytes: int
    restrictions: tuple[str, ...]
    terms: str | None

    @property
    def url(self) -> str:
        """Download URL, derived so that it always names the pinned revision."""
        quoted = quote(self.path, safe="/")
        if self.host == "github":
            return f"https://raw.githubusercontent.com/{self.repository}/{self.revision}/{quoted}"
        return f"https://huggingface.co/datasets/{self.repository}/resolve/{self.revision}/{quoted}"

    @property
    def local_relpath(self) -> str:
        """POSIX path of the local copy, relative to the download root (``DATA_ROOT/raw``).

        The first 12 hex digits of the revision keep revisions apart while leaving
        room under the Windows path-length limit; the record written next to the
        file holds the full revision.
        """
        return f"{self.host}/{self.repository}/{self.revision[:12]}/{self.path}"


@dataclass(frozen=True)
class PinSet:
    pins: tuple[Pin, ...]
    terms: Mapping[str, Terms]

    def by_name(self, name: str) -> Pin:
        for pin in self.pins:
            if pin.name == name:
                return pin
        known = ", ".join(pin.name for pin in self.pins)
        raise KeyError(f"unknown pin {name!r}; known pins: {known}")

    def by_blob(self, git_blob_sha1: str) -> tuple[Pin, ...]:
        return tuple(pin for pin in self.pins if pin.git_blob_sha1 == git_blob_sha1)


def _path_problems(where: str, path: Any) -> list[str]:
    if not isinstance(path, str) or not path:
        return [f"{where}.path must be a non-empty string"]
    found = []
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        found.append(f"{where}.path must be relative: {path!r}")
    if any(part in {"", ".", ".."} for part in path.split("/")):
        found.append(f"{where}.path has an empty, '.' or '..' segment: {path!r}")
    if _BAD_PATH_CHARS & set(path) or any(ord(ch) < 32 for ch in path):
        found.append(f"{where}.path has a character that is unsafe in a file name: {path!r}")
    return found


def _terms_problems(key: str, raw: Any) -> list[str]:
    where = f"terms.{key}"
    if not isinstance(raw, Mapping):
        return [f"{where} must be a table"]
    found = []
    for field in ("title", "agreement_url"):
        if not isinstance(raw.get(field), str) or not raw.get(field):
            found.append(f"{where}.{field} must be a non-empty string")
    if isinstance(raw.get("agreement_url"), str) and not raw["agreement_url"].startswith(
        "https://"
    ):
        found.append(f"{where}.agreement_url must be an https URL")
    if not isinstance(raw.get("agreement_git_blob_sha1"), str) or not _SHA1.match(
        raw["agreement_git_blob_sha1"]
    ):
        found.append(f"{where}.agreement_git_blob_sha1 must be 40 lowercase hex digits")
    clauses = raw.get("clauses")
    if (
        not isinstance(clauses, list)
        or not clauses
        or not all(isinstance(c, str) and c for c in clauses)
    ):
        found.append(f"{where}.clauses must be a non-empty list of strings")
    unknown = sorted(set(raw) - {"title", "agreement_url", "agreement_git_blob_sha1", "clauses"})
    if unknown:
        found.append(f"{where} has unknown keys: {unknown}")
    return found


def _source_problems(index: int, raw: Any, terms_keys: set[str]) -> list[str]:
    where = f"sources[{index}]"
    if not isinstance(raw, Mapping):
        return [f"{where} must be a table"]
    found = []
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        found.append(f"{where}.name must match {_NAME.pattern}: {name!r}")
    if raw.get("role") not in ROLES:
        found.append(f"{where}.role must be one of {list(ROLES)}: {raw.get('role')!r}")
    if raw.get("host") not in HOSTS:
        found.append(f"{where}.host must be one of {list(HOSTS)}: {raw.get('host')!r}")
    repository = raw.get("repository")
    if not isinstance(repository, str) or not _REPOSITORY.match(repository):
        found.append(f"{where}.repository must look like owner/name: {repository!r}")
    for field in ("revision", "git_blob_sha1"):
        value = raw.get(field)
        if not isinstance(value, str) or not _SHA1.match(value):
            found.append(f"{where}.{field} must be 40 lowercase hex digits: {value!r}")
    found.extend(_path_problems(where, raw.get("path")))
    size = raw.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        found.append(f"{where}.bytes must be a positive integer: {size!r}")
    restrictions = raw.get("restrictions")
    if not isinstance(restrictions, list):
        found.append(f"{where}.restrictions must be a list")
    else:
        for problem in contracts.def_problems(restrictions, "restrictions", "restrictions"):
            found.append(f"{where}.restrictions: {problem}")
    terms = raw.get("terms")
    if terms is not None and terms not in terms_keys:
        found.append(f"{where}.terms names unknown terms {terms!r}")
    allowed = {
        "name",
        "role",
        "host",
        "repository",
        "revision",
        "path",
        "git_blob_sha1",
        "bytes",
        "restrictions",
        "terms",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        found.append(f"{where} has unknown keys: {unknown}")
    return found


def parse_pins(data: Mapping[str, Any]) -> PinSet:
    """Validate parsed ``pins.toml`` data and build a :class:`PinSet`.

    Raises :class:`PinsError` listing every problem found.
    """
    problems: list[str] = []
    if data.get("schema_version") != 1:
        problems.append(f"schema_version must be 1, got {data.get('schema_version')!r}")
    raw_terms = data.get("terms", {})
    if not isinstance(raw_terms, Mapping):
        problems.append("terms must be a table")
        raw_terms = {}
    for key in sorted(raw_terms):
        problems.extend(_terms_problems(key, raw_terms[key]))
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        problems.append("sources must be a non-empty array of tables")
        raw_sources = []
    for index, raw in enumerate(raw_sources):
        problems.extend(_source_problems(index, raw, set(raw_terms)))
    names = [raw.get("name") for raw in raw_sources if isinstance(raw, Mapping)]
    duplicates = sorted({str(n) for n in names if names.count(n) > 1})
    if duplicates:
        problems.append(f"duplicate source names: {duplicates}")
    unknown = sorted(set(data) - {"schema_version", "terms", "sources"})
    if unknown:
        problems.append(f"unknown top-level keys: {unknown}")
    if problems:
        raise PinsError(problems)

    terms = {
        key: Terms(
            key=key,
            title=raw["title"],
            agreement_url=raw["agreement_url"],
            agreement_git_blob_sha1=raw["agreement_git_blob_sha1"],
            clauses=tuple(raw["clauses"]),
        )
        for key, raw in raw_terms.items()
    }
    pins = tuple(
        Pin(
            name=raw["name"],
            role=raw["role"],
            host=raw["host"],
            repository=raw["repository"],
            revision=raw["revision"],
            path=raw["path"],
            git_blob_sha1=raw["git_blob_sha1"],
            bytes=raw["bytes"],
            restrictions=tuple(sorted(raw["restrictions"])),
            terms=raw.get("terms"),
        )
        for raw in raw_sources
    )
    return PinSet(pins=pins, terms=terms)


def load_pins(path: str | Path = PINS_PATH) -> PinSet:
    """Read and validate a pins file (the packaged ``pins.toml`` by default)."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return parse_pins(data)
