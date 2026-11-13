"""Checks over the registries and the public receipts.

registries:

- every ``registries/<kind>.toml`` must be UTF-8 TOML that validates against schema
  ``urn:gbf:schema:registry_<kind>`` (TOML dates and times are validated as ISO 8601
  strings). A registry without its schema fails as ``schema-missing``;
- every kind listed under ``required`` in ``gate/registries.json`` must exist;
- every registry entry that names a candidate file of the repository together with a
  sha256 is re-hashed over the file's raw bytes. An entry names a file through one of
  the keys ``path``, ``relpath``, ``file`` or ``repo_path``, or through the key of the
  table that holds its ``sha256``. Entries naming files outside the repository (for
  example under DATA_ROOT) are not re-hashed here.

receipts: every ``receipts/public/**/*.json`` must be UTF-8 JSON without NaN or
Infinity and validate against ``urn:gbf:schema:receipt``.

Schema problems print the JSON Schema keyword and a JSON pointer whose parts are masked
as ``?`` unless they are short identifiers; instance values are never printed.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from gbf import contracts
from gbf.gate.model import CheckResult
from gbf.gate.repo import Workspace, to_json_compatible, toml_error_line

REGISTRY_DIR = "registries"
REQUIRED_REGISTRIES_FILE = "gate/registries.json"
RECEIPTS_DIR = "receipts/public/"
PATH_KEYS = ("path", "relpath", "file", "repo_path")

_KIND = re.compile(r"[a-z][a-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_POINTER_PART = re.compile(r"[A-Za-z0-9_.-]{1,64}")


@dataclass(frozen=True)
class RegistryEntry:
    """A registry table that names a file. ``path`` is POSIX with any leading ./ removed;
    ``sha256`` is the registered value as written (None if absent); ``pointer`` is the
    entry's JSON pointer inside the registry."""

    path: str
    sha256: str | None
    restrictions: tuple[str, ...]
    pointer: str


def normalize_registered_path(value: str) -> str:
    path = value.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def registry_entries(data: Any) -> list[RegistryEntry]:
    """Every table in ``data`` that names a file (see the module docstring)."""
    entries: list[RegistryEntry] = []

    def walk(node: Any, pointer: str, key: str | None) -> None:
        if isinstance(node, dict):
            named = next((node[k] for k in PATH_KEYS if isinstance(node.get(k), str)), None)
            if named is None and "sha256" in node and key is not None:
                named = key
            if named is not None:
                sha = node.get("sha256")
                tags = node.get("restrictions")
                entries.append(
                    RegistryEntry(
                        path=normalize_registered_path(named),
                        sha256=sha if isinstance(sha, str) else None,
                        restrictions=(
                            tuple(t for t in tags if isinstance(t, str))
                            if isinstance(tags, list)
                            else ()
                        ),
                        pointer=pointer or "/",
                    )
                )
            for child_key, child in node.items():
                walk(child, f"{pointer}/{child_key}", str(child_key))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{pointer}/{index}", None)

    walk(data, "", None)
    return entries


def _safe_part(part: object) -> str:
    text = str(part)
    return text if _SAFE_POINTER_PART.fullmatch(text) else "?"


def _pointer(parts: Iterable[object]) -> str:
    return "/" + "/".join(_safe_part(p) for p in parts)


def schema_problems(instance: Any, schema_name: str) -> list[tuple[str, str]]:
    """Sorted, de-duplicated (keyword, pointer) pairs for every schema violation.

    For ``required`` the pointer names the missing property (a name from the schema).
    """
    found: set[tuple[str, str]] = set()
    for error in contracts.validator(schema_name).iter_errors(instance):
        parts = list(error.absolute_path)
        keyword = str(error.validator)
        if (
            keyword == "required"
            and isinstance(error.instance, dict)
            and isinstance(error.validator_value, list)
        ):
            for name in error.validator_value:
                if isinstance(name, str) and name not in error.instance:
                    found.add((keyword, _pointer([*parts, name])))
        else:
            found.add((keyword, _pointer(parts)))
    return sorted(found)


def is_registry_file(rel: str) -> bool:
    parts = rel.split("/")
    return len(parts) == 2 and parts[0] == REGISTRY_DIR and parts[1].endswith(".toml")


def load_registry(ws: Workspace, rel: str, result: CheckResult) -> dict[str, Any] | None:
    """Parse one registry; add a problem and return None if it cannot be read or parsed."""
    try:
        raw = ws.read(rel)
    except OSError as error:
        result.add(rel, "unreadable", detail=type(error).__name__)
        return None
    if raw is None:
        result.add(rel, "missing")
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        result.add(rel, "not-utf8")
        return None
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        result.add(rel, "toml-parse-error", line=toml_error_line(error))
        return None


def _required_kinds(ws: Workspace, result: CheckResult) -> set[str]:
    rel = REQUIRED_REGISTRIES_FILE
    if rel not in ws.file_set:
        return set()
    try:
        raw = ws.read(rel)
        config = json.loads(raw.decode("utf-8")) if raw is not None else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        result.add(rel, "invalid", detail=type(error).__name__)
        return set()
    issues: list[str] = []
    if not isinstance(config, dict):
        result.add(rel, "invalid", detail="top level must be an object")
        return set()
    if config.get("schema_version") != 1:
        issues.append("schema_version must be 1")
    required = config.get("required")
    kinds: set[str] = set()
    if not isinstance(required, list):
        issues.append("required must be a list of registry kinds")
    else:
        for item in required:
            if isinstance(item, str) and _KIND.fullmatch(item):
                kinds.add(item)
            else:
                issues.append("required holds an invalid kind")
        if len(kinds) != len(required):
            issues.append("required must list each kind once")
    for issue in sorted(set(issues)):
        result.add(rel, "invalid", detail=issue)
    return kinds


def check_registries(root: Path, workspace: Workspace | None = None) -> CheckResult:
    """Validate every registry and re-hash every repository file it registers."""
    ws = workspace if workspace is not None else Workspace(root)
    result = CheckResult("registries")
    registry_files = [rel for rel in ws.files if is_registry_file(rel)]
    present = {PurePosixPath(rel).stem for rel in registry_files}
    for kind in sorted(_required_kinds(ws, result) - present):
        result.add(f"{REGISTRY_DIR}/{kind}.toml", "missing")
    known_schemas = set(contracts.schema_names())
    for rel in registry_files:
        kind = PurePosixPath(rel).stem
        if not _KIND.fullmatch(kind):
            result.add(rel, "invalid-kind-name")
            continue
        data = load_registry(ws, rel, result)
        if data is None:
            continue
        result.scanned += 1
        schema = f"registry_{kind}"
        if schema not in known_schemas:
            result.add(rel, "schema-missing", detail=contracts.ID_PREFIX + schema)
        else:
            for keyword, pointer in schema_problems(to_json_compatible(data), schema):
                result.add(rel, f"schema-{keyword}", detail=pointer)
        for entry in registry_entries(data):
            if entry.sha256 is None or entry.path not in ws.file_set:
                continue
            where = f"registered in {rel}"
            if not _SHA256.fullmatch(entry.sha256.lower()):
                result.add(entry.path, "sha256-malformed", detail=where)
                continue
            try:
                actual = ws.sha256(entry.path)
            except OSError as error:
                result.add(entry.path, "unreadable", detail=type(error).__name__)
                continue
            if actual != entry.sha256.lower():
                result.add(entry.path, "sha256-mismatch", detail=where)
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(name)


def check_receipts(root: Path, workspace: Workspace | None = None) -> CheckResult:
    """Validate every public receipt against the receipt schema."""
    ws = workspace if workspace is not None else Workspace(root)
    result = CheckResult("receipts")
    receipt_files = [
        rel
        for rel in ws.files
        if rel.startswith(RECEIPTS_DIR) and PurePosixPath(rel).suffix.lower() == ".json"
    ]
    if receipt_files and "receipt" not in contracts.schema_names():
        for rel in receipt_files:
            result.add(rel, "schema-missing", detail=contracts.ID_PREFIX + "receipt")
        return result
    for rel in receipt_files:
        try:
            raw = ws.read(rel)
        except OSError as error:
            result.add(rel, "unreadable", detail=type(error).__name__)
            continue
        if raw is None:
            continue
        result.scanned += 1
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            result.add(rel, "not-utf8")
            continue
        try:
            instance = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError as error:
            result.add(rel, "json-parse-error", line=error.lineno)
            continue
        except ValueError:
            result.add(rel, "json-nonstandard-number")
            continue
        for keyword, pointer in schema_problems(instance, "receipt"):
            result.add(rel, f"schema-{keyword}", detail=pointer)
    return result
