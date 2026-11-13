"""JSON Schema contracts for every artifact the pipeline writes or accepts.

Schemas live next to this module in ``schemas/<name>.schema.json`` and use ids
of the form ``urn:gbf:schema:<name>`` so they can reference each other.
Validation reports every problem at once instead of stopping at the first one.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SCHEMA_DIR = Path(__file__).parent / "schemas"
ID_PREFIX = "urn:gbf:schema:"


class ContractError(ValueError):
    """An artifact does not satisfy its schema. ``problems`` lists every violation."""

    def __init__(self, schema_name: str, problems: list[str]):
        self.schema_name = schema_name
        self.problems = problems
        joined = "\n  - ".join(problems)
        super().__init__(f"{schema_name}: {len(problems)} problem(s)\n  - {joined}")


def schema_names() -> list[str]:
    return sorted(p.name.removesuffix(".schema.json") for p in SCHEMA_DIR.glob("*.schema.json"))


@cache
def _registry() -> Registry:
    resources = []
    for name in schema_names():
        with open(SCHEMA_DIR / f"{name}.schema.json", "rb") as fh:
            contents = json.loads(fh.read().decode("utf-8"))
        expected_id = ID_PREFIX + name
        if contents.get("$id") != expected_id:
            raise ContractError(name, [f"$id must be {expected_id!r}, got {contents.get('$id')!r}"])
        resources.append((expected_id, Resource.from_contents(contents)))
    return Registry().with_resources(resources)


def load_schema(name: str) -> dict[str, Any]:
    resource = _registry().get_or_retrieve(ID_PREFIX + name).value
    return dict(resource.contents)


def validator(name: str) -> Draft202012Validator:
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema, registry=_registry(), format_checker=Draft202012Validator.FORMAT_CHECKER
    )


def problems(instance: Any, name: str) -> list[str]:
    found = []
    for error in validator(name).iter_errors(instance):
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        found.append(f"{location}: {error.message}")
    return sorted(found)


def validate(instance: Any, name: str) -> None:
    found = problems(instance, name)
    if found:
        raise ContractError(name, found)


def def_problems(instance: Any, name: str, definition: str) -> list[str]:
    """Problems of ``instance`` against ``$defs/<definition>`` of schema ``name``."""
    sub_schema = {"$ref": f"{ID_PREFIX}{name}#/$defs/{definition}"}
    checker = Draft202012Validator(
        sub_schema, registry=_registry(), format_checker=Draft202012Validator.FORMAT_CHECKER
    )
    found = []
    for error in checker.iter_errors(instance):
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        found.append(f"{location}: {error.message}")
    return sorted(found)
