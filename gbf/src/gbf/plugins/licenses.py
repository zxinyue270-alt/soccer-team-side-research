"""SPDX licence expressions and the core's permissive allowlist (ADR 0001 section 4).

A plugin declares its licence as an SPDX expression such as ``MIT``,
``MIT OR Apache-2.0`` or ``GPL-3.0-only WITH Classpath-exception-2.0``. The core accepts it
when the expression is satisfied by the allowed identifiers: PERMISSIVE_LICENSES plus
whatever the user named explicitly (``--allow-license``). Rules:

- an identifier is allowed when it is in the allowed set (compared case-insensitively, as
  SPDX identifiers are); ``GPL-2.0+`` and other deprecated or ``LicenseRef-`` forms count
  only when named exactly; ``NONE`` and ``NOASSERTION`` never pass implicitly;
- ``A OR B`` needs one allowed side, ``A AND B`` needs both;
- ``L WITH E`` is allowed when ``L`` is: an SPDX exception only adds permissions;
- operators are upper case (``AND``, ``OR``, ``WITH``); ``WITH`` binds tighter than ``AND``,
  which binds tighter than ``OR``; parentheses group.

These are engineering rules, not legal advice.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

PERMISSIVE_LICENSES: tuple[str, ...] = (
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "ISC",
    "MIT",
    "PSF-2.0",
    "Unlicense",
    "Zlib",
)
"""Licences the core accepts without an explicit opt-in."""

_ID = re.compile(r"^(?:DocumentRef-[A-Za-z0-9.-]+:)?[A-Za-z0-9][A-Za-z0-9.-]*\+?$")
_TOKEN = re.compile(r"\(|\)|[^\s()]+")
_OPERATORS = {"AND", "OR", "WITH"}


class LicenseExpressionError(ValueError):
    """A licence expression or an allowlist entry is not valid SPDX syntax."""


@dataclass(frozen=True)
class LicenseId:
    name: str


@dataclass(frozen=True)
class WithException:
    license: LicenseId
    exception: str


@dataclass(frozen=True)
class AllOf:
    parts: tuple[LicenseNode, ...]


@dataclass(frozen=True)
class AnyOf:
    parts: tuple[LicenseNode, ...]


LicenseNode = LicenseId | WithException | AllOf | AnyOf


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _TOKEN.findall(text)
        self.position = 0

    def fail(self, message: str) -> LicenseExpressionError:
        return LicenseExpressionError(f"license expression {self.text!r}: {message}")

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise self.fail("ends too early")
        self.position += 1
        return token

    def parse(self) -> LicenseNode:
        if not self.tokens:
            raise self.fail("is empty")
        node = self.expression()
        if self.peek() is not None:
            raise self.fail(f"unexpected {self.peek()!r}")
        return node

    def expression(self) -> LicenseNode:
        parts = [self.conjunction()]
        while self.peek() == "OR":
            self.take()
            parts.append(self.conjunction())
        return parts[0] if len(parts) == 1 else AnyOf(tuple(parts))

    def conjunction(self) -> LicenseNode:
        parts = [self.with_expression()]
        while self.peek() == "AND":
            self.take()
            parts.append(self.with_expression())
        return parts[0] if len(parts) == 1 else AllOf(tuple(parts))

    def with_expression(self) -> LicenseNode:
        node = self.atom()
        if self.peek() == "WITH":
            self.take()
            exception = self.take()
            if exception in _OPERATORS or not _ID.match(exception):
                raise self.fail(f"{exception!r} is not an exception identifier")
            if not isinstance(node, LicenseId):
                raise self.fail("WITH must follow a single licence identifier")
            return WithException(node, exception)
        return node

    def atom(self) -> LicenseNode:
        token = self.take()
        if token == "(":
            node = self.expression()
            if self.take() != ")":
                raise self.fail("missing ')'")
            return node
        if token in _OPERATORS or token == ")":
            raise self.fail(f"unexpected {token!r}")
        if token.lower() in {"and", "or", "with"}:
            raise self.fail(f"operator {token!r} must be upper case")
        if not _ID.match(token):
            raise self.fail(f"{token!r} is not a licence identifier")
        return LicenseId(token)


def parse_license_expression(text: str) -> LicenseNode:
    """Parse an SPDX licence expression; raises LicenseExpressionError on bad syntax."""
    if not isinstance(text, str):
        raise LicenseExpressionError(f"license expression must be a string, got {text!r}")
    return _Parser(text).parse()


def license_ids(node: LicenseNode) -> list[str]:
    """Licence identifiers in ``node`` (exceptions excluded), in order of appearance."""
    if isinstance(node, LicenseId):
        return [node.name]
    if isinstance(node, WithException):
        return [node.license.name]
    return [name for part in node.parts for name in license_ids(part)]


def _satisfied(node: LicenseNode, allowed: frozenset[str]) -> bool:
    if isinstance(node, LicenseId):
        return node.name.casefold() in allowed
    if isinstance(node, WithException):
        return _satisfied(node.license, allowed)
    if isinstance(node, AllOf):
        return all(_satisfied(part, allowed) for part in node.parts)
    return any(_satisfied(part, allowed) for part in node.parts)


def checked_allowlist(allow_licenses: Iterable[str]) -> list[str]:
    """The explicit opt-ins as a sorted list of unique identifiers.

    Each entry must be one SPDX identifier (no operators, no spaces); raises
    LicenseExpressionError naming every invalid entry.
    """
    if isinstance(allow_licenses, str):
        raise LicenseExpressionError(
            f"allow_licenses must be a list of identifiers, not the string {allow_licenses!r}"
        )
    entries = list(allow_licenses)
    bad = [entry for entry in entries if not isinstance(entry, str) or not _ID.match(entry)]
    if bad:
        raise LicenseExpressionError(
            "allow_licenses entries must be single SPDX identifiers: "
            + ", ".join(repr(entry) for entry in bad)
        )
    return sorted(set(entries), key=str.casefold)


def effective_allowlist(allow_licenses: Iterable[str] = ()) -> list[str]:
    """PERMISSIVE_LICENSES plus the explicit opt-ins, sorted (case-insensitively), unique."""
    merged: dict[str, str] = {}
    for name in (*PERMISSIVE_LICENSES, *checked_allowlist(allow_licenses)):
        merged.setdefault(name.casefold(), name)
    return sorted(merged.values(), key=str.casefold)


def license_problems(expression: str, allowed: Iterable[str]) -> list[str]:
    """Why ``expression`` is not satisfied by the identifiers in ``allowed`` ([] if it is).

    Reports a syntax error, or the identifiers that are not allowed.
    """
    try:
        node = parse_license_expression(expression)
    except LicenseExpressionError as exc:
        return [str(exc)]
    allowed_set = frozenset(name.casefold() for name in allowed)
    if _satisfied(node, allowed_set):
        return []
    missing = sorted(
        {name for name in license_ids(node) if name.casefold() not in allowed_set},
        key=str.casefold,
    )
    return [
        f"license {expression!r} is not satisfied by the allowed licenses "
        f"(not allowed: {', '.join(missing)})"
    ]
