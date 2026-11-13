"""Result types shared by every gate check.

Conventions:

- A problem prints as ``<check> <relative path>[:line] <rule id>``, optionally followed by
  ``(detail)``. The path is POSIX and relative to the repository root (``-`` when a problem
  is not about one file); lines are 1-based. Details are composed by the gate itself
  (counts, schema keywords, JSON pointers); they never hold the matched text of a secret
  or a local path.
- Durations are seconds (float).
"""

from __future__ import annotations

from dataclasses import dataclass, field

NO_PATH = "-"


@dataclass(frozen=True)
class Problem:
    """One finding of one check."""

    check: str
    path: str
    rule: str
    line: int | None = None
    detail: str = ""

    def sort_key(self) -> tuple[str, str, int, str, str]:
        return (self.check, self.path, self.line or 0, self.rule, self.detail)

    def format(self) -> str:
        location = self.path if self.line is None else f"{self.path}:{self.line}"
        text = f"{self.check} {location} {self.rule}"
        if self.detail:
            text += f" ({self.detail})"
        return text


@dataclass
class CheckResult:
    """Outcome of one check.

    ``notes`` explain coverage (printed on every run), ``hints`` suggest an improvement
    that is not required, ``warnings`` flag something worth attention that does not fail
    the gate. Only ``problems`` fail it. ``scanned`` counts the files the check examined.
    """

    name: str
    problems: list[Problem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    scanned: int = 0

    @property
    def passed(self) -> bool:
        return not self.problems

    def add(self, path: str, rule: str, *, line: int | None = None, detail: str = "") -> None:
        problem = Problem(self.name, path, rule, line, detail)
        if problem not in self.problems:
            self.problems.append(problem)

    def sorted_problems(self) -> list[Problem]:
        return sorted(self.problems, key=Problem.sort_key)


class GateError(RuntimeError):
    """A check cannot run at all, for example because the root is not a git work tree.

    ``rule`` is a short id that is safe to print. The message never contains local paths.
    """

    def __init__(self, rule: str, message: str) -> None:
        self.rule = rule
        super().__init__(message)
