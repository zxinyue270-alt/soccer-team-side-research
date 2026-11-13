"""Credential and local-path patterns used by the gate.

Credential rules (rule ids in brackets):

- AWS access key ids [aws-access-key-id]; PEM/PGP private key blocks [private-key-block];
- GitHub tokens with the classic ghp/gho/ghu/ghs/ghr prefixes or the fine-grained
  github_pat prefix [github-token];
- keys starting with sk and a dash (OpenAI, Anthropic, DeepSeek, DashScope style)
  [sk-dash-key] or sk and an underscore (ElevenLabs, Stripe style) [sk-underscore-key];
  Hugging Face tokens [hf-token]; Google API keys [google-api-key];
- Kaggle key JSON, i.e. a "username" next to a 32-hex "key" [kaggle-key-json];
- names ending in API_KEY, TOKEN, SECRET or PASSWORD (any case) assigned a literal value
  [secret-assignment]. Environment lookups and placeholders do not count; in source
  files (.py, .json, .toml, ...) only quoted literals count;
- ElevenLabs xi-api-key headers [xi-api-key-header] and Authorization Bearer values
  [bearer-token].

A token must contain a run of at least 16 letters and digits that mixes two of: lower
case, upper case, digits. Placeholders (template markers, "your", "example", "xxxx", ...)
never count. So prose such as a long hyphenated word does not match.

Local paths (:func:`find_local_paths`, which the abs_paths check calls;
:func:`find_home_paths` gives its user-home findings alone):

- User-home paths [user-home-path]: a drive letter followed by Users and a name (any
  separators, escaped or not), the same behind a POSIX drive prefix such as /c/, and
  /Users/<name> or /home/<name>. A name starting with <, {, $, % or * is a placeholder
  and allowed, as are the shared Windows profiles Public and Default and a name of dots
  only.
- Private-data paths [private-data-path]: an absolute path that runs through a folder
  named gbf-data (the default name of the data root next to the checkout, see
  gbf.models) or private, in any letter case. A path is absolute when it starts with a
  drive letter and a separator (escaped or not) or with a file: URL, or when a slash, ~
  or two backslashes (a UNC share) start a word: at the start of a line or after white
  space, a quote, a backtick, =, (, [, a comma or a semicolon. So /c/... and /mnt/c/...
  count, while URLs and paths that continue a placeholder ($DATA_ROOT, <data root>,
  {root}) do not. A path is read on its line up to a quote, a backtick, a bar, an angle
  bracket, a comma, a semicolon, a parenthesis or a tab, spaces included (folder names
  may hold spaces), and split at slashes and backslashes. The folder must be a whole part
  (gbf-data-old and private.txt do not count) that begins within PATH_SCAN_CHARS
  characters of the root; a part that goes on after a space counts by its first word,
  without trailing punctuation. A part before the folder that starts with <, {, $, % or
  * or is an ellipsis (three or more dots, or …) is a placeholder and allows the path,
  so generic examples read $DATA_ROOT/private/..., <data root>/private/... or
  C:/.../gbf-data/.... The parts . and .. are not placeholders: a path that steps into
  the folder through them still counts. Relative paths are not checked.

Not covered by the path rules: a user-home path with a part of dots only right after
Users (C:/Users/./<name>); private material in a folder with another name
(D:/footage/clip7.mp4) or below a placeholder part ($DATA_ROOT/private, C:/.../private,
C:/$RECYCLE.BIN/private); a folder that begins more than PATH_SCAN_CHARS characters after
the root; a path with a character that ends the text read, such as a parenthesis or a
comma, before the folder (C:/Program Files (x86)/gbf-data/x, D:/Footage (2026)/private/x);
a folder name followed by ], } or a query ([C:/x/private], C:/x/private?a=1); a slash
root after any character but those listed above, such as :, *, { or >
(<td>/srv/private/x</td>); a drive-relative path, which starts with a single backslash;
and a path wrapped onto the next line. False positives, the price of reading a slash at
the start of a word as a root: a spaced slash or a double slash before the word private
in prose, a comment that starts with two slashes and that word, and root-relative or
protocol-relative URL paths through a folder named so.

Every pattern is assembled from fragments, so this module never matches its own rules
(tests/gate checks that). Callers report rule ids and line numbers only and never print
the matched text; :func:`redact` is a last line of defence for anything printed and hides
a private-data path from its root to the end of the text read with it.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath

# --- fragments -------------------------------------------------------------------------
_DASHES = "-" * 5
_AWS_PREFIXES = "AK" + "IA|AS" + "IA"
_GH = "g" + "h"
_SK = "s" + "k"
_HF = "h" + "f"
_GOOGLE = "AI" + "za"
_XI_HEADER = "xi-" + "api-key"
_AUTHORIZATION = "author" + "ization"
_BEARER = "bea" + "rer"
_USERS = "Us" + "ers"
_HOME = "ho" + "me"
_SECRET_SUFFIXES = "API_?KEY|TOK" + "EN|SECR" + "ET|PASSW" + "(?:OR)?D"


@dataclass(frozen=True)
class Finding:
    """A rule that matched, located by 1-based line. The matched text is not kept."""

    rule: str
    line: int


# --- token rules -----------------------------------------------------------------------
# (rule id, pattern, whether the named group "body" must contain a random-looking run)
_TOKEN_RULES: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    (
        "aws-access-key-id",
        re.compile(r"\b(?:" + _AWS_PREFIXES + r")(?P<body>[A-Z0-9]{16})\b"),
        False,
    ),
    (
        "private-key-block",
        re.compile(_DASHES + "BEGIN (?P<body>(?:[A-Z0-9]+ )*)PRIVATE KEY(?: BLOCK)?" + _DASHES),
        False,
    ),
    ("github-token", re.compile(r"\b" + _GH + r"[pousr]_(?P<body>[A-Za-z0-9]{30,})\b"), True),
    ("github-token", re.compile(r"\bgithub_" + r"pat_(?P<body>[A-Za-z0-9_]{30,})"), True),
    ("sk-dash-key", re.compile(r"\b" + _SK + r"-(?P<body>[A-Za-z0-9_-]{20,})"), True),
    ("sk-underscore-key", re.compile(r"\b" + _SK + r"_(?P<body>[A-Za-z0-9_]{20,})"), True),
    ("hf-token", re.compile(r"\b" + _HF + r"_(?P<body>[A-Za-z0-9]{30,})\b"), True),
    (
        "google-api-key",
        re.compile(_GOOGLE + r"(?P<body>[0-9A-Za-z_-]{35})(?![0-9A-Za-z_-])"),
        True,
    ),
)
_RANDOM_RUN = re.compile(r"[A-Za-z0-9]{16,}")

_KAGGLE_KEY = re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"')
_KAGGLE_USER = re.compile(r'"username"\s*:\s*"')

_ASSIGNMENT = re.compile(
    r"(?<![A-Za-z0-9_])(?P<name>[A-Za-z0-9_]*(?:" + _SECRET_SUFFIXES + r"))(?![A-Za-z0-9_])"
    r"[\"']?\s*(?:=(?!=)|:=|:(?![:/]))\s*"
    r"(?P<value>\"[^\"\n]*\"|'[^'\n]*'|[^\s\"'#;,)}\]]+)",
    re.IGNORECASE,
)
_XI_DICT = re.compile(
    r"[\"']" + _XI_HEADER + r"[\"']\s*[:=,]\s*[\"'](?P<value>[^\"'\s]{16,})[\"']", re.IGNORECASE
)
_XI_LINE = re.compile(
    r"(?<![\w-])" + _XI_HEADER + r"\s*:\s*(?P<value>[A-Za-z0-9_-]{16,})", re.IGNORECASE
)
_BEARER_VALUE = re.compile(
    _AUTHORIZATION
    + r"[\"']?\s*[:=,]\s*[\"']?"
    + _BEARER
    + r"\s+(?P<value>[A-Za-z0-9_.~+/=-]{16,})",
    re.IGNORECASE,
)

# Files in which an unquoted value can be a literal (shell, env, ini, yaml, prose).
_UNQUOTED_SUFFIXES = frozenset(
    {
        "",
        ".env",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
        ".psm1",
        ".bat",
        ".cmd",
        ".ini",
        ".cfg",
        ".conf",
        ".properties",
        ".yaml",
        ".yml",
        ".txt",
        ".md",
        ".rst",
    }
)

_PLACEHOLDER_CHARS = "<>{}$%"
_PLACEHOLDER_MARKERS = (
    "your",
    "example",
    "placeholder",
    "changeme",
    "change_me",
    "change-me",
    "dummy",
    "redacted",
    "replace",
    "insert",
    "sample",
    "fake",
    "xxxx",
    "****",
    "....",
    "…",
)
_PLACEHOLDER_WORDS = frozenset({"none", "null", "nil", "true", "false", "secret", "password"})
_ENV_LOOKUP = re.compile(r"os\.environ|getenv|process\.env|secrets\.|env\(", re.IGNORECASE)
_ENV_VAR_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
_AUTH_SCHEMES = ("bearer ", "basic ", "token ")

# --- user-home paths -------------------------------------------------------------------
_SEP = r"[\\/]+"
_NAME = r"(?P<name>[^\\/\s\"'`|,;)]+)"
_HOME_RULES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:" + _SEP + "(?i:" + _USERS + ")" + _SEP + _NAME),
    re.compile(r"(?<![\w.-])/[A-Za-z]/(?i:" + _USERS + ")/" + _NAME),
    re.compile(r"(?<![\w.:-])/" + _USERS + "/" + _NAME),
    re.compile(r"(?<![\w.:-])/" + _HOME + "/" + _NAME),
)
_PLACEHOLDER_NAME_START = "<{$%*"
_SHARED_PROFILES = frozenset({"public", "default"})
# A redacted home path runs to the end of the name, which may contain spaces.
_NAME_TAIL = re.compile(r"[^\\/\n\"'`|]*")
REDACTED = "<redacted>"

# --- private-data paths ----------------------------------------------------------------
#: Folder names (lower case) that mark private material inside an absolute path.
PRIVATE_DIR_NAMES = frozenset({"gbf" + "-data", "priv" + "ate"})
#: The private folder must begin within this many characters after the path's root
#: (Windows' classic MAX_PATH), so a root does not reach far into a long line.
PATH_SCAN_CHARS = 260
# Start of a word: the start of the text, or after white space, a quote, a backtick, =,
# (, [, a comma or a semicolon (not after a letter, :, /, > or }, which keeps URLs and
# paths behind a placeholder out).
_WORD_START = r"(?<![^\s\"'`=(\[,;])"
_PATH_ROOT = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:(?=[\\/])"
    + r"|(?<![A-Za-z0-9])(?i:file):/+"
    + "|"
    + _WORD_START
    + r"(?:\\\\|/|~[\w.-]*(?=[\\/]))"
)
# A stretch of path text: up to a quote, a backtick, |, <, >, a comma, a semicolon, a
# parenthesis, a tab or a line break.
_PATH_STRETCH = re.compile(r"[^\n\r\t\"'`|<>,;()]+")
_PATH_PART = re.compile(r"[^\\/]+")
_PRIVATE_WORD = re.compile("|".join(sorted(PRIVATE_DIR_NAMES)), re.IGNORECASE)
_PART_END_PUNCTUATION = ".:!?*_"


def looks_random(value: str) -> bool:
    """True if ``value`` mixes at least two of: lower case, upper case, digits (ASCII)."""
    lower = any("a" <= c <= "z" for c in value)
    upper = any("A" <= c <= "Z" for c in value)
    digit = any("0" <= c <= "9" for c in value)
    return lower + upper + digit >= 2


def has_random_run(value: str) -> bool:
    """True if ``value`` holds 16+ consecutive letters/digits that look random."""
    return any(looks_random(run) for run in _RANDOM_RUN.findall(value))


def is_placeholder(value: str) -> bool:
    """True for empty values, template markers and words that mark example values."""
    stripped = value.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered in _PLACEHOLDER_WORDS:
        return True
    if any(ch in stripped for ch in _PLACEHOLDER_CHARS):
        return True
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def _secret_like_literal(value: str) -> bool:
    candidate = value.strip()
    lowered = candidate.lower()
    for scheme in _AUTH_SCHEMES:
        if lowered.startswith(scheme):
            candidate = candidate[len(scheme) :].strip()
            break
    if len(candidate) < 8 or any(ch.isspace() for ch in candidate):
        return False
    if is_placeholder(candidate) or _ENV_LOOKUP.search(candidate):
        return False
    if _ENV_VAR_NAME.fullmatch(candidate):
        return False
    return looks_random(candidate)


def _unquoted_ok(filename: str) -> bool:
    name = PurePosixPath(filename).name.lower()
    return name.startswith(".env") or PurePosixPath(name).suffix in _UNQUOTED_SUFFIXES


def _token_spans(text: str) -> Iterator[tuple[str, int, int]]:
    for rule, pattern, must_look_random in _TOKEN_RULES:
        for match in pattern.finditer(text):
            if is_placeholder(match.group(0)):
                continue
            if must_look_random and not has_random_run(match.group("body")):
                continue
            yield rule, match.start(), match.end()


def _home_spans(text: str) -> Iterator[tuple[int, int]]:
    for pattern in _HOME_RULES:
        for match in pattern.finditer(text):
            name = match.group("name")
            if name[0] in _PLACEHOLDER_NAME_START or name.lower() in _SHARED_PROFILES:
                continue
            if set(name) <= {".", "…"}:
                continue
            tail = _NAME_TAIL.match(text, match.start("name"))
            yield match.start(), tail.end() if tail is not None else match.end()


def _is_ellipsis(word: str) -> bool:
    """Whether ``word`` is an ellipsis: three or more dots, or a … with or without dots
    (. and .. are not)."""
    return set(word) <= {".", "…"} and ("…" in word or len(word) >= 3)


def _part_kind(part: str) -> str:
    """Kind of a path part: "folder" (a private folder), "placeholder" or "" (docstring)."""
    words = part.split(None, 1)
    if not words:
        return ""
    first = words[0]
    if first[0] in _PLACEHOLDER_NAME_START or _is_ellipsis(first):
        return "placeholder"
    if first.rstrip(_PART_END_PUNCTUATION).lower() in PRIVATE_DIR_NAMES:
        return "folder"
    return ""


def _private_spans(text: str) -> Iterator[tuple[int, int]]:
    """(root start, end of its stretch) of every private-data path (module docstring).

    A stretch of path text is split into parts once; every root in it then looks up the
    first part after it that is a private folder or a placeholder, so a line with many
    roots costs no more than one pass over it."""
    if _PRIVATE_WORD.search(text) is None:
        return
    for stretch in _PATH_STRETCH.finditer(text):
        lo, hi = stretch.span()
        if _PRIVATE_WORD.search(text, lo, hi) is None:
            continue
        starts: list[int] = []
        kinds: list[str] = []
        for part in _PATH_PART.finditer(text, lo, hi):
            starts.append(part.start())
            kinds.append(_part_kind(part.group(0)))
        # following[i]: the first part from i on that is a folder or a placeholder.
        following = [len(starts)] * (len(starts) + 1)
        for i in range(len(starts) - 1, -1, -1):
            following[i] = i if kinds[i] else following[i + 1]
        for root in _PATH_ROOT.finditer(text, lo, hi):
            # A root ends where a part starts or at a separator, so the parts after it
            # are exactly the stretch's parts that start at or after its end.
            i = following[bisect_left(starts, root.end())]
            if (
                i < len(starts)
                and kinds[i] == "folder"
                and starts[i] - root.end() < PATH_SCAN_CHARS
            ):
                yield root.start(), hi


class _Lines:
    """1-based line numbers of offsets in one text. Each count goes on from the offset
    asked for last, so the matches of a pattern, which come in increasing order, cost one
    pass over the text in all, not one pass up to each match."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._offset = 0
        self._line = 1

    def of(self, offset: int) -> int:
        if offset < self._offset:
            self._offset, self._line = 0, 1
        self._line += self._text.count("\n", self._offset, offset)
        self._offset = offset
        return self._line


def find_credentials(text: str, filename: str = "") -> list[Finding]:
    """Credential findings in ``text`` (the content of ``filename``), one per rule and line."""
    lines = _Lines(text)
    found: set[Finding] = set()
    for rule, start, _ in _token_spans(text):
        found.add(Finding(rule, lines.of(start)))
    key = _KAGGLE_KEY.search(text)
    if key is not None and _KAGGLE_USER.search(text):
        found.add(Finding("kaggle-key-json", lines.of(key.start())))
    unquoted_ok = _unquoted_ok(filename)
    for match in _ASSIGNMENT.finditer(text):
        raw = match.group("value")
        if raw[:1] in "\"'":
            value = raw[1:-1]
        elif unquoted_ok:
            value = raw
        else:
            continue
        if _secret_like_literal(value):
            found.add(Finding("secret-assignment", lines.of(match.start())))
    for pattern in (_XI_DICT, _XI_LINE):
        for match in pattern.finditer(text):
            if _secret_like_literal(match.group("value")):
                found.add(Finding("xi-api-key-header", lines.of(match.start())))
    for match in _BEARER_VALUE.finditer(text):
        if _secret_like_literal(match.group("value")):
            found.add(Finding("bearer-token", lines.of(match.start())))
    return sorted(found, key=lambda f: (f.line, f.rule))


def find_local_paths(text: str) -> list[Finding]:
    """User-home and private-data path findings in ``text``, one per rule and line."""
    lines = _Lines(text)
    found = {Finding("user-home-path", lines.of(start)) for start, _ in _home_spans(text)}
    found.update(Finding("private-data-path", lines.of(start)) for start, _ in _private_spans(text))
    return sorted(found, key=lambda f: (f.line, f.rule))


def find_home_paths(text: str) -> list[Finding]:
    """User-home path findings in ``text``, one per line (no private-data paths)."""
    lines = _Lines(text)
    numbers = {lines.of(start) for start, _ in _home_spans(text)}
    return [Finding("user-home-path", line) for line in sorted(numbers)]


def redact(text: str) -> str:
    """Replace every credential token and local path in ``text`` with ``<redacted>``."""
    spans = [(start, end) for _, start, end in _token_spans(text)]
    spans.extend(_home_spans(text))
    spans.extend(_private_spans(text))
    if not spans:
        return text
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        pieces.append(REDACTED)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)
