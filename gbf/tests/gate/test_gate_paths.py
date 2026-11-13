"""Private-data paths in the abs_paths check: absolute paths through a gbf-data or private
folder in Windows (forward and back slashes, escaped or not), POSIX, UNC, file-URL and
quoted spellings, also through . and .. parts; placeholders, relative paths and
look-alikes that must not match; redaction; the check and the command line on a temporary
repository; and the repository's own files. Every path that must match is assembled at
run time under neutral roots such as D:/work, so this file never matches the rule itself
and names no real local root."""

from __future__ import annotations

import string
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gbf.gate import patterns
from gbf.gate.content import check_abs_paths
from gbf.gate.model import CheckResult
from gbf.gate.patterns import REDACTED, find_home_paths, find_local_paths, redact
from gbf.gate.repo import Workspace
from gbf.util.io import write_text
from gbf.util.paths import repo_root

RunGate = Callable[..., tuple[int, str]]
NoLeak = Callable[..., None]

DATA = "gbf" + "-data"
PRIVATE = "priv" + "ate"
BACKSLASH = chr(92)
RULE = "private-data-path"


def slash(*parts: str) -> str:
    return "/".join(parts)


def back(*parts: str) -> str:
    return BACKSLASH.join(parts)


def lines(result: CheckResult) -> list[str]:
    return [p.format() for p in result.sorted_problems()]


# Built at import time, so the source text of this file holds none of them.
SPELLINGS: dict[str, str] = {
    "windows-forward": slash("D:", "work", DATA, PRIVATE, "clip7", "report.html"),
    "windows-backslash": back("D:", "work", DATA, PRIVATE, "clip7", "labels.json"),
    "windows-escaped": (BACKSLASH * 2).join(["D:", "work", DATA, "runs", "a.json"]),
    "windows-mixed": "D:" + BACKSLASH + slash("work", DATA, "x.json"),
    "lower-case-drive": slash("d:", "srv", DATA, "x"),
    "posix-drive-prefix": slash("", "d", "work", DATA, PRIVATE, "labels", "x.json"),
    "wsl-mount": slash("", "mnt", "d", "work", DATA, "x"),
    "posix": slash("", "srv", "footage", PRIVATE, "clip7.mp4"),
    "unc-share": BACKSLASH * 2 + back("nas", "share", PRIVATE, "clip7.mp4"),
    "file-url": "file://" + slash("", "srv", DATA, "x"),
    "file-url-drive": "file:///" + slash("D:", "work", DATA, "x"),
    "home-tilde": slash("~", DATA, PRIVATE, "x"),
    "upper-case-folder": slash("D:", "Footage", PRIVATE.upper(), "x.mp4"),
    "space-in-a-folder": slash("D:", "My Data", DATA, "runs", "x"),
    "ends-at-the-folder": "kept in " + slash("D:", "clips", PRIVATE) + " for now",
    "end-of-sentence": "kept in " + slash("D:", "clips", PRIVATE) + ".",
    "double-quoted": 'out = "' + slash("D:", "work", DATA, PRIVATE, "x.json") + '"',
    "single-quoted": "out = '" + slash("", "d", "work", DATA, "x") + "'",
    "backticks": "see `" + back("D:", "work", DATA, "x") + "`",
    "json-value": '{"out": "' + (BACKSLASH * 2).join(["D:", "work", DATA, PRIVATE]) + '"}',
    "docstring": '    """Reads ' + slash("D:", "work", DATA, PRIVATE, "x.json") + '."""',
    "option-value": "--out=" + slash("D:", "work", DATA, PRIVATE, "run7"),
    "assignment": "DATA_ROOT=" + slash("", "srv", DATA),
    "markdown-link": "[report](" + slash("D:", "work", DATA, "report.html") + ")",
    "bold": "**" + slash("D:", "clips", PRIVATE) + "**",
    # Fifth review: a . or .. part before the folder counted as a placeholder.
    "parent-part": slash("D:", "work", "gbf", "..", DATA, PRIVATE, "y"),
    "parent-part-backslash": back("D:", "work", "gbf", "..", DATA, PRIVATE, "y"),
    "current-part-posix": slash("", "d", "work", ".", DATA, "y"),
    "current-part-drive": slash("D:", "work", ".", DATA, "y"),
}


@pytest.mark.parametrize("text", SPELLINGS.values(), ids=SPELLINGS.keys())
def test_private_data_paths_in_every_spelling_are_found(text: str) -> None:
    assert [(f.rule, f.line) for f in find_local_paths("first line\n" + text + "\n")] == [(RULE, 2)]


@pytest.mark.parametrize(
    "text",
    [
        "gbf-data/models",
        "../gbf-data/private/clip7/x.json",
        "gbf/../gbf-data/private/x",
        "receipts/private/x.json",
        "docs/private/notes.md",
        "$DATA_ROOT/private/clip7/labels.json",
        "${DATA_ROOT}/private/clip7",
        "%DATA_ROOT%\\private\\clip7",
        "<data root>/private/clip7",
        "<parent of the gbf checkout>/gbf-data/models",
        "C:/.../gbf-data/models",
        "C:/…/gbf-data/models",
        "D:/..../private/x",
        "C:/<root>/private/x",
        "D:/{name}/private/x",
        "/home/{user}/gbf-data/x",
        "D:/work/gbf-data-old/x",
        "D:/work/private.txt",
        "D:/work/privateer/x",
        "/srv/privately/x",
        "C:/data/a.json",
        "C:/Windows/Fonts/arial.ttf",
        "https://example.com/private/x",
        "https://github.com/someone/gbf-data/tree/main",
        "keep private material under DATA_ROOT",
        "public/private key pairs",
        "C:/a and the gbf-data folder",
        "a / b / c private",
    ],
)
def test_placeholders_relative_paths_and_look_alikes_do_not_match(text: str) -> None:
    assert find_local_paths(text) == []


def test_each_rule_reports_a_line_once(make_token: Callable[..., str]) -> None:
    name = make_token("gbfuser", 5, string.ascii_lowercase)
    home = slash("C:", "Users", name, "x")
    first = SPELLINGS["windows-forward"]
    second = SPELLINGS["posix"]
    text = f"{first} and {second}\nok\n{home} {SPELLINGS['unc-share']}\n"
    assert [(f.rule, f.line) for f in find_local_paths(text)] == [
        (RULE, 1),
        (RULE, 3),
        ("user-home-path", 3),
    ]


def test_find_local_paths_is_what_the_abs_paths_check_runs(
    gate_repo: Path, make_token: Callable[..., str]
) -> None:
    # Fifth review: find_home_paths also returned private-data paths, under a name and
    # docstrings that promised user-home paths only. The check calls find_local_paths, and
    # find_home_paths gives the user-home findings alone again.
    name = make_token("gbfuser", 5, string.ascii_lowercase)
    text = slash("", "home", name, "a") + "\n" + SPELLINGS["windows-backslash"] + "\n"
    found = [(f.rule, f.line) for f in find_local_paths(text)]
    assert found == [("user-home-path", 1), (RULE, 2)]
    assert [(f.rule, f.line) for f in find_home_paths(text)] == [("user-home-path", 1)]
    write_text(gate_repo / "docs" / "both.md", text)
    assert lines(check_abs_paths(gate_repo)) == [
        f"abs_paths docs/both.md:{line} {rule}" for rule, line in found
    ]


def test_the_folder_must_begin_within_the_scan_limit() -> None:
    limit = patterns.PATH_SCAN_CHARS
    near = "C:/" + "a" * (limit - 10) + "/" + PRIVATE + "/x"
    far = "C:/" + "a" * limit + "/" + PRIVATE + "/x"
    assert [f.rule for f in find_local_paths(near)] == [RULE]
    assert find_local_paths(far) == []


def test_a_line_with_many_roots_is_split_into_parts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    kind = patterns._part_kind

    def counting(part: str) -> str:
        calls.append(part)
        return kind(part)

    monkeypatch.setattr(patterns, "_part_kind", counting)
    roots = 2000
    text = "/a " * roots + "/b/" + PRIVATE + "/x"
    assert [f.rule for f in find_local_paths(text)] == [RULE]
    # One call per part ("a " for each root, then b, the folder and x), not per root.
    assert len(calls) == roots + 3


def test_line_numbers_cost_one_pass_however_many_paths_a_text_holds(
    make_token: Callable[..., str], counting_text: Callable[[str], Any]
) -> None:
    # Fifth review: each finding counted the line breaks from the start of the text, so a
    # text full of paths took time proportional to their number times its length (35 s for
    # 1 MB with a path on every line). The matches of each pattern come in order, so the
    # count goes on from the last one: a pass over the text per pattern at most.
    name = make_token("gbfuser", 5, string.ascii_lowercase)
    row = "a " + SPELLINGS["windows-forward"] + " b " + slash("", "home", name, "x") + "\n"
    text = counting_text(row * 2000)
    found = find_local_paths(text)
    assert len(found) == 4000
    assert [(f.rule, f.line) for f in found[-2:]] == [(RULE, 2000), ("user-home-path", 2000)]
    assert text.scanned < 10 * len(text)


def test_redact_hides_a_private_path_from_its_root_on() -> None:
    path = SPELLINGS["windows-forward"]
    assert redact("a " + path + " b") == "a " + REDACTED
    assert redact('x "' + path + '" y') == f'x "{REDACTED}" y'
    assert redact("x " + SPELLINGS["unc-share"]) == "x " + REDACTED
    for kept in (
        "see $DATA_ROOT/private/clip7",
        "abs_paths docs/private/notes.md:3 private-data-path",
        "restricted_data private/cases/7.json nda-case-file",
    ):
        assert redact(kept) == kept


def _notes() -> str:
    return "\n".join(
        [
            "# Notes",
            "",
            "Placeholders are fine: $DATA_ROOT/private/clip7/labels.json.",
            "Run written to " + SPELLINGS["windows-forward"] + ".",
            "Labels: " + SPELLINGS["windows-backslash"],
            "Copy: " + SPELLINGS["posix-drive-prefix"],
            "",
        ]
    )


def test_the_check_fails_private_paths_in_a_repository(gate_repo: Path) -> None:
    write_text(gate_repo / "docs" / "notes.md", _notes())
    source = '"""Settings."""\n\nOUT = "' + SPELLINGS["windows-escaped"] + '"\n'
    write_text(gate_repo / "src" / "demo" / "where.py", source)
    assert lines(check_abs_paths(gate_repo)) == [
        f"abs_paths docs/notes.md:{line} {RULE}" for line in (4, 5, 6)
    ] + [f"abs_paths src/demo/where.py:3 {RULE}"]


def test_the_gate_reports_private_paths_without_printing_them(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    write_text(gate_repo / "docs" / "notes.md", _notes())
    code, output = run_gate(gate_repo)
    assert code == 1
    for line in (4, 5, 6):
        assert f"abs_paths docs/notes.md:{line} {RULE}" in output
    assert "[gate] abs_paths FAILED (3)" in output
    no_leak(
        output,
        gate_repo,
        SPELLINGS["windows-forward"],
        SPELLINGS["windows-backslash"],
        SPELLINGS["posix-drive-prefix"],
        "clip7",
    )


def test_the_repositorys_own_files_hold_no_private_data_paths() -> None:
    root = repo_root()
    workspace = Workspace(root)
    assert len(workspace.files) >= 100
    result = check_abs_paths(root, workspace)
    assert [p.format() for p in result.problems if p.rule == RULE] == []
