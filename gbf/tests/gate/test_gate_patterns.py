"""Credential and user-home-path patterns: true positives built at runtime, placeholders
and look-alikes that must not match, redaction, and self-scans of the gate's sources and
of the repository's own data files and docs."""

from __future__ import annotations

import string
from collections.abc import Callable
from typing import Any

import pytest

from gbf.gate.content import check_abs_paths, check_credentials
from gbf.gate.datafiles import check_restricted_data
from gbf.gate.imports import check_forbidden_imports
from gbf.gate.patterns import (
    REDACTED,
    find_credentials,
    find_home_paths,
    has_random_run,
    is_placeholder,
    redact,
)
from gbf.gate.repo import Workspace
from gbf.util.paths import repo_root

HEX = "0123456789abcdef"
ALNUM = string.ascii_letters + string.digits
SK = "s" + "k"
BACKSLASH = chr(92)

# (rule, builder of the text, file name); builders get make_token.
POSITIVES: list[tuple[str, Callable[[Callable[..., str]], str], str]] = [
    (
        "aws-access-key-id",
        lambda t: t("AK" + "IA", 16, string.ascii_uppercase + string.digits),
        "a.txt",
    ),
    ("private-key-block", lambda t: "-" * 5 + "BEGIN OPENSSH PRIVATE KEY" + "-" * 5, "id.txt"),
    ("github-token", lambda t: "pushed with " + t("gh" + "p_", 36), "a.txt"),
    ("github-token", lambda t: t("github_" + "pat_", 22) + "_" + t("", 59), "a.txt"),
    ("sk-dash-key", lambda t: t(SK + "-", 48), "a.txt"),
    ("sk-dash-key", lambda t: t(SK + "-proj-", 64, ALNUM + "-_"), "a.py"),
    ("sk-dash-key", lambda t: t(SK + "-ant-api03-", 93, ALNUM + "-_") + "AA", "a.md"),
    ("sk-underscore-key", lambda t: t(SK + "_", 48, HEX), "a.txt"),
    ("hf-token", lambda t: t("h" + "f_", 34, string.ascii_letters), "a.txt"),
    ("google-api-key", lambda t: t("AI" + "za", 35, ALNUM + "-_"), "a.txt"),
    (
        "kaggle-key-json",
        lambda t: '{"username": "someone", "key": "' + t("", 32, HEX) + '"}',
        "kaggle-copy.json",
    ),
    ("secret-assignment", lambda t: "OPENAI_API_KEY=" + t("", 40), ".env.local"),
    ("secret-assignment", lambda t: "export HF_TOKEN=" + t("", 30), "setup.sh"),
    ("secret-assignment", lambda t: "db_password: " + t("", 14), "config.yaml"),
    ("secret-assignment", lambda t: '{"ARK_API_KEY": "' + t("", 36, HEX + "-") + '"}', "a.json"),
    ("secret-assignment", lambda t: 'CLIENT_SECRET = "' + t("", 20) + '"', "a.py"),
    ("xi-api-key-header", lambda t: 'curl -H "xi-api-key: ' + t("", 32, HEX) + '" u', "a.sh"),
    ("xi-api-key-header", lambda t: 'headers = {"xi-api-key": "' + t("", 32, HEX) + '"}', "a.py"),
    ("bearer-token", lambda t: "Authorization: Bearer " + t("", 40, ALNUM + "._-"), "a.http"),
    ("bearer-token", lambda t: '{"Authorization": "Bearer ' + t("", 40) + '"}', "a.py"),
]

NEGATIVES: list[tuple[str, str]] = [
    (SK + "-" + "x" * 48, "a.txt"),
    (SK + "-your-openai-key-here-0123456789abcdef", "a.txt"),
    (SK + "-learn-intelex-optimized-version-2024", "a.md"),
    ("h" + "f_hub_download_and_cache_files_for_testing", "a.py"),
    (SK + "_learn_estimator_wrapper_class_v2", "a.py"),
    ("AK" + "IAIOSFODNN7EXAMPLE", "a.md"),
    ("OPENAI_API_KEY=<your key>", ".env.example"),
    ("export HF_TOKEN=$HF_TOKEN", "setup.sh"),
    ("GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}", "ci.yml"),
    ('token = os.environ["TOKEN"]', "a.py"),
    ("API_KEY = some_variable_name_that_is_long123", "a.py"),
    ('HF_TOKEN_ENV = "HF_TOKEN_V2"', "a.py"),
    ("max_tokens = 4096", "a.py"),
    ("PASSWORD = 'changeme'", "a.py"),
    ("password: correcthorsebattery", "a.yaml"),
    ('{"xi-api-key": elevenlabs_api_key_value}', "a.py"),
    ('headers = {"Authorization": f"Bearer {token}"}', "a.py"),
    ("Authorization: Bearer $TOKEN", "a.sh"),
]


@pytest.mark.parametrize(("rule", "build", "filename"), POSITIVES)
def test_credential_rules_match_runtime_tokens(
    rule: str,
    build: Callable[[Callable[..., str]], str],
    filename: str,
    make_token: Callable[..., str],
) -> None:
    text = "first line\n" + build(make_token) + "\n"
    findings = find_credentials(text, filename)
    assert [(f.rule, f.line) for f in findings] == [(rule, 2)]


@pytest.mark.parametrize(("text", "filename"), NEGATIVES)
def test_placeholders_lookups_and_lookalikes_do_not_match(text: str, filename: str) -> None:
    assert find_credentials(text, filename) == []


def test_unquoted_values_count_only_outside_source_files(make_token: Callable[..., str]) -> None:
    line = "API_TOKEN = " + make_token("", 24)
    assert [f.rule for f in find_credentials(line, "notes.md")] == ["secret-assignment"]
    assert find_credentials(line, "module.py") == []


def test_random_run_and_placeholder_helpers() -> None:
    assert has_random_run("abcDEF0123456789xyz")
    assert not has_random_run("learn-intelex-optimized-version-2024")
    assert not has_random_run("abcdefghijklmnopqrstu")
    assert is_placeholder("<api key>")
    assert is_placeholder("${{ secrets.X }}")
    assert is_placeholder("sk-xxxxxxxxxxxxxxxxxxxx")
    assert not is_placeholder("q7Zk2Lm9")


def _home_variants(name: str) -> list[str]:
    return [
        "C:/Users/" + name + "/data",
        BACKSLASH.join(["D:", "Users", name, "x"]),
        (BACKSLASH * 2).join(["C:", "Users", name, "x"]),
        "c:/users/" + name + "/x",
        "/c/Users/" + name + "/x",
        "/" + "Users/" + name + "/x",
        "/" + "home/" + name + "/x",
        "file:///" + "home/" + name + "/x",
        '"/' + "home/" + name + '"',
    ]


def test_home_paths_in_every_spelling_are_found(make_token: Callable[..., str]) -> None:
    name = make_token("gbfuser", 5, string.ascii_lowercase)
    for variant in _home_variants(name):
        assert [(f.rule, f.line) for f in find_home_paths("x\n" + variant)] == [
            ("user-home-path", 2)
        ], variant


@pytest.mark.parametrize(
    "text",
    [
        "C:/Users/<name>/data",
        "C:/Users/{user}/data",
        "/home/$USER/data",
        "/Users/${USER}/data",
        "C:/Users/%USERNAME%/data",
        "C:/Users/*/AppData",
        "C:/Users/.../AppData",
        "C:/Users/Public/Documents",
        "C:/Users",
        "https://example.com/home/page/",
        "https://example.com/Users/someone/",
        "docs/home/alice/notes.md",
        "./home/bob/notes.md",
        "~/data",
    ],
)
def test_placeholders_and_relative_paths_are_not_home_paths(text: str) -> None:
    assert find_home_paths(text) == []


def test_redact_hides_tokens_and_whole_home_paths(make_token: Callable[..., str]) -> None:
    key = make_token(SK + "-", 48)
    name = make_token("First", 4, string.ascii_lowercase) + " Last"
    line = f"a C:/Users/{name}/AppData b {key} c docs/home/alice/x"
    redacted = redact(line)
    assert key not in redacted
    assert name not in redacted
    assert "Last" not in redacted
    assert redacted == f"a {REDACTED}/AppData b {REDACTED} c docs/home/alice/x"
    assert redact("credentials src/x.py:3 sk-dash-key") == "credentials src/x.py:3 sk-dash-key"


def test_gate_sources_never_match_their_own_rules() -> None:
    root = repo_root()
    own = [
        p.relative_to(root).as_posix()
        for base in (root / "src" / "gbf" / "gate", root / "tests" / "gate")
        for p in sorted(base.rglob("*.py"))
    ]
    assert len(own) >= 10
    workspace = Workspace(root, files=own)
    for check in (check_credentials, check_abs_paths, check_forbidden_imports):
        result = check(root, workspace)
        assert result.problems == [], [p.format() for p in result.problems]
        assert result.scanned == len(own)


def test_the_repositorys_own_data_and_docs_pass_the_restricted_data_check() -> None:
    # JSON Schemas list every tag in enums, gate and preregistration files are JSON, the
    # registries assign tags, the docs explain them: none of it is restricted material.
    root = repo_root()
    bases = ("src/gbf/contracts/schemas", "gate", "prereg", "registries", "docs", "tests/gate")
    own = [
        p.relative_to(root).as_posix()
        for base in bases
        for p in sorted((root / base).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ]
    assert len([rel for rel in own if rel.endswith(".json")]) >= 10
    result = check_restricted_data(root, Workspace(root, files=own))
    assert result.problems == [], [p.format() for p in result.problems]


def test_find_reports_one_finding_per_rule_and_line(make_token: Callable[..., str]) -> None:
    key = make_token(SK + "-", 48)
    text = f'OPENAI_API_KEY = "{key}"  # and again {key}\n'
    findings = find_credentials(text, "settings.py")
    assert [(f.rule, f.line) for f in findings] == [("secret-assignment", 1), ("sk-dash-key", 1)]


def test_line_numbers_count_lf_lines(make_token: Callable[..., str]) -> None:
    key = make_token("h" + "f_", 34, string.ascii_letters)
    text = "\n" * 9 + key + "\n"
    assert [f.line for f in find_credentials(text)] == [10]


def test_line_numbers_cost_one_pass_however_many_tokens_a_text_holds(
    make_token: Callable[..., str], counting_text: Callable[[str], Any]
) -> None:
    # Fifth review: each finding counted the line breaks from the start of the text, which
    # took time proportional to the number of findings times the length of the text.
    key = make_token("h" + "f_", 34, string.ascii_letters)
    text = counting_text(f"x {key}\n" * 2000)
    found = find_credentials(text)
    assert [(f.rule, f.line) for f in found[-1:]] == [("hf-token", 2000)]
    assert len(found) == 2000
    assert text.scanned < 20 * len(text)
