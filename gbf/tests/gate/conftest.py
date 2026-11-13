"""Fixtures for the gate tests.

``gate_repo`` is a minimal git repository (files added to the index, no commit) under a
path with non-ASCII characters and a space. It passes every static check: placeholders
for keys and home paths, a valid public receipt, and dependencies that are allowed.
git runs with an empty global config so a machine's own ignore rules or autocrlf
setting cannot change what the gate sees. Secrets and home paths used by the red tests
are assembled at runtime; none appears literally in this directory.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import string
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from gbf.gate import cli
from gbf.gate.patterns import is_placeholder
from gbf.util.io import write_text

PYPROJECT = """\
[project]
name = "gate-fixture"
version = "0.0.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0",
    "jsonschema>=4.23",
]

[dependency-groups]
dev = ["pytest>=8.3"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
"""

UV_LOCK = """\
version = 1
requires-python = ">=3.11"

[[package]]
name = "gate-fixture"
version = "0.0.0"
source = { editable = "." }

[[package]]
name = "numpy"
version = "2.5.4"
source = { registry = "https://pypi.org/simple" }
"""

README = """\
# Demo

Set keys in your shell, never in files: `export OPENAI_API_KEY=<your key>`, `HF_TOKEN=$HF_TOKEN`.
Local data may live in C:/Users/<name>/data, /home/{user}/data, /Users/$USER/data or
%USERPROFILE%/data; the repository only stores paths relative to DATA_ROOT.
"""

SETTINGS = '''\
"""Settings read from the environment."""

import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
MAX_TOKENS = 4096
'''

BASELINE_FILES = {
    ".gitignore": "__pycache__/\n.local/\n",
    ".gitattributes": "* text=auto eol=lf\n",
    "pyproject.toml": PYPROJECT,
    "uv.lock": UV_LOCK,
    "README.md": README,
    "src/demo/__init__.py": '"""Demo package for the gate tests."""\n',
    "src/demo/settings.py": SETTINGS,
    "tests/test_demo.py": "def test_demo() -> None:\n    assert 1 + 1 == 2\n",
}

GIT_IDENTITY = ("-c", "user.name=gbf gate test", "-c", "user.email=gate-test@example.invalid")


def valid_receipt_data() -> dict[str, object]:
    """A receipt that satisfies urn:gbf:schema:receipt."""
    return {
        "schema_version": 1,
        "kind": "eval",
        "status": "passed",
        "created_at": "2026-10-21T00:00:00Z",
        "producer": "tests.gate",
        "code": {"gbf_version": "0.0.1", "git_commit": "unknown", "git_dirty": False},
        "inputs": [{"name": "demo test", "relpath": "tests/test_demo.py", "sha256": "0" * 64}],
        "scope": "A demo receipt for the gate tests.",
        "limitations": ["Synthetic content only."],
        "restrictions": [],
        "result": {"answer": 42},
    }


GIT_REDIRECTS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")


def run_git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "init.defaultBranch=main", *GIT_IDENTITY, *args],
        cwd=root,
        env=env,
        capture_output=True,
        check=True,
    )
    return completed.stdout.decode("utf-8", "replace")


@pytest.fixture
def isolated_git(unicode_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make git ignore the machine's global and system configuration."""
    config = unicode_tmp_path / "empty.gitconfig"
    config.write_bytes(b"")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in GIT_REDIRECTS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def git() -> Callable[..., str]:
    """``git(root, *args)`` runs git in ``root`` with a fixed test identity."""
    return run_git


@pytest.fixture(scope="session")
def gate_repo_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The baseline repository, built once per session (copied by ``gate_repo``)."""
    base = tmp_path_factory.mktemp("gate-template")
    config = base / "empty.gitconfig"
    config.write_bytes(b"")
    env = {k: v for k, v in os.environ.items() if k.upper() not in GIT_REDIRECTS}
    env.update({"GIT_CONFIG_GLOBAL": str(config), "GIT_CONFIG_NOSYSTEM": "1"})
    root = base / "template"
    for rel, text in BASELINE_FILES.items():
        write_text(root / rel, text)
    receipt = json.dumps(valid_receipt_data(), indent=2, sort_keys=True) + "\n"
    write_text(root / "receipts" / "public" / "demo.json", receipt)
    run_git(root, "init", "-q", "--template=", env=env)
    run_git(root, "add", "-A", env=env)
    return root


@pytest.fixture
def gate_repo(unicode_tmp_path: Path, isolated_git: None, gate_repo_template: Path) -> Path:
    """Root of a fresh copy of the baseline repository (passes every static check)."""
    root = unicode_tmp_path / "repo 仓库"
    shutil.copytree(gate_repo_template, root)
    return root


@pytest.fixture
def valid_receipt() -> dict[str, object]:
    return valid_receipt_data()


@pytest.fixture
def run_gate(capsys: pytest.CaptureFixture[str]) -> Callable[..., tuple[int, str]]:
    """``run_gate(root, *args)`` runs the CLI and returns (exit code, stdout + stderr)."""

    def run(root: Path, *args: str) -> tuple[int, str]:
        code = cli.main(["--root", str(root), *args])
        captured = capsys.readouterr()
        return code, captured.out + captured.err

    return run


@pytest.fixture
def make_token() -> Callable[..., str]:
    """``make_token(prefix, length, alphabet)``: a deterministic random-looking token.

    The body mixes lower case, upper case and digits (within the alphabet) and is never a
    placeholder, so the gate must treat it as a real credential.
    """

    def make(prefix: str, length: int, alphabet: str = string.ascii_letters + string.digits) -> str:
        rng = random.Random(f"{prefix}:{length}:{alphabet}")
        while True:
            body = "".join(rng.choice(alphabet) for _ in range(length))
            classes = [
                set(alphabet) & set(group)
                for group in (string.ascii_lowercase, string.ascii_uppercase, string.digits)
            ]
            if all(set(body) & group for group in classes if group) and not is_placeholder(body):
                return prefix + body

    return make


class CountingText(str):
    """Text whose count method adds up, in ``scanned``, the characters it scans."""

    scanned = 0

    def count(self, sub: str, start: int = 0, end: int | None = None) -> int:  # type: ignore[override]
        stop = len(self) if end is None else end
        self.scanned += max(stop - start, 0)
        return super().count(sub, start, stop)


@pytest.fixture
def counting_text() -> Callable[[str], CountingText]:
    """``counting_text(text)``: ``text`` as a CountingText, to measure what numbering the
    lines of findings costs."""
    return CountingText


def assert_no_leak(output: str, root: Path, *values: str) -> None:
    """Fail if ``output`` contains any of ``values`` or the repository's absolute path."""
    for value in (*values, str(root), root.as_posix(), str(root.resolve())):
        assert value not in output


@pytest.fixture
def no_leak() -> Callable[..., None]:
    """``no_leak(output, root, *values)``: the gate printed none of the values or paths."""
    return assert_no_leak
