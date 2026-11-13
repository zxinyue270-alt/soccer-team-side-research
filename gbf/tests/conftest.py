"""Shared test setup.

- Network access is blocked for the whole session (loopback excepted).
- ``unicode_tmp_path`` gives a temporary directory whose path contains
  non-ASCII characters and a space, so path handling is exercised on every run.
- Tests never skip: a missing tool is a failure, not a skip (the gate rejects skips).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gbf.testing import offline

offline.install()


@pytest.fixture
def unicode_tmp_path(tmp_path: Path) -> Path:
    path = tmp_path / "非ASCII 路径-ü"
    path.mkdir()
    return path


@pytest.fixture(scope="session")
def ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if found is None:
        pytest.fail("ffmpeg is required on PATH (see registries/tools.toml)")
    return found


@pytest.fixture(scope="session")
def ffprobe_bin() -> str:
    found = shutil.which("ffprobe")
    if found is None:
        pytest.fail("ffprobe is required on PATH (see registries/tools.toml)")
    return found
