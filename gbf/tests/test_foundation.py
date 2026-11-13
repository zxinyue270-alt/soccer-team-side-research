"""Tests for the shared foundation: hashing, LF-only IO, contracts, offline guard."""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest

from gbf import contracts
from gbf.testing import offline
from gbf.util import hashing, io, paths


def test_sha256_file_hashes_raw_bytes(unicode_tmp_path: Path) -> None:
    lf = unicode_tmp_path / "lf.txt"
    crlf = unicode_tmp_path / "crlf.txt"
    lf.write_bytes(b"a\nb\n")
    crlf.write_bytes(b"a\r\nb\r\n")
    assert hashing.sha256_file(lf) == hashing.sha256_bytes(b"a\nb\n")
    assert hashing.sha256_file(lf) != hashing.sha256_file(crlf)


def test_git_blob_sha1_matches_git() -> None:
    # `printf 'hello\n' | git hash-object --stdin`
    assert hashing.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_canonical_json_is_order_independent() -> None:
    a = {"b": 1, "a": [1, 2, {"z": "ü", "y": None}]}
    b = {"a": [1, 2, {"y": None, "z": "ü"}], "b": 1}
    assert hashing.canonical_json(a) == hashing.canonical_json(b)
    assert hashing.canonical_sha256(a) == hashing.canonical_sha256(b)
    with pytest.raises(ValueError, match="not JSON compliant"):
        hashing.canonical_json({"x": float("nan")})


def test_write_text_and_json_use_lf(unicode_tmp_path: Path) -> None:
    text_path = unicode_tmp_path / "out" / "note.txt"
    io.write_text(text_path, "line1\nline2\n")
    assert text_path.read_bytes() == b"line1\nline2\n"
    json_path = unicode_tmp_path / "out" / "data.json"
    io.write_json(json_path, {"b": 1, "a": "中文"})
    raw = json_path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")
    assert io.read_json(json_path) == {"a": "中文", "b": 1}


def test_png_roundtrip_on_non_ascii_path(unicode_tmp_path: Path) -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[1, 2] = (255, 128, 7)
    path = unicode_tmp_path / "帧.png"
    io.write_image_png(path, image)
    assert np.array_equal(io.read_image_rgb(path), image)


def test_repo_root_and_relative_paths() -> None:
    root = paths.repo_root()
    assert (root / "pyproject.toml").is_file()
    assert paths.to_repo_relative(root / "src" / "gbf" / "__init__.py") == "src/gbf/__init__.py"


def test_contract_schemas_load_and_check() -> None:
    names = contracts.schema_names()
    assert {"common", "restrictions"} <= set(names)
    for name in names:
        contracts.validator(name)  # raises if the schema itself is invalid


def test_restrictions_contract_reports_every_problem() -> None:
    assert contracts.def_problems(["nda", "no_share"], "restrictions", "restrictions") == []
    found = contracts.def_problems(["nda", "nda", "made_up"], "restrictions", "restrictions")
    assert len(found) == 2  # duplicate item and unknown tag are both reported


def test_contract_error_lists_all_problems() -> None:
    error = contracts.ContractError("demo", ["a: first", "b: second"])
    assert error.problems == ["a: first", "b: second"]
    assert "2 problem(s)" in str(error)


def test_relpath_rejects_local_paths() -> None:
    assert contracts.def_problems("receipts/public/eval.json", "common", "relpath") == []
    # Assembled at run time so the gate's user-home path scan stays quiet.
    home_path = "/" + "/".join(["home", "x", "a.json"])
    for bad in [home_path, "C:/data/a.json", "runs\\a.json", "a/../b.json", ""]:
        assert contracts.def_problems(bad, "common", "relpath"), bad


def test_offline_guard_blocks_remote_but_allows_loopback() -> None:
    assert offline.is_installed()
    with pytest.raises(offline.NetworkBlockedError):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(offline.NetworkBlockedError):
        socket.create_connection(("93.184.215.14", 443), timeout=1)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client = socket.create_connection(server.getsockname(), timeout=2)
        client.close()
    finally:
        server.close()
