"""Tests for gbf.models: the pinned table, the models dir, fetch/verify and the CLI.

Nothing here touches the network: fetch gets a fake opener that serves a small zip built
at test time, with a pin computed from it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
import tomllib
import urllib.request
import zipfile
from pathlib import Path

import pytest

from gbf import models
from gbf.models import (
    DEFAULT_MODEL_ID,
    PINS,
    ModelDownloadError,
    ModelMissingError,
    ModelPin,
    ModelsDirError,
    ModelVerificationError,
    UnknownModelError,
    check_models_dir,
    default_models_dir,
    fetch,
    model_files,
    onnx_path,
    pin_problems,
    resolve_pin,
    table_problems,
    verify,
)

STEM = "rtmpose-t_simcc-test_pt-test-halpe26_700e-256x192-0123abcd_20260101"
MEMBER = f"20260101/rtmpose_onnx/{STEM}/end2end.onnx"
NETWORK = b"not really an onnx graph, just bytes to hash " * 64


def build_zip(network: bytes = NETWORK) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in (
            (MEMBER, network),
            (f"20260101/rtmpose_onnx/{STEM}/pipeline.json", b"{}"),
        ):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue()


ZIP = build_zip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_pin(**changes) -> ModelPin:
    pin = ModelPin(
        model_id="rtmpose-t-test-256x192",
        alias="testing",
        keypoints="halpe26",
        input_size=(192, 256),
        url=f"https://download.openmmlab.com/mmpose/test/{STEM}.zip",
        zip_bytes=len(ZIP),
        zip_sha256=sha256(ZIP),
        onnx_member=MEMBER,
        onnx_bytes=len(NETWORK),
        onnx_sha256=sha256(NETWORK),
        pinned_on="2026-10-21",
    )
    return dataclasses.replace(pin, **changes)


class FakeResponse:
    def __init__(self, data: bytes, status: int = 200) -> None:
        self._data = io.BytesIO(data)
        self.status = status
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._data.read(size)

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    def __init__(self, data: bytes = ZIP, status: int = 200) -> None:
        self.data = data
        self.status = status
        self.requests: list[str] = []
        self.responses: list[FakeResponse] = []

    def open(self, request: urllib.request.Request, timeout: float = 0.0) -> FakeResponse:
        self.requests.append(request.full_url)
        response = FakeResponse(self.data, self.status)
        self.responses.append(response)
        return response


class NoNetwork:
    def open(self, request: urllib.request.Request, timeout: float = 0.0) -> FakeResponse:
        raise AssertionError(f"no download expected, got {request.full_url}")


def outside(tmp: Path) -> Path:
    """A repo_root for fetch that does not contain ``tmp`` (pytest's basetemp may be inside
    the real checkout)."""
    return tmp / "pretend-checkout"


# ------------------------------------------------------------------------ pin table


def test_pinned_table_is_well_formed() -> None:
    assert table_problems(PINS) == []
    assert [pin.model_id for pin in PINS] == [
        "rtmpose-x-halpe26-384x288",
        "rtmpose-m-halpe26-256x192",
    ]
    assert DEFAULT_MODEL_ID == "rtmpose-x-halpe26-384x288"
    for pin in PINS:
        assert re.fullmatch(r"[0-9a-f]{64}", pin.zip_sha256)
        assert re.fullmatch(r"[0-9a-f]{64}", pin.onnx_sha256)
        assert pin.url.startswith("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/")
        assert pin.onnx_member.endswith(f"{pin.zip_name.removesuffix('.zip')}/end2end.onnx")
        assert pin.keypoints == "halpe26"
        assert pin.status == "experimental"
        assert "not verified" in pin.licence
        assert pin.onnx_bytes > pin.zip_bytes * 0.9  # float weights barely compress
    x, m = resolve_pin("performance"), resolve_pin("balanced")
    assert x.input_size == (288, 384)  # width, height: the 384x288 model
    assert m.input_size == (192, 256)
    assert "halpe26_700e-384x288-7fb6e239_20230606.zip" in x.url
    assert "halpe26_700e-256x192-4d3e73dd_20230605.zip" in m.url


def test_pin_problems_are_all_listed() -> None:
    broken = make_pin(
        model_id="Bad Id",
        url="http://example.com/a.zip?x=1",
        zip_sha256="ABC",
        onnx_bytes=0,
        input_size=(256, 192),
        pinned_on="21.10.2026",
        onnx_member="../end2end.onnx",
    )
    problems = "\n".join(pin_problems(broken))
    for fragment in (
        "model_id must be",
        "url must be https on download.openmmlab.com",
        "query or fragment",
        "zip_sha256 must be 64 lowercase hex",
        "onnx_bytes must be a positive integer",
        "onnx_member must be",
        "disagrees with the HxW token",
        "pinned_on must be YYYY-MM-DD",
    ):
        assert fragment in problems
    twice = table_problems([make_pin(), make_pin(alias="other")])
    assert any("is used 2 times" in p for p in twice)
    assert any("pinned twice" in p for p in twice)


def test_resolve_pin_by_id_or_alias() -> None:
    assert resolve_pin("rtmpose-m-halpe26-256x192").alias == "balanced"
    assert resolve_pin("performance").model_id == DEFAULT_MODEL_ID
    with pytest.raises(UnknownModelError, match="rtmpose-x-halpe26-384x288 \\(performance\\)"):
        resolve_pin("lightweight")


# ------------------------------------------------------------------------ locations


def test_default_models_dir_precedence(unicode_tmp_path: Path, monkeypatch) -> None:
    assert default_models_dir({"GBF_MODELS": "a", "DATA_ROOT": "b"}) == Path("a")
    assert default_models_dir({"DATA_ROOT": str(unicode_tmp_path)}) == unicode_tmp_path / "models"
    checkout = unicode_tmp_path / "checkout"
    monkeypatch.setattr(models.paths, "repo_root", lambda start=None: checkout)
    assert default_models_dir({}) == unicode_tmp_path / "gbf-data" / "models"

    def no_checkout(start=None):
        raise models.paths.RepoRootNotFoundError("installed, not a checkout")

    monkeypatch.setattr(models.paths, "repo_root", no_checkout)
    with pytest.raises(ModelsDirError, match="set GBF_MODELS or DATA_ROOT"):
        default_models_dir({})


def test_models_dir_inside_the_checkout_is_refused(unicode_tmp_path: Path) -> None:
    with pytest.raises(ModelsDirError, match="inside the gbf checkout"):
        check_models_dir(unicode_tmp_path / "weights", repo_root=unicode_tmp_path)
    assert (
        check_models_dir(unicode_tmp_path / "w", repo_root=outside(unicode_tmp_path))
        == (unicode_tmp_path / "w").resolve()
    )


# -------------------------------------------------------------------------- fetching


def test_fetch_downloads_verifies_and_extracts(unicode_tmp_path: Path) -> None:
    pin = make_pin()
    opener = FakeOpener()
    result = fetch(
        pin.model_id,
        unicode_tmp_path,
        pins=[pin],
        opener=opener,
        repo_root=outside(unicode_tmp_path),
        now=lambda: "2026-10-21T00:00:00Z",
    )
    files = model_files(pin, unicode_tmp_path)
    assert result.status == "downloaded"
    assert opener.requests == [pin.url]
    assert opener.responses[0].closed
    assert files.onnx_path.read_bytes() == NETWORK
    assert files.zip_path.read_bytes() == ZIP
    record = json.loads(files.record_path.read_bytes().decode("utf-8"))
    assert record["onnx_sha256"] == pin.onnx_sha256
    assert record["zip_sha256"] == pin.zip_sha256
    assert record["verified_at"] == "2026-10-21T00:00:00Z"
    assert sorted(p.name for p in files.directory.iterdir()) == sorted(
        [files.zip_path.name, files.onnx_path.name, files.record_path.name]
    )
    # A second fetch finds the verified network and never opens a connection.
    again = fetch(
        pin.model_id,
        unicode_tmp_path,
        pins=[pin],
        opener=NoNetwork(),
        repo_root=outside(unicode_tmp_path),
    )
    assert again.status == "present"
    assert verify(pin.model_id, unicode_tmp_path, pins=[pin]) == files.onnx_path


def test_fetch_extracts_from_a_verified_zip_without_downloading(unicode_tmp_path: Path) -> None:
    pin = make_pin()
    files = model_files(pin, unicode_tmp_path)
    files.directory.mkdir(parents=True)
    files.zip_path.write_bytes(ZIP)
    result = fetch(
        pin.model_id,
        unicode_tmp_path,
        pins=[pin],
        opener=NoNetwork(),
        repo_root=outside(unicode_tmp_path),
    )
    assert result.status == "extracted"
    assert files.onnx_path.read_bytes() == NETWORK


def test_fetch_refuses_a_download_that_does_not_match(unicode_tmp_path: Path) -> None:
    pin = make_pin()
    for opener, message in (
        (FakeOpener(build_zip(NETWORK + b"x")), "sha256|bytes"),  # other bytes
        (FakeOpener(ZIP + b"trailing"), "more than"),  # longer than pinned
        (FakeOpener(ZIP[:100]), "100 bytes"),  # truncated
    ):
        with pytest.raises(ModelVerificationError, match=message):
            fetch(
                pin.model_id,
                unicode_tmp_path,
                pins=[pin],
                opener=opener,
                repo_root=outside(unicode_tmp_path),
            )
        directory = model_files(pin, unicode_tmp_path).directory
        assert list(directory.iterdir()) == []  # nothing kept, no .part left behind


def test_fetch_refuses_a_network_that_does_not_match(unicode_tmp_path: Path) -> None:
    pin = make_pin(onnx_sha256="0" * 64)
    with pytest.raises(ModelVerificationError, match="does not match the pinned"):
        fetch(
            pin.model_id,
            unicode_tmp_path,
            pins=[pin],
            opener=FakeOpener(),
            repo_root=outside(unicode_tmp_path),
        )
    files = model_files(pin, unicode_tmp_path)
    assert not files.onnx_path.exists()
    assert not files.onnx_path.with_name(files.onnx_path.name + ".part").exists()
    wrong_member = make_pin(onnx_member=f"elsewhere/{STEM}/end2end.onnx")
    files.zip_path.unlink()
    with pytest.raises(ModelVerificationError, match="no such member"):
        fetch(
            wrong_member.model_id,
            unicode_tmp_path,
            pins=[wrong_member],
            opener=FakeOpener(),
            repo_root=outside(unicode_tmp_path),
        )


def test_fetch_reports_http_errors(unicode_tmp_path: Path) -> None:
    pin = make_pin()
    with pytest.raises(ModelDownloadError, match="HTTP 404"):
        fetch(
            pin.model_id,
            unicode_tmp_path,
            pins=[pin],
            opener=FakeOpener(status=404),
            repo_root=outside(unicode_tmp_path),
        )


def test_fetch_never_replaces_a_mismatched_file_without_force(unicode_tmp_path: Path) -> None:
    pin = make_pin()
    files = model_files(pin, unicode_tmp_path)
    files.directory.mkdir(parents=True)
    files.onnx_path.write_bytes(b"tampered")
    with pytest.raises(ModelVerificationError, match="without --force"):
        fetch(
            pin.model_id,
            unicode_tmp_path,
            pins=[pin],
            opener=NoNetwork(),
            repo_root=outside(unicode_tmp_path),
        )
    assert files.onnx_path.read_bytes() == b"tampered"
    result = fetch(
        pin.model_id,
        unicode_tmp_path,
        pins=[pin],
        opener=FakeOpener(),
        force=True,
        repo_root=outside(unicode_tmp_path),
    )
    assert result.status == "downloaded"
    assert files.onnx_path.read_bytes() == NETWORK


def test_fetch_refuses_a_malformed_pin_before_any_access(unicode_tmp_path: Path) -> None:
    pin = make_pin(zip_sha256="not-a-hash")
    with pytest.raises(ModelVerificationError, match="malformed pin"):
        fetch(pin.model_id, unicode_tmp_path, pins=[pin], opener=NoNetwork())


# ------------------------------------------------------------------------ verifying


def test_verify_lists_missing_and_mismatched_files(unicode_tmp_path: Path) -> None:
    pin = make_pin()
    with pytest.raises(ModelMissingError, match="models fetch"):
        verify(pin.model_id, unicode_tmp_path, pins=[pin])
    files = model_files(pin, unicode_tmp_path)
    files.directory.mkdir(parents=True)
    files.onnx_path.write_bytes(NETWORK[:-1] + b"?")
    files.zip_path.write_bytes(b"short")
    with pytest.raises(ModelVerificationError) as error:
        verify(pin.model_id, unicode_tmp_path, pins=[pin])
    assert len(error.value.problems) == 2
    files.onnx_path.write_bytes(NETWORK)
    files.zip_path.unlink()
    assert verify(pin.model_id, unicode_tmp_path, pins=[pin]) == files.onnx_path


def test_onnx_path_without_hashing_still_checks_presence_and_size(unicode_tmp_path) -> None:
    with pytest.raises(ModelMissingError):
        onnx_path("balanced", unicode_tmp_path, verify_hash=False)
    pin = resolve_pin("balanced")
    path = model_files(pin, unicode_tmp_path).onnx_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"too short")
    with pytest.raises(ModelVerificationError, match="bytes, pinned"):
        onnx_path("balanced", unicode_tmp_path, verify_hash=False)


# ------------------------------------------------------------------------------- CLI


def test_cli_list_and_verify(unicode_tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(models.paths, "repo_root", lambda start=None: outside(unicode_tmp_path))
    assert models.main(["--models-dir", str(unicode_tmp_path), "list"]) == 0
    out = capsys.readouterr().out
    for pin in PINS:
        assert pin.model_id in out
        assert pin.onnx_sha256 in out
    assert "local: missing" in out
    assert models.main(["--models-dir", str(unicode_tmp_path), "verify", "balanced"]) == 1
    assert "failed: rtmpose-m-halpe26-256x192" in capsys.readouterr().err
    assert models.main(["--models-dir", str(unicode_tmp_path), "verify", "nonsense"]) == 2


def test_cli_refuses_a_models_dir_inside_the_checkout(unicode_tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(models.paths, "repo_root", lambda start=None: unicode_tmp_path)
    assert models.main(["--models-dir", str(unicode_tmp_path / "m"), "list"]) == 2
    assert "inside the gbf checkout" in capsys.readouterr().err


def test_pipeline_pose_tags_are_the_registry_restrictions() -> None:
    # Every run whose pose comes from a pinned model carries pipeline.POSE_MODEL_RESTRICTIONS;
    # they must be the registry's restrictions of the model family the weights belong to.
    from gbf import pipeline
    from gbf.measure import default_registry_path

    with open(default_registry_path().with_name("models.toml"), "rb") as handle:
        registry = tomllib.load(handle)
    family = {v: m for m in registry["models"] for v in m.get("variants", ())}
    pinned = [model_id for model_id in pipeline.POSE_MODELS.values() if model_id is not None]
    assert pinned
    for model_id in pinned:
        assert model_id in family, model_id
        assert tuple(family[model_id]["restrictions"]) == pipeline.POSE_MODEL_RESTRICTIONS
