"""Pinned pose-model weights, kept outside the repository: ``python -m gbf models``.

Commands: ``list`` (the pins and what is on disk), ``fetch [model ...]`` (download, verify,
extract) and ``verify [model ...]`` (re-hash what is on disk). A model is named by its
``model_id`` or by its rtmlib mode alias (``performance``, ``balanced``).

What is pinned
    :data:`PINS` holds the two RTMPose Halpe-26 ONNX models that rtmlib's ``BodyWithFeet``
    solution downloads for its "performance" and "balanced" modes (rtmlib 0.0.16): the
    upstream zip URL on download.openmmlab.com, the zip's size and sha256, the zip member
    that holds the network (``.../end2end.onnx``) and that file's size and sha256. All
    hashes were computed on 2026-10-21 from a download whose MD5 matched the server's
    ``Content-MD5`` header. The input sizes come from the ONNX graphs themselves; the
    ``pipeline.json`` inside the 384x288 zip still says 192x256, a stale export setting.

Licences (engineering judgement, not legal advice)
    rtmlib, which defines these URLs and the preprocessing, is Apache-2.0, as is mmpose,
    whose RTMPose models these are. The weights were trained on the "body7" collection of
    public human-pose datasets; their licences have not been verified by this project. The
    models are therefore experimental and for private use only: nothing derived from them
    may be published until that is resolved (see registries/models.toml).

Where the files go
    ``<models dir>/<model_id>/`` holds the verified upstream zip, the extracted network
    ``<model_id>.onnx`` and a ``<model_id>.fetch.json`` record. The models dir is the first
    of: the ``--models-dir`` argument, ``$GBF_MODELS``, ``$DATA_ROOT/models``, and
    ``gbf-data/models`` next to the gbf checkout (the directory that holds the checkout's
    folder). A models dir inside the gbf checkout is refused: weights are never committed.

Network access is only the HTTP GET of a pinned URL (stdlib urllib) during ``fetch``, and
only for a model whose verified files are not already on disk. Every byte is checked
before it is used: a zip or ONNX file whose size or sha256 differs from its pin is refused,
and an existing file that does not match is never replaced without ``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol
from urllib.parse import urlsplit

import gbf
from gbf.util import paths
from gbf.util.io import write_json

MODELS_ENV = "GBF_MODELS"
"""Environment variable naming the models dir (highest priority after --models-dir)."""

DATA_ROOT_ENV = "DATA_ROOT"
"""Environment variable naming the data root; the models dir is then ``$DATA_ROOT/models``."""

SIBLING_DATA_DIRNAME = "gbf-data"
"""Data folder next to the gbf checkout used when neither variable is set."""

MODELS_DIRNAME = "models"
RECORD_SUFFIX = ".fetch.json"
USER_AGENT = f"gbf-models/{gbf.__version__}"
DEFAULT_TIMEOUT_S = 60.0
"""Socket timeout for the download, seconds (per blocking operation, not in total)."""

PINNED_HOST = "download.openmmlab.com"
_CHUNK = 1 << 20
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$")
_SIZE_TOKEN = re.compile(r"-(\d+)x(\d+)-")

LICENCE_NOTE = (
    "Runtime rtmlib (Apache-2.0) and model code mmpose RTMPose (Apache-2.0). The weights "
    "were trained on the 'body7' collection of public human-pose datasets, whose licences "
    "this project has not verified: experimental, private use only."
)


@dataclass(frozen=True)
class ModelPin:
    """One pinned model: where it comes from and the exact bytes expected.

    Attributes:
        model_id: stable id used in file names and in ``pose.parquet``'s ``model_id``.
        alias: rtmlib ``BodyWithFeet`` mode that uses this model ("performance", ...).
        keypoints: keypoint layout of the network output ("halpe26").
        input_size: network input (width, height) in pixels, read from the ONNX graph;
            rtmlib's ``model_input_size`` takes the same (width, height) order.
        url: upstream zip URL (https, on download.openmmlab.com).
        zip_bytes, zip_sha256: size in bytes and sha256 (lowercase hex) of the zip.
        onnx_member: path of the network inside the zip (POSIX, as zipfile lists it).
        onnx_bytes, onnx_sha256: size and sha256 of that member once extracted.
        pinned_on: date (YYYY-MM-DD) the hashes were computed from a download.
        status: registry status; "experimental" until the training-data licences are
            verified (registries/models.toml conventions).
        licence: licence note for code, weights and training data.
    """

    model_id: str
    alias: str
    keypoints: str
    input_size: tuple[int, int]
    url: str
    zip_bytes: int
    zip_sha256: str
    onnx_member: str
    onnx_bytes: int
    onnx_sha256: str
    pinned_on: str
    status: str = "experimental"
    licence: str = LICENCE_NOTE

    @property
    def zip_name(self) -> str:
        """File name of the zip: the last segment of the URL path."""
        return PurePosixPath(urlsplit(self.url).path).name

    @property
    def onnx_name(self) -> str:
        """File name of the extracted network inside ``<models dir>/<model_id>/``."""
        return f"{self.model_id}.onnx"


PINS: tuple[ModelPin, ...] = (
    ModelPin(
        model_id="rtmpose-x-halpe26-384x288",
        alias="performance",
        keypoints="halpe26",
        input_size=(288, 384),
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip"
        ),
        zip_bytes=186_431_237,
        zip_sha256="0c33d9ba030e7011686e6a7f48fd85df383378216f2f5765ae389c1828685e68",
        onnx_member=(
            "20230831/rtmpose_onnx/"
            "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606/end2end.onnx"
        ),
        onnx_bytes=199_918_283,
        onnx_sha256="21bdaefd1a9b7934160987e754d8eca191bc44cccbfa3ba2ef71142c6dd05a9a",
        pinned_on="2026-10-21",
    ),
    ModelPin(
        model_id="rtmpose-m-halpe26-256x192",
        alias="balanced",
        keypoints="halpe26",
        input_size=(192, 256),
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip"
        ),
        zip_bytes=52_089_522,
        zip_sha256="55b81170e236040b59fc792ad0a8315301ac4c079a3bdb1095d838aad3088d18",
        onnx_member=(
            "20230831/rtmpose_onnx/"
            "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605/end2end.onnx"
        ),
        onnx_bytes=55_685_444,
        onnx_sha256="26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf",
        pinned_on="2026-10-21",
    ),
)

DEFAULT_MODEL_ID = PINS[0].model_id
"""The model the pose stage uses unless told otherwise (rtmpose-x, rtmlib "performance")."""


# --------------------------------------------------------------------------- errors


class ModelError(RuntimeError):
    """A models command was refused or failed. ``problems`` lists every reason found."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


class UnknownModelError(ModelError):
    """A model name is neither a pinned model_id nor an alias."""


class ModelsDirError(ModelError):
    """The models dir cannot be determined or lies inside the gbf checkout."""


class ModelDownloadError(ModelError):
    """The HTTP download failed (status, connection or timeout)."""


class ModelVerificationError(ModelError):
    """Bytes on disk or from the network do not match their pin."""


class ModelMissingError(ModelError):
    """A model's extracted ONNX file is not on disk."""


class Opener(Protocol):
    """What fetch needs from urllib's opener; tests pass a fake one (no network)."""

    def open(self, request: urllib.request.Request, timeout: float = ...) -> Any: ...


def _build_opener() -> Opener:
    """The real HTTP opener (stdlib urllib, follows redirects)."""
    return urllib.request.build_opener()


# ---------------------------------------------------------------------- pin table


def pin_problems(pin: ModelPin) -> list[str]:
    """Every inconsistency of one pin (formats, URL, names); empty when it is well formed."""
    problems: list[str] = []
    name = pin.model_id or "<empty model_id>"
    if not _MODEL_ID.match(pin.model_id):
        problems.append(f"{name}: model_id must be lowercase letters, digits, '.' and '-'")
    if not pin.alias or pin.alias == pin.model_id:
        problems.append(f"{name}: alias must be non-empty and differ from the model_id")
    parts = urlsplit(pin.url)
    if parts.scheme != "https" or parts.hostname != PINNED_HOST:
        problems.append(f"{name}: url must be https on {PINNED_HOST}, got {pin.url!r}")
    if parts.query or parts.fragment:
        problems.append(f"{name}: url must not carry a query or fragment")
    if not pin.zip_name.endswith(".zip"):
        problems.append(f"{name}: url must name a .zip file, got {pin.zip_name!r}")
    for label, digest in (("zip_sha256", pin.zip_sha256), ("onnx_sha256", pin.onnx_sha256)):
        if not _SHA256.match(digest):
            problems.append(f"{name}: {label} must be 64 lowercase hex digits")
    for label, size in (("zip_bytes", pin.zip_bytes), ("onnx_bytes", pin.onnx_bytes)):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            problems.append(f"{name}: {label} must be a positive integer, got {size!r}")
    member = PurePosixPath(pin.onnx_member)
    stem = pin.zip_name.removesuffix(".zip")
    if member.name != "end2end.onnx" or stem not in member.parts or ".." in member.parts:
        problems.append(
            f"{name}: onnx_member must be <...>/{stem}/end2end.onnx, got {pin.onnx_member!r}"
        )
    if pin.onnx_member.startswith("/") or "\\" in pin.onnx_member:
        problems.append(f"{name}: onnx_member must be a relative POSIX path")
    size = pin.input_size
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size)
    ):
        problems.append(f"{name}: input_size must be (width, height) positive ints")
    else:
        token = _SIZE_TOKEN.search(pin.zip_name)
        if token is None or (int(token.group(2)), int(token.group(1))) != size:
            problems.append(
                f"{name}: input_size {size} (width, height) disagrees with the HxW token"
                f" of {pin.zip_name!r}"
            )
    if not _DATE.match(pin.pinned_on):
        problems.append(f"{name}: pinned_on must be YYYY-MM-DD, got {pin.pinned_on!r}")
    if pin.status not in {"planned", "experimental", "research_only", "blocked", "plugin_only"}:
        problems.append(f"{name}: unknown status {pin.status!r}")
    if not pin.licence.strip():
        problems.append(f"{name}: licence note must not be empty")
    return problems


def table_problems(pins: Sequence[ModelPin] = PINS) -> list[str]:
    """Problems of every pin plus duplicate ids, aliases, URLs and hashes across pins."""
    problems = [problem for pin in pins for problem in pin_problems(pin)]
    names: dict[str, int] = {}
    for pin in pins:
        for value in (pin.model_id, pin.alias):
            names[value] = names.get(value, 0) + 1
    problems += [f"name {value!r} is used {n} times" for value, n in names.items() if n > 1]
    for label, values in (
        ("url", [pin.url for pin in pins]),
        ("zip_sha256", [pin.zip_sha256 for pin in pins]),
        ("onnx_sha256", [pin.onnx_sha256 for pin in pins]),
    ):
        repeated = sorted({v for v in values if values.count(v) > 1})
        problems += [f"{label} {value} is pinned twice" for value in repeated]
    return problems


def resolve_pin(name: str, pins: Sequence[ModelPin] = PINS) -> ModelPin:
    """The pin whose model_id or alias is ``name``; UnknownModelError lists the known names."""
    for pin in pins:
        if name in (pin.model_id, pin.alias):
            return pin
    known = ", ".join(f"{pin.model_id} ({pin.alias})" for pin in pins)
    raise UnknownModelError([f"unknown model {name!r}; known: {known}"])


# ------------------------------------------------------------------------ locations


@dataclass(frozen=True)
class ModelFiles:
    """Where one model's files live: ``<models dir>/<model_id>/``."""

    directory: Path
    zip_path: Path
    onnx_path: Path
    record_path: Path


def model_files(pin: ModelPin, models_dir: str | Path) -> ModelFiles:
    """File locations of ``pin`` under ``models_dir`` (nothing is created or checked)."""
    directory = Path(models_dir) / pin.model_id
    return ModelFiles(
        directory=directory,
        zip_path=directory / pin.zip_name,
        onnx_path=directory / pin.onnx_name,
        record_path=directory / f"{pin.model_id}{RECORD_SUFFIX}",
    )


def default_models_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The models dir when none is given: ``$GBF_MODELS``, ``$DATA_ROOT/models``, sibling.

    The sibling default is ``<parent of the gbf checkout>/gbf-data/models``. Raises
    ModelsDirError when no variable is set and the code does not run from a checkout.
    """
    env = os.environ if environ is None else environ
    if env.get(MODELS_ENV):
        return Path(env[MODELS_ENV])
    if env.get(DATA_ROOT_ENV):
        return Path(env[DATA_ROOT_ENV]) / MODELS_DIRNAME
    try:
        checkout = paths.repo_root()
    except paths.RepoRootNotFoundError as error:
        raise ModelsDirError(
            [f"set {MODELS_ENV} or {DATA_ROOT_ENV}: gbf does not run from a checkout ({error})"]
        ) from error
    return checkout.parent / SIBLING_DATA_DIRNAME / MODELS_DIRNAME


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def check_models_dir(
    models_dir: str | Path | None = None, *, repo_root: str | Path | None = None
) -> Path:
    """Resolve the models dir (default: :func:`default_models_dir`) and refuse the checkout.

    ``repo_root`` is the checkout to keep weights out of; by default the one this code runs
    from, if any (tests pass another directory). Nothing is created here.
    """
    chosen = default_models_dir() if models_dir is None else Path(models_dir)
    resolved = Path(os.path.abspath(chosen)).resolve()
    if repo_root is not None:
        checkout: Path | None = Path(repo_root).resolve()
    else:
        try:
            checkout = paths.repo_root()
        except paths.RepoRootNotFoundError:
            checkout = None
    if checkout is not None and _is_within(resolved, checkout):
        raise ModelsDirError(
            [
                f"models dir {resolved} is inside the gbf checkout {checkout}; weights are "
                f"never committed: set {MODELS_ENV} or {DATA_ROOT_ENV} to a directory outside it"
            ]
        )
    return resolved


# --------------------------------------------------------------------- verification


def _hash_file(path: Path, limit: int) -> tuple[int, str]:
    """(size, sha256) of a file, reading at most ``limit + 1`` bytes."""
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while total <= limit:
            chunk = handle.read(min(_CHUNK, limit + 1 - total))
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _mismatches(
    what: str, size: int, sha256: str, expected_size: int, expected_sha256: str
) -> list[str]:
    problems = []
    if size != expected_size:
        shown = f"more than {expected_size}" if size > expected_size else str(size)
        problems.append(f"{what}: {shown} bytes, pinned {expected_size}")
    elif sha256 != expected_sha256:
        problems.append(f"{what}: sha256 {sha256} does not match the pinned {expected_sha256}")
    return problems


def _check_file(path: Path, what: str, size: int, sha256: str) -> list[str]:
    found_size, found_sha = _hash_file(path, size)
    return _mismatches(what, found_size, found_sha, size, sha256)


def verify(
    model: str,
    models_dir: str | Path | None = None,
    *,
    pins: Sequence[ModelPin] = PINS,
    check_zip: bool = True,
) -> Path:
    """Re-hash a fetched model and return the path of its verified ``.onnx`` file.

    The extracted network must exist and match its pin; the zip is checked too when it is
    present and ``check_zip`` is set. Raises ModelMissingError when the network is not on
    disk and ModelVerificationError listing every mismatch.
    """
    pin = resolve_pin(model, pins)
    root = default_models_dir() if models_dir is None else Path(models_dir)
    files = model_files(pin, root)
    if not files.onnx_path.is_file():
        raise ModelMissingError(
            [
                f"{pin.model_id}: {files.onnx_path} is missing; run "
                f"`python -m gbf models fetch {pin.model_id}`"
            ]
        )
    problems = _check_file(
        files.onnx_path, f"{pin.model_id}: {pin.onnx_name}", pin.onnx_bytes, pin.onnx_sha256
    )
    if check_zip and files.zip_path.is_file():
        problems += _check_file(
            files.zip_path, f"{pin.model_id}: {pin.zip_name}", pin.zip_bytes, pin.zip_sha256
        )
    if problems:
        raise ModelVerificationError(problems)
    return files.onnx_path


# -------------------------------------------------------------------------- fetching


@dataclass(frozen=True)
class FetchResult:
    """Outcome for one model.

    status is "present" (verified files were already there, no network), "extracted"
    (a verified zip was already there, the network was extracted from it) or
    "downloaded" (the zip came from the network, then was verified and extracted).
    """

    model_id: str
    status: str
    onnx_path: Path
    zip_sha256: str
    onnx_sha256: str


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_record(pin: ModelPin, verified_at: str) -> dict[str, Any]:
    """The ``<model_id>.fetch.json`` record written after both hashes matched."""
    return {
        "schema_version": 1,
        "model_id": pin.model_id,
        "alias": pin.alias,
        "url": pin.url,
        "zip_name": pin.zip_name,
        "zip_bytes": pin.zip_bytes,
        "zip_sha256": pin.zip_sha256,
        "onnx_member": pin.onnx_member,
        "onnx_name": pin.onnx_name,
        "onnx_bytes": pin.onnx_bytes,
        "onnx_sha256": pin.onnx_sha256,
        "input_size": list(pin.input_size),
        "pinned_on": pin.pinned_on,
        "status": pin.status,
        "licence": pin.licence,
        "verified_at": verified_at,
        "gbf_version": gbf.__version__,
    }


def _copy_capped(source: BinaryIO, part: Path, limit: int) -> tuple[int, str]:
    """Stream at most ``limit + 1`` bytes into ``part``; return (size, sha256)."""
    digest = hashlib.sha256()
    total = 0
    with open(part, "wb") as sink:
        while total <= limit:
            chunk = source.read(min(_CHUNK, limit + 1 - total))
            if not chunk:
                break
            digest.update(chunk)
            sink.write(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _download_zip(pin: ModelPin, target: Path, opener: Opener, timeout: float) -> None:
    """GET the pinned URL into ``target`` (atomic: ``.part`` then rename) after verifying."""
    request = urllib.request.Request(pin.url, headers={"User-Agent": USER_AGENT})
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        raise ModelDownloadError([f"{pin.model_id}: HTTP {error.code} for {pin.url}"]) from error
    except (urllib.error.URLError, OSError) as error:
        raise ModelDownloadError([f"{pin.model_id}: cannot fetch {pin.url}: {error}"]) from error
    part = target.with_name(target.name + ".part")
    try:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        if status != 200:
            raise ModelDownloadError([f"{pin.model_id}: HTTP {status} for {pin.url}"])
        size, sha256 = _copy_capped(response, part, pin.zip_bytes)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()
    problems = _mismatches(
        f"{pin.model_id}: downloaded {pin.zip_name}", size, sha256, pin.zip_bytes, pin.zip_sha256
    )
    if problems:
        part.unlink(missing_ok=True)
        raise ModelVerificationError(problems)
    os.replace(part, target)


def _extract_onnx(pin: ModelPin, zip_path: Path, onnx_path: Path) -> None:
    """Extract exactly the pinned member into ``onnx_path`` after verifying its bytes.

    Only the pinned member is read and it is written under our own file name, so archive
    paths never reach the filesystem.
    """
    part = onnx_path.with_name(onnx_path.name + ".part")
    what = f"{pin.model_id}: {pin.onnx_member} in {pin.zip_name}"
    try:
        with zipfile.ZipFile(zip_path) as archive:
            try:
                info = archive.getinfo(pin.onnx_member)
            except KeyError:
                raise ModelVerificationError([f"{what}: no such member"]) from None
            if info.file_size != pin.onnx_bytes:
                raise ModelVerificationError(
                    [f"{what}: declares {info.file_size} bytes, pinned {pin.onnx_bytes}"]
                )
            with archive.open(info) as source:
                size, sha256 = _copy_capped(source, part, pin.onnx_bytes)
    except zipfile.BadZipFile as error:
        part.unlink(missing_ok=True)
        raise ModelVerificationError([f"{what}: unreadable archive ({error})"]) from error
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    problems = _mismatches(what, size, sha256, pin.onnx_bytes, pin.onnx_sha256)
    if problems:
        part.unlink(missing_ok=True)
        raise ModelVerificationError(problems)
    os.replace(part, onnx_path)


def fetch(
    model: str,
    models_dir: str | Path | None = None,
    *,
    pins: Sequence[ModelPin] = PINS,
    opener: Opener | None = None,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT_S,
    repo_root: str | Path | None = None,
    now: Callable[[], str] = _utc_now,
) -> FetchResult:
    """Make ``<models dir>/<model_id>/<model_id>.onnx`` exist and match its pin.

    Order: resolve the pin and check it is well formed, check the models dir (never inside
    the checkout), then use what is already there if it verifies: the extracted network
    (status "present"), else the zip ("extracted"). Only then is the zip downloaded
    ("downloaded"). Both the zip and the extracted network are verified (size and sha256)
    before they are renamed into place. An existing file that does not match its pin is
    an error unless ``force`` is set, in which case it is replaced.

    Raises UnknownModelError, ModelsDirError, ModelDownloadError or
    ModelVerificationError (each lists every problem it found).
    """
    pin = resolve_pin(model, pins)
    problems = pin_problems(pin)
    if problems:
        raise ModelVerificationError([f"malformed pin: {problem}" for problem in problems])
    root = check_models_dir(models_dir, repo_root=repo_root)
    files = model_files(pin, root)

    if files.onnx_path.is_file():
        problems = _check_file(
            files.onnx_path,
            f"{pin.model_id}: existing {pin.onnx_name}",
            pin.onnx_bytes,
            pin.onnx_sha256,
        )
        if not problems:
            if not files.record_path.is_file():
                write_json(files.record_path, fetch_record(pin, now()))
            return FetchResult(
                pin.model_id, "present", files.onnx_path, pin.zip_sha256, pin.onnx_sha256
            )
        if not force:
            raise ModelVerificationError(
                [*problems, f"{pin.model_id}: refusing to replace it without --force"]
            )

    status = "extracted"
    zip_ok = False
    if files.zip_path.is_file():
        problems = _check_file(
            files.zip_path,
            f"{pin.model_id}: existing {pin.zip_name}",
            pin.zip_bytes,
            pin.zip_sha256,
        )
        if problems and not force:
            raise ModelVerificationError(
                [*problems, f"{pin.model_id}: refusing to replace it without --force"]
            )
        zip_ok = not problems
    files.directory.mkdir(parents=True, exist_ok=True)
    if not zip_ok:
        active = opener if opener is not None else _build_opener()
        _download_zip(pin, files.zip_path, active, timeout)
        status = "downloaded"
    _extract_onnx(pin, files.zip_path, files.onnx_path)
    write_json(files.record_path, fetch_record(pin, now()))
    return FetchResult(pin.model_id, status, files.onnx_path, pin.zip_sha256, pin.onnx_sha256)


def onnx_path(
    model: str = DEFAULT_MODEL_ID,
    models_dir: str | Path | None = None,
    *,
    verify_hash: bool = True,
) -> Path:
    """Path of a fetched model's network, re-hashed first unless ``verify_hash`` is False.

    Without hashing, the file must still exist and have the pinned size.
    """
    if verify_hash:
        return verify(model, models_dir, check_zip=False)
    pin = resolve_pin(model)
    root = default_models_dir() if models_dir is None else Path(models_dir)
    path = model_files(pin, root).onnx_path
    if not path.is_file():
        raise ModelMissingError(
            [f"{pin.model_id}: {path} is missing; run `python -m gbf models fetch {pin.model_id}`"]
        )
    if path.stat().st_size != pin.onnx_bytes:
        raise ModelVerificationError(
            [f"{pin.model_id}: {path.stat().st_size} bytes, pinned {pin.onnx_bytes}"]
        )
    return path


# ------------------------------------------------------------------------------- CLI


def _local_state(pin: ModelPin, root: Path) -> str:
    """Cheap on-disk state for ``list`` (sizes only; ``verify`` re-hashes)."""
    files = model_files(pin, root)
    if not files.onnx_path.is_file():
        return "missing"
    if files.onnx_path.stat().st_size != pin.onnx_bytes:
        return "size mismatch (run verify)"
    return "present (size ok; run verify to re-hash)"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gbf models",
        description="Pinned pose-model weights, kept outside the repository.",
    )
    parser.add_argument(
        "--models-dir",
        help=f"models dir (default: ${MODELS_ENV}, $DATA_ROOT/{MODELS_DIRNAME}, or "
        f"{SIBLING_DATA_DIRNAME}/{MODELS_DIRNAME} next to the checkout)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the pinned models and what is on disk")
    fetch_parser = sub.add_parser("fetch", help="download, verify and extract models")
    fetch_parser.add_argument("models", nargs="*", help="model ids or aliases (default: all)")
    fetch_parser.add_argument(
        "--force", action="store_true", help="replace local files that do not match their pin"
    )
    fetch_parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="socket timeout, seconds"
    )
    verify_parser = sub.add_parser("verify", help="re-hash fetched models")
    verify_parser.add_argument("models", nargs="*", help="model ids or aliases (default: all)")
    return parser


def _selected(names: Sequence[str]) -> list[ModelPin]:
    if not names:
        return list(PINS)
    problems = []
    selected = []
    for name in names:
        try:
            pin = resolve_pin(name)
        except UnknownModelError as error:
            problems += error.problems
            continue
        if pin not in selected:
            selected.append(pin)
    if problems:
        raise UnknownModelError(problems)
    return selected


def main(argv: list[str]) -> int:
    """``python -m gbf models list|fetch|verify``; returns 0, 1 (failure) or 2 (usage)."""
    args = _parser().parse_args(argv)
    try:
        selected = _selected(getattr(args, "models", []))
        root = check_models_dir(args.models_dir)
    except (UnknownModelError, ModelsDirError) as error:
        print("error: " + "\n  ".join(error.problems), file=sys.stderr)
        return 2
    print(f"models dir: {root}")
    if args.command == "list":
        for pin in selected:
            width, height = pin.input_size
            print(
                f"{pin.model_id} ({pin.alias}): {pin.keypoints}, input {width}x{height} (w x h),"
                f" status {pin.status}, pinned {pin.pinned_on}\n"
                f"  {pin.url}\n"
                f"  zip  {pin.zip_bytes} bytes sha256 {pin.zip_sha256}\n"
                f"  onnx {pin.onnx_bytes} bytes sha256 {pin.onnx_sha256}\n"
                f"  local: {_local_state(pin, root)}\n"
                f"  licence: {pin.licence}"
            )
        return 0
    failed = False
    for pin in selected:
        try:
            if args.command == "fetch":
                result = fetch(pin.model_id, root, force=args.force, timeout=args.timeout)
                print(
                    f"{result.status}: {pin.model_id} -> <models dir>/{pin.model_id}/"
                    f"{pin.onnx_name} (onnx sha256 {result.onnx_sha256})"
                )
            else:
                verify(pin.model_id, root)
                print(f"ok: {pin.model_id} (onnx sha256 {pin.onnx_sha256})")
        except ModelError as error:
            failed = True
            print(f"failed: {pin.model_id}\n  " + "\n  ".join(error.problems), file=sys.stderr)
    return 1 if failed else 0
