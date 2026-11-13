"""``python -m gbf fetch``: download pinned open data and verify every byte.

Rules (ADR 0002):

- Only sources pinned in ``pins.toml`` are downloaded, from URLs that name the
  pinned revision. A download is accepted only if its size and git blob id
  (sha1 of ``b"blob <size>\\0" + content``) equal the pin; its sha256 is then
  recorded next to the file in ``<file>.fetch.json``.
- Files are written only under the download root, ``$DATA_ROOT/raw`` by default.
  A root inside the gbf repository, or inside any other git work tree, is refused
  before anything is downloaded.
- StatsBomb sources are refused unless ``--accept-statsbomb-terms`` is given;
  without it the command prints the agreement URL and a summary of the clauses
  the project relies on, and downloads nothing.
- Writes are atomic (``.part`` file, then rename); an existing file that differs
  from its pin is never overwritten without ``--force``.

Network access is only the HTTP GET of the pinned URLs, through stdlib urllib.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import gbf
from gbf import contracts
from gbf.events._meta import is_within, same_path, utc_timestamp
from gbf.events.pins import Pin, PinSet, Terms, load_pins
from gbf.util import hashing, paths
from gbf.util.io import write_json

DATA_ROOT_ENV = "DATA_ROOT"
RAW_DIRNAME = "raw"
RECORD_SUFFIX = ".fetch.json"
USER_AGENT = f"gbf-fetch/{gbf.__version__}"
DEFAULT_TIMEOUT_S = 60.0
_CHUNK = 1 << 20


class FetchError(RuntimeError):
    """A fetch was refused or failed. ``problems`` lists every reason found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("\n".join(problems))


class TermsNotAcceptedError(FetchError):
    """A selected source needs terms the caller has not accepted."""


class DestinationError(FetchError):
    """The download root lies inside a git work tree."""


class VerificationError(FetchError):
    """Downloaded or existing bytes do not match their pin."""


class Opener(Protocol):
    def open(self, request: urllib.request.Request, timeout: float = ...) -> Any: ...


def _build_opener() -> Opener:
    """The real HTTP opener (stdlib urllib, follows redirects). Tests replace it."""
    return urllib.request.build_opener()


@dataclass(frozen=True)
class FetchResult:
    """Outcome for one pin: ``downloaded`` or ``present`` (already there and verified)."""

    pin: str
    status: str
    local_relpath: str
    bytes: int
    sha256: str


# ------------------------------------------------------------------------------ terms


def terms_notice(terms: Terms, pin_names: Iterable[str]) -> str:
    """Text shown when a source needing ``terms`` was requested without acceptance."""
    lines = [
        f"Refusing to download {terms.key} data without --accept-{terms.key}-terms.",
        f"Pins that need it: {', '.join(sorted(pin_names))}.",
        f"Agreement: {terms.title}",
        f"  {terms.agreement_url}",
        "Summary of the clauses this project relies on (not a substitute for the agreement):",
        *(f"  {clause}" for clause in terms.clauses),
        f"Re-run with --accept-{terms.key}-terms only after you have read and accepted it.",
    ]
    return "\n".join(lines)


def check_terms(selected: Sequence[Pin], pin_set: PinSet, accepted: Iterable[str]) -> None:
    """Raise :class:`TermsNotAcceptedError` if a selected pin needs unaccepted terms."""
    accepted_set = set(accepted)
    missing: dict[str, list[str]] = {}
    for pin in selected:
        if pin.terms is not None and pin.terms not in accepted_set:
            missing.setdefault(pin.terms, []).append(pin.name)
    if missing:
        raise TermsNotAcceptedError(
            [terms_notice(pin_set.terms[key], names) for key, names in sorted(missing.items())]
        )


# ------------------------------------------------------------------------ destination


def _enclosing_git_work_tree(path: Path, ceiling: Path | None) -> Path | None:
    """Nearest ``path`` or parent holding a ``.git`` entry, searching below ``ceiling``."""
    for candidate in [path, *path.parents]:
        if ceiling is not None and (
            same_path(candidate, ceiling) or not is_within(candidate, ceiling)
        ):
            return None
        if (candidate / ".git").exists():
            return candidate
    return None


def check_destination(
    root: str | Path,
    *,
    repo_root: str | Path | None = None,
    git_ceiling: str | Path | None = None,
) -> Path:
    """Resolve the download root and refuse it when it lies inside a git work tree.

    Refused are the gbf repository (``repo_root``; by default the checkout this
    code runs from, when there is one) and any directory that has a ``.git``
    entry in itself or a parent. Raw provider data must never sit where
    ``git add`` could pick it up. ``git_ceiling`` stops the ``.git`` search before
    that directory, like ``GIT_CEILING_DIRECTORIES``; it exists for tests, which
    must keep their scratch files inside the repository.

    Nothing is created here. Raises :class:`DestinationError` listing every reason.
    """
    resolved = Path(os.path.abspath(root)).resolve()
    problems = []
    repos: list[Path] = []
    if repo_root is not None:
        repos.append(Path(repo_root).resolve())
    else:
        try:
            repos.append(paths.repo_root())
        except paths.RepoRootNotFoundError:
            pass
    for repo in repos:
        if is_within(resolved, repo):
            problems.append(
                f"destination {resolved} is inside the gbf repository {repo}; "
                f"set {DATA_ROOT_ENV} (or --dest) to a directory outside it"
            )
    ceiling = Path(git_ceiling).resolve() if git_ceiling is not None else None
    work_tree = _enclosing_git_work_tree(resolved, ceiling)
    if work_tree is not None and not any(same_path(work_tree, repo) for repo in repos):
        problems.append(
            f"destination {resolved} is inside the git work tree {work_tree}; "
            "raw provider data must stay outside every repository"
        )
    if problems:
        raise DestinationError(problems)
    return resolved


# --------------------------------------------------------------------------- download


def verify_bytes(pin: Pin, data: bytes) -> list[str]:
    """Problems with ``data`` against ``pin`` (size and git blob id); empty if it matches."""
    problems = []
    if len(data) != pin.bytes:
        problems.append(f"{pin.name}: {len(data)} bytes, pinned {pin.bytes}")
    blob = hashing.git_blob_sha1(data)
    if blob != pin.git_blob_sha1:
        problems.append(f"{pin.name}: git blob {blob} does not match pinned {pin.git_blob_sha1}")
    return problems


def fetch_record(pin: Pin, sha256: str, terms: Terms | None, fetched_at: str) -> dict[str, Any]:
    """The ``<file>.fetch.json`` record for a verified file.

    Validated against ``urn:gbf:schema:align#/$defs/fetch_record``; ``fetched_at`` is
    when the bytes were verified (RFC 3339 UTC).
    """
    record: dict[str, Any] = {
        "schema_version": 1,
        "pin": pin.name,
        "role": pin.role,
        "host": pin.host,
        "repository": pin.repository,
        "revision": pin.revision,
        "path": pin.path,
        "url": pin.url,
        "local_relpath": pin.local_relpath,
        "bytes": pin.bytes,
        "git_blob_sha1": pin.git_blob_sha1,
        "sha256": sha256,
        "fetched_at": fetched_at,
        "restrictions": list(pin.restrictions),
        "terms": None
        if terms is None
        else {"key": terms.key, "agreement_url": terms.agreement_url, "accepted": True},
    }
    problems = contracts.def_problems(record, "align", "fetch_record")
    if problems:
        raise FetchError([f"{pin.name}: invalid fetch record: {p}" for p in problems])
    return record


def _read_capped(response: Any, limit: int) -> bytes:
    """Read at most ``limit + 1`` bytes, so an oversized body is detected, not stored."""
    chunks = []
    total = 0
    while total <= limit:
        chunk = response.read(min(_CHUNK, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _http_get(pin: Pin, opener: Opener, timeout: float) -> bytes:
    request = urllib.request.Request(pin.url, headers={"User-Agent": USER_AGENT})
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        raise FetchError([f"{pin.name}: HTTP {error.code} for {pin.url}"]) from error
    except (urllib.error.URLError, OSError) as error:
        raise FetchError([f"{pin.name}: cannot fetch {pin.url}: {error}"]) from error
    try:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        if status != 200:
            raise FetchError([f"{pin.name}: HTTP {status} for {pin.url}"])
        return _read_capped(response, pin.bytes)
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()


def download_pin(
    pin: Pin,
    root: Path,
    *,
    opener: Opener,
    terms: Terms | None,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT_S,
    now: Callable[[], str] = utc_timestamp,
) -> FetchResult:
    """Download one pin into ``root`` (already checked) and verify it before writing.

    An existing file that matches the pin is kept (status ``present``). One that
    does not match raises :class:`VerificationError` unless ``force`` is set.
    """
    target = root.joinpath(*PurePosixPath(pin.local_relpath).parts)
    record_path = target.with_name(target.name + RECORD_SUFFIX)
    if target.exists():
        existing = target.read_bytes()
        problems = verify_bytes(pin, existing)
        if not problems:
            sha256 = hashing.sha256_bytes(existing)
            if not record_path.exists():
                write_json(record_path, fetch_record(pin, sha256, terms, now()))
            return FetchResult(pin.name, "present", pin.local_relpath, len(existing), sha256)
        if not force:
            raise VerificationError(
                [
                    f"{pin.name}: the existing local copy does not match the pin; "
                    "refusing to overwrite it without --force",
                    *problems,
                ]
            )
    data = _http_get(pin, opener, timeout)
    problems = verify_bytes(pin, data)
    if problems:
        raise VerificationError(problems)
    sha256 = hashing.sha256_bytes(data)
    record = fetch_record(pin, sha256, terms, now())  # validated before anything is written
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        part.write_bytes(data)
        os.replace(part, target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    write_json(record_path, record)
    return FetchResult(pin.name, "downloaded", pin.local_relpath, len(data), sha256)


def fetch(
    names: Sequence[str],
    *,
    root: str | Path,
    accepted_terms: Iterable[str] = (),
    pin_set: PinSet | None = None,
    opener: Opener | None = None,
    repo_root: str | Path | None = None,
    git_ceiling: str | Path | None = None,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> list[FetchResult]:
    """Fetch the named pins (all pins when ``names`` is empty) into ``root``.

    Order of checks: unknown names, terms, destination; only then any network
    access. Pins are fetched in ``pins.toml`` order.
    """
    pin_set = pin_set if pin_set is not None else load_pins()
    known = {pin.name for pin in pin_set.pins}
    unknown = [name for name in names if name not in known]
    if unknown:
        listing = ", ".join(sorted(known))
        raise FetchError([f"unknown pin {name!r}; known: {listing}" for name in unknown])
    wanted = set(names)
    selected = [pin for pin in pin_set.pins if not wanted or pin.name in wanted]
    check_terms(selected, pin_set, accepted_terms)
    resolved = check_destination(root, repo_root=repo_root, git_ceiling=git_ceiling)
    active = opener if opener is not None else _build_opener()
    return [
        download_pin(
            pin,
            resolved,
            opener=active,
            terms=pin_set.terms.get(pin.terms) if pin.terms else None,
            force=force,
            timeout=timeout,
        )
        for pin in selected
    ]


# -------------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gbf fetch",
        description="Download pinned open data (pins.toml) and verify it byte for byte.",
    )
    parser.add_argument("names", nargs="*", help="pin names to fetch (default: all pins)")
    parser.add_argument("--list", action="store_true", help="list the pins and exit")
    parser.add_argument(
        "--data-root",
        help=f"data root (default: ${DATA_ROOT_ENV}); files go to <data-root>/{RAW_DIRNAME}/...",
    )
    parser.add_argument("--dest", help="download root, overriding <data-root>/raw")
    parser.add_argument(
        "--accept-statsbomb-terms",
        action="store_true",
        help="confirm that you have read and accepted the StatsBomb user agreement",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace a local copy that does not match its pin"
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds")
    return parser


def _list(pin_set: PinSet) -> None:
    for pin in pin_set.pins:
        terms = f", terms: {pin.terms}" if pin.terms else ""
        print(
            f"{pin.name}\n  {pin.host}:{pin.repository}@{pin.revision} {pin.path}\n"
            f"  {pin.bytes} bytes, git blob {pin.git_blob_sha1}, "
            f"restrictions: {', '.join(pin.restrictions)}{terms}"
        )


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    pin_set = load_pins()
    if args.list:
        _list(pin_set)
        return 0
    known = sorted(pin.name for pin in pin_set.pins)
    unknown = [name for name in args.names if name not in known]
    if unknown:
        for name in unknown:
            print(f"error: unknown pin {name!r}; known: {', '.join(known)}", file=sys.stderr)
        return 2
    if args.dest:
        root = Path(args.dest)
    else:
        data_root = args.data_root or os.environ.get(DATA_ROOT_ENV)
        if not data_root:
            print(f"error: set {DATA_ROOT_ENV} or pass --data-root / --dest", file=sys.stderr)
            return 2
        root = Path(data_root) / RAW_DIRNAME
    accepted = ["statsbomb"] if args.accept_statsbomb_terms else []
    try:
        results = fetch(
            args.names,
            root=root,
            accepted_terms=accepted,
            pin_set=pin_set,
            force=args.force,
            timeout=args.timeout,
        )
    except (TermsNotAcceptedError, DestinationError) as error:
        print("\n".join(error.problems), file=sys.stderr)
        return 2
    except FetchError as error:
        print("fetch failed:\n  " + "\n  ".join(error.problems), file=sys.stderr)
        return 1
    for result in results:
        print(
            f"{result.status}: {result.pin} ({result.bytes} bytes, git blob verified, "
            f"sha256 {result.sha256}) -> <download root>/{result.local_relpath}"
        )
    return 0
