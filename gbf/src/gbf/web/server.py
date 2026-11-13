"""``python -m gbf web``: a local web page to run the highlight analysis and look at its results.

Usage::

    python -m gbf web [--runs-dir DIR] [--port N] [--no-open]

``--runs-dir`` is the folder whose subfolders are runs (a run is a subfolder with a
run.json); the default is ``$DATA_ROOT/runs`` when DATA_ROOT is set (read as gbf.models
reads it), else the option is required. ``--port 0``, the default, lets the system pick a
free port. The command prints the page's address, which carries a token drawn for this
start, opens it in the default browser unless ``--no-open``, and serves until Ctrl+C,
which also stops a running analysis (docs/design/web.md).

What it serves
    ``/`` and ``/static/<file>``: the page (gbf/web/static: HTML, CSS and JavaScript; no
    external font, script or request). ``/api/...``: JSON (:meth:`WebHandler._api`).
    ``/runs/<run>/<path>``: files of a run (gbf.web.runs.resolve_artifact), with HTTP Range
    support (206 Partial Content) so that videos can seek; never a directory listing.

Security (any web page open in the same browser can send requests to 127.0.0.1)
    - It listens on 127.0.0.1 only; on Windows the port is bound exclusively.
    - Every start draws a random token (secrets.token_urlsafe). The printed address carries
      it in the fragment (``#t=...``), which a browser never sends to a server; the page's
      script reads it and sends it as the ``X-GBF-Token`` header with every API call.
      Images, videos and the report's own links cannot carry a header, so
      ``GET /api/session`` (with the header) also sets an HttpOnly, SameSite=Strict cookie
      for ``/runs/``; a file of a run needs the header or that cookie. Tokens are compared
      in constant time.
    - Host must be ``127.0.0.1:<port>`` or ``localhost:<port>`` (DNS rebinding). A request
      with any other Origin is refused, and so is a request to the API or to a run's files
      that the browser marks as cross-site or same-site (``Sec-Fetch-Site``).
    - JSON POST bodies are UTF-8 and at most :data:`MAX_BODY_BYTES`; authenticated video
      uploads instead accept a length-bounded binary stream.
      No CORS header is ever sent and OPTIONS is refused, so a script of another origin can
      neither read an answer nor send a JSON body.
    - A refused request ends its connection. Its body, when it has a length of at most
      :data:`MAX_BODY_BYTES` and no Transfer-Encoding, is read and dropped first: a
      connection closed with a body unread is reset, and the client can lose the answer.
    - Every answer carries ``X-Content-Type-Options: nosniff``, ``Referrer-Policy:
      no-referrer``, ``Cache-Control: no-store``, ``Cross-Origin-Resource-Policy:
      same-origin``, ``X-Frame-Options: DENY`` and a Content-Security-Policy; a run's files
      get one that allows no script at all.
    - Requests are checked before the file system is touched, and the names a run's path
      resolves to once more after (gbf.web.runs). Reasons go to the console, never into an
      answer, and requests are otherwise not logged.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import secrets
import shutil
import socket
import socketserver
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

import gbf
from gbf.models import DATA_ROOT_ENV
from gbf.pipeline import STAGES
from gbf.web.automatic import MAX_UPLOAD_BYTES, runtime_status
from gbf.web.jobs import (
    FIELD_NAMES_ZH,
    STAGE_NAMES_ZH,
    JobManager,
    JobRefused,
    JobRequest,
    JobRequestError,
    network_path,
)
from gbf.web.runs import SERVED_SUFFIXES, RunPathError, list_runs, resolve_artifact, run_detail

HOST = "127.0.0.1"
DEFAULT_PORT = 0
RUNS_DIRNAME = "runs"
TOKEN_HEADER = "X-GBF-Token"
MAX_BODY_BYTES = 64 * 1024
COPY_CHUNK = 256 * 1024
STATIC_DIR = Path(__file__).with_name("static")
STATIC_FILES: Mapping[str, str] = MappingProxyType(
    {
        "index.html": "text/html; charset=utf-8",
        "app.js": "text/javascript; charset=utf-8",
        "style.css": "text/css; charset=utf-8",
    }
)
"""The page's own files, served from STATIC_DIR (the only names ``/static/`` answers)."""

PAGE_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
    "form-action 'none'"
)
FILE_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; media-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)
"""A run's files: the report's inline CSS, its images and videos, and no script at all."""
API_CSP = "default-src 'none'; frame-ancestors 'none'"

SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("Cache-Control", "no-store"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("X-Frame-Options", "DENY"),
)

PRIVACY_ZH = (
    "只在本机运行：本页由这台电脑上的 gbf 服务提供，只监听 127.0.0.1，不连外网，不加载任何外部"
    "字体或脚本，也不调用大模型或其他在线服务；视频、图片和数据都留在这台电脑上。地址里的访问令牌"
    "只对这次启动有效，请不要分享。"
)
TOKEN_MESSAGE = (
    "缺少或错误的访问令牌：请用终端里打印的完整地址（带 #t=…）打开本页。"
    "服务每次启动都会换一个新令牌。"
)
FOREIGN_MESSAGE = "拒绝访问：这个请求来自别的网页或别的地址。"

_UNSATISFIABLE = (-1, -1)
# At most 18 digits (no file comes near 10**18 bytes); int() refuses more than 4300.
_RANGE = re.compile(r"bytes=(?P<first>[0-9]{0,18})-(?P<last>[0-9]{0,18})")
_LENGTH = re.compile(r"[0-9]{1,10}")


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """The byte span (first, last, inclusive) a ``Range`` header asks for in a file of
    ``size`` bytes; None to answer with the whole file (no header, a header this server
    does not read, such as several ranges or a number of more than 18 digits, or an invalid
    one: RFC 9110 lets a server ignore it); ``(-1, -1)`` when the range cannot be satisfied
    (416)."""
    if header is None:
        return None
    match = _RANGE.fullmatch(header.strip())
    if match is None:
        return None
    first, last = match["first"], match["last"]
    if not first and not last:
        return None
    if not first:  # the last N bytes
        count = int(last)
        if count == 0 or size == 0:
            return _UNSATISFIABLE
        return max(0, size - count), size - 1
    start = int(first)
    if start >= size:
        return _UNSATISFIABLE
    end = size - 1 if not last else min(int(last), size - 1)
    if end < start:
        return None
    return start, end


class _Refusal(Exception):
    """An answer other than the one asked for: ``status``, the Chinese ``message`` (and
    ``problems``) the client sees, and the ``reason`` the console sees."""

    def __init__(
        self,
        status: int,
        message: str,
        reason: str,
        problems: Sequence[str] = (),
        allow: str | None = None,
    ) -> None:
        self.status = status
        self.message = message
        self.reason = reason
        self.problems = list(problems)
        self.allow = allow
        super().__init__(f"{status} {reason}")


def _not_allowed(allow: str) -> _Refusal:
    return _Refusal(405, "不支持这种请求方法。", "method not allowed", allow=allow)


def _console(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


class WebServer(ThreadingHTTPServer):
    """The local server: 127.0.0.1 only, one token per start (see the module docstring).

    ``runs_dir`` must exist. ``jobs`` defaults to a JobManager running
    ``python -m gbf highlights``; ``token`` to a fresh random one; ``log`` (the console by
    default) receives one line per refused request.
    """

    daemon_threads = True
    block_on_close = False
    # On Windows SO_REUSEADDR would let a second server bind the same port; the port is
    # bound exclusively there instead (server_bind).
    allow_reuse_address = os.name != "nt"

    def __init__(
        self,
        runs_dir: Path,
        *,
        port: int = DEFAULT_PORT,
        jobs: JobManager | None = None,
        token: str | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir).resolve(strict=True)
        self.token = token or secrets.token_urlsafe(32)
        self.jobs = jobs if jobs is not None else JobManager(self.runs_dir)
        self.upload_lock = threading.Lock()
        self.log = log or _console
        super().__init__((HOST, port), WebHandler)
        self.port = int(self.server_address[1])
        self.allowed_hosts = frozenset({f"127.0.0.1:{self.port}", f"localhost:{self.port}"})
        self.allowed_origins = frozenset(f"http://{host}" for host in self.allowed_hosts)
        self.cookie_name = f"gbf_web_{self.port}"

    def server_bind(self) -> None:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if os.name == "nt" and exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        socketserver.TCPServer.server_bind(self)
        # http.server would look up the host's name here; nothing needs it.
        self.server_name = HOST
        self.server_port = int(self.server_address[1])

    def handle_error(self, request: Any, client_address: Any) -> None:
        if isinstance(sys.exc_info()[1], ConnectionError | TimeoutError):
            return  # a browser that closed a connection (videos do it all the time)
        super().handle_error(request, client_address)

    @property
    def url(self) -> str:
        """The page's address with this start's token in the fragment."""
        return f"http://{HOST}:{self.port}/#t={self.token}"


class WebHandler(BaseHTTPRequestHandler):
    """One request (see the module docstring for the rules it applies)."""

    server: WebServer
    protocol_version = "HTTP/1.1"
    server_version = "gbf-web"
    sys_version = ""
    timeout = 60
    error_message_format = "%(code)d %(message)s\n"
    error_content_type = "text/plain; charset=utf-8"
    _csp_sent = False
    _headers_done = False
    _body_read = False

    # ---------------------------------------------------------------- entry points

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("HEAD")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def do_PATCH(self) -> None:
        self._dispatch("PATCH")

    def do_OPTIONS(self) -> None:
        self._dispatch("OPTIONS")

    def log_message(self, format: str, *args: Any) -> None:
        return  # requests are not logged; refusals are, by _refuse

    def end_headers(self) -> None:
        for name, value in SECURITY_HEADERS:
            self.send_header(name, value)
        if not self._csp_sent:
            self.send_header("Content-Security-Policy", API_CSP)
        self._csp_sent = False
        super().end_headers()
        self._headers_done = True

    def _dispatch(self, method: str) -> None:
        self._headers_done = False
        self._body_read = False
        api = self.path.startswith("/api/")
        try:
            self._route(method)
        except _Refusal as refusal:
            self._refuse(method, refusal, api=api)
        except (ConnectionError, TimeoutError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 - answer 500 and keep serving
            self.server.log(
                f"gbf web：内部错误 {method} {self._path_only()}：{type(exc).__name__}: {exc}"
            )
            if self._headers_done:
                self.close_connection = True
            else:
                internal = _Refusal(500, "服务内部错误，详见运行服务的终端。", type(exc).__name__)
                self._refuse(method, internal, api=api, logged=True)

    # ---------------------------------------------------------------- checks

    def _path_only(self) -> str:
        return self.path.split("?", 1)[0][:200]

    def _check_host(self) -> None:
        values = self.headers.get_all("Host") or []
        if len(values) != 1 or values[0].strip().lower() not in self.server.allowed_hosts:
            raise _Refusal(403, FOREIGN_MESSAGE, "Host is not 127.0.0.1:<port> or localhost:<port>")

    def _check_origin(self) -> None:
        for value in self.headers.get_all("Origin") or []:
            if value.strip().lower() not in self.server.allowed_origins:
                raise _Refusal(403, FOREIGN_MESSAGE, "foreign Origin")

    def _check_fetch_site(self) -> None:
        site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if site and site not in ("same-origin", "none"):
            raise _Refusal(403, FOREIGN_MESSAGE, f"Sec-Fetch-Site {site}")

    def _cookie_token(self) -> str | None:
        jar: SimpleCookie = SimpleCookie()
        for header in self.headers.get_all("Cookie") or []:
            try:
                jar.load(header)
            except CookieError:
                continue
        morsel = jar.get(self.server.cookie_name)
        return morsel.value if morsel is not None else None

    def _check_token(self, *, cookie: bool) -> None:
        given = self.headers.get(TOKEN_HEADER)
        if given is None and cookie:
            given = self._cookie_token()
        expected = self.server.token.encode("ascii")
        if given is None or not hmac.compare_digest(given.encode("utf-8", "replace"), expected):
            raise _Refusal(403, TOKEN_MESSAGE, "missing or wrong token")

    def _segments(self, path: str, prefix: str) -> list[str]:
        try:
            return [unquote(part, errors="strict") for part in path[len(prefix) :].split("/")]
        except UnicodeDecodeError:
            raise _Refusal(400, "地址无效。", "path is not UTF-8") from None

    # ---------------------------------------------------------------- routing

    def _route(self, method: str) -> None:
        target = self.path
        if not target.startswith("/") or target.startswith("//"):
            raise _Refusal(400, "地址无效。", "request target is not a path")
        path = urlsplit(target).path
        self._check_host()
        self._check_origin()
        if path == "/" or path.startswith("/static/"):
            self._static(method, path)
        elif path.startswith("/api/"):
            self._check_fetch_site()
            self._check_token(cookie=False)
            self._api(method, self._segments(path, "/api/"))
        elif path.startswith("/runs/"):
            self._check_fetch_site()
            self._check_token(cookie=True)
            self._run_file(method, self._segments(path, "/runs/"))
        else:
            raise _Refusal(404, "找不到这个地址。", "unknown path")

    def _static(self, method: str, path: str) -> None:
        if method not in ("GET", "HEAD"):
            raise _not_allowed("GET, HEAD")
        name = "index.html" if path == "/" else path.removeprefix("/static/")
        if name not in STATIC_FILES:
            raise _Refusal(404, "找不到这个地址。", "unknown static file")
        self._send_file(STATIC_DIR / name, STATIC_FILES[name], PAGE_CSP, ranges=False)

    def _run_file(self, method: str, segments: list[str]) -> None:
        if method not in ("GET", "HEAD"):
            raise _not_allowed("GET, HEAD")
        if len(segments) < 2:
            raise _Refusal(404, "找不到这个文件。", "no file named")
        try:
            path = resolve_artifact(self.server.runs_dir, segments[0], segments[1:])
        except RunPathError as exc:
            message = "找不到这个文件。" if exc.status == 404 else "拒绝访问这个文件。"
            raise _Refusal(exc.status, message, exc.reason) from None
        self._send_file(path, SERVED_SUFFIXES[path.suffix.lower()], FILE_CSP, ranges=True)

    def _api(self, method: str, segments: list[str]) -> None:
        """``GET session`` (and the cookie), ``GET runs``, ``GET runs/<run>``, ``GET jobs``,
        ``POST jobs``, ``GET jobs/<id>``."""
        root = self.server.runs_dir
        jobs = self.server.jobs
        if segments == ["session"]:
            if method != "GET":
                raise _not_allowed("GET")
            self._session()
        elif segments == ["runs"]:
            if method != "GET":
                raise _not_allowed("GET")
            self._send_json(200, {"runs_dir": str(root), "runs": list_runs(root)})
        elif len(segments) == 2 and segments[0] == "runs":
            if method != "GET":
                raise _not_allowed("GET")
            try:
                detail = run_detail(root, segments[1])
            except RunPathError as exc:
                message = "找不到这次运行。" if exc.status == 404 else "拒绝访问。"
                raise _Refusal(exc.status, message, exc.reason) from None
            self._send_json(200, detail)
        elif segments == ["jobs"]:
            if method == "GET":
                current = jobs.current()
                payload = {"jobs": jobs.recent(), "current": current.id if current else None}
                self._send_json(200, payload)
            elif method == "POST":
                self._start_job()
            else:
                raise _not_allowed("GET, POST")
        elif len(segments) == 2 and segments[0] == "jobs":
            if method != "GET":
                raise _not_allowed("GET")
            job = jobs.get(segments[1])
            if job is None:
                raise _Refusal(404, "找不到这个分析任务。", "unknown job")
            self._send_json(200, job.snapshot(root))
        elif len(segments) == 3 and segments[0] == "jobs" and segments[2] == "cancel":
            if method != "POST":
                raise _not_allowed("POST")
            self._json_body()
            job = jobs.cancel(segments[1])
            if job is None:
                raise _Refusal(404, "找不到这个分析任务。", "unknown job")
            self._send_json(200, job.snapshot(root))
        elif segments == ["uploads"]:
            if method != "POST":
                raise _not_allowed("POST")
            self._upload_video()
        else:
            raise _Refusal(404, "找不到这个地址。", "unknown API path")

    def _session(self) -> None:
        cookie = (
            f"{self.server.cookie_name}={self.server.token}; Path=/runs/; HttpOnly; SameSite=Strict"
        )
        payload = {
            "version": gbf.__version__,
            "runs_dir": str(self.server.runs_dir),
            "privacy_zh": PRIVACY_ZH,
            "stages": [
                {"name": name, "name_zh": STAGE_NAMES_ZH.get(name, name)} for name in STAGES
            ],
            "fields_zh": dict(FIELD_NAMES_ZH),
            "automatic": runtime_status(self.server.jobs.environ),
        }
        self._send_json(200, payload, extra=[("Set-Cookie", cookie)])

    def _json_body(self) -> Any:
        media, _, parameters = (self.headers.get("Content-Type") or "").partition(";")
        charset = parameters.strip().lower().removeprefix("charset=").strip('"')
        if media.strip().lower() != "application/json" or charset not in ("", "utf-8"):
            raise _Refusal(415, "请求体必须是 JSON（application/json，UTF-8）。", "content type")
        if self.headers.get("Transfer-Encoding") is not None:
            raise _Refusal(411, "请求体需要写明长度。", "transfer encoding")
        length = (self.headers.get("Content-Length") or "").strip()
        if not _LENGTH.fullmatch(length):
            raise _Refusal(411, "请求体需要写明长度。", "no content length")
        if int(length) > MAX_BODY_BYTES:
            raise _Refusal(413, "请求体太大。", "body too large")
        body = self.rfile.read(int(length))
        self._body_read = True
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):  # nested too deeply
            raise _Refusal(400, "请求体不是有效的 JSON。", "invalid JSON") from None

    def _drop_unread_body(self) -> None:
        """Read and drop the body of a request that is about to be refused, when nothing has
        read it and its length is known and at most MAX_BODY_BYTES (one Content-Length, no
        Transfer-Encoding): closing a connection with a body unread resets it, and the
        client can lose the answer. Other bodies stay unread; the connection ends anyway."""
        if self._body_read or self.headers.get("Transfer-Encoding") is not None:
            return
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not _LENGTH.fullmatch(lengths[0].strip()):
            return
        length = int(lengths[0].strip())
        if 0 < length <= MAX_BODY_BYTES:
            self._body_read = True
            try:
                self.rfile.read(length)
            except OSError:  # the client stopped sending (or timed out): answer all the same
                pass

    def _start_job(self) -> None:
        data = self._json_body()
        try:
            request = JobRequest.from_json(data)
        except JobRequestError as exc:
            raise _Refusal(
                400, "这些内容需要改一下：", "invalid job request", exc.problems
            ) from None
        try:
            job = self.server.jobs.start(request)
        except JobRefused as exc:
            raise _Refusal(exc.status, "不能开始分析：", "job refused", exc.problems) from None
        self._send_json(202, job.snapshot(self.server.runs_dir))

    def _finish_large_refusal(self) -> None:
        """Send FIN before a bounded drain so Windows does not reset a pending answer.

        Oversized requests get their answer immediately, even if the caller sends no
        body. Only after answering do we discard up to 128 KiB, for at most 0.5 seconds.
        """
        if self._body_read or self.headers.get("Transfer-Encoding") is not None:
            return
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not _LENGTH.fullmatch(lengths[0].strip()):
            return
        length = int(lengths[0].strip())
        if length <= MAX_BODY_BYTES:
            return
        try:
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_WR)
            remaining = min(length, 2 * MAX_BODY_BYTES)
            deadline = time.monotonic() + 0.5
            while remaining:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                self.connection.settimeout(timeout)
                chunk = self.rfile.read1(min(COPY_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            pass  # a stopped sender must not delay or replace the refusal

    def _upload_video(self) -> None:
        """A bounded, streaming local upload, authenticated by the normal API checks."""
        if self.headers.get_content_type() != "application/octet-stream":
            raise _Refusal(415, "请选择一个视频文件。", "upload content type")
        lengths = self.headers.get_all("Content-Length") or []
        if (
            self.headers.get("Transfer-Encoding") is not None
            or len(lengths) != 1
            or not _LENGTH.fullmatch(lengths[0].strip())
        ):
            raise _Refusal(411, "视频上传需要明确的文件长度。", "upload length")
        length = int(lengths[0])
        if not 0 < length <= MAX_UPLOAD_BYTES:
            raise _Refusal(413, "视频为空或超过 512 MB，请截取短片后重试。", "upload size")
        if self.server.jobs.current() is not None:
            raise _Refusal(409, "已有分析正在运行，请先等待完成或停止它。", "busy")
        status = runtime_status(self.server.jobs.environ)
        if not status["available"]:
            raise _Refusal(400, status["message_zh"], "runtime unavailable")
        if not self.server.upload_lock.acquire(blocking=False):
            raise _Refusal(409, "已有视频正在传入，请稍后再试。", "upload busy")
        partial = None
        try:
            from gbf.pipeline import check_out_dir

            folder = self.server.runs_dir / ".gbf-web" / "uploads"
            if check_out_dir(folder):
                raise _Refusal(400, "视频保存目录必须位于代码仓库外。", "upload in repository")
            folder.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(folder).free < length + 512 * 1024**2:
                raise _Refusal(413, "磁盘空间不足，无法保存这个视频。", "upload disk space")
            dest = folder / (secrets.token_hex(16) + ".mp4")
            partial = dest.with_suffix(".partial")
            self.connection.settimeout(60)
            remaining = length
            with partial.open("xb") as handle:
                while remaining:
                    chunk = self.rfile.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        raise _Refusal(400, "视频传入中断，请重新选择文件。", "incomplete upload")
                    handle.write(chunk)
                    remaining -= len(chunk)
            self._body_read = True
            os.replace(partial, dest)
            self._send_json(201, {"video": str(dest), "bytes": length})
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
            self.server.upload_lock.release()

    # ---------------------------------------------------------------- answers

    def _begin(
        self,
        status: int,
        content_type: str,
        length: int,
        csp: str,
        extra: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Security-Policy", csp)
        self._csp_sent = True
        for name, value in extra:
            self.send_header(name, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()

    def _send_json(self, status: int, payload: Any, extra: Sequence[tuple[str, str]] = ()) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._begin(status, "application/json; charset=utf-8", len(body), API_CSP, extra)
        if self.command != "HEAD":
            self.wfile.write(body)

    def _refuse(self, method: str, refusal: _Refusal, *, api: bool, logged: bool = False) -> None:
        self._drop_unread_body()
        self.close_connection = True  # a refused request ends its connection
        if not logged and refusal.status != 404:
            self.server.log(
                f"gbf web：拒绝 {method} {self._path_only()}（{refusal.status}，{refusal.reason}）"
            )
        extra = [("Allow", refusal.allow)] if refusal.allow else []
        if api:
            payload = {"error_zh": refusal.message, "problems_zh": refusal.problems}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        else:
            body = refusal.message.encode("utf-8")
            content_type = "text/plain; charset=utf-8"
        self._begin(refusal.status, content_type, len(body), API_CSP, extra)
        if method != "HEAD":
            self.wfile.write(body)
        self._finish_large_refusal()

    def _send_file(self, path: Path, content_type: str, csp: str, *, ranges: bool) -> None:
        try:
            handle = open(path, "rb")
        except OSError:
            raise _Refusal(404, "找不到这个文件。", "cannot open") from None
        with handle:
            size = os.fstat(handle.fileno()).st_size
            span = parse_range(self.headers.get("Range"), size) if ranges else None
            extra = [("Content-Disposition", "inline")]
            if ranges:
                extra.append(("Accept-Ranges", "bytes"))
            if span == _UNSATISFIABLE:
                extra.append(("Content-Range", f"bytes */{size}"))
                self.close_connection = True
                self._begin(416, "text/plain; charset=utf-8", 0, API_CSP, extra)
                return
            if span is None:
                status, start, length = 200, 0, size
            else:
                status, start, length = 206, span[0], span[1] - span[0] + 1
                extra.append(("Content-Range", f"bytes {span[0]}-{span[1]}/{size}"))
            self._begin(status, content_type, length, csp, extra)
            if self.command == "HEAD":
                return
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(COPY_CHUNK, remaining))
                if not chunk:  # the file shrank: end the connection with a short body
                    self.close_connection = True
                    return
                self.wfile.write(chunk)
                remaining -= len(chunk)


# --------------------------------------------------------------------------- the command


def runs_dir_from(value: Path | None, environ: Mapping[str, str]) -> tuple[Path | None, list[str]]:
    """The runs folder: ``value``, else ``$DATA_ROOT/runs`` (gbf.models reads DATA_ROOT the
    same way). Returns it resolved, or the problems (in Chinese) when there is none, it is a
    network path or it is not an existing folder."""
    if value is not None:
        candidate = Path(value)
    elif environ.get(DATA_ROOT_ENV):
        candidate = Path(environ[DATA_ROOT_ENV]) / RUNS_DIRNAME
    else:
        return None, [
            "需要用 --runs-dir 指定运行目录（它的每个子文件夹是一次运行）；也可以设置环境变量 "
            f"{DATA_ROOT_ENV}，默认用 ${DATA_ROOT_ENV}/{RUNS_DIRNAME}。"
        ]
    if network_path(str(candidate)):
        return None, [f"运行目录不能是网络路径：{candidate}"]
    if not candidate.is_dir():
        return None, [
            f"运行目录不存在：{candidate}（先建好这个文件夹，或用 --runs-dir 指定已有的文件夹）"
        ]
    return candidate.resolve(), []


def _port(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {text!r}") from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port must be in 0..65535, got {port}")
    return port


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gbf web",
        description="Local web page (127.0.0.1 only) to start the highlight analysis and look "
        "at its results (docs/design/web.md).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=f"folder whose subfolders are runs (default ${DATA_ROOT_ENV}/{RUNS_DIRNAME})",
    )
    parser.add_argument(
        "--port", type=_port, default=DEFAULT_PORT, help="port on 127.0.0.1 (0: a free one)"
    )
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    return parser


def main(argv: list[str]) -> int:
    """``python -m gbf web ...``; returns 0 after Ctrl+C, 2 for refused arguments, 1 when the
    port cannot be bound."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    runs_dir, problems = runs_dir_from(args.runs_dir, os.environ)
    if runs_dir is None:
        for problem in problems:
            print(f"gbf web：{problem}", file=sys.stderr)
        return 2
    try:
        server = WebServer(runs_dir, port=args.port)
    except OSError as exc:
        print(
            f"gbf web：无法在 {HOST}:{args.port} 上监听（{exc.strerror or exc}）。换一个端口"
            "（--port），或用 --port 0 让系统挑一个空闲端口。",
            file=sys.stderr,
        )
        return 1
    print(
        "gbf web：本机分析台已启动（只监听 127.0.0.1，不连外网）。\n"
        f"  运行目录：{server.runs_dir}\n"
        f"  在浏览器里打开：{server.url}\n"
        "  这个地址带着本次启动的访问令牌，只给自己用，不要分享。按 Ctrl+C 停止服务"
        "（正在运行的分析也会停止）。",
        flush=True,
    )
    if not args.no_open:
        try:
            opened = webbrowser.open(server.url)
        except webbrowser.Error:
            opened = False
        if not opened:
            print("gbf web：没能自动打开浏览器，请手动复制上面的地址。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("gbf web：正在停止……", flush=True)
    finally:
        server.jobs.close()
        server.server_close()
    print("gbf web：已停止。", flush=True)
    return 0


__all__ = [
    "DEFAULT_PORT",
    "HOST",
    "STATIC_DIR",
    "STATIC_FILES",
    "TOKEN_HEADER",
    "WebHandler",
    "WebServer",
    "main",
    "parse_range",
    "runs_dir_from",
]
