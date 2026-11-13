"""Block network access during tests and gate runs.

Loopback stays allowed: on Windows, asyncio and multiprocessing build their
socket pairs through 127.0.0.1. Everything else raises NetworkBlockedError,
including DNS lookups of non-local names.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", ""}
_installed = False


class NetworkBlockedError(RuntimeError):
    pass


def _is_local(host: Any) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    if not isinstance(host, str):
        return False
    host = host.strip("[]").split("%", 1)[0].lower()
    if host in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _address_host(address: Any) -> Any:
    if isinstance(address, tuple) and address:
        return address[0]
    return address


def install() -> None:
    """Patch the socket module in this process. Safe to call more than once."""
    global _installed
    if _installed:
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(
            _address_host(address)
        ):
            raise NetworkBlockedError(f"network access is blocked here: {address!r}")
        return original_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(
            _address_host(address)
        ):
            raise NetworkBlockedError(f"network access is blocked here: {address!r}")
        return original_connect_ex(self, address)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_local(host):
            raise NetworkBlockedError(f"DNS lookup is blocked here: {host!r}")
        return original_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = guarded_getaddrinfo
    _installed = True


def is_installed() -> bool:
    return _installed
