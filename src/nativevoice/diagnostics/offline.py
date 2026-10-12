"""Offline-compliance proof: a runtime socket guard plus a static import scan.

TWO INDEPENDENT PROOFS BECAUSE EACH MISSES WHAT THE OTHER CATCHES. A runtime guard
only sees code that actually executed - it proves nothing about a branch that was
never reached in the acceptance run. A static scan only sees imports - a
`getattr(socket, "sock" + "et")` style call or a compiled dependency bypasses it.
The specification demands provable offline behaviour, so both are reported, and a
clean bill of health requires both to be clean.

The guard refuses before any syscall that could leave the machine: a hostname is
rejected without resolving it, because DNS *is* a network operation and "we blocked
the connection but looked the name up first" is not offline. Loopback (127.0.0.0/8,
::1, `localhost`) stays reachable so that local IPC and a local server under test keep
working; everything else raises `NetworkBlocked` and is recorded in `attempts`.
"""

from __future__ import annotations

import inspect
import ipaddress
import os
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Iterable

from ..core.errors import VoiceError

LOOPBACK_NAMES = frozenset({"localhost", "ip6-localhost", "ip6-loopback", ""})
LOOPBACK_NETS = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))


class NetworkBlocked(VoiceError):
    """An outbound connection attempt was refused by the offline guard."""


@dataclass(frozen=True, slots=True)
class NetworkAttempt:
    """One refused (or allowed) target, recorded with enough detail to act on."""

    kind: str
    target: str
    port: int | None
    allowed: bool
    at_ns: int
    stack_hint: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "target": self.target,
            "port": self.port,
            "allowed": self.allowed,
            "at_ns": self.at_ns,
            "stack_hint": self.stack_hint,
        }

    def __str__(self) -> str:
        host = self.target if self.port is None else f"{self.target}:{self.port}"
        return f"{self.kind} -> {host}"


def _is_loopback_target(host: object) -> bool:
    """Loopback by literal address or by loopback name; anything else is external.

    A name that merely *looks* local (`example.localhost`, a hosts-file alias) is
    resolved by the OS, not trusted here; only the exact loopback spellings pass.
    """
    text = str(host or "")
    if text.lower() in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(text) in LOOPBACK_NETS[0] or ipaddress.ip_address(text) in LOOPBACK_NETS[1]
    except ValueError:
        return False


def _describe(address: object) -> tuple[str, int | None]:
    if isinstance(address, (tuple, list)) and address:
        host = str(address[0])
        port = address[1] if len(address) > 1 else None
        return host, int(port) if isinstance(port, (int,)) else port
    return str(address), None


@dataclass(slots=True)
class OfflineGuard:
    """Context manager that makes non-loopback networking impossible inside its body.

    Nest-safe: the innermost-entered, outermost-exited instance owns the patch, and
    restoration happens in `__exit__` even when the body raised, because a guard that
    leaks patched sockets turns the *next* test into a false positive.
    """

    allow_loopback: bool = True
    extra_allowed: frozenset = frozenset()
    attempts: list = field(default_factory=list)
    installed: bool = False
    _restore: tuple = field(default=(), init=False, repr=False)

    def _permitted(self, host: str) -> bool:
        return host in self.extra_allowed or (self.allow_loopback and _is_loopback_target(host))

    def check(self, kind: str, address: object) -> None:
        """Record the attempt, then raise unless the target is permitted.

        Recording happens before raising and regardless of the verdict, so `attempts`
        is a complete picture of what the code under test tried to do.
        """
        host, port = _describe(address)
        allowed = self._permitted(host)
        self.attempts.append(
            NetworkAttempt(kind=kind, target=host, port=port, allowed=allowed,
                           at_ns=time.monotonic_ns(), stack_hint=_caller_hint())
        )
        if allowed:
            return
        raise NetworkBlocked("offline guard refused a non-loopback target", kind=kind, target=host, port=port)

    @property
    def blocked(self) -> tuple[NetworkAttempt, ...]:
        return tuple(a for a in self.attempts if not a.allowed)

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(a.target for a in self.blocked)

    def assert_clean(self) -> None:
        """Raise if anything tried to leave the machine during the guarded block."""
        if self.blocked:
            rendered = ", ".join(str(a) for a in self.blocked)
            raise NetworkBlocked(
                f"{len(self.blocked)} outbound network attempt(s) recorded", targets=rendered
            )

    # -- patching ---------------------------------------------------------------

    def install(self) -> "OfflineGuard":
        if self.installed:
            return self
        guard = self
        real_socket = socket.socket
        real_create = socket.create_connection
        real_getaddrinfo = socket.getaddrinfo

        class GuardedSocket(real_socket):
            """Real socket whose connect-time targets are vetted, never post-hoc."""

            def connect(self, address):
                guard.check("socket.connect", address)
                return real_socket.connect(self, address)

            def connect_ex(self, address):
                guard.check("socket.connect_ex", address)
                return real_socket.connect_ex(self, address)

            def sendto(self, data, *args, **kwargs):
                target = args[0] if args else kwargs.get("address")
                if target is not None:
                    guard.check("socket.sendto", target)
                return real_socket.sendto(self, data, *args, **kwargs)

        def guarded_create_connection(address, *args, **kwargs):
            # The address is vetted here, before any resolution or syscall. Delegation
            # then rebuilds the connection through `real_create`, whose internal
            # `socket(...)` lookup lands on `GuardedSocket` in the pure-Python build, so
            # the socket that comes back is itself guarded. On builds where the helper is
            # native and bypasses that lookup, the connect target was already refused or
            # permitted above, so the guarantee does not weaken.
            guard.check("socket.create_connection", address)
            return real_create(address, *args, **kwargs)

        def guarded_getaddrinfo(host, *args, **kwargs):
            # Refuse before resolution: the lookup itself would already have left home.
            guard.check("socket.getaddrinfo", (host, None))
            return real_getaddrinfo(host, *args, **kwargs)

        self._restore = (real_socket, real_create, real_getaddrinfo)
        socket.socket = GuardedSocket
        socket.create_connection = guarded_create_connection
        socket.getaddrinfo = guarded_getaddrinfo
        self.installed = True
        return self

    def uninstall(self) -> "OfflineGuard":
        if not self.installed:
            return self
        real_socket, real_create, real_getaddrinfo = self._restore
        socket.socket = real_socket
        socket.create_connection = real_create
        socket.getaddrinfo = real_getaddrinfo
        self.installed = False
        return self

    def __enter__(self) -> "OfflineGuard":
        return self.install()

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Unconditional restore: an exception in the body must not leave sockets patched.
        self.uninstall()
        return False

def _caller_hint(depth: int = 3) -> str:
    """file:line of the code that attempted the connection, walked from the stack.

    Cheap enough to run on every attempt (only attempts matter), and it makes a
    failed offline assertion actionable instead of naming only the target.
    """
    try:
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is None or frame.f_back is None:
                return ""
            frame = frame.f_back
        return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
    except Exception:  # a hint is never worth failing a guard for
        return 


# -- static proof ---------------------------------------------------------------

NETWORK_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("urllib.request", re.compile(r"^\s*(?:import|from)\s+urllib\.request|^\s*from\s+urllib\s+import\s+.*request")),
    ("http.client", re.compile(r"^\s*(?:import|from)\s+http\.client|^\s*from\s+http\s+import\s+.*client")),
    ("socket", re.compile(r"^\s*(?:import|from)\s+socket\b")),
    ("requests", re.compile(r"^\s*(?:import|from)\s+requests\b")),
    ("httpx", re.compile(r"^\s*(?:import|from)\s+httpx\b")),
    ("urllib3", re.compile(r"^\s*(?:import|from)\s+urllib3\b")),
    ("ftplib", re.compile(r"^\s*(?:import|from)\s+ftplib\b")),
    ("smtplib", re.compile(r"^\s*(?:import|from)\s+smtplib\b")),
    ("asyncio networking", re.compile(r"open_connection|start_server|create_connection")),
    ("url opener", re.compile(r"\burlopen\s*\(")),
)

# This module is the guard: importing `socket` is its whole job, so it is allowlisted
# by relative path rather than by exception-per-finding, which would hide real calls.
STATIC_SCAN_ALLOWLIST = frozenset({"diagnostics/offline.py", "nativevoice/diagnostics/offline.py"})


@dataclass(frozen=True, slots=True)
class Finding:
    file: str
    line: int
    module: str
    text: str

    def as_dict(self) -> dict:
        return {"file": self.file, "line": self.line, "module": self.module, "text": self.text}

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.module} ({self.text})"


def static_scan(package_path: str, *, allowlist: Iterable[str] = STATIC_SCAN_ALLOWLIST) -> tuple[Finding, ...]:
    """Grep .py files under `package_path` for network-capable imports and calls.

    Line-oriented rather than AST-oriented on purpose: an unapproved third-party
    library is usually introduced by a plain import line, and a text scan reports the
    exact `file:line` a human can open. `allowlist` entries are compared as suffixes so
    the guard module can be excused without excusing a directory of findings.
    """
    allowed = tuple(str(a).replace("\\", "/") for a in allowlist)
    findings: list[Finding] = []
    for root, dirs, files in os.walk(package_path):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git") and not d.endswith(".egg-info")]
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            full = os.path.join(root, name).replace("\\", "/")
            # Reported relative to the directory actually being scanned. Splitting the
            # path on the package name was cheaper until a checkout lived in a directory
            # called `nativevoice`, at which point the first match was the repository and
            # every finding came back with a mangled prefix.
            try:
                relative = os.path.relpath(full, package_path).replace("\\", "/")
            except ValueError:
                relative = full
            if any(relative.endswith(a) or full.endswith(a) for a in allowed):
                continue
            try:
                with open(full, "r", encoding="utf-8") as handle:
                    lines = handle.read().splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for number, text in enumerate(lines, start=1):
                stripped = text.strip()
                if stripped.startswith("#"):
                    continue
                for label, pattern in NETWORK_PATTERNS:
                    if pattern.search(stripped):
                        findings.append(Finding(relative, number, label, stripped[:120]))
                        break
    return tuple(findings)


def scan_package(package_path: str, *, allowlist: Iterable[str] = STATIC_SCAN_ALLOWLIST) -> dict:
    """Scan plus the counts the CLI and reports need. Clean means zero findings."""
    findings = static_scan(package_path, allowlist=allowlist)
    return {
        "root": package_path,
        "findings": [f.as_dict() for f in findings],
        "count": len(findings),
        "clean": not findings,
        "allowlisted": sorted(allowlist),
    }


def package_source_root() -> str:
    """Absolute path of the `nativevoice` package directory this module lives in."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
