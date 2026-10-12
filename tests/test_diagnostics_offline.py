"""Offline proof: a runtime guard that really stops a connection, and a static scan."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import urllib.request

import pytest

from nativevoice.core.errors import VoiceError
from nativevoice.diagnostics.offline import (
    NetworkAttempt,
    NetworkBlocked,
    OfflineGuard,
    package_source_root,
    scan_package,
    static_scan,
)

EXTERNAL_IP = "93.184.68.12"  # a public address; never contacted - the guard refuses first


def test_raw_socket_connect_to_a_public_address_is_refused():
    guard = OfflineGuard()
    original = socket.socket
    with guard:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(NetworkBlocked) as excinfo:
            probe.connect((EXTERNAL_IP, 443))
        probe.close()
    assert excinfo.value.details["target"] == EXTERNAL_IP
    assert [attempt.target for attempt in guard.blocked] == [EXTERNAL_IP]
    assert guard.blocked[0].port == 443
    assert guard.blocked[0].kind == "socket.connect"
    assert socket.socket is original, "the guard leaked its patch past __exit__"
    with pytest.raises(NetworkBlocked):
        guard.assert_clean()


def test_a_real_dns_resolving_attempt_never_reaches_the_resolver():
    """Blocking must precede resolution: a DNS lookup is already outbound traffic."""
    calls = []
    real_getaddrinfo = socket.getaddrinfo
    guard = OfflineGuard()
    with guard:
        # `socket.getaddrinfo` is patched by the guard; the *real* resolver is captured
        # above, so if the guard let a lookup through we would see it here.
        def tripwire(host, *args, **kwargs):
            calls.append(host)
            raise AssertionError("resolution reached the OS resolver inside the guard")

        socket.getaddrinfo = tripwire
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("example.com", 443), timeout=2)
        socket.getaddrinfo = real_getaddrinfo
    assert calls == [], "the guard resolved an external name before refusing it"
    assert guard.targets == ("example.com",)


def test_http_library_attempt_is_recorded_and_blocked():
    guard = OfflineGuard()
    with guard:
        with pytest.raises((NetworkBlocked, VoiceError)):
            urllib.request.urlopen("http://example.com/", timeout=2)  # noqa: S310 - must be refused
        client = http.client.HTTPConnection("example.com", 80, timeout=2)
        # http.client connects lazily via socket.socket, so the refusal lands at request time.
        with pytest.raises(NetworkBlocked):
            client.request("GET", "/")
            client.close()
    assert len(guard.blocked) >= 2, "both HTTP stacks must be refused"
    assert {attempt.target for attempt in guard.blocked} == {"example.com"}
    assert all(attempt.kind.startswith("socket.") for attempt in guard.blocked)


def test_tls_attempt_is_refused_before_any_handshake_traffic():
    """HTTPS goes through the same socket layer, so a TLS client cannot bypass the guard."""
    guard = OfflineGuard()
    context = ssl.create_default_context()
    with guard:
        with pytest.raises(NetworkBlocked):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((EXTERNAL_IP, 443))
        secure = http.client.HTTPSConnection("example.org", context=context, timeout=2)
        with pytest.raises(NetworkBlocked):
            secure.request("GET", "/")
    assert guard.targets == (EXTERNAL_IP, "example.org")


def test_loopback_still_works_inside_the_guard():
    """Local IPC must survive, or the guard is unusable for real sessions and tests."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    guard = OfflineGuard()
    with guard:
        client = socket.create_connection(("127.0.0.1", port), timeout=2)
        connection, _ = server.accept()
        client.sendall(b"ping")
        assert connection.recv(4) == b"ping"
        client.close()
        connection.close()
    guard.assert_clean()
    # One loopback connection is recorded more than once (the helper, the resolver it
    # calls, and the socket's own connect), which is the point: `attempts` is a full
    # audit trail, and an allowed target is marked allowed rather than omitted.
    assert [attempt.target for attempt in guard.attempts] == ["127.0.0.1"] * len(guard.attempts)
    assert len(guard.attempts) >= 2
    assert all(attempt.allowed for attempt in guard.attempts)
    assert {"socket.create_connection", "socket.connect"} <= {attempt.kind for attempt in guard.attempts}
    server.close()


def test_guard_restores_sockets_when_the_body_raises():
    original_socket = socket.socket
    original_create = socket.create_connection
    with pytest.raises(RuntimeError):
        with OfflineGuard():
            raise RuntimeError("body failed")
    assert socket.socket is original_socket
    assert socket.create_connection is original_create


def test_nested_guard_does_not_unpatch_while_the_outer_scope_is_active():
    original = socket.socket
    outer = OfflineGuard()
    with outer:
        assert socket.socket is not original
        with OfflineGuard():
            assert socket.socket is not original
        assert socket.socket is not original, "the inner guard must not restore while an outer one is live"
        with pytest.raises(NetworkBlocked):
            socket.create_connection((EXTERNAL_IP, 443), timeout=1)
    assert socket.socket is original


def test_allowlist_extends_permitted_targets():
    guard = OfflineGuard(extra_allowed=frozenset({"10.0.0.7"}))
    with guard:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(NetworkBlocked):
            probe.connect(("10.0.0.8", 9))
        probe.close()
    assert guard.targets == ("10.0.0.8",)


def test_clean_run_asserts_quietly():
    guard = OfflineGuard()
    with guard:
        socket.socketpair()
    guard.assert_clean()
    assert guard.blocked == ()


def test_attempt_records_are_serialisable_and_actionable():
    guard = OfflineGuard()
    with guard:
        with pytest.raises(NetworkBlocked):
            socket.create_connection((EXTERNAL_IP, 8080), timeout=1)
    payload = json.loads(json.dumps([a.as_dict() for a in guard.attempts]))
    assert payload[0]["target"] == EXTERNAL_IP
    assert payload[0]["port"] == 8080
    assert payload[0]["allowed"] is False
    assert ":" in payload[0]["stack_hint"], "a finding without a file:line is not actionable"


def test_static_scan_finds_network_imports_in_a_temp_tree(tmp_path):
    (tmp_path / "sneaky.py").write_text(
        "import urllib.request\n\nimport socket\n\n\ndef fetch():\n    return urllib.request.urlopen('http://x')\n",
        encoding="utf-8",
    )
    (tmp_path / "clean.py").write_text("import math\n\nVALUE = math.pi\n", encoding="utf-8")
    findings = static_scan(str(tmp_path))
    by_line = {finding.line: finding.module for finding in findings}
    assert by_line == {1: "urllib.request", 3: "socket", 7: "url opener"}
    assert "sneaky.py:1" in str(findings[0])
    assert all("clean.py" not in finding.file for finding in findings)


def test_static_scan_reports_third_party_and_mail_transports(tmp_path):
    (tmp_path / "bad.py").write_text(
        "import requests\nimport httpx\nimport urllib3\nimport ftplib\nimport smtplib\nimport http.client\n",
        encoding="utf-8",
    )
    modules = {finding.module for finding in static_scan(str(tmp_path))}
    assert {"requests", "httpx", "urllib3", "ftplib", "smtplib", "http.client"} <= modules


def test_static_scan_catches_asyncio_stream_helpers(tmp_path):
    (tmp_path / "async_bad.py").write_text(
        "async def go():\n    reader, writer = await open_connection('1.2.3.4', 443)\n", encoding="utf-8"
    )
    assert any(finding.module == "asyncio networking" for finding in static_scan(str(tmp_path)))


def test_comments_are_not_findings(tmp_path):
    (tmp_path / "talks.py").write_text("# we never import socket here\nX = 1\n", encoding="utf-8")
    assert static_scan(str(tmp_path)) == ()


def test_shipped_package_is_clean_except_for_the_guard_itself():
    report = scan_package(package_source_root())
    assert report["clean"], report["findings"]
    assert report["count"] == 0
    assert any("offline.py" in entry for entry in report["allowlisted"])


def test_the_guard_module_is_the_only_socket_importer_in_the_package():
    """Prove the allowlist is load-bearing: without it, offline.py is a finding."""
    unallowlisted = static_scan(package_source_root(), allowlist=())
    assert unallowlisted, "the guard imports socket, so an unallowlisted scan must see it"
    assert {f.file for f in unallowlisted} == {"diagnostics/offline.py"}


def test_attempt_dataclass_shapes_targets_for_humans():
    attempt = NetworkAttempt(kind="socket.connect", target=EXTERNAL_IP, port=443, allowed=False, at_ns=1, stack_hint="x.py:3")
    assert str(attempt) == f"socket.connect -> {EXTERNAL_IP}:443"


def test_findings_are_relative_to_the_scanned_root_not_to_the_package_name(tmp_path):
    """A checkout whose *directory* happens to be named like the package must not shift
    every reported path - that is what a relocation exposed, and it failed silently
    everywhere except here."""
    from pathlib import Path

    nested = tmp_path / "nativevoice" / "src" / "nativevoice"
    (nested / "diagnostics").mkdir(parents=True)
    (nested / "diagnostics" / "offline.py").write_text("import socket\n", encoding="utf-8")
    (nested / "elsewhere.py").write_text("import socket\n", encoding="utf-8")
    findings = static_scan(str(nested), allowlist=())
    assert {finding.file for finding in findings} == {"diagnostics/offline.py", "elsewhere.py"}
    assert all(not finding.file.startswith("src/") for finding in findings)
