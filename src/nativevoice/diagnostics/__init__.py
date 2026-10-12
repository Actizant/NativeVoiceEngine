"""Diagnostics: structured logs, offline proof, latency trace.

Standard library only, and no sink in this package may reach the network - the
offline guarantee in `nativevoice.diagnostics.offline` is what makes the rest of the
project's "local-first" claim checkable, so adding a remote transport here would
contradict the module that hosts it.
"""

from __future__ import annotations

from .log import DEFAULT_REDACTED_FIELDS, REDACTED, DiagnosticSink, DiagnosticSinkError
from .offline import (
    Finding,
    NetworkAttempt,
    NetworkBlocked,
    OfflineGuard,
    static_scan,
    package_source_root,
    scan_package,
)
from .timeline import STAGES, Span, StageStats, Trace

__all__ = [
    "DEFAULT_REDACTED_FIELDS",
    "REDACTED",
    "STAGES",
    "DiagnosticSink",
    "DiagnosticSinkError",
    "Finding",
    "NetworkAttempt",
    "NetworkBlocked",
    "OfflineGuard",
    "Span",
    "StageStats",
    "Trace",
    "package_source_root",
    "scan_package",
    "static_scan",
]
