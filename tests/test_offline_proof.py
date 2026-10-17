"""The offline-proof tool's own reporting line: one static finding, formatted.

No model, no inference, no network. The unit under test is the label a finding gets on its
way into the published record. The defect these tests exist for: `tools/offline_proof.py`
read `finding.path` and `finding.lineno` while `Finding` defines `file` and `line`
(`src/nativevoice/diagnostics/offline.py`), so the line raised the moment a finding existed.
The published record's historically empty list is why no run ever proved the formatting
worked - an absent finding cannot be formatted, and a present one crashed the tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import tools.offline_proof as offline_proof  # noqa: E402
from nativevoice.diagnostics.offline import Finding, static_scan  # noqa: E402


def test_a_synthetic_finding_formats_without_an_attribute_error() -> None:
    """The exact dataclass, the exact field names, the label the record publishes."""
    finding = Finding(file="net/helper.py", line=7, module="socket", text="import socket")
    assert offline_proof.finding_label(finding) == "net/helper.py:7:socket"
    assert str(finding) == "net/helper.py:7: socket (import socket)", (
        "the dataclass keeps its own human-readable form; the label is the machine-readable one"
    )


def test_a_real_scan_produces_findings_that_all_format(tmp_path: Path) -> None:
    """A finding that came out of the scanner, not out of a fixture.

    `static_scan` is what the proof runs, and its output is what the record lists, so the
    round trip has to hold for the object type it actually returns.
    """
    (tmp_path / "net_tool.py").write_text("import socket\n", encoding="utf-8")
    findings = static_scan(str(tmp_path))
    assert findings, "NOT RUN: the scan found nothing, so there is no finding to format"
    assert all(isinstance(finding, Finding) for finding in findings)

    labelled = [offline_proof.finding_label(finding) for finding in findings]
    assert all(label.count(":") == 2 for label in labelled), labelled
    assert any("net_tool.py" in label for label in labelled), labelled
    assert not hasattr(findings[0], "path"), "the tool must not be written against a field it does not have"
