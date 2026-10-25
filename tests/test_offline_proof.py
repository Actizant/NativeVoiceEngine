"""The offline-proof tool's own reporting line: one static finding, formatted.

No model, no inference, no network. The unit under test is the label a finding gets on its
way into the published record. The defect these tests exist for: `tools/offline_proof.py`
read `finding.path` and `finding.lineno` while `Finding` defines `file` and `line`
(`src/nativevoice/diagnostics/offline.py`), so the line raised the moment a finding existed.
The published record's historically empty list is why no run ever proved the formatting
worked - an absent finding cannot be formatted, and a present one crashed the tool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


# --- the publication contract --------------------------------------------------------

#: The committed evidence record the tool publishes to when no destination is named. Every
#: test in this section relocates it into a temporary repository before `main()` can be
#: reached, so no test here can write to the real record even by forgetting to.
EVIDENCE_PATH = "artifacts/audit/offline-proof.json"

HOME = str(Path.home())
PLACEHOLDER = "[USER_HOME]"

#: A completed run's published shape, read from the record's own keys.
COMPLETED = {
    "tts_asset": "kokoro-int8-multi-lang-v1_1",
    "stt_asset": "faster-whisper-base",
    "language": "en",
    "tts_load_seconds": 0.1,
    "tts_wall_seconds": 0.2,
    "tts_audio_seconds": 1.5,
    "tts_real_time_factor": 0.1333,
    "stt_load_seconds": 0.1,
    "stt_input_format": "AudioFormat(sample_rate=16000, channels=1, sample_width_bits=16)",
    "stt_wall_seconds": 0.2,
    "heard": "The voice engine is running locally on my machine.",
    "heard_language": "en",
    "attempts": [],
    "guard_clean": True,
    "static_findings": [],
    "total_seconds": 0.6,
    "verdict": "PASS",
}

NOT_RUN = {"verdict": "NOT RUN", "reason": "missing manifest: tts=None stt=None"}


@pytest.fixture(autouse=True)
def temporary_repository(tmp_path: Path, monkeypatch) -> Path:
    """Point the tool's default destination at a throwaway repo, for every test in this file.

    `ROOT` is patched as well as `DEFAULT_OUT` because the destination label, and the static
    scan's package root, are both derived from it. `artifacts/audit/` is deliberately not
    created: a refused run must not have made it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(offline_proof, "ROOT", repo)
    default = repo / EVIDENCE_PATH
    monkeypatch.setattr(offline_proof, "DEFAULT_OUT", default)
    return default


def _stub_run(monkeypatch, report: dict) -> None:
    """Replace the measurement with a verdict, so no engine is imported and no run happens."""
    monkeypatch.setattr(offline_proof, "run", lambda *args: dict(report))


def test_a_not_run_result_writes_nothing_by_default(temporary_repository, monkeypatch, capsys):
    """The defect this change exists to close: a non-measurement must not replace evidence.

    Before the gate, a machine without the two approved assets wrote a two-key stub over the
    seventeen-key published record and then exited 1 - the destruction happened first, so the
    exit code only reported it afterwards.
    """
    _stub_run(monkeypatch, NOT_RUN)

    assert offline_proof.main([]) == 1
    assert not temporary_repository.exists(), "a NOT RUN run published a stub over the record"
    assert not temporary_repository.parent.exists(), "a NOT RUN run created the artefact directory"
    printed = capsys.readouterr().out
    assert "artefact NOT written" in printed, printed
    assert "missing manifest" in printed, printed


def test_allow_incomplete_alone_cannot_reach_the_default_destination(temporary_repository, monkeypatch, capsys):
    """The flag authorises an incomplete report elsewhere; it never authorises this path.

    Naming the destination is what makes an incomplete publication the operator's act, so the
    flag by itself must leave the committed record exactly where it was.
    """
    _stub_run(monkeypatch, NOT_RUN)

    assert offline_proof.main(["--allow-incomplete"]) == 1
    assert not temporary_repository.exists()
    assert not temporary_repository.parent.exists()
    assert "artefact NOT written" in capsys.readouterr().out


def test_an_explicit_destination_alone_still_refuses_the_incomplete_report(temporary_repository, monkeypatch):
    """Both halves of the permission are required; naming a path is not enough."""
    _stub_run(monkeypatch, NOT_RUN)
    elsewhere = temporary_repository.parent.parent / "elsewhere.json"

    assert offline_proof.main(["--out", str(elsewhere)]) == 1
    assert not elsewhere.exists()
    assert not temporary_repository.exists()


def test_an_explicit_destination_with_the_flag_publishes_the_incomplete_report(temporary_repository, monkeypatch, capsys):
    """The one permitted incomplete case: named destination, flag set, and still a failed exit."""
    _stub_run(monkeypatch, NOT_RUN)
    elsewhere = temporary_repository.parent.parent / "elsewhere.json"

    assert offline_proof.main(["--out", str(elsewhere), "--allow-incomplete"]) == 1
    rows = json.loads(elsewhere.read_text(encoding="utf-8"))
    assert rows["verdict"] == "NOT RUN"
    assert not temporary_repository.exists(), "the incomplete report reached the committed record"
    assert "incomplete report" in capsys.readouterr().out


def test_a_pass_result_still_publishes_to_the_default_destination(temporary_repository, monkeypatch, capsys):
    """The documented behaviour of a bare run is unchanged: it publishes, and exits 0."""
    _stub_run(monkeypatch, COMPLETED)

    assert offline_proof.main([]) == 0
    assert temporary_repository.is_file()
    rows = json.loads(temporary_repository.read_text(encoding="utf-8"))
    assert rows == COMPLETED, "the published record lost or gained a field"
    printed = capsys.readouterr().out
    assert f"artefact -> {EVIDENCE_PATH}" in printed, printed


def test_a_fail_result_is_published_and_exits_one(temporary_repository, monkeypatch):
    """A completed run that failed is evidence, not a reason to write nothing."""
    _stub_run(monkeypatch, {**COMPLETED, "verdict": "FAIL",
                            "attempts": ["10.0.0.1:443"], "guard_clean": False})

    assert offline_proof.main([]) == 1
    rows = json.loads(temporary_repository.read_text(encoding="utf-8"))
    assert rows["verdict"] == "FAIL" and rows["attempts"] == ["10.0.0.1:443"]


def test_a_home_path_in_the_report_is_published_scrubbed(temporary_repository, monkeypatch):
    """The publication boundary is now structural, not a claim about the payload.

    This writer used to be exempted from the boundary on the grounds that a static finding is
    reported relative to the scanned directory. The payload is handed straight to the shared
    serialiser now, so a field that does name a file leaves the account name behind.
    """
    leaky = {**COMPLETED, "static_findings": [f"{HOME}\\Documents\\evidence.json:3:socket"]}
    _stub_run(monkeypatch, leaky)
    elsewhere = temporary_repository.parent.parent / "published.json"

    assert offline_proof.main(["--out", str(elsewhere)]) == 0
    written = elsewhere.read_text(encoding="utf-8")
    assert HOME not in written and HOME.replace("\\", "\\\\") not in written, "the account name was published"
    assert PLACEHOLDER in written, written
    assert json.loads(written)["verdict"] == "PASS"


def test_a_non_ascii_transcript_is_published_literally(temporary_repository, monkeypatch):
    """Parity with every other committed artefact: readable text, not escape sequences."""
    _stub_run(monkeypatch, {**COMPLETED, "heard": "语音引擎运行在本地 — ok é"})
    elsewhere = temporary_repository.parent.parent / "unicode.json"

    assert offline_proof.main(["--out", str(elsewhere)]) == 0
    written = elsewhere.read_text(encoding="utf-8")
    assert "语音引擎运行在本地 — ok é" in written, written
    assert "\\u" not in written, "the shared serialiser's ensure_ascii default was overridden"


def test_an_empty_model_root_reports_not_run_without_publishing(temporary_repository, monkeypatch, tmp_path, capsys):
    """The real early return, not a stub: no manifests, so no measurement and no artefact.

    `run()` returns before either engine adapter is reached, so this proves the gate holds on
    the path that actually happens on a fresh machine without loading a model.
    """
    empty = tmp_path / "models"
    empty.mkdir()
    monkeypatch.setattr(offline_proof, "models_dir", lambda: empty)

    assert offline_proof.main([]) == 1
    assert not temporary_repository.exists()
    assert not temporary_repository.parent.exists()
    assert "missing manifest" in capsys.readouterr().out


def test_the_publication_contract_is_written_down_where_it_is_used():
    """The docstring is the operator's only warning that a bare run publishes evidence."""
    source = (ROOT / "tools" / "offline_proof.py").read_text(encoding="utf-8")
    assert f'DEFAULT_OUT = ROOT / "artifacts" / "audit" / "offline-proof.json"' in source
    for phrase in (EVIDENCE_PATH, "NOT RUN", "artefact NOT written", "nothing was overwritten",
                   PLACEHOLDER, "CRLF", "capability, not a restriction", "README.md:53"):
        assert phrase in source, f"the documented contract no longer mentions {phrase!r}"
    assert "--allow-incomplete" in source and "write_json(out, report)" in source
    assert ".write_text(json.dumps(" not in source, "the writer left the shared redacting boundary"
