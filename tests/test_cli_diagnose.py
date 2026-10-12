"""The diagnose command: exit code is the contract, so it is tested as a process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")


def run(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "nativevoice.cli.diagnose", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=cwd or str(ROOT),
    )


def _capabilities(path: Path, rows) -> str:
    path.write_text(json.dumps({"capabilities": rows}), encoding="utf-8")
    return str(path)


def test_clean_run_with_no_capability_file_exits_zero_but_prints_not_run(tmp_path):
    result = run()
    assert result.returncode == 0, result.stderr
    assert "== host ==" in result.stdout
    assert "Windows" in result.stdout or "Linux" in result.stdout or "Darwin" in result.stdout
    assert "NOT_RUN" in result.stdout and "capability data" in result.stdout
    assert "offline static scan" in result.stdout
    assert "verdict: INCOMPLETE" in result.stdout


def test_measured_capability_under_threshold_passes(tmp_path):
    path = _capabilities(tmp_path / "caps.json", [
        {"language": "en", "task": "stt", "status": "measured", "engine": "shadow",
         "metric_name": "WER", "metric_value": 0.083, "measured_on": "2026-10-12", "model_id": "m1"},
    ])
    result = run("--capabilities", path, "--max-wer", "0.15")
    assert result.returncode == 0, result.stdout
    assert "PASS     en/stt WER [en]: 0.0830" in result.stdout


def test_measured_capability_over_threshold_fails(tmp_path):
    path = _capabilities(tmp_path / "caps.json", [
        {"language": "bn", "task": "stt", "status": "measured", "engine": "shadow",
         "metric_name": "WER", "metric_value": 0.62, "measured_on": "2026-10-12", "model_id": "m1"},
    ])
    result = run("--capabilities", path, "--max-wer", "0.15")
    assert result.returncode == 1
    assert "FAIL" in result.stdout and "exceeds the max threshold" in result.stdout


def test_declared_capability_is_never_reported_as_installed(tmp_path):
    path = _capabilities(tmp_path / "caps.json", [
        {"language": "bn", "task": "stt", "status": "declared", "engine": "shadow", "metric_name": "WER", "metric_value": 0.2},
    ])
    result = run("--capabilities", path, "--max-wer", "0.15")
    assert "NOT_RUN" in result.stdout
    assert "declared by upstream only" in result.stdout
    assert "d" in result.stdout.split("legend:")[0]  # the matrix symbol, not the word


def test_missing_and_awaiting_capabilities_are_listed_as_gaps(tmp_path):
    path = _capabilities(tmp_path / "caps.json", [
        {"language": "bn", "task": "tts", "status": "no_model", "engine": "pocket"},
        {"language": "hi", "task": "stt", "status": "unsupported", "engine": "shadow"},
    ])
    result = run("--capabilities", path)
    assert result.returncode == 0
    assert "awaiting an approved model" in result.stdout
    assert "no_model (1): bn/tts" in result.stdout
    assert "unsupported (1): hi/stt" in result.stdout


def test_broken_capability_file_is_a_failure(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    result = run("--capabilities", str(bad))
    assert result.returncode == 1
    assert "FAIL     capability data" in result.stdout
    assert "not valid JSON" in result.stdout


def test_capability_entry_with_unknown_status_fails_loudly(tmp_path):
    path = tmp_path / "caps.json"
    path.write_text(json.dumps([{"language": "en", "task": "stt", "status": "probably"}]), encoding="utf-8")
    result = run("--capabilities", str(path))
    assert result.returncode == 1
    assert "unknown task or status" in result.stdout


def test_network_import_in_a_scanned_tree_fails_the_run(tmp_path):
    (tmp_path / "leaky.py").write_text("import socket\n", encoding="utf-8")
    result = run("--scan-path", str(tmp_path))
    assert result.returncode == 1
    assert "FAIL" in result.stdout
    assert "leaky.py:1" in result.stdout


def test_strict_upgrades_not_run_to_a_nonzero_exit(tmp_path):
    path = _capabilities(tmp_path / "caps.json", [
        {"language": "bn", "task": "tts", "status": "no_model", "engine": "pocket"},
    ])
    assert run("--capabilities", path).returncode == 0
    strict = run("--capabilities", path, "--strict")
    assert strict.returncode == 2
    assert "NOT_RUN rows, escalated by --strict" in strict.stdout


def test_json_output_is_machine_readable(tmp_path):
    path = _capabilities(tmp_path / "caps.json", [
        {"language": "en", "task": "stt", "status": "measured", "engine": "e", "metric_name": "WER",
         "metric_value": 0.1, "measured_on": "2026-10-12"},
    ])
    result = run("--capabilities", path, "--max-wer", "0.2", "--json")
    payload = json.loads(result.stdout)
    assert payload["exit_code"] == 0
    assert payload["report"]["summary"]["passed"] >= 1
    assert payload["capabilities"].startswith("language")
    assert payload["report"]["host"]


def test_first_audio_threshold_is_a_separate_criterion(tmp_path):
    path = _capabilities(tmp_path / "caps.json", [
        {"language": "en", "task": "tts", "status": "measured", "engine": "e", "metric_name": "first_audio_ms",
         "metric_value": 420.0, "measured_on": "2026-10-12"},
    ])
    passing = run("--capabilities", path, "--max-first-audio-ms", "500")
    assert passing.returncode == 0 and "PASS     en/tts first_audio_ms" in passing.stdout
    failing = run("--capabilities", path, "--max-first-audio-ms", "300")
    assert failing.returncode == 1


def test_version_flag_short_circuits():
    result = run("--version")
    assert result.returncode == 0
    assert "ultron-native-voice" in result.stdout


def test_help_lists_the_capability_flag():
    result = run("--help")
    assert result.returncode == 0
    for flag in ("--capabilities", "--scan-path", "--max-wer", "--strict", "--json"):
        assert flag in result.stdout


@pytest.mark.parametrize("status", ["measured", "declared", "no_model", "unsupported", "partial"])
def test_every_status_produces_exactly_one_verdict_row(tmp_path, status):
    path = _capabilities(tmp_path / f"caps-{status}.json", [
        {"language": "en", "task": "stt", "status": status, "engine": "e",
         "metric_name": "WER", "metric_value": 0.05, "measured_on": "2026-10-12"},
    ])
    result = run("--capabilities", path, "--max-wer", "0.1")
    rows = [line for line in result.stdout.splitlines() if line.startswith(("PASS ", "FAIL ", "NOT_RUN ")) and "en/stt" in line]
    assert len(rows) == 1, result.stdout
    if status == "measured":
        assert rows[0].startswith("PASS")
    else:
        assert rows[0].startswith("NOT_RUN")
