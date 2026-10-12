"""Acceptance report invariants, above all: NOT_RUN never counts as passing."""

from __future__ import annotations

import json
import math

from nativevoice.eval.report import (
    AcceptanceReport,
    CriterionResult,
    CriterionStatus,
    Evidence,
    build_report,
    host_summary,
    judge,
    not_run,
)


def test_judge_passes_and_fails_on_max_criteria():
    assert judge("first audio", 180.0, 250.0, unit=" ms").status is CriterionStatus.PASS
    assert judge("first audio", 900.0, 250.0, unit=" ms").status is CriterionStatus.FAIL
    assert judge("first audio", 250.0, 250.0, unit=" ms").status is CriterionStatus.PASS


def test_judge_min_direction_is_the_mirror_image():
    assert judge("coverage", 0.9, 0.8, direction="min").status is CriterionStatus.PASS
    assert judge("coverage", 0.4, 0.8, direction="min").status is CriterionStatus.FAIL


def test_no_measurement_is_never_zero_and_never_a_pass():
    row = judge("wer", None, 0.15, unit="")
    assert row.status is CriterionStatus.NOT_RUN
    assert row.measured is None
    assert row.passes is None
    assert "no measurement" in row.reason


def test_zero_is_a_real_value_and_is_judged():
    assert judge("wer", 0.0, 0.15).status is CriterionStatus.PASS
    assert judge("clipping", 0.0, 0.0).status is CriterionStatus.PASS


def test_missing_threshold_cannot_be_met():
    row = judge("latency", 12.0, None, unit=" ms")
    assert row.status is CriterionStatus.NOT_RUN
    assert "no threshold" in row.reason


def test_unknown_direction_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        judge("x", 1.0, 2.0, direction="sideways")


def test_summarise_counts_the_three_statuses_separately():
    report = AcceptanceReport("acceptance")
    report.add(judge("a", 1.0, 2.0))
    report.add(judge("b", 5.0, 2.0))
    report.add(not_run("c", "no approved model"))
    report.add(not_run("d", "no corpus with consented audio"))
    summary = report.summarise()
    assert (summary.passed, summary.failed, summary.not_run, summary.total) == (1, 1, 2, 4)
    assert summary.verdict == "REJECTED"
    assert math.isclose(summary.coverage, 0.5)


def test_not_run_blocks_acceptance_even_with_no_failures():
    report = AcceptanceReport("acceptance")
    report.add(judge("a", 1.0, 2.0))
    report.add(not_run("b", "no data"))
    summary = report.summarise()
    assert summary.failed == 0
    assert summary.verdict == "INCOMPLETE"
    rendered = report.render()
    assert "NOT_RUN" in rendered and "verdict: INCOMPLETE" in rendered


def test_not_run_never_enters_the_passing_set_under_any_view():
    """The invariant the specification turns on: unknown is not success."""
    report = AcceptanceReport("acceptance")
    rows = [not_run(f"gap-{i}", "no measurement") for i in range(5)]
    rows.append(judge("real", 0.1, 0.2))
    report.extend(rows)
    summary = report.summarise()
    assert summary.not_run == 5 and summary.passed == 1
    assert summary.passed + summary.failed + summary.not_run == summary.total
    assert all(row.passes is None for row in report.by_status(CriterionStatus.NOT_RUN))
    assert report.by_status(CriterionStatus.PASS) == (rows[-1],)
    assert not any(row.passes for row in report.by_status(CriterionStatus.NOT_RUN))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["summary"]["passed"] == 1
    assert [row["status"] for row in payload["criteria"]].count("NOT_RUN") == 5
    # Rendering keeps the words distinct so a scanner cannot merge NOT_RUN into PASS:
    # one status word per row line, and the tally line is spelled differently again.
    rendered = report.render()
    status_lines = [line.split()[0] for line in rendered.splitlines() if line.startswith(("PASS ", "FAIL ", "NOT_RUN "))]
    assert status_lines.count("NOT_RUN") == 5
    assert status_lines.count("PASS") == 1
    assert "PASS=1 FAIL=0 NOT_RUN=5" in rendered


def test_evidence_carries_command_timestamp_host_and_before_after():
    before = {"baseline": "none"}
    evidence = Evidence.capture("python -m nativevoice.cli.diagnose --capabilities x.json", before=before, started_s=0.0)
    assert evidence.command.startswith("python -m")
    assert evidence.captured_at and evidence.host == host_summary()
    assert evidence.before == before
    assert "pid" in evidence.after and "cwd" in evidence.after
    assert evidence.duration_s is not None and evidence.duration_s >= 0


def test_report_write_json_is_rereadable_and_complete(tmp_path):
    report = build_report("acceptance", [judge("wer", 0.1, 0.2, evidence=Evidence.capture("cmd")), not_run("tts-bn", "no model")])
    path = report.write_json(str(tmp_path / "out" / "report.json"))
    payload = json.load(open(path, encoding="utf-8"))
    assert payload["summary"]["passed"] == 1
    assert payload["criteria"][0]["evidence"]["command"] == "cmd"
    assert payload["criteria"][1]["measured"] is None
    assert payload["criteria"][1]["reason"] == "no model"


def test_rendered_report_names_the_measured_value_and_bound():
    report = build_report("latency gate", [judge("first audio", 210.0, 200.0, unit=" ms", language="bn")])
    rendered = report.render()
    assert "FAIL" in rendered
    assert "210.0000 ms" in rendered and "limit max 200.0 ms" in rendered
    assert "[bn]" in rendered


def test_criterion_result_defaults_keep_passes_none_for_not_run():
    row = CriterionResult("x", CriterionStatus.NOT_RUN, None, None, "", "max")
    assert row.passes is None
    assert row.measured_text == "no measurement"
