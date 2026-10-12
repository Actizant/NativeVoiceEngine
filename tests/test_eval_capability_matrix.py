"""The matrix must never collapse DECLARED into MEASURED."""

from __future__ import annotations

import json

import pytest

from nativevoice.core.language import Capability, CapabilityStatus, Task
from nativevoice.eval.capability_matrix import STATUS_SYMBOL, build_matrix

def cap(task: Task, status: CapabilityStatus, language: str, **kw) -> tuple[str, Capability]:
    return (language, Capability(task=task, status=status, language=language, **kw))


@pytest.fixture
def entries():
    return [
        cap(Task.STT, CapabilityStatus.MEASURED, "en", engine="shadow", metric_name="WER", metric_value=0.083,
            measured_on="2026-10-12", model_id="shadow-en", evidence="eval:en-stt:1"),
        cap(Task.STT, CapabilityStatus.DECLARED, "bn", engine="shadow", metric_name="WER", metric_value=0.31,
            model_id="shadow-multilingual"),
        cap(Task.TTS, CapabilityStatus.NO_MODEL, "bn", engine="pocket"),
        cap(Task.STT, CapabilityStatus.UNSUPPORTED, "hi", engine="shadow"),
        cap(Task.TTS, CapabilityStatus.PARTIAL, "en", engine="pocket", limitations=("no prosody control",)),
        cap(Task.LID, CapabilityStatus.MEASURED, "en", engine="shadow", metric_name="accuracy", metric_value=0.91,
            measured_on="2026-10-12"),
    ]


def test_matrix_shape_and_lookup(entries):
    matrix = build_matrix(entries)
    assert matrix.languages == ("en", "bn", "hi")
    assert set(matrix.tasks) >= {"stt", "tts", "lid"}
    measured = matrix.cell("en", "stt")
    assert measured.status is CapabilityStatus.MEASURED
    assert measured.metric_value == pytest.approx(0.083)
    assert measured.evidence == "eval:en-stt:1"
    assert matrix.cell("hi", "tts") is None


def test_pinned_task_columns_keep_absent_capabilities_visible(entries):
    matrix = build_matrix(entries, tasks=[Task.STT, Task.TTS, Task.VOICE_CLONE])
    assert "voice_clone" in matrix.tasks
    rendered = matrix.render_text()
    assert "(none)" in rendered


def test_declared_and_measured_render_differently(entries):
    """This is the exact collapse the project must not reproduce."""
    matrix = build_matrix(entries)
    assert matrix.cell("en", "stt").symbol == STATUS_SYMBOL[CapabilityStatus.MEASURED]
    assert matrix.cell("bn", "stt").symbol == STATUS_SYMBOL[CapabilityStatus.DECLARED]
    assert matrix.cell("en", "stt").symbol != matrix.cell("bn", "stt").symbol
    declared = matrix.cell("bn", "stt").describe()
    assert "declared only" in declared and "not evaluated on this machine" in declared
    # A vendor number is shown but labelled as unverified instead of being dropped.
    assert "unverified" in declared and "0.3100" in declared
    rendered = matrix.render_text()
    assert "M=measured here" in rendered and "d=declared by upstream, unverified" in rendered
    assert "declared only" in rendered


def test_verdict_separates_measured_from_implemented_from_awaiting(entries):
    verdict = build_matrix(entries).verdict()
    assert verdict.measured == 2
    assert verdict.declared == 1
    assert verdict.partial == 1
    assert verdict.unsupported == 1
    assert verdict.awaiting_model == 1
    assert verdict.total == 6
    assert verdict.implemented == verdict.measured + verdict.partial == 3
    assert verdict.unmeasured == 4


def test_measured_pairs_filter_and_dict_shape(entries):
    matrix = build_matrix(entries)
    assert [cell.language for cell in matrix.measured_pairs("stt")] == ["en"]
    payload = json.loads(json.dumps(matrix.to_dict()))
    assert payload["verdict"]["measured"] == 2
    assert payload["symbols"]["declared"] == STATUS_SYMBOL[CapabilityStatus.DECLARED]
    cell = next(row for row in payload["cells"] if row["language"] == "bn" and row["task"] == "stt")
    assert cell["status"] == "declared"
    assert cell["described_as"].startswith("declared only")


def test_missing_metric_on_a_measured_cell_is_called_out():
    matrix = build_matrix([cap(Task.STT, CapabilityStatus.MEASURED, "en", engine="shadow")])
    assert matrix.verdict().measured == 1
    assert matrix.cell("en", "stt").has_metric is False
    assert "claimed measured but no metric value was recorded" in matrix.cell("en", "stt").describe()
    # The bookkeeping failure is surfaced in the human view, not hidden by the symbol.
    assert "no metric value" in matrix.render_text()


def test_partial_cell_must_state_its_gaps():
    matrix = build_matrix([cap(Task.TTS, CapabilityStatus.PARTIAL, "hi", engine="pocket")])
    assert matrix.cell("hi", "tts").describe() == "partial: gaps unstated"
    stated = build_matrix([cap(Task.TTS, CapabilityStatus.PARTIAL, "hi", engine="pocket", limitations=("no nasal vowels",))])
    assert "no nasal vowels" in stated.cell("hi", "tts").describe()


def test_empty_matrix_renders_without_claiming_anything():
    matrix = build_matrix([])
    assert matrix.verdict().total == 0
    assert matrix.render_text().startswith("language")
    assert matrix.languages == () and matrix.tasks == ()


def test_every_capability_status_is_fully_wired():
    """Adding a status without wiring it is how a report silently misrenders a cell.

    A KeyError in the renderer, an uncounted bucket in the verdict, or a cell that
    falls through to the default description all make a capability report read as
    though nothing changed. This test is the seam that notices.
    """
    from nativevoice.core.language import CapabilityStatus, Task
    from nativevoice.eval.capability_matrix import STATUS_SYMBOL, CapabilityMatrix, Cell

    cells = []
    for status in CapabilityStatus:
        assert status in STATUS_SYMBOL, f"{status.value} has no symbol"
        cells.append(
            Cell(
                language="xx",
                task=Task.STT.value,
                status=status,
                metric_name="wer" if status is not CapabilityStatus.NO_MODEL else "",
                metric_value=None if status is CapabilityStatus.NO_MODEL else 0.42,
                limitations=("detail",),
            )
        )
    matrix = CapabilityMatrix(languages=("xx",), tasks=(Task.STT.value,), cells=tuple(cells))
    for cell in cells:
        described = cell.describe()
        assert described and described != "awaiting an approved model" or cell.status is CapabilityStatus.NO_MODEL
    verdict = matrix.verdict()
    assert verdict.total == len(cells)
    assert verdict.measured_failed == 1
    assert verdict.implemented == 2  # MEASURED + PARTIAL only; a failed measurement is not implemented
    assert verdict.awaiting_model == 1
