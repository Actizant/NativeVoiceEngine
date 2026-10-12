"""Diagnostics: log sink privacy/filtering and the per-stage latency trace."""

from __future__ import annotations

import json

import pytest
from fixtures import silence, speech_like

from nativevoice.core.events import DiagnosticEvent
from nativevoice.core.language import LanguageTag
from nativevoice.core.time import Timings, monotonic_ns
from nativevoice.diagnostics.log import REDACTED, DiagnosticSink, DiagnosticSinkError
from nativevoice.diagnostics.timeline import STAGES, Trace


# --- log ----------------------------------------------------------------------


def test_sink_is_a_ring_buffer_with_an_overflow_counter():
    sink = DiagnosticSink(capacity=3)
    for index in range(6):
        sink.emit("vad", "speech_start", seq=index)
    assert len(sink.events) == 3
    assert sink.accepted == 6
    assert sink.dropped == 3
    assert [event.fields["seq"] for event in sink.events] == [3, 4, 5]


def test_query_filters_by_component_event_and_level():
    sink = DiagnosticSink(minimum_level="debug")
    sink.emit("stt", "decode_start", "debug")
    sink.emit("stt", "decode_done", "info", ms=12)
    sink.emit("tts", "decode_done", "warn")
    assert len(sink.query(component="stt")) == 2
    assert len(sink.query(event="decode_done")) == 2
    assert len(sink.query(level="warn")) == 1
    assert sink.query(component="nope") == ()
    assert sink.query(limit=1)[0].event == "decode_done"


def test_level_threshold_filters_before_storage():
    sink = DiagnosticSink(minimum_level="warn")
    assert sink.emit("stt", "quiet", "info") is None
    assert sink.emit("stt", "loud", "error") is not None
    assert len(sink.events) == 1


def test_transcripts_and_audio_are_redacted_by_default():
    sink = DiagnosticSink()
    sink.emit("stt", "final", text="the user's private sentence", samples=bytes(4096), ms=31.5)
    stored = sink.events[0]
    assert stored.fields["text"] == REDACTED
    assert stored.fields["text_length"] == len("the user's private sentence")
    assert stored.fields["samples"] == REDACTED
    assert stored.fields["samples_length"] == 4096
    assert stored.fields["ms"] == 31.5
    assert "private sentence" not in sink.to_jsonl()


def test_redaction_can_be_opted_into_per_field():
    sink = DiagnosticSink(keep_fields=frozenset({"text"}))
    sink.emit("stt", "final", text="kept because the operator opted in", ms=1)
    assert sink.events[0].fields["text"].startswith("kept")


def test_nested_fields_are_redacted_too():
    sink = DiagnosticSink()
    sink.emit("session", "turn", detail={"text": "secret", "ms": 4})
    nested = sink.events[0].fields["detail"]
    assert nested["text"] == REDACTED and nested["ms"] == 4
    # A container whose own name is user content is dropped wholesale, not walked into.
    sink.emit("session", "turn", utterance={"text": "secret"})
    assert sink.events[1].fields["utterance"] == REDACTED


def test_jsonl_output_is_one_object_per_line_and_matches_the_file(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    with DiagnosticSink(path=str(path)) as sink:
        sink.emit("capture", "device_open", device="default")
        sink.emit("capture", "device_close", device="default")
        rendered = sink.to_jsonl().strip().splitlines()
    assert len(rendered) == 2
    rows = [json.loads(line) for line in rendered]
    assert rows[0]["component"] == "capture" and rows[0]["fields"]["device"] == "default"
    assert rows[0]["timestamp_ns"] > 0
    on_disk = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert on_disk == rows
    assert sink.written == 2


def test_unwritable_destination_is_reported_not_swallowed(tmp_path):
    """`tmp_path` is a directory, so opening it as a log file must raise, not drop."""
    with pytest.raises(DiagnosticSinkError):
        DiagnosticSink(path=str(tmp_path))


def test_counts_are_per_component():
    sink = DiagnosticSink()
    sink.emit("stt", "a")
    sink.emit("stt", "b")
    sink.emit("tts", "c")
    assert sink.counts() == {"stt": 2, "tts": 1}


def test_record_accepts_a_prebuilt_diagnostic_event():
    sink = DiagnosticSink()
    sink.record(DiagnosticEvent(component="router", event="fallback", level="warn", fields={"reason": "language unsupported"}))
    assert sink.query(component="router")[0].fields["reason"] == "language unsupported"


# --- timeline ----------------------------------------------------------------


def _timings(ms: float) -> Timings:
    return Timings(started_ns=0, first_result_ns=int(ms * 1e6) // 2, finished_ns=int(ms * 1e6), audio_duration_s=1.0)


def test_trace_aggregates_median_and_p95_per_stage_per_language():
    trace = Trace()
    for index, ms in enumerate([10.0, 20.0, 30.0, 40.0, 200.0]):
        trace.record("stt", _timings(ms), language="en", operation_id=f"op{index}")
    for ms in [800.0, 900.0]:
        trace.record("stt", _timings(ms), language="bn", operation_id=f"op{ms}")
    english = trace.stats("stt", "en")
    bengali = trace.stats("stt", "bn")
    assert english.count == 5 and bengali.count == 2
    assert english.median_ms == 30.0
    assert english.p95_ms == 200.0
    assert bengali.p95_ms == 900.0
    # The per-language split is what makes the regression attributable at all.
    assert trace.bottleneck("bn").stage == "stt"
    assert trace.bottleneck("bn").p95_ms > trace.bottleneck("en").p95_ms


def test_unknown_stage_is_rejected():
    trace = Trace()
    with pytest.raises(ValueError):
        trace.record("magic", _timings(5.0))
    with pytest.raises(ValueError):
        with trace.timed("magic"):
            pass


def test_timed_context_records_wall_time_and_annotations():
    trace = Trace()
    with trace.timed("tts", language=LanguageTag("hi"), operation_id="op-1") as out:
        total = sum(range(5000))
        out["first_result_ms"] = 3.0
        out["real_time_factor"] = 0.42
    assert total == sum(range(5000))
    span = trace.spans[0]
    assert span.stage == "tts" and span.language == "hi" and span.ms >= 0
    assert span.first_result_ms == 3.0 and span.real_time_factor == 0.42


def test_missing_stages_are_named_instead_of_defaulted_to_zero():
    trace = Trace()
    trace.record("capture", _timings(5.0), language="en", operation_id="op")
    trace.record("stt", _timings(50.0), language="en", operation_id="op")
    assert trace.missing_stages("op") == tuple(s for s in STAGES if s not in ("capture", "stt"))
    assert trace.stats("vad", "en") is None
    assert "vad" not in [row.stage for row in trace.all_stats()]


def test_compare_reports_no_baseline_instead_of_a_fake_percentage():
    old = Trace()
    old.record("stt", _timings(100.0), language="en", operation_id="a")
    new = Trace()
    new.record("stt", _timings(400.0), language="en", operation_id="b")
    new.record("playback", _timings(20.0), language="en", operation_id="b")
    rows = {row["stage"]: row for row in new.compare(old, tolerance_pct=25.0)}
    assert rows["stt"]["verdict"] == "regressed"
    assert rows["stt"]["delta_pct"] == pytest.approx(300.0)
    assert rows["playback"]["verdict"] == "no baseline"
    assert rows["playback"]["delta_pct"] is None
    assert rows["playback"]["baseline_median_ms"] is None


def test_empty_trace_asserts_nothing():
    trace = Trace()
    assert "no spans recorded" in trace.render_text()
    assert trace.to_dict()["spans"] == []
    assert trace.bottleneck("en") is None


def test_first_audio_latency_survives_into_the_stats():
    trace = Trace()
    trace.record("tts", Timings(started_ns=0, first_result_ns=45_000_000, finished_ns=150_000_000, audio_duration_s=3.0), language="en", operation_id="x")
    span = trace.spans[0]
    assert span.first_result_ms == pytest.approx(45.0)
    assert span.real_time_factor == pytest.approx(0.05)


def test_timeline_serialises_for_a_report():
    trace = Trace()
    trace.record("vad", _timings(4.0), language="bn", operation_id="op")
    payload = json.loads(json.dumps(trace.to_dict()))
    assert payload["stats"][0]["stage"] == "vad"
    assert payload["missing_stages"]["op"]


def test_quiet_input_still_records_a_span():
    """A silent utterance is a real measurement, not an absence of one."""
    trace = Trace()
    buffer = silence(seconds=0.4)
    with trace.timed("capture", language="en", operation_id="silence"):
        assert buffer.frame_count > 0
    with trace.timed("vad", language="en", operation_id="silence"):
        assert speech_like().frame_count > 0
    assert {span.stage for span in trace.spans} == {"capture", "vad"}
    assert monotonic_ns() > 0
