"""Synthesis measurement: latency, RTF, integrity flags, and round-trip honesty."""

from __future__ import annotations

import array
import math
from dataclasses import dataclass, field

import pytest
from fixtures import sine, silence

from nativevoice.core.audio import AudioBuffer, AudioFormat
from nativevoice.core.events import SynthesisEvent, SynthesisKind
from nativevoice.core.language import Capability, CapabilityStatus, LanguageTag, Task
from nativevoice.eval.report import CriterionStatus
from nativevoice.eval.synthesis import (
    StreamAccumulator,
    clipping_fraction,
    intelligibility_proxy_check,
    measure_outcome,
    measure_synthesis,
    silence_only,
)

RATE = 16000


def _buffer(seconds: float, amplitude: float = 0.2, rate: int = RATE) -> AudioBuffer:
    """Exact-length audible tone: `speech_like` pads a trailing tail, which would make
    every duration assertion in this file test the fixture instead of the metric."""
    return sine(rate=rate, seconds=seconds, amplitude=amplitude)


@dataclass(frozen=True, slots=True)
class Request:
    """Duck-typed stand-in for `tts.api.SynthesisRequest` (only the ceiling is read)."""

    maximum_duration_s: float = 5.0


def test_latency_rtf_and_duration_ratio_are_derived_from_audio():
    audio = _buffer(2.0)
    measured = measure_synthesis(
        audio,
        wall_time_ms=1000.0,
        first_audio_ms=120.0,
        maximum_duration_s=4.0,
        engine="test-engine",
        model_id="test-model",
        language="en",
    )
    assert math.isclose(measured.audio_duration_s, 2.0, rel_tol=1e-6)
    assert math.isclose(measured.real_time_factor, 0.5, rel_tol=1e-6)
    assert measured.is_faster_than_real_time is True
    assert math.isclose(measured.duration_ratio, 0.5, rel_tol=1e-6)
    assert measured.first_audio_ms == 120.0
    assert measured.overran_maximum is False
    assert measured.silence_only is False


def test_missing_first_audio_mark_stays_none_with_a_reason():
    measured = measure_synthesis(_buffer(1.0), wall_time_ms=500.0)
    assert measured.first_audio_ms is None
    assert any("first-audio latency NOT RUN" in note for note in measured.notes)


def test_slower_than_real_time_is_reported_not_rounded_away():
    measured = measure_synthesis(_buffer(1.0), wall_time_ms=4000.0, maximum_duration_s=2.0)
    assert math.isclose(measured.real_time_factor, 4.0, rel_tol=1e-6)
    assert measured.is_faster_than_real_time is False


def test_zero_audio_produces_no_fabricated_rtf():
    empty = AudioBuffer.zeros(AudioFormat(RATE, 1), 0)
    measured = measure_synthesis(empty, wall_time_ms=10.0, maximum_duration_s=2.0)
    assert measured.real_time_factor is None
    # A ratio of 0.0 is a true measurement here (nothing produced against a 2 s ceiling);
    # it is the RTF that cannot exist, and the note says exactly that.
    assert measured.duration_ratio == 0.0
    assert measured.silence_only is True
    assert any("zero frames" in note for note in measured.notes)
    assert any("real-time factor NOT RUN" in note for note in measured.notes)


def test_eos_overrun_is_detected_from_produced_duration():
    """A run-on generation exceeds its ceiling; the ratio near 1.0 is the tell."""
    audio = _buffer(6.0)
    measured = measure_synthesis(audio, wall_time_ms=6000.0, maximum_duration_s=5.0, eos_seen=False)
    assert measured.overran_maximum is True
    assert math.isclose(measured.duration_ratio, 1.2, rel_tol=1e-6)
    assert any("end-of-sequence" in note for note in measured.notes)


def test_clipping_fraction_is_measured_not_assumed():
    loud = AudioBuffer(array.array("h", [32767, -32768] * 800), AudioFormat(RATE, 1))
    quiet = _buffer(0.2, amplitude=0.05)
    assert clipping_fraction(loud) > 0.9
    assert clipping_fraction(quiet) == 0.0
    assert silence_only(quiet) is False
    assert silence_only(silence(seconds=0.5)) is True
    assert silence_only(AudioBuffer.zeros(AudioFormat(RATE, 1), 0)) is True


def test_measure_outcome_reads_engine_published_timings():
    @dataclass
    class Outcome:
        audio: AudioBuffer
        engine: str = "eng"
        model_id: str = "mdl"
        language: LanguageTag = field(default_factory=lambda: LanguageTag("hi"))
        timings: dict = field(default_factory=lambda: {"first_audio_ms": 42.0, "total_ms": 210.0})
        truncated: bool = False

    outcome = Outcome(audio=_buffer(1.0))
    measured = measure_outcome(outcome, Request(maximum_duration_s=3.0))
    assert measured.first_audio_ms == 42.0
    assert math.isclose(measured.wall_time_ms, 210.0, rel_tol=1e-9)
    assert math.isclose(measured.real_time_factor, 0.21, rel_tol=1e-6)
    assert measured.language == "hi" and measured.model_id == "mdl"


def test_measure_outcome_falls_back_to_buffer_timestamps():
    audio = _buffer(0.5)
    audio.started_at_ns = 1_000_000

    @dataclass
    class Outcome:
        audio: AudioBuffer
        timings: dict = field(default_factory=lambda: {"first_audio_ns": 51_000_000, "finished_ns": 200_000_000})

    measured = measure_outcome(Outcome(audio=audio), Request(maximum_duration_s=2.0))
    assert math.isclose(measured.first_audio_ms, 50.0, rel_tol=1e-9)
    assert math.isclose(measured.wall_time_ms, 199.0, rel_tol=1e-9)


def test_stream_accumulator_takes_first_audio_from_the_first_real_chunk():
    started = 10_000_000
    accumulator = StreamAccumulator(started_ns=started)
    events = [
        SynthesisEvent(kind=SynthesisKind.FIRST_AUDIO, sequence=0, audio=AudioBuffer.zeros(AudioFormat(RATE, 1), 0)),
        SynthesisEvent(kind=SynthesisKind.AUDIO, sequence=1, audio=_buffer(0.25)),
        SynthesisEvent(kind=SynthesisKind.AUDIO, sequence=3, audio=_buffer(0.25)),
        SynthesisEvent(kind=SynthesisKind.EOS, sequence=4),
    ]
    accumulator.consume(iter(events))
    assert accumulator.eos_seen is True
    assert accumulator.sequence_gaps == 1
    assert accumulator.merged().duration_s == pytest.approx(0.5, abs=1e-6)


def test_stream_without_eos_is_reported_as_cut_off():
    accumulator = StreamAccumulator(started_ns=0)
    accumulator.consume(iter([SynthesisEvent(kind=SynthesisKind.AUDIO, sequence=0, audio=_buffer(0.2))]))
    measured = accumulator.measurement(request=Request(maximum_duration_s=5.0), engine="e", model_id="m")
    assert measured.eos_seen is False
    assert any("without EOS" in note for note in measured.notes)


def test_stream_that_never_emits_audio_is_not_zero_latency():
    accumulator = StreamAccumulator(started_ns=0)
    accumulator.consume(iter([SynthesisEvent(kind=SynthesisKind.FAILED, sequence=0, reason="engine died")]))
    measured = accumulator.measurement(request=Request(maximum_duration_s=1.0), engine="e")
    assert measured.first_audio_ms is None
    assert measured.real_time_factor is None
    assert any("engine died" in note for note in measured.notes)


# --- round-trip intelligibility ------------------------------------------------


@dataclass
class StubTranscriber:
    capability: Capability | None
    text: str = "the quick brown fox"
    engine_id: str = "stub-stt"
    model_id: str = "stub-model-1"
    raises: Exception | None = None

    def language_capability(self, language: LanguageTag) -> Capability:
        if self.raises is not None:
            raise self.raises
        if self.capability is None:
            raise LookupError("no capability registered")
        return self.capability

    def transcribe(self, request):
        @dataclass
        class Result:
            text: str

        return Result(self.text)


@dataclass
class FakeRequest:
    audio: AudioBuffer
    language: LanguageTag
    max_duration_s: float
    token: object


def _capability(status: CapabilityStatus, *, model_id: str = "stub-model-1") -> Capability:
    return Capability(task=Task.STT, status=status, language="en", engine="stub-stt", model_id=model_id)


def _build(audio, tag, max_duration_s, token):
    return FakeRequest(audio=audio, language=tag, max_duration_s=max_duration_s, token=token)


def test_round_trip_without_a_transcriber_is_not_run():
    check = intelligibility_proxy_check(_buffer(1.0), "the quick brown fox", "en")
    assert check.status is CriterionStatus.NOT_RUN
    assert check.round_trip_wer is None
    assert "no transcriber supplied" in check.reason
    assert check.measured is False


def test_round_trip_with_no_approved_model_is_not_run():
    transcriber = StubTranscriber(_capability(CapabilityStatus.NO_MODEL))
    check = intelligibility_proxy_check(_buffer(1.0), "the quick brown fox", "en", transcriber=transcriber)
    assert check.status is CriterionStatus.NOT_RUN
    assert "no approved model" in check.reason
    assert check.round_trip_wer is None


def test_round_trip_refuses_a_transcriber_that_names_no_model():
    transcriber = StubTranscriber(_capability(CapabilityStatus.MEASURED, model_id=""))
    transcriber.model_id = ""
    check = intelligibility_proxy_check(_buffer(1.0), "the quick brown fox", "en", transcriber=transcriber)
    assert check.status is CriterionStatus.NOT_RUN
    assert "model_id" in check.reason


def test_round_trip_needs_an_installed_engine_not_a_vendor_claim_alone():
    """UNSUPPORTED/NO_MODEL refuse; an engine that actually transcribes yields a number.

    DECLARED still executes because a model physically present produced the text - the
    status qualifies *who asserted* the language support, and the round-trip result
    itself is measured here rather than inherited from a model card.
    """
    declared = intelligibility_proxy_check(
        _buffer(1.0), "the quick brown fox", "en", transcriber=StubTranscriber(_capability(CapabilityStatus.DECLARED)), request_builder=_build
    )
    assert declared.round_trip_wer == 0.0
    unsupported = intelligibility_proxy_check(
        _buffer(1.0), "the quick brown fox", "en", transcriber=StubTranscriber(_capability(CapabilityStatus.UNSUPPORTED))
    )
    assert unsupported.status is CriterionStatus.NOT_RUN


def test_round_trip_passes_only_on_a_perfect_real_transcription():
    check = intelligibility_proxy_check(
        _buffer(1.0), "the quick brown fox", "en", transcriber=StubTranscriber(_capability(CapabilityStatus.MEASURED)), request_builder=_build
    )
    assert check.status is CriterionStatus.PASS
    assert check.round_trip_wer == 0.0
    assert check.model_id == "stub-model-1"


def test_round_trip_fail_carries_the_measured_wer():
    transcriber = StubTranscriber(_capability(CapabilityStatus.MEASURED), text="the quick brown cat")
    check = intelligibility_proxy_check(_buffer(1.0), "the quick brown fox", "en", transcriber=transcriber, request_builder=_build)
    assert check.status is CriterionStatus.FAIL
    assert math.isclose(check.round_trip_wer, 0.25, rel_tol=1e-12)
    assert check.hypotheses == ("the quick brown cat",)


def test_round_trip_is_not_run_when_no_audio_came_out():
    empty = AudioBuffer.zeros(AudioFormat(RATE, 1), 0)
    check = intelligibility_proxy_check(empty, "whatever", "en", transcriber=StubTranscriber(_capability(CapabilityStatus.MEASURED)))
    assert check.status is CriterionStatus.NOT_RUN
    assert "no audio" in check.reason


def test_round_trip_is_not_run_when_the_capability_probe_itself_fails():
    transcriber = StubTranscriber(None)
    check = intelligibility_proxy_check(_buffer(1.0), "text", "en", transcriber=transcriber)
    assert check.status is CriterionStatus.NOT_RUN
    assert "capability" in check.reason


def test_round_trip_is_not_run_when_the_request_cannot_be_built():
    transcriber = StubTranscriber(_capability(CapabilityStatus.MEASURED))

    def broken(audio, tag, max_duration_s, token):
        raise RuntimeError("recognition contracts unavailable: no module")

    check = intelligibility_proxy_check(_buffer(1.0), "text", "en", transcriber=transcriber, request_builder=broken)
    assert check.status is CriterionStatus.NOT_RUN
    assert "no recognition request" in check.reason


def test_proxy_check_serialises_for_a_report():
    import json

    check = intelligibility_proxy_check(_buffer(1.0), "the quick brown fox", "en")
    payload = json.loads(json.dumps(check.as_dict()))
    assert payload["status"] == "NOT_RUN"
    assert payload["round_trip_wer"] is None
