"""Pipeline routing, streaming, cancellation and assembly contracts."""

from __future__ import annotations

import array
import math

import pytest

from nativevoice.core.audio import AudioBuffer, AudioFormat
from nativevoice.core.cancel import CancellationToken
from nativevoice.core.errors import (
    EngineUnavailable,
    LanguageUnsupported,
    ModelNotInstalled,
    ProfileInvalid,
)
from nativevoice.core.events import DiagnosticEvent, SynthesisEvent, SynthesisKind
from nativevoice.core.language import Capability, CapabilityStatus, LanguageTag, Task
from nativevoice.tts.adapters.self_test import PLUMBING_LANGUAGE, ScriptedSynthesisEngine
from nativevoice.tts.api import Prosody, SynthesisRequest, VoiceReference
from nativevoice.tts.chunking import TextChunker
from nativevoice.tts.engine_registry import EngineRegistry
from nativevoice.tts.normalise import TextNormaliser
from nativevoice.tts.pipeline import SpeechPipeline

FORMAT = AudioFormat(24000, 1)
BENGALI = LanguageTag("bn")
THREE_SENTENCES = "One sentence here. Another sentence follows. And a third one ends it."


def tone(freq: float, seconds: float, rate: int = 24000, amplitude: int = 8000) -> AudioBuffer:
    frames = int(rate * seconds)
    return AudioBuffer(
        array.array("h", (int(amplitude * math.sin(2 * math.pi * freq * i / rate)) for i in range(frames))),
        AudioFormat(rate, 1),
    )


def request_for(
    text: str = THREE_SENTENCES,
    language: LanguageTag = LanguageTag(PLUMBING_LANGUAGE),
    **kwargs,
) -> SynthesisRequest:
    return SynthesisRequest(text=text, language=language, **kwargs)


class DeclaredOnlyEngine:
    """Vendor-claim-only engine: exists to prove `strict` means something."""

    engine_id = "declared-only"
    model_id = "unverified-card"
    output_format = FORMAT
    supports_streaming = False
    controllable = ()

    def __init__(self, status: CapabilityStatus = CapabilityStatus.DECLARED) -> None:
        self.status = status

    def capabilities(self):
        return (
            Capability(task=Task.TTS, status=self.status, language="bn", engine=self.engine_id),
        )

    def voices(self):
        return ()

    def condition(self, reference):
        return reference

    def warm(self):
        return None

    def synthesize(self, request):
        raise AssertionError("a refused engine must never be asked to synthesise")

    def stream(self, request):
        return iter(())

    def close(self):
        return None


class MeasuredEngine(DeclaredOnlyEngine):
    engine_id = "measured-engine"

    def __init__(self) -> None:
        super().__init__(CapabilityStatus.MEASURED)


def pipeline_for(engine: ScriptedSynthesisEngine, **kwargs) -> SpeechPipeline:
    registry = EngineRegistry()
    registry.register(engine)
    return SpeechPipeline(TextNormaliser(), TextChunker(kwargs.pop("max_tokens", 50)), registry, **kwargs)


class TestEngineRegistry:
    def test_registry_starts_empty(self):
        registry = EngineRegistry()
        assert len(registry) == 0
        assert registry.describe() == ()
        assert registry.engines() == ()

    def test_register_get_and_unregister(self):
        engine = ScriptedSynthesisEngine()
        registry = EngineRegistry()
        registry.register(engine)
        assert registry.get("self-test") is engine
        assert registry.unregister("self-test") is True
        assert len(registry) == 0

    def test_duplicate_id_is_refused(self):
        registry = EngineRegistry()
        registry.register(ScriptedSynthesisEngine())
        with pytest.raises(EngineUnavailable):
            registry.register(ScriptedSynthesisEngine())

    def test_best_for_strict_requires_measured_or_partial_not_a_vendor_claim(self):
        registry = EngineRegistry()
        registry.register(DeclaredOnlyEngine())
        with pytest.raises(EngineUnavailable) as refused:
            registry.best_for("bn", Task.TTS, strict=True)
        assert "model card" in str(refused.value)
        assert refused.value.details["unverified_engines"] == ("declared-only",)
        assert registry.best_for("bn", Task.TTS, strict=False).engine_id == "declared-only"

    def test_measured_wins_over_declared_when_strict_is_relaxed(self):
        registry = EngineRegistry()
        registry.register(DeclaredOnlyEngine())
        registry.register(MeasuredEngine())
        assert registry.best_for("bn", Task.TTS, strict=False).engine_id == "measured-engine"

    def test_no_model_declaration_becomes_model_not_installed(self):
        registry = EngineRegistry()
        registry.register(DeclaredOnlyEngine(CapabilityStatus.NO_MODEL))
        with pytest.raises(ModelNotInstalled) as refused:
            registry.best_for("bn")
        assert "no installed model" in str(refused.value)

    def test_empty_registry_names_the_engine_that_is_missing(self):
        with pytest.raises(EngineUnavailable) as refused:
            EngineRegistry().best_for(BENGALI)
        assert refused.value.details["language"] == "bn"
        assert "remedy" in refused.value.details

    def test_capability_of_unregistered_language_is_no_model_not_absent(self):
        capability = EngineRegistry().capability("bn")
        assert capability.status is CapabilityStatus.NO_MODEL
        assert capability.limitations

    def test_describe_lists_declared_languages(self):
        registry = EngineRegistry()
        registry.register(ScriptedSynthesisEngine())
        described = registry.describe()
        assert described[0].engine_id == "self-test"
        assert described[0].streaming is True
        assert ("bn", "unsupported") in described[0].languages


class TestCapabilityGate:
    def test_engine_is_never_called_for_an_unsupported_language(self):
        engine = ScriptedSynthesisEngine()
        pipeline = pipeline_for(engine)
        with pytest.raises(LanguageUnsupported):
            pipeline.synthesize(request_for(language=BENGALI))
        assert engine.calls == []

    def test_streaming_gate_refuses_before_any_audio_reaches_the_caller(self):
        engine = ScriptedSynthesisEngine()
        pipeline = pipeline_for(engine)
        events: list[SynthesisEvent] = []
        with pytest.raises(LanguageUnsupported):
            for event in pipeline.stream(request_for(language=LanguageTag("hi"))):
                events.append(event)
        assert events and events[-1].kind is SynthesisKind.FAILED
        assert all(event.audio is None for event in events)
        assert engine.calls == []

    def test_language_capability_is_reported_from_declarations(self):
        pipeline = pipeline_for(ScriptedSynthesisEngine())
        assert pipeline.language_capability(BENGALI).status is CapabilityStatus.UNSUPPORTED

    def test_pinned_engine_id_still_checks_the_language(self):
        engine = ScriptedSynthesisEngine()
        pipeline = pipeline_for(engine, engine_id="self-test")
        with pytest.raises(LanguageUnsupported):
            pipeline.synthesize(request_for(language=BENGALI))
        assert engine.calls == []

    def test_voice_without_consent_is_refused(self):
        engine = ScriptedSynthesisEngine()
        pipeline = pipeline_for(engine)
        unconsented = VoiceReference(voice_id="someone-elses-voice", audio=tone(300, 0.2))
        with pytest.raises(ProfileInvalid):
            pipeline.synthesize(request_for(voice=unconsented))
        assert engine.calls == []


class TestAssembly:
    def test_three_chunks_are_crossfaded_not_butted(self):
        buffers = [tone(200, 0.5), tone(300, 0.5), tone(400, 0.5)]
        engine = ScriptedSynthesisEngine(buffers=buffers, events_per_chunk=1)
        pipeline = pipeline_for(engine, max_tokens=4)
        outcome = pipeline.synthesize(request_for(voice=engine.voices()[0]))
        assert len(engine.calls) == 3
        expected_frames = sum(buffer.frame_count for buffer in buffers) - 2 * 480
        assert outcome.audio.frame_count == expected_frames
        assert outcome.audio.duration_s == pytest.approx(1.46, abs=1e-4)
        blend = outcome.audio.samples[11520:12000]
        assert blend != buffers[1].samples[0:480]
        assert any(blend)

    def test_stream_pieces_concatenate_to_the_same_audio_as_synthesize(self):
        buffers = [tone(200, 0.4), tone(320, 0.4)]
        engine = ScriptedSynthesisEngine(buffers=buffers, events_per_chunk=1)
        pipeline = pipeline_for(engine, max_tokens=6)
        voice = engine.voices()[0]
        outcome = pipeline.synthesize(request_for(voice=voice))
        streamed = [
            event.audio
            for event in pipeline.stream(request_for(voice=voice))
            if event.audio is not None
        ]
        joined = streamed[0]
        for piece in streamed[1:]:
            joined = joined.appended(piece)
        assert joined.to_bytes() == outcome.audio.to_bytes()

    def test_pause_after_s_is_rendered_as_silence_between_chunks(self):
        buffers = [tone(200, 0.3), tone(300, 0.3)]
        engine = ScriptedSynthesisEngine(buffers=buffers, events_per_chunk=1)
        pipeline = pipeline_for(engine, max_tokens=6)
        voice = engine.voices()[0]
        without = pipeline.synthesize(request_for(voice=voice))
        with_pause = pipeline.synthesize(
            request_for(voice=voice, prosody=Prosody(pause_after_s=0.25))
        )
        # The pause is itself a splice, so it costs one overlap of its own: the net
        # gain is the requested pause minus `crossfade_s`, not the full pause.
        assert with_pause.audio.duration_s - without.audio.duration_s == pytest.approx(
            0.25 - 0.02, abs=0.001
        )

    def test_prosody_is_clamped_before_the_engine_sees_it(self):
        engine = ScriptedSynthesisEngine()
        pipeline = pipeline_for(engine)
        wild = Prosody(rate=99.0, pitch_semitones=-400.0, volume_db=1000.0, pause_after_s=-5.0)
        pipeline.synthesize(request_for(prosody=wild, voice=engine.voices()[0]))
        seen = engine.calls[0].prosody
        assert seen == Prosody(rate=2.0, pitch_semitones=-12.0, volume_db=12.0, pause_after_s=0.0)

    def test_duration_ceiling_truncates_and_reports_dropped_text(self):
        buffers = [tone(200, 1.0), tone(300, 1.0), tone(400, 1.0)]
        engine = ScriptedSynthesisEngine(buffers=buffers, events_per_chunk=1)
        pipeline = pipeline_for(engine, max_tokens=6)
        outcome = pipeline.synthesize(
            request_for(voice=engine.voices()[0], maximum_duration_s=1.0)
        )
        assert outcome.truncated is True
        assert outcome.audio.duration_s <= 1.0 + 1e-6
        assert outcome.dropped_text

    def test_timings_include_first_audio_and_real_time_factor(self):
        engine = ScriptedSynthesisEngine(events_per_chunk=2)
        pipeline = pipeline_for(engine)
        outcome = pipeline.synthesize(request_for(voice=engine.voices()[0]))
        assert outcome.timings["first_result_ms"] is not None
        assert outcome.timings["audio_duration_s"] > 0
        assert outcome.timings["real_time_factor"] is not None
        assert outcome.timings["normalisation"]["language"] == PLUMBING_LANGUAGE


class TestStreamingEvents:
    def test_sequence_numbers_are_monotonic_and_one_terminal_kind_wins(self):
        engine = ScriptedSynthesisEngine(events_per_chunk=3)
        pipeline = pipeline_for(engine)
        events = list(pipeline.stream(request_for(voice=engine.voices()[0])))
        sequences = [event.sequence for event in events]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)
        assert events[0].kind is SynthesisKind.FIRST_AUDIO
        assert sum(1 for event in events if event.terminal) == 1
        assert events[-1].kind is SynthesisKind.EOS
        assert events[-1].audio is None

    def test_cancelling_mid_stream_emits_no_further_audio(self):
        engine = ScriptedSynthesisEngine(events_per_chunk=8)
        pipeline = pipeline_for(engine)
        token = CancellationToken()
        events: list[SynthesisEvent] = []
        for event in pipeline.stream(request_for(token=token, voice=engine.voices()[0])):
            events.append(event)
            if len(events) == 3:
                token.cancel("user interrupted")
        assert events[-1].kind is SynthesisKind.CANCELLED
        cancelled_at = max(index for index, event in enumerate(events) if event.terminal)
        assert all(event.audio is None for event in events[cancelled_at:])
        assert any(event.audio is not None for event in events[:cancelled_at])
        sequences = [event.sequence for event in events]
        assert sequences == sorted(sequences)

    def test_pre_cancelled_token_yields_no_audio_events(self):
        engine = ScriptedSynthesisEngine()
        pipeline = pipeline_for(engine)
        token = CancellationToken()
        token.cancel("already stopped")
        events = list(pipeline.stream(request_for(token=token, voice=engine.voices()[0])))
        assert all(event.audio is None for event in events)
        assert engine.calls == []

    def test_empty_text_is_refused(self):
        pipeline = pipeline_for(ScriptedSynthesisEngine())
        with pytest.raises(Exception):
            pipeline.synthesize(request_for(text="   ", voice=VoiceReference("v", consent_token="t", owner="o")))


class TestLedger:
    def test_pipeline_posts_structured_diagnostics(self):
        recorded: list[DiagnosticEvent] = []

        class Ledger:
            def record(self, event: DiagnosticEvent) -> None:
                recorded.append(event)

        engine = ScriptedSynthesisEngine(events_per_chunk=1)
        registry = EngineRegistry()
        registry.register(engine)
        pipeline = SpeechPipeline(
            TextNormaliser(), TextChunker(4), registry, ledger=Ledger()
        )
        pipeline.synthesize(request_for(voice=engine.voices()[0]))
        names = [event.event for event in recorded]
        assert "synthesis_started" in names and "synthesis_completed" in names
        first = recorded[0].fields
        assert first["chunks"] == 3
        assert first["engine"] == "self-test"
        assert first["budget_utilisation"] > 0
        assert recorded[0].component == "tts.pipeline"
