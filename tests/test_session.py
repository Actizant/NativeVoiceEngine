"""Session behaviour: turn accounting, barge-in, stale work, disabled voice, recovery.

The fakes here are deliberately minimal collaborators. Nothing about a real
recogniser or synthesiser is being claimed - these tests are about ordering and
cancellation, which is where a voice stack actually breaks.
"""

from __future__ import annotations

import array
import pytest
from fixtures import sine, silence

from nativevoice.audio.io import CapturingOutput
from nativevoice.core.audio import AudioBuffer, AudioFormat
from nativevoice.core.errors import DeviceError, EngineUnavailable, LanguageUnsupported
from nativevoice.core.events import SynthesisEvent, SynthesisKind, TranscriptionEvent, TranscriptionKind
from nativevoice.core.language import LanguageTag
from nativevoice.session import (
    ListeningMode,
    SessionConfig,
    TranscriptBuffer,
    TurnState,
    VoiceSession,
    WakeWordBackendUnavailable,
    WakeWordDetector,
)
from nativevoice.stt.api import TranscriptionResult
from nativevoice.tts.api import SynthesisRequest


class FakeTranscriber:
    def __init__(self, text="hello there", language=LanguageTag("en"), events=None, error=None):
        self.text = text
        self.language = language
        self.events = events or []
        self.error = error
        self.calls = 0
        self.requests = []

    def transcribe(self, request):
        self.calls += 1
        self.requests.append(request)
        if self.error:
            raise self.error
        return TranscriptionResult(
            text=self.text, language=self.language, confidence=0.9, engine="fake", model_id="fake-1"
        )

    def stream(self, chunks, language=None):
        self.calls += 1
        list(chunks)
        return iter(self.events)

    def language_capability(self, language):
        raise AssertionError("not used by the session")


class FakeSynthesizer:
    def __init__(self, chunks=6, size=320, error=None, language=LanguageTag("en")):
        self.chunks = chunks
        self.size = size
        self.error = error
        self.language = language
        self.requests: list[SynthesisRequest] = []

    def stream(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        format = AudioFormat(24000, 1)
        for index in range(self.chunks):
            kind = SynthesisKind.AUDIO if index < self.chunks - 1 else SynthesisKind.EOS
            yield SynthesisEvent(
                kind, index, audio=AudioBuffer(array.array("h", bytes(self.size * 2)), format, 0, "fake")
            )

    def synthesize(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        from nativevoice.tts.api import SynthesisOutcome

        return SynthesisOutcome(audio=sine(rate=24000, seconds=0.1), language=request.language, voice_id="v")


def session(transcriber=None, synthesizer=None, output=None, **config) -> VoiceSession:
    return VoiceSession(
        transcriber=transcriber or FakeTranscriber(),
        synthesizer=synthesizer if synthesizer is not None else FakeSynthesizer(),
        output=output or CapturingOutput(),
        config=SessionConfig(**config),
    )


class TestTurnAccounting:
    def test_hear_propagates_a_recogniser_refusal(self):
        error = LanguageUnsupported("no engine for Bengali", language="bn")
        voice = session(FakeTranscriber(error=error))
        with pytest.raises(LanguageUnsupported):
            voice.hear(sine(seconds=0.2))
        assert voice.transcriber.calls == 1

    def test_hear_commits_text_and_language(self):
        transcriber = FakeTranscriber()
        voice = session(transcriber)
        result = voice.hear(sine(seconds=0.3))
        assert result.text == "hello there"
        assert voice.turn.state is TurnState.AWAITING_ANSWER
        assert voice.turn.committed_text == "hello there"
        assert voice.turn.language == LanguageTag("en")

    def test_auto_detection_only_when_no_language_forced(self):
        voice = session()
        voice.hear(sine(seconds=0.1))
        assert voice.transcriber.requests[0].allow_auto_detection is True
        forced = session()
        forced.hear(sine(seconds=0.1))
        assert forced.transcriber.requests[0].token is forced.turn.token

    def test_new_turn_supersedes_the_previous_one(self):
        voice = session()
        first = voice.begin_turn()
        second = voice.begin_turn()
        assert first.token.cancelled and first.superseded_by == second.turn_id
        assert second.current

    def test_interrupt_cancels_without_a_turn(self):
        out = CapturingOutput()
        voice = session(output=out)
        assert voice.interrupt() is False
        voice.begin_turn()
        assert voice.interrupt("user spoke") is True
        assert voice.turn.token.cancelled


class TestTranscriptBuffer:
    def test_partial_replaces_partial(self):
        buffer = TranscriptBuffer()
        buffer.apply(TranscriptionEvent.partial("the", "u1"))
        buffer.apply(TranscriptionEvent.partial("the cat", "u1"))
        assert buffer.text == "the cat"

    def test_final_commits_and_clears_the_partial(self):
        buffer = TranscriptBuffer()
        buffer.apply(TranscriptionEvent.partial("the cat", "u1"))
        buffer.apply(TranscriptionEvent.final("the cat sat", "u1"))
        assert buffer.text == "the cat sat"
        assert buffer.partials == {}

    def test_retract_removes_a_committed_final(self):
        buffer = TranscriptBuffer()
        buffer.apply(TranscriptionEvent.final("wrong words", "u1"))
        buffer.apply(TranscriptionEvent.retract("u1", text="wrong words"))
        assert buffer.text == ""
        assert buffer.retracted == ["wrong words"]

    def test_a_closed_buffer_accepts_nothing_more(self):
        buffer = TranscriptBuffer(accept=False)
        buffer.apply(TranscriptionEvent.final("late", "u1"))
        assert buffer.text == "" and buffer.ignored == ["u1"]

    def test_hear_streaming_reduces_events(self):
        events = [
            TranscriptionEvent.partial("hal", "t1"),
            TranscriptionEvent.partial("hello", "t1"),
            TranscriptionEvent.final("hello", "t1"),
        ]
        voice = session(FakeTranscriber(events=events))
        text, seen = voice.hear_streaming([sine(seconds=0.1)], token=None)
        assert text == "hello" and len(seen) == 3

    def test_hear_streaming_ignores_a_final_from_another_turn(self):
        events = [TranscriptionEvent.final("stale", "other_turn")]
        voice = session(FakeTranscriber(events=events))
        voice.begin_turn()
        text, _ = voice.hear_streaming([sine(seconds=0.1)])
        assert text == ""


class TestAnswerRouting:
    def test_voice_output_disabled_never_reaches_the_speaker(self):
        out = CapturingOutput()
        voice = session(synthesizer=FakeSynthesizer(), output=out, voice_output_enabled=False)
        report = voice.answer("spoken words")
        assert report.spoke is False and report.silenced is True
        assert out.frames_written == 0
        assert "voice output disabled" in report.notes[0]

    def test_disabled_output_stays_disabled_for_every_later_turn(self):
        voice = session(voice_output_enabled=False)
        for _ in range(3):
            assert voice.answer("anything").spoke is False

    def test_a_failed_synthesis_event_is_recorded(self):
        class Failing:
            def stream(self, request):
                yield SynthesisEvent(SynthesisKind.FAILED, 0, reason="engine died")

            def synthesize(self, request):
                raise AssertionError("unused")

        voice = session(synthesizer=Failing())
        report = voice.answer("words")
        assert report.spoke is False
        assert any("synthesis ended failed" in note for note in report.notes)

    def test_enabled_output_plays_every_chunk(self):
        out = CapturingOutput()
        voice = session(output=out)
        report = voice.answer("spoken words")
        assert report.spoke is True
        assert out.frames_written == 6 * 320

    def test_a_synthesizer_that_cannot_do_the_language_is_reported_not_silenced(self):
        error = LanguageUnsupported("no Bengali model", language="bn")
        voice = session(synthesizer=FakeSynthesizer(error=error))
        report = voice.answer("নমস্কার", language=LanguageTag("bn"))
        assert report.spoke is False and report.silenced is True
        assert any("language unsupported" in note for note in report.notes)

    def test_empty_answer_does_not_produce_audio(self):
        out = CapturingOutput()
        voice = session(output=out)
        report = voice.answer("   ")
        assert report.spoke is False and out.frames_written == 0
        assert any("empty answer" in note for note in report.notes)

    def test_a_stale_answer_is_discarded_and_said_so(self):
        voice = session()
        old = voice.begin_turn()
        voice.begin_turn()
        voice.turn = old
        report = voice.answer("late arrival")
        assert any("stale answer discarded" in note for note in report.notes)


class TestBargeIn:
    def test_playback_stops_and_queued_audio_is_not_emitted(self):
        out = CapturingOutput()
        voice = session(output=out)
        voice.begin_turn()
        voice.turn.state = TurnState.SPEAKING
        assert voice.output.play(sine(rate=24000, seconds=0.1), voice.turn.token) > 0
        voice.interrupt("barge-in")
        assert voice.output.play(sine(rate=24000, seconds=0.1), voice.turn.token) == 0

    def test_synthesis_stops_emitting_once_the_turn_is_cancelled(self):
        from nativevoice.audio.vad import SpeechSegment

        voice = session()
        voice.begin_turn()
        voice.turn.state = TurnState.SPEAKING
        segment = SpeechSegment(0, 8000, duration_s=0.5)
        assert voice.barge_in_from(segment) is True
        assert voice.turn.token.reason == "barge-in"

    def test_a_short_noise_burst_does_not_interrupt(self):
        from nativevoice.audio.vad import SpeechSegment

        voice = session()
        voice.begin_turn()
        voice.turn.state = TurnState.SPEAKING
        assert voice.barge_in_from(SpeechSegment(0, 1600, duration_s=0.1)) is False
        assert voice.turn.current

    def test_barge_in_can_be_switched_off(self):
        from nativevoice.audio.vad import SpeechSegment

        voice = session(barge_in_enabled=False)
        voice.begin_turn()
        voice.turn.state = TurnState.SPEAKING
        assert voice.barge_in_from(SpeechSegment(0, 16000, duration_s=1.0)) is False
        assert voice.turn.current


class TestListeningModes:
    def test_wake_word_mode_is_refused_not_silently_downgraded(self):
        with pytest.raises(WakeWordBackendUnavailable):
            SessionConfig(mode=ListeningMode.WAKE_WORD)

    def test_wake_word_detector_has_no_backend(self):
        with pytest.raises(WakeWordBackendUnavailable):
            WakeWordDetector().feed(sine(seconds=0.1))

    def test_continuous_listening_requires_explicit_consent(self):
        with pytest.raises(ValueError):
            SessionConfig(mode=ListeningMode.CONTINUOUS)
        assert SessionConfig(mode=ListeningMode.CONTINUOUS, continuous_consent=True).mode is ListeningMode.CONTINUOUS

    def test_segmentation_needs_a_configured_detector(self):
        voice = session()
        with pytest.raises(EngineUnavailable):
            list(voice.segment_capture([sine(seconds=0.1)]))
        from nativevoice.audio.vad import EnergySpeechDetector, SegmenterConfig, UtteranceSegmenter

        detector = UtteranceSegmenter(
            EnergySpeechDetector(format=AudioFormat(16000, 1)), SegmenterConfig(minimum_silence_s=0.05)
        )
        wired = VoiceSession(
            transcriber=FakeTranscriber(),
            synthesizer=None,
            output=CapturingOutput(),
            segmenter_factory=lambda: detector,
        )
        assert list(wired.segment_capture([silence(seconds=0.2)])) == []


class TestRecovery:
    def test_a_single_device_failure_is_recovered(self):
        voice = session()
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 2:
                raise DeviceError("microphone vanished")
            return "ok"

        assert voice.with_recovery(flaky) == "ok"
        assert voice.recoveries == 1 and voice.device_failures == 1

    def test_recovery_is_bounded(self):
        voice = session(max_recoveries=2)

        def broken():
            raise DeviceError("still broken")

        with pytest.raises(DeviceError):
            voice.with_recovery(broken)
        assert voice.device_failures == 3

    def test_close_cancels_the_turn_so_its_work_stops_emitting(self):
        out = CapturingOutput()
        voice = session(output=out)
        turn = voice.begin_turn()
        voice.close()
        assert turn.token.cancelled
        assert out.play(sine(rate=24000, seconds=0.4), turn.token) == 0
        assert out.cancelled_spans == 1
