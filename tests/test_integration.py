"""End-to-end workflow and independence proofs.

Two things are asserted here that no other test in the suite covers: the whole
path from microphone-shaped bytes to speaker-shaped bytes with a host in the
middle, and the negative claim that this engine owns no memory of its own and
imports nothing belonging to a host application.
"""

from __future__ import annotations

import array
import re
from pathlib import Path

import pytest
from fixtures import silence, speech_like, sine

from nativevoice.audio.io import CapturingOutput
from nativevoice.audio.vad import EnergySpeechDetector, SegmenterConfig, UtteranceSegmenter
from nativevoice.core.audio import AudioBuffer, AudioFormat
from nativevoice.core.cancel import CancellationToken
from nativevoice.core.errors import EngineUnavailable, LanguageUnsupported
from nativevoice.core.events import SynthesisEvent, SynthesisKind, TranscriptionEvent, TranscriptionKind
from nativevoice.core.language import Capability, CapabilityStatus, LanguageTag, Task
from nativevoice.integration import HostReply, VoiceInterface, VoiceHost
from nativevoice.session.state import ListeningMode, SessionConfig
from nativevoice.stt.api import TranscriptionResult
from nativevoice.tts.api import SynthesisOutcome

ROOT = Path(__file__).resolve().parents[1] / "src" / "nativevoice"


class FakeStt:
    def __init__(self, text="switch to hindi please", language=LanguageTag("hi")):
        self.text, self.language = text, language
        self.invocations = []

    def transcribe(self, request):
        self.invocations.append(request)
        return TranscriptionResult(text=self.text, language=self.language, confidence=0.87, engine="fake")

    def stream(self, chunks, language=None, token=None):
        list(chunks)
        yield TranscriptionEvent.partial(self.text[:6], "u1")
        yield TranscriptionEvent.final(self.text, "u1", language=self.language.bcp47)

    def language_capability(self, language):
        return Capability(Task.STT, CapabilityStatus.MEASURED, language.code, metric_name="wer", metric_value=0.1)


class FakeTts:
    def __init__(self, chunks=4, error=None):
        self.chunks, self.error = chunks, error
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        for index in range(self.chunks):
            kind = SynthesisKind.AUDIO if index < self.chunks - 1 else SynthesisKind.EOS
            yield SynthesisEvent(
                kind,
                index,
                audio=AudioBuffer(array.array("h", bytes(480 * 2)), AudioFormat(24000, 1), 0, "fake"),
            )

    def synthesize(self, request):
        self.requests.append(request)
        return SynthesisOutcome(audio=sine(rate=24000, seconds=0.1), language=request.language, voice_id="v")


class MockHost:
    """The only thing a host has to provide. Nothing else about the host application exists here."""

    def __init__(self, prefix="HOST: "):
        self.seen = []
        self.prefix = prefix

    def respond(self, utterance, language=None):
        self.seen.append((utterance, None if language is None else language.code))
        return f"{self.prefix}{utterance}"


def interface(**kw) -> VoiceInterface:
    kw.setdefault("transcriber", FakeStt())
    kw.setdefault("synthesizer", FakeTts())
    kw.setdefault("output", CapturingOutput())
    return VoiceInterface(**kw)


class TestEndToEndExchange:
    def test_one_turn_traverses_capture_to_playback(self):
        out = CapturingOutput()
        host = MockHost()
        voice = interface(output=out)
        reply = voice.exchange(sine(seconds=0.3), host)
        assert isinstance(reply, HostReply)
        assert reply.utterance == "switch to hindi please"
        assert reply.reply == "HOST: switch to hindi please"
        assert reply.spoken is True and reply.silenced is False
        assert out.frames_written == 4 * 480
        assert host.seen == [("switch to hindi please", "hi")]

    def test_the_host_decides_the_language_of_the_reply_not_the_engine(self):
        voice = interface(transcriber=FakeStt(language=LanguageTag("bn")))
        reply = voice.exchange(sine(seconds=0.2), MockHost())
        assert reply.language == "bn"
        assert voice.synthesizer.requests[0].language == LanguageTag("bn")

    def test_voice_output_disabled_stays_disabled_across_the_whole_path(self):
        out = CapturingOutput()
        voice = interface(output=out, config=SessionConfig(voice_output_enabled=False))
        reply = voice.exchange(sine(seconds=0.2), MockHost())
        assert reply.reply.startswith("HOST:")
        assert reply.spoken is False and reply.silenced is True
        assert out.frames_written == 0

    def test_a_silent_answer_produces_no_audio(self):
        out = CapturingOutput()
        voice = interface(output=out)

        class QuietHost:
            def respond(self, utterance, language=None):
                return ""

        reply = voice.exchange(sine(seconds=0.2), QuietHost())
        assert reply.reply == "" and out.frames_written == 0

    def test_streaming_hearing_yields_partials_then_a_final(self):
        voice = interface()
        events = list(voice.stream_hearing(iter([sine(seconds=0.1)]), None))
        assert [event.kind for event in events] == [TranscriptionKind.PARTIAL, TranscriptionKind.FINAL]

    def test_hear_transcribes_one_finished_utterance(self):
        stt = FakeStt()
        voice = interface(transcriber=stt)
        result = voice.hear(sine(seconds=0.2))
        assert result.text == "switch to hindi please"
        assert len(stt.invocations) == 1

    def test_hear_applies_the_language_it_is_asked_to_force(self):
        stt = FakeStt()
        voice = interface(transcriber=stt)
        voice.hear(sine(seconds=0.2), language=LanguageTag("en"))
        request = stt.invocations[0]
        assert request.language == LanguageTag("en")
        assert request.allow_auto_detection is False

    def test_converse_segments_a_continuous_capture(self):
        detector = UtteranceSegmenter(
            EnergySpeechDetector(format=AudioFormat(16000, 1)),
            SegmenterConfig(minimum_silence_s=0.05, padding_s=0.02),
        )
        host = MockHost()
        voice = interface(segmenter_factory=lambda: detector,
                          config=SessionConfig(mode=ListeningMode.CONTINUOUS, continuous_consent=True))
        signal = speech_like(rate=16000, bursts=((0.1, 0.3), (0.6, 0.8)))
        replies = list(voice.converse(signal.frames(320), host))
        assert len(replies) == 2
        assert all(reply.spoken for reply in replies)
        assert len(host.seen) == 2

    def test_converse_reports_a_refusal_instead_of_aborting_the_conversation(self):
        detector = UtteranceSegmenter(
            EnergySpeechDetector(format=AudioFormat(16000, 1)), SegmenterConfig(minimum_silence_s=0.05)
        )
        voice = interface(
            synthesizer=FakeTts(error=LanguageUnsupported("no Bengali synthesis model", language="bn")),
            segmenter_factory=lambda: detector,
            transcriber=FakeStt(language=LanguageTag("bn")),
            config=SessionConfig(mode=ListeningMode.CONTINUOUS, continuous_consent=True),
        )
        signal = speech_like(rate=16000, bursts=((0.1, 0.3), (0.6, 0.8)))
        replies = list(voice.converse(signal.frames(320), MockHost()))
        assert len(replies) == 2, "a refusal must end one turn, not the conversation"
        assert all(reply.silenced for reply in replies)
        assert any("language unsupported for synthesis" in note for note in replies[0].notes)


class TestCancellationAndReplacement:
    def test_barge_in_stops_playback_and_no_queued_audio_escapes(self):
        out = CapturingOutput()
        voice = interface(output=out)
        voice.session.begin_turn()
        events = []
        for event in voice.speak("a long answer that will be interrupted"):
            events.append(event)
            if len(events) == 1:
                voice.cancel("user spoke over it")
        assert out.frames_written <= 480
        assert voice.session.turn.token.cancelled

    def test_cancel_without_a_turn_is_a_no_op_on_the_device(self):
        out = CapturingOutput()
        voice = interface(output=out)
        assert voice.cancel() is False
        assert out.halts == 1  # no turn to cancel, but the device is still silenced

    def test_speak_without_a_synthesizer_refuses_rather_than_lies(self):
        voice = interface(synthesizer=None)
        with pytest.raises(EngineUnavailable):
            list(voice.speak("words"))

    def test_any_engine_satisfying_the_contract_can_be_swapped_in(self):
        """Replaceability is the architectural claim; prove it by substitution."""

        class OtherStt:
            def transcribe(self, request):
                return TranscriptionResult(text="replaced", language=LanguageTag("en"), engine="other")

            def stream(self, chunks, language=None, token=None):
                return iter([])

            def language_capability(self, language):
                raise AssertionError("unused")

        voice = interface(transcriber=OtherStt())
        reply = voice.exchange(sine(seconds=0.2), MockHost())
        assert reply.utterance == "replaced" and reply.spoken is True


class TestIndependenceAndMemory:
    def test_no_host_application_module_is_imported_anywhere(self):
        offenders = []
        pattern = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z_][\w.]*)", re.M)
        for path in ROOT.rglob("*.py"):
            for name in pattern.findall(path.read_text(encoding="utf-8")):
                head = name.split(".")[0]
                if head in {"nativevoice", "ultron", "fieldmind", "orukeet", "openwhispr", "pocket_tts"} and head != "nativevoice":
                    offenders.append(f"{path.name}:{head}")
        assert offenders == []

    def test_the_engine_persists_nothing_during_a_conversation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        voice = interface()
        host = MockHost()
        for _ in range(3):
            voice.exchange(sine(seconds=0.2), host)
        assert sorted(path.name for path in tmp_path.iterdir()) == []
        assert voice.diagnose()["history_persisted"] is False

    def test_the_conversation_window_is_bounded_and_in_memory(self):
        voice = interface(max_history=2)
        for _ in range(5):
            voice.exchange(sine(seconds=0.2), MockHost())
        assert len(voice.history) == 2
        assert all(isinstance(reply, HostReply) for reply in voice.history)

    def test_no_network_module_is_imported_by_the_engine_packages(self):
        """`diagnostics/offline.py` imports socket on purpose - it is the guard that
        refuses sockets - and `cli`/`diagnostics` are not on the speech path, so the
        ban applies to the packages that actually handle an utterance."""
        banned = {"socket", "requests", "httpx", "urllib3", "ftplib", "smtplib", "urllib"}
        engine_packages = ("core", "audio", "models", "hardware", "config", "language", "stt", "tts", "profiles", "session", "eval")
        offenders = []
        paths = [found for name in engine_packages for found in (ROOT / name).rglob("*.py")]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for name in re.findall(r"^\s*(?:from|import)\s+([\w.]+)", text, re.M):
                if name.split(".")[0] in banned:
                    offenders.append(f"{path.name}:{name}")
        assert offenders == []

    def test_capabilities_are_reported_from_the_source_not_invented(self):
        voice = interface()
        assert voice.capabilities(Task.STT) == {}

        def source(task):
            return {"en": Capability(task, CapabilityStatus.MEASURED, "en", metric_name="wer", metric_value=0.04)}

        voice = interface(capability_source=source)
        recorded = voice.capabilities(Task.STT)
        assert recorded["en"].status is CapabilityStatus.MEASURED

    def test_diagnose_describes_the_running_system(self):
        voice = interface()
        voice.exchange(sine(seconds=0.2), MockHost())
        payload = voice.diagnose()
        assert payload["mode"] == "push_to_tap"
        assert payload["voice_output_enabled"] is True
        assert payload["turns_in_window"] == 1
        assert payload["transcriber"] == "FakeStt"
        assert set(payload["capabilities"]) == {"stt", "tts", "lid", "voice_clone"}

    def test_close_releases_the_turn_so_its_span_cannot_finish(self):
        out = CapturingOutput()
        voice = interface(output=out)
        voice.exchange(sine(seconds=0.2), MockHost())
        turn = voice.session.turn
        voice.close()
        assert turn.token.cancelled
        assert out.play(sine(rate=24000, seconds=0.4), turn.token) == 0


def test_a_voice_host_needs_only_one_method():
    class MinimalHost:
        def respond(self, utterance, language=None):
            return "ok"

    voice = interface()
    reply = voice.exchange(sine(seconds=0.2), MinimalHost())
    assert reply.reply == "ok" and reply.spoken is True
