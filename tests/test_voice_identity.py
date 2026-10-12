"""Voice identity packages, host selection, and multi-host isolation.

The claim under test is the product's central structural promise: one engine,
many hosts, each with its own voice selection and its own data. A voice identity
that carries invented metadata, or a host that can read another host's
recordings, fails that promise regardless of how good the synthesis is.
"""

from __future__ import annotations

import array
import json

import pytest
from fixtures import sine

from nativevoice.audio.io import CapturingOutput
from nativevoice.core.audio import AudioBuffer, AudioFormat
from nativevoice.core.errors import (
    ConfigurationError,
    LanguageUnsupported,
    ProfileInvalid,
    StorageError,
)
from nativevoice.core.language import CapabilityStatus, LanguageTag
from nativevoice.integration import VoiceInterface
from nativevoice.session.state import SessionConfig
from nativevoice.tts.adapters.self_test import PLUMBING_LANGUAGE, ScriptedSynthesisEngine
from nativevoice.tts.api import SynthesisOutcome
from nativevoice.tts.chunking import TextChunker
from nativevoice.tts.engine_registry import EngineRegistry
from nativevoice.tts.normalise import TextNormaliser
from nativevoice.tts.pipeline import SpeechPipeline
from nativevoice.voices import (
    PLATFORM_DEFAULT,
    Compatibility,
    HostProfile,
    InstallRecord,
    LanguageSupport,
    Measurements,
    VoiceIdentity,
    VoiceProvenance,
    VoiceRegistry,
    identity_from_json,
    identity_to_json,
)

RATE = 24000


def identity(voice_id=PLATFORM_DEFAULT, languages=(("en", CapabilityStatus.MEASURED),), state="installed", **extra):
    forced_install = extra.pop("install", None)
    return VoiceIdentity(
        voice_id=voice_id,
        display_name=extra.pop("display_name", voice_id),
        named=extra.pop("named", False),
        languages=tuple(
            item if isinstance(item, LanguageSupport) else LanguageSupport(language=item[0], status=item[1])
            for item in languages
        ),
        compatibility=Compatibility(engine_id="sherpa-onnx-kokoro", model_id="kokoro-int8", speaker_slot=0),
        install=forced_install or InstallRecord(state=state),
        **extra,
    )


class TestIdentityPackage:
    def test_round_trip_through_json_preserves_every_field(self):
        original = identity(
            languages=(
                LanguageSupport(
                    language="en",
                    status=CapabilityStatus.MEASURED,
                    accent="us",
                    style="neutral",
                    metric_name="roundtrip_wer",
                    metric_value=0.049,
                    measured_on="2026-10-12",
                    evidence="artifacts/evaluation/roundtrip-2026-10-12.json",
                    limitations=("measured on machine-generated speech",),
                ),
                LanguageSupport(language="zh", status=CapabilityStatus.MEASURED_FAILED, metric_name="roundtrip_cer", metric_value=0.463),
            ),
            provenance=VoiceProvenance(creator="hexgrad", licence_spdx="Apache-2.0", redistributable=False),
            measurements=Measurements(host="rtx3050", real_time_factor=1.74, first_audio_ms=4360.0, peak_rss_mib=708.8),
            install=InstallRecord(state="installed", version="3", digests={"model.int8.onnx": "ab" * 32}),
            accents=("us",),
            styles=("neutral", "calm"),
            tags=("platform",),
        )
        restored = identity_from_json(identity_to_json(original))
        assert restored == original

    def test_a_newer_schema_is_refused_rather_than_half_read(self):
        payload = identity().as_dict()
        payload["schema"] = 99
        with pytest.raises(ConfigurationError, match="newer schema"):
            identity_from_json(payload)

    def test_an_unnamed_voice_may_not_carry_an_invented_display_name(self):
        with pytest.raises(ConfigurationError, match="invented display name"):
            identity(display_name="Serene Narrator")

    def test_a_named_voice_may_carry_one(self):
        assert identity(display_name="Serene Narrator", named=True).display_name == "Serene Narrator"

    def test_voice_id_is_path_safe(self):
        for bad in ("", "../evil", "a/b", "a\\b"):
            with pytest.raises(ConfigurationError):
                VoiceIdentity(voice_id=bad, display_name=bad or "x", compatibility=Compatibility("e", "m"))


class TestCapabilitySemantics:
    def test_measured_speaks_and_measured_failed_does_not(self):
        works = identity(languages=(("en", CapabilityStatus.MEASURED), ("zh", CapabilityStatus.MEASURED_FAILED)))
        assert works.speaks("en") is True
        assert works.speaks("zh") is False
        with pytest.raises(LanguageUnsupported) as caught:
            works.require_speaks("zh")
        assert caught.value.details["status"] == "measured_failed"

    def test_declared_is_not_enough_to_speak(self):
        voice = identity(languages=(("de", CapabilityStatus.DECLARED),))
        assert voice.speaks("de") is False

    def test_an_unrecorded_language_names_what_is_recorded(self):
        voice = identity(languages=(("en", CapabilityStatus.MEASURED),))
        with pytest.raises(LanguageUnsupported) as caught:
            voice.require_speaks("bn")
        assert caught.value.details["known"] == ("en",)

    def test_a_staged_identity_cannot_speak_at_all(self):
        voice = identity(state="staged")
        assert voice.speaks("en") is False
        with pytest.raises(LanguageUnsupported, match="not installed"):
            voice.require_speaks("en")

    def test_reference_carries_the_slot_not_the_host(self):
        reference = identity(languages=(("en", CapabilityStatus.MEASURED),)).as_reference()
        assert reference.voice_id == PLATFORM_DEFAULT
        assert "kokoro-int8 speaker 0" in reference.source
        assert reference.audio is None


class TestRegistryAndSelection:
    def test_discovery_save_and_remove(self, tmp_path):
        registry = VoiceRegistry(tmp_path)
        assert registry.all() == ()
        registry.save(identity())
        registry.save(
            identity(
                voice_id="voice.host.fieldmind",
                languages=(("en", CapabilityStatus.MEASURED), ("de", CapabilityStatus.MEASURED)),
            )
        )
        assert {item.voice_id for item in registry.all()} == {PLATFORM_DEFAULT, "voice.host.fieldmind"}
        assert registry.remove("voice.host.fieldmind") is True
        assert registry.get("voice.host.fieldmind") is None

    def test_an_unreadable_identity_is_reported_not_silently_dropped(self, tmp_path):
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        registry = VoiceRegistry(tmp_path)
        registry.save(identity())
        assert [item.voice_id for item in registry.all()] == [PLATFORM_DEFAULT]
        assert "broken.json" in registry.unreadable

    def test_precedence_is_explicit_then_host_then_platform(self, tmp_path):
        registry = VoiceRegistry(tmp_path)
        registry.save(identity())
        registry.save(
            identity(
                voice_id="voice.host.fieldmind",
                languages=(("en", CapabilityStatus.MEASURED), ("de", CapabilityStatus.MEASURED)),
            )
        )
        host = HostProfile(host_id="fieldmind", default_voice="voice.host.fieldmind", root=tmp_path / "hosts/fieldmind")
        assert registry.select(language="en", host=host).voice_id == "voice.host.fieldmind"
        assert registry.select(requested=PLATFORM_DEFAULT, language="en", host=host).voice_id == PLATFORM_DEFAULT
        hostless = HostProfile(host_id="ultron", root=tmp_path / "hosts/ultron")
        assert registry.select(language="en", host=hostless).voice_id == PLATFORM_DEFAULT

    def test_german_resolves_only_to_the_voice_that_records_it(self, tmp_path):
        registry = VoiceRegistry(tmp_path)
        registry.save(identity())
        registry.save(
            identity(
                voice_id="voice.host.fieldmind",
                languages=(("en", CapabilityStatus.MEASURED), ("de", CapabilityStatus.MEASURED)),
            )
        )
        host = HostProfile(host_id="fieldmind", default_voice="voice.host.fieldmind", root=tmp_path / "hosts/fieldmind")
        assert registry.select(language="de", host=host).voice_id == "voice.host.fieldmind"
        with pytest.raises(LanguageUnsupported) as caught:
            registry.select(requested=PLATFORM_DEFAULT, language="de", host=None)
        assert caught.value.details["other_installed"] == ("voice.host.fieldmind",)

    def test_no_installed_voice_at_all_explains_where_it_looked(self, tmp_path):
        registry = VoiceRegistry(tmp_path / "absent")
        with pytest.raises(StorageError, match="no voice identity is installed"):
            registry.select(language="en")

    def test_requesting_an_unknown_voice_is_not_silently_defaulted(self, tmp_path):
        registry = VoiceRegistry(tmp_path)
        registry.save(identity())
        with pytest.raises(StorageError, match="not installed"):
            registry.select(requested="voice.does.not.exist")


class TestHostIsolation:
    def test_host_roots_are_separate_and_reject_unsafe_ids(self, tmp_path):
        first = HostProfile(host_id="ultron", root=tmp_path / "ultron")
        second = HostProfile(host_id="fieldmind", root=tmp_path / "fieldmind")
        first.ensure()
        assert first.recordings != second.recordings
        assert first.recordings.is_dir()
        assert not second.recordings.exists(), "ensuring one host must not create another host's store"
        second.ensure()
        assert second.profiles.is_dir() and first.profiles.is_dir()
        assert first.profiles != second.profiles
        with pytest.raises(ConfigurationError):
            HostProfile(host_id="../elsewhere", root=tmp_path)

    def test_two_hosts_speak_the_same_answer_with_different_voices(self, tmp_path):
        registry = VoiceRegistry(tmp_path)
        registry.save(identity())
        registry.save(
            identity(
                voice_id="voice.host.fieldmind.speaker",
                languages=(("en", CapabilityStatus.MEASURED),),
            )
        )

        class Stt:
            def transcribe(self, request):
                from nativevoice.stt.api import TranscriptionResult

                return TranscriptionResult(text="hello", language=LanguageTag("en"), engine="fake")

            def stream(self, chunks, language=None, token=None):
                return iter([])

        class Tts:
            def __init__(self):
                self.seen = []

            def stream(self, request):
                self.seen.append(request.voice.voice_id if request.voice else "")
                from nativevoice.core.events import SynthesisEvent, SynthesisKind

                yield SynthesisEvent(
                    SynthesisKind.EOS,
                    0,
                    audio=AudioBuffer(array.array("h", bytes(480 * 2)), AudioFormat(RATE, 1)),
                )

            def synthesize(self, request):
                return SynthesisOutcome(
                    audio=AudioBuffer(array.array("h", bytes(480 * 2)), AudioFormat(RATE, 1)),
                    language=request.language,
                    voice_id="",
                )

        class Host:
            def respond(self, utterance, language=None):
                return "reply"

        synthesizer = Tts()
        ultron = VoiceInterface(
            transcriber=Stt(),
            synthesizer=synthesizer,
            output=CapturingOutput(),
            voices=registry,
            host=HostProfile(host_id="ultron", root=tmp_path / "hosts/ultron"),
            config=SessionConfig(),
        )
        fieldmind = VoiceInterface(
            transcriber=Stt(),
            synthesizer=synthesizer,
            output=CapturingOutput(),
            voices=registry,
            host=HostProfile(
                host_id="fieldmind",
                default_voice="voice.host.fieldmind.speaker",
                root=tmp_path / "hosts/fieldmind",
            ),
            config=SessionConfig(),
        )
        ultron.exchange(sine(seconds=0.2), Host())
        fieldmind.exchange(sine(seconds=0.2), Host())
        assert synthesizer.seen == [PLATFORM_DEFAULT, "voice.host.fieldmind.speaker"]
        assert ultron.history[0].reply == fieldmind.history[0].reply == "reply"
        assert len(ultron.history) == 1 and len(fieldmind.history) == 1

    def test_a_host_without_a_recorded_voice_for_the_language_is_refused_not_substituted(self, tmp_path):
        registry = VoiceRegistry(tmp_path)
        registry.save(identity())

        class Stt:
            def transcribe(self, request):
                from nativevoice.stt.api import TranscriptionResult

                return TranscriptionResult(text="hello", language=LanguageTag("bn"), engine="fake")

            def stream(self, chunks, language=None, token=None):
                return iter([])

        class Tts:
            def __init__(self):
                self.calls = 0

            def stream(self, request):
                self.calls += 1
                raise LanguageUnsupported("no Bengali front-end", language="bn")

            def synthesize(self, request):
                raise AssertionError("unused")

        out = CapturingOutput()
        voice = VoiceInterface(
            transcriber=Stt(),
            synthesizer=Tts(),
            output=out,
            voices=registry,
            host=HostProfile(host_id="ultron", root=tmp_path / "hosts/ultron"),
        )
        reply = voice.exchange(sine(seconds=0.2), _QuietHost())
        assert reply.spoken is False and out.frames_written == 0
        assert voice.voice_refusal.startswith("LanguageUnsupported")


class _QuietHost:
    def respond(self, utterance, language=None):
        return "answer"


class TestConsentBoundary:
    def _pipeline(self):

        return SpeechPipeline(
            normaliser=TextNormaliser(),
            chunker=TextChunker(),
            registry=EngineRegistry(),
        )

    def test_an_installed_identity_needs_no_consent_record(self):

        engine = _SlotEngine()
        registry = EngineRegistry()
        registry.register(engine)
        pipeline = SpeechPipeline(TextNormaliser(), TextChunker(), registry)
        outcome = pipeline.synthesize(_request(identity().as_reference()))
        assert outcome.audio.frame_count > 0
        assert outcome.voice_id == PLATFORM_DEFAULT

    def test_a_recorded_voice_without_consent_is_still_refused(self):


        engine = ScriptedSynthesisEngine()
        registry = EngineRegistry()
        registry.register(engine)
        pipeline = SpeechPipeline(TextNormaliser(), TextChunker(), registry)
        recorded = _recorded_reference()
        with pytest.raises(ProfileInvalid, match="needs consent"):
            pipeline.synthesize(_request(recorded, _tag(PLUMBING_LANGUAGE)))

    def test_a_recorded_voice_with_consent_passes_the_gate(self):

        engine = ScriptedSynthesisEngine()
        registry = EngineRegistry()
        registry.register(engine)
        pipeline = SpeechPipeline(TextNormaliser(), TextChunker(), registry, strict=False)
        consented = _recorded_reference(consent_token="granted-by-owner", owner="recorded-donor")
        outcome = pipeline.synthesize(_request(consented, _tag(PLUMBING_LANGUAGE)))
        assert outcome.audio.frame_count > 0


def _tag(value):
    return value if isinstance(value, LanguageTag) else LanguageTag(str(value))


def _request(voice, language=LanguageTag("en")):
    from nativevoice.tts.api import SynthesisRequest

    return SynthesisRequest(text="one sentence.", language=language, voice=voice)


def _recorded_reference(**kwargs):
    from nativevoice.tts.api import VoiceReference

    return VoiceReference(
        voice_id="voice.recorded.donor-01",
        audio=sine(seconds=0.2),
        **kwargs,
    )


class _SlotEngine:
    """Stands in for a real engine: proves the pipeline routes a slot reference through."""

    engine_id = "slot-engine"
    model_id = "slot-model"
    output_format = AudioFormat(RATE, 1)
    supports_streaming = True
    controllable = ("rate",)

    def capabilities(self):
        from nativevoice.core.language import Capability, Task

        return (Capability(Task.TTS, CapabilityStatus.MEASURED, "en", engine="slot-engine"),)

    def voices(self):
        return ()

    def condition(self, reference):
        raise AssertionError("unused")

    def warm(self):
        return None

    def synthesize(self, request):
        self.last = request.voice
        return SynthesisOutcome(
            audio=AudioBuffer(array.array("h", bytes(480 * 2)), AudioFormat(RATE, 1)),
            language=request.language,
            voice_id=request.voice.voice_id if request.voice else "",
        )

    def stream(self, request):
        from nativevoice.core.events import SynthesisEvent, SynthesisKind

        outcome = self.synthesize(request)
        yield SynthesisEvent(SynthesisKind.EOS, 0, audio=outcome.audio)

    def close(self):
        return None
