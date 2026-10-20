"""The pronunciation gap and the scripted transport: the refusal surface, asserted."""

from __future__ import annotations

import importlib
import sys

import pytest

from nativevoice.core.errors import CapabilityUnavailable, LanguageUnsupported
from nativevoice.core.language import CapabilityStatus, LanguageTag, Task
from nativevoice.tts.adapters.self_test import PLUMBING_LANGUAGE, ScriptedSynthesisEngine
from nativevoice.tts.api import SynthesisRequest, VoiceReference
from nativevoice.tts.engine_registry import EngineRegistry
from nativevoice.tts.phonemiser import (
    G2P_MISSING_ASSET,
    TRANSLITERATION_MISSING_ASSET,
    GraphemeToPhonemeRegistry,
    NoTransliteration,
)

BENGALI = LanguageTag("bn")
HINDI = LanguageTag("hi")


class TestScriptedEngine:
    def test_declares_streaming_and_no_real_language(self):
        engine = ScriptedSynthesisEngine()
        assert engine.supports_streaming is True
        statuses = {
            capability.language: capability.status for capability in engine.capabilities()
        }
        assert statuses["bn"] is CapabilityStatus.UNSUPPORTED
        assert statuses["hi"] is CapabilityStatus.UNSUPPORTED
        assert statuses["en"] is CapabilityStatus.UNSUPPORTED
        assert statuses[PLUMBING_LANGUAGE] is CapabilityStatus.PARTIAL

    def test_can_never_satisfy_a_production_decision(self):
        registry = EngineRegistry()
        registry.register(ScriptedSynthesisEngine())
        for code in ("bn", "hi", "en"):
            with pytest.raises(LanguageUnsupported):
                registry.best_for(code)
        assert registry.best_for(PLUMBING_LANGUAGE).engine_id == "self-test"

    def test_scripted_output_is_deterministic(self):
        first = ScriptedSynthesisEngine(chunk_duration_s=0.1)
        second = ScriptedSynthesisEngine(chunk_duration_s=0.1)
        for text in ("alpha", "beta", "gamma"):
            one = first.synthesize(SynthesisRequest(text, BENGALI)).audio
            two = second.synthesize(SynthesisRequest(text, BENGALI)).audio
            assert one.to_bytes() == two.to_bytes()

    def test_stream_yields_monotonic_events_ending_in_eos(self):
        engine = ScriptedSynthesisEngine(chunk_duration_s=0.2, events_per_chunk=3)
        events = list(engine.stream(SynthesisRequest("words", BENGALI, voice=engine.voices()[0])))
        assert [event.sequence for event in events] == sorted(event.sequence for event in events)
        assert events[-1].kind.value == "eos"
        assert sum(1 for event in events if event.audio is not None) == 3

    def test_condition_requires_reference_audio_and_consent(self):
        engine = ScriptedSynthesisEngine()
        with pytest.raises(Exception):
            engine.condition(VoiceReference(voice_id="x", consent_token="t", owner="o"))


class TestPronunciationGap:
    @pytest.mark.parametrize("code", ["bn", "hi", "en"])
    def test_every_language_reports_no_model(self, code):
        capability = GraphemeToPhonemeRegistry().capability(code)
        assert capability.status is CapabilityStatus.NO_MODEL
        assert capability.task is Task.TTS
        assert G2P_MISSING_ASSET in capability.limitations

    def test_bengali_limitation_names_the_absent_language(self):
        limitations = GraphemeToPhonemeRegistry().capability(BENGALI).limitations
        assert any("Bengali is absent" in note for note in limitations)

    def test_hindi_limitation_names_community_only_status(self):
        limitations = GraphemeToPhonemeRegistry().capability(HINDI).limitations
        assert any("community-only" in note for note in limitations)

    def test_registry_knows_both_target_languages_as_codes(self):
        assert {"bn", "hi"} <= set(GraphemeToPhonemeRegistry().languages())

    def test_asking_for_phonemes_refuses_instead_of_reading_letters(self):
        registry = GraphemeToPhonemeRegistry()
        with pytest.raises(CapabilityUnavailable) as refused:
            registry.phonemes("আমি কথা বলছি", BENGALI)
        assert refused.value.details["language"] == "bn"
        assert "graphemes" in str(refused.value)

    def test_unregistered_language_is_still_no_model_not_supported(self):
        registry = GraphemeToPhonemeRegistry()
        registry.declare("ja")
        assert registry.capability("ja").status is CapabilityStatus.NO_MODEL
        assert "no ja grapheme-to-phoneme asset" in registry.capability("ja").limitations[-1]

    def test_transliteration_refuses_and_says_why(self):
        with pytest.raises(CapabilityUnavailable) as refused:
            NoTransliteration().transliterate("hello", LanguageTag("en"), BENGALI)
        assert TRANSLITERATION_MISSING_ASSET in str(refused.value)

    def test_transliteration_capability_is_no_model(self):
        capability = NoTransliteration().capability(BENGALI, LanguageTag("en"))
        assert capability.status is CapabilityStatus.NO_MODEL

    def test_registry_transliteration_is_the_refusal(self):
        assert isinstance(GraphemeToPhonemeRegistry().transliteration, NoTransliteration)

    def test_provider_registration_requires_a_real_declaration(self):
        registry = GraphemeToPhonemeRegistry()
        with pytest.raises(CapabilityUnavailable):
            registry.register("en", GraphemeToPhonemeRegistry())

    def test_describe_covers_declared_languages(self):
        described = GraphemeToPhonemeRegistry().describe()
        assert {capability.language for capability in described} == {"bn", "en", "hi"}


class TestNoUpstreamSynthesisRuntime:
    """The invariant the retired reference adapter used to assert on its own."""

    def test_no_upstream_tts_package_is_loaded(self):
        assert "pocket_tts" not in sys.modules

    def test_the_retired_reference_adapter_is_not_reachable(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("nativevoice.tts.adapters.pocket_reference")

    def test_the_adapters_package_exports_only_live_engines(self):
        adapters = importlib.import_module("nativevoice.tts.adapters")
        assert set(adapters.__all__) == {
            "PLUMBING_LANGUAGE",
            "SHERPA_KOKORO_ENGINE_ID",
            "ScriptedSynthesisEngine",
            "SherpaKokoroEngine",
        }
