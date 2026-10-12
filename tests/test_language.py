"""Tests for language identity, capability honesty, scripts, LID and routing.

Covers the invariants the spec calls out: identity is separate from capability,
an unrecorded language reports NO_MODEL, script separation is exact, lexicon LID
is never presented as calibrated, romanised Hindi/Urdu is flagged low-confidence,
and routing refuses rather than mis-invoking an engine.
"""

from __future__ import annotations

import pytest

from nativevoice.core.language import (
    Capability,
    CapabilityStatus,
    LanguageTag,
    Task,
)
from nativevoice.language import (
    CapabilityLedger,
    LanguageRegistry,
    LanguageRouter,
    LexiconLanguageIdentifier,
    RoutingKind,
    ScriptOnlyIdentifier,
    STOPWORDS,
    TransliteratedTextRisk,
    analyze,
    detect_script,
    is_mixed,
    script_fractions,
)
from nativevoice.language.corpora import SAMPLE_SENTENCES


@pytest.fixture
def registry() -> LanguageRegistry:
    return LanguageRegistry()


@pytest.fixture
def ledger() -> CapabilityLedger:
    # The single most important honesty invariant: a fresh ledger has ZERO
    # measured entries, so nothing is supported until proven otherwise.
    return CapabilityLedger()


def _cap(status, language, engine="eng", metric=None):
    return Capability(
        task=Task.STT,
        status=status,
        language=language,
        engine=engine,
        metric_name="wer" if metric is not None else "",
        metric_value=metric,
        model_id=f"{engine}-{language}" if metric is not None else "",
    )


# --- registry / identity -----------------------------------------------------
def test_registry_has_61_unique_languages(registry):
    codes = registry.all_codes()
    assert len(registry) == 61
    assert len(codes) == len(set(codes))
    assert len(registry.profiles()) == 61


def test_bengali_and_hindi_present_with_native_names_and_scripts(registry):
    bn = registry.lookup("bn")
    hi = registry.lookup("hi")
    assert bn.native_name == "বাংলা"
    assert hi.native_name == "हिन्दी"
    assert bn.script == "Bengali"
    assert hi.script == "Devanagari"
    assert registry.known("bn") and registry.known("hi")


def test_lookup_accepts_language_tag_and_region(registry):
    assert registry.lookup(LanguageTag.parse("hi-IN")).code == "hi"
    assert registry.lookup("zh-Hans").script == "Han"


def test_registry_records_identity_not_capability(registry):
    # A profile never carries a supported boolean; capability lives only in the
    # ledger. There is no attribute on LanguageProfile to make a support claim.
    profile = registry.lookup("bn")
    assert not hasattr(profile, "supported")
    assert profile.english_name == "Bengali"


# --- ledger / capability honesty --------------------------------------------
def test_unrecorded_language_reports_no_model(ledger):
    capability = ledger.best(Task.STT, "bn")
    assert capability.status is CapabilityStatus.NO_MODEL


def test_all_registry_languages_default_to_no_model(ledger, registry):
    for code in registry.all_codes():
        assert ledger.best(Task.STT, code).status is CapabilityStatus.NO_MODEL


def test_best_prefers_measured_over_declared(ledger):
    ledger.record(_cap(CapabilityStatus.DECLARED, "en", engine="a"))
    ledger.record(_cap(CapabilityStatus.MEASURED, "en", engine="b", metric=0.03))
    assert ledger.best(Task.STT, "en").status is CapabilityStatus.MEASURED
    assert ledger.best(Task.STT, "en").engine == "b"


def test_usable_policy_excludes_declared_when_strict(ledger):
    ledger.record(_cap(CapabilityStatus.DECLARED, "hi", engine="a"))
    assert ledger.usable(Task.STT, "hi", allow_declared=False) is None
    assert ledger.usable(Task.STT, "hi", allow_declared=True).status is CapabilityStatus.DECLARED


def test_unsupported_never_usable(ledger):
    ledger.record(_cap(CapabilityStatus.UNSUPPORTED, "bn", engine="a"))
    assert ledger.usable(Task.STT, "bn") is None


def test_matrix_shape(ledger):
    ledger.record(_cap(CapabilityStatus.MEASURED, "en", engine="alpha", metric=0.02))
    ledger.record(_cap(CapabilityStatus.PARTIAL, "en", engine="beta", metric=0.10))
    ledger.record(_cap(CapabilityStatus.MEASURED, "hi", engine="alpha", metric=0.05))
    matrix = ledger.matrix(Task.STT)
    assert matrix == {
        "en": {"alpha": "measured", "beta": "partial"},
        "hi": {"alpha": "measured"},
    }


def test_report_is_sorted_and_names_status(ledger, registry):
    ledger.record(_cap(CapabilityStatus.MEASURED, "en", engine="alpha", metric=0.02))
    ledger.record(_cap(CapabilityStatus.NO_MODEL, "bn", engine="beta"))
    report = ledger.report(Task.STT, registry)
    assert "measured" in report and "no_model" in report
    assert "English" in report
    # empty ledger for an unrelated task states the honest emptiness
    assert "no capabilities recorded" in CapabilityLedger().report(Task.TTS)


# --- scripts -----------------------------------------------------------------
def test_detect_script_separates_devanagari_bengali_latin():
    assert detect_script("hello world") == ("Latin", 1.0)
    assert detect_script(SAMPLE_SENTENCES["hi"]) == ("Devanagari", 1.0)
    assert detect_script(SAMPLE_SENTENCES["bn"]) == ("Bengali", 1.0)


def test_detect_script_returns_coverage():
    script, coverage = detect_script("aआ")  # 1 Latin letter, 1 Devanagari letter
    assert script in {"Latin", "Devanagari"}
    assert coverage == 0.5


def test_script_fractions_and_is_mixed():
    text = "hello world " + SAMPLE_SENTENCES["hi"]
    fractions = script_fractions(text)
    assert fractions["Latin"] >= 0.15 and fractions["Devanagari"] >= 0.15
    assert is_mixed(text) is True
    assert is_mixed(SAMPLE_SENTENCES["hi"]) is False


def test_every_registry_script_is_classifiable(registry):
    from nativevoice.language import scripts

    for profile in registry.profiles():
        assert profile.script in scripts.SCRIPT_RANGES


# --- lexicon LID -------------------------------------------------------------
@pytest.fixture
def identifier() -> LexiconLanguageIdentifier:
    return LexiconLanguageIdentifier()


def test_lexicon_is_never_calibrated(identifier):
    for sample in SAMPLE_SENTENCES.values():
        assert identifier.identify(sample).is_calibrated is False


def test_lexicon_separates_devanagari_from_bengali(identifier):
    assert identifier.identify(SAMPLE_SENTENCES["hi"]).language.code == "hi"
    assert identifier.identify(SAMPLE_SENTENCES["bn"]).language.code == "bn"
    assert identifier.identify(SAMPLE_SENTENCES["ta"]).language.code == "ta"
    assert identifier.identify(SAMPLE_SENTENCES["te"]).language.code == "te"


def test_lexicon_hindi_vs_marathi(identifier):
    assert identifier.identify(SAMPLE_SENTENCES["hi"]).language.code == "hi"
    assert identifier.identify(SAMPLE_SENTENCES["mr"]).language.code == "mr"


def test_lexicon_english_and_confidence(identifier):
    decision = identifier.identify(SAMPLE_SENTENCES["en"])
    assert decision.language.code == "en"
    assert decision.confidence > 0.5
    assert decision.method == "lexicon-ngram+script"


def test_lexicon_no_letters_returns_und(identifier):
    decision = identifier.identify("123   456 !!!")
    assert decision.language.code == "und"
    assert decision.confidence == 0.0


def test_transliterated_hinglish_low_confidence_with_note(identifier):
    hinglish = "aap kaise hain mujhe thoda paani chahiye kal milte hain"
    decision = identifier.identify(hinglish)
    assert decision.confidence < 0.5
    assert decision.is_calibrated is False
    assert decision.metadata.get("transliterated") is True
    assert "Hinglish" in decision.metadata.get("note", "") or "Roman" in decision.metadata.get("note", "")
    # both Hindi and Urdu offered as alternatives; never silently guessed
    alt_codes = {code for code, _ in decision.alternatives}
    assert {"hi", "ur"}.issubset(alt_codes)


def test_transliterated_risk_returns_none_for_plain_english():
    risk = TransliteratedTextRisk().assess(SAMPLE_SENTENCES["en"])
    assert risk is None


def test_script_only_identifier_for_unlexiconed_script(registry):
    identifier = ScriptOnlyIdentifier(registry)
    decision = identifier.identify("Ελληνικά γλώσσα")  # Greek, no lexicon
    assert decision.method == "script-only"
    assert decision.is_calibrated is False
    assert decision.language.code == "el"


def test_script_only_ambiguous_multi_language_script(registry):
    # Cyrillic maps to several registry languages and has no lexicon -> und.
    decision = ScriptOnlyIdentifier(registry).identify("Русский текст здесь")
    assert decision.language.code == "und"
    assert len(decision.alternatives) >= 2


# --- routing -----------------------------------------------------------------
@pytest.fixture
def router() -> LanguageRouter:
    return LanguageRouter()


def test_unsupported_language_yields_unavailable_with_fallback(router, ledger):
    ledger.record(_cap(CapabilityStatus.MEASURED, "en", engine="alpha", metric=0.02))
    outcome = router.route(ledger=ledger, requested_language=LanguageTag("bn"))
    assert outcome.kind is RoutingKind.UNAVAILABLE
    assert outcome.may_invoke_engine is False
    assert outcome.fallback is not None and outcome.fallback.code == "en"


def test_declared_insufficient_in_strict_mode(router, ledger):
    ledger.record(_cap(CapabilityStatus.DECLARED, "bn", engine="alpha"))
    assert router.route(ledger=ledger, requested_language=LanguageTag("bn")).kind is RoutingKind.UNAVAILABLE
    assert router.route(
        ledger=ledger, requested_language=LanguageTag("bn"), allow_declared=True
    ).kind is RoutingKind.ROUTED


def test_explicit_language_wins_when_usable(router, ledger):
    ledger.record(_cap(CapabilityStatus.MEASURED, "hi", engine="alpha", metric=0.05))
    outcome = router.route(ledger=ledger, requested_language=LanguageTag("hi"))
    assert outcome.kind is RoutingKind.ROUTED
    assert outcome.engine_id == "alpha"


def test_low_confidence_yields_ambiguous(router, ledger):
    # Cyrillic has no lexicon: script-only confidence is low, so ambiguous.
    outcome = router.route(ledger=ledger, spoken_text="Русский текст здесь", audio_duration_s=8.0)
    assert outcome.kind is RoutingKind.AMBIGUOUS
    assert outcome.threshold is not None
    assert outcome.may_invoke_engine is False


def test_autodetect_routes_when_confident(router, ledger):
    ledger.record(_cap(CapabilityStatus.MEASURED, "bn", engine="alpha", metric=0.05))
    outcome = router.route(ledger=ledger, spoken_text=SAMPLE_SENTENCES["bn"], audio_duration_s=8.0)
    assert outcome.kind is RoutingKind.ROUTED
    assert outcome.language.code == "bn"


def test_threshold_scales_down_with_duration(router):
    short = router.threshold_for(0.5)
    long = router.threshold_for(8.0)
    assert short > long >= 0.5


def test_code_switch_detection(router, ledger):
    ledger.record(_cap(CapabilityStatus.MEASURED, "en", engine="alpha", metric=0.02))
    ledger.record(_cap(CapabilityStatus.PARTIAL, "hi", engine="alpha", metric=0.08))
    mixed = "hello my friend " + SAMPLE_SENTENCES["hi"]
    outcome = router.route(ledger=ledger, spoken_text=mixed, audio_duration_s=6.0)
    assert outcome.kind is RoutingKind.CODE_SWITCH
    assert outcome.language is not None and outcome.secondary_language is not None
    named = {outcome.language.code, outcome.secondary_language.code}
    assert "hi" in named and "en" in named


def test_no_text_defers_to_engine_auto_detection(router, ledger):
    ledger.record(_cap(CapabilityStatus.MEASURED, "en", engine="alpha", metric=0.02))
    outcome = router.route(ledger=ledger, audio_duration_s=3.0)
    assert outcome.kind is RoutingKind.ROUTED
    assert outcome.language is None  # engine names the language itself
    assert outcome.metadata.get("auto_detect") is True


def test_no_capability_anywhere_is_unavailable(router, ledger):
    outcome = router.route(ledger=ledger, audio_duration_s=3.0)
    assert outcome.kind is RoutingKind.UNAVAILABLE


def test_corpus_covers_required_languages():
    required = {"en", "hi", "bn", "ta", "te", "mr", "ur", "es", "fr", "de", "ja", "zh", "ar"}
    assert required.issubset(set(STOPWORDS))
