"""Transcription scoring: aggregation, per-language worst case, hallucination flags."""

from __future__ import annotations

import math

import pytest

from fixtures import noise, silence, speech_like
from nativevoice.core.audio import AudioBuffer, AudioFormat
from nativevoice.core.errors import CapabilityUnavailable
from nativevoice.core.language import LanguageTag
from nativevoice.eval.samples import EvalSample, builtin_fixtures
from nativevoice.eval.transcription import (
    Pair,
    dominant_token_share,
    evaluate_pairs,
    hallucination_flags,
    repetition_ratio,
    score,
)

EN_REFERENCE = "the quick brown fox jumps over the lazy dog"
EN_HYPOTHESIS = "the quick brown fox jumped over lazy dog"


def test_score_reports_the_hand_checkable_alignment():
    result = score(EN_REFERENCE, EN_HYPOTHESIS, "en-US")
    assert result.language == "en-US"
    assert math.isclose(result.wer, 2 / 9, rel_tol=1e-12)
    assert result.word.alignment.substitutions == 1
    assert result.word.alignment.deletions == 1
    assert result.reference_tokens == 9
    assert result.sentence_error_rate == 1.0
    assert result.perfect is False


def test_score_accepts_a_language_tag():
    result = score("আমি বাংলায় কথা বলি", "আমি বাংলায় কথা বলি", LanguageTag("bn"))
    assert result.wer == 0.0 and result.cer == 0.0 and result.perfect


def test_plain_strings_are_scored_but_cannot_certify_quality():
    aggregate = evaluate_pairs([(EN_REFERENCE, EN_HYPOTHESIS, "en")])
    assert aggregate.certifies_quality is False
    assert any("EvalSample" in note for note in aggregate.provenance_notes)
    assert math.isclose(aggregate.overall.micro_wer, 2 / 9, rel_tol=1e-12)


def _pair(language: str, reference: str, hypothesis: str) -> Pair:
    return Pair.from_sample(
        EvalSample(
            path=f"{language}.wav",
            language=language,
            transcript=reference,
            licence="CC-BY-4.0",
            consent_basis="recorded with speaker consent",
        ),
        hypothesis,
    )


def test_evaluation_groups_per_language_and_keeps_worst_case():
    aggregate = evaluate_pairs(
        [
            _pair("en", "one two three", "one two three"),
            _pair("en", "one two three", "one two four"),
            _pair("bn", "আমি বাংলায় কথা বলি", "complete nonsense here now"),
        ]
    )
    assert aggregate.certifies_quality is True
    english = aggregate.language("en")
    bengali = aggregate.language("bn")
    assert english.pairs == 2 and bengali.pairs == 1
    assert math.isclose(english.micro_wer, 1 / 6, rel_tol=1e-12)
    assert math.isclose(english.worst_wer, 1 / 3, rel_tol=1e-12)
    assert bengali.micro_wer >= 1.0
    # The worst language is the one that a corpus average would have buried.
    assert aggregate.worst_language().language == "bn"
    assert bengali.worst_hypothesis == "complete nonsense here now"


def test_synthetic_samples_are_refused_before_any_metric():
    manifest = builtin_fixtures()
    pairs = [Pair.from_sample(sample, "anything at all") for sample in manifest.samples[:2]]
    with pytest.raises(CapabilityUnavailable) as excinfo:
        evaluate_pairs(pairs)
    assert "synthetic" in str(excinfo.value)
    # Escaping the rule is possible only for diagnostics, and then quality is not certified.
    diagnostic = evaluate_pairs(pairs, enforce_synthetic_rule=False)
    assert diagnostic.certifies_quality is False
    assert any("synthetic" in note for note in diagnostic.provenance_notes)


def test_unlicensed_real_sample_does_not_certify():
    pairs = [
        Pair(
            "one two",
            "one two",
            "en",
            sample=EvalSample(path="x.wav", language="en", transcript="one two", licence="CC0"),
        )
    ]
    aggregate = evaluate_pairs(pairs)
    assert aggregate.certifies_quality is False
    assert any("missing licence or consent" in note for note in aggregate.provenance_notes)


def test_a_bare_sample_is_not_a_scored_pair():
    with pytest.raises(CapabilityUnavailable):
        evaluate_pairs([EvalSample(path="x.wav", language="en", transcript="one", licence="CC0", consent_basis="ok")])


def test_repetition_ratio_detects_a_looping_decoder():
    assert repetition_ratio("the the the the") == 1.0
    assert repetition_ratio("one two three four") == 0.0
    assert dominant_token_share("system system system error") == 0.75


def test_hallucination_flags_repetition_over_near_silence():
    flags = hallucination_flags("", "the the the the the", audio=silence(seconds=1.0))
    kinds = {flag.kind for flag in flags}
    assert {"repeated_token_run", "text_over_empty_reference", "text_over_near_silence"} <= kinds
    rendered = [flag.as_dict() for flag in flags]
    assert all(row["value"] is not None for row in rendered)


def test_hallucination_flags_single_dictionary_word_loop():
    # Four identical adjacent tokens trip the repetition flag too, and that is correct;
    # this pair isolates the *dominant token* signal with no adjacent repeats at all.
    flags = hallucination_flags("nothing said at all here", "system error system warning system")
    assert {flag.kind for flag in flags} == {"single_dominant_token"}
    adjacent = hallucination_flags("nothing said at all here", "appappapp appappapp appappapp appappapp")
    assert {flag.kind for flag in adjacent} == {"single_dominant_token", "repeated_token_run"}


def test_quiet_audio_with_a_clean_transcript_is_not_flagged():
    buffer: AudioBuffer = speech_like()
    flags = hallucination_flags("hello there", "hello there", audio=buffer)
    assert flags == ()


def test_low_energy_noise_without_reference_text_is_flagged():
    flags = hallucination_flags("", "i am ready", audio=noise(seconds=1.0, amplitude=0.001))
    assert any(flag.kind == "text_over_near_silence" for flag in flags)


def test_silence_only_buffer_can_still_be_scored_for_transport():
    """A zero-frame buffer has no reference tokens, so invented text is infinite error."""
    empty = AudioBuffer.zeros(AudioFormat(16000, 1), 0)
    result = score("", "hallucinated sentence")
    assert result.wer == math.inf
    assert empty.frame_count == 0


def test_aggregate_serialises_without_infinity_leaking_into_json():
    import json

    aggregate = evaluate_pairs([_pair("en", "", "invented words")])
    payload = json.loads(json.dumps(aggregate.as_dict()))
    assert payload["overall"]["worst_wer"] is None
    assert payload["overall"]["reference_tokens"] == 0
