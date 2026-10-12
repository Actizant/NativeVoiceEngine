"""Metrics are only useful if the arithmetic can be checked by hand, so these tests
state the expected alignment explicitly rather than snapshotting a value.
"""

from __future__ import annotations

import math

import pytest

from nativevoice.core.errors import CapabilityUnavailable
from nativevoice.eval.metrics import (
    aggregate_rates,
    character_error_rate,
    mcd_placeholder,
    normalise,
    sentence_error_rate,
    tokenize,
    word_alignment,
    word_error_rate,
)

REFERENCE = "the quick brown fox jumps over the lazy dog"
HYPOTHESIS = "the quick brown fox jumped over lazy dog"


def test_wer_matches_hand_alignment():
    # 9 reference tokens; one substitution (jumps->jumped) and one deletion (extra "the").
    result = word_alignment(REFERENCE, HYPOTHESIS, level="word")
    assert result.alignment.substitutions == 1
    assert result.alignment.deletions == 1
    assert result.alignment.insertions == 0
    assert result.alignment.correct == 7
    assert result.alignment.distance == 2
    assert result.alignment.reference_length == 9
    assert math.isclose(word_error_rate(REFERENCE, HYPOTHESIS), 2 / 9, rel_tol=1e-12)
    assert math.isclose(result.rate, 2 / 9, rel_tol=1e-12)


def test_insertion_only_pair():
    result = word_alignment("one two", "one two three four", level="word")
    assert result.alignment.insertions == 2
    assert result.alignment.distance == 2
    # Normalised by the reference length (2 tokens), so the rate exceeds 1.0 by design.
    assert math.isclose(result.rate, 1.0, rel_tol=1e-12)


def test_empty_reference_with_invented_text_is_infinite_not_zero():
    result = word_alignment("", "the the the", level="word")
    assert result.rate == math.inf
    assert word_error_rate("", "something") == math.inf


def test_empty_both_scores_zero():
    assert word_error_rate("", "") == 0.0
    assert character_error_rate("", "") == 0.0


def test_identical_text_scores_zero():
    assert word_error_rate(REFERENCE, REFERENCE) == 0.0
    assert word_alignment(REFERENCE, REFERENCE).alignment.distance == 0


def test_punctuation_and_case_are_not_tokens():
    assert word_error_rate("Hello, world!", "hello world") == 0.0
    assert tokenize("Hello, world!") == ["hello", "world"]


def test_character_error_rate_ignores_spaces_but_counts_letters():
    # "cat" -> "cot": one character substitution out of three characters.
    assert math.isclose(character_error_rate("cat dog", "cot dog"), 1 / 6, rel_tol=1e-12)
    assert math.isclose(character_error_rate("abc", "a bc"), 0.0, rel_tol=1e-12)


def test_devanagari_danda_is_a_boundary_not_a_token():
    assert tokenize("राम जाता है।") == ["राम", "जाता", "है"]
    assert tokenize("राम जाता है॥ सीता आती है॥") == ["राम", "जाता", "है", "सीता", "आती", "है"]
    assert word_error_rate("मैं हिंदी में बोलता हूँ।", "मैं हिंदी मे बोलता हूँ।") == 0.2


def test_bengali_script_and_zero_width_marks_are_normalised():
    assert tokenize("আমি বাংলায় কথা বলি।") == ["আমি", "বাংলায়", "কথা", "বলি"]
    # ZWJ/ZWNJ carry no pronunciation; a transcript differing only in them is correct.
    # Built from code points so the literals cannot silently differ in a matra.
    left = "\u09aa\u09cd\u09b0\u09a4\u09cd"
    right = "\u09af\u09bc"
    joined = left + chr(0x200D) + right
    disjoined = left + chr(0x200C) + right
    plain = left + right
    assert word_error_rate(joined, plain) == 0.0
    assert word_error_rate(disjoined, plain) == 0.0
    assert word_error_rate("আমি বাংলায় কথা বলি।", "আমি বাংলায় কথা বলে।") == 0.25


def test_normalise_is_conservative_and_never_rewrites_words():
    assert normalise("  A\u00a0B\tC  ") == "A B C"
    assert normalise("नमस्ते\u200c, dunia।") == "नमस्ते dunia"
    # No transliteration, no digit folding, no lemma: the tokens survive untouched.
    assert normalise("1984 रोटी") == "1984 रोटी"


def test_sentence_error_rate_counts_wrong_sentences_only():
    assert sentence_error_rate("aaaa। bbbb।", "aaaa। bbbb।") == 0.0
    assert sentence_error_rate("aaaa। bbbb।", "aaaa। cccc।") == 0.5
    # A dropped final sentence is one more wrong sentence out of two.
    assert sentence_error_rate("aaaa। bbbb।", "aaaa।") == 0.5
    assert sentence_error_rate("", "") == 0.0
    assert sentence_error_rate("", " invented") == 1.0


def test_alignment_breakdown_is_algebraically_consistent():
    for reference, hypothesis in [
        (REFERENCE, HYPOTHESIS),
        ("a b c d e", "a x c"),
        ("", "one two"),
        ("one two", ""),
    ]:
        result = word_alignment(reference, hypothesis, level="word")
        alignment = result.alignment
        assert alignment.distance == alignment.substitutions + alignment.insertions + alignment.deletions
        assert alignment.reference_length == alignment.correct + alignment.substitutions + alignment.deletions


def test_alignment_dict_is_json_serialisable_for_reporting():
    import json

    payload = json.loads(json.dumps(word_alignment(REFERENCE, HYPOTHESIS, level="word").as_dict()))
    assert payload["substitutions"] == 1
    assert payload["rate_infinite"] is False
    assert math.isclose(payload["rate"], 2 / 9, rel_tol=1e-5)


def test_aggregate_rates_is_a_macro_mean():
    assert math.isclose(aggregate_rates([0.0, 1.0]), 0.5)
    assert math.isnan(aggregate_rates([]))


def test_mcd_is_refused_not_invented():
    with pytest.raises(CapabilityUnavailable) as excinfo:
        mcd_placeholder()
    message = str(excinfo.value)
    assert "mel-cepstral" in message
    assert "external, unapproved" in message
