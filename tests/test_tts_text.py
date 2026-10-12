"""Normalisation and chunking: the front-end contracts, asserted.

Assertions about Devanagari/Bengali word *content* are made against the authored
rule tables rather than retyped literals: those tables are flagged `verified=False`,
so a test that hard-codes my spelling would only prove I can paste twice. What is
verified here is the wiring - native digits reach the table, the table's words reach
the output, and nothing silently keeps a digit.
"""

from __future__ import annotations

import pytest

from nativevoice.core.errors import ConfigurationError
from nativevoice.tts.chunking import TextChunker
from nativevoice.tts.normalise import (
    BENGALI_DIGITS,
    DEFAULT_RULES,
    DEVANAGARI_DIGITS,
    SPOKEN_NUMBER_CEILING,
    NormalisationReport,
    TextNormaliser,
)


@pytest.fixture
def normaliser() -> TextNormaliser:
    return TextNormaliser()


def _in_native_script(value: int, digits: str) -> str:
    return "".join(digits[int(character)] for character in str(value))


class TestNormalisation:
    def test_collapses_whitespace_and_guarantees_terminal_punctuation(self, normaliser):
        report = normaliser.normalise("hello    there\nfriend", "en")
        assert report.text == "hello there friend."
        assert {"whitespace", "terminal_punctuation"} <= set(report.kinds())

    def test_missing_punctuation_is_the_documented_mispronunciation_bug(self, normaliser):
        report = normaliser.normalise("Open the pod bay doors", "en")
        assert report.text.endswith(".")
        assert any(change.kind == "terminal_punctuation" for change in report.changes)

    def test_existing_terminators_are_left_alone(self, normaliser):
        for text, code in (
            ("Stay here.", "en"),
            ("Really?", "en"),
            ("Stop!", "en"),
            ("ঠিক আছে।", "bn"),
        ):
            assert normaliser.normalised_text(text, code) == text

    def test_dangling_comma_becomes_a_stop_not_a_second_stop(self, normaliser):
        assert normaliser.normalised_text("one, two, three,", "en") == "one, two, three."

    def test_scripts_get_their_own_terminal(self, normaliser):
        assert normaliser.normalised_text("ठीक আছে", "hi").endswith("।")
        assert normaliser.normalised_text("ঠিক আছে", "bn").endswith("।")

    def test_control_characters_are_rejected(self, normaliser):
        with pytest.raises(ConfigurationError) as refused:
            normaliser.normalise("bell\x07 in the text", "en")
        assert "control character" in str(refused.value)

    def test_html_and_entities_are_rejected(self, normaliser):
        for payload in ("<p>hello</p>", "a &amp; b", "<br/>"):
            with pytest.raises(ConfigurationError):
                normaliser.normalise(payload, "en")

    def test_english_digit_expansion(self, normaliser):
        assert normaliser.normalised_text("There are 42 items.", "en") == "There are forty two items."
        assert normaliser.normalised_text("0 degrees", "en") == "zero degrees."
        assert normaliser.normalised_text("room 101", "en") == "room one hundred and one."

    @pytest.mark.parametrize("language,digits", [("hi", DEVANAGARI_DIGITS), ("bn", BENGALI_DIGITS)])
    @pytest.mark.parametrize("value", [0, 5, 12, 20, 21, 42, 60, 99, 100, 305, 999])
    def test_native_digits_are_replaced_by_the_authored_words(self, normaliser, language, digits, value):
        grammar = DEFAULT_RULES[language].numbers
        expected = normaliser.number_words(value, grammar)
        assert expected
        spoken = normaliser.normalised_text(_in_native_script(value, digits), language)
        assert not any(character in spoken for character in digits)
        for word in expected.split(grammar.separator):
            assert word in spoken

    @pytest.mark.parametrize("language", ["en", "hi", "bn"])
    def test_every_number_up_to_the_ceiling_has_an_authored_form(self, normaliser, language):
        grammar = normaliser.rules_for(language).numbers
        missing = [
            value
            for value in range(0, SPOKEN_NUMBER_CEILING + 1)
            if not normaliser.number_words(value, grammar)
        ]
        assert missing == []

    @pytest.mark.parametrize("language", ["en", "hi", "bn"])
    def test_above_the_ceiling_digits_survive_and_are_reported(self, normaliser, language):
        report = normaliser.normalise("the year 1000 arrived", language)
        assert "1000" in report.text
        assert any("exceeds the authored spoken range" in note for note in report.limitations)

    def test_decimal_is_spoken_not_split(self, normaliser):
        assert "point one four" in normaliser.normalise("pi is 3.14 today", "en").text

    def test_abbreviations_are_expanded_and_reported(self, normaliser):
        report = normaliser.normalise("Dr. Roy said etc. was fine", "en")
        assert "doctor" in report.text and "et cetera" in report.text
        assert any(change.kind == "abbreviation" for change in report.changes)

    def test_currency_and_percent(self, normaliser):
        assert normaliser.normalised_text("$1.50", "en") == "one point five zero dollars."
        assert normaliser.normalised_text("12.5%", "en") == "twelve point five percent."
        assert normaliser.normalised_text("1 dollar", "en") == "one dollar."
        assert "USD" not in normaliser.normalised_text("costs 5 USD", "en")

    def test_taka_and_rupee_symbols_are_expanded_in_their_own_scripts(self, normaliser):
        hindi = normaliser.normalised_text("₹" + _in_native_script(500, DEVANAGARI_DIGITS), "hi")
        assert "रुपया" in hindi or "रुपये" in hindi
        bengali = normaliser.normalised_text("৳" + _in_native_script(50, BENGALI_DIGITS), "bn")
        assert "টাকা" in bengali

    def test_report_lists_every_change_made(self, normaliser):
        report = normaliser.normalise("Mr. Smith paid $5 for 3 things", "en")
        assert isinstance(report, NormalisationReport)
        assert set(report.kinds()) == {
            "abbreviation", "currency", "number", "terminal_punctuation"
        }
        assert report.changed()
        assert report.as_dict()["language"] == "en"
        assert report.as_dict()["changes"][0]["before"]

    def test_unauthored_language_reports_the_gap_instead_of_guessing(self, normaliser):
        report = normaliser.normalise("bonjour 42", "fr")
        assert "42" in report.text
        assert any("no numeral grammar" in note for note in report.limitations)

    @pytest.mark.parametrize("language", ["hi", "bn"])
    def test_fused_numerals_are_flagged_unverified(self, normaliser, language):
        report = normaliser.normalise("some words", language)
        assert any("not verified against a licensed lexicon" in note for note in report.limitations)

    def test_english_grammar_is_not_flagged_as_unverified(self, normaliser):
        assert normaliser.normalise("seven words only", "en").limitations == ()


class TestChunking:
    def test_decimal_ellipsis_and_initialism_do_not_end_a_sentence(self):
        chunker = TextChunker(max_tokens=50)
        sentences = chunker.sentences("The ratio is 3.14 today. The U.S.A. waited... then left.")
        assert sentences[0] == "The ratio is 3.14 today."
        assert "U.S.A." in sentences[1]
        assert "then left." in sentences[1]

    def test_oversized_sentence_sub_splits_on_commas(self):
        chunker = TextChunker(max_tokens=6)
        text = "alpha one, beta two, gamma three, delta four, epsilon five."
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        assert all(chunk.tokens <= 6 for chunk in chunks)
        assert TextChunker.reconstruct(chunks) == text

    def test_budget_is_respected_across_many_sentences(self):
        chunker = TextChunker(max_tokens=10)
        text = " ".join(f"Sentence number {index} has words." for index in range(20))
        chunks = chunker.chunk(text)
        assert all(chunk.tokens <= 10 for chunk in chunks)
        assert len(chunks) > 1

    def test_a_single_unsplittable_sentence_is_never_truncated(self):
        chunker = TextChunker(max_tokens=1)
        text = "supercalifragilistic expialidocious."
        chunks = chunker.chunk(text)
        assert TextChunker.reconstruct(chunks) == text
        assert chunks[0].tokens > chunker.max_tokens

    def test_lossless_for_five_hundred_words(self):
        words = [f"word{index}" for index in range(500)]
        text = " ".join(" ".join(words[start : start + 10]) + "." for start in range(0, 500, 10))
        assert len(text.split()) == 500
        chunker = TextChunker(max_tokens=25)
        chunks = chunker.chunk(text)
        assert "".join(chunk.text for chunk in chunks) == text
        assert sum(len(chunk.text.split()) for chunk in chunks) == 500

    def test_bengali_sentence_boundaries_stay_lossless(self):
        chunker = TextChunker(max_tokens=8)
        text = "প্রথম বাক্য। দ্বিতীয় বাক্য। তৃতীয় বাক্য শেষ।"
        chunks = chunker.chunk(text)
        assert TextChunker.reconstruct(chunks) == text
        assert len(chunks) >= 1

    def test_empty_input_yields_no_chunks(self):
        assert TextChunker().chunk("") == ()

    def test_budget_utilisation_is_reported(self):
        chunker = TextChunker(max_tokens=10)
        chunks = chunker.chunk("one two three. four five six.")
        assert 0.0 < chunker.budget_utilisation(chunks) <= 1.0
