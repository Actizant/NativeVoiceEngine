"""The maths the multilingual instrument is made of, tested apart from any model.

No inference, no weights, no download, no asset gate. The functions here turn a truncated
filename cue into the only two signals this evaluation has for "did the right language come
back", and `tools/multilingual_compare.py` consumes all of them per sample. They are not
accuracy and never claim to be; they are the gross-error check that stands in for accuracy,
so a wrong number here silently becomes a published comparison table.

This file exists because the measurement-seam rewrite of `tests/test_multilingual_eval.py`
deleted the classes that pinned these helpers, and the comparator went on calling them
untested - `char_recall`, `cue_overlap` and `quote_from_name` from `tools/multilingual_eval.py`
and `script_of` from `tools/multilingual_compare.py` had no assertion anywhere. The cases were
reconstructed from the implementations and from finding A48, which is the reason two of them
are shaped the way they are: a cue extractor that had read the speaker's prefix instead of the
quoted sentence, and a "longest common subsequence" that was a greedy first-match scan and
scored a verbatim Chinese transcript at 0.04.

Every fixture is synthetic. The sample identifiers the deleted tests embedded are third-party
filenames that appear in `tools/multilingual_eval.py` and in the committed evaluation
artefacts; they are not reproduced here, and every expected value below is derived from the
algorithm by hand rather than read back out of the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.multilingual_compare import script_of  # noqa: E402
from tools.multilingual_eval import (  # noqa: E402
    char_recall,
    cue_overlap,
    quote_from_name,
    quote_tail_from_name,
    repeated,
)

# A filename of the shape this source platform produces: a speaker prefix, a timestamp, then
# the first words of the clip joined by hyphens. The prefix and the timestamp are noise.
SPACED_NAME = "Speaker-One-2026-10-13-18-26-Hi-friends,-I'm-happy-to-share-a-few-ideas.mp3"
SPACED_TAIL = "Hi-friends,-I'm-happy-to-share-a-few-ideas"
# An unspaced cue: the whole quoted sentence is one token, which is exactly the A48 trap.
UNSPACED_NAME = "話者一-2026-01-02-03-04-あいうえおかきくけこさしす.mp3"
UNSPACED_TAIL = "あいうえおかきくけこさしす"
UNTIMED_NAME = "voice_preview_draft (2).wav"


class TestQuoteFromName:
    """A48: the cue lives after the timestamp, and only after it."""

    def test_the_cue_is_every_word_after_the_timestamp_in_order(self):
        # "Hi" and "to" are two characters, so the length filter drops them; the speaker
        # prefix sits before the stamp and must never reach the cue.
        assert quote_from_name(SPACED_NAME) == ["friends", "I'm", "happy", "share", "few", "ideas"]

    def test_the_speaker_prefix_is_not_the_cue(self):
        words = quote_from_name(SPACED_NAME)
        assert "Speaker" not in words and "One" not in words

    def test_a_name_without_a_timestamp_has_no_cue_rather_than_a_fake_one(self):
        assert quote_from_name(UNTIMED_NAME) == []
        assert quote_tail_from_name(UNTIMED_NAME) == ""

    def test_a_partial_timestamp_does_not_open_a_cue(self):
        # The stamp needs date, hour, minute and its trailing dash; a shorter one is ordinary
        # text, and matching it anyway would invent a cue boundary.
        assert quote_from_name("clip-2026-10-13-18.mp3") == []
        assert quote_from_name("clip-2026-10-13-18-31.mp3") == []
        assert quote_from_name("2026-10-13-18-26-Hello-there.mp3") == ["Hello", "there"]

    def test_digits_and_short_words_are_not_cue_words(self):
        assert quote_from_name("clip-2026-10-13-18-31-ab-12-cd-hello.wav") == ["hello"]

    def test_an_unspaced_sentence_arrives_as_exactly_one_token(self):
        words = quote_from_name(UNSPACED_NAME)
        assert words == [UNSPACED_TAIL], "one 13-character 'word' is what word overlap then scores 0.0 against"

    def test_the_tail_is_the_raw_text_before_any_splitting(self):
        assert quote_tail_from_name(SPACED_NAME) == SPACED_TAIL
        assert quote_tail_from_name(UNSPACED_NAME) == UNSPACED_TAIL


class TestCharacterRecall:
    """The other half of A48: an unspaced cue needs a real longest common subsequence."""

    def test_the_subsequence_is_longest_not_first_match(self):
        # The cue opens 你们 and a transcript that opens 你知道吗 shares 你知道吗 with it. A
        # first-match scan spends its one 你 then hunts 们 through the rest and returns 1/5;
        # the dynamic program keeps 4 of 5. This single number is the defect that was fixed.
        assert char_recall("你们知道吗", "你知道吗") == 0.8

    def test_punctuation_a_model_inserts_or_drops_is_not_a_mismatch(self):
        assert char_recall("a,b.c!d", "abcd") == 1.0
        assert char_recall("a b_c-d.e", "abcde") == 1.0

    def test_fullwidth_punctuation_is_not_in_the_skip_set_while_ascii_is(self):
        # CUE_SKIP_CHARACTERS covers the ASCII marks and a handful of typographic ones, but
        # not U+FF1F. The same four shared characters therefore score differently, and a
        # reader comparing two cues has to know the denominator moved.
        assert char_recall("你们知道吗?", "你知道吗") == 0.8
        assert char_recall("你们知道吗？", "你知道吗") == 0.6667

    def test_recall_is_normalised_by_the_cue_not_by_the_transcript(self):
        assert char_recall("abc", "abcdefghij") == 1.0
        assert char_recall("abcdefghij", "abc") == pytest.approx(0.3)

    def test_an_empty_cue_or_an_empty_transcript_is_none_not_zero(self):
        # None is the honest answer: nothing was measurable. A zero here would read as a
        # transcript in the wrong language.
        assert char_recall("", "anything") is None
        assert char_recall("   ", "anything") is None
        assert char_recall(".,!?,,", "anything") is None
        assert char_recall("abc", "") is None
        assert char_recall("abc", " .,. ") is None

    def test_case_is_not_folded_and_that_is_a_limitation(self):
        # `cue_overlap` lowercases both sides; this function compares code points, so a
        # Latin transcript that differs only in case scores nothing. Pinned because it is a
        # property of the measure, not a preference.
        assert char_recall("ABC", "abc") == 0.0

    def test_the_same_language_in_a_different_script_lowers_recall_without_breaking(self):
        assert char_recall("我們学习", "我们学习") == pytest.approx(0.75)

    def test_recall_is_bounded_and_monotonic_in_shared_characters(self):
        cue = "abcdefghij"
        assert char_recall(cue, cue) == 1.0
        assert char_recall(cue, "abc") < char_recall(cue, "abcdefghij")
        assert 0.0 <= char_recall(cue, "zzzabczzz") <= 1.0
        assert char_recall("aabb", "abab") == 0.75, "a shared multiset is not a subsequence: three of four"


class TestCueOverlap:
    """Word presence against a truncated cue - indicative, and honest about being indicative."""

    def test_spaced_scripts_are_measured_by_word_presence(self):
        result = cue_overlap(["friends", "happy", "share"], "Hi friends, I am happy to tell you")
        assert result["filename_cue_words"] == 3
        assert result["filename_cue_matched"] == 2
        assert result["filename_cue_overlap"] == pytest.approx(0.667)

    def test_no_cue_words_is_none_rather_than_a_division_by_zero(self):
        result = cue_overlap([], "any transcript at all")
        assert result["filename_cue_words"] == 0
        assert result["filename_cue_matched"] == 0
        assert result["filename_cue_overlap"] is None

    def test_an_empty_transcript_scores_zero_because_there_was_a_cue_to_match(self):
        result = cue_overlap(["hello"], "")
        assert result["filename_cue_overlap"] == 0.0
        assert result["filename_cue_matched"] == 0

    def test_matching_is_case_insensitive(self):
        assert cue_overlap(["Friends"], "friends said it")["filename_cue_matched"] == 1

    def test_a_cue_word_can_match_inside_a_longer_word(self):
        # The check is a substring test, so "share" is satisfied by "shared". Deliberately
        # pinned: an overlap of 1.0 is not proof that the word was spoken.
        assert cue_overlap(["share"], "they shared it")["filename_cue_matched"] == 1

    def test_a_repeated_cue_word_is_counted_twice(self):
        result = cue_overlap(["hi", "hi"], "hi there")
        assert result["filename_cue_words"] == 2
        assert result["filename_cue_matched"] == 2, "one spoken word satisfies both copies"
        assert result["filename_cue_overlap"] == 1.0

    def test_punctuation_is_stripped_at_the_edges_of_a_cue_word_only(self):
        assert cue_overlap(["ideas,"], "a few ideas here")["filename_cue_matched"] == 1
        assert cue_overlap(["it's"], "it s")["filename_cue_matched"] == 0

    def test_the_result_refuses_to_call_itself_a_reference(self):
        assert "not annotated references" in cue_overlap(["a"], "a")["note"]


class TestRepeatedClauseDetector:
    """Whisper's loop failure, and the size of clause it cannot see.

    The detector compares comma-separated clauses against a uniqueness ratio. The baseline's
    Chinese output repeated a phrase *inside* one clause, which this does not see, so
    `repeated_clause: false` in an artefact is not a claim that no repetition happened.
    """

    def test_a_repeated_clause_is_caught(self):
        assert repeated("hello world, hello world, hello world, hello world") is True

    def test_normal_distinct_clauses_are_not_flagged(self):
        assert repeated("Hi friends, I am happy to share, a few ideas") is False

    def test_intra_clause_repetition_is_not_caught_known_gap(self):
        # One clause, one token looped inside it: the detector splits on commas, sees a single part, and
        # returns before any ratio is computed. Synthetic wording - the shape of the baseline's failure,
        # not its transcript.
        assert repeated("今天天气天气天气天气很好") is False

    def test_two_of_three_identical_clauses_is_below_the_ratio(self):
        # 3 clauses, 2 distinct: the test is `2 < 3 * 0.6`, which is false.
        assert repeated("one, two, two") is False

    def test_half_the_clauses_distinct_is_above_the_ratio(self):
        # 4 clauses, 2 distinct: `2 < 4 * 0.6` is true.
        assert repeated("one, one, two, two") is True

    def test_the_ratio_is_strict_at_three_of_five(self):
        # 5 clauses, 3 distinct: `3 < 3.0` is false, so the boundary is not inclusive.
        assert repeated("a, a, b, b, c") is False

    def test_fewer_than_three_clauses_is_never_flagged(self):
        assert repeated("yes, yes") is False
        assert repeated("") is False
        assert repeated(",,,") is False

    def test_a_fullwidth_comma_separates_clauses_like_an_ascii_one(self):
        assert repeated("好的，好的，好的，好的") is True

    def test_clause_comparison_ignores_case_and_padding(self):
        assert repeated("Hello World, hello world, HELLO WORLD") is True


class TestScriptOf:
    """A language code and a writing system are different facts; the baseline conflated them."""

    def test_devanagari_and_bengali_are_told_apart_from_arabic(self):
        assert script_of("मैं अक्सर आसमान से") == "devanagari"
        assert script_of("আমি প্রায় আকাশ থেকে") == "bengali"
        assert script_of("میں اکسر آسوان سے") == "arabic"

    def test_text_in_one_language_written_in_another_script_reports_the_script(self):
        # The A48 finding in miniature: right language, wrong writing system, and the label
        # the instrument gives is the one a reader can actually verify.
        assert script_of("میں آج اسکول گیا") == "arabic"

    def test_latin_cyrillic_hangul_kana_and_cjk_are_each_identified(self):
        for text, expected in [("Hello world", "latin"), ("Я не прошу любви", "cyrillic"),
                               ("고양이가 창문을", "hangul"), ("みなさんこんにちは", "hiragana-katakana"),
                               ("你们知道吗", "cjk")]:
            assert script_of(text) == expected, text

    def test_the_majority_script_wins_a_mixed_transcript(self):
        # मैं is three code points (U+092E U+0948 U+0902), so latin only wins here as a strict
        # majority of four against three - and it wins from the devanagari character whether it
        # came first or last, which is what separates a majority rule from encounter order.
        assert script_of("मैं abcd") == "latin"
        assert script_of("abcd मैं") == "latin"

    def test_a_tie_is_broken_by_which_character_came_first(self):
        # Nothing ranks the scripts, so a 50/50 transcript is labelled by encounter order.
        # Recorded as a limitation of the label, not as desired behaviour.
        assert script_of("अ a") == "devanagari"
        assert script_of("a अ") == "latin"

    def test_punctuation_and_digits_alone_have_no_script(self):
        assert script_of("...!?") == "none"
        assert script_of("123 456") == "none"
        assert script_of("") == "none"

    def test_the_underscore_falls_inside_the_latin_range(self):
        # U+005F sits in 0x0041-0x007A, so a file-name slug is reported as latin script.
        assert script_of("_ _") == "latin"
