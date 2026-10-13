"""Tests for the multilingual evaluation instrument, not for any model.

The comparison it produces is only as trustworthy as these functions, and two of them were wrong when first
written - a cue extractor that read the speaker's name instead of the quoted sentence, and a "longest common
subsequence" that was a greedy scan. Both are pinned here so they cannot silently return.

No microphone, no loudspeaker, no model, no network.
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

CHINESE_NAME = ("黄丽萍长自述1-2026-10-13-18-31-你们知道吗？每次照镜子的时候，我都会数一数又长出多少根白发。我现在才明白，这些白发就像是我们家族.mp3")
ENGLISH_NAME = ("Paula-2026-10-13-18-26-Hi-everyone,-I'm-excited-to-share-some-insights.mp3")
UNTIMED_NAME = "voice_preview_ultron (2).wav"

# the real transcript that the greedy version scored at 0.04 despite being verbatim
VERBATIM = ("你知道吗,每次照镜子的时候,我都会数一数又长出多少根白发,我现在才明白,这些白发就像是我们家族的印记,"
            "以前觉得是负担,现在慢慢学会接纳了,其实每个人都有自己的不完美,重要的是怎么去")
TRADITIONALISED = ("您知道嗎?每次照鏡子的時候我都會屬於屬於屬於長出多少根白髮 我現在才明白這些白髮就像是我們家族的印記 "
                   "以前覺得是負擔,現在慢慢學會階納了")


class TestCueExtraction:
    """A48: the cue lives after the timestamp, not in the speaker prefix."""

    def test_the_quoted_sentence_is_taken_from_after_the_timestamp(self):
        words = quote_from_name(ENGLISH_NAME)
        assert {"everyone", "excited", "share", "insights"} <= set(words)
        assert "Paula" not in words

    def test_the_cue_keeps_the_original_word_order_for_subsequence_checks(self):
        tail = quote_tail_from_name(CHINESE_NAME)
        assert tail.startswith("你们知道吗")
        assert "2026-10-13" not in tail

    def test_an_unstamped_filename_has_no_cue_rather_than_a_fake_one(self):
        assert quote_from_name(UNTIMED_NAME) == []
        assert quote_tail_from_name(UNTIMED_NAME) == ""
        assert char_recall(quote_tail_from_name(UNTIMED_NAME), "anything") is None

    def test_punctuation_in_the_name_is_not_treated_as_a_word(self):
        words = quote_from_name(CHINESE_NAME)
        assert len(words) == 1  # an unspaced sentence is one token, which is exactly the A48 trap


class TestCharacterRecall:
    """A48 again, from the other side: an unspaced cue needs a subsequence, not a substring."""

    def test_a_verbatim_transcript_with_restyled_punctuation_scores_high(self):
        recall = char_recall(quote_tail_from_name(CHINESE_NAME), VERBATIM)
        assert recall is not None and recall >= 0.85

    def test_a_greedy_scan_would_have_failed_here_and_now_does_not(self):
        # the cue opens 你们 and the transcript opens 你知道吗: a first-match scan stalls at one character
        # of nine, and a first-match scan stalls at one character out of ten. The proper subsequence keeps
        # nine of ten, losing only the 们 the model never said.
        assert char_recall("你们知道吗每次照镜子", "你知道吗每次照镜子") == 0.9

    def test_an_unrelated_transcript_scores_low(self):
        assert char_recall(quote_tail_from_name(CHINESE_NAME), "completely unrelated english text") < 0.10

    def test_the_same_script_written_differently_lowers_recall_without_breaking(self):
        recall = char_recall(quote_tail_from_name(CHINESE_NAME), TRADITIONALISED)
        assert 0.3 < recall < 0.95

    def test_recall_is_bounded_and_monotonic(self):
        cue = "abcdefghij"
        assert char_recall(cue, "") is None
        assert char_recall(cue, cue) == 1.0
        assert char_recall(cue, "abc") < char_recall(cue, "abcdefghij")


class TestWordCueOverlap:
    def test_spaced_scripts_are_measured_by_word_presence(self):
        result = cue_overlap(["everyone", "excited", "share"], "Hi everyone, I am excited to tell you")
        assert result["filename_cue_words"] == 3
        assert result["filename_cue_matched"] == 2
        assert result["filename_cue_overlap"] == pytest.approx(0.667)

    def test_the_result_admits_it_is_not_a_reference(self):
        assert "not annotated references" in cue_overlap(["a"], "a")["note"]


class TestScriptOf:
    """A language code and a writing system are different facts, and the baseline conflated them."""

    def test_devanagari_and_bengali_are_told_apart_from_arabic(self):
        assert script_of("मैं अक्सर आसमान से") == "devanagari"
        assert script_of("আমি প্রায় আকাশ থেকে") == "bengali"
        assert script_of("میں اکسر آسوان سے") == "arabic"

    def test_a_hindi_transcript_written_in_arabic_script_is_reported_as_arabic_script(self):
        # this is the real finding: asserted `hi`, written in Perso-Arabic - right language, wrong script
        assert script_of("میں اکسل آسوان سے گیتی ہوئی") == "arabic"

    def test_latin_cyrillic_hangul_kana_and_cjk_are_each_identified(self):
        for text, expected in [("Hello world", "latin"), ("Я не прошу любви", "cyrillic"),
                               ("고양이가 창문을", "hangul"), ("みなさんこんにちは", "hiragana-katakana"),
                               ("你们知道吗", "cjk")]:
            assert script_of(text) == expected, text

    def test_punctuation_only_text_has_no_script(self):
        assert script_of("...!?") == "none"


class TestRepeatedClauseDetector:
    """Documents a known gap rather than pretending it does not exist.

    The detector looks for repeated comma-separated clauses. The baseline's Chinese output repeated a
    *phrase inside one clause* (都會屬於屬於屬於), which it does not see - so "repeated_clause: false" in the
    artefacts does not mean no repetition happened. Pinned so the limitation stays visible instead of
    becoming an invisible assurance.
    """

    def test_a_repeated_clause_is_caught(self):
        assert repeated("hello world, hello world, hello world, hello world") is True

    def test_normal_distinct_clauses_are_not_flagged(self):
        assert repeated("Hi everyone, I am excited to share, some insights") is False

    def test_intra_clause_repetition_is_not_caught_known_gap(self):
        assert repeated("我都會屬於屬於屬於長出多少根白髮") is False
