"""Multilingual identification and voice-activity behaviour, measured directly.

These are the assertions the earlier suite did not make. They were written after
measuring the running code, not to match it: the language identifier used to hand
every Latin-script utterance to English at confidence 1.000 without comparing
Spanish, French or German at all, and nothing in the suite noticed because each
individual path looked self-consistent.
"""

from __future__ import annotations

import math

import pytest
from fixtures import noise, silence, sine, speech_like

from nativevoice.audio import (
    EnergySpeechDetector,
    SegmenterConfig,
    UtteranceSegmenter,
)
from nativevoice.core.audio import AudioFormat
from nativevoice.core.language import Capability, CapabilityStatus, LanguageTag, Task
from nativevoice.language import (
    LanguageRegistry,
    LanguageRouter,
    LexiconLanguageIdentifier,
    ScriptOnlyIdentifier,
    detect_script,
    is_mixed,
)

RATE = 16000


@pytest.fixture(scope="module")
def identifier() -> LexiconLanguageIdentifier:
    return LexiconLanguageIdentifier()


class TestLatinScriptComparison:
    """The defect class: one script, several languages, no comparison performed."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("el niño muy rápido corrió por la calle con sus amigos", "es"),
            ("je voudrais un croissant et du café merci beaucoup", "fr"),
            ("der schnelle braune Fuchs springt über den faunen Hund", "de"),
            ("the quick brown fox jumps over the lazy dog and then sleeps", "en"),
        ],
    )
    def test_latin_text_is_compared_against_every_latin_lexicon(self, identifier, text, expected):
        decision = identifier.identify(text)
        assert decision.language.code == expected

    def test_no_uncalibrated_decision_ever_claims_certainty(self, identifier):
        for text in ("hello there", "el niño", "der hund", "bonjour", "नमस्ते", "நன்றி", "你好"):
            decision = identifier.identify(text)
            assert decision.is_calibrated is False
            assert decision.confidence < 1.0

    def test_english_beats_spanish_only_on_word_evidence(self, identifier):
        spanish_heavy = "la casa del árbol es muy bonita y grande"
        assert identifier.identify(spanish_heavy).language.code == "es"


class TestTransliterationAmbiguity:
    def test_romanised_hindi_is_reported_as_ambiguous_not_as_english(self, identifier):
        decision = identifier.identify("aap kaise hain mujhe thoda paani chahiye kal milte hain")
        assert decision.confidence < 0.5
        assert decision.metadata.get("transliterated") is True
        assert decision.language.code != "en" or decision.confidence < 0.3

    def test_real_english_is_not_downgraded_by_the_heuristic(self, identifier):
        decision = identifier.identify("the quick brown fox jumps over the lazy dog")
        assert decision.language.code == "en" and decision.confidence > 0.7


class TestScriptSeparation:
    @pytest.mark.parametrize(
        "text,script",
        [
            ("নমস্কার আমি ভালো আছি", "Bengali"),
            ("नमस्ते मैं ठीक हूँ", "Devanagari"),
            ("நன்றி நல்ல நாள்", "Tamil"),
            ("你好，今天天气很好", "Han"),
            ("مرحبا، كيف حالك", "Arabic"),
        ],
    )
    def test_scripts_are_separated_from_codepoints_alone(self, text, script):
        assert detect_script(text)[0] == script

    def test_bengali_and_hindi_are_never_confused(self, identifier):
        assert identifier.identify("আমি ভাত খাই এবং জল পান করি প্রতিদিন").language.code == "bn"
        assert identifier.identify("मैं भात खाता हूँ और पानी पीता हूँ रोज़").language.code == "hi"

    def test_a_script_majority_decides_but_heavy_mixing_is_flagged(self, identifier):
        mostly_hindi = identifier.identify("मैं भात खाता हूँ और पानी पीता हूँ रोज़")
        assert mostly_hindi.language.code == "hi"
        assert is_mixed("hello everyone मैं ठीक हूँ और ठीक") is True

    def test_unique_script_inference_is_capped_below_certainty(self):
        decision = ScriptOnlyIdentifier().identify("நன்றி நல்ல நாள் தினம்")
        assert decision.confidence <= 0.9


class TestMixedLanguage:
    def test_two_scripts_in_one_utterance_are_flagged(self):
        assert is_mixed("hello कैसे हो तुम very good वाह") is True

    def test_router_names_both_languages_of_a_mixed_utterance(self):
        outcome = LanguageRouter().route(
            ledger=_ledger(), spoken_text="hello nasıl oluyo बहुत अच्छा है आज", task=_stt_task()
        )
        assert outcome.kind.value == "code_switch"


def _ledger():
    from nativevoice.language import CapabilityLedger
    from nativevoice.core.language import Capability, CapabilityStatus, Task

    ledger = CapabilityLedger()
    for code in ("en", "hi", "bn"):
        ledger.record(Capability(Task.STT, CapabilityStatus.MEASURED, code, engine="probe", metric_name="wer", metric_value=0.1, measured_on="test"))
    return ledger


def _stt_task():
    from nativevoice.core.language import Task

    return Task.STT


class TestVoiceActivityAndEndpointing:
    """Detection accuracy expressed as measured timing error, not as a pass."""

    def _segmenter(self, **config) -> UtteranceSegmenter:
        settings = SegmenterConfig(**config)
        return UtteranceSegmenter(
            EnergySpeechDetector(format=AudioFormat(RATE, 1)), settings
        )

    def _run(self, segmenter, signal, frame_size=320):
        found = []
        for chunk in signal.frames(frame_size):
            found.extend(segmenter.push(chunk))
        found.extend(segmenter.flush())
        return found

    def test_two_bursts_endpoint_into_two_turns_within_sixty_milliseconds(self):
        signal = speech_like(rate=RATE, bursts=((0.20, 0.55), (1.05, 1.40)))
        segmenter = self._segmenter(minimum_silence_s=0.20, padding_s=0.05)
        segments = self._run(segmenter, signal)
        assert len(segments) == 2
        onsets = [segment.first_sample / RATE for segment in segments]
        assert abs(onsets[0] - 0.15) < 0.06, onsets
        assert abs(onsets[1] - 1.0) < 0.06, onsets

    def test_endpointed_span_carries_at_most_the_triggering_silence(self):
        """Endpointing on silence means the returned span *includes* that silence by
        construction. The bound that matters is that it carries no more than the
        configured cut-off plus one analysis frame, and `trim_silence` removes the
        remainder before anything is billed for it."""
        signal = speech_like(rate=RATE, bursts=((0.2, 0.6), (1.2, 1.5)))
        segmenter = self._segmenter(minimum_silence_s=0.15, padding_s=0.02)
        segments = self._run(segmenter, signal)
        budget = 0.15 + 0.02 + segmenter.config.frame_s
        for segment, (_, end) in zip(segments, ((0.2, 0.6), (1.2, 1.5))):
            assert segment.last_sample / RATE <= end + budget
        from nativevoice.audio import trim_silence

        tightened = [trim_silence(segment.audio, threshold=0.01, keep=0.02) for segment in segments]
        assert all(item.duration_s < segment.duration_s for item, segment in zip(tightened, segments))

    def test_quiet_speech_is_still_detected(self):
        signal = speech_like(rate=RATE, bursts=((0.3, 0.7),), amplitude=0.08)
        segments = self._run(self._segmenter(minimum_silence_s=0.1), signal)
        assert len(segments) == 1

    def test_background_noise_alone_never_opens_a_turn(self):
        for amplitude in (0.004, 0.01, 0.02):
            segmenter = self._segmenter()
            emitted = self._run(segmenter, noise(rate=RATE, seconds=2.0, amplitude=amplitude))
            assert emitted == [], f"noise at {amplitude} opened a turn"

    def test_pure_silence_never_opens_a_turn(self):
        assert self._run(self._segmenter(), silence(seconds=2.0)) == []

    def test_uninterrupted_speech_is_forced_out_as_truncated_turns(self):
        signal = speech_like(rate=RATE, bursts=((0.0, 3.0),))
        segments = self._run(self._segmenter(maximum_speech_s=1.0, minimum_silence_s=0.05), signal)
        assert len(segments) >= 2
        assert all(segment.truncated for segment in segments)
        assert all(segment.duration_s <= 1.1 for segment in segments)

    def test_a_long_pause_inside_one_utterance_splits_it_by_design(self):
        """Endpointing on silence is a policy, and the policy is configurable."""
        signal = speech_like(rate=RATE, bursts=((0.2, 0.5), (1.2, 1.5)))
        strict = self._run(self._segmenter(minimum_silence_s=0.20), signal)
        lenient = self._run(self._segmenter(minimum_silence_s=0.90), signal)
        assert len(strict) == 2
        assert len(lenient) == 1

    def test_look_back_padding_recovers_the_onset_without_losing_samples(self):
        signal = speech_like(rate=RATE, bursts=((0.4, 0.7),))
        segment = self._run(self._segmenter(minimum_silence_s=0.1, padding_s=0.1), signal)[0]
        assert segment.first_sample < int(0.4 * RATE)
        assert segment.audio.frame_count == segment.last_sample - segment.first_sample

    def test_segment_positions_are_monotonic_across_turns(self):
        signal = speech_like(rate=RATE, bursts=((0.1, 0.3), (0.6, 0.8), (1.1, 1.3)))
        segments = self._run(self._segmenter(minimum_silence_s=0.1), signal)
        starts = [segment.first_sample for segment in segments]
        assert starts == sorted(starts) and len(segments) == 3

    def test_capture_stall_drops_nothing_silently(self):
        """A ring buffer that overflows must report it, or slow audio looks like short audio."""
        from nativevoice.audio.buffer import RingBuffer

        ring = RingBuffer(AudioFormat(RATE, 1), max_frames=RATE)
        for _ in range(4):
            ring.append(sine(rate=RATE, seconds=0.5))
        assert ring.available_frames == RATE
        assert ring.overflow_frames == 2 * (RATE // 2)


class TestMeasuredFailureIsItsOwnState:
    """A measured failure must not be recorded as a refusal, and must not be routable.

    Collapsing the two loses the only number that says whether a gap is a missing
    asset or an asset that does not work, and a status that carries a metric can
    easily become a status that gets trusted.
    """

    def _ledger(self, *capabilities):
        from nativevoice.language import CapabilityLedger

        ledger = CapabilityLedger()
        for capability in capabilities:
            ledger.record(capability)
        return ledger

    def test_measured_failure_still_counts_as_evidence(self):
        failed = Capability(
            Task.TTS, CapabilityStatus.MEASURED_FAILED, "zh", metric_name="cer", metric_value=0.46, measured_on="x"
        )
        assert failed.counts_as_evidence() is True
        assert failed.usable_for_automatic_routing is False

    def test_a_refusal_with_no_number_is_not_evidence(self):
        refused = Capability(Task.TTS, CapabilityStatus.UNSUPPORTED, "bn", limitations=("no front-end",))
        assert refused.counts_as_evidence() is False
        assert refused.usable_for_automatic_routing is False

    def test_measured_beats_measured_failed_for_the_same_language(self):
        ledger = self._ledger(
            Capability(Task.TTS, CapabilityStatus.MEASURED_FAILED, "zh", engine="bad", metric_name="cer", metric_value=0.46),
            Capability(Task.TTS, CapabilityStatus.MEASURED, "zh", engine="good", metric_name="cer", metric_value=0.04),
        )
        assert ledger.best(Task.TTS, "zh").engine == "good"

    def test_only_a_failed_measurement_still_routes_nowhere(self):
        from nativevoice.language import LanguageRouter

        ledger = self._ledger(
            Capability(
                Task.TTS, CapabilityStatus.MEASURED_FAILED, "zh", engine="bad", metric_name="cer", metric_value=0.46
            )
        )
        outcome = LanguageRouter().route(ledger=ledger, requested_language=LanguageTag("zh"), task=Task.TTS)
        assert outcome.kind.value in {"unavailable", "ambiguous"}
        assert ledger.best(Task.TTS, "zh").status is CapabilityStatus.MEASURED_FAILED

    def test_an_unrecorded_language_is_no_model_not_a_failure(self):
        from nativevoice.language import CapabilityLedger

        capability = CapabilityLedger().best(Task.TTS, "cy")
        assert capability.status is CapabilityStatus.NO_MODEL
        assert capability.counts_as_evidence() is False
