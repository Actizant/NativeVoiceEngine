"""Contract tests for the shared core types: no engine required."""

from __future__ import annotations

import array
import threading

import pytest

from fixtures import sine
from nativevoice.core.audio import AudioBuffer, AudioFormat, Recognition
from nativevoice.core.cancel import CancellationToken, guarded
from nativevoice.core.errors import (
    CapabilityUnavailable,
    ConfigurationError,
    ModelNotApproved,
    ResourceLimitExceeded,
)
from nativevoice.core.events import SynthesisEvent, SynthesisKind, TranscriptionEvent, TranscriptionKind
from nativevoice.core.language import (
    Capability,
    CapabilityStatus,
    LanguageTag,
    Task,
)
from nativevoice.core.time import Timings, new_id


class TestAudioFormat:
    def test_rejects_unsupported_width(self):
        with pytest.raises(ConfigurationError):
            AudioFormat(16000, 1, 24)

    def test_rejects_nonsense_rate_and_channels(self):
        with pytest.raises(ConfigurationError):
            AudioFormat(0, 1)
        with pytest.raises(ConfigurationError):
            AudioFormat(16000, 0)

    def test_bytes_per_frame_tracks_channels(self):
        assert AudioFormat(48000, 2).bytes_per_frame == 4

    def test_allowed_preset_marks_uncommon_rate(self):
        assert AudioFormat(16000).is_allowed_preset()
        assert not AudioFormat(192000).is_allowed_preset()


class TestAudioBuffer:
    def test_frame_math_is_exact(self):
        buffer = sine(rate=16000, seconds=0.1)
        assert buffer.frame_count == 1600
        assert pytest.approx(buffer.duration_s, abs=1e-9) == 0.1

    def test_partial_frame_is_rejected_not_silently_truncated(self):
        AudioBuffer(array.array("h", [0, 1, 2]), AudioFormat(16000, 1))
        with pytest.raises(ConfigurationError):
            AudioBuffer(array.array("h", [0, 1, 2]), AudioFormat(16000, 2))
        with pytest.raises(ConfigurationError):
            AudioBuffer.from_bytes(b"\x00\x01\x02", AudioFormat(16000, 2))

    def test_stereo_downmix_averages_channels(self):
        left_right = array.array("h", [100, -100, 200, -200])
        buffer = AudioBuffer(left_right, AudioFormat(16000, 2))
        mono = buffer.to_mono()
        assert list(mono.samples) == [0, 0]
        assert mono.format.channels == 1

    def test_slicing_is_clamped_to_available_frames(self):
        buffer = sine(seconds=0.1)
        assert buffer.sliced(-5, 10 ** 6).frame_count == buffer.frame_count
        assert buffer.sliced(500, 100).frame_count == 0

    def test_frames_iterates_a_short_final_chunk(self):
        buffer = sine(seconds=0.1)
        chunks = list(buffer.frames(700))
        assert [chunk.frame_count for chunk in chunks] == [700, 700, 200]
        assert sum(chunk.frame_count for chunk in chunks) == buffer.frame_count

    def test_append_requires_matching_format(self):
        with pytest.raises(ConfigurationError):
            sine(rate=16000).appended(sine(rate=24000))

    def test_bytes_roundtrip_preserves_samples(self):
        buffer = sine(seconds=0.05, channels=2)
        restored = AudioBuffer.from_bytes(buffer.to_bytes(), buffer.format)
        assert list(restored.samples) == list(buffer.samples)

    def test_silence_detection(self):
        assert AudioBuffer.zeros(AudioFormat(16000, 1), 320).is_silent()
        assert not sine().is_silent()


class TestCancellation:
    def test_cleanups_run_once_in_reverse_order(self):
        token = CancellationToken()
        order = []
        token.on_cleanup(lambda: order.append("first"))
        token.on_cleanup(lambda: order.append("second"))
        token.cancel("user barged in")
        token.cancel("again")
        assert order == ["second", "first"]

    def test_cleanup_registered_after_cancel_fires_immediately(self):
        token = CancellationToken()
        token.cancel()
        fired = []
        token.on_cleanup(lambda: fired.append(True))
        assert fired == [True]

    def test_guard_raises_at_the_yield_boundary(self):
        token = CancellationToken()
        items = [1, 2, 3]

        def consume():
            for value in guarded(items, token):
                if value == 2:
                    token.cancel("barge-in")
                yield value

        with pytest.raises(Exception) as caught:
            list(consume())
        assert "cancelled" in str(caught.value)

    def test_children_are_cancelled_with_the_parent(self):
        parent = CancellationToken()
        child = CancellationToken()
        parent.link(child)
        parent.cancel("turn over")
        assert child.cancelled and child.reason.startswith("parent:")

    def test_pre_cancelled_parent_cancels_a_later_child(self):
        parent = CancellationToken()
        parent.cancel("stop")
        child = CancellationToken()
        parent.link(child)
        assert child.cancelled

    def test_sleep_is_interruptible(self):
        token = CancellationToken()
        threading.Timer(0.01, lambda: token.cancel("interrupted")).start()
        assert token.sleep(5.0) is True


class TestLanguageModel:
    def test_parsing_plain_and_regional_tags(self):
        assert LanguageTag.parse("en") == LanguageTag("en")
        assert LanguageTag.parse("EN-IN") == LanguageTag("en", region="IN")
        assert LanguageTag.parse("bn-Beng-IN").bcp47 == "bn-Beng-IN"

    def test_rejects_empty_tag(self):
        with pytest.raises(ConfigurationError):
            LanguageTag("")

    def test_capability_usability_rule(self):
        def status(value: CapabilityStatus) -> bool:
            return Capability(Task.STT, value, "en").usable_for_automatic_routing

        assert status(CapabilityStatus.MEASURED)
        assert status(CapabilityStatus.PARTIAL)
        assert not status(CapabilityStatus.DECLARED)
        assert not status(CapabilityStatus.NO_MODEL)
        assert not status(CapabilityStatus.UNSUPPORTED)

    def test_declared_without_measurement_is_not_evidence(self):
        declared = Capability(Task.STT, CapabilityStatus.DECLARED, "en", evidence="model card")
        assert not declared.counts_as_evidence()
        measured = Capability(
            Task.STT, CapabilityStatus.MEASURED, "en", metric_name="wer", metric_value=0.11, evidence="run:1"
        )
        assert measured.counts_as_evidence()

    def test_error_categories_are_distinguishable(self):
        assert issubclass(ModelNotApproved, CapabilityUnavailable)
        message = str(ModelNotApproved("blocked", licence="unreviewed"))
        assert "licence=" in message and "unreviewed" in message
        error = ResourceLimitExceeded("too big", explanation="needs 4096 MB, only 512 MB free", needs_mb=4096)
        # the sentence a user can act on must be in the rendered message, not only in details
        assert str(error) == "too big: needs 4096 MB, only 512 MB free (needs_mb=4096)"
        assert error.details["needs_mb"] == 4096
        assert ResourceLimitExceeded("bare").__str__() == "bare"


class TestEvents:
    def test_transcription_kinds_are_named_separately(self):
        assert TranscriptionEvent.retract("u1").kind is TranscriptionKind.RETRACT
        assert TranscriptionEvent.final("hi", "u1").is_final
        assert not TranscriptionEvent.partial("hi", "u1").is_final

    def test_from_recognition_carries_confidence_and_metadata(self):
        event = TranscriptionEvent.from_recognition(
            Recognition("hello", language="en", confidence=0.8, metadata={"engine": "x"}), "u1", TranscriptionKind.FINAL
        )
        assert (event.text, event.language, event.confidence, event.metadata["engine"]) == ("hello", "en", 0.8, "x")

    def test_synthesis_terminal_kinds(self):
        for kind in (SynthesisKind.EOS, SynthesisKind.CANCELLED, SynthesisKind.FAILED):
            assert SynthesisEvent(kind, 3).terminal
        assert not SynthesisEvent(SynthesisKind.AUDIO, 3).terminal


class TestTimings:
    def test_first_result_is_recorded_once(self):
        timings = Timings()
        timings.mark_first_result()
        recorded = timings.first_result_ns
        timings.mark_first_result()
        assert timings.first_result_ns == recorded

    def test_real_time_factor_needs_audio_duration(self):
        timings = Timings(audio_duration_s=0.0)
        assert timings.real_time_factor is None
        timings.audio_duration_s = 1.0
        timings.finish()
        assert 0 < timings.real_time_factor < 1

    def test_as_dict_exposes_the_four_reported_fields(self):
        payload = Timings(audio_duration_s=0.5).finish().as_dict()
        assert set(payload) == {"total_ms", "first_result_ms", "audio_duration_s", "real_time_factor"}

    def test_ids_are_unique_and_prefixed(self):
        generated = {new_id("utt") for _ in range(500)}
        assert len(generated) == 500
        assert all(value.startswith("utt_") for value in generated)


class TestResourceRequirement:
    def test_as_dict_is_a_lossless_view_of_every_field(self):
        """A measurement-provenance field dropped by the serializer would silently
        turn a measured number into an unsourced one."""
        from dataclasses import fields

        from nativevoice.models.api import ResourceRequirement

        requirement = ResourceRequirement(ram_mb=1200, measured=True, measured_on="host:7", rtf_cpu=0.42)
        payload = requirement.as_dict()
        declared = {field.name for field in fields(ResourceRequirement)}
        assert declared <= set(payload), sorted(declared - set(payload))
        assert payload["measured_on"] == "host:7"
