"""Tests for the audio transport layer. Everything here runs without hardware."""

from __future__ import annotations

import math

import pytest
from fixtures import noise, sine, silence, speech_like

from nativevoice.audio import (
    MODE_BAND_LIMITED,
    MODE_LINEAR,
    CapturingOutput,
    EnergySpeechDetector,
    Preprocessor,
    RingBuffer,
    SegmenterConfig,
    SyntheticInput,
    UtteranceSegmenter,
    WavFileInput,
    crossfade,
    fade_edges,
    inspect_wav,
    list_devices,
    measure,
    normalize_peak,
    peak,
    read_wav,
    remove_dc,
    resample,
    rms,
    trim_silence,
    write_wav,
)
from nativevoice.core.audio import AudioBuffer, AudioFormat
from nativevoice.core.cancel import CancellationToken
from nativevoice.core.errors import ConfigurationError, DeviceDisconnected, DeviceError, StorageError


class TestMeasurement:
    def test_rms_tracks_amplitude(self):
        assert rms(sine(amplitude=0.3).samples) == pytest.approx(0.3 / math.sqrt(2), rel=0.02)

    def test_peak_of_silence_is_zero(self):
        assert peak(silence().samples) == 0.0

    def test_saturated_run_is_reported_as_clipping(self):
        square = AudioBuffer(
            __import__("array").array("h", [32767 if index % 2 == 0 else -32768 for index in range(2000)]),
            AudioFormat(16000, 1),
        )
        assert measure(square).clipped_fraction > 0.9

    def test_silence_fraction_of_a_gated_signal(self):
        level = measure(speech_like(rate=16000, bursts=((0.2, 0.6),)))
        assert 0.2 < level.silence_fraction < 0.6
        assert level.rms > 0


class TestProcessing:
    def test_dc_blocker_centres_a_biased_capture(self, dc_offset_capture):
        assert abs(rms(dc_offset_capture.samples)) > 0.02
        cleaned = remove_dc(dc_offset_capture, coefficient=0.9)
        assert abs(rms(cleaned.samples)) < abs(rms(dc_offset_capture.samples))

    def test_normalize_reaches_target_without_exceeding_full_scale(self):
        buffer = normalize_peak(sine(amplitude=0.1, seconds=0.05), target=0.9)
        assert peak(buffer.samples) == pytest.approx(0.9, rel=0.02)
        assert max(buffer.samples) <= 32767

    def test_normalize_can_attenuate_only(self):
        louder = normalize_peak(sine(amplitude=0.001, seconds=0.05), target=0.9, amplify=False)
        assert peak(louder.samples) < 0.01

    def test_fade_edges_silence_the_boundaries(self):
        faded = fade_edges(sine(amplitude=0.5, seconds=0.1), fade_s=0.01)
        assert abs(faded.samples[0]) < 200
        assert abs(faded.samples[-1]) < 200
        assert peak(faded.samples[400:-400]) > 0.4

    def test_crossfade_length_and_continuity(self):
        left = sine(seconds=0.1, frequency=200, amplitude=0.2)
        right = sine(seconds=0.1, frequency=800, amplitude=0.2)
        overlap = int(0.02 * 16000)
        merged = crossfade(left, right, overlap_s=0.02)
        assert merged.frame_count == left.frame_count + right.frame_count - overlap
        jump = max(abs(merged.samples[i] - merged.samples[i - 1]) for i in range(overlap * 4 - 40, overlap * 4 + 40))
        assert jump < 3000


class TestResampling:
    @pytest.mark.parametrize(
        "source,target",
        [(48000, 16000), (44100, 16000), (16000, 24000), (22050, 44100), (8000, 48000)],
    )
    def test_duration_is_preserved(self, source, target):
        buffer = sine(rate=source, seconds=0.2)
        out = resample(buffer, target)
        assert out.format.sample_rate == target
        assert pytest.approx(out.duration_s, rel=0.01) == 0.2

    def test_band_limited_preserves_a_passband_tone(self):
        buffer = sine(rate=48000, seconds=0.3, frequency=500, amplitude=0.3)
        out = resample(buffer, 16000, mode=MODE_BAND_LIMITED)
        assert rms(out.samples) == pytest.approx(0.3 / math.sqrt(2), rel=0.15)

    def test_band_limited_attenuates_an_aliasing_tone(self):
        # 12 kHz at 48 kHz is above the 8 kHz target Nyquist, so it must be filtered out.
        buffer = sine(rate=48000, seconds=0.3, frequency=12000, amplitude=0.3)
        stopped = resample(buffer, 16000, mode=MODE_BAND_LIMITED, taps=8)
        assert rms(stopped.samples) < 0.02

    def test_linear_is_noisier_on_downsampling_by_design(self):
        buffer = sine(rate=48000, seconds=0.3, frequency=12000, amplitude=0.3)
        leaked = rms(resample(buffer, 16000, mode=MODE_LINEAR).samples)
        filtered = rms(resample(buffer, 16000, taps=8).samples)
        assert leaked > filtered * 5

    def test_stereo_resamples_both_channels(self):
        buffer = sine(rate=44100, seconds=0.05, channels=2)
        out = resample(buffer, 16000)
        assert out.format.channels == 2
        assert abs(rms(out.channel(0)) - rms(out.channel(1))) < 1e-6

    def test_same_rate_is_a_noop_and_bad_modes_are_rejected(self):
        buffer = sine(rate=16000, seconds=0.02)
        assert resample(buffer, 16000).samples is buffer.samples
        with pytest.raises(ConfigurationError):
            resample(buffer, 16000, mode="magic")
        with pytest.raises(ConfigurationError):
            resample(buffer, 0)


class TestRingBuffer:
    def test_overflow_drops_oldest_and_counts_it(self):
        ring = RingBuffer(AudioFormat(16000, 1), max_frames=1000)
        assert ring.append(sine(seconds=0.05)) == 800
        assert ring.available_frames == 800 and ring.overflow_frames == 0
        ring.append(sine(seconds=0.05))
        assert ring.available_frames == 1000
        assert ring.overflow_frames == 600
        ring.append(sine(seconds=0.05))
        assert ring.overflow_frames == 1400
        # what survives is the newest audio, not the oldest
        tail = ring.peek_frames(1000)
        assert tail.frame_count == 1000
        assert tail.samples[-1] == sine(seconds=0.05).samples[-1]

    def test_drain_advances_and_clear_resets(self):
        ring = RingBuffer(AudioFormat(16000, 1), max_frames=5000)
        ring.append(sine(seconds=0.1))
        assert ring.drain(800).frame_count == 800
        assert ring.available_frames == 800
        ring.clear()
        assert ring.available_frames == 0

    def test_format_mismatch_is_fatal(self):
        ring = RingBuffer(AudioFormat(16000, 1), max_frames=100)
        with pytest.raises(ConfigurationError):
            ring.append(sine(rate=24000, seconds=0.01))

    def test_staleness_is_visible_to_the_session(self):
        ring = RingBuffer(AudioFormat(16000, 1), max_frames=16000)
        assert ring.available_frames == 0
        ring.append(sine(seconds=0.2))
        assert 0.15 < ring.oldest_age_s <= 0.21


class TestVoiceActivity:
    def _segmenter(self, rate: int = 16000, **config) -> UtteranceSegmenter:
        return UtteranceSegmenter(
            EnergySpeechDetector(format=AudioFormat(rate, 1)),
            SegmenterConfig(**config),
        )

    def test_two_bursts_become_two_segments(self):
        segmenter = self._segmenter(minimum_silence_s=0.05, padding_s=0.02)
        signal = speech_like(rate=16000, bursts=((0.1, 0.3), (0.6, 0.8)))
        segments = []
        for chunk in signal.frames(320):
            segments.extend(segmenter.push(chunk))
        segments.extend(segmenter.flush())
        assert len(segments) == 2
        assert all(0.1 < segment.duration_s < 0.35 for segment in segments)

    def test_pure_silence_produces_nothing(self):
        segmenter = self._segmenter()
        emitted = []
        for chunk in silence(seconds=0.5).frames(320):
            emitted.extend(segmenter.push(chunk))
        assert emitted == []

    def test_padding_recovers_the_onset(self):
        segmenter = self._segmenter(padding_s=0.05, minimum_silence_s=0.05)
        signal = speech_like(rate=16000, bursts=((0.2, 0.4),))
        collected = []
        for chunk in signal.frames(320):
            collected.extend(segmenter.push(chunk))
        collected.extend(segmenter.flush())
        assert len(collected) == 1
        onset = int(0.2 * 16000)
        # within one 320-sample frame of the true onset, and 50 ms of look-back kept
        assert onset - 800 - 320 <= collected[0].first_sample <= onset - 800 + 320
        assert collected[0].audio.frame_count > 0.2 * 16000

    def test_long_speech_is_forced_out_as_truncated(self):
        segmenter = self._segmenter(maximum_speech_s=0.2, minimum_silence_s=0.05)
        signal = speech_like(rate=16000, bursts=((0.0, 1.0),))
        collected = []
        for chunk in signal.frames(320):
            collected.extend(segmenter.push(chunk))
        assert collected and all(segment.truncated for segment in collected)

    def test_hysteresis_prevents_chatter(self):
        detector = EnergySpeechDetector(format=AudioFormat(16000, 1))
        scores = [detector.probe(frame) for frame in noise(amplitude=0.004, seconds=1.0).frames(320)]
        assert set(scores) == {0.0}

    def test_detector_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError):
            EnergySpeechDetector(format=AudioFormat(16000, 1), rise=0.01, fall=0.05)

    def test_cancelled_token_stops_segmentation(self):
        token = CancellationToken()
        segmenter = UtteranceSegmenter(
            EnergySpeechDetector(format=AudioFormat(16000, 1)), SegmenterConfig(), token=token
        )
        signal = speech_like(rate=16000, bursts=((0.0, 0.5),))
        chunks = list(signal.frames(320))
        token.cancel("push-to-talk released")
        assert segmenter.push(chunks[0]) == []


class TestPreprocessing:
    def test_capture_is_brought_to_canonical_form(self, dc_offset_capture):
        working, report = Preprocessor().process(dc_offset_capture)
        assert working.format == AudioFormat(16000, 1)
        assert "mono" not in report.stages or working.format.channels == 1
        assert "resample:band_limited" not in report.stages
        assert report.duration_s == pytest.approx(working.duration_s)

    def test_report_names_the_stages_that_ran(self):
        capture = sine(rate=44100, seconds=0.2, amplitude=0.05, channels=2)
        working, report = Preprocessor().process(capture)
        assert report.stages == ["mono", "dc_block", "normalize", "resample:band_limited"]
        assert working.format.sample_rate == 16000

    def test_trim_keeps_a_margin_around_speech(self):
        padded = silence(seconds=0.3).appended(speech_like(rate=16000, bursts=((0.0, 0.2),)))
        trimmed = trim_silence(padded, threshold=0.01, keep=0.05)
        assert 0.2 < trimmed.duration_s < 0.35


class TestWaveContainer:
    def test_roundtrip_preserves_pcm(self, tmp_path):
        original = sine(rate=16000, seconds=0.05, amplitude=0.4)
        path = write_wav(str(tmp_path / "tone.wav"), original)
        restored = read_wav(path)
        assert list(restored.samples) == list(original.samples)
        assert restored.format == original.format

    def test_stereo_roundtrip(self, tmp_path):
        original = sine(rate=44100, seconds=0.05, channels=2)
        path = write_wav(str(tmp_path / "stereo.wav"), original)
        assert inspect_wav(path).channels == 2
        assert list(read_wav(path).samples) == list(original.samples)

    def test_non_wave_payload_is_rejected(self, tmp_path):
        bogus = tmp_path / "bogus.wav"
        bogus.write_bytes(b"not a wave file at all")
        with pytest.raises(StorageError):
            read_wav(str(bogus))

    def test_unknown_codec_is_rejected_with_a_reason(self, tmp_path):
        # RIFF/WAVE header claiming IEEE float 24-bit, which the reader refuses.
        header = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
        header += (0xFFFE).to_bytes(2, "little") + (1).to_bytes(2, "little") + (16000).to_bytes(4, "little")
        header += (32000).to_bytes(4, "little") + (2).to_bytes(2, "little") + (24).to_bytes(2, "little")
        header += b"data" + (0).to_bytes(4, "little")
        path = tmp_path / "weird.wav"
        path.write_bytes(header)
        with pytest.raises(ConfigurationError, match="unsupported WAVE encoding"):
            read_wav(str(path))

    def test_writer_refuses_unsupported_layouts(self, tmp_path):
        with pytest.raises(ConfigurationError):
            write_wav(str(tmp_path / "x.wav"), sine(seconds=0.01, channels=4))
class TestInputs:
    def test_wav_input_resamples_and_chunks(self, tmp_path):
        path = write_wav(str(tmp_path / "in.wav"), sine(rate=44100, seconds=0.2, amplitude=0.3))
        source = WavFileInput(path=path, frame_size=1600, target_rate=16000)
        source.open()
        chunks = list(source.frames())
        assert all(chunk.format.sample_rate == 16000 for chunk in chunks)
        assert sum(chunk.frame_count for chunk in chunks) == 3200
        source.close()

    def test_wav_input_requires_open(self):
        with pytest.raises(DeviceError):
            list(WavFileInput(path="unused").frames())

    def test_synthetic_input_can_simulate_a_disconnected_device(self):
        source = SyntheticInput(buffers=[sine(seconds=0.05)] * 4, stall_after=2)
        source.open()
        iterator = source.frames()
        assert next(iterator) is not None
        assert next(iterator) is not None
        with pytest.raises(DeviceDisconnected):
            next(iterator)


class TestOutputs:
    def test_capturing_output_accumulates(self):
        sink = CapturingOutput()
        sink.play(sine(seconds=0.1))
        assert sink.frames_written == 1600
        assert sink.audio.duration_s == pytest.approx(0.1)

    def test_cancellation_mid_playback_emits_nothing_further(self):
        token = CancellationToken()
        sink = CapturingOutput()

        def cancel_later():
            token.cancel("barge-in")

        import threading

        timer = threading.Timer(0.0, cancel_later)
        timer.start()
        timer.join()
        played = sink.play(sine(seconds=0.4), token=token)
        assert played == 0
        assert sink.frames_written == 0
        assert sink.cancelled_spans == 1

    def test_stop_clears_the_current_span_and_later_spans_still_emit(self):
        sink = CapturingOutput()
        assert sink.play(sine(seconds=0.2)) == 3200
        sink.stop()
        assert sink.halts == 1 and sink.frames_written == 0 and sink.played == []
        assert sink.play(sine(seconds=0.1)) == 1600

def test_device_enumeration_never_raises():
    devices = list_devices()
    assert isinstance(devices, list)
