"""Tests for the log-mel frontend (`nativevoice.audio.features`).

Every assertion here is hand-checkable arithmetic on a deterministic signal, in the
spirit of `tests/test_audio.py`. **None of it demonstrates anything about speech
recognition.** The signals are tones, DC and envelopes — `tests/fixtures.py` says so
in its own words — so these tests prove that the frontend computes what it documents,
and prove nothing whatsoever about accuracy on real audio. Accuracy is a measurement
question and stays `NOT RUN` until approved real speech and a trained model both
exist.

Reference status, stated rather than implied:
* the DFT is cross-checked against an independent naive O(N²) transform written here,
  and — if `numpy` happens to be importable — against `numpy.fft.rfft`. Both check
  *arithmetic*.
* **mel scale, filterbank shape and log-floor behaviour are NOT verified against any
  external speech frontend** (Kaldi, librosa, torchaudio, Whisper's own filters). None
  of those is installed, and installing or fetching one is out of scope. Parity with a
  trusted speech toolkit remains an open item, recorded in `docs/asr-architecture.md`.
"""

from __future__ import annotations

import array
import ast
import cmath
import math
import re
from pathlib import Path

import pytest

from nativevoice.audio.features import (
    FBankConfig,
    FeatureGrid,
    MelFilter,
    center_frequencies,
    log_fbank,
    mel_filters,
    power_spectrum,
    window_function,
)
from nativevoice.core.audio import AudioBuffer, AudioFormat
from nativevoice.core.errors import ConfigurationError

from fixtures import noise, silence, sine, speech_like

RATE = 16000
LOG_FLOOR = 1e-10
NATURAL_LOG_OF_FLOOR = math.log(LOG_FLOOR)  # -23.025850929940457, natural log, not log10


def tone(frequency: float, seconds: float = 0.5, amplitude: float = 0.3, rate: int = RATE) -> AudioBuffer:
    count = int(rate * seconds)
    samples = array.array(
        "h", (int(amplitude * 32767.0 * math.sin(2.0 * math.pi * frequency * index / rate)) for index in range(count))
    )
    return AudioBuffer(samples, AudioFormat(rate, 1))


def small_config(**overrides) -> FBankConfig:
    """A config whose FFT is small enough for a cheap hand-checkable frame."""
    base = {"n_mels": 8, "fft_size": 64, "window_ms": 2.0, "hop_ms": 1.0}
    base.update(overrides)
    return FBankConfig(**base)


class TestDocumentedGeometry:
    def test_defaults_are_the_stated_frame_and_hop_sizes(self):
        config = FBankConfig()
        assert config.sample_rate == 16000
        assert config.window_samples == 400, "25 ms at 16 kHz"
        assert config.hop_samples == 160, "10 ms at 16 kHz"
        assert config.fft_size == 1024
        assert config.n_mels == 80
        assert config.bin_count == 513
        assert config.freq_max == 8000.0
        assert config.f_min == 20.0

    def test_one_second_of_audio_yields_exactly_ninety_eight_frames(self):
        config = FBankConfig()
        # (16000 - 400) // 160 + 1 = 98, with no padding at either end.
        assert config.frame_count(16000) == 98
        assert log_fbank(sine(seconds=1.0, frequency=440.0), config).frames == 98

    def test_the_no_padding_boundary_is_exact_on_both_sides(self):
        config = FBankConfig()
        assert config.frame_count(399) == 0
        assert config.frame_count(400) == 1
        assert config.frame_count(401) == 1
        assert config.frame_count(559) == 1
        assert config.frame_count(560) == 2
        assert config.frame_count(719) == 2
        assert config.frame_count(720) == 3
        assert config.frame_count(1) == 0

    def test_a_shorter_than_window_buffer_is_empty_not_erroring(self):
        grid = log_fbank(tone(440.0, seconds=0.01))
        assert grid.frames == 0
        assert grid.is_empty is True
        assert len(grid.values) == 0

    def test_an_empty_buffer_is_refused(self):
        empty = AudioBuffer(array.array("h"), AudioFormat(RATE, 1))
        with pytest.raises(ConfigurationError):
            log_fbank(empty)

    def test_frame_time_and_extent_use_the_documented_unpadded_grid(self):
        config = FBankConfig()
        grid = log_fbank(tone(1000.0, seconds=0.5), config)
        assert grid.frame_time_ms(0) == 0.0
        assert grid.frame_time_ms(3) == pytest.approx(3 * 160 * 1000.0 / 16000.0)
        assert grid.extent_s(3) == pytest.approx((0.03, 0.055))

    def test_a_window_exactly_as_long_as_the_transform_is_supported(self):
        # Exercises the branch where no zero padding happens at all: window_samples == fft_size.
        config = FBankConfig(sample_rate=8000, fft_size=64, window_ms=8.0, hop_ms=4.0, n_mels=4)
        assert config.window_samples == config.fft_size == 64
        buffer = AudioBuffer(array.array("h", (int(6000 * math.sin(2 * math.pi * 1000 * i / 8000)) for i in range(8000))), AudioFormat(8000, 1))
        grid = log_fbank(buffer, config)
        # 1 s at 8 kHz: 1 + (8000 - 64)//32 = 249 frames.
        assert grid.frames == 249 and len(grid.values) == 249 * 4

    def test_contiguous_framing_when_the_hop_equals_the_window(self):
        config = FBankConfig(window_ms=20.0, hop_ms=20.0, fft_size=512, n_mels=24)
        assert config.window_samples == config.hop_samples == 320
        assert config.frame_count(8000) == 25

    def test_a_hop_longer_than_the_window_skips_audio_rather_than_reusing_it(self):
        # Sparse framing is legal and must be honest about it: frame 1 starts at 25 ms and
        # covers only its own 10 ms, so samples in between are never seen by any frame.
        config = FBankConfig(window_ms=10.0, hop_ms=25.0, fft_size=512, n_mels=24)
        buffer = AudioBuffer(array.array("h", (0 for _ in range(8000))), AudioFormat(RATE, 1))
        grid = log_fbank(buffer, config)
        assert config.window_samples == 160 and config.hop_samples == 400
        assert grid.frames == config.frame_count(8000) == 20
        assert grid.extent_s(1) == pytest.approx((0.025, 0.035))
        assert grid.frame_time_ms(1) == pytest.approx(25.0)

    def test_samples_between_sparse_frames_are_simply_not_observed(self):
        # The counterpart of the test above, so the skip is a measured property and not an
        # accident: an energy burst placed strictly between two frame windows contributes
        # nothing, which is what "no padding, no interpolation" costs.
        config = FBankConfig(window_ms=10.0, hop_ms=25.0, fft_size=512, n_mels=24)
        quiet = array.array("h", (0 for _ in range(8000)))
        # hop 400, window 160 -> samples 200..300 sit inside frame 0 (0..160)? no: frame 0
        # covers 0..160, frame 1 covers 400..560, so 200..390 are never observed.
        burst = array.array("h", (20000 if 200 <= index < 390 else 0 for index in range(8000)))
        grid_quiet = log_fbank(AudioBuffer(quiet, AudioFormat(RATE, 1)), config)
        grid_burst = log_fbank(AudioBuffer(burst, AudioFormat(RATE, 1)), config)
        assert grid_burst.values == grid_quiet.values, "an unobserved burst must not change the grid"
        # Positive control, so the equality above is not a vacuous "nothing ever changes":
        # the same burst placed inside frame 1 (400..560) must move that frame's bins.
        inside = array.array("h", (20000 if 400 <= index < 550 else 0 for index in range(8000)))
        grid_inside = log_fbank(AudioBuffer(inside, AudioFormat(RATE, 1)), config)
        assert grid_inside.row(1) != grid_quiet.row(1)
        assert max(grid_inside.row(1)) > max(grid_quiet.row(1))


class TestOutputShapeAndContract:
    def test_values_length_is_frames_times_n_mels(self):
        config = FBankConfig()
        grid = log_fbank(tone(1000.0, seconds=0.5), config)
        assert len(grid.values) == grid.frames * config.n_mels == 48 * 80

    def test_values_are_stored_in_float64_array(self):
        grid = log_fbank(tone(440.0, seconds=0.2))
        assert isinstance(grid.values, array.array)
        assert grid.values.typecode == "d"
        assert isinstance(grid, FeatureGrid)

    def test_row_and_value_agree_and_bounds_are_refused(self):
        grid = log_fbank(tone(1000.0, seconds=0.5))
        row = grid.row(3)
        assert len(row) == grid.n_mels == 80
        assert row[7] == grid.value(3, 7)
        with pytest.raises(ConfigurationError):
            grid.value(3, 80)
        with pytest.raises(ConfigurationError):
            grid.value(grid.frames, 0)
        with pytest.raises(ConfigurationError):
            grid.row(-1)

    def test_filter_count_and_public_helpers_agree_with_the_config(self):
        config = FBankConfig(n_mels=40)
        filters = mel_filters(config)
        assert len(filters) == 40
        assert center_frequencies(config) == tuple(f.center_hz for f in filters)
        assert all(isinstance(f, MelFilter) for f in filters)


class TestDeterminismAndLinearity:
    def test_the_same_input_produces_identical_values_twice(self):
        buffer = speech_like()
        first = log_fbank(buffer)
        second = log_fbank(buffer)
        assert first.values == second.values
        assert first.values is not second.values, "no shared mutable state between calls"

    def test_extraction_is_repeatable_across_two_instantiations_of_the_config(self):
        buffer = noise(seconds=0.2, seed=99)
        assert log_fbank(buffer, FBankConfig()).values == log_fbank(buffer, FBankConfig()).values

    def test_doubling_sample_amplitude_raises_every_bin_by_exactly_ln_four(self):
        # The frontend is linear in amplitude, so energy scales by 4 and every log bin
        # shifts by ln(4). Integer doubling is exact, which makes this hand-checkable.
        base = tone(1000.0, seconds=0.3, amplitude=0.3)
        assert max(abs(v) for v in base.samples) <= 16383, "doubling must stay in int16 range"
        doubled = AudioBuffer(array.array("h", (v * 2 for v in base.samples)), base.format)
        low = log_fbank(base).row(3)
        high = log_fbank(doubled).row(3)
        expected = math.log(4.0)
        assert all(abs(h - l - expected) < 1e-9 for l, h in zip(low, high))

    def test_the_same_tone_at_double_rate_is_refused_rather_than_resampled(self):
        with pytest.raises(ConfigurationError) as caught:
            log_fbank(tone(1000.0, seconds=0.2, rate=32000))
        assert "resample" in str(caught.value)

    def test_stereo_is_downmixed_the_way_the_orchestrator_downmixes(self):
        mono = tone(1000.0, seconds=0.2)
        both = array.array("h")
        for value in mono.samples:
            both.append(value)
            both.append(value)
        stereo = AudioBuffer(both, AudioFormat(RATE, 2))
        assert log_fbank(stereo).values == log_fbank(mono).values


class TestSilenceDCAndToneBehaviour:
    def test_silence_is_exactly_the_log_floor_in_every_bin_and_never_negative_infinity(self):
        grid = log_fbank(silence(seconds=0.3))
        assert grid.frames == 28
        assert all(value == NATURAL_LOG_OF_FLOOR for value in grid.values)
        assert all(math.isfinite(value) for value in grid.values)

    def test_the_floor_is_natural_log_and_not_base_ten(self):
        assert NATURAL_LOG_OF_FLOOR != math.log10(LOG_FLOOR)
        assert log_fbank(silence(seconds=0.05)).values[0] == NATURAL_LOG_OF_FLOOR

    def test_no_fixture_produces_a_non_finite_feature(self):
        for buffer in (silence(seconds=0.2), noise(seconds=0.2), speech_like(), tone(37.0), tone(7999.0)):
            assert all(math.isfinite(value) for value in log_fbank(buffer).values), buffer.source

    def test_a_one_khz_tone_peaks_in_the_filter_centred_on_it(self):
        config = FBankConfig()
        filters = mel_filters(config)
        grid = log_fbank(tone(1000.0, seconds=0.5), config)
        row = grid.row(grid.frames // 2)
        peak = row.index(max(row))
        assert abs(filters[peak].center_hz - 1000.0) < 60.0, "filter centres are ~100 Hz apart here"

    def test_energy_falls_away_from_the_peak_filter_on_both_sides(self):
        config = FBankConfig()
        grid = log_fbank(tone(1000.0, seconds=0.5), config)
        row = grid.row(grid.frames // 2)
        peak = row.index(max(row))
        assert all(row[peak - d] > row[peak - d - 1] for d in range(1, 4))
        assert all(row[peak + d] > row[peak + d + 1] for d in range(1, 4))

    def test_pure_dc_concentrates_in_the_lowest_filter_and_not_in_the_high_end(self):
        # f_min=20 Hz means DC is not a passband, but a windowed DC step still leaks;
        # what must hold is that the leak falls with frequency, not that it is zero.
        dc = AudioBuffer(array.array("h", (1000 for _ in range(RATE))), AudioFormat(RATE, 1))
        row = log_fbank(dc).row(0)
        assert row.index(max(row)) == 0
        assert sum(row[:5]) > sum(row[-5:])
        assert row[-1] < row[0] - 5.0

    def test_the_lowest_filter_center_is_at_least_f_min(self):
        config = FBankConfig()
        assert center_frequencies(config)[0] > config.f_min
        assert center_frequencies(config)[-1] < config.freq_max


class TestFilterbankGeometry:
    def test_filters_are_ordered_in_frequency_and_disjoint_by_center(self):
        config = FBankConfig()
        centers = center_frequencies(config)
        assert centers == tuple(sorted(centers))
        assert len(set(centers)) == len(centers)

    def test_every_filter_is_a_rising_then_falling_triangle_peaking_at_one(self):
        config = FBankConfig()
        for item in mel_filters(config):
            assert item.weights[0] == 0.0
            assert item.weights[-1] == 0.0
            assert max(item.weights) == 1.0
            assert all(0.0 <= weight <= 1.0 for weight in item.weights)
            middle = item.weights.index(1.0)
            assert all(b >= a for a, b in zip(item.weights[: middle + 1], item.weights[1 : middle + 1]))
            assert all(b <= a for a, b in zip(item.weights[middle:], item.weights[middle + 1 :]))

    def test_no_filter_reaches_beyond_the_last_dft_bin(self):
        config = FBankConfig()
        assert all(item.last_bin <= config.bin_count - 1 for item in mel_filters(config))

    def test_the_default_config_builds_eighty_filters_without_collapsing(self):
        # Regression guard for the geometry defect found while writing this frontend:
        # at fft_size=512 the 25 ms window gives 31.25 Hz per bin and three of the
        # eighty low-frequency triangles collapse onto one bin. The refusal below is
        # what catches that; the default must never reach it.
        assert len(mel_filters(FBankConfig())) == 80
        with pytest.raises(ConfigurationError) as caught:
            mel_filters(FBankConfig(fft_size=512))
        assert "no width" in str(caught.value)

    def test_the_dc_bin_is_outside_the_first_filter(self):
        config = FBankConfig()
        assert mel_filters(config)[0].first_bin >= 1

    def test_coarser_grids_are_still_usable(self):
        config = FBankConfig(n_mels=40, window_ms=20.0, hop_ms=10.0)
        assert len(mel_filters(config)) == 40
        # 0.2 s at 16 kHz is 3200 samples: 1 + (3200-320)//160 = 19 frames.
        assert log_fbank(tone(1000.0, seconds=0.2), config).frames == 19


class TestWindowFunctions:
    def test_povey_is_zero_at_both_ends_symmetric_and_bounded(self):
        config = small_config()
        values = window_function(config)
        assert len(values) == config.window_samples == 32
        assert values[0] == 0.0
        assert values[-1] == 0.0
        # An even-length window has no sample exactly at the centre, so the peak is just
        # under 1.0 rather than equal to it — asserted as a bound, which is the claim that
        # actually matters (no gain, no negative lobe).
        assert max(values) <= 1.0
        assert max(values) > 0.999
        assert all(abs(values[i] - values[len(values) - 1 - i]) < 1e-12 for i in range(len(values)))
        assert all(0.0 <= value <= 1.0 for value in values)

    def test_povey_reaches_unity_exactly_when_the_window_has_a_centre_sample(self):
        config = FBankConfig(window_ms=2.0625, hop_ms=1.0, n_mels=4, fft_size=128)
        assert config.window_samples == 33, "an odd-length window has a sample at the centre"
        values = window_function(config)
        assert values[0] == 0.0 and values[-1] == 0.0
        assert max(values) == pytest.approx(1.0, abs=1e-12)
        assert values[16] == max(values)

    def test_hamming_endpoints_are_the_hand_checkable_0p08(self):
        values = window_function(small_config(window="hamming"))
        assert values[0] == pytest.approx(0.54 - 0.46, abs=1e-15)
        assert values[-1] == pytest.approx(0.08, abs=1e-15)

    def test_hann_endpoints_are_zero(self):
        values = window_function(small_config(window="hann"))
        assert values[0] == 0.0
        assert values[-1] == pytest.approx(0.0, abs=1e-15)

    def test_a_one_sample_window_is_defined_not_divided_by_zero(self):
        assert window_function(FBankConfig(window_ms=0.0625, hop_ms=1.0, n_mels=4)) == (1.0,)


class TestSpectrumParity:
    """Arithmetic parity for the transform. This is not speech-frontend parity."""

    def test_fft_matches_an_independent_naive_transform(self):
        config = small_config()
        values = [math.sin(2.0 * math.pi * 3 * i / 64) + 0.3 * i / 64 for i in range(64)]
        naive = [
            sum(values[n] * cmath.exp(-2.0j * math.pi * k * n / 64) for n in range(64)) for k in range(config.bin_count)
        ]
        ours = power_spectrum(values, config)
        error = max(abs(computed - abs(reference) ** 2) for computed, reference in zip(ours, naive))
        scale = max(abs(reference) ** 2 for reference in naive)
        assert error / scale < 1e-12

    def test_parseval_holds_over_the_two_sided_spectrum(self):
        config = small_config()
        values = [((i * 37) % 61) / 61.0 - 0.5 for i in range(32)]
        spectrum = power_spectrum(values, config)
        two_sided = sum(spectrum) + sum(spectrum[k] for k in range(1, config.fft_size - config.bin_count + 1))
        assert two_sided == pytest.approx(config.fft_size * sum(v * v for v in values), rel=1e-12)

    def test_measure_magnitude_is_the_elementwise_root_of_measure_power(self):
        power = power_spectrum([math.cos(2.0 * math.pi * 5 * i / 64) for i in range(64)], small_config())
        magnitude = power_spectrum(
            [math.cos(2.0 * math.pi * 5 * i / 64) for i in range(64)], small_config(power="magnitude")
        )
        assert max(abs(m - math.sqrt(p)) for m, p in zip(magnitude, power)) == 0.0

    def test_a_frame_longer_than_the_transform_is_refused(self):
        with pytest.raises(ConfigurationError):
            power_spectrum([0.1] * 65, small_config())

    def test_numpy_reference_if_present(self):
        """Optional cross-check against a trusted numerical library, arithmetic only."""
        numpy = pytest.importorskip("numpy", reason="NOT RUN: numpy is not installed on this machine")
        config = small_config()
        values = [math.sin(2.0 * math.pi * 7 * i / 32) for i in range(32)]
        expected = numpy.abs(numpy.fft.rfft(values, n=config.fft_size)) ** 2
        ours = power_spectrum(values, config)
        assert max(abs(float(a) - float(b)) for a, b in zip(ours, expected)) < 1e-9


class TestGridLevelBehaviourAgainstDirectComputation:
    def test_a_grid_row_recomputes_from_the_spectrum_and_filters(self):
        config = small_config()
        buffer = tone(1000.0, seconds=0.1)
        grid = log_fbank(buffer, config)
        window = window_function(config)
        frame = [(buffer.samples[n] / 32768.0) * window[n] for n in range(config.window_samples)]
        spectrum = power_spectrum(frame, config)
        expected = []
        for mel_filter in mel_filters(config):
            energy = sum(weight * spectrum[mel_filter.first_bin + offset] for offset, weight in enumerate(mel_filter.weights))
            expected.append(math.log(energy if energy > config.log_floor else config.log_floor))
        actual = grid.row(0)
        assert len(actual) == len(expected) == config.n_mels
        # Tolerated rather than exact: summation order is an implementation detail, and a
        # last-bit difference is not a contract. 1e-12 nats is ~1e-7 in energy terms.
        assert max(abs(a - e) for a, e in zip(actual, expected)) < 1e-12


class TestConfigValidation:
    @pytest.mark.parametrize(
        "overrides,fragment",
        [
            ({"fft_size": 8}, "minimum transform length"),
            ({"center": "circular"}, "unknown centering"),
            ({"mel_scale": "fraud"}, "unknown mel scale"),
            ({"edge_mode": "jitter"}, "unknown mel edge mode"),
            ({"log": "log2"}, "unknown log base"),
            ({"dynamic_floor": 0.0}, "dynamic_floor"),
            ({"post_scale": (4.0, 0.0)}, "non-zero divisor"),
            ({"tail_pad_samples": -1}, "cannot be negative"),
            ({"center": "reflect", "window_ms": 25.0, "fft_size": 1024}, "window to equal the transform"),
            ({"mel_scale": "slaney", "f_min": 20.0, "f_max": None, "fft_size": 400, "window_ms": 25.0}, "spans 0 Hz to Nyquist"),
            ({"window": "kaiser"}, "unknown window"),
            ({"power": "energy"}, "unknown power"),
            ({"n_mels": 0}, "n_mels"),
            ({"log_floor": 0.0}, "log_floor"),
            ({"hop_ms": 0.0}, "positive"),
            ({"window_ms": -1.0}, "positive"),
            ({"f_min": -5.0}, "f_min"),
            ({"f_max": 20.0}, "Nyquist"),
            ({"sample_rate": 11025}, "sample_rate"),
            ({"window_ms": 100.0}, "no longer than the FFT"),
        ],
    )
    def test_an_impossible_config_is_refused_at_construction(self, overrides, fragment):
        with pytest.raises(ConfigurationError) as caught:
            FBankConfig(**overrides)
        assert fragment in str(caught.value)

    def test_f_min_above_f_max_is_refused(self):
        with pytest.raises(ConfigurationError):
            FBankConfig(f_min=9000.0, f_max=8000.0)


class TestSpectrumClosedFormValues:
    """Assertions computable on paper, independent of any helper in the module."""

    def test_an_impulse_has_flat_unit_spectrum(self):
        config = small_config()
        frame = [1.0] + [0.0] * (config.fft_size - 1)
        spectrum = power_spectrum(frame, config)
        assert len(spectrum) == config.bin_count
        assert all(value == pytest.approx(1.0, abs=1e-12) for value in spectrum)

    def test_an_all_zero_frame_has_an_all_zero_spectrum(self):
        config = small_config()
        assert power_spectrum([0.0] * config.fft_size, config) == tuple([0.0] * config.bin_count)

    def test_a_dc_frame_lands_entirely_in_bin_zero(self):
        config = small_config()
        spectrum = power_spectrum([1.0] * 64, config)
        assert spectrum[0] == pytest.approx(64.0 * 64.0, rel=1e-12)
        assert spectrum[1] == pytest.approx(0.0, abs=1e-12)
        assert max(spectrum[2:]) < 1e-12

    def test_a_bin_centred_cosine_peaks_in_its_own_bin_with_the_exact_predicted_power(self):
        # x[n] = cos(2*pi*5*n/64) has |X[5]| = |X[59]| = N/2 = 32, so power = 1024 and
        # every other bin is 0. This is the transform's definition, not a property of
        # the filterbank, and it is checked with no reference implementation at all.
        config = small_config()
        frame = [math.cos(2.0 * math.pi * 5 * n / 64) for n in range(64)]
        spectrum = power_spectrum(frame, config)
        assert spectrum.index(max(spectrum)) == 5
        assert spectrum[5] == pytest.approx(1024.0, rel=1e-9)
        for bin_index in (4, 6, 7, 20):
            assert spectrum[bin_index] == pytest.approx(0.0, abs=1e-9)

    def test_measure_magnitude_of_the_same_cosine_is_n_over_two(self):
        config = small_config(power="magnitude")
        frame = [math.cos(2.0 * math.pi * 5 * n / 64) for n in range(64)]
        spectrum = power_spectrum(frame, config)
        assert spectrum[5] == pytest.approx(32.0, rel=1e-9)

    def test_a_negative_amplitude_input_does_not_produce_negative_power(self):
        config = small_config()
        frame = [-math.cos(2.0 * math.pi * 5 * n / 64) for n in range(64)]
        assert all(value >= 0.0 for value in power_spectrum(frame, config))


class TestFilterbankCoverageAndRounding:
    def test_positive_weight_bins_are_contiguous_and_exclude_dc_and_nyquist(self):
        config = FBankConfig()
        weighted = {
            mel_filter.first_bin + offset
            for mel_filter in mel_filters(config)
            for offset, weight in enumerate(mel_filter.weights)
            if weight > 0.0
        }
        assert 0 not in weighted, "f_min=20 Hz must keep the DC bin out of every filter"
        assert config.bin_count - 1 not in weighted, "a triangle's right endpoint has zero weight"
        # Bin 1 is 15.625 Hz, below f_min=20 Hz, so it lies in the first filter's stop band
        # and gets no positive weight; the first positively-weighted bin is 2 (31.25 Hz).
        assert min(weighted) == 2
        assert max(weighted) == config.bin_count - 2
        assert weighted == set(range(min(weighted), max(weighted) + 1)), "no interior bin is unobserved"

    def test_no_interior_bin_is_between_two_filters(self):
        config = FBankConfig()
        for left, right in zip(mel_filters(config), mel_filters(config)[1:]):
            assert right.first_bin <= left.last_bin, "adjacent triangles must overlap, not leave a gap"

    def test_edge_rounding_does_not_move_the_visible_band_materially(self):
        # Snapping mel edges to FFT bins is the one approximation in the filterbank, so
        # its error is measured rather than waved through: a pure tone must be estimated
        # from the mel grid to within a small fraction of its own frequency.
        config = FBankConfig()
        centres = center_frequencies(config)
        for frequency, tolerance_hz in ((200.0, 12.0), (1000.0, 25.0), (5000.0, 60.0)):
            grid = log_fbank(tone(frequency, seconds=0.5), config)
            row = grid.row(grid.frames // 2)
            order = sorted(range(config.n_mels), key=lambda i: -row[i])[:5]
            weights = [math.exp(row[i]) for i in order]
            total = sum(weights)
            estimate = sum(w * centres[i] for w, i in zip(weights, order)) / total
            assert abs(estimate - frequency) <= tolerance_hz, f"{frequency} Hz estimated at {estimate:.1f}"

    def test_energy_increases_with_tone_amplitude_in_the_peak_band(self):
        config = FBankConfig()
        quiet = log_fbank(tone(1000.0, seconds=0.3, amplitude=0.05), config)
        loud = log_fbank(tone(1000.0, seconds=0.3, amplitude=0.30), config)
        peak = max(range(config.n_mels), key=lambda i: loud.row(1)[i])
        assert loud.row(1)[peak] > quiet.row(1)[peak]


class TestSampleRateGeometryContract:
    @pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000])
    def test_the_default_geometry_works_up_to_24khz(self, rate):
        config = FBankConfig(sample_rate=rate)
        assert config.window_samples <= config.fft_size
        assert len(mel_filters(config)) == config.n_mels

    @pytest.mark.parametrize("rate", [32000, 44100, 48000])
    def test_rates_above_24khz_are_refused_at_the_default_fft_size(self, rate):
        # Documented limit, not an accident: at 44.1/48 kHz a 25 ms window does not fit a
        # 1024-point transform at all, and at 32 kHz the 80-filter ladder collapses. The
        # two fail at different moments — 44.1/48 kHz in the constructor, 32 kHz only when
        # the filterbank is built — which is why both calls are inside the expectation.
        with pytest.raises(ConfigurationError) as caught:
            mel_filters(FBankConfig(sample_rate=rate))
        message = str(caught.value)
        assert ("fft_size" in message or "FFT" in message), message

    def test_a_larger_transform_makes_48khz_work(self):
        config = FBankConfig(sample_rate=48000, fft_size=2048)
        assert config.window_samples == 1200
        assert len(mel_filters(config)) == config.n_mels

    def test_window_and_hop_round_at_the_documented_half_up_rule(self):
        assert FBankConfig(sample_rate=22050).window_samples == 551, "25 ms = 551.25 samples, rounded"
        assert FBankConfig(sample_rate=22050).hop_samples == 221, "10 ms = 220.5 samples, half-up to 221"
        assert FBankConfig(sample_rate=44100, fft_size=2048).window_samples == 1103

    def test_the_frontend_uses_the_config_rate_and_never_assumes_16khz(self):
        config = FBankConfig(sample_rate=48000, fft_size=2048)
        samples = array.array("h", (int(3000 * math.sin(2 * math.pi * 440 * i / 48000)) for i in range(4800)))
        grid = log_fbank(AudioBuffer(samples, AudioFormat(48000, 1)), config)
        # 4800 samples, window 1200, hop 480: 1 + (4800-1200)//480 = 8 frames.
        assert config.window_samples == 1200 and config.hop_samples == 480
        assert grid.frames == config.frame_count(4800) == 8


class TestGridShapeIsEnforced:
    def test_a_grid_whose_values_do_not_match_its_frames_is_refused(self):
        # Review finding: row() used to return a silently short tuple for such a grid.
        with pytest.raises(ConfigurationError) as caught:
            FeatureGrid(FBankConfig(), 2, array.array("d", [0.0] * 10), AudioFormat(RATE, 1))
        assert "shape" in str(caught.value)

    def test_negative_frame_count_is_refused(self):
        with pytest.raises(ConfigurationError):
            FeatureGrid(FBankConfig(), -1, array.array("d"), AudioFormat(RATE, 1))

    def test_every_row_of_a_valid_grid_is_exactly_n_mels_long(self):
        grid = log_fbank(tone(1000.0, seconds=0.3))
        assert all(len(grid.row(f)) == grid.n_mels for f in range(grid.frames))
        assert len(grid.values) == grid.frames * grid.n_mels

    def test_an_empty_grid_refuses_row_zero(self):
        grid = log_fbank(tone(440.0, seconds=0.01))
        assert grid.frames == 0 and grid.is_empty
        with pytest.raises(ConfigurationError):
            grid.row(0)
        with pytest.raises(ConfigurationError):
            grid.value(0, 0)

    def test_the_values_buffer_is_mutable_by_design_and_says_so(self):
        # `frozen` protects the identity of the grid, not its bytes — the same convention
        # as AudioBuffer.samples. Pinned so nobody reads `frozen=True` as immutability.
        grid = log_fbank(tone(1000.0, seconds=0.2))
        original = grid.values[0]
        grid.values[0] = 0.0
        assert grid.values[0] == 0.0
        grid.values[0] = original


class TestDownmixIsTheDocumentedAveraging:
    def test_two_identical_channels_equal_the_mono_grid(self):
        mono = tone(1000.0, seconds=0.2)
        both = array.array("h")
        for value in mono.samples:
            both.append(value)
            both.append(value)
        stereo = AudioBuffer(both, AudioFormat(RATE, 2))
        assert log_fbank(stereo).values == log_fbank(mono).values

    def test_anti_phase_stereo_downmixes_to_silence_exactly(self):
        samples = [int(12000 * math.sin(2.0 * math.pi * 300 * i / RATE)) for i in range(2000)]
        interleaved = array.array("h")
        for value in samples:
            interleaved.append(value)
            interleaved.append(-value)
        stereo = AudioBuffer(interleaved, AudioFormat(RATE, 2))
        grid = log_fbank(stereo)
        silent = log_fbank(AudioBuffer.zeros(AudioFormat(RATE, 1), 2000))
        assert grid.frames == silent.frames == 11
        assert grid.values == silent.values
        assert grid.values[0] == NATURAL_LOG_OF_FLOOR

    def test_stereo_downmix_averages_and_does_not_take_the_left_channel(self):
        # L=2000, R=0 averages to 1000 by AudioBuffer.to_mono(); taking channel 0 would
        # double the energy and every log bin would shift by ln(4).
        mixed = AudioBuffer(array.array("h", [v for pair in zip([2000] * 2000, [0] * 2000) for v in pair]), AudioFormat(RATE, 2))
        reference = AudioBuffer(array.array("h", [1000] * 2000), AudioFormat(RATE, 1))
        assert log_fbank(mixed).values == log_fbank(reference).values


class TestFloorAndBandAreConfigurable:
    def test_a_custom_log_floor_is_applied_in_energy_units_before_the_log(self):
        config = FBankConfig(log_floor=1e-4)
        grid = log_fbank(silence(seconds=0.1), config)
        assert all(value == math.log(1e-4) for value in grid.values)
        assert grid.values[0] != math.log(1e-10)

    def test_an_explicit_f_max_keeps_every_filter_inside_the_band(self):
        config = FBankConfig(f_max=4000.0, n_mels=24)
        centres = center_frequencies(config)
        assert all(config.f_min < centre < 4000.0 for centre in centres)
        assert len(centres) == 24

    def test_the_two_measures_are_independent_not_a_scaled_copy(self):
        # sum(|X|) over a filter is not sqrt(sum(|X|^2)); if a reader assumed the two
        # measures differed only by a factor of two in the log, they would be wrong.
        power_grid = log_fbank(tone(1000.0, seconds=0.2), FBankConfig())
        magnitude_grid = log_fbank(tone(1000.0, seconds=0.2), FBankConfig(power="magnitude"))
        row_p, row_m = power_grid.row(1), magnitude_grid.row(1)
        assert any(abs(m - 0.5 * p) > 0.5 for p, m in zip(row_p, row_m))
        assert all(math.isfinite(value) for value in row_m)


class TestModuleHygiene:
    def test_the_frontend_imports_only_the_standard_library_and_this_package(self):
        source = Path(__file__).resolve().parents[1] / "src" / "nativevoice" / "audio" / "features.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module)
        assert imported <= {"__future__", "array", "math", "dataclasses", "typing"}, sorted(imported)

    def test_the_frontend_never_opens_a_file_or_touches_a_socket(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "nativevoice" / "audio" / "features.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("open(", "socket", "urlopen", "subprocess", "requests"):
            assert forbidden not in source, forbidden

    def test_the_frontend_claims_no_capability_and_reports_no_metric(self):
        # A frontend is not a model. It must not import the capability vocabulary or
        # emit a metric a later reader could cite as evidence of recognition quality;
        # the docstring is allowed to say in prose that it proves nothing.
        source = (Path(__file__).resolve().parents[1] / "src" / "nativevoice" / "audio" / "features.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("CapabilityStatus", "Capability", "MEASURED", "Task", "error_rate", "transcribe"):
            assert forbidden not in source, forbidden
        assert re.search(r"\bWER\b|\bwer\b", source) is None
