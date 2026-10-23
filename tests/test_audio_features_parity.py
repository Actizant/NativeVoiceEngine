"""Phase 1B: numerical parity between `nativevoice.audio.features` and the Whisper
feature extractor the planned teacher actually runs.

Two files, because the two claims have different standing:

* `TestParityContract` needs **no third-party package**. It pins the parity
  configuration, the mel-ladder constant, the frame arithmetic and the reflect-pad
  order — all hand-checkable, all runnable anywhere.
* `TestAgainstReference` imports `faster_whisper` and `numpy`. If either is absent it
  reports **NOT RUN** and parity is asserted nowhere else in this repository: the
  student geometry is a different contract and no arithmetic-only test may be read as
  teacher parity.

Tolerance, justified rather than tuned: the reference computes in float32 (periodic
Hanning cast to float32, complex64 STFT, float32 mel matrix) while this frontend
computes in float64. Measured worst case across every input below is about 1e-5 in the
reference's own post-normalised units, and the same reference algorithm restated in
float64 agrees with this implementation to ~5e-9. That second number is what
distinguishes "different precision" from "different formula"; 1e-4 leaves an order of
magnitude of margin over the former and would still catch the latter.
"""

from __future__ import annotations

import array
import cmath
import math

import pytest

from nativevoice.audio.features import (
    FBankConfig,
    _padded_sequence,
    mel_edges_hz,
    mel_filters,
    power_spectrum,
    slaney_hz_to_mel,
    slaney_mel_to_hz,
    log_fbank,
)
from nativevoice.core.audio import AudioBuffer, AudioFormat
from nativevoice.core.errors import ConfigurationError

RATE = 16000
#: The top of the reference mel ladder, hardcoded in `get_mel_filters` rather than
#: derived. It is exactly Slaney's hz_to_mel(8000), which is why deriving it is a
#: stronger contract than copying it.
REFERENCE_MAX_MEL = 45.245640471924965
TOLERANCE = 1e-4


def tone_buffer(frequency: float, samples: int, amplitude: float = 0.3) -> AudioBuffer:
    return AudioBuffer(
        array.array(
            "h",
            (max(-32768, min(32767, int(round(amplitude * 32767 * math.sin(2.0 * math.pi * frequency * i / RATE))))) for i in range(samples)),
        ),
        AudioFormat(RATE, 1),
    )


def generated_buffer(values, samples: int) -> AudioBuffer:
    return AudioBuffer(
        array.array("h", (max(-32768, min(32767, int(round(0.3 * 32767 * values(i))))) for i in range(samples))),
        AudioFormat(RATE, 1),
    )


def reference_config() -> FBankConfig:
    return FBankConfig.whisper_parity()


class TestParityContract:
    def test_every_field_matches_the_reference_geometry(self):
        config = reference_config()
        assert (config.sample_rate, config.fft_size, config.window_samples, config.hop_samples) == (16000, 400, 400, 160)
        assert config.n_mels == 80
        assert config.window == "hann_periodic"
        assert config.power == "power"
        assert config.center == "reflect"
        assert config.tail_pad_samples == 160
        assert config.drop_last_frame is True
        assert config.mel_scale == "slaney"
        assert config.normalize_filters is True
        assert config.edge_mode == "interpolate"
        assert config.log == "log10"
        assert config.log_floor == 1e-10
        assert config.dynamic_floor == 8.0
        assert config.post_scale == (4.0, 4.0)

    def test_the_derived_slaney_ladder_lands_on_the_reference_constant(self):
        assert slaney_hz_to_mel(8000.0) == REFERENCE_MAX_MEL
        assert slaney_mel_to_hz(REFERENCE_MAX_MEL) == pytest.approx(8000.0, abs=1e-9)
        assert slaney_hz_to_mel(1000.0) == pytest.approx(15.0, abs=1e-12)
        assert slaney_mel_to_hz(0.0) == 0.0
        edges = mel_edges_hz(reference_config())
        assert edges[0] == 0.0
        assert edges[-1] == pytest.approx(8000.0, abs=1e-9)
        assert len(edges) == 82
        # The first centre is the reference's first band centre; 37.239... Hz.
        assert edges[1] == pytest.approx(37.2392, abs=1e-3)

    def test_frame_counts_follow_the_reference_arithmetic(self):
        config = reference_config()
        # 1 + (n + 160 tail + 400 centre - 400) // 160, then one frame dropped.
        assert config.frame_count(RATE) == 101
        assert config.frame_count(4000) == 26
        assert config.frame_count(8000) == 51
        assert config.frame_count(400) == 3
        assert config.frame_count(160) == 2
        assert config.frame_count(4) == 1
        assert config.frame_count(0) == 1, "the reference pads 160 zeros, then reflects, then transforms"

    def test_the_student_geometry_is_a_different_contract_by_exactly_two_frames_per_second(self):
        assert FBankConfig().frame_count(RATE) == 98
        assert reference_config().frame_count(RATE) == 101

    def test_reflect_padding_mirrors_without_repeating_the_edge_sample(self):
        # The order of this mirror was the one real defect found while building the
        # parity path: a reversed right-hand mirror is invisible in the padding alone and
        # shows up only as a wrong spectrum in the later frames. Pinned without any
        # third-party package, against numpy's documented 'reflect' order.
        config = FBankConfig(sample_rate=16000, fft_size=64, window_ms=4.0, hop_ms=2.0, n_mels=8, center="reflect")
        assert config.pad_samples == 32
        data = [float(i) for i in range(1, 41)]
        padded = _padded_sequence(data, config)
        assert len(padded) == 40 + 64
        assert padded[:32] == [float(33 - i) for i in range(32)], "a[32] down to a[1]: 33 ... 2"
        assert padded[32:72] == data
        assert padded[72:] == [float(40 - 1 - i) for i in range(32)], "a[38] down to a[7]: 39 ... 8"
        assert padded[72] == 39.0 and padded[-1] == 8.0, "no duplicated edge sample on either side"
        assert padded[31] == 2.0 and padded[71] == 40.0

    def test_reflect_padding_keeps_bouncing_beyond_the_data(self):
        # A pad longer than the audio is not an error: numpy keeps bouncing, and the
        # expected sequence below was taken from numpy and then reproduced without it.
        # An earlier draft refused this case outright, which was a fabricated limitation.
        config = FBankConfig(sample_rate=16000, fft_size=16, window_ms=1.0, hop_ms=0.5, n_mels=4, center="reflect")
        assert config.pad_samples == 8
        assert _padded_sequence([1.0, 2.0, 3.0], config) == [
            1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0,
            2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0,
        ]

    def test_a_one_sample_buffer_reflects_to_itself(self):
        config = FBankConfig(sample_rate=16000, fft_size=16, window_ms=1.0, hop_ms=0.5, n_mels=4, center="reflect")
        assert _padded_sequence([7.0], config) == [7.0] * 17

    def test_zero_centering_is_available_and_pads_with_real_zeros(self):
        config = FBankConfig(sample_rate=16000, fft_size=64, window_ms=4.0, hop_ms=2.0, n_mels=8, center="zero")
        padded = _padded_sequence([1.0, 2.0, 3.0], config)
        assert padded[:32] == [0.0] * 32 and padded[-32:] == [0.0] * 32

    def test_a_non_power_of_two_length_uses_the_general_transform(self):
        config = reference_config()
        assert config.uses_radix2 is False and config.bin_count == 201
        assert FBankConfig().uses_radix2 is True
        assert FBankConfig(sample_rate=16000, fft_size=100, window_ms=6.0, hop_ms=2.0, n_mels=8).uses_radix2 is False

    def test_the_general_transform_matches_a_naive_dft_at_length_400(self):
        config = FBankConfig(sample_rate=16000, fft_size=400, window_ms=25.0, hop_ms=10.0, n_mels=8)
        frame = [math.sin(2.0 * math.pi * 7 * n / 400) + 0.25 * n / 400 for n in range(400)]
        ours = power_spectrum(frame, config)
        worst = 0.0
        for bin_index in range(config.bin_count):
            acc = sum(frame[n] * cmath.exp(-2j * math.pi * bin_index * n / 400) for n in range(400))
            worst = max(worst, abs(ours[bin_index] - abs(acc) ** 2))
        assert worst < 1e-6, f"general DFT deviates from the definition by {worst:.3e}"

    def test_the_dynamic_floor_and_rescale_are_whole_buffer_statistics(self):
        config = reference_config()
        assert config.global_postprocessing is True
        assert FBankConfig().global_postprocessing is False
        loud = log_fbank(tone_buffer(1000.0, RATE, amplitude=0.3), config)
        silence = log_fbank(AudioBuffer.zeros(AudioFormat(RATE, 1), 4000), config)
        # The floor is relative to the buffer's own maximum, so one frame's values depend
        # on the loudest frame anywhere in the call: the reason the student contract
        # keeps this switched off. 8 mel units become 2.0 after the /4 rescale.
        assert min(loud.values) >= max(loud.values) - 2.0 - 1e-12
        assert sum(1 for value in loud.values if abs(value - (max(loud.values) - 2.0)) < 1e-12) > 7000
        # And the absolute clip floor in the reference's units is hand-computable:
        # (log10(1e-10) + 4) / 4 == -1.5 exactly, which is all a silent grid contains.
        assert set(silence.values) == {-1.5}

    def test_slaney_scale_refuses_band_limits(self):
        with pytest.raises(ConfigurationError) as caught:
            FBankConfig(mel_scale="slaney", fft_size=400, window_ms=25.0, f_min=20.0)
        assert "0 Hz to Nyquist" in str(caught.value)

    def test_centering_refuses_a_window_shorter_than_the_transform(self):
        with pytest.raises(ConfigurationError) as caught:
            FBankConfig(center="reflect")
        assert "window to equal the transform" in str(caught.value)


def _reference_extractor():
    pytest.importorskip("numpy", reason="NOT RUN: numpy is not installed")
    pytest.importorskip(
        "faster_whisper",
        reason="NOT RUN: the reference feature extractor (faster-whisper) is not installed on this machine",
    )
    from faster_whisper.feature_extractor import FeatureExtractor

    return FeatureExtractor(feature_size=80)


def _numpy():
    return pytest.importorskip("numpy", reason="NOT RUN: numpy is not installed")


def _waveform(buffer: AudioBuffer):
    """The reference's input convention, stated: int16 samples scaled by 1/32768 as float32."""
    numpy = _numpy()
    return numpy.asarray(buffer.samples, dtype=numpy.float32) / 32768.0


def _reference_grid(extractor, buffer: AudioBuffer, **kwargs):
    return numpy_asarray(extractor(_waveform(buffer), **kwargs))


def numpy_asarray(value):
    return _numpy().asarray(value)


def _grid_matrix(grid):
    numpy = _numpy()
    return numpy.array(grid.values).reshape(grid.frames, grid.n_mels).T


def _max_abs(numpy, left, right):
    if left.shape != right.shape:
        raise AssertionError(f"shape mismatch {left.shape} vs {right.shape}")
    return float(numpy.abs(left - right).max())


class TestAgainstReference:
    @pytest.mark.parametrize(
        "label, builder",
        [
            ("one-khz-tone", lambda: tone_buffer(1000.0, RATE)),
            ("off-centre-tone", lambda: tone_buffer(1010.0, 8000)),
            ("frequency-sweep", lambda: generated_buffer(lambda i: math.sin(2 * math.pi * (300 + 4 * i) * i / RATE), 8000)),
            ("impulse", lambda: AudioBuffer(array.array("h", (9000 if i == 320 else 0 for i in range(8000))), AudioFormat(RATE, 1))),
            ("dc-offset", lambda: AudioBuffer(array.array("h", (6500 for _ in range(4000))), AudioFormat(RATE, 1))),
            ("silence", lambda: AudioBuffer.zeros(AudioFormat(RATE, 1), 4000)),
            ("two-tone-mixture", lambda: generated_buffer(lambda i: 0.6 * math.sin(2 * math.pi * 130 * i / RATE) + 0.35 * math.sin(2 * math.pi * 520 * i / RATE), 8000)),
            ("boundary-exactly-one-window", lambda: tone_buffer(1000.0, 400)),
            ("boundary-one-past-a-window", lambda: tone_buffer(1000.0, 401)),
            ("boundary-shorter-than-the-pad", lambda: tone_buffer(500.0, 160)),
        ],
    )
    def test_grid_matches_the_reference_on_identical_input(self, label, builder):
        extractor = _reference_extractor()
        numpy = _numpy()
        buffer = builder()
        grid = log_fbank(buffer, reference_config())
        shipped = _reference_grid(extractor, buffer)
        assert grid.frames == shipped.shape[1], f"{label}: frame count diverges"
        assert grid.n_mels == shipped.shape[0]
        worst = _max_abs(numpy, _grid_matrix(grid), shipped)
        assert worst < TOLERANCE, f"{label}: worst difference {worst:.3e} exceeds {TOLERANCE:.0e}"

    def test_mel_matrix_matches_the_reference_matrix(self):
        extractor = _reference_extractor()
        numpy = _numpy()
        config = reference_config()
        dense = numpy.zeros((config.n_mels, config.bin_count))
        for item in mel_filters(config):
            for offset, weight in enumerate(item.weights):
                dense[item.index, item.first_bin + offset] = weight
        worst = _max_abs(numpy, dense, extractor.mel_filters.astype(numpy.float64))
        assert worst < 1e-8, f"mel matrix differs by {worst:.3e}"

    def test_stereo_downmix_matches_the_reference_on_the_same_mono_signal(self):
        # The reference has no concept of channels, so this pins that `to_mono()` is the
        # downmix in force: the same averaged signal must give the same grid.
        extractor = _reference_extractor()
        numpy = _numpy()
        mono = tone_buffer(700.0, 4000)
        interleaved = array.array("h")
        for value in mono.samples:
            interleaved.append(value)
            interleaved.append(value)
        grid = log_fbank(AudioBuffer(interleaved, AudioFormat(RATE, 2)), reference_config())
        shipped = _reference_grid(extractor, mono)
        assert grid.frames == shipped.shape[1]
        assert _max_abs(numpy, _grid_matrix(grid), shipped) < TOLERANCE

    def test_parity_still_requires_the_parity_config(self):
        # The negative control for the whole file: the student geometry must NOT pass
        # this comparison. Were it ever to pass, the two contracts would have merged and
        # one of them would no longer mean what it says.
        extractor = _reference_extractor()
        numpy = _numpy()
        buffer = tone_buffer(1000.0, RATE)
        student = log_fbank(buffer, FBankConfig())
        shipped = _reference_grid(extractor, buffer)
        assert student.frames == 98 and shipped.shape[1] == 101
        padded = numpy.pad(_grid_matrix(student), ((0, 0), (0, shipped.shape[1] - 98)), constant_values=numpy.min(shipped))
        assert _max_abs(numpy, padded, shipped) > 0.5


class TestReferenceParameters:
    """The reference's own optional arguments, pinned to what they actually do.

    `__call__(waveform, padding=160, chunk_length=None)`: `padding` is the tail zero pad
    this frontend models as `tail_pad_samples`, so it must be testable at a second value;
    `chunk_length` only assigns `n_samples`/`nb_max_frames`, which `__call__` never reads,
    so it must NOT change the output. Both were checked experimentally rather than assumed
    — the second one was wrongly assumed in an earlier draft of this work.
    """

    @pytest.mark.parametrize("samples", [1, 5, 20, 40, 160, 400])
    def test_short_buffers_agree_with_the_reference_rather_than_being_refused(self, samples):
        extractor = _reference_extractor()
        numpy = _numpy()
        buffer = tone_buffer(500.0, samples)
        grid = log_fbank(buffer, reference_config())
        shipped = _reference_grid(extractor, buffer)
        assert grid.frames == shipped.shape[1]
        assert _max_abs(numpy, _grid_matrix(grid), shipped) < TOLERANCE

    def test_a_different_tail_pad_matches_a_different_reference_padding(self):
        extractor = _reference_extractor()
        numpy = _numpy()
        config = FBankConfig.whisper_parity()
        no_tail = FBankConfig(
            sample_rate=16000, window_ms=25.0, hop_ms=10.0, n_mels=80, fft_size=400, f_min=0.0, f_max=None,
            log_floor=1e-10, window="hann_periodic", power="power", center="reflect", tail_pad_samples=0,
            drop_last_frame=True, mel_scale="slaney", normalize_filters=True, edge_mode="interpolate",
            log="log10", dynamic_floor=8.0, post_scale=(4.0, 4.0),
        )
        assert no_tail.tail_pad_samples == 0 and no_tail.fft_size == 400
        buffer = tone_buffer(1000.0, 4000)
        mine = _grid_matrix(log_fbank(buffer, no_tail))
        shipped = _reference_grid(extractor, buffer, padding=0)
        assert mine.shape == shipped.shape, "the frame count must move with the tail pad"
        assert mine.shape[1] < _grid_matrix(log_fbank(buffer, config)).shape[1]
        assert _max_abs(numpy, mine, shipped) < TOLERANCE

    def test_chunk_length_does_not_change_the_reference_output(self):
        extractor = _reference_extractor()
        numpy = _numpy()
        buffer = tone_buffer(1000.0, 4000)
        default = _reference_grid(extractor, buffer)
        with_chunk = _reference_grid(extractor, buffer, chunk_length=30)
        assert default.shape == with_chunk.shape
        assert numpy.array_equal(default, with_chunk), (
            "chunk_length only assigns n_samples/nb_max_frames, which __call__ never reads; "
            "if a future version starts using them, this test must fail and §3.3 must be re-measured"
        )
