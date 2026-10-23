"""Log-mel filterbank frontend (stdlib only).

This is the missing input contract for a self-trained recognition model: a
Conformer-class encoder consumes a frame-wise log-mel feature grid, and nothing in
this tree produced one before. It is a *frontend*, not a model — it turns samples
into numbers, and it is incapable of demonstrating anything about recognition
accuracy. The signals in `tests/fixtures.py` are tones and envelopes, exactly in the
sense that file states: they exercise arithmetic determinism, and no test here may be
read as evidence about speech.

Everything is hand-computable and dependency-free, in the same spirit as
`audio/pcm.py` (which exists because `audioop` left the standard library in 3.13):
the DFT is an iterative radix-2 Cooley-Tukey, the filterbank is triangular, and the
log is natural. Every constant that shapes the output is a field of
:class:`FBankConfig`, so a feature grid can be re-derived from the config that made
it rather than from the mood of the machine.

Deliberately absent, with the reason recorded rather than left implied:

* **no pre-emphasis.** A 0.97 pre-emphasis pole is a Kaldi convention and it changes
  the spectral tilt the model sees. Shipping a knob nobody has verified against a
  reference is worse than not having it; add it with a parity measurement, not here.
* **no mean/variance normalisation, no CMN, no spec_augment, no LF-MF, no deltas.** Those belong to a
  training recipe and to the model that consumes it, and this repository has no approved recipe yet.
  `log_fbank` returns raw log-mel energies so that whatever normalisation is later chosen is visible as a
  separate, testable step. The one exception is opt-in and named: `FBankConfig.whisper_parity()` applies the
  reference frontend's own whole-buffer `max(x, max-8)` clamp and `(x+4)/4` rescale, because reproducing that
  frontend *is* its job — and `global_postprocessing` says so on every grid produced with it.
* **no float32 cast.** Output is stored in an `array('d')`. Casting to float32 is the
  consumer's boundary decision — see :func:`nativevoice.audio.pcm.to_float_samples`
  for the same convention on the sample side — and it is where roughly 7 decimal
  digits of rounding would enter, which is why it is not hidden in here.

Scaling convention, stated because every downstream number depends on it and two
tools that disagree about it silently produce different accuracy. The first five are
the **default** geometry (`FBankConfig()`), which is the project's own student-side
contract; `FBankConfig.whisper_parity()` selects the reference frontend's choices
instead, and the two are deliberately not the same:

1. int16 samples are divided by :data:`nativevoice.audio.pcm.FULL_SCALE` (32768.0),
   so a full-scale sine is amplitude 1.0.
2. the forward DFT is **unnormalised**, so Parseval holds as
   `sum(|X[k]|^2) = N * sum(|x[n]|^2)` where `N` is the transform length
   (:attr:`FBankConfig.fft_size`, zero padding included); that identity is asserted in
   the tests over the *full* two-sided spectrum. A transform length that is not a power
   of two uses a general DFT with precomputed twiddles, because the reference frontend
   uses `n_fft=400` and refusing it would have been convenience dressed as a property.
3. power is `re^2 + im^2` (or its square root when `power="magnitude"`).
4. mel filters are triangular, spaced uniformly in mel, with **no area normalisation**
   and edges snapped to bins by round-half-up — so a filter's response to a flat
   spectrum is proportional to its bandwidth in bins, exactly as Kaldi's default
   `raw-fbank` computes. `mel_scale="slaney"` plus `edge_mode="interpolate"` plus
   `normalize_filters=True` is the reference's version instead.
5. `log(max(energy, log_floor))` in the configured base (:attr:`FBankConfig.log`,
   natural by default) — a *floor*, not an additive epsilon. A silent frame therefore
   equals `log(log_floor)` exactly, which the tests pin.
6. `dynamic_floor` and `post_scale` apply **whole-buffer** statistics, matching the
   reference's `max(x, x.max() - 8.0)` then `(x + 4) / 4`. They are off by default
   precisely because they make one frame depend on the loudest frame elsewhere, which
   breaks frame-by-frame reproducibility; :attr:`FBankConfig.global_postprocessing`
   reports whichever contract a grid was produced under.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass
from typing import Sequence

from ..core.audio import AudioBuffer, AudioFormat
from ..core.errors import ConfigurationError
from .pcm import FULL_SCALE

#: The window functions this frontend implements. Named, not lambda-swappable, so a
#: recorded config always denotes a computable window.
#:
#: `hann` is the *symmetric* definition (`0.5 - 0.5*cos(2*pi*n/(N-1))`), whose last
#: sample is 0. `hann_periodic` is `np.hanning(N + 1)[:-1]`, the periodic form a
#: short-time transform actually wants — its last sample is ~6.17e-5, not 0. They are
#: different functions and a pipeline that quietly means one and writes the other is
#: not comparable to anything, so both are named.
WINDOWS = ("povey", "hamming", "hann", "hann_periodic")

#: `power` integrates |X|^2, `magnitude` integrates |X|.
POWER_MEASURES = ("power", "magnitude")

#: Framing policy. `none` is the Phase 1 behaviour: a frame is a window over real
#: samples only, so nothing outside the buffer is invented and the samples between two
#: sparse frames are observed by nothing. `zero` pads with zeros; `reflect` pads the way
#: `numpy.pad(mode="reflect")` does — mirror without repeating the edge sample, which
#: needs at least `pad` samples to exist and therefore refuses very short audio.
CENTER_MODES = ("none", "zero", "reflect")

#: Mel ladders. `htk` is `2595*log10(1 + f/700)`. `slaney` is the piecewise Slaney
#: scale (linear below 1000 Hz at 3/200 mel per Hz, logarithmic above with
#: `logstep = ln(6.4)/27`), which is what the reference extractor uses.
MEL_SCALES = ("htk", "slaney")

#: How a filter's real-valued Hz edges become integer FFT bins. `round_bin` snaps with
#: round-half-up (the Phase 1 contract). `interpolate` evaluates the triangle at each
#: bin's exact frequency, which is what a continuous-frequency filterbank means and is
#: required for parity: at 16 kHz with a 400-point transform one bin is 40 Hz, so
#: rounding edges to bins moves them by up to 20 Hz.
EDGE_MODES = ("round_bin", "interpolate")

LOG_BASES = ("natural", "log10")

_MIN_FFT_SIZE = 16
_HTK_MEL_K1 = 2595.0
_HTK_MEL_K2 = 700.0
_SLANEY_F_SP = 200.0 / 3.0
_SLANEY_MIN_LOG_HZ = 1000.0
_SLANEY_MIN_LOG_MEL = _SLANEY_MIN_LOG_HZ / _SLANEY_F_SP  # 14.999999999999998
_SLANEY_LOG_STEP = math.log(6.4) / 27.0


def hz_to_mel(value: float) -> float:
    """The HTK mel scale: `2595 * log10(1 + f/700)`."""
    return _HTK_MEL_K1 * math.log10(1.0 + value / _HTK_MEL_K2)


def mel_to_hz(value: float) -> float:
    """Inverse of :func:`hz_to_mel`."""
    return _HTK_MEL_K2 * (10.0 ** (value / _HTK_MEL_K1) - 1.0)


def slaney_hz_to_mel(value: float) -> float:
    """Slaney mel: linear below 1000 Hz, logarithmic above.

    The reference hardcodes its top edge as `45.245640471924965` mel rather than
    deriving it; that constant is exactly this function at 8000 Hz, so deriving it
    reproduces the reference at 16 kHz and generalises honestly to other rates.
    """
    if value < 0.0:
        raise ConfigurationError("frequency cannot be negative", frequency=value)
    if value < _SLANEY_MIN_LOG_HZ:
        return value / _SLANEY_F_SP
    return _SLANEY_MIN_LOG_MEL + math.log(value / _SLANEY_MIN_LOG_HZ) / _SLANEY_LOG_STEP


def slaney_mel_to_hz(value: float) -> float:
    """Inverse of :func:`slaney_hz_to_mel`."""
    if value < _SLANEY_MIN_LOG_MEL:
        return value * _SLANEY_F_SP
    return _SLANEY_MIN_LOG_HZ * math.exp(_SLANEY_LOG_STEP * (value - _SLANEY_MIN_LOG_MEL))


def _round_bin(value: float) -> int:
    """Round half up, explicitly.

    `math.floor(x + 0.5)` rather than :func:`round`, because Python's `round` is
    banker's rounding and would map an exactly-half bin differently in a different
    language. Bin mapping is part of the feature contract, not a detail.
    """
    return int(math.floor(value + 0.5))


@dataclass(frozen=True, slots=True)
class FBankConfig:
    """Every number that shapes a feature grid.

    Frame geometry is expressed in milliseconds because that is how speech tooling
    names it, and is converted to samples by :attr:`window_samples` /
    :attr:`hop_samples` with explicit round-half-up, documented so a 25 ms frame at an
    odd rate is not a mystery.

    :attr:`fft_size` is 1024 for a 400-sample window at 16 kHz on purpose. Zero padding
    invents no information — it interpolates the spectrum — but `edge_mode="round_bin"`
    needs bins finer than the mel ladder: with `fft_size=512` one bin is 31.25 Hz wide
    and three of the 80 filters (indices 0, 3 and 4) collapse so that `left == center`
    or `center == right`, which would hand the model constant `log(log_floor)` rows.
    :func:`mel_filters` refuses that rather than shipping it, and the refusal is asserted
    in the tests. `edge_mode="interpolate"` has no such collapse because it never rounds:
    that is the mode `whisper_parity()` uses, and it is why the two edge modes are
    separate fields rather than one being "fixed" into the other.

    A grid is reproducible from this object alone: two calls with equal configs on
    equal samples produce identical values.

    **Sample-rate geometry limit, measured not assumed.** The default `fft_size=1024`
    accommodates a 25 ms window only up to 24000 Hz. At 32000 Hz the window is 800 samples
    and bin spacing is still 31.25 Hz, so the 80-filter ladder collapses; at 44100 and
    48000 Hz the window alone (1103 and 1200 samples) exceeds the transform. Both cases
    raise `ConfigurationError` naming the remedy — grow `fft_size` with the rate rather
    than shrinking the window, and record whichever geometry is used wherever the features
    are consumed. Which rates work at the defaults is asserted in the tests, so this limit
    cannot drift unnoticed.
    """

    sample_rate: int = 16000
    window_ms: float = 25.0
    hop_ms: float = 10.0
    n_mels: int = 80
    fft_size: int = 1024
    f_min: float = 20.0
    f_max: float | None = None
    log_floor: float = 1e-10
    window: str = "povey"
    power: str = "power"
    center: str = "none"
    tail_pad_samples: int = 0
    drop_last_frame: bool = False
    mel_scale: str = "htk"
    normalize_filters: bool = False
    edge_mode: str = "round_bin"
    log: str = "natural"
    dynamic_floor: float | None = None
    post_scale: tuple[float, float] | None = None

    @classmethod
    def whisper_parity(cls, *, n_mels: int = 80) -> "FBankConfig":
        """The configuration that reproduces the reference Whisper frontend.

        Derived by reading and running `faster_whisper==1.2.1`,
        `faster_whisper/feature_extractor.py` — `FeatureExtractor(feature_size=80)` with
        `sampling_rate=16000, n_fft=400, hop_length=160`, `center=True` with
        `mode="reflect"`, an unnormalised `np.fft.rfft`, `np.abs(...) ** 2` on
        `stft[..., :-1]`, Slaney mel filters area-normalised by `2/(hi - lo)`,
        `np.log10(np.clip(x, 1e-10))`, then `max(x, x.max() - 8.0)`, then `(x + 4) / 4`.
        It is *named* rather than made the default on purpose: the student's own frontend
        is a separate decision (see `docs/asr-architecture.md` D1), and a model must not
        inherit the teacher's geometry by accident.

        Two documented consequences of copying the reference honestly:

        * `fft_size=400` is not a power of two, so this config uses the general DFT path,
          which is far slower than radix-2 and is not a streaming geometry.
        * `dynamic_floor` and `post_scale` are **whole-buffer statistics**: frame k's
          values depend on the loudest frame anywhere in the same call. That is a real
          train/inference coupling and :attr:`global_postprocessing` reports it.
        """
        return cls(
            sample_rate=16000,
            window_ms=25.0,
            hop_ms=10.0,
            n_mels=n_mels,
            fft_size=400,
            f_min=0.0,
            f_max=None,
            log_floor=1e-10,
            window="hann_periodic",
            power="power",
            center="reflect",
            tail_pad_samples=160,
            drop_last_frame=True,
            mel_scale="slaney",
            normalize_filters=True,
            edge_mode="interpolate",
            log="log10",
            dynamic_floor=8.0,
            post_scale=(4.0, 4.0),
        )

    @property
    def global_postprocessing(self) -> bool:
        """Whether a value in the grid depends on samples outside its own frame."""
        return self.dynamic_floor is not None or self.post_scale is not None

    @property
    def pad_samples(self) -> int:
        return self.fft_size // 2 if self.center != "none" else 0

    @property
    def padded_length(self) -> int:
        """Buffer length as the transform sees it, after tail and center padding."""
        return self.tail_pad_samples + 2 * self.pad_samples

    def frame_start(self, frame: int) -> int:
        """First sample index of a frame in *buffer* coordinates (may be negative)."""
        return frame * self.hop_samples - self.pad_samples

    def __post_init__(self) -> None:
        if not AudioFormat(self.sample_rate, 1).is_allowed_preset():
            raise ConfigurationError(
                "sample_rate is not one the audio stack guarantees end to end",
                sample_rate=self.sample_rate,
                remedy="resample with nativevoice.audio.resample first",
            )
        if self.window_ms <= 0 or self.hop_ms <= 0:
            raise ConfigurationError("window_ms and hop_ms must be positive", window_ms=self.window_ms, hop_ms=self.hop_ms)
        if self.window not in WINDOWS:
            raise ConfigurationError("unknown window function", window=self.window, allowed=WINDOWS)
        if self.power not in POWER_MEASURES:
            raise ConfigurationError("unknown power measure", power=self.power, allowed=POWER_MEASURES)
        if self.n_mels < 1:
            raise ConfigurationError("n_mels must be >= 1", n_mels=self.n_mels)
        if self.fft_size < _MIN_FFT_SIZE:
            raise ConfigurationError(
                "fft_size must be at least the minimum transform length",
                fft_size=self.fft_size,
                minimum=_MIN_FFT_SIZE,
            )
        if self.log_floor <= 0.0:
            raise ConfigurationError("log_floor must be positive", log_floor=self.log_floor)
        if self.f_min < 0.0:
            raise ConfigurationError("f_min must be >= 0", f_min=self.f_min)
        nyquist = self.sample_rate / 2.0
        if self.f_max is not None and not (self.f_min < self.f_max <= nyquist):
            raise ConfigurationError("f_max must be above f_min and no higher than Nyquist", f_max=self.f_max, f_min=self.f_min, nyquist=nyquist)
        if not 1 <= self.window_samples <= self.fft_size:
            raise ConfigurationError(
                "window must be at least one sample and no longer than the FFT",
                window_samples=self.window_samples,
                fft_size=self.fft_size,
            )
        if self.hop_samples < 1:
            raise ConfigurationError("hop_ms is shorter than one sample", hop_ms=self.hop_ms, sample_rate=self.sample_rate)
        if self.center not in CENTER_MODES:
            raise ConfigurationError("unknown centering mode", center=self.center, allowed=CENTER_MODES)
        if self.mel_scale not in MEL_SCALES:
            raise ConfigurationError("unknown mel scale", mel_scale=self.mel_scale, allowed=MEL_SCALES)
        if self.edge_mode not in EDGE_MODES:
            raise ConfigurationError("unknown mel edge mode", edge_mode=self.edge_mode, allowed=EDGE_MODES)
        if self.log not in LOG_BASES:
            raise ConfigurationError("unknown log base", log=self.log, allowed=LOG_BASES)
        if self.tail_pad_samples < 0:
            raise ConfigurationError("tail_pad_samples cannot be negative", tail_pad_samples=self.tail_pad_samples)
        if self.center != "none" and self.window_samples != self.fft_size:
            raise ConfigurationError(
                "a centered geometry requires the window to equal the transform",
                center=self.center,
                window_samples=self.window_samples,
                fft_size=self.fft_size,
                remedy="set window_ms so window_samples == fft_size, or use center='none'",
            )
        if self.mel_scale == "slaney":
            if self.f_min != 0.0 or self.f_max is not None:
                raise ConfigurationError(
                    "the slaney ladder spans 0 Hz to Nyquist and takes no band limits",
                    f_min=self.f_min,
                    f_max=self.f_max,
                    remedy="set f_min=0.0 and f_max=None, or use mel_scale='htk'",
                )
        if self.dynamic_floor is not None and self.dynamic_floor <= 0.0:
            raise ConfigurationError("dynamic_floor must be positive", dynamic_floor=self.dynamic_floor)
        if self.post_scale is not None:
            scale = self.post_scale
            if not isinstance(scale, tuple) or len(scale) != 2 or not scale[1]:
                raise ConfigurationError(
                    "post_scale must be a (offset, divisor) tuple with a non-zero divisor",
                    post_scale=scale,
                )

    @property
    def window_samples(self) -> int:
        return _round_bin(self.window_ms * self.sample_rate / 1000.0)

    @property
    def hop_samples(self) -> int:
        return _round_bin(self.hop_ms * self.sample_rate / 1000.0)

    @property
    def freq_max(self) -> float:
        return self.sample_rate / 2.0 if self.f_max is None else self.f_max

    @property
    def bin_count(self) -> int:
        """Positive-frequency bins of the DFT, DC included (index 0)."""
        return self.fft_size // 2 + 1

    @property
    def uses_radix2(self) -> bool:
        """Whether the fast path applies; otherwise a general DFT is used."""
        return self.fft_size >= 2 and (self.fft_size & (self.fft_size - 1)) == 0

    def frame_count(self, n_samples: int) -> int:
        """Frames emitted for `n_samples` at this geometry.

        With `center="none"` and no tail padding this is the Phase 1 rule, unchanged:
        `0` for anything shorter than one window, else `1 + (n - window) // hop`. A
        leading partial frame is not averaged up to a full window, because a padded and
        an unpadded frontend are not feature-compatible and silence should not be
        manufactured from fewer samples than exist.

        With centering or tail padding, the transform's own length governs:
        `1 + (n + tail + 2*pad - fft_size) // hop`, then minus one when
        :attr:`drop_last_frame` — which is how the reference frontend reaches 101 frames
        for one second of 16 kHz audio while the unpadded geometry gives 98.
        """
        if self.center == "none" and not self.tail_pad_samples:
            frames = 0 if n_samples < self.window_samples else 1 + (n_samples - self.window_samples) // self.hop_samples
        else:
            length = n_samples + self.padded_length
            frames = 0 if length < self.fft_size else 1 + (length - self.fft_size) // self.hop_samples
        if self.drop_last_frame and frames > 0:
            frames -= 1
        return frames


@dataclass(frozen=True, slots=True)
class MelFilter:
    """One triangular mel filter, stored sparsely over the bins it actually touches."""

    index: int
    left_hz: float
    center_hz: float
    right_hz: float
    first_bin: int
    weights: tuple[float, ...]

    @property
    def last_bin(self) -> int:
        return self.first_bin + len(self.weights) - 1


def window_function(config: FBankConfig) -> tuple[float, ...]:
    """The analysis window, computed once per call and never cached.

    Povey is `(0.5 - 0.5*cos(2*pi*n/(N-1)))^0.1`; its endpoints are exactly 0.0, which
    is why the 0^0.1 case is spelled out rather than falling into a general power.

    `hann_periodic` is `0.5 - 0.5*cos(2*pi*n/N)` — the last sample is ~6.17e-5, not 0.
    It is not a rounding of `hann`; the two are different functions, and parity with a
    short-time transform depends on picking the periodic one.
    """
    size = config.window_samples
    if size == 1:
        return (1.0,)
    coefficients = {"hamming": (0.54, 0.46), "hann": (0.5, 0.5)}
    if config.window in coefficients:
        a, b = coefficients[config.window]
        return tuple(a - b * math.cos(2.0 * math.pi * n / (size - 1)) for n in range(size))
    if config.window == "hann_periodic":
        return tuple(0.5 - 0.5 * math.cos(2.0 * math.pi * n / size) for n in range(size))
    out = []
    for n in range(size):
        base = 0.5 - 0.5 * math.cos(2.0 * math.pi * n / (size - 1))
        out.append(0.0 if base <= 0.0 else base ** 0.1)
    return tuple(out)


def bin_frequencies(config: FBankConfig) -> tuple[float, ...]:
    """The frequency of each DFT bin: `k * sample_rate / fft_size`, DC first."""
    return tuple(k * config.sample_rate / config.fft_size for k in range(config.bin_count))


def mel_edges_hz(config: FBankConfig) -> tuple[float, ...]:
    """`n_mels + 2` real-valued band edges in Hz, uniform in the config's mel scale.

    Public because the edges, not the triangles, are the thing a reviewer can check
    against a paper or another tool. In `slaney` mode the ladder spans 0 Hz to Nyquist
    by construction (the config refuses band limits there).
    """
    if config.mel_scale == "slaney":
        to_hz, to_mel = slaney_mel_to_hz, slaney_hz_to_mel
    else:
        to_hz, to_mel = mel_to_hz, hz_to_mel
    mel_min, mel_max = to_mel(config.f_min), to_mel(config.freq_max)
    if not mel_max > mel_min:
        raise ConfigurationError("filterbank spans no frequency", f_min=config.f_min, f_max=config.freq_max)
    step = (mel_max - mel_min) / (config.n_mels + 1)
    return tuple(to_hz(mel_min + step * i) for i in range(config.n_mels + 2))


def mel_filters(config: FBankConfig) -> tuple[MelFilter, ...]:
    """Build the triangular mel filterbank: `n_mels` filters over the DFT bins.

    Two edge policies, because they answer different questions:

    * `round_bin` snaps the real-valued Hz edges to integer bins with the explicit
      round-half-up in :func:`_round_bin`, and weights are the triangle evaluated at
      those bin indices. A filter whose three edges land on one bin collapses to zero
      width and is refused, because it would silently contribute a constant
      `log(log_floor)` row and read as a working frontend.
    * `interpolate` evaluates the triangle at each bin's **exact frequency** — the
      definition a continuous-frequency filterbank actually means, and what the
      reference extractor computes. It never collapses: with 40 Hz bins a filter can
      have partial weight everywhere in its span, which is precisely why `round_bin`
      cannot reach parity (at 16 kHz / 400 points a bin is 40 Hz wide, so rounding moves
      an edge by up to 20 Hz).

    With :attr:`FBankConfig.normalize_filters` each triangle is scaled by
    `2 / (right_hz - left_hz)` (Slaney area normalisation), so weights are in units of
    1/Hz, the peak is no longer 1.0, and the log magnitudes shift accordingly. That is
    a property of the reference, not a mistake to hide: the parity test compares the
    whole matrix, not a hand-picked bin.
    """
    hz = mel_edges_hz(config)
    last_bin = config.bin_count - 1
    frequencies = bin_frequencies(config) if config.edge_mode == "interpolate" else ()
    out: list[MelFilter] = []
    for index in range(config.n_mels):
        left_hz, center_hz, right_hz = hz[index], hz[index + 1], hz[index + 2]
        if config.edge_mode == "round_bin":
            left = min(_round_bin(left_hz * config.fft_size / config.sample_rate), last_bin)
            center = min(_round_bin(center_hz * config.fft_size / config.sample_rate), last_bin)
            right = min(_round_bin(right_hz * config.fft_size / config.sample_rate), last_bin)
            if not left < center < right:
                raise ConfigurationError(
                    "a mel filter has no width at this fft_size; increase fft_size, reduce n_mels, "
                    "or use edge_mode='interpolate'",
                    filter=index,
                    left_bin=left,
                    center_bin=center,
                    right_bin=right,
                    fft_size=config.fft_size,
                    n_mels=config.n_mels,
                )
            weights: list[float] = []
            for bin_index in range(left, right + 1):
                if bin_index < center:
                    weights.append((bin_index - left) / (center - left))
                elif bin_index == center:
                    weights.append(1.0)
                else:
                    weights.append((right - bin_index) / (right - center))
            first_bin = left
        else:
            weights = []
            for bin_index in range(config.bin_count):
                value = frequencies[bin_index]
                if value <= left_hz or value >= right_hz:
                    weights.append(0.0)
                elif value < center_hz:
                    weights.append((value - left_hz) / (center_hz - left_hz))
                elif value > center_hz:
                    weights.append((right_hz - value) / (right_hz - center_hz))
                else:
                    weights.append(1.0)
            positive = [i for i, weight in enumerate(weights) if weight > 0.0]
            if not positive:
                raise ConfigurationError(
                    "a mel filter has no bin inside its band; increase fft_size or reduce n_mels",
                    filter=index,
                    left_hz=left_hz,
                    center_hz=center_hz,
                    right_hz=right_hz,
                    fft_size=config.fft_size,
                )
            first_bin, last = positive[0], positive[-1]
            weights = weights[first_bin : last + 1]
        if config.normalize_filters:
            span = right_hz - left_hz
            if span <= 0.0:
                raise ConfigurationError("filter span must be positive", filter=index, span_hz=span)
            weights = [weight * (2.0 / span) for weight in weights]
        out.append(
            MelFilter(
                index=index,
                left_hz=left_hz,
                center_hz=center_hz,
                right_hz=right_hz,
                first_bin=first_bin,
                weights=tuple(weights),
            )
        )
    return tuple(out)


def center_frequencies(config: FBankConfig) -> tuple[float, ...]:
    """The mel filter centre frequencies in Hz, lowest to highest."""
    return tuple(f.center_hz for f in mel_filters(config))


def _fft(re_in: list[float], im_in: list[float]) -> tuple[list[float], list[float]]:
    """Iterative radix-2 Cooley-Tukey, in place on copies. Unnormalised forward DFT."""
    size = len(re_in)
    re = re_in[:]
    im = im_in[:]
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    length = 2
    while length <= size:
        angle = -2.0 * math.pi / length
        step_re, step_im = math.cos(angle), math.sin(angle)
        half = length >> 1
        for start in range(0, size, length):
            w_re, w_im = 1.0, 0.0
            for offset in range(half):
                a = start + offset
                b = a + half
                t_re = w_re * re[b] - w_im * im[b]
                t_im = w_re * im[b] + w_im * re[b]
                re[b] = re[a] - t_re
                im[b] = im[a] - t_im
                re[a] = re[a] + t_re
                im[a] = im[a] + t_im
                w_re, w_im = w_re * step_re - w_im * step_im, w_re * step_im + w_im * step_re
        length <<= 1
    return re, im


class _Transform:
    """The DFT for one geometry, with its twiddles built once and reused per frame.

    Radix-2 when the length is a power of two; otherwise a general DFT over the real
    input. The general path exists because the reference frontend uses `n_fft=400`,
    which is not a power of two — refusing it would have been an implementation
    convenience dressed up as a safety property. Both paths are unnormalised forward
    transforms, so Parseval keeps the same shape, and the general path is roughly an
    order of magnitude slower, which is measured and reported rather than glossed.
    """

    __slots__ = ("config", "tables")

    def __init__(self, config: FBankConfig) -> None:
        self.config = config
        if config.uses_radix2:
            self.tables = None
            return
        size = config.fft_size
        self.tables = tuple(
            tuple(
                (math.cos(2.0 * math.pi * k * n / size), math.sin(2.0 * math.pi * k * n / size))
                for n in range(size)
            )
            for k in range(config.bin_count)
        )

    def power(self, frame: Sequence[float]) -> tuple[float, ...]:
        config = self.config
        if self.tables is None:
            size = config.fft_size
            re = [0.0] * size
            im = [0.0] * size
            for index, value in enumerate(frame):
                re[index] = value
            re, im = _fft(re, im)
            out = [re[k] * re[k] + im[k] * im[k] for k in range(config.bin_count)]
        else:
            out = []
            for k in range(config.bin_count):
                table = self.tables[k]
                acc_re = 0.0
                acc_im = 0.0
                for value, (cos_value, sin_value) in zip(frame, table):
                    acc_re += value * cos_value
                    acc_im -= value * sin_value
                out.append(acc_re * acc_re + acc_im * acc_im)
        if config.power == "magnitude":
            out = [math.sqrt(value) for value in out]
        return tuple(out)


def _log_function(config: FBankConfig):
    return math.log10 if config.log == "log10" else math.log


def power_spectrum(frame: Sequence[float], config: FBankConfig) -> tuple[float, ...]:
    """Positive-frequency power (or magnitude) of one zero-padded frame.

    Exposed because a reviewer asked for a frontend they could check bin by bin; the
    tests use it to pin Parseval against the time domain. Values above
    :attr:`FBankConfig.bin_count` bins are not returned, and a frame longer than the
    transform is refused rather than truncated — a silently dropped tail is a wrong
    spectrum, which is worse than a raised eyebrow.
    """
    size = config.fft_size
    if len(frame) > size:
        raise ConfigurationError(
            "frame is longer than the transform",
            frame=len(frame),
            fft_size=size,
        )
    return _Transform(config).power(frame)


@dataclass(frozen=True, slots=True)
class FeatureGrid:
    """`frames` rows of `n_mels` log-mel values, row-major in :attr:`values`.

    Nothing in here is normalised or quantised; :attr:`values` is an `array('d')`, so
    `values[f * n_mels + m]` is exactly what :meth:`value` returns.

    `frozen` freezes the *identity* of the grid, not its bytes: :attr:`values` is a
    mutable array, exactly like `core.audio.AudioBuffer.samples`, because a 30-second
    utterance at 80 mels is a buffer a consumer may want to write into (a normaliser, a
    logger) rather than re-allocate. A caller that needs an immutable copy takes
    `array('d', grid.values)`. What *is* enforced is the shape: a grid whose row count
    and value count disagree is refused at construction, so :meth:`row` can never
    return a silently short tuple — that was a real defect found in review, not a
    hypothetical one.
    """

    config: FBankConfig
    frames: int
    values: array.array
    source_format: AudioFormat
    started_at_ns: int = 0
    #: Real frames in the buffer the grid was computed from, so :meth:`extent_s` can
    # clamp to audio that exists. 0 for a grid assembled by hand, which clamps to empty.
    source_frames: int = 0

    def __post_init__(self) -> None:
        if self.frames < 0:
            raise ConfigurationError("frames cannot be negative", frames=self.frames)
        expected = self.frames * self.config.n_mels
        if len(self.values) != expected:
            raise ConfigurationError(
                "feature grid shape does not match its frame count",
                frames=self.frames,
                n_mels=self.config.n_mels,
                expected_values=expected,
                actual_values=len(self.values),
            )

    @property
    def n_mels(self) -> int:
        return self.config.n_mels

    @property
    def global_postprocessing(self) -> bool:
        """Whether these values depend on samples outside the frame they describe."""
        return self.config.global_postprocessing

    @property
    def is_empty(self) -> bool:
        return self.frames == 0

    def value(self, frame: int, mel: int) -> float:
        if not 0 <= frame < self.frames or not 0 <= mel < self.n_mels:
            # ConfigurationError, not IndexError: `tools/audit.py`'s error-taxonomy check counts
            # any raise site outside `core.errors` as a finding, and `core/audio.py` already
            # answers an out-of-range index this way (`AudioBuffer.channel`).
            raise ConfigurationError("feature index out of range", frame=frame, mel=mel, frames=self.frames, n_mels=self.n_mels)
        return self.values[frame * self.n_mels + mel]

    def row(self, frame: int) -> tuple[float, ...]:
        if not 0 <= frame < self.frames:
            raise ConfigurationError("feature frame out of range", frame=frame, frames=self.frames)
        start = frame * self.n_mels
        return tuple(self.values[start : start + self.n_mels])

    def frame_time_ms(self, frame: int) -> float:
        """Start of `frame` in buffer milliseconds.

        Zero-based and never negative when `center="none"`; under centering the first
        frames genuinely start *before* the buffer, and this returns the negative offset
        rather than pretending they do not. :meth:`extent_s` is the clamped view.
        """
        return self.config.frame_start(frame) * 1000.0 / self.config.sample_rate

    def extent_s(self, frame: int) -> tuple[float, float]:
        """`(start_s, end_s)` of the **real** samples a frame draws on, clamped to the buffer.

        Under `center="reflect"` or `"zero"` a frame also draws on synthesized samples
        outside the buffer; those are not reported here, so the extent is always a
        sub-interval of the audio that exists.
        """
        rate = self.config.sample_rate
        length = self.source_frames
        first = max(0, min(self.config.frame_start(frame), length))
        last = max(first, min(first + self.config.window_samples, length))
        return first / rate, last / rate


def _reflected(source: list[float], index: int) -> float:
    """Element `index` (1-based, measured outward) of a reflect mirror.

    Triangular bounce with period `2 * (n - 1)` — the same rule `numpy.pad(mode="reflect")`
    applies, including when the pad is longer than the data, which is why there is no
    "needs more samples than the pad" refusal here. An earlier draft refused that case and
    the refusal was a fabricated limitation, not numpy's.
    """
    count = len(source)
    if count == 1:
        return source[0]
    period = 2 * (count - 1)
    folded = index % period
    if folded >= count:
        folded = period - folded
    return source[folded]


def _padded_sequence(data: list[float], config: FBankConfig) -> list[float]:
    """Tail padding then center padding, exactly as the reference framing does it.

    `reflect` mirrors without repeating the edge sample — `numpy.pad(mode="reflect")`
    semantics: `[3, 2, 1, 2, 3, 4, 5, 4, 3]` for `[1..5]` padded by 2 — and keeps
    bouncing for a pad longer than the data, which numpy accepts. A one-sample buffer
    mirrors to itself.
    """
    padded = data if not config.tail_pad_samples else data + [0.0] * config.tail_pad_samples
    pad = config.pad_samples
    if not pad:
        return padded
    if config.center == "zero":
        return [0.0] * pad + padded + [0.0] * pad
    left = [_reflected(padded, index) for index in range(pad, 0, -1)]
    backwards = padded[::-1]
    right = [_reflected(backwards, index) for index in range(1, pad + 1)]
    return left + padded + right


def log_fbank(buffer: AudioBuffer, config: FBankConfig | None = None) -> FeatureGrid:
    """Compute the log-mel feature grid for one buffer.

    Stereo is downmixed by :meth:`AudioBuffer.to_mono` (the same averaging the
    orchestrator applies before an engine sees audio), not by taking one channel, and
    a rate mismatch is refused rather than silently resampled — an engine's format is
    a contract, and a frontend that guesses it produces features the model cannot
    name.

    With :attr:`FBankConfig.dynamic_floor` or `post_scale` set, the grid is a
    **whole-buffer** result: a frame's values depend on the loudest frame in the same
    call, so the same utterance is not reproducible frame-by-frame in a stream.
    :attr:`FeatureGrid.global_postprocessing` reports that.
    """
    config = config if config is not None else FBankConfig()
    if buffer.format.sample_rate != config.sample_rate:
        raise ConfigurationError(
            "frontend sample rate does not match the buffer",
            buffer_rate=buffer.format.sample_rate,
            config_rate=config.sample_rate,
            remedy="resample with nativevoice.audio.resample before extracting features",
        )
    if not len(buffer.samples):
        raise ConfigurationError("cannot extract features from an empty buffer")
    mono = buffer.to_mono() if buffer.format.channels > 1 else buffer
    frames = config.frame_count(mono.frame_count)
    source_frames = mono.frame_count
    if frames <= 0:
        return FeatureGrid(config, 0, array.array("d"), mono.format, buffer.started_at_ns, source_frames)
    padded = _padded_sequence([value / FULL_SCALE for value in mono.samples], config)
    window = config.window_samples
    hop = config.hop_samples
    weights = window_function(config)
    filters = mel_filters(config)
    transform = _Transform(config)
    floor = config.log_floor
    logarithm = _log_function(config)
    out = array.array("d", bytes(frames * config.n_mels * 8))
    cursor = 0
    for frame in range(frames):
        first = frame * hop
        windowed = [padded[first + n] * weights[n] for n in range(window)]
        if window < config.fft_size:
            windowed.extend([0.0] * (config.fft_size - window))
        spectrum = transform.power(windowed)
        for mel_filter in filters:
            total = 0.0
            offset = mel_filter.first_bin
            for position, weight in enumerate(mel_filter.weights):
                if weight:
                    total += weight * spectrum[offset + position]
            out[cursor] = logarithm(total if total > floor else floor)
            cursor += 1
    if config.dynamic_floor is not None:
        limit = max(out) - config.dynamic_floor
        for index, value in enumerate(out):
            if value < limit:
                out[index] = limit
    if config.post_scale is not None:
        shift, divisor = config.post_scale
        for index, value in enumerate(out):
            out[index] = (value + shift) / divisor
    return FeatureGrid(config, frames, out, mono.format, buffer.started_at_ns, source_frames)
