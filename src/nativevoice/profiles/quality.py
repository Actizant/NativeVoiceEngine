"""Recording-quality gating for candidate voice profiles.

Why a gate exists at all: a conditioning asset built from a clipped, silent or
single-tone recording does not fail loudly - it produces a voice that quietly
tracks the artifact. Rejecting at intake is the only cheap place to catch it.

The metrics are `audio.pcm.measure` plus two derived statistics, chosen because
each one kills a specific failure mode that a plain RMS check cannot see:

* `snr_db` - a percentile split of short-frame energy: the bottom 30% of frames is
  treated as the noise floor and the top 30% as speech. A tone or steady hiss has
  no such split (ratio ~1), so it fails, while real speech with pauses has a large
  one.
* `energy_contrast` - p90 minus p10 of frame energy. This is the pure-tone and
  white-noise detector: a constant-envelope signal has essentially none.

Thresholds are PROVISIONAL. They were measured against this repository's synthetic
fixtures (see the docstring on `PROVISIONAL_MINIMUMS`), and `speech_like` is tone
bursts, not speech. They must be re-measured on approved real recordings before a
profile is accepted in production; the report says so instead of implying a
clinical basis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..audio.pcm import measure, rms
from ..core.audio import AudioBuffer
from ..core.errors import ConfigurationError

FRAME_S = 0.025
_HISTOGRAM_FLOOR = 0.30
_HISTOGRAM_SPEECH = 0.30
_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class QualityMinimums:
    """The line a candidate recording has to clear."""

    duration_s: float
    min_rms: float
    max_rms: float
    max_clipped_fraction: float
    max_silence_fraction: float
    min_snr_db: float
    min_energy_contrast: float
    label: str = "provisional-until-measured-on-real-speech"

    def explain(self) -> str:
        return (
            f"{self.label}: duration>={self.duration_s}s, rms in [{self.min_rms}, {self.max_rms}], "
            f"clipped<={self.max_clipped_fraction}, silence<={self.max_silence_fraction}, "
            f"snr>={self.min_snr_db}dB, energy contrast>={self.min_energy_contrast}"
        )


#: Set from measurements on tests/fixtures.py at 16 kHz:
#:   speech_like  rms 0.073, silence 0.588, contrast 0.118, snr ~inf (silent gaps)
#:   sine 2 s     rms 0.212, silence 0.022, contrast 0.000, snr 0.0 dB
#:   noise 2 s    rms 0.012, silence 0.496, contrast 0.001, snr 0.5 dB
#:   silence 2 s  rms 0.000, silence 1.000, contrast 0.000
#: The bands below sit between those clusters with margin, and are wrong until a
#: real corpus says otherwise.
PROVISIONAL_MINIMUMS = QualityMinimums(
    duration_s=0.8,
    min_rms=0.02,
    max_rms=0.45,
    max_clipped_fraction=0.005,
    max_silence_fraction=0.75,
    min_snr_db=6.0,
    min_energy_contrast=0.03,
)


@dataclass(frozen=True, slots=True)
class QualityCheck:
    name: str
    value: float
    threshold: float
    passed: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "threshold": round(self.threshold, 6),
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Measurements, per-check verdicts, and the reason behind each verdict."""

    duration_s: float
    rms: float
    peak: float
    clipped_fraction: float
    silence_fraction: float
    snr_db: float
    energy_contrast: float
    checks: tuple[QualityCheck, ...]
    minimums: QualityMinimums

    @property
    def usable(self) -> bool:
        return all(check.passed for check in self.checks) and bool(self.checks)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(check.reason for check in self.checks if not check.passed)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed)

    def as_dict(self) -> dict:
        return {
            "usable": self.usable,
            "duration_s": round(self.duration_s, 4),
            "rms": round(self.rms, 6),
            "peak": round(self.peak, 6),
            "clipped_fraction": round(self.clipped_fraction, 6),
            "silence_fraction": round(self.silence_fraction, 6),
            "snr_db": None if math.isinf(self.snr_db) else round(self.snr_db, 3),
            "energy_contrast": round(self.energy_contrast, 6),
            "checks": [check.as_dict() for check in self.checks],
            "minimums": self.minimums.explain(),
        }


def frame_energies(buffer: AudioBuffer, frame_s: float = FRAME_S) -> list[float]:
    window = max(1, int(frame_s * buffer.format.sample_rate) * buffer.format.samples_per_frame)
    samples = buffer.samples
    return [
        rms(samples[start : start + window])
        for start in range(0, max(1, len(samples) - window + 1), window)
    ]


def estimate_snr_db(buffer: AudioBuffer) -> float:
    """Speech-to-noise from a percentile split of frame energies.

    The split is deliberately crude: without a VAD model there is no labelled
    speech/noise partition, so the quietest 30% of frames stand in for the noise
    floor and the loudest 30% for speech. It reports a *very* high number for a
    recording with real silence in it and a near-zero number for tone or hiss,
    which is exactly the discrimination intake needs.
    """
    ordered = sorted(frame_energies(buffer))
    if not ordered:
        return 0.0
    take = max(1, int(len(ordered) * _HISTOGRAM_FLOOR))
    floor = rms(ordered[:take])
    speech = rms(ordered[len(ordered) - max(1, int(len(ordered) * _HISTOGRAM_SPEECH)) :])
    if floor <= _EPSILON:
        return float("inf") if speech > _EPSILON else 0.0
    return 20.0 * math.log10(speech / floor)


def energy_contrast(buffer: AudioBuffer) -> float:
    """p90 minus p10 of frame energy; the constant-envelope detector."""
    ordered = sorted(frame_energies(buffer))
    if not ordered:
        return 0.0
    low = ordered[max(0, int(len(ordered) * 0.10))]
    high = ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))]
    return high - low


def assess(buffer: AudioBuffer, minimums: QualityMinimums = PROVISIONAL_MINIMUMS) -> QualityReport:
    """Measure one candidate recording against the intake minimums."""
    if buffer.frame_count == 0:
        raise ConfigurationError("cannot assess empty audio", frames=0)
    level = measure(buffer)
    snr = estimate_snr_db(buffer)
    contrast = energy_contrast(buffer)
    checks = (
        _check(
            "duration", buffer.duration_s, minimums.duration_s,
            buffer.duration_s >= minimums.duration_s, "below minimum duration for conditioning",
        ),
        _check(
            "loudness_floor", level.rms, minimums.min_rms, level.rms >= minimums.min_rms,
            "too quiet: the conditioner would learn the noise floor",
        ),
        _check(
            "loudness_ceiling", level.rms, minimums.max_rms, level.rms <= minimums.max_rms,
            "too hot for a representative level",
        ),
        _check(
            "clipping", level.clipped_fraction, minimums.max_clipped_fraction,
            level.clipped_fraction <= minimums.max_clipped_fraction,
            "saturated runs present; harmonics would be invented by the model",
        ),
        _check(
            "silence", level.silence_fraction, minimums.max_silence_fraction,
            level.silence_fraction <= minimums.max_silence_fraction,
            "mostly silence: no speaker to condition on",
        ),
        _check(
            "signal_to_noise", snr, minimums.min_snr_db, snr >= minimums.min_snr_db,
            "no speech/noise separation found: constant envelope suggests tone or hiss",
        ),
        _check(
            "energy_contrast", contrast, minimums.min_energy_contrast,
            contrast >= minimums.min_energy_contrast,
            "flat energy profile: a pure tone or steady noise, not an utterance",
        ),
    )
    return QualityReport(
        duration_s=buffer.duration_s,
        rms=level.rms,
        peak=level.peak,
        clipped_fraction=level.clipped_fraction,
        silence_fraction=level.silence_fraction,
        snr_db=snr,
        energy_contrast=contrast,
        checks=checks,
        minimums=minimums,
    )


def _check(name: str, value: float, threshold: float, passed: bool, failure: str) -> QualityCheck:
    return QualityCheck(
        name=name,
        value=value,
        threshold=threshold,
        passed=passed,
        reason="" if passed else f"{name}: {value:.4f} vs {threshold:.4f} - {failure}",
    )
