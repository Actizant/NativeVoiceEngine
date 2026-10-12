"""Capture-side preprocessing chain.

Ordering is fixed and matters: DC removal before level measurement, measurement
before gain, resampling last so the rate conversion sees a clean signal. The
chain records what it did so a transcript can be explained by the audio that was
actually fed to the engine rather than by the audio the microphone produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.audio import AudioBuffer, AudioFormat
from .pcm import measure, normalize_peak, remove_dc
from .resample import MODE_BAND_LIMITED, resample


@dataclass(slots=True)
class PreprocessReport:
    stages: list[str] = field(default_factory=list)
    input_level: float = 0.0
    output_level: float = 0.0
    clipped_fraction: float = 0.0
    silence_fraction: float = 0.0
    duration_s: float = 0.0


@dataclass(slots=True)
class Preprocessor:
    """Brings arbitrary captured audio to the engine's canonical form."""

    target_format: AudioFormat = field(default_factory=lambda: AudioFormat(16000, 1))
    dc_blocker: bool = True
    normalize: bool = True
    peak_target: float = 0.95
    amplify: bool = False
    resample_mode: str = MODE_BAND_LIMITED

    def process(self, buffer: AudioBuffer, report: PreprocessReport | None = None) -> tuple[AudioBuffer, PreprocessReport]:
        result = PreprocessReport() if report is None else report
        working = buffer
        result.input_level = measure(working).rms
        if working.format.channels > 1:
            working = working.to_mono()
            result.stages.append("mono")
        if self.dc_blocker:
            working = remove_dc(working)
            result.stages.append("dc_block")
        if self.normalize:
            before = measure(working)
            working = normalize_peak(working, self.peak_target, amplify=self.amplify)
            if before.rms > 0:
                result.stages.append("normalize")
        if working.format.sample_rate != self.target_format.sample_rate:
            working = resample(working, self.target_format.sample_rate, mode=self.resample_mode)
            result.stages.append(f"resample:{self.resample_mode}")
        if working.format.channels != self.target_format.channels:
            working = working.to_mono()
            result.stages.append("mono")
        level = measure(working)
        result.output_level = level.rms
        result.clipped_fraction = level.clipped_fraction
        result.silence_fraction = level.silence_fraction
        result.duration_s = working.duration_s
        return working, result


def trim_silence(buffer: AudioBuffer, threshold: float = 0.01, keep: float = 0.05) -> AudioBuffer:
    """Drop leading and trailing near-silence, keeping a short margin.

    Engines bill and score by decoded duration, so a two-second dead-air prefix
    costs latency without adding information. The margin exists because trimming
    to the exact onset clips the first consonant.
    """
    stride = buffer.format.samples_per_frame
    margin = int(keep * buffer.format.sample_rate)
    from .pcm import rms

    frames = buffer.frame_count
    first = 0
    last = frames
    for index in range(frames):
        base = index * stride
        if rms(buffer.samples[base : base + stride]) >= threshold:
            first = max(0, index - margin)
            break
    for index in range(frames - 1, -1, -1):
        base = index * stride
        if rms(buffer.samples[base : base + stride]) >= threshold:
            last = min(frames, index + 1 + margin)
            break
    return buffer.sliced(first, last)
