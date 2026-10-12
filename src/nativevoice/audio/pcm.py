"""PCM measurement and sample-domain operations (stdlib only).

`audioop` was removed from the standard library in 3.13, so the amplitude maths
this stack needs is implemented here instead. Everything operates on the
interleaved int16 `array` of `core.AudioBuffer`.
"""

from __future__ import annotations

import array
import math

from ..core.audio import AudioBuffer, AudioFormat, AudioLevel
from ..core.errors import ConfigurationError

FULL_SCALE = 32768.0


def to_float_samples(buffer: AudioBuffer) -> array.array:
    return array.array("d", (sample / FULL_SCALE for sample in buffer.samples))


def from_float_samples(values, format: AudioFormat, started_at_ns: int = 0) -> AudioBuffer:
    clipped = array.array("h", (max(-32768, min(32767, int(round(v * FULL_SCALE)))) for v in values))
    return AudioBuffer(clipped, format, started_at_ns)


def rms(values) -> float:
    """Root-mean-square in full-scale units, 0.0 for an empty span."""
    count = 0
    total = 0.0
    for value in values:
        total += value * value
        count += 1
    if not count:
        return 0.0
    return math.sqrt(total / count) / FULL_SCALE


def peak(values) -> float:
    highest = 0
    for value in values:
        magnitude = -value if value < 0 else value
        if magnitude > highest:
            highest = magnitude
    return highest / FULL_SCALE


def measure(buffer: AudioBuffer, silence_threshold: float = 0.01, clip_threshold: float = 0.999) -> AudioLevel:
    """Amplitude description used by recording-quality gating and the energy VAD.

    `clipped_fraction` counts consecutive saturated frames rather than single
    samples: one stray sample is a transient, a run of them is a bad recording.
    """
    samples = buffer.samples
    total = 0.0
    highest = 0
    count = len(samples)
    clipped_frames = 0
    silent_frames = 0
    run = 0
    stride = buffer.format.samples_per_frame
    for index in range(0, count, stride):
        frame_peak = 0
        frame_total = 0.0
        for channel in range(stride):
            value = samples[index + channel]
            frame_total += value * value
            magnitude = -value if value < 0 else value
            if magnitude > frame_peak:
                frame_peak = magnitude
        frame_rms = math.sqrt(frame_total / stride) / FULL_SCALE
        if frame_peak >= clip_threshold * FULL_SCALE - 1:
            run += 1
            if run >= 2:
                clipped_frames += 1
        else:
            run = 0
        if frame_rms < silence_threshold:
            silent_frames += 1
        total += frame_total
        if frame_peak > highest:
            highest = frame_peak
    frame_count = max(1, count // stride)
    return AudioLevel(
        rms=math.sqrt(total / max(1, count)) / FULL_SCALE,
        peak=highest / FULL_SCALE,
        clipped_fraction=clipped_frames / frame_count,
        silence_fraction=silent_frames / frame_count,
    )


def gain(buffer: AudioBuffer, factor: float) -> AudioBuffer:
    scaled = array.array("h", (max(-32768, min(32767, int(round(sample * factor)))) for sample in buffer.samples))
    return AudioBuffer(scaled, buffer.format, buffer.started_at_ns, buffer.source)


def normalize_peak(buffer: AudioBuffer, target: float = 0.95, amplify: bool = True) -> AudioBuffer:
    current = peak(buffer.samples)
    if current <= 1e-9:
        return buffer
    factor = target / current
    if not amplify and factor > 1.0:
        return buffer
    return gain(buffer, factor)


def fade_edges(buffer: AudioBuffer, fade_s: float = 0.005) -> AudioBuffer:
    """Short fades at both ends.

    A codec decoder's first sample frequently lands off-zero, which is heard as
    a click; the reference TTS studied patches this with exactly this 5 ms fade.
    Cheap, and prevents an artifact that quality metrics then fail to explain.
    """
    stride = buffer.format.samples_per_frame
    fade_frames = min(int(fade_s * buffer.format.sample_rate), buffer.frame_count // 2)
    if fade_frames <= 0:
        return buffer
    data = array.array("h", buffer.samples)
    last_frame = buffer.frame_count - 1
    for index in range(fade_frames):
        rising = index / fade_frames
        falling = 1.0 - rising
        for channel in range(stride):
            data[index * stride + channel] = int(data[index * stride + channel] * rising)
            tail = (last_frame - fade_frames + 1 + index) * stride + channel
            data[tail] = int(data[tail] * falling)
    return AudioBuffer(data, buffer.format, buffer.started_at_ns, buffer.source)


def remove_dc(buffer: AudioBuffer, coefficient: float = 0.995) -> AudioBuffer:
    """One-pole DC blocker.

    Condenser capture chains commonly sit a few samples off centre; a DC offset
    biases every RMS decision downstream, including the energy gate.
    """
    data = array.array("h", buffer.samples)
    state = [0.0] * buffer.format.channels
    previous = [0.0] * buffer.format.channels
    stride = buffer.format.samples_per_frame
    for index in range(0, len(data), stride):
        for channel in range(stride):
            position = index + channel
            sample = data[position]
            state[channel] = coefficient * (state[channel] + sample - previous[channel])
            previous[channel] = sample
            data[position] = max(-32768, min(32767, int(round(state[channel]))))
    return AudioBuffer(data, buffer.format, buffer.started_at_ns, buffer.source)


def crossfade(left: AudioBuffer, right: AudioBuffer, overlap_s: float = 0.0) -> AudioBuffer:
    if left.format != right.format:
        raise ConfigurationError("crossfade requires matching formats", left=left.format, right=right.format)
    if overlap_s <= 0:
        return left.appended(right)
    stride = left.format.samples_per_frame
    overlap = min(int(overlap_s * left.format.sample_rate), left.frame_count, right.frame_count)
    if overlap <= 0:
        return left.appended(right)
    head = array.array("h", left.samples[: (left.frame_count - overlap) * stride])
    tail_start = left.frame_count - overlap
    merged = array.array("h", head)
    for index in range(overlap):
        factor = index / overlap
        for channel in range(stride):
            first = left.samples[(tail_start + index) * stride + channel]
            second = right.samples[index * stride + channel]
            merged.append(max(-32768, min(32767, int(first * (1 - factor) + second * factor))))
    merged.extend(right.samples[overlap * stride :])
    return AudioBuffer(merged, left.format, left.started_at_ns, left.source or right.source)
