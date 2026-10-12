"""Sample-rate conversion.

Two modes, chosen deliberately rather than automatically:

`band_limited` - windowed-sinc polyphase filter, correct for any rational rate
ratio, the default for anything whose audio will be judged (synthesis output,
evaluation fixtures, recording import).

`linear` - one multiply-accumulate per output sample. It aliases on
downsampling, which is audible and measurable, so it exists only as the
explicitly-labelled cheap tier for live capture on constrained machines.

Realtime capture at 16 kHz costs `O(taps * down/up)` Python operations per
output sample, which is why the mode is a user-visible choice instead of an
implementation detail. The measured cost is recorded in the benchmark report
rather than estimated here.
"""

from __future__ import annotations

import array
import math
from math import gcd

from ..core.audio import AudioBuffer, AudioFormat
from ..core.errors import ConfigurationError

MODE_BAND_LIMITED = "band_limited"
MODE_LINEAR = "linear"
DEFAULT_TAPS = 6


def resample(buffer: AudioBuffer, target_rate: int, *, mode: str = MODE_BAND_LIMITED, taps: int = DEFAULT_TAPS) -> AudioBuffer:
    source_rate = buffer.format.sample_rate
    if target_rate <= 0:
        raise ConfigurationError("target rate must be positive", target_rate=target_rate)
    if mode not in (MODE_LINEAR, MODE_BAND_LIMITED):
        raise ConfigurationError("unknown resample mode", mode=mode)
    if taps < 1:
        raise ConfigurationError("taps must be >= 1", taps=taps)
    if target_rate == source_rate:
        return buffer
    if mode == MODE_LINEAR:
        data = _linear(buffer.samples, source_rate, target_rate, buffer.format.channels)
    else:
        data = _polyphase(buffer.samples, source_rate, target_rate, buffer.format.channels, taps)
    return AudioBuffer(data, AudioFormat(target_rate, buffer.format.channels), buffer.started_at_ns, buffer.source)


def _linear(samples: array.array, source_rate: int, target_rate: int, channels: int) -> array.array:
    count = len(samples)
    frames_out = max(1, round(count / channels * target_rate / source_rate))
    step = source_rate / target_rate
    out = array.array("h", bytes(frames_out * channels * 2))
    last = count - channels
    for index in range(frames_out * channels):
        position = index * step
        whole = int(position)
        if whole >= last:
            out[index] = samples[min(count - 1, last + channels - 1)] if count else 0
            continue
        fraction = position - whole
        out[index] = int(round(samples[whole] * (1 - fraction) + samples[whole + channels] * fraction))
    return out


def _hann(weight: float) -> float:
    if abs(weight) >= 1.0:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * weight))


def _sinc(value: float) -> float:
    if value == 0.0:
        return 1.0
    scaled = math.pi * value
    return math.sin(scaled) / scaled


def _polyphase(samples: array.array, source_rate: int, target_rate: int, channels: int, taps: int) -> array.array:
    divisor = gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    if channels == 1:
        return array.array("h", _resample_channel(samples, up, down, taps))
    planar = [array.array("h") for _ in range(channels)]
    stride = channels
    for index in range(0, len(samples), stride):
        for channel in range(channels):
            planar[channel].append(samples[index + channel])
    out_channels = [_resample_channel(planar[channel], up, down, taps) for channel in range(channels)]
    frames_out = len(out_channels[0])
    interleaved = array.array("h", bytes(frames_out * channels * 2))
    for frame in range(frames_out):
        for channel in range(channels):
            interleaved[frame * channels + channel] = out_channels[channel][frame]
    return interleaved


def _resample_channel(samples: array.array, up: int, down: int, taps: int) -> array.array:
    """Band-limited resampling via a precomputed table of `up` phase filters."""
    count = len(samples)
    if count == 0:
        return array.array("h")
    frames_out = max(1, round(count * up / down))
    scale = max(1.0, down / up)
    half_width = max(1, int(math.ceil(taps * scale)))
    table: list[array.array] = []
    for phase in range(up):
        row = array.array("d", bytes((2 * half_width + 1) * 8))
        for offset in range(-half_width, half_width + 1):
            distance = (offset - phase / up) / scale
            weight = _sinc(distance) * _hann(distance / taps) / scale
            row[offset + half_width] = weight
        table.append(row)
    out = array.array("h", bytes(frames_out * 2))
    for index in range(frames_out):
        numerator = index * down
        position = numerator // up
        phase = numerator - position * up
        row = table[phase]
        start = position - half_width
        total = 0.0
        lower = start if start > 0 else 0
        upper = start + len(row)
        if upper > count:
            upper = count
        cursor = lower - start
        for column in range(cursor, cursor + (upper - lower)):
            total += row[column] * samples[lower + (column - cursor)]
        out[index] = max(-32768, min(32767, int(round(total))))
    return out
