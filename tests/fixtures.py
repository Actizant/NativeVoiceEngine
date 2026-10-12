"""Synthetic fixtures.

These are tone and envelope shapes, not speech. They exist to exercise transport,
buffering, cancellation and metric plumbing deterministically. They must never
be used to claim recognition or synthesis quality - every quality test that needs
real speech is marked and reports NOT RUN until approved audio exists.
"""

from __future__ import annotations

import array
import math

import pytest

from nativevoice.core.audio import AudioBuffer, AudioFormat


def sine(rate: int = 16000, seconds: float = 0.25, frequency: float = 220.0, amplitude: float = 0.3, channels: int = 1, phase: float = 0.0) -> AudioBuffer:
    count = int(rate * seconds)
    peak_value = amplitude * 32767
    samples = array.array("h")
    for index in range(count):
        value = int(peak_value * math.sin(2 * math.pi * frequency * index / rate + phase))
        for _ in range(channels):
            samples.append(value)
    return AudioBuffer(samples, AudioFormat(rate, channels))


def silence(rate: int = 16000, seconds: float = 0.25, channels: int = 1) -> AudioBuffer:
    return AudioBuffer.zeros(AudioFormat(rate, channels), int(rate * seconds))


def speech_like(rate: int = 16000, bursts: tuple[tuple[float, float], ...] = ((0.1, 0.35), (0.6, 0.9)), amplitude: float = 0.25) -> AudioBuffer:
    """Tone bursts separated by silence: a stand-in for utterance timing.

    Deliberately includes vibrato and harmonics so RMS gating is not testing a
    signal that no real capture could produce.
    """
    count = int(rate * (bursts[-1][1] + 0.3))
    samples = array.array("h")
    for index in range(count):
        moment = index / rate
        envelope = 0.0
        for start, end in bursts:
            if start <= moment <= end:
                edge = min(moment - start, end - moment, 0.03) / 0.03
                envelope = max(envelope, max(0.0, edge))
        if envelope <= 0:
            samples.append(0)
            continue
        vibrato = 1.0 + 0.02 * math.sin(2 * math.pi * 5.0 * moment)
        value = (
            math.sin(2 * math.pi * 130.0 * vibrato * moment) * 0.6
            + math.sin(2 * math.pi * 260.0 * vibrato * moment) * 0.25
            + math.sin(2 * math.pi * 520.0 * moment) * 0.1
        )
        samples.append(int(max(-1.0, min(1.0, value * envelope * amplitude)) * 32767))
    return AudioBuffer(samples, AudioFormat(rate, 1))


def noise(rate: int = 16000, seconds: float = 0.2, amplitude: float = 0.02, seed: int = 1234) -> AudioBuffer:
    state = seed & 0xFFFFFFFF
    count = int(rate * seconds)
    samples = array.array("h")
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        samples.append(int(((state / 0x7FFFFFFF) - 0.5) * 2 * amplitude * 32767))
    return AudioBuffer(samples, AudioFormat(rate, 1))


@pytest.fixture
def dc_offset_capture() -> AudioBuffer:
    """A capture sitting 1200 samples off centre - what a cheap mic actually produces."""
    base = sine(seconds=0.2, amplitude=0.05)
    biased = array.array("h", (max(-32768, min(32767, value + 1200)) for value in base.samples))
    return AudioBuffer(biased, base.format)
