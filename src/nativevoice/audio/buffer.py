"""Bounded audio buffering for the streaming path.

The live path is producer-consumer with mismatched rates: the device callback
arrives on a timer, the engine consumes when it can. An unbounded queue here
turns a slow engine into unbounded memory, and a bounded queue that silently
drops frames turns a slow engine into missing speech. Both are handled by
making the loss explicit: `overflow_frames` is a counter the session can read,
and `pre_rolls` keeps a bounded look-back so a segment started after speech
onset still contains its first phonemes.
"""

from __future__ import annotations

import array
import threading
import time

from ..core.audio import AudioBuffer, AudioFormat
from ..core.errors import ConfigurationError


class RingBuffer:
    """Thread-safe FIFO of PCM16 frames with a hard frame ceiling."""

    def __init__(self, format: AudioFormat, max_frames: int) -> None:
        if max_frames <= 0:
            raise ConfigurationError("max_frames must be positive", max_frames=max_frames)
        self.format = format
        self.max_frames = max_frames
        self._data = array.array("h")
        self._start = 0
        self._lock = threading.Lock()
        self.overflow_frames = 0
        self._last_write_ns = 0

    @property
    def stride(self) -> int:
        return self.format.samples_per_frame

    @property
    def available_frames(self) -> int:
        with self._lock:
            return (len(self._data) - self._start) // self.stride

    @property
    def oldest_age_s(self) -> float:
        with self._lock:
            available = (len(self._data) - self._start) // self.stride
            if available == 0 or self._last_write_ns == 0:
                return 0.0
            newest_s = available / self.format.sample_rate
            return max(0.0, (time.monotonic_ns() - self._last_write_ns) / 1e9 + newest_s)

    def append(self, buffer: AudioBuffer) -> int:
        """Push frames, returning how many were retained. Drops oldest on overflow."""
        if buffer.format != self.format:
            raise ConfigurationError("buffer format mismatch", got=buffer.format, want=self.format)
        incoming = buffer.samples
        with self._lock:
            self._data.extend(incoming)
            self._last_write_ns = time.monotonic_ns()
            capacity = self.max_frames * self.stride
            surplus = len(self._data) - self._start - capacity
            dropped = 0
            if surplus > 0:
                surplus -= surplus % self.stride
                self._start += surplus
                dropped = surplus // self.stride
                self.overflow_frames += dropped
            return max(0, len(incoming) // self.stride - dropped)

    def peek_frames(self, frames: int) -> AudioBuffer:
        with self._lock:
            available = (len(self._data) - self._start) // self.stride
            take = min(frames, available)
            payload = self._data[self._start : self._start + take * self.stride]
        return AudioBuffer(array.array("h", payload), self.format)

    def drain(self, frames: int) -> AudioBuffer:
        with self._lock:
            available = (len(self._data) - self._start) // self.stride
            take = min(frames, available)
            end = self._start + take * self.stride
            payload = array.array("h", self._data[self._start : end])
            self._start = end
            if self._start == len(self._data):
                self._data = array.array("h")
                self._start = 0
        return AudioBuffer(payload, self.format)

    def drain_all(self) -> AudioBuffer:
        return self.drain(self.available_frames)

    def clear(self) -> None:
        with self._lock:
            self._data = array.array("h")
            self._start = 0


class PreRoll:
    """Fixed-length look-back of the most recent audio.

    Speech-start detection is never instant, so the first 200-500 ms of an
    utterance is already gone by the time it is acted on. Retaining a bounded
    tail costs one buffer and recovers the onset.
    """

    def __init__(self, format: AudioFormat, duration_s: float = 0.4) -> None:
        self._ring = RingBuffer(format, max(1, int(duration_s * format.sample_rate)))

    @property
    def format(self) -> AudioFormat:
        return self._ring.format

    def push(self, buffer: AudioBuffer) -> None:
        self._ring.append(buffer)

    def snapshot(self) -> AudioBuffer:
        return self._ring.peek_frames(self._ring.available_frames)

    def clear(self) -> None:
        self._ring.clear()
