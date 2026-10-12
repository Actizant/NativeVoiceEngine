"""Device capture and playback.

`sounddevice` is optional and imported lazily: the whole control path, the test
suite and every metric must run on a machine with no usable audio hardware,
because that is the common case in CI and on a laptop with a revoked microphone
permission. `WavFileInput` and `CapturingOutput` provide the deterministic ends
of the same contracts, so the streaming, cancellation and interruption tests
drive real code paths rather than mocks.

Failure model: a callback that stops arriving is treated as a disconnected
device rather than as silence. The reference stack documented a Teams capture
path that delivered digital silence for the whole meeting, which is exactly the
failure that looks like success. `capture_stall_s` makes that detectable.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Protocol

from ..core.audio import AudioBuffer, AudioFormat
from ..core.cancel import CancellationToken
from ..core.errors import DeviceDisconnected, DeviceError
from .resample import resample
from .wav import read_wav


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    index: int
    name: str
    channels: int
    sample_rate: int
    kind: str
    is_default: bool = False
    backend: str = ""


class AudioInput(Protocol):
    format: AudioFormat

    def open(self) -> None: ...

    def frames(self) -> Iterator[AudioBuffer]: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class WavFileInput:
    """Deterministic input used by tests, offline evaluation and file import."""

    path: str
    frame_size: int = 1600
    target_rate: int | None = None
    realtime_pacing: bool = False
    format: AudioFormat = field(default=AudioFormat(16000, 1), repr=False)
    _buffer: AudioBuffer | None = None
    _cursor: int = 0
    _opened: bool = False

    def open(self) -> None:
        self._buffer = read_wav(self.path)
        if self._buffer.format.channels > 1:
            self._buffer = self._buffer.to_mono()
        if self.target_rate and self._buffer.format.sample_rate != self.target_rate:
            self._buffer = resample(self._buffer, self.target_rate)
        self.format = self._buffer.format
        self._cursor = 0
        self._opened = True

    def frames(self) -> Iterator[AudioBuffer]:
        if not self._opened or self._buffer is None:
            raise DeviceError("input used before open()", path=self.path)
        while self._cursor < self._buffer.frame_count:
            chunk = self._buffer.sliced(self._cursor, self._cursor + self.frame_size)
            self._cursor += chunk.frame_count
            if self.realtime_pacing:
                time.sleep(chunk.duration_s)
            yield chunk

    def close(self) -> None:
        self._buffer = None
        self._opened = False


@dataclass(slots=True)
class SounddeviceInput:
    """Live microphone capture, resampled to the engine's canonical rate."""

    device: int | None = None
    target_rate: int = 16000
    frame_size: int = 1600
    channels: int = 1
    capture_stall_s: float = 2.0
    format: AudioFormat = field(default=AudioFormat(16000, 1), repr=False)

    _queue: "queue.SimpleQueue" = field(default_factory=queue.SimpleQueue, repr=False)
    _stream: object = None
    _last_callback_ns: int = 0
    _error: BaseException | None = None
    _closed: bool = False

    @property
    def available(self) -> bool:
        return _sounddevice() is not None

    def open(self) -> None:
        module = _sounddevice()
        if module is None:
            raise DeviceError("sounddevice is not installed; live capture is unavailable")
        self.format = AudioFormat(self.target_rate, self.channels)
        self._queue = queue.SimpleQueue()
        self._last_callback_ns = time.monotonic_ns()
        self._error = None
        self._closed = False

        def callback(indata, frames, time_info, status):
            self._last_callback_ns = time.monotonic_ns()
            if status:
                self._error = DeviceError("capture reported status", status=str(status))
            payload = bytes(indata[:frames, 0].tobytes()) if self.channels == 1 else bytes(indata.tobytes())
            self._queue.put((payload, frames))

        self._stream = module.InputStream(
            samplerate=self.target_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.frame_size,
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def frames(self) -> Iterator[AudioBuffer]:
        if self._stream is None:
            raise DeviceError("input used before open()")
        stride = self.format.bytes_per_frame
        assembled = bytearray()
        while not self._closed:
            silence_gap = (time.monotonic_ns() - self._last_callback_ns) / 1e9
            if silence_gap > self.capture_stall_s:
                raise DeviceDisconnected(
                    "no audio arriving from the capture device",
                    device=self.device,
                    silent_for_s=round(silence_gap, 3),
                )
            try:
                payload, _frames = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._error is not None:
                raise self._error
            assembled.extend(payload)
            while len(assembled) >= stride * self.frame_size:
                take = stride * self.frame_size
                chunk = bytes(assembled[:take])
                del assembled[:take]
                yield AudioBuffer.from_bytes(chunk, self.format)

    def close(self) -> None:
        self._closed = True
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


@dataclass(slots=True)
class SyntheticInput:
    """Feeds pre-built buffers on a thread; used to test capture failure paths."""

    buffers: list[AudioBuffer]
    format: AudioFormat = field(default=AudioFormat(16000, 1), repr=False)
    stall_after: int | None = None

    _consumed: int = 0

    def open(self) -> None:
        self._consumed = 0

    def frames(self) -> Iterator[AudioBuffer]:
        for index, buffer in enumerate(self.buffers):
            if self.stall_after is not None and index >= self.stall_after:
                raise DeviceDisconnected("capture stalled", after=index)
            self._consumed = index + 1
            yield buffer

    def close(self) -> None:
        pass


class AudioOutput(Protocol):
    def play(self, buffer: AudioBuffer, token: CancellationToken | None = None) -> int: ...

    def stop(self) -> None: ...

    @property
    def playing(self) -> bool: ...


@dataclass(slots=True)
class CapturingOutput:
    """A sink that records instead of emitting sound.

    Doubles as the `voice output disabled` path required by §9: text-only mode
    keeps the synthesis pipeline intact but routes it here, so disabling speech
    is a routing decision and not a code path that skips the checks.
    """

    played: list[AudioBuffer] = field(default_factory=list)
    frames_written: int = 0
    cancelled_spans: int = 0
    halts: int = 0
    _generation: int = 0
    _playing: bool = False

    def play(self, buffer: AudioBuffer, token: CancellationToken | None = None) -> int:
        """Emit a span, dropping it if it is superseded mid-flight.

        Two independent things can end a span: the caller's token (the user
        barged in on this answer) and a newer `stop()` (the session moved on).
        Both are observed per chunk, and neither permanently disables the sink -
        a device that cannot speak again after being interrupted is a worse bug
        than one that plays a stale syllable.
        """
        generation = self._generation
        self._playing = True
        emitted = 0
        try:
            for chunk in buffer.frames(max(1, buffer.format.sample_rate // 20)):
                if (token is not None and token.cancelled) or generation != self._generation:
                    self.cancelled_spans += 1
                    return emitted
                self.played.append(chunk)
                self.frames_written += chunk.frame_count
                emitted += chunk.frame_count
        finally:
            self._playing = False
        return emitted

    def stop(self) -> None:
        """Abandon whatever is playing right now. Later spans are allowed."""
        self._generation += 1
        self.halts += 1
        self._playing = False
        self.played.clear()
        self.frames_written = 0

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def audio(self) -> AudioBuffer | None:
        if not self.played:
            return None
        merged = self.played[0]
        for extra in self.played[1:]:
            merged = merged.appended(extra)
        return merged


@dataclass(slots=True)
class SounddeviceOutput:
    """Playback through the default rendering device with hard-stop semantics."""

    device: int | None = None
    blocksize: int = 2400
    _stream: object = None
    _generation: int = 0
    _opened_format: AudioFormat | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _playing: bool = False
    format: AudioFormat = field(default=AudioFormat(24000, 1), repr=False)

    @property
    def available(self) -> bool:
        return _sounddevice() is not None

    def _open(self, module, format: AudioFormat) -> None:
        """(Re)create a raw stream at the format about to be played.

        A raw output stream is not started by construction, and writing to a
        stopped stream raises; recreating on a format change is also what keeps a
        24 kHz answer from being fed to a stream opened for 16 kHz, which is heard
        as a pitch shift rather than as an error.
        """
        if self._stream is not None:
            if getattr(self, "_opened_format", None) == format:
                return
            self.close()
        self._stream = module.RawOutputStream(
            samplerate=format.sample_rate,
            channels=format.channels,
            dtype="int16",
            blocksize=self.blocksize,
            device=self.device,
        )
        self._stream.start()
        self._opened_format = format
        self.format = format

    def play(self, buffer: AudioBuffer, token: CancellationToken | None = None) -> int:
        module = _sounddevice()
        if module is None:
            raise DeviceError("sounddevice is not installed; playback is unavailable")
        emitted = 0
        generation = self._generation
        self._playing = True
        try:
            with self._lock:
                try:
                    self._open(module, buffer.format)
                except module.PortAudioError as error:
                    raise DeviceError("output stream could not be opened", error=str(error)) from error
                stream = self._stream
                assert stream is not None
                for chunk in buffer.frames(self.blocksize):
                    if token is not None and token.cancelled or generation != self._generation:
                        return emitted
                    try:
                        stream.write(chunk.to_bytes())
                    except module.PortAudioError as error:
                        # A barge-in from another thread aborts the stream while a
                        # write is in flight. That is the requested outcome, not a
                        # device fault, so a generation change ends the span quietly;
                        # only an unrequested failure is reported as a disconnection.
                        if (token is not None and token.cancelled) or generation != self._generation:
                            return emitted
                        self.close()
                        raise DeviceDisconnected("playback device stopped", error=str(error)) from error
                    emitted += chunk.frame_count
        finally:
            self._playing = False
        return emitted

    def stop(self) -> None:
        """Abandon the current span. The next span reopens and plays normally."""
        self._generation += 1
        self._playing = False
        self.close()

    def close(self) -> None:
        stream, self._stream = self._stream, None
        self._opened_format = None
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass

    @property
    def playing(self) -> bool:
        return self._playing


def list_devices() -> list[DeviceInfo]:
    module = _sounddevice()
    if module is None:
        return []
    default_input, default_output = module.default.device
    found: list[DeviceInfo] = []
    for index, device in enumerate(module.query_devices()):
        if device["max_input_channels"]:
            found.append(
                DeviceInfo(
                    index,
                    device["name"],
                    int(device["max_input_channels"]),
                    int(device["default_samplerate"]),
                    "input",
                    index == default_input,
                    device.get("hostapi", ""),
                )
            )
        if device["max_output_channels"]:
            found.append(
                DeviceInfo(
                    index,
                    device["name"],
                    int(device["max_output_channels"]),
                    int(device["default_samplerate"]),
                    "output",
                    index == default_output,
                    device.get("hostapi", ""),
                )
            )
    return found


_module_cache: object = ...


def _sounddevice():
    global _module_cache
    if _module_cache is ...:
        try:
            import sounddevice as module  # type: ignore[import-not-found]

            _module_cache = module
        except Exception:
            _module_cache = None
    return _module_cache
