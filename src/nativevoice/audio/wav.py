"""Minimal RIFF/WAVE reader and writer.

`wave` is on the standard library's deprecation list (PEP 748) and its removal
would break recording import, so the container parser is written directly. It is
short, handles PCM8/16/32 and IEEE float32, and rejects what it cannot decode
with a specific error instead of producing silence.

Compressed containers (mp3, m4a, ogg) are not supported here by design: the
usual route is an external ffmpeg binary, and shipping one is a provenance
decision that has not been approved. Import of such files is therefore an
explicit gap, not a silent fallback.
"""

from __future__ import annotations

import array
import os
import struct
from dataclasses import dataclass

from ..core.audio import AudioBuffer, AudioFormat
from ..core.errors import ConfigurationError, StorageError

CODEC_PCM = 1
CODEC_FLOAT = 3


@dataclass(frozen=True, slots=True)
class WavInfo:
    sample_rate: int
    channels: int
    frame_count: int
    codec: str
    bits_per_sample: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.sample_rate


def _chunks(payload: bytes):
    position = 0
    while position + 8 <= len(payload):
        label, size = struct.unpack_from("<4sI", payload, position)
        body = payload[position + 8 : position + 8 + size]
        yield label, body
        position += 8 + size + (size % 2)


def _parse(payload: bytes, path: str = "") -> tuple[bytes, bytes]:
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise StorageError("not a RIFF/WAVE file", path=path)
    fmt = None
    data = b""
    for label, body in _chunks(payload[12:]):
        if label == b"fmt ":
            fmt = body
        elif label == b"data":
            data = body
    if fmt is None or len(fmt) < 16:
        raise StorageError("WAVE file has no fmt chunk", path=path)
    return fmt, data


def _header(fmt: bytes, path: str = "") -> tuple[int, int, int, int, int]:
    codec, channels, rate, _byte_rate, align, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if (codec, bits) not in ((CODEC_PCM, 8), (CODEC_PCM, 16), (CODEC_PCM, 32), (CODEC_FLOAT, 32)):
        raise ConfigurationError(
            "unsupported WAVE encoding; decode it with an approved tool first",
            path=path,
            codec=codec,
            bits=bits,
        )
    return codec, channels, rate, align, bits


def inspect(path: str) -> WavInfo:
    with open(path, "rb") as handle:
        payload = handle.read()
    fmt, data = _parse(payload, path)
    codec, channels, rate, align, bits = _header(fmt, path)
    stride = max(1, align)
    if stride != channels * bits // 8:
        raise StorageError("WAVE block alignment disagrees with layout", path=path, align=stride)
    return WavInfo(rate, channels, len(data) // stride, "float32" if codec == CODEC_FLOAT else "pcm", bits)


def read_wav(path: str) -> AudioBuffer:
    if not os.path.exists(path):
        raise StorageError("WAVE file not found", path=path)
    with open(path, "rb") as handle:
        payload = handle.read()
    fmt, data = _parse(payload, path)
    codec, channels, rate, align, bits = _header(fmt, path)
    format = AudioFormat(rate, channels, 16)
    if codec == CODEC_FLOAT:
        values = array.array("f")
        values.frombytes(data[: len(data) - (len(data) % 4)])
        samples = array.array("h", (max(-32768, min(32767, int(round(v * 32767)))) for v in values))
        return AudioBuffer(samples, format)
    if bits == 16:
        samples = array.array("h")
        samples.frombytes(data[: len(data) - (len(data) % 2)])
    elif bits == 8:
        samples = array.array("h", ((value - 128) * 256 for value in data))
    else:
        raw = array.array("i")
        raw.frombytes(data[: len(data) - (len(data) % 4)])
        samples = array.array("h", (max(-32768, min(32767, value >> 16)) for value in raw))
    return AudioBuffer(samples, format)


def write_wav(path: str, buffer: AudioBuffer) -> str:
    if buffer.format.sample_width_bits != 16 or buffer.format.channels > 2:
        raise ConfigurationError("WAVE writer emits 16-bit mono or stereo", format=buffer.format)
    payload = buffer.to_bytes()
    frame = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(payload),
        b"WAVE",
        b"fmt ",
        16,
        CODEC_PCM,
        buffer.format.channels,
        buffer.format.sample_rate,
        buffer.format.sample_rate * buffer.format.bytes_per_frame,
        buffer.format.bytes_per_frame,
        16,
        b"data",
        len(payload),
    )
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(frame)
        handle.write(payload)
    return path
