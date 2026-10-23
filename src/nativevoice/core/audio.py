"""Audio data contracts.

Canonical in-memory representation is interleaved PCM16 little-endian. int16 was
chosen over float32 because capture, resampling and playback on the control path
all stay exact and dependency-free, and every engine family studied in the
reference projects ultimately consumes 16-bit PCM anyway.

Float lives at the boundary, not in this class: `AudioBuffer` deliberately exposes
no float method. Conversion is `nativevoice.audio.pcm.to_float_samples` (an
`array('d')` divided by 32768.0) for anything in the control path, and an engine
that needs a tensor converts inside its own adapter —
`stt/adapters/faster_whisper_engine.py` turns a buffer into float32 numpy, and
`audio/features.py` builds its log-mel grid in float64. Naming the helper here
rather than inventing a method keeps one rounding convention, not two.

`array` is used rather than `bytes` so that per-frame arithmetic never pays for a
decode/re-encode pair.
"""

from __future__ import annotations

import array
from dataclasses import dataclass, field
from typing import Iterator

from .errors import ConfigurationError

SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
SUPPORTED_SAMPLE_WIDTH_BITS = (16,)


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    channels: int = 1
    sample_width_bits: int = 16

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ConfigurationError("sample_rate must be positive", sample_rate=self.sample_rate)
        if self.channels < 1:
            raise ConfigurationError("channels must be >= 1", channels=self.channels)
        if self.sample_width_bits not in SUPPORTED_SAMPLE_WIDTH_BITS:
            raise ConfigurationError(
                "unsupported sample width", sample_width_bits=self.sample_width_bits
            )

    @property
    def bytes_per_frame(self) -> int:
        """Wire size of one frame. Use for byte buffers and WAVE block alignment."""
        return self.channels * self.sample_width_bits // 8

    @property
    def samples_per_frame(self) -> int:
        """Elements per frame inside an `array('h')`. Use for sample indexing.

        Kept distinct from `bytes_per_frame` on purpose: every stride bug in an
        audio stack is a confusion between these two, and for 16-bit PCM they
        differ by exactly a factor of two. Equals `channels` because int16 is
        the only supported width.
        """
        return self.channels * (self.sample_width_bits // 16)

    def is_allowed_preset(self) -> bool:
        """True when the rate is one the stack guarantees end-to-end."""
        return self.sample_rate in SUPPORTED_SAMPLE_RATES


CANONICAL_16K_MONO = AudioFormat(16000, 1)
CANONICAL_24K_MONO = AudioFormat(24000, 1)
CANONICAL_48K_MONO = AudioFormat(48000, 1)


@dataclass(slots=True)
class AudioBuffer:
    """A fixed span of interleaved PCM16 samples plus the format that gives them meaning.

    `started_at_ns` is a monotonic timestamp carried alongside the audio so that
    latency measurement does not have to guess where a chunk began.
    """

    samples: array.array
    format: AudioFormat
    started_at_ns: int = 0
    source: str = ""

    def __post_init__(self) -> None:
        if self.samples.typecode != "h":
            raise ConfigurationError("samples must be int16", typecode=self.samples.typecode)
        if len(self.samples) % self.format.samples_per_frame != 0:
            raise ConfigurationError(
                "sample count is not a whole number of frames",
                samples=len(self.samples),
                samples_per_frame=self.format.samples_per_frame,
            )

    @classmethod
    def from_frames(cls, frames: list[int] | array.array, format: AudioFormat, started_at_ns: int = 0) -> "AudioBuffer":
        return cls(array.array("h", frames), format, started_at_ns)

    @classmethod
    def zeros(cls, format: AudioFormat, frames: int) -> "AudioBuffer":
        return cls(array.array("h", bytes(frames * format.bytes_per_frame)), format)

    @property
    def frame_count(self) -> int:
        return len(self.samples) // self.format.samples_per_frame

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.format.sample_rate

    def is_silent(self) -> bool:
        return any(self.samples) is False

    def frame(self, index: int) -> array.array:
        stride = self.format.samples_per_frame
        return self.samples[index * stride : (index + 1) * stride]

    def channel(self, index: int) -> array.array:
        channels = self.format.channels
        if not 0 <= index < channels:
            raise ConfigurationError("channel index out of range", channel=index, channels=channels)
        return array.array("h", self.samples[index::channels])

    def to_mono(self) -> "AudioBuffer":
        if self.format.channels == 1:
            return self
        channels = self.format.channels
        mix = array.array("h", bytes(self.frame_count * 2))
        for frame_index in range(self.frame_count):
            base = frame_index * channels
            total = 0
            for channel in range(channels):
                total += self.samples[base + channel]
            mix[frame_index] = max(-32768, min(32767, total // channels))
        return AudioBuffer(mix, AudioFormat(self.format.sample_rate, 1), self.started_at_ns, self.source)

    def sliced(self, first_frame: int, last_frame: int) -> "AudioBuffer":
        stride = self.format.samples_per_frame
        lo = max(0, min(first_frame, self.frame_count))
        hi = max(lo, min(last_frame, self.frame_count))
        return AudioBuffer(self.samples[lo * stride : hi * stride], self.format, self.started_at_ns, self.source)

    def frames(self, size: int) -> Iterator["AudioBuffer"]:
        """Iterate fixed-size chunks; the last chunk may be shorter."""
        if size <= 0:
            raise ConfigurationError("chunk size must be positive", size=size)
        for start in range(0, max(self.frame_count, 1), size):
            if self.frame_count == 0:
                return
            yield self.sliced(start, min(start + size, self.frame_count))

    def appended(self, other: "AudioBuffer") -> "AudioBuffer":
        if other.format != self.format:
            raise ConfigurationError("cannot append mismatched formats", left=self.format, right=other.format)
        merged = array.array("h", self.samples)
        merged.extend(other.samples)
        return AudioBuffer(merged, self.format, self.started_at_ns or other.started_at_ns, self.source or other.source)

    def to_bytes(self) -> bytes:
        return self.samples.tobytes()

    @classmethod
    def from_bytes(cls, payload: bytes, format: AudioFormat, started_at_ns: int = 0) -> "AudioBuffer":
        if len(payload) % format.bytes_per_frame:
            raise ConfigurationError(
                "payload is not a whole number of frames",
                bytes=len(payload),
                bytes_per_frame=format.bytes_per_frame,
            )
        data = array.array("h")
        data.frombytes(payload)
        if data.itemsize != 2:
            raise ConfigurationError("platform int16 width unexpected", itemsize=data.itemsize)
        return cls(data, format, started_at_ns)

    def __len__(self) -> int:
        return self.frame_count


@dataclass(frozen=True, slots=True)
class AudioLevel:
    """Measured amplitude description of a buffer, used by quality gating."""

    rms: float
    peak: float
    clipped_fraction: float
    silence_fraction: float


@dataclass(slots=True)
class Recognition:
    """One engine's answer for a time span of audio."""

    text: str
    language: str = ""
    confidence: float | None = None
    first_sample: int = 0
    last_sample: int = 0
    is_final: bool = False
    alternatives: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class SynthesisResult:
    """Audio produced for one text span, with the timings needed for honest reporting."""

    audio: AudioBuffer
    language: str = ""
    voice_id: str = ""
    first_audio_ns: int = 0
    total_ns: int = 0
    sampled: bool = False
    metadata: dict = field(default_factory=dict)
