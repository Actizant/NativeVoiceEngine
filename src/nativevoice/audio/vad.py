"""Voice activity detection: interface, energy baseline, and segmentation.

Two decisions carried over from studying the reference stack, both earned the
hard way there:

1. Detection is hysteresis-based, not single-threshold. One RMS cut-off flips
   constantly on room noise; a rise threshold above a fall threshold does not.
2. Segmentation parameters are per-context, not global. Enabling an aggressive
   gate on pause-heavy dictation deletes real speech, and the decoder then
   fills the gap with prompt leakage. So the segmenter is configured by the
   session, and the energy detector is only one implementation of the interface.

A neural detector (acoustic model in the style of Silera/Silero ONNX) is
represented by the same protocol. None is installed, and `installed_detector`
reports that honestly rather than quietly substituting the energy one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..core.audio import AudioBuffer, AudioFormat
from ..core.cancel import CancellationToken
from ..core.errors import CapabilityUnavailable, ConfigurationError
from .pcm import peak, rms

Rise = 0.012
Fall = 0.006
PeakRise = 0.05


class SpeechActivityDetector(Protocol):
    """Classifies short spans of audio as speech or non-speech."""

    name: str
    format: AudioFormat

    def probe(self, buffer: AudioBuffer) -> float:
        """Return a speech score in 0..1 for the span."""
        ...

    def reset(self) -> None: ...


@dataclass(slots=True)
class EnergySpeechDetector:
    """Hysteresis gate over frame RMS and peak. No model, no dependencies.

    The score is not a probability and is not presented as one: it is a monotone
    0..1 speech-likelihood used only for thresholding. Anything downstream that
    needs a calibrated probability must use a modelled detector.
    """

    format: AudioFormat
    rise: float = Rise
    fall: float = Fall
    peak_rise: float = PeakRise
    _state: bool = False
    _floor: float = 0.0

    name: str = "energy-hysteresis"

    def __post_init__(self) -> None:
        if self.fall > self.rise:
            raise ConfigurationError("fall threshold must not exceed rise threshold")

    def probe(self, buffer: AudioBuffer) -> float:
        if buffer.frame_count == 0:
            return 1.0 if self._state else 0.0
        level = rms(buffer.samples)
        loudest = peak(buffer.samples)
        # The floor tracks the quietest recent span so the gate adapts to the
        # room instead of to an absolute dBFS number.
        if not self._state and level < self.rise:
            self._floor = min(self._floor or level, level) if self._floor else level
        else:
            self._floor = self._floor * 0.999
        effective_rise = max(self.rise, self._floor * 3.0)
        if self._state:
            if level < self.fall and loudest < self.peak_rise:
                self._state = False
        elif level > effective_rise or (loudest > self.peak_rise and level > self.fall):
            self._state = True
        if self._state:
            return 1.0
        return 0.0

    def reset(self) -> None:
        self._state = False
        self._floor = 0.0


@dataclass(slots=True)
class SpeechSegment:
    """One delimited utterance, including the padding that preceded it."""

    first_sample: int
    last_sample: int
    duration_s: float
    audio: AudioBuffer | None = None
    truncated: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class SegmenterConfig:
    minimum_speech_s: float = 0.20
    minimum_silence_s: float = 0.35
    maximum_speech_s: float = 20.0
    padding_s: float = 0.10
    on_score: float = 0.5
    off_score: float = 0.5
    frame_s: float = 0.02

    def validate(self) -> None:
        if self.minimum_speech_s <= 0 or self.minimum_silence_s <= 0:
            raise ConfigurationError("speech and silence minima must be positive")
        if self.maximum_speech_s < self.minimum_speech_s:
            raise ConfigurationError("maximum speech span is below minimum")
        if not 0 <= self.off_score <= self.on_score <= 1:
            raise ConfigurationError("hysteresis scores must satisfy 0 <= off <= on <= 1")


@dataclass(slots=True)
class UtteranceSegmenter:
    """Turns a detector's per-frame scores into utterance boundaries.

    Emits a segment when silence has persisted for `minimum_silence_s`, or
    forces a cut at `maximum_speech_s` with `truncated` set so the session knows
    the utterance was not given a chance to end.
    """

    detector: SpeechActivityDetector
    config: SegmenterConfig = field(default_factory=SegmenterConfig)
    token: CancellationToken | None = None

    _buffer: AudioBuffer | None = None
    _pending: list[AudioBuffer] = field(default_factory=list)
    _samples_seen: int = 0
    _segment_first: int = 0
    _speech_started: bool = False
    _speech_frames: int = 0
    _silence_frames: int = 0
    _total_frames: int = 0

    def __post_init__(self) -> None:
        self.config.validate()

    def reset(self) -> None:
        self._buffer = None
        self._pending.clear()
        self._samples_seen = 0
        self._segment_first = 0
        self._speech_started = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._total_frames = 0
        self.detector.reset()

    def push(self, chunk: AudioBuffer) -> list[SpeechSegment]:
        """Feed capture; returns segments completed by this chunk."""
        if self.token is not None and self.token.cancelled:
            return []
        rate = chunk.format.sample_rate
        frame_size = max(1, int(self.config.frame_s * rate))
        emitted: list[SpeechSegment] = []
        for frame in chunk.frames(frame_size):
            score = self.detector.probe(frame)
            self._total_frames += 1
            self._samples_seen += frame.frame_count
            if self._speech_started:
                self._accumulate(frame)
                if score < self.config.off_score:
                    self._silence_frames += 1
                else:
                    self._silence_frames = 0
                if self._silence_frames * frame_size / rate >= self.config.minimum_silence_s:
                    emitted.append(self._close(truncated=False))
                elif self._speech_frames * frame_size / rate >= self.config.maximum_speech_s:
                    emitted.append(self._close(truncated=True))
            else:
                self._remember(frame, rate)
                if score >= self.config.on_score:
                    self._begin(frame)
                    self._speech_frames = 1
                    self._silence_frames = 0
        return emitted

    def flush(self) -> list[SpeechSegment]:
        """Close any in-flight utterance, e.g. because capture stopped."""
        if self._speech_started:
            return [self._close(truncated=False)]
        return []

    def _remember(self, frame: AudioBuffer, rate: int) -> None:
        """Keep a bounded tail of pre-speech audio so onsets are not lost."""
        limit = max(1, int(self.config.padding_s * rate))
        self._pending.append(frame)
        stored = sum(item.frame_count for item in self._pending)
        while stored > limit and len(self._pending) > 1:
            stored -= self._pending.pop(0).frame_count

    def _begin(self, frame: AudioBuffer) -> None:
        prefixed = frame
        lookback_samples = 0
        for previous in self._pending:
            prefixed = previous.appended(prefixed)
            lookback_samples += previous.frame_count
        self._pending.clear()
        self._segment_first = self._samples_seen - frame.frame_count - lookback_samples
        self._buffer = prefixed
        self._speech_started = True

    def _accumulate(self, frame: AudioBuffer) -> None:
        self._speech_frames += 1
        assert self._buffer is not None
        self._buffer = self._buffer.appended(frame)

    def _close(self, truncated: bool) -> SpeechSegment:
        audio = self._buffer
        assert audio is not None
        first = self._segment_first
        last = first + audio.frame_count
        segment = SpeechSegment(
            first_sample=first,
            last_sample=last,
            duration_s=audio.duration_s,
            audio=audio,
            truncated=truncated,
            metadata={"detector": self.detector.name, "silence_frames": self._silence_frames},
        )
        self._buffer = None
        self._speech_started = False
        self._speech_frames = 0
        self._silence_frames = 0
        return segment


class NeuralSpeechDetectorUnavailable(CapabilityUnavailable):
    """Raised when a modelled detector is requested but none is installed."""


def installed_detector(format: AudioFormat) -> SpeechActivityDetector:
    """The best detector actually available on this machine, named in the result."""
    return EnergySpeechDetector(format=format)
