"""Synthesis engine contracts.

Voice conditioning is expressed as a `VoiceReference`, never as a bare file
path, because the consent and ownership facts have to travel with the audio that
was consented for. `SynthesisRequest` carries prosody as named controls rather
than an open dict so an engine that cannot do something can say so, and the
session can report the gap instead of silently ignoring it.

An engine that has no approved model does not pretend to speak: `synthesize`
raises `CapabilityUnavailable`, and `voices()` reports empty rather than listing
presets it cannot render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

from ..core.audio import AudioBuffer, AudioFormat
from ..core.cancel import CancellationToken
from ..core.events import SynthesisEvent
from ..core.language import Capability, LanguageTag, Task


@dataclass(frozen=True, slots=True)
class VoiceReference:
    """A speaker condition plus the record of why using it is permitted."""

    voice_id: str
    audio: AudioBuffer | None = None
    state_path: str = ""
    speaker_slot: int | None = None
    language: LanguageTag | None = None
    consent_token: str = ""
    owner: str = ""
    source: str = ""

    def has_consent(self) -> bool:
        return bool(self.consent_token and self.owner)


@dataclass(frozen=True, slots=True)
class Prosody:
    rate: float = 1.0
    pitch_semitones: float = 0.0
    volume_db: float = 0.0
    pause_after_s: float = 0.0
    style: str = ""

    def clamp(self) -> "Prosody":
        return Prosody(
            rate=min(2.0, max(0.5, self.rate)),
            pitch_semitones=min(12.0, max(-12.0, self.pitch_semitones)),
            volume_db=min(12.0, max(-24.0, self.volume_db)),
            pause_after_s=max(0.0, self.pause_after_s),
            style=self.style,
        )


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    text: str
    language: LanguageTag
    voice: VoiceReference | None = None
    prosody: Prosody = field(default_factory=Prosody)
    seed: int | None = None
    maximum_duration_s: float = 120.0
    token: CancellationToken | None = None

    @property
    def required_task(self) -> Task:
        return Task.TTS


@dataclass(frozen=True, slots=True)
class SynthesisOutcome:
    audio: AudioBuffer
    language: LanguageTag
    voice_id: str
    engine: str = ""
    model_id: str = ""
    timings: dict = field(default_factory=dict)
    dropped_text: str = ""
    truncated: bool = False


class SynthesisEngine(Protocol):
    """One installed synthesis backend."""

    engine_id: str
    model_id: str
    output_format: AudioFormat
    supports_streaming: bool
    controllable: tuple[str, ...]

    def capabilities(self) -> tuple[Capability, ...]: ...

    def voices(self) -> tuple[VoiceReference, ...]:
        """Voices this engine can actually render right now."""
        ...

    def condition(self, reference: VoiceReference) -> VoiceReference:
        """Turn recorded audio into a reusable voice, or raise."""
        ...

    def warm(self) -> None: ...

    def synthesize(self, request: SynthesisRequest) -> SynthesisOutcome: ...

    def stream(self, request: SynthesisRequest) -> Iterator[SynthesisEvent]: ...

    def close(self) -> None: ...


class SpeechSynthesizer(Protocol):
    def synthesize(self, request: SynthesisRequest) -> SynthesisOutcome: ...

    def stream(self, request: SynthesisRequest) -> Iterator[SynthesisEvent]: ...

    def language_capability(self, language: LanguageTag) -> Capability: ...
