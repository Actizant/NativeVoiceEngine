"""Recognition engine contracts.

An engine is described by what it can actually do, per language and per task,
and reports `Capability` objects rather than a boolean. The runtime refuses to
route to a capability the engine itself has not declared, which is what keeps an
English-only model from being used for a Bengali utterance and reporting success.

`transcribe` and `stream` are deliberately both on the protocol: chunked
decoding after an utterance and true partial streaming are different engine
abilities, and a system that offers only one must say so instead of pretending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

from ..core.audio import AudioBuffer, AudioFormat, Recognition
from ..core.cancel import CancellationToken
from ..core.events import TranscriptionEvent
from ..core.language import Capability, LanguageTag, Task


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    audio: AudioBuffer
    language: LanguageTag | None = None
    allow_auto_detection: bool = True
    hint_terms: tuple[str, ...] = ()
    context: str = ""
    max_duration_s: float = 30.0
    token: CancellationToken | None = None

    @property
    def required_task(self) -> Task:
        return Task.STT


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: LanguageTag | None = None
    confidence: float | None = None
    alternatives: tuple[str, ...] = ()
    segments: tuple[Recognition, ...] = ()
    engine: str = ""
    model_id: str = ""
    timings: dict = field(default_factory=dict)
    truncated: bool = False

    def as_recognition(self) -> Recognition:
        return Recognition(
            text=self.text,
            language=self.language.bcp47 if self.language else "",
            confidence=self.confidence,
            is_final=True,
            metadata={"engine": self.engine, "model_id": self.model_id, **self.timings},
        )


class TranscriptionEngine(Protocol):
    """One installed recognition backend."""

    engine_id: str
    model_id: str
    format: AudioFormat
    supports_streaming: bool

    def capabilities(self) -> tuple[Capability, ...]:
        """Declared per-language support, with status and provenance."""
        ...

    def warm(self) -> None:
        """Load weights and pay first-run cost before the first utterance."""
        ...

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult: ...

    def stream(
        self,
        chunks: Iterator[AudioBuffer],
        language: LanguageTag | None = None,
        token: CancellationToken | None = None,
    ) -> Iterator[TranscriptionEvent]:
        """Yield partials then a final. Only when `supports_streaming`.

        `token` is part of the signature because cancellation has to be observable at
        every yield boundary; leaving it to the implementation to invent made the rule
        unenforceable for anything but the happy path.
        """
        ...

    def close(self) -> None: ...


class Transcriber(Protocol):
    """What a host consumes: something that turns audio into text, whichever engine answers."""

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult: ...

    def stream(
        self,
        chunks: Iterator[AudioBuffer],
        language: LanguageTag | None = None,
        token: CancellationToken | None = None,
    ) -> Iterator[TranscriptionEvent]: ...

    def language_capability(self, language: LanguageTag) -> Capability: ...
