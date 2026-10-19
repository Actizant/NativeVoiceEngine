"""The contract a host application implements and the voice engine consumes.

This file exists to make the boundary explicit and auditable in both directions:

* the host provides an answer for an utterance and may persist that conversation
  however it likes. The engine holds **no persistent memory of its own** - not
  transcripts, not answers, not embeddings - because the host already owns
  persistent memory and a second host owns its own. A second store inside the
  voice layer would be a second source of truth, and those diverge silently.
* the engine provides typed events, capability truth and cancellation. The host
  never imports an engine, a model id, a licence decision or a file path.

`VoiceHost` is deliberately tiny: one method that turns text into text. Anything
else the host can do, it can do after receiving the utterance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

from ..core.audio import AudioBuffer
from ..core.cancel import CancellationToken
from ..core.events import SynthesisEvent, TranscriptionEvent
from ..core.language import Capability, LanguageTag, Task
from ..stt.api import TranscriptionResult


class VoiceHost(Protocol):
    """The whole of what a host must supply."""

    def respond(self, utterance: str, language: LanguageTag | None = None) -> str:
        """Return the text to be spoken. Empty string means 'say nothing'."""
        ...


@dataclass(slots=True)
class HostReply:
    """What the engine hands back for one exchange, in both directions."""

    utterance: str
    reply: str
    language: str
    spoken: bool
    silenced: bool
    notes: tuple[str, ...] = ()
    timings: dict = field(default_factory=dict)

    @property
    def transcript(self) -> str:
        return self.utterance


class VoiceEngine(Protocol):
    """What the engine offers a host. `integration.VoiceInterface` is the implementation."""

    def hear(self, audio: AudioBuffer, language: LanguageTag | None = None) -> TranscriptionResult: ...

    def stream_hearing(self, chunks: Iterator[AudioBuffer], language: LanguageTag | None = None) -> Iterator[TranscriptionEvent]: ...

    def speak(self, text: str, language: LanguageTag | None = None, token: CancellationToken | None = None) -> Iterator[SynthesisEvent]: ...

    def exchange(self, audio: AudioBuffer, host: VoiceHost, language: LanguageTag | None = None) -> HostReply: ...

    def cancel(self, reason: str = "cancelled") -> bool: ...

    def capabilities(self, task: Task) -> dict[str, Capability]: ...
