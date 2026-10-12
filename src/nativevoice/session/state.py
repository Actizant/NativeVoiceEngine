"""Session state and turn accounting.

Kept separate from the orchestrator so the interesting parts - which turn is
current, what may still be committed, what must be discarded - are a pure,
exhaustively testable model with no threads and no devices in it. Most
"the assistant talked over itself" bugs are state bugs pretending to be timing
bugs, and a pure reducer is the only way to prove them out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..core.errors import CapabilityUnavailable, ConfigurationError
from ..core.cancel import CancellationToken
from ..core.events import TranscriptionEvent, TranscriptionKind
from ..core.language import LanguageTag
from ..core.time import new_id


class ListeningMode(str, Enum):
    PUSH_TO_TAP = "push_to_tap"
    HOLD_TO_TALK = "hold_to_talk"
    """Explicitly started and stopped by the user. The default: nothing hears anything
    until the user says so, which is the only mode that needs no privacy argument."""

    CONTINUOUS = "continuous"
    """VAD-delimited always-on capture. Requires `continuous_consent` in
    SessionConfig; constructing it without that is refused rather than defaulted."""

    WAKE_WORD = "wake_word"
    """Declared, not implemented. `WakeWordDetector` below is an interface with no
    working backend, and `SessionConfig.allow_wake_word` cannot be set True."""

    OFF = "off"


class WakeWordBackendUnavailable(CapabilityUnavailable):
    """Raised on any attempt to enable wake-word listening.

    A wake word is a separate capability from continuous listening, and the
    specification forbids reporting the two as interchangeable. The interface
    exists so the architecture is real; the detector does not. Subclasses
    `CapabilityUnavailable` so a host catches it with every other "this is not
    implemented" answer rather than with a bare `Exception`.
    """


class TurnState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    AWAITING_ANSWER = "awaiting_answer"
    SPEAKING = "speaking"


class WakeWordDetector:
    """Interface placeholder. Every method refuses; nothing is installed."""

    name = "none"

    def feed(self, audio) -> bool:
        raise WakeWordBackendUnavailable("no wake-word detector is installed")

    def reset(self) -> None:
        raise WakeWordBackendUnavailable("no wake-word detector is installed")


@dataclass(slots=True)
class SessionConfig:
    mode: ListeningMode = ListeningMode.PUSH_TO_TAP
    continuous_consent: bool = False
    barge_in_enabled: bool = True
    interrupt_requires_speech_ms: int = 200
    voice_output_enabled: bool = True
    stale_answer_policy: str = "discard"
    max_recoveries: int = 3

    def __post_init__(self) -> None:
        if self.stale_answer_policy not in ("discard", "speak_anyway"):
            raise ConfigurationError("unknown stale answer policy")
        if self.mode is ListeningMode.WAKE_WORD:
            raise WakeWordBackendUnavailable(
                "wake-word listening is specified but not implemented; choose push_to_tap or continuous"
            )
        if self.mode is ListeningMode.CONTINUOUS and not self.continuous_consent:
            raise ConfigurationError("continuous listening requires explicit user consent")


@dataclass(slots=True)
class Turn:
    """One exchange. Its token is the handle every late worker checks."""

    turn_id: str = field(default_factory=lambda: new_id("turn"))
    state: TurnState = TurnState.IDLE
    token: CancellationToken = field(default_factory=CancellationToken)
    language: LanguageTag | None = None
    committed_text: str = ""
    superseded_by: str = ""
    voice: object | None = None
    started_ns: int = 0
    attempts: int = 0

    @property
    def current(self) -> bool:
        return not self.token.cancelled


@dataclass(slots=True)
class TranscriptBuffer:
    """Reducer over `partial` / `final` / `retract`.

    A partial replaces the previous partial for the same utterance; a final
    commits. A retract removes an earlier final by exact text match, which is
    what the diarizing reference pipeline needed once more context arrived. A
    final from a turn that is no longer current is dropped here rather than
    committed, which is the whole of stale-response cancellation.
    """

    finals: list[str] = field(default_factory=list)
    partials: dict[str, str] = field(default_factory=dict)
    retracted: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    accept: bool = True

    def apply(self, event: TranscriptionEvent) -> str:
        if not self.accept:
            self.ignored.append(event.utterance_id)
            return self.text
        if event.kind is TranscriptionKind.RETRACT:
            for index, existing in enumerate(self.finals):
                if existing == event.text or (event.text and event.text in existing):
                    self.finals.pop(index)
                    self.retracted.append(event.text or existing)
                    break
            self.partials.pop(event.utterance_id, None)
        elif event.kind is TranscriptionKind.PARTIAL:
            self.partials[event.utterance_id] = event.text
        else:
            self.partials.pop(event.utterance_id, None)
            if event.text and event.text not in self.finals:
                self.finals.append(event.text)
        return self.text

    @property
    def text(self) -> str:
        parts = list(self.finals)
        parts.extend(value for value in self.partials.values() if value)
        return " ".join(part for part in parts if part).strip()
