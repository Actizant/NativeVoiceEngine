"""Streaming event contracts for recognition and synthesis.

Partial hypotheses are revision-shaped, not append-shaped: an engine that
re-decodes with more context may *retract* what it previously showed. The event
kinds below make that explicit so the session layer can hold one reducer instead
of each UI guessing. This mirrors the `partial`/`final`/`retract` triple that the
OpenWhispr meeting pipeline arrived at, expressed as a typed contract.

Audio output uses a monotonic `sequence` so that a stale or duplicated chunk can
be detected by the player rather than played, and every terminal event names its
reason - including `cancelled`, which must be distinguishable from `completed`
in diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .audio import AudioBuffer, Recognition


class TranscriptionKind(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"
    RETRACT = "retract"


class SynthesisKind(str, Enum):
    FIRST_AUDIO = "first_audio"
    AUDIO = "audio"
    EOS = "eos"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TurnKind(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    TURN_COMMITTED = "turn_committed"
    INTERRUPTED = "interrupted"
    IDLE = "idle"


@dataclass(slots=True)
class TranscriptionEvent:
    kind: TranscriptionKind
    text: str
    utterance_id: str
    sequence: int = 0
    language: str = ""
    confidence: float | None = None
    is_final: bool = False
    elapsed_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def partial(cls, text: str, utterance_id: str, **kw) -> "TranscriptionEvent":
        return cls(TranscriptionKind.PARTIAL, text, utterance_id, **kw)

    @classmethod
    def final(cls, text: str, utterance_id: str, **kw) -> "TranscriptionEvent":
        return cls(TranscriptionKind.FINAL, text, utterance_id, is_final=True, **kw)

    @classmethod
    def retract(cls, utterance_id: str, text: str = "", **kw) -> "TranscriptionEvent":
        """`text` names the span being withdrawn; empty means "the last final"."""
        return cls(TranscriptionKind.RETRACT, text, utterance_id, **kw)

    @classmethod
    def from_recognition(cls, rec: Recognition, utterance_id: str, kind: TranscriptionKind) -> "TranscriptionEvent":
        return cls(
            kind,
            rec.text,
            utterance_id,
            language=rec.language,
            confidence=rec.confidence,
            is_final=rec.is_final,
            metadata=dict(rec.metadata),
        )


@dataclass(slots=True)
class SynthesisEvent:
    kind: SynthesisKind
    sequence: int
    audio: AudioBuffer | None = None
    reason: str = ""
    text: str = ""
    voice_id: str = ""
    language: str = ""
    elapsed_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.kind in (SynthesisKind.EOS, SynthesisKind.CANCELLED, SynthesisKind.FAILED)


@dataclass(slots=True)
class TurnEvent:
    kind: TurnKind
    turn_id: str
    reason: str = ""
    audio: AudioBuffer | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class DiagnosticEvent:
    """Structured log record. Diagnostics are data, not printed strings."""

    component: str
    event: str
    level: str = "info"
    fields: dict = field(default_factory=dict)
    timestamp_ns: int = 0
