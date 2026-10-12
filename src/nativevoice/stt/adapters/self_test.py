"""A deterministic engine used only to prove the transport works.

The docstring is the contract: ``DeterministicEchoEngine`` does NOT recognise
speech. It returns a fixed, content-derived transcript for whatever buffer it is
handed, and it declares *every* language ``UNSUPPORTED`` so that it can never
satisfy a production routing decision. It exists to exercise buffer plumbing,
streaming event ordering and the retract reducer in isolation - not to pretend
that anything is understood. Any test that treats it as a real recogniser would
be testing a lie, so the capability table makes that impossible by construction.
"""

from __future__ import annotations

import hashlib

from ...core.audio import AudioBuffer, AudioFormat
from ...core.events import TranscriptionEvent, TranscriptionKind
from ...core.language import Capability, CapabilityStatus, LanguageTag, Task
from ...language.registry import LanguageRegistry
from ..api import TranscriptionRequest, TranscriptionResult


def _signature(buffer: AudioBuffer) -> str:
    payload = bytes(buffer.samples) + str(buffer.format).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


class DeterministicEchoEngine:
    """Returns a stable transcript per input buffer; claims no real capability."""

    engine_id = "self-test"
    model_id = "deterministic-echo"
    supports_streaming = True

    def __init__(self, sample_rate: int = 16000, registry: LanguageRegistry | None = None) -> None:
        self.format = AudioFormat(sample_rate, 1)
        self._registry = registry or LanguageRegistry()
        self._transcripts: dict[str, str] = {}
        self.invocations: list[str] = []

    def set_transcript(self, buffer: AudioBuffer, text: str) -> None:
        self._transcripts[_signature(buffer)] = text

    def transcript_for(self, buffer: AudioBuffer) -> str:
        return self._transcripts.get(_signature(buffer), f"echo frames={buffer.frame_count} samples={len(buffer.samples)}")

    def capabilities(self) -> tuple[Capability, ...]:
        # Every language UNSUPPORTED: this engine must never be selectable, no
        # matter which language is asked for.
        return tuple(
            Capability(
                task=Task.STT,
                status=CapabilityStatus.UNSUPPORTED,
                language=code,
                engine=self.engine_id,
                limitations=("plumbing fixture; does not recognise speech",),
            )
            for code in self._registry.all_codes()
        )

    def warm(self) -> None:
        return None

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        text = self.transcript_for(request.audio)
        self.invocations.append("transcribe")
        return TranscriptionResult(
            text=text,
            language=request.language,
            engine=self.engine_id,
            model_id=self.model_id,
        )

    def stream(self, chunks, language: LanguageTag | None = None):
        collected = AudioBuffer.zeros(self.format, 0)
        total = 0
        for chunk in chunks:
            collected = collected.appended(chunk) if collected.frame_count else chunk
            total += chunk.frame_count
        text = self.transcript_for(collected) if collected.frame_count else ""
        self.invocations.append("stream")
        # Demonstrate the retract path: a stale partial is replaced, not appended.
        yield TranscriptionEvent(TranscriptionKind.PARTIAL, text[:4], "echo", language=language.code if language else "")
        yield TranscriptionEvent(TranscriptionKind.RETRACT, "", "echo")
        yield TranscriptionEvent(TranscriptionKind.FINAL, text, "echo", is_final=True, language=language.code if language else "")

    def close(self) -> None:
        return None
