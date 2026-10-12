"""Transcription orchestration: route, resample, stream, reduce, time, cancel.

``TranscriptionService`` is the thing ULTRON talks to. It never invokes an engine
before the router has approved the language, resamples audio into the selected
engine's declared format, folds the engine's revision-shaped partial/final/retract
events into one growing transcript, attaches measured ``Timings``, honours the
``CancellationToken`` at every yield boundary, and - the honesty point - raises a
specific ``LanguageUnsupported`` / ``EngineUnavailable`` / ``CapabilityUnavailable``
rather than ever returning empty or unrelated text.

Partial handling is a small reducer, because a multilingual engine may *retract*
a stale hypothesis and replace it. The reducer makes that explicit and testable
instead of leaving each UI to guess.
"""

from __future__ import annotations

from dataclasses import replace

from ..audio.resample import resample
from ..core.audio import AudioBuffer, Recognition
from ..core.cancel import CancellationToken, guard
from ..core.errors import (
    CapabilityUnavailable,
    EngineCrashed,
    EngineUnavailable,
    LanguageUnsupported,
)
from ..core.events import TranscriptionEvent, TranscriptionKind
from ..core.language import Capability, LanguageTag, Task
from ..core.time import Timings, new_id
from ..language.registry import CapabilityLedger
from ..language.routing import LanguageRouter, RoutingKind
from .api import TranscriptionRequest, TranscriptionResult
from .engine_registry import EngineRegistry
from .text_context import build_prompt


class TranscriptReducer:
    """Fold partial/final/retract events into a single visible transcript."""

    def __init__(self, joiner: str = " ") -> None:
        self._committed: list[str] = []
        self._partial: str = ""
        self._joiner = joiner
        self.retracts = 0

    def feed(self, event: TranscriptionEvent) -> str:
        if event.kind is TranscriptionKind.PARTIAL:
            # A new partial *replaces* the stale one; it is never appended.
            self._partial = event.text
        elif event.kind is TranscriptionKind.RETRACT:
            self._partial = ""
            self.retracts += 1
        elif event.kind is TranscriptionKind.FINAL:
            if event.text:
                self._committed.append(event.text)
            self._partial = ""
        return self.visible()

    def visible(self) -> str:
        parts = list(self._committed)
        if self._partial:
            parts.append(self._partial)
        return self._joiner.join(parts)


def result_from_recognition(recognition: Recognition, engine_id: str, model_id: str = "") -> TranscriptionResult:
    """Convert one engine ``Recognition`` into a full ``TranscriptionResult``."""
    language = None
    if recognition.language:
        language = LanguageTag.parse(recognition.language)
    return TranscriptionResult(
        text=recognition.text,
        language=language,
        confidence=recognition.confidence,
        segments=(recognition,),
        alternatives=recognition.alternatives,
        engine=engine_id,
        model_id=model_id,
    )


class TranscriptionService:
    """Implements the ``Transcriber`` protocol over a registry + router + ledger."""

    def __init__(
        self,
        registry: EngineRegistry,
        router: LanguageRouter | None = None,
        ledger: CapabilityLedger | None = None,
        *,
        strict: bool = True,
    ) -> None:
        self.registry = registry
        self.ledger = ledger if ledger is not None else registry.ledger
        self.router = router if router is not None else LanguageRouter()
        self.strict = bool(strict)

    # -- selection ------------------------------------------------------------
    def _select(self, outcome):
        pair = self.registry.best_for(outcome.language, strict=self.strict)
        if pair is None:
            label = outcome.language.bcp47 if outcome.language else "(auto-detect)"
            raise EngineUnavailable(
                "the routed language has no installed engine that can serve it",
                language=label,
                strict=self.strict,
            )
        return pair

    def _raise_for_outcome(self, outcome) -> None:
        if outcome.kind is RoutingKind.UNAVAILABLE:
            raise LanguageUnsupported(
                outcome.reason or "no engine has usable capability for this language",
                language=outcome.language.bcp47 if outcome.language else None,
                fallback=outcome.fallback.bcp47 if outcome.fallback else None,
                kind=outcome.kind.value,
            )
        if outcome.kind is RoutingKind.AMBIGUOUS:
            raise CapabilityUnavailable(
                outcome.reason or "language detection is below threshold; ask rather than guess",
                confidence=round(outcome.confidence, 4),
                kind=outcome.kind.value,
            )

    def _prepare(self, buffer: AudioBuffer, engine) -> AudioBuffer:
        target = engine.format
        if target.channels == 1 and buffer.format.channels != 1:
            buffer = buffer.to_mono()
        elif target.channels != buffer.format.channels:
            raise CapabilityUnavailable(
                "channel layout cannot be adapted to the engine",
                target=str(target),
                source=str(buffer.format),
            )
        if buffer.format.sample_rate != target.sample_rate:
            buffer = resample(buffer, target.sample_rate)
        return buffer

    @staticmethod
    def _truncate(buffer: AudioBuffer, max_duration_s: float) -> tuple[AudioBuffer, bool]:
        if max_duration_s and buffer.duration_s > max_duration_s:
            keep = int(max(0, round(max_duration_s * buffer.format.sample_rate)))
            return buffer.sliced(0, keep), True
        return buffer, False

    @staticmethod
    def _normalize(raw, engine) -> TranscriptionResult:
        if isinstance(raw, TranscriptionResult):
            return raw
        if isinstance(raw, Recognition):
            return result_from_recognition(raw, engine.engine_id, getattr(engine, "model_id", ""))
        raise EngineCrashed(
            "engine returned an unexpected result type",
            engine=engine.engine_id,
            returned=type(raw).__name__,
        )

    # -- Transcriber API ------------------------------------------------------
    def language_capability(self, language: LanguageTag) -> Capability:
        code = language.code if isinstance(language, LanguageTag) else str(language)
        return self.ledger.best(Task.STT, code)

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        token = request.token
        guard(token)
        if not self.registry.has_engines():
            raise EngineUnavailable(
                "no transcription engines are installed", task=Task.STT.value
            )
        audio, truncated = self._truncate(request.audio, request.max_duration_s)
        duration = audio.duration_s
        outcome = self.router.route(
            ledger=self.ledger,
            requested_language=request.language,
            spoken_text=None,
            audio_duration_s=duration,
            allow_declared=not self.strict,
            allow_auto_detection=request.allow_auto_detection,
        )
        self._raise_for_outcome(outcome)
        engine, capability = self._select(outcome)
        prepared = self._prepare(audio, engine)
        prompt = build_prompt(request.hint_terms, request.context)
        engine_request = replace(
            request,
            audio=prepared,
            language=outcome.language,
            context=prompt,
            hint_terms=(),
        )
        timings = Timings(audio_duration_s=duration)
        guard(token)
        raw = engine.transcribe(engine_request)
        result = self._normalize(raw, engine)
        guard(token)
        if not result.text or not result.text.strip():
            raise CapabilityUnavailable(
                "engine returned an empty transcription",
                engine=engine.engine_id,
                model_id=result.model_id,
            )
        timings.finish()
        # TranscriptionResult is frozen: rebuild it rather than mutate.
        return replace(
            result,
            engine=engine.engine_id,
            model_id=engine.model_id or result.model_id,
            language=result.language if result.language is not None else outcome.language,
            truncated=truncated,
            timings={
                **timings.as_dict(),
                "engine": engine.engine_id,
                "capability_status": capability.status.value,
            },
        )

    def stream(
        self,
        chunks,
        language: LanguageTag | None = None,
        *,
        token: CancellationToken | None = None,
        allow_auto_detection: bool = True,
    ):
        raw_chunks: list[AudioBuffer] = []
        for chunk in chunks:
            guard(token)
            raw_chunks.append(chunk)
        if not self.registry.has_engines():
            raise EngineUnavailable("no transcription engines are installed", task=Task.STT.value)
        if not raw_chunks:
            raise CapabilityUnavailable("stream received no audio chunks", task=Task.STT.value)
        duration = sum(chunk.duration_s for chunk in raw_chunks)
        outcome = self.router.route(
            ledger=self.ledger,
            requested_language=language,
            spoken_text=None,
            audio_duration_s=duration,
            allow_declared=not self.strict,
            allow_auto_detection=allow_auto_detection,
        )
        self._raise_for_outcome(outcome)
        engine, _capability = self._select(outcome)
        prepared = [self._prepare(chunk, engine) for chunk in raw_chunks]
        utterance_id = new_id("utt")
        timings = Timings(audio_duration_s=duration)
        language_code = outcome.language.code if outcome.language else ""

        if engine.supports_streaming and hasattr(engine, "stream"):
            reducer = TranscriptReducer()
            sequence = 0
            for event in engine.stream(iter(prepared), outcome.language):
                guard(token)
                reducer.feed(event)
                timings.mark_first_result()
                sequence += 1
                event.sequence = sequence
                event.elapsed_ms = timings.total_ms
                if event.kind is TranscriptionKind.FINAL:
                    # Suppress the engine's own final; a single authoritative,
                    # reduced final is emitted once the stream is complete.
                    continue
                guard(token)
                yield event
            final_text = reducer.visible()
            if not final_text.strip():
                raise CapabilityUnavailable(
                    "engine produced no final transcript", engine=engine.engine_id
                )
            timings.finish()
            yield TranscriptionEvent.final(
                final_text,
                utterance_id,
                sequence=sequence + 1,
                language=language_code,
                elapsed_ms=timings.total_ms,
                metadata={
                    "engine": engine.engine_id,
                    "model_id": getattr(engine, "model_id", ""),
                    "retracts": reducer.retracts,
                },
            )
        else:
            merged = prepared[0]
            for extra in prepared[1:]:
                merged = merged.appended(extra)
            engine_request = TranscriptionRequest(
                audio=merged,
                language=outcome.language,
                allow_auto_detection=False,
                token=token,
            )
            guard(token)
            result = self._normalize(engine.transcribe(engine_request), engine)
            if not result.text.strip():
                raise CapabilityUnavailable("engine produced no text", engine=engine.engine_id)
            timings.finish()
            guard(token)
            yield TranscriptionEvent.final(
                result.text,
                utterance_id,
                language=result.language.code if result.language else language_code,
                confidence=result.confidence,
                elapsed_ms=timings.total_ms,
                metadata={
                    "engine": engine.engine_id,
                    "model_id": getattr(engine, "model_id", ""),
                },
            )
