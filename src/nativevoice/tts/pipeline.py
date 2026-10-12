"""The synthesis pipeline: gate, normalise, chunk, render, assemble.

Order matters here for two reasons that are easy to get wrong.

First, the language gate runs before an engine is touched. The reference study
found registries that advertised support flags no engine could honour, so routing
here asks the engine to declare the language itself, and a refusal means the
engine's `synthesize` is never called - which is what makes it safe to hand a
Bengali request to an English-only model.

Second, chunk audio is assembled with an overlapping crossfade rather than plain
concatenation. Independent chunk generation (the documented behaviour of the
reference architecture) resets prosody at every boundary, and a hard splice turns
that reset into an audible click. The crossfade is short by design: a longer one
would swallow real speech to hide a discontinuity.

The render path is a single generator that yields *contiguous* pieces: `stream()`
emits them as events and `synthesize()` concatenates the same pieces, so a cached
utterance and a live one are byte-identical. That property is what makes latency
measurements comparable instead of two different code paths.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from ..audio.pcm import crossfade
from ..audio.resample import resample
from ..core.audio import AudioBuffer, AudioFormat
from ..core.cancel import guard
from ..core.errors import (
    CancellationRequested,
    ConfigurationError,
    LanguageUnsupported,
    ProfileInvalid,
    VoiceError,
)
from ..core.events import DiagnosticEvent, SynthesisEvent, SynthesisKind
from ..core.language import Capability, LanguageTag, Task
from ..core.time import Timings, monotonic_ns
from .api import Prosody, SynthesisEngine, SynthesisOutcome, SynthesisRequest
from .chunking import TextChunk, TextChunker
from .engine_registry import EngineRegistry
from .normalise import NormalisationReport, TextNormaliser


class DiagnosticsLedger(Protocol):
    """Where the pipeline posts structured events instead of printing them."""

    def record(self, event: DiagnosticEvent) -> None: ...


@dataclass(slots=True)
class _RenderState:
    """Per-request render outcome, so nothing is smuggled through instance state."""

    truncated: bool = False
    dropped_text: str = ""
    emitted_frames: int = 0


@dataclass(slots=True)
class _Preparation:
    engine: SynthesisEngine
    chunks: tuple[TextChunk, ...]
    report: NormalisationReport
    prosody: Prosody = field(default_factory=Prosody)
    render: _RenderState = field(default_factory=_RenderState)


class SpeechPipeline:
    """A `SpeechSynthesizer` built from normaliser + chunker + engine registry."""

    def __init__(
        self,
        normaliser: TextNormaliser,
        chunker: TextChunker,
        registry: EngineRegistry,
        ledger: DiagnosticsLedger | None = None,
        *,
        crossfade_s: float = 0.02,
        strict: bool = True,
        engine_id: str | None = None,
        target_format: AudioFormat | None = None,
    ) -> None:
        if crossfade_s < 0:
            raise ConfigurationError("crossfade_s cannot be negative", crossfade_s=crossfade_s)
        self.normaliser = normaliser
        self.chunker = chunker
        self.registry = registry
        self.ledger = ledger
        self.crossfade_s = crossfade_s
        self.strict = strict
        self.engine_id = engine_id
        self.target_format = target_format

    def language_capability(self, language: LanguageTag) -> Capability:
        return self.registry.capability(language.root(), Task.TTS)

    def synthesize(self, request: SynthesisRequest) -> SynthesisOutcome:
        timings = Timings()
        prepared = self._prepare(request)
        self._post("synthesis_started", request, prepared, timings)
        pieces = list(self._rendered_pieces(request, prepared, timings))
        audio = self._join(pieces, prepared.engine.output_format)
        timings.audio_duration_s = audio.duration_s
        timings.finish()
        self._post("synthesis_completed", request, prepared, timings)
        return SynthesisOutcome(
            audio=audio,
            language=request.language,
            voice_id=request.voice.voice_id if request.voice else "",
            engine=prepared.engine.engine_id,
            model_id=prepared.engine.model_id,
            timings={**timings.as_dict(), "normalisation": prepared.report.as_dict()},
            dropped_text=prepared.render.dropped_text,
            truncated=prepared.render.truncated,
        )

    def stream(self, request: SynthesisRequest) -> Iterator[SynthesisEvent]:
        """Yield audio events with monotonic sequence numbers and one terminal kind.

        Cancellation is observed before every yield, and the terminal `cancelled`
        event carries no audio: a player that keys off `kind` alone would otherwise
        play a chunk the user had already interrupted.
        """
        timings = Timings()
        sequence = 0
        prepared: _Preparation | None = None
        try:
            prepared = self._prepare(request)
            self._post("synthesis_started", request, prepared, timings)
            first = True
            for piece in self._rendered_pieces(request, prepared, timings):
                if self._cancelled(request):
                    break
                kind = SynthesisKind.FIRST_AUDIO if first else SynthesisKind.AUDIO
                first = False
                sequence += 1
                yield self._event(request, prepared, kind, sequence, piece, timings)
            if self._cancelled(request):
                sequence += 1
                yield self._event(
                    request,
                    prepared,
                    SynthesisKind.CANCELLED,
                    sequence,
                    None,
                    timings,
                    reason=self._cancel_reason(request),
                )
                self._post("synthesis_cancelled", request, prepared, timings)
                return
            sequence += 1
            yield self._event(
                request, prepared, SynthesisKind.EOS, sequence, None, timings, reason="completed"
            )
            self._post("synthesis_completed", request, prepared, timings)
        except CancellationRequested as cancelled:
            sequence += 1
            yield self._event(
                request,
                prepared,
                SynthesisKind.CANCELLED,
                sequence,
                None,
                timings,
                reason=str(cancelled.details.get("reason") or "cancelled"),
            )
            if prepared is not None:
                self._post("synthesis_cancelled", request, prepared, timings)
        except VoiceError as failure:
            sequence += 1
            yield self._event(
                request, prepared, SynthesisKind.FAILED, sequence, None, timings, reason=str(failure)
            )
            if prepared is not None:
                self._post("synthesis_failed", request, prepared, timings)
            raise

    # -- preparation ---------------------------------------------------------

    def _prepare(self, request: SynthesisRequest) -> _Preparation:
        """Gate on language, then normalise and chunk. No engine call happens here."""
        guard(request.token)
        if not request.text.strip():
            raise ConfigurationError("nothing to synthesise", language=request.language.bcp47)
        if request.voice is not None and request.voice.audio is not None and not request.voice.has_consent():
            raise ProfileInvalid(
                "a recorded voice needs consent on record before it can speak",
                voice_id=request.voice.voice_id,
            )
        engine = self._select(request)
        report = self.normaliser.normalise(request.text, request.language)
        chunks = self.chunker.chunk(report.text)
        if not chunks:
            raise ConfigurationError(
                "normalisation produced nothing renderable", language=request.language.bcp47
            )
        return _Preparation(
            engine=engine, chunks=chunks, report=report, prosody=request.prosody.clamp()
        )

    def _select(self, request: SynthesisRequest) -> SynthesisEngine:
        """Capability gate: an engine is only used if it declares this language."""
        if self.engine_id is None:
            return self.registry.best_for(request.language.root(), Task.TTS, strict=self.strict)
        engine = self.registry.get(self.engine_id)
        capability = self._declared(engine, request.language.root())
        if capability is None or not capability.usable_for_automatic_routing:
            raise LanguageUnsupported(
                "the selected engine cannot synthesise this language",
                engine_id=self.engine_id,
                language=request.language.bcp47,
                declared="" if capability is None else capability.status.value,
            )
        return engine

    @staticmethod
    def _declared(engine: SynthesisEngine, language: LanguageTag) -> Capability | None:
        for capability in engine.capabilities():
            if capability.task is Task.TTS and capability.language == language.code:
                return capability
        return None

    # -- rendering -----------------------------------------------------------

    def _rendered_pieces(
        self, request: SynthesisRequest, prepared: _Preparation, timings: Timings
    ) -> Iterator[AudioBuffer]:
        """Audio spans that concatenate, in order, into the final utterance.

        The tail of each span is held back by `crossfade_s` so the next span blends
        into it; nothing else about the audio is altered.
        """
        engine = prepared.engine
        format = self.target_format or engine.output_format
        overlap = min(int(round(self.crossfade_s * format.sample_rate)), format.sample_rate // 2)
        budget_frames = int(request.maximum_duration_s * format.sample_rate)
        pending: AudioBuffer | None = None
        first_chunk = True

        for position, chunk in enumerate(prepared.chunks):
            guard(request.token)
            text = chunk.text.strip()
            if not text:
                continue
            if not first_chunk and prepared.prosody.pause_after_s > 0:
                silence = AudioBuffer.zeros(
                    format, int(prepared.prosody.pause_after_s * format.sample_rate)
                )
                pending, emit = self._hold(pending, silence, overlap)
                if emit is not None:
                    limited = self._budget(emit, budget_frames, prepared)
                    if limited is None:
                        self._drop_from(prepared, position)
                        return
                    if limited.frame_count:
                        prepared.render.emitted_frames += limited.frame_count
                        yield limited
            remaining_s = max(
                0.1,
                request.maximum_duration_s - prepared.render.emitted_frames / format.sample_rate,
            )
            sub_request = replace(
                request, text=text, prosody=prepared.prosody, maximum_duration_s=remaining_s
            )
            for buffer in self._engine_audio(engine, sub_request):
                buffer = self._conform(buffer, format)
                timings.mark_first_result()
                pending, emit = self._hold(pending, buffer, overlap)
                if emit is None:
                    continue
                emit = self._budget(emit, budget_frames, prepared)
                if emit is None:
                    self._drop_from(prepared, position)
                    return
                prepared.render.emitted_frames += emit.frame_count
                yield emit
            first_chunk = False

        if pending is not None and pending.frame_count:
            tail = self._budget(pending, budget_frames, prepared)
            if tail is not None:
                prepared.render.emitted_frames += tail.frame_count
                yield tail
            else:
                self._drop_from(prepared, len(prepared.chunks) - 1)

    @staticmethod
    def _budget(piece: AudioBuffer, budget_frames: int, prepared: _Preparation) -> AudioBuffer | None:
        """Clip a piece to the duration ceiling; None means the budget is spent."""
        left = budget_frames - prepared.render.emitted_frames
        if left <= 0:
            prepared.render.truncated = True
            return None
        if piece.frame_count <= left:
            return piece
        prepared.render.truncated = True
        return piece.sliced(0, left)

    def _drop_from(self, prepared: _Preparation, position: int) -> None:
        prepared.render.truncated = True
        prepared.render.dropped_text = " ".join(
            chunk.text.strip() for chunk in prepared.chunks[position:] if chunk.text.strip()
        )

    @staticmethod
    def _hold(
        pending: AudioBuffer | None, buffer: AudioBuffer, overlap: int
    ) -> tuple[AudioBuffer | None, AudioBuffer | None]:
        """Blend the held tail of the previous span into the head of this one."""
        if buffer.frame_count == 0:
            return pending, None
        if pending is None:
            if overlap <= 0 or buffer.frame_count <= overlap:
                return None, buffer
            return buffer.sliced(buffer.frame_count - overlap, buffer.frame_count), buffer.sliced(
                0, buffer.frame_count - overlap
            )
        head_frames = min(overlap, buffer.frame_count)
        head = buffer.sliced(0, head_frames)
        blend = crossfade(pending, head, overlap_s=head.duration_s)
        rest = buffer.sliced(head_frames, buffer.frame_count)
        if overlap <= 0 or rest.frame_count <= overlap:
            return None, blend.appended(rest) if rest.frame_count else blend
        return rest.sliced(rest.frame_count - overlap, rest.frame_count), blend.appended(
            rest.sliced(0, rest.frame_count - overlap)
        )

    def _engine_audio(self, engine: SynthesisEngine, request: SynthesisRequest) -> Iterator[AudioBuffer]:
        if engine.supports_streaming:
            for event in engine.stream(request):
                guard(request.token)
                if event.audio is not None:
                    yield event.audio
            return
        outcome = engine.synthesize(request)
        guard(request.token)
        yield outcome.audio

    @staticmethod
    def _conform(buffer: AudioBuffer, format: AudioFormat) -> AudioBuffer:
        if buffer.format.channels != format.channels:
            if format.channels > buffer.format.channels:
                raise ConfigurationError(
                    "engine audio has fewer channels than the pipeline format",
                    engine=buffer.format,
                    pipeline=format,
                )
            buffer = buffer.to_mono()
        if buffer.format.sample_rate != format.sample_rate:
            buffer = resample(buffer, format.sample_rate)
        if buffer.format != format:
            raise ConfigurationError(
                "engine audio cannot be matched to the pipeline format",
                engine=buffer.format,
                pipeline=format,
            )
        return buffer

    @staticmethod
    def _join(pieces: Sequence[AudioBuffer], format: AudioFormat) -> AudioBuffer:
        if not pieces:
            return AudioBuffer.zeros(format, 0)
        merged = pieces[0]
        for piece in pieces[1:]:
            merged = merged.appended(piece)
        return merged

    # -- events and diagnostics ----------------------------------------------

    def _event(
        self,
        request: SynthesisRequest,
        prepared: _Preparation | None,
        kind: SynthesisKind,
        sequence: int,
        audio: AudioBuffer | None,
        timings: Timings,
        reason: str = "",
    ) -> SynthesisEvent:
        return SynthesisEvent(
            kind=kind,
            sequence=sequence,
            audio=audio,
            reason=reason,
            voice_id=request.voice.voice_id if request.voice else "",
            language=request.language.bcp47,
            elapsed_ms=(monotonic_ns() - timings.started_ns) / 1e6,
            text="" if prepared is None else prepared.report.text,
        )

    @staticmethod
    def _cancelled(request: SynthesisRequest) -> bool:
        return request.token is not None and request.token.cancelled

    @staticmethod
    def _cancel_reason(request: SynthesisRequest) -> str:
        if request.token is None:
            return ""
        return request.token.reason or "cancelled"

    def _post(
        self, event: str, request: SynthesisRequest, prepared: _Preparation, timings: Timings
    ) -> None:
        if self.ledger is None:
            return
        self.ledger.record(
            DiagnosticEvent(
                component="tts.pipeline",
                event=event,
                timestamp_ns=monotonic_ns(),
                fields={
                    "language": request.language.bcp47,
                    "engine": prepared.engine.engine_id,
                    "model_id": prepared.engine.model_id,
                    "chunks": len(prepared.chunks),
                    "budget_utilisation": round(self.chunker.budget_utilisation(prepared.chunks), 3),
                    "rate": prepared.prosody.rate,
                    "pitch_semitones": prepared.prosody.pitch_semitones,
                    "volume_db": prepared.prosody.volume_db,
                    "normalisation_changes": len(prepared.report.changes),
                    "normalisation_limitations": list(prepared.report.limitations),
                    "first_result_ms": timings.first_result_ms,
                    "truncated": prepared.render.truncated,
                },
            )
        )
