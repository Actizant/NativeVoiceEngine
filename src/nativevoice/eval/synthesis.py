"""Synthesis latency, RTF and output-integrity measurement.

WHY A SEPARATE MODULE FROM `core.Timings`. `Timings` records what an engine chose
to report; this module *re-derives* the claims from the audio that actually arrived.
An engine can say `real_time_factor=0.4` while emitting half a second of silence, so
the honest numbers are produced from the buffer: duration from frame count,
first-audio latency from the first chunk timestamp, clipping and silence from
`audio.pcm.measure`, and the produced-versus-requested duration ratio from both ends
of the request. Where a number cannot be derived (no first-audio mark was recorded,
no audio at all), the field stays `None` and a note says why, rather than defaulting
to a flattering zero.

Intelligibility is the same discipline: a round-trip WER is only meaningful if a real
transcriber with an approved model did the listening, so `intelligibility_proxy_check`
answers NOT_RUN with the specific missing ingredient unless it can cite a measured
capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Iterator

from ..audio.pcm import measure, peak, rms
from ..core.audio import AudioBuffer
from ..core.cancel import CancellationToken
from ..core.events import SynthesisKind
from ..core.language import CapabilityStatus, LanguageTag
from ..core.time import monotonic_ns
from ..core.errors import EngineUnavailable
from .metrics import word_error_rate
from .report import CriterionStatus

if TYPE_CHECKING:  # contracts only; the measurement layer must not import an engine
    from ..stt.api import Transcriber

MS_PER_S = 1000.0


def _default_request(audio: AudioBuffer, tag: LanguageTag, max_duration_s: float, token: CancellationToken | None):
    """Build a frozen `stt.api.TranscriptionRequest`, lazily.

    Lazy on purpose: `nativevoice.stt` pulls in engine adapters, and the whole
    measurement layer would become unimportable if any of that is mid-flight or
    missing. Recognition contracts are typed here, not instantiated eagerly, and an
    unimportable subsystem is reported through our own taxonomy rather than crashing
    an evaluation with a third-party or bare exception.
    """
    try:
        from ..stt.api import TranscriptionRequest
    except ImportError as exc:
        raise EngineUnavailable(f"recognition contracts unavailable: {exc}") from exc
    return TranscriptionRequest(
        audio=audio, language=tag, allow_auto_detection=False, max_duration_s=max_duration_s, token=token
    )


@dataclass(frozen=True, slots=True)
class SynthesisMeasurement:
    """Every number §14 asks for about one synthesis, or `None` with a reason."""

    engine: str
    model_id: str
    language: str
    first_audio_ms: float | None
    wall_time_ms: float
    audio_duration_s: float
    requested_duration_s: float | None
    real_time_factor: float | None
    duration_ratio: float | None
    silence_only: bool
    silence_fraction: float
    clipped_fraction: float
    rms: float
    peak: float
    eos_seen: bool
    overran_maximum: bool
    truncated: bool
    notes: tuple[str, ...] = ()

    @property
    def is_faster_than_real_time(self) -> bool | None:
        """None when RTF was not measurable - which is not the same as False."""
        if self.real_time_factor is None:
            return None
        return self.real_time_factor < 1.0

    def as_dict(self) -> dict:
        return {
            "engine": self.engine,
            "model_id": self.model_id,
            "language": self.language,
            "first_audio_ms": None if self.first_audio_ms is None else round(self.first_audio_ms, 3),
            "wall_time_ms": round(self.wall_time_ms, 3),
            "audio_duration_s": round(self.audio_duration_s, 6),
            "requested_duration_s": self.requested_duration_s,
            "real_time_factor": None if self.real_time_factor is None else round(self.real_time_factor, 4),
            "duration_ratio": None if self.duration_ratio is None else round(self.duration_ratio, 4),
            "silence_only": self.silence_only,
            "silence_fraction": round(self.silence_fraction, 4),
            "clipped_fraction": round(self.clipped_fraction, 6),
            "rms": round(self.rms, 6),
            "peak": round(self.peak, 6),
            "eos_seen": self.eos_seen,
            "overran_maximum": self.overran_maximum,
            "truncated": self.truncated,
            "notes": list(self.notes),
        }


def _duration_ratio(produced: float, requested: float | None) -> float | None:
    """Produced audio seconds divided by the span the request asked for.

    `requested` is `maximum_duration_s`, i.e. a ceiling, so a ratio near 1.0 means
    generation hit the ceiling (usually an EOS failure), not that the engine is
    accurate. The ratio is `None` when no ceiling was given: inventing an expected
    duration from the text length would be a guess dressed as a measurement.
    """
    if not requested or requested <= 0:
        return None
    return produced / requested


def measure_synthesis(
    audio: AudioBuffer,
    *,
    wall_time_ms: float,
    first_audio_ms: float | None = None,
    requested_duration_s: float | None = None,
    maximum_duration_s: float | None = None,
    eos_seen: bool = True,
    truncated: bool = False,
    engine: str = "",
    model_id: str = "",
    language: str = "",
    silence_threshold: float = 0.01,
) -> SynthesisMeasurement:
    """The one primitive: turn audio + clock facts into the reported numbers."""
    level = measure(audio, silence_threshold=silence_threshold) if audio.frame_count else None
    produced = audio.duration_s
    notes: list[str] = []
    if first_audio_ms is None:
        notes.append("first-audio latency NOT RUN: the caller recorded no first-chunk mark")
    if audio.frame_count == 0:
        notes.append("engine returned zero frames: no audio was produced")
    real_time_factor = None
    if produced > 0 and wall_time_ms >= 0:
        real_time_factor = (wall_time_ms / MS_PER_S) / produced
    else:
        notes.append("real-time factor NOT RUN: produced audio has no duration to divide by")
    ceiling = maximum_duration_s if maximum_duration_s else requested_duration_s
    overran = bool(ceiling and produced > ceiling * 1.001)
    if overran:
        notes.append(f"generation overran the {ceiling:.3f}s ceiling: end-of-sequence was not honoured")
    return SynthesisMeasurement(
        engine=engine,
        model_id=model_id,
        language=language,
        first_audio_ms=first_audio_ms,
        wall_time_ms=wall_time_ms,
        audio_duration_s=produced,
        requested_duration_s=ceiling,
        real_time_factor=real_time_factor,
        duration_ratio=_duration_ratio(produced, ceiling),
        silence_only=level is None or level.rms <= silence_threshold,
        silence_fraction=level.silence_fraction if level else 1.0,
        clipped_fraction=level.clipped_fraction if level else 0.0,
        rms=level.rms if level else 0.0,
        peak=level.peak if level else 0.0,
        eos_seen=eos_seen,
        overran_maximum=overran,
        truncated=truncated,
        notes=tuple(notes),
    )


def measure_outcome(outcome, request) -> SynthesisMeasurement:
    """Measure a `tts.api.SynthesisOutcome` against its `SynthesisRequest`.

    Timings are read from `outcome.timings` if the engine published them
    (`first_audio_ms` / `total_ms`); otherwise they are derived from the buffer's
    own monotonic marks. Nothing is assumed about units beyond what the engine states.
    """
    timings = dict(getattr(outcome, "timings", {}) or {})
    audio: AudioBuffer = outcome.audio
    first_audio_ms = timings.get("first_audio_ms")
    if first_audio_ms is None:
        started = getattr(audio, "started_at_ns", 0)
        mark = timings.get("first_audio_ns") or 0
        first_audio_ms = (mark - started) / 1e6 if mark and started else None
    wall_time_ms = timings.get("total_ms")
    if wall_time_ms is None:
        finished = timings.get("finished_ns") or 0
        started = getattr(audio, "started_at_ns", 0)
        wall_time_ms = (finished - started) / 1e6 if finished and started else 0.0
    return measure_synthesis(
        audio,
        wall_time_ms=float(wall_time_ms),
        first_audio_ms=None if first_audio_ms is None else float(first_audio_ms),
        requested_duration_s=getattr(request, "maximum_duration_s", None),
        maximum_duration_s=getattr(request, "maximum_duration_s", None),
        eos_seen=not bool(getattr(outcome, "truncated", False)),
        truncated=bool(getattr(outcome, "truncated", False)),
        engine=getattr(outcome, "engine", "") or "",
        model_id=getattr(outcome, "model_id", "") or "",
        language=getattr(getattr(outcome, "language", None), "bcp47", "") or str(getattr(outcome, "language", "") or ""),
    )


@dataclass(slots=True)
class StreamAccumulator:
    """Collapses a synthesis event stream into the facts `measure_synthesis` needs.

    First-audio latency is taken from the first chunk that actually carried samples,
    not from a `first_audio` label, because a mislabelled or empty leading chunk is a
    real engine behaviour and the listener experiences the buffer, not the label.
    """

    started_ns: int
    token: CancellationToken | None = None
    chunks: list = field(default_factory=list)
    first_audio_ns: int = 0
    eos_seen: bool = False
    failed_reason: str = ""
    cancelled: bool = False
    sequence_gaps: int = 0
    _last_sequence: int = -1

    def consume(self, events: Iterator) -> "StreamAccumulator":
        for event in events:
            self._observe(event)
        return self

    def _observe(self, event) -> None:
        sequence = getattr(event, "sequence", self._last_sequence + 1)
        if self._last_sequence >= 0 and sequence > self._last_sequence + 1:
            self.sequence_gaps += 1
        self._last_sequence = sequence
        kind = getattr(event, "kind", None)
        audio = getattr(event, "audio", None)
        if audio is not None and audio.frame_count:
            if not self.first_audio_ns:
                self.first_audio_ns = getattr(event, "timestamp_ns", 0) or monotonic_ns()
            self.chunks.append(audio)
        if kind is SynthesisKind.EOS:
            self.eos_seen = True
        elif kind is SynthesisKind.FAILED:
            self.failed_reason = getattr(event, "reason", "") or "engine reported failure"
        elif kind is SynthesisKind.CANCELLED:
            self.cancelled = True

    def merged(self) -> AudioBuffer | None:
        if not self.chunks:
            return None
        joined = self.chunks[0]
        for extra in self.chunks[1:]:
            joined = joined.appended(extra)
        return joined

    def measurement(self, *, request, engine: str = "", model_id: str = "", language: str = "") -> SynthesisMeasurement:
        audio = self.merged()
        finished = monotonic_ns()
        wall = (finished - self.started_ns) / 1e6
        first = None if not self.first_audio_ns else (self.first_audio_ns - self.started_ns) / 1e6
        if audio is None:
            notes = ["no audio chunk ever arrived"]
            if self.failed_reason:
                notes.append(f"stream failed: {self.failed_reason}")
            return SynthesisMeasurement(
                engine=engine, model_id=model_id, language=language,
                first_audio_ms=None, wall_time_ms=wall, audio_duration_s=0.0,
                requested_duration_s=getattr(request, "maximum_duration_s", None),
                real_time_factor=None, duration_ratio=None,
                silence_only=True, silence_fraction=1.0, clipped_fraction=0.0, rms=0.0, peak=0.0,
                eos_seen=self.eos_seen, overran_maximum=False, truncated=False,
                notes=tuple(notes),
            )
        result = measure_synthesis(
            audio,
            wall_time_ms=wall,
            first_audio_ms=first,
            requested_duration_s=getattr(request, "maximum_duration_s", None),
            maximum_duration_s=getattr(request, "maximum_duration_s", None),
            eos_seen=self.eos_seen,
            engine=engine,
            model_id=model_id,
            language=language,
        )
        if not self.eos_seen and not self.cancelled:
            result = _add_note(result, "stream ended without EOS: synthesis was cut off, not completed")
        if self.sequence_gaps:
            result = _add_note(result, f"{self.sequence_gaps} sequence gap(s) observed in the audio stream")
        return result


def _add_note(measurement: SynthesisMeasurement, note: str) -> SynthesisMeasurement:
    return replace(measurement, notes=measurement.notes + (note,))


@dataclass(frozen=True, slots=True)
class ProxyCheck:
    """Round-trip intelligibility result, or the precise reason it was NOT RUN."""

    status: CriterionStatus
    round_trip_wer: float | None
    transcriber: str
    model_id: str
    language: str
    reason: str
    hypotheses: tuple[str, ...] = ()

    @property
    def measured(self) -> bool:
        return self.status is not CriterionStatus.NOT_RUN and self.round_trip_wer is not None

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "round_trip_wer": None if self.round_trip_wer is None or self.round_trip_wer == float("inf") else round(self.round_trip_wer, 6),
            "transcriber": self.transcriber,
            "model_id": self.model_id,
            "language": self.language,
            "reason": self.reason,
            "hypotheses": list(self.hypotheses),
        }


def intelligibility_proxy_check(
    audio: AudioBuffer,
    spoken_text: str,
    language: "LanguageTag | str",
    *,
    transcriber: "Transcriber | None" = None,
    token: CancellationToken | None = None,
    max_duration_s: float = 30.0,
    request_builder: Callable[..., object] | None = None,
) -> ProxyCheck:
    """WER of "synthesise then transcribe", or NOT_RUN naming exactly what is missing.

    A round-trip number is a *quality* claim about two engines at once, so it is only
    emitted when the listening engine reports a real capability: `NO_MODEL`,
    `UNSUPPORTED` or an absent capability record all refuse, as does a missing model id
    on an engine that should have one. Never `PASS` without a live transcriber call.
    """
    tag = language if isinstance(language, LanguageTag) else LanguageTag.parse(str(language))
    if transcriber is None:
        return ProxyCheck(
            CriterionStatus.NOT_RUN, None, "", "", tag.bcp47,
            "no transcriber supplied: round-trip intelligibility needs a real listening engine",
        )
    if audio.frame_count == 0:
        return ProxyCheck(
            CriterionStatus.NOT_RUN, None, _describe(transcriber), "", tag.bcp47,
            "synthesis produced no audio: there is nothing to transcribe",
        )
    capability = None
    probe = getattr(transcriber, "language_capability", None)
    if callable(probe):
        try:
            capability = probe(tag)
        except Exception as exc:  # a transcriber that cannot answer is not an approved one
            return ProxyCheck(
                CriterionStatus.NOT_RUN, None, _describe(transcriber), "", tag.bcp47,
                f"transcriber refused the capability query: {type(exc).__name__}: {exc}",
            )
    if capability is None:
        return ProxyCheck(
            CriterionStatus.NOT_RUN, None, _describe(transcriber), "", tag.bcp47,
            "transcriber reports no capability record for this language",
        )
    status = CapabilityStatus(capability.status)
    model_id = capability.model_id or _model_id(transcriber)
    if status in (CapabilityStatus.NO_MODEL, CapabilityStatus.UNSUPPORTED):
        return ProxyCheck(
            CriterionStatus.NOT_RUN, None, capability.engine or _describe(transcriber), model_id, tag.bcp47,
            f"transcriber has no approved model for this language (status={status.value})",
        )
    if not model_id:
        return ProxyCheck(
            CriterionStatus.NOT_RUN, None, capability.engine or _describe(transcriber), "", tag.bcp47,
            "transcriber is usable but names no model_id: the result could not be attributed",
        )
    builder = request_builder or _default_request
    try:
        request = builder(audio, tag, max_duration_s, token)
    except Exception as exc:
        return ProxyCheck(
            CriterionStatus.NOT_RUN, None, capability.engine or _describe(transcriber), model_id, tag.bcp47,
            f"no recognition request could be built: {exc}",
        )
    result = transcriber.transcribe(request)
    hypothesis = result.text or ""
    wer = word_error_rate(spoken_text, hypothesis)
    return ProxyCheck(
        CriterionStatus.PASS if wer == 0.0 else CriterionStatus.FAIL,
        wer,
        capability.engine or _describe(transcriber),
        model_id,
        tag.bcp47,
        "measured round trip" if wer == 0.0 else f"round-trip WER {wer:.4f} is not zero",
        hypotheses=(hypothesis,) if hypothesis else (),
    )


def _describe(transcriber) -> str:
    return getattr(transcriber, "engine_id", "") or type(transcriber).__name__


def _model_id(transcriber) -> str:
    return getattr(transcriber, "model_id", "") or ""


def silence_only(audio: AudioBuffer, *, threshold: float = 0.01) -> bool:
    """Standalone gate used by report writers: did the engine emit mere silence?"""
    if audio.frame_count == 0:
        return True
    return rms(audio.samples) <= threshold and peak(audio.samples) <= threshold


def clipping_fraction(audio: AudioBuffer) -> float:
    if audio.frame_count == 0:
        return 0.0
    return measure(audio).clipped_fraction
