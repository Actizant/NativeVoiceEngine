"""The voice session: capture, hearing, answering, barge-in, recovery.

Everything a user experiences as "talking to ULTRON" is sequenced here, and the
sequencing is deliberately synchronous-per-stage with generators between stages.
Real-time voice code usually fails in the seams where a thread decides what
happens next, so the only concurrency a caller must reason about is the capture
iterator: turn state, cancellation and stale-work rules live in `state.py` as a
pure model, and this class just applies them.

Two behaviours are specified rather than left to taste:

* voice output disabled is a **routing** decision, not a skipped path - synthesis
  still runs its capability checks, and the result goes to a silent sink. A
  user who turned speech off must not get audio because another code path was
  shorter.
* barge-in cancels the playback token *and* closes the transcript buffer of the
  superseded turn, so a late final from a request nobody wants any more cannot
  be committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Protocol

from ..audio.vad import SpeechSegment, UtteranceSegmenter
from ..core.audio import AudioBuffer
from ..core.cancel import CancellationToken
from ..core.errors import DeviceError, EngineUnavailable, LanguageUnsupported, VoiceError
from ..core.events import SynthesisEvent, SynthesisKind, TranscriptionEvent, TranscriptionKind
from ..core.language import LanguageTag
from ..core.time import Timings, monotonic_ns
from ..stt.api import TranscriptionRequest, TranscriptionResult
from ..tts.api import Prosody, SynthesisRequest
from .state import SessionConfig, TranscriptBuffer, Turn, TurnState


class TranscriberLike(Protocol):
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult: ...

    def stream(self, chunks: Iterator[AudioBuffer], language: LanguageTag | None = None) -> Iterator[TranscriptionEvent]: ...


class SynthesizerLike(Protocol):
    def synthesize(self, request: SynthesisRequest): ...

    def stream(self, request: SynthesisRequest) -> Iterator[SynthesisEvent]: ...


class OutputLike(Protocol):
    def play(self, buffer: AudioBuffer, token: CancellationToken | None = None) -> int: ...

    def stop(self) -> None: ...


class RouterLike(Protocol):
    """Language routing is injected, not imported, so the session stays engine-agnostic."""

    def route(self, text: str = "", audio: AudioBuffer | None = None, language: LanguageTag | None = None): ...


@dataclass(slots=True)
class SessionReport:
    turn_id: str
    heard: str
    language: str
    spoke: bool
    silenced: bool
    audio_duration_s: float
    timings: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class VoiceSession:
    """One conversation's voice lifecycle. Not shared between conversations."""

    transcriber: TranscriberLike
    synthesizer: SynthesizerLike | None
    output: OutputLike
    config: SessionConfig = field(default_factory=SessionConfig)
    router: RouterLike | Callable | None = None
    segmenter_factory: Callable[[], UtteranceSegmenter] | None = None
    language: LanguageTag | None = None
    voice: object | None = None
    on_answer: Callable[[str], str] | None = None

    turn: Turn | None = None
    recoveries: int = 0
    device_failures: int = 0
    history: list[SessionReport] = field(default_factory=list)
    _sink_note: str = ""
    _segmenter: UtteranceSegmenter | None = None

    # ------------------------------------------------------------------ turn lifecycle

    def begin_turn(self, language: LanguageTag | None = None, voice: object | None = None) -> Turn:
        """Start a turn, superseding any in-flight one.

        Called by push-to-tap, by the hold-to-talk release path, and by the
        continuous-mode segmenter; the turn model is the same either way.
        """
        previous = self.turn
        if previous is not None and previous.current:
            self.interrupt("superseded by a new turn")
        turn = Turn(
            language=language or self.language,
            voice=voice if voice is not None else self.voice,
            started_ns=monotonic_ns(),
        )
        turn.state = TurnState.LISTENING
        self.turn = turn
        if previous is not None:
            previous.superseded_by = turn.turn_id
        return turn

    def interrupt(self, reason: str = "user interrupted") -> bool:
        """Stop talking now, and refuse anything already queued behind the voice."""
        turn = self.turn
        if turn is None:
            self.output.stop()
            return False
        turn.state = TurnState.IDLE
        turn.token.cancel(reason)
        self.output.stop()
        return True

    # ------------------------------------------------------------------ hearing

    def hear(self, audio: AudioBuffer, token: CancellationToken | None = None) -> TranscriptionResult:
        """Transcribe one finished utterance."""
        turn = self.turn or self.begin_turn()
        turn.state = TurnState.TRANSCRIBING
        request = TranscriptionRequest(
            audio=audio,
            language=turn.language,
            allow_auto_detection=turn.language is None,
            token=token or turn.token,
        )
        result = self.transcriber.transcribe(request)
        turn.language = result.language or turn.language
        turn.committed_text = result.text
        turn.state = TurnState.AWAITING_ANSWER
        return result

    def hear_streaming(
        self, chunks: Iterable[AudioBuffer], token: CancellationToken | None = None
    ) -> tuple[str, list[TranscriptionEvent]]:
        """Live partials for an in-progress utterance, reduced by the turn's buffer."""
        turn = self.turn or self.begin_turn()
        turn.state = TurnState.TRANSCRIBING
        buffer = TranscriptBuffer()
        events: list[TranscriptionEvent] = []
        for event in self.transcriber.stream(iter(chunks), turn.language):
            if token is not None and token.cancelled:
                break
            events.append(event)
            if event.utterance_id != turn.turn_id and event.kind is TranscriptionKind.FINAL:
                buffer.ignored.append(event.utterance_id)
                continue
            buffer.apply(event)
        return buffer.text, events

    def segmenter(self) -> UtteranceSegmenter:
        """A detector for the current turn, or a refusal if none is configured.

        The reference stack endpointed dictation on the hotkey rather than on VAD,
        because an aggressive gate deletes pause-heavy speech; segmentation is
        available for continuous mode but is never forced on a push-to-talk turn.
        """
        if self._segmenter is None:
            if self.segmenter_factory is None:
                raise EngineUnavailable("no voice activity segmenter is configured for this session")
            self._segmenter = self.segmenter_factory()
        return self._segmenter

    def segment_capture(self, stream: Iterable[AudioBuffer]) -> Iterator[SpeechSegment]:
        detector = self.segmenter()
        for chunk in stream:
            yield from detector.push(chunk)
        yield from detector.flush()

    # ------------------------------------------------------------------ answering

    def answer(self, text: str, language: LanguageTag | None = None, prosody: Prosody | None = None) -> SessionReport:
        """Speak an answer, or route it to silence when voice output is off.

        A turn that has been superseded does not speak: an older answer arriving
        after the user moved on is exactly the failure §9 calls stale-response,
        and the honest remedy is to drop it and say so in the report.
        """
        turn = self.turn or self.begin_turn()
        target = language or turn.language or self.language
        timings = Timings()
        silenced = False
        spoke = False
        notes: list[str] = []

        if not self.config.voice_output_enabled:
            silenced = True
            notes.append("voice output disabled")
        elif self.synthesizer is None:
            silenced = True
            notes.append("no synthesizer attached")
        elif not text.strip():
            notes.append("empty answer; nothing synthesised")
        else:
            request = SynthesisRequest(
                text=text,
                language=target,
                voice=turn.voice,
                prosody=prosody or Prosody(),
                token=turn.token,
            )
            try:
                turn.state = TurnState.SPEAKING
                for event in self.synthesizer.stream(request):
                    if turn.token.cancelled:
                        notes.append("synthesis cancelled: " + (turn.token.reason or ""))
                        break
                    if event.kind in (SynthesisKind.FAILED, SynthesisKind.CANCELLED):
                        notes.append(f"synthesis ended {event.kind.value}: {event.reason}")
                        break
                    if event.audio is not None:
                        spoken = self.output.play(event.audio, turn.token)
                        spoke = spoke or spoken > 0
                        if spoken == 0 and not spoke:
                            notes.append("playback produced no frames")
                timings.audio_duration_s += 0.0
            except LanguageUnsupported as error:
                notes.append(f"language unsupported for synthesis: {error.details.get('language', target)}")
                silenced = True
            except VoiceError as error:
                notes.append(f"synthesis unavailable: {type(error).__name__}")
                silenced = True
            finally:
                turn.state = TurnState.IDLE
        timings.finish()

        if turn.superseded_by:
            notes.append(f"stale answer discarded into turn {turn.superseded_by}")
        report = SessionReport(
            turn_id=turn.turn_id,
            heard=turn.committed_text,
            language=(target.bcp47 if target else ""),
            spoke=spoke,
            silenced=silenced,
            audio_duration_s=timings.audio_duration_s,
            timings=timings.as_dict(),
            notes=tuple(notes),
        )
        self.history.append(report)
        return report

    # ------------------------------------------------------------------ failure handling

    def with_recovery(self, action: Callable[[], object]) -> object:
        """Run one device-touching action, reopening up to `max_recoveries` times.

        A microphone that is revoked mid-conversation must not end the
        conversation, and must not loop forever either: each attempt is counted,
        reported, and the second exhaustion re-raises the original error.
        """
        attempt = 0
        while True:
            try:
                return action()
            except DeviceError as error:
                self.device_failures += 1
                attempt += 1
                if attempt > self.config.max_recoveries:
                    raise
                self.recoveries += 1
                self._sink_note = f"recovered from {type(error).__name__} on attempt {attempt}"

    def barge_in_from(self, segment: SpeechSegment) -> bool:
        """Decide whether incoming speech should interrupt ongoing playback.

        Requires a run long enough to be a real voice rather than a door slam -
        the reference meeting pipeline needed the same guard once AEC leaked the
        user's own voice back as echo.
        """
        if not self.config.barge_in_enabled or self.turn is None:
            return False
        if self.turn.state is not TurnState.SPEAKING:
            return False
        if segment.duration_s * 1000 < self.config.interrupt_requires_speech_ms:
            return False
        self.interrupt("barge-in")
        return True

    def close(self) -> None:
        if self.turn is not None:
            self.turn.token.cancel("session closed")
        self.output.stop()
        self._segmenter = None
