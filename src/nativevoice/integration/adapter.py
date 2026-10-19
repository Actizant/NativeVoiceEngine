"""`VoiceInterface` - the single object a host application talks to.

Composition, not framework: it is given a transcriber, a synthesizer and an
output sink, and it sequences them through `session.VoiceSession`. A host that
wants a different engine swaps the object; nothing in here knows which model is
answering, and nothing in here writes to disk.

The conversation window is bounded and **in-memory only**. Persistent memory
belongs to the host: one host has its own store and another has its own, and a
voice layer keeping a second, silently-diverging transcript history is how a
system ends up with two truths. `history` exists so a caller can read back the
turns of the current conversation, and `max_history` caps it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Deque, Iterator

from collections import deque

from ..audio.io import CapturingOutput
from ..audio.vad import SpeechSegment
from ..core.audio import AudioBuffer
from ..core.cancel import CancellationToken
from ..core.errors import EngineUnavailable, VoiceError
from ..core.events import SynthesisEvent, TranscriptionEvent
from ..core.language import Capability, LanguageTag, Task
from ..session.orchestrator import SessionReport, VoiceSession
from ..session.state import SessionConfig, TurnState
from ..stt.api import TranscriptionRequest, TranscriptionResult
from ..tts.api import Prosody, SynthesisRequest
from .contract import HostReply, VoiceHost


@dataclass(slots=True)
class VoiceInterface:
    transcriber: object
    synthesizer: object | None = None
    output: object = None
    config: SessionConfig = field(default_factory=SessionConfig)
    language: LanguageTag | None = None
    voice: object | None = None
    host: object | None = None
    voices: object | None = None
    segmenter_factory: Callable[[], object] | None = None
    capability_source: Callable[[Task], dict[str, Capability]] | None = None
    max_history: int = 20

    session: VoiceSession = field(init=False)
    voice_refusal: str = ""
    history: Deque[HostReply] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.output is None:
            self.output = CapturingOutput()
        self.session = VoiceSession(
            transcriber=self.transcriber,
            synthesizer=self.synthesizer,
            output=self.output,
            config=self.config,
            language=self.language,
            voice=self.voice,
            segmenter_factory=self.segmenter_factory,
        )
        self.history = deque(maxlen=self.max_history)

    # ------------------------------------------------------------------ voice identity

    def voice_for(self, language: LanguageTag | None):
        """Resolve which voice speaks this answer, or explain why nothing can.

        Resolution is the registry's job - explicit request, then host default, then
        platform default - and an unrecorded language is a refusal, never a silent
        substitution of whichever voice happened to be installed.
        """
        if self.voices is None:
            return self.voice
        selected = self.voices.select(requested=self.voice, language=language, host=self.host)
        return selected.as_reference()

    # ------------------------------------------------------------------ hearing

    def hear(self, audio: AudioBuffer, language: LanguageTag | None = None) -> TranscriptionResult:
        """Transcribe one finished utterance in the language the host asked for.

        The session takes a language when a turn opens, not when it hears, so forcing
        one here means opening the turn with it - the same way `exchange()` and
        `stream_hearing()` do. Forwarding it to `hear()` would ignore the request.
        """
        self.session.begin_turn(language)
        return self.session.hear(audio)

    def stream_hearing(
        self, chunks: Iterator[AudioBuffer], language: LanguageTag | None = None
    ) -> Iterator[TranscriptionEvent]:
        """Live partials while the user is still talking."""
        turn = self.session.begin_turn(language)
        for event in self.transcriber.stream(iter(chunks), language or turn.language, turn.token):
            yield event

    # ------------------------------------------------------------------ speaking

    def speak(
        self, text: str, language: LanguageTag | None = None, token: CancellationToken | None = None
    ) -> Iterator[SynthesisEvent]:
        if self.synthesizer is None:
            raise EngineUnavailable("no synthesizer is attached", task=Task.TTS.value)
        turn = self.session.turn or self.session.begin_turn(language)
        turn.state = TurnState.SPEAKING
        target = language or turn.language or self.language
        request = SynthesisRequest(
            text=text,
            language=target,
            voice=turn.voice if turn.voice is not None else self.voice_for(target),
            prosody=Prosody(),
            token=token or turn.token,
        )
        for event in self.synthesizer.stream(request):
            if turn.token.cancelled:
                break
            if event.audio is not None:
                self.output.play(event.audio, turn.token)
            yield event

    def cancel(self, reason: str = "user interrupted") -> bool:
        return self.session.interrupt(reason)

    # ------------------------------------------------------------------ end to end

    def exchange(
        self, audio: AudioBuffer, host: VoiceHost, language: LanguageTag | None = None
    ) -> HostReply:
        """One complete turn: hear, ask the host, speak the answer.

        A host that only wants text can pass `voice_output_enabled=False` in the
        session config and this same path runs unchanged - the capability checks
        still execute, and the audio goes to the silent sink. Text-only is a
        routing choice, not a different code path.
        """
        turn = self.session.begin_turn(language)
        result = self.session.hear(audio)
        reply_text = host.respond(result.text, result.language) or ""
        try:
            turn.voice = self.voice_for(result.language or language)
        except VoiceError as error:
            self.voice_refusal = f"{type(error).__name__}: {error}"
        report: SessionReport = self.session.answer(reply_text, language=result.language)
        reply = HostReply(
            utterance=result.text,
            reply=reply_text,
            language=(result.language.bcp47 if result.language else report.language),
            spoken=report.spoke,
            silenced=report.silenced,
            notes=report.notes,
            timings=report.timings,
        )
        self.history.append(reply)
        return reply

    def converse(
        self, capture: Iterator[AudioBuffer], host: VoiceHost, token: CancellationToken | None = None
    ) -> Iterator[HostReply]:
        """Segment a continuous capture stream into exchanges.

        Barge-in is wired here rather than in the session so that a host reading
        this loop sees the whole contract: incoming speech during playback ends
        the current reply and starts the next turn.
        """
        segments: Iterator[SpeechSegment] = self.session.segment_capture(capture)
        for segment in segments:
            if token is not None and token.cancelled:
                return
            if segment.audio is None:
                continue
            try:
                yield self.exchange(segment.audio, host)
            except VoiceError as error:
                yield HostReply(
                    utterance="",
                    reply="",
                    language="",
                    spoken=False,
                    silenced=True,
                    notes=(f"{type(error).__name__}: {error}",),
                )

    # ------------------------------------------------------------------ truth about the system

    def capabilities(self, task: Task = Task.STT) -> dict[str, Capability]:
        if self.capability_source is None:
            return {}
        return self.capability_source(task)

    def diagnose(self) -> dict:
        """Everything an operator needs to answer 'what can this do right now'."""
        return {
            "mode": self.config.mode.value,
            "voice_output_enabled": self.config.voice_output_enabled,
            "barge_in": self.config.barge_in_enabled,
            "transcriber": type(self.transcriber).__name__,
            "synthesizer": None if self.synthesizer is None else type(self.synthesizer).__name__,
            "output": type(self.output).__name__,
            "turn": None if self.session.turn is None else self.session.turn.state.value,
            "device_failures": self.session.device_failures,
            "recoveries": self.session.recoveries,
            "turns_in_window": len(self.history),
            "voice_refusal": self.voice_refusal,
            "history_persisted": False,
            "capabilities": {task.value: sorted(self.capabilities(task)) for task in Task},
        }

    def close(self) -> None:
        self.session.close()
