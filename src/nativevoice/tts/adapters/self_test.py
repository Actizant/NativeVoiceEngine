"""Plumbing verification engine: it proves the pipe, not the speech.

This is *not* a text-to-speech synthesizer and must never be reachable in a
production decision. It emits deterministic pre-built buffers so that streaming,
cancellation, prosody routing, crossfade assembly and duration truncation can be
tested without a model. Its capability table is the safety mechanism: every real
language is declared `UNSUPPORTED`, so a router that insists on a usable
declaration can never select it.

One synthetic language code is declared `PARTIAL` - `zxx`, the ISO 639-2 code for
"no linguistic content". That is what lets the pipeline be exercised end to end
while still being unable to satisfy a real request: a test asking for Bengali
gets refused exactly as production would.
"""

from __future__ import annotations

import array
import math
from collections.abc import Iterator, Sequence

from ...core.audio import AudioBuffer, AudioFormat
from ...core.cancel import guard
from ...core.errors import ConfigurationError, ProfileInvalid
from ...core.events import SynthesisEvent, SynthesisKind
from ...core.language import Capability, CapabilityStatus, LanguageTag, Task
from ...core.time import monotonic_ns
from ..api import SynthesisOutcome, SynthesisRequest, VoiceReference

PLUMBING_LANGUAGE = "zxx"

#: Languages a real engine might claim; all are refused here so the registry can
#: never route a user request to this adapter.
PRODUCTION_LANGUAGES = (
    "en", "bn", "hi", "zh", "fr", "de", "es", "it", "pt", "ja", "ko", "ru", "ar", "tr",
)

_LIMITATION = "plumbing verification only: emits scripted buffers, contains no speech"


class ScriptedSynthesisEngine:
    """Deterministic buffers for testing the pipeline. Not speech synthesis."""

    def __init__(
        self,
        buffers: Sequence[AudioBuffer] | None = None,
        *,
        engine_id: str = "self-test",
        model_id: str = "scripted-plumbing",
        sample_rate: int = 24000,
        events_per_chunk: int = 4,
        chunk_duration_s: float = 0.25,
    ) -> None:
        self.engine_id = engine_id
        self.model_id = model_id
        self.output_format = AudioFormat(sample_rate, 1)
        self.supports_streaming = True
        self.controllable = ("rate", "pitch_semitones", "volume_db", "pause_after_s")
        self.events_per_chunk = max(1, events_per_chunk)
        self.chunk_duration_s = chunk_duration_s
        self._scripted = list(buffers) if buffers is not None else []
        self._cursor = 0
        self.calls: list[SynthesisRequest] = []
        self.conditions: list[VoiceReference] = []
        self.warm_calls = 0
        self.closed = False

    # -- declarations --------------------------------------------------------

    def capabilities(self) -> tuple[Capability, ...]:
        out = [
            Capability(
                task=Task.TTS,
                status=CapabilityStatus.UNSUPPORTED,
                language=code,
                engine=self.engine_id,
                limitations=(_LIMITATION,),
                evidence="hard refusal: this adapter is not a speech model",
            )
            for code in PRODUCTION_LANGUAGES
        ]
        out.append(
            Capability(
                task=Task.TTS,
                status=CapabilityStatus.PARTIAL,
                language=PLUMBING_LANGUAGE,
                engine=self.engine_id,
                model_id=self.model_id,
                limitations=(_LIMITATION,),
                evidence="deterministic scripted audio, used only to test transport",
            )
        )
        return tuple(out)

    def voices(self) -> tuple[VoiceReference, ...]:
        return (
            VoiceReference(
                voice_id=f"{self.engine_id}-default",
                consent_token="plumbing-consent",
                owner="plumbing",
                source="self_test adapter default condition",
            ),
        )

    def condition(self, reference: VoiceReference) -> VoiceReference:
        if reference.audio is None:
            raise ConfigurationError("a scripted condition still needs reference audio")
        if not reference.has_consent():
            raise ProfileInvalid("reference audio has no consent record", voice_id=reference.voice_id)
        self.conditions.append(reference)
        return reference

    def warm(self) -> None:
        self.warm_calls += 1

    def close(self) -> None:
        self.closed = True

    # -- rendering -----------------------------------------------------------

    def synthesize(self, request: SynthesisRequest) -> SynthesisOutcome:
        guard(request.token)
        if self.closed:
            raise ConfigurationError("engine is closed", engine_id=self.engine_id)
        self.calls.append(request)
        started = monotonic_ns()
        audio = self._next_buffer(request)
        return SynthesisOutcome(
            audio=audio,
            language=request.language,
            voice_id=request.voice.voice_id if request.voice else "",
            engine=self.engine_id,
            model_id=self.model_id,
            timings={
                "total_ms": (monotonic_ns() - started) / 1e6,
                "prosody": {
                    "rate": request.prosody.rate,
                    "pitch_semitones": request.prosody.pitch_semitones,
                    "volume_db": request.prosody.volume_db,
                    "pause_after_s": request.prosody.pause_after_s,
                },
                "text": request.text,
            },
        )

    def stream(self, request: SynthesisRequest) -> Iterator[SynthesisEvent]:
        guard(request.token)
        self.calls.append(request)
        audio = self._next_buffer(request)
        span = max(1, audio.frame_count // self.events_per_chunk)
        sequence = 0
        for piece in audio.frames(span):
            guard(request.token)
            sequence += 1
            yield SynthesisEvent(
                kind=SynthesisKind.AUDIO,
                sequence=sequence,
                audio=piece,
                language=request.language.bcp47,
                voice_id=request.voice.voice_id if request.voice else "",
            )
        sequence += 1
        yield SynthesisEvent(
            kind=SynthesisKind.EOS, sequence=sequence, reason="completed",
            language=request.language.bcp47,
        )

    def _next_buffer(self, request: SynthesisRequest) -> AudioBuffer:
        index = self._cursor % max(1, len(self._scripted))
        if self._scripted:
            self._cursor += 1
            return self._scripted[index]
        self._cursor += 1
        return self._generated(request, index)

    def _generated(self, request: SynthesisRequest, index: int) -> AudioBuffer:
        """A labelled test tone: constant amplitude, one frequency per call index.

        Deliberately a tone rather than a formant fake. It has to be recognisable
        as *not speech* in a spectrogram, otherwise a broken pipeline could pass
        review by looking plausible.
        """
        rate = self.output_format.sample_rate
        frames = max(1, int(self.chunk_duration_s * rate))
        frequency = 180.0 + 60.0 * index
        amplitude = 0.2 * (32767.0 / 32768.0)
        samples = array.array(
            "h",
            (
                int(amplitude * 32768.0 * math.sin(2 * math.pi * frequency * position / rate))
                for position in range(frames)
            ),
        )
        return AudioBuffer(samples, self.output_format, monotonic_ns(), f"{self.engine_id}:{request.text[:24]}")
