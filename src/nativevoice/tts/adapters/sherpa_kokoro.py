"""Real synthesis engine: Kokoro-82M through sherpa-onnx's offline TTS.

This is the first adapter that genuinely speaks. It turns an *approved* Kokoro
ONNX asset into audible int16 audio, routed through the same gates the rest of the
stack uses: the product-scope guard (``models.approval.ensure_product_scope``), the
licence gate (``models.approval.ensure_approved``) and integrity verification
(``models.integrity.inspect``) all run before sherpa-onnx is imported or
a single weight is read, and ``ModelNotApproved`` / ``IntegrityError`` /
``ModelNotInstalled`` propagate unchanged so a caller sees the real reason a voice
is silent.

Honesty about what "streaming" means here. sherpa-onnx's ``OfflineTts.generate`` is
whole-utterance: it returns a finished buffer, not frames as they are produced. So
``supports_streaming`` claims *chunk-level* streaming only - the long text is split
by ``tts.chunking`` and one ``SynthesisEvent`` is emitted per chunk, which gives
real first-audio latency savings but not frame-level synthesis. A caller that needs
true frame streaming does not have it, and the docstrings say so rather than
implying otherwise.

Languages are not invented. The asset ships an ``en`` (espeak-ng ``en-us``) path and
a ``zh`` (lexicon + jieba dict) path; those are the two the provisioned manifest
records and the two the front-end can actually render. A Kokoro speaker id is just
an index into ``voices.bin`` - it carries no language metadata - so per-speaker
language is *recorded from the manifest*, never fabricated. ``condition`` raises:
this is a fixed-speaker model, not a zero-shot cloner.
"""

from __future__ import annotations

from pathlib import Path

from ..api import Prosody, SynthesisOutcome, SynthesisRequest, VoiceReference
from ..chunking import TextChunker
from ...audio.pcm import from_float_samples
from ...core.audio import AudioBuffer, AudioFormat
from ...core.cancel import guard
from ...core.errors import (
    CapabilityUnavailable,
    ConfigurationError,
    IntegrityError,
    LanguageUnsupported,
    ModelNotInstalled,
)
from ...core.events import SynthesisEvent, SynthesisKind
from ...core.language import Capability, CapabilityStatus, LanguageTag, Task
from ...core.time import Timings, monotonic_ns
from ...config.paths import models_dir
from ...models.api import ModelManifest, ReviewStatus
from ...models.approval import ensure_approved, ensure_product_scope
from ...models.integrity import base_directory
from ...models.integrity import inspect as inspect_integrity

ENGINE_ID = "sherpa-onnx-kokoro"
SPEAKER_ID_PREFIX = "kokoro-"

# Which asset files the sherpa Kokoro front-end needs, and the language each
# renders. English uses the espeak-ng data dir + US lexicon; Mandarin uses the
# Han lexer + jieba dict + pinyin/fst files. These paths are relative to the
# asset directory and are checked for presence, never invented.
_LANG_LEXICON = {
    "en": ("lexicon-us-en.txt", "espeak-ng-data"),
    "zh": ("lexicon-zh.txt", "dict"),
}
# sherpa rejects lang="us"; the verified working eSpeak-ng voice name is "en-us".
_LANG_ESPEAK = {"en": "en-us"}

# sherpa's `speed` is inversely related to Prosody.rate: a larger speed factor
# speaks faster. The clamp is the engine's own, applied after the pipeline clamp.
_MIN_SPEED = 0.1
_MAX_SPEED = 5.0


def resolve_asset_dir(manifest: ModelManifest, root: Path | None = None) -> Path:
    """The directory a provisioned asset lives in, derived from config.

    The registry resolves integrity against ``models/<model_id>/``; the absolute
    path is never hardcoded in library code - it comes from ``config.paths``.
    """
    base = Path(root) if root is not None else Path(models_dir())
    return base_directory(manifest, base)


def _required_files(asset_dir: Path, languages: tuple[str, ...]) -> tuple[str, ...]:
    """Named asset pieces the front-end needs for the requested languages.

    Returns the missing names (empty when all present). The check is by path name
    so the error tells the user exactly which file is absent from the asset.
    """
    needed = ["model.int8.onnx", "voices.bin", "tokens.txt", "LICENSE"]
    for code in languages:
        for piece in _LANG_LEXICON.get(code, ()):
            needed.append(piece)
    missing = []
    for name in needed:
        if name in missing:
            continue
        if not (asset_dir / name).exists():
            missing.append(name)
    return tuple(missing)


def _clamp_speed(rate: float) -> float:
    """Map ``Prosody.rate`` (higher == faster) onto sherpa's ``speed`` factor."""
    return max(_MIN_SPEED, min(_MAX_SPEED, float(rate)))


def _volume_gain(volume_db: float) -> float:
    """Decibels to a linear int16 headroom scale, bounded so it cannot clip hard."""
    if volume_db == 0.0:
        return 1.0
    gain = 10.0 ** (volume_db / 20.0)
    return max(0.0, min(gain, 1.5))


class SherpaKokoroEngine:
    """A ``tts.api.SynthesisEngine`` backed by sherpa-onnx + Kokoro weights."""

    engine_id = ENGINE_ID
    output_format = AudioFormat(24000, 1)
    # Chunk-level streaming, NOT frame-level (see module docstring).
    supports_streaming = True
    controllable = ("rate", "volume_db", "pause_after_s")

    def __init__(
        self,
        manifest: ModelManifest,
        asset_dir: Path | str | None = None,
        *,
        num_threads: int = 2,
        max_num_sentences: int = 1,
        chunker: TextChunker | None = None,
    ) -> None:
        if manifest.task is not Task.TTS:
            raise ConfigurationError(
                "manifest is not a TTS asset", model_id=manifest.model_id, task=manifest.task.value
            )
        self.manifest = manifest
        self.model_id = manifest.model_id
        self.asset_dir = Path(asset_dir) if asset_dir is not None else resolve_asset_dir(manifest)
        self.num_threads = max(1, int(num_threads))
        self.max_num_sentences = max(1, int(max_num_sentences))
        # 24 tokens, not the 48 a quality-first reading would pick: measured on this
        # machine at the best thread count, time-to-first-audio for a two-sentence
        # reply fell from 11,545 ms to 5,061 ms (artifacts/benchmarks/tts-latency-lab.json).
        # RTF barely moved, so the cost is more crossfade seams, not throughput. A host
        # that prefers continuity over responsiveness passes its own chunker.
        self.chunker = chunker or TextChunker(max_tokens=24)
        self._tts = None
        self._capability_override: Capability | None = None
        self._closed = False

    # -- gates ----------------------------------------------------------------

    def _require_approved(self) -> None:
        # Propagates ModelNotApproved unchanged. A manifest that has not passed the
        # licence gate never reaches the runtime - and scope is asked first, because
        # an approval override says nothing about which path an asset may serve.
        ensure_product_scope(self.manifest)
        ensure_approved(self.manifest)

    def _require_verified(self) -> None:
        if not self.asset_dir.is_dir():
            raise ModelNotInstalled(
                "Kokoro asset directory is absent",
                model_id=self.model_id,
                expected_path=str(self.asset_dir),
            )
        report = inspect_integrity(self.manifest, self.asset_dir)
        if report.missing:
            raise ModelNotInstalled(
                "Kokoro asset files are missing on disk",
                model_id=self.model_id,
                missing=report.missing,
                reasons=report.reasons,
            )
        if not report.ok:
            raise IntegrityError(
                "Kokoro asset failed integrity verification",
                model_id=self.model_id,
                reasons=report.reasons,
            )

    def _require_present(self) -> None:
        if not self.asset_dir.is_dir():
            raise ModelNotInstalled(
                "no Kokoro asset directory is present",
                model_id=self.model_id,
                expected_path=str(self.asset_dir),
            )
        missing = _required_files(self.asset_dir, self._renderable_languages())
        if missing:
            raise ModelNotInstalled(
                "Kokoro asset directory is missing required files",
                model_id=self.model_id,
                expected_path=str(self.asset_dir),
                missing=missing,
            )

    # -- construction of the sherpa config (no runtime import yet) ------------

    def _lang_for(self, language: LanguageTag | None) -> str:
        """Map a request language to a front-end this asset can render.

        Refuses rather than silently falling back to English: an espeak-ng
        English lexer reading Devanagari or Bengali produces romanised gibberish
        that *is* audio but is *not* the language, which is the exact trap a
        "supported: true" flag hides behind.
        """
        code = (language or LanguageTag("en")).root().code
        if code not in _LANG_LEXICON or code not in self._renderable_languages():
            raise LanguageUnsupported(
                "this Kokoro asset has no front-end for this language",
                language=code,
                task=Task.TTS.value,
                renderable=self._renderable_languages(),
                remedy="only the languages recorded and present in the asset are synthesable",
            )
        return code

    def _renderable_languages(self) -> tuple[str, ...]:
        declared = tuple(code for code in (self.manifest.languages or ()) if code in _LANG_LEXICON)
        return declared or ("en",)

    def _build_config(self, language: str):
        import sherpa_onnx  # lazy: only touched after the gates pass

        kokoro_kwargs = dict(
            model=str(self.asset_dir / "model.int8.onnx"),
            voices=str(self.asset_dir / "voices.bin"),
            tokens=str(self.asset_dir / "tokens.txt"),
        )
        if language == "en":
            kokoro_kwargs["lexicon"] = str(self.asset_dir / _LANG_LEXICON["en"][0])
            kokoro_kwargs["data_dir"] = str(self.asset_dir / "espeak-ng-data")
            # sherpa rejects lang="us"; the verified working eSpeak-ng voice is "en-us".
            kokoro_kwargs["lang"] = _LANG_ESPEAK["en"]
        elif language == "zh":
            kokoro_kwargs["lexicon"] = str(self.asset_dir / _LANG_LEXICON["zh"][0])
            kokoro_kwargs["dict_dir"] = str(self.asset_dir / "dict")
            # The pinyin/number/date .fst files sit at the asset root for Mandarin.
            kokoro_kwargs["data_dir"] = str(self.asset_dir)
        else:
            raise LanguageUnsupported(
                "no front-end is wired for this language", language=language, task=Task.TTS.value
            )
        kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(**kokoro_kwargs)
        model = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=kokoro, num_threads=self.num_threads, provider="cpu"
        )
        return sherpa_onnx.OfflineTtsConfig(
            model=model, max_num_sentences=self.max_num_sentences
        )

    def warm(self, language: str | None = None) -> None:
        """Load the sherpa TTS instance, gated behind scope, approval and integrity."""
        if self._closed:
            raise ConfigurationError("engine is closed", engine_id=self.engine_id)
        if self._tts is not None and getattr(self, "_loaded_language", None) == (
            language if language in _LANG_LEXICON else self._renderable_languages()[0]
        ):
            return
        self._require_approved()
        self._require_verified()
        self._require_present()
        import sherpa_onnx

        lang = language if language in _LANG_LEXICON else self._renderable_languages()[0]
        self._tts = sherpa_onnx.OfflineTts(self._build_config(lang))
        self._loaded_language = lang

    # -- declarations ---------------------------------------------------------

    def capabilities(self) -> tuple[Capability, ...]:
        # A measured override (set by an eval run) wins for the language it covers;
        # everything else stays DECLARED. No status is upgraded without a number.
        measured = {}
        if self._capability_override is not None:
            measured[self._capability_override.language] = self._capability_override
        out: list[Capability] = []
        for code in self._renderable_languages():
            override = measured.get(code)
            if override is not None:
                out.append(override)
                continue
            out.append(
                Capability(
                    task=Task.TTS,
                    status=CapabilityStatus.DECLARED,
                    language=code,
                    engine=self.engine_id,
                    model_id=self.model_id,
                    limitations=(
                        "asset is installed, approved and front-end present, but intelligibility "
                        "has not been measured on this machine",
                    ),
                    evidence=f"provisioned manifest {self.manifest.model_id}",
                )
            )
        return tuple(out)

    def record_measured(self, capability: Capability) -> None:
        """Attach a MEASURED capability produced by an evaluation run.

        Only an actual measurement may call this; it is the single way an engine
        capability moves from DECLARED to MEASURED, and the caller must supply the
        metric and evidence the measurement produced.
        """
        if capability.status is not CapabilityStatus.MEASURED:
            raise ConfigurationError(
                "only a measured capability can be recorded", status=capability.status.value
            )
        if capability.metric_value is None or not capability.metric_name:
            raise ConfigurationError("a measured capability needs a metric name and value")
        self._capability_override = capability

    def voices(self) -> tuple[VoiceReference, ...]:
        """Real speaker ids enumerated from the loaded instance.

        Kokoro's ``voices.bin`` is an anonymous array of speaker embeddings: it
        gives no names and no per-speaker language tag. So the voice id is the
        stable ``kokoro-<sid>`` and the language is recorded *from the manifest*,
        never invented. Returns empty before ``warm`` - an engine that cannot yet
        render must not list voices it cannot speak.
        """
        if self._tts is None:
            return ()
        count = int(getattr(self._tts, "num_speakers", 0) or 0)
        language = self._language_of_loaded()
        return tuple(
            VoiceReference(
                voice_id=f"{SPEAKER_ID_PREFIX}{sid}",
                language=language,
                source=f"{self.model_id} speaker {sid}",
            )
            for sid in range(count)
        )

    def _language_of_loaded(self) -> LanguageTag | None:
        code = getattr(self, "_loaded_language", None) or self._renderable_languages()[0]
        return LanguageTag(code)

    def condition(self, reference: VoiceReference) -> VoiceReference:
        raise CapabilityUnavailable(
            "Kokoro is a fixed multi-speaker model, not a zero-shot voice cloner",
            engine=self.engine_id,
            model_id=self.model_id,
            voice_id=reference.voice_id,
            remedy="select one of the built-in speaker ids; a reference clip cannot condition this asset",
        )

    # -- rendering ------------------------------------------------------------

    def _speaker_id(self, request: SynthesisRequest) -> int:
        """Which speaker embedding to render with.

        An installed voice identity supplies the slot directly. Parsing the display
        id is only a compatibility path for callers that name a slot themselves,
        because an engine that reads names to find embeddings couples identity to
        engine and breaks the moment a host chooses its own labelling.
        """
        voice = request.voice
        if voice is None or not voice.voice_id:
            return 0
        if voice.speaker_slot is not None:
            token = str(voice.speaker_slot)
            return self._checked(int(token), voice.voice_id)
        token = voice.voice_id
        if token.startswith(SPEAKER_ID_PREFIX):
            token = token[len(SPEAKER_ID_PREFIX) :]
        try:
            sid = int(token)
        except ValueError as exc:
            raise CapabilityUnavailable(
                "voice identity supplied neither a speaker slot nor a parseable speaker id",
                voice_id=voice.voice_id,
                expected=f"{SPEAKER_ID_PREFIX}<int> or an explicit speaker_slot",
            ) from exc
        return self._checked(sid, voice.voice_id)

    def _checked(self, sid: int, voice_id: str) -> int:
        count = int(getattr(self._tts, "num_speakers", 0) or 0)
        if count and sid >= count:
            raise CapabilityUnavailable(
                "speaker id out of range for this asset", voice_id=voice.voice_id, num_speakers=count
            )
        return sid

    def _generate(self, text: str, sid: int, speed: float):
        return self._tts.generate(text=text, sid=sid, speed=speed)

    def _buffer_for(self, text: str, request: SynthesisRequest, timings: Timings) -> AudioBuffer | None:
        """Render one chunk into an int16 AudioBuffer, or None if it was silent."""
        guard(request.token)
        result = self._generate(text, self._speaker_id(request), _clamp_speed(request.prosody.rate))
        samples = list(getattr(result, "samples", []) or [])
        if not samples:
            return None
        buffer = from_float_samples(samples, self.output_format, started_at_ns=monotonic_ns())
        gain = _volume_gain(request.prosody.volume_db)
        if gain != 1.0:
            from ...audio.pcm import gain as _scale

            buffer = _scale(buffer, gain)
        timings.mark_first_result()
        return buffer

    def synthesize(self, request: SynthesisRequest) -> SynthesisOutcome:
        guard(request.token)
        if self._closed:
            raise ConfigurationError("engine is closed", engine_id=self.engine_id)
        if not request.text.strip():
            raise ConfigurationError("nothing to synthesise", language=request.language.bcp47)
        self.warm(self._lang_for(request.language))
        timings = Timings()
        chunks = self.chunker.chunk(request.text)
        buffers: list[AudioBuffer] = []
        for chunk in chunks:
            text = chunk.text.strip()
            if not text:
                continue
            buffer = self._buffer_for(text, request, timings)
            if buffer is not None and buffer.frame_count:
                buffers.append(buffer)
        if not buffers:
            raise CapabilityUnavailable(
                "synthesis produced no audio",
                engine=self.engine_id,
                model_id=self.model_id,
                language=request.language.bcp47,
            )
        audio = buffers[0]
        for extra in buffers[1:]:
            audio = audio.appended(extra)
        timings.audio_duration_s = audio.duration_s
        timings.finish()
        return SynthesisOutcome(
            audio=audio,
            language=request.language,
            voice_id=(request.voice.voice_id if request.voice else f"{SPEAKER_ID_PREFIX}0"),
            engine=self.engine_id,
            model_id=self.model_id,
            timings={
                **timings.as_dict(),
                "first_audio_ms": timings.first_result_ms,
                "total_ms": timings.total_ms,
                "chunks": len(buffers),
                "speed": _clamp_speed(request.prosody.rate),
            },
        )

    def stream(self, request: SynthesisRequest):
        """Yield one FIRST_AUDIO/AUDIO event per text chunk, then EOS.

        Chunk-level only: sherpa generates each chunk whole, so the first event
        arrives after the first chunk is rendered, not after the first frame. The
        token is guarded before every generation and every yield, so no audio
        escapes after cancellation.
        """
        guard(request.token)
        if self._closed:
            raise ConfigurationError("engine is closed", engine_id=self.engine_id)
        if not request.text.strip():
            raise ConfigurationError("nothing to synthesise", language=request.language.bcp47)
        self.warm(self._lang_for(request.language))
        timings = Timings()
        chunks = self.chunker.chunk(request.text)
        sequence = 0
        first = True
        for chunk in chunks:
            text = chunk.text.strip()
            if not text:
                continue
            guard(request.token)
            buffer = self._buffer_for(text, request, timings)
            if buffer is None or not buffer.frame_count:
                continue
            guard(request.token)
            sequence += 1
            kind = SynthesisKind.FIRST_AUDIO if first else SynthesisKind.AUDIO
            first = False
            yield SynthesisEvent(
                kind=kind,
                sequence=sequence,
                audio=buffer,
                language=request.language.bcp47,
                voice_id=(request.voice.voice_id if request.voice else f"{SPEAKER_ID_PREFIX}0"),
                text=text,
                elapsed_ms=timings.total_ms,
            )
        guard(request.token)
        sequence += 1
        yield SynthesisEvent(
            kind=SynthesisKind.EOS,
            sequence=sequence,
            reason="completed",
            language=request.language.bcp47,
            elapsed_ms=timings.total_ms,
        )

    def close(self) -> None:
        self._tts = None
        self._closed = True
