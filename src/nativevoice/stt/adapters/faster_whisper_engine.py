"""Real recognition engine: Whisper via CTranslate2, loaded from a manifest.

The lightweight ``FasterWhisperAdapter`` in this package refuses to run unless a
caller hands it ``approved=True``; it exists so the plumbing can be tested without
a model. This module is the genuine path: a ``TranscriptionEngine`` that resolves
its weights through a reviewed ``ModelManifest``, runs them past the licence gate
and integrity verifier before importing ``faster_whisper``, and propagates
``ModelNotApproved`` / ``IntegrityError`` / ``ModelNotInstalled`` unchanged.

Where the weights live is *not* hardcoded. The asset directory is derived from the
configured Hugging Face cache root (``default_cache_root``) by matching a
``models--*--<model_id>`` repository, and the exact CT2 snapshot sub-directory is
read from the manifest's own recorded relative paths. No absolute path appears in
library code.

CPU by default. onnxruntime on this box has no CUDA provider, and 4 GB of VRAM is
not a promise, so ``device`` defaults to ``"cpu"`` with an explicit ``compute_type``
default; ``"float16"`` on GPU is offered but never assumed. Whisper's ~99-language
list is a vendor claim, so every capability starts ``DECLARED`` - a language only
becomes ``MEASURED`` when an evaluation run actually produced a number for it.

``stream`` here is buffered partials, not true microphone streaming: the caller
supplies whole chunks, they are concatenated, and the segment generator yields a
partial per decoded segment before one final. That is documented honestly rather
than passed off as frame-level streaming.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..api import TranscriptionRequest, TranscriptionResult
from ..text_context import build_prompt
from ...core.audio import AudioBuffer, AudioFormat, Recognition
from ...core.cancel import guard
from ...core.errors import (
    CapabilityUnavailable,
    ConfigurationError,
    IntegrityError,
    ModelNotInstalled,
)
from ...core.events import TranscriptionEvent
from ...core.language import Capability, CapabilityStatus, LanguageTag, Task
from ...core.time import Timings, new_id
from ...models.api import ModelManifest
from ...models.approval import ensure_approved
from ...models.integrity import inspect as inspect_integrity

ENGINE_ID = "faster-whisper"


def default_cache_root() -> str:
    """HF hub cache prefix, from the environment, never a literal path."""
    override = os.environ.get("HF_HOME")
    base = override if override else os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(base, "hub")


def resolve_asset_dir(manifest: ModelManifest, cache_root: str | None = None) -> Path:
    """The repository directory a CT2 model was cached under, matched by model id.

    A provisioned manifest records the absolute directory it was verified against,
    and that wins; otherwise the cache is scanned for ``models--*--<id>`` so the
    organisation prefix is never hardcoded. Raises ``ModelNotInstalled`` naming the
    cache root when no such repo is present.
    """
    if manifest.asset_root:
        declared = Path(manifest.asset_root)
        if declared.is_dir():
            return declared
    root = Path(cache_root or default_cache_root())
    suffix = f"--{manifest.model_id}"
    matches = sorted(p for p in root.glob(f"models--*{suffix}") if p.is_dir())
    if not matches:
        raise ModelNotInstalled(
            "no cached CT2 model repository is present for this manifest",
            model_id=manifest.model_id,
            expected_path=str(root),
            detail="looked for a models--*--<model_id> directory; nothing will be downloaded",
        )
    return matches[0]


def _snapshot_dir(repo_dir: Path, manifest: ModelManifest) -> Path:
    """The exact CT2 snapshot directory, taken from the manifest's own paths.

    The provisioner recorded each asset relative to the repository directory as
    ``snapshots/<hash>/file``. Recovering the hash from that (rather than trusting
    ``snapshots`` mtime) is what lets integrity root and load path stay identical.
    """
    for asset in manifest.assets:
        parts = Path(asset.relative_path).parts
        if len(parts) >= 2 and parts[0] == "snapshots":
            candidate = repo_dir / "snapshots" / parts[1]
            if candidate.is_dir():
                return candidate
    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        subs = sorted((p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        if subs:
            return subs[-1]
    raise ModelNotInstalled(
        "no CT2 snapshot directory could be located for this manifest",
        model_id=manifest.model_id,
        expected_path=str(snapshots),
    )


def _to_float32(buffer: AudioBuffer):
    """int16 interleaved mono -> float32 waveform in [-1, 1]."""
    import numpy as np

    mono = buffer.to_mono() if buffer.format.channels > 1 else buffer
    return np.asarray(mono.samples, dtype=np.float32) / 32768.0


class FasterWhisperEngine:
    """A manifest-gated, integrity-verified Whisper recognition engine."""

    engine_id = ENGINE_ID
    format = AudioFormat(16000, 1)
    # Whisper is VAD/chunk based, not microphone-frame streaming.
    supports_streaming = False

    def __init__(
        self,
        manifest: ModelManifest,
        *,
        asset_dir: Path | str | None = None,
        device: str = "cpu",
        compute_type: str = "default",
        cpu_threads: int | None = None,
        beam_size: int = 5,
        vad_filter: bool = False,
        language_codes: tuple[str, ...] | None = None,
    ) -> None:
        if manifest.task is not Task.STT:
            raise ConfigurationError(
                "manifest is not an STT asset", model_id=manifest.model_id, task=manifest.task.value
            )
        self.manifest = manifest
        self.model_id = manifest.model_id
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.beam_size = max(1, int(beam_size))
        self.vad_filter = bool(vad_filter)
        self._language_codes = tuple(language_codes) if language_codes is not None else tuple(manifest.languages)
        self._asset_dir = Path(asset_dir) if asset_dir is not None else None
        self._model = None
        self._capability_override: Capability | None = None
        self._closed = False

    # -- gates ----------------------------------------------------------------

    def _require_approved(self) -> None:
        ensure_approved(self.manifest)

    def _resolved_asset_dir(self) -> Path:
        if self._asset_dir is not None:
            return self._asset_dir
        self._asset_dir = resolve_asset_dir(self.manifest)
        return self._asset_dir

    def _require_verified(self) -> Path:
        repo_dir = self._resolved_asset_dir()
        report = inspect_integrity(self.manifest, repo_dir)
        if report.missing:
            raise ModelNotInstalled(
                "CT2 model assets are missing on disk",
                model_id=self.model_id,
                missing=report.missing,
                reasons=report.reasons,
            )
        if not report.ok:
            raise IntegrityError(
                "CT2 model failed integrity verification",
                model_id=self.model_id,
                reasons=report.reasons,
            )
        return _snapshot_dir(repo_dir, self.manifest)

    def _import_backend(self):
        try:
            import faster_whisper
        except Exception as exc:  # a missing package is an installation gap, named exactly
            raise ModelNotInstalled(
                "faster-whisper backend package is not installed",
                package="faster-whisper",
                engine=self.engine_id,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        return faster_whisper

    # -- lifecycle ------------------------------------------------------------

    def warm(self) -> None:
        if self._closed:
            raise ConfigurationError("engine is closed", engine_id=self.engine_id)
        if self._model is not None:
            return
        self._require_approved()
        snapshot = self._require_verified()
        module = self._import_backend()
        kwargs = dict(device=self.device, compute_type=self.compute_type)
        if self.cpu_threads:
            kwargs["cpu_threads"] = int(self.cpu_threads)
        self._model = module.WhisperModel(str(snapshot), local_files_only=True, **kwargs)

    def close(self) -> None:
        self._model = None
        self._closed = True

    # -- declarations ---------------------------------------------------------

    def capabilities(self) -> tuple[Capability, ...]:
        measured = {}
        if self._capability_override is not None:
            measured[self._capability_override.language] = self._capability_override
        codes = self._language_codes or self._vendor_codes()
        out: list[Capability] = []
        for code in codes:
            override = measured.get(code)
            if override is not None:
                out.append(override)
                continue
            out.append(
                Capability(
                    task=Task.STT,
                    status=CapabilityStatus.DECLARED,
                    language=code,
                    engine=self.engine_id,
                    model_id=self.model_id,
                    limitations=("vendor model-card language claim; not measured on this machine",),
                    evidence=self.manifest.provenance.source,
                )
            )
        return tuple(out)

    @staticmethod
    def _vendor_codes() -> tuple[str, ...]:
        # The Whisper language set is a vendor claim; the codes we surface are the
        # ones the manifest recorded, and nothing is invented when the manifest is
        # silent. An empty tuple means "we assert no languages we have not seen".
        return ()

    def record_measured(self, capability: Capability) -> None:
        if capability.status is not CapabilityStatus.MEASURED:
            raise ConfigurationError(
                "only a measured capability can be recorded", status=capability.status.value
            )
        if capability.metric_value is None or not capability.metric_name:
            raise ConfigurationError("a measured capability needs a metric name and value")
        self._capability_override = capability

    # -- recognition ----------------------------------------------------------

    def _run(self, request: TranscriptionRequest):
        if request.audio.format.sample_rate != self.format.sample_rate:
            raise CapabilityUnavailable(
                "engine expects 16 kHz audio; resample before requesting",
                format=str(request.audio.format),
            )
        prompt = build_prompt(request.hint_terms, request.context)
        force = request.language.code if (request.language and not request.allow_auto_detection) else None
        kwargs = dict(
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        if force:
            kwargs["language"] = force
        if prompt:
            kwargs["initial_prompt"] = prompt
        waveform = _to_float32(request.audio)
        timings = Timings(audio_duration_s=request.audio.duration_s)
        segment_iter, info = self._model.transcribe(waveform, **kwargs)
        segments: list[Recognition] = []
        texts: list[str] = []
        no_speech: list[float] = []
        for segment in segment_iter:
            guard(request.token)
            texts.append(segment.text)
            segments.append(
                Recognition(
                    text=segment.text.strip(),
                    language=getattr(info, "language", "") or "",
                    confidence=None,
                    first_sample=int(round(segment.start * self.format.sample_rate)),
                    last_sample=int(round(segment.end * self.format.sample_rate)),
                    is_final=True,
                    metadata={
                        "start": segment.start,
                        "end": segment.end,
                        "no_speech_prob": getattr(segment, "no_speech_prob", None),
                    },
                )
            )
            value = getattr(segment, "no_speech_prob", None)
            if value is not None:
                no_speech.append(float(value))
        timings.finish()
        text = " ".join(t.strip() for t in texts).strip()
        detected = LanguageTag(getattr(info, "language")) if getattr(info, "language", None) else request.language
        confidence = self._confidence(info, no_speech)
        timings_dict = {
            **timings.as_dict(),
            "language_probability": getattr(info, "language_probability", None),
            "segments": len(segments),
        }
        return text, detected, confidence, tuple(segments), timings_dict

    @staticmethod
    def _confidence(info, no_speech: list[float]) -> float | None:
        probs = [1.0 - value for value in no_speech]
        if probs:
            return round(sum(probs) / len(probs), 4)
        return getattr(info, "language_probability", None)

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        guard(request.token)
        if self._closed:
            raise ConfigurationError("engine is closed", engine_id=self.engine_id)
        if request.audio.frame_count == 0:
            raise CapabilityUnavailable("no audio to transcribe", engine=self.engine_id)
        self.warm()
        text, language, confidence, segments, timings = self._run(request)
        return TranscriptionResult(
            text=text,
            language=language,
            confidence=confidence,
            segments=segments,
            engine=self.engine_id,
            model_id=self.model_id,
            timings=timings,
        )

    def stream(
        self,
        chunks,
        language: LanguageTag | None = None,
        token=None,
    ):
        """Buffered partials then one final; not microphone-frame streaming.

        ``chunks`` are concatenated into one utterance, the CT2 segment generator
        is drained yielding a PARTIAL per decoded segment, then a single FINAL. The
        whole utterance must be present before decoding starts, which is why this is
        documented as buffered partials.
        """
        if not self.supports_streaming:
            raise CapabilityUnavailable(
                "faster-whisper exposes buffered partials only, not true streaming",
                engine=self.engine_id,
                remedy="call transcribe(), or read stream() as buffered partials",
            )
        merged: AudioBuffer | None = None
        for chunk in chunks:
            guard(token)
            merged = chunk if merged is None else merged.appended(chunk)
        if merged is None or merged.frame_count == 0:
            raise CapabilityUnavailable("stream received no audio", engine=self.engine_id)
        self.warm()
        request = TranscriptionRequest(audio=merged, language=language, token=token)
        utterance_id = new_id("utt")
        text, detected, confidence, segments, _timings = self._run(request)
        sequence = 0
        accumulated: list[str] = []
        for recognition in segments:
            guard(token)
            accumulated.append(recognition.text)
            sequence += 1
            yield TranscriptionEvent.partial(
                " ".join(accumulated).strip(),
                utterance_id,
                sequence=sequence,
                language=detected.bcp47 if detected else "",
            )
        guard(token)
        sequence += 1
        yield TranscriptionEvent.final(
            text,
            utterance_id,
            sequence=sequence,
            language=detected.bcp47 if detected else "",
            confidence=confidence,
        )
