"""OPTIONAL adapter for a locally cached CT2 Whisper model.

Nothing here runs at import time or downloads anything: ``faster_whisper`` is
imported lazily *inside* methods, and the adapter refuses to execute at all
unless ``approved=True``. The 6 Whisper CT2 models already in ``~/.cache`` are
THIRD-PARTY ASSETS THAT HAVE NOT BEEN APPROVED for shipping, so this class carries
``ReviewStatus.UNREVIEWED`` provenance and gates every load behind an explicit
approval flag. A missing package raises ``ModelNotInstalled`` naming the package with
``import_state`` ``package-absent``; a package that is present but will not import keeps
the same category and says so, because telling a host to install what is already there
hides a corrupt environment. A missing model directory raises ``ModelNotInstalled``
naming the exact cache path; both are checked with a path *name*, never by reading the
cache contents.

The languages it can claim come only from the upstream model card, so they are
``DECLARED`` - unverified - and strict routing therefore will not select this
adapter until something real is measured.
"""

from __future__ import annotations

import io
import os
import wave

from ...core.audio import AudioFormat
from ...core.errors import CapabilityUnavailable, ModelNotApproved, ModelNotInstalled
from ...core.language import Capability, CapabilityStatus, LanguageTag, Task
from ...models.api import ModelManifest, Provenance, ResourceRequirement, ReviewStatus
from ...language.registry import LanguageRegistry
from ..api import TranscriptionRequest, TranscriptionResult
from .optional_backend import ABSENT, import_state

ENGINE_ID = "faster-whisper"

# Vendor claim only. faster-whisper inherits Whisper's ~99-language support; we
# record the subset our registry knows, and always as DECLARED (never measured).
_PROVENANCE = Provenance(
    source="Systran/faster-whisper on Hugging Face",
    upstream_project="Systran/faster-whisper (Whisper via CTranslate2)",
    licence_spdx="MIT (code) / BSD-3-Clause (Whisper weights, to be verified)",
    redistributable=None,
    notes="Third-party CT2 weights, NOT approved for shipping; licence verification pending.",
)


def default_cache_root() -> str:
    """Cache path prefix used to *name* a model, never to list its contents."""
    override = os.environ.get("HF_HOME")
    base = override if override else os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(base, "hub")


class FasterWhisperAdapter:
    """Whisper via CTranslate2, gated behind an explicit approval flag."""

    engine_id = ENGINE_ID
    supports_streaming = False

    def __init__(
        self,
        model_name: str = "base",
        *,
        approved: bool = False,
        device: str = "cpu",
        compute_type: str = "int8",
        registry: LanguageRegistry | None = None,
        cache_root: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.approved = bool(approved)
        self.device = device
        self.compute_type = compute_type
        self.format = AudioFormat(16000, 1)
        self.model_id = f"faster-whisper-{model_name}"
        self.review = ReviewStatus.UNREVIEWED
        self.provenance = _PROVENANCE
        self._registry = registry or LanguageRegistry()
        self._cache_root = cache_root or default_cache_root()
        self._model = None

    def model_path(self) -> str:
        """The directory a CT2 model would live in, constructed as a name only."""
        return os.path.join(self._cache_root, f"models--Systran--faster-whisper-{self.model_name}")

    def manifest(self) -> ModelManifest:
        return ModelManifest(
            model_id=self.model_id,
            engine=ENGINE_ID,
            task=Task.STT,
            display_name=f"Faster-Whisper {self.model_name}",
            languages=tuple(self._registry.all_codes()),
            sample_rate=16000,
            quantization=f"ct2-{self.compute_type}",
            provenance=self.provenance,
            review=self.review,
            requirements=ResourceRequirement(measured=False),
            notes="Unreviewed third-party asset; requires explicit approval to run.",
        )

    @staticmethod
    def backend_available() -> bool:
        """True if ``faster_whisper`` is importable. No network, no download."""
        try:
            import faster_whisper  # noqa: F401
        except Exception:
            return False
        return True

    def _require_approval(self) -> None:
        if not self.approved:
            raise ModelNotApproved(
                "faster-whisper weights are unreviewed third-party assets and are not approved",
                engine=ENGINE_ID,
                model_id=self.model_id,
                review=self.review.value,
                hint="pass approved=True only after models/registry has approved the asset",
            )

    def _import_backend(self):
        try:
            import faster_whisper
        except Exception as exc:
            state = import_state("faster_whisper", exc)
            raise ModelNotInstalled(
                (
                    "faster-whisper backend package is not installed"
                    if state == ABSENT
                    else "faster-whisper backend package is present but failed to import"
                ),
                package="faster-whisper",
                engine=ENGINE_ID,
                import_state=state,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        return faster_whisper

    def _require_model_dir(self) -> str:
        path = self.model_path()
        # Existence of the directory is checked by stat, never by reading contents.
        if not os.path.isdir(path):
            raise ModelNotInstalled(
                "no cached faster-whisper model asset is present",
                model_id=self.model_id,
                expected_path=path,
                detail="the CT2 model dir is absent; nothing will be downloaded",
            )
        return path

    def capabilities(self) -> tuple[Capability, ...]:
        # Vendor-declared only; status DECLARED so strict routing cannot pick it.
        return tuple(
            Capability(
                task=Task.STT,
                status=CapabilityStatus.DECLARED,
                language=code,
                engine=ENGINE_ID,
                model_id=self.model_id,
                limitations=(
                    "vendor model-card claim, not measured by us",
                    "asset UNREVIEWED and unapproved by default",
                ),
                evidence=_PROVENANCE.source,
            )
            for code in self._registry.all_codes()
        )

    def warm(self) -> None:
        self._require_approval()
        module = self._import_backend()
        self._require_model_dir()
        if self._model is None:
            self._model = module.WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self._cache_root,
                local_files_only=True,
            )

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        # Approval is enforced first: we refuse to touch the backend without it.
        self._require_approval()
        module = self._import_backend()
        self._require_model_dir()
        self.warm()
        audio = request.audio
        if audio.format.sample_rate != 16000 or audio.format.channels != 1:
            raise CapabilityUnavailable(
                "faster-whisper adapter expects 16 kHz mono input; resample before requesting",
                format=str(audio.format),
            )
        segments, info = self._model.transcribe(self._wav_stream(audio), language=(request.language.code if request.language else None))
        text = "".join(segment.text for segment in segments).strip()
        detected = LanguageTag(info.language) if getattr(info, "language", None) else request.language
        return TranscriptionResult(
            text=text,
            language=detected,
            confidence=getattr(info, "language_probability", None),
            engine=ENGINE_ID,
            model_id=self.model_id,
        )

    def close(self) -> None:
        self._model = None

    @staticmethod
    def _wav_stream(buffer) -> io.BytesIO:
        container = io.BytesIO()
        with wave.open(container, "wb") as handle:
            handle.setnchannels(buffer.format.channels)
            handle.setsampwidth(2)
            handle.setframerate(buffer.format.sample_rate)
            handle.writeframes(buffer.to_bytes())
        container.seek(0)
        return container
