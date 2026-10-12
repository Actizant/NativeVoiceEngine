"""The reference synthesis architecture, expressed as a refusal.

This adapter documents *what would have to be true* for a Pocket-TTS-class model
to run here, and enforces that nothing is claimed until it is. It imports nothing
from any upstream synthesis package: there is no torch runtime on this machine, no
approved flow-matching model asset, and no approved voice-conditioning asset, so a
working adapter would be a lie held together by a tone generator.

The architecture it describes is worth stating precisely, because it shapes the
interfaces in this repository:

* roughly 100M parameters, a flow-matching language model over continuous Mimi VAE
  latents at 12.5 Hz frames, 24 kHz output;
* ~200 ms to first audio chunk and 3-6x real time on CPU once attention and FFN
  layers are int8-dynamic-quantised - which is why `Timings.first_result_ms` and
  `real_time_factor` are first-class fields rather than something a caller measures;
* a "voice" is an exported KV-cache state stored as safetensors, not a speaker
  embedding. That is why `VoiceReference.state_path` exists beside `audio`, and why
  consent has to travel with the reference: the exported state *is* the speaker;
* text handling is grapheme-based through a ~4000-entry SentencePiece tokenizer with
  no phonemizer, which is why `tts/normalise.py` insists on fully-written-form text;
* long text is split on sentence boundaries with a decimal-aware splitter and
  regrouped under 50 tokens per chunk, each chunk generated independently - the
  documented source of cross-chunk prosody drift that `tts/chunking.py` budgets for
  and `tts/pipeline.py` crossfades;
* seven languages ship, Bengali is absent and Hindi is community-only, so this
  project's two target languages have *no* model today.

Two refusals, in order, and they mean different things: `ModelNotApproved` says an
asset exists but no human has signed off its provenance and licence; the second
`CapabilityUnavailable` says that even with approval there is still no runtime, no
weights and no conditioning asset installed. Fixing one does not fix the other.
"""

from __future__ import annotations

from collections.abc import Iterator

from ...core.audio import AudioFormat
from ...core.errors import (
    CapabilityUnavailable,
    ModelNotApproved,
)
from ...core.language import Capability, CapabilityStatus, LanguageTag, Task
from ...models.api import ModelManifest, ReviewStatus
from ..api import SynthesisEvent, SynthesisOutcome, SynthesisRequest, VoiceReference

ENGINE_ID = "pocket-reference"
MISSING_COMPONENTS = (
    "torch runtime (not installed on this machine)",
    "an approved flow-matching synthesis model asset with reviewed provenance",
    "an approved voice-conditioning asset (exported KV-cache state as safetensors)",
)
REFERENCE_LANGUAGES = ("en", "fr", "de", "es", "it", "pt", "zh")
TARGET_LANGUAGES = ("bn", "hi")
_NO_MODEL = (
    "the reference model covers seven languages and Bengali is not among them; "
    "Hindi exists only as an unreviewed community asset"
)


def development_plan() -> dict:
    """Itemised requirements to make this adapter real.

    Every number here is an *external requirement taken from the published
    reference architecture*, not a measurement we made by training. That
    distinction matters: nothing below has been verified in this repository, and a
    plan that reads like a measurement is how a project promises a voice it cannot
    build.
    """
    return {
        "status": "external_requirements_unverified_by_training",
        "architecture": {
            "parameter_scale": "approximately 100M parameters, flow-matching language model",
            "acoustic_representation": "continuous Mimi VAE latents",
            "frame_rate_hz": 12.5,
            "output_sample_rate_hz": 24000,
            "quantisation": "int8 dynamic quantisation of attention and FFN layers only",
            "cpu_targets": {
                "first_audio_ms": 200,
                "real_time_factor_range": [3.0, 6.0],
            },
            "text_front_end": {
                "tokenizer": "approximately 4000-entry SentencePiece over graphemes",
                "phonemizer": "none - grapheme-only, so normalisation must produce spoken form",
                "chunk_budget_tokens": 50,
                "known_defect": "independent per-chunk generation causes cross-chunk prosody drift",
            },
        },
        "data_requirements": {
            "per_language_read_speech_hours": {
                "existing_reference_languages": "as published upstream; not re-measured here",
                "bn": "unknown - no public corpus of sufficient quality has been reviewed",
                "hi": "unknown - community assets exist without licence clarity",
                "minimum_target_hours": 50,
                "minimum_speakers": 20,
                "provenance_required": True,
            },
            "voice_conditioning": {
                "reference_audio_seconds": 10,
                "requires_consent_record": True,
                "exported_state_format": "safetensors KV-cache state per voice",
            },
        },
        "compute_requirements": {
            "training": "not attempted in this project; multi-GPU days per language, unquantified here",
            "inference": {
                "ram_mb_minimum": 1024,
                "threads": 4,
                "needs_gpu": False,
                "note": "int8 CPU path is the shipping target; measured per machine by hardware profiling",
            },
        },
        "evaluation_protocol": {
            "intelligibility": "round-trip through an approved STT engine; report WER per language",
            "naturalness": "blind MOS with at least 10 native listeners per language",
            "speaker_similarity": "embedding distance between reference audio and output, per condition",
            "prosody_continuity": "pitch and tempo drift measured across chunk boundaries",
            "latency": "first-audio ms and real-time factor measured on the shipping machine",
            "regression_corpus": "fixed sentence set per language, including numerals and code-switch",
            "gate": "a language ships only at CapabilityStatus.MEASURED with a cited metric",
        },
        "unknowns": [
            "Bengali and Hindi training data availability and licensing",
            "whether cross-language identity survives a non-native conditioning asset",
            "real CPU latency on the target hardware with int8 quantisation",
        ],
    }


class PocketReferenceAdapter:
    """Architecture documentation that refuses to synthesise.

    Registered so `describe()` can show the user what the project is aiming at and
    exactly why it is not usable; it declares NO_MODEL for every language so no
    router can ever select it, and every entry point raises with the missing
    component named.
    """

    engine_id = ENGINE_ID
    model_id = ""
    output_format = AudioFormat(24000, 1)
    supports_streaming = False
    controllable = ("rate", "pitch_semitones", "volume_db", "pause_after_s")

    def __init__(self, manifest: ModelManifest | None = None) -> None:
        self.manifest = manifest
        if manifest is not None:
            self.model_id = manifest.model_id

    def capabilities(self) -> tuple[Capability, ...]:
        codes = tuple(dict.fromkeys(REFERENCE_LANGUAGES + TARGET_LANGUAGES))
        return tuple(
            Capability(
                task=Task.TTS,
                status=CapabilityStatus.NO_MODEL,
                language=code,
                engine=self.engine_id,
                model_id=self.model_id,
                limitations=self._limitation_for(code),
                evidence="adapter is documentation; no approved asset is installed",
            )
            for code in codes
        )

    def voices(self) -> tuple[VoiceReference, ...]:
        return ()

    def condition(self, reference: VoiceReference) -> VoiceReference:
        self._gate()
        raise self._refusal("voice conditioning", reference.language)

    def warm(self) -> None:
        self._gate()

    def close(self) -> None:
        return None

    def synthesize(self, request: SynthesisRequest) -> SynthesisOutcome:
        self._gate()
        raise self._refusal("synthesis", request.language)

    def stream(self, request: SynthesisRequest) -> Iterator[SynthesisEvent]:
        self._gate()
        raise self._refusal("streaming synthesis", request.language)

    # -- refusal machinery ---------------------------------------------------

    def _gate(self) -> None:
        """Approval first: an unreviewed asset must not be reported as merely absent."""
        if self.manifest is None:
            raise ModelNotApproved(
                "the reference synthesis architecture has no model manifest",
                engine=self.engine_id,
                remedy="register a manifest and complete a provenance and licence review",
            )
        if self.manifest.review is not ReviewStatus.APPROVED:
            missing = self.manifest.provenance.incomplete()
            raise ModelNotApproved(
                "the reference synthesis model is not approved for shipping",
                engine=self.engine_id,
                model_id=self.manifest.model_id,
                review=self.manifest.review.value,
                incomplete_provenance=missing,
                remedy="complete the provenance and licence review in the model registry",
            )

    def _refusal(self, operation: str, language: LanguageTag | None) -> CapabilityUnavailable:
        return CapabilityUnavailable(
            f"{operation} is unavailable in the reference synthesis adapter",
            engine=self.engine_id,
            language=language.code if language is not None else "",
            missing_components=MISSING_COMPONENTS,
            detail=(
                "approval of the manifest alone does not make this work: the torch runtime, "
                "an approved flow-matching synthesis model and an approved voice-conditioning "
                "asset must all be installed and measured first"
            ),
        )

    @staticmethod
    def _limitation_for(code: str) -> tuple[str, ...]:
        if code in TARGET_LANGUAGES:
            return (_NO_MODEL,)
        return (
            "declared only to document the reference architecture; no installed asset backs it",
        )
