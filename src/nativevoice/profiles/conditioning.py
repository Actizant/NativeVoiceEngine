"""Voice conditioning: the interface, and the fact that we cannot perform it.

Conditioning is the step that turns reference audio into whatever the synthesis
model consumes. In the reference architecture that artifact is an exported
KV-cache state saved as safetensors - literally a re-usable speaker, which is why
the pipeline refuses to render against a voice without consent attached.

Nothing here can produce that state. There is no approved synthesis model, no
approved conditioning asset, and no torch runtime on this machine, so
`NoApprovedConditioner` raises and reports `NO_MODEL` instead of returning a
speaker embedding invented from MFCC averages. A fake conditioner would be worse
than none: it would let a profile reach the engine and make "it works" untestable.

The evaluation plan below states what would have to be *measured* before this
project could claim cross-language identity preservation. It is a plan, not a
result: we do not claim a voice conditioned on Bengali still sounds like the same
person in English, because we have never produced either utterance.
"""

from __future__ import annotations

from typing import Protocol

from ..core.errors import CapabilityUnavailable
from ..core.language import Capability, CapabilityStatus, LanguageTag, Task
from ..tts.api import VoiceReference

MISSING_APPROVED_MODEL = (
    "no approved synthesis model exists, so there is nothing to condition on: "
    "a voice state requires the flow-matching model asset that has not been reviewed or installed"
)
MISSING_CONDITIONING_ASSET = (
    "no approved voice-conditioning asset format exists (the reference architecture stores an "
    "exported KV-cache state as safetensors; nothing here can read or write one)"
)
MISSING_RUNTIME = "no torch runtime installed on this machine"

_UNVERIFIED = "authored as a plan; not measured, because no synthesis has run in this repository"


class VoiceConditioner(Protocol):
    """Turns consented reference audio into a reusable voice for one engine."""

    conditioner_id: str

    def capability(self, language: LanguageTag) -> Capability: ...

    def condition(self, reference: VoiceReference) -> VoiceReference: ...

    def verify(self, reference: VoiceReference) -> bool:
        """Whether a stored voice state still matches its consented audio."""
        ...


class NoApprovedConditioner:
    """The only conditioner that exists today: it refuses, naming what is missing."""

    conditioner_id = "none"

    def capability(self, language: LanguageTag) -> Capability:
        return Capability(
            task=Task.VOICE_CLONE,
            status=CapabilityStatus.NO_MODEL,
            language=language.root().code,
            engine=self.conditioner_id,
            limitations=(MISSING_APPROVED_MODEL, MISSING_CONDITIONING_ASSET, MISSING_RUNTIME),
            evidence=_UNVERIFIED,
        )

    def condition(self, reference: VoiceReference) -> VoiceReference:
        if not reference.has_consent():
            raise CapabilityUnavailable(
                "voice conditioning refused: the reference carries no consent record",
                voice_id=reference.voice_id,
                reason="consent must be recorded before a speaker state can exist",
            )
        raise CapabilityUnavailable(
            "voice conditioning is unavailable",
            voice_id=reference.voice_id,
            reason=MISSING_APPROVED_MODEL,
            also_missing=(MISSING_CONDITIONING_ASSET, MISSING_RUNTIME),
        )

    def verify(self, reference: VoiceReference) -> bool:
        raise CapabilityUnavailable(
            "voice-state verification is unavailable",
            voice_id=reference.voice_id,
            reason=MISSING_CONDITIONING_ASSET,
        )


def cross_language_evaluation_plan() -> dict:
    """What would have to be measured to *claim* cross-language identity preservation.

    Ordered so a reader can see the dependency: without a working single-language
    voice, nothing downstream is meaningful. Every item is labelled unverified;
    claiming "the same speaker across languages" requires listener evidence, not
    an embedding distance we chose ourselves.
    """
    return {
        "status": _UNVERIFIED,
        "prerequisites": [
            "an approved synthesis model that renders one language with measured intelligibility",
            "an approved conditioning path that produces a reusable speaker state",
            "consented multi-language recordings of the same speakers",
        ],
        "measures": [
            {
                "name": "speaker_similarity_within_language",
                "method": "embedding distance between reference audio and synthesised output, same language",
                "purpose": "baseline; an across-language claim without this is meaningless",
            },
            {
                "name": "speaker_similarity_across_language",
                "method": "same embedding distance where the reference is language A and the output language B",
                "report": "mean and 10th percentile per language pair, never a single average",
            },
            {
                "name": "listener_identification",
                "method": "AX identification: same-speaker-different-language vs different-speaker pairs",
                "report": "d-prime or percent correct against a same-language control",
                "minimum_listeners": 10,
                "note": "native listeners of both languages; machine metrics cannot settle identity",
            },
            {
                "name": "intelligibility_per_language",
                "method": "round-trip through an approved STT engine, report WER/CER with the corpus named",
            },
            {
                "name": "naturalness_per_language",
                "method": "blind MOS with native listeners, reported per language rather than overall",
            },
            {
                "name": "prosody_drift_across_chunks",
                "method": "pitch and tempo discontinuity measured at chunk boundaries",
                "note": "the reference architecture generates chunks independently, so this is a known risk",
            },
        ],
        "language_pairs": [("bn", "en"), ("en", "bn"), ("hi", "en"), ("en", "hi"), ("bn", "hi")],
        "corpus_requirements": {
            "speakers": 20,
            "minutes_per_speaker_per_language": 10,
            "licensing": "consent recorded per speaker per language and per permitted use",
        },
        "refusals": [
            "no cross-language identity claim may be made from one language's audio",
            "no claim may cite a metric that has not been computed on approved assets",
            "a community model without provenance review cannot supply the baseline",
        ],
    }
