"""Voice profile storage, consent, quality gating, exchange and conditioning.

Storage and governance only. Conditioning - turning a profile into something a
synthesis model can speak with - requires an approved model that does not exist
here, so `conditioning.py` refuses by name rather than faking a speaker embedding.
"""

from __future__ import annotations

from .conditioning import (
    MISSING_APPROVED_MODEL,
    MISSING_CONDITIONING_ASSET,
    MISSING_RUNTIME,
    NoApprovedConditioner,
    VoiceConditioner,
    cross_language_evaluation_plan,
)
from .consent import (
    CONSENT_SCOPE_SYNTHESIS,
    CONSENT_SCOPE_TRAINING,
    DISCLAIMER,
    ConsentRecord,
    digest_statement,
)
from .exchange import FORMAT_VERSION, export_profile, import_profile
from .quality import (
    PROVISIONAL_MINIMUMS,
    QualityCheck,
    QualityMinimums,
    QualityReport,
    assess,
    energy_contrast,
    estimate_snr_db,
)
from .store import (
    RemovedArtifacts,
    RetentionPolicy,
    VoiceProfile,
    VoiceProfileStore,
    audio_digest,
)

__all__ = [
    "CONSENT_SCOPE_SYNTHESIS",
    "CONSENT_SCOPE_TRAINING",
    "DISCLAIMER",
    "FORMAT_VERSION",
    "MISSING_APPROVED_MODEL",
    "MISSING_CONDITIONING_ASSET",
    "MISSING_RUNTIME",
    "ConsentRecord",
    "NoApprovedConditioner",
    "PROVISIONAL_MINIMUMS",
    "QualityCheck",
    "QualityMinimums",
    "QualityReport",
    "RemovedArtifacts",
    "RetentionPolicy",
    "VoiceConditioner",
    "VoiceProfile",
    "VoiceProfileStore",
    "assess",
    "audio_digest",
    "cross_language_evaluation_plan",
    "digest_statement",
    "energy_contrast",
    "estimate_snr_db",
    "export_profile",
    "import_profile",
]
