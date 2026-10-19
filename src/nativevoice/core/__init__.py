"""Shared, dependency-free contracts used by every voice subsystem here.

Nothing in `core` imports from a sibling subsystem; the dependency direction is
always `subsystem -> core`.
"""

from .audio import (
    AudioBuffer,
    AudioFormat,
    AudioLevel,
    CANONICAL_16K_MONO,
    CANONICAL_24K_MONO,
    Recognition,
    SynthesisResult,
)
from .cancel import CancellationToken, guard, guarded
from .errors import (
    CapabilityUnavailable,
    CancellationRequested,
    ConfigurationError,
    DeviceDisconnected,
    DeviceError,
    EngineCrashed,
    EngineTimeout,
    EngineUnavailable,
    IntegrityError,
    LanguageUnsupported,
    ModelNotApproved,
    ModelNotInstalled,
    ProfileInvalid,
    ResourceLimitExceeded,
    StorageError,
    VoiceError,
)
from .events import (
    DiagnosticEvent,
    SynthesisEvent,
    SynthesisKind,
    TranscriptionEvent,
    TranscriptionKind,
    TurnEvent,
    TurnKind,
)
from .language import (
    Capability,
    CapabilityStatus,
    LanguageDecision,
    LanguageProfile,
    LanguageTag,
    Task,
)
from .time import Timings, monotonic_ns, new_id

__all__ = [
    "AudioBuffer",
    "AudioFormat",
    "AudioLevel",
    "CANONICAL_16K_MONO",
    "CANONICAL_24K_MONO",
    "Capability",
    "CapabilityStatus",
    "CancellationRequested",
    "CancellationToken",
    "ConfigurationError",
    "DeviceDisconnected",
    "DeviceError",
    "DiagnosticEvent",
    "EngineCrashed",
    "EngineTimeout",
    "EngineUnavailable",
    "IntegrityError",
    "LanguageDecision",
    "LanguageProfile",
    "LanguageTag",
    "LanguageUnsupported",
    "ModelNotApproved",
    "ModelNotInstalled",
    "ProfileInvalid",
    "Recognition",
    "ResourceLimitExceeded",
    "StorageError",
    "SynthesisEvent",
    "SynthesisKind",
    "SynthesisResult",
    "Task",
    "Timings",
    "TranscriptionEvent",
    "TranscriptionKind",
    "TurnEvent",
    "TurnKind",
    "VoiceError",
    "guard",
    "guarded",
    "monotonic_ns",
    "new_id",
]
