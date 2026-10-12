"""Voice identity as an installable, versioned package - separate from the engine.

Why this exists as its own layer: a synthesis engine enumerates *speaker slots*,
and a slot is not a voice. Kokoro's `voices.bin` is an anonymous array of 103
embeddings with no names, no per-speaker language tag and no accent information,
so treating "sid 47" as an identity would mean inventing metadata and then
trusting it. An identity package is the thing a host selects: a stable id, the
engine and slot it maps to, what it has actually been measured to speak, and the
permissions under which it may be used.

The consequence for the platform is the point. ULTRON can select one identity,
FieldMind another, and a third-party host a third - all against the same
compiled engine, with no voice name, no application identity and no host default
baked into the synthesis path. Per-language capability lives here too, because a
voice that speaks English well and Mandarin badly is not one "multilingual" flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..core.errors import ConfigurationError, LanguageUnsupported
from ..core.language import CapabilityStatus, LanguageTag, Task
from ..tts.api import VoiceReference

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LanguageSupport:
    """What this voice has actually been shown to do in one language."""

    language: str
    status: CapabilityStatus
    accent: str = ""
    style: str = ""
    metric_name: str = ""
    metric_value: float | None = None
    measured_on: str = ""
    evidence: str = ""
    limitations: tuple[str, ...] = ()

    def usable(self) -> bool:
        return self.status in (CapabilityStatus.MEASURED, CapabilityStatus.PARTIAL)

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "status": self.status.value,
            "accent": self.accent,
            "style": self.style,
            "metric": {
                "name": self.metric_name,
                "value": self.metric_value,
                "measured_on": self.measured_on,
            },
            "evidence": self.evidence,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class VoiceProvenance:
    """Where a voice came from and what may be done with it.

    Training method and permission are recorded separately because a permitted
    training run does not imply a redistributable artefact, and an Apache-2.0
    model does not imply the voices inside it carry the same terms.
    """

    creator: str = ""
    training_method: str = ""
    licence_spdx: str = ""
    permissions: str = ""
    base_asset: str = ""
    redistributable: bool | None = None
    consent_basis: str = ""
    notes: str = ""

    def incomplete(self) -> tuple[str, ...]:
        missing = [name for name in ("creator", "training_method", "licence_spdx", "permissions") if not getattr(self, name)]
        if self.redistributable is None:
            missing.append("redistributable")
        return tuple(missing)


@dataclass(frozen=True, slots=True)
class Compatibility:
    """What this voice needs from the runtime, stated rather than discovered by failure."""

    engine_id: str
    model_id: str
    speaker_slot: int | str | None = None
    min_platform_version: str = "0.1.0"
    required_quantization: str = ""
    sample_rate: int = 24000


@dataclass(frozen=True, slots=True)
class Measurements:
    """Performance of this voice, with the machine it was measured on."""

    host: str = ""
    measured_on: str = ""
    real_time_factor: float | None = None
    first_audio_ms: float | None = None
    peak_rss_mib: float | None = None
    load_seconds: float | None = None
    quality: dict = field(default_factory=dict)
    source: str = ""

    def measured(self) -> bool:
        return any(
            value is not None
            for value in (self.real_time_factor, self.first_audio_ms, self.peak_rss_mib, self.load_seconds)
        )


@dataclass(frozen=True, slots=True)
class InstallRecord:
    """Lifecycle facts a host needs before trusting an installed voice."""

    state: str = "installed"  # installed | staged | removed | broken
    version: str = "1"
    installed_on: str = ""
    source: str = ""
    digests: dict = field(default_factory=dict)
    updated_from: str = ""
    updated_on: str = ""
    removal: str = ""

    def installed(self) -> bool:
        return self.state == "installed"


@dataclass(frozen=True, slots=True)
class VoiceIdentity:
    voice_id: str
    display_name: str
    compatibility: Compatibility
    languages: tuple[LanguageSupport, ...] = ()
    named: bool = True
    accents: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()
    provenance: VoiceProvenance = field(default_factory=VoiceProvenance)
    measurements: Measurements = field(default_factory=Measurements)
    install: InstallRecord = field(default_factory=InstallRecord)
    tags: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.voice_id or any(separator in self.voice_id for separator in ("/", "\\", "..")):
            raise ConfigurationError("voice_id must be a safe, non-empty identifier", voice_id=self.voice_id)
        if not self.named and self.display_name and self.display_name != self.voice_id:
            raise ConfigurationError(
                "an unnamed voice may not carry an invented display name", voice_id=self.voice_id
            )

    def support_for(self, language: LanguageTag | str) -> LanguageSupport | None:
        code = language.code if isinstance(language, LanguageTag) else str(language).lower()
        return next((item for item in self.languages if item.language == code), None)

    def speaks(self, language: LanguageTag | str) -> bool:
        entry = self.support_for(language)
        return bool(entry and entry.usable() and self.install.installed())

    def require_speaks(self, language: LanguageTag | str) -> None:
        entry = self.support_for(language)
        code = language.code if isinstance(language, LanguageTag) else str(language).lower()
        if entry is None:
            raise LanguageUnsupported(
                "this voice records no capability for the requested language",
                voice_id=self.voice_id,
                language=code,
                known=tuple(item.language for item in self.languages),
            )
        if not self.install.installed():
            raise LanguageUnsupported(
                "voice is not installed in this state",
                voice_id=self.voice_id,
                language=code,
                install_state=self.install.state,
            )
        if not entry.usable():
            raise LanguageUnsupported(
                f"voice {entry.status.value} for this language",
                voice_id=self.voice_id,
                language=code,
                status=entry.status.value,
                limitations=entry.limitations,
                evidence=entry.evidence,
            )

    def as_reference(self) -> VoiceReference:
        """The engine-facing form. Slots stay inside the engine's vocabulary."""
        first = next((item for item in self.languages if item.usable()), None)
        slot = self.compatibility.speaker_slot
        return VoiceReference(
            voice_id=self.voice_id,
            state_path="",
            speaker_slot=int(slot) if isinstance(slot, int) or (isinstance(slot, str) and slot.isdigit()) else None,
            language=LanguageTag(first.language) if first else None,
            owner=self.provenance.creator,
            source=f"{self.compatibility.model_id} speaker {slot}",
        )

    def with_install(self, **changes: Any) -> "VoiceIdentity":
        return replace(self, install=replace(self.install, **changes))

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "voice_id": self.voice_id,
            "display_name": self.display_name,
            "named": self.named,
            "languages": [item.as_dict() for item in self.languages],
            "accents": list(self.accents),
            "styles": list(self.styles),
            "compatibility": {
                "engine_id": self.compatibility.engine_id,
                "model_id": self.compatibility.model_id,
                "speaker_slot": self.compatibility.speaker_slot,
                "min_platform_version": self.compatibility.min_platform_version,
                "required_quantization": self.compatibility.required_quantization,
                "sample_rate": self.compatibility.sample_rate,
            },
            "provenance": {
                "creator": self.provenance.creator,
                "training_method": self.provenance.training_method,
                "licence_spdx": self.provenance.licence_spdx,
                "permissions": self.provenance.permissions,
                "base_asset": self.provenance.base_asset,
                "redistributable": self.provenance.redistributable,
                "consent_basis": self.provenance.consent_basis,
                "notes": self.provenance.notes,
            },
            "measurements": {
                "host": self.measurements.host,
                "measured_on": self.measurements.measured_on,
                "real_time_factor": self.measurements.real_time_factor,
                "first_audio_ms": self.measurements.first_audio_ms,
                "peak_rss_mib": self.measurements.peak_rss_mib,
                "load_seconds": self.measurements.load_seconds,
                "quality": dict(self.measurements.quality),
                "source": self.measurements.source,
            },
            "install": {
                "state": self.install.state,
                "version": self.install.version,
                "installed_on": self.install.installed_on,
                "source": self.install.source,
                "digests": dict(self.install.digests),
                "updated_from": self.install.updated_from,
                "updated_on": self.install.updated_on,
                "removal": self.install.removal,
            },
            "tags": list(self.tags),
            "notes": self.notes,
        }
