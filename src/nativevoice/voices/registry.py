"""Voice identity registry and host selection.

Two things are kept apart deliberately. The *catalogue* of identities is shared -
every host reads the same installed voices - while each host's selection and data
live in its own root, so ULTRON's chosen voice, its recordings and its logs are
never FieldMind's, and neither host can corrupt the other by writing to a shared
default.

Selection resolves in a fixed order: explicit request, then host default, then
platform default. It never falls back to a voice that has not been recorded as
able to speak the requested language, because a silent substitution is how a
platform ends up answering in a voice and a language nobody chose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import data_root
from ..core.errors import ConfigurationError, LanguageUnsupported, StorageError
from ..core.language import CapabilityStatus, LanguageTag, Task
from .identity import (
    SCHEMA_VERSION,
    Compatibility,
    InstallRecord,
    LanguageSupport,
    Measurements,
    VoiceIdentity,
    VoiceProvenance,
)

PLATFORM_DEFAULT = "voice.platform.default"


def _status(value) -> CapabilityStatus:
    try:
        return CapabilityStatus(str(value))
    except ValueError:
        return CapabilityStatus.NO_MODEL


def identity_to_json(identity: VoiceIdentity) -> str:
    return json.dumps(identity.as_dict(), indent=2, ensure_ascii=False)


def identity_from_json(payload) -> VoiceIdentity:
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    schema = int(data.get("schema", 0))
    if schema > SCHEMA_VERSION:
        raise ConfigurationError(
            "identity package declares a newer schema than this platform reads",
            schema=schema,
            supported=SCHEMA_VERSION,
        )
    compatibility = data.get("compatibility") or {}
    provenance = data.get("provenance") or {}
    readings = data.get("measurements") or {}
    install = data.get("install") or {}
    languages = tuple(
        LanguageSupport(
            language=str(item.get("language", "")).lower(),
            status=_status(item.get("status")),
            accent=item.get("accent", "") or "",
            style=item.get("style", "") or "",
            metric_name=(item.get("metric") or {}).get("name", "") or "",
            metric_value=(item.get("metric") or {}).get("value"),
            measured_on=(item.get("metric") or {}).get("measured_on", "") or "",
            evidence=item.get("evidence", "") or "",
            limitations=tuple(item.get("limitations") or ()),
        )
        for item in (data.get("languages") or ())
    )
    return VoiceIdentity(
        voice_id=data["voice_id"],
        display_name=data.get("display_name", data["voice_id"]),
        named=bool(data.get("named", True)),
        languages=languages,
        accents=tuple(data.get("accents") or ()),
        styles=tuple(data.get("styles") or ()),
        compatibility=Compatibility(
            engine_id=compatibility.get("engine_id", "") or "",
            model_id=compatibility.get("model_id", "") or "",
            speaker_slot=compatibility.get("speaker_slot"),
            min_platform_version=compatibility.get("min_platform_version", "0.1.0") or "0.1.0",
            required_quantization=compatibility.get("required_quantization", "") or "",
            sample_rate=int(compatibility.get("sample_rate", 24000)),
        ),
        provenance=VoiceProvenance(
            creator=provenance.get("creator", "") or "",
            training_method=provenance.get("training_method", "") or "",
            licence_spdx=provenance.get("licence_spdx", "") or "",
            permissions=provenance.get("permissions", "") or "",
            base_asset=provenance.get("base_asset", "") or "",
            redistributable=provenance.get("redistributable"),
            consent_basis=provenance.get("consent_basis", "") or "",
            notes=provenance.get("notes", "") or "",
        ),
        measurements=Measurements(
            host=readings.get("host", "") or "",
            measured_on=readings.get("measured_on", "") or "",
            real_time_factor=readings.get("real_time_factor"),
            first_audio_ms=readings.get("first_audio_ms"),
            peak_rss_mib=readings.get("peak_rss_mib"),
            load_seconds=readings.get("load_seconds"),
            quality=dict(readings.get("quality") or {}),
            source=readings.get("source", "") or "",
        ),
        install=InstallRecord(
            state=install.get("state", "installed") or "installed",
            version=str(install.get("version", "1")),
            installed_on=install.get("installed_on", "") or "",
            source=install.get("source", "") or "",
            digests=dict(install.get("digests") or {}),
            updated_from=install.get("updated_from", "") or "",
            updated_on=install.get("updated_on", "") or "",
            removal=install.get("removal", "") or "",
        ),
        tags=tuple(data.get("tags") or ()),
        notes=data.get("notes", "") or "",
    )


@dataclass(slots=True)
class VoiceRegistry:
    """The shared catalogue of installed voice identities."""

    root: Path = field(default_factory=lambda: Path(data_root()) / "voices")
    unreadable: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def discover(self) -> tuple[VoiceIdentity, ...]:
        found: list[VoiceIdentity] = []
        self.unreadable = {}
        if not self.root.is_dir():
            return ()
        for path in sorted(self.root.glob("*.json")):
            try:
                found.append(identity_from_json(path.read_text(encoding="utf-8")))
            except Exception as error:
                # A file that fails to parse is reported, never dropped quietly: an
                # identity that vanishes from the catalogue is indistinguishable from
                # one that was deliberately removed, and those mean different things.
                self.unreadable[path.name] = f"{type(error).__name__}: {error}"
        return tuple(found)

    def all(self) -> tuple[VoiceIdentity, ...]:
        return self.discover()

    def get(self, voice_id: str) -> VoiceIdentity | None:
        return next((item for item in self.discover() if item.voice_id == voice_id), None)

    def save(self, identity: VoiceIdentity) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{identity.voice_id}.json"
        path.write_text(identity_to_json(identity), encoding="utf-8")
        return path

    def remove(self, voice_id: str) -> bool:
        path = self.root / f"{voice_id}.json"
        if not path.exists() and self.get(voice_id) is None:
            return False
        path.unlink(missing_ok=True)
        return True

    def candidates(self, language: LanguageTag | str) -> tuple[VoiceIdentity, ...]:
        return tuple(item for item in self.discover() if item.speaks(language))

    def select(
        self,
        requested: VoiceIdentity | str | None = None,
        language: LanguageTag | str | None = None,
        host: "HostProfile | None" = None,
    ) -> VoiceIdentity:
        chosen = self._resolve(requested, host)
        if language is None:
            return chosen
        try:
            chosen.require_speaks(language)
        except LanguageUnsupported as error:
            raise LanguageUnsupported(
                "the selected voice is not recorded as able to render this language",
                voice_id=chosen.voice_id,
                language=getattr(error, "details", {}).get("language", str(language)),
                recorded=tuple(item.language for item in chosen.languages),
                other_installed=tuple(item.voice_id for item in self.candidates(language)),
            ) from error
        return chosen

    def _resolve(self, requested, host: "HostProfile | None") -> VoiceIdentity:
        if isinstance(requested, VoiceIdentity):
            return requested
        if requested is not None:
            found = self.get(str(requested))
            if found is None:
                # An explicitly requested voice is never quietly replaced: a host that
                # asks for its own narrator and gets somebody else's has been told a
                # lie in a language it can hear and a log it will not read.
                raise StorageError(
                    "requested voice is not installed",
                    voice_id=str(requested),
                    root=str(self.root),
                    installed=tuple(item.voice_id for item in self.discover()),
                )
            return found
        for candidate_id in filter(None, (host.default_voice if host else None, PLATFORM_DEFAULT)):
            found = self.get(str(candidate_id))
            if found is not None:
                return found
        raise StorageError(
            "no voice identity is installed",
            looked_for=PLATFORM_DEFAULT,
            root=str(self.root),
            detail="install one with tools/seed_voices.py or register a package per host",
        )


@dataclass(slots=True)
class HostProfile:
    """One host application's own slice of the platform: its defaults and its data."""

    host_id: str
    display_name: str = ""
    default_voice: str = PLATFORM_DEFAULT
    default_language: str = ""
    root: Path | None = None

    def __post_init__(self) -> None:
        if not self.host_id or any(bad in self.host_id for bad in ("/", "\\", "..", " ", " ")):
            raise ConfigurationError("host_id must be a safe identifier", host_id=self.host_id)
        self.root = Path(self.root) if self.root is not None else Path(data_root()) / "hosts" / self.host_id

    @property
    def recordings(self) -> Path:
        return self.root / "recordings"

    @property
    def profiles(self) -> Path:
        return self.root / "profiles"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> Path:
        for path in (self.root, self.recordings, self.profiles, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        return self.root
