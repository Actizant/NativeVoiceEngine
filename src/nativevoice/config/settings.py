"""Local settings for the voice stack.

The specification (§11) insists that this system never phones home. There is
deliberately no ``telemetry_enabled`` field here, and no plan to add one: a knob
that promises "off" but sits next to a real telemetry path would be a lie, and
this module is the honest record of what the system *can* be told to do. Adding
that name here later would mean the privacy claim in the specification is
already false; the field itself is intentionally absent so that claim is not
undercut by a config schema.

Retention, output, and offline posture are the settings the user actually
controls: how long recordings live, whether audio ever reaches a speaker, and
whether the runtime may contact anything over the network. Unknown keys are
preserved on round-trip, not silently dropped, so a future version does not
corrupt the file of a user rolling back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ConfigurationError, StorageError
from .paths import data_root

_VALID_RETENTION = ("none", "session", "keep")
_DEFAULT_FILENAME = "settings.json"

_TELEMETRY_FORBIDDEN = "telemetry_enabled"


@dataclass(slots=True)
class Settings:
    recordings_retention: str = "session"
    voice_output_enabled: bool = True
    offline_only: bool = False
    default_profile: str = "balanced"
    max_resident_mib: int | None = None
    max_threads: int | None = None
    unknown_values: dict = field(default_factory=dict)

    @property
    def unknown_keys(self) -> tuple[str, ...]:
        """Keys the loader saw in the file that this version does not understand."""
        return tuple(sorted(self.unknown_values))

    @classmethod
    def defaults(cls) -> "Settings":
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        if not isinstance(data, dict):
            raise ConfigurationError("settings must be a JSON object", got=type(data).__name__)
        if _TELEMETRY_FORBIDDEN in data:
            raise ConfigurationError(
                "telemetry is not a supported concept in this system; refusing to record it as a knob",
                key=_TELEMETRY_FORBIDDEN,
            )
        known_map = {
            "recordings_retention": _as_str("recordings_retention", data.get("recordings_retention", "session")),
            "voice_output_enabled": _as_bool("voice_output_enabled", data.get("voice_output_enabled", True)),
            "offline_only": _as_bool("offline_only", data.get("offline_only", False)),
            "default_profile": _as_str("default_profile", data.get("default_profile", "balanced")),
            "max_resident_mib": _as_positive("max_resident_mib", data.get("max_resident_mib"), minimum=1),
            "max_threads": _as_positive("max_threads", data.get("max_threads"), minimum=1),
        }
        retention = known_map["recordings_retention"]
        if retention not in _VALID_RETENTION:
            raise ConfigurationError(
                "recordings_retention must be one of none/session/keep",
                value=retention,
                allowed=list(_VALID_RETENTION),
            )
        profile = known_map["default_profile"]
        if not profile:
            raise ConfigurationError("default_profile must be a non-empty string", value=profile)
        unknown = {k: v for k, v in data.items() if k not in known_map}
        return cls(unknown_values=unknown, **known_map)

    def to_dict(self) -> dict:
        payload: dict[str, Any] = {
            "recordings_retention": self.recordings_retention,
            "voice_output_enabled": self.voice_output_enabled,
            "offline_only": self.offline_only,
            "default_profile": self.default_profile,
            "max_resident_mib": self.max_resident_mib,
            "max_threads": self.max_threads,
        }
        payload.update(self.unknown_values)
        return payload


def _as_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigurationError("setting must be a string", key=name, got=type(value).__name__)
    return value


def _as_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError("setting must be a boolean", key=name, got=type(value).__name__)
    return value


def _as_positive(name: str, value: Any, *, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError("setting must be an integer or null", key=name, got=type(value).__name__)
    if value < minimum:
        raise ConfigurationError("setting below allowed minimum", key=name, value=value, minimum=minimum)
    return value


def default_settings_path() -> Path:
    return data_root() / _DEFAULT_FILENAME


def load_settings(path: str | Path | None = None) -> Settings:
    resolved = Path(path) if path is not None else default_settings_path()
    if not resolved.exists():
        return Settings.defaults()
    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise StorageError("cannot read settings", path=str(resolved)) from exc
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ConfigurationError("settings file is not valid JSON", path=str(resolved)) from exc
    return Settings.from_dict(data)


def save_settings(settings: Settings, path: str | Path | None = None) -> Path:
    resolved = Path(path) if path is not None else default_settings_path()
    payload = json.dumps(settings.to_dict(), indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise StorageError("cannot write settings", path=str(resolved)) from exc
    return resolved


__all__ = [
    "Settings",
    "default_settings_path",
    "load_settings",
    "save_settings",
]
