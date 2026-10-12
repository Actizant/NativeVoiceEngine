"""Settings, paths and the promise that telemetry is not a knob."""

from __future__ import annotations

import json

import pytest

from nativevoice.config.paths import (
    SUBDIRS,
    cache_dir,
    data_root,
    ensure_root,
    ensure_subdir,
    logs_dir,
    models_dir,
    subpath,
    voices_dir,
)
from nativevoice.config.settings import Settings, load_settings, save_settings
from nativevoice.core.errors import ConfigurationError, StorageError


def test_defaults_are_privacy_first(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path))
    settings = load_settings()
    assert settings.recordings_retention == "session"
    assert settings.voice_output_enabled is True
    assert settings.offline_only is False


def test_no_telemetry_knob_exists_on_class():
    fields = set(Settings.__dataclass_fields__)
    assert "telemetry_enabled" not in fields


def test_settings_from_dict_refuses_telemetry_key():
    with pytest.raises(ConfigurationError):
        Settings.from_dict({"telemetry_enabled": False})


def test_round_trip_preserves_known_fields():
    original = Settings(
        recordings_retention="keep",
        voice_output_enabled=False,
        offline_only=True,
        default_profile="performance",
        max_resident_mib=2048,
        max_threads=4,
    )
    restored = Settings.from_dict(original.to_dict())
    assert restored == original


def test_round_trip_preserves_unknown_keys(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"future_flag": {"a": 1}, "unknown_count": 5, "offline_only": True}))
    settings = load_settings(path)
    assert set(settings.unknown_keys) == {"future_flag", "unknown_count"}
    saved = save_settings(settings, path)
    reloaded = load_settings(saved)
    assert set(reloaded.unknown_keys) == {"future_flag", "unknown_count"}


def test_bad_retention_rejected():
    with pytest.raises(ConfigurationError):
        Settings.from_dict({"recordings_retention": "forever"})


def test_bool_must_be_bool():
    with pytest.raises(ConfigurationError):
        Settings.from_dict({"voice_output_enabled": 1})


def test_int_limits_must_be_positive():
    with pytest.raises(ConfigurationError):
        Settings.from_dict({"max_resident_mib": 0})
    with pytest.raises(ConfigurationError):
        Settings.from_dict({"max_threads": -1})


def test_int_limits_may_be_null():
    settings = Settings.from_dict({"max_resident_mib": None, "max_threads": None})
    assert settings.max_resident_mib is None
    assert settings.max_threads is None


def test_data_root_honours_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path))
    assert data_root() == tmp_path
    assert models_dir() == tmp_path / "models"
    assert voices_dir() == tmp_path / "voices"
    assert cache_dir() == tmp_path / "cache"
    assert logs_dir() == tmp_path / "logs"
    for name in SUBDIRS:
        assert subpath(name).parent == tmp_path


def test_data_root_windows_localappdata(monkeypatch):
    monkeypatch.delenv("NATIVEVOICE_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", r"C:\\Users\\tester\\AppData\\Local")
    monkeypatch.setattr("platform.system", lambda: "Windows")
    root = data_root()
    assert root.name == "ultron-voice"
    assert root.parent.name == "Local"


def test_data_root_posix_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("NATIVEVOICE_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("platform.system", lambda: "Linux")
    root = data_root()
    assert str(root).endswith(".local/share/ultron-voice") or str(root).endswith(".local\\share\\ultron-voice")


def test_unknown_subdir_rejected():
    with pytest.raises(ConfigurationError):
        subpath("secret")


def test_ensure_root_and_subdir_create_inside_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path))
    assert ensure_root() == tmp_path
    assert (tmp_path / "tmp_existing").exists() is False
    models = ensure_subdir("models")
    assert models == tmp_path / "models"
    assert models.is_dir()


def test_refuses_to_escape_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path))
    from nativevoice.config.paths import _ensure_within_root
    from pathlib import Path

    outside = Path(tmp_path.parent / "escape")
    with pytest.raises(StorageError):
        _ensure_within_root(outside)
