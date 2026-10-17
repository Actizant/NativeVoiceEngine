"""Local configuration: settings and data-directory resolution.

Standard-library only. ``paths`` owns where files live (honouring
``NATIVEVOICE_HOME``), ``settings`` owns the knobs the user controls - retention,
output, offline posture, hardware limits - and the absence of a telemetry knob is
itself a documented guarantee, not an oversight.
"""

from __future__ import annotations

from .paths import (
    SUBDIRS,
    cache_dir,
    data_root,
    ensure_root,
    ensure_subdir,
    hf_hub_cache_root,
    logs_dir,
    measurement_data_root,
    measurement_models_dir,
    models_dir,
    subpath,
    voices_dir,
)
from .settings import Settings, default_settings_path, load_settings, save_settings

__all__ = [
    "SUBDIRS",
    "Settings",
    "cache_dir",
    "data_root",
    "default_settings_path",
    "ensure_root",
    "ensure_subdir",
    "hf_hub_cache_root",
    "load_settings",
    "logs_dir",
    "measurement_data_root",
    "measurement_models_dir",
    "models_dir",
    "save_settings",
    "subpath",
    "voices_dir",
]
