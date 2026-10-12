"""Local data directory resolution.

Everything this system persists - approved models, captured voices, cache, logs -
lives under one root so a user can find it, back it up, or delete it as a unit.
The root honours ``NATIVEVOICE_HOME`` when set (the portable override), otherwise
the platform-conventional location. Nothing in this module creates a directory
outside that root unless the caller passes an explicit absolute path, which is
the whole guarantee behind "local-first, no surprise writes".
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

from ..core.errors import ConfigurationError, StorageError

SUBDIRS = ("models", "voices", "cache", "logs")
_ENV_HOME = "NATIVEVOICE_HOME"
_DIR_NAME = "ultron-voice"


def data_root() -> Path:
    env = os.environ.get(_ENV_HOME)
    if env:
        return Path(env).expanduser()
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / _DIR_NAME
        return Path.home() / "AppData" / "Local" / _DIR_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / _DIR_NAME
    return Path.home() / ".local" / "share" / _DIR_NAME


def subpath(name: str) -> Path:
    if name not in SUBDIRS:
        raise ConfigurationError("unknown data subdirectory", name=name, known=list(SUBDIRS))
    return data_root() / name


def _ensure_within_root(path: Path) -> None:
    root = data_root().resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise StorageError("refusing to create a directory outside the data root", path=str(resolved), root=str(root))


def ensure_root() -> Path:
    root = data_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError("cannot create data root", path=str(root)) from exc
    return root


def ensure_subdir(name: str) -> Path:
    path = subpath(name)
    _ensure_within_root(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError("cannot create data subdirectory", path=str(path)) from exc
    return path


def models_dir() -> Path:
    return subpath("models")


def voices_dir() -> Path:
    return subpath("voices")


def cache_dir() -> Path:
    return subpath("cache")


def logs_dir() -> Path:
    return subpath("logs")
