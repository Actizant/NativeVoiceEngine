"""Local data directory resolution.

Everything this system persists - approved models, captured voices, cache, logs -
lives under one root so a user can find it, back it up, or delete it as a unit.
The root honours ``NATIVEVOICE_HOME`` when set (the portable override), otherwise
the platform-conventional location. Nothing in this module creates a directory
outside that root unless the caller passes an explicit absolute path, which is
the whole guarantee behind "local-first, no surprise writes".

Measurement-only assets live in a root of their own beside it (see
:func:`measurement_data_root`). This module computes that name and creates
nothing there; the isolation is a separate directory tree, not a flag inside one
tree that a tool can forget to read.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

from ..core.errors import ConfigurationError, StorageError

SUBDIRS = ("models", "voices", "cache", "logs")
_ENV_HOME = "NATIVEVOICE_HOME"
_DIR_NAME = "ultron-voice"
_MEASUREMENT_SUFFIX = "-measurement"


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


def measurement_data_root() -> Path:
    """The root for measurement-only assets: a sibling of :func:`data_root`.

    Deliberately a different tree rather than a subdirectory of the product root,
    so the two never share a manifest directory, a backup, or an accidental
    recursive delete - and so ``_ensure_within_root`` cannot be talked into
    creating product-path content from a measurement tool.
    """
    root = data_root()
    return root.with_name(root.name + _MEASUREMENT_SUFFIX)


def measurement_models_dir() -> Path:
    """Measurement counterpart of :func:`models_dir`, same internal layout.

    ``<measurement root>/models/manifests/*.json`` is discovered exactly like the
    product models directory, so a measurement registry is the same class with a
    different root and scope - not a parallel implementation to keep in sync.
    """
    return measurement_data_root() / "models"


def hf_hub_cache_root() -> Path:
    """The Hugging Face hub cache directory, from the environment only.

    One implementation, because the tool that digests a snapshot and the engine that
    loads one have to be looking at the same directory: if they disagree, the digest
    in the provenance record describes bytes no loader will ever read. Precedence
    follows ``huggingface_hub`` itself, so an environment set for the library is
    honoured rather than half-read - ``HF_HUB_CACHE`` names the hub directory
    directly, ``HF_HOME`` is its parent, and neither is a literal path here.

    Creates nothing, and resolves nothing: a cache that has never been written stays
    absent, and the caller reports it as an installation gap.
    """
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser()
    home = os.environ.get("HF_HOME")
    base = Path(home).expanduser() if home else Path(os.path.expanduser("~")) / ".cache" / "huggingface"
    return base / "hub"
