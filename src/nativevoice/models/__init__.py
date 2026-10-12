"""Model registry: asset contracts, discovery, integrity, approval and lifecycle.

The ``api`` module is the frozen contract; this package supplies everything that
enforces it at load time - a file-based :class:`ModelRegistry`, the SHA-256
:class:`~nativevoice.models.integrity` checks, the default-deny
:class:`~nativevoice.models.approval` gate and the :class:`ModelManager`
lifecycle. Standard-library only: nothing here imports a neural runtime, which
is exactly the point - the gate that decides *whether* to run a model must not
itself require running one.
"""

from __future__ import annotations

from .api import (
    Asset,
    AssetKind,
    ModelLoader,
    ModelManifest,
    ModelStatus,
    Provenance,
    ResourceRequirement,
    ReviewStatus,
)
from .approval import (
    ApprovalGate,
    ApprovalOverride,
    approval_blockers,
    ensure_approved,
    is_approved,
)
from .integrity import IntegrityReport, inspect, resolve_asset_path, verify
from .lifecycle import ModelManager
from .manifest_io import (
    from_dict,
    load_manifest,
    save_manifest,
    to_dict,
)
from .registry import ModelRegistry, RegistryIssue

__all__ = [
    "ApprovalGate",
    "ApprovalOverride",
    "Asset",
    "AssetKind",
    "IntegrityReport",
    "ModelLoader",
    "ModelManifest",
    "ModelManager",
    "ModelRegistry",
    "ModelStatus",
    "Provenance",
    "RegistryIssue",
    "ResourceRequirement",
    "ReviewStatus",
    "approval_blockers",
    "ensure_approved",
    "from_dict",
    "inspect",
    "is_approved",
    "load_manifest",
    "manifest_with_notes",
    "resolve_asset_path",
    "save_manifest",
    "to_dict",
    "verify",
]
