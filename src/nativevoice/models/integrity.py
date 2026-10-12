"""Content integrity: does the file on disk match what was reviewed?

Approval answers "is this licensed for us to ship"; integrity answers "is this
still the exact bytes that were approved". They are orthogonal - a corrupt file
from an approved source is dangerous in a different way from an unapproved one,
so it gets a different error and a different remedy (re-download, not re-review).

The honesty rule that drives this module: an asset with no recorded digest can
never be reported as *verified*. Absence of evidence is not evidence of
correctness, so such an asset is classified ``unverifiable`` and the whole
manifest fails verification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .api import Asset, ModelManifest

_CHUNK = 1024 * 1024


def base_directory(manifest: ModelManifest, registry_root: str | Path) -> Path:
    """Where this model's assets actually are.

    Provisioned assets do not all live under one tree: a sherpa model unpacks to
    `<models>/<model_id>/`, while a Hugging Face snapshot lives in the HF cache. A
    single integrity root silently reported the second as missing, so a manifest may
    name its own base - absolute, or relative to the registry root - and the default
    stays `<registry_root>/<model_id>`.
    """
    if manifest.asset_root:
        candidate = Path(manifest.asset_root)
        return candidate if candidate.is_absolute() else Path(registry_root) / candidate
    return Path(registry_root) / manifest.model_id


def resolve_asset_path(root: str | Path, asset: Asset, manifest: ModelManifest | None = None) -> Path:
    base = base_directory(manifest, root) if manifest is not None else Path(root)
    return base / asset.relative_path


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            hasher.update(block)
    return hasher.hexdigest()


@dataclass(slots=True)
class IntegrityReport:
    """Per-asset classification so callers can pick the right error to raise.

    ``missing`` files are an installation problem; ``unverifiable`` and
    ``mismatched`` are integrity problems, but only ``verified`` counts as
    positive evidence of correctness.
    """

    verified: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unverifiable: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()
    reasons: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unverifiable and not self.mismatched and bool(self.verified)


def inspect(manifest: ModelManifest, root: str | Path) -> IntegrityReport:
    """Classify every declared asset against size + digest on disk."""
    verified: list[str] = []
    missing: list[str] = []
    unverifiable: list[str] = []
    mismatched: list[str] = []
    reasons: list[str] = []

    if not manifest.assets:
        reasons.append("manifest declares no assets - nothing to verify")
        return IntegrityReport(reasons=tuple(reasons))

    base = base_directory(manifest, root)
    for asset in manifest.assets:
        path = base / asset.relative_path
        label = asset.relative_path
        if not path.is_file():
            missing.append(label)
            reasons.append(f"asset missing: {label}")
            continue
        if not asset.sha256:
            unverifiable.append(label)
            reasons.append(f"asset has no recorded digest (unverifiable): {label}")
            continue
        if asset.size_bytes and path.stat().st_size != asset.size_bytes:
            mismatched.append(label)
            reasons.append(
                f"size mismatch for {label}: recorded {asset.size_bytes} bytes, found {path.stat().st_size} bytes"
            )
            continue
        actual = _digest(path)
        if actual.lower() != asset.sha256.lower():
            mismatched.append(label)
            reasons.append(f"digest mismatch for {label}: recorded {asset.sha256}, computed {actual}")
            continue
        verified.append(label)

    if not verified:
        reasons.append("no asset passed verification")

    return IntegrityReport(
        verified=tuple(verified),
        missing=tuple(missing),
        unverifiable=tuple(unverifiable),
        mismatched=tuple(mismatched),
        reasons=tuple(reasons),
    )


def verify(manifest: ModelManifest, root: str | Path) -> tuple[bool, tuple[str, ...]]:
    """Return ``(ok, reasons)`` for the whole manifest."""
    report = inspect(manifest, root)
    return report.ok, report.reasons
