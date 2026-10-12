"""Manifest (de)serialisation between ``ModelManifest`` and human-editable JSON.

Manifests are the durable record of what a model is, where it came from and what
it costs to run, so they are edited by humans during licence review. That means
two properties the loader must guarantee:

* a file with extra keys - a field added by a newer version, a note from a
  reviewer - loads without crashing, and the unrecognised keys are surfaced in
  ``notes`` rather than silently dropped, so nothing is lost across an edit;
* ``to_dict``/``from_dict`` round-trips every field, so rewriting a manifest
  never corrupts the provenance or review state it is guarding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.errors import ConfigurationError, StorageError
from ..core.language import Task
from .api import Asset, AssetKind, ModelManifest, Provenance, ResourceRequirement, ReviewStatus

_UNKNOWN_NOTE = "unknown keys preserved"


def _provenance_to_dict(value: Provenance) -> dict:
    return {
        "source": value.source,
        "upstream_project": value.upstream_project,
        "licence_spdx": value.licence_spdx,
        "licence_notice": value.licence_notice,
        "consent_basis": value.consent_basis,
        "redistributable": value.redistributable,
        "notes": value.notes,
    }


def _provenance_from_dict(data: dict) -> Provenance:
    return Provenance(
        source=data.get("source", ""),
        upstream_project=data.get("upstream_project", ""),
        licence_spdx=data.get("licence_spdx", ""),
        licence_notice=data.get("licence_notice", ""),
        consent_basis=data.get("consent_basis", ""),
        redistributable=data.get("redistributable"),
        notes=data.get("notes", ""),
    )


def _requirements_to_dict(value: ResourceRequirement) -> dict:
    return {
        "ram_mb": value.ram_mb,
        "vram_mb": value.vram_mb,
        "disk_mb": value.disk_mb,
        "threads": value.threads,
        "needs_gpu": value.needs_gpu,
        "measured": value.measured,
        "rtf_cpu": value.rtf_cpu,
        "rtf_gpu": value.rtf_gpu,
        "first_audio_ms": value.first_audio_ms,
        "measured_on": value.measured_on,
    }


def _requirements_from_dict(data: dict) -> ResourceRequirement:
    return ResourceRequirement(
        ram_mb=int(data.get("ram_mb", 0) or 0),
        vram_mb=int(data.get("vram_mb", 0) or 0),
        disk_mb=int(data.get("disk_mb", 0) or 0),
        threads=int(data.get("threads", 1) or 1),
        needs_gpu=bool(data.get("needs_gpu", False)),
        measured=bool(data.get("measured", False)),
        rtf_cpu=None if data.get("rtf_cpu") is None else float(data["rtf_cpu"]),
        rtf_gpu=None if data.get("rtf_gpu") is None else float(data["rtf_gpu"]),
        first_audio_ms=None if data.get("first_audio_ms") is None else float(data["first_audio_ms"]),
        measured_on=data.get("measured_on", ""),
    )


def _asset_to_dict(value: Asset) -> dict:
    return {
        "kind": value.kind.value,
        "relative_path": value.relative_path,
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "url": value.url,
    }


def _asset_from_dict(data: dict) -> Asset:
    kind_raw = data.get("kind", AssetKind.MODEL.value)
    try:
        kind = AssetKind(kind_raw)
    except ValueError as exc:
        raise ConfigurationError("unknown asset kind", kind=kind_raw) from exc
    return Asset(
        kind=kind,
        relative_path=data.get("relative_path", ""),
        sha256=data.get("sha256", ""),
        size_bytes=int(data.get("size_bytes", 0) or 0),
        url=data.get("url", ""),
    )


_KNOWN_FIELDS = frozenset(
    {
        "model_id",
        "engine",
        "task",
        "display_name",
        "version",
        "languages",
        "sample_rate",
        "quantization",
        "provenance",
        "review",
        "approved_by",
        "approved_on",
        "requirements",
        "assets",
        "asset_root",
        "minimum_profile",
        "notes",
    }
)


def to_dict(manifest: ModelManifest) -> dict:
    """Full, lossless dict form - every field present, enums as their values."""
    return {
        "model_id": manifest.model_id,
        "engine": manifest.engine,
        "task": manifest.task.value,
        "display_name": manifest.display_name,
        "version": manifest.version,
        "languages": list(manifest.languages),
        "sample_rate": manifest.sample_rate,
        "quantization": manifest.quantization,
        "provenance": _provenance_to_dict(manifest.provenance),
        "review": manifest.review.value,
        "approved_by": manifest.approved_by,
        "approved_on": manifest.approved_on,
        "requirements": _requirements_to_dict(manifest.requirements),
        "assets": [_asset_to_dict(a) for a in manifest.assets],
        "asset_root": manifest.asset_root,
        "minimum_profile": manifest.minimum_profile,
        "notes": manifest.notes,
    }


def from_dict(data: dict) -> ModelManifest:
    """Build a manifest from a dict, folding unknown keys into ``notes``.

    Unknown keys are never dropped: a licence reviewer editing a manifest must
    not lose fields a newer writer added. They are recorded so the loss is
    visible rather than silent.
    """
    if not isinstance(data, dict):
        raise ConfigurationError("manifest must be a mapping", got=type(data).__name__)
    model_id = data.get("model_id")
    if not model_id:
        raise ConfigurationError("manifest requires model_id")
    task_raw = data.get("task", Task.STT.value)
    try:
        task = Task(task_raw)
    except ValueError as exc:
        raise ConfigurationError("unknown task", task=task_raw) from exc
    review_raw = data.get("review", ReviewStatus.UNREVIEWED.value)
    try:
        review = ReviewStatus(review_raw)
    except ValueError as exc:
        raise ConfigurationError("unknown review status", review=review_raw) from exc
    provenance_data = data.get("provenance") or {}
    requirements_data = data.get("requirements") or {}
    assets_data = data.get("assets") or []
    notes = data.get("notes", "")
    unknown = sorted(k for k in data if k not in _KNOWN_FIELDS)
    if unknown:
        folded = f"{_UNKNOWN_NOTE}: {', '.join(unknown)}"
        notes = f"{notes}; {folded}" if notes else folded
    languages_raw = data.get("languages") or ()
    return ModelManifest(
        model_id=str(model_id),
        engine=str(data.get("engine", "")),
        task=task,
        display_name=str(data.get("display_name", "")),
        version=str(data.get("version", "")),
        languages=tuple(str(x) for x in languages_raw),
        sample_rate=int(data.get("sample_rate", 16000) or 16000),
        quantization=str(data.get("quantization", "")),
        provenance=_provenance_from_dict(provenance_data),
        review=review,
        approved_by=str(data.get("approved_by", "")),
        approved_on=str(data.get("approved_on", "")),
        requirements=_requirements_from_dict(requirements_data),
        assets=tuple(_asset_from_dict(a) for a in assets_data),
        asset_root=str(data.get("asset_root", "")),
        minimum_profile=str(data.get("minimum_profile", "lite")),
        notes=notes,
    )


def load_manifest(path: str | Path) -> ModelManifest:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StorageError("cannot read manifest", path=str(path)) from exc
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("manifest is not valid JSON", path=str(path)) from exc
    return from_dict(data)


def save_manifest(manifest: ModelManifest, path: str | Path) -> Path:
    path = Path(path)
    payload = json.dumps(to_dict(manifest), indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise StorageError("cannot write manifest", path=str(path)) from exc
    return path
