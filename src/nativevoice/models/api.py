"""Model asset contracts: provenance, review state, and resource requirements.

The specification forbids shipping third-party weights without an explicit
provenance and licence review, so that rule is expressed as data and enforced at
load time rather than left to documentation. Three axes are kept separate
because the reference study showed them being conflated:

* `ReviewStatus` - has a human approved this asset? starts `UNREVIEWED`, and
  nothing in the loader will proceed past it.
* integrity - does the file on disk match the digest recorded at review time?
* `requirements` - does it fit the machine, per measurement rather than guess?

An asset can be present and corrupt, present and unapproved, or approved and
not installable offline; each of those is a different error with a different
remedy, so each gets its own field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from ..core.language import Task


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class AssetKind(str, Enum):
    MODEL = "model"
    TOKENIZER = "tokenizer"
    VOICE_ASSET = "voice_asset"
    RUNTIME_BINARY = "runtime_binary"
    EVALUATION_DATA = "evaluation_data"
    TRAINING_DATA = "training_data"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where an asset came from and on what terms.

    `licence_spdx` of "" means unknown, which is not the same as permissive;
    `ReviewStatus.UNREVIEWED` is the only state consistent with it.
    """

    source: str
    upstream_project: str = ""
    licence_spdx: str = ""
    licence_notice: str = ""
    consent_basis: str = ""
    permissions: str = ""
    redistributable: bool | None = None
    notes: str = ""

    def incomplete(self) -> tuple[str, ...]:
        missing = []
        if not self.source:
            missing.append("source")
        if not self.licence_spdx:
            missing.append("licence_spdx")
        if self.redistributable is None:
            missing.append("redistributable")
        return tuple(missing)


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    """Measured or declared cost of running one model.

    `measured` distinguishes the two. A declared figure may gate a suggestion;
    only a measured one may gate an automatic profile decision.
    """

    ram_mb: int = 0
    vram_mb: int = 0
    disk_mb: int = 0
    threads: int = 1
    needs_gpu: bool = False
    measured: bool = False
    rtf_cpu: float | None = None
    rtf_gpu: float | None = None
    first_audio_ms: float | None = None
    measured_on: str = ""

    def as_dict(self) -> dict:
        return {
            "ram_mb": self.ram_mb,
            "vram_mb": self.vram_mb,
            "disk_mb": self.disk_mb,
            "threads": self.threads,
            "needs_gpu": self.needs_gpu,
            "measured": self.measured,
            "rtf_cpu": self.rtf_cpu,
            "rtf_gpu": self.rtf_gpu,
            "first_audio_ms": self.first_audio_ms,
            "measured_on": self.measured_on,
        }


@dataclass(frozen=True, slots=True)
class Asset:
    kind: AssetKind
    relative_path: str
    sha256: str = ""
    size_bytes: int = 0
    url: str = ""


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Everything needed to decide whether and how to load one model."""

    model_id: str
    engine: str
    task: Task
    display_name: str = ""
    version: str = ""
    languages: tuple[str, ...] = ()
    sample_rate: int = 16000
    quantization: str = ""
    provenance: Provenance = field(default_factory=lambda: Provenance(source=""))
    review: ReviewStatus = ReviewStatus.UNREVIEWED
    approved_by: str = ""
    approved_on: str = ""
    requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    assets: tuple[Asset, ...] = ()
    asset_root: str = ""
    minimum_profile: str = "lite"
    notes: str = ""

    @property
    def key(self) -> str:
        return self.model_id


@dataclass(frozen=True, slots=True)
class ModelStatus:
    manifest: ModelManifest
    installed: bool = False
    verified: bool = False
    usable: bool = False
    reasons: tuple[str, ...] = ()

    def explain(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "ready"


class ModelLoader(Protocol):
    def prepare(self, manifest: ModelManifest) -> None: ...

    def load(self, manifest: ModelManifest, budget: object = None) -> object: ...

    def unload(self, model_id: str) -> None: ...
