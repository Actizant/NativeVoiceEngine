"""Model asset contracts: provenance, review state, and resource requirements.

The specification forbids shipping third-party weights without an explicit
provenance and licence review, so that rule is expressed as data and enforced at
load time rather than left to documentation. Four axes are kept separate
because the reference study showed them being conflated:

* `ReviewStatus` - has a human approved this asset? starts `UNREVIEWED`, and
  nothing in the loader will proceed past it.
* integrity - does the file on disk match the digest recorded at review time?
* `requirements` - does it fit the machine, per measurement rather than guess?
* `AssetScope` - which registry may serve the asset at all. This is not part of
  approval: an asset can be reviewed and approved and still be confined to
  measurement work, because "a human approved this" answers a different question
  from "this may be routed to a customer". `review` alone cannot express the
  second, so conflating them would turn an approval record into a distribution
  claim.

An asset can be present and corrupt, present and unapproved, approved and not
installable offline, or approved and not meant for the product path at all; each
of those is a different error with a different remedy, so each gets its own field.
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


class AssetScope(str, Enum):
    """Which registry may serve this asset, independent of whether it is approved.

    `PRODUCT` is the shipped voice path: routing, the model manager and anything
    a customer can reach. `MEASUREMENT` is local research only - an asset that is
    on disk and may even be fully approved, but whose permission to be *served*
    has not been established. A manifest with no `use_scope` is `PRODUCT`, so
    existing manifests keep loading exactly as before; isolation always has to be
    asserted, never inferred from absence.
    """

    PRODUCT = "product"
    MEASUREMENT = "measurement"


class AssetKind(str, Enum):
    """What one file inside an asset package *is*.

    The four trailing values exist because a package is not only model data: it
    carries its licence text, its README, its VCS attributes and occasionally a
    helper script. Before them the choice was to label those as a tokenizer asset or
    to refuse the whole package, and both answers destroy evidence - the first puts
    a wrong description into the provenance record a later review reads, the second
    leaves the file out of the digest list entirely. Recorded with their own digest
    and size, they are pinned like any other file and loaded by nothing.
    """

    MODEL = "model"
    MODEL_CONFIG = "model_config"  # config.json / preprocessor_config.json: not weights, not a vocabulary
    TOKENIZER = "tokenizer"
    VOICE_ASSET = "voice_asset"
    RUNTIME_BINARY = "runtime_binary"
    EVALUATION_DATA = "evaluation_data"
    TRAINING_DATA = "training_data"
    LICENCE = "licence"  # the licence text shipped with the asset, read rather than summarised
    DOCUMENTATION = "documentation"  # README / model card: prose, never a runtime input
    PACKAGE_METADATA = "package_metadata"  # .gitattributes and friends: packaging, not content
    SOURCE = "source"  # executable code shipped inside a package, whose licence may differ


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
    use_scope: AssetScope = AssetScope.PRODUCT
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
