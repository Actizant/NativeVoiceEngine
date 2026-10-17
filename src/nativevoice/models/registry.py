"""The model registry: discovery, indexing and per-model status.

A registry points at a directory of human-edited manifest JSON files and answers
three questions without ever touching a runtime: *what do we have*, *which of it
serves this task and language*, and *is each one actually loadable right now*.
The last is the ``ModelStatus`` roll-up of the other two packages - installation
(files on disk), integrity (digests match) and approval (licence gate) - so a
caller can decide to route, fall back, or report the limitation from one object.

Load failures are collected rather than raised: a single malformed manifest in a
curated directory must not blind the registry to the other forty models.

A registry also has a :class:`~nativevoice.models.api.AssetScope`, and the scope
tag on each manifest is checked at the one place every manifest passes through -
indexing. A measurement-scoped manifest is therefore not merely hidden from a
product registry, it is never in its index, so there is nothing for a later
caller to route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import CapabilityUnavailable, ModelNotApproved
from ..core.language import Task
from .api import AssetScope, ModelManifest, ModelStatus
from .approval import approval_blockers, scope_blockers
from .integrity import inspect as inspect_integrity
from .manifest_io import load_manifest


@dataclass(slots=True)
class RegistryIssue:
    """A manifest file that could not be read, kept separate from valid ones."""

    path: str
    reason: str


@dataclass(slots=True)
class ModelRegistry:
    root: Path
    scope: AssetScope = AssetScope.PRODUCT
    manifests: dict[str, ModelManifest] = field(default_factory=dict)
    _by_language: dict[tuple[Task, str], list[str]] = field(default_factory=dict)
    issues: list[RegistryIssue] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._discover()

    def _discover(self) -> None:
        self.manifests.clear()
        self._by_language.clear()
        self.issues.clear()
        if not self.root.is_dir():
            return
        # `<root>/*.json` and `<root>/manifests/*.json`: provisioning writes the
        # second, hand-written manifests live in the first, and discovering only one
        # of them reports an empty registry rather than an error - the worst failure
        # mode, because everything downstream then says "no model installed".
        for path in sorted(self.root.glob("manifests/*.json")) + sorted(self.root.glob("*.json")):
            self._ingest_path(path)

    def _ingest_path(self, path: Path) -> ModelManifest | None:
        try:
            manifest = load_manifest(path)
            return self._index(manifest)
        except Exception as exc:  # a bad file is reported, never fatal to the scan
            self.issues.append(RegistryIssue(path=str(path), reason=str(exc)))
            return None

    def _index(self, manifest: ModelManifest) -> ModelManifest:
        blockers = scope_blockers(manifest, self.scope)
        if blockers:
            raise ModelNotApproved(
                "model is not indexed by this registry",
                model_id=manifest.model_id,
                reasons=blockers,
                registry_scope=self.scope.value,
            )
        self.manifests[manifest.model_id] = manifest
        for language in manifest.languages:
            self._by_language.setdefault((manifest.task, language), []).append(manifest.model_id)
        return manifest

    def register(self, manifest: ModelManifest) -> ModelManifest:
        """Add or replace a manifest in the index without touching disk.

        Scope-checked like discovery: this is the path a hand-built manifest takes,
        so refusing here is what makes the isolation a property of the index rather
        than of the files on disk.
        """
        return self._index(manifest)

    def reload(self) -> None:
        self._discover()

    def all(self) -> tuple[ModelManifest, ...]:
        return tuple(self.manifests[k] for k in sorted(self.manifests))

    def get(self, model_id: str) -> ModelManifest:
        try:
            return self.manifests[model_id]
        except KeyError:
            raise CapabilityUnavailable(
                "no such model in registry", model_id=model_id, root=str(self.root)
            ) from None

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.manifests))

    def for_task(self, task: Task) -> tuple[ModelManifest, ...]:
        return tuple(m for m in self.all() if m.task is task)

    def for_language(self, task: Task, language: str) -> tuple[ModelManifest, ...]:
        ids = self._by_language.get((task, language), [])
        return tuple(self.manifests[i] for i in sorted(ids))

    def model_ids_for(self, task: Task, language: str) -> tuple[str, ...]:
        return tuple(m.model_id for m in self.for_language(task, language))

    def status(self, model_id: str) -> ModelStatus:
        """Installed + integrity + approval rolled up for one model."""
        manifest = self.get(model_id)
        return self._status_for(manifest)

    def _status_for(self, manifest: ModelManifest) -> ModelStatus:
        report = inspect_integrity(manifest, self.root)
        installed = bool(manifest.assets) and not report.missing
        verified = report.ok
        blockers = approval_blockers(manifest)
        reasons: list[str] = list(report.reasons)
        if blockers:
            reasons.extend(f"not approved: {b}" for b in blockers)
        usable = installed and verified and not blockers
        return ModelStatus(
            manifest=manifest,
            installed=installed,
            verified=verified,
            usable=usable,
            reasons=tuple(reasons),
        )

    def status_all(self) -> dict[str, ModelStatus]:
        return {model_id: self._status_for(self.manifests[model_id]) for model_id in sorted(self.manifests)}

    def usable(self, task: Task | None = None, language: str | None = None) -> tuple[ModelStatus, ...]:
        result: list[ModelStatus] = []
        for model_id in sorted(self.manifests):
            manifest = self.manifests[model_id]
            if task is not None and manifest.task is not task:
                continue
            if language is not None and language not in manifest.languages:
                continue
            status = self._status_for(manifest)
            if status.usable:
                result.append(status)
        return tuple(result)
