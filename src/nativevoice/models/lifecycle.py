"""Load / unload lifecycle for the model registry.

``ModelManager`` turns a registry's *declarations* into running engine objects
under three constraints the specification insists on:

* a model is never handed to an engine until it is installed, integrity-verified
  and approved, each with its own error so the remedy is unambiguous;
* a failed load leaves no half-loaded state - either the engine object exists and
  is tracked, or nothing was recorded and the failure is visible via ``status``;
* the loaded set stays inside a per-task concurrency cap and a resource budget,
  evicting the least-recently-used idle model when a new one is required.

The engine itself is supplied as a factory, so the entire lifecycle - approval,
integrity, budget, eviction, teardown - is testable with a stub and no neural
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..core.errors import (
    EngineUnavailable,
    IntegrityError,
    ModelNotInstalled,
    ResourceLimitExceeded,
)
from .api import ModelManifest, ModelStatus
from .approval import ApprovalGate
from .integrity import inspect as inspect_integrity
from .registry import ModelRegistry

EngineFactory = Callable[[ModelManifest, object], object]


@dataclass(slots=True)
class _LoadedRecord:
    model_id: str
    task: object
    engine: object
    used_at: int


@dataclass(slots=True)
class ModelManager:
    registry: ModelRegistry
    engine_factory: EngineFactory
    budget: object | None = None
    max_per_task: int = 1
    approvals: ApprovalGate = field(default_factory=ApprovalGate)

    _loaded: dict[str, _LoadedRecord] = field(default_factory=dict)
    _last_errors: dict[str, tuple[str, ...]] = field(default_factory=dict)
    _clock: int = 0

    def __post_init__(self) -> None:
        if self.max_per_task < 1:
            raise ResourceLimitExceeded(
                "per-task concurrency cap must be at least 1",
                max_per_task=self.max_per_task,
            )

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    @property
    def loaded_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaded))

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def loaded_for_task(self, task: object) -> tuple[str, ...]:
        return tuple(sorted(m for m, r in self._loaded.items() if r.task == task))

    def engine(self, model_id: str) -> object:
        record = self._loaded.get(model_id)
        if record is None:
            raise EngineUnavailable("model is not loaded", model_id=model_id)
        record.used_at = self._tick()
        return record.engine

    def _gate(self, manifest: ModelManifest) -> None:
        self.approvals.check(manifest)

    def _require_installed_and_verified(self, manifest: ModelManifest) -> None:
        report = inspect_integrity(manifest, self.registry.root)
        if report.missing:
            raise ModelNotInstalled(
                "model assets are missing on disk",
                model_id=manifest.model_id,
                missing=report.missing,
                reasons=report.reasons,
            )
        if not report.ok:
            raise IntegrityError(
                "model assets failed integrity verification",
                model_id=manifest.model_id,
                reasons=report.reasons,
            )

    def _require_budget(self, manifest: ModelManifest) -> None:
        if self.budget is None:
            return
        verdict = self.budget.fits(manifest.requirements)
        if not verdict.ok:
            raise ResourceLimitExceeded(
                "model does not fit the active resource budget",
                explanation=getattr(verdict, "explanation", ""),
                model_id=manifest.model_id,
                reasons=getattr(verdict, "reasons", ()),
            )

    def _make_room(self, manifest: ModelManifest) -> None:
        same_task = self.loaded_for_task(manifest.task)
        if manifest.model_id in self._loaded or len(same_task) < self.max_per_task:
            return
        evictable = sorted(same_task, key=lambda m: self._loaded[m].used_at)
        if evictable:
            self.unload(evictable[0])

    def load(self, model_id: str) -> object:
        manifest = self.registry.get(model_id)
        if model_id in self._loaded:
            self._loaded[model_id].used_at = self._tick()
            return self._loaded[model_id].engine
        self._last_errors.pop(model_id, None)
        try:
            self._gate(manifest)
            self._require_installed_and_verified(manifest)
            self._require_budget(manifest)
            self._make_room(manifest)
            engine = self.engine_factory(manifest, self.budget)
        except Exception as exc:
            self._loaded.pop(model_id, None)
            self._last_errors[model_id] = (str(exc),)
            raise
        self._loaded[model_id] = _LoadedRecord(model_id, manifest.task, engine, self._tick())
        self._last_errors.pop(model_id, None)
        return engine

    def unload(self, model_id: str) -> bool:
        record = self._loaded.pop(model_id, None)
        if record is None:
            return False
        closer = getattr(record.engine, "close", None)
        if callable(closer):
            closer()
        return True

    def evict_idle(self) -> str | None:
        """Unload the least-recently-used loaded model. Returns its id or None."""
        if not self._loaded:
            return None
        victim = min(self._loaded.values(), key=lambda r: r.used_at)
        self.unload(victim.model_id)
        return victim.model_id

    def status(self, model_id: str) -> ModelStatus:
        manifest = self.registry.get(model_id)
        base = self.registry.status(model_id)
        reasons = list(base.reasons)
        if model_id in self._loaded:
            reasons.append("loaded")
            return ModelStatus(
                manifest=manifest,
                installed=True,
                verified=base.verified,
                usable=True,
                reasons=tuple(reasons),
            )
        error = self._last_errors.get(model_id)
        if error:
            failure = tuple(f"load failed: {r}" for r in error)
            return ModelStatus(
                manifest=manifest,
                installed=base.installed,
                verified=base.verified,
                usable=False,
                reasons=failure,
            )
        return ModelStatus(
            manifest=manifest,
            installed=base.installed,
            verified=base.verified,
            usable=base.usable,
            reasons=tuple(reasons),
        )

    def close(self) -> None:
        """Deterministic teardown: unload everything in sorted id order."""
        for model_id in sorted(self._loaded):
            self.unload(model_id)
        self._loaded.clear()
