"""Installed-engine catalogue with capability-ledger-aware selection.

Zero engines exist at import time: an engine only becomes routable once something
explicitly registers it together with the capabilities it truly declares. Selection
then honours the same ledger semantics as routing - a ``DECLARED`` capability only
satisfies ``best_for`` when ``strict=False``, and a ``NO_MODEL``/``UNSUPPORTED``
language matches nothing. Crucially, ``best_for`` returns a capability's engine
only when that engine is actually installed, so a claimed-but-missing adapter
surfaces as "no engine" rather than an attempt to call something absent.
"""

from __future__ import annotations

from dataclasses import replace

from ..core.errors import ConfigurationError
from ..core.language import Capability, CapabilityStatus, LanguageTag, Task
from ..language.registry import CapabilityLedger
from .api import TranscriptionEngine


def _code_of(language: str | LanguageTag | None) -> str | None:
    if language is None:
        return None
    if isinstance(language, LanguageTag):
        return language.code
    return language.split("-")[0].split("_")[0].lower()


class EngineRegistry:
    """Maps engine instances to capabilities and answers "who can do this?"."""

    def __init__(self, ledger: CapabilityLedger | None = None) -> None:
        self._engines: dict[str, TranscriptionEngine] = {}
        self.ledger = ledger if ledger is not None else CapabilityLedger()

    def register(self, engine: TranscriptionEngine, capabilities=None) -> TranscriptionEngine:
        engine_id = getattr(engine, "engine_id", "") or ""
        if not engine_id:
            raise ConfigurationError("engine has no engine_id", engine=repr(engine))
        if engine_id in self._engines:
            raise ConfigurationError("engine already registered", engine=engine_id)
        caps = tuple(capabilities) if capabilities is not None else tuple(engine.capabilities())
        self._engines[engine_id] = engine
        for capability in caps:
            stamped = capability if capability.engine == engine_id else replace(capability, engine=engine_id)
            self.ledger.record(stamped)
        return engine

    def unregister(self, engine_id: str) -> None:
        self._engines.pop(engine_id, None)

    def close_all(self) -> None:
        for engine in list(self._engines.values()):
            closer = getattr(engine, "close", None)
            if callable(closer):
                closer()

    def has_engines(self) -> bool:
        return bool(self._engines)

    def get(self, engine_id: str) -> TranscriptionEngine | None:
        return self._engines.get(engine_id)

    def engine_ids(self) -> list[str]:
        return sorted(self._engines)

    def for_task(self, task: Task = Task.STT) -> list[TranscriptionEngine]:
        """Installed engines that have at least one recorded capability for the task."""
        task_value = task.value if isinstance(task, Task) else str(task)
        result: list[TranscriptionEngine] = []
        for engine_id in sorted(self._engines):
            for code in self.ledger.recorded_languages(task_value):
                if self.ledger.get(task_value, code, engine_id) is not None:
                    result.append(self._engines[engine_id])
                    break
        return result

    def best_for(self, language, strict: bool = True, task: Task = Task.STT):
        """Return ``(engine, capability)`` or ``None``.

        ``strict`` maps to the ledger's usability policy: strict means a
        ``DECLARED``-only capability is not good enough.
        """
        code = _code_of(language)
        if code is None:
            return self.best_any(strict=strict, task=task)
        capability = self.ledger.usable(task, code, allow_declared=not strict)
        if capability is None:
            return None
        engine = self._engines.get(capability.engine)
        if engine is None:
            return None
        return engine, capability

    def best_any(self, strict: bool = True, task: Task = Task.STT):
        """Broadest capable installed engine, for auto-detection requests.

        Chosen by number of usable capabilities (a multilingual engine wins),
        tie-broken by engine id for determinism.
        """
        allow_declared = not strict
        best_engine = None
        best_cap: Capability | None = None
        best_count = -1
        for engine_id in sorted(self._engines):
            count = 0
            representative: Capability | None = None
            for code in self.ledger.recorded_languages(task):
                cap = self.ledger.get(task, code, engine_id)
                if cap is None:
                    continue
                usable = cap.status in (CapabilityStatus.MEASURED, CapabilityStatus.PARTIAL) or (
                    allow_declared and cap.status is CapabilityStatus.DECLARED
                )
                if usable:
                    count += 1
                    if representative is None:
                        representative = cap
            if representative is not None and count > best_count:
                best_count = count
                best_engine = self._engines[engine_id]
                best_cap = representative
        if best_engine is None or best_cap is None:
            return None
        return best_engine, best_cap

    def describe(self) -> str:
        """Human-readable "installed vs missing" report."""
        lines = ["Engine registry:"]
        if not self._engines:
            lines.append("  (no engines installed - every request raises EngineUnavailable)")
        for engine_id in sorted(self._engines):
            engine = self._engines[engine_id]
            streaming = getattr(engine, "supports_streaming", False)
            model = getattr(engine, "model_id", "") or "(no model id)"
            covered = sorted(task.value for task in Task if self._engine_covers(task, engine_id))
            lines.append(
                f"  installed: {engine_id} model={model} streaming={streaming} "
                f"tasks={','.join(covered) or '-'}"
            )
        for task in Task:
            serving = len(self.for_task(task))
            note = "" if serving else " [MISSING: no installed engine for this task]"
            lines.append(f"  task {task.value}: {serving} engine(s){note}")
        lines.append(
            "  optional adapters (faster-whisper) are NOT auto-registered and are "
            "unreviewed; they must be installed and explicitly approved."
        )
        return "\n".join(lines)

    def _engine_covers(self, task: Task, engine_id: str) -> bool:
        task_value = task.value if isinstance(task, Task) else str(task)
        return engine_id in self.ledger.engines_for(task_value)

