"""Synthesis engine registry: three-state routing with no ambient trust.

Routing is the place where a hopeful registry does damage: an engine that
"supports Hindi" because its model card says so gets used, and the user hears
English-flavoured garbage reported as success. So selection is decided by the
status an engine declares for itself, per language and per task, and `strict`
controls whether an unverified vendor claim is enough.

Precedence when nothing is routable is chosen to give the *most actionable* error:
`ModelNotInstalled` (an asset is recognised but absent) beats
`LanguageUnsupported` (an engine exists and says it cannot) beats
`EngineUnavailable` (nothing is registered at all). Each one has a different
remedy, so reporting the wrong one wastes the user's time.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import (
    EngineUnavailable,
    LanguageUnsupported,
    ModelNotInstalled,
)
from ..core.language import Capability, CapabilityStatus, LanguageTag, Task
from .api import SynthesisEngine

_ROUTABLE = (CapabilityStatus.MEASURED, CapabilityStatus.PARTIAL)
_PREFERENCE = {
    CapabilityStatus.MEASURED: 0,
    CapabilityStatus.PARTIAL: 1,
    CapabilityStatus.DECLARED: 2,
    CapabilityStatus.NO_MODEL: 3,
    CapabilityStatus.UNSUPPORTED: 4,
}


@dataclass(frozen=True, slots=True)
class EngineDescription:
    """Human-readable summary of one registered engine; safe to print in a UI."""

    engine_id: str
    model_id: str
    streaming: bool
    controllable: tuple[str, ...]
    languages: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict:
        return {
            "engine_id": self.engine_id,
            "model_id": self.model_id,
            "streaming": self.streaming,
            "controllable": list(self.controllable),
            "languages": [{"language": code, "status": status} for code, status in self.languages],
        }


class EngineRegistry:
    """Registration and selection of synthesis engines.

    Empty at construction: this module never pre-registers an engine, so nothing
    can be routed to before a manifest has been reviewed and an adapter installed.
    """

    def __init__(self) -> None:
        self._engines: dict[str, SynthesisEngine] = {}

    def __len__(self) -> int:
        return len(self._engines)

    def register(self, engine: SynthesisEngine) -> str:
        if not getattr(engine, "engine_id", ""):
            raise EngineUnavailable("an engine must declare an engine_id before registration")
        engine_id = engine.engine_id
        if engine_id in self._engines:
            raise EngineUnavailable("engine_id already registered", engine_id=engine_id)
        self._engines[engine_id] = engine
        return engine_id

    def unregister(self, engine_id: str, *, close: bool = True) -> bool:
        engine = self._engines.pop(engine_id, None)
        if engine is None:
            return False
        if close:
            engine.close()
        return True

    def get(self, engine_id: str) -> SynthesisEngine:
        try:
            return self._engines[engine_id]
        except KeyError:
            raise EngineUnavailable("no engine registered under that id", engine_id=engine_id) from None

    def engines(self) -> tuple[SynthesisEngine, ...]:
        return tuple(self._engines.values())

    def capability(self, language: LanguageTag | str, task: Task = Task.TTS) -> Capability:
        """Best declared capability across engines, or an explicit empty answer."""
        code = self._code(language)
        matches = [
            capability
            for engine in self._engines.values()
            for capability in engine.capabilities()
            if capability.task is task and capability.language == code
        ]
        if not matches:
            return Capability(
                task=task,
                status=CapabilityStatus.NO_MODEL,
                language=code,
                engine="",
                limitations=(
                    "no registered engine declares this language for this task",
                    "engine registry is empty" if not self._engines else "language not declared",
                ),
                evidence="capability answers come from engine declarations, never from a default",
            )
        return self._best_capability(matches)

    def best_for(
        self,
        language: LanguageTag | str,
        task: Task = Task.TTS,
        strict: bool = True,
    ) -> SynthesisEngine:
        """The engine to use, or the specific reason there isn't one."""
        code = self._code(language)
        accepted = _ROUTABLE + ((CapabilityStatus.DECLARED,) if not strict else ())
        scored: list[tuple[int, str, SynthesisEngine]] = []
        absent = False
        refused = False
        claimed_by: list[str] = []
        for engine in self._engines.values():
            for capability in engine.capabilities():
                if capability.task is not task or capability.language != code:
                    continue
                if capability.status in accepted:
                    scored.append((_PREFERENCE[capability.status], engine.engine_id, engine))
                elif capability.status is CapabilityStatus.NO_MODEL:
                    absent = True
                elif capability.status is CapabilityStatus.UNSUPPORTED:
                    refused = True
                elif capability.status is CapabilityStatus.DECLARED:
                    claimed_by.append(engine.engine_id)
        if scored:
            scored.sort(key=lambda item: (item[0], item[1]))
            return scored[0][2]
        if claimed_by:
            raise EngineUnavailable(
                "engines claim this language only on their model card, which strict routing will not act on",
                language=code,
                task=task.value,
                unverified_engines=tuple(claimed_by),
                remedy="measure the language or route with strict=False and warn the user",
            )
        if absent:
            raise ModelNotInstalled(
                "the only engine for this language has no installed model",
                language=code,
                task=task.value,
                remedy="install and approve a model asset for this language",
            )
        if refused:
            raise LanguageUnsupported(
                "the registered engines cannot synthesise this language",
                language=code,
                task=task.value,
                engines=tuple(self._engines),
            )
        if not self._engines:
            raise EngineUnavailable(
                "no synthesis engine is installed",
                language=code,
                task=task.value,
                remedy="install a synthesis adapter with an approved model asset",
            )
        raise EngineUnavailable(
            "no registered engine declares this language",
            language=code,
            task=task.value,
            engines=tuple(self._engines),
        )

    def describe(self) -> tuple[EngineDescription, ...]:
        out = []
        for engine in self._engines.values():
            pairs = {
                (capability.language, capability.status.value)
                for capability in engine.capabilities()
                if capability.task is Task.TTS
            }
            out.append(
                EngineDescription(
                    engine_id=engine.engine_id,
                    model_id=engine.model_id,
                    streaming=engine.supports_streaming,
                    controllable=tuple(engine.controllable),
                    languages=tuple(sorted(pairs)),
                )
            )
        return tuple(sorted(out, key=lambda item: item.engine_id))

    @staticmethod
    def _code(language: LanguageTag | str) -> str:
        tag = language if isinstance(language, LanguageTag) else LanguageTag(language)
        return tag.root().code

    @staticmethod
    def _best_capability(matches: list[Capability]) -> Capability:
        return min(matches, key=lambda capability: _PREFERENCE.get(capability.status, 9))
