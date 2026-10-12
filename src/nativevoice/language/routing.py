"""Which engine may handle an utterance, and when it must not be invoked.

Routing is where the honesty invariant becomes operational. A language with no
usable capability must never reach an engine - not because the engine would fail
loudly, but because a multilingual model will happily *succeed* at transcribing
Bengali into something wrong. So the router's job is as much about refusing as
about selecting:

- An explicit user language wins, but only if some engine genuinely has usable
  capability for it; otherwise we return ``unavailable`` with the best fallback
  and never invoke an unsupported engine.
- Automatic detection is gated by a confidence threshold that rises as the audio
  gets shorter, because a two-second clip carries less evidence than a six-second
  one (see the reference accuracy curve). Below threshold we return ``ambiguous``
  and expect the caller to ask rather than emit unrelated speech.
- A DECLARED-only capability is not sufficient in strict mode.
- Two scripts both over the mixture threshold is a real code-switch; we name the
  dominant and secondary languages rather than pick one silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..core.language import (
    Capability,
    CapabilityStatus,
    LanguageDecision,
    LanguageTag,
    Task,
)
from .lid import LexiconLanguageIdentifier, LanguageIdentifier
from .registry import CapabilityLedger, LanguageRegistry


class RoutingKind(str, Enum):
    ROUTED = "routed"
    UNAVAILABLE = "unavailable"
    AMBIGUOUS = "ambiguous"
    CODE_SWITCH = "code_switch"


@dataclass(frozen=True, slots=True)
class RoutingOutcome:
    kind: RoutingKind
    language: LanguageTag | None = None
    secondary_language: LanguageTag | None = None
    capability: Capability | None = None
    decision: LanguageDecision | None = None
    confidence: float = 0.0
    reason: str = ""
    fallback: LanguageTag | None = None
    threshold: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def engine_id(self) -> str | None:
        return self.capability.engine if self.capability else None

    @property
    def may_invoke_engine(self) -> bool:
        return self.kind is RoutingKind.ROUTED and self.capability is not None


def _step_threshold(duration_s: float) -> float:
    """Required detection confidence as a function of available evidence.

    Short utterances carry less evidence, so we demand more before acting. The
    reference classifier's own accuracy is ~52% at 2 s, ~78% at 3 s, ~93% at 6 s;
    a threshold that climbs as duration falls is the honest response to that.
    """
    if duration_s < 1.0:
        return 0.90
    if duration_s < 2.0:
        return 0.80
    if duration_s < 3.0:
        return 0.72
    if duration_s < 6.0:
        return 0.65
    return 0.60


class LanguageRouter:
    def __init__(
        self,
        registry: LanguageRegistry | None = None,
        identifier: LanguageIdentifier | None = None,
        *,
        mixture_threshold: float = 0.15,
        threshold_fn=None,
    ) -> None:
        self.registry = registry or LanguageRegistry()
        self.identifier = identifier or LexiconLanguageIdentifier(self.registry)
        self.mixture_threshold = mixture_threshold
        self._threshold_fn = threshold_fn or _step_threshold

    def threshold_for(self, duration_s: float) -> float:
        return self._threshold_fn(duration_s)

    def route(
        self,
        *,
        ledger: CapabilityLedger,
        requested_language: LanguageTag | None = None,
        spoken_text: str | None = None,
        audio_duration_s: float = 0.0,
        allow_declared: bool = False,
        allow_auto_detection: bool = True,
        task: Task = Task.STT,
    ) -> RoutingOutcome:
        if requested_language is not None:
            return self._route_explicit(ledger, requested_language, allow_declared, task)

        text = (spoken_text or "").strip()
        if text:
            return self._route_from_text(
                ledger, text, audio_duration_s, allow_declared, task
            )

        if not allow_auto_detection:
            return RoutingOutcome(
                kind=RoutingKind.AMBIGUOUS,
                reason="no requested language and automatic detection is disabled",
            )

        code, cap = self._best_supported_language(ledger, task, allow_declared)
        if code is None or cap is None:
            return RoutingOutcome(
                kind=RoutingKind.UNAVAILABLE,
                reason="no engine has usable capability for any language; cannot auto-detect",
            )
        # No text yet, so we cannot name a language; language stays None and the
        # selected engine performs detection itself. `code` only proves that some
        # engine with real capability exists.
        return RoutingOutcome(
            kind=RoutingKind.ROUTED,
            language=None,
            capability=cap,
            confidence=1.0,
            reason="engine-side auto-detection deferred to the selected multilingual engine",
            metadata={"auto_detect": True, "candidate_language": code},
        )

    def _route_explicit(
        self, ledger: CapabilityLedger, language: LanguageTag, allow_declared: bool, task: Task
    ) -> RoutingOutcome:
        code = language.code
        cap = ledger.usable(task, code, allow_declared=allow_declared)
        if cap is not None:
            return RoutingOutcome(
                kind=RoutingKind.ROUTED,
                language=LanguageTag(code),
                capability=cap,
                confidence=1.0,
                reason=f"explicit language {code} has usable capability ({cap.status.value})",
            )
        status = ledger.best(task, code).status.value
        fallback_language, _ = self._best_supported_language(
            ledger, task, allow_declared, exclude=code
        )
        return RoutingOutcome(
            kind=RoutingKind.UNAVAILABLE,
            language=LanguageTag(code),
            reason=f"no usable engine capability for '{code}' (best status: {status})",
            fallback=LanguageTag(fallback_language) if fallback_language else None,
            capability=None,
            metadata={"best_status": status, "strict": not allow_declared},
        )

    def _route_from_text(
        self,
        ledger: CapabilityLedger,
        text: str,
        duration_s: float,
        allow_declared: bool,
        task: Task,
    ) -> RoutingOutcome:
        from .scripts import analyze

        evidence = analyze(text)
        decision = self.identifier.identify(text)
        mixtures = evidence.mixtures(self.mixture_threshold)

        if len(mixtures) >= 2:
            dominant_script = mixtures[0][0]
            secondary_script = mixtures[1][0]
            dominant, dominant_fraction = self._dominant_pair(mixtures[0], text, allow_declared, task, ledger)
            secondary, _ = self._language_for_script(ledger, secondary_script, allow_declared, task)
            return RoutingOutcome(
                kind=RoutingKind.CODE_SWITCH,
                language=dominant,
                secondary_language=secondary,
                decision=decision,
                confidence=decision.confidence,
                reason=(
                    f"mixed scripts {dominant_script} ({dominant_fraction:.0%}) and "
                    f"{secondary_script}; named dominant and secondary rather than guessing"
                ),
                metadata={"dominant_script": dominant_script, "secondary_script": secondary_script},
            )

        threshold = self.threshold_for(duration_s)
        if decision.below(threshold):
            return RoutingOutcome(
                kind=RoutingKind.AMBIGUOUS,
                language=decision.language,
                decision=decision,
                confidence=decision.confidence,
                threshold=threshold,
                reason=(
                    f"detection confidence {decision.confidence:.2f} is below the "
                    f"{threshold:.2f} threshold required for {duration_s:.1f}s of audio; "
                    "ask the caller instead of emitting unrelated speech"
                ),
            )

        code = decision.language.code
        cap = ledger.usable(task, code, allow_declared=allow_declared)
        if cap is None:
            fallback_language, _ = self._best_supported_language(ledger, task, allow_declared, exclude=code)
            return RoutingOutcome(
                kind=RoutingKind.UNAVAILABLE,
                language=decision.language,
                decision=decision,
                confidence=decision.confidence,
                reason=f"detected '{code}' but no engine has usable capability for it",
                fallback=LanguageTag(fallback_language) if fallback_language else None,
            )
        return RoutingOutcome(
            kind=RoutingKind.ROUTED,
            language=decision.language,
            capability=cap,
            decision=decision,
            confidence=decision.confidence,
            threshold=threshold,
            reason=f"auto-detected {code} above threshold {threshold:.2f}",
        )

    def _dominant_pair(self, pair, text, allow_declared, task, ledger):
        script, fraction = pair
        language, _cap = self._language_for_script(ledger, script, allow_declared, task)
        if language is None:
            detected = self.identifier.identify(text)
            language = detected.language
        return language, fraction

    def _language_for_script(
        self, ledger: CapabilityLedger, script: str, allow_declared: bool, task: Task
    ) -> tuple[LanguageTag | None, Capability | None]:
        """Name a language from its script, preferring one with usable capability."""
        candidates = self.registry.codes_for_script(script)
        for code in candidates:
            cap = ledger.usable(task, code, allow_declared=allow_declared)
            if cap is not None:
                return LanguageTag(code), cap
        if candidates:
            return LanguageTag(candidates[0]), None
        return None, None

    def _best_supported_language(
        self,
        ledger: CapabilityLedger,
        task: Task,
        allow_declared: bool,
        exclude: str | None = None,
    ) -> tuple[str | None, Capability | None]:
        """Highest-status language with usable capability; English preferred on ties."""
        best: tuple[str, Capability] | None = None
        rank = _RANK
        for code in ledger.recorded_languages(task):
            if code == exclude:
                continue
            cap = ledger.usable(task, code, allow_declared=allow_declared)
            if cap is None:
                continue
            if best is None:
                best = (code, cap)
                continue
            current_rank = rank[cap.status]
            best_rank = rank[best[1].status]
            if current_rank > best_rank or (
                current_rank == best_rank and code == "en" and best[0] != "en"
            ):
                best = (code, cap)
        if best is None:
            return None, None
        return best


_RANK = {
    CapabilityStatus.MEASURED: 2,
    CapabilityStatus.PARTIAL: 1,
    CapabilityStatus.DECLARED: 0,
}
