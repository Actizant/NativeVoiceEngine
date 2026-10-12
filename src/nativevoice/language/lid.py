"""Text-domain language identification.

The acoustic story is sobering: the reference cloud classifier reports ~52% at
two seconds and ~93% at six seconds over 33 languages, and its own documentation
says the number is "not a calibrated probability". We do not pretend otherwise.
This module works on *text* (an ASR hypothesis, a transcript, a typed query)
where writing script is an exact, offline signal, and it never claims a score is
a probability: every decision is emitted with ``is_calibrated = False`` and a
normalised, explicitly-labelled *relative* score.

The design is layered on top of :mod:`nativevoice.language.scripts`:
- ``ScriptOnlyIdentifier`` when a script has no lexicon (Greek, Thai, ...).
- ``LexiconLanguageIdentifier`` which first narrows candidates by the dominant
  script, then scores them with a stopword hit ratio plus a character-ngram
  overlap, so Hindi vs Marathi (both Devanagari) separates on word choice.
- ``TransliteratedTextRisk`` for the genuinely hard case: Hindi and Urdu written
  in Latin letters collide with English. We return a low-confidence decision
  with an explicit note instead of guessing silently.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..core.language import LanguageDecision, LanguageTag
from . import corpora
from .registry import LanguageRegistry
from .scripts import UNKNOWN_SCRIPT, analyze

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
UND_TAG = LanguageTag("und")


class LanguageIdentifier(Protocol):
    """Anything that maps written text to a language decision."""

    def identify(self, text: str) -> LanguageDecision:
        ...


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens; used for space-delimited stopword matching."""
    return [token.lower() for token in _WORD_RE.findall(text)]


def _char_ngrams(text: str, sizes: tuple[int, ...] = (2, 3)) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    grams: set[str] = set()
    for size in sizes:
        for index in range(0, max(0, len(compact) - size + 1)):
            grams.add(compact[index : index + size])
    return grams


class _Lexicon:
    """Precomputed scoring tables so ``identify`` stays cheap."""

    def __init__(self, registry: LanguageRegistry | None = None) -> None:
        self.registry = registry or LanguageRegistry()
        self._stopwords: dict[str, set[str]] = {}
        self._grams: dict[str, set[str]] = {}
        self._space_delimited: dict[str, bool] = {}
        for code, words in corpora.STOPWORDS.items():
            lowered = {word.lower() if code == "en" else word for word in words}
            self._stopwords[code] = lowered
            seed_text = "".join(words) + "".join(corpora.CHAR_NGRAM_SEED.get(code, ()))
            self._grams[code] = _char_ngrams(seed_text, sizes=(1, 2, 3))
            profile = self.registry.lookup(code) if self.registry.known(code) else None
            self._space_delimited[code] = bool(profile.space_delimited) if profile else True

    def candidates(self, script: str) -> list[str]:
        """Codes in the corpus that are written in ``script``."""
        return [code for code in self._stopwords if self._script_of(code) == script]

    def _script_of(self, code: str) -> str:
        if self.registry.known(code):
            return self.registry.lookup(code).script
        return corpora.LEXICON_SCRIPTS.get(code, UNKNOWN_SCRIPT)

    def score(self, text: str, tokens: list[str], grams: set[str], code: str) -> float:
        words = self._stopwords[code]
        if self._space_delimited[code]:
            hits = sum(1 for token in tokens if token in words)
            ratio = hits / len(tokens) if tokens else 0.0
        else:
            present = sum(1 for word in words if word in text)
            ratio = present / len(words) if words else 0.0
        overlap = len(grams & self._grams[code]) / len(grams) if grams else 0.0
        return 2.0 * ratio + overlap


WORD_EVIDENCE_FLOOR = 0.4


class LexiconLanguageIdentifier:
    """Stopword + character-ngram scoring, gated by the dominant script.

    Confidence is a normalised *relative* score, never a probability; every
    decision is returned with ``is_calibrated = False``.
    """

    method = "lexicon-ngram+script"

    def __init__(self, registry: LanguageRegistry | None = None) -> None:
        self._lexicon = _Lexicon(registry)
        self.registry = self._lexicon.registry
        self.transliteration = TransliteratedTextRisk(self.registry)

    def identify(self, text: str) -> LanguageDecision:
        evidence = analyze(text)
        if evidence.total_letters == 0:
            return LanguageDecision(
                language=UND_TAG,
                confidence=0.0,
                method=self.method,
                is_calibrated=False,
                metadata={"reason": "no alphabetic content"},
            )

        script = evidence.dominant
        tokens = tokenize(text)
        grams = _char_ngrams(text)

        risk = self.transliteration.assess(text) if script == "Latin" else None

        candidates = self._lexicon.candidates(script)
        if not candidates:
            return ScriptOnlyIdentifier(self.registry).identify(text)

        scored = sorted(
            ((code, self._lexicon.score(text, tokens, grams, code)) for code in candidates),
            key=lambda pair: -pair[1],
        )
        if risk is not None and scored[0][1] < WORD_EVIDENCE_FLOOR:
            # Romanised Hindi/Urdu shares its alphabet with English. A Latin-script
            # winner with no word-level support is a letter pattern, not evidence
            # about a language, so the ambiguity is reported instead of resolved.
            return risk
        decision = self._decision(scored[0][0], evidence, scored)
        if risk is not None:
            decision.confidence = min(decision.confidence, 0.6)
            decision.metadata["transliteration_risk"] = risk.metadata.get("note", "")
            decision.metadata["alternatives"] = list(risk.alternatives)
        return decision

    def _decision(
        self,
        best_code: str,
        evidence,
        scored: list[tuple[str, float]],
    ) -> LanguageDecision:
        top = scored[0][1] if scored else 0.0
        second = scored[1][1] if len(scored) > 1 else 0.0
        coverage = evidence.coverage
        if top <= 0:
            # Nothing matched lexically; the script is all we know.
            confidence = 0.35 * coverage
        else:
            margin = top / (top + second) if (top + second) else 1.0
            confidence = 0.5 * coverage + 0.5 * margin
        # A relative score is not a probability; withholding 1.0 keeps a caller from
        # reading "certain" out of a number that only means "beat the runner-up".
        confidence = max(0.0, min(0.95, confidence))
        alternatives = tuple(
            (code, round(score / top, 4) if top else 0.0) for code, score in scored[1:]
        )
        return LanguageDecision(
            language=LanguageTag(best_code),
            confidence=confidence,
            alternatives=alternatives,
            method=self.method,
            is_calibrated=False,
            metadata={
                "script": evidence.dominant,
                "script_coverage": round(coverage, 4),
                "relative": True,
                "top_score": round(top, 4),
            },
        )


class ScriptOnlyIdentifier:
    """Identification from writing script alone, for scripts with no lexicon."""

    method = "script-only"

    def __init__(self, registry: LanguageRegistry | None = None) -> None:
        self.registry = registry or LanguageRegistry()

    def identify(self, text: str) -> LanguageDecision:
        evidence = analyze(text)
        if evidence.total_letters == 0:
            return LanguageDecision(
                language=UND_TAG,
                confidence=0.0,
                method=self.method,
                is_calibrated=False,
                metadata={"reason": "no alphabetic content"},
            )
        script = evidence.dominant
        candidates = self.registry.codes_for_script(script) if script != UNKNOWN_SCRIPT else []
        coverage = evidence.coverage
        if len(candidates) == 1:
            language = LanguageTag(candidates[0])
            # A script shared by exactly one registered language is strong evidence,
            # but it is inferred from writing system alone and cross-checked against
            # no lexicon, so it is not allowed to reach certainty.
            confidence = min(0.9, coverage)
            note = (
                f"language inferred from unique script {script}; lexicon unavailable "
                "and this is unverified"
            )
            alternatives: tuple[tuple[str, float], ...] = ()
        else:
            language = UND_TAG
            confidence = 0.5 * coverage
            note = f"script {script} maps to multiple candidate languages; not resolved"
            alternatives = tuple((code, round(coverage / max(1, len(candidates)), 4)) for code in candidates)
        return LanguageDecision(
            language=language,
            confidence=confidence,
            alternatives=alternatives,
            method=self.method,
            is_calibrated=False,
            metadata={"script": script, "script_coverage": round(coverage, 4), "note": note},
        )


class TransliteratedTextRisk:
    """Detect romanised Hindi/Urdu that would otherwise be mistaken for English.

    Hinglish and Roman Urdu are written in Latin letters, so a lexicon that only
    knows native scripts cannot separate them from English reliably. Rather than
    guess, this returns a deliberately low-confidence decision naming the risk.
    """

    def __init__(self, registry: LanguageRegistry | None = None) -> None:
        self.registry = registry or LanguageRegistry()
        self._urdu = set(corpora.ROMAN_URDU_MARKERS)
        self._hindi = set(corpora.ROMAN_HINDI_MARKERS)
        self._english = set(word.lower() for word in corpora.STOPWORDS["en"])

    def assess(self, text: str) -> LanguageDecision | None:
        evidence = analyze(text)
        if evidence.dominant != "Latin" or evidence.total_letters == 0:
            return None
        tokens = tokenize(text)
        hindi_hits = sum(1 for token in tokens if token in self._hindi)
        urdu_hits = sum(1 for token in tokens if token in self._urdu)
        english_hits = sum(1 for token in tokens if token in self._english)
        marker_hits = hindi_hits + urdu_hits
        # Only a genuine transliteration when Indic markers clearly dominate.
        if marker_hits < 2 or english_hits >= marker_hits:
            return None
        hi = round(hindi_hits / marker_hits, 4)
        ur = round(urdu_hits / marker_hits, 4)
        en = round(english_hits / marker_hits, 4)
        best = "hi" if hindi_hits >= urdu_hits else "ur"
        alternatives = (("hi", hi), ("ur", ur), ("en", en))
        confidence = min(0.35, 0.15 + 0.05 * marker_hits / max(1, len(tokens)))
        return LanguageDecision(
            language=LanguageTag(best),
            confidence=round(confidence, 4),
            alternatives=alternatives,
            method="lexicon-ngram+script",
            is_calibrated=False,
            metadata={
                "script": "Latin",
                "transliterated": True,
                "hindi_markers": hindi_hits,
                "urdu_markers": urdu_hits,
                "english_markers": english_hits,
                "note": (
                    "Romanised Hindi/Urdu (Hinglish/Roman Urdu): Latin script collides "
                    "with English and cannot be separated with confidence; caller should "
                    "ask rather than assume."
                ),
            },
        )
