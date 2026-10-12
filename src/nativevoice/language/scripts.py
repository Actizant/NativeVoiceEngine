"""Deterministic, offline Unicode-range script classification.

Script is the single most reliable signal for separating languages that share a
region but not an alphabet: Devanagari, Bengali and Latin text are trivially and
exactly distinguishable from each other, whereas an acoustic classifier on a two
second clip reports barely-above-chance accuracy. Because every character in the
supported scripts falls inside a compact set of Unicode blocks, this is pure
stdlib arithmetic with no model, no download and no ambiguity.

A "letter" here is any character for which ``str.isalpha`` is true. Everything
else (spaces, digits, punctuation, combining marks) is ignored by the counts, so
coverage fractions describe the alphabetic content of an utterance, which is
exactly what mixed-language detection needs.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# ISO 15924 title, mapped to the closed Unicode ranges that contain its letters.
# Order is not significant because Indic blocks are laid out in the same
# direction as their phonetic tables, so the codepoint ordering matches the
# block ordering; each codepoint is checked against every range regardless.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "Latin": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00D6), (0x00D8, 0x00F6),
              (0x00F8, 0x02AF), (0x1E00, 0x1EFF)),
    "Greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "Cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F)),
    "Armenian": ((0x0530, 0x058F), (0xFB13, 0xFB17)),
    "Hebrew": ((0x05D0, 0x05FF), (0xFB1D, 0xFB4F)),
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
               (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "Syriac": ((0x0700, 0x074F),),
    "Thaana": ((0x0780, 0x07BF),),
    "NKo": ((0x07C0, 0x07FF),),
    "Devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "Bengali": ((0x0980, 0x09FF),),
    "Gurmukhi": ((0x0A00, 0x0A7F),),
    "Gujarati": ((0x0A80, 0x0AFF),),
    "Odia": ((0x0B00, 0x0B7F),),
    "Tamil": ((0x0B80, 0x0BFF),),
    "Telugu": ((0x0C00, 0x0C7F),),
    "Kannada": ((0x0C80, 0x0CFF),),
    "Malayalam": ((0x0D00, 0x0D7F),),
    "Sinhala": ((0x0D80, 0x0DFF),),
    "Thai": ((0x0E00, 0x0E7F),),
    "Lao": ((0x0E80, 0x0EFF),),
    "Tibetan": ((0x0F00, 0x0FFF),),
    "Myanmar": ((0x1000, 0x109F),),
    "Georgian": ((0x10A0, 0x10FF), (0x1C90, 0x1CBA)),
    "Hangul": ((0x1100, 0x11FF), (0x302E, 0x302F), (0xA4D0, 0xA4FF), (0xAC00, 0xD7A3)),
    "Ethiopic": ((0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DFF), (0x1E800, 0x1EAFF)),
    "Cherokee": ((0x13A0, 0x13FF), (0xAB70, 0xABBF)),
    "Khmer": ((0x1780, 0x17FF), (0x19E0, 0x19FF)),
    "Mongolian": ((0x1800, 0x18AF),),
    "Yi": ((0xA000, 0xA4CF),),
    "Hiragana": ((0x3041, 0x309F),),
    "Katakana": ((0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)),
    "Han": ((0x2E80, 0x2FDF), (0x3005, 0x3007), (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)),
}

# Some ranges above are only relevant to astral-plane characters; keep a fast
# per-character decision path that checks the compact blocks first.
_RANGES: list[tuple[str, int, int]] = []
for _script, _spans in SCRIPT_RANGES.items():
    for _lo, _hi in _spans:
        _RANGES.append((_script, _lo, _hi))
_RANGES.sort(key=lambda row: row[1])

UNKNOWN_SCRIPT = "Unknown"


@dataclass(frozen=True, slots=True)
class ScriptEvidence:
    """How much of an utterance's alphabetic content belongs to each script."""

    total_letters: int
    counts: dict[str, int]

    @property
    def fractions(self) -> dict[str, float]:
        if self.total_letters == 0:
            return {}
        return {
            script: count / self.total_letters
            for script, count in self._sorted_counts()
        }

    def _sorted_counts(self) -> list[tuple[str, int]]:
        return sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))

    @property
    def ranked(self) -> list[tuple[str, int]]:
        return self._sorted_counts()

    @property
    def dominant(self) -> str:
        ranked = self.ranked
        return ranked[0][0] if ranked else UNKNOWN_SCRIPT

    @property
    def coverage(self) -> float:
        ranked = self.ranked
        if not ranked or self.total_letters == 0:
            return 0.0
        return ranked[0][1] / self.total_letters

    def mixtures(self, threshold: float = 0.15) -> list[tuple[str, float]]:
        """Scripts whose share of letters is at least ``threshold``."""
        return [
            (script, fraction)
            for script, fraction in self.fractions.items()
            if fraction >= threshold and script != UNKNOWN_SCRIPT
        ]


def char_script(char: str) -> str:
    """Return the ISO 15924 title for one alphabetic character, or Unknown."""
    if not char.isalpha():
        return UNKNOWN_SCRIPT
    point = ord(char)
    lo, hi = 0, len(_RANGES) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        _, start, end = _RANGES[mid]
        if point < start:
            hi = mid - 1
        elif point > end:
            lo = mid + 1
        else:
            return _RANGES[mid][0]
    return UNKNOWN_SCRIPT


def dominant_script(text: str) -> str:
    """Script holding the largest share of ``text``'s letters."""
    return analyze(text).dominant


def analyze(text: str) -> ScriptEvidence:
    """Count letters per script, ignoring case marks, digits and punctuation."""
    counts: dict[str, int] = {}
    total = 0
    for char in text:
        if not char.isalpha():
            continue
        script = char_script(char)
        if script == UNKNOWN_SCRIPT:
            script = unicodedata.name(char, UNKNOWN_SCRIPT).split(" ")[0]
        counts[script] = counts.get(script, 0) + 1
        total += 1
    return ScriptEvidence(total_letters=total, counts=counts)


def detect_script(text: str) -> tuple[str, float]:
    """Return ``(script, coverage)``: the dominant script and the fraction of
    letters it accounts for. ``(Unknown, 0.0)`` when there are no letters."""
    evidence = analyze(text)
    if evidence.total_letters == 0:
        return UNKNOWN_SCRIPT, 0.0
    return evidence.dominant, evidence.coverage


def script_fractions(text: str) -> dict[str, float]:
    """Share of letters per script; sums to 1.0 whenever letters exist."""
    return analyze(text).fractions


def is_mixed(text: str, threshold: float = 0.15) -> bool:
    """True when at least two scripts each clear ``threshold`` of the letters.

    Mixed-language utterances are separable by script far more reliably than by
    an acoustic model, so this predicate drives code-switch routing directly.
    """
    return len(analyze(text).mixtures(threshold)) >= 2
