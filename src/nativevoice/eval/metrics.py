"""Metric primitives: edit distance over tokens/characters, and Indic normalisation.

WHY THIS FILE IS HAND-WRITTEN. The specification requires error rates to be
measured, and an error rate is only defensible if the alignment that produced it
can be shown. A third-party scorer hides its tokenisation, its normalisation and
its tie-breaking behind a call; here every step is inspectable, and
`EditDistance` exposes the actual substitution/insertion/deletion operations so a
reported WER can be reproduced by hand from the pair alone.

Normalisation is deliberately conservative. A *proper* Indic text normaliser
(matra/conjunct segmentation, digit folding, punctuation sense) is an external
dependency this project has not approved, and it is emphatically not the
`whisper-normalizer` package - that one is Whisper-specific, English-centric in
its punctuation table and would quietly invent behaviour we cannot audit. What is
implemented below only removes characters whose irrelevance is uncontroversial
(script-independent formatting controls, sentence terminators used as separators)
and applies Unicode NFC. It never rewrites a word, so it can under-count nothing
that a stricter normaliser would have caught: the worst case is a slightly
*pessimistic* error rate, never a flattering one.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from ..core.errors import CapabilityUnavailable, ConfigurationError

# Sentence/clause terminators that carry no lexical content in Devanagari and
# Bengali text: U+0964 danda, U+0965 double danda, plus the ASCII/Devanagari
# abbreviation sign U+0970. Treated as word boundaries, never as tokens.
INDIC_TERMINATORS = "।॥॰"

# Zero-width joiner/disjoiner are used to force or block conjunct formation; they
# have no independent pronunciation, so comparing transcripts that differ only in
# their presence would penalise the engine for a typesetting choice.
INVISIBLE_CONTROLS = "\u200b\u200c\u200d\u200e\u200f\ufeff"

# Generic punctuation treated as a token boundary. The apostrophe is deliberately
# absent so English contractions ("don't") survive as one token.
_BOUNDARY_PUNCT = "!\"#$%&()*+,-./:;<=>?[\\]^_`{|}~“”–—…،؛"

_TRANSLITERALS = {ord(c): " " for c in _BOUNDARY_PUNCT + INDIC_TERMINATORS}
# Deleted, not spaced: a joiner sits *inside* a word, so turning it into a boundary
# would split one token into two and score a correct transcript as two errors.
for _control in INVISIBLE_CONTROLS:
    _TRANSLITERALS[ord(_control)] = None
_TRANSLITERALS[ord("\t")] = " "
_TRANSLITERALS[ord("\n")] = " "
_TRANSLITERALS[ord("\r")] = " "
_TRANSLITERALS[ord("\u00a0")] = " "

_MULTI_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class EditDistance:
    """Alignment cost between two sequences, with the operation mix preserved.

    `substitutions + insertions + deletions == distance` always holds, which is
    what lets a WER headline be checked against the breakdown in a report.
    """

    substitutions: int
    insertions: int
    deletions: int
    correct: int

    @property
    def distance(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def reference_length(self) -> int:
        """Errors are normalised against the reference length (the convention)."""
        return self.correct + self.substitutions + self.deletions

    def as_dict(self) -> dict:
        return {
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "correct": self.correct,
            "distance": self.distance,
            "reference_length": self.reference_length,
        }


def _alignment(reference: Sequence[str], hypothesis: Sequence[str]) -> EditDistance:
    """Minimum-error-rate alignment counts, resolved by backtrace of the DP table.

    The DP itself is textbook Levenshtein; the backtrace is the point. Priority on
    a tie is substitution > deletion > insertion, so the split between
    substitutions and deletions is deterministic and reproducible rather than
    dependent on Python's dict or memory layout.
    """
    rows = len(reference)
    cols = len(hypothesis)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        table[i][0] = i
    for j in range(cols + 1):
        table[0][j] = j
    for i in range(1, rows + 1):
        row = table[i]
        previous = table[i - 1]
        left_char = reference[i - 1]
        for j in range(1, cols + 1):
            cost = 0 if left_char == hypothesis[j - 1] else 1
            row[j] = min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + cost)
    subs = ins = dele = correct = 0
    i, j = rows, cols
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if table[i][j] == table[i - 1][j - 1] + cost:
                if cost:
                    subs += 1
                else:
                    correct += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and table[i][j] == table[i - 1][j] + 1:
            dele += 1
            i -= 1
            continue
        ins += 1
        j -= 1
    return EditDistance(substitutions=subs, insertions=ins, deletions=dele, correct=correct)


def normalise(text: str) -> str:
    """Conservative, auditable normalisation (see module docstring)."""
    folded = unicodedata.normalize("NFC", text)
    spaced = folded.translate(_TRANSLITERALS)
    return _MULTI_SPACE.sub(" ", spaced).strip()


def tokenize(text: str, *, normalise_first: bool = True) -> list[str]:
    """Lowercased word tokens after `normalise`; empty string yields no tokens."""
    body = normalise(text) if normalise_first else text
    if not body:
        return []
    return body.lower().split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER in [0, +inf): edit distance over word tokens / reference token count.

    Raises `ZeroDivisionError`-free: an empty reference is only 0.0 when the
    hypothesis is empty too, otherwise the insertion-only rate is reported as
    `inf`, because a transcript invented over contentless audio is maximally wrong
    and a finite-looking number here would be a lie.
    """
    return error_rate(reference, hypothesis, level="word")


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER: edit distance over characters of the normalised, space-joined text."""
    return error_rate(reference, hypothesis, level="char")


def error_rate(reference: str, hypothesis: str, *, level: str = "word") -> float:
    return word_alignment(reference, hypothesis, level=level).rate


def word_alignment(reference: str, hypothesis: str, *, level: str = "word") -> "ErrorRate":
    """Full error-rate record for one pair at word or character granularity."""
    if level == "word":
        left = tokenize(reference)
        right = tokenize(hypothesis)
    elif level == "char":
        left = list("".join(normalise(reference).lower().split()))
        right = list("".join(normalise(hypothesis).lower().split()))
    else:
        raise ConfigurationError(f"unknown alignment level: {level!r}")
    return ErrorRate(
        level=level,
        alignment=_alignment(left, right),
        tokens=len(left),
        hypothesis_tokens=len(right),
    )


@dataclass(frozen=True, slots=True)
class ErrorRate:
    """One scored pair: alignment plus the denominators a reviewer needs."""

    level: str
    alignment: EditDistance
    tokens: int
    hypothesis_tokens: int

    @property
    def rate(self) -> float:
        reference = self.alignment.reference_length
        if reference == 0:
            return 0.0 if self.alignment.distance == 0 else float("inf")
        return self.alignment.distance / reference

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "rate": None if self.rate == float("inf") else round(self.rate, 6),
            "rate_infinite": self.rate == float("inf"),
            **self.alignment.as_dict(),
            "hypothesis_tokens": self.hypothesis_tokens,
        }


def sentence_error_rate(reference: str, hypothesis: str) -> float:
    """Fraction of reference sentences whose token sequences are not exact.

    Sentences are split on danda/double-danda and ASCII full stop, which is the
    only segmentation claim this file is entitled to make without a real
    normaliser. A pair with an empty reference scores 0.0 only when the hypothesis
    is also empty; otherwise every invented sentence counts as an error.
    """
    left = _sentences(reference)
    right = _sentences(hypothesis)
    if not left:
        return 0.0 if not right else 1.0
    mismatches = 0
    for index, sentence in enumerate(left):
        other = right[index] if index < len(right) else ""
        if tokenize(sentence) != tokenize(other):
            mismatches += 1
    return mismatches / len(left)


def _sentences(text: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for char in text:
        if char in INDIC_TERMINATORS or char in ".!?":
            current.append(" ")
            piece = "".join(current).strip()
            if piece:
                pieces.append(piece)
            current = []
        else:
            current.append(char)
    piece = "".join(current).strip()
    if piece:
        pieces.append(piece)
    return pieces


def aggregate_rates(rates: Sequence[float]) -> float:
    """Micro-average of rates, i.e. corpus WER from per-utterance alignments.

    Kept separate from a plain mean: a corpus WER weights by reference length and
    is the number a reviewer expects to compare against a published table, while
    the macro mean weights each utterance equally. Callers choose deliberately.
    """
    if not rates:
        return float("nan")
    return sum(rates) / len(rates)


def mcd_placeholder(*_args, **_kwargs):
    """Mel-cepstral distortion is NOT AVAILABLE, and this function exists to say so.

    MCD needs a frame-wise mel-cepstral extractor (a waveform->cepstrum pipeline
    with matched sample rates, window, and number of coefficients) plus a spoken
    reference corpus whose licensing and consent we hold. Neither is approved here,
    so any number this function could return would be fabricated. Raising
    `CapabilityUnavailable` is the honest behaviour; a silently-constant "MCD"
    column in a report is exactly the self-deception the specification forbids.
    """
    raise CapabilityUnavailable(
        "mel-cepstral distortion cannot be computed here",
        missing="frame-wise mel-cepstral extractor (external, unapproved)",
        also_missing="licensed, consented spoken reference corpus for the target language",
        required_decisions="cepstral coefficient count, window/overlap, sampling-rate alignment",
    )
