"""Transcription quality: per-utterance scoring, aggregation, hallucination flags.

WHY THE GUARD LIVES HERE. `metrics.py` is honest arithmetic; arithmetic cannot tell
whether its input was speech. This module is where an utterance carries its provenance
(an `EvalSample`) far enough that the scorer can refuse it. Scoring a synthetic tone
manifest, or reporting a quality verdict from pairs that nobody licensed, are both
refused at the entry point rather than at the rendering step where the temptation to
explain the number away is strongest.

Per-language grouping plus worst-case reporting is deliberate: a corpus average hides
that one language can be at 90% WER while the headline reads 12%, which is precisely
how the reference projects' READMEs ended up describing Bengali as "supported".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Union

from ..audio.pcm import measure
from ..core.audio import AudioBuffer
from ..core.errors import CapabilityUnavailable
from ..core.language import LanguageTag
from .metrics import (
    EditDistance,
    ErrorRate,
    aggregate_rates,
    sentence_error_rate,
    tokenize,
    word_alignment,
)
from .samples import EvalSample, synthetic_guard

PairLike = Union["Pair", Sequence[str]]


@dataclass(frozen=True, slots=True)
class TranscriptionScore:
    """One reference/hypothesis pair at every granularity we claim to measure."""

    reference: str
    hypothesis: str
    language: str
    wer: float
    cer: float
    sentence_error_rate: float
    word: ErrorRate
    character: ErrorRate
    reference_tokens: int
    hypothesis_tokens: int

    @property
    def language_tag(self) -> LanguageTag:
        return LanguageTag.parse(self.language)

    @property
    def perfect(self) -> bool:
        return self.wer == 0.0 and self.cer == 0.0

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "wer": None if self.wer == float("inf") else round(self.wer, 6),
            "cer": None if self.cer == float("inf") else round(self.cer, 6),
            "sentence_error_rate": round(self.sentence_error_rate, 6),
            "reference_tokens": self.reference_tokens,
            "hypothesis_tokens": self.hypothesis_tokens,
            "word": self.word.as_dict(),
            "character": self.character.as_dict(),
        }


def score(reference: str, hypothesis: str, language: "LanguageTag | str" = "") -> TranscriptionScore:
    """WER, CER, sentence-error rate and full alignment for a single utterance.

    `language` is only used to label the record and route normalisation; the
    normalisation itself is script-driven (see `metrics.normalise`), never
    model-driven, so nothing here depends on an approved model being installed.
    """
    tag = language if isinstance(language, str) else language.bcp47
    words = word_alignment(reference, hypothesis, level="word")
    chars = word_alignment(reference, hypothesis, level="char")
    return TranscriptionScore(
        reference=reference,
        hypothesis=hypothesis,
        language=tag,
        wer=words.rate,
        cer=chars.rate,
        sentence_error_rate=sentence_error_rate(reference, hypothesis),
        word=words,
        character=chars,
        reference_tokens=words.alignment.reference_length,
        hypothesis_tokens=words.hypothesis_tokens,
    )


@dataclass(frozen=True, slots=True)
class Pair:
    """A scored utterance plus the provenance needed to trust it."""

    reference: str
    hypothesis: str
    language: str
    sample: EvalSample | None = None
    audio: AudioBuffer | None = None

    @classmethod
    def from_sample(cls, sample: EvalSample, hypothesis: str, audio: AudioBuffer | None = None) -> "Pair":
        return cls(sample.transcript, hypothesis, sample.language, sample=sample, audio=audio)

    def resolved_language(self) -> str:
        if self.language:
            return self.language
        if self.sample is not None:
            return self.sample.language
        raise CapabilityUnavailable("scoring pair has no language", reference=self.reference[:40])


@dataclass(frozen=True, slots=True)
class LanguageAggregate:
    """Micro + macro averages for one language, plus the worst utterance."""

    language: str
    pairs: int
    reference_tokens: int
    micro_wer: float
    macro_wer: float
    micro_cer: float
    mean_sentence_error_rate: float
    worst_wer: float
    worst_reference: str
    worst_hypothesis: str
    hallucinating: int

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "pairs": self.pairs,
            "reference_tokens": self.reference_tokens,
            "micro_wer": round(self.micro_wer, 6),
            "macro_wer": round(self.macro_wer, 6),
            "micro_cer": round(self.micro_cer, 6),
            "mean_sentence_error_rate": round(self.mean_sentence_error_rate, 6),
            "worst_wer": None if self.worst_wer == float("inf") else round(self.worst_wer, 6),
            "worst_reference": self.worst_reference,
            "worst_hypothesis": self.worst_hypothesis,
            "hallucinating_pairs": self.hallucinating,
        }


@dataclass(frozen=True, slots=True)
class TranscriptionAggregate:
    """Corpus-level result. `certifies_quality` is False unless provenance allowed it."""

    scores: tuple[tuple[Pair, TranscriptionScore], ...]
    overall: LanguageAggregate
    by_language: tuple[LanguageAggregate, ...]
    certifies_quality: bool
    provenance_notes: tuple[str, ...]

    def language(self, tag: str) -> LanguageAggregate | None:
        root = LanguageTag.parse(tag).code
        for entry in self.by_language:
            if entry.language == root:
                return entry
        return None

    def worst_language(self) -> LanguageAggregate | None:
        if not self.by_language:
            return None
        return max(self.by_language, key=lambda entry: (entry.worst_wer, entry.micro_wer))

    def as_dict(self) -> dict:
        return {
            "pairs": len(self.scores),
            "certifies_quality": self.certifies_quality,
            "provenance_notes": list(self.provenance_notes),
            "overall": self.overall.as_dict(),
            "by_language": [entry.as_dict() for entry in self.by_language],
            "worst_language": None if self.worst_language() is None else self.worst_language().as_dict(),
        }


def _normalised(items: Iterable[PairLike]) -> tuple[Pair, ...]:
    pairs = []
    for item in items:
        if isinstance(item, Pair):
            pairs.append(item)
            continue
        if isinstance(item, EvalSample):
            raise CapabilityUnavailable(
                "an EvalSample alone is not a scored pair; pass Pair.from_sample(sample, hypothesis)",
                sample=item.sample_id,
            )
        if len(item) not in (3, 4):
            raise CapabilityUnavailable(
                "pair must be (reference, hypothesis, language[, audio])", got=len(item)
            )
        audio = item[3] if len(item) == 4 else None
        pairs.append(Pair(str(item[0]), str(item[1]), str(item[2]), audio=audio))
    return tuple(pairs)


def evaluate_pairs(
    pairs: Iterable[PairLike],
    *,
    enforce_synthetic_rule: bool = True,
) -> TranscriptionAggregate:
    """Score many pairs, grouped per language, with worst-case and provenance verdict.

    `enforce_synthetic_rule` exists so a caller can ask for diagnostic arithmetic on
    synthetic fixtures (transport tests, alignment tests); it cannot be turned off to
    produce a *quality* claim, because the resulting aggregate has
    `certifies_quality=False` and no report writer will accept one.
    """
    resolved = _normalised(pairs)
    samples = [p.sample for p in resolved if p.sample is not None]
    if enforce_synthetic_rule:
        synthetic_guard(samples, activity="evaluate_pairs")

    notes: list[str] = []
    grouped: dict[str, list[tuple[Pair, TranscriptionScore]]] = {}
    for pair in resolved:
        entry = score(pair.reference, pair.hypothesis, pair.resolved_language())
        grouped.setdefault(LanguageTag.parse(pair.resolved_language()).code, []).append((pair, entry))

    synthetic_count = sum(1 for s in samples if s.synthetic)
    unattached = sum(1 for p in resolved if p.sample is None)
    unlicensed = sum(1 for s in samples if not s.usable and not s.synthetic)
    if synthetic_count:
        notes.append(f"{synthetic_count} pair(s) came from synthetic fixtures and cannot certify quality")
    if unattached:
        notes.append(f"{unattached} pair(s) carry no EvalSample: licence and consent are unverifiable")
    if unlicensed:
        notes.append(f"{unlicensed} pair(s) have a sample missing licence or consent")
    certifies = bool(resolved) and not synthetic_count and not unattached and not unlicensed

    all_entries = [entry for group in grouped.values() for _, entry in group]
    all_pairs = [pair for group in grouped.values() for pair, _ in group]
    aggregates = [_aggregate(code, group) for code, group in sorted(grouped.items())]
    overall = _aggregate("all", [(p, e) for p, e in zip(all_pairs, all_entries)])
    return TranscriptionAggregate(
        scores=tuple(zip(all_pairs, all_entries)),
        overall=overall,
        by_language=tuple(aggregates),
        certifies_quality=certifies,
        provenance_notes=tuple(notes),
    )


def _aggregate(language: str, group: Sequence[tuple[Pair, TranscriptionScore]]) -> LanguageAggregate:
    # Micro-average: total edits over total reference tokens, i.e. corpus WER.
    total_distance = sum(s.word.alignment.distance for _, s in group)
    total_reference = sum(s.reference_tokens for _, s in group)
    total_char_distance = sum(s.character.alignment.distance for _, s in group)
    total_char_reference = sum(s.character.alignment.reference_length for _, s in group)
    micro_wer = float("inf") if total_reference == 0 and total_distance else total_distance / max(1, total_reference)
    micro_cer = total_char_distance / max(1, total_char_reference)
    worst = max(group, key=lambda item: item[1].wer, default=(None, None))
    hallucinating = sum(1 for pair, entry in group if hallucination_flags(entry.reference, entry.hypothesis, audio=pair.audio))
    return LanguageAggregate(
        language=language,
        pairs=len(group),
        reference_tokens=total_reference,
        micro_wer=micro_wer,
        macro_wer=aggregate_rates([s.wer for _, s in group]) if group else float("nan"),
        micro_cer=micro_cer,
        mean_sentence_error_rate=aggregate_rates([s.sentence_error_rate for _, s in group]) if group else float("nan"),
        worst_wer=worst[1].wer if worst[0] is not None else float("nan"),
        worst_reference=worst[0].reference if worst[0] is not None else "",
        worst_hypothesis=worst[1].hypothesis if worst[1] is not None else "",
        hallucinating=hallucinating,
    )


@dataclass(frozen=True, slots=True)
class HallucinationFlag:
    """One suspected decoder confabulation, with the measurement behind it."""

    kind: str
    detail: str
    value: float | None = None

    def as_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "value": None if self.value is None else round(self.value, 4)}


def repetition_ratio(text: str) -> float:
    """Share of tokens that repeat the token immediately before them, in [0, 1].

    A streaming decoder stuck in a loop produces "the the the the" or one dictionary
    word many times; that shape is visible without any language model, which is why
    it is the primary signal here.
    """
    tokens = tokenize(text)
    if len(tokens) < 2:
        return 0.0
    repeats = sum(1 for index in range(1, len(tokens)) if tokens[index] == tokens[index - 1])
    return repeats / (len(tokens) - 1)


def dominant_token_share(text: str) -> float:
    """Fraction of tokens taken by the most frequent token: catches looped words."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return max(counts.values()) / len(tokens)


def hallucination_flags(
    reference: str,
    hypothesis: str,
    *,
    audio: AudioBuffer | None = None,
    repetition_threshold: float = 0.5,
    dominance_threshold: float = 0.6,
    silence_threshold: float = 0.01,
    near_silent_share: float = 0.9,
) -> tuple[HallucinationFlag, ...]:
    """Documented failure mode: dictionary words or repeated tokens over no speech.

    Three independent triggers, because the observed decoders fail in different ways:
    (a) repetition ratio / one dominant token, (b) output invented against an empty
    reference, (c) non-empty output when the *audio* is measurably near-silence - the
    last requires the caller to pass the buffer, since text alone cannot prove silence.
    """
    flags: list[HallucinationFlag] = []
    hypothesis_tokens = tokenize(hypothesis)
    ratio = repetition_ratio(hypothesis)
    if ratio >= repetition_threshold and len(hypothesis_tokens) >= 3:
        flags.append(
            HallucinationFlag(
                "repeated_token_run",
                f"{ratio:.0%} of adjacent tokens repeat the previous token",
                value=ratio,
            )
        )
    share = dominant_token_share(hypothesis)
    if share >= dominance_threshold and len(hypothesis_tokens) >= 4:
        flags.append(
            HallucinationFlag(
                "single_dominant_token",
                f"one token accounts for {share:.0%} of the hypothesis",
                value=share,
            )
        )
    if hypothesis_tokens and not tokenize(reference):
        flags.append(
            HallucinationFlag(
                "text_over_empty_reference",
                f"{len(hypothesis_tokens)} token(s) emitted against an empty reference",
                value=float(len(hypothesis_tokens)),
            )
        )
    if audio is not None and audio.frame_count > 0 and hypothesis_tokens:
        level = measure(audio, silence_threshold=silence_threshold)
        if level.silence_fraction >= near_silent_share or level.rms <= silence_threshold:
            flags.append(
                HallucinationFlag(
                    "text_over_near_silence",
                    f"non-empty output over audio that is {level.silence_fraction:.0%} silent (rms {level.rms:.5f})",
                    value=level.silence_fraction,
                )
            )
    return tuple(flags)


def alignment_of(reference: str, hypothesis: str) -> EditDistance:
    """Raw word alignment, exposed so a reviewer can reproduce the WER by hand."""
    return word_alignment(reference, hypothesis, level="word").alignment
