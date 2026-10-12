"""Per-operation latency trace, aggregated per stage and per language.

WHY A TRACE AND NOT ONE TOTAL. "The system is slow" cannot be fixed; "Bengali STT
p95 moved from 780 ms to 2100 ms while capture and TTS did not move" can. So each
operation records named spans (`capture`, `vad`, `stt`, `routing`, `tts`, `playback`)
tagged with its language, and aggregation is per stage *and* per language - a
corpus-wide percentile would blend a fast language into a slow one and hide exactly
the regression this exists to find.

Spans come from `core.time.Timings` when the caller has one (so `first_result_ms` and
`real_time_factor` are preserved rather than re-derived), and from the monotonic clock
when the caller only has a code block. Missing stages stay absent rather than being
recorded as 0 ms, because a zero-latency stage would flatter the totals.
"""

from __future__ import annotations

import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from ..core.errors import ConfigurationError
from ..core.language import LanguageTag
from ..core.time import Timings, monotonic_ns

STAGES = ("capture", "vad", "stt", "routing", "tts", "playback")


@dataclass(frozen=True, slots=True)
class Span:
    """One measured stage of one operation."""

    stage: str
    operation_id: str
    language: str
    ms: float
    first_result_ms: float | None = None
    real_time_factor: float | None = None
    source: str = "timings"

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "operation": self.operation_id,
            "language": self.language,
            "ms": round(self.ms, 3),
            "first_result_ms": None if self.first_result_ms is None else round(self.first_result_ms, 3),
            "real_time_factor": None if self.real_time_factor is None else round(self.real_time_factor, 4),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class StageStats:
    stage: str
    language: str
    count: int
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "language": self.language,
            "count": self.count,
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
        }

    def drifted_beyond(self, previous: "StageStats | None", tolerance_pct: float) -> bool | None:
        """None when either side has no data - which is not "no regression"."""
        if previous is None or previous.median_ms <= 0:
            return None
        return (self.median_ms - previous.median_ms) / previous.median_ms * 100.0 > tolerance_pct


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-0.95 * len(ordered) // 1))))
    return ordered[rank - 1]


@dataclass(slots=True)
class Trace:
    """Append-only span store with per-stage/per-language aggregation."""

    spans: list[Span] = field(default_factory=list)

    def record(self, stage: str, timings: Timings, *, language: str = "", operation_id: str = "") -> Span:
        """Attach a `Timings` object as one stage span."""
        if stage not in STAGES:
            raise ConfigurationError(f"unknown stage {stage!r}; declared stages are {STAGES}")
        span = Span(
            stage=stage,
            operation_id=operation_id,
            language=_language_text(language),
            ms=timings.total_ms,
            first_result_ms=timings.first_result_ms,
            real_time_factor=timings.real_time_factor,
        )
        self.spans.append(span)
        return span

    def add(self, stage: str, ms: float, *, language: str = "", operation_id: str = "", first_result_ms: float | None = None) -> Span:
        """Attach a plain millisecond figure, e.g. from a hand-timed block."""
        span = Span(stage=stage, operation_id=operation_id, language=_language_text(language),
                    ms=float(ms), first_result_ms=first_result_ms, source="manual")
        self.spans.append(span)
        return span

    @contextmanager
    def timed(self, stage: str, *, language: str = "", operation_id: str = "") -> Iterator[dict]:
        """`with trace.timed("stt", language="bn"): ...` records the wall span.

        Yields a dict the body can annotate (`out["first_audio_ms"] = ...`) so the
        stage's own sub-latency travels with it instead of being lost.
        """
        if stage not in STAGES:
            raise ConfigurationError(f"unknown stage {stage!r}; declared stages are {STAGES}")
        started = monotonic_ns()
        out: dict = {}
        try:
            yield out
        finally:
            elapsed = (monotonic_ns() - started) / 1e6
            self.spans.append(
                Span(
                    stage=stage,
                    operation_id=operation_id,
                    language=_language_text(language),
                    ms=elapsed,
                    first_result_ms=out.get("first_result_ms"),
                    real_time_factor=out.get("real_time_factor"),
                    source="stopwatch",
                )
            )

    def operations(self) -> tuple[str, ...]:
        seen: list[str] = []
        for span in self.spans:
            if span.operation_id not in seen:
                seen.append(span.operation_id)
        return tuple(seen)

    def for_operation(self, operation_id: str) -> tuple[Span, ...]:
        return tuple(s for s in self.spans if s.operation_id == operation_id)

    def missing_stages(self, operation_id: str) -> tuple[str, ...]:
        """Stages this operation never reported - the gaps a totals-only view hides."""
        present = {s.stage for s in self.for_operation(operation_id)}
        return tuple(stage for stage in STAGES if stage not in present)

    def languages(self) -> tuple[str, ...]:
        seen: list[str] = []
        for span in self.spans:
            if span.language not in seen:
                seen.append(span.language)
        return tuple(seen)

    def stats(self, stage: str, language: str = "") -> StageStats | None:
        wanted = [s.ms for s in self.spans if s.stage == stage and (not language or s.language == _language_text(language))]
        if not wanted:
            return None
        return StageStats(
            stage=stage,
            language=language or "*",
            count=len(wanted),
            median_ms=statistics.median(wanted),
            p95_ms=_p95(wanted),
            min_ms=min(wanted),
            max_ms=max(wanted),
            mean_ms=statistics.fmean(wanted),
        )

    def all_stats(self) -> tuple[StageStats, ...]:
        rows: list[StageStats] = []
        for language in self.languages():
            for stage in STAGES:
                entry = self.stats(stage, language)
                if entry is not None:
                    rows.append(entry)
        return tuple(rows)

    def bottleneck(self, language: str = "") -> StageStats | None:
        """The stage with the highest p95 for that language: where to look first."""
        rows = [row for row in self.all_stats() if not language or row.language == _language_text(language)]
        return max(rows, key=lambda row: row.p95_ms, default=None)

    def compare(self, previous: "Trace", *, tolerance_pct: float = 25.0) -> tuple[dict, ...]:
        """Per (stage, language) drift against an earlier trace, with unknowns named.

        A stage present in one trace but not the other yields `"no baseline"` rather
        than a percentage, so an attribution never rests on a comparison that could
        not have been made.
        """
        rows = []
        for current in self.all_stats():
            baseline = previous.stats(current.stage, current.language)
            drift = None if baseline is None or baseline.median_ms <= 0 else (current.median_ms - baseline.median_ms) / baseline.median_ms * 100.0
            rows.append(
                {
                    "stage": current.stage,
                    "language": current.language,
                    "current_median_ms": round(current.median_ms, 3),
                    "baseline_median_ms": None if baseline is None else round(baseline.median_ms, 3),
                    "delta_pct": None if drift is None else round(drift, 2),
                    "verdict": "no baseline" if drift is None else ("regressed" if drift > tolerance_pct else ("improved" if drift < -tolerance_pct else "unchanged")),
                }
            )
        return tuple(rows)

    def render_text(self) -> str:
        if not self.spans:
            return "trace: empty (no spans recorded - nothing is asserted about latency)"
        lines = [f"trace: {len(self.spans)} span(s) over {len(self.operations())} operation(s)"]
        for row in self.all_stats():
            lines.append(
                f"  {row.language:<8} {row.stage:<9} n={row.count:<4} median {row.median_ms:8.1f} ms  "
                f"p95 {row.p95_ms:8.1f} ms  max {row.max_ms:8.1f} ms"
            )
        for language in self.languages():
            top = self.bottleneck(language)
            if top is not None:
                lines.append(f"  bottleneck for {language}: {top.stage} (p95 {top.p95_ms:.1f} ms)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "spans": [s.as_dict() for s in self.spans],
            "stats": [row.as_dict() for row in self.all_stats()],
            "missing_stages": {op: self.missing_stages(op) for op in self.operations()},
            "bottlenecks": {language: (None if self.bottleneck(language) is None else self.bottleneck(language).as_dict()) for language in self.languages()},
        }


def _language_text(language: LanguageTag | str | None) -> str:
    if language is None or language == "":
        return ""
    if isinstance(language, LanguageTag):
        return language.code
    return LanguageTag.parse(str(language)).code
