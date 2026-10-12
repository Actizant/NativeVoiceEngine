"""Benchmark harness, regression baseline checks, and a fault-isolating suite.

WHY MIN/MEDIAN/P95 AND NOT A MEAN. Latency distributions for local speech work are
long-tailed: a model load, a page fault or a Windows audio callback can add hundreds
of milliseconds once in twenty runs. An average hides that behind a respectable number,
so the reported set is min/median/p95/max plus every raw sample, and the p95 uses the
nearest-rank definition (no interpolated value that never actually occurred).

A first run has no baseline, and this module says `no_baseline` rather than
"improved": comparing against a file that does not exist is how a regression check
becomes decoration. Suite rows keep their own status, so one crashing benchmark yields
an `ERROR` row instead of aborting the run and silently deleting the other results.
"""

from __future__ import annotations

import inspect
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..core.errors import ConfigurationError
from ..core.errors import CapabilityUnavailable
from .report import Evidence, host_summary

RowStatus = ("OK", "ERROR", "NOT_RUN")


@dataclass(frozen=True, slots=True)
class Benchmark:
    """What to run, how often, and with what parameters (all recorded in output)."""

    name: str
    subject: str = ""
    parameters: dict = field(default_factory=dict)
    repeats: int = 5
    warmup: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ConfigurationError("a benchmark needs at least one timed repeat")
        if self.warmup < 0:
            raise ConfigurationError("warmup repeats cannot be negative")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    subject: str
    parameters: dict
    repeats: int
    warmup: int
    seed: int | None
    min_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float
    mean_ms: float
    samples_ms: tuple[float, ...]
    status: str = "OK"
    error: str = ""
    host: str = ""
    measured_at: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "subject": self.subject,
            "parameters": self.parameters,
            "repeats": self.repeats,
            "warmup": self.warmup,
            "seed": self.seed,
            "min_ms": round(self.min_ms, 4),
            "median_ms": round(self.median_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "mean_ms": round(self.mean_ms, 4),
            "samples_ms": [round(v, 4) for v in self.samples_ms],
            "status": self.status,
            "error": self.error,
            "host": self.host,
            "measured_at": self.measured_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BenchmarkResult":
        return cls(
            name=payload["name"],
            subject=payload.get("subject", ""),
            parameters=dict(payload.get("parameters", {})),
            repeats=int(payload.get("repeats", 0)),
            warmup=int(payload.get("warmup", 0)),
            seed=payload.get("seed"),
            min_ms=float(payload["min_ms"]),
            median_ms=float(payload["median_ms"]),
            p95_ms=float(payload["p95_ms"]),
            max_ms=float(payload["max_ms"]),
            mean_ms=float(payload.get("mean_ms", payload["median_ms"])),
            samples_ms=tuple(float(v) for v in payload.get("samples_ms", ())),
            status=payload.get("status", "OK"),
            error=payload.get("error", ""),
            host=payload.get("host", ""),
            measured_at=payload.get("measured_at", ""),
        )


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Inputs must be sorted; no interpolation.

    Interpolated percentiles report a time that no run ever took, which matters when
    the number is quoted as "worst observed latency users can expect".
    """
    if not sorted_values:
        raise ConfigurationError("cannot take a percentile of an empty sample set")
    if not 0.0 < fraction <= 1.0:
        raise ConfigurationError("percentile fraction must be in (0, 1]")
    rank = max(1, min(len(sorted_values), int(-(-fraction * len(sorted_values) // 1))))
    return sorted_values[rank - 1]


def _invoke(fn: Callable, rng: random.Random | None) -> None:
    """Call `fn`, handing it a deterministic RNG only if it asks for one.

    A benchmark that consumes randomness must not depend on process-global state, or
    two runs of the same command produce numbers that differ for no measurable reason.
    """
    if rng is not None and "rng" in inspect.signature(fn).parameters:
        fn(rng=rng)
        return
    fn()


def run_benchmark(benchmark: Benchmark | Callable, fn: Callable | None = None, *, repeats: int = 5, warmup: int = 1, seed: int | None = None, subject: str = "", parameters: dict | None = None) -> BenchmarkResult:
    """Time `fn` `repeats` times after `warmup` untimed runs.

    Accepts either `run_benchmark(Benchmark(...), fn)` or `run_benchmark(fn, ...)`;
    the second form exists so a one-off measurement does not have to construct a
    descriptor whose fields it then ignores.
    """
    if isinstance(benchmark, Benchmark):
        descriptor = benchmark
        if fn is None:
            raise CapabilityUnavailable(
                "benchmark descriptor given but nothing to time", name=benchmark.name
            )
        callable_under_test = fn
    else:
        callable_under_test = benchmark
        descriptor = Benchmark(name=getattr(callable_under_test, "__name__", "benchmark") or subject,
                               subject=subject, parameters=dict(parameters or {}),
                               repeats=repeats, warmup=warmup, seed=seed)
    rng = None if descriptor.seed is None else random.Random(descriptor.seed)
    for _ in range(descriptor.warmup):
        _invoke(callable_under_test, rng)
    samples: list[float] = []
    for _ in range(descriptor.repeats):
        started = time.perf_counter()
        _invoke(callable_under_test, rng)
        samples.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples)
    return BenchmarkResult(
        name=descriptor.name,
        subject=descriptor.subject or descriptor.name,
        parameters=descriptor.parameters,
        repeats=descriptor.repeats,
        warmup=descriptor.warmup,
        seed=descriptor.seed,
        min_ms=ordered[0],
        median_ms=statistics.median(ordered),
        p95_ms=percentile(ordered, 0.95),
        max_ms=ordered[-1],
        mean_ms=statistics.fmean(ordered),
        samples_ms=tuple(samples),
        host=host_summary(),
        measured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )


@dataclass(frozen=True, slots=True)
class RegressionOutcome:
    """Comparison against a stored baseline, including the no-baseline case."""

    kind: str  # no_baseline | improved | unchanged | regressed | incompatible | unmeasurable
    name: str
    tolerance_pct: float
    baseline: BenchmarkResult | None
    current: BenchmarkResult | None
    delta_pct: float | None
    detail: str

    @property
    def has_baseline(self) -> bool:
        return self.baseline is not None

    @property
    def accept(self) -> bool | None:
        """None means "cannot say", which is not approval. NOT_RUN, again."""
        if self.kind in ("improved", "unchanged"):
            return True
        if self.kind == "regressed":
            return False
        return None

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "has_baseline": self.has_baseline,
            "tolerance_pct": self.tolerance_pct,
            "delta_pct": None if self.delta_pct is None else round(self.delta_pct, 3),
            "accept": self.accept,
            "detail": self.detail,
            "baseline_median_ms": None if self.baseline is None else round(self.baseline.median_ms, 4),
            "current_median_ms": None if self.current is None else round(self.current.median_ms, 4),
            "baseline_measured_at": "" if self.baseline is None else self.baseline.measured_at,
            "baseline_host": "" if self.baseline is None else self.baseline.host,
        }


def save_baseline(path: str, result: BenchmarkResult) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.as_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_baseline(path: str) -> BenchmarkResult | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return BenchmarkResult.from_dict(json.load(handle))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CapabilityUnavailable(
            "baseline file exists but is not a readable BenchmarkResult", path=path, error=str(exc)
        ) from exc


def compare(current: BenchmarkResult, baseline: BenchmarkResult | None, *, tolerance_pct: float = 10.0) -> RegressionOutcome:
    """Median-versus-median drift. Parameter sets must match or the comparison is refused.

    Comparing a 44.1 kHz run against a stored 16 kHz baseline would produce a
    "regression" that is really a different experiment, so mismatched parameters yield
    `incompatible` with both sides named instead of a number.
    """
    if baseline is None:
        return RegressionOutcome(
            "no_baseline", current.name, tolerance_pct, None, current, None,
            "no stored baseline at this path; the first run only records one and asserts nothing",
        )
    if baseline.parameters != current.parameters or baseline.subject != current.subject:
        return RegressionOutcome(
            "incompatible", current.name, tolerance_pct, baseline, current, None,
            f"baseline describes subject={baseline.subject!r} params={baseline.parameters!r}, "
            f"current is subject={current.subject!r} params={current.parameters!r}",
        )
    if baseline.median_ms <= 0:
        return RegressionOutcome(
            "unmeasurable", current.name, tolerance_pct, baseline, current, None,
            "baseline median is zero; percentage drift is undefined",
        )
    delta = (current.median_ms - baseline.median_ms) / baseline.median_ms * 100.0
    if delta > tolerance_pct:
        kind = "regressed"
    elif delta < -tolerance_pct:
        kind = "improved"
    else:
        kind = "unchanged"
    return RegressionOutcome(kind, current.name, tolerance_pct, baseline, current, delta,
                             f"median {baseline.median_ms:.3f} ms -> {current.median_ms:.3f} ms ({delta:+.2f}%)")


def check_regression(
    benchmark: Benchmark,
    fn: Callable,
    baseline_path: str,
    *,
    tolerance_pct: float = 10.0,
    write_baseline_when_absent: bool = True,
) -> "tuple[BenchmarkResult, RegressionOutcome]":
    """Run, compare to `baseline_path`, and (only when absent) record a baseline.

    A stored baseline is never overwritten implicitly: silently promoting today's
    slow run into tomorrow's baseline is the standard way a regression suite decays
    into a suite that always passes.
    """
    result = run_benchmark(benchmark, fn)
    baseline = load_baseline(baseline_path)
    outcome = compare(result, baseline, tolerance_pct=tolerance_pct)
    if baseline is None and write_baseline_when_absent:
        save_baseline(baseline_path, result)
    return result, outcome


@dataclass(frozen=True, slots=True)
class SuiteRow:
    name: str
    status: str
    result: BenchmarkResult | None
    outcome: RegressionOutcome | None
    reason: str
    duration_s: float | None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "duration_s": None if self.duration_s is None else round(self.duration_s, 6),
            "result": None if self.result is None else self.result.as_dict(),
            "regression": None if self.outcome is None else self.outcome.as_dict(),
        }


@dataclass(slots=True)
class Suite:
    """A list of benchmarks that cannot abort each other.

    `skip_reason` is data supplied by the caller (usually "no approved model for this
    language"), and a skipped row is NOT_RUN with that reason - not an omitted row,
    which is how gaps disappear from a report.
    """

    name: str
    items: list = field(default_factory=list)
    rows: list[SuiteRow] = field(default_factory=list)

    def add(self, benchmark: Benchmark, fn: Callable | None = None, *, skip_reason: str = "", baseline_path: str = "", tolerance_pct: float = 10.0) -> "Suite":
        self.items.append((benchmark, fn, skip_reason, baseline_path, tolerance_pct))
        return self

    def run(self) -> "Suite":
        self.rows = []
        for benchmark, fn, skip_reason, baseline_path, tolerance in self.items:
            if skip_reason:
                self.rows.append(SuiteRow(benchmark.name, "NOT_RUN", None, None, skip_reason, None))
                continue
            if fn is None:
                self.rows.append(SuiteRow(benchmark.name, "NOT_RUN", None, None, "benchmark was registered without a callable", None))
                continue
            started = time.monotonic()
            try:
                result = run_benchmark(benchmark, fn)
            except Exception as exc:  # one broken benchmark must not end the suite
                self.rows.append(
                    SuiteRow(benchmark.name, "ERROR", None, None, f"{type(exc).__name__}: {exc}", time.monotonic() - started)
                )
                continue
            outcome = None
            if baseline_path:
                try:
                    baseline = load_baseline(baseline_path)
                except CapabilityUnavailable as exc:
                    self.rows.append(SuiteRow(benchmark.name, "ERROR", result, None, str(exc), time.monotonic() - started))
                    continue
                outcome = compare(result, baseline, tolerance_pct=tolerance)
                if baseline is None:
                    save_baseline(baseline_path, result)
            self.rows.append(SuiteRow(benchmark.name, "OK", result, outcome, "", time.monotonic() - started))
        return self

    def counts(self) -> dict:
        return {status: sum(1 for row in self.rows if row.status == status) for status in RowStatus}

    def render_text(self) -> str:
        lines = [f"suite {self.name!r}: {len(self.rows)} benchmark(s)"]
        for row in self.rows:
            if row.result is not None and row.status == "OK":
                drift = "" if row.outcome is None else f" [{row.outcome.kind}]"
                lines.append(
                    f"  OK      {row.name}: median {row.result.median_ms:.3f} ms, p95 {row.result.p95_ms:.3f} ms, "
                    f"min {row.result.min_ms:.3f}, max {row.result.max_ms:.3f} ({row.result.repeats} repeats){drift}"
                )
            else:
                lines.append(f"  {row.status:<8} {row.name}: {row.reason}")
        counts = self.counts()
        lines.append(f"  totals: ok={counts['OK']} error={counts['ERROR']} not_run={counts['NOT_RUN']}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "suite": self.name,
            "counts": self.counts(),
            "host": host_summary(),
            "evidence": Evidence.capture(f"python -m nativevoice.eval.runner {self.name}").as_dict(),
            "rows": [row.as_dict() for row in self.rows],
        }
