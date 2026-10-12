"""Acceptance report: PASS / FAIL / NOT RUN per criterion, with re-runnable evidence.

WHY NOT_RUN IS A FIRST-CLASS STATUS. The natural failure of an acceptance report is
that a criterion with no data reads like a pass in a summary that only counts
pass/fail, or that `None` gets coerced to `0.0` and then beats a "max latency"
threshold. Here `NOT_RUN` is a distinct enum member, `CriterionResult.passes` is
`bool | None` (never a defaulted False/True), and `summarise()` counts the three
statuses separately so a report of "3 PASS, 0 FAIL, 9 NOT RUN" is visibly 9 criteria
we have no idea about. Every criterion also carries the command, timestamp and host
that produced it, so a claim can be re-run instead of re-litigated.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


class CriterionStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"

    @property
    def counts_as_pass(self) -> bool:
        """The single place that decides what passing means. NOT_RUN is never passing."""
        return self is CriterionStatus.PASS


@dataclass(frozen=True, slots=True)
class Evidence:
    """How this number came to exist, at a level that lets someone re-run it."""

    command: str = ""
    captured_at: str = ""
    host: str = ""
    duration_s: float | None = None
    artifacts: tuple[str, ...] = ()
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def capture(cls, command: str, *, before: dict | None = None, started_s: float | None = None, notes: str = "") -> "Evidence":
        return cls(
            command=command,
            captured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            host=host_summary(),
            duration_s=None if started_s is None else max(0.0, time.monotonic() - started_s),
            before=dict(before or {}),
            after=dict(measurements_snapshot()),
            notes=notes,
        )

    def as_dict(self) -> dict:
        return {
            "command": self.command,
            "captured_at": self.captured_at,
            "host": self.host,
            "duration_s": None if self.duration_s is None else round(self.duration_s, 6),
            "artifacts": list(self.artifacts),
            "before": dict(self.before),
            "after": dict(self.after),
            "notes": self.notes,
        }


def host_summary() -> str:
    """One-line machine description: enough to tell which box produced a number."""
    release = " ".join(filter(None, (platform.system(), platform.release(), platform.machine())))
    try:
        cores = os.cpu_count() or 0
    except Exception:  # platform probes must never break a report
        cores = 0
    return f"{release} / {cores} cpus / python {platform.python_version()}".strip()


def measurements_snapshot() -> dict:
    """Cheap, dependency-free environment facts recorded alongside a measurement.

    Kept to what stdlib answers instantly; deliberately NOT importing the resource
    sampler, because a report writer that can fail must not be able to take the
    acceptance run down with it.
    """
    return {"cwd": os.getcwd(), "pid": os.getpid(), "python": platform.python_version()}


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """One acceptance criterion and the verdict on it."""

    name: str
    status: CriterionStatus
    measured: float | None
    threshold: float | None
    unit: str
    direction: str  # "max" (lower is better) or "min" (higher is better)
    reason: str = ""
    evidence: Evidence | None = None
    language: str = ""

    @property
    def passes(self) -> bool | None:
        """None for NOT_RUN: 'unknown' is not a boolean and must not be coerced."""
        if self.status is CriterionStatus.NOT_RUN:
            return None
        return self.status is CriterionStatus.PASS

    @property
    def measured_text(self) -> str:
        if self.measured is None:
            return "no measurement"
        if not self.unit:
            return f"{self.measured:.4f}"
        separator = "" if self.unit.startswith(" ") else " "
        return f"{self.measured:.4f}{separator}{self.unit}"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "measured": self.measured,
            "threshold": self.threshold,
            "unit": self.unit,
            "direction": self.direction,
            "reason": self.reason,
            "language": self.language,
            "evidence": None if self.evidence is None else self.evidence.as_dict(),
        }


def judge(
    name: str,
    measured: float | None,
    threshold: float | None,
    *,
    unit: str = "",
    direction: str = "max",
    evidence: Evidence | None = None,
    language: str = "",
    reason: str = "",
) -> CriterionResult:
    """Turn a number and a threshold into a verdict, with no numeric defaults.

    A missing measurement is NOT_RUN naming what is missing; a missing threshold is
    NOT_RUN because an unbounded criterion cannot be met. `measured=0.0` is a real
    value and is judged, which is exactly why "no data" must never be encoded as 0.
    """
    if direction not in ("max", "min"):
        raise ConfigurationError(f"unknown direction: {direction!r}")
    if measured is None:
        return CriterionResult(
            name, CriterionStatus.NOT_RUN, None, threshold, unit, direction,
            reason or f"no measurement recorded; nothing to compare against the {direction} threshold"
            + (f" of {threshold}{unit}" if threshold is not None else ""),
            evidence, language,
        )
    if threshold is None:
        return CriterionResult(
            name, CriterionStatus.NOT_RUN, measured, None, unit, direction,
            reason or "criterion has no threshold, so it cannot be judged", evidence, language,
        )
    ok = measured <= threshold if direction == "max" else measured >= threshold
    return CriterionResult(
        name,
        CriterionStatus.PASS if ok else CriterionStatus.FAIL,
        measured, threshold, unit, direction,
        reason or ("" if ok else f"{measured:.4f}{unit} exceeds the {direction} threshold {threshold}{unit}"),
        evidence, language,
    )


@dataclass(frozen=True, slots=True)
class ReportSummary:
    total: int
    passed: int
    failed: int
    not_run: int

    @property
    def verdict(self) -> str:
        """ACCEPT only with no FAIL and nothing unmeasured. NOT_RUN blocks acceptance."""
        if self.failed:
            return "REJECTED"
        if self.not_run:
            return "INCOMPLETE"
        return "ACCEPTED" if self.passed else "EMPTY"

    @property
    def coverage(self) -> float:
        return 0.0 if self.total == 0 else (self.passed + self.failed) / self.total

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "not_run": self.not_run,
            "verdict": self.verdict,
            "coverage": round(self.coverage, 4),
        }


@dataclass(slots=True)
class AcceptanceReport:
    """An ordered set of criteria, summarised and renderable without a template."""

    title: str
    criteria: list[CriterionResult] = field(default_factory=list)
    generated_at: str = ""
    host: str = ""

    def add(self, criterion: CriterionResult) -> CriterionResult:
        self.criteria.append(criterion)
        return criterion

    def extend(self, criteria: Iterable[CriterionResult]) -> "AcceptanceReport":
        self.criteria.extend(criteria)
        return self

    def summarise(self) -> ReportSummary:
        passed = sum(1 for c in self.criteria if c.status.counts_as_pass)
        failed = sum(1 for c in self.criteria if c.status is CriterionStatus.FAIL)
        not_run = sum(1 for c in self.criteria if c.status is CriterionStatus.NOT_RUN)
        return ReportSummary(len(self.criteria), passed, failed, not_run)

    def by_status(self, status: CriterionStatus) -> tuple[CriterionResult, ...]:
        return tuple(c for c in self.criteria if c.status is status)

    def render(self) -> str:
        """Human-readable text. Statuses spelled out so a scanner cannot merge them."""
        stamp = self.generated_at or time.strftime("%Y-%m-%dT%H:%M:%S%z")
        lines = [
            f"# {self.title}",
            f"generated: {stamp}",
            f"host: {self.host or host_summary()}",
            "",
        ]
        for criterion in self.criteria:
            marker = criterion.status.value
            detail = criterion.measured_text
            bound = ""
            if criterion.threshold is not None:
                bound = f" (limit {criterion.direction} {criterion.threshold}{criterion.unit})"
            language = f" [{criterion.language}]" if criterion.language else ""
            lines.append(f"{marker:<8} {criterion.name}{language}: {detail}{bound}")
            if criterion.reason:
                lines.append(f"         reason: {criterion.reason}")
            if criterion.evidence and criterion.evidence.command:
                lines.append(f"         command: {criterion.evidence.command}")
        summary = self.summarise()
        lines += [
            "",
            f"PASS={summary.passed} FAIL={summary.failed} NOT_RUN={summary.not_run} total={summary.total}",
            f"verdict: {summary.verdict} (coverage {summary.coverage:.2%})",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "host": self.host,
            "summary": self.summarise().as_dict(),
            "criteria": [c.as_dict() for c in self.criteria],
        }

    def write_json(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        return path


def build_report(title: str, criteria: Sequence[CriterionResult], *, host: str = "") -> AcceptanceReport:
    """Freeze a criteria list into a report with generation metadata filled in."""
    return AcceptanceReport(
        title=title,
        criteria=list(criteria),
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        host=host or host_summary(),
    )


def not_run(name: str, reason: str, *, unit: str = "", threshold: float | None = None, language: str = "", evidence: Evidence | None = None) -> CriterionResult:
    """Constructor for the honest empty cell: never a zero, always a reason."""
    return CriterionResult(
        name=name,
        status=CriterionStatus.NOT_RUN,
        measured=None,
        threshold=threshold,
        unit=unit,
        direction="max",
        reason=reason,
        evidence=evidence,
        language=language,
    )
from ..core.errors import ConfigurationError
