"""Capability matrix: language x task, with declared and measured never merged.

WHY THE DISTINCTION IS THE WHOLE POINT. A registry that stores "supported: true"
cannot express the difference between "this machine ran this language yesterday and
produced WER 0.18" and "the upstream model card mentions this language". The studied
reference projects got exactly this wrong, and the consequence was users routing
Bengali audio into a model that had never been evaluated on it. So a cell keeps the
`CapabilityStatus`, the metric name/value pair if there is one, the engine, the model
id, the measurement date and the evidence pointer, and rendering uses different
symbols per status. Any cell without a measured metric is stated as unmeasured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..core.language import Capability, CapabilityStatus, Task

STATUS_SYMBOL = {
    CapabilityStatus.MEASURED: "M",
    CapabilityStatus.DECLARED: "d",
    CapabilityStatus.PARTIAL: "P",
    CapabilityStatus.MEASURED_FAILED: "!",
    CapabilityStatus.UNSUPPORTED: "x",
    CapabilityStatus.NO_MODEL: "-",
}


@dataclass(frozen=True, slots=True)
class Cell:
    """One language/task intersection as reported by an engine or an eval run."""

    language: str
    task: str
    status: CapabilityStatus
    engine: str = ""
    model_id: str = ""
    metric_name: str = ""
    metric_value: float | None = None
    measured_on: str = ""
    evidence: str = ""
    limitations: tuple[str, ...] = ()

    @classmethod
    def from_capability(cls, language: str, capability: Capability) -> "Cell":
        return cls(
            language=language,
            task=capability.task.value,
            status=CapabilityStatus(capability.status),
            engine=capability.engine,
            model_id=capability.model_id,
            metric_name=capability.metric_name,
            metric_value=capability.metric_value,
            measured_on=capability.measured_on,
            evidence=capability.evidence,
            limitations=tuple(capability.limitations),
        )

    @property
    def symbol(self) -> str:
        return STATUS_SYMBOL[self.status]

    @property
    def has_metric(self) -> bool:
        """A metric on a DECLARED cell is a vendor number, so it is labelled as such."""
        return self.metric_value is not None and bool(self.metric_name)

    def describe(self) -> str:
        if self.status is CapabilityStatus.MEASURED:
            if self.has_metric:
                return f"{self.metric_name}={self.metric_value:.4f} on {self.measured_on or 'unknown date'} via {self.engine or 'unknown engine'}"
            # A MEASURED cell with no number is a bookkeeping failure, not a pass:
            # saying "awaiting a model" here would be actively false.
            return "claimed measured but no metric value was recorded"
        if self.status is CapabilityStatus.DECLARED:
            claimed = f" (upstream figure {self.metric_name}={self.metric_value:.4f}, unverified)" if self.has_metric else ""
            return f"declared only{claimed}: not evaluated on this machine"
        if self.status is CapabilityStatus.PARTIAL:
            gaps = "; ".join(self.limitations) or "gaps unstated"
            return f"partial: {gaps}"
        if self.status is CapabilityStatus.MEASURED_FAILED:
            number = f"{self.metric_name}={self.metric_value:.4f}" if self.has_metric else "no metric recorded"
            gaps = "; ".join(self.limitations) or "reason unstated"
            return f"measured and failed the bar ({number} on {self.measured_on or 'unknown date'}): {gaps}"
        if self.status is CapabilityStatus.UNSUPPORTED:
            return "unsupported by the installed engine"
        return "awaiting an approved model"

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "task": self.task,
            "status": self.status.value,
            "engine": self.engine,
            "model_id": self.model_id,
            "metric_name": self.metric_name or None,
            "metric_value": self.metric_value,
            "measured_on": self.measured_on or None,
            "evidence": self.evidence or None,
            "limitations": list(self.limitations),
            "described_as": self.describe(),
        }


@dataclass(frozen=True, slots=True)
class MatrixVerdict:
    implemented: int
    measured: int
    declared: int
    partial: int
    unsupported: int
    measured_failed: int = 0
    awaiting_model: int = 0
    total: int = 0

    @property
    def unmeasured(self) -> int:
        """Everything that is not MEASURED. This is the number a report should lead with."""
        return self.total - self.measured

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "implemented": self.implemented,
            "measured": self.measured,
            "declared_only": self.declared,
            "partial": self.partial,
            "unsupported": self.unsupported,
            "measured_failed": self.measured_failed,
            "awaiting_model": self.awaiting_model,
            "unmeasured": self.unmeasured,
        }


@dataclass(frozen=True, slots=True)
class CapabilityMatrix:
    languages: tuple[str, ...]
    tasks: tuple[str, ...]
    cells: tuple[Cell, ...]

    def cell(self, language: str, task: str) -> Cell | None:
        for entry in self.cells:
            if entry.language == language and entry.task == task:
                return entry
        return None

    def verdict(self) -> MatrixVerdict:
        measured = sum(1 for c in self.cells if c.status is CapabilityStatus.MEASURED)
        declared = sum(1 for c in self.cells if c.status is CapabilityStatus.DECLARED)
        partial = sum(1 for c in self.cells if c.status is CapabilityStatus.PARTIAL)
        unsupported = sum(1 for c in self.cells if c.status is CapabilityStatus.UNSUPPORTED)
        measured_failed = sum(1 for c in self.cells if c.status is CapabilityStatus.MEASURED_FAILED)
        awaiting = sum(1 for c in self.cells if c.status is CapabilityStatus.NO_MODEL)
        return MatrixVerdict(
            implemented=measured + partial,
            measured=measured,
            declared=declared,
            partial=partial,
            unsupported=unsupported,
            measured_failed=measured_failed,
            awaiting_model=awaiting,
            total=len(self.cells),
        )

    def measured_pairs(self, task: str) -> tuple[Cell, ...]:
        return tuple(c for c in self.cells if c.task == task and c.status is CapabilityStatus.MEASURED)

    def render_text(self) -> str:
        """Grid plus a legend. Symbols differ per status; DECLARED never reads as MEASURED."""
        header = "language".ljust(14) + "".join(task.ljust(14) for task in self.tasks)
        lines = [header, "-" * len(header)]
        for language in self.languages:
            row = language.ljust(14)
            for task in self.tasks:
                entry = self.cell(language, task)
                row += ("(none)").ljust(14) if entry is None else entry.symbol.ljust(14)
            lines.append(row)
        lines.append("")
        lines.append(
            "legend: M=measured here  d=declared by upstream, unverified  P=partial  "
            "!=measured and failed the bar  x=unsupported  -=awaiting model"
        )
        verdict = self.verdict()
        lines.append(
            f"totals: measured={verdict.measured} declared={verdict.declared} partial={verdict.partial} "
            f"unsupported={verdict.unsupported} awaiting_model={verdict.awaiting_model} unmeasured={verdict.unmeasured}/{verdict.total}"
        )
        detail = [f"  {c.language}/{c.task}: {c.describe()}" for c in self.cells if not (c.status is CapabilityStatus.MEASURED and c.has_metric)]
        if detail:
            lines.append("cells that are not measured:")
            lines.extend(detail)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "languages": list(self.languages),
            "tasks": list(self.tasks),
            "verdict": self.verdict().as_dict(),
            "symbols": {status.value: symbol for status, symbol in STATUS_SYMBOL.items()},
            "cells": [c.as_dict() for c in self.cells],
        }


def build_matrix(capabilities: Iterable[Capability], *, tasks: Iterable[Task | str] | None = None) -> CapabilityMatrix:
    """Collapse (language, Capability) pairs into the matrix view.

    `tasks` lets a caller pin the column set so an absent capability shows as `(none)`
    rather than vanishing - the deletion of a column is how "no Bengali TTS" hides.
    """
    entries: list[Cell] = []
    languages: list[str] = []
    for language, capability in capabilities:
        tag = str(language)
        if tag not in languages:
            languages.append(tag)
        entries.append(Cell.from_capability(tag, capability))
    columns = [t.value if isinstance(t, Task) else str(t) for t in tasks] if tasks is not None else []
    for entry in entries:
        if entry.task not in columns:
            columns.append(entry.task)
    return CapabilityMatrix(tuple(languages), tuple(columns), tuple(entries))
