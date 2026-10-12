"""Time, identifiers and measured-latency records."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field


def monotonic_ns() -> int:
    return time.monotonic_ns()


def new_id(prefix: str) -> str:
    """Sortable, collision-resistant identifier: timestamp plus entropy."""
    return f"{prefix}_{time.time_ns():x}{secrets.token_hex(4)}"


@dataclass(slots=True)
class Timings:
    """Elapsed spans of one operation, in milliseconds.

    `first_audio_ms` and `real_time_factor` are the two numbers §14 of the
    specification demands for synthesis, so they are computed here rather than
    re-derived by every caller from raw counters.
    """

    started_ns: int = field(default_factory=monotonic_ns)
    first_result_ns: int = 0
    finished_ns: int = 0
    audio_duration_s: float = 0.0

    def mark_first_result(self) -> None:
        if not self.first_result_ns:
            self.first_result_ns = monotonic_ns()

    def finish(self) -> "Timings":
        self.finished_ns = monotonic_ns()
        return self

    @property
    def total_ms(self) -> float:
        end = self.finished_ns or monotonic_ns()
        return (end - self.started_ns) / 1e6

    @property
    def first_result_ms(self) -> float | None:
        if not self.first_result_ns:
            return None
        return (self.first_result_ns - self.started_ns) / 1e6

    @property
    def real_time_factor(self) -> float | None:
        """Generation wall-time divided by generated audio duration. <1 is faster than real time."""
        end = self.finished_ns or monotonic_ns()
        if self.audio_duration_s <= 0:
            return None
        return ((end - self.started_ns) / 1e9) / self.audio_duration_s

    def as_dict(self) -> dict:
        return {
            "total_ms": round(self.total_ms, 3),
            "first_result_ms": None if self.first_result_ms is None else round(self.first_result_ms, 3),
            "audio_duration_s": round(self.audio_duration_s, 6),
            "real_time_factor": None if self.real_time_factor is None else round(self.real_time_factor, 4),
        }
