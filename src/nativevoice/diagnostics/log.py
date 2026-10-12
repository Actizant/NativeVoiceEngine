"""Structured diagnostics: in-memory ring buffer and optional local JSONL, never a network.

WHY REDACTION IS THE DEFAULT. A diagnostic sink in a voice stack is one `fields=text`
away from writing the user's transcripts to disk, and transcripts are the most
sensitive artefact the system produces (names, addresses, health, code). So recording
audio-derived or speech-derived payloads requires an explicit opt-in per field name:
anything in `redacted_fields` is replaced by a placeholder and its byte/character
length is kept instead, which is enough for debugging and useless for surveillance.

There is deliberately no network sink in this module, and none may be added: the
offline-compliance proof in `diagnostics.offline` (runtime guard plus static scan)
would be a lie if a log path could phone home. `path` is a local file; `query()` reads
the ring buffer. Nothing else is touched.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

from ..core.errors import VoiceError
from ..core.events import DiagnosticEvent

REDACTED = "[redacted]"

# Field names whose values are user content. Length is preserved so a 4-char and a
# 4000-char transcript are still distinguishable when chasing a truncation bug.
DEFAULT_REDACTED_FIELDS = frozenset(
    {
        "audio",
        "buffer",
        "chunks",
        "frame",
        "frames",
        "hypothesis",
        "pcm",
        "payload",
        "prompt",
        "reference",
        "samples",
        "text",
        "transcript",
        "utterance",
        "voice_sample",
    }
)

LEVEL_ORDER = ("debug", "info", "warn", "error")


class DiagnosticSinkError(VoiceError):
    """The JSONL destination could not be written - reported, never swallowed."""


def _level_rank(level: str) -> int:
    try:
        return LEVEL_ORDER.index(level)
    except ValueError:
        return LEVEL_ORDER.index("info")


@dataclass(slots=True)
class DiagnosticSink:
    """Ring buffer of `DiagnosticEvent`s with an optional local JSONL mirror.

    `capacity` bounds memory the way a long session needs it to; `dropped` counts the
    events that fell off, so "we have no record of that" is itself measurable rather
    than implied by an empty query result.
    """

    capacity: int = 2000
    path: str = ""
    minimum_level: str = "info"
    redact: bool = True
    redacted_fields: frozenset = DEFAULT_REDACTED_FIELDS
    keep_fields: frozenset = frozenset()
    events: deque = field(default_factory=deque)
    accepted: int = 0
    written: int = 0
    # slots=True forbids undeclared instance attributes, so the file handle is a
    # declared non-init field rather than something assigned lazily.
    _handle: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.capacity = max(1, int(self.capacity))
        self.events = deque(maxlen=self.capacity)
        if self.path:
            self._open()

    def _open(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        try:
            os.makedirs(directory, exist_ok=True)
            self._handle = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 (owned by sink)
        except OSError as exc:
            raise DiagnosticSinkError("diagnostic JSONL destination unwritable", path=self.path, error=str(exc)) from exc

    def _redacted(self, fields: dict) -> dict:
        if not self.redact:
            return dict(fields)
        out = {}
        for key, value in fields.items():
            if key in self.redacted_fields and key not in self.keep_fields:
                out[key] = REDACTED
                out[f"{key}_length"] = len(value) if hasattr(value, "__len__") else 0
            elif isinstance(value, dict):
                out[key] = self._redacted(value)
            else:
                out[key] = value
        return out

    @property
    def dropped(self) -> int:
        """Events that fell off the ring. 'Not in the buffer' must be provable, not implied."""
        return max(0, self.accepted - len(self.events))

    def record(self, event: DiagnosticEvent) -> DiagnosticEvent | None:
        """Append one event if its level passes the threshold. Returns None if filtered."""
        if _level_rank(event.level) < _level_rank(self.minimum_level):
            return None
        if not event.timestamp_ns:
            event.timestamp_ns = time.time_ns()
        stored = replace(event, fields=self._redacted(event.fields))
        self.events.append(stored)
        self.accepted += 1
        if self._handle is not None:
            self._handle.write(_line(stored) + "\n")
            self._handle.flush()
            self.written += 1
        return stored

    def emit(
        self,
        component: str,
        event: str,
        level: str = "info",
        fields: dict | None = None,
        **named,
    ) -> DiagnosticEvent | None:
        """Convenience producer: build the event and record it in one call.

        `fields` and keyword arguments are merged so a caller can pass a pre-built
        dict (an event relay) or name the fields inline, without a signature that
        forces one style.
        """
        payload = dict(fields or {})
        payload.update(named)
        return self.record(DiagnosticEvent(component=component, event=event, level=level, fields=payload))

    def extend(self, events: Iterable[DiagnosticEvent]) -> int:
        return sum(1 for e in events if self.record(e) is not None)

    def query(
        self,
        component: str = "",
        event: str = "",
        level: str = "",
        *,
        limit: int | None = None,
    ) -> tuple[DiagnosticEvent, ...]:
        """Filter the ring buffer. Empty strings mean "any", not "match nothing"."""
        rows: Iterator[DiagnosticEvent] = iter(self.events)
        matched = [
            stored
            for stored in rows
            if (not component or stored.component == component)
            and (not event or stored.event == event)
            and (not level or stored.level == level)
        ]
        if limit is not None:
            matched = matched[-limit:] if limit > 0 else []
        return tuple(matched)

    def counts(self) -> dict:
        result: dict[str, int] = {}
        for stored in self.events:
            result[stored.component] = result.get(stored.component, 0) + 1
        return result

    def to_jsonl(self) -> str:
        """Serialise the buffer, oldest first. Same shape as the file mirror."""
        return "\n".join(_line(stored) for stored in self.events) + ("\n" if self.events else "")

    def clear(self) -> None:
        self.events.clear()
        self.accepted = 0

    def close(self) -> None:
        handle = self._handle
        if handle is not None:
            handle.close()
            self._handle = None

    def __enter__(self) -> "DiagnosticSink":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _line(event: DiagnosticEvent) -> str:
    return json.dumps(
        {
            "component": event.component,
            "event": event.event,
            "level": event.level,
            "timestamp_ns": event.timestamp_ns,
            "fields": event.fields,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
