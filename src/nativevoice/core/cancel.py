"""Cancellation primitives.

Streaming speech work is long-running and externally interruptible: the user
speaks over ULTRON, or a newer answer supersedes an older one. A bare boolean
is not enough for that, because queued audio can still reach the DAC after the
flag flips. Two rules are enforced here:

1. Every generator/iterator in this package checks its token at each yield
   boundary (`guard`), so cancellation is observed before more work escapes.
2. Cancellation runs registered cleanups exactly once, in reverse order, so a
   playback stream is stopped rather than merely abandoned.
"""

from __future__ import annotations

import threading
from typing import Callable, Iterable, TypeVar

from .errors import CancellationRequested

T = TypeVar("T")

Cleanup = Callable[[], None]


class CancellationToken:
    """A one-way, thread-safe cancel signal with deterministic cleanup."""

    __slots__ = ("_event", "_lock", "_reason", "_cleanups", "_children")

    def __init__(self, reason: str | None = None) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = reason
        self._cleanups: list[Cleanup] = []
        self._children: list[CancellationToken] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "cancelled") -> bool:
        """Signal cancellation and run cleanups. True if this call did it."""
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            cleanups = list(reversed(self._cleanups))
            children = list(self._children)
            self._cleanups.clear()
        self._event.set()
        for cleanup in cleanups:
            try:
                cleanup()
            except Exception:  # a broken cleanup must not strand the rest
                pass
        for child in children:
            child.cancel(f"parent:{reason}")
        return True

    def on_cleanup(self, callback: Cleanup) -> Cleanup:
        """Register teardown. Fires immediately if already cancelled."""
        with self._lock:
            if self._event.is_set():
                already = True
            else:
                already = False
                self._cleanups.append(callback)
        if already:
            callback()
        return callback

    def link(self, child: CancellationToken) -> None:
        with self._lock:
            self._children.append(child)
            cascade = self._event.is_set()
        if cascade:
            child.cancel("parent:pre-cancelled")

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancellationRequested("operation cancelled", reason=self._reason)

    def sleep(self, seconds: float) -> bool:
        """Interruptible sleep. True if it returned early because of cancel."""
        return self._event.wait(seconds)


def guard(token: CancellationToken | None) -> None:
    if token is not None:
        token.raise_if_cancelled()


def guarded(
    items: Iterable[T], token: CancellationToken | None
) -> "Iterable[T]":
    """Yield items, observing cancellation at every boundary."""
    for item in items:
        guard(token)
        yield item
