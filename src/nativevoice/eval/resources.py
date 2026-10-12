"""Process memory and CPU measurement without psutil.

WHY NO DEPENDENCY. The specification forbids new third-party code on the control
path, and a measurement library is exactly the kind of dependency that quietly
becomes required at import time. Everything here goes through `ctypes` to the OS
calls that already exist: `K32GetProcessMemoryInfo`/`GetProcessMemoryInfo` and
`GetProcessTimes` on Windows, `getrusage` plus `/proc/self/status` on POSIX.

The contract that makes the rest of the evaluation layer trustworthy is
`measured: bool`. When a call is missing, the structure fails, or the platform is
unknown, the reading comes back `measured=False` with the OS error text in `reason`
and numeric fields as `None`. Nothing in this module substitutes an estimate for a
measurement, so "peak RSS 1.0 GB" can only ever appear if the kernel said so.
"""

from __future__ import annotations

import ctypes
import os
import platform
import threading
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")

_IS_WINDOWS = os.name == "nt"
_TICKS_PER_SECOND = 100_000_000  # Windows GetProcessTimes reports 100 ns intervals


@dataclass(frozen=True, slots=True)
class MemoryReading:
    """Peak and current working set of this process, or why neither is known."""

    measured: bool
    peak_bytes: int | None
    current_bytes: int | None
    source: str
    reason: str = ""

    @property
    def available(self) -> bool:
        """Alias of `measured`, spelled for report writers."""
        return self.measured

    @property
    def peak_mb(self) -> float | None:
        return None if self.peak_bytes is None else self.peak_bytes / (1024 * 1024)

    @property
    def current_mb(self) -> float | None:
        return None if self.current_bytes is None else self.current_bytes / (1024 * 1024)

    def as_dict(self) -> dict:
        return {
            "measured": self.measured,
            "peak_bytes": self.peak_bytes,
            "peak_mb": None if self.peak_mb is None else round(self.peak_mb, 3),
            "current_bytes": self.current_bytes,
            "current_mb": None if self.current_mb is None else round(self.current_mb, 3),
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CpuTimeReading:
    """User and kernel CPU time consumed by this process. Never wall-clock."""

    measured: bool
    user_s: float | None
    system_s: float | None
    source: str
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.measured

    @property
    def total_s(self) -> float | None:
        if self.user_s is None and self.system_s is None:
            return None
        return (self.user_s or 0.0) + (self.system_s or 0.0)

    def as_dict(self) -> dict:
        return {
            "measured": self.measured,
            "user_s": None if self.user_s is None else round(self.user_s, 6),
            "system_s": None if self.system_s is None else round(self.system_s, 6),
            "total_s": None if self.total_s is None else round(self.total_s, 6),
            "source": self.source,
            "reason": self.reason,
        }


def _failure(kind: str, reason: str):
    if kind == "memory":
        return MemoryReading(False, None, None, f"{platform.system().lower()}:{'psapi' if _IS_WINDOWS else 'rusage'}", reason)
    return CpuTimeReading(False, None, None, f"{platform.system().lower()}:cpu", reason)


class _ProcessMemoryCountersEx(ctypes.Structure):
    """Layout of Windows PROCESS_MEMORY_COUNTERS_EX; a wrong field order returns garbage."""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _windows_kernel32():
    """kernel32 with last-error capture; `use_last_error` is what makes failures legible."""
    import ctypes.wintypes as wintypes

    if not hasattr(_windows_kernel32, "_cache"):
        dll = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = wintypes.HANDLE
        current = dll.GetCurrentProcess
        current.restype = handle
        current.argtypes = []
        _windows_kernel32._cache = (dll, current())
    return _windows_kernel32._cache


def _windows_memory() -> MemoryReading:
    import ctypes.wintypes as wintypes

    try:
        kernel32, process = _windows_kernel32()
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        function = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        source = "windows:k32GetProcessMemoryInfo"
        if function is None:
            function = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            source = "windows:getProcessMemoryInfo"
        function.restype = wintypes.BOOL
        # argtypes are not optional here: without them ctypes marshals the pseudo-handle
        # as a 32-bit int and the call fails with ERROR_INVALID_HANDLE on 64-bit Windows.
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCountersEx), wintypes.DWORD]
        if not function(process, ctypes.byref(counters), counters.cb):
            return _failure("memory", f"GetProcessMemoryInfo failed with error {ctypes.get_last_error()}")
        return MemoryReading(True, int(counters.PeakWorkingSetSize), int(counters.WorkingSetSize), source)
    except (OSError, AttributeError, ValueError) as exc:
        return _failure("memory", f"Windows memory API unavailable: {exc}")


def _proc_status_value(name: str) -> int | None:
    """`VmHWM`/`VmRSS` from /proc/self/status, in bytes; None when the file lies flat."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(name):
                    parts = line.split()
                    if len(parts) >= 2:
                        value = int(parts[1])
                        return value * 1024 if line.rstrip().endswith("kB") else value
    except OSError:
        return None
    return None


def _rusage_maxrss() -> tuple[int | None, str]:
    """(peak bytes, error text). `ru_maxrss` is kilobytes on Linux, bytes on macOS.

    Other POSIX systems are not documented here, so the unit stays unknown and the
    reading is reported unavailable rather than multiplied by a hopeful 1024.
    """
    try:
        import resource
    except ImportError as exc:
        return None, f"resource module unavailable: {exc}"
    system = platform.system()
    try:
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except OSError as exc:
        return None, f"getrusage failed: {exc}"
    if system == "Linux":
        return raw * 1024, ""
    if system == "Darwin":
        return raw, ""
    return None, f"ru_maxrss unit is not defined for {system}; refusing to guess a scale"


def _posix_memory() -> MemoryReading:
    peak = _proc_status_value("VmHWM:")
    current = _proc_status_value("VmRSS:")
    source = "posix:/proc/self/status"
    if peak is None:
        raw, error = _rusage_maxrss()
        if raw is None:
            return _failure("memory", error or "no memory source available")
        peak = raw
        source = "posix:getrusage"
        if current is None:
            current = raw
    if peak is None:
        return _failure("memory", "no memory source returned a peak working set")
    return MemoryReading(True, int(peak), None if current is None else int(current), source)


def read_memory() -> MemoryReading:
    """Peak RSS of this process, via the platform's own API."""
    return _windows_memory() if _IS_WINDOWS else _posix_memory()


def _windows_cpu() -> CpuTimeReading:
    import ctypes.wintypes as wintypes

    try:
        kernel32, process = _windows_kernel32()
        creation = ctypes.c_uint64()
        exit_ = ctypes.c_uint64()
        kernel = ctypes.c_uint64()
        user = ctypes.c_uint64()
        function = kernel32.GetProcessTimes
        function.restype = wintypes.BOOL
        quad = ctypes.POINTER(ctypes.c_uint64)
        function.argtypes = [wintypes.HANDLE, quad, quad, quad, quad]
        if not function(process, ctypes.byref(creation), ctypes.byref(exit_), ctypes.byref(kernel), ctypes.byref(user)):
            return _failure("cpu", f"GetProcessTimes failed with error {ctypes.get_last_error()}")
        return CpuTimeReading(True, user.value / _TICKS_PER_SECOND, kernel.value / _TICKS_PER_SECOND, "windows:getProcessTimes")
    except (OSError, AttributeError, ValueError) as exc:
        return _failure("cpu", f"Windows CPU-time API unavailable: {exc}")


def _posix_cpu() -> CpuTimeReading:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return CpuTimeReading(True, float(usage.ru_utime), float(usage.ru_stime), "posix:getrusage")
    except (ImportError, OSError, ValueError) as exc:
        return _failure("cpu", f"POSIX CPU-time API unavailable: {exc}")


def read_cpu_time() -> CpuTimeReading:
    """Cumulative user+system CPU seconds for this process."""
    return _windows_cpu() if _IS_WINDOWS else _posix_cpu()


def cpu_utilisation(start: CpuTimeReading, end: CpuTimeReading, wall_s: float) -> float | None:
    """CPU fraction over a span: CPU delta / wall delta. None if either side unmeasured.

    Deliberately not a percentage of one core or of all cores: the divisor is the wall
    span, so 1.0 means "one core busy the whole time". Multi-core work can exceed 1.0,
    and reporting that as ">100%" is the honest reading rather than a normalised lie.
    """
    if not (start.measured and end.measured) or wall_s <= 0:
        return None
    delta = (end.total_s or 0.0) - (start.total_s or 0.0)
    return delta / wall_s


@dataclass(frozen=True, slots=True)
class ResourceSample:
    at_ns: int
    memory: MemoryReading
    cpu: CpuTimeReading


@dataclass(frozen=True, slots=True)
class ResourceRun:
    """What a `Sampler` observed while the callable ran, plus what it could not see."""

    samples: tuple[ResourceSample, ...]
    peak_bytes: int | None
    cpu_delta_s: float | None
    wall_s: float
    polls: int
    measured: bool
    reasons: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.measured

    @property
    def peak_mb(self) -> float | None:
        return None if self.peak_bytes is None else self.peak_bytes / (1024 * 1024)

    def as_dict(self) -> dict:
        return {
            "measured": self.measured,
            "polls": self.polls,
            "wall_s": round(self.wall_s, 6),
            "peak_bytes": self.peak_bytes,
            "peak_mb": None if self.peak_mb is None else round(self.peak_mb, 3),
            "cpu_delta_s": None if self.cpu_delta_s is None else round(self.cpu_delta_s, 6),
            "cpu_utilisation": None if self.cpu_utilisation is None else round(self.cpu_utilisation, 4),
            "reasons": list(self.reasons),
        }

    @property
    def cpu_utilisation(self) -> float | None:
        if self.cpu_delta_s is None or self.wall_s <= 0:
            return None
        return self.cpu_delta_s / self.wall_s

    @property
    def sampled(self) -> bool:
        """True when at least one poll succeeded: polling failure is not zero usage."""
        return self.polls > 0


class Sampler:
    """Polls memory/CPU on a background thread while a callable runs.

    The peak that matters on Windows and Linux is the process's own high-water mark,
    which the OS already tracks; polling adds visibility into *when* it happened and
    gives a CPU delta for the span. Polling alone under-reads a spike between
    intervals, so `ResourceRun` reports both the OS peak and the observed samples.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        if interval_s <= 0:
            raise ConfigurationError("sampler interval must be positive")
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[ResourceSample] = []
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            sample = ResourceSample(time.monotonic_ns(), read_memory(), read_cpu_time())
            with self._lock:
                self._samples.append(sample)
            self._stop.wait(self.interval_s)

    def run(self, fn: Callable[[], T], *args, **kwargs) -> "tuple[T, ResourceRun]":
        """Execute `fn` while sampling. Exceptions from `fn` propagate unrecorded.

        Sampling teardown happens in `finally`, including when `fn` raises, because a
        leaked poller thread would contaminate the next benchmark's CPU numbers.
        """
        with self._lock:
            self._samples = []
        self._stop.clear()
        first = read_cpu_time()
        self._thread = threading.Thread(target=self._poll, name="ultron-resource-sampler", daemon=True)
        self._thread.start()
        started = time.monotonic()
        try:
            result = fn(*args, **kwargs)
        finally:
            wall = time.monotonic() - started
            self._stop.set()
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        last = read_cpu_time()
        with self._lock:
            collected = list(self._samples)
        reasons = []
        if not first.measured or not last.measured:
            reasons.append(f"cpu baseline unmeasured: {first.reason or last.reason}")
        peaks = [s.memory.peak_bytes for s in collected if s.memory.measured and s.memory.peak_bytes is not None]
        if collected and not peaks:
            reasons.append("every memory poll failed: " + (collected[-1].memory.reason or "unknown"))
        observed_peak = max(peaks, default=None)
        os_peak = read_memory()
        if os_peak.measured:
            observed_peak = max(filter(None, [observed_peak, os_peak.peak_bytes]))
        elif not reasons:
            reasons.append("final memory read failed: " + (os_peak.reason or "unknown"))
        cpu_delta = None
        if first.measured and last.measured:
            cpu_delta = (last.total_s or 0.0) - (first.total_s or 0.0)
        return result, ResourceRun(
            samples=tuple(collected),
            peak_bytes=observed_peak,
            cpu_delta_s=cpu_delta,
            wall_s=wall,
            polls=len(collected),
            measured=bool(collected) or (os_peak.measured and cpu_delta is not None),
            reasons=tuple(reasons),
        )
from ..core.errors import ConfigurationError
