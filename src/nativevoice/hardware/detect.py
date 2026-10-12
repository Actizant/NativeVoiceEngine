"""Machine detection for the control path, with no third-party imports.

The specification requires the system to adapt to the real box it is installed on
- Ryzen 12C/16T, RTX 3050 Laptop, ~16 GB - but it also forbids the control path
from *requiring* a dependency. So every fact here is gathered through the
standard library with ``psutil`` as an optional accelerator: if it is present we
use it for CPU/RAM, if not we fall back to ``os.cpu_count``, a Windows
``GlobalMemoryStatusEx`` call, or ``/proc/meminfo``. When a number cannot be
obtained it is left ``None`` and a note explains why, rather than guessed.

GPU details come from ``nvidia-smi`` queried through ``subprocess``. Its absence
is a normal outcome, recorded as a reason string; detection must never raise.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

_NVIDIA_QUERY = "name,memory.total,driver_version"
_TIMEOUT_S = 3.0


@dataclass(slots=True)
class SystemReport:
    os_name: str = ""
    os_release: str = ""
    machine: str = ""
    python_version: str = ""
    cpu_physical: int | None = None
    cpu_logical: int | None = None
    ram_total_mb: int | None = None
    ram_available_mb: int | None = None
    free_disk_mb: int | None = None
    gpu_name: str = ""
    gpu_vram_mb: int | None = None
    gpu_driver: str = ""
    psutil_used: bool = False
    notes: tuple[str, ...] = field(default=())

    def missing(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in ("cpu_physical", "cpu_logical", "ram_total_mb", "ram_available_mb", "free_disk_mb", "gpu_vram_mb")
            if getattr(self, name) is None
        )

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpu_name)

    @property
    def is_partial(self) -> bool:
        """True when the facts a profile decision depends on are not all present."""
        return self.cpu_logical is None or self.ram_total_mb is None

    def note_for(self, needle: str) -> str:
        for note in self.notes:
            if needle in note:
                return note
        return ""


def _bytes_to_mb(value: int) -> int:
    return int(round(value / (1024 * 1024)))


def _try_psutil() -> tuple[dict | None, str]:
    """Import psutil lazily; the control path must not require it."""
    try:
        import psutil  # type: ignore
    except Exception as exc:  # absence, or a broken install, is not fatal
        return None, f"psutil unavailable ({exc.__class__.__name__}); using stdlib fallbacks"
    return psutil, ""


def _windows_memory() -> tuple[int, int] | None:
    if platform.system() != "Windows":
        return None
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except Exception:
        return None
    if not ok:
        return None
    return _bytes_to_mb(status.ullTotalPhys), _bytes_to_mb(status.ullAvailPhys)


def _linux_memory() -> tuple[int, int] | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            fields = {}
            for line in handle:
                parts = line.split()
                if parts:
                    fields[parts[0].rstrip(":")] = int(parts[1])
    except OSError:
        return None
    total_kb = fields.get("MemTotal")
    if total_kb is None:
        return None
    avail_kb = fields.get("MemAvailable", fields.get("MemFree", 0))
    return int(total_kb // 1024), int(avail_kb // 1024)


def _detect_memory(notes: list[str]) -> tuple[int | None, int | None]:
    psutil, reason = _try_psutil()
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return _bytes_to_mb(vm.total), _bytes_to_mb(vm.available)
        except Exception as exc:
            notes.append(f"psutil.virtual_memory failed ({exc.__class__.__name__}); using fallback")
    windows = _windows_memory()
    if windows is not None:
        return windows
    linux = _linux_memory()
    if linux is not None:
        return linux
    notes.append("memory size could not be determined on this platform")
    return None, None


def _detect_cpu(notes: list[str]) -> tuple[int | None, int | None, bool]:
    psutil, _ = _try_psutil()
    if psutil is not None:
        try:
            return psutil.cpu_count(logical=False), psutil.cpu_count(logical=True), True
        except Exception as exc:
            notes.append(f"psutil.cpu_count failed ({exc.__class__.__name__}); using os.cpu_count")
    logical = os.cpu_count()
    if logical is None:
        notes.append("cpu count could not be determined")
    else:
        notes.append("physical core count unknown without psutil; logical cores used")
    return None, logical, False


def _detect_gpu(notes: list[str]) -> tuple[str, int | None, str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
    except FileNotFoundError:
        notes.append("nvidia-smi not found: GPU details unavailable")
        return "", None, ""
    except Exception as exc:
        notes.append(f"nvidia-smi query failed ({exc.__class__.__name__}): GPU details unavailable")
        return "", None, ""
    if completed.returncode != 0:
        notes.append(f"nvidia-smi returned {completed.returncode}: GPU details unavailable")
        return "", None, ""
    line = (completed.stdout or "").strip().splitlines()
    if not line:
        notes.append("nvidia-smi produced no output: GPU details unavailable")
        return "", None, ""
    fields = [column.strip() for column in line[0].split(",")]
    name = fields[0] if fields else ""
    vram: int | None = None
    if len(fields) > 1:
        try:
            vram = int(float(fields[1]))
        except ValueError:
            vram = None
    driver = fields[2] if len(fields) > 2 else ""
    return name, vram, driver


def _detect_disk(notes: list[str]) -> int | None:
    try:
        usage = shutil.disk_usage(os.getcwd())
    except OSError as exc:
        notes.append(f"free disk space could not be measured ({exc.__class__.__name__})")
        return None
    return _bytes_to_mb(usage.free)


def detect() -> SystemReport:
    """Gather compute, memory, disk and GPU facts without raising or importing heavy deps."""
    notes: list[str] = []
    _, psutil_reason = _try_psutil()
    if psutil_reason:
        notes.append(psutil_reason)
    cpu_physical, cpu_logical, psutil_used = _detect_cpu(notes)
    ram_total, ram_available = _detect_memory(notes)
    gpu_name, gpu_vram, gpu_driver = _detect_gpu(notes)
    free_disk = _detect_disk(notes)
    return SystemReport(
        os_name=platform.system(),
        os_release=platform.release(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        cpu_physical=cpu_physical,
        cpu_logical=cpu_logical,
        ram_total_mb=ram_total,
        ram_available_mb=ram_available,
        free_disk_mb=free_disk,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram,
        gpu_driver=gpu_driver,
        psutil_used=psutil_used,
        notes=tuple(notes),
    )
