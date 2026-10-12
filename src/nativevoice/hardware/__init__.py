"""Hardware detection and performance profiles: compute, memory, disk and GPU.

This package keeps strictly to CPU / RAM / disk / GPU; audio device enumeration
lives in ``nativevoice.audio`` and is intentionally not imported here, so the
control path carries no audio coupling. Import-time dependencies are standard
library only - ``psutil`` and ``nvidia-smi`` are consulted lazily and their
absence degrades to a note, never an error.
"""

from __future__ import annotations

from .budget import ResourceBudget, Verdict, build_budget
from .detect import SystemReport, detect
from .profiles import (
    BALANCED,
    LITE,
    PERFORMANCE,
    PROFILES,
    ProfilePolicy,
    get_profile,
    mark_measured,
    profile_names,
)
from .recommend import Recommendation, recommend, select_profile

__all__ = [
    "BALANCED",
    "LITE",
    "PERFORMANCE",
    "PROFILES",
    "ProfilePolicy",
    "Recommendation",
    "ResourceBudget",
    "SystemReport",
    "Verdict",
    "build_budget",
    "detect",
    "get_profile",
    "mark_measured",
    "profile_names",
    "recommend",
    "select_profile",
]
