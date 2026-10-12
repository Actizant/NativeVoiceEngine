"""The three performance profiles, as data with an honesty guard.

``lite`` / ``balanced`` / ``performance`` are the ceilings a run may not exceed.
The values below are *provisional placeholders*: §10 of the specification says
"do not assign arbitrary RAM thresholds without measurement", and no benchmark
has run yet, so every profile carries ``measured=False`` and
``provisional=True``. ``unmeasured_fields()`` makes which numbers are guesses
explicit, and ``require_validated()`` raises unless a benchmark has recorded
evidence - so nothing can present a profile as validated while it is a seed.

When a real benchmark produces the thresholds, ``mark_measured`` returns a new
policy with the evidence attached; the provisional seeds stay untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.errors import CapabilityUnavailable, ConfigurationError

_THRESHOLD_FIELDS = (
    "max_resident_mib",
    "max_threads",
    "max_audio_buffer_frames",
    "concurrency",
    "allowed_quantizations",
)

QUANTIZATIONS = ("fp32", "fp16", "int16", "int8")


@dataclass(frozen=True, slots=True)
class ProfilePolicy:
    name: str
    max_resident_mib: int
    max_threads: int
    allowed_quantizations: tuple[str, ...]
    allow_gpu: bool
    max_audio_buffer_frames: int
    streaming: bool
    concurrency: int
    measured: bool = False
    measured_fields: tuple[str, ...] = ()
    provisional: bool = True
    validation_evidence: str = ""
    description: str = ""

    def unmeasured_fields(self) -> list[str]:
        """Threshold field names no benchmark has actually set.

        Reported per field, because marking one threshold measured is not the same
        as validating a profile. A bool here would let a single resident-memory
        figure certify thread counts and buffer sizes that were never benchmarked.
        """
        covered = set(self.measured_fields)
        if not covered and self.measured:
            return list(_THRESHOLD_FIELDS)
        return [field for field in _THRESHOLD_FIELDS if field not in covered]

    @property
    def is_validated(self) -> bool:
        """Validated only when every threshold field has benchmark evidence behind it."""
        return (
            self.measured
            and not self.provisional
            and bool(self.validation_evidence.strip())
            and not self.unmeasured_fields()
        )

    def require_validated(self) -> None:
        if not self.is_validated:
            raise CapabilityUnavailable(
                "profile thresholds are provisional and have not been benchmarked",
                profile=self.name,
                unmeasured=self.unmeasured_fields(),
            )

    def allows_quantization(self, quantization: str) -> bool:
        return quantization in self.allowed_quantizations


LITE = ProfilePolicy(
    name="lite",
    max_resident_mib=1024,
    max_threads=2,
    allowed_quantizations=("int8",),
    allow_gpu=False,
    max_audio_buffer_frames=16000,
    streaming=False,
    concurrency=1,
    description="Smallest footprint, CPU only; the safe default when detection is partial.",
)

BALANCED = ProfilePolicy(
    name="balanced",
    max_resident_mib=3072,
    max_threads=4,
    allowed_quantizations=("int8", "int16", "fp16"),
    allow_gpu=True,
    max_audio_buffer_frames=48000,
    streaming=True,
    concurrency=2,
    description="Mid footprint, GPU where available.",
)

PERFORMANCE = ProfilePolicy(
    name="performance",
    max_resident_mib=8192,
    max_threads=8,
    allowed_quantizations=("int8", "fp16", "fp32"),
    allow_gpu=True,
    max_audio_buffer_frames=160000,
    streaming=True,
    concurrency=4,
    description="Largest footprint the box may commit; only when measured to fit.",
)

PROFILES: dict[str, ProfilePolicy] = {p.name: p for p in (LITE, BALANCED, PERFORMANCE)}

__all__ = [
    "BALANCED",
    "LITE",
    "PERFORMANCE",
    "PROFILES",
    "QUANTIZATIONS",
    "ProfilePolicy",
    "get_profile",
    "mark_measured",
    "profile_names",
]


def profile_names() -> tuple[str, ...]:
    return tuple(PROFILES)


def get_profile(name: str) -> ProfilePolicy:
    try:
        return PROFILES[name]
    except KeyError:
        raise ConfigurationError(
            "unknown performance profile", profile=name, known=list(PROFILES)
        ) from None


def mark_measured(profile: ProfilePolicy, evidence: str, **thresholds: object) -> ProfilePolicy:
    """Return a copy of ``profile`` marked measured, with the benchmark evidence.

    The seed instances are frozen and left as-is, so switching a profile to
    validated is a new value, never a mutation of the provisional record.
    """
    if not evidence.strip():
        raise ConfigurationError("marking a profile measured requires benchmark evidence", profile=profile.name)
    safe = {k: v for k, v in thresholds.items() if k in _THRESHOLD_FIELDS}
    unknown = set(thresholds) - set(safe)
    if unknown:
        raise ConfigurationError("cannot benchmark-set non-threshold fields", fields=sorted(unknown))
    covered = tuple(dict.fromkeys((*profile.measured_fields, *safe)))
    complete = len(covered) == len(_THRESHOLD_FIELDS)
    return replace(
        profile,
        **safe,
        measured=True,
        measured_fields=covered,
        provisional=not complete,
        validation_evidence=f"{profile.validation_evidence} | {evidence}".strip(" |"),
    )
