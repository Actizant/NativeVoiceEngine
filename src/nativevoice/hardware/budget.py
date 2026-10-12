"""Resource budget: does a model fit *this* machine under *this* profile?

A budget is the intersection of three inputs - the measured :class:`SystemReport`,
the selected :class:`ProfilePolicy` ceilings, and any explicit user limits - and
`fits` compares a model's declared or measured `ResourceRequirement` against it.
Two honesty rules are baked in:

* `requirement.measured is False` always shows up in the verdict reasons, so a
  guess from a model card is never silently treated as a measurement (the spec
  lets a declared figure gate only a suggestion);
* the verdict `explanation` names both sides' numbers, so a rejection is
  actionable rather than a bare "too big".

`switch_profile` returns a brand-new budget and never mutates the old one, which
is what lets a profile change leave an already-loaded model set intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.errors import ConfigurationError, ResourceLimitExceeded
from .detect import SystemReport
from .profiles import ProfilePolicy, get_profile

_LIMIT_KEYS = ("max_resident_mib", "max_threads", "allow_gpu")


@dataclass(frozen=True, slots=True)
class Verdict:
    ok: bool
    reasons: tuple[str, ...]
    explanation: str

    def raise_if_rejected(self, model_id: str = "") -> None:
        if not self.ok:
            raise ResourceLimitExceeded(
                "model does not fit the resource budget",
                explanation=self.explanation,
                model_id=model_id,
                reasons=self.reasons,
            )


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    report: SystemReport
    profile: ProfilePolicy
    user_limits: dict = field(default_factory=dict)

    @property
    def max_resident_mib(self) -> int:
        return int(self.user_limits.get("max_resident_mib", self.profile.max_resident_mib))

    @property
    def max_threads(self) -> int:
        return int(self.user_limits.get("max_threads", self.profile.max_threads))

    @property
    def allow_gpu(self) -> bool:
        wanted = bool(self.user_limits.get("allow_gpu", self.profile.allow_gpu))
        return wanted and self.report.has_gpu

    def switch_profile(self, name: str) -> "ResourceBudget":
        """Return a new budget on another profile; this one is unchanged."""
        return ResourceBudget(report=self.report, profile=get_profile(name), user_limits=dict(self.user_limits))

    def fits(self, requirement: object) -> Verdict:
        hard: list[str] = []
        advisory: list[str] = []
        notes: list[str] = []

        ram_mb = int(getattr(requirement, "ram_mb", 0) or 0)
        vram_mb = int(getattr(requirement, "vram_mb", 0) or 0)
        disk_mb = int(getattr(requirement, "disk_mb", 0) or 0)
        threads = int(getattr(requirement, "threads", 1) or 1)
        needs_gpu = bool(getattr(requirement, "needs_gpu", False))
        measured = bool(getattr(requirement, "measured", False))

        available = self.report.ram_available_mb
        if available is None:
            hard.append("available RAM is unmeasured, so a fit cannot be confirmed")
        else:
            ceiling = min(self.max_resident_mib, available)
            notes.append(f"RAM need {ram_mb} MiB vs budget {ceiling} MiB (profile cap {self.max_resident_mib}, free {available})")
            if ram_mb > ceiling:
                hard.append(f"RAM {ram_mb} MiB exceeds budget {ceiling} MiB")

        if vram_mb > 0 or needs_gpu:
            if not self.allow_gpu:
                hard.append(f"model wants GPU but budget has GPU disabled (profile {self.profile.name}, gpu present={self.report.has_gpu})")
            gpu_vram = self.report.gpu_vram_mb
            if vram_mb > 0:
                if gpu_vram is None:
                    hard.append("GPU VRAM is unmeasured, so the VRAM fit cannot be confirmed")
                else:
                    notes.append(f"VRAM need {vram_mb} MiB vs GPU {gpu_vram} MiB")
                    if vram_mb > gpu_vram:
                        hard.append(f"VRAM {vram_mb} MiB exceeds GPU capacity {gpu_vram} MiB")

        notes.append(f"threads need {threads} vs budget {self.max_threads}")
        if threads > self.max_threads:
            hard.append(f"threads {threads} exceed budget {self.max_threads}")

        if disk_mb > 0:
            free_disk = self.report.free_disk_mb
            if free_disk is None:
                hard.append("free disk is unmeasured, so the disk fit cannot be confirmed")
            else:
                notes.append(f"disk need {disk_mb} MiB vs free {free_disk} MiB")
                if disk_mb > free_disk:
                    hard.append(f"disk {disk_mb} MiB exceeds free space {free_disk} MiB")

        if not measured:
            advisory.append("requirement is declared, not measured on this machine")
        if not self.profile.measured:
            advisory.append(f"profile '{self.profile.name}' thresholds are provisional (unmeasured)")

        explanation = "; ".join(notes + hard + advisory) if (notes or hard or advisory) else "no constraints evaluated"
        return Verdict(ok=not hard, reasons=tuple(hard + advisory), explanation=explanation)

    def require_fits(self, requirement: object, model_id: str = "") -> Verdict:
        verdict = self.fits(requirement)
        verdict.raise_if_rejected(model_id=model_id)
        return verdict


def build_budget(
    report: SystemReport,
    profile: str = "balanced",
    *,
    max_resident_mib: int | None = None,
    max_threads: int | None = None,
    allow_gpu: bool | None = None,
) -> ResourceBudget:
    """Derive a budget from a report, a profile name and optional explicit limits."""
    policy = get_profile(profile)
    user_limits: dict = {}
    if max_resident_mib is not None:
        if max_resident_mib <= 0:
            raise ConfigurationError("max_resident_mib must be positive", value=max_resident_mib)
        user_limits["max_resident_mib"] = int(max_resident_mib)
    if max_threads is not None:
        if max_threads < 1:
            raise ConfigurationError("max_threads must be at least 1", value=max_threads)
        user_limits["max_threads"] = int(max_threads)
    if allow_gpu is not None:
        user_limits["allow_gpu"] = bool(allow_gpu)
    return ResourceBudget(report=report, profile=policy, user_limits=user_limits)
