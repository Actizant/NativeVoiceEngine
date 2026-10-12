"""Profile recommendation from a detected machine.

`recommend` picks the most demanding profile the box can host, using only
*measured* facts from :func:`detect.detect` (total RAM, core count, GPU presence)
compared against the profile ceilings - and it flags the result as provisional,
because those ceilings are themselves unmeasured seeds. It degrades to ``lite``
whenever detection is partial, so an unknown machine never gets an aggressive
default.

Selecting or switching a profile is deliberately a pure operation: `select_profile`
hands back a policy and the resulting :class:`ResourceBudget` produces a *new*
budget on switch, so changing the profile can never invalidate an already-loaded
set of models.
"""

from __future__ import annotations

from dataclasses import dataclass

from .budget import ResourceBudget, build_budget
from .detect import SystemReport
from .profiles import ProfilePolicy, get_profile, profile_names

_ORDER = ("lite", "balanced", "performance")
_RANK = {name: index for index, name in enumerate(_ORDER)}


def select_profile(name: str) -> ProfilePolicy:
    """Manual override entry point: resolve a profile name to its policy."""
    return get_profile(name)


def _rank(name: str) -> int:
    return _RANK.get(name, 0)


@dataclass(frozen=True, slots=True)
class Recommendation:
    profile: ProfilePolicy
    runner_up: ProfilePolicy | None
    reasons: tuple[str, ...]
    budget: ResourceBudget
    provisional: bool = True

    @property
    def profile_name(self) -> str:
        return self.profile.name

    def switch_profile(self, name: str) -> "Recommendation":
        """Re-point the recommendation at another profile without touching loaded models."""
        policy = get_profile(name)
        return Recommendation(
            profile=policy,
            runner_up=self.profile if policy.name != self.profile.name else self.runner_up,
            reasons=tuple(self.reasons) + (f"manually overridden to '{name}'",),
            budget=self.budget.switch_profile(name),
            provisional=not policy.measured,
        )


def _supported(policy: ProfilePolicy, report: SystemReport) -> tuple[bool, str]:
    cores = report.cpu_physical if report.cpu_physical else report.cpu_logical
    if cores is not None and policy.max_threads > cores:
        return False, f"'{policy.name}' wants {policy.max_threads} threads but only {cores} cores detected"
    if report.ram_total_mb is not None and policy.max_resident_mib > report.ram_total_mb:
        return False, f"'{policy.name}' caps {policy.max_resident_mib} MiB which exceeds total RAM {report.ram_total_mb} MiB"
    return True, f"'{policy.name}' within detected cores={cores}, ram_total={report.ram_total_mb} MiB"


def recommend(
    report: SystemReport,
    manifests: tuple | list = (),
    *,
    max_resident_mib: int | None = None,
    max_threads: int | None = None,
    allow_gpu: bool | None = None,
) -> Recommendation:
    reasons: list[str] = []

    required_floor = 0
    for manifest in manifests:
        floor = _rank(getattr(manifest, "minimum_profile", "lite"))
        if floor > required_floor:
            required_floor = floor

    if report.is_partial:
        reasons.append(
            "detection is partial (" + ", ".join(report.missing()) + "); degrading to 'lite' rather than guessing a larger footprint"
        )
        chosen_name = "lite"
    else:
        chosen_name = "lite"
        for candidate in reversed(_ORDER):
            policy = get_profile(candidate)
            ok, note = _supported(policy, report)
            reasons.append(note)
            if ok:
                chosen_name = candidate
                break
        else:
            reasons.append("no profile fully supported; using 'lite' as the safest available")

    chosen = get_profile(chosen_name)
    if required_floor > _rank(chosen_name):
        reasons.append(
            f"installed manifests request at least '{_ORDER[required_floor]}', which the detected machine does not support; "
            "recommending the highest safe profile and expecting fallback"
        )

    if report.is_partial:
        runner_up_name = "lite"
    else:
        lower = [n for n in _ORDER if _rank(n) < _rank(chosen_name)]
        runner_up_name = lower[-1] if lower else chosen_name
    runner_up = get_profile(runner_up_name) if runner_up_name != chosen_name else None

    if not chosen.measured:
        reasons.append("profile thresholds are provisional (unmeasured); this is a suggestion, not a validated assignment")

    budget = build_budget(
        report,
        chosen_name,
        max_resident_mib=max_resident_mib,
        max_threads=max_threads,
        allow_gpu=allow_gpu,
    )
    return Recommendation(
        profile=chosen,
        runner_up=runner_up,
        reasons=tuple(reasons),
        budget=budget,
        provisional=not chosen.measured,
    )


__all__ = ["Recommendation", "profile_names", "recommend", "select_profile"]
