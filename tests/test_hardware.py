"""Hardware detection and profiles: honesty guards and pure budget switching."""

from __future__ import annotations

import subprocess
import sys

import pytest

from nativevoice.core.errors import (
    CapabilityUnavailable,
    ConfigurationError,
    ResourceLimitExceeded,
)
from nativevoice.hardware.budget import build_budget
from nativevoice.hardware.detect import SystemReport, detect
from nativevoice.hardware.profiles import (
    BALANCED,
    LITE,
    PERFORMANCE,
    PROFILES,
    get_profile,
    mark_measured,
    profile_names,
)
from nativevoice.hardware.recommend import recommend, select_profile
from nativevoice.models.api import ResourceRequirement

# the re-exported ``detect`` function shadows the submodule on the package, so
# reach the module object itself for subprocess monkeypatching
detect_mod = sys.modules["nativevoice.hardware.detect"]

FULL_REPORT = SystemReport(
    os_name="Windows",
    machine="AMD64",
    cpu_physical=12,
    cpu_logical=16,
    ram_total_mb=16085,
    ram_available_mb=9000,
    free_disk_mb=300000,
    gpu_name="NVIDIA GeForce RTX 3050 Laptop GPU",
    gpu_vram_mb=4096,
    gpu_driver="617.42",
)

PARTIAL_REPORT = SystemReport(os_name="Linux", machine="x86_64")


# --- detect ------------------------------------------------------------------

def test_detect_returns_report_without_raising():
    report = detect()
    assert report.os_name
    assert report.python_version
    assert isinstance(report.notes, tuple)


def test_nvidia_smi_absence_is_tolerated(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not on PATH")

    monkeypatch.setattr(detect_mod.subprocess, "run", fake_run)
    report = detect()
    assert report.gpu_name == ""
    assert report.gpu_vram_mb is None
    assert any("nvidia-smi" in n for n in report.notes)


def test_nvidia_smi_nonzero_exit_is_tolerated(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(detect_mod.subprocess, "run", fake_run)
    report = detect()
    assert report.gpu_name == ""
    assert any("nvidia-smi" in n for n in report.notes)


def test_parse_gpu_line_shape(monkeypatch):
    class Result:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 617.42\n"
        stderr = ""

    monkeypatch.setattr(detect_mod.subprocess, "run", lambda *a, **k: Result())
    report = detect()
    assert report.gpu_name == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert report.gpu_vram_mb == 4096
    assert report.gpu_driver == "617.42"


def test_control_path_imports_without_third_party(monkeypatch):
    # A missing psutil must not break detection; it degrades to stdlib fallbacks.
    monkeypatch.setitem(__import__("sys").modules, "psutil", None)
    report = detect()
    assert report.os_name  # still populated
    assert any("psutil" in n for n in report.notes)


# --- profiles: honesty guard -------------------------------------------------

def test_all_profiles_ship_unmeasured():
    for policy in (LITE, BALANCED, PERFORMANCE):
        assert policy.measured is False
        assert policy.provisional is True
        assert policy.unmeasured_fields()


def test_lite_ships_cpu_only_no_gpu():
    assert LITE.allow_gpu is False


def test_profile_not_validated_without_evidence():
    assert LITE.is_validated is False
    with pytest.raises(CapabilityUnavailable):
        LITE.require_validated()


def test_marking_one_threshold_does_not_validate_the_profile():
    """Benchmarking resident memory says nothing about buffer size or concurrency.

    Marking a profile validated on a partial run is precisely how a provisional
    number turns into an advertised one, so the policy refuses it.
    """
    partial = mark_measured(LITE, "benchmark run 2026-10-12", max_resident_mib=1400, max_threads=3)
    assert partial.measured is True
    assert partial.max_resident_mib == 1400
    assert partial.measured_fields == ("max_resident_mib", "max_threads")
    assert partial.provisional is True
    assert partial.is_validated is False
    assert "max_audio_buffer_frames" in partial.unmeasured_fields()
    assert "concurrency" in partial.unmeasured_fields()
    with pytest.raises(CapabilityUnavailable):
        partial.require_validated()
    assert LITE.max_resident_mib != 1400  # the provisional seed is never mutated


def test_only_a_complete_benchmark_set_validates_a_profile():
    from nativevoice.hardware.profiles import _THRESHOLD_FIELDS, get_profile

    seed = get_profile("balanced")
    values = {
        "max_resident_mib": 2048,
        "max_threads": 4,
        "allowed_quantizations": ("int8", "fp32"),
        "max_audio_buffer_frames": 48000,
        "concurrency": 2,
    }
    missing = set(_THRESHOLD_FIELDS) - set(values)
    assert not missing, f"the fixture must cover every threshold: {sorted(missing)}"
    validated = mark_measured(seed, "benchmark run 2026-10-12 host=rtx3050", **values)
    assert validated.unmeasured_fields() == []
    assert validated.provisional is False
    assert validated.is_validated is True
    validated.require_validated()


def test_mark_measured_requires_evidence():
    with pytest.raises(ConfigurationError):
        mark_measured(LITE, "   ")


def test_profile_registry_and_lookup():
    assert set(profile_names()) == {"lite", "balanced", "performance"}
    assert get_profile("performance") is PERFORMANCE
    with pytest.raises(ConfigurationError):
        get_profile("turbo")


# --- budget ------------------------------------------------------------------

def test_budget_fits_within_lite():
    b = build_budget(FULL_REPORT, "lite")
    verdict = b.fits(ResourceRequirement(ram_mb=200, threads=1, measured=True))
    assert verdict.ok is True
    assert "RAM need" in verdict.explanation
    assert "200" in verdict.explanation


def test_budget_rejects_oversized_model_with_numbers():
    b = build_budget(FULL_REPORT, "lite")
    verdict = b.fits(ResourceRequirement(ram_mb=8000, threads=1, measured=True))
    assert verdict.ok is False
    assert "8000" in verdict.explanation
    assert "exceeds" in verdict.explanation.lower()


def test_budget_rejects_thread_overrun():
    b = build_budget(FULL_REPORT, "lite")
    verdict = b.fits(ResourceRequirement(ram_mb=100, threads=16, measured=True))
    assert verdict.ok is False
    assert any("threads" in r for r in verdict.reasons)


def test_budget_gpu_required_but_profile_denies():
    b = build_budget(FULL_REPORT, "lite")
    verdict = b.fits(ResourceRequirement(ram_mb=100, needs_gpu=True, measured=True))
    assert verdict.ok is False
    assert any("GPU" in r for r in verdict.reasons)


def test_budget_gpu_fits_under_performance():
    b = build_budget(FULL_REPORT, "performance")
    verdict = b.fits(ResourceRequirement(ram_mb=1000, vram_mb=3000, needs_gpu=True, threads=4, measured=True))
    assert verdict.ok is True


def test_budget_rejects_vram_over_gpu_capacity():
    b = build_budget(FULL_REPORT, "performance")
    verdict = b.fits(ResourceRequirement(ram_mb=100, vram_mb=8192, needs_gpu=True, measured=True))
    assert verdict.ok is False
    assert any("VRAM" in r for r in verdict.reasons)


def test_unmeasured_requirement_always_flagged_in_reasons():
    b = build_budget(FULL_REPORT, "balanced")
    verdict = b.fits(ResourceRequirement(ram_mb=100, threads=1, measured=False))
    assert verdict.ok is True  # declared fits, but ...
    assert any("not measured" in r for r in verdict.reasons)


def test_partial_report_cannot_confirm_ram_fit():
    b = build_budget(PARTIAL_REPORT, "lite")
    verdict = b.fits(ResourceRequirement(ram_mb=100, threads=1, measured=True))
    assert verdict.ok is False
    assert any("unmeasured" in r.lower() for r in verdict.reasons)


def test_user_limits_override_profile():
    b = build_budget(FULL_REPORT, "lite", max_resident_mib=9000, max_threads=8)
    verdict = b.fits(ResourceRequirement(ram_mb=8000, threads=6, measured=True))
    assert verdict.ok is True


def test_build_budget_rejects_bad_limits():
    with pytest.raises(ConfigurationError):
        build_budget(FULL_REPORT, "lite", max_resident_mib=0)
    with pytest.raises(ConfigurationError):
        build_budget(FULL_REPORT, "lite", max_threads=0)


def test_require_fits_raises_resource_limit():
    b = build_budget(FULL_REPORT, "lite")
    with pytest.raises(ResourceLimitExceeded) as exc:
        b.require_fits(ResourceRequirement(ram_mb=99999, threads=1, measured=True))
    assert "exceeds" in exc.value.explanation.lower()


# --- budget switch purity ----------------------------------------------------

def test_switch_profile_is_pure():
    lite = build_budget(FULL_REPORT, "lite")
    perf = lite.switch_profile("performance")
    assert lite.profile is LITE  # untouched
    assert perf.profile is PERFORMANCE
    assert lite.max_resident_mib == LITE.max_resident_mib
    assert perf.max_resident_mib == PERFORMANCE.max_resident_mib
    assert lite is not perf


def test_switch_profile_preserves_user_limits():
    lite = build_budget(FULL_REPORT, "lite", max_threads=8)
    perf = lite.switch_profile("performance")
    assert perf.max_threads == 8


def test_profile_switch_does_not_invalidate_loaded_model(tmp_path):
    from nativevoice.core.language import Task
    from nativevoice.models.api import Asset, AssetKind, ModelManifest, Provenance, ReviewStatus
    from nativevoice.models.lifecycle import ModelManager
    from nativevoice.models.manifest_io import save_manifest
    from nativevoice.models.registry import ModelRegistry
    import hashlib

    blob = b"weights"
    root = tmp_path
    (root / "m").mkdir(parents=True, exist_ok=True)
    (root / "m" / "w.bin").write_bytes(blob)
    manifest = ModelManifest(
        model_id="m",
        engine="stub",
        task=Task.STT,
        languages=("en",),
        provenance=Provenance(source="s", licence_spdx="MIT", redistributable=True),
        review=ReviewStatus.APPROVED,
        approved_by="a",
        requirements=ResourceRequirement(ram_mb=100, threads=1, measured=True),
        assets=(Asset(kind=AssetKind.MODEL, relative_path="w.bin", sha256=hashlib.sha256(blob).hexdigest(), size_bytes=len(blob)),),
    )
    save_manifest(manifest, root / "m.json")
    registry = ModelRegistry(root)
    budget = build_budget(FULL_REPORT, "lite")
    manager = ModelManager(registry, lambda mf, bg: object(), budget=budget)
    manager.load("m")
    manager.load("m")  # budget upgraded to a heavier profile, still-loaded set must survive
    heavier = budget.switch_profile("performance")
    assert manager.is_loaded("m") is True
    assert heavier.fits(manifest.requirements).ok is True


# --- recommend ---------------------------------------------------------------

def test_recommend_degrades_to_lite_on_partial_detection():
    rec = recommend(PARTIAL_REPORT, [])
    assert rec.profile is LITE
    assert rec.provisional is True
    assert any("lite" in r.lower() for r in rec.reasons)


def test_recommend_full_machine_gets_high_profile():
    rec = recommend(FULL_REPORT, [])
    assert rec.profile_name in {"balanced", "performance"}
    assert rec.runner_up is not None
    assert rec.provisional is True  # thresholds are seeds until a benchmark


def test_select_profile_manual_override():
    assert select_profile("performance") is PERFORMANCE
    with pytest.raises(ConfigurationError):
        select_profile("bogus")


def test_recommendation_switch_profile_is_pure():
    rec = recommend(FULL_REPORT, [])
    moved = rec.switch_profile("lite")
    assert moved.profile is LITE
    assert rec.profile is not LITE  # original untouched
    assert rec.budget.profile is rec.profile


def test_recommend_honours_min_profile_floor_reason():
    from nativevoice.core.language import Task
    from nativevoice.models.api import ModelManifest

    heavy = ModelManifest(
        model_id="h",
        engine="e",
        task=Task.STT,
        minimum_profile="performance",
        requirements=ResourceRequirement(ram_mb=100),
    )
    tiny = SystemReport(
        os_name="Windows",
        machine="AMD64",
        cpu_physical=2,
        cpu_logical=2,
        ram_total_mb=1024,
        ram_available_mb=512,
    )
    rec = recommend(tiny, [heavy])
    assert rec.profile_name in profile_names()
    assert any("does not support" in r or "fallback" in r for r in rec.reasons)
