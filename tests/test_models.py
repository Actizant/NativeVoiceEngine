"""Registry, integrity, approval and lifecycle - enforced without any neural runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nativevoice.core.errors import (
    ConfigurationError,
    IntegrityError,
    ModelNotApproved,
    ModelNotInstalled,
    ResourceLimitExceeded,
)
from nativevoice.core.language import Task
from nativevoice.models.api import (
    Asset,
    AssetKind,
    ModelManifest,
    Provenance,
    ResourceRequirement,
    ReviewStatus,
)
from nativevoice.models.approval import ApprovalGate, approval_blockers, ensure_approved
from nativevoice.models.integrity import inspect, verify
from nativevoice.models.lifecycle import ModelManager
from nativevoice.models.manifest_io import from_dict, load_manifest, save_manifest, to_dict
from nativevoice.models.registry import ModelRegistry

COMPLETE = Provenance(source="upstream", licence_spdx="MIT", redistributable=True)
BLOB = b"ultron-weight-bytes-0123456789"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_manifest(**overrides) -> ModelManifest:
    base = dict(
        model_id="test-model",
        engine="stub",
        task=Task.STT,
        languages=("en", "bn"),
        provenance=COMPLETE,
        review=ReviewStatus.APPROVED,
        approved_by="reviewer@example.com",
        approved_on="2026-01-01",
        requirements=ResourceRequirement(ram_mb=200, threads=2, measured=True),
        assets=(Asset(kind=AssetKind.MODEL, relative_path="weights.bin", sha256=sha(BLOB), size_bytes=len(BLOB)),),
    )
    base.update(overrides)
    return ModelManifest(**base)


def write_asset(tmp_path: Path, name: str, data: bytes, model_id: str = "test-model") -> Path:
    """Assets resolve under `<root>/<model_id>/`, so tests must place them there."""
    path = tmp_path / model_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_installed_dir(root: Path, manifest: ModelManifest) -> None:
    root.mkdir(parents=True, exist_ok=True)
    from nativevoice.models.integrity import base_directory

    base = base_directory(manifest, root)
    for asset in manifest.assets:
        path = base / asset.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(BLOB)
    save_manifest(manifest, root / f"{manifest.model_id}.json")


class FailingEngine:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def factory_ok(manifest, budget):
    return FailingEngine()


# --- approval gate -----------------------------------------------------------

def test_default_deny_rejects_unreviewed():
    manifest = make_manifest(review=ReviewStatus.UNREVIEWED, approved_by="")
    with pytest.raises(ModelNotApproved):
        ensure_approved(manifest)


def test_approval_requires_named_approver():
    manifest = make_manifest(review=ReviewStatus.APPROVED, approved_by="   ")
    with pytest.raises(ModelNotApproved):
        ensure_approved(manifest)
    blockers = approval_blockers(manifest)
    assert any("approver" in b for b in blockers)


def test_approval_requires_complete_provenance():
    manifest = make_manifest(provenance=Provenance(source=""), approved_by="alice")
    assert manifest.review is ReviewStatus.APPROVED
    with pytest.raises(ModelNotApproved):
        ensure_approved(manifest)


def test_approved_manifest_passes():
    ensure_approved(make_manifest())


def test_override_needs_matching_id_and_reason():
    gate = ApprovalGate()
    manifest = make_manifest(review=ReviewStatus.UNREVIEWED, approved_by="")
    with pytest.raises(ConfigurationError):
        gate.override(manifest, model_id="some-other-id", reason="urgent")
    with pytest.raises(ConfigurationError):
        gate.override(manifest, model_id=manifest.model_id, reason="   ")


def test_override_grants_and_is_logged(caplog):
    gate = ApprovalGate()
    manifest = make_manifest(review=ReviewStatus.UNREVIEWED, approved_by="")
    with pytest.raises(ModelNotApproved):
        gate.check(manifest)
    record = gate.override(manifest, model_id=manifest.model_id, reason="eval under NDA")
    assert record.reason == "eval under NDA"
    gate.check(manifest)  # now allowed
    assert any("override" in rec.getMessage().lower() for rec in caplog.records)


# --- integrity ---------------------------------------------------------------

def test_integrity_verified_ok(tmp_path):
    manifest = make_manifest()
    make_installed_dir(tmp_path, manifest)
    ok, reasons = verify(manifest, tmp_path)
    assert ok is True
    assert reasons == ()


def test_integrity_digest_mismatch(tmp_path):
    manifest = make_manifest()
    make_installed_dir(tmp_path, manifest)
    write_asset(tmp_path, "weights.bin", b"X" * len(BLOB))  # same size, different bytes
    ok, reasons = verify(manifest, tmp_path)
    assert ok is False
    assert any("digest mismatch" in r for r in reasons)


def test_integrity_missing_file(tmp_path):
    manifest = make_manifest()
    (tmp_path / "dummy.txt").write_text("x")
    ok, reasons = verify(manifest, tmp_path)
    assert ok is False
    report = inspect(manifest, tmp_path)
    assert report.missing == ("weights.bin",)
    assert any("missing" in r for r in reasons)


def test_integrity_unrecorded_digest_is_unverifiable(tmp_path):
    manifest = make_manifest(assets=(Asset(kind=AssetKind.MODEL, relative_path="weights.bin"),))
    write_asset(tmp_path, "weights.bin", BLOB)
    ok, reasons = verify(manifest, tmp_path)
    assert ok is False
    report = inspect(manifest, tmp_path)
    assert report.unverifiable == ("weights.bin",)
    assert report.verified == ()
    assert any("unverifiable" in r for r in reasons)


# --- manifest_io round-trip --------------------------------------------------

def test_manifest_round_trip_preserves_every_field():
    manifest = make_manifest(notes="hello")
    data = to_dict(manifest)
    restored = from_dict(data)
    assert restored == manifest
    assert to_dict(restored) == data


def test_manifest_unknown_keys_folded_into_notes():
    data = to_dict(make_manifest())
    data["future_field"] = {"a": 1}
    data["reviewer_note"] = "keep me"
    restored = from_dict(data)
    assert "future_field" in restored.notes
    assert "reviewer_note" in restored.notes


def test_save_load_json_is_human_editable(tmp_path):
    path = tmp_path / "m.json"
    manifest = make_manifest()
    save_manifest(manifest, path)
    text = path.read_text(encoding="utf-8")
    assert "\n" in text and "  " in text  # indented, not a one-liner
    json.loads(text)  # valid JSON
    assert load_manifest(path) == manifest


# --- registry ----------------------------------------------------------------

def test_registry_indexes_by_task_and_language(tmp_path):
    en = make_manifest(model_id="en-model", languages=("en",))
    bn = make_manifest(model_id="bn-model", task=Task.TTS, languages=("bn",))
    for manifest in (en, bn):
        make_installed_dir(tmp_path, manifest)
    registry = ModelRegistry(tmp_path)
    assert set(registry.ids()) == {"en-model", "bn-model"}
    assert registry.model_ids_for(Task.STT, "en") == ("en-model",)
    assert registry.model_ids_for(Task.TTS, "bn") == ("bn-model",)
    assert registry.model_ids_for(Task.STT, "bn") == ()


def test_registry_status_usable(tmp_path):
    manifest = make_manifest()
    make_installed_dir(tmp_path, manifest)
    registry = ModelRegistry(tmp_path)
    status = registry.status("test-model")
    assert status.installed and status.verified and status.usable
    assert status.explain() == "ready"


def test_registry_unusable_when_corrupt(tmp_path):
    manifest = make_manifest()
    make_installed_dir(tmp_path, manifest)
    write_asset(tmp_path, "weights.bin", b"b" * len(BLOB))
    registry = ModelRegistry(tmp_path)
    status = registry.status("test-model")
    assert status.usable is False
    assert any("digest mismatch" in r for r in status.reasons)


def test_registry_reports_bad_manifest_without_crashing(tmp_path):
    make_installed_dir(tmp_path, make_manifest())
    (tmp_path / "broken.json").write_text("{not json")
    registry = ModelRegistry(tmp_path)
    assert registry.ids() == ("test-model",)
    assert len(registry.issues) == 1


# --- lifecycle ---------------------------------------------------------------

def build_manager(tmp_path, manifest, *, budget=None, max_per_task=1, factory=factory_ok):
    make_installed_dir(tmp_path, manifest)
    registry = ModelRegistry(tmp_path)
    return ModelManager(registry, factory, budget=budget, max_per_task=max_per_task)


def test_load_ok_and_close(tmp_path):
    manager = build_manager(tmp_path, make_manifest())
    engine = manager.load("test-model")
    assert manager.is_loaded("test-model")
    manager.close()
    assert engine.closed is True
    assert manager.loaded_ids == ()


def test_load_missing_assets_raises_not_installed(tmp_path):
    manifest = make_manifest()
    registry = ModelRegistry(tmp_path)  # no asset files written
    registry.register(manifest)
    manager = ModelManager(registry, factory_ok)
    with pytest.raises(ModelNotInstalled):
        manager.load("test-model")


def test_load_integrity_failure_raises(tmp_path):
    manifest = make_manifest()
    make_installed_dir(tmp_path, manifest)
    write_asset(tmp_path, "weights.bin", b"corrupt")
    registry = ModelRegistry(tmp_path)
    manager = ModelManager(registry, factory_ok)
    with pytest.raises(IntegrityError):
        manager.load("test-model")


def test_load_unapproved_raises(tmp_path):
    manifest = make_manifest(review=ReviewStatus.UNREVIEWED, approved_by="")
    make_installed_dir(tmp_path, manifest)
    registry = ModelRegistry(tmp_path)
    manager = ModelManager(registry, factory_ok)
    with pytest.raises(ModelNotApproved):
        manager.load("test-model")


def test_failed_load_leaves_no_state(tmp_path):
    calls = []

    def boom(manifest, budget):
        calls.append(manifest.model_id)
        raise RuntimeError("engine exploded")

    manager = build_manager(tmp_path, make_manifest(), factory=boom)
    with pytest.raises(RuntimeError):
        manager.load("test-model")
    assert calls == ["test-model"]
    assert manager.is_loaded("test-model") is False
    status = manager.status("test-model")
    assert status.usable is False
    assert any("load failed" in r for r in status.reasons)


def test_budget_rejection_raises(tmp_path):
    class RejectingBudget:
        def fits(self, requirement):
            return type(
                "V",
                (),
                {"ok": False, "explanation": "RAM 200 exceeds budget 50", "reasons": ("RAM too big",)},
            )()

    manager = build_manager(tmp_path, make_manifest(), budget=RejectingBudget())
    with pytest.raises(ResourceLimitExceeded) as exc:
        manager.load("test-model")
    assert "budget" in exc.value.explanation
    assert manager.is_loaded("test-model") is False


def test_idle_eviction_on_per_task_cap(tmp_path):
    a = make_manifest(model_id="a-model")
    b = make_manifest(model_id="b-model")
    make_installed_dir(tmp_path, a)
    make_installed_dir(tmp_path, b)
    registry = ModelRegistry(tmp_path)
    manager = ModelManager(registry, factory_ok, max_per_task=1)
    engine_a = manager.load("a-model")
    manager.load("b-model")
    assert manager.is_loaded("a-model") is False
    assert engine_a.closed is True  # evicted cleanly
    assert manager.loaded_ids == ("b-model",)


def test_evict_idle_picks_least_recently_used(tmp_path):
    a = make_manifest(model_id="a-model")
    b = make_manifest(model_id="b-model")
    make_installed_dir(tmp_path, a)
    make_installed_dir(tmp_path, b)
    registry = ModelRegistry(tmp_path)
    manager = ModelManager(registry, factory_ok, max_per_task=2)
    manager.load("a-model")
    manager.load("b-model")
    manager.engine("a-model")  # touch a, so b is now least-recently-used
    victim = manager.evict_idle()
    assert victim == "b-model"
    assert manager.loaded_ids == ("a-model",)


def test_close_is_idempotent_and_deterministic(tmp_path):
    manifest = make_manifest()
    manager = build_manager(tmp_path, manifest)
    manager.load("test-model")
    manager.close()
    manager.close()
    assert manager.loaded_ids == ()
