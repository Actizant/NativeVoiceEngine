"""The evaluation tool's only door to a model: the measurement seam, at a pinned revision.

No inference, no real weights, no download. The unit under test is *which manifest is
allowed to become an engine in this tool, and what the resulting record is permitted to
claim*. Synthetic `model.bin` placeholders stand in for bytes; the assertions are about
refusals, ordering and metadata.

The tool used to build its own `WhisperModel` from whichever cached snapshot came first
in a sorted listing, under a hardcoded home-directory cache root. That route checked no
scope, no authoriser, no provenance and no revision, and it could read a different
commit than the one the repository's own evidence names - so a number written down here
could be attributed to a model nobody had verified. Every case below is one of the ways
that is now impossible.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import tools.multilingual_eval as mle  # noqa: E402
from nativevoice.config.paths import models_dir  # noqa: E402
from nativevoice.core.errors import (  # noqa: E402
    CapabilityUnavailable,
    ConfigurationError,
    ModelNotApproved,
)
from nativevoice.core.language import Task  # noqa: E402
from nativevoice.models.api import (  # noqa: E402
    Asset,
    AssetKind,
    AssetScope,
    ModelManifest,
    Provenance,
    ResourceRequirement,
    ReviewStatus,
)
from nativevoice.models.registry import ModelRegistry  # noqa: E402
from tools.provision import CATALOGUE  # noqa: E402

BLOB = b"synthetic-ct2-bytes-not-a-real-model"
TEACHER = "faster-whisper-large-v3"
PIN = CATALOGUE[TEACHER]["revision"]
PROVENANCE = Provenance(source="Systran/faster-whisper-large-v3", licence_spdx="NOASSERTION", redistributable=False)


def teacher_manifest(**overrides) -> ModelManifest:
    """A measurement manifest in the state provisioning records it: unreviewed, named authoriser."""
    base = dict(
        model_id=TEACHER,
        engine="faster-whisper",
        task=Task.STT,
        display_name="recognition teacher, measurement only",
        languages=(),
        provenance=PROVENANCE,
        review=ReviewStatus.UNREVIEWED,
        approved_by="researcher@example.com",
        requirements=ResourceRequirement(ram_mb=3300, threads=4, measured=False),
        assets=(
            Asset(
                kind=AssetKind.MODEL,
                relative_path=f"snapshots/{PIN}/model.bin",
                sha256=hashlib.sha256(BLOB).hexdigest(),
                size_bytes=len(BLOB),
            ),
        ),
        use_scope=AssetScope.MEASUREMENT,
    )
    base.update(overrides)
    return ModelManifest(**base)


def hub(tmp_path: Path, monkeypatch, revision: str = PIN, inside_product_tree: bool = False) -> Path:
    """A cache root holding one pinned snapshot of the teacher, with synthetic bytes."""
    if inside_product_tree:
        # Set before `models_dir()` is read: the product tree has to be the patched one, or
        # the redirect lands beside a stale tree and the case this exists to test never arises.
        monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path / "nv"))
    root = models_dir() / "hub" if inside_product_tree else tmp_path / "hub"
    snapshot = root / "models--Systran--faster-whisper-large-v3" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(BLOB)
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    return root


def serving(monkeypatch, *manifests: ModelManifest) -> ModelRegistry:
    """What the tool's `measurement.registry()` call returns: these manifests, measurement-scoped."""
    registry = ModelRegistry(Path("unused") / "measurement-models", scope=AssetScope.MEASUREMENT)
    for manifest in manifests:
        registry.register(manifest)
    monkeypatch.setattr(mle.measurement, "registry", lambda: registry)
    return registry


# --- which manifest may become an engine -------------------------------------------

def test_a_pass_with_nothing_in_the_measurement_registry_is_refused(monkeypatch, tmp_path):
    """No manifest, no run - and no quieter fallback to a cached directory.

    `serving()` hands back an empty registry, so the only thing this can be refusing on
    is the absence of an authorised record for the model id the pass names.
    """
    hub(tmp_path, monkeypatch)
    serving(monkeypatch)
    with pytest.raises(CapabilityUnavailable) as exc:
        mle.load_pass("large-v3-int8")
    assert "no such model in registry" in str(exc.value)
    assert exc.value.details["model_id"] == TEACHER


def test_an_unauthorised_measurement_manifest_is_refused(monkeypatch, tmp_path):
    """The licence gate's own question, on the evaluation route: who authorised this?"""
    hub(tmp_path, monkeypatch)
    serving(monkeypatch, teacher_manifest(approved_by=""))
    with pytest.raises(ModelNotApproved) as exc:
        mle.load_pass("large-v3-int8")
    assert any("authoriser" in reason for reason in exc.value.details["reasons"])


def test_a_product_registry_is_refused_before_any_model_id_is_resolved(monkeypatch, tmp_path):
    """Pointing the tool at the shipped registry is refused as a registry problem.

    The manifest is not even looked up, so this is not the scope of the asset speaking:
    it is the tool being unable to aim itself at the product path.
    """
    hub(tmp_path, monkeypatch)
    product = ModelRegistry(tmp_path / "product-models", scope=AssetScope.PRODUCT)
    monkeypatch.setattr(mle.measurement, "registry", lambda: product)
    with pytest.raises(ConfigurationError) as exc:
        mle.load_pass("large-v3-int8")
    assert "will not read this registry" in str(exc.value)


def test_a_refusal_happens_before_the_samples_are_touched(monkeypatch, tmp_path, capsys):
    """The refusal's position: no audio read, no workspace written, no partial artefact.

    The old tool resolved a snapshot directory before it asked anyone's permission, so a
    refused run had already decoded every sample. This asserts the order by making the
    sample scan leave a trace, and names the refusal in the output because the
    incomplete-evaluation-set branch returns the same exit code from a later point.
    """
    hub(tmp_path, monkeypatch)
    serving(monkeypatch)
    scanned: list[str] = []
    monkeypatch.setattr(mle, "find_sources", lambda: scanned.append("scanned") or [])
    monkeypatch.setattr(mle, "_guard_console", lambda: None)
    assert mle.main(["--pass", "large-v3-int8"]) == 1
    assert scanned == []
    assert "REFUSED" in capsys.readouterr().out


# --- which snapshot the run is about ------------------------------------------------

def test_a_manifest_that_digested_another_revision_is_refused(monkeypatch, tmp_path):
    """Two pins, one answer: the catalogue's revision and the manifest's must agree.

    The manifest here records a different commit than the teacher is pinned to, so the
    number would be attributed to bytes the evidence record does not name.
    """
    hub(tmp_path, monkeypatch, revision=PIN)
    serving(monkeypatch, teacher_manifest(assets=(
        Asset(kind=AssetKind.MODEL, relative_path=f"snapshots/{'f' * 40}/model.bin",
              sha256=hashlib.sha256(BLOB).hexdigest(), size_bytes=len(BLOB)),
    )))
    with pytest.raises(ConfigurationError) as exc:
        mle.load_pass("large-v3-int8")
    assert "different snapshot than this pass is pinned to" in str(exc.value)
    assert exc.value.details["pinned"] == PIN


def test_an_absent_pinned_snapshot_is_refused_rather_than_substituted(monkeypatch, tmp_path):
    """The exact defect the directory scan had: another snapshot exists, so it was used.

    Only a *different* revision is on disk here, and the manifest names the pinned one.
    The run refuses, because the substitute would be a different model.
    """
    hub(tmp_path, monkeypatch, revision="f" * 40)
    serving(monkeypatch, teacher_manifest())
    with pytest.raises(ConfigurationError) as exc:
        mle.load_pass("large-v3-int8")
    assert "no cached snapshot for the revision" in str(exc.value)
    assert exc.value.details["revision"] == PIN
    assert Path(exc.value.details["cache_root"]).name == "hub"


def test_an_unpinned_pass_uses_the_revision_the_manifest_recorded(monkeypatch, tmp_path):
    """No catalogue pin is not no pin: the digest run already wrote down which directory.

    `snapshot_files` asked by that name returns bytes on disk, and the basis says where
    the revision came from, so the record never implies a catalogue pin that does not
    exist.
    """
    hub(tmp_path, monkeypatch)
    manifest = teacher_manifest()
    resolved = mle.resolve_revision({"repo": "Systran/faster-whisper-large-v3", "revision": ""}, manifest)
    assert resolved["revision"] == PIN
    assert resolved["revision_basis"] == "the snapshot the manifest recorded when it was digested"
    assert resolved["snapshot_files_present"] == 1
    assert mle.manifest_revision(teacher_manifest(assets=(
        Asset(kind=AssetKind.MODEL, relative_path="model.bin", sha256="x", size_bytes=1),
    ))) == ""


def test_a_manifest_naming_no_revision_at_all_is_refused(monkeypatch, tmp_path):
    """A flat manifest beside an unpinned pass leaves nothing to pin the run to."""
    hub(tmp_path, monkeypatch)
    flat = teacher_manifest(assets=(
        Asset(kind=AssetKind.MODEL, relative_path="model.bin", sha256=hashlib.sha256(BLOB).hexdigest(),
              size_bytes=len(BLOB)),
    ))
    with pytest.raises(ConfigurationError) as exc:
        mle.resolve_revision({"repo": "Systran/faster-whisper-large-v3", "revision": ""}, flat)
    assert "names a revision" in str(exc.value)


# --- where the resolved directory turned out to be ---------------------------------

def test_the_route_this_tool_uses_refuses_a_cache_redirected_into_the_product_tree(monkeypatch, tmp_path):
    """N1 seen from the evaluation side: nothing declared the product tree, the cache did.

    The manifest declares no root and the tool passes no `asset_dir`, so the loader has
    nothing to refuse and the pass resolves normally - until the engine turns the
    environment's cache root into the directory it is about to open, and the same rule
    stops it before a digest is read.
    """
    inside = hub(tmp_path, monkeypatch, inside_product_tree=True)
    serving(monkeypatch, teacher_manifest())
    engine, pin = mle.load_pass("large-v3-int8")
    assert engine.manifest.asset_root == ""
    assert str(inside) in pin["repository"]  # the tool really did resolve into the product tree
    with pytest.raises(ConfigurationError) as exc:
        engine._require_verified()
    assert any("product scope's tree" in reason for reason in exc.value.details["reasons"])


# --- what the artefact is allowed to say -------------------------------------------

def test_the_record_names_the_model_and_revision_and_denies_being_the_product_path(monkeypatch, tmp_path):
    """Output metadata, because a number without its provenance becomes a product claim."""
    root = hub(tmp_path, monkeypatch)
    serving(monkeypatch, teacher_manifest())
    engine, pin = mle.load_pass("large-v3-int8")
    record = mle.pass_record("large-v3-int8", engine, pin)
    assert record["model_id"] == TEACHER
    assert record["revision"] == PIN
    assert record["snapshot_recorded"] == f"snapshots/{PIN}"
    assert record["repository"] == str(root / "models--Systran--faster-whisper-large-v3")
    assert record["hub_cache_root"] == str(root)
    assert record["serve_scope"] == "measurement"
    assert record["review_status"] == "unreviewed"
    assert record["engine"] == "faster-whisper"
    assert record["compute_type"] == "int8"
    assert record["asset_root"] == ""
    assert record["local_files_only"] is True
    assert record["production_configuration"] is False
    assert "measurement" in record["production_configuration_basis"]
    assert record["snapshot_model_bin_bytes"] == len(BLOB)


def test_no_pass_is_labelled_a_product_configuration(monkeypatch, tmp_path):
    """The table no longer carries the flag that made a research run look shipped.

    `base-default` used to record `production_configuration: true` while reading the same
    cache any other pass did. Every record this tool writes is now a measurement, and the
    settings it mirrors are said in a basis string instead of a boolean a reader trusts.
    """
    assert all("production" not in spec for spec in mle.PASSES.values())
    hub(tmp_path, monkeypatch)
    serving(monkeypatch, teacher_manifest())
    engine, pin = mle.load_pass("large-v3-int8")
    assert mle.pass_record("base-default", engine, pin)["production_configuration"] is False


# --- the route itself, held statically ----------------------------------------------

def test_the_tool_has_no_way_to_reach_the_runtime_except_through_the_manifest(monkeypatch):
    """A fence, not a style preference: the bypass was the defect.

    The source is read once and asked for the shapes that used to exist - a direct
    backend import, a model constructor, the manifest-free adapter's `approved=True`, a
    hardcoded cache root, and the directory scan that picked a snapshot by sort order.
    The constructor is matched with its call parentheses, because the module docstring
    names the thing it refuses to do.
    """
    source = (ROOT / "tools" / "multilingual_eval.py").read_text(encoding="utf-8")
    for gone in ("import faster_whisper", "WhisperModel(", "approved=True", "FasterWhisperAdapter",
                 ".cache", "locate_snapshot"):
        assert gone not in source, gone
    assert "measurement.load(" in source
    assert "hf_hub_cache_root()" in source
    assert "snapshot_files(" in source
    assert not hasattr(mle, "locate_snapshot")
    assert not hasattr(mle, "_float32")
