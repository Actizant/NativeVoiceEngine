"""The measurement seam: what it loads, what it refuses, and what it never touches.

Stub engines and synthetic bytes. No Large-v3, no download, no inference: the unit under
test is *which manifest is allowed to reach an engine at all*, and the cheapest way to
learn that a guard works is to make everything else a constant.

The seam exists because a scope tag nobody can act on is a label. Provisioning can record
`measurement`, the product path can refuse it, and the research number still cannot be
taken - which is how a boundary gets widened by whoever needs it next, with a comment
about it in the changelog. Three refusals are checked here rather than one, in both
directions: a research loader that only declined to *widen* approval would happily run a
product-scoped asset with no approval record at all. The third is about where the bytes
are, because the registry guard says where a manifest may be read from and says nothing
about the directory it points into.

"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from nativevoice.config.paths import (  # noqa: E402
    hf_hub_cache_root,
    measurement_models_dir,
    models_dir,
)
from nativevoice.core.errors import ConfigurationError, ModelNotApproved, ModelNotInstalled  # noqa: E402
from nativevoice.core.language import Task  # noqa: E402
from nativevoice.models import measurement  # noqa: E402
from nativevoice.models.api import (  # noqa: E402
    Asset,
    AssetKind,
    AssetScope,
    ModelManifest,
    Provenance,
    ResourceRequirement,
    ReviewStatus,
)
from nativevoice.models.approval import ApprovalGate, ensure_measurement_use, measurement_blockers  # noqa: E402
from nativevoice.models.integrity import inspect as inspect_integrity  # noqa: E402
from nativevoice.models.registry import ModelRegistry  # noqa: E402
from nativevoice.models.roots import asset_root_blockers, ensure_asset_root_isolated  # noqa: E402
from nativevoice.stt.adapters.faster_whisper_engine import FasterWhisperEngine  # noqa: E402

BLOB = b"synthetic-ct2-bytes-not-a-real-model"
# Recorded as NOASSERTION by the provisioner for this asset: unresolved, and a research
# use does not resolve it. Complete provenance is required, an answered licence is not.
PROVENANCE = Provenance(source="Systran/faster-whisper-large-v3", licence_spdx="NOASSERTION", redistributable=False)


def make_manifest(**overrides) -> ModelManifest:
    base = dict(
        model_id="teacher-candidate",
        engine="faster-whisper",
        task=Task.STT,
        display_name="recognition teacher, measurement only",
        languages=(),
        provenance=PROVENANCE,
        # Unreviewed on purpose: this is the state a measurement asset is recorded in,
        # and the axis under test must not care.
        review=ReviewStatus.UNREVIEWED,
        approved_by="researcher@example.com",
        requirements=ResourceRequirement(ram_mb=3300, threads=4, measured=False),
        assets=(
            Asset(
                kind=AssetKind.MODEL,
                relative_path="model.bin",
                sha256=hashlib.sha256(BLOB).hexdigest(),
                size_bytes=len(BLOB),
            ),
        ),
        use_scope=AssetScope.MEASUREMENT,
    )
    base.update(overrides)
    return ModelManifest(**base)


class StubEngine:
    """What an engine is for these tests: an object that records what reached it."""

    def __init__(self, manifest: ModelManifest, **kwargs) -> None:
        self.manifest = manifest
        self.kwargs = kwargs


def trial_root(tmp_path: Path, monkeypatch) -> Path:
    """A configuration where the product and measurement trees are siblings."""
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path / "nv"))
    return tmp_path / "trial"


def indexed(root: Path, *manifests: ModelManifest, scope: AssetScope = AssetScope.MEASUREMENT) -> ModelRegistry:
    registry = ModelRegistry(root, scope=scope)
    for manifest in manifests:
        registry.register(manifest)
    return registry


# --- the registry it will read --------------------------------------------------

def test_the_measurement_registry_is_rooted_in_the_measurement_tree(monkeypatch, tmp_path):
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path))
    registry = measurement.registry()
    assert registry.scope is AssetScope.MEASUREMENT
    assert registry.root == measurement_models_dir().resolve()
    assert not registry.root.exists()  # asking where research assets live creates nothing


def test_the_measurement_loader_refuses_a_product_registry_before_looking_anything_up(tmp_path, monkeypatch):
    """Scope of the *registry*, checked before the manifest is even read.

    The registry is empty, so the only way this can raise is the registry refusal: a
    CapabilityUnavailable would mean it opened the product index first.
    """
    product = ModelRegistry(tmp_path / "product-models", scope=AssetScope.PRODUCT)
    with pytest.raises(ConfigurationError) as exc:
        measurement.load(product, "teacher-candidate", engine_factory=StubEngine)
    assert "will not read this registry" in str(exc.value)
    assert exc.value.details["registry_scope"] == "product"


def test_the_measurement_registry_cannot_be_pointed_at_the_product_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path / "nv"))
    with pytest.raises(ConfigurationError):
        measurement.registry(models_dir())
    # A parent directory counts: a registry also globs `<root>/*.json`, so pointing at
    # the whole product data tree would read and write the shipped assets' manifests.
    with pytest.raises(ConfigurationError):
        measurement.registry(str(tmp_path / "nv"))


# --- the manifest it will serve ---------------------------------------------------

def test_a_product_scoped_manifest_is_refused_even_when_it_is_already_indexed(tmp_path, monkeypatch):
    """Defence in depth, tested as such: the registry filter is not the only guard.

    Writing the index directly bypasses `register`, which is the seam a future refactor
    could weaken - so the loader's own scope check has to hold on its own.
    """
    root = trial_root(tmp_path, monkeypatch)
    registry = indexed(root)
    product = make_manifest(model_id="shipped-asset", use_scope=AssetScope.PRODUCT, review=ReviewStatus.APPROVED, approved_on="2026-10-14")
    registry.manifests[product.model_id] = product
    with pytest.raises(ModelNotApproved) as exc:
        measurement.load(registry, product.model_id, engine_factory=StubEngine)
    assert any("use scope" in reason for reason in exc.value.details["reasons"])


def test_a_product_scoped_manifest_is_never_indexed_by_the_measurement_registry(tmp_path, monkeypatch):
    root = trial_root(tmp_path, monkeypatch)
    product = make_manifest(use_scope=AssetScope.PRODUCT)
    with pytest.raises(ModelNotApproved):
        indexed(root, product)


def test_an_unreviewed_measurement_asset_reaches_the_engine(tmp_path, monkeypatch):
    """The control the refusals above depend on: the research path actually works.

    Unreviewed, unapproved, licence unresolved, nothing measured - and still loadable,
    because what authorises local measurement is scope plus a named authoriser plus a
    provenance record with no blanks in it.
    """
    root = trial_root(tmp_path, monkeypatch)
    manifest = make_manifest(asset_root=str(root))
    registry = indexed(root, manifest)
    engine = measurement.load(registry, manifest.model_id, engine_factory=StubEngine, beam_size=5)
    assert engine.manifest is manifest
    assert engine.kwargs == {"beam_size": 5}
    assert measurement.usable(manifest) is True


def test_the_licence_gate_is_not_on_this_path_in_either_direction(tmp_path, monkeypatch):
    """A gate the loader never consults, so its override can neither open nor close it.

    The same manifest is refused by the approval gate and loaded by the measurement
    seam; an override on a product-scoped asset is then refused by the measurement
    seam, which is the direction that matters: an approval record is about one asset's
    review state, not about which path it may serve.
    """
    root = trial_root(tmp_path, monkeypatch)
    manifest = make_manifest()
    registry = indexed(root, manifest)
    gate = ApprovalGate()
    with pytest.raises(ModelNotApproved):
        gate.check(manifest)
    assert measurement.load(registry, manifest.model_id, engine_factory=StubEngine).manifest is manifest

    product = make_manifest(model_id="shipped-asset", use_scope=AssetScope.PRODUCT)
    gate.override(product, model_id=product.model_id, reason="eval under NDA")
    gate.check(product)  # the approval axis is wide open here
    with pytest.raises(ModelNotApproved):
        ensure_measurement_use(product)
    assert measurement.usable(product) is False


# --- the rule behind the loader ---------------------------------------------------

def test_the_review_record_can_never_authorize_measurement_use():
    """`review` is not an entry condition on this axis: APPROVED buys nothing here.

    An approved manifest with no named authoriser is still refused, and an unreviewed one
    with an authoriser is accepted - the two states a conflation of the axes would swap.
    """
    unreviewed = make_manifest(review=ReviewStatus.UNREVIEWED)
    assert measurement_blockers(unreviewed) == ()
    approved_but_anonymous = make_manifest(review=ReviewStatus.APPROVED, approved_by="", approved_on="2026-10-14")
    assert any("authoriser" in reason for reason in measurement_blockers(approved_but_anonymous))


def test_a_recorded_rejection_bars_measurement_use_as_well_as_the_product_path():
    """The one thing on the review axis that does reach this seam, and its direction.

    `REJECTED` is not an absent approval; it is a human's decision that the asset is not
    to be used. A research route that overruled it would be the second door the licence
    gate was built not to have, while a route that could not read it would turn every
    rejection into a routing argument. Nothing else on the axis changes: the same
    manifest, unreviewed, is loadable.
    """
    rejected = make_manifest(review=ReviewStatus.REJECTED)
    assert any("rejected" in reason for reason in measurement_blockers(rejected))
    assert measurement.usable(rejected) is False
    with pytest.raises(ModelNotApproved) as exc:
        ensure_measurement_use(rejected)
    assert any("rejected" in reason for reason in exc.value.details["reasons"])

    # Nothing about the rejection is inherited from the shipped path's own rules: this is
    # the measurement gate refusing on its axis, with its own reason set.
    assert measurement_blockers(make_manifest(review=ReviewStatus.UNREVIEWED)) == ()


def test_a_rejection_is_indexable_and_still_unloadable(tmp_path, monkeypatch):
    """The refusal belongs to the seam, not the index - and an override cannot reopen it.

    Discovery and `register()` are scope-only by design, so a rejected measurement asset
    is in the index and the loader is the thing that says no. The control is the product
    gate opened for the same manifest: an override is a decision about the approval
    record, never written back to the manifest, so it has no effect on this axis.
    """
    root = trial_root(tmp_path, monkeypatch)
    rejected = make_manifest(model_id="refused-by-a-human", review=ReviewStatus.REJECTED)
    registry = indexed(root, rejected)
    assert rejected.model_id in registry.ids()

    gate = ApprovalGate()
    gate.override(rejected, model_id=rejected.model_id, reason="local measurement only")
    gate.check(rejected)  # the approval axis is open, which is why the next line proves something
    with pytest.raises(ModelNotApproved):
        measurement.load(registry, rejected.model_id, engine_factory=StubEngine)
    assert measurement.usable(rejected) is False


@pytest.mark.parametrize(
    "overrides",
    [
        dict(use_scope=AssetScope.PRODUCT),
        dict(review=ReviewStatus.REJECTED),
        dict(approved_by=""),
        dict(approved_by="   "),
        dict(provenance=Provenance(source="", licence_spdx="NOASSERTION", redistributable=False)),
        dict(provenance=Provenance(source="hf", licence_spdx="", redistributable=False)),
        dict(provenance=Provenance(source="hf", licence_spdx="NOASSERTION")),
    ],
    ids=["product-scope", "rejected-review", "no-authoriser", "whitespace-authoriser", "no-source", "no-licence-stated", "redistribution-unknown"],
)
def test_what_is_refused_is_reported_and_then_refused(overrides):
    """`usable` is a read-out of the guard, not a second rule that can drift from it."""
    manifest = make_manifest(**overrides)
    assert measurement.usable(manifest) is False
    assert measurement_blockers(manifest)
    with pytest.raises(ModelNotApproved):
        ensure_measurement_use(manifest)


def test_the_gate_answers_without_the_neural_runtime_being_importable(tmp_path, monkeypatch):
    """A gate that decides whether to run a model must not need a runtime to answer.

    Two properties, and the previous version of this test could not see either. First the
    import is inside the call, not at module scope: the engine module is poisoned and the
    loader is then executed *afresh*, so an import hoisted to the top of the module would
    raise here rather than quietly passing - which is what made the old assertions
    vacuous, since this file had already imported the engine by the time they ran.

    Then the refusals themselves: they arrive for their own reasons while the poison is
    live, and the last case shows the poison really is armed, because the same seam that
    refuses an unauthorised manifest reaches the runtime for an authorised one.
    """
    monkeypatch.setitem(sys.modules, "nativevoice.stt.adapters.faster_whisper_engine", None)
    spec = importlib.util.find_spec("nativevoice.models.measurement")
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)

    root = trial_root(tmp_path, monkeypatch)
    registry = indexed(root)
    product = make_manifest(model_id="shipped-asset", use_scope=AssetScope.PRODUCT)
    registry.manifests[product.model_id] = product
    with pytest.raises(ModelNotApproved):
        fresh.load(registry, product.model_id)
    with pytest.raises(ConfigurationError):
        fresh.load(ModelRegistry(root, scope=AssetScope.PRODUCT), "teacher-candidate")
    # The control: with the gate open, the same poisoned import is what stops the load.
    authorized = make_manifest()
    registry.manifests[authorized.model_id] = authorized
    with pytest.raises(ImportError):
        fresh.load(registry, authorized.model_id, asset_dir=str(tmp_path / "trial-data"))


def test_the_default_factory_builds_the_manifest_bearing_engine_on_the_measurement_axis(tmp_path, monkeypatch):
    """Only the construction is exercised here: no weights, no warm, no inference."""
    manifest = make_manifest(asset_root=str(trial_root(tmp_path, monkeypatch)))
    engine = measurement.stt_engine(manifest, asset_dir=tmp_path, compute_type="int8")
    assert engine.serve_scope is AssetScope.MEASUREMENT
    assert engine.manifest is manifest
    assert engine._model is None


def test_the_adapter_serves_measurement_and_product_as_two_asserted_paths(tmp_path):
    """Same manifest, same bytes, one argument different: each axis keeps its own owner.

    The measurement route accepts an unreviewed measurement asset and refuses a
    product-scoped one; the product route refuses the measurement asset exactly as it
    always did. Nothing is verified on disk here, so the only variable is the gate.
    """
    directory = tmp_path / "repo"
    (directory / "model.bin").parent.mkdir(parents=True, exist_ok=True)
    (directory / "model.bin").write_bytes(BLOB)
    research = make_manifest(asset_root=str(directory))
    shipped = make_manifest(model_id="shipped-asset", asset_root=str(directory), use_scope=AssetScope.PRODUCT, review=ReviewStatus.APPROVED, approved_on="2026-10-14")

    FasterWhisperEngine(research, asset_dir=directory, serve_scope=AssetScope.MEASUREMENT)._require_approved()
    with pytest.raises(ModelNotApproved):
        FasterWhisperEngine(research, asset_dir=directory)._require_approved()
    with pytest.raises(ModelNotApproved):
        FasterWhisperEngine(shipped, asset_dir=directory, serve_scope=AssetScope.MEASUREMENT)._require_approved()
    FasterWhisperEngine(shipped, asset_dir=directory)._require_approved()


def test_a_serve_scope_that_is_not_a_scope_raises_rather_than_defaulting(tmp_path):
    """An argument that silently fell back to the product path would be the whole defect.

    `pytest.raises(ValueError)` on its own would not notice: ConfigurationError subclasses
    ValueError, so any configuration complaint from this constructor - including the new
    asset-root rule - would pass for enum coercion. The type is asserted exactly, the
    message has to name both the class and the value refused, and the valid spelling of the
    same argument is shown to be accepted, so the failure below cannot be a typo in the
    test rather than a property of the code.
    """
    manifest = make_manifest()
    with pytest.raises(ValueError) as exc:
        FasterWhisperEngine(manifest, asset_dir=tmp_path, serve_scope="measurement ")
    assert type(exc.value) is ValueError, f"expected raw enum coercion, got {type(exc.value).__name__}: {exc.value}"
    assert not isinstance(exc.value, ConfigurationError)
    assert "AssetScope" in str(exc.value) and "measurement " in str(exc.value)
    assert FasterWhisperEngine(manifest, asset_dir=tmp_path, serve_scope="measurement").serve_scope is AssetScope.MEASUREMENT


# --- which directory the bytes are actually in -------------------------------------

def test_a_measurement_asset_root_may_not_be_the_product_tree(tmp_path, monkeypatch):
    """The registry guard says where a *manifest* may be read from, not where its bytes are.

    Every authorisation fact is correct in this manifest and its index is rooted in the
    measurement tree, so the only thing left to refuse is the root it names: the product
    user-data tree, which the shipped path, its backups and its cleanup all reach.
    """
    root = trial_root(tmp_path, monkeypatch)
    models_dir().mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(asset_root=str(models_dir()))
    registry = indexed(root, manifest)
    with pytest.raises(ConfigurationError) as exc:
        measurement.load(registry, manifest.model_id, engine_factory=StubEngine)
    assert exc.value.details["scope"] == "measurement"
    assert any("product scope's tree" in reason for reason in exc.value.details["reasons"])
    # The same manifest pointed beside both trees is the one that loads.
    registry.manifests[manifest.model_id] = make_manifest(asset_root=str(root / "assets"))
    assert measurement.load(registry, manifest.model_id, engine_factory=StubEngine).manifest.asset_root == str(root / "assets")


def test_an_asset_dir_argument_cannot_carry_product_bytes_past_the_manifest(tmp_path, monkeypatch):
    """The override a caller passes is the directory the engine opens, so it is what is checked.

    A manifest declaring no root is the case that matters: registry, scope and
    authorisation all correct, and an `asset_dir` alone would otherwise be enough to read
    the shipped tree from a run recorded as measurement.
    """
    root = trial_root(tmp_path, monkeypatch)
    models_dir().mkdir(parents=True, exist_ok=True)
    manifest = make_manifest()
    registry = indexed(root, manifest)
    with pytest.raises(ConfigurationError) as exc:
        measurement.load(registry, manifest.model_id, engine_factory=StubEngine, asset_dir=models_dir())
    assert any("product scope's tree" in reason for reason in exc.value.details["reasons"])
    assert measurement.load(registry, manifest.model_id, engine_factory=StubEngine, asset_dir=tmp_path / "trial-data") is not None


@pytest.mark.parametrize(
    "relative, allowed",
    [
        ("", True),  # nothing declared: the registry's own tree is implied, and checked there
        ("trial", True),  # a trial directory beside both trees
        ("nv", False),  # the product root itself
        ("nv/models", False),  # the product models directory
        ("nv/cache", False),  # anywhere under the product root, not only its models tree
        ("nv-measurement/models", True),  # this scope's own tree, one name prefix away
    ],
)
def test_the_asset_root_rule_compares_trees_and_not_name_prefixes(tmp_path, monkeypatch, relative, allowed):
    """`ultron-voice` and `ultron-voice-measurement` are siblings, not an overlap.

    A lexical prefix test would read the second as inside the first and refuse the
    measurement scope's own tree, so the comparison is component-wise in both directions:
    the candidate inside the forbidden tree, and the forbidden tree inside the candidate.
    """
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path / "nv"))
    candidate = tmp_path / relative if relative else ""
    blockers = asset_root_blockers(AssetScope.MEASUREMENT, candidate, base=tmp_path / "nv-measurement")
    assert bool(blockers) is not allowed


def test_a_relative_asset_root_is_resolved_before_it_is_judged(tmp_path, monkeypatch):
    """Relative is not innocent: the base decides which tree a declared root really is.

    Without a base there is nothing to resolve against, and the refusal is explicit rather
    than a guess about the current directory - a path check that invents a base is a path
    check that can be argued out of its own answer.
    """
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path / "nv"))
    assert asset_root_blockers(AssetScope.MEASUREMENT, "models", base=models_dir())
    assert asset_root_blockers(AssetScope.MEASUREMENT, "models", base=measurement_models_dir()) == ()
    with pytest.raises(ConfigurationError) as exc:
        ensure_asset_root_isolated(AssetScope.MEASUREMENT, "models")
    assert "relative" in str(exc.value.details["reasons"][0])


@pytest.mark.skipif(sys.platform != "win32", reason="the spellings this normalises are Windows ones")
def test_equivalent_windows_spellings_of_one_product_tree_are_all_refused(tmp_path, monkeypatch):
    """One tree, seven spellings, no way through.

    Two paths that differ only in case, separators, a trailing separator or a `..` segment
    are the same directory to the loader, and a per-user or Store redirect can make two
    names one tree as well. The product tree is created first because resolving a path that
    exists is what recovers the on-disk spelling.
    """
    root = trial_root(tmp_path, monkeypatch)
    product = models_dir()
    product.mkdir(parents=True)
    registry = indexed(root)
    spellings = [
        str(product),
        str(product) + "\\",
        product.as_posix(),
        str(product) + "\\..\\models",
        str(product.parent) + "\\.\\models",
        str(product).swapcase(),
        str(product).upper(),
    ]
    for index, spelling in enumerate(spellings):
        model_id = f"spelling-{index}"
        registry.manifests[model_id] = make_manifest(model_id=model_id, asset_root=spelling)
        with pytest.raises(ConfigurationError) as exc:
            measurement.load(registry, model_id, engine_factory=StubEngine)
        assert any("product scope's tree" in reason for reason in exc.value.details["reasons"]), spelling


def test_the_shared_hub_cache_is_not_the_product_tree_and_a_redirect_of_it_is(tmp_path, monkeypatch):
    """The pair that keeps the claim honest, in both directions.

    A pinned Hugging Face asset is read from a cache the two scopes share by design, and
    that is not an overlap: nothing here asserts that cached weights are physically outside
    every directory product code can open. Point ``HF_HUB_CACHE`` under the product root
    and the same rule refuses it, which is the case the easy claim would have hidden.
    """
    for variable in ("HF_HOME", "HF_HUB_CACHE"):
        monkeypatch.delenv(variable, raising=False)
    root = trial_root(tmp_path, monkeypatch)
    registry = indexed(root)
    shared = make_manifest(asset_root=str(hf_hub_cache_root() / "models--Systran--faster-whisper-large-v3"))
    registry.manifests[shared.model_id] = shared
    assert measurement.load(registry, shared.model_id, engine_factory=StubEngine) is not None

    models_dir().mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(models_dir() / "hub"))
    redirected = make_manifest(model_id="redirected-cache", asset_root=str(hf_hub_cache_root() / "models--x"))
    registry.manifests[redirected.model_id] = redirected
    with pytest.raises(ConfigurationError):
        measurement.load(registry, redirected.model_id, engine_factory=StubEngine)


# --- the directory nobody declared --------------------------------------------------

def test_a_redirected_cache_is_refused_when_no_path_names_it(tmp_path, monkeypatch):
    """N1: the resolved directory is the one that gets opened, so it is the one guarded.

    Neither the manifest nor the call names a root, so every declared-root check has
    nothing to refuse and the loader is satisfied. The only path left is the cache root
    the environment supplies; pointed inside the product tree it is refused at the
    verification seam, before one digest is read, instead of measuring the shipped
    asset's bytes under a measurement record.
    """
    root = trial_root(tmp_path, monkeypatch)
    hub = models_dir() / "hub"
    (hub / "models--Systran--teacher-candidate").mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    registry = indexed(root, make_manifest())
    engine = measurement.load(registry, "teacher-candidate")
    assert engine.manifest.asset_root == ""  # nothing was declared: this is the gap N1 closed
    with pytest.raises(ConfigurationError) as exc:
        engine.warm()
    assert any("product scope's tree" in reason for reason in exc.value.details["reasons"])
    assert engine._model is None


def test_a_cache_beside_both_trees_reaches_the_work_the_guard_precedes(tmp_path, monkeypatch):
    """The control the refusal above needs: the same resolution passes outside the tree.

    Without this, the test above could be passing because resolution always fails. Here
    the run gets to integrity and reports the missing bytes, which is what it reported
    before this guard existed.
    """
    root = trial_root(tmp_path, monkeypatch)
    hub = tmp_path / "hub-cache"
    (hub / "models--Systran--teacher-candidate").mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    engine = measurement.load(indexed(root, make_manifest()), "teacher-candidate")
    with pytest.raises(ModelNotInstalled):
        engine.warm()


def test_the_resolved_root_check_is_not_asked_of_the_product_axis(tmp_path, monkeypatch):
    """Product scope keeps its own question - approved or not - and never this path rule.

    The shipped path reads the product tree by definition, so a check that fired there
    would refuse every real product asset. Aimed at the same redirected cache, the
    product axis reaches integrity, which is where its bytes were always decided.
    """
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path / "nv"))
    models_dir().mkdir(parents=True)
    (models_dir() / "models--x--product-asset").mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(models_dir()))
    manifest = make_manifest(
        model_id="product-asset",
        use_scope=AssetScope.PRODUCT,
        review=ReviewStatus.APPROVED,
        approved_on="2026-10-14",
    )
    engine = FasterWhisperEngine(manifest)
    assert engine.serve_scope is AssetScope.PRODUCT
    with pytest.raises(ModelNotInstalled):
        engine._require_verified()


def test_the_engine_applies_the_same_root_rule_as_the_loader_on_its_own_axis(tmp_path, monkeypatch):
    """Both seams, one rule: constructing the engine directly is not the way around it.

    The product branch is untouched by this check - the shipped path is not asked about a
    measurement directory - so the cases are the measurement refusal, the measurement
    acceptance, a root this seam cannot resolve, and the product path still refusing a
    measurement asset.
    """
    root = trial_root(tmp_path, monkeypatch)
    models_dir().mkdir(parents=True, exist_ok=True)
    inside = make_manifest(asset_root=str(models_dir()))
    with pytest.raises(ConfigurationError):
        FasterWhisperEngine(inside, serve_scope=AssetScope.MEASUREMENT)._require_approved()
    outside = make_manifest(model_id="outside", asset_root=str(root / "assets"))
    FasterWhisperEngine(outside, serve_scope=AssetScope.MEASUREMENT)._require_approved()
    relative = make_manifest(model_id="relative", asset_root="models")
    with pytest.raises(ConfigurationError) as exc:
        FasterWhisperEngine(relative, serve_scope=AssetScope.MEASUREMENT)._require_approved()
    assert "relative" in str(exc.value.details["reasons"][0])
    with pytest.raises(ModelNotApproved):
        FasterWhisperEngine(outside, serve_scope=AssetScope.PRODUCT)._require_approved()


# --- the order the guards are checked in -------------------------------------------

def test_the_measurement_gate_refuses_before_integrity_or_the_backend_is_reached(tmp_path, monkeypatch):
    """C5: the guard's *position* in the load seam, not only its existence.

    The assets are present and digest-correct, which is asserted separately below rather
    than assumed, so an integrity or installation failure is not available as the reason
    anything is refused. The backend import is poisoned, and the control at the end shows
    what reaching it would have looked like: a measurement-scoped object whose gate really
    does open gets ModelNotInstalled, not ModelNotApproved. A refactor that moved
    `_require_approved()` after `_require_verified()` or after `_import_backend()` fails
    here, and nothing else in the suite would notice.
    """
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    root = trial_root(tmp_path, monkeypatch)
    repository = tmp_path / "hub-cache" / ("models--" + "Systran--faster-whisper-large-v3")
    snapshot = repository / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(BLOB)
    assets = (
        Asset(
            kind=AssetKind.MODEL,
            relative_path=f"snapshots/{snapshot.name}/model.bin",
            sha256=hashlib.sha256(BLOB).hexdigest(),
            size_bytes=len(BLOB),
        ),
    )

    refused = make_manifest(asset_root=str(repository), assets=assets, approved_by="")
    authorized = make_manifest(model_id="authorized", asset_root=str(repository), assets=assets)
    # The fixture is the proof, not a hope: these bytes verify, so a refusal here can only
    # have come from the gate.
    assert inspect_integrity(authorized, repository).ok is True

    engine = FasterWhisperEngine(refused, asset_dir=repository, serve_scope=AssetScope.MEASUREMENT)
    with pytest.raises(ModelNotApproved) as exc:
        engine.warm()
    assert any("authoriser" in reason for reason in exc.value.details["reasons"])
    assert engine._model is None

    control = FasterWhisperEngine(authorized, asset_dir=repository, serve_scope=AssetScope.MEASUREMENT)
    with pytest.raises(ModelNotInstalled) as reached:
        control.warm()
    assert reached.value.details["package"] == "faster-whisper"
    assert control._model is None
