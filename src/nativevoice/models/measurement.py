"""The measurement-only seam: how a research asset gets loaded, and by what.

Nothing in the product path is allowed to serve a measurement-scoped manifest, so
the other half has to exist somewhere real. Without it the scope tag is a label with
no reader: provisioning can record `measurement`, every product seam can refuse it,
and the asset still cannot be *used* - which is how a boundary ends up quietly
widened by whoever needs the number next.

Three separate guarantees, deliberately not one:

* :func:`registry` - a registry rooted in the measurement tree. An explicit root is
  allowed for a trial directory, and goes through the same resolved overlap rule
  provisioning uses, so this loader cannot be pointed at the product registry. A
  measurement run that could write or read the product tree would not be a
  measurement run.
* :func:`load` - refuses a registry that is not measurement-scoped, then refuses a
  manifest that is not authorised for measurement use, then refuses an asset root that
  sits in the product tree. Both scope directions are checked because a weak gate that
  only declined to *widen* approval would let a product asset - one a customer is
  supposed to reach - be run through the research seam with no approval record at all.
  The root check is the same guarantee about *bytes* rather than about manifests: a
  measurement manifest is read from the measurement tree, and that says nothing about
  where the weights it names live, nor about an ``asset_dir`` a caller hands in over
  its head.
* no :class:`~nativevoice.models.approval.ApprovalGate` here, and no
  :class:`~nativevoice.models.lifecycle.ModelManager` either. An override is a
  decision about the product approval record and must not reach this axis; the
  manager stays the product lifecycle, so a measurement asset cannot acquire budget,
  eviction or routing simply by being loadable for research. A manifest a human marked
  ``REJECTED`` is refused here too, which is the one way the review axis speaks: it can
  close this door and cannot open it.

What is *not* done here: no download, no provisioning, no manifest written, no
network. The hub cache is shared between the two scopes by design, and a pinned
Hugging Face asset is read from it on that footing: this check does not claim cached
weights are physically outside every directory product code can reach. What it compares
is the root the manifest declares and the ``asset_dir`` the caller passes - so a cache
that a manifest *names*, or an argument that points into the product tree, is refused
here, and nothing about a cache no path mentions is decided at this seam. Asset
resolution is the engine's own, and it re-runs the same rule on the directory it
actually resolved before reading a single digest, then verifies integrity there rather
than caching it here, so a file that changed after this function returned - or a cache
redirected into the product tree by ``HF_HOME`` with the manifest declaring nothing - is
still refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..core.errors import ConfigurationError
from .api import AssetScope, ModelManifest
from .approval import ensure_measurement_use, measurement_blockers
from .registry import ModelRegistry
from .roots import ensure_asset_root_isolated, resolve_registry_root

#: Builds an engine object for one verified measurement manifest.
EngineFactory = Callable[..., object]


def registry(root: str | Path | None = None) -> ModelRegistry:
    """A registry serving measurement-scoped manifests, from the measurement tree.

    The default root is :func:`~nativevoice.config.paths.measurement_models_dir`; an
    explicit one is for a trial directory on the same condition provisioning imposes:
    it may not be, or sit inside, the product tree. Nothing is created here, so asking
    what the measurement registry would serve leaves no empty directory behind.
    """
    resolved = resolve_registry_root(AssetScope.MEASUREMENT, "" if root is None else str(root))
    return ModelRegistry(resolved, scope=AssetScope.MEASUREMENT)


def stt_engine(manifest: ModelManifest, **kwargs) -> object:
    """The manifest-bearing recognition engine, on the measurement axis.

    Imported inside the call because a gate that decides *whether* to run a model must
    not require a neural runtime to be installed to answer.
    """
    from ..stt.adapters.faster_whisper_engine import FasterWhisperEngine

    return FasterWhisperEngine(manifest, serve_scope=AssetScope.MEASUREMENT, **kwargs)


def load(
    source: ModelRegistry,
    model_id: str,
    *,
    engine_factory: EngineFactory | None = None,
    **engine_kwargs,
) -> object:
    """Build an engine for one measurement-scoped asset, refusing anything else.

    The registry is a parameter rather than something constructed here so the caller's
    root is the thing under review: a measurement evaluation that passed the product
    registry is refused by scope, not by convention. ``engine_factory`` exists for
    tests with a stub engine and synthetic bytes; the default is the real
    manifest-bearing STT engine, which still runs its own approval-free integrity gate
    when it warms. An ``asset_dir`` is checked, not just the manifest's own
    ``asset_root``, because it is the one the engine opens - and the engine re-checks
    whatever directory it resolves, so a root neither this call nor the manifest named
    is refused there rather than here.
    """
    if source.scope is not AssetScope.MEASUREMENT:
        raise ConfigurationError(
            "the measurement loader will not read this registry",
            registry_scope=source.scope.value,
            remedy="build the registry with nativevoice.models.measurement.registry()",
            root=str(source.root),
        )
    manifest = source.get(model_id)
    ensure_measurement_use(manifest)
    # Last of the three, and on both roots the engine will actually open: what the
    # manifest declares and what this call overrides it with. Ordered after the
    # authorisation gate because the two refusals answer different questions - a path
    # check that fired first would let a boundary test pass on a manifest that was
    # never authorised, which is how a green suite stops proving the thing it names.
    ensure_asset_root_isolated(
        AssetScope.MEASUREMENT, manifest.asset_root, engine_kwargs.get("asset_dir"), base=source.root
    )
    factory = engine_factory or stt_engine
    return factory(manifest, **engine_kwargs)


def usable(manifest: ModelManifest) -> bool:
    """Whether this manifest would be accepted by :func:`load` - reported, not raised.

    A tool that lists what it can measure needs the blockers as data; the refusal
    itself stays in :func:`~nativevoice.models.approval.ensure_measurement_use` so
    there is one rule, not a summary that can drift from it. This is the authorisation
    axis alone: :func:`load` additionally refuses an asset root that overlaps the
    product tree, which a manifest on its own cannot answer - it depends on which
    ``asset_dir`` the caller passes.
    """
    return not measurement_blockers(manifest)
