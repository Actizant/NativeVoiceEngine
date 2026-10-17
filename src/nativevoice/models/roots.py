"""Which directory a scope may use, compared resolved.

Both the provisioner and the measurement loader need the same rule: one scope's
registry may not be pointed into the other scope's tree. Two copies of it drift,
and the guard that survives is the weaker one, so it lives here and callers
delegate. :func:`resolve_registry_root` applies that rule to the tree a scope
provisions or reads; :func:`ensure_asset_root_isolated` applies the same resolved
overlap test to the tree a manifest's *bytes* live in, which the registry rule
cannot see.

Nothing in this module creates a directory. The tree for a scope that has never
been provisioned does not exist yet, and a tool that was refused should not have
left an empty one behind; ``resolve()`` defaults to ``strict=False`` so an absent
root can still be compared.

Paths are compared *resolved*, never as spelled: two roots can carry different
names and still be one tree through a junction, a symlink or a per-user redirect,
and Windows' Store build redirects ``LOCALAPPDATA`` in exactly that way. A
lexical comparison proves only that the strings differ.
"""

from __future__ import annotations

from pathlib import Path

from ..config.paths import data_root, measurement_data_root, measurement_models_dir, models_dir
from ..core.errors import ConfigurationError
from .api import AssetScope


def default_registry_root(scope: AssetScope) -> Path:
    """The models root the given scope is served from."""
    return measurement_models_dir() if scope is AssetScope.MEASUREMENT else models_dir()


def scope_tree(scope: AssetScope) -> Path:
    """The whole directory tree reserved to one scope, root included.

    Overlap is tested against the tree rather than the models directory because the
    two scopes are meant to share nothing at all: no manifest directory, no backup
    job, and no accidental recursive delete. A root under the *other* scope's root
    would pass a models-directory test and still put measurement evidence inside the
    product user data.
    """
    return measurement_data_root() if scope is AssetScope.MEASUREMENT else data_root()


def other_scope(scope: AssetScope) -> AssetScope:
    """The scope this one must not share a directory with."""
    return AssetScope.PRODUCT if scope is AssetScope.MEASUREMENT else AssetScope.MEASUREMENT


def resolve_registry_root(scope: AssetScope, explicit: str = "") -> Path:
    """The root to provision into, refusing one that overlaps the other scope's tree.

    Two scopes sharing a directory is not a tidiness question: discovery reads every
    ``manifests/*.json`` and every ``*.json`` under its root, so a measurement
    manifest written into the product tree is a file the product registry has to
    reject on sight rather than one it never sees.

    The result is resolved, because an unresolved one is two different answers to
    "where did this go": the overlap test below runs on resolved paths, and a
    caller that then used the spelled form (``--root ~``, a relative path, or a root
    under a redirected ``LOCALAPPDATA``) would write somewhere the guard had not
    looked at.
    """
    own = default_registry_root(scope)
    root = Path(explicit) if explicit else own
    resolved = root.expanduser().resolve()
    forbidden = scope_tree(other_scope(scope)).expanduser().resolve()
    if overlaps(resolved, forbidden):
        raise ConfigurationError(
            "provisioning root overlaps the other asset scope's tree",
            root=str(resolved),
            scope=scope.value,
            forbidden=str(forbidden),
            remedy="give this scope its own tree, or omit --root and use the configured one",
        )
    return resolved


def overlaps(resolved: Path, forbidden: Path) -> bool:
    """Whether one resolved path is the other, or contains it, or sits inside it.

    One predicate, shared by the registry rule and the asset-root rule, so the two
    cannot disagree about what "overlaps" means while both claim to enforce it. The
    comparison is component-wise, which is what makes the two sibling trees -
    ``ultron-voice`` and ``ultron-voice-measurement`` - distinct rather than a
    string prefix that a lexical test would read as an overlap.
    """
    return resolved == forbidden or forbidden in resolved.parents or resolved in forbidden.parents


def asset_root_blockers(
    scope: AssetScope, asset_root: str | Path | None, base: str | Path = ""
) -> tuple[str, ...]:
    """Why an asset root may not be opened under ``scope`` (empty when it may).

    The registry guard says where a *manifest* may be read from; this one says where the
    *bytes* it points at may be. Both are needed, and neither implies the other: a
    measurement-scoped manifest can sit correctly in the measurement tree and still
    declare an ``asset_root`` inside the product user-data tree, where the shipped path,
    its backups and its cleanup all reach - and a caller can hand an engine an
    ``asset_dir`` that the manifest never mentioned at all.

    Deliberately not a claim about the Hugging Face hub cache: that directory is shared
    by design, product and measurement alike, and passes this test because it is not the
    product *user-data* tree. It does not follow that cached bytes are outside every
    directory product code can open - ``HF_HOME`` or ``HF_HUB_CACHE`` set under the
    product root puts them exactly there. What decides whether that is caught is which
    directories callers hand to this predicate: the measurement loader passes the
    declared ``asset_root`` and any ``asset_dir``, and the engine additionally passes
    the directory it resolved from the cache root, so a redirected cache is refused
    before its digests are read rather than only when a manifest names it.

    An empty root is not a blocker: the caller's own tree is implied, and that is the
    case the registry guard already covers.
    """
    text = str(asset_root or "").strip()
    if not text:
        return ()
    candidate = Path(text)
    if not candidate.is_absolute():
        if not str(base).strip():
            return (f"asset root {text!r} is relative and there is no registry root to resolve it against",)
        candidate = Path(base) / candidate
    resolved = candidate.expanduser().resolve()
    forbidden = scope_tree(other_scope(scope)).expanduser().resolve()
    if overlaps(resolved, forbidden):
        return (
            f"asset root {resolved} is the {other_scope(scope).value} scope's tree or shares it: "
            f"{forbidden}",
        )
    return ()


def ensure_asset_root_isolated(scope: AssetScope, *roots: str | Path | None, base: str | Path = "") -> None:
    """Refuse an asset root the other scope also reads, naming every offending one.

    Every declared root is checked together so a manifest and an ``asset_dir`` override
    cannot be judged separately - the pair, not either half, is what the loader opens.
    """
    blockers = [reason for root in roots for reason in asset_root_blockers(scope, root, base)]
    if blockers:
        raise ConfigurationError(
            "asset root overlaps the other asset scope's tree",
            scope=scope.value,
            reasons=tuple(blockers),
            remedy=(
                "point the asset at this scope's own tree, or at a directory neither scope owns such as "
                "the shared hub cache; if the hub cache itself was configured under the product root, "
                "move HF_HOME/HF_HUB_CACHE rather than widening this check"
            ),
        )
