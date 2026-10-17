"""The licence gate: default-deny approval of model assets.

Nothing may be loaded unless a human has approved it. Approval requires three
independent facts to hold at once, because each closes a different loophole:

* ``review is APPROVED`` - the review decision itself;
* ``approved_by`` is non-empty - a *named* approver, so "approved" can never be
  anonymous or stamped by a bot that filled the field with whitespace;
* ``provenance.incomplete()`` is empty - the licence/source/redistribution facts
  the decision rests on are actually on record.

There is deliberately no environment variable, config flag, or ambient state that
flips this open. A caller who genuinely needs to run an unapproved asset must go
through :meth:`ApprovalGate.override`, which demands the model id *and* a reason
string and writes a diagnostic record; it cannot be reached by accident and it
never goes silent.

An override opens the approval gate and nothing else. Which *path* an asset may
serve is a different question, checked separately by :func:`ensure_product_scope`,
so overriding a measurement-scoped asset cannot make it reachable from the product
path - and an override on an unreviewed product-scoped asset still does what the
contract above promises.

:func:`ensure_measurement_use` is the same boundary seen from the other side: what
authorises running a research asset locally is its measurement scope, a named
authoriser and a complete provenance record, none of which is the product approval
this module's ``review`` axis describes. ``review`` therefore grants nothing here -
and reads once, in the refusing direction only: a recorded ``REJECTED`` bars
measurement use as well as the shipped path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..core.errors import ConfigurationError, ModelNotApproved
from .api import AssetScope, ModelManifest, ReviewStatus

logger = logging.getLogger("nativevoice.models.approval")


def approval_blockers(manifest: ModelManifest) -> tuple[str, ...]:
    """The concrete reasons this manifest is not approved (empty when it is)."""
    blockers: list[str] = []
    if manifest.review is not ReviewStatus.APPROVED:
        blockers.append(f"review status is {manifest.review.value}, not approved")
    if not manifest.approved_by.strip():
        blockers.append("no named approver recorded")
    missing = manifest.provenance.incomplete()
    if missing:
        blockers.append("provenance incomplete: " + ", ".join(missing))
    return tuple(blockers)


def is_approved(manifest: ModelManifest) -> bool:
    return not approval_blockers(manifest)


def ensure_approved(manifest: ModelManifest) -> None:
    """Raise ``ModelNotApproved`` unless the manifest passes every approval axis."""
    blockers = approval_blockers(manifest)
    if blockers:
        raise ModelNotApproved(
            "model is not approved for use",
            model_id=manifest.model_id,
            reasons=blockers,
        )


def scope_blockers(
    manifest: ModelManifest, registry_scope: AssetScope = AssetScope.PRODUCT
) -> tuple[str, ...]:
    """Why this manifest may not be served by a registry of `registry_scope` (empty when it may)."""
    if manifest.use_scope is registry_scope:
        return ()
    return (
        f"use scope is {manifest.use_scope.value}; this registry serves {registry_scope.value}-scoped assets only",
    )


def product_blockers(manifest: ModelManifest) -> tuple[str, ...]:
    """Approval *and* scope - the full reason set for the shipped voice path.

    A review record is not evidence of distribution permission, so a measurement
    asset is blocked here even when :func:`approval_blockers` returns nothing.
    """
    return approval_blockers(manifest) + scope_blockers(manifest)


def ensure_product_scope(manifest: ModelManifest) -> None:
    """Raise ``ModelNotApproved`` unless this asset's *scope* allows the product path.

    Scope only, so the two axes keep separate owners: this guard cannot be widened
    by an :class:`ApprovalGate` override, and it cannot swallow the override's job.
    A product-scoped asset that is merely unreviewed passes here and is then
    refused - or overridden - by :meth:`ApprovalGate.check`, exactly as before.
    """
    blockers = scope_blockers(manifest)
    if blockers:
        raise ModelNotApproved(
            "model is not eligible for the product path",
            model_id=manifest.model_id,
            reasons=blockers,
        )


def measurement_blockers(manifest: ModelManifest) -> tuple[str, ...]:
    """Why this manifest may not be loaded for local measurement (empty when it may).

    Three facts, and deliberately *not* the review state's permission. ``review``
    answers whether a human cleared the asset for the shipped path; measuring a file
    that is already on this machine is a different decision, and tying it to
    ``APPROVED`` would make an approval record the only way to do research - which is
    how the two axes got conflated before. What is required is that the asset announced
    itself as measurement-scoped, that a named person authorised this use, and that the
    provenance it rests on is recorded rather than assumed: an untagged licence stays a
    visible gap, and ``NOASSERTION`` is a recorded answer to a question, not a blank.

    One thing on the review axis does reach this path, and only in the refusing
    direction: ``REJECTED``. That is not an absent approval, it is a recorded decision
    that the asset is not to be used, and a research seam that overruled it would make
    this the second door the licence gate was built to not have. ``UNREVIEWED`` records
    no such decision, so it stays loadable - which is the distinction the axis exists
    to keep.
    """
    blockers: list[str] = []
    if manifest.review is ReviewStatus.REJECTED:
        blockers.append("review status is rejected: a recorded refusal bars measurement use, not only the product path")
    blockers += list(scope_blockers(manifest, AssetScope.MEASUREMENT))
    if not manifest.approved_by.strip():
        blockers.append("no named authoriser recorded for measurement use")
    missing = manifest.provenance.incomplete()
    if missing:
        blockers.append("provenance incomplete: " + ", ".join(missing))
    return tuple(blockers)


def ensure_measurement_use(manifest: ModelManifest) -> None:
    """Raise ``ModelNotApproved`` unless this asset may be loaded for measurement.

    A product-scoped manifest is refused here too, and that direction matters: a gate
    that simply declined to *widen* approval would let an asset on the shipped path be
    run through the measurement seam without an approval record at all. Nothing on
    this axis consults :class:`ApprovalGate`, so there is no override to reach for -
    the refusal is the whole answer, and it holds for a recorded rejection as well:
    overrides live in the gate and are never written back to a manifest, so a human who
    rejected an asset cannot be quietly overruled by this route.
    """
    blockers = measurement_blockers(manifest)
    if blockers:
        raise ModelNotApproved(
            "model is not authorized for measurement use",
            model_id=manifest.model_id,
            reasons=blockers,
        )


def ensure_product_eligible(manifest: ModelManifest) -> None:
    """Raise ``ModelNotApproved`` unless this asset is approved *and* scoped for product.

    The combined question, for callers that want one answer rather than two:
    :func:`product_blockers` is the reporting form of the same rule. The load seam
    deliberately does *not* use this - it calls :func:`ensure_product_scope` and
    leaves the approval axis to :class:`ApprovalGate`, whose override is a human
    decision about one asset's approval record, not a licence to route it.
    """
    blockers = product_blockers(manifest)
    if blockers:
        raise ModelNotApproved(
            "model is not eligible for the product path",
            model_id=manifest.model_id,
            reasons=blockers,
        )


@dataclass(slots=True)
class ApprovalOverride:
    """A logged, explicit decision to run an asset that failed the gate."""

    model_id: str
    reason: str
    blockers: tuple[str, ...]


@dataclass(slots=True)
class ApprovalGate:
    """Decides whether a manifest may load, with a non-accidental override path.

    An empty gate is pure default-deny: it behaves exactly like
    :func:`ensure_approved`. Overrides are keyed by model id and each carries the
    reason a human gave, so the audit trail survives.
    """

    overrides: dict[str, ApprovalOverride] = field(default_factory=dict)

    def check(self, manifest: ModelManifest) -> None:
        try:
            ensure_approved(manifest)
            return
        except ModelNotApproved:
            override = self.overrides.get(manifest.model_id)
            if override is None:
                raise
            logger.warning(
                "approval override active for %s: %s (blocked by: %s)",
                override.model_id,
                override.reason,
                "; ".join(override.blockers),
            )

    def override(self, manifest: ModelManifest, *, model_id: str, reason: str) -> ApprovalOverride:
        """Record an explicit override. Requires the matching model id and a reason.

        The doubled model-id argument is intentional: the caller has to spell out
        the id they are approving, so a copy-paste of the wrong manifest cannot
        silently authorise it.
        """
        if not isinstance(model_id, str) or model_id != manifest.model_id:
            raise ConfigurationError(
                "override model_id must match the manifest",
                requested=model_id,
                manifest=manifest.model_id,
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigurationError("override requires a non-empty reason", model_id=model_id)
        # Both axes are recorded even though only the approval axis is being opened.
        # A fully approved measurement asset has no approval blockers, so logging
        # those alone would print "blocked by: none" over a decision that is being
        # taken for an asset the product path must still refuse - the one reading
        # this record must not be able to form.
        blockers = product_blockers(manifest)
        record = ApprovalOverride(model_id=model_id, reason=reason.strip(), blockers=blockers)
        self.overrides[model_id] = record
        logger.warning(
            "approval override granted for %s: %s (blocked by: %s)",
            model_id,
            record.reason,
            "; ".join(blockers) or "none",
        )
        return record
