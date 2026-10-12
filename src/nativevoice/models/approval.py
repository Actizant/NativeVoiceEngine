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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..core.errors import ConfigurationError, ModelNotApproved
from .api import ModelManifest, ReviewStatus

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
        blockers = approval_blockers(manifest)
        record = ApprovalOverride(model_id=model_id, reason=reason.strip(), blockers=blockers)
        self.overrides[model_id] = record
        logger.warning(
            "approval override granted for %s: %s (blocked by: %s)",
            model_id,
            record.reason,
            "; ".join(blockers) or "none",
        )
        return record
