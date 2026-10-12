"""Consent records for voice profiles.

A voice profile is biometric-adjacent material: the audio *is* a person, and in
this project's target architecture the conditioning asset (an exported KV-cache
state) is even more literally the speaker than a phone number is. So consent is
modelled as data that must exist before storage, is checked before every use, and
can be withdrawn.

What this is: a statement of permission plus a digest binding it to the exact audio
that was consented for, so that swapping the reference audio later is detectable.
What this is not: legal advice, a lawyer-reviewed release form, or a substitute for
one. The wording is preserved verbatim alongside the digest because a normalised or
paraphrased consent is a different document from the one a person agreed to.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from ..core.errors import ProfileInvalid

CONSENT_SCOPE_SYNTHESIS = "synthesis"
CONSENT_SCOPE_TRAINING = "model_training"
CONSENT_SCOPE_REDACTABLE_DELETE = "delete_on_request"

#: Carried in every record so nobody mistakes the presence of this file for legal
#: sufficiency. It is a record of an assertion, not the assertion's validity.
DISCLAIMER = (
    "This record stores the consent statement as given plus a digest of the audio it "
    "covers. It is evidence of what was claimed, not legal advice and not a release form."
)


@dataclass(frozen=True, slots=True)
class ConsentRecord:
    """Who permitted what, when, over which audio, and for how long."""

    owner: str
    basis: str
    granted_at: float
    scope: tuple[str, ...] = (CONSENT_SCOPE_SYNTHESIS, CONSENT_SCOPE_REDACTABLE_DELETE)
    revocable: bool = True
    expires_at: float | None = None
    statement: str = ""
    audio_digest: str = ""
    revoked_at: float | None = None
    revocation_reason: str = ""

    def __post_init__(self) -> None:
        if not self.owner.strip():
            raise ProfileInvalid("a consent record needs an owner", basis=self.basis)
        if not self.basis.strip():
            raise ProfileInvalid("a consent record needs a stated basis", owner=self.owner)
        if not self.scope:
            raise ProfileInvalid("a consent record needs at least one scope", owner=self.owner)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scope

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None

    def expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at

    def usable_for(self, scope: str, now: float) -> tuple[bool, str]:
        """Whether this consent still authorises `scope`, with the reason when not."""
        if self.revoked:
            return False, f"consent was revoked ({self.revocation_reason or 'no reason recorded'})"
        if self.expired(now):
            return False, "consent has expired"
        if not self.has_scope(scope):
            return False, f"consent does not cover {scope!r}; scopes recorded: {', '.join(self.scope)}"
        if self.audio_digest:
            return True, "consent valid"
        return False, "consent has no audio digest, so it cannot be bound to this recording"

    def binds_audio(self, digest: str) -> bool:
        return bool(self.audio_digest) and self.audio_digest == digest

    def revoke(self, reason: str = "", at: float | None = None) -> "ConsentRecord":
        if not self.revocable:
            raise ProfileInvalid(
                "this consent is recorded as irrevocable; treat the profile as unusable and "
                "delete it rather than rewriting the record",
                owner=self.owner,
            )
        return replace(self, revoked_at=at, revocation_reason=reason)

    def as_dict(self) -> dict:
        return {
            "owner": self.owner,
            "basis": self.basis,
            "granted_at": self.granted_at,
            "scope": list(self.scope),
            "revocable": self.revocable,
            "expires_at": self.expires_at,
            "statement": self.statement,
            "audio_digest": self.audio_digest,
            "revoked_at": self.revoked_at,
            "revocation_reason": self.revocation_reason,
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def from_dict(payload: dict) -> "ConsentRecord":
        try:
            return ConsentRecord(
                owner=payload["owner"],
                basis=payload["basis"],
                granted_at=float(payload["granted_at"]),
                scope=tuple(payload.get("scope") or ()),
                revocable=bool(payload.get("revocable", True)),
                expires_at=payload.get("expires_at"),
                statement=payload.get("statement", ""),
                audio_digest=payload.get("audio_digest", ""),
                revoked_at=payload.get("revoked_at"),
                revocation_reason=payload.get("revocation_reason", ""),
            )
        except (KeyError, TypeError, ValueError) as invalid:
            raise ProfileInvalid("consent record is malformed", error=str(invalid)) from invalid


def digest_statement(record: ConsentRecord) -> str:
    """Content digest of the statement itself, so an edit is detectable."""
    canonical = json.dumps(record.as_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
