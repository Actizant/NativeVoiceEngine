"""Local storage and governance for voice profiles.

A profile is three things kept together: metadata JSON, the 16-bit mono reference
WAV, and the consent record that permits its use. They live in one directory per
profile so `delete()` can remove *every* trace in a single operation - the honest
answer to "please erase my voice" - and so a partial state is visible rather than
assumed.

Governance enforced here rather than in a caller's discipline:

* creation without a consent record is refused;
* a revoked or expired consent makes the profile unusable for synthesis even
  though the bytes are still on disk (deletion is a separate, explicit act);
* the consent's audio digest is bound to the stored recording, so swapping the WAV
  under an existing profile is detected;
* identifiers are validated and the resolved path is checked against the store
  root, because profile ids arrive from callers and a `..` in one is a filesystem
  escape.

Conditioning itself is not here: turning audio into a usable voice state needs an
approved synthesis model, which does not exist yet (see `conditioning.py`).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..audio.wav import read_wav, write_wav
from ..core.audio import AudioBuffer
from ..core.errors import ProfileInvalid, StorageError
from ..core.language import LanguageTag
from ..core.time import new_id
from ..tts.api import VoiceReference
from .consent import CONSENT_SCOPE_SYNTHESIS, ConsentRecord
from .quality import PROVISIONAL_MINIMUMS, QualityMinimums, assess

PROFILE_ID_PREFIX = "voice"
METADATA_FILE = "profile.json"
AUDIO_FILE = "reference.wav"
CONSENT_FILE = "consent.json"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """How long a profile is held. Consent expiry is separate and tighter."""

    days: int | None = None
    basis: str = "explicit_consent"

    def expired(self, created_at: float, now: float) -> bool:
        return self.days is not None and now >= created_at + self.days * 86400.0

    def as_dict(self) -> dict:
        return {"days": self.days, "basis": self.basis}

    @staticmethod
    def from_dict(payload: dict) -> "RetentionPolicy":
        return RetentionPolicy(days=payload.get("days"), basis=payload.get("basis", "explicit_consent"))


@dataclass(frozen=True, slots=True)
class RemovedArtifacts:
    """Exactly what a delete removed, so the operation can be reported truthfully."""

    profile_id: str
    paths: tuple[str, ...]
    directory_removed: bool

    def as_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "paths": list(self.paths),
            "directory_removed": self.directory_removed,
        }


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    """One stored speaker reference plus the record of why it may be used."""

    id: str
    display_name: str
    created_at: float
    languages: tuple[str, ...]
    model_compatibility: tuple[str, ...]
    source_digest: str
    sample_rate: int
    channels: int
    duration_s: float
    consent: ConsentRecord
    retention: RetentionPolicy = RetentionPolicy()
    audio_file: str = AUDIO_FILE
    quality: dict = field(default_factory=dict)

    def consent_usable(self, scope: str = CONSENT_SCOPE_SYNTHESIS, now: float | None = None) -> tuple[bool, str]:
        """Whether this profile may speak right now."""
        moment = time.time() if now is None else now
        if self.retention.expired(self.created_at, moment):
            return False, f"retention policy of {self.retention.days} days has elapsed"
        return self.consent.usable_for(scope, moment)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "languages": list(self.languages),
            "model_compatibility": list(self.model_compatibility),
            "source_digest": self.source_digest,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_s": round(self.duration_s, 6),
            "consent": self.consent.as_dict(),
            "retention": self.retention.as_dict(),
            "audio_file": self.audio_file,
            "quality": self.quality,
        }

    @staticmethod
    def from_dict(payload: dict) -> "VoiceProfile":
        try:
            return VoiceProfile(
                id=payload["id"],
                display_name=payload["display_name"],
                created_at=float(payload["created_at"]),
                languages=tuple(payload.get("languages") or ()),
                model_compatibility=tuple(payload.get("model_compatibility") or ()),
                source_digest=payload.get("source_digest", ""),
                sample_rate=int(payload["sample_rate"]),
                channels=int(payload["channels"]),
                duration_s=float(payload.get("duration_s", 0.0)),
                consent=ConsentRecord.from_dict(payload["consent"]),
                retention=RetentionPolicy.from_dict(payload.get("retention") or {}),
                audio_file=payload.get("audio_file", AUDIO_FILE),
                quality=dict(payload.get("quality") or {}),
            )
        except (KeyError, TypeError, ValueError) as invalid:
            raise ProfileInvalid("profile metadata is malformed", error=str(invalid)) from invalid


def audio_digest(buffer: AudioBuffer) -> str:
    """Content hash of the PCM, the thing consent is actually about."""
    return hashlib.sha256(buffer.to_bytes()).hexdigest()


class VoiceProfileStore:
    """Create, read, list and delete voice profiles under one root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.problems: tuple[str, ...] = ()

    def create(
        self,
        display_name: str,
        audio: AudioBuffer,
        consent: ConsentRecord | None,
        *,
        languages: tuple[str, ...] = (),
        model_compatibility: tuple[str, ...] = (),
        retention: RetentionPolicy | None = None,
        minimums: QualityMinimums = PROVISIONAL_MINIMUMS,
        scope: str = CONSENT_SCOPE_SYNTHESIS,
    ) -> VoiceProfile:
        """Intake a profile: consent first, then quality, then write.

        Order matters. An unconsented recording must not reach disk even if it is
        perfect audio, and a bad recording must not be stored with a consent bound
        to bytes we are throwing away. `scope` is what the intake is *for*: a
        training-only consent may store audio, but the same profile stays
        unusable for synthesis because the scopes are checked again on use.
        """
        if not display_name.strip():
            raise ProfileInvalid("a voice profile needs a display name")
        if consent is None:
            raise ProfileInvalid(
                "a voice profile cannot be created without a consent record",
                display_name=display_name,
                remedy="capture consent (owner, basis, scope) before storing reference audio",
            )
        if audio.format.channels != 1:
            audio = audio.to_mono()
        digest = audio_digest(audio)
        bound = consent if consent.audio_digest == digest else replace(consent, audio_digest=digest)
        usable, reason = bound.usable_for(scope, time.time())
        if not usable:
            raise ProfileInvalid(
                "consent does not permit storing this voice",
                reason=reason,
                owner=bound.owner,
                scope=scope,
            )
        report = assess(audio, minimums)
        if not report.usable:
            raise ProfileInvalid(
                "candidate recording fails intake quality checks",
                failed=report.failed_checks,
                reasons=list(report.reasons),
                minimums=minimums.explain(),
            )
        profile = VoiceProfile(
            id=new_id(PROFILE_ID_PREFIX),
            display_name=display_name.strip(),
            created_at=time.time(),
            languages=tuple(LanguageTag(code).root().code for code in languages),
            model_compatibility=tuple(model_compatibility),
            source_digest=digest,
            sample_rate=audio.format.sample_rate,
            channels=audio.format.channels,
            duration_s=audio.duration_s,
            consent=bound,
            retention=retention or RetentionPolicy(),
            quality=report.as_dict(),
        )
        return self.restore(profile, audio)

    def restore(self, profile: VoiceProfile, audio: AudioBuffer) -> VoiceProfile:
        """Write a fully-formed profile to disk; used by intake and by import.

        The consent is bound to the exact bytes being stored, so a profile whose
        metadata and audio disagree cannot be written, imported or resurrected.
        """
        directory = self._directory(profile.id)
        if directory.exists():
            raise ProfileInvalid("profile id already exists in this store", profile_id=profile.id)
        if audio_digest(audio) != profile.source_digest:
            raise ProfileInvalid(
                "reference audio does not match the recorded digest",
                profile_id=profile.id,
                expected=profile.source_digest[:12],
                measured=audio_digest(audio)[:12],
            )
        directory.mkdir(parents=True)
        write_wav(str(directory / AUDIO_FILE), audio)
        self._write(directory / CONSENT_FILE, profile.consent.as_dict())
        self._write(directory / METADATA_FILE, profile.as_dict())
        return profile

    def get(self, profile_id: str) -> VoiceProfile:
        directory = self._directory(profile_id)
        metadata = directory / METADATA_FILE
        if not metadata.exists():
            raise ProfileInvalid("no such voice profile", profile_id=profile_id)
        return VoiceProfile.from_dict(self._read_json(metadata))

    def load_audio(self, profile_id: str) -> AudioBuffer:
        profile = self.get(profile_id)
        path = self._directory(profile_id) / profile.audio_file
        if not path.exists():
            raise StorageError("profile metadata exists but its audio does not", profile_id=profile_id)
        audio = read_wav(str(path))
        if audio_digest(audio) != profile.source_digest:
            raise ProfileInvalid(
                "stored audio no longer matches its consent-bound digest",
                profile_id=profile_id,
            )
        return audio

    def list(self) -> tuple[VoiceProfile, ...]:
        """Every readable profile; unreadable directories land in `problems`.

        A silently-skipped profile would make "delete everything of mine" look
        complete when it was not, so the failures are surfaced rather than dropped.
        """
        found: list[VoiceProfile] = []
        problems: list[str] = []
        for entry in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not entry.is_dir():
                continue
            try:
                found.append(VoiceProfile.from_dict(self._read_json(entry / METADATA_FILE)))
            except (OSError, ProfileInvalid, StorageError) as failure:
                problems.append(f"{entry.name}: {failure}")
        self.problems = tuple(problems)
        return tuple(found)

    def delete(self, profile_id: str) -> RemovedArtifacts:
        directory = self._directory(profile_id)
        if not directory.exists():
            raise ProfileInvalid("no such voice profile", profile_id=profile_id)
        removed: list[str] = []
        for stray in sorted(path for path in directory.rglob("*") if path.is_file()):
            stray.unlink()
            removed.append(str(stray))
        for nested in sorted((path for path in directory.rglob("*") if path.is_dir()), reverse=True):
            nested.rmdir()
        directory.rmdir()
        return RemovedArtifacts(profile_id, tuple(removed), not directory.exists())

    def revoke(self, profile_id: str, reason: str = "", at: float | None = None) -> VoiceProfile:
        """Record a withdrawal. The audio stays until `delete()`, but cannot be used."""
        profile = self.get(profile_id)
        revoked = replace(profile.consent, revoked_at=at if at is not None else time.time(),
                          revocation_reason=reason or "revoked by owner")
        updated = replace(profile, consent=revoked)
        directory = self._directory(profile_id)
        self._write(directory / CONSENT_FILE, updated.consent.as_dict())
        self._write(directory / METADATA_FILE, updated.as_dict())
        return updated

    def voice_reference(
        self, profile_id: str, scope: str = CONSENT_SCOPE_SYNTHESIS, now: float | None = None
    ) -> VoiceReference:
        """The consent-checked reference the pipeline is allowed to condition on."""
        profile = self.get(profile_id)
        usable, reason = profile.consent_usable(scope, now)
        if not usable:
            raise ProfileInvalid(
                "this voice profile may not be used for synthesis",
                profile_id=profile_id,
                reason=reason,
            )
        audio = self.load_audio(profile_id)
        return VoiceReference(
            voice_id=profile.id,
            audio=audio,
            language=LanguageTag(profile.languages[0]) if profile.languages else None,
            consent_token=profile.consent.audio_digest or profile.id,
            owner=profile.consent.owner,
            source=f"profile:{profile.id}",
        )

    def model_compatible(self, profile_id: str, model_key: str) -> bool:
        """Whether a profile was captured for a specific model family."""
        return model_key in self.get(profile_id).model_compatibility

    def _directory(self, profile_id: str) -> Path:
        if not _SAFE_ID.match(profile_id):
            raise ProfileInvalid(
                "profile id is not a safe directory name",
                profile_id=profile_id,
                rule="alphanumeric plus - and _, 3-64 characters, no separators",
            )
        candidate = (self.root / profile_id).resolve()
        if not candidate.is_relative_to(self.root.resolve()):
            raise ProfileInvalid("profile id escapes the store root", profile_id=profile_id)
        return candidate

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as failure:
            raise StorageError("could not write profile metadata", path=str(path), error=str(failure)) from failure

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError as failure:
            raise StorageError("profile file unreadable", path=str(path), error=str(failure)) from failure
        except json.JSONDecodeError as failure:
            raise ProfileInvalid("profile JSON is corrupt", path=str(path), error=str(failure)) from failure
