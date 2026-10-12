"""Portable voice-profile exchange: one file out, one file in.

Why a zip and not a database dump: users move recordings between machines, and the
transfer has to carry the consent with the audio. A WAV copied on its own loses
exactly the information that made copying it legitimate. So the container is
manifest + WAV + consent record + checksums, and import re-verifies all three
against each other rather than trusting the file it was handed.

`FORMAT_VERSION` guards against a future reader silently misinterpreting today's
export: a minor bump stays readable, an unknown major version is refused.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ..audio.wav import read_wav, write_wav
from ..core.errors import ConfigurationError, IntegrityError, ProfileInvalid, StorageError
from .consent import ConsentRecord
from .store import (
    AUDIO_FILE,
    VoiceProfile,
    VoiceProfileStore,
    audio_digest,
)

FORMAT_VERSION = "1.0"
MANIFEST_MEMBER = "manifest.json"
CONSENT_MEMBER = "consent.json"
AUDIO_MEMBER = AUDIO_FILE
CHECKSUM_MEMBER = "CHECKSUMS.json"
REQUIRED_MEMBERS = (MANIFEST_MEMBER, AUDIO_MEMBER, CONSENT_MEMBER, CHECKSUM_MEMBER)


def _major(version: str) -> int:
    try:
        return int(str(version).split(".")[0])
    except (TypeError, ValueError):
        raise ConfigurationError("exchange format version is not parseable", version=version) from None


def export_profile(store: VoiceProfileStore, profile_id: str, path: str | Path) -> str:
    """Write one profile as a portable, self-describing archive."""
    profile = store.get(profile_id)
    audio = store.load_audio(profile_id)
    if audio_digest(audio) != profile.source_digest:
        raise IntegrityError(
            "refusing to export a profile whose audio no longer matches its digest",
            profile_id=profile_id,
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": FORMAT_VERSION,
        "exported_at": _utc_stamp(),
        "profile": profile.as_dict(),
    }
    # Consent travels as its own member so a reader can audit permission without
    # parsing profile metadata, and so a stripped-consent archive fails verification.
    consent = profile.consent.as_dict()
    consent["format_version"] = FORMAT_VERSION
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        audio_bytes = _wav_bytes(audio)
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        consent_bytes = json.dumps(consent, ensure_ascii=False, indent=2).encode("utf-8")
        checksums = {
            "format_version": FORMAT_VERSION,
            "members": {
                MANIFEST_MEMBER: hashlib.sha256(manifest_bytes).hexdigest(),
                AUDIO_MEMBER: hashlib.sha256(audio_bytes).hexdigest(),
                CONSENT_MEMBER: hashlib.sha256(consent_bytes).hexdigest(),
            },
        }
        bundle.writestr(MANIFEST_MEMBER, manifest_bytes)
        bundle.writestr(AUDIO_MEMBER, audio_bytes)
        bundle.writestr(CONSENT_MEMBER, consent_bytes)
        bundle.writestr(CHECKSUM_MEMBER, json.dumps(checksums, indent=2).encode("utf-8"))
    return str(target)


def import_profile(
    store: VoiceProfileStore, path: str | Path, *, overwrite: bool = False
) -> VoiceProfile:
    """Load an archive, verify it, and install the profile under its original id."""
    source = Path(path)
    if not source.exists():
        raise StorageError("exchange file not found", path=str(source))
    with zipfile.ZipFile(source) as bundle:
        names = set(bundle.namelist())
        missing = [member for member in REQUIRED_MEMBERS if member not in names]
        if missing:
            raise IntegrityError("exchange archive is incomplete", missing=missing)
        unexpected = sorted(names.difference(REQUIRED_MEMBERS))
        if unexpected:
            raise IntegrityError("exchange archive carries unknown members", members=unexpected)
        payload = {member: bundle.read(member) for member in REQUIRED_MEMBERS}
    _verify_checksums(payload[CHECKSUM_MEMBER], payload)
    manifest = json.loads(payload[MANIFEST_MEMBER].decode("utf-8"))
    version = manifest.get("format_version", "")
    if _major(version) != _major(FORMAT_VERSION):
        raise ConfigurationError(
            "exchange format major version is not supported",
            found=version,
            supported=FORMAT_VERSION,
        )
    profile = VoiceProfile.from_dict(manifest["profile"])
    consent = ConsentRecord.from_dict(json.loads(payload[CONSENT_MEMBER].decode("utf-8")))
    if profile.consent.as_dict() != consent.as_dict():
        raise IntegrityError(
            "the consent record in this archive does not match the profile metadata",
            profile_id=profile.id,
        )
    audio = _read_wav_bytes(payload[AUDIO_MEMBER])
    if audio_digest(audio) != profile.source_digest:
        raise IntegrityError(
            "reference audio does not match the digest recorded in the manifest",
            profile_id=profile.id,
        )
    if not consent.binds_audio(profile.source_digest):
        raise ProfileInvalid(
            "consent in this archive is not bound to the audio it carries",
            profile_id=profile.id,
        )
    existing = any(stored.id == profile.id for stored in store.list())
    if existing:
        if not overwrite:
            raise ProfileInvalid(
                "a profile with this id already exists in the store",
                profile_id=profile.id,
                remedy="pass overwrite=True to replace it, after confirming consent still holds",
            )
        store.delete(profile.id)
    return store.restore(profile, audio)


def _verify_checksums(checksum_bytes: bytes, payload: dict[str, bytes]) -> None:
    recorded = json.loads(checksum_bytes.decode("utf-8"))
    members = recorded.get("members") or {}
    for name, expected in members.items():
        measured = hashlib.sha256(payload[name]).hexdigest()
        if measured != expected:
            raise IntegrityError(
                "exchange member does not match its recorded checksum",
                member=name,
                expected=expected[:12],
                measured=measured[:12],
            )


def _wav_bytes(audio) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "reference.wav")
        write_wav(path, audio)
        return Path(path).read_bytes()


def _read_wav_bytes(payload: bytes):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reference.wav"
        path.write_bytes(payload)
        return read_wav(str(path))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
