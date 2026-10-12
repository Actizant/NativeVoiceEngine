"""Portable exchange: verification is the point of the format."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import pytest

from fixtures import speech_like
from nativevoice.core.errors import ConfigurationError, IntegrityError, ProfileInvalid
from nativevoice.profiles.consent import ConsentRecord
from nativevoice.profiles.exchange import (
    AUDIO_MEMBER,
    CHECKSUM_MEMBER,
    CONSENT_MEMBER,
    FORMAT_VERSION,
    MANIFEST_MEMBER,
    export_profile,
    import_profile,
)
from nativevoice.profiles.store import VoiceProfileStore


def consent(**overrides) -> ConsentRecord:
    payload = {
        "owner": "Aparna Bose",
        "basis": "explicit_recording_session",
        "granted_at": time.time() - 30,
        "statement": "I permit this recording to be used to synthesise my voice.",
    }
    payload.update(overrides)
    return ConsentRecord(**payload)


@pytest.fixture
def source(tmp_path: Path) -> VoiceProfileStore:
    return VoiceProfileStore(tmp_path / "source")


@pytest.fixture
def target(tmp_path: Path) -> VoiceProfileStore:
    return VoiceProfileStore(tmp_path / "target")


@pytest.fixture
def exported(tmp_path: Path, source: VoiceProfileStore) -> tuple[VoiceProfileStore, VoiceProfileStore, str, Path]:
    profile = source.create(
        "Aparna", speech_like(rate=16000), consent(), languages=("bn",), model_compatibility=("pocket-class",)
    )
    path = tmp_path / "aparna.ultronvoice"
    export_profile(source, profile.id, path)
    return source, VoiceProfileStore(tmp_path / "target"), profile.id, path


def _rewrite(path: Path, replacements: dict[str, bytes]) -> None:
    """Rebuild an archive with some members replaced; untouched members keep their bytes."""
    with zipfile.ZipFile(path) as bundle:
        members = {name: bundle.read(name) for name in bundle.namelist()}
    members.update(replacements)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)


def _rehash(path: Path, replacements: dict[str, bytes]) -> None:
    """Replace members *and* fix the checksums, so a version bump is not a tamper."""
    import hashlib

    with zipfile.ZipFile(path) as bundle:
        members = {name: bundle.read(name) for name in bundle.namelist()}
    members.update(replacements)
    checksums = json.loads(members[CHECKSUM_MEMBER])
    checksums["members"].update(
        {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in members.items()
            if name != CHECKSUM_MEMBER
        }
    )
    members[CHECKSUM_MEMBER] = json.dumps(checksums).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)


class TestExchange:
    def test_round_trip_preserves_profile_and_audio(self, exported):
        source_store, target_store, profile_id, path = exported
        imported = import_profile(target_store, path)
        assert imported.id == profile_id
        assert imported.source_digest == source_store.get(profile_id).source_digest
        assert target_store.load_audio(profile_id).to_bytes() == source_store.load_audio(profile_id).to_bytes()

    def test_consent_travels_with_the_audio(self, exported):
        _, target_store, profile_id, path = exported
        imported = import_profile(target_store, path)
        assert imported.consent.owner == "Aparna Bose"
        assert imported.consent.binds_audio(imported.source_digest)

    def test_the_file_is_a_zip_with_the_four_members(self, exported):
        *_, path = exported
        with zipfile.ZipFile(path) as bundle:
            assert set(bundle.namelist()) == {
                MANIFEST_MEMBER, AUDIO_MEMBER, CONSENT_MEMBER, CHECKSUM_MEMBER
            }
        assert FORMAT_VERSION.startswith("1.")

    def test_tampered_audio_is_rejected(self, exported):
        _, target_store, _, path = exported
        with zipfile.ZipFile(path) as bundle:
            audio = bytearray(bundle.read(AUDIO_MEMBER))
        audio[-200] ^= 0xFF
        _rewrite(path, {AUDIO_MEMBER: bytes(audio)})
        with pytest.raises(IntegrityError) as refused:
            import_profile(target_store, path)
        assert "checksum" in str(refused.value)

    def test_tampered_manifest_is_rejected(self, exported):
        _, target_store, profile_id, path = exported
        with zipfile.ZipFile(path) as bundle:
            manifest = json.loads(bundle.read(MANIFEST_MEMBER))
        manifest["profile"]["consent"]["owner"] = "Someone Else"
        _rehash(path, {MANIFEST_MEMBER: json.dumps(manifest).encode("utf-8")})
        with pytest.raises(IntegrityError) as refused:
            import_profile(target_store, path)
        assert "consent" in str(refused.value)

    def test_stripped_consent_member_is_rejected(self, exported):
        _, target_store, _, path = exported
        with zipfile.ZipFile(path) as bundle:
            members = {name: bundle.read(name) for name in bundle.namelist()}
        del members[CONSENT_MEMBER]
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name, payload in members.items():
                bundle.writestr(name, payload)
        with pytest.raises(IntegrityError) as refused:
            import_profile(target_store, path)
        assert "incomplete" in str(refused.value)

    def test_extra_member_is_refused(self, exported):
        _, target_store, _, path = exported
        with zipfile.ZipFile(path, "a") as bundle:
            bundle.writestr("payload.txt", b"not part of the format")
        with pytest.raises(IntegrityError) as refused:
            import_profile(target_store, path)
        assert "unknown members" in str(refused.value)

    def test_unknown_major_version_is_refused(self, exported):
        _, target_store, _, path = exported
        with zipfile.ZipFile(path) as bundle:
            manifest = json.loads(bundle.read(MANIFEST_MEMBER))
        manifest["format_version"] = "2.0"
        _rehash(path, {MANIFEST_MEMBER: json.dumps(manifest).encode("utf-8")})
        with pytest.raises(ConfigurationError) as refused:
            import_profile(target_store, path)
        assert refused.value.details["supported"] == FORMAT_VERSION

    def test_minor_version_stays_readable(self, exported):
        _, target_store, _, path = exported
        with zipfile.ZipFile(path) as bundle:
            manifest = json.loads(bundle.read(MANIFEST_MEMBER))
        manifest["format_version"] = "1.4"
        _rehash(path, {MANIFEST_MEMBER: json.dumps(manifest).encode("utf-8")})
        assert import_profile(target_store, path).id == manifest["profile"]["id"]

    def test_duplicate_id_needs_an_explicit_overwrite(self, exported):
        _, target_store, profile_id, path = exported
        import_profile(target_store, path)
        with pytest.raises(ProfileInvalid) as refused:
            import_profile(target_store, path)
        assert "already exists" in str(refused.value)
        replaced = import_profile(target_store, path, overwrite=True)
        assert replaced.id == profile_id
        assert len(target_store.list()) == 1

    def test_export_refuses_when_the_stored_audio_no_longer_matches(self, source, tmp_path):
        profile = source.create("Aparna", speech_like(rate=16000), consent())
        wav = Path(source.root) / profile.id / "reference.wav"
        payload = bytearray(wav.read_bytes())
        payload[-10] ^= 0xFF
        wav.write_bytes(bytes(payload))
        with pytest.raises(Exception):
            export_profile(source, profile.id, tmp_path / "bad.ultronvoice")

    def test_imported_profile_is_readable_from_the_new_store(self, exported):
        _, target_store, profile_id, path = exported
        imported = import_profile(target_store, path)
        assert (Path(target_store.root) / profile_id / "profile.json").exists()
        assert imported.consent.statement.startswith("I permit")
