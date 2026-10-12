"""Voice profile storage, consent, quality intake and conditioning refusal."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from fixtures import noise, silence, sine, speech_like
from nativevoice.core.errors import CapabilityUnavailable, ProfileInvalid
from nativevoice.core.language import CapabilityStatus, LanguageTag, Task
from nativevoice.profiles.conditioning import (
    MISSING_APPROVED_MODEL,
    NoApprovedConditioner,
    cross_language_evaluation_plan,
)
from nativevoice.profiles.consent import (
    CONSENT_SCOPE_SYNTHESIS,
    CONSENT_SCOPE_TRAINING,
    DISCLAIMER,
    ConsentRecord,
    digest_statement,
)
from nativevoice.profiles.quality import (
    PROVISIONAL_MINIMUMS,
    assess,
    energy_contrast,
    estimate_snr_db,
)
from nativevoice.profiles.store import RetentionPolicy, VoiceProfileStore, audio_digest
from nativevoice.tts.api import VoiceReference


def consent(**overrides) -> ConsentRecord:
    payload = {
        "owner": "Aparna Bose",
        "basis": "explicit_recording_session",
        "granted_at": time.time() - 60,
        "statement": "I permit this recording to be used to synthesise my voice.",
    }
    payload.update(overrides)
    return ConsentRecord(**payload)


@pytest.fixture
def store(tmp_path: Path) -> VoiceProfileStore:
    return VoiceProfileStore(tmp_path / "voices")


@pytest.fixture
def profile_store(store: VoiceProfileStore) -> tuple[VoiceProfileStore, str]:
    profile = store.create(
        "Aparna", speech_like(rate=16000), consent(), languages=("bn",), model_compatibility=("pocket-class",)
    )
    return store, profile.id


class TestConsent:
    def test_creation_without_consent_is_refused(self, store):
        with pytest.raises(ProfileInvalid) as refused:
            store.create("Aparna", speech_like(), None)
        assert "consent" in str(refused.value)

    def test_a_stored_profile_carries_its_consent_record(self, profile_store):
        store, profile_id = profile_store
        profile = store.get(profile_id)
        assert profile.consent.owner == "Aparna Bose"
        assert profile.consent.audio_digest == profile.source_digest
        assert (Path(store.root) / profile_id / "consent.json").exists()

    def test_consent_is_digest_bound_to_the_exact_audio(self, profile_store):
        store, profile_id = profile_store
        profile = store.get(profile_id)
        other = speech_like(rate=16000, bursts=((0.05, 0.45), (0.55, 0.9)))
        assert audio_digest(other) != profile.source_digest
        assert not profile.consent.binds_audio(audio_digest(other))

    def test_revoked_consent_makes_the_profile_unusable_for_synthesis(self, profile_store):
        store, profile_id = profile_store
        store.revoke(profile_id, "owner asked to be forgotten")
        with pytest.raises(ProfileInvalid) as refused:
            store.voice_reference(profile_id)
        assert "revoked" in refused.value.details["reason"]

    def test_revocation_does_not_delete_the_bytes(self, profile_store):
        store, profile_id = profile_store
        store.revoke(profile_id)
        assert (Path(store.root) / profile_id / "reference.wav").exists()

    def test_expired_consent_is_refused_at_intake(self, store):
        with pytest.raises(ProfileInvalid) as refused:
            store.create("Late", speech_like(rate=16000), consent(expires_at=time.time() - 1))
        assert "expired" in refused.value.details["reason"]

    def test_consent_that_expires_later_blocks_use_afterwards(self, store):
        profile = store.create(
            "Temporary", speech_like(rate=16000), consent(expires_at=time.time() + 3600)
        )
        with pytest.raises(ProfileInvalid) as refused:
            store.voice_reference(profile.id, now=time.time() + 7200)
        assert "expired" in refused.value.details["reason"]

    def test_scope_is_enforced(self, store):
        profile = store.create(
            "Training only",
            speech_like(rate=16000),
            consent(scope=(CONSENT_SCOPE_TRAINING,)),
            scope=CONSENT_SCOPE_TRAINING,
        )
        usable, reason = store.get(profile.id).consent_usable(CONSENT_SCOPE_SYNTHESIS)
        assert usable is False
        assert "does not cover" in reason

    def test_intake_cannot_be_done_under_an_unrelated_scope(self, store):
        with pytest.raises(ProfileInvalid):
            store.create(
                "Wrong scope",
                speech_like(rate=16000),
                consent(scope=(CONSENT_SCOPE_TRAINING,)),
                scope=CONSENT_SCOPE_SYNTHESIS,
            )

    def test_consent_is_a_statement_plus_digest_not_legal_advice(self):
        record = consent()
        assert DISCLAIMER in json.dumps(record.as_dict())
        assert len(digest_statement(record)) == 64

    def test_malformed_consent_is_rejected(self):
        with pytest.raises(ProfileInvalid):
            ConsentRecord.from_dict({"owner": "x"})
        with pytest.raises(ProfileInvalid):
            ConsentRecord(owner="  ", basis="explicit", granted_at=time.time())
        with pytest.raises(ProfileInvalid):
            ConsentRecord(owner="Aparna", basis="", granted_at=time.time())
        with pytest.raises(ProfileInvalid):
            ConsentRecord(owner="Aparna", basis="explicit", granted_at=time.time(), scope=())


class TestQuality:
    def test_speech_like_fixture_clears_the_minimums(self):
        report = assess(speech_like(rate=16000))
        assert report.usable, report.reasons
        assert report.minimums.label.startswith("provisional")

    @pytest.mark.parametrize(
        "candidate",
        [
            pytest.param(sine(rate=16000, seconds=2.0, amplitude=0.3), id="pure-tone"),
            pytest.param(noise(rate=16000, seconds=2.0, amplitude=0.02), id="low-noise"),
            pytest.param(noise(rate=16000, seconds=2.0, amplitude=0.3), id="loud-noise"),
            pytest.param(silence(rate=16000, seconds=2.0), id="silence"),
            pytest.param(speech_like(rate=16000, bursts=((0.1, 0.35),)), id="too-short"),
        ],
    )
    def test_rejects_non_utterances(self, candidate):
        report = assess(candidate)
        assert report.usable is False
        assert report.failed_checks

    def test_tone_is_caught_by_envelope_not_by_loudness(self):
        tone = sine(rate=16000, seconds=2.0, amplitude=0.3)
        report = assess(tone)
        assert "energy_contrast" in report.failed_checks
        assert estimate_snr_db(tone) < PROVISIONAL_MINIMUMS.min_snr_db
        assert energy_contrast(tone) == pytest.approx(0.0, abs=1e-6)

    def test_clipped_recording_is_rejected(self):
        clipped = sine(rate=16000, seconds=1.5, amplitude=1.0)
        assert "clipping" in assess(clipped).failed_checks or not assess(clipped).usable

    def test_store_refuses_a_bad_recording_at_intake(self, store):
        with pytest.raises(ProfileInvalid) as refused:
            store.create("Hiss", noise(seconds=2.0), consent())
        assert refused.value.details["failed"]

    def test_report_is_serialisable(self):
        payload = assess(speech_like(rate=16000)).as_dict()
        assert payload["usable"] is True
        assert {check["name"] for check in payload["checks"]}


class TestStoreLifecycle:
    def test_create_get_list_delete(self, profile_store):
        store, profile_id = profile_store
        assert store.get(profile_id).id == profile_id
        assert [profile.id for profile in store.list()] == [profile_id]
        removed = store.delete(profile_id)
        assert removed.directory_removed is True
        assert len(removed.paths) == 3
        assert store.list() == ()
        assert not (Path(store.root) / profile_id).exists()

    def test_delete_removes_every_trace_including_strays(self, store):
        profile = store.create("Aparna", speech_like(rate=16000), consent())
        stray = Path(store.root) / profile.id / "notes.txt"
        stray.write_text("leftover", encoding="utf-8")
        removed = store.delete(profile.id)
        assert not Path(store.root, profile.id).exists()
        assert any("notes.txt" in path for path in removed.paths)

    def test_stored_audio_round_trips_byte_for_byte(self, profile_store):
        store, profile_id = profile_store
        original = speech_like(rate=16000)
        assert store.load_audio(profile_id).to_bytes() == original.to_bytes()

    @pytest.mark.parametrize(
        "bad_id", ["../../etc/passwd", "a/b", "..", "with sep", "semi;colon", "x" * 80]
    )
    def test_path_traversal_ids_are_rejected(self, store, bad_id):
        with pytest.raises(ProfileInvalid):
            store.get(bad_id)
        with pytest.raises(ProfileInvalid):
            store.delete(bad_id)

    def test_traversal_cannot_reach_outside_the_root(self, store, tmp_path):
        sibling = tmp_path / "innocent"
        sibling.mkdir()
        (sibling / "profile.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ProfileInvalid):
            store.get("../innocent")
        assert (sibling / "profile.json").exists()

    def test_tampered_audio_is_detected_on_load(self, profile_store):
        store, profile_id = profile_store
        path = Path(store.root) / profile_id / "reference.wav"
        payload = bytearray(path.read_bytes())
        payload[-100] ^= 0xFF
        path.write_bytes(bytes(payload))
        with pytest.raises(ProfileInvalid):
            store.load_audio(profile_id)

    def test_ids_are_prefixed_and_sortable(self, store):
        first = store.create("One", speech_like(rate=16000), consent())
        second = store.create("Two", speech_like(rate=16000), consent())
        assert first.id.startswith("voice_")
        assert first.id < second.id

    def test_retention_policy_expiry_blocks_use(self, store):
        profile = store.create(
            "Short lived", speech_like(rate=16000), consent(), retention=RetentionPolicy(days=1)
        )
        stored = store.get(profile.id)
        assert stored.consent_usable(now=time.time() + 3 * 86400)[0] is False

    def test_model_compatibility_keys_are_stored(self, profile_store):
        store, profile_id = profile_store
        assert store.model_compatible(profile_id, "pocket-class")
        assert not store.model_compatible(profile_id, "some-other-model")

    def test_voice_reference_carries_consent_for_the_pipeline(self, profile_store):
        store, profile_id = profile_store
        reference = store.voice_reference(profile_id)
        assert isinstance(reference, VoiceReference)
        assert reference.has_consent()
        assert reference.language == LanguageTag("bn")

    def test_metadata_is_readable_json_on_disk(self, profile_store):
        store, profile_id = profile_store
        payload = json.loads((Path(store.root) / profile_id / "profile.json").read_text("utf-8"))
        assert payload["id"] == profile_id
        assert payload["languages"] == ["bn"]
        assert payload["retention"]["basis"] == "explicit_consent"

    def test_list_reports_unreadable_directories_instead_of_hiding_them(self, store):
        profile = store.create("Aparna", speech_like(rate=16000), consent())
        broken = Path(store.root) / "voice_broken0000"
        broken.mkdir()
        (broken / "profile.json").write_text("{not json", encoding="utf-8")
        assert [stored.id for stored in store.list()] == [profile.id]
        assert store.problems and "voice_broken0000" in store.problems[0]


class TestConditioningRefusal:
    def test_conditioner_names_the_missing_approved_model(self):
        reference = VoiceReference(
            voice_id="v", audio=speech_like(), consent_token="c", owner="o", language=LanguageTag("bn")
        )
        with pytest.raises(CapabilityUnavailable) as refused:
            NoApprovedConditioner().condition(reference)
        assert "approved synthesis model" in str(refused.value)

    def test_unconsented_reference_is_refused_before_the_model_question(self):
        with pytest.raises(CapabilityUnavailable) as refused:
            NoApprovedConditioner().condition(VoiceReference(voice_id="v", audio=speech_like()))
        assert "consent" in str(refused.value)

    def test_capability_is_no_model_for_every_language(self):
        conditioner = NoApprovedConditioner()
        for code in ("bn", "hi", "en"):
            capability = conditioner.capability(LanguageTag(code))
            assert capability.task is Task.VOICE_CLONE
            assert capability.status is CapabilityStatus.NO_MODEL
        assert MISSING_APPROVED_MODEL in conditioner.capability(LanguageTag("bn")).limitations

    def test_verify_cannot_confirm_a_state_that_cannot_exist(self):
        with pytest.raises(CapabilityUnavailable):
            NoApprovedConditioner().verify(VoiceReference("v", consent_token="c", owner="o"))

    def test_cross_language_plan_is_a_plan_not_a_claim(self):
        plan = cross_language_evaluation_plan()
        assert "not measured" in plan["status"]
        assert plan["prerequisites"]
        assert {measure["name"] for measure in plan["measures"]} >= {
            "speaker_similarity_across_language",
            "listener_identification",
        }
        assert ("bn", "en") in plan["language_pairs"]
        assert plan["refusals"]
