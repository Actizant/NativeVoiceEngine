"""Manifests carry licence and consent; synthetic fixtures must never be usable."""

from __future__ import annotations

import json

import pytest

from nativevoice.audio.wav import write_wav
from nativevoice.core.errors import CapabilityUnavailable, ConfigurationError
from nativevoice.eval.samples import (
    EvalSample,
    SampleManifest,
    builtin_fixtures,
    synthetic_guard,
)
from fixtures import speech_like, sine


def _write_corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    entries = []
    for index, (language, buffer, transcript) in enumerate(
        [
            ("en", sine(seconds=0.2), "the quick brown fox"),
            ("hi", speech_like(), "मैं हिंदी में बोलता हूँ"),
            ("bn", speech_like(), "আমি বাংলায় কথা বলি"),
        ]
    ):
        path = root / f"{language}-{index}.wav"
        write_wav(str(path), buffer)
        entries.append(
            {
                "sample_id": path.name,
                "path": str(path),
                "language": language,
                "transcript": transcript,
                "licence": "CC-BY-4.0",
                "consent_basis": "speaker signed recording consent",
            }
        )
    manifest_path = tmp_path / "corpus.json"
    manifest_path.write_text(json.dumps({"name": "mini", "source": "test", "samples": entries}), encoding="utf-8")
    return str(manifest_path)


def test_usable_is_derived_from_licence_and_consent():
    assert EvalSample("a.wav", "en", "hello", licence="MIT", consent_basis="signed").usable is True
    assert EvalSample("a.wav", "en", "hello", licence="MIT").usable is False
    assert EvalSample("a.wav", "en", "hello", consent_basis="signed").usable is False
    assert EvalSample("a.wav", "en", "hello").unusable_reasons == ("licence missing", "consent basis missing")


def test_synthetic_beats_a_perfect_licence():
    """A tone with paperwork is still a tone: it cannot be a quality corpus."""
    sample = EvalSample("synthetic", "en", "", licence="internal", consent_basis="n/a", synthetic=True)
    assert sample.usable is False
    assert "synthetic fixtures cannot be scored for quality" in sample.unusable_reasons


def test_manifest_round_trip_and_audio_resolution(tmp_path):
    manifest = SampleManifest.load(_write_corpus(tmp_path))
    assert manifest.name == "mini"
    assert manifest.languages() == ("en", "hi", "bn")
    assert len(manifest.quality_usable()) == 3
    assert manifest.blocked() == ()
    buffer = manifest.audio(manifest.by_language("en")[0])
    assert buffer.frame_count > 0 and buffer.format.sample_rate == 16000
    manifest.require_quality_corpus()


def test_gaps_report_missing_paperwork(tmp_path):
    payload = {
        "name": "patchy",
        "samples": [
            {"path": "a.wav", "language": "en", "transcript": "one", "licence": "CC0", "consent_basis": "public"},
            {"path": "b.wav", "language": "en", "transcript": "two", "licence": ""},
            {"path": "c.wav", "language": "bn", "transcript": "তিন", "licence": "CC0", "consent_basis": "public", "synthetic": True},
        ],
    }
    path = tmp_path / "patchy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    gaps = SampleManifest.load(str(path)).gaps()
    assert gaps["en"] == {"samples": 2, "usable": 1, "synthetic": 0, "licence_missing": 1, "consent_missing": 1}
    assert gaps["bn"]["usable"] == 0 and gaps["bn"]["synthetic"] == 1


def test_unknown_manifest_key_is_rejected_not_ignored(tmp_path):
    """A typo in `licence`/`synthetic` must fail loudly rather than change a verdict."""
    path = tmp_path / "typo.json"
    path.write_text(
        json.dumps({"samples": [{"path": "a.wav", "language": "en", "transcript": "x", "licnese": "CC0"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        SampleManifest.load(str(path))


def test_missing_manifest_file_is_a_specific_error(tmp_path):
    with pytest.raises(ConfigurationError):
        SampleManifest.load(str(tmp_path / "absent.json"))


def test_builtin_fixtures_are_all_synthetic_and_refused():
    manifest = builtin_fixtures()
    assert manifest.samples
    assert all(sample.synthetic for sample in manifest.samples)
    assert manifest.quality_usable() == ()
    with pytest.raises(CapabilityUnavailable) as excinfo:
        manifest.require_quality_corpus()
    assert "no usable evaluation samples" in str(excinfo.value)


def test_builtin_fixtures_produce_real_buffers(tmp_path):
    """Plumbing tests need actual samples; these are deterministic across runs."""
    manifest = builtin_fixtures()
    first = manifest.audio(manifest.samples[0])
    again = SampleManifest(name=manifest.name, samples=manifest.samples).audio(manifest.samples[0])
    assert first.samples == again.samples and first.frame_count > 0
    envelope = next(s for s in manifest.samples if "envelope" in s.sample_id)
    assert manifest.audio(envelope).frame_count > 0


def test_synthetic_guard_names_the_offenders():
    samples = [EvalSample("s", "en", "", synthetic=True, sample_id="tone-en-0")]
    with pytest.raises(CapabilityUnavailable) as excinfo:
        synthetic_guard(samples, activity="unit test")
    assert "tone-en-0" in str(excinfo.value)
    assert synthetic_guard([EvalSample("r", "en", "hi", licence="CC0", consent_basis="ok")], activity="x") is None


def test_sample_without_language_is_rejected():
    with pytest.raises(ConfigurationError):
        EvalSample(path="a.wav", language="", transcript="x")
