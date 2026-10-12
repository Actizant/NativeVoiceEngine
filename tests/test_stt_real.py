"""Tests for the real manifest-driven faster-whisper recognition engine.

Unit tiers verify gating, capability bookkeeping and refusal semantics without a
model; the real-inference tests are ``requires_approved_asset`` and skipped unless
``--run-approved-assets`` (or ``ULTRON_RUN_APPROVED_ASSET=1``) is set and the
approved faster-whisper asset is resident. A skipped test is never a pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fixtures import sine
from nativevoice.core.cancel import CancellationToken
from nativevoice.core.errors import (
    CancellationRequested,
    CapabilityUnavailable,
    ConfigurationError,
    ModelNotApproved,
    ModelNotInstalled,
)
from nativevoice.core.language import Capability, CapabilityStatus, LanguageTag, Task
from nativevoice.config.paths import models_dir
from nativevoice.models.api import ModelManifest, Provenance, ReviewStatus
from nativevoice.models.manifest_io import load_manifest
from nativevoice.stt.adapters.faster_whisper_engine import (
    ENGINE_ID,
    FasterWhisperEngine,
    resolve_asset_dir,
)
from nativevoice.stt.api import TranscriptionRequest

requires_assets = pytest.mark.skipif(
    not os.environ.get("ULTRON_RUN_APPROVED_ASSET"),
    reason=(
        "NOT RUN by default: real inference needs --run-approved-assets (or "
        "ULTRON_RUN_APPROVED_ASSET=1) and the approved faster-whisper asset in the HF cache"
    ),
)

_FULL = Provenance(source="hf-test", licence_spdx="MIT", redistributable=True)


def _stt_manifest(**overrides) -> ModelManifest:
    base = dict(
        model_id="faster-whisper-base",
        engine=ENGINE_ID,
        task=Task.STT,
        display_name="whisper test",
        languages=("en", "hi", "zh"),
        sample_rate=16000,
        provenance=_FULL,
        review=ReviewStatus.APPROVED,
        approved_by="tester",
    )
    base.update(overrides)
    return ModelManifest(**base)


# --- contract + bookkeeping (no model) ---------------------------------------

def test_engine_declares_contract():
    engine = FasterWhisperEngine(_stt_manifest(), asset_dir=Path("."))
    assert engine.engine_id == ENGINE_ID
    assert engine.model_id == "faster-whisper-base"
    assert (engine.format.sample_rate, engine.format.channels) == (16000, 1)
    # Whisper is buffered/VAD, not microphone-frame streaming: reported honestly.
    assert engine.supports_streaming is False


def test_capabilities_declared_from_manifest_languages():
    engine = FasterWhisperEngine(_stt_manifest(), asset_dir=Path("."))
    caps = {c.language: c for c in engine.capabilities()}
    assert set(caps) == {"en", "hi", "zh"}
    assert all(c.status is CapabilityStatus.DECLARED for c in caps.values())


def test_no_languages_recorded_means_no_claim():
    engine = FasterWhisperEngine(_stt_manifest(languages=()), asset_dir=Path("."))
    assert engine.capabilities() == ()


def test_record_measured_requires_metric():
    engine = FasterWhisperEngine(_stt_manifest(), asset_dir=Path("."))
    with pytest.raises(ConfigurationError):
        engine.record_measured(Capability(task=Task.STT, status=CapabilityStatus.DECLARED, language="en"))
    engine.record_measured(
        Capability(
            task=Task.STT,
            status=CapabilityStatus.MEASURED,
            language="en",
            engine=ENGINE_ID,
            model_id="faster-whisper-base",
            metric_name="roundtrip_wer",
            metric_value=0.05,
        )
    )
    caps = {c.language: c for c in engine.capabilities()}
    assert caps["en"].status is CapabilityStatus.MEASURED
    assert caps["hi"].status is CapabilityStatus.DECLARED


def test_resolve_asset_dir_matches_repo_by_model_id(tmp_path):
    repo = tmp_path / "models--Systran--faster-whisper-base"
    repo.mkdir()
    assert resolve_asset_dir(_stt_manifest(), cache_root=str(tmp_path)) == repo


def test_resolve_asset_dir_missing_raises_model_not_installed(tmp_path):
    with pytest.raises(ModelNotInstalled):
        resolve_asset_dir(_stt_manifest(model_id="ghost-model"), cache_root=str(tmp_path))


def test_warm_unapproved_raises_model_not_approved():
    engine = FasterWhisperEngine(
        _stt_manifest(review=ReviewStatus.UNREVIEWED, approved_by=""), asset_dir=Path(".")
    )
    with pytest.raises(ModelNotApproved):
        engine.warm()


def test_warm_missing_repo_raises_model_not_installed(tmp_path):
    engine = FasterWhisperEngine(
        _stt_manifest(model_id="absent-model"),
    )
    # Point the resolver at an empty cache root so no repository matches.
    engine._asset_dir = None
    import nativevoice.stt.adapters.faster_whisper_engine as fw

    original = fw.default_cache_root
    fw.default_cache_root = lambda: str(tmp_path)
    try:
        with pytest.raises(ModelNotInstalled):
            engine.warm()
    finally:
        fw.default_cache_root = original


def test_non_streaming_stream_refuses():
    engine = FasterWhisperEngine(_stt_manifest(), asset_dir=Path("."))
    with pytest.raises(CapabilityUnavailable):
        next(engine.stream(iter([sine(rate=16000, seconds=0.5)])))


def test_transcribe_observes_cancel_before_loading():
    engine = FasterWhisperEngine(_stt_manifest(), asset_dir=Path(".")
    )
    token = CancellationToken()
    token.cancel("stop")
    with pytest.raises(CancellationRequested):
        engine.transcribe(TranscriptionRequest(audio=sine(rate=16000, seconds=0.5), token=token))


# --- real inference (skipped unless opted in) --------------------------------

@pytest.fixture(scope="module")
def whisper_manifest():
    path = Path(models_dir()) / "manifests" / "faster-whisper-base.json"
    if not path.is_file():
        pytest.skip(f"faster-whisper manifest not present at {path}")
    return load_manifest(path)


@requires_assets
@pytest.mark.requires_approved_asset
@pytest.mark.slow
def test_real_transcribe_recognises_synthesised_english(whisper_manifest):
    """Round-trip: our TTS speaks, this engine listens - the intelligibility measurement."""
    from nativevoice.audio.resample import resample
    from nativevoice.eval.transcription import score
    from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine

    speech = "Please turn off the lights before you leave the room."
    tts_path = Path(models_dir()) / "manifests" / "kokoro-int8-multi-lang-v1_1.json"
    if not tts_path.is_file():
        pytest.skip("round-trip needs the approved Kokoro asset too")
    from nativevoice.tts.api import SynthesisRequest
    from nativevoice.tts.adapters.sherpa_kokoro import resolve_asset_dir as tts_dir

    tts = SherpaKokoroEngine(load_manifest(tts_path), tts_dir(load_manifest(tts_path)))
    outcome = tts.synthesize(SynthesisRequest(text=speech, language=LanguageTag("en")))
    tts.close()

    engine = FasterWhisperEngine(whisper_manifest, device="cpu", cpu_threads=2, vad_filter=True)
    try:
        result = engine.transcribe(
            TranscriptionRequest(audio=resample(outcome.audio, 16000), language=LanguageTag("en"), allow_auto_detection=False)
        )
        assert result.engine == ENGINE_ID
        assert result.text
        assert result.timings.get("segments", 0) >= 1
        wer = score(speech, result.text, "en").wer
        assert wer <= 0.15
    finally:
        engine.close()


@requires_assets
@pytest.mark.requires_approved_asset
@pytest.mark.slow
def test_real_forced_language_and_detection(whisper_manifest):
    from nativevoice.audio.resample import resample
    from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine, resolve_asset_dir
    from nativevoice.tts.api import SynthesisRequest

    tts_manifest = load_manifest(Path(models_dir()) / "manifests" / "kokoro-int8-multi-lang-v1_1.json")
    tts = SherpaKokoroEngine(tts_manifest, resolve_asset_dir(tts_manifest))
    outcome = tts.synthesize(SynthesisRequest(text="今天天气很好。", language=LanguageTag("zh")))
    tts.close()

    engine = FasterWhisperEngine(whisper_manifest, device="cpu", cpu_threads=2, vad_filter=True)
    try:
        audio16 = resample(outcome.audio, 16000)
        detected = engine.transcribe(TranscriptionRequest(audio=audio16, allow_auto_detection=True))
        forced = engine.transcribe(
            TranscriptionRequest(audio=audio16, language=LanguageTag("zh"), allow_auto_detection=False)
        )
        assert detected.language is not None
        assert forced.language is None or forced.language.code == "zh"
        assert forced.text
    finally:
        engine.close()
