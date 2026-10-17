"""Tests for the real sherpa-onnx Kokoro synthesis engine.

Two tiers, deliberately: unit tests that verify gating, prosody mapping, refusal
semantics and capability bookkeeping WITHOUT loading a model, and real-inference
tests marked ``requires_approved_asset`` that are skipped unless the operator opts
in with ``--run-approved-assets`` (or ``ULTRON_RUN_APPROVED_ASSET=1``) and the
approved Kokoro asset is installed. A skipped real test never counts as passed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from fixtures import sine
from nativevoice.core.errors import (
    CancellationRequested,
    CapabilityUnavailable,
    ConfigurationError,
    IntegrityError,
    LanguageUnsupported,
    ModelNotApproved,
    ModelNotInstalled,
)
from nativevoice.core.language import Capability, CapabilityStatus, LanguageTag, Task
from nativevoice.config.paths import models_dir
from nativevoice.models.api import Asset, AssetKind, AssetScope, ModelManifest, Provenance, ReviewStatus
from nativevoice.models.approval import ApprovalGate
from nativevoice.models.manifest_io import load_manifest
from nativevoice.tts.api import Prosody, SynthesisRequest, VoiceReference
from nativevoice.tts.adapters.sherpa_kokoro import (
    ENGINE_ID,
    SherpaKokoroEngine,
    _clamp_speed,
    _volume_gain,
    resolve_asset_dir,
)

requires_assets = pytest.mark.skipif(
    not os.environ.get("ULTRON_RUN_APPROVED_ASSET"),
    reason=(
        "NOT RUN by default: real inference needs --run-approved-assets (or "
        "ULTRON_RUN_APPROVED_ASSET=1) and the approved Kokoro asset under "
        "models/kokoro-int8-multi-lang-v1_1"
    ),
)

_FULL = Provenance(
    source="local-test",
    licence_spdx="Apache-2.0",
    redistributable=True,
)


def _tts_manifest(**overrides) -> ModelManifest:
    base = dict(
        model_id="kokoro-test",
        engine=ENGINE_ID,
        task=Task.TTS,
        display_name="test",
        languages=("en", "zh"),
        sample_rate=24000,
        provenance=_FULL,
        review=ReviewStatus.APPROVED,
        approved_by="tester",
    )
    base.update(overrides)
    return ModelManifest(**base)


BLOB = b"synthetic-onnx-bytes-not-a-real-model"


def _tts_manifest_with_asset(directory: Path, **overrides) -> ModelManifest:
    """A manifest pinned to the synthetic bytes this test writes into ``directory``.

    It records `asset_root` the way a provisioned manifest does, so the bytes the
    test installs are the bytes integrity looks for: a refusal below can then only
    be about scope, never about an accidentally missing or mismatched asset.
    """
    base = dict(
        asset_root=str(directory),
        assets=(
            Asset(
                kind=AssetKind.MODEL,
                relative_path="model.int8.onnx",
                sha256=hashlib.sha256(BLOB).hexdigest(),
                size_bytes=len(BLOB),
            ),
        ),
    )
    base.update(overrides)
    return _tts_manifest(**base)


def _install_asset(manifest: ModelManifest, directory: Path) -> Path:
    """Put digest-correct bytes where the manifest says they belong.

    Integrity then cannot be the accidental reason a load was refused: the only
    variable left is which path the manifest is scoped to.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for asset in manifest.assets:
        path = directory / asset.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(BLOB)
    return directory


# --- construction and pure mapping (no model, no IO) -------------------------

def test_engine_declares_contract():
    engine = SherpaKokoroEngine(_tts_manifest(), asset_dir=Path("."))
    assert engine.engine_id == ENGINE_ID
    assert engine.model_id == "kokoro-test"
    assert (engine.output_format.sample_rate, engine.output_format.channels) == (24000, 1)
    # chunk-level streaming is claimed; frame-level is not (see docstring).
    assert engine.supports_streaming is True
    assert "rate" in engine.controllable


def test_resolve_asset_dir_uses_model_id_under_root(tmp_path):
    manifest = _tts_manifest(model_id="asset-x")
    assert resolve_asset_dir(manifest, root=tmp_path) == tmp_path / "asset-x"


def test_clamp_speed_maps_rate_and_clamps():
    assert _clamp_speed(1.0) == 1.0
    assert _clamp_speed(2.0) == 2.0
    assert _clamp_speed(9.0) == 5.0  # upper clamp
    assert _clamp_speed(0.01) == 0.1  # lower clamp


def test_volume_gain_mapping():
    assert _volume_gain(0.0) == 1.0
    assert _volume_gain(6.0) > 1.0
    assert _volume_gain(6.0) <= 1.5  # bounded so it cannot hard-clip
    assert _volume_gain(-6.0) < 1.0


def test_lang_for_accepts_renderable_and_refuses_others():
    engine = SherpaKokoroEngine(_tts_manifest(), asset_dir=Path("."))
    assert engine._lang_for(LanguageTag("en")) == "en"
    assert engine._lang_for(LanguageTag("zh")) == "zh"
    with pytest.raises(LanguageUnsupported):
        engine._lang_for(LanguageTag("hi"))


def test_capabilities_declared_until_measured():
    engine = SherpaKokoroEngine(_tts_manifest(), asset_dir=Path("."))
    caps = {c.language: c for c in engine.capabilities()}
    assert set(caps) == {"en", "zh"}
    assert all(c.status is CapabilityStatus.DECLARED for c in caps.values())


def test_record_measured_flows_into_capabilities():
    engine = SherpaKokoroEngine(_tts_manifest(), asset_dir=Path("."))
    measured = Capability(
        task=Task.TTS,
        status=CapabilityStatus.MEASURED,
        language="en",
        engine=ENGINE_ID,
        model_id="kokoro-test",
        metric_name="roundtrip_wer",
        metric_value=0.02,
        evidence="artifacts/evaluation/roundtrip.json",
    )
    engine.record_measured(measured)
    caps = {c.language: c for c in engine.capabilities()}
    assert caps["en"].status is CapabilityStatus.MEASURED
    assert caps["zh"].status is CapabilityStatus.DECLARED


def test_record_measured_rejects_non_measured_or_unmetric():
    engine = SherpaKokoroEngine(_tts_manifest(), asset_dir=Path("."))
    with pytest.raises(ConfigurationError):
        engine.record_measured(
            Capability(task=Task.TTS, status=CapabilityStatus.DECLARED, language="en")
        )
    with pytest.raises(ConfigurationError):
        engine.record_measured(
            Capability(task=Task.TTS, status=CapabilityStatus.MEASURED, language="en")
        )


def test_voices_empty_before_warm():
    engine = SherpaKokoroEngine(_tts_manifest(), asset_dir=Path("."))
    assert engine.voices() == ()


def test_condition_refuses_voice_clone():
    engine = SherpaKokoroEngine(_tts_manifest(), asset_dir=Path("."))
    reference = VoiceReference(voice_id="someone", audio=sine(rate=24000, seconds=1.0))
    with pytest.raises(CapabilityUnavailable) as exc:
        engine.condition(reference)
    assert "clone" in str(exc.value).lower()


# --- gate error propagation (no model needed) --------------------------------

def test_unapproved_manifest_raises_model_not_approved():
    engine = SherpaKokoroEngine(
        _tts_manifest(review=ReviewStatus.UNREVIEWED, approved_by=""), asset_dir=Path(".")
    )
    with pytest.raises(ModelNotApproved):
        engine.warm("en")


def test_a_measurement_scoped_manifest_never_reaches_the_speaker_weights(tmp_path):
    """The synthesis path asks the scope question first, override included.

    The same two axes that keep a research model off the recognition path apply here:
    an approval record - or a human override of it - says nothing about which path an
    asset may serve.
    """
    directory = tmp_path / "kokoro"
    manifest = _tts_manifest_with_asset(directory, use_scope=AssetScope.MEASUREMENT)
    engine = SherpaKokoroEngine(manifest, asset_dir=_install_asset(manifest, directory))

    gate = ApprovalGate()
    gate.override(manifest, model_id=manifest.model_id, reason="local measurement only")
    gate.check(manifest)  # the approval axis is wide open here
    with pytest.raises(ModelNotApproved) as exc:
        engine.warm("en")
    assert any("use scope" in reason for reason in exc.value.details["reasons"])
    assert engine._tts is None


def test_an_approved_product_manifest_still_passes_both_guards(tmp_path):
    """The control the refusal above depends on: nothing about a product asset changed.

    Same synthetic bytes, same digest, only `use_scope` different - so the gate opens
    and the run fails downstream for the reason it always did, which is the missing
    front-end files. sherpa-onnx is never imported and no weight is read either way.
    """
    directory = tmp_path / "kokoro"
    manifest = _tts_manifest_with_asset(directory)
    engine = SherpaKokoroEngine(manifest, asset_dir=_install_asset(manifest, directory))
    engine._require_approved()  # approved and product-scoped: open exactly as before
    with pytest.raises(ModelNotInstalled):
        engine.warm("en")  # integrity passes; the front-end files this engine needs do not exist
    assert engine._tts is None


def test_missing_asset_dir_raises_model_not_installed(tmp_path):
    engine = SherpaKokoroEngine(_tts_manifest(), asset_dir=tmp_path / "absent")
    with pytest.raises(ModelNotInstalled):
        engine.warm("en")


def test_integrity_mismatch_raises_integrity_error(tmp_path):
    (tmp_path / "model.int8.onnx").write_bytes(b"not the approved bytes")
    manifest = _tts_manifest(
        assets=(Asset(kind="model", relative_path="model.int8.onnx", sha256="00" * 32),),
        asset_root=str(tmp_path),
    )
    engine = SherpaKokoroEngine(manifest, asset_dir=tmp_path)
    with pytest.raises(IntegrityError):
        engine.warm("en")


def test_provenance_incomplete_blocks():
    engine = SherpaKokoroEngine(
        _tts_manifest(provenance=Provenance(source="x"), approved_by="tester"), asset_dir=Path(".")
    )
    with pytest.raises(ModelNotApproved):
        engine.warm("en")


def test_synthesize_empty_text_raises():
    engine = SherpaKokoroEngine(_tts_manifest(), asset_dir=Path("."))
    with pytest.raises(ConfigurationError):
        engine.synthesize(SynthesisRequest(text="   ", language=LanguageTag("en")))


def test_stream_observations_cancel_before_model():
    engine = SherpaKokoroEngine(_tts_manifest(), asset_dir=Path("."))
    from nativevoice.core.cancel import CancellationToken

    token = CancellationToken()
    token.cancel("user interrupted")
    stream = engine.stream(SynthesisRequest(text="hello world", language=LanguageTag("en"), token=token))
    with pytest.raises(CancellationRequested):
        next(stream)


# --- real inference (skipped unless opted in) --------------------------------

@pytest.fixture(scope="module")
def kokoro_manifest():
    path = Path(models_dir()) / "manifests" / "kokoro-int8-multi-lang-v1_1.json"
    if not path.is_file():
        pytest.skip(f"approved Kokoro manifest not present at {path}")
    return load_manifest(path)


@requires_assets
@pytest.mark.requires_approved_asset
@pytest.mark.slow
def test_real_english_synthesis_produces_audio(kokoro_manifest):
    engine = SherpaKokoroEngine(kokoro_manifest)
    engine.warm("en")
    try:
        assert int(engine._tts.num_speakers) == 103
        voices = engine.voices()
        assert len(voices) == 103
        assert voices[0].voice_id == "kokoro-0"
        outcome = engine.synthesize(
            SynthesisRequest(text="Please turn off the lights before you leave.", language=LanguageTag("en"))
        )
        assert outcome.audio.format.sample_rate == 24000
        assert outcome.audio.frame_count > 0
        assert not outcome.audio.is_silent()
    finally:
        engine.close()


@requires_assets
@pytest.mark.requires_approved_asset
@pytest.mark.slow
def test_real_english_streaming_emits_chunk_events_then_eos(kokoro_manifest):
    from nativevoice.core.events import SynthesisKind

    engine = SherpaKokoroEngine(kokoro_manifest)
    try:
        events = list(
            engine.stream(
                SynthesisRequest(
                    text="The weather is pleasant today. Let us go for a walk in the park.",
                    language=LanguageTag("en"),
                )
            )
        )
        kinds = [e.kind for e in events]
        assert kinds[0] is SynthesisKind.FIRST_AUDIO
        assert kinds[-1] is SynthesisKind.EOS
        assert any(e.audio is not None and e.audio.frame_count for e in events)
    finally:
        engine.close()


@requires_assets
@pytest.mark.requires_approved_asset
@pytest.mark.slow
def test_real_chinese_synthesis_produces_audio(kokoro_manifest):
    engine = SherpaKokoroEngine(kokoro_manifest)
    try:
        outcome = engine.synthesize(SynthesisRequest(text="今天天气很好。", language=LanguageTag("zh")))
        assert outcome.audio.frame_count > 0
    finally:
        engine.close()


@requires_assets
@pytest.mark.requires_approved_asset
def test_real_refuses_hindi_without_producing_audio(kokoro_manifest):
    engine = SherpaKokoroEngine(kokoro_manifest)
    try:
        with pytest.raises(LanguageUnsupported):
            engine.synthesize(SynthesisRequest(text="नमस्ते", language=LanguageTag("hi")))
    finally:
        engine.close()


def test_default_chunk_budget_is_the_measured_one():
    """24 tokens was chosen from a measurement, so it must not drift back silently."""
    from pathlib import Path as _Path

    from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine
    from nativevoice.tts.chunking import TextChunker

    engine = SherpaKokoroEngine(_tts_manifest(), asset_dir=_Path("."))
    assert engine.chunker.max_tokens == 24

    slower = SherpaKokoroEngine(
        _tts_manifest(), asset_dir=_Path("."), chunker=TextChunker(max_tokens=48)
    )
    assert slower.chunker.max_tokens == 48, "a host that prefers continuity may still ask for it"
