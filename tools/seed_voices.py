"""Create the platform's first voice identity package from measured evidence.

Every field in the resulting identity is copied from something that already
exists: the engine and model from the provisioned manifest, the digests from the
verified assets, the per-language status and numbers from the round-trip
evaluation artefact. Nothing is asserted about a voice that has not been
observed - which is why the default identity is *unnamed*: Kokoro's `voices.bin`
is an anonymous array of speaker embeddings, so a display name would be a
fabrication, and an invented name is exactly the kind of metadata a later reader
trusts because it looks like provenance.

    python tools/seed_voices.py [--slot 0] [--out <voices root>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nativevoice.config import models_dir  # noqa: E402
from nativevoice.core.language import CapabilityStatus, Task  # noqa: E402
from nativevoice.models.registry import ModelRegistry  # noqa: E402
from nativevoice.voices.identity import (  # noqa: E402
    Compatibility,
    InstallRecord,
    LanguageSupport,
    Measurements,
    VoiceIdentity,
    VoiceProvenance,
)
from nativevoice.voices.registry import PLATFORM_DEFAULT, VoiceRegistry  # noqa: E402

DEFAULT_ARTIFACT = ROOT / "artifacts" / "evaluation"


def measured_from_artifact(path: Path) -> dict:
    """Per-language results, or an empty dict when no evaluation has run."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["language"]: row for row in payload.get("results", [])}


def voice_registry(out: str) -> VoiceRegistry:
    """Where the package is written, and whether that place is published evidence.

    With no ``--out`` the identity goes to the data root, which is runtime state the product
    reads back: its asset root must name the directory that exists. A root inside the
    repository is a tracked file, and an identity records that root and its evidence file by
    absolute path, so that copy is redacted as it is serialized. Resolved, because a
    relative ``--out`` can still point inside the repository.
    """
    if not out:
        return VoiceRegistry()
    root = Path(out)
    return VoiceRegistry(root, committed=root.resolve().is_relative_to(ROOT))


def build(manifest, artifact: dict, slot: int, host_summary: str, evidence: str = "") -> VoiceIdentity:
    languages = []
    for code, row in sorted(artifact.items()):
        verdict = row.get("verdict")
        name = row.get("primary_metric") or ("cer" if row.get("forced_wer") == 1.0 and row.get("forced_cer") is not None else "wer")
        metric_name = f"roundtrip_{name}"
        metric_value = row.get("primary_value")
        if metric_value is None:
            metric_value = row.get("forced_cer") if name == "cer" else row.get("forced_wer")
        if verdict == "PASS":
            status = CapabilityStatus.MEASURED
            limitations = ("measured on machine-generated speech via round-trip recognition",)
        elif verdict == "FAIL":
            status = CapabilityStatus.MEASURED_FAILED
            limitations = tuple(row.get("limitations") or ()) or ("measured; did not clear the bar",)
        elif verdict == "UNSUPPORTED":
            status = CapabilityStatus.UNSUPPORTED
            metric_name, metric_value = "", None
            limitations = (row.get("tts_error") or "no front-end for this language in this asset",)
        else:
            status = CapabilityStatus.NO_MODEL
            metric_name, metric_value = "", None
            limitations = (f"not evaluated ({verdict})",)
        languages.append(
            LanguageSupport(
                language=code,
                status=status,
                metric_name=metric_name,
                metric_value=metric_value,
                measured_on=time.strftime("%Y-%m-%d"),
                evidence=evidence,
                limitations=limitations,
            )
        )
    if not languages:
        languages = [
            LanguageSupport(
                language=code,
                status=CapabilityStatus.NO_MODEL,
                limitations=("no evaluation has run for this language",),
            )
            for code in manifest.languages
        ]

    readings = next(iter(artifact.values()), {})
    return VoiceIdentity(
        voice_id=PLATFORM_DEFAULT,
        display_name=PLATFORM_DEFAULT,
        named=False,
        languages=tuple(languages),
        compatibility=Compatibility(
            engine_id=manifest.engine,
            model_id=manifest.model_id,
            speaker_slot=slot,
            min_platform_version="0.1.0",
            required_quantization=manifest.quantization,
            sample_rate=manifest.sample_rate,
        ),
        provenance=VoiceProvenance(
            creator=manifest.provenance.upstream_project or "unknown",
            training_method="third-party pretrained checkpoint; no voice model trained by this project exists yet",
            licence_spdx=manifest.provenance.licence_spdx,
            permissions=(
                "local use approved; redistribution requires the espeak-ng GPL-3.0 pass and per-voice "
                "consent review"
            ),
            base_asset=manifest.provenance.source,
            redistributable=False,
            consent_basis="no personal data in the checkpoint; speaker slots are not donated voices",
            notes=manifest.provenance.licence_notice,
        ),
        measurements=Measurements(
            host=host_summary,
            measured_on=time.strftime("%Y-%m-%d"),
            real_time_factor=readings.get("mean_tts_rtf"),
            first_audio_ms=readings.get("mean_tts_first_audio_ms"),
            peak_rss_mib=readings.get("peak_rss_mib"),
            quality={
                code: {row.get("primary_metric") or "wer": row.get("primary_value")} for code, row in artifact.items()
            },
            source=str(DEFAULT_ARTIFACT),
        ),
        install=InstallRecord(
            state="installed",
            version="1",
            installed_on=time.strftime("%Y-%m-%dT%H:%M:%S"),
            source=manifest.asset_root or str(Path(models_dir()) / manifest.model_id),
            digests={asset.relative_path: asset.sha256 for asset in manifest.assets if asset.sha256},
            removal="delete the identity JSON from the voices root; the model asset is untouched",
        ),
        tags=("platform", "default", "unnamed"),
        notes=(
            "Speaker slot selected by index because the asset provides no voice names. "
            "Hosts that want a distinct identity install their own package rather than editing this one."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slot", type=int, default=0, help="speaker index inside the asset's voice array")
    parser.add_argument("--model", default="kokoro", help="substring of the TTS model id to bind to")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--out", default="")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    registry = ModelRegistry(models_dir())
    manifest = next((item for item in registry.all() if args.model in item.model_id and item.task is Task.TTS), None)
    if manifest is None:
        print(f"no provisioned TTS asset matching {args.model!r}; run tools/provision.py first")
        return 1
    status = registry.status(manifest.model_id)
    if not status.usable:
        print(f"refusing to bind an identity to an unverifiable asset: {status.explain()}")
        return 1

    artifact_path = Path(args.artifact)
    evidence_file = artifact_path if artifact_path.is_file() else artifact_path / f"roundtrip-{time.strftime('%Y-%m-%d')}.json"
    measured = measured_from_artifact(evidence_file)
    if not measured:
        print(f"no evaluation evidence at {evidence_file}; the identity will record NO_MODEL for every language")

    identity = build(
        manifest, measured, args.slot, host_summary=f"models_root={registry.root}", evidence=str(evidence_file)
    )
    if args.print_only:
        print(json.dumps(identity.as_dict(), indent=2, ensure_ascii=False))
        return 0
    target = voice_registry(args.out)
    path = target.save(identity)
    print(f"identity {identity.voice_id} -> {path}")
    for item in identity.languages:
        value = f" {item.metric_name}={item.metric_value}" if item.metric_name else ""
        print(f"  {item.language}: {item.status.value}{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
