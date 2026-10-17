"""End-to-end platform run: two hosts, two voices, one engine, offline.

This is the product claim made visible. ULTRON and FieldMind are represented here
as two `HostProfile` objects driving the *same* compiled engines through the same
`VoiceInterface`, each with its own answer function, its own output sink and its
own data root. Nothing in the synthesis path knows which host is talking.

A second voice is only offered after it has been measured. Slot selection is free
in this asset, so a name or a status for speaker 7 would otherwise be a guess:
the tool synthesises a sentence with that speaker, transcribes it back with the
approved ASR model, and records MEASURED only when the round-trip clears the bar -
otherwise the identity is written as MEASURED_FAILED and selection refuses it.

    python tools/platform_demo.py [--slot-b 7] [--play]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nativevoice.audio import resample, write_wav  # noqa: E402
from nativevoice.audio.io import CapturingOutput  # noqa: E402
from nativevoice.config import models_dir  # noqa: E402
from nativevoice.core.cancel import CancellationToken  # noqa: E402
from nativevoice.core.language import CapabilityStatus, LanguageTag, Task  # noqa: E402
from nativevoice.diagnostics.evidence_paths import write_json  # noqa: E402
from nativevoice.diagnostics.offline import OfflineGuard  # noqa: E402
from nativevoice.eval.transcription import score as score_pair  # noqa: E402
from nativevoice.integration import VoiceInterface  # noqa: E402
from nativevoice.models.registry import ModelRegistry  # noqa: E402
from nativevoice.session.state import SessionConfig  # noqa: E402
from nativevoice.stt.api import TranscriptionRequest  # noqa: E402
from nativevoice.tts.api import SynthesisRequest  # noqa: E402
from nativevoice.voices.identity import (  # noqa: E402
    Compatibility,
    InstallRecord,
    LanguageSupport,
    Measurements,
    VoiceIdentity,
    VoiceProvenance,
)
from nativevoice.voices.registry import PLATFORM_DEFAULT, HostProfile, VoiceRegistry  # noqa: E402

PASS_WER = 0.15
PROBE = "The voice platform serves several applications from one engine."
INPUT_SENTENCE = "Please tell me what the weather will do this afternoon."
#: The tracked evidence directory: a run with no ``--out`` writes its artefacts into the repository.
DEFAULT_OUT = ROOT / "artifacts" / "platform"


def engines(registry: ModelRegistry, threads: int):
    from nativevoice.stt.adapters.faster_whisper_engine import FasterWhisperEngine
    from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine

    manifests = {item.model_id: item for item in registry.all()}
    tts_id = next(key for key in manifests if "kokoro" in key)
    stt_id = next(key for key in manifests if "whisper-base" in key)
    synthesizer = SherpaKokoroEngine(manifest=manifests[tts_id], num_threads=threads)
    transcriber = FasterWhisperEngine(manifest=manifests[stt_id], cpu_threads=threads)
    return synthesizer, transcriber, manifests[tts_id], manifests[tts_id].asset_root or ""


def round_trip(synthesizer, transcriber, text: str, slot_identity: VoiceIdentity) -> tuple[str, float]:
    """Synthesise with `slot_identity`, then transcribe back. Returns (heard, WER)."""
    outcome = synthesizer.synthesize(
        SynthesisRequest(text=text, language=LanguageTag("en"), voice=slot_identity.as_reference())
    )
    feed = resample(outcome.audio, 16000) if outcome.audio.format.sample_rate != 16000 else outcome.audio
    heard = transcriber.transcribe(
        TranscriptionRequest(
            audio=feed, language=LanguageTag("en"), allow_auto_detection=False, max_duration_s=30.0
        )
    )
    return heard.text, float(score_pair(reference=text, hypothesis=heard.text, language=LanguageTag("en")).wer)


def make_identity(voice_id: str, display: str, slot: int, manifest, languages) -> VoiceIdentity:
    return VoiceIdentity(
        voice_id=voice_id,
        display_name=display,
        named=bool(display and display != voice_id),
        languages=tuple(languages),
        compatibility=Compatibility(
            engine_id=manifest.engine,
            model_id=manifest.model_id,
            speaker_slot=slot,
            required_quantization=manifest.quantization,
            sample_rate=manifest.sample_rate,
        ),
        provenance=VoiceProvenance(
            creator=manifest.provenance.upstream_project or "unknown",
            training_method="third-party pretrained checkpoint; speaker slot selected, not trained",
            licence_spdx=manifest.provenance.licence_spdx,
            permissions="local use approved; redistribution requires the GPL-3.0 espeak-ng pass",
            base_asset=manifest.provenance.source,
            redistributable=False,
            consent_basis="no personal data; slots are model embeddings, not donated voices",
        ),
        install=InstallRecord(
            state="installed",
            version="1",
            installed_on=time.strftime("%Y-%m-%dT%H:%M:%S"),
            source=str(manifest.asset_root or ""),
            digests={asset.relative_path: asset.sha256 for asset in manifest.assets if asset.sha256},
            removal="delete this identity JSON; the model asset is untouched",
        ),
    )


def measured_support(wer: float, heard: str, evidence: str) -> LanguageSupport:
    status = CapabilityStatus.MEASURED if wer <= PASS_WER else CapabilityStatus.MEASURED_FAILED
    return LanguageSupport(
        language="en",
        status=status,
        metric_name="roundtrip_wer",
        metric_value=wer,
        measured_on=time.strftime("%Y-%m-%d"),
        evidence=evidence,
        limitations=(
            "round-trip on machine-generated speech; measures the TTS+STT pair, not human speech",
            f"heard as: {heard[:80]}",
        ),
    )


class _Answer:
    """The entire host contract: one function from utterance to reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def respond(self, utterance, language=None):
        return self.reply


def write_run(report: dict, target: Path) -> Path:
    """Serialize the run record as committed evidence, host directories redacted.

    The report keeps its real paths in memory: the verdict above compares the two hosts'
    data roots for distinctness, and that comparison has to run on the paths the hosts
    actually resolved to, not on a placeholder that would make them look identical.
    """
    return write_json(target, report)


def voice_registry(voices_root: Path) -> VoiceRegistry:
    """Build the catalogue for this run, marking it as evidence when the root is tracked.

    The default output directory is inside the repository, so the identities a routine
    demonstration writes are published files and go out redacted. Resolved, because
    ``--out`` may name the tracked directory relatively; the registry still writes to the
    root it was given.
    """
    return VoiceRegistry(voices_root, committed=Path(voices_root).resolve().is_relative_to(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slot-b", type=int, default=7, help="speaker slot for the second host")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    voices_root = out / "voices"
    registry = ModelRegistry(models_dir())
    voices = voice_registry(voices_root)
    report: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "hosts": {}, "measurements": {}}

    guard = OfflineGuard()
    synthesizer = transcriber = None
    with guard:
        synthesizer, transcriber, tts_manifest, _ = engines(registry, args.threads)
        synthesizer.warm("en")
        transcriber.warm()

        # platform default: bind to slot 0, measured now so the status is earned here
        default_identity = make_identity(
            PLATFORM_DEFAULT, PLATFORM_DEFAULT, 0, tts_manifest, [LanguageSupport("en", CapabilityStatus.NO_MODEL)]
        )
        heard_probe, probe_wer = round_trip(synthesizer, transcriber, PROBE, default_identity)
        default_identity = make_identity(
            PLATFORM_DEFAULT,
            PLATFORM_DEFAULT,
            0,
            tts_manifest,
            [measured_support(probe_wer, heard_probe, str(out / "platform-run.json"))],
        )
        voices.save(default_identity)

        # second host's voice: same engine, different slot, status from its own measurement
        slot_b_id = f"voice.host.fieldmind.speaker{args.slot_b}"
        candidate = make_identity(slot_b_id, "", args.slot_b, tts_manifest, [LanguageSupport("en", CapabilityStatus.NO_MODEL)])
        heard_b, wer_b = round_trip(synthesizer, transcriber, PROBE, candidate)
        slot_b_identity = make_identity(
            slot_b_id, "", args.slot_b, tts_manifest, [measured_support(wer_b, heard_b, str(out / "platform-run.json"))]
        )
        # display_name stays empty because the asset gives no speaker names; naming it would be invention
        voices.save(slot_b_identity)
        report["measurements"] = {
            "slot_0": {"wer": round(probe_wer, 4), "heard": heard_probe, "status": default_identity.languages[0].status.value},
            f"slot_{args.slot_b}": {
                "wer": round(wer_b, 4),
                "heard": heard_b,
                "status": slot_b_identity.languages[0].status.value,
            },
            "pass_bar": PASS_WER,
        }

        class SttAdapter:
            """Adapts the engine to the `Transcriber` shape the session expects.

            `stt.orchestrator.TranscriptionService` normally sits here and does the
            format negotiation; this demo wires the engine directly, so it performs
            the one step the service would have handled - the recogniser takes
            16 kHz mono and refuses anything else.
            """

            def transcribe(self, request):
                audio = request.audio
                if audio.format.sample_rate != 16000 or audio.format.channels != 1:
                    from dataclasses import replace as _replace

                    audio = resample(audio.to_mono(), 16000)
                    request = _replace(request, audio=audio)
                return transcriber.transcribe(request)

            def stream(self, chunks, language=None, token=None):
                return transcriber.stream(chunks, language, token)

        answers = {
            "ultron": "Your summary is ready: three decisions and one follow-up for tomorrow.",
            "fieldmind": "Field report: soil moisture is adequate and the next irrigation window is at six.",
        }

        input_audio = synthesizer.synthesize(
            SynthesisRequest(text=INPUT_SENTENCE, language=LanguageTag("en"), voice=default_identity.as_reference())
        ).audio
        report["spoken_input"] = {"text": INPUT_SENTENCE, "seconds": round(input_audio.duration_s, 3)}

        for host_id, voice_id in (("ultron", PLATFORM_DEFAULT), ("fieldmind", slot_b_id)):
            sink = CapturingOutput()
            host = HostProfile(host_id=host_id, default_voice=voice_id, root=out / "hosts" / host_id)
            host.ensure()
            interface = VoiceInterface(
                transcriber=SttAdapter(),
                synthesizer=synthesizer,
                output=sink,
                voices=voices,
                host=host,
                config=SessionConfig(),
            )

            reply = interface.exchange(input_audio, _Answer(answers[host_id]))
            path = write_wav(str(out / f"{host_id}-reply.wav"), sink.audio) if sink.audio else ""
            report["hosts"][host_id] = {
                "default_voice": host.default_voice,
                "heard": reply.utterance,
                "answer": reply.reply,
                "spoken": reply.spoken,
                "silenced": reply.silenced,
                "language": reply.language,
                "frames_played": sink.frames_written,
                "audio_file": path,
                "notes": list(reply.notes),
                "voice_refusal": interface.voice_refusal,
                "data_root": str(host.root),
                "store_isolated": str(host.recordings),
            }
            interface.close()

    report["identities"] = {
        item.voice_id: {
            "slot": item.compatibility.speaker_slot,
            "named": item.named,
            "en": item.languages[0].status.value,
            "installed": item.install.state,
        }
        for item in voices.all()
    }
    report["network_attempts"] = [f"{attempt.host}:{attempt.port}" for attempt in guard.attempts]
    both_spoke = all(row["frames_played"] > 0 for row in report["hosts"].values())
    distinct = report["hosts"]["ultron"]["data_root"] != report["hosts"]["fieldmind"]["data_root"]
    report["verdict"] = "PASS" if both_spoke and distinct and not guard.attempts else "FAIL"
    target = out / "platform-run.json"
    write_run(report, target)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"artefact -> {target}")
    return 0 if report["verdict"] == "PASS" else 1


    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
