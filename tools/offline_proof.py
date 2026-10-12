"""Offline compliance proved against the real engines, not against the test suite.

The specification demands that offline behaviour be verified with networking
deliberately disabled. A unit test with no model loaded proves nothing about the
model's own loading path, so this runs the actual engines - synthesis, then
recognition of what was just synthesised - while a guard makes every
non-loopback connection impossible and records any attempt.

Three things have to hold for this to say PASS: the work completes, the guard
records zero attempts, and the static scan finds no network import on the speech
path. Two independent proofs, because a runtime guard only catches code that
actually ran, and a static scan only catches code that was written.

    python tools/offline_proof.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nativevoice.audio import write_wav  # noqa: E402
from nativevoice.config import models_dir  # noqa: E402
from nativevoice.core.language import Task  # noqa: E402
from nativevoice.diagnostics.offline import OfflineGuard, static_scan  # noqa: E402
from nativevoice.models.registry import ModelRegistry  # noqa: E402


def run(voice_text: str, language: str, threads: int) -> dict:
    from nativevoice.stt.api import TranscriptionRequest
    from nativevoice.tts.api import SynthesisRequest
    from nativevoice.core.language import LanguageTag

    registry = ModelRegistry(models_dir())
    manifests = {manifest.model_id: manifest for manifest in registry.all()}
    tts_id = next((key for key in manifests if "kokoro" in key), None)
    stt_id = next((key for key in manifests if "whisper" in key), None)
    if not tts_id or not stt_id:
        return {"verdict": "NOT RUN", "reason": f"missing manifest: tts={tts_id} stt={stt_id}"}

    from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine
    from nativevoice.stt.adapters.faster_whisper_engine import FasterWhisperEngine

    guard = OfflineGuard()
    result: dict = {"tts_asset": tts_id, "stt_asset": stt_id, "language": language}
    started = time.perf_counter()
    with guard:
        synthesizer = SherpaKokoroEngine(manifest=manifests[tts_id], num_threads=threads)
        load_started = time.perf_counter()
        synthesizer.warm()
        result["tts_load_seconds"] = round(time.perf_counter() - load_started, 3)

        synth_started = time.perf_counter()
        outcome = synthesizer.synthesize(
            SynthesisRequest(text=voice_text, language=LanguageTag(language)),
        )
        result["tts_wall_seconds"] = round(time.perf_counter() - synth_started, 3)
        result["tts_audio_seconds"] = round(outcome.audio.duration_s, 3)
        result["tts_real_time_factor"] = round(result["tts_wall_seconds"] / max(0.001, result["tts_audio_seconds"]), 4)

        transcriber = FasterWhisperEngine(manifest=manifests[stt_id], cpu_threads=threads)
        load_started = time.perf_counter()
        transcriber.warm()
        result["stt_load_seconds"] = round(time.perf_counter() - load_started, 3)

        # The adapter takes 16 kHz mono and refuses anything else; the service layer
        # resamples, so this tool does the same step explicitly.
        from nativevoice.audio import resample
        from nativevoice.core.audio import AudioFormat

        for_feed = resample(outcome.audio, 16000) if outcome.audio.format.sample_rate != 16000 else outcome.audio
        result["stt_input_format"] = str(for_feed.format)
        hearing_started = time.perf_counter()
        heard = transcriber.transcribe(
            TranscriptionRequest(
                audio=for_feed,
                language=None,
                allow_auto_detection=True,
                max_duration_s=max(5.0, outcome.audio.duration_s + 5),
            )
        )
        result["stt_wall_seconds"] = round(time.perf_counter() - hearing_started, 3)
        result["heard"] = heard.text
        result["heard_language"] = heard.language.code if heard.language else ""
        synthesizer.close()
        transcriber.close()

    result["attempts"] = [f"{attempt.host}:{attempt.port}" for attempt in guard.attempts]
    result["guard_clean"] = not guard.attempts
    findings = static_scan(str(ROOT / "src" / "nativevoice"))
    result["static_findings"] = [f"{finding.path}:{finding.lineno}:{finding.module}" for finding in findings]
    result["total_seconds"] = round(time.perf_counter() - started, 2)
    verdict = "PASS" if result["guard_clean"] and not result["static_findings"] and heard.text.strip() else "FAIL"
    result["verdict"] = verdict
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", default="The voice engine is running locally on my machine.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "audit" / "offline-proof.json"))
    args = parser.parse_args(argv)

    report = run(args.text, args.language, args.threads)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"artefact -> {args.out}")
    return 0 if report.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
