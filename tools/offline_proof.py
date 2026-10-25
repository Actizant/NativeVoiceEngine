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

What is proved is a capability, not a restriction: the speech path can complete with
no network access. The hosted API is separate planned work (`README.md:53`) and this
tool says nothing about it.

What this tool does is publish. The default destination is
`artifacts/audit/offline-proof.json`, committed evidence, so reaching it is a
deliberate act rather than a side effect: a completed run - PASS or FAIL - publishes,
while a run that never measured anything (NOT RUN, for want of an approved asset)
writes nothing at all, not even a directory, and exits 1. `--allow-incomplete`
publishes such a report only when `--out` names another destination; it can never
authorise a write to the default one. The payload goes through the shared redacting
serializer, so a home-rooted path is published as `[USER_HOME]`; like every writer on
that boundary it writes text, so a Windows working copy carries CRLF while the
repository stores the LF form.

    python tools/offline_proof.py                # publishes the evidence record
    python tools/offline_proof.py --out PATH     # publishes elsewhere instead
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
from nativevoice.diagnostics.evidence_paths import write_json  # noqa: E402
from nativevoice.diagnostics.offline import OfflineGuard, static_scan  # noqa: E402
from nativevoice.models.registry import ModelRegistry  # noqa: E402


def finding_label(finding) -> str:
    """Render one static-scan finding as the published record names it: `file:line:module`.

    `Finding` defines `file` and `line` (`src/nativevoice/diagnostics/offline.py`), and the
    label has to agree with that. Reading `path` and `lineno` instead raised the moment a
    finding actually existed, which is why the record's historically empty list concealed it.
    """
    return f"{finding.file}:{finding.line}:{finding.module}"


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
    result["static_findings"] = [finding_label(finding) for finding in findings]
    result["total_seconds"] = round(time.perf_counter() - started, 2)
    verdict = "PASS" if result["guard_clean"] and not result["static_findings"] and heard.text.strip() else "FAIL"
    result["verdict"] = verdict
    return result


#: The committed evidence record this tool publishes to when the operator names no
#: destination. Written in one place so the refusal and the publication agree on it.
DEFAULT_OUT = ROOT / "artifacts" / "audit" / "offline-proof.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", default="The voice engine is running locally on my machine.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", metavar="PATH",
                        help=f"publish to PATH instead of the default {DEFAULT_OUT.relative_to(ROOT).as_posix()}")
    parser.add_argument("--allow-incomplete", action="store_true", dest="allow_incomplete",
                        help="publish a NOT RUN report, and only where --out already names a destination")
    args = parser.parse_args(argv)

    # Whether a destination was *named* is what gates an incomplete report, not whether the
    # resolved path happens to look like somewhere safe: `--allow-incomplete` alone must
    # never be able to reach the committed record.
    out = Path(args.out) if args.out is not None else DEFAULT_OUT
    label = out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else str(out)
    report = run(args.text, args.language, args.threads)
    verdict = report.get("verdict")

    if verdict == "NOT RUN" and not (args.out is not None and args.allow_incomplete):
        print(json.dumps(report, indent=2, ensure_ascii=False))
        reason = report.get("reason", "no reason was given")
        print(f"artefact NOT written: verdict is NOT RUN ({reason}). {label} is unchanged: no directory "
              "was created, nothing was overwritten. To publish an incomplete report elsewhere, pass "
              "--out PATH --allow-incomplete")
        return 1

    write_json(out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"artefact -> {label}"
          + (" (incomplete report, never the default destination)" if verdict == "NOT RUN" else ""))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
