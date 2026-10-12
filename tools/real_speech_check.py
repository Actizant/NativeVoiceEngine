"""Step 2 of the real-speech protocol: read-aloud English, nothing persisted.

Run it and speak the three sentences when the countdown ends, at roughly arm's length and a natural pace:

    python tools/real_speech_check.py

What is guaranteed by construction rather than by intention:

* **local only** - recognition runs on the approved local CT2 model inside an `OfflineGuard`, and any
  connection attempt fails the run;
* **retention `none`** - audio lives in memory, is scored, and is never written; the data directory is
  listed before and after to prove it rather than to assert it;
* **predeclared criteria** - the pass bars are the ones written in `docs/real-speech-protocol.md` before
  any audio existed, and anything that cannot be evaluated from a manual read-aloud run is reported
  `NOT RUN` with the reason, never silently dropped or quietly passed.

Results here are human-speech measurements and are kept separate from the synthetic round-trip numbers:
read-aloud speech is cleaner than spontaneous speech, so these WER figures are optimistic for real
conversation and are labelled read-aloud wherever they appear.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nativevoice.audio import measure, rms  # noqa: E402
from nativevoice.audio.buffer import RingBuffer  # noqa: E402
from nativevoice.audio.io import SounddeviceInput  # noqa: E402
from nativevoice.audio.preprocess import Preprocessor  # noqa: E402
from nativevoice.audio.vad import (  # noqa: E402
    EnergySpeechDetector,
    SegmenterConfig,
    UtteranceSegmenter,
)
from nativevoice.config import data_root, models_dir  # noqa: E402
from nativevoice.core.audio import AudioFormat  # noqa: E402
from nativevoice.core.language import LanguageTag  # noqa: E402
from nativevoice.diagnostics.offline import OfflineGuard  # noqa: E402
from nativevoice.eval.transcription import score as score_pair  # noqa: E402
from nativevoice.models.registry import ModelRegistry  # noqa: E402

SENTENCES = (
    "The quick brown fox jumps over the lazy dog.",
    "Please turn off the lights before you leave the room.",
    "I would like a cup of coffee and a slice of bread.",
)

# Bars fixed in docs/real-speech-protocol.md before any audio existed.
CRITERIA = {
    "false_turn_openings_max": 0,
    "detected_segments_expected": f"{len(SENTENCES)} to {2 * len(SENTENCES)}",
    "read_aloud_wer_reported": True,
    "language_detection_top1_reported": True,
}


def listing(root: str) -> set[str]:
    if not os.path.isdir(root):
        return set()
    return {
        os.path.relpath(os.path.join(directory, name), root)
        for directory, _, names in os.walk(root)
        for name in names
    }


def countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"\r  starting in {remaining:>2}s - be ready to speak", end="", flush=True)
        time.sleep(1)
    print("\r  listening now. Read each sentence once, pausing between them.   ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--countdown", type=int, default=12)
    parser.add_argument("--limit", type=float, default=75.0, help="maximum capture window in seconds")
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--retention", choices=("none", "keep"), default="none")
    parser.add_argument("--recordings-dir", default="")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "device" / "real-speech-read-aloud.json"))
    args = parser.parse_args(argv)

    root = str(data_root())
    before = listing(root)

    print("Read these three sentences aloud, one at a time, at a natural pace:")
    for index, sentence in enumerate(SENTENCES, 1):
        print(f"  {index}. {sentence}")

    registry = ModelRegistry(models_dir())
    manifest = next(item for item in registry.all() if "whisper-base" in item.model_id)
    status = registry.status(manifest.model_id)
    if not status.usable:
        print(f"refusing to score with an unverifiable model: {status.explain()}")
        return 1

    from nativevoice.stt.adapters.faster_whisper_engine import FasterWhisperEngine
    from nativevoice.stt.api import TranscriptionRequest

    report: dict = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": "human read-aloud speech",
        "criteria_predeclared": CRITERIA,
        "retention": args.retention,
        "recordings_written": [],
        "targets": list(SENTENCES),
    }
    guard = OfflineGuard()
    with guard:
        counter = FasterWhisperEngine(manifest=manifest, cpu_threads=4)
        counter.warm()

        fmt = AudioFormat(args.rate, 1)
        source = SounddeviceInput(device=args.device, target_rate=args.rate, frame_size=args.rate // 50)
        source.open()
        segmenter = UtteranceSegmenter(
            EnergySpeechDetector(format=fmt),
            SegmenterConfig(minimum_silence_s=0.6, minimum_speech_s=0.15, maximum_speech_s=12.0, padding_s=0.15),
        )
        pre = Preprocessor(target_format=fmt)
        ring = RingBuffer(fmt, max_frames=args.rate * 120)
        levels: list[float] = []
        peaks: list[float] = []
        segments = []
        idle_segments: list = []
        started = time.perf_counter()
        warm_window = 0.0
        try:
            countdown(args.countdown)
            # a short quiet window after the countdown so pre-speech room noise is measured
            quiet_deadline = time.perf_counter() + 2.0
            in_quiet = True
            for chunk in source.frames():
                processed, measured = pre.process(chunk)
                ring.append(processed)
                levels.append(measured.output_level)
                peaks.append(measured.clipped_fraction)
                if in_quiet:
                    idle_segments.extend(segmenter.push(processed))
                    if time.perf_counter() >= quiet_deadline:
                        in_quiet = False
                        warm_window = processed.duration_s
                else:
                    segments.extend(segmenter.push(processed))
                    if len(segments) >= len(SENTENCES) or time.perf_counter() - started > args.limit:
                        break
            segments.extend(segmenter.flush())
        finally:
            source.close()
        capture_wall = time.perf_counter() - started
        captured = ring.drain_all()

        results = []
        for index, sentence in enumerate(SENTENCES):
            if index >= len(segments):
                results.append({"target": sentence, "heard": None, "status": "MISSING_SEGMENT"})
                continue
            audio = segments[index].audio
            heard = counter.transcribe(
                TranscriptionRequest(
                    audio=audio, language=None, allow_auto_detection=True, max_duration_s=30.0
                )
            )
            mark = score_pair(reference=sentence, hypothesis=heard.text, language=LanguageTag("en"))
            detail = mark.as_dict()
            results.append(
                {
                    "target": sentence,
                    "heard": heard.text,
                    "wer": round(float(mark.wer or 0.0), 4),
                    "cer": round(float(mark.cer or 0.0), 4),
                    "word_errors": {k: detail["word"][k] for k in ("substitutions", "insertions", "deletions", "correct")},
                    "sentence_error": detail["sentence_error_rate"] == 1.0,
                    "detected_language": heard.language.code if heard.language else "",
                    "confidence": heard.confidence,
                    "segment_seconds": round(audio.duration_s, 3),
                    "segment_rms": round(rms(audio.samples), 4),
                    "truncated": segments[index].truncated,
                    "status": "ok",
                }
            )
        counter.close()

    scored = [item for item in results if item.get("status") == "ok"]
    report.update(
        {
            "device_index": args.device,
            "capture": {
                "wall_seconds": round(capture_wall, 3),
                "audio_seconds": round(captured.duration_s, 3),
                "frames": captured.frame_count,
                "median_input_level": (sorted(levels)[len(levels) // 2] if levels else 0.0),
                "peak_input_level": round(max(levels), 4) if levels else 0.0,
                "quiet_window_seconds": round(warm_window, 3),
                "false_turn_openings_in_quiet_window": len(idle_segments),
                "segments_detected": len(segments),
                "segment_onsets": [
                    {"first_sample": s.first_sample, "seconds": round(s.duration_s, 3), "truncated": s.truncated}
                    for s in segments
                ],
            },
            "results": results,
            "summary": {
                "sentences_expected": len(SENTENCES),
                "sentences_scored": len(scored),
                "read_aloud_wer_mean": round(statistics.fmean(i["wer"] for i in scored), 4) if scored else None,
                "read_aloud_cer_mean": round(statistics.fmean(i["cer"] for i in scored), 4) if scored else None,
                "exact_matches": sum(1 for i in scored if i["wer"] == 0.0),
                "language_detection_top1": (
                    sum(1 for i in scored if i["detected_language"] == "en") / len(scored) if scored else None
                ),
            },
            "network_attempts": [f"{attempt.host}:{attempt.port}" for attempt in guard.attempts],
        }
    )

    after = listing(root)
    created = sorted(after - before)
    report["persistence"] = {
        "data_root": root,
        "files_before": len(before),
        "files_after": len(after),
        "created_during_run": created,
        "no_audio_written": not created,
    }
    if args.retention == "keep" and scored:
        from nativevoice.audio import write_wav

        destination = Path(args.recordings_dir or (Path(root) / "hosts" / "real-speech" / "recordings"))
        destination.mkdir(parents=True, exist_ok=True)
        for index, sentence in enumerate(SENTENCES):
            if index < len(segments) and segments[index].audio is not None:
                path = destination / f"readaloud_{int(time.time())}_{index:02d}.wav"
                write_wav(str(path), segments[index].audio)
                report["recordings_written"].append(str(path))
        report["persistence"]["created_during_run"] = sorted(listing(root) - before)

    verdicts = {
        "no_audio_persisted": "PASS" if report["persistence"]["no_audio_written"] else "FAIL",
        "offline_no_attempts": "PASS" if not report["network_attempts"] else "FAIL",
        "false_turn_openings": (
            "PASS" if report["capture"]["false_turn_openings_in_quiet_window"] <= CRITERIA["false_turn_openings_max"] else "FAIL"
        ),
        "all_sentences_detected": "PASS" if len(scored) == len(SENTENCES) else "FAIL",
        "read_aloud_wer": "MEASURED (no pass bar set, by design)" if scored else "NOT RUN (no speech segment per sentence)",
        "language_detection": "MEASURED" if scored else "NOT RUN",
        "vad_onset_error_ms": "NOT RUN - a manual read-aloud run has no ground-truth onset timestamp to measure against",
        "endpointing_on_long_pause": "NOT RUN - step 3 of the protocol covers it",
        "noise_and_barge_in": "NOT RUN - steps 4 and 6",
    }
    report["acceptance"] = verdicts
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("capture", "summary", "acceptance", "persistence", "network_attempts")}, indent=2))
    for item in results:
        if item.get("status") == "ok":
            print(f"  target: {item['target']}\n  heard : {item['heard']}  (WER {item['wer']}, CER {item['cer']}, lang {item['detected_language']})")
    print(f"artefact -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
