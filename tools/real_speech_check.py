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
  `NOT RUN` with the reason, never silently dropped or quietly passed;
* **history is never clobbered** - an existing artefact is copied to a `.previous` file before a new run
  writes over it, so a corrected rerun cannot erase the evidence it is correcting.

Speech is matched to sentences **by content, not by position**: a run that produces one utterance out of
order is scored against the sentence it actually says, and anything that matches nothing is reported as
extra rather than forced onto a target. The quiet-room measurement and speech capture share one audio
stream, so nothing spoken is ever discarded on the way to being measured.

Results here are human-speech measurements and are kept separate from the synthetic round-trip numbers:
read-aloud speech is cleaner than spontaneous speech, so these WER figures are optimistic for real
conversation and are labelled read-aloud wherever they appear.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
from nativevoice.diagnostics.evidence_paths import write_json  # noqa: E402
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

# Beyond this word-error distance a recognised utterance is not called the same sentence at all. The cap
# exists so an unrecognised or unrelated segment is reported as extra instead of being scored against
# whichever target is nearest to it.
ALIGN_WER_CAP = 0.60

QUIET_WINDOW_SECONDS = 2.0


def configure_console() -> None:
    """Give the console a chance to carry whatever the recogniser heard.

    A Windows console defaults to cp1252, and a transcription that drifts into another script would
    otherwise raise on the way to the screen.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            continue  # a captured or non-text stream - `emit` is the fallback for those


def emit(line: str = "") -> None:
    """Print, and never let an undisplayable character abort a run.

    The artefact is the record; the console is a courtesy. If even the UTF-8 attempt fails - a stream that
    cannot be reconfigured, or one closed mid-report - the line is reproduced as ASCII escapes, and a
    hostile console still cannot raise past here.
    """
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        try:
            print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
        except (UnicodeEncodeError, ValueError, OSError):
            pass
    except (ValueError, OSError):
        pass


def emit_block(text: str) -> None:
    """Print a multi-line passage line by line, so one bad glyph cannot swallow the rest of it."""
    for line in text.splitlines() or [""]:
        emit(line)


def listing(root: str) -> set[str]:
    if not os.path.isdir(root):
        return set()
    return {
        os.path.relpath(os.path.join(directory, name), root)
        for directory, _, names in os.walk(root)
        for name in names
    }


def countdown(seconds: int) -> None:
    """Count down with one line per second and an unmistakable GO marker.

    Carriage-return ticks disappear in a scrollback-fed console: the whole sequence arrives as one mashed
    line *after* the window has already opened, which is how a speaker in the room missed the start. There
    is deliberately no audible cue either - the input stream is already open here, so a beep would land in
    the captured audio and manufacture a turn opening inside the quiet baseline.
    """
    emit(f"  window opens in {seconds} seconds - get ready, arm's length, natural pace")
    for remaining in range(seconds, 0, -1):
        emit(f"  ... {remaining}")
        time.sleep(1)
    emit("")
    emit("  " + "=" * 58)
    emit("  GO - START READING NOW, one sentence at a time")
    emit("  " + "=" * 58)


def word_error(reference: str, hypothesis: str) -> float:
    """WER of one recognised string against one target, as the evaluation harness scores it."""
    mark = score_pair(reference=reference, hypothesis=hypothesis, language=LanguageTag("en"))
    return float(mark.wer or 0.0)


def align_targets(targets: list[str], heard: list[str]) -> dict:
    """Pair recognised segments to the intended sentences by content, never by position.

    Each target takes its closest unused segment while it is within `ALIGN_WER_CAP`; ties break on
    segment order then target order, so the same inputs always produce the same pairing. A target with
    no acceptable segment is *missing*, a segment nobody claimed is *extra* - duplicates included, and
    named as such - and neither is ever invented into a match.
    """
    costs = [
        (word_error(target, text), target_index, segment_index)
        for target_index, target in enumerate(targets)
        for segment_index, text in enumerate(heard)
    ]
    costs.sort(key=lambda item: (item[0], item[2], item[1]))

    taken_targets: set[int] = set()
    taken_segments: set[int] = set()
    matches: list[dict] = []
    for wer, target_index, segment_index in costs:
        if wer > ALIGN_WER_CAP:
            break
        if target_index in taken_targets or segment_index in taken_segments:
            continue
        taken_targets.add(target_index)
        taken_segments.add(segment_index)
        matches.append(
            {"target_index": target_index, "segment_index": segment_index, "wer": round(wer, 4)}
        )

    missing: list[dict] = []
    for target_index, _ in enumerate(targets):
        if target_index in taken_targets:
            continue
        options = [item for item in costs if item[1] == target_index]
        closest = min(options, key=lambda item: item[0]) if options else None
        missing.append(
            {
                "target_index": target_index,
                "closest_segment_index": closest[2] if closest else None,
                "closest_wer": round(closest[0], 4) if closest else None,
                "reason": (
                    "no segment recognised"
                    if not options
                    else "no segment within the alignment cap"
                ),
            }
        )

    extra: list[dict] = []
    for segment_index, text in enumerate(heard):
        if segment_index in taken_segments:
            continue
        options = [item for item in costs if item[2] == segment_index]
        closest = min(options, key=lambda item: item[0]) if options else None
        duplicate = bool(closest) and closest[0] <= ALIGN_WER_CAP
        extra.append(
            {
                "segment_index": segment_index,
                "text": text,
                "closest_target_index": closest[1] if closest else None,
                "closest_wer": round(closest[0], 4) if closest else None,
                "reason": (
                    f"duplicate of sentence {closest[1] + 1}"
                    if duplicate
                    else "recognised speech that matches no target sentence"
                ),
            }
        )

    return {
        "method": "content alignment (minimum WER, greedy, capped)",
        "cap": ALIGN_WER_CAP,
        "matches": sorted(matches, key=lambda item: item["target_index"]),
        "missing_targets": missing,
        "extra_segments": extra,
    }


def classify_openings(
    segments: list, rate: int, quiet_window_seconds: float, taken_segments: set[int], texts: list[str] | None = None
) -> dict:
    """Describe every turn opening the detector made, and which of them can honestly be called false.

    An opening inside the quiet window counts as false only when it claimed no sentence: speech spoken
    immediately after the countdown is genuine input, not room noise, and it is transcribed and aligned
    like any other utterance.

    `texts`, when supplied, also counts openings **outside** that window that produced an empty
    transcript. That is a separate measure with its own name - the quiet-window bar was agreed before any
    audio existed, and a later finding must not silently move it (A44).
    """
    openings = []
    for segment_index, segment in enumerate(segments):
        onset_seconds = segment.first_sample / rate
        opening = {
            "segment_index": segment_index,
            "onset_seconds": round(onset_seconds, 3),
            "duration_seconds": round(segment.duration_s, 3),
            "within_quiet_window": onset_seconds < quiet_window_seconds,
            "matched_a_sentence": segment_index in taken_segments,
            "truncated": segment.truncated,
        }
        if texts is not None:
            opening["transcript_empty"] = not (texts[segment_index] or "").strip()
        openings.append(opening)
    inside = [item for item in openings if item["within_quiet_window"]]
    report = {
        "openings": openings,
        "openings_in_quiet_window": len(inside),
        "false_turn_openings": sum(1 for item in inside if not item["matched_a_sentence"]),
        "speech_inside_quiet_window": sum(1 for item in inside if item["matched_a_sentence"]),
    }
    if texts is None:
        report["empty_transcript_openings_outside_quiet_window"] = None
        report["empty_transcript_openings_outside_quiet_window_onsets"] = None
        return report
    empty_outside = [
        item for item in openings if not item["within_quiet_window"] and item["transcript_empty"]
    ]
    report["empty_transcript_openings_outside_quiet_window"] = len(empty_outside)
    report["empty_transcript_openings_outside_quiet_window_onsets"] = [
        item["onset_seconds"] for item in empty_outside
    ]
    return report


def annotate_alignment(alignment: dict, heard: list[dict]) -> dict:
    """Attach the language and confidence the engine asserted to every segment the report mentions.

    A sentence that came back in the wrong language is only evidence if the run records what language was
    claimed. Unmatched segments used to carry the text alone, which left a whole-sentence substitution
    unattributable after the fact (A43). Values are copied from what the engine returned; an empty or
    missing code stays empty rather than being filled in.
    """
    for item in alignment["extra_segments"]:
        voice = heard[item["segment_index"]]
        item["detected_language"] = voice.get("language", "")
        item["confidence"] = voice.get("confidence")
    for item in alignment["missing_targets"]:
        index = item.get("closest_segment_index")
        if index is None or index >= len(heard):
            continue
        voice = heard[index]
        item["closest_segment_text"] = voice.get("text", "")
        item["closest_segment_language"] = voice.get("language", "")
        item["closest_segment_confidence"] = voice.get("confidence")
    return alignment


def collect_segments(
    *,
    chunks,
    pre,
    segmenter,
    ring,
    quiet_window_s: float,
    limit_s: float,
    stop_at_segments: int,
) -> dict:
    """Feed capture through one segmenter stream while measuring the quiet room alongside it.

    The quiet window only *samples levels*: every chunk still goes to the segmenter, so a sentence begun
    immediately after the countdown is captured like any other instead of being logged as noise and
    discarded. `quiet_seconds` is the window actually measured, in audio seconds.
    """
    segments: list = []
    levels: list[float] = []
    quiet_levels: list[float] = []
    quiet_seconds = 0.0
    audio_seconds = 0.0
    stopped_because = "the capture source ended"
    for chunk in chunks:
        processed, measured = pre.process(chunk)
        ring.append(processed)
        levels.append(measured.output_level)
        if audio_seconds < quiet_window_s:
            quiet_levels.append(measured.output_level)
            quiet_seconds += processed.duration_s
        audio_seconds += processed.duration_s
        segments.extend(segmenter.push(processed))
        if len(segments) >= stop_at_segments:
            stopped_because = f"reached {stop_at_segments} turn openings"
            break
        if audio_seconds >= limit_s:
            stopped_because = f"reached the {limit_s:.0f} s capture limit"
            break
    segments.extend(segmenter.flush())
    return {
        "segments": segments,
        "levels": levels,
        "quiet_levels": quiet_levels,
        "quiet_seconds": quiet_seconds,
        "audio_seconds": audio_seconds,
        "stopped_because": stopped_because,
    }


def write_artefact(report: dict, path: Path) -> str:
    """Save the record, and return where the run it replaced was kept.

    Always called before anything is printed: the console is presentation, this file is the evidence, and
    a terminal that cannot display a Japanese hallucination must not be able to cost us either.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = preserve_previous(path)
    if kept:
        report["previous_artefact_preserved"] = kept
    write_json(path, report)
    return kept


def open_and_capture(
    *,
    source,
    pre,
    segmenter,
    ring,
    quiet_window_s: float,
    limit_s: float,
    stop_at_segments: int,
    on_ready,
) -> dict:
    """Give the start cue, then open the microphone, so nothing from before GO reaches the analysis.

    The input callback queues audio from the instant the stream opens. Opening it *before* the countdown
    therefore poured up to a minute of pre-GO silence into the analysis at CPU speed: it ate the capture
    limit, moved the quiet baseline onto the room as it was during the countdown rather than as it was
    when the speaker began, and left the speaker a much shorter window than promised.
    """
    on_ready()
    started = time.perf_counter()
    source.open()
    try:
        stats = collect_segments(
            chunks=source.frames(),
            pre=pre,
            segmenter=segmenter,
            ring=ring,
            quiet_window_s=quiet_window_s,
            limit_s=limit_s,
            stop_at_segments=stop_at_segments,
        )
    finally:
        source.close()
    stats["wall_seconds"] = time.perf_counter() - started
    return stats


def preserve_previous(path: Path) -> str:
    """Copy the artefact a run is about to replace, named by the run it holds.

    Correction must never mean erasure: the backup carries the earlier run's own timestamp, so a chain of
    reruns leaves one file per generation instead of overwriting the last one.
    """
    if not path.exists():
        return ""
    try:
        stamp = str(json.loads(path.read_text(encoding="utf-8")).get("generated", ""))
    except (ValueError, OSError):
        stamp = ""
    safe = stamp.replace("-", "").replace(":", "").replace("T", "-") or "undated"
    previous = path.with_name(f"{path.stem}.{safe}.previous{path.suffix}")
    shutil.copyfile(path, previous)
    return str(previous)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--countdown", type=int, default=12)
    parser.add_argument("--limit", type=float, default=75.0, help="maximum capture window in seconds")
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--quiet-window", type=float, default=QUIET_WINDOW_SECONDS)
    parser.add_argument("--retention", choices=("none", "keep"), default="none")
    parser.add_argument("--recordings-dir", default="")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "device" / "real-speech-read-aloud.json"))
    args = parser.parse_args(argv)
    configure_console()

    root = str(data_root())
    before = listing(root)

    emit("Read these three sentences aloud, one at a time, at a natural pace:")
    for index, sentence in enumerate(SENTENCES, 1):
        emit(f"  {index}. {sentence}")

    registry = ModelRegistry(models_dir())
    manifest = next(item for item in registry.all() if "whisper-base" in item.model_id)
    status = registry.status(manifest.model_id)
    if not status.usable:
        emit(f"refusing to score with an unverifiable model: {status.explain()}")
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
        segmenter = UtteranceSegmenter(
            EnergySpeechDetector(format=fmt),
            SegmenterConfig(minimum_silence_s=0.6, minimum_speech_s=0.15, maximum_speech_s=12.0, padding_s=0.15),
        )
        pre = Preprocessor(target_format=fmt)
        ring = RingBuffer(fmt, max_frames=args.rate * 120)
        stats = open_and_capture(
            source=source,
            pre=pre,
            segmenter=segmenter,
            ring=ring,
            quiet_window_s=args.quiet_window,
            limit_s=args.limit,
            stop_at_segments=2 * len(SENTENCES),
            on_ready=lambda: countdown(args.countdown),
        )
        capture_wall = stats["wall_seconds"]
        captured = ring.drain_all()
        segments = stats["segments"]
        levels = stats["levels"]
        quiet_levels = stats["quiet_levels"]
        quiet_seconds = stats["quiet_seconds"]

        heard: list[dict] = []
        for segment in segments:
            transcription = counter.transcribe(
                TranscriptionRequest(
                    audio=segment.audio,
                    language=None,
                    allow_auto_detection=True,
                    max_duration_s=30.0,
                )
            )
            heard.append(
                {
                    "text": transcription.text,
                    "language": transcription.language.code if transcription.language else "",
                    "confidence": transcription.confidence,
                }
            )
        counter.close()

    alignment = align_targets(list(SENTENCES), [item["text"] for item in heard])
    alignment = annotate_alignment(alignment, heard)
    taken = {item["segment_index"] for item in alignment["matches"]}
    opening_report = classify_openings(
        segments, args.rate, quiet_seconds, taken, texts=[item["text"] for item in heard]
    )

    results = []
    for target_index, sentence in enumerate(SENTENCES):
        match = next((item for item in alignment["matches"] if item["target_index"] == target_index), None)
        if match is None:
            closest = next(
                (item for item in alignment["missing_targets"] if item["target_index"] == target_index), {}
            )
            results.append(
                {
                    "target": sentence,
                    "heard": None,
                    "status": "MISSING_SEGMENT",
                    "why": closest.get("reason", "no segment recognised"),
                    "closest_wer": closest.get("closest_wer"),
                    "closest_segment_said": closest.get("closest_segment_text", ""),
                    "closest_segment_language": closest.get("closest_segment_language", ""),
                    "closest_segment_confidence": closest.get("closest_segment_confidence"),
                }
            )
            continue
        segment_index = match["segment_index"]
        segment = segments[segment_index]
        voice = heard[segment_index]
        mark = score_pair(reference=sentence, hypothesis=voice["text"], language=LanguageTag("en"))
        detail = mark.as_dict()
        results.append(
            {
                "target": sentence,
                "heard": voice["text"],
                "wer": round(float(mark.wer or 0.0), 4),
                "cer": round(float(mark.cer or 0.0), 4),
                "word_errors": {
                    key: detail["word"][key]
                    for key in ("substitutions", "insertions", "deletions", "correct")
                },
                "sentence_error": detail["sentence_error_rate"] == 1.0,
                "detected_language": voice["language"],
                "confidence": voice["confidence"],
                "segment_index": segment_index,
                "segment_seconds": round(segment.audio.duration_s, 3),
                "segment_rms": round(rms(segment.audio.samples), 4),
                "truncated": segment.truncated,
                "status": "ok",
            }
        )

    scored = [item for item in results if item.get("status") == "ok"]
    report.update(
        {
            "device_index": args.device,
            "capture": {
                "countdown_seconds": args.countdown,
                "stream_opened_at": "GO",
                "wall_seconds": round(capture_wall, 3),
                "audio_seconds": round(captured.duration_s, 3),
                "frames": captured.frame_count,
                "median_input_level": (sorted(levels)[len(levels) // 2] if levels else 0.0),
                "peak_input_level": round(max(levels), 4) if levels else 0.0,
                "quiet_window_seconds": round(quiet_seconds, 3),
                "stopped_because": stats["stopped_because"],
                "quiet_window_median_level": (
                    round(sorted(quiet_levels)[len(quiet_levels) // 2], 6) if quiet_levels else 0.0
                ),
                "quiet_window_peak_level": round(max(quiet_levels), 4) if quiet_levels else 0.0,
                "segments_detected": len(segments),
                **opening_report,
            },
            "alignment": alignment,
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
                "unmatched_segments": len(alignment["extra_segments"]),
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
        for item in results:
            if item.get("status") != "ok":
                continue
            path = destination / f"readaloud_{int(time.time())}_{item['segment_index']:02d}.wav"
            write_wav(str(path), segments[item["segment_index"]].audio)
            report["recordings_written"].append(str(path))
        report["persistence"]["created_during_run"] = sorted(listing(root) - before)
        report["persistence"]["no_audio_written"] = False

    verdicts = {
        "no_audio_persisted": "PASS" if report["persistence"]["no_audio_written"] else "FAIL",
        "offline_no_attempts": "PASS" if not report["network_attempts"] else "FAIL",
        "false_turn_openings": (
            "PASS"
            if opening_report["false_turn_openings"] <= CRITERIA["false_turn_openings_max"]
            else "FAIL"
        ),
        "all_sentences_detected": "PASS" if len(scored) == len(SENTENCES) else "FAIL",
        "empty_transcript_openings": (
            "NOT RUN - no transcripts were taken"
            if opening_report["empty_transcript_openings_outside_quiet_window"] is None
            else (
                "MEASURED (informational, no pass bar set) - "
                f"{opening_report['empty_transcript_openings_outside_quiet_window']} opening(s) outside "
                "the quiet window produced no transcript"
            )
        ),
        "read_aloud_wer": "MEASURED (no pass bar set, by design)" if scored else "NOT RUN (no segment matched a sentence)",
        "language_detection": "MEASURED" if scored else "NOT RUN",
        "vad_onset_error_ms": "NOT RUN - a manual read-aloud run has no ground-truth onset timestamp to measure against",
        "endpointing_on_long_pause": "NOT RUN - step 3 of the protocol covers it",
        "noise_and_barge_in": "NOT RUN - steps 4 and 6",
    }
    report["acceptance"] = verdicts
    destination = Path(args.out)
    kept = write_artefact(report, destination)

    emit_block(
        json.dumps(
            {
                key: report[key]
                for key in ("capture", "alignment", "summary", "acceptance", "persistence", "network_attempts")
            },
            indent=2,
        )
    )
    for item in results:
        if item.get("status") == "ok":
            emit_block(
                f"  target: {item['target']}\n  heard : {item['heard']}"
                f"  (WER {item['wer']}, CER {item['cer']}, lang {item['detected_language']})"
            )
        else:
            said = item.get("closest_segment_said") or ""
            nearest = (
                f" | closest segment said {said!r} as"
                f" {item.get('closest_segment_language') or '(no language asserted)'}"
                f" at confidence {item.get('closest_segment_confidence')}"
                if said
                else ""
            )
            emit_block(f"  target: {item['target']}\n  heard : none ({item['why']}){nearest}")
    for item in alignment["extra_segments"]:
        emit(
            f"  extra : {item['text']}  ({item['reason']}, closest WER {item['closest_wer']},"
            f" heard as {item['detected_language'] or '(no language asserted)'}"
            f" at confidence {item['confidence']})"
        )
    emit(f"artefact -> {args.out}" + (f"  (previous kept at {kept})" if kept else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
