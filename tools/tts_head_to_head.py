"""Head-to-head TTS comparison: installed Kokoro vs the KittenTTS trial candidate.

Run from the project root:

    python tools/tts_head_to_head.py

Nothing here changes a default, registers an identity in the shared catalogue, or promotes anything. The
candidate is loaded from an isolated trial root, gated by the same approval and integrity checks as the
incumbent, measured with the same probes, and written out for Shreshtha to judge.

Both engines are timed on **one sentence at a time**, which is the only basis on which they are comparable:
Kitten's runtime exposes whole-utterance generation only, so measuring Kokoro through its sentence
chunker would be comparing our pipeline against their model. Kokoro's chunked streaming figure is
reported separately and labelled as such, never mixed into the head-to-head column.

Stages are timed apart because they fail differently: engine init (loader, once per language),
first audio (what a listener perceives), synthesis total and RTF (whether a reply can keep up),
playback startup (a device cost, not a model cost), and round-trip recognition (quality, on our own
synthetic audio - never presented as human-speech accuracy).
"""

from __future__ import annotations

import argparse
import array
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sherpa_onnx  # noqa: E402
from nativevoice.audio import resample  # noqa: E402
from nativevoice.config import models_dir  # noqa: E402
from nativevoice.core.audio import AudioBuffer, AudioFormat  # noqa: E402
from nativevoice.core.language import LanguageTag  # noqa: E402
from nativevoice.diagnostics.offline import OfflineGuard  # noqa: E402
from nativevoice.eval.transcription import score as score_pair  # noqa: E402
from nativevoice.models.approval import ensure_approved  # noqa: E402
from nativevoice.models.integrity import inspect as inspect_integrity  # noqa: E402
from nativevoice.models.registry import ModelRegistry  # noqa: E402
from nativevoice.stt.adapters.faster_whisper_engine import FasterWhisperEngine  # noqa: E402
from nativevoice.stt.api import TranscriptionRequest  # noqa: E402
from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine  # noqa: E402
from nativevoice.tts.api import SynthesisRequest  # noqa: E402

SENTENCES = (
    "The quick brown fox jumps over the lazy dog.",
    "Please turn off the lights before you leave the room.",
    "I would like a cup of coffee and a slice of bread.",
    "The weather today is pleasant and the sky is clear.",
    "My favourite book is about a young engineer in the city.",
    "We walked to the park and watched the children play.",
    "Could you tell me the time and the nearest train station?",
    "She sells seashells by the seashore every summer.",
)
LONG_REPLY = (
    "The quick brown fox jumps over the lazy dog, and then it walks home through the park "
    "while the rain falls softly on the empty street."
)
REPEATS = 2


class KittenTrialEngine:
    """The candidate, driven directly and kept inside this tool.

    Deliberately not a package adapter: if the candidate is promoted, that happens as its own reviewed
    change with its own adapter and tests, not as a side effect of a benchmark run.
    """

    engine_id = "sherpa-onnx-kitten-trial"
    output_format = AudioFormat(24000, 1)

    def __init__(self, manifest, num_threads: int = 4) -> None:
        ensure_approved(manifest)
        base = Path(manifest.asset_root)
        report = inspect_integrity(manifest, base)
        if not report.ok:
            raise RuntimeError(f"candidate failed integrity verification: {report.reasons}")
        self.manifest = manifest
        self.directory = base
        self.num_threads = num_threads
        self.tts = None

    def warm(self) -> float:
        started = time.perf_counter()
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kitten=sherpa_onnx.OfflineTtsKittenModelConfig(
                    model=str(self.directory / "model.int8.onnx"),
                    voices=str(self.directory / "voices.bin"),
                    tokens=str(self.directory / "tokens.txt"),
                    data_dir=str(self.directory / "espeak-ng-data"),
                ),
                num_threads=self.num_threads,
                provider="cpu",
            ),
            max_num_sentences=1,
        )
        self.tts = sherpa_onnx.OfflineTts(config)
        return time.perf_counter() - started

    def generate(self, text: str, sid: int = 0) -> AudioBuffer:
        generated = self.tts.generate(text=text, sid=sid, speed=1.0)
        values = array.array("h", (max(-32768, min(32767, int(round(s * 32767)))) for s in generated.samples))
        return AudioBuffer(values, AudioFormat(generated.sample_rate, 1))

    @property
    def num_speakers(self) -> int:
        return int(getattr(self.tts, "num_speakers", 0) or 0)

    def close(self) -> None:
        self.tts = None


def peak_sampler():
    state = {"peak": 0, "stop": False}

    def poll():
        try:
            import psutil

            process = psutil.Process()
            while not state["stop"]:
                state["peak"] = max(state["peak"], process.memory_info().rss)
                time.sleep(0.01)
        except Exception:
            while not state["stop"]:
                time.sleep(0.05)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    return state, thread


def summarise(firsts, walls, audios, wer, cer, exact) -> dict:
    med_first = statistics.median(firsts)
    med_wall = statistics.median(walls)
    med_audio = statistics.median(audios) or 0.001
    return {
        "first_audio_ms_median": round(med_first * 1000, 1),
        "first_audio_ms_best": round(min(firsts) * 1000, 1),
        "first_audio_ms_worst": round(max(firsts) * 1000, 1),
        "synthesis_ms_median": round(med_wall * 1000, 1),
        "audio_seconds_median": round(med_audio, 3),
        "rtf_median": round(med_wall / med_audio, 4),
        "rtf_best": round(min(w / a for w, a in zip(walls, audios) if a), 4),
        "roundtrip_wer_mean": round(statistics.fmean(wer), 4),
        "roundtrip_cer_mean": round(statistics.fmean(cer), 4),
        "exact_matches_of": f"{sum(exact)}/{len(exact)}",
        "trials": len(firsts),
    }


def bench_kokoro(manifest, listen, threads: int) -> dict:
    state, sampler = peak_sampler()
    init_start = time.perf_counter()
    engine = SherpaKokoroEngine(manifest=manifest, num_threads=threads)
    engine.warm("en")
    init_s = time.perf_counter() - init_start

    firsts, walls, audios, wers, cers, exact, chunk_counts = [], [], [], [], [], [], []
    for _ in range(REPEATS):
        for sentence in SENTENCES:
            outcome = engine.synthesize(SynthesisRequest(text=sentence, language=LanguageTag("en")))
            timings = outcome.timings or {}
            # single-sentence inputs must stay single-chunk for the comparison to be fair;
            # the count is recorded rather than assumed
            chunk_counts.append(int(timings.get("chunks", 0) or 0))
            firsts.append((timings.get("first_audio_ms") or timings.get("first_result_ms") or 0) / 1000)
            walls.append((timings.get("total_ms") or 0) / 1000)
            audios.append(outcome.audio.duration_s)
            heard = listen(outcome.audio)
            mark = score_pair(reference=sentence, hypothesis=heard, language=LanguageTag("en"))
            wers.append(mark.wer or 0.0)
            cers.append(mark.cer or 0.0)
            exact.append((mark.wer or 0.0) == 0.0)
    streamed_first = None
    for _ in range(REPEATS):
        span = time.perf_counter()
        for event in engine.stream(SynthesisRequest(text=LONG_REPLY, language=LanguageTag("en"))):
            if event.audio is not None:
                streamed_first = (time.perf_counter() - span) * 1000
                break
    row_speakers = int(getattr(engine, "num_speakers", None) or getattr(engine._tts, "num_speakers", 0) or 0)
    engine.close()
    state["stop"] = True
    sampler.join(timeout=0.5)
    row = summarise(firsts, walls, audios, wers, cers, exact)
    row.update(
        {
            "chunks_per_sentence_max": max(chunk_counts) if chunk_counts else None,
            "asset": manifest.model_id,
            "licence": manifest.provenance.licence_spdx,
            "redistributable": manifest.provenance.redistributable,
            "engine_init_s": round(init_s, 3),
            "peak_rss_mib": round(state["peak"] / 2**20, 1) if state["peak"] else None,
            "languages": list(manifest.languages),
            "num_speakers": row_speakers,
            "voices_named_in_asset": False,
            "chunked_stream_first_audio_ms_median": None if streamed_first is None else round(streamed_first, 1),
            "measurement_basis": "single sentence, whole-utterance generate on both engines",
        }
    )
    return row


def bench_kitten(manifest, listen, threads: int) -> dict:
    state, sampler = peak_sampler()
    engine = KittenTrialEngine(manifest=manifest, num_threads=threads)
    init_s = engine.warm()

    firsts, walls, audios, wers, cers, exact = [], [], [], [], [], []
    for _ in range(REPEATS):
        for sentence in SENTENCES:
            span = time.perf_counter()
            buffer = engine.generate(sentence)
            wall = time.perf_counter() - span
            firsts.append(wall)
            walls.append(wall)
            audios.append(buffer.duration_s)
            heard = listen(buffer)
            mark = score_pair(reference=sentence, hypothesis=heard, language=LanguageTag("en"))
            wers.append(mark.wer or 0.0)
            cers.append(mark.cer or 0.0)
            exact.append((mark.wer or 0.0) == 0.0)
    speakers = engine.num_speakers
    row_speakers = int(getattr(engine, "num_speakers", None) or getattr(engine._tts, "num_speakers", 0) or 0)
    engine.close()
    state["stop"] = True
    sampler.join(timeout=0.5)
    row = summarise(firsts, walls, audios, wers, cers, exact)
    row.update(
        {
            "asset": manifest.model_id,
            "licence": manifest.provenance.licence_spdx,
            "redistributable": manifest.provenance.redistributable,
            "engine_init_s": round(init_s, 3),
            "peak_rss_mib": round(state["peak"] / 2**20, 1) if state["peak"] else None,
            "languages": list(manifest.languages),
            "num_speakers": speakers,
            "voices_named_in_asset": True,
            "chunked_stream_first_audio_ms_median": None,
            "measurement_basis": "single sentence, whole-utterance generate on both engines",
        }
    )
    return row


def playback_startup() -> dict:
    """Opening the rendering stream is a device cost, timed apart from any model."""
    try:
        import sounddevice as sd
    except Exception as error:
        return {"available": False, "reason": str(error)}
    stamps = []
    for _ in range(3):
        started = time.perf_counter()
        stream = sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16")
        stamps.append((time.perf_counter() - started) * 1000)
        stream.abort()
    return {"available": True, "stream_open_ms_median": round(statistics.median(stamps), 1), "audio_played": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "benchmarks" / "tts-head-to-head.json"))
    args = parser.parse_args(argv)

    incumbent = ModelRegistry(models_dir())
    trial = ModelRegistry(ROOT / ".trials" / "kitten" / "models")
    kokoro_manifest = next(item for item in incumbent.all() if "kokoro" in item.model_id)
    kitten_manifest = next(item for item in trial.all() if "kitten" in item.model_id)

    results: dict = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "threads": args.threads,
        "sentences": len(SENTENCES),
        "repeats": REPEATS,
        "engines": {},
        "conditions": {},
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        results["conditions"] = {
            "free_ram_gb_at_start": round(vm.available / 2**30, 2),
            "processes": len(list(psutil.process_iter())),
            "logical_cpus": os.cpu_count(),
        }
    except Exception:
        pass

    guard = OfflineGuard()
    with guard:
        whisper = next(item for item in incumbent.all() if "whisper-base" in item.model_id)
        listener = FasterWhisperEngine(manifest=whisper, cpu_threads=args.threads)
        listener.warm()

        def listen(buffer: AudioBuffer) -> str:
            feed = resample(buffer, 16000) if buffer.format.sample_rate != 16000 else buffer
            return listener.transcribe(
                TranscriptionRequest(
                    audio=feed, language=LanguageTag("en"), allow_auto_detection=False, max_duration_s=30.0
                )
            ).text

        results["engines"]["incumbent_kokoro"] = bench_kokoro(kokoro_manifest, listen, args.threads)
        results["engines"]["candidate_kitten"] = bench_kitten(kitten_manifest, listen, args.threads)
        results["playback_startup"] = playback_startup()
        listener.close()

    inc, cand = results["engines"]["incumbent_kokoro"], results["engines"]["candidate_kitten"]
    results["comparison"] = {
        "first_audio_speedup_x": round(inc["first_audio_ms_median"] / max(1, cand["first_audio_ms_median"]), 2),
        "rtf_ratio_candidate_over_incumbent": round(cand["rtf_median"] / max(0.001, inc["rtf_median"]), 3),
        "quality_wer_delta": round(cand["roundtrip_wer_mean"] - inc["roundtrip_wer_mean"], 4),
        "candidate_reaches_realtime": bool(cand["rtf_median"] < 1.0),
        "candidate_under_500ms_first_audio": bool(cand["first_audio_ms_median"] < 500),
        "language_coverage": {"incumbent": inc["languages"], "candidate": cand["languages"]},
        "distribution": {
            "incumbent_redistributable": inc["redistributable"],
            "candidate_redistributable": cand["redistributable"],
            "note": "both bundle GPL-3.0 espeak-ng data; the candidate additionally discloses no training-data or voice-actor provenance",
        },
        "status": "report only - default engine and shared catalogue unchanged, promotion needs approval",
    }
    results["network_attempts"] = [f"{attempt.host}:{attempt.port}" for attempt in guard.attempts]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for key, row in results["engines"].items():
        print(f"{key}: init={row['engine_init_s']}s first={row['first_audio_ms_median']}ms rtf={row['rtf_median']} "
              f"wer={row['roundtrip_wer_mean']} exact={row['exact_matches_of']} ram={row['peak_rss_mib']}MiB "
              f"voices={row['num_speakers']}")
    print("comparison:", json.dumps(results["comparison"], indent=2))
    print(f"artefact -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
