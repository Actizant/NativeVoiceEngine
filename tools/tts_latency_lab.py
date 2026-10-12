"""TTS latency laboratory: what can be improved without a new asset, measured.

The headline finding from the audit is that synthesis runs slower than real time on
this machine (RTF 1.8-2.4) and first audio takes 4-8 s. Before proposing a different
model - which needs approval, a download and a licence review - every lever that
exists in the *current* implementation gets measured, so the replacement case is
built on numbers rather than on preference.

Levers examined, all available without new assets:

* inference threads, past the point of diminishing returns;
* the size of the first chunk handed to the engine, which is what actually bounds
  time-to-first-audio when generation is per-chunk;
* the cost of switching language, which reloads the engine and is invisible until a
  conversation changes language in mid-flight;
* whether the runtime offers a streaming callback at all, because if it does,
  chunk-level latency is a choice we are making rather than a limit we inherited.

Every number is labelled measured, with the free-memory and process context beside
it, and nothing here is presented as a profile threshold.

    python tools/tts_latency_lab.py
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

from nativevoice.config import models_dir  # noqa: E402  (rewritten below if namespace differs)

TEXT_LONG = (
    "The quick brown fox jumps over the lazy dog, and then it walks slowly home "
    "through the quiet park while the rain falls softly on the empty streets."
)
TEXT_SHORT = "The voice platform is running locally."


def median(values):
    return round(statistics.median(values), 4) if values else None


def probe_engine_config():
    """Does the installed runtime expose a per-sentence callback? Reported, not assumed."""
    import sherpa_onnx as so

    doc = (so.OfflineTts.generate.__doc__ or "")
    lines = [line.strip() for line in doc.splitlines() if line.strip()]
    return {"signature_lines": lines[:6], "mentions_callback": any("callback" in line.lower() for line in lines)}


def run(threads: int, chunk_tokens: int, registry_manifest, repeats: int):
    from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine
    from nativevoice.tts.chunking import TextChunker
    from nativevoice.tts.api import SynthesisRequest
    from nativevoice.core.language import LanguageTag

    firsts, fulls, durations = [], [], []
    started = time.perf_counter()
    engine = SherpaKokoroEngine(
        manifest=registry_manifest,
        num_threads=threads,
        chunker=TextChunker(max_tokens=chunk_tokens),
    )
    engine.warm("en")
    load_s = time.perf_counter() - started
    for _ in range(repeats):
        span = time.perf_counter()
        first = None
        total_audio = 0.0
        for event in engine.stream(SynthesisRequest(text=TEXT_LONG, language=LanguageTag("en"))):
            if event.audio is None:
                continue
            if first is None:
                first = time.perf_counter() - span
            total_audio += event.audio.duration_s
        firsts.append(first if first is not None else time.perf_counter() - span)
        fulls.append(time.perf_counter() - span)
        durations.append(total_audio)
    engine.close()
    return {
        "threads": threads,
        "chunk_tokens": chunk_tokens,
        "load_seconds": round(load_s, 3),
        "first_audio_ms_median": round(median(firsts) * 1000, 1),
        "first_audio_ms_best": round(min(firsts) * 1000, 1),
        "utterance_wall_ms_median": round(median(fulls) * 1000, 1),
        "audio_seconds_per_utterance": median(durations),
        "rtf_median": median([wall / audio for wall, audio in zip(fulls, durations)]),
        "measure": "measured",
    }


def language_switch_cost(manifest):
    """Switching language mid-conversation rebuilds the engine; measure what that costs."""
    from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine

    engine = SherpaKokoroEngine(manifest=manifest, num_threads=4)
    engine.warm("en")
    started = time.perf_counter()
    try:
        engine.warm("zh")
        reload_s = time.perf_counter() - started
        ok = True
    except Exception as error:  # an unavailable language must fail, not hang
        reload_s = time.perf_counter() - started
        ok = False
        reason = f"{type(error).__name__}: {error}"
    engine.close()
    out = {"switch_en_to_zh_seconds": round(reload_s, 3), "succeeded": ok}
    if not ok:
        out["reason"] = reason
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "benchmarks" / "tts-latency-lab.json"))
    args = parser.parse_args(argv)

    from nativevoice.models.registry import ModelRegistry

    registry = ModelRegistry(models_dir())
    manifest = next(item for item in registry.all() if "kokoro" in item.model_id)
    status = registry.status(manifest.model_id)
    if not status.usable:
        print(f"refusing to benchmark an unverifiable asset: {status.explain()}")
        return 1

    try:
        import psutil

        context = {
            "free_ram_gb": round(psutil.virtual_memory().available / 2**30, 2),
            "processes": len(list(psutil.process_iter())),
        }
    except Exception:
        context = {"free_ram_gb": None, "processes": -1}
    context["logical_cpus"] = os.cpu_count()

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "asset": manifest.model_id,
        "context": context,
        "runtime_streaming_support": probe_engine_config(),
        "single_short_sentence_reference_ms": None,
        "thread_sweep": [],
        "chunk_sweep": [],
        "language_switch": language_switch_cost(manifest),
    }

    for threads in (1, 2, 4, 6, 8, 12):
        row = run(threads, 50, manifest, args.repeats)
        payload["thread_sweep"].append(row)
        print(f"  threads={threads:2d} first_audio={row['first_audio_ms_median']:8.1f}ms rtf={row['rtf_median']}", flush=True)

    best_threads = min(payload["thread_sweep"], key=lambda row: row["rtf_median"] or 9e9)["threads"]
    for tokens in (50, 24, 12):
        row = run(best_threads, tokens, manifest, args.repeats)
        payload["chunk_sweep"].append(row)
        print(f"  chunk<={tokens:2d} tokens first_audio={row['first_audio_ms_median']:8.1f}ms rtf={row['rtf_median']}", flush=True)

    from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine
    from nativevoice.tts.api import SynthesisRequest
    from nativevoice.core.language import LanguageTag

    engine = SherpaKokoroEngine(manifest=manifest, num_threads=best_threads)
    engine.warm("en")
    stamps = []
    for _ in range(args.repeats):
        span = time.perf_counter()
        outcome = engine.synthesize(SynthesisRequest(text=TEXT_SHORT, language=LanguageTag("en")))
        stamps.append((time.perf_counter() - span) / max(0.001, outcome.audio.duration_s))
    payload["single_short_sentence_reference_ms"] = round(min(stamps) * 1000, 1)
    payload["best_rtf_short_sentence"] = round(min(stamps), 4)
    engine.close()

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"artefact -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
