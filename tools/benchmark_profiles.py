"""Hardware-profile benchmarks: measure, then derive, never assert.

§10 forbids arbitrary RAM thresholds, and the reference stack studied here shipped
no local-model latency figure at all. So this runs the real engines once per profile
and writes down what the machine actually did - load time, first-chunk latency,
real-time factor, peak resident memory - and only then proposes ceilings, each of
which is a documented margin above an observed number rather than a guess.

The machine is not required to be idle, but it is required to be described: the
process count, free memory and thread settings are recorded beside every number,
because a benchmark without its conditions is an anecdote.

    python tools/benchmark_profiles.py [--profiles lite,balanced,performance]
"""

from __future__ import annotations

import argparse
import array
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nativevoice.audio import resample  # noqa: E402
from nativevoice.config import models_dir  # noqa: E402
from nativevoice.core.audio import AudioBuffer  # noqa: E402
from nativevoice.core.cancel import CancellationToken  # noqa: E402
from nativevoice.core.language import LanguageTag  # noqa: E402
from nativevoice.hardware import profiles as PROFILES  # noqa: E402
from nativevoice.models.registry import ModelRegistry  # noqa: E402
from nativevoice.stt.api import TranscriptionRequest  # noqa: E402
from nativevoice.tts.api import SynthesisRequest  # noqa: E402

SENTENCES = (
    "The voice engine is running locally on my machine.",
    "Please summarise the meeting notes and remind me about the four o'clock call.",
    "A quick brown fox jumps over the lazy dog while the rain falls softly outside.",
)

HEARDWARE_NOTE = "AMD64 / 16 logical cores / RTX 3050 4 GB / onnxruntime CPU-only"


def peak_rss_mb(sample_ms: float = 10.0):
    """Sampler that reports peak resident set in MiB for the current process."""
    state = {"peak": 0, "stop": False}

    def poll():
        try:
            import psutil

            process = psutil.Process()
            while not state["stop"]:
                state["peak"] = max(state["peak"], process.memory_info().rss)
                time.sleep(sample_ms / 1000)
        except Exception:
            while not state["stop"]:
                time.sleep(sample_ms / 1000)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    return state, thread


def background_processes() -> int:
    try:
        import psutil

        return len(list(psutil.process_iter()))
    except Exception:
        return -1


def measure_profile(name: str, registry: ModelRegistry, repeats: int) -> dict:
    from nativevoice.stt.adapters.faster_whisper_engine import FasterWhisperEngine
    from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine

    profile = PROFILES.get_profile(name)
    manifests = {manifest.model_id: manifest for manifest in registry.all()}
    tts_id = next(key for key in manifests if "kokoro" in key)
    stt_id = next(key for key in manifests if "whisper-base" in key)
    threads = profile.max_threads

    report: dict = {
        "profile": name,
        "threads": threads,
        "quantizations": list(profile.allowed_quantizations),
        "gpu_allowed": profile.allow_gpu,
        "buffer_frames": profile.max_audio_buffer_frames,
        "concurrency": profile.concurrency,
    }

    state, sampler = peak_rss_mb()
    started = time.perf_counter()
    engine = SherpaKokoroEngine(manifest=manifests[tts_id], num_threads=threads)
    engine.warm("en")
    report["tts_load_seconds"] = round(time.perf_counter() - started, 3)

    synthesised: AudioBuffer | None = None
    firsts, walls, durations = [], [], []
    for index in range(repeats):
        for sentence in SENTENCES:
            request = SynthesisRequest(text=sentence, language=LanguageTag("en"), token=CancellationToken())
            span = time.perf_counter()
            first = None
            for event in engine.stream(request):
                if event.audio is not None and first is None:
                    first = time.perf_counter() - span
                if event.audio is not None:
                    durations.append(event.audio.duration_s)
            walls.append(time.perf_counter() - span)
            firsts.append(first if first is not None else walls[-1])
            if synthesised is None:
                synthesised = engine.synthesize(request).audio
    report["tts_rtf_median"] = round(sorted(walls)[len(walls) // 2] / max(0.001, sum(durations) / len(durations)), 4)
    report["tts_first_chunk_ms_median"] = round(sorted(firsts)[len(firsts) // 2] * 1000, 1)
    report["tts_audio_seconds_total"] = round(sum(durations), 3)

    listener = FasterWhisperEngine(manifest=manifests[stt_id], cpu_threads=threads)
    load_started = time.perf_counter()
    listener.warm()
    report["stt_load_seconds"] = round(time.perf_counter() - load_started, 3)

    feed = resample(synthesised, 16000)
    heard_walls = []
    texts = []
    for _ in range(repeats):
        span = time.perf_counter()
        result = listener.transcribe(
            TranscriptionRequest(audio=feed, language=LanguageTag("en"), allow_auto_detection=False, max_duration_s=30.0)
        )
        heard_walls.append(time.perf_counter() - span)
        texts.append(result.text)
    report["stt_rtf_median"] = round(sorted(heard_walls)[len(heard_walls) // 2] / feed.duration_s, 4)
    report["stt_heard_example"] = texts[0]
    report["stt_exact_match_rate"] = round(sum(text == texts[0] for text in texts) / len(texts), 3)

    engine.close()
    listener.close()
    state["stop"] = True
    sampler.join(timeout=1.0)
    report["peak_rss_mb"] = round(state["peak"] / 2**20, 1) if state["peak"] else None
    return report


def derive(profile: dict, margin: float) -> dict:
    """Proposed ceilings, each a documented margin above an observed number."""
    observed_rss = profile.get("peak_rss_mb") or 0
    return {
        "max_resident_mib": int(-(-observed_rss * margin // 256) * 256) if observed_rss else None,
        "observed_rss_mb": observed_rss,
        "observed_tts_rtf": profile.get("tts_rtf_median"),
        "observed_stt_rtf": profile.get("stt_rtf_median"),
        "observed_first_chunk_ms": profile.get("tts_first_chunk_ms_median"),
        "headroom_factor": margin,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profiles", default="lite,balanced,performance")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "benchmarks" / "profiles.json"))
    args = parser.parse_args(argv)

    registry = ModelRegistry(models_dir())
    free_before = None
    try:
        import psutil

        free_before = round(psutil.virtual_memory().available / 2**30, 2)
    except Exception:
        pass

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": f"{platform.system()} {platform.release()} / python {platform.python_version()} / {HEARDWARE_NOTE}",
        "free_ram_gb_at_start": free_before,
        "background_processes": background_processes(),
        "logical_cpus": os.cpu_count(),
        "repeats_per_profile": args.repeats,
        "sentences": len(SENTENCES),
        "profiles": [],
    }
    for name in [item.strip() for item in args.profiles.split(",") if item.strip()]:
        print(f"benchmarking {name} ...", flush=True)
        profile = measure_profile(name, registry, args.repeats)
        profile["proposed_limits"] = derive(profile, args.margin)
        payload["profiles"].append(profile)
        print(
            f"  threads={profile['threads']} tts_rtf={profile['tts_rtf_median']} "
            f"first_chunk={profile['tts_first_chunk_ms_median']}ms stt_rtf={profile['stt_rtf_median']} "
            f"peak_rss={profile['peak_rss_mb']}MiB",
            flush=True,
        )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"artefact -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
