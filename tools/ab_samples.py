"""Matched A/B listening samples: incumbent Kokoro against the Kitten trial voice.

Run:

    python tools/ab_samples.py

Both engines render the **same sentences at the same speed and sample rate**, so the only difference a
listener can hear is the model and its voice, not the plumbing. Files land in `artifacts/ab/` as plain
16-bit WAVs - double-click them in Explorer, or use the printed per-pair command - because a listening
comparison should not need a toolchain to operate.

The sentences are chosen to expose the things a WER number cannot see: punctuation handling, a question,
numbers and an abbreviation, a long utterance where prosody drift shows up, and a short imperative where
clipping or a click at the edge would be audible.

Nothing here promotes anything: the candidate is read from `.trials/kitten/`, the default engine and the
shared voice catalogue are untouched, and the manifest written alongside the audio records which slot and
which licence each clip came from so a listening judgement can be traced back to a model.
"""

from __future__ import annotations

import array
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nativevoice.audio import measure, resample, write_wav  # noqa: E402
from nativevoice.config import models_dir  # noqa: E402
from nativevoice.core.audio import AudioFormat  # noqa: E402
from nativevoice.core.language import LanguageTag  # noqa: E402
from nativevoice.models.registry import ModelRegistry  # noqa: E402
from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine  # noqa: E402
from nativevoice.tts.api import SynthesisRequest  # noqa: E402

OUT = ROOT / "artifacts" / "ab"

SENTENCES = (
    ("punctuation-list", "Stop! Don't go out without your keys, will you?"),
    ("question", "Could you tell me the time and the nearest train station?"),
    ("numbers-abbreviation", "The meeting is at 9:30 a.m. on 5 March, room 204, not 240."),
    ("long-utterance", (
        "The quick brown fox jumps over the lazy dog, and then it walks slowly home through the "
        "quiet park, while the rain falls softly on the empty streets and the lights go out one by one."
    )),
    ("short-imperative", "Turn it off now."),
    ("names-and-places", "Anita and Ravi live near the harbour, beside the old bridge in Kochi."),
)

# Playback order is declared, never derived from filenames: sorting `01_*.wav` alphabetically would put
# the trial candidate first and bias every pair toward whatever was heard first.
PLAYBACK_ORDER = ("kokoro", "kitten")
LABELS = {
    "kokoro": "A  Kokoro (incumbent default)",
    "kitten": "B  KittenTTS (isolated trial)",
}


def pair_clips(directory: Path, pair: int) -> list[tuple[str, Path]]:
    """The clips of one pair, in the order a listener must hear them: incumbent first, candidate second."""
    ordered: list[tuple[str, Path]] = []
    for engine in PLAYBACK_ORDER:
        found = sorted(directory.glob(f"{pair:02d}_{engine}_*.wav"))
        if found:
            ordered.append((engine, found[0]))
    return ordered


def render_kokoro(manifest, text: str) -> tuple[object, float]:
    engine = SherpaKokoroEngine(manifest=manifest, num_threads=4)
    engine.warm("en")
    started = time.perf_counter()
    outcome = engine.synthesize(SynthesisRequest(text=text, language=LanguageTag("en")))
    wall = time.perf_counter() - started
    engine.close()
    return outcome.audio, wall


def render_kitten(manifest, text: str) -> tuple[object, float]:
    sys.path.insert(0, str(ROOT / "tools"))
    from tts_head_to_head import KittenTrialEngine

    engine = KittenTrialEngine(manifest=manifest, num_threads=4)
    engine.warm()
    started = time.perf_counter()
    buffer = engine.generate(text)
    wall = time.perf_counter() - started
    engine.close()
    return buffer, wall


def main() -> int:
    incumbent = ModelRegistry(models_dir())
    trial = ModelRegistry(ROOT / ".trials" / "kitten" / "models")
    kokoro = next(item for item in incumbent.all() if "kokoro" in item.model_id)
    kitten = next(item for item in trial.all() if "kitten" in item.model_id)

    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.wav"):
        stale.unlink()

    rows = []
    for index, (slug, text) in enumerate(SENTENCES, 1):
        entry: dict = {"pair": index, "slug": slug, "text": text}
        for label, renderer, manifest, voice_note in (
            ("kokoro", render_kokoro, kokoro, "slot 0 (asset gives no voice names)"),
            ("kitten", render_kitten, kitten, "slot 0 = 'Bella' per the upstream card"),
        ):
            audio, wall = renderer(manifest, text)
            play = resample(audio, 24000) if audio.format.sample_rate != 24000 else audio
            path = OUT / f"{index:02d}_{label}_{slug}.wav"
            write_wav(str(path), play)
            level = measure(play)
            entry[label] = {
                "file": path.name,
                "seconds": round(play.duration_s, 3),
                "wall_seconds": round(wall, 3),
                "rtf": round(wall / max(0.001, play.duration_s), 3),
                "peak": round(level.peak, 4),
                "rms": round(level.rms, 4),
                "clipped_fraction": round(level.clipped_fraction, 5),
                "model": manifest.model_id,
                "licence": manifest.provenance.licence_spdx,
                "voice": voice_note,
            }
            print(f"  {index:02d} {label:7s} {play.duration_s:5.2f}s audio in {wall:5.2f}s -> {path.name}")
        rows.append(entry)

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "human listening comparison; not a quality metric",
        "identical_conditions": {
            "text": "same sentences for both engines",
            "speed": "1.0 for both",
            "output_sample_rate": 24000,
            "threads": 4,
            "chain": "same normalisation and WAVE writer for both",
        },
        "engines": {
            "kokoro": {"model": kokoro.model_id, "licence": kokoro.provenance.licence_spdx, "languages": list(kokoro.languages)},
            "kitten": {"model": kitten.model_id, "licence": kitten.provenance.licence_spdx, "languages": list(kitten.languages)},
        },
        "pairs": rows,
        "status": "trial evidence only; default engine and shared catalogue unchanged",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} matched pairs -> {OUT}")
    print("Play one pair back-to-back, incumbent first, then the trial voice:")
    print("  python tools/ab_samples.py --play 01      (pair numbers 01 .. %02d, one pair per call)" % len(SENTENCES))
    return 0


def play(pair: int) -> int:
    import sounddevice as sd

    from nativevoice.audio import read_wav

    clips = pair_clips(OUT, pair)
    if not clips:
        print(f"no clips for pair {pair:02d}")
        return 1
    for engine, path in clips:
        buffer = read_wav(path)
        print(f"{LABELS[engine]}  [{path.name}, {buffer.duration_s:.2f}s]", flush=True)
        with sd.RawOutputStream(samplerate=buffer.format.sample_rate, channels=buffer.format.channels, dtype="int16") as stream:
            for chunk in buffer.frames(2400):
                stream.write(chunk.to_bytes())
        time.sleep(0.4)
    return 0


def play_argument(argv: list[str]) -> int:
    """The pair number after `--play`, or a refusal. There is no all-pairs switch to advertise."""
    position = argv.index("--play") + 1
    if position >= len(argv):
        raise ValueError("--play needs a pair number, for example: --play 1")
    try:
        return int(argv[position])
    except ValueError:
        raise ValueError(
            f"--play takes a pair number 01 .. {len(SENTENCES):02d}, not {argv[position]!r}"
        ) from None


if __name__ == "__main__":
    if "--play" in sys.argv:
        try:
            pair = play_argument(sys.argv)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        raise SystemExit(play(pair))
    raise SystemExit(main())
