"""Isolated multilingual ASR comparison over user-supplied voice samples.

Run:

    python tools/multilingual_eval.py --pass base-default
    python tools/multilingual_eval.py --pass base-int8
    python tools/multilingual_eval.py --pass large-v3-int8

Why this is a separate tool rather than a run of the platform: the objective is to measure recognition
across languages under **equivalent** configuration, and the only way to keep one model's compute type from
confounding the comparison is to state it per pass. `base-default` mirrors the settings the production
adapter uses; `base-int8` and `large-v3-int8` share one quantisation so the two differ by model size and
nothing else. Mirroring those settings is not the same as measuring the shipped path, and no pass here does
that: every pass loads from the measurement registry, so `production_configuration` is false on all of them.

Ground truth: **there are none.** The samples carry no verified reference transcripts, and a filename that
appears to quote the utterance is marketing text, not an annotation. So this tool never computes WER or CER.
It reports transcribed text, segment structure, empty and repeated output, the language the model asserted,
and a clearly-labelled *filename cue overlap* - the share of the filename's words that also appear in the
transcript. That number is a gross-error detector. It is not accuracy, and it is not a quality score.

Boundaries kept by construction: the model comes only from
:func:`nativevoice.models.measurement.load` over :func:`~nativevoice.models.measurement.registry`, so a pass
that has no authorised measurement manifest is refused before any audio is read, and the scope guard, the
named authoriser, the provenance record, the asset-root rule and the integrity verifier all run ahead of the
weights. This tool does not import the backend, does not construct ``WhisperModel`` itself, and does not use
the manifest-free adapter - that path fabricates a product-scoped manifest with no assets and no provenance,
so nothing in it would be verified. The revision is pinned: the pass's own pin comes from the provisioner's
catalogue, and ``snapshot_files`` yields nothing for a revision that is not on disk, so a missing snapshot is
a refusal rather than a quieter one for whatever newest directory the cache happens to hold. Everything runs
under `OfflineGuard` with local-only loading, so no model is fetched and no connection is attempted; the
source files are opened read-only and hashed before and after; conversion writes isolated copies outside the
repository; nothing here registers a manifest, changes a default or touches the production engine.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from nativevoice.audio import read_wav, write_wav  # noqa: E402
from nativevoice.config.paths import hf_hub_cache_root  # noqa: E402
from nativevoice.core.audio import AudioBuffer, AudioFormat  # noqa: E402
from nativevoice.core.errors import ConfigurationError, VoiceError  # noqa: E402
from nativevoice.diagnostics.evidence_paths import write_json  # noqa: E402
from nativevoice.diagnostics.offline import OfflineGuard  # noqa: E402
from nativevoice.models import measurement  # noqa: E402
from nativevoice.stt.api import TranscriptionRequest  # noqa: E402
from tools.provision import CATALOGUE, hf_snapshot_dir, snapshot_files  # noqa: E402

DOWNLOADS = Path(os.path.expanduser("~")) / "Downloads"
WORKSPACE = Path(os.environ.get("ML_EVAL_WORK", str(ROOT.parent / "_multilingual_eval")))
OUT = ROOT / "artifacts" / "evaluation"
BEAM_SIZE = 5  # the production adapter's default
CPU_THREADS = 4  # the production default for interactive work on this machine

SAMPLES = [
    ("Energetic-Male", "en", "English male voice"),
    ("Paula", "en", "English female voice"),
    ("黄丽萍长自述1", "zh", "Chinese"),
    ("5.28达人", "de", "German"),
    ("Voz-Femenina-Español", "es", "Spanish"),
    ("元気な女性", "ja", "Japanese"),
    ("Foot", "fr", "French"),
    ("보이스1", "ko", "Korean"),
    ("بنت-سورية-77-الف", "ar", "Arabic"),
    ("Рената-Литвинова", "ru", "Russian"),
    ("Matilde", "pt", "Portuguese"),
    ("Ultron-2026-10-04-10-20", "hi", "Hindi"),
    ("voice_preview_ultron (2).wav", "unknown", "identity to be determined"),
]

#: One entry per pass. `model_id` names the manifest this tool loads from the measurement
#: registry, and `repo`/`revision` are read from the provisioner's own catalogue so a pin
#: has exactly one home - the one its digests were verified against. `settings_basis` says
#: what a pass is equivalent to, which is a statement about settings and not about the
#: shipped path: no pass here measures the shipped path, because none of them can load
#: anything but a measurement-scoped manifest.
def _pass(model_id: str, compute_type: str, settings_basis: str) -> dict:
    spec = CATALOGUE[model_id]
    return {
        "model_id": model_id,
        "repo": spec["hf_repo"],
        "revision": str(spec.get("revision") or ""),
        "compute_type": compute_type,
        "settings_basis": settings_basis,
    }


PASSES = {
    "base-default": _pass(
        "faster-whisper-base", "default", "the settings the production adapter uses: beam 5, no VAD, the publisher's default quantisation"
    ),
    "base-int8": _pass(
        "faster-whisper-base", "int8", "the same model as base-default at the quantisation the large-v3 pass uses"
    ),
    "large-v3-int8": _pass(
        "faster-whisper-large-v3", "int8", "the selected teacher, one quantisation shared with base-int8 so the two differ by model size and nothing else"
    ),
}


def manifest_revision(manifest) -> str:
    """The snapshot the manifest's own recorded paths name, or "" when they name none.

    The provisioner records each asset as ``snapshots/<revision>/<file>``, so the digest
    run says which directory the bytes were verified in. That is a recorded fact about
    this manifest, and it is what an unpinned pass falls back on - not a directory scan.
    """
    for asset in manifest.assets:
        parts = Path(asset.relative_path).parts
        if len(parts) >= 3 and parts[0] == "snapshots":
            return parts[1]
    return ""


def resolve_revision(spec: dict, manifest) -> dict:
    """Pin this pass to one revision, and refuse when the cache does not hold it.

    Two pins are compared before anything is loaded: the catalogue's and the manifest's.
    If they disagree, the manifest was digested from a different commit than the pass
    claims to measure, and the run would produce a number attributed to the wrong file.
    Then the pinned snapshot is asked for by name: `snapshot_files` returns nothing for a
    revision that is not on disk, which is the refusal this tool needs in place of the
    directory scan it used to do, where the first sorted snapshot holding a `model.bin`
    silently decided which model was measured.
    """
    pin = str(spec.get("revision") or "")
    recorded = manifest_revision(manifest)
    if pin and recorded and pin != recorded:
        raise ConfigurationError(
            "the manifest records a different snapshot than this pass is pinned to",
            model_id=manifest.model_id,
            pinned=pin,
            recorded=recorded,
            remedy="provision the pinned revision into the measurement registry, or point this pass at the manifest that digested it",
        )
    revision = pin or recorded
    if not revision:
        raise ConfigurationError(
            "neither this pass nor its manifest names a revision",
            model_id=manifest.model_id,
            remedy="pin the catalogue entry, or re-provision so the manifest records the snapshot it digested",
        )
    present = snapshot_files(spec["repo"], revision)
    if not present:
        raise ConfigurationError(
            "no cached snapshot for the revision this pass is pinned to; this tool does not download and will not substitute another snapshot",
            repo=spec["repo"],
            revision=revision,
            cache_root=str(hf_hub_cache_root()),
        )
    return {
        "revision": revision,
        "revision_basis": (
            "pinned by the provisioner's catalogue and matched by the manifest's recorded paths"
            if pin and recorded
            else "pinned by the provisioner's catalogue"
            if pin
            else "the snapshot the manifest recorded when it was digested"
        ),
        "repository": str(hf_snapshot_dir(spec["repo"])),
        "snapshot_files_present": len(present),
    }


def load_pass(name: str):
    """The one engine this tool will run, with its pin resolved: no other route exists.

    :func:`nativevoice.models.measurement.load` is the only way a model reaches this
    process, so the registry scope, the manifest's measurement scope, the named
    authoriser, the provenance record and the asset-root rule are all settled before the
    object exists, and the integrity verifier plus the resolved-directory guard settle
    the bytes when it warms. Constructing it reads no weights: the refusal a missing or
    unauthorised manifest produces here happens before a single sample is opened.
    """
    spec = PASSES[name]
    engine = measurement.load(
        measurement.registry(),
        spec["model_id"],
        compute_type=spec["compute_type"],
        beam_size=BEAM_SIZE,
        cpu_threads=CPU_THREADS,
        vad_filter=False,
    )
    return engine, resolve_revision(spec, engine.manifest)


QUOTE_STAMP = __import__("re").compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-")

# Punctuation a filename may quote and a transcript may restyle. The ASCII quote and apostrophe are added
# by code point so this literal does not have to escape them.
CUE_SKIP_CHARACTERS = frozenset("_.!?,()[]{}-–—«»“”‘’؛،،:;") | {chr(34), chr(39)}


def quote_from_name(name: str) -> list[str]:
    """The spoken sentence some filenames quote, after their timestamp. A truncated cue, not a reference.

    Filenames from this source platform append the first words of the clip, which is marketing text cut
    mid-sentence - useful for spotting a transcript that is in the wrong language entirely, and worthless
    as a reference for accuracy.
    """
    import re

    stem = Path(name).stem
    match = QUOTE_STAMP.search(stem)
    if not match:
        return []
    tail = stem[match.end():]
    return [word for word in re.split(r"[-_,.!?;: ()]+", tail) if len(word) > 2 and not word.isdigit()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _guard_console() -> None:
    """Same guard `real_speech_check` needed: a cp1252 console cannot print CJK, Arabic or Cyrillic.

    The defect matters more here than anywhere else in this project - the evaluation set is defined by
    filenames in exactly those scripts, so an unguarded print would abort the run over a *label*.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass


def emit(line: str = "") -> None:
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        try:
            print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
        except (UnicodeEncodeError, ValueError, OSError):
            pass
    except (ValueError, OSError):
        pass


def find_sources() -> list[dict]:
    """Match each expected sample to its attachment by prefix, read-only, and record its digest."""
    found = []
    for prefix, expected, category in SAMPLES:
        hits = sorted(p for p in DOWNLOADS.iterdir() if p.name.startswith(prefix) and p.suffix.lower() in (".mp3", ".wav"))
        if prefix.endswith(".wav"):
            hits = [p for p in hits if p.suffix.lower() == ".wav"]
        else:
            hits = [p for p in hits if p.suffix.lower() == ".mp3"]
        found.append({
            "prefix": prefix,
            "expected_language": expected,
            "voice_category": category,
            "paths": [str(p) for p in hits],
            "accessible": bool(hits),
            "sha256_before": [sha256(p) for p in hits],
            "bytes": [p.stat().st_size for p in hits],
            "filename_words": quote_from_name(hits[0].name) if hits else [],
            "expected_language_basis": "inferred from the filename script and speaker name - not verified",
            "reference_transcript": "none - no verified reference exists for any sample",
        })
    return found


def to_working_copy(path: Path) -> dict:
    """Decode to an isolated 16 kHz mono copy. The source is only ever read.

    Decoding and resampling are done by ffmpeg through PyAV, identically for every file and every model, so
    a preprocessing difference can never be mistaken for a recognition difference. ffmpeg is also the
    component that handles planar multichannel layouts correctly - flattening raw frames by hand would have
    interleaved them wrongly.
    """
    import av
    import numpy as np

    info: dict = {"source": str(path), "decoded": False}
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"no audio stream in {path.name}")
        info["source_codec"] = stream.codec_context.name
        info["source_rate"] = stream.sample_rate
        info["source_channels"] = stream.channels
        layout = getattr(stream.codec_context, "layout", None)
        info["source_layout"] = str(layout.name) if layout is not None else "unknown"
        info["declared_duration_s"] = (round(float(container.duration / av.time_base), 3)
                                        if container.duration else None)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        info["decode_error"] = "decoded no audio samples"
        return info
    flat = np.concatenate(chunks).astype(np.int16)
    samples = array.array("h", flat.tobytes())
    working = AudioBuffer(samples, AudioFormat(16000, 1))
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    target = WORKSPACE / f"{path.stem[:60].replace(' ', '_')}.16k.wav"
    write_wav(str(target), working)
    info.update({
        "decoded": True,
        "working_copy": str(target),
        "working_frames": working.frame_count,
        "audio_seconds": round(working.duration_s, 3),
        "working_sha256": sha256(target),
        "source_sha256": sha256(path),
    })
    return info


class PeakMemory:
    """Sample this process's RSS from a thread while a call runs."""

    def __init__(self, interval: float = 0.05):
        import psutil

        self._process = psutil.Process()
        self._interval = interval
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        def poll():
            while not self._stop.is_set():
                self._peak = max(self._peak, self._process.memory_info().rss)
                time.sleep(self._interval)

        self._peak = self._process.memory_info().rss
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._peak = max(self._peak, self._process.memory_info().rss)

    @property
    def peak_mib(self) -> float:
        return round(self._peak / 2**20, 1)


def quote_tail_from_name(name: str) -> str:
    """The raw quoted tail of a filename, before any splitting. Used by the character-level check."""
    import re

    stem = Path(name).stem
    match = QUOTE_STAMP.search(stem)
    return stem[match.end():] if match else ""


def char_recall(cue: str, text: str) -> float | None:
    """Longest-common-subsequence coverage of a cue by the transcript, for scripts without spaces.

    A CJK or Japanese filename arrives as one 40-character "word", so the word-substring test scores 0.0
    against a transcript that is in fact verbatim (finding A48). A longest common subsequence tolerates the
    punctuation a model inserts or drops. It is still not accuracy and still not a reference - it asks only
    how much of the cued wording is present.

    The subsequence is computed properly, by dynamic programming. A greedy single pass "advance on the first
    match" is not a longest common subsequence and collapses on the first inserted word: it scored the
    verbatim Chinese transcript at 0.04 because the cue opens with 你们 and the model said 你知道吗.
    """
    cue_chars = [ch for ch in cue if not ch.isspace() and ch not in CUE_SKIP_CHARACTERS]
    target = [ch for ch in text if not ch.isspace() and ch not in CUE_SKIP_CHARACTERS]
    if not cue_chars or not target:
        return None
    previous = [0] * (len(target) + 1)
    for cue_char in cue_chars:
        current = [0]
        for index, target_char in enumerate(target):
            current.append(previous[index] + 1 if cue_char == target_char else max(current[index], previous[index + 1]))
        previous = current
    return round(previous[-1] / len(cue_chars), 4)


def cue_overlap(words: list[str], text: str) -> dict:
    """How much of the filename's wording reappears in the transcript. Indicative only - never a score.

    A48: unreliable for unspaced scripts, where the whole sentence becomes one word. `char_recall` is the
    measure to read for Chinese and Japanese.
    """
    lowered = text.lower()
    matched = [w for w in words if w.lower().strip(",.!?") in lowered]
    return {
        "filename_cue_words": len(words),
        "filename_cue_matched": len(matched),
        "filename_cue_overlap": round(len(matched) / len(words), 3) if words else None,
        "note": "indicative gross-error check only; filenames are not annotated references",
    }


def repeated(text: str) -> bool:
    """Whisper's signature failure: the same clause looped."""
    parts = [p.strip().lower() for p in text.replace("，", ",").split(",") if p.strip()]
    if len(parts) < 3:
        return False
    return len({*parts}) < len(parts) * 0.6


def pass_record(name: str, engine, pin: dict) -> dict:
    """The skeleton every artefact carries: which model, which revision, which scope.

    Read from the engine and the manifest rather than from `PASSES` alone, so a record
    cannot describe a configuration the object that ran did not have.
    `production_configuration` is false unconditionally and says why: the only route to a
    model here is the measurement registry, so nothing this tool produces measures the
    shipped path, whatever settings a pass mirrors.
    """
    manifest = engine.manifest
    weights = Path(pin["repository"]) / "snapshots" / pin["revision"] / "model.bin"
    return {
        "pass": name,
        "model_id": manifest.model_id,
        "engine": engine.engine_id,
        "repo": PASSES[name]["repo"],
        "hub_cache_root": str(hf_hub_cache_root()),
        "repository": pin["repository"],
        "revision": pin["revision"],
        "revision_basis": pin["revision_basis"],
        "snapshot_recorded": f"snapshots/{pin['revision']}",
        "snapshot_files_present": pin["snapshot_files_present"],
        "snapshot_model_bin_bytes": weights.stat().st_size if weights.is_file() else None,
        "asset_root": manifest.asset_root,
        "serve_scope": manifest.use_scope.value,
        "review_status": manifest.review.value,
        "compute_type": engine.compute_type,
        "beam_size": engine.beam_size,
        "cpu_threads": engine.cpu_threads,
        "vad_filter": engine.vad_filter,
        "local_files_only": True,
        "production_configuration": False,
        "production_configuration_basis": (
            "loaded through nativevoice.models.measurement from a measurement-scoped manifest, so no line in "
            "this file measures the shipped path; " + PASSES[name]["settings_basis"]
        ),
        "results": [],
    }


def run_pass(name: str, sources: list[dict], engine, pin: dict) -> dict:
    record = pass_record(name, engine, pin)
    guard = OfflineGuard()
    with guard:
        with PeakMemory() as peak:
            started = time.perf_counter()
            # `warm` is the gate, the digests, the backend import and the model constructor
            # together, so it is named for what it times rather than as a load time.
            engine.warm()
            record["engine_warm_seconds"] = round(time.perf_counter() - started, 3)
            for info in sources:
                audio = read_wav(info["working_copy"])
                wall = time.perf_counter()
                result = engine.transcribe(TranscriptionRequest(audio=audio))
                elapsed = time.perf_counter() - wall
                segments = list(result.segments)
                text = result.text
                asserted = result.language.code if result.language else ""
                row = {
                    "sample": Path(info["source"]).name,
                    "expected_language": info["expected_language"],
                    "voice_category": info["voice_category"],
                    "audio_seconds": info["audio_seconds"],
                    "wall_seconds": round(elapsed, 3),
                    "real_time_factor": round(elapsed / max(0.001, info["audio_seconds"]), 4),
                    "asserted_language": asserted,
                    "language_probability": round(float(result.timings.get("language_probability") or 0.0), 4),
                    "expected_language_asserted": asserted == info["expected_language"],
                    "segments": len(segments),
                    "first_segment_start": round(float(segments[0].metadata["start"]), 3) if segments else None,
                    "last_segment_end": round(float(segments[-1].metadata["end"]), 3) if segments else None,
                    "silent_tail_seconds": (round(info["audio_seconds"] - float(segments[-1].metadata["end"]), 3)
                                            if segments else None),
                    "empty_transcript": not text,
                    "repeated_clause": repeated(text),
                    "peak_rss_mib": peak.peak_mib,
                    "text": text,
                }
                row.update(cue_overlap(info["filename_words"], text))
                record["results"].append(row)
                _write_partial(name, record)
        record["peak_rss_mib_whole_pass"] = peak.peak_mib
    record["network_attempts"] = [f"{a.host}:{a.port}" for a in guard.attempts]
    return record


def _write_partial(name: str, record: dict) -> None:
    write_json(OUT / f"multilingual-{name}.json", record)


def _write_inventory(inventory: dict) -> Path:
    """Serialize the inventory as committed evidence.

    `paths`, `source`, `working_copy`, `downloads` and `workspace` are absolute here and
    stay absolute in memory - the pass opens the working copy by the value in this very
    dict - so the redaction belongs to this write and nowhere else.
    """
    return write_json(OUT / "multilingual-inventory.json", inventory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pass", dest="pass_name", choices=sorted(PASSES), required=True)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args(argv)
    _guard_console()

    if not args.inventory_only:
        try:
            # Resolved before a single sample is opened: a pass with no authorised
            # measurement manifest, or without its pinned snapshot, is refused here
            # rather than after the audio has been decoded into the workspace. This call
            # builds the gated object and reads no weights.
            engine, pin = load_pass(args.pass_name)
        except VoiceError as error:
            emit(f"REFUSED: {error}")
            emit("no pass ran: this tool loads a model only through the measurement registry")
            return 1

    sources_meta = find_sources()
    inventory = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "downloads": str(DOWNLOADS),
                 "workspace": str(WORKSPACE), "samples": sources_meta}
    accessible = [s for s in sources_meta if s["accessible"]]
    OUT.mkdir(parents=True, exist_ok=True)
    if len(accessible) != len(SAMPLES):
        missing = [s["prefix"] for s in sources_meta if not s["accessible"]]
        inventory["missing"] = missing
        _write_inventory(inventory)
        emit(f"NOT ACCESSIBLE: {missing}")
        emit("stopping: the evaluation set is incomplete and no substitute sample may be invented")
        return 1

    prepared = []
    for sample in accessible:
        path = Path(sample["paths"][0])
        info = to_working_copy(path)
        info.update({"expected_language": sample["expected_language"],
                     "voice_category": sample["voice_category"],
                     "filename_words": sample["filename_words"]})
        prepared.append(info)
        emit(f"prepared {Path(info['source']).name[:46]:48s} {info['audio_seconds']:6.2f}s "
             f"{info['source_codec']} {info['source_rate']}Hz ch={info['source_channels']}")

    inventory["working_copies"] = prepared
    inventory["total_audio_seconds"] = round(sum(p["audio_seconds"] for p in prepared), 2)
    _write_inventory(inventory)
    if args.inventory_only:
        emit(f"inventory only -> {OUT / 'multilingual-inventory.json'}")
        return 0

    record = run_pass(args.pass_name, prepared, engine, pin)
    write_json(OUT / f"multilingual-{args.pass_name}.json", record)

    after = find_sources()
    tampered = [s["prefix"] for s, b in zip(after, sources_meta) if s["sha256_before"] != b["sha256_before"]]
    emit(f"sources unchanged: {not tampered} | tampered: {tampered or 'none'}")
    emit(f"pass {args.pass_name}: {record['model_id']}@{record['revision'][:12]} "
         f"warm={record['engine_warm_seconds']}s "
         f"peak={record['peak_rss_mib_whole_pass']} MiB network={len(record['network_attempts'])} "
         f"-> {OUT / ('multilingual-' + args.pass_name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
