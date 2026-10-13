"""Compare the multilingual ASR passes side by side. Analysis only - it loads stored results.

Run after the passes:

    python tools/multilingual_compare.py

It exists separately because the passes must not share a process or overlap in time: a comparison that
re-ran inference would contaminate the latency it is supposed to be reporting. Two things it deliberately
refuses to do:

* **it does not compute WER or CER.** No sample has a verified reference transcript. The filename quotes are
  recomputed here with the fixed extractor as an indicative gross-error cue, labelled as such.
* **it does not treat an asserted language as measured support.** `expected_language` was inferred from the
  filename's script and speaker name, so agreement is a hint about the sample, not proof about the model.

Failure kinds are reported apart - decoding, preprocessing, segmentation structure, empty output, repeated
output, script mismatch and language assertion - because they have different owners and different fixes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from multilingual_eval import (  # noqa: E402
    PASSES,
    _guard_console,
    char_recall,
    cue_overlap,
    emit,
    quote_from_name,
    quote_tail_from_name,
)

EVAL = ROOT / "artifacts" / "evaluation"
PASSES_IN_ORDER = ("base-default", "base-int8", "large-v3-int8")


def script_of(text: str) -> str:
    """Which writing system the transcript came back in. A language claim and a script are not the same thing."""
    ranges = (
        ("devanagari", 0x0900, 0x097F), ("bengali", 0x0980, 0x09FF), ("arabic", 0x0600, 0x06FF),
        ("cjk", 0x4E00, 0x9FFF), ("hiragana-katakana", 0x3040, 0x30FF), ("hangul", 0xAC00, 0xD7AF),
        ("cyrillic", 0x0400, 0x04FF), ("latin", 0x0041, 0x007A),
    )
    tally: dict[str, int] = {}
    for char in text:
        code = ord(char)
        for name, low, high in ranges:
            if low <= code <= high:
                tally[name] = tally.get(name, 0) + 1
                break
    return max(tally, key=tally.get) if tally else "none"


def load(name: str) -> dict | None:
    path = EVAL / f"multilingual-{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    _guard_console()
    inventory = json.loads((EVAL / "multilingual-inventory.json").read_text(encoding="utf-8"))
    cues = {}
    for copy in inventory.get("working_copies", []):
        source = Path(copy["source"])
        cues[source.name] = {
            "words": quote_from_name(source.name),
            "tail": quote_tail_from_name(source.name),
            "expected": copy.get("expected_language", "unknown"),
            "category": copy.get("voice_category", "unknown"),
            "audio_seconds": copy["audio_seconds"],
            "decoded": copy.get("decoded"),
            "source_codec": copy.get("source_codec"),
            "source_rate": copy.get("source_rate"),
        }

    passes = {name: load(name) for name in PASSES_IN_ORDER}
    missing = [name for name, data in passes.items() if data is None]
    rows = []
    for sample, cue in cues.items():
        row = {
            "sample": sample,
            "expected_language_inferred": cue["expected"],
            "voice_category": cue["category"],
            "audio_seconds": cue["audio_seconds"],
            "reference_transcript": "none",
            "models": {},
        }
        for name, data in passes.items():
            if data is None:
                continue
            found = next((r for r in data["results"] if r["sample"] == sample), None)
            if found is None:
                row["models"][name] = {"status": "not run in this pass"}
                continue
            fixed_cue = cue_overlap(cue["words"], found["text"])
            row["models"][name] = {
                "status": "ok",
                "asserted_language": found["asserted_language"],
                "language_probability": found["language_probability"],
                "segments": found["segments"],
                "empty_transcript": found["empty_transcript"],
                "repeated_clause": found["repeated_clause"],
                "transcript_script": script_of(found["text"]),
                "real_time_factor": found["real_time_factor"],
                "wall_seconds": found["wall_seconds"],
                "peak_rss_mib": found["peak_rss_mib"],
                "first_segment_start": found["first_segment_start"],
                "last_segment_end": found["last_segment_end"],
                "cue_words_available": fixed_cue["filename_cue_words"],
                "cue_overlap_fixed": fixed_cue["filename_cue_overlap"],
                "cue_char_recall": char_recall(cue["tail"], found["text"]),
                "text_excerpt": found["text"][:90],
            }
        rows.append(row)

    summary = {"generated": json.loads((EVAL / "multilingual-inventory.json").read_text(encoding="utf-8"))["generated"],
               "passes_available": [n for n in PASSES_IN_ORDER if passes[n] is not None],
               "passes_missing": missing,
               "configurations": {name: {k: passes[name][k] for k in
                                         ("repo", "compute_type", "beam_size", "cpu_threads", "vad_filter",
                                          "model_load_seconds", "peak_rss_mib_whole_pass", "network_attempts",
                                          "production_configuration")}
                                  for name in PASSES_IN_ORDER if passes[name] is not None},
               "samples": rows,
               "caveats": [
                   "no verified reference transcript exists for any sample, so no WER or CER is reported",
                   "expected_language is inferred from filename script and speaker name, not verified",
                   "language_probability is the model's own confidence, not a measured accuracy",
                   "large-v3 ran while physical free RAM was below its resident size; check the memory note",
                   "cue_char_recall is a longest-common-subsequence check against a truncated filename "
                   "quote: it detects gross error and script choice, and is not a word or character error rate",
               ]}
    out = EVAL / "multilingual-comparison.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    emit(f"comparison -> {out}")

    for row in rows:
        line = f"{row['sample'][:38]:40s} {row['expected_language_inferred']:8s}"
        for name in summary["passes_available"]:
            model = row["models"].get(name, {})
            if model.get("status") != "ok":
                line += f" {name[:9]:9s}=--"
                continue
            line += (f" {name[:9]:9s}={str(model['asserted_language']):4s}/"
                     f"{model['transcript_script'][:5]:5s}"
                     f" cr{model['cue_char_recall'] if model['cue_char_recall'] is not None else -1:.2f}"
                     f"{' E' if model['empty_transcript'] else ''}"
                     f"{' R' if model['repeated_clause'] else ''}"
                     f" rt{model['real_time_factor']:.2f}")
        emit(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
