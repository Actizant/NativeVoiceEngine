"""The comparator reads finished pass records, and the record format changed under it.

No inference, no real weights, no model load. Every record here is a dict written to a
temporary directory; the unit under test is *which field a timing number may be read
from, and what the comparator is allowed to print when a field is not there*.

The regression these tests exist for: `tools/multilingual_eval.py` began writing
`engine_warm_seconds` when passes moved behind the measurement seam, while
`tools/multilingual_compare.py` kept reading `model_load_seconds`. The stale artefacts on
disk still carried the old name, so the mismatch stayed invisible - and the two numbers
do not measure the same span. A legacy `model_load_seconds` timed the model constructor
alone; `engine_warm_seconds` times approval, digest verification, the backend import and
that constructor together. Copying one into the other's slot would have been worse than
the `KeyError`: a 0.6-second constructor would have read as a verified warm-up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import tools.multilingual_compare as mc  # noqa: E402
from nativevoice.core.errors import ConfigurationError  # noqa: E402

SAMPLE = "sample-one.wav"
TEXT = "the quick brown fox"


def record_row(sample: str = SAMPLE, text: str = TEXT) -> dict:
    """One finished sample row. Identical under both schemas - only the pass timing moved."""
    return {
        "sample": sample,
        "expected_language": "en",
        "voice_category": "synthetic",
        "audio_seconds": 2.0,
        "wall_seconds": 0.4,
        "real_time_factor": 0.2,
        "asserted_language": "en",
        "language_probability": 0.9,
        "expected_language_asserted": True,
        "segments": 1,
        "first_segment_start": 0.0,
        "last_segment_end": 2.0,
        "silent_tail_seconds": 0.0,
        "empty_transcript": False,
        "repeated_clause": False,
        "peak_rss_mib": 120.0,
        "text": text,
    }


def current_record(name: str = "large-v3-int8", **overrides) -> dict:
    """The schema `pass_record` plus `run_pass` write now: warm span, scope metadata."""
    record = {
        "pass": name,
        "model_id": "faster-whisper-large-v3",
        "engine": "faster-whisper",
        "repo": "Systran/faster-whisper-large-v3",
        "revision": "e" * 40,
        "serve_scope": "measurement",
        "review_status": "unreviewed",
        "compute_type": "int8",
        "beam_size": 5,
        "cpu_threads": 4,
        "vad_filter": False,
        "local_files_only": True,
        "production_configuration": False,
        "engine_warm_seconds": 12.5,
        "peak_rss_mib_whole_pass": 1800.0,
        "network_attempts": [],
        "results": [record_row()],
    }
    record.update(overrides)
    return record


def legacy_record(name: str = "base-default", **overrides) -> dict:
    """The schema already on disk: constructor-only timing, no scope metadata, and the
    `production_configuration` flag the old pass table carried per name."""
    record = {
        "pass": name,
        "repo": "Systran/faster-whisper-base",
        "snapshot": "models--Systran--faster-whisper-base/snapshots/aaa",
        "compute_type": "default",
        "beam_size": 5,
        "cpu_threads": 4,
        "vad_filter": False,
        "local_files_only": True,
        "production_configuration": True,
        "model_load_seconds": 0.612,
        "peak_rss_mib_whole_pass": 400.0,
        "network_attempts": [],
        "results": [record_row()],
    }
    record.update(overrides)
    return record


def evaluation_tree(tmp_path: Path, *records: dict, sample: str = SAMPLE) -> Path:
    """A synthetic `artifacts/evaluation` holding these records and a matching inventory.

    `sample` names the working copy in the inventory, so a test that wants the comparator's
    filename-cue wiring exercised passes a stamped name; the default unstamped one is what
    makes the timing tests independent of cue extraction.
    """
    out = tmp_path / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    inventory = {"generated": "2026-10-14T00:00:00",
                 "working_copies": [{"source": str(Path("/synthetic") / sample),
                                     "audio_seconds": 2.0,
                                     "expected_language": "en",
                                     "voice_category": "synthetic",
                                     "decoded": True,
                                     "source_codec": "pcm_s16le",
                                     "source_rate": 16000}]}
    (out / "multilingual-inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    for record in records:
        (out / f"multilingual-{record['pass']}.json").write_text(json.dumps(record), encoding="utf-8")
    return out


# --- the timing field, read by name --------------------------------------------------

def test_a_current_record_is_read_from_the_warm_field_it_was_written_to():
    config = mc.pass_configuration("large-v3-int8", current_record())
    assert config["engine_warm_seconds"] == 12.5
    assert config["legacy_model_load_seconds"] is None
    assert "engine.warm()" in config["timing_basis"]
    assert "integrity digest" in config["timing_basis"]


def test_a_legacy_record_keeps_its_own_name_and_its_own_slot():
    """Historical compatibility, stated rather than absorbed.

    The value survives, the field it came from is said out loud, and `engine_warm_seconds`
    is None instead of a constructor time wearing a warm-up's label.
    """
    config = mc.pass_configuration("base-default", legacy_record())
    assert config["engine_warm_seconds"] is None
    assert config["legacy_model_load_seconds"] == 0.612
    assert config["timing_basis"] == mc.TIMING_BASIS[mc.LEGACY_LOAD_FIELD]
    for said in ("legacy", "constructor", "not comparable"):
        assert said in config["timing_basis"]


def test_a_record_carrying_both_names_reports_both_instead_of_choosing_silently():
    """The rename boundary: nothing is dropped just because one field won the basis."""
    both = current_record(model_load_seconds=0.612)
    config = mc.pass_configuration("large-v3-int8", both)
    assert config["engine_warm_seconds"] == 12.5
    assert config["legacy_model_load_seconds"] == 0.612
    assert config["timing_basis"] == mc.TIMING_BASIS[mc.ENGINE_WARM_FIELD]


# --- what a missing field does -------------------------------------------------------

def test_a_record_with_no_timing_field_at_all_is_refused_by_name(monkeypatch, tmp_path):
    """No `KeyError`, and no invented zero: the artefact and both accepted names are printed.

    The stripped record still has every other field, so the refusal cannot be about
    anything except the missing span.
    """
    monkeypatch.setattr(mc, "EVAL", evaluation_tree(tmp_path))
    bare = {k: v for k, v in current_record().items() if k not in mc.TIMING_BASIS}
    with pytest.raises(ConfigurationError) as exc:
        mc.pass_configuration("large-v3-int8", bare)
    assert exc.value.details["artefact"].endswith("multilingual-large-v3-int8.json")
    assert exc.value.details["missing"] == [f"{mc.ENGINE_WARM_FIELD} (current) or {mc.LEGACY_LOAD_FIELD} (legacy)"]
    assert exc.value.details["remedy"]


def test_an_interrupted_record_names_every_field_it_still_lacks(monkeypatch, tmp_path):
    """One refusal per artefact, not one per missing field.

    A partial record is a real shape here: `_write_partial` runs per sample, while the
    whole-pass peak and the network attempt list are only attached when the pass ends.
    """
    monkeypatch.setattr(mc, "EVAL", evaluation_tree(tmp_path))
    partial = current_record()
    del partial["engine_warm_seconds"], partial["peak_rss_mib_whole_pass"], partial["network_attempts"]
    with pytest.raises(ConfigurationError) as exc:
        mc.pass_configuration("large-v3-int8", partial)
    assert set(exc.value.details["missing"]) == {
        "peak_rss_mib_whole_pass", "network_attempts",
        f"{mc.ENGINE_WARM_FIELD} (current) or {mc.LEGACY_LOAD_FIELD} (legacy)",
    }


def test_the_run_refuses_before_any_comparison_is_written(monkeypatch, tmp_path, capsys):
    """Exit 1, a stated reason, and no half-written document for someone else to quote.

    The output file is pre-filled with a sentinel: the refusal branch returns before the
    write, so the sentinel still being there proves validation came first rather than
    after the comparison had already been published.
    """
    out = evaluation_tree(tmp_path, {k: v for k, v in current_record().items() if k not in mc.TIMING_BASIS})
    monkeypatch.setattr(mc, "EVAL", out)
    monkeypatch.setattr(mc, "_guard_console", lambda: None)
    (out / "multilingual-comparison.json").write_text("PRE-EXISTING", encoding="utf-8")
    assert mc.main() == 1
    assert "REFUSED" in capsys.readouterr().out
    assert (out / "multilingual-comparison.json").read_text(encoding="utf-8") == "PRE-EXISTING"


# --- the whole comparison, both schemas present --------------------------------------

def test_a_finished_pair_writes_each_schema_under_its_own_timing(monkeypatch, tmp_path, capsys):
    """A legacy pass and a current pass side by side, neither one's number moved into the
    other's field."""
    out = evaluation_tree(tmp_path, current_record(), legacy_record("base-int8", production_configuration=False))
    monkeypatch.setattr(mc, "EVAL", out)
    monkeypatch.setattr(mc, "_guard_console", lambda: None)
    assert mc.main() == 0
    summary = json.loads((out / "multilingual-comparison.json").read_text(encoding="utf-8"))
    warm = summary["configurations"]["large-v3-int8"]
    old = summary["configurations"]["base-int8"]
    assert (warm["engine_warm_seconds"], warm["legacy_model_load_seconds"]) == (12.5, None)
    assert (old["engine_warm_seconds"], old["legacy_model_load_seconds"]) == (None, 0.612)
    assert summary["generated"] == "2026-10-14T00:00:00"
    assert summary["passes_available"] == ["base-int8", "large-v3-int8"]
    assert summary["passes_missing"] == ["base-default"]
    assert any("legacy_model_load_seconds" in caveat for caveat in summary["caveats"])
    row = summary["samples"][0]["models"]["large-v3-int8"]
    assert row["status"] == "ok" and row["text_excerpt"] == TEXT


def test_the_comparator_wires_the_filename_cue_into_every_sample_row(monkeypatch, tmp_path):
    """The row fields must come from the fixed extractor, not from the speaker prefix.

    `Hello there friend` is the quoted tail of this synthetic filename; `Recorder` and the
    timestamp digits are not part of the cue, because the extractor starts reading after the
    stamp. Each expectation is derived by hand rather than copied from a run:

    * words - `Hello`, `there`, `friend`: three tokens longer than two characters, with the
      bare digit groups of the timestamp dropped, so `cue_words_available` is 3.
    * overlap - against `Hello there stranger`, `hello` and `there` reappear and `friend`
      does not: 2 of 3, rounded to 0.667.
    * recall - cue `Hellotherefriend` (16 characters) against `Hellotherestranger`: the
      common subsequence is the 10-character `Hellothere` prefix plus two more (`re`, or
      `rn`), so 12/16 = 0.75.
    * script - latin characters and nothing else.
    """
    source = "Recorder-2026-10-13-18-26-Hello there friend.mp3"
    out = evaluation_tree(tmp_path, current_record(
        results=[record_row(source, "Hello there stranger")]), sample=source)
    monkeypatch.setattr(mc, "EVAL", out)
    monkeypatch.setattr(mc, "_guard_console", lambda: None)
    assert mc.main() == 0
    summary = json.loads((out / "multilingual-comparison.json").read_text(encoding="utf-8"))
    assert [row["sample"] for row in summary["samples"]] == [source]
    row = summary["samples"][0]["models"]["large-v3-int8"]
    assert row["cue_words_available"] == 3
    assert row["cue_overlap_fixed"] == 0.667
    assert row["cue_char_recall"] == 0.75
    assert row["transcript_script"] == "latin"


def test_an_unstamped_filename_produces_no_cue_fields_rather_than_zeroes(monkeypatch, tmp_path):
    """A missing cue stays `None`: a zero would read as a measured gross-error rate of none."""
    out = evaluation_tree(tmp_path, current_record())
    monkeypatch.setattr(mc, "EVAL", out)
    monkeypatch.setattr(mc, "_guard_console", lambda: None)
    assert mc.main() == 0
    summary = json.loads((out / "multilingual-comparison.json").read_text(encoding="utf-8"))
    row = summary["samples"][0]["models"]["large-v3-int8"]
    assert row["cue_words_available"] == 0
    assert row["cue_overlap_fixed"] is None
    assert row["cue_char_recall"] is None


def test_a_legacy_product_claim_is_carried_verbatim_and_called_out(monkeypatch, tmp_path):
    """The comparator neither launders nor rewrites what a run said about itself.

    `production_configuration: true` stays exactly as the record wrote it - fixing history
    is not this tool's job - but naming it in the caveats keeps a fresh comparison
    document from lending a stale flag the authority of a current measurement.
    """
    out = evaluation_tree(tmp_path, current_record(), legacy_record())
    monkeypatch.setattr(mc, "EVAL", out)
    monkeypatch.setattr(mc, "_guard_console", lambda: None)
    assert mc.main() == 0
    summary = json.loads((out / "multilingual-comparison.json").read_text(encoding="utf-8"))
    assert summary["configurations"]["base-default"]["production_configuration"] is True
    assert summary["configurations"]["large-v3-int8"]["production_configuration"] is False
    named = [c for c in summary["caveats"] if "production_configuration true" in c]
    assert len(named) == 1 and "base-default" in named[0]
    assert "large-v3-int8" not in named[0]


def test_a_set_of_records_that_all_denies_being_the_product_path_adds_no_claim(monkeypatch, tmp_path):
    """The caveat is computed from the data, not a standing sentence about old artefacts."""
    out = evaluation_tree(tmp_path, current_record(), legacy_record("base-int8", production_configuration=False))
    monkeypatch.setattr(mc, "EVAL", out)
    monkeypatch.setattr(mc, "_guard_console", lambda: None)
    assert mc.main() == 0
    summary = json.loads((out / "multilingual-comparison.json").read_text(encoding="utf-8"))
    assert not [c for c in summary["caveats"] if "production_configuration true" in c]


# --- the writer and the reader, held together ----------------------------------------

def test_the_writer_and_the_reader_name_the_same_field():
    """Drift guard, because the defect was two files agreeing on nothing.

    If the evaluation tool renames its timing again, the comparator's constant stops
    matching its source and this fails at once - instead of the mismatch waiting for a
    pass to finish before it surfaces as a `KeyError`.
    """
    writer = (ROOT / "tools" / "multilingual_eval.py").read_text(encoding="utf-8")
    assert f'record["{mc.ENGINE_WARM_FIELD}"] =' in writer
    assert f'record["{mc.LEGACY_LOAD_FIELD}"]' not in writer
    assert mc.LEGACY_LOAD_FIELD in mc.TIMING_BASIS and mc.ENGINE_WARM_FIELD in mc.TIMING_BASIS


def test_the_comparator_no_longer_indexes_a_record_field_it_has_not_checked():
    """The bare `passes[name][k]` read was how the rename became an unexplained crash."""
    source = (ROOT / "tools" / "multilingual_compare.py").read_text(encoding="utf-8")
    assert "passes[name][" not in source
    assert "pass_configuration(" in source
