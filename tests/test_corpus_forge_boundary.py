"""The wall between training text and evaluation text, tested without an engine.

No inference, no weights, no download, no asset gate, and nothing here may construct a
synthesiser or a transcriber: every test below runs on text, on the manifest builder, or on
a `main()` that is asserted to refuse *before* it reaches a model.

This file exists because the Legacy Dataset Isolation Audit of 2026-10-15 measured a leak
rather than imagined one. Of the sixteen sentences `tools/corpus_forge.py` forged
`artifacts/corpus/en-synthetic-01` from, eight are the same example as one of the eight
English sentences the round-trip evaluation scores - one identical once punctuation and
case are set aside, seven near-duplicates that differ by an added clause or a changed tail.
Four of those eight were *kept* clips, including `en_slot0_006.wav`, whose sentence is
character-for-character the evaluation sentence "Could you tell me the time and the nearest
train station?". A model trained on that clip and then scored on that evaluation is not
being evaluated, and nothing in the repository could see this, because both lists were
hand-written in different files and nothing compared them.

The audit also recorded why the exclusion cannot be a deletion: removing `artifacts/evaluation/
multilingual-*.json` would remove no legacy identifier, since the same thirteen third-party
speaker names are hard-coded in tracked source. So the contract here is declarative and
enforced at generation time - named excluded inputs, a documented collision rule, and a
`main()` that refuses - rather than a path list someone might read as cleanup.

Every expected value is derived by hand from the rule in `DISJOINTNESS_RULE`, not read back
out of `collisions()`.

The owner's follow-up of the same date (P1-P4) moved the gate from "the manifest declares X" to
"X is verified against this code and against the tracked register in
`artifacts/corpus/legacy-batches.json`": a boundary-contract header bound to the live pools, a
named list of pre-contract batches matched by the digest of the file as read from disk, a closed
`usable_for` vocabulary, and quarantine flags demoted from evidence to records.
`corpus_training_eligible()` is the single named door a future trainer must walk through, and
there is no trainer: `TestTheTrainingEntryPoint` pins that door's contract on a temporary
filesystem, which is a tested agreement rather than runtime enforcement.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "src"), str(ROOT / "tools")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from tools import corpus_forge as forge  # noqa: E402

CORPUS_MANIFEST = ROOT / "artifacts" / "corpus" / "en-synthetic-01" / "manifest.json"

#: The clip the audit named, and the sentence it repeats from the evaluation.
COLLIDING_CLIP = "en_slot0_006.wav"
COLLIDING_SENTENCE = "Could you tell me the time and the nearest train station?"

#: Every kept clip in the committed corpus whose text cannot be separated from an evaluation
#: sentence. Four of the eight kept clips: this is the leak, pinned by name so it cannot be
#: quietly re-added to the training pool by someone reading only the sentence list.
LEAKED_CLIPS = {
    "en_slot0_001.wav",
    "en_slot0_002.wav",
    "en_slot0_003.wav",
    COLLIDING_CLIP,
}

EXCLUDED = [
    "artifacts/evaluation/multilingual-inventory.json",
    "artifacts/evaluation/multilingual-base-default.json",
    "artifacts/evaluation/multilingual-base-int8.json",
    "artifacts/evaluation/multilingual-large-v3-int8.json",
    "artifacts/evaluation/multilingual-comparison.json",
]


def corpus_samples() -> list[dict]:
    """The committed corpus descriptor, read read-only. Never written, never regenerated."""
    return json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))["samples"]


def evaluate_english_sentences() -> list[str]:
    """The evaluation's English sentences, read from `tools/evaluate.py` by parsing it.

    Imported as text on purpose: `tools/evaluate.py` pulls the synthesis and recognition
    adapters in at module scope, so importing it to read a sentence list would drag two
    engines into a test that must not touch either.
    """
    tree = ast.parse((ROOT / "tools" / "evaluate.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "CORPUS":
            for key, value in zip(node.value.keys, node.value.values):
                if key.value == "en":
                    return [element.value for element in value.elts]
    raise AssertionError("tools/evaluate.py no longer names an English CORPUS list")


def stub_manifest(model_id: str, engine: str, licence_spdx: str) -> SimpleNamespace:
    return SimpleNamespace(model_id=model_id, engine=engine, provenance=SimpleNamespace(licence_spdx=licence_spdx))


#: A kept sentence of the quarantined batch that is clean against the evaluation pool. It is
#: the text a gate that only checks train/eval disjointness would happily re-accept, so the
#: reuse check is tested against it rather than against the already-known collision.
QUARANTINED_SENTENCE = "Silence in the archive allowed her to read the letter twice."


#: A sentence that is clean against the evaluation pool and against every quarantined record,
#: and the pool built by swapping it in for one live line. Used to show what an *unregistered*
#: pool does: it is not a leak, it is simply not something a reviewer approved.
NOVEL_SENTENCE = "The cartographer redrew the coastline after every storm."
NOVEL_TRAINING_POOL = tuple(
    NOVEL_SENTENCE if sentence.startswith("The museum charged") else sentence for sentence in forge.TRAIN_SENTENCES
)


def clean_training_manifest(name: str = "en-synthetic-02", texts=None) -> dict:
    """A batch that satisfies every declared field, built by the project's own builder.

    One sentence per training-pool line, all kept: the counts, clip names and digests are the
    ones the builder produces, so a blocker raised here can only come from the gate and not
    from a hand-written fixture that forgot a field.
    """
    sentences = list(texts if texts is not None else forge.TRAIN_SENTENCES)
    return forge.build_corpus_manifest(
        name=name,
        rows=[row(index, index, text, kept=True) for index, text in enumerate(sentences)],
        wall_seconds=24.0,
        slots=list(range(len(sentences))),
        tts_manifest=stub_manifest("kokoro-int8-multi-lang-v1_1", "sherpa-onnx-kokoro", "Apache-2.0"),
        stt_manifest=stub_manifest("faster-whisper-base", "faster-whisper", "MIT"),
        max_wer=0.0,
    )


def recorded_digests(manifest: dict) -> dict[str, str]:
    """The digests a caller would hand back after reading the audio, i.e. the manifest's own.

    Tolerant of a fixture the mutation tests have already damaged: it answers for whatever
    clip names survive, so the blocker under test is the one the mutation introduced.
    """
    rows = manifest.get("samples")
    if not isinstance(rows, list):
        return {}
    return {
        sample["clip"]: sample["sha256"]
        for sample in rows
        if isinstance(sample, dict) and isinstance(sample.get("clip"), str) and sample["clip"]
    }


def corpus_manifest() -> dict:
    """The committed quarantined batch descriptor, read read-only."""
    return json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))


#: The register the repository ships, and the pool digest it pins. Tests that need an
#: ineligible-by-construction register write their own copy under `tmp_path`; tests that ask
#: "does the shipped evidence hold up" read this one, because that is the question an owner
#: can actually review in a diff.
REPO_REGISTER = ROOT / "artifacts" / "corpus" / "legacy-batches.json"


def live_pool_sha() -> str:
    return forge.pool_set_sha256(forge.TRAIN_SENTENCES, forge.EVAL_SENTENCES)


def pool_entry(**overrides) -> dict:
    entry = {
        "pool_set_sha256": live_pool_sha(),
        "boundary_contract_version": forge.SUPPORTED_BOUNDARY_VERSION,
        "disjointness_rule": forge.DISJOINTNESS_RULE,
        "training_sentences": len(forge.TRAIN_SENTENCES),
        "evaluation_sentences": len(forge.EVAL_SENTENCES),
        "issued": "2026-10-15",
        "ruling": "fixture",
    }
    entry.update(overrides)
    return entry


def batch_entry(name: str = "en-synthetic-01", manifest_sha256: str = "ab" * 32, status: str = "QUARANTINED", **overrides) -> dict:
    entry = {
        "name": name,
        "manifest_sha256": manifest_sha256,
        "status": status,
        "ruling_date": "2026-10-15",
        "reason": "fixture",
    }
    entry.update(overrides)
    return entry


def write_register(directory: Path, *, pool_sets=None, batches=None) -> Path:
    """A register in a temporary directory: the shape the gate reads, without touching the tracked one."""
    document = {
        "schema_version": forge.REGISTER_SCHEMA_VERSION,
        "pool_sets": [pool_entry()] if pool_sets is None else list(pool_sets),
        "pre_contract_batches": [] if batches is None else list(batches),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "legacy-batches.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def missing_field(entry: dict, field: str) -> dict:
    """A register record that is complete except for one required field."""
    entry.pop(field)
    return entry


def register_document(pool_sets, pre_contract_batches) -> str:
    """Register text for the damage cases, built from complete fixtures with one field removed.

    A case that breaks several requirements at once passes against any validator, so each entry
    here is a well-formed record minus the single field whose absence the register must refuse.
    """
    return json.dumps(
        {
            "schema_version": forge.REGISTER_SCHEMA_VERSION,
            "pool_sets": pool_sets,
            "pre_contract_batches": pre_contract_batches,
        }
    )


def register_batch_of(path: Path, index: int = 0) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    return document["pre_contract_batches"][index]


def manifest_digest_of(manifest_path: Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def clips_on_disk(manifest: dict, directory: Path) -> dict[str, str]:
    """Write one file per clip and re-point the manifest's digests at the bytes just written.

    The gate asks a caller for digests read from audio, so a fixture that wants to be accepted
    has to have audio. Each clip's contents are derived from its own name, which makes a
    single-byte edit to any file a digest mismatch rather than a coincidence.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for sample in manifest["samples"]:
        payload = f"wav:{sample['clip']}".encode("utf-8")
        (directory / sample["clip"]).write_bytes(payload)
        sample["sha256"] = hashlib.sha256(payload).hexdigest()
    return {sample["clip"]: sample["sha256"] for sample in manifest["samples"]}


def publish(manifest: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def row(index: int, slot: int, text: str, *, kept: bool) -> dict:
    heard = text if kept else text.replace("archive", "arrive")
    return {
        "index": index,
        "slot": slot,
        "text": text,
        "heard": heard,
        "normalised_reference": forge.normalise(text),
        "normalised_heard": forge.normalise(heard),
        "wer": 0.0 if kept else 0.125,
        "cer": 0.0 if kept else 0.03,
        "exact_match": kept,
        "kept": kept,
        "clip": f"en_slot{slot}_{index:03d}.wav" if kept else "",
        "sha256": "ab" * 32 if kept else "",
        "seconds": 3.0,
        "sample_rate": 16000,
        "status": "ok",
    }


class TestTheKnownCollision:
    """The specific leak the audit measured, asserted against the committed corpus."""

    def test_the_committed_corpus_names_the_colliding_clip_and_its_sentence(self):
        rows = {sample["clip"]: sample["text"] for sample in corpus_samples()}
        assert rows[COLLIDING_CLIP] == COLLIDING_SENTENCE

    def test_the_colliding_sentence_belongs_to_the_evaluation_pool(self):
        assert COLLIDING_SENTENCE in forge.EVAL_SENTENCES

    def test_the_colliding_sentence_is_no_longer_a_training_sentence(self):
        assert COLLIDING_SENTENCE not in forge.TRAIN_SENTENCES
        assert all(forge.normalise(s) != forge.normalise(COLLIDING_SENTENCE) for s in forge.TRAIN_SENTENCES)

    def test_the_check_reports_it_as_the_same_sentence_rather_than_a_near_duplicate(self):
        breaches = forge.collisions([COLLIDING_SENTENCE], forge.EVAL_SENTENCES)
        assert len(breaches) == 1, "one evaluation sentence matches, and it matches exactly"
        candidate, scored, reason = breaches[0]
        assert candidate == COLLIDING_SENTENCE and scored == COLLIDING_SENTENCE
        assert reason == forge.IDENTICAL_REASON

    def test_every_leaked_kept_clip_in_the_committed_corpus_is_out_of_the_training_pool(self):
        leaked = {
            sample["clip"]
            for sample in corpus_samples()
            if forge.collisions([sample["text"]], forge.EVAL_SENTENCES)
        }
        assert leaked == LEAKED_CLIPS
        for clip in sorted(leaked):
            text = next(sample["text"] for sample in corpus_samples() if sample["clip"] == clip)
            assert text not in forge.TRAIN_SENTENCES, f"{clip} would re-open the leak the audit found"


class TestNormalisationReachesPastPunctuation:
    """A sentence leaks through a comma, a capital or a double space just as easily as a
    copy, and those are the variants that hid the original collision from a diff."""

    def test_case_punctuation_and_spacing_variants_are_the_same_sentence(self):
        assert forge.collisions(["Hello, there!"], ["hello THERE"]) == [
            ("Hello, there!", "hello THERE", forge.IDENTICAL_REASON)
        ]

    def test_a_repeated_space_or_a_tab_is_not_a_different_sentence(self):
        assert forge.collisions(["one  two\tthree"], ["one two three"]) != []

    def test_an_apostrophe_and_a_trailing_period_are_ignored(self):
        assert forge.collisions(["I would like a slice of banana bread."], ["i would like a slice of banana bread"]) != []

    def test_reordering_the_words_still_collides_because_the_vocabulary_is_identical(self):
        # Word order is not a new sentence for a model that has memorised the words.
        breaches = forge.collisions(["the cat sat on the mat"], ["mat the on cat sat"])
        assert len(breaches) == 1
        assert "contains" in breaches[0][2]

    def test_genuinely_different_sentences_do_not_collide(self):
        assert forge.collisions(["Silence in the archive allowed her to read the letter twice."], ["The train departed late."]) == []

    def test_an_empty_sides_sentence_collides_only_with_itself_and_crashes_on_neither(self):
        assert forge.collisions([""], [""]) == [("", "", forge.IDENTICAL_REASON)]
        assert forge.collisions([""], ["the cat sat on the mat"]) == []
        assert forge.near_duplicate("", "") == ""


class TestTheNearDuplicateRule:
    """Three mechanisms, documented in `DISJOINTNESS_RULE`, each covering what the others miss."""

    def test_word_set_containment_is_a_collision(self):
        breaches = forge.collisions(["the cat sat on the mat"], ["the cat sat on a warm mat"])
        assert len(breaches) == 1
        assert breaches[0][2] == "one sentence's word set contains the other's"

    def test_shared_vocabulary_at_the_documented_threshold_is_a_collision(self):
        # 9 shared words out of an 11-word union is 0.818, so this is the bar being met, and the
        # openings differ, which means the leading-word rule cannot be what fires here.
        first = "one two three four five six seven eight nine ten"
        second = "nine ten three four five six seven eight two alpha"
        breaches = forge.collisions([first], [second])
        assert len(breaches) == 1
        assert breaches[0][2].startswith("word sets share 0.82")

    def test_shared_vocabulary_below_the_threshold_is_not_a_collision(self):
        first = "one two three four five six seven eight nine ten"
        second = "nine ten three four five six seven alpha beta gamma"
        assert forge.collisions([first], [second]) == []

    def test_a_shared_leading_phrase_with_a_divergent_tail_is_a_collision(self):
        # The historical pair. Containment misses it, because each side has words the other
        # lacks, and Jaccard misses it at 0.67 - which is exactly why the leading-word test exists.
        training = "My favourite book is about a young engineer building a bridge."
        evaluation = "My favourite book is about a young engineer in the city."
        breaches = forge.collisions([training], [evaluation])
        assert len(breaches) == 1
        assert breaches[0][2] == "8 leading words are identical, bar is 5"

    def test_the_rule_is_written_down_in_the_form_the_code_applies(self):
        rule = forge.DISJOINTNESS_RULE
        for mechanism in ("identical", "containing", "Jaccard >= 0.8", "5 leading words", "normalisation"):
            assert mechanism in rule, f"{mechanism} is enforced but not documented"

    def test_the_two_thresholds_are_the_documented_ones(self):
        assert forge.NEAR_DUPLICATE_JACCARD == 0.8
        assert forge.SHARED_PREFIX_TOKENS == 5


class TestTheLivePools:
    def test_the_shipped_pools_are_disjoint(self):
        assert forge.collisions(forge.TRAIN_SENTENCES, forge.EVAL_SENTENCES) == []

    def test_both_pools_are_populated_and_internally_clean(self):
        for pool in (forge.TRAIN_SENTENCES, forge.EVAL_SENTENCES):
            assert len(pool) == 8
            assert len({forge.normalise(sentence) for sentence in pool}) == len(pool)

    def test_the_evaluation_pool_is_the_english_evaluation_of_record(self):
        # Drift guard. The two lists live in different files, and the leak began exactly there:
        # a paraphrase in one file was invisible to the other.
        assert list(forge.EVAL_SENTENCES) == evaluate_english_sentences()

    def test_no_training_sentence_is_a_relabelled_evaluation_sentence(self):
        evaluation = {forge.normalise(sentence) for sentence in forge.EVAL_SENTENCES}
        assert evaluation.isdisjoint({forge.normalise(sentence) for sentence in forge.TRAIN_SENTENCES})


class TestGenerationRefuses:
    """A guard that only exists in a comment is not a guard: these run `main()` and stop it."""

    @pytest.fixture(autouse=True)
    def no_model_may_be_reached(self, monkeypatch):
        def explode(*_args, **_kwargs):
            raise AssertionError("the boundary check let the run reach a model registry")

        monkeypatch.setattr(forge, "ModelRegistry", explode)

    def test_a_colliding_training_pool_refuses_the_run_and_writes_nothing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(forge, "TRAIN_SENTENCES", (COLLIDING_SENTENCE,))
        assert forge.main(["--out", str(tmp_path / "corpus")]) == 2
        captured = capsys.readouterr()
        assert "not disjoint" in captured.err
        assert "training:   " + COLLIDING_SENTENCE in captured.err
        assert "evaluation: " + COLLIDING_SENTENCE in captured.err
        assert not (tmp_path / "corpus").exists()

    def test_disjoint_pools_are_not_a_refusal_and_still_reach_no_model(self, tmp_path):
        with pytest.raises(AssertionError, match="model registry"):
            forge.main(["--out", str(tmp_path / "corpus")])

    def test_a_pool_that_reuses_quarantined_corpus_text_refuses_the_run(self, tmp_path, monkeypatch, capsys):
        # Decision 4 at generation time: the quarantined batch's sentences are excluded from
        # the pool even where the evaluation pool is clean, which is the case
        # QUARANTINED_SENTENCE is in.
        monkeypatch.setattr(forge, "TRAIN_SENTENCES", (QUARANTINED_SENTENCE,))
        assert forge.main(["--out", str(tmp_path / "corpus")]) == 2
        captured = capsys.readouterr()
        assert "reuses text from a batch that is not cleared for training" in captured.err
        assert QUARANTINED_SENTENCE in captured.err
        assert not (tmp_path / "corpus").exists()

    def test_an_unreadable_quarantine_record_refuses_the_run_rather_than_guessing(self, tmp_path, monkeypatch, capsys):
        def explode(**_kwargs):
            raise forge.CorpusBoundaryError("cannot read the corpus record at …")

        monkeypatch.setattr(forge, "quarantined_corpus_texts", explode)
        assert forge.main(["--out", str(tmp_path / "corpus")]) == 2
        captured = capsys.readouterr()
        assert "quarantine records are unreadable" in captured.err
        assert not (tmp_path / "corpus").exists()

    def test_a_pool_the_register_does_not_record_refuses_the_run(self, tmp_path, monkeypatch, capsys):
        # Editing `TRAIN_SENTENCES` is not the same as approving new text. The register has to
        # name the pool set the manifest will be stamped with, so a batch cannot authorise
        # itself by being generated from whatever the code happens to hold today.
        assert NOVEL_TRAINING_POOL != forge.TRAIN_SENTENCES
        assert forge.collisions(NOVEL_TRAINING_POOL, list(forge.EVAL_SENTENCES)) == []
        assert forge.collisions(NOVEL_TRAINING_POOL, list(forge.quarantined_corpus_texts())) == []
        monkeypatch.setattr(forge, "TRAIN_SENTENCES", NOVEL_TRAINING_POOL)
        assert forge.main(["--out", str(tmp_path / "corpus")]) == 2
        captured = capsys.readouterr()
        assert "not recorded in artifacts/corpus/legacy-batches.json" in captured.err
        assert not (tmp_path / "corpus").exists()

    def test_a_register_that_cannot_be_read_refuses_the_run(self, tmp_path, monkeypatch, capsys):
        def explode(*_args, **_kwargs):
            raise forge.CorpusBoundaryError("cannot read the boundary register at …")

        monkeypatch.setattr(forge, "boundary_register", explode)
        assert forge.main(["--out", str(tmp_path / "corpus")]) == 2
        captured = capsys.readouterr()
        assert "no pool set is authorised" in captured.err
        assert not (tmp_path / "corpus").exists()

    def test_a_pool_recorded_under_a_different_rule_refuses_the_run(self, tmp_path, monkeypatch, capsys):
        # The rule travels with the pool set in the register: re-tightening or loosening
        # `DISJOINTNESS_RULE` without saying so in the tracked file is refused here as well as
        # at the gate.
        doctored = {"pool_sets": [pool_entry(disjointness_rule="identical sentences only")], "pre_contract_batches": []}
        monkeypatch.setattr(forge, "boundary_register", lambda *_a, **_k: doctored)
        assert forge.main(["--out", str(tmp_path / "corpus")]) == 2
        captured = capsys.readouterr()
        assert "disjointness rule are not recorded" in captured.err
        assert not (tmp_path / "corpus").exists()

    def test_a_pool_recorded_with_the_wrong_sizes_refuses_the_run(self, tmp_path, monkeypatch, capsys):
        doctored = {"pool_sets": [pool_entry(training_sentences=800)], "pre_contract_batches": []}
        monkeypatch.setattr(forge, "boundary_register", lambda *_a, **_k: doctored)
        assert forge.main(["--out", str(tmp_path / "corpus")]) == 2
        captured = capsys.readouterr()
        assert "do not match the pools in code" in captured.err
        assert not (tmp_path / "corpus").exists()

    def test_asking_for_teacher_targets_refuses_with_the_rights_that_are_missing(self, tmp_path, capsys):
        assert forge.main(["--out", str(tmp_path / "corpus"), "--teacher-targets"]) == 2
        captured = capsys.readouterr()
        assert "NOASSERTION" in captured.err and "redistributable=False" in captured.err
        assert "not treat this refusal as a licence finding" in captured.err
        assert not (tmp_path / "corpus").exists()

    def test_the_refusal_is_returned_rather_than_raised(self, tmp_path):
        # The tool is a command: a refusal must exit non-zero and stay legible, not traceback.
        assert forge.main(["--out", str(tmp_path / "corpus"), "--teacher-targets"]) == 2


class TestTheManifestContract:
    """What a future training tool reads off the artefact, asserted on the built manifest."""

    @pytest.fixture()
    def manifest(self) -> dict:
        return forge.build_corpus_manifest(
            name="en-synthetic-02",
            rows=[
                row(0, 0, "Silence in the archive allowed her to read the letter twice.", kept=True),
                row(1, 7, "The library closes at eight and returns must be made by then.", kept=False),
            ],
            wall_seconds=12.3456,
            slots=[0, 7],
            tts_manifest=stub_manifest("kokoro-int8-multi-lang-v1_1", "sherpa-onnx-kokoro", "Apache-2.0"),
            stt_manifest=stub_manifest("faster-whisper-base", "faster-whisper", "MIT"),
            max_wer=0.0,
        )

    def test_the_manifest_names_the_legacy_evaluation_records_as_excluded_inputs(self, manifest):
        boundary = manifest["training_boundary"]
        assert boundary["excluded_inputs"] == EXCLUDED
        assert boundary["excluded_inputs"] == list(forge.LEGACY_EVAL_ONLY_INPUTS)

    def test_every_excluded_input_is_a_real_evaluation_record_and_not_an_audit_file(self):
        for path in EXCLUDED:
            assert (ROOT / path).is_file(), f"{path} is named as evidence but does not exist"
            assert path.startswith("artifacts/evaluation/")
        assert "artifacts/audit/offline-proof.json" not in forge.LEGACY_EVAL_ONLY_INPUTS

    def test_teacher_targets_are_recorded_as_blocked_not_absent(self, manifest):
        targets = manifest["training_boundary"]["teacher_targets"]
        assert targets["enabled"] is False
        assert targets["status"] == "BLOCKED"
        assert len(targets["blocked_until"]) == 2
        assert "NOASSERTION" in targets["refusal"]

    def test_the_manifest_states_the_rule_it_was_checked_under(self, manifest):
        boundary = manifest["training_boundary"]
        assert boundary["disjointness_rule"] == forge.DISJOINTNESS_RULE
        assert boundary["sentence_pools"] == {"training": 8, "evaluation": 8, "collisions_found": 0}
        assert manifest["split"] == "training"

    def test_the_scope_fields_the_audit_relied_on_are_unchanged(self, manifest):
        # What this corpus may be used for predates the boundary work, and licensing reports
        # quote these strings, so they are pinned rather than assumed stable.
        assert manifest["usable_for"] == ["training", "adaptation", "asr-domain-evaluation"]
        assert manifest["not_usable_for"] == ["human speech quality claims", "intelligibility scoring of real speech"]
        assert manifest["licence_basis"].startswith("Audio derived from an Apache-2.0 pretrained checkpoint")
        assert manifest["consent_basis"].startswith("not applicable: no human speaker")
        assert manifest["synthetic"] is True

    def test_the_run_bookkeeping_survived_the_extraction(self, manifest):
        assert manifest["counts"] == {"generated": 2, "kept": 1, "dropped": 1}
        assert [sample["clip"] for sample in manifest["samples"]] == ["en_slot0_000.wav"]
        assert manifest["dropped"][0]["clip"] == "en_slot7_001.wav"
        assert manifest["generator"]["slots"] == [0, 7]
        assert manifest["generator"]["licence"] == "Apache-2.0"
        assert manifest["verifier"]["policy"] == "clip kept only when round-trip WER <= 0.0"
        assert manifest["wall_seconds"] == 12.35

    def test_the_contract_is_serialisable_as_written(self, manifest):
        assert json.loads(json.dumps(manifest, ensure_ascii=False)) == manifest

    def test_two_built_manifests_do_not_share_the_teacher_target_record(self):
        # Found by the gate's tests, not by intent: the boundary block was taken from a
        # module-level constant by reference, so an edit to one manifest's `enabled` flag
        # changed every other manifest built afterwards - including the refusal a trainer
        # reads. A record that gates training must be per-batch.
        first = clean_training_manifest()
        second = clean_training_manifest()
        assert first["training_boundary"]["teacher_targets"] is not forge.TEACHER_TARGET_RECORD
        assert second["training_boundary"]["teacher_targets"] is not first["training_boundary"]["teacher_targets"]
        first["training_boundary"]["teacher_targets"]["enabled"] = True
        first["training_boundary"]["teacher_targets"]["blocked_until"].append("edited")
        assert second["training_boundary"]["teacher_targets"]["enabled"] is False
        assert forge.TEACHER_TARGET_RECORD["enabled"] is False
        assert len(forge.TEACHER_TARGET_RECORD["blocked_until"]) == 2


class TestExistingEvidenceIsUntouched:
    def built_manifest(self) -> dict:
        return forge.build_corpus_manifest(
            name="en-synthetic-02",
            rows=[row(0, 0, "Silence in the archive allowed her to read the letter twice.", kept=True)],
            wall_seconds=1.0,
            slots=[0],
            tts_manifest=stub_manifest("kokoro-int8-multi-lang-v1_1", "sherpa-onnx-kokoro", "Apache-2.0"),
            stt_manifest=stub_manifest("faster-whisper-base", "faster-whisper", "MIT"),
            max_wer=0.0,
        )

    def test_the_builder_never_rewrites_the_committed_corpus_descriptor(self):
        committed = CORPUS_MANIFEST.read_text(encoding="utf-8")
        assert json.loads(committed)["name"] == "en-synthetic-01"
        assert json.loads(committed)["samples"][0]["clip"] != ""
        # Building a manifest opens no path, so the committed descriptor cannot be its output.
        # The assertion that matters is the byte comparison after the call, not the call itself.
        assert self.built_manifest()["name"] == "en-synthetic-02"
        assert CORPUS_MANIFEST.read_text(encoding="utf-8") == committed

    def test_the_default_output_is_a_new_batch_not_the_committed_corpus(self):
        # Running the tool with no `--out` must not be a way to overwrite the corpus the
        # audit already reported on. Read off the parser's own default, not off help text:
        # argparse does not print string defaults, so a `--help` assertion would pass
        # whether the default was right or wrong.
        default = Path(forge.build_parser().parse_args([]).out)
        assert default.name == "en-synthetic-02"
        assert default != CORPUS_MANIFEST.parent


class TestTrainingPoolReplacement:
    """Decision 4: the quarantined sentences are out of the pool, and cannot come back.

    The pool has been rewritten twice - once to drop the eight evaluation colliders, then
    again to drop eight replacements that recycled text the quarantined batch had already
    used. These tests pin the third version against both records, because a pool that only
    checks itself against the evaluation set is how the second rewrite got in.
    """

    def test_no_pool_sentence_appears_anywhere_in_the_quarantined_batch(self):
        recorded = {row["text"] for row in corpus_manifest()["samples"]} | {
            row["text"] for row in corpus_manifest()["dropped"]
        }
        assert set(forge.TRAIN_SENTENCES).isdisjoint(recorded)
        assert len(forge.TRAIN_SENTENCES) == 8
        assert len({forge.normalise(sentence) for sentence in forge.TRAIN_SENTENCES}) == 8

    def test_the_pool_stays_disjoint_from_both_records_it_could_leak_into(self):
        assert forge.collisions(forge.TRAIN_SENTENCES, list(forge.EVAL_SENTENCES)) == []
        assert forge.collisions(forge.TRAIN_SENTENCES, list(forge.quarantined_corpus_texts())) == []

    def test_the_known_collision_stays_a_known_collision(self):
        # The clip's sentence remains recorded in the quarantine, so re-adding it to the pool
        # is caught by the same check rather than by memory of what used to be there.
        assert COLLIDING_SENTENCE in forge.quarantined_corpus_texts()
        assert COLLIDING_SENTENCE not in forge.TRAIN_SENTENCES
        assert forge.collisions([COLLIDING_SENTENCE], list(forge.quarantined_corpus_texts()))

    def test_a_near_duplicate_of_a_quarantined_sentence_is_rejected(self):
        # Punctuation-only and clause-extended variants of a kept clip's sentence.
        variants = [
            "could you tell me the time and the nearest train station",
            COLLIDING_SENTENCE.replace("?", " and the platform number as well?"),
            "Please turn off the lights before you leave the meeting room quietly.",
            "please turn off the lights before you leave the meeting room",
        ]
        for variant in variants:
            assert forge.collisions([variant], list(forge.quarantined_corpus_texts())), variant

    def test_the_new_pool_text_is_not_something_the_quarantine_used_in_a_dropped_attempt(self):
        dropped = [row["text"] for row in corpus_manifest()["dropped"]]
        assert len(dropped) == 8
        for sentence in forge.TRAIN_SENTENCES:
            assert not forge.collisions([sentence], dropped), sentence


class TestTrainingEligibility:
    """`corpus_training_blockers()` decides whether a batch may move weights.

    The contract the owner set: no self-declared flag is accepted on its own, and a clean
    manifest is accepted only when its boundary evidence is complete *and* the audio behind
    its digests has been read by the caller. So the anti-vacuity controls matter as much as
    the refusals - if the gate refused everything, every refusal test below would pass while
    the real training path stayed dead.
    """

    def test_the_quarantined_batch_is_refused_for_training(self):
        blockers = forge.corpus_training_blockers(corpus_manifest())
        assert blockers, "a quarantined batch must never read as training-eligible"
        for expected in (
            "usable_for does not claim 'training'",
            "usable_for does not claim 'adaptation'",
            "training_quarantine status is 'QUARANTINED', not 'CLEARED'",
            "training_quarantine records digests_verified as anything other than true",
            "no split designation of 'training' is recorded",
            "training_boundary is absent or malformed: this batch predates the disjointness contract",
            "no caller-supplied digest of the manifest as read from disk, so its registration "
            "cannot be confirmed: a pre-contract batch is not eligible",
            "no clip digests were supplied by the caller: a manifest cannot prove its own audio exists",
        ):
            assert expected in blockers
        # The count is pinned because this descriptor is a frozen historical record: if it
        # changes, either the quarantine or the leak it documents has been edited. It moved from
        # 11 to 12 when the register anchor landed, because a pre-contract batch now also has to
        # show the caller read the file it is registered under.
        assert len(blockers) == 12

    def test_the_refusal_names_every_colliding_sentence_of_the_quarantined_batch(self):
        blockers = forge.corpus_training_blockers(corpus_manifest())
        leaked_texts = {
            sample["text"]
            for sample in corpus_manifest()["samples"]
            if sample["clip"] in LEAKED_CLIPS
        }
        assert len(leaked_texts) == 4
        for text in leaked_texts:
            assert any(text in blocker and "collides with the evaluation pool" in blocker for blocker in blockers), text

    def test_the_quarantine_record_names_the_four_clips_and_the_measured_duration(self):
        record = corpus_manifest()["training_quarantine"]
        assert set(record["colliding_clips"]) == LEAKED_CLIPS
        assert record["colliding_audio_seconds"] == 11.2355
        assert record["rule"] == forge.DISJOINTNESS_RULE
        assert record["digests_verified"] is False
        assert record["audio_present"] is False

    def test_editing_the_status_field_alone_does_not_release_the_legacy_batch(self):
        # The owner's ruling is enforceable rather than declarative: the legacy batch has no
        # split and no boundary block, so a `status: CLEARED` written into it still cannot
        # reach a trainer.
        manifest = corpus_manifest()
        manifest["training_quarantine"]["status"] = "CLEARED"
        manifest["usable_for"] = ["training", "adaptation", "asr-domain-evaluation"]
        blockers = forge.corpus_training_blockers(manifest)
        assert "training_boundary is absent or malformed: this batch predates the disjointness contract" in blockers
        assert "no split designation of 'training' is recorded" in blockers
        assert any("collides with the evaluation pool" in blocker for blocker in blockers)
        assert "no clip digests were supplied by the caller: a manifest cannot prove its own audio exists" in blockers

    def test_a_clean_batch_is_accepted_once_the_caller_has_read_the_audio(self):
        manifest = clean_training_manifest()
        assert forge.corpus_training_blockers(manifest, clip_digests=recorded_digests(manifest)) == ()

    def test_a_manifest_cannot_prove_that_its_own_audio_exists(self):
        # The one blocker a fully self-consistent manifest still earns, and it is the reason
        # this gate is not a schema check: the JSON describes bytes, it does not show them.
        manifest = clean_training_manifest()
        assert forge.corpus_training_blockers(manifest) == (
            "no clip digests were supplied by the caller: a manifest cannot prove its own audio exists",
        )

    def test_a_digest_the_caller_read_differently_is_a_blocker(self):
        manifest = clean_training_manifest()
        digests = recorded_digests(manifest)
        first = next(iter(digests))
        digests[first] = "00" * 32
        blockers = forge.corpus_training_blockers(manifest, clip_digests=digests)
        assert len(blockers) == 1
        assert blockers[0].startswith(f"{first} digest mismatch: the caller read ")
        assert "the manifest records" in blockers[0]

    def test_a_clip_the_caller_never_read_is_a_blocker(self):
        manifest = clean_training_manifest()
        digests = recorded_digests(manifest)
        first = next(iter(digests))
        digests.pop(first)
        blockers = forge.corpus_training_blockers(manifest, clip_digests=digests)
        assert blockers == (f"{first} was not read by the caller: its bytes are unverified",)

    def test_hashing_is_offered_as_an_answer_the_caller_can_actually_give(self, tmp_path):
        # `read_clip_digests` is what turns the gate's demand into a command, so its two
        # behaviours are pinned: it recomputes from bytes, and a missing clip raises rather
        # than reporting the batch as merely unverified.
        clip = "en_slot0_000.wav"
        payload = b"RIFFfake-wave-bytes"
        (tmp_path / clip).write_bytes(payload)
        assert forge.read_clip_digests(tmp_path, [clip]) == {clip: hashlib.sha256(payload).hexdigest()}
        with pytest.raises(FileNotFoundError):
            forge.read_clip_digests(tmp_path, [clip, "en_slot0_001.wav"])

    def test_a_declared_collision_count_above_zero_is_refused(self):
        manifest = clean_training_manifest()
        manifest["training_boundary"]["sentence_pools"]["collisions_found"] = 3
        blockers = forge.corpus_training_blockers(manifest, clip_digests=recorded_digests(manifest))
        assert blockers == ("training_boundary declares 3 collisions, not zero",)

    @pytest.mark.parametrize(
        ("mutate", "expected"),
        [
            (lambda m: m["training_boundary"].pop("disjointness_rule"), "training_boundary.disjointness_rule is not the rule this code enforces"),
            (lambda m: m["training_boundary"].update(disjointness_rule="   "), "training_boundary.disjointness_rule is not the rule this code enforces"),
            (
                lambda m: m["training_boundary"].update(disjointness_rule="identical sentences only"),
                "training_boundary.disjointness_rule is not the rule this code enforces",
            ),
            (lambda m: m["training_boundary"].pop("sentence_pools"), "training_boundary.sentence_pools is missing or malformed"),
            (lambda m: m["training_boundary"]["sentence_pools"].update(training=0), "training_boundary.sentence_pools.training is not a positive integer"),
            (lambda m: m["training_boundary"]["sentence_pools"].update(evaluation="eight"), "training_boundary.sentence_pools.evaluation is not a positive integer"),
            (lambda m: m["training_boundary"]["sentence_pools"].update(training=800), "declares 800 sentences, this code defines 8"),
            (lambda m: m["training_boundary"]["sentence_pools"].update(evaluation=800), "declares 800 sentences, this code defines 8"),
            (lambda m: m["training_boundary"]["sentence_pools"].pop("collisions_found"), "training_boundary.sentence_pools.collisions_found is not an integer"),
            (lambda m: m["training_boundary"].pop("excluded_inputs"), "training_boundary.excluded_inputs is missing or malformed"),
            (
                lambda m: m["training_boundary"].update(excluded_inputs=["artifacts/evaluation/multilingual-inventory.json"]),
                "training_boundary.excluded_inputs omits ",
            ),
            (
                lambda m: m["training_boundary"].update(excluded_inputs=[*EXCLUDED, "artifacts/audit/offline-proof.json"]),
                "training_boundary.excluded_inputs names inputs no review has recorded as excluded",
            ),
            (lambda m: m["training_boundary"].pop("teacher_targets"), "training_boundary.teacher_targets is missing or malformed"),
            (
                lambda m: m["training_boundary"]["teacher_targets"].update(enabled=True),
                "training_boundary.teacher_targets.enabled is not false while teacher-use and source-data rights are unresolved",
            ),
            (lambda m: m.pop("training_boundary"), "training_boundary is absent or malformed"),
            (lambda m: m.update(split="evaluation"), "no split designation of 'training' is recorded"),
            (lambda m: m.update(synthetic=False), "synthetic is not recorded as true"),
            (lambda m: m.update(licence_basis=""), "licence_basis is missing or empty"),
            (lambda m: m.update(consent_basis=None), "consent_basis is missing or empty"),
            (lambda m: m["generator"].pop("model_id"), "generator.model_id is missing"),
            (lambda m: m["verifier"].update(model_id="  "), "verifier.model_id is missing"),
            (lambda m: m.update(usable_for="training"), "usable_for is missing, empty or not a list of non-empty strings"),
            (lambda m: m.update(usable_for=[]), "usable_for is missing, empty or not a list of non-empty strings"),
            (lambda m: m.update(counts=None), "counts is absent or malformed"),
            (lambda m: m.pop("samples"), "samples is absent, malformed or empty"),
            (lambda m: m.update(samples=[]), "samples is absent, malformed or empty"),
            (lambda m: m["counts"].update(generated=9), "counts.generated 9 is not kept 8 plus dropped 0"),
            (lambda m: m["counts"].update(kept=7), "counts.kept 7 does not match the 8 listed samples"),
            (lambda m: m.update(dropped=None), "dropped is absent or malformed"),
        ],
    )
    def test_a_self_declared_field_that_is_malformed_is_a_blocker(self, mutate, expected):
        manifest = clean_training_manifest()
        mutate(manifest)
        blockers = forge.corpus_training_blockers(manifest, clip_digests=recorded_digests(manifest))
        assert any(expected in blocker for blocker in blockers), blockers

    @pytest.mark.parametrize(
        ("mutate", "expected"),
        [
            (lambda s: s.update(sha256="deadbeef"), "carries no 64-character hexadecimal digest"),
            (lambda s: s.update(sha256="ab" * 31 + "g"), "carries no 64-character hexadecimal digest"),
            (lambda s: s.update(seconds=0), "records no positive duration"),
            (lambda s: s.update(kept=False), "is listed as a training sample but is not marked kept"),
            (lambda s: s.update(text=""), "records no transcript text"),
            (lambda s: s.update(clip=""), "records no clip name"),
        ],
    )
    def test_a_sample_row_that_does_not_describe_a_real_clip_is_a_blocker(self, mutate, expected):
        manifest = clean_training_manifest()
        mutate(manifest["samples"][0])
        digests = {sample["clip"]: sample["sha256"] for sample in manifest["samples"] if sample["clip"]}
        blockers = forge.corpus_training_blockers(manifest, clip_digests=digests)
        assert any(expected in blocker for blocker in blockers), blockers

    def test_listing_one_clip_twice_is_a_blocker(self):
        manifest = clean_training_manifest()
        manifest["samples"][1]["clip"] = manifest["samples"][0]["clip"]
        blockers = forge.corpus_training_blockers(manifest, clip_digests=recorded_digests(manifest))
        assert any("is listed more than once" in blocker for blocker in blockers)

    def test_a_sample_row_that_is_not_an_object_is_refused_without_raising(self):
        manifest = clean_training_manifest()
        manifest["samples"].insert(0, "en_slot9_009.wav")
        blockers = forge.corpus_training_blockers(manifest)
        assert "samples[0] is not an object" in blockers

    def test_the_gate_refuses_a_descriptor_that_is_not_a_json_object(self):
        assert forge.corpus_training_blockers(["not", "a", "manifest"]) == (
            "a corpus manifest must be a JSON object",
        )

    def test_a_clip_digests_argument_that_is_not_a_mapping_is_refused(self):
        manifest = clean_training_manifest()
        blockers = forge.corpus_training_blockers(manifest, clip_digests=[s["clip"] for s in manifest["samples"]])
        assert blockers == (
            "clip_digests must map a clip name to the digest the caller read from its bytes",
        )

    def test_reusing_quarantined_corpus_text_is_a_blocker_even_when_the_evaluation_pool_is_clean(self):
        # This is decision 4 made executable: the train/eval check would pass this batch, and
        # only the quarantine read catches it. QUARANTINED_SENTENCE is a kept clip of
        # en-synthetic-01 and appears in neither shipped pool.
        assert QUARANTINED_SENTENCE not in forge.TRAIN_SENTENCES
        assert forge.collisions([QUARANTINED_SENTENCE], list(forge.EVAL_SENTENCES)) == []
        manifest = clean_training_manifest(texts=[QUARANTINED_SENTENCE])
        blockers = forge.corpus_training_blockers(manifest, clip_digests=recorded_digests(manifest))
        assert any(
            blocker.startswith("a sample reuses quarantined corpus text") and QUARANTINED_SENTENCE in blocker
            for blocker in blockers
        ), blockers

    def test_a_near_duplicate_of_quarantined_text_is_refused_by_the_same_check(self):
        extended = QUARANTINED_SENTENCE + " and then she slept."
        assert forge.collisions([extended], list(forge.quarantined_corpus_texts()))
        manifest = clean_training_manifest(texts=[extended])
        blockers = forge.corpus_training_blockers(manifest, clip_digests=recorded_digests(manifest))
        assert any(blocker.startswith("a sample reuses quarantined corpus text") for blocker in blockers)

    def test_an_unreadable_quarantine_record_blocks_rather_than_being_skipped(self, monkeypatch):
        def explode(**_kwargs):
            raise forge.CorpusBoundaryError("cannot read the corpus record at …")

        monkeypatch.setattr(forge, "quarantined_corpus_texts", explode)
        manifest = clean_training_manifest()
        blockers = forge.corpus_training_blockers(manifest, clip_digests=recorded_digests(manifest))
        assert any(
            blocker.startswith("the quarantine records cannot be read, so reuse cannot be ruled out")
            for blocker in blockers
        ), blockers

    def test_a_batch_is_not_reported_for_reusing_its_own_sentences(self):
        # Without this the quarantined batch would answer its own record, and the reuse check
        # would bury the one blocker that identifies the leak.
        assert forge.quarantined_corpus_texts(exclude="en-synthetic-01") == ()
        blockers = forge.corpus_training_blockers(corpus_manifest())
        assert not any("reuses quarantined corpus text" in blocker for blocker in blockers)


class TestEvaluationUseIsIndependent:
    """Decision 3: quarantining a batch from training must not retire what it measured."""

    def test_the_quarantined_batch_keeps_its_evaluation_use(self):
        assert forge.corpus_evaluation_uses(corpus_manifest()) == ["asr-domain-evaluation"]

    def test_a_training_eligible_batch_reports_the_same_evaluation_use(self):
        assert forge.corpus_evaluation_uses(clean_training_manifest()) == ["asr-domain-evaluation"]

    def test_training_and_adaptation_are_never_reported_as_evaluation_uses(self):
        # A batch that IS eligible for training still exposes only its evaluation scope here,
        # so a caller cannot read this function as a general "what may I do" answer.
        training_eligible = clean_training_manifest()
        assert "training" in training_eligible["usable_for"]
        assert forge.corpus_evaluation_uses(training_eligible) == ["asr-domain-evaluation"]
        assert forge.corpus_evaluation_uses({"usable_for": ["training", "adaptation"]}) == []

    def test_a_refused_training_claim_does_not_change_the_evaluation_answer(self):
        manifest = corpus_manifest()
        assert forge.corpus_evaluation_uses(manifest) == forge.corpus_evaluation_uses(
            {**manifest, "usable_for": ["training", "adaptation", "asr-domain-evaluation"]}
        ) == ["asr-domain-evaluation"]

    def test_a_malformed_scope_list_reads_as_no_permitted_use(self):
        assert forge.corpus_evaluation_uses({"usable_for": "asr-domain-evaluation"}) == []
        assert forge.corpus_evaluation_uses({}) == []
        assert forge.corpus_evaluation_uses({"usable_for": [""]}) == []


class TestQuarantineRecords:
    """What the gate reads, tested against a corpus root that is not the repository's."""

    def test_the_committed_quarantine_yields_every_sentence_the_batch_forged(self):
        texts = forge.quarantined_corpus_texts()
        manifest = corpus_manifest()
        assert len(texts) == 16
        assert set(texts) == {row["text"] for row in manifest["samples"]} | {
            row["text"] for row in manifest["dropped"]
        }

    def test_a_batch_cleared_for_training_contributes_no_text(self, tmp_path):
        batch = tmp_path / "en-synthetic-77"
        batch.mkdir()
        (batch / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "en-synthetic-77",
                    "samples": [{"text": "Cleared batches are not a reuse source."}],
                    "dropped": [],
                    "training_quarantine": {"status": forge.QUARANTINE_CLEARED},
                }
            ),
            encoding="utf-8",
        )
        assert forge.quarantined_corpus_texts(tmp_path) == ()

    def test_a_batch_with_no_quarantine_record_is_not_a_reuse_source(self, tmp_path):
        batch = tmp_path / "en-synthetic-78"
        batch.mkdir()
        (batch / "manifest.json").write_text(json.dumps({"name": "x", "samples": [], "dropped": []}), encoding="utf-8")
        assert forge.quarantined_corpus_texts(tmp_path) == ()

    @pytest.mark.parametrize(
        "document",
        [
            "{not json at all",
            json.dumps([1, 2, 3]),
            json.dumps({"name": "n", "samples": {"clip": "a"}, "dropped": [], "training_quarantine": {"status": "QUARANTINED"}}),
            json.dumps({"name": "n", "samples": [], "dropped": "none", "training_quarantine": {"status": "QUARANTINED"}}),
        ],
    )
    def test_a_damaged_quarantine_record_raises_instead_of_reading_as_empty(self, tmp_path, document):
        batch = tmp_path / "en-synthetic-79"
        batch.mkdir()
        (batch / "manifest.json").write_text(document, encoding="utf-8")
        with pytest.raises(forge.CorpusBoundaryError):
            forge.quarantined_corpus_texts(tmp_path)

    def test_a_quarantine_record_that_cannot_be_opened_raises(self, tmp_path, monkeypatch):
        batch = tmp_path / "en-synthetic-80"
        batch.mkdir()
        (batch / "manifest.json").write_text(json.dumps({"samples": [], "dropped": []}), encoding="utf-8")
        monkeypatch.setattr(Path, "read_text", lambda *_a, **_k: (_ for _ in ()).throw(OSError("locked")))
        with pytest.raises(forge.CorpusBoundaryError):
            forge.quarantined_corpus_texts(tmp_path)

    def test_no_corpus_root_reads_as_no_quarantine_at_all(self, tmp_path):
        # A documented limitation, pinned so it cannot be mistaken for a guarantee: the reuse
        # check is exactly as good as the records on the machine running it. Deleting the
        # corpus directory would silence it, which is why the quarantine record, the audit
        # finding and this repository's copy of the manifest all carry the same text.
        assert forge.quarantined_corpus_texts(tmp_path / "absent") == ()

    def test_the_gate_and_the_generator_read_the_same_quarantine(self):
        # main() refuses a run before any model is constructed and the gate refuses after the
        # fact. Both call this one reader, so the shipped pool has to answer cleanly to both:
        # a pool that passed the run check but failed the gate would mean the generator's
        # refusal is a different rule from the trainer's.
        quarantined = list(forge.quarantined_corpus_texts())
        assert forge.collisions(forge.TRAIN_SENTENCES, quarantined) == []
        manifest = clean_training_manifest()
        blockers = forge.corpus_training_blockers(manifest, clip_digests=recorded_digests(manifest))
        assert not any("reuses quarantined corpus text" in blocker for blocker in blockers), blockers


class TestTheBoundaryContract:
    """P1: a manifest's boundary claims are verified against this code and the tracked register.

    The report measured six bypasses that all had one shape - the manifest declared something
    and the gate agreed. These tests answer each of them, and they pin the shipped register
    against the shipped pools so a pool or rule edit cannot land as a quiet one-file change.
    """

    def test_the_builder_stamps_the_live_pool_and_the_supported_version(self):
        manifest = clean_training_manifest()
        contract = manifest["training_boundary"]["boundary_contract"]
        assert contract == {
            "version": forge.SUPPORTED_BOUNDARY_VERSION,
            "issued_by": forge.CONTRACT_ISSUED_BY,
            "pool_set_sha256": live_pool_sha(),
        }
        assert forge.SUPPORTED_BOUNDARY_VERSION == 1

    def test_the_shipped_register_records_the_live_pool_set(self):
        # The drift detector. Rewrite a sentence pool, or re-word `DISJOINTNESS_RULE`, without
        # editing artifacts/corpus/legacy-batches.json and this fails - which is the mechanism
        # behind "a pool change requires a reviewed, tracked update", not a promise in prose.
        register = forge.boundary_register(REPO_REGISTER)
        recorded = [entry for entry in register["pool_sets"] if entry["pool_set_sha256"] == live_pool_sha()]
        assert len(recorded) == 1, "the live pool set is registered exactly once"
        entry = recorded[0]
        assert entry["boundary_contract_version"] == forge.SUPPORTED_BOUNDARY_VERSION
        assert entry["disjointness_rule"] == forge.DISJOINTNESS_RULE
        assert entry["training_sentences"] == len(forge.TRAIN_SENTENCES)
        assert entry["evaluation_sentences"] == len(forge.EVAL_SENTENCES)

    def test_the_pool_digest_is_stable_and_sensitive_to_what_it_covers(self):
        assert live_pool_sha() == forge.pool_set_sha256(forge.TRAIN_SENTENCES, forge.EVAL_SENTENCES)
        assert live_pool_sha() != forge.pool_set_sha256(NOVEL_TRAINING_POOL, forge.EVAL_SENTENCES)
        assert live_pool_sha() != forge.pool_set_sha256(forge.EVAL_SENTENCES, forge.TRAIN_SENTENCES)
        assert live_pool_sha() != forge.pool_set_sha256(forge.TRAIN_SENTENCES[:1], forge.EVAL_SENTENCES)
        # Normalised, so a re-cased pool is the same pool and a shuffled one is not.
        assert forge.pool_set_sha256(("Hello There",), ("Good Day",)) == forge.pool_set_sha256(
            ("hello, there!",), ("good day",)
        )
        assert forge.pool_set_sha256(("one two", "three"), ("x",)) != forge.pool_set_sha256(("three", "one two"), ("x",))

    def test_a_pool_this_code_no_longer_defines_fails_closed_even_when_it_is_registered(self, tmp_path, monkeypatch):
        # The retired-pool case: a manifest stamped under the old pool, and a register that still
        # records that old pool. Registration is not the anchor - the live pools are - so a
        # retired sentence set cannot be revived by a sympathetic register entry, and the same
        # edit run backwards (register the new pool, ship the old text) fails on the same check.
        manifest = clean_training_manifest()
        stamped = live_pool_sha()
        register = write_register(tmp_path, pool_sets=[pool_entry()])
        monkeypatch.setattr(forge, "TRAIN_SENTENCES", NOVEL_TRAINING_POOL)
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=register
        )
        assert any("names a pool set this code no longer defines" in blocker for blocker in blockers), blockers
        assert live_pool_sha() != stamped, "the fixture must really be a different pool"

    @pytest.mark.parametrize(
        ("mutate", "expected"),
        [
            (lambda c: c.update(version=0), "boundary_contract.version is 0, not the supported contract version 1"),
            (lambda c: c.update(version=2), "boundary_contract.version is 2, not the supported contract version 1"),
            (lambda c: c.update(version="1"), "boundary_contract.version is '1', not the supported contract version 1"),
            (lambda c: c.pop("version"), "boundary_contract.version is None, not the supported contract version 1"),
            (lambda c: c.update(pool_set_sha256="abc"), "pool_set_sha256 is missing or is not a 64-character hexadecimal digest"),
            (lambda c: c.pop("pool_set_sha256"), "pool_set_sha256 is missing or is not a 64-character hexadecimal digest"),
            (lambda c: c.update(pool_set_sha256="cd" * 32), "names a pool set this code no longer defines"),
        ],
    )
    def test_a_contract_header_that_does_not_match_this_code_is_refused(self, mutate, expected, tmp_path):
        manifest = clean_training_manifest()
        contract = manifest["training_boundary"]["boundary_contract"]
        mutate(contract)
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        )
        assert any(expected in blocker for blocker in blockers), blockers

    def test_a_boundary_contract_that_is_not_an_object_is_refused(self, tmp_path):
        manifest = clean_training_manifest()
        manifest["training_boundary"]["boundary_contract"] = "approved, trust me"
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        )
        assert blockers == ("training_boundary.boundary_contract is present but is not an object",)

    def test_a_pool_set_that_is_not_in_the_register_is_refused(self, tmp_path):
        # The claimed digest matches the live pools, so the only thing missing is the reviewed
        # record of them - and that is the whole point of P2.
        manifest = clean_training_manifest()
        register = write_register(tmp_path, pool_sets=[])
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=register
        )
        assert any(
            "is not recorded in artifacts/corpus/legacy-batches.json" in blocker for blocker in blockers
        ), blockers

    def test_a_register_binding_the_pool_to_another_version_is_refused(self, tmp_path):
        manifest = clean_training_manifest()
        register = write_register(tmp_path, pool_sets=[pool_entry(boundary_contract_version=2)])
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=register
        )
        assert any("binds this pool set to contract version 2" in blocker for blocker in blockers), blockers

    def test_a_register_recording_another_rule_is_refused(self, tmp_path):
        manifest = clean_training_manifest()
        register = write_register(tmp_path, pool_sets=[pool_entry(disjointness_rule="identical sentences only")])
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=register
        )
        assert any("different disjointness rule than this code enforces" in blocker for blocker in blockers), blockers

    def test_a_register_recording_other_pool_sizes_is_refused(self, tmp_path):
        manifest = clean_training_manifest()
        register = write_register(tmp_path, pool_sets=[pool_entry(evaluation_sentences=99)])
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=register
        )
        assert any("pool sizes do not match the pools in code" in blocker for blocker in blockers), blockers

    def test_a_sample_that_is_not_a_sentence_of_the_registered_pool_is_refused(self, tmp_path):
        # Declaring the approved pool and drawing from it are different claims. This batch
        # declares the live pool and carries one novel sentence, so the header is true and the
        # audio is not what the header describes.
        texts = [NOVEL_SENTENCE, *forge.TRAIN_SENTENCES[1:]]
        manifest = clean_training_manifest(texts=texts)
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        )
        assert any(
            blocker.startswith("a sample is not a sentence of the registered training pool")
            and NOVEL_SENTENCE in blocker
            for blocker in blockers
        ), blockers

    def test_issued_by_is_recorded_and_never_believed(self, tmp_path):
        # Pinned in both directions: the field is inert evidence (renaming the issuer changes
        # nothing), and naming an authoritative issuer does not rescue a header that does not
        # match this code.
        manifest = clean_training_manifest()
        manifest["training_boundary"]["boundary_contract"]["issued_by"] = "the project owner, personally"
        assert (
            forge.corpus_training_blockers(
                manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
            )
            == ()
        )
        manifest["training_boundary"]["boundary_contract"]["version"] = 7
        manifest["training_boundary"]["boundary_contract"]["issued_by"] = forge.CONTRACT_ISSUED_BY
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        )
        assert any("not the supported contract version" in blocker for blocker in blockers), blockers

    def test_the_committed_corpus_declares_no_contract_and_is_refused_on_provenance(self):
        manifest = corpus_manifest()
        blockers = forge.corpus_training_blockers(
            manifest,
            manifest_sha256=manifest_digest_of(CORPUS_MANIFEST),
            register_path=REPO_REGISTER,
        )
        assert any(
            "the register records en-synthetic-01 as 'QUARANTINED'" in blocker for blocker in blockers
        ), blockers


class TestTheLegacyRegister:
    """P2: 'legacy' is a named list in a tracked file, never an absent field.

    The register anchors a batch's *provenance* - this exact file, with these exact bytes, was
    read by a reviewer and ruled on. It supplies no evidence: a batch that records no boundary
    block is refused even under a `CLEARED` ruling, and one whose bytes differ from the recorded
    digest is refused however kind the ruling reads. Both halves are tested below, because the
    useful statement is not "the register can clear a batch" but exactly what clearing it takes.
    """

    def legacy(self, tmp_path, *, drop_boundary: bool = True) -> tuple[dict, Path, str]:
        """A batch with no contract header, published to a temporary directory.

        `drop_boundary` also removes the whole boundary block, which is what a real pre-contract
        artefact such as `en-synthetic-01` looks like. Keeping the block but stripping the header
        separates the two things the register can and cannot grant: provenance (this file is the
        reviewed one) and evidence (this file describes its own boundary at all).
        """
        manifest = clean_training_manifest()
        manifest["training_boundary"].pop("boundary_contract")
        if drop_boundary:
            manifest.pop("training_boundary")
        path = publish(manifest, tmp_path)
        return manifest, path, manifest_digest_of(path)

    def test_a_registered_batch_with_a_matching_digest_is_no_longer_unanchored(self, tmp_path):
        manifest, _path, sha = self.legacy(tmp_path)
        register = write_register(
            tmp_path / "reg", batches=[batch_entry(name=manifest["name"], manifest_sha256=sha, status="QUARANTINED")]
        )
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), manifest_sha256=sha, register_path=register
        )
        assert not any("the provenance of its boundary evidence is unknown" in blocker for blocker in blockers), blockers
        assert any(f"the register records {manifest['name']} as 'QUARANTINED'" in blocker for blocker in blockers), blockers

    def test_an_explicit_cleared_ruling_is_required_not_a_pending_one(self, tmp_path):
        manifest, _path, sha = self.legacy(tmp_path, drop_boundary=False)
        register = write_register(
            tmp_path / "reg", batches=[batch_entry(name=manifest["name"], manifest_sha256=sha, status="PENDING REVIEW")]
        )
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), manifest_sha256=sha, register_path=register
        )
        assert any(
            "not 'CLEARED': clearing a pre-contract batch takes an explicit ruling in the tracked register" in blocker
            for blocker in blockers
        ), blockers

    def test_a_matching_digest_and_an_explicit_ruling_release_a_batch_whose_evidence_is_intact(self, tmp_path):
        # What a real release looks like: the register names this exact file, rules it CLEARED,
        # and every other check is re-derived from code and disk. No batch in the shipped
        # register is in this position, and the digest here is over a temporary fixture.
        manifest, _path, sha = self.legacy(tmp_path, drop_boundary=False)
        register = write_register(
            tmp_path / "reg", batches=[batch_entry(name=manifest["name"], manifest_sha256=sha, status="CLEARED")]
        )
        assert (
            forge.corpus_training_blockers(
                manifest, clip_digests=recorded_digests(manifest), manifest_sha256=sha, register_path=register
            )
            == ()
        )

    def test_a_cleared_ruling_does_not_supply_missing_boundary_evidence(self, tmp_path):
        manifest, _path, sha = self.legacy(tmp_path)
        register = write_register(
            tmp_path / "reg", batches=[batch_entry(name=manifest["name"], manifest_sha256=sha, status="CLEARED")]
        )
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), manifest_sha256=sha, register_path=register
        )
        assert blockers == (
            "training_boundary is absent or malformed: this batch predates the disjointness contract",
        )

    def test_a_cleared_ruling_is_worthless_without_the_digest_that_identifies_the_file(self, tmp_path):
        manifest, _path, sha = self.legacy(tmp_path, drop_boundary=False)
        register = write_register(
            tmp_path / "reg", batches=[batch_entry(name=manifest["name"], manifest_sha256=sha, status="CLEARED")]
        )
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=register
        )
        assert blockers == (
            "no caller-supplied digest of the manifest as read from disk, so its registration cannot be "
            "confirmed: a pre-contract batch is not eligible",
        )

    def test_editing_a_registered_manifest_invalidates_its_own_entry(self, tmp_path):
        manifest, path, _sha = self.legacy(tmp_path, drop_boundary=False)
        register = write_register(
            tmp_path / "reg", batches=[batch_entry(name=manifest["name"], manifest_sha256="ef" * 32, status="CLEARED")]
        )
        blockers = forge.corpus_training_blockers(
            manifest,
            clip_digests=recorded_digests(manifest),
            manifest_sha256=manifest_digest_of(path),
            register_path=register,
        )
        assert any(
            "no longer matches the record that was reviewed, so its boundary evidence is unanchored" in blocker
            for blocker in blockers
        ), blockers

    def test_a_batch_that_claims_a_contract_while_the_register_quarantines_its_name_is_refused(self, tmp_path):
        # The bypass this closes: a quarantined batch re-labelled with a valid-looking contract
        # header. The register answers for the *name*, so the header buys nothing.
        manifest = corpus_manifest()
        manifest["training_boundary"] = clean_training_manifest()["training_boundary"]
        blockers = forge.corpus_training_blockers(
            manifest, manifest_sha256=manifest_digest_of(CORPUS_MANIFEST), register_path=REPO_REGISTER
        )
        assert any(
            "a boundary contract written into a batch that is still quarantined on record does not release it"
            in blocker
            for blocker in blockers
        ), blockers

    def test_an_unregistered_batch_with_no_contract_is_refused_by_name(self, tmp_path):
        manifest = clean_training_manifest(name="en-synthetic-55")
        manifest["training_boundary"].pop("boundary_contract")
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        )
        assert blockers == (
            "en-synthetic-55 has no supported boundary_contract and is not a registered pre-contract batch: "
            "the provenance of its boundary evidence is unknown",
        )

    def test_an_absent_quarantine_field_is_not_a_release(self, tmp_path):
        # The report's core finding: `if quarantine is not None` meant a batch with nothing
        # declared was treated as nothing wrong. Absence now has to be answered by a contract or
        # a register ruling, and this batch has neither.
        manifest = clean_training_manifest(name="en-synthetic-56")
        assert forge.QUARANTINE_FIELD not in manifest
        manifest["training_boundary"].pop("boundary_contract")
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        )
        assert not any("training_quarantine" in blocker for blocker in blockers), "nothing is declared, so nothing fires"
        assert any("is not a registered pre-contract batch" in blocker for blocker in blockers), blockers

    def test_a_batch_with_no_name_cannot_be_matched_to_a_ruling(self, tmp_path):
        manifest = clean_training_manifest()
        manifest.pop("name")
        manifest["training_boundary"].pop("boundary_contract")
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        )
        assert any("records no name, so no ruling in the register can be matched" in blocker for blocker in blockers), blockers

    @pytest.mark.parametrize(
        ("document", "refusal"),
        [
            pytest.param("{not json at all", "cannot read the boundary register at", id="not-json"),
            pytest.param(json.dumps([1, 2, 3]), "is not a JSON object", id="not-an-object"),
            pytest.param(
                json.dumps({"schema_version": 99, "pool_sets": [], "pre_contract_batches": []}),
                "declares schema_version",
                id="wrong-schema-version",
            ),
            pytest.param(register_document({}, []), "records no pool_sets list", id="pool-sets-not-a-list"),
            pytest.param(register_document([], {}), "records no pre_contract_batches list", id="batches-not-a-list"),
            pytest.param(
                json.dumps({"schema_version": 1, "pre_contract_batches": []}),
                "records no pool_sets list",
                id="pool-sets-absent",
            ),
            pytest.param(
                json.dumps({"schema_version": 1, "pool_sets": []}),
                "records no pre_contract_batches list",
                id="batches-absent",
            ),
            pytest.param(
                register_document([{"pool_set_sha256": "abc"}], []), "pool set with no valid digest", id="pool-digest-malformed"
            ),
            pytest.param(
                register_document([missing_field(pool_entry(), "boundary_contract_version")], []),
                "pool set with no contract version",
                id="pool-version-missing",
            ),
            pytest.param(
                register_document([missing_field(pool_entry(), "disjointness_rule")], []),
                "no recorded disjointness rule",
                id="pool-rule-missing",
            ),
            pytest.param(
                register_document([missing_field(pool_entry(), "training_sentences")], []),
                "pool set with no training_sentences",
                id="pool-training-count-missing",
            ),
            pytest.param(
                register_document([missing_field(pool_entry(), "evaluation_sentences")], []),
                "pool set with no evaluation_sentences",
                id="pool-evaluation-count-missing",
            ),
            pytest.param(
                register_document([], [{"manifest_sha256": "ab" * 32, "status": "CLEARED"}]),
                "batch record with no name",
                id="batch-name-missing",
            ),
            pytest.param(
                register_document([], [missing_field(batch_entry(), "manifest_sha256")]),
                "without a valid manifest digest",
                id="batch-digest-missing",
            ),
            pytest.param(
                register_document([], [missing_field(batch_entry(), "status")]),
                "with no status ruling",
                id="batch-status-missing",
            ),
            pytest.param(
                register_document([], ["a string rather than an object"]),
                "batch record that is not an object",
                id="batch-not-an-object",
            ),
        ],
    )
    def test_a_damaged_register_is_a_refusal_not_an_empty_one(self, tmp_path, document, refusal):
        # Each case names the single requirement it breaks, and asserts that requirement's own
        # message: a register broken in three ways refuses for any one of them, so a generic
        # assertion here would pass even after the check under test had been removed.
        path = tmp_path / "legacy-batches.json"
        path.write_text(document, encoding="utf-8")
        manifest = clean_training_manifest()
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=path
        )
        assert any(refusal in blocker for blocker in blockers), (refusal, blockers)
        with pytest.raises(forge.CorpusBoundaryError) as raised:
            forge.boundary_register(path)
        assert refusal in str(raised.value)

    def test_an_absent_register_file_is_a_refusal(self, tmp_path):
        manifest = clean_training_manifest()
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=tmp_path / "absent.json"
        )
        assert any("cannot be read, so no batch can be anchored" in blocker for blocker in blockers), blockers

    def test_the_register_is_read_on_every_call_and_never_cached(self, tmp_path):
        # A pool registered after a first refusal has to change the second answer, and one
        # withdrawn after an acceptance has to revoke it. Caching either would make the anchor a
        # snapshot rather than a check.
        manifest = clean_training_manifest()
        register = write_register(tmp_path, pool_sets=[])
        assert forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=register
        ) != ()
        register.write_text(
            json.dumps(
                {
                    "schema_version": forge.REGISTER_SCHEMA_VERSION,
                    "pool_sets": [pool_entry()],
                    "pre_contract_batches": [],
                }
            ),
            encoding="utf-8",
        )
        assert forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=register
        ) == ()


class TestClosedUseVocabulary:
    """P3: `usable_for` is a closed set, so an invented scope is not a permission."""

    def test_the_vocabulary_is_the_three_named_uses(self):
        assert forge.ALLOWED_USES == ("training", "adaptation", "asr-domain-evaluation")

    def test_the_builder_emits_only_supported_uses(self):
        assert set(clean_training_manifest()["usable_for"]) <= set(forge.ALLOWED_USES)

    @pytest.mark.parametrize(
        "extra",
        ["speaker-cloning-training", "quality-scoring-of-human-speech", "training ", "synthetic-evaluation"],
    )
    def test_an_invented_use_is_a_blocker_named_by_value(self, tmp_path, extra):
        manifest = clean_training_manifest()
        manifest["usable_for"] = [*manifest["usable_for"], extra]
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        )
        assert any(blocker.startswith("usable_for names an unsupported use") and extra in blocker for blocker in blockers), blockers

    def test_a_single_supported_use_list_is_not_a_refusal(self, tmp_path):
        manifest = clean_training_manifest()
        assert forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        ) == ()

    def test_the_evaluation_reader_reports_only_a_supported_use(self):
        # The other half of P3: the function an evaluation reader consults must not hand back an
        # invented scope as if it were a permission it had checked.
        manifest = clean_training_manifest()
        manifest["usable_for"] = [*manifest["usable_for"], "speaker-cloning-training"]
        assert forge.corpus_evaluation_uses(manifest) == [forge.EVALUATION_USE]


class TestStatusFlagsAreRecordsNotEvidence:
    """P4: `status: CLEARED` and `digests_verified: true` are required and prove nothing."""

    def test_self_declared_clearance_with_no_caller_digests_is_still_blocked(self, tmp_path):
        manifest = clean_training_manifest()
        manifest["training_quarantine"] = {"status": forge.QUARANTINE_CLEARED, "digests_verified": True}
        blockers = forge.corpus_training_blockers(manifest, register_path=write_register(tmp_path))
        assert blockers == (
            "no clip digests were supplied by the caller: a manifest cannot prove its own audio exists",
        )

    def test_a_forged_clearance_block_changes_an_eligible_batch_not_at_all(self, tmp_path):
        # Acceptance came from the contract, the register and the caller's digests. A clearance
        # record sitting beside them contributes nothing - that is what "record, not evidence"
        # means in code, and it is what would break if anyone wrote `if not blockers: return`
        # after reading the quarantine.
        manifest = clean_training_manifest()
        manifest["training_quarantine"] = {"status": forge.QUARANTINE_CLEARED, "digests_verified": True}
        assert forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        ) == ()

    @pytest.mark.parametrize(
        ("record", "expected"),
        [
            ({"status": "CLEARED"}, "training_quarantine records digests_verified as anything other than true"),
            ({"status": "cleared", "digests_verified": True}, "training_quarantine status is 'cleared'"),
            ({"status": "", "digests_verified": True}, "training_quarantine status is ''"),
            ({"digests_verified": True}, "training_quarantine status is None"),
            ("CLEARED", "training_quarantine is present but is not an object"),
        ],
    )
    def test_a_quarantine_record_that_is_not_an_explicit_clearance_still_refuses(self, tmp_path, record, expected):
        manifest = clean_training_manifest()
        manifest["training_quarantine"] = record
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        )
        assert any(expected in blocker for blocker in blockers), blockers

    def test_digests_verified_true_does_not_stand_in_for_the_caller(self, tmp_path):
        manifest = clean_training_manifest()
        manifest["training_quarantine"] = {"status": forge.QUARANTINE_CLEARED, "digests_verified": True}
        blockers = forge.corpus_training_blockers(manifest, register_path=write_register(tmp_path))
        assert any("a manifest cannot prove its own audio exists" in blocker for blocker in blockers), blockers

    def test_the_register_digest_is_required_as_well_as_the_clip_digests(self, tmp_path):
        # Two different callers, two different kinds of byte: the audio and the descriptor. A
        # pre-contract batch needs both, and supplying only one leaves the other blocker in place.
        manifest = clean_training_manifest(name="en-synthetic-57")
        manifest["training_boundary"].pop("boundary_contract")
        register = write_register(
            tmp_path, batches=[batch_entry(name="en-synthetic-57", manifest_sha256="ab" * 32, status="CLEARED")]
        )
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=register
        )
        assert blockers == (
            "no caller-supplied digest of the manifest as read from disk, so its registration cannot be "
            "confirmed: a pre-contract batch is not eligible",
        )

    def test_invented_rights_claims_are_records_and_the_batch_stays_blocked(self, tmp_path):
        # The report's "licence_basis: trust me" column. P5 - resolving the named assets against
        # the model registry - was declined, so these fields stay declarations. What is pinned
        # here is that they are never the *whole* answer: a batch that forges them and nothing
        # else is still refused by evidence outside them, and no registry is consulted.
        manifest = clean_training_manifest(name="en-synthetic-58")
        manifest["training_boundary"].pop("boundary_contract")
        manifest["licence_basis"] = "trust me"
        manifest["consent_basis"] = "agreed"
        manifest["generator"]["model_id"] = "the-best-model-ever"
        manifest["verifier"]["model_id"] = "also-the-best-model-ever"
        blockers = forge.corpus_training_blockers(
            manifest, clip_digests=recorded_digests(manifest), register_path=write_register(tmp_path)
        )
        assert blockers == (
            "en-synthetic-58 has no supported boundary_contract and is not a registered pre-contract batch: "
            "the provenance of its boundary evidence is unknown",
        )


class TestTheTrainingEntryPoint:
    """`corpus_training_eligible()` is the named door - and there is no room behind it yet.

    No trainer calls this. Every test here is the gate's contract on a real filesystem: audio
    that cannot be read, a descriptor that cannot be parsed and a register that cannot be found
    each refuse rather than degrade into an "unverified but allowed" answer.
    """

    def an_eligible_batch(self, tmp_path) -> tuple[Path, Path]:
        manifest = clean_training_manifest()
        clips_on_disk(manifest, tmp_path / "clips")
        return publish(manifest, tmp_path / "batch"), tmp_path / "clips"

    def test_a_fully_anchored_batch_with_audio_on_disk_is_eligible(self, tmp_path):
        batch, clips = self.an_eligible_batch(tmp_path)
        assert forge.corpus_training_eligible(batch, clips, register_path=REPO_REGISTER) is None

    def test_the_exception_is_part_of_the_boundary_family(self):
        assert issubclass(forge.CorpusNotEligible, forge.CorpusBoundaryError)
        assert issubclass(forge.CorpusBoundaryError, RuntimeError)

    def test_a_rewritten_clip_is_a_mismatch_and_not_an_acceptance(self, tmp_path):
        batch, clips = self.an_eligible_batch(tmp_path)
        victim = next(iter(clips.iterdir()))
        victim.write_bytes(b"wav:" + victim.name.encode("utf-8") + b"-tampered")
        with pytest.raises(forge.CorpusNotEligible) as caught:
            forge.corpus_training_eligible(batch, clips, register_path=REPO_REGISTER)
        assert "digest mismatch" in str(caught.value)
        assert "byte identity only" in str(caught.value), "the refusal must not read as a rights clearance"

    def test_a_deleted_clip_raises_rather_than_reporting_the_batch_as_merely_unverified(self, tmp_path):
        batch, clips = self.an_eligible_batch(tmp_path)
        next(iter(clips.iterdir())).unlink()
        with pytest.raises(forge.CorpusNotEligible, match="cannot be read"):
            forge.corpus_training_eligible(batch, clips, register_path=REPO_REGISTER)

    def test_an_empty_audio_directory_refuses_the_batch(self, tmp_path):
        manifest = clean_training_manifest()
        batch = publish(manifest, tmp_path / "batch")
        (tmp_path / "clips").mkdir()
        with pytest.raises(forge.CorpusNotEligible, match="cannot be read"):
            forge.corpus_training_eligible(batch, tmp_path / "clips", register_path=REPO_REGISTER)

    def test_an_absent_manifest_raises(self, tmp_path):
        with pytest.raises(forge.CorpusNotEligible, match="cannot be read"):
            forge.corpus_training_eligible(tmp_path / "nowhere.json", tmp_path, register_path=REPO_REGISTER)

    @pytest.mark.parametrize("payload", ["{ not json", json.dumps([1, 2, 3])])
    def test_a_malformed_manifest_raises_rather_than_reading_as_no_blockers(self, tmp_path, payload):
        batch = tmp_path / "manifest.json"
        batch.write_text(payload, encoding="utf-8")
        with pytest.raises(forge.CorpusNotEligible, match="not readable JSON|not a JSON object"):
            forge.corpus_training_eligible(batch, tmp_path, register_path=REPO_REGISTER)

    def test_an_unreadable_register_refuses_an_otherwise_eligible_batch(self, tmp_path):
        batch, clips = self.an_eligible_batch(tmp_path)
        with pytest.raises(forge.CorpusNotEligible, match="boundary register"):
            forge.corpus_training_eligible(batch, clips, register_path=tmp_path / "absent.json")

    def test_the_quarantined_batch_is_refused_by_the_entry_point(self, tmp_path):
        # A copy with bytes attached to its clip names, so the refusal comes from the gate rather
        # than from the missing audio: what is pinned is that the historical leak is still named,
        # and that attaching bytes to the descriptor is itself an edit that invalidates its
        # registration - the committed file's digest is the one the register holds.
        manifest = corpus_manifest()
        clips_on_disk(manifest, tmp_path / "clips")
        batch = publish(manifest, tmp_path / "batch")
        with pytest.raises(forge.CorpusNotEligible) as caught:
            forge.corpus_training_eligible(batch, tmp_path / "clips", register_path=REPO_REGISTER)
        message = str(caught.value)
        assert "en-synthetic-01 is not eligible for training" in message
        assert "collides with the evaluation pool" in message, "the leak itself has to be named"
        assert "no longer matches the record that was reviewed" in message

    def test_the_committed_record_is_refused_on_its_absent_audio_before_anything_else(self, tmp_path):
        # The order matters: the entry point reads the bytes first, so the quarantined batch
        # cannot even reach the argument about its fields on a machine that lacks its audio.
        (tmp_path / "clips").mkdir()
        with pytest.raises(forge.CorpusNotEligible, match="cannot be read"):
            forge.corpus_training_eligible(CORPUS_MANIFEST, tmp_path / "clips", register_path=REPO_REGISTER)

    def test_the_refusal_lists_every_blocker_not_just_the_first(self, tmp_path):
        manifest = clean_training_manifest()
        manifest["training_boundary"]["boundary_contract"]["version"] = 404
        manifest["usable_for"] = ["speaker-cloning-training"]
        clips_on_disk(manifest, tmp_path / "clips")
        batch = publish(manifest, tmp_path / "batch")
        with pytest.raises(forge.CorpusNotEligible) as caught:
            forge.corpus_training_eligible(batch, tmp_path / "clips", register_path=REPO_REGISTER)
        message = str(caught.value)
        assert "usable_for names an unsupported use" in message
        assert "not the supported contract version 1" in message
