"""M1: who may read a corpus batch, and who may move weights. Two fences over the working tree.

The measurement behind this file is not a hypothesis. `tools/corpus_forge.py` is the only module
in the repository that opens a corpus descriptor at all: `tools/evaluate.py` synthesises from a
`CORPUS` dict written into its own source and reads no manifest, and `SampleManifest.load()` - the
only evaluation-manifest reader in `src/` - refuses a corpus batch outright, because every field
of a corpus row (`clip`, `sha256`, `seconds`, `wer`, and the rest) is an unknown sample key. So
the permitted use `asr-domain-evaluation` is a permission on record, not a capability: **no tool
in this repository scores a corpus batch today**, and nothing here pretends otherwise.

That leaves two ways the arrangement can rot, and a comment cannot stop either:

* a new module opens a corpus descriptor and reads its `usable_for` as it likes - which is how a
  quarantined batch would end up scored, or a training pool would end up read as evaluation text;
* a training path appears that updates weights from a corpus without consulting
  `corpus_training_blockers()` / `corpus_training_eligible()`, making the gate decorative.

Both are fenced by scanning the **working tree**, not `git ls-files`: an untracked file is exactly
the case a tracked-file scan would miss.

**These fences are review triggers, not enforcement.** They run in the test suite, so a violation
is caught before a change is accepted rather than at runtime, and a caller that never appears in a
test run is a caller the fence has not seen. `GATED_TRAINING_CALLERS` is empty because no training
path exists yet - that emptiness is the honest state of the repository, and the fixture tests
below show the fence firing on a synthetic trainer so the list cannot be quietly emptied further.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "src"), str(ROOT / "tools")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from nativevoice.core.errors import ConfigurationError  # noqa: E402
from nativevoice.eval.samples import EvalSample, SampleManifest  # noqa: E402

#: Anything that names a corpus descriptor location reads a corpus, whatever it then does with it.
CORPUS_MARKERS = ("artifacts/corpus", "CORPUS_ROOT", "legacy-batches.json")

#: The two accessors a corpus reader is required to consult: the training gate, and the
#: evaluation-scope reader that applies the closed `usable_for` vocabulary.
GATE_MARKERS = ("corpus_training_blockers", "corpus_training_eligible")
EVALUATION_MARKER = "corpus_evaluation_uses"

#: Reviewed readers. A new name here is a tracked-file change that a reviewer has to accept, and
#: listing a file is not a pass: it still has to consult both accessors.
CORPUS_MANIFEST_READERS = {"tools/corpus_forge.py"}

#: A deliberately narrow vocabulary. These are the shapes a training loop takes in the libraries
#: this project would use, not prose about training: `trainer` and `gradient` appear in the
#: corpus gate's own comments, so the wide words are excluded on purpose and a real loop cannot
#: hide behind that exclusion.
WEIGHT_UPDATE_MARKERS = (
    "optimizer",
    ".backward(",
    "torch.optim",
    "training_step",
    "save_pretrained",
    "Trainer",
    "train(",
    "fine_tune",
    "fine-tune",
    "num_epochs",
    "requires_grad",
)

#: Modules that update model weights and must therefore consult the training gate. Empty is the
#: current truth: this repository has no trainer.
GATED_TRAINING_CALLERS: set[str] = set()


def source_files(root: Path) -> list[Path]:
    """Every `.py` under `src/` and `tools/` in the tree handed in.

    The working tree, deliberately, rather than `git ls-files`: a new untracked tool is the case
    this fence exists for, and git would not report it.
    """
    found: list[Path] = []
    for base in ("src", "tools"):
        directory = Path(root) / base
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            parts = path.relative_to(root).parts
            if "__pycache__" in parts or any(part.startswith(".") for part in parts):
                continue
            found.append(path)
    return found


def source_of(path: Path, root: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def corpus_readers(root: Path = ROOT) -> set[str]:
    return {rel(path, root) for path in source_files(root) if any(marker in source_of(path, root) for marker in CORPUS_MARKERS)}


def readers_that_skip_the_accessors(root: Path = ROOT, allowlist: set[str] = CORPUS_MANIFEST_READERS) -> set[str]:
    """Listed readers that never call the gate, or never call the evaluation-scope reader.

    Being on the allowlist is the review step; consulting the accessors is the behaviour the
    review approved. A reader that does neither is unreviewed in the only way that matters.
    """
    offenders: set[str] = set()
    for path in source_files(root):
        name = rel(path, root)
        if name not in allowlist:
            continue
        text = source_of(path, root)
        consults_gate = any(marker in text for marker in GATE_MARKERS)
        consults_scope = EVALUATION_MARKER in text
        if not (consults_gate and consults_scope):
            offenders.add(name)
    return offenders


def training_callers_without_the_gate(root: Path = ROOT, allowlist: set[str] = GATED_TRAINING_CALLERS) -> set[str]:
    """Modules that look like a training loop and do not name the training gate anywhere."""
    offenders: set[str] = set()
    for path in source_files(root):
        text = source_of(path, root)
        if not any(marker in text for marker in WEIGHT_UPDATE_MARKERS):
            continue
        if any(marker in text for marker in GATE_MARKERS):
            continue
        offenders.add(rel(path, root))
    return offenders - set(allowlist)


def write_tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


class TestTheCorpusReaderFence:
    def test_exactly_one_module_reads_corpus_descriptors_today(self):
        assert corpus_readers() == CORPUS_MANIFEST_READERS, (
            "a new module opens a corpus descriptor; add it to CORPUS_MANIFEST_READERS only after "
            "reviewing what it does with usable_for, and make it consult both accessors first"
        )

    def test_the_listed_reader_consults_both_accessors(self):
        assert readers_that_skip_the_accessors() == set()

    def test_an_unlisted_reader_is_a_fence_failure_not_a_pass(self, tmp_path):
        # Non-vacuity: the scan is pointed at a tree with one new reader and it catches it. A
        # fence that only ever sees the file it was written for would pass forever.
        root = write_tree(
            tmp_path,
            {
                "tools/corpus_forge.py": "CORPUS_ROOT = 1\ncorpus_training_blockers\ncorpus_evaluation_uses\n",
                "tools/quiet_reader.py": "json.load(open('artifacts/corpus/en-synthetic-01/manifest.json'))\n",
            },
        )
        assert corpus_readers(root) == {"tools/corpus_forge.py", "tools/quiet_reader.py"}
        assert "tools/quiet_reader.py" not in CORPUS_MANIFEST_READERS

    def test_a_listed_reader_that_ignores_the_accessors_is_still_refused(self, tmp_path):
        # Being reviewed is not the same as behaving: an allowlisted file that opens descriptors
        # and never calls the gate or the scope reader fails on its own.
        root = write_tree(tmp_path, {"tools/scores_anyway.py": "artifacts/corpus/... json.load(...)\n"})
        assert readers_that_skip_the_accessors(root, allowlist={"tools/scores_anyway.py"}) == {"tools/scores_anyway.py"}

    def test_the_fence_survives_a_reader_that_only_calls_the_evaluation_side(self, tmp_path):
        root = write_tree(tmp_path, {"tools/scorer.py": "corpus_evaluation_uses({})\nartifacts/corpus\n"})
        assert readers_that_skip_the_accessors(root, allowlist={"tools/scorer.py"}) == {"tools/scorer.py"}

    def test_the_gate_is_reachable_from_the_reader_it_governs(self):
        # The allowlisted file is the one that *defines* the accessors, so the previous test could
        # pass on a self-reference. This one checks the definitions are really there and callable.
        from tools import corpus_forge as forge

        for name in ("corpus_training_blockers", "corpus_training_eligible", "corpus_evaluation_uses", "boundary_register"):
            assert callable(getattr(forge, name)), f"{name} is named by the fence but is not a function"


class TestTheTrainingCallerFence:
    def test_no_module_in_the_repository_updates_weights(self):
        assert training_callers_without_the_gate() == set(), (
            "a training-shaped module appeared; it must call corpus_training_eligible() (or "
            "corpus_training_blockers) and handle CorpusNotEligible before touching weights"
        )

    def test_a_trainer_that_never_names_the_gate_is_refused(self, tmp_path):
        root = write_tree(
            tmp_path,
            {"src/nativevoice/train/loop.py": "import torch.optim\noptimizer = build(params)\nloss.backward()\n"},
        )
        assert training_callers_without_the_gate(root) == {"src/nativevoice/train/loop.py"}

    def test_a_trainer_that_consults_the_gate_passes_the_fence(self, tmp_path):
        # The other side of the same fixture: the fence is a naming check, so the compliant shape
        # is shown rather than assumed. It still says nothing about whether that trainer is right.
        root = write_tree(
            tmp_path,
            {
                "src/nativevoice/train/loop.py": (
                    "from tools.corpus_forge import corpus_training_eligible\n"
                    "corpus_training_eligible(batch, clips)\nloss.backward()\n"
                )
            },
        )
        assert training_callers_without_the_gate(root) == set()

    def test_listing_a_trainer_does_not_excuse_it_from_the_gate(self, tmp_path):
        # `GATED_TRAINING_CALLERS` subtracts reviewed names, so the empty list is checked here as
        # well: reviewing a file that never names the gate must not silence the fence either.
        root = write_tree(tmp_path, {"tools/train.py": "optimizer = 1\n"})
        assert training_callers_without_the_gate(root, allowlist=set()) == {"tools/train.py"}

    def test_the_gate_marker_vocabulary_is_not_self_referential(self):
        # A marker list that matches prose in the gate's own comments would report the gate as a
        # trainer. `tools/corpus_forge.py` talks about gradients and trainers and must not trip it.
        from tools import corpus_forge as forge

        text = (Path(forge.__file__)).read_text(encoding="utf-8")
        assert any(word in text.lower() for word in ("trainer", "gradient")), "the corpus file is meant to contain the wide words"
        assert not [marker for marker in WEIGHT_UPDATE_MARKERS if marker in text], "the gate would be fenced by itself"


class TestTheQualityPathStillRefusesASyntheticCorpus:
    """M1's third pin: the evaluation-manifest reader cannot be mistaken for a domain scorer."""

    def corpus_rows(self) -> dict:
        manifest = json.loads((ROOT / "artifacts" / "corpus" / "en-synthetic-01" / "manifest.json").read_text(encoding="utf-8"))
        return manifest

    def test_sample_manifest_load_refuses_a_corpus_batch(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(self.corpus_rows()), encoding="utf-8")
        with pytest.raises(ConfigurationError) as caught:
            SampleManifest.load(str(path))
        message = str(caught.value)
        assert "unknown sample manifest key" in message
        for key in ("clip", "sha256", "seconds"):
            assert key in message, f"{key} is a corpus field and has to be named as the reason"

    def test_the_refusal_is_the_closed_schema_naming_every_corpus_field(self, tmp_path):
        # The sentence above is incidental to this reader: what the refusal actually rests on is
        # `EvalSample`'s closed field set, so the structured payload is pinned instead of the prose.
        # A corpus row shares no field name with an evaluation sample, which is why one read cannot
        # be mistaken for the other - and if a later change admits any of these keys to the schema,
        # this fails and the reason has to be re-reviewed rather than quietly inherited.
        fields = set(EvalSample.__dataclass_fields__)
        rows = self.corpus_rows()["samples"]
        unknown = {key for row in rows for key in row} - fields
        assert {"clip", "sha256", "seconds", "text", "wer", "cer", "kept", "status"} <= unknown
        assert not unknown & fields, "a corpus field has become a schema field"
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(self.corpus_rows()), encoding="utf-8")
        with pytest.raises(ConfigurationError) as caught:
            SampleManifest.load(str(path))
        assert caught.value.details["keys"] == sorted(unknown)
        assert caught.value.details["path"] == str(path)

    def test_the_refusal_is_the_schema_and_not_a_broken_fixture(self, tmp_path):
        # Non-vacuity, pointing the other way: the same loader on the same kind of path accepts a row
        # whose keys are all schema fields. So the corpus batch is turned away for being a corpus
        # batch, and no unrelated defect in this test is what makes the refusal look like a pass.
        path = tmp_path / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "name": "control",
                    "samples": [{"path": "a.wav", "language": "en", "transcript": "one two three"}],
                }
            ),
            encoding="utf-8",
        )
        manifest = SampleManifest.load(str(path))
        assert [sample.sample_id for sample in manifest.samples] == ["a.wav"]
        # Recorded as a limitation, not a design goal: the strictness is per row, so a descriptor's
        # batch-level rights fields (`usable_for`, `training_quarantine`) are simply never read here.
        # This reader is not a corpus gate; `corpus_training_blockers()` is.
        with_extra = tmp_path / "with-rights.json"
        with_extra.write_text(
            json.dumps(
                {
                    "name": "control",
                    "usable_for": ["training", "adaptation"],
                    "samples": [{"path": "a.wav", "language": "en", "transcript": "one two three"}],
                }
            ),
            encoding="utf-8",
        )
        assert [s.sample_id for s in SampleManifest.load(str(with_extra)).samples] == ["a.wav"]

    def test_the_corpus_tool_never_routes_a_batch_into_that_reader(self):
        from tools import corpus_forge as forge

        text = (Path(forge.__file__)).read_text(encoding="utf-8")
        assert "SampleManifest" not in text and "eval.samples" not in text

    def test_tools_evaluate_reads_no_corpus_descriptor(self):
        # The M1 measurement, pinned: `tools/evaluate.py` hand-writes its sentences, so there is no
        # code path in which a corpus batch is scored, and the permitted use stays a record.
        text = (ROOT / "tools" / "evaluate.py").read_text(encoding="utf-8")
        assert not any(marker in text for marker in CORPUS_MARKERS)
        tree = ast.parse(text)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "CORPUS" in names

    def test_nothing_outside_the_eval_package_references_sample_manifest(self):
        # `samples.py` defines it and `eval/__init__.py` re-exports it; neither calls it. A name
        # appearing anywhere else would be a caller, and today there is none - which is exactly why
        # `asr-domain-evaluation` is a permission on record rather than a runnable tool.
        definers = {"src/nativevoice/eval/samples.py", "src/nativevoice/eval/__init__.py"}
        offenders = {
            rel(path, ROOT) for path in source_files(ROOT) if "SampleManifest" in source_of(path, ROOT) and rel(path, ROOT) not in definers
        }
        assert offenders == set()

    def test_no_command_line_tool_loads_an_evaluation_manifest(self):
        tools = [rel(path, ROOT) for path in source_files(ROOT) if rel(path, ROOT).startswith("tools/")]
        assert tools, "the scan would otherwise prove nothing by looking at an empty set"
        assert not [name for name in tools if "SampleManifest" in (ROOT / name).read_text(encoding="utf-8")]
