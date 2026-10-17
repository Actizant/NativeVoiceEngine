"""Generate a speech corpus by synthesis, then gate it against the training boundary register.

Every clip is rendered by the approved TTS checkpoint, transcribed back by the ASR
engine, kept only when the transcription matches, and pinned with a SHA-256. The route
is the point: the sentences are authored in this repository, so no clip came from a
scraped dataset or a donor recording whose consent would have to be reconstructed. What
that does not settle is rights. A checkpoint's licence, the data bundled with it and any
redistribution right over the audio it renders are recorded as open questions, not as
permissions granted here: the boundary register decides whether a batch may train on it,
and a batch stays blocked until its boundary evidence and recomputable clip digests
exist. Nothing in this tool asserts that its output is owned, licensed or cleared.

The corpus is for **training and adaptation**, never for scoring quality. Synthetic
audio cannot measure intelligibility of human speech, and `eval/samples.py` already
refuses synthetic fixtures in a quality path - so `usable_for` is drawn from one closed
vocabulary, `ALLOWED_USES` ("training", "adaptation", "asr-domain-evaluation"), and a use
outside it is a blocker rather than a permission. The domain-evaluation entry is a permission
on record, not a capability: no tool in this repository scores a corpus batch today.

Training and evaluation text live in two named pools, `TRAIN_SENTENCES` and
`EVAL_SENTENCES`, and the tool refuses to forge unless they are disjoint under
`collisions()` - see the rule documented at `DISJOINTNESS_RULE`. This exists because of a
measured leak, not as a stylistic preference: `artifacts/corpus/en-synthetic-01` shipped a
clip whose sentence is *the same sentence* the round-trip evaluation scores
(`en_slot0_006.wav`, "Could you tell me the time and the nearest train station?"), and
three more kept clips were near-duplicates of evaluation sentences. A model trained on
text the evaluation then scores is not being evaluated.

That batch is now **quarantined from training** and its own sentences are excluded from the
pool: `corpus_training_blockers()` refuses any manifest that collides with a quarantined
record, and `TRAIN_SENTENCES` was rewritten from text the quarantined corpus never used.
The pool is a claim about *new* text only if the gate can read the quarantine; refusing to
forge when it cannot is deliberate. See `quarantined_corpus_texts()`.

A manifest's own boundary fields authorise nothing. `corpus_training_blockers()` anchors each
batch outside itself: either a `boundary_contract` header whose version and `pool_set_sha256`
match **this code's live pools and the tracked register** in
`artifacts/corpus/legacy-batches.json`, or a registered pre-contract ruling matched by the
digest of the manifest as the caller read it from disk - so editing a manifest invalidates its
own registration. See `boundary_provenance_blockers()`. Both are reviewable provenance, not
attestation: a digest proves byte identity and nothing else, certainly not licensing, consent,
authorship or a right to redistribute, and the register is a tracked file whose every change is
a diff a reviewer sees.

`corpus_training_eligible()` is the single named door a future trainer is required to walk
through. **No training path calls it yet**, so today it is a contract under test rather than
runtime enforcement, and this repository has no trainer to enforce it with.

Language scope is honest: English only, because that is what the approved synthesis
asset can render. Hindi and Bengali rows would be fabricated, and the schema is
written so they can be added the day a front-end exists for them.

Teacher targets are not generated here and the tool refuses the flag that would ask for
them: the recognition teacher's own terms are `NOASSERTION` and `redistributable=False`,
and no source-data rights have been resolved for any candidate audio. See
`TEACHER_TARGET_REFUSAL`.

    python tools/corpus_forge.py --slots 0 7 --out artifacts/corpus/en-synthetic-02
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nativevoice.audio import read_wav, resample, write_wav  # noqa: E402
from nativevoice.config import models_dir  # noqa: E402
from nativevoice.core.language import LanguageTag, Task  # noqa: E402
from nativevoice.models.registry import ModelRegistry  # noqa: E402
from nativevoice.stt.api import TranscriptionRequest  # noqa: E402
from nativevoice.tts.api import SynthesisRequest, VoiceReference  # noqa: E402

#: The published English round-trip evaluation sentences. Their source of truth is
#: `tools/evaluate.py` as `CORPUS["en"]`; the test suite compares this copy against that file
#: without importing it, because a pool that merely *paraphrases* the evaluation is the leak.
EVAL_SENTENCES = (
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore every summer.",
    "Please turn off the lights before you leave the room.",
    "I would like a cup of coffee and a slice of bread.",
    "The weather today is pleasant and the sky is clear.",
    "My favourite book is about a young engineer in the city.",
    "We walked to the park and watched the children play.",
    "Could you tell me the time and the nearest train station?",
)

#: Sentences rendered into training clips. This pool has been rewritten twice, for two
#: different reasons. Eight of the original sixteen went first: each is identical or a
#: near-duplicate under the rule below to a sentence in `EVAL_SENTENCES`, and four of those
#: eight had already been kept in `en-synthetic-01`, whose manifest still records them. The
#: replacement eight went next, on the owner's ruling of 2026-10-15: every one of them was a
#: sentence the quarantined batch had already generated - four as kept clips, four as dropped
#: records - and a pool that recycles quarantined text only re-derives audio for it. Nothing
#: below appears in that corpus in any form. `TestTrainingPoolReplacement` proves that against the
#: committed descriptor read off disk - `test_no_pool_sentence_appears_anywhere_in_the_quarantined_batch`
#: and `test_the_new_pool_text_is_not_something_the_quarantine_used_in_a_dropped_attempt` - rather
#: than against a list someone could edit to match, while
#: `TestTheLivePools::test_the_shipped_pools_are_disjoint` keeps the two pools apart.
TRAIN_SENTENCES = (
    "Our neighbour repairs bicycles in the garage every weekend.",
    "Two interns documented the outage timeline before Monday's review.",
    "The museum charged six euros for a guided evening tour.",
    "Grandmother labelled each jar with dates and a handwritten note.",
    "This corridor echoes whenever the ventilation fans start running.",
    "He parked the van behind the bakery to unload the crates.",
    "Nobody expected the ferry to arrive during the fog.",
    "We rehearsed the ensemble for forty minutes before sunset.",
)

#: Where committed corpus batches live. The gate reads it; nothing in this tool writes
#: anywhere except the directory named by `--out`.
CORPUS_ROOT = ROOT / "artifacts" / "corpus"

#: The batch the 2026-10-15 quarantine was applied to. Named because the mechanism below is
#: generic on purpose: a future batch can be quarantined the same way, and the gate must not
#: need editing to notice. Its audio is absent from this machine, so its digests cannot be
#: recomputed here and it stays blocked whatever else changes.
QUARANTINED_CORPUS = CORPUS_ROOT / "en-synthetic-01"

#: The legacy third-party evaluation records: the 13 vendor voice-preview names, their
#: digests, and the transcripts models produced from them. They are evidence about language
#: coverage and they are not training inputs, so every corpus manifest names them.
LEGACY_EVAL_ONLY_INPUTS = (
    "artifacts/evaluation/multilingual-inventory.json",
    "artifacts/evaluation/multilingual-base-default.json",
    "artifacts/evaluation/multilingual-base-int8.json",
    "artifacts/evaluation/multilingual-large-v3-int8.json",
    "artifacts/evaluation/multilingual-comparison.json",
)

#: Containment and Jaccard catch a sentence re-recorded with a clause added or trimmed. The
#: leading-word test catches what they both miss: two sentences that share their opening and
#: diverge in the tail - "…a young engineer in the city" against "…a young engineer building
#: a bridge" - share 8 of 10 words, which is below the Jaccard bar.
NEAR_DUPLICATE_JACCARD = 0.8
SHARED_PREFIX_TOKENS = 5
DISJOINTNESS_RULE = (
    "identical, or one word set containing the other, or word-set Jaccard >= 0.8, or at least "
    "5 leading words in common, after case folding and punctuation-to-space normalisation"
)

TRAINING_USE = "training"
ADAPTATION_USE = "adaptation"
EVALUATION_USE = "asr-domain-evaluation"
#: P3's closed vocabulary. An unrecognised string is a blocker rather than a pass-through: a
#: scope a tool has never heard of reads, to the next caller, as a permission nobody refused.
ALLOWED_USES = (TRAINING_USE, ADAPTATION_USE, EVALUATION_USE)
QUARANTINE_FIELD = "training_quarantine"
#: The only status that releases a batch. Anything else - including a field a hand edit left
#: blank or renamed - is treated as still quarantined, so clearing takes a deliberate word.
QUARANTINE_CLEARED = "CLEARED"
HEX_DIGEST_LENGTH = 64

#: The boundary contract this code defines. `DISJOINTNESS_RULE`, the two sentence pools and
#: this number move together: a rule edited in code is refused by
#: `TestTheBoundaryContract::test_a_register_recording_another_rule_is_refused`, and a pool edited
#: without bumping the version is refused by
#: `test_a_pool_this_code_no_longer_defines_fails_closed_even_when_it_is_registered`, because the
#: live pools - not the register - are the anchor. Bumping the version instead leaves the batch
#: refused by `test_a_pool_set_that_is_not_in_the_register_is_refused` and
#: `test_a_register_binding_the_pool_to_another_version_is_refused` until
#: `artifacts/corpus/legacy-batches.json` records the new pool set. That is a tracked file, so a
#: pool change cannot be authored inside a batch manifest.
SUPPORTED_BOUNDARY_VERSION = 1
CONTRACT_ISSUED_BY = "tools/corpus_forge.py"
#: The tracked register. A file rather than a `*/manifest.json` glob member, so the quarantine
#: reader never mistakes it for a batch.
BOUNDARY_REGISTER = CORPUS_ROOT / "legacy-batches.json"
REGISTER_SCHEMA_VERSION = 1
#: The two pools are digested as one value, so a manifest cannot pair a training pool from one
#: revision with an evaluation pool from another.
POOL_SET_SEPARATOR = "\x1f"

TEACHER_TARGET_REFUSAL = (
    "teacher-target generation is disabled: the recognition teacher is recorded as "
    "licence_spdx=NOASSERTION with redistributable=False, and no source-data rights are "
    "resolved for any candidate audio. Resolve both in docs/provenance.md before a single "
    "pseudo-label is written, and do not treat this refusal as a licence finding."
)

TEACHER_TARGET_RECORD = {
    "enabled": False,
    "status": "BLOCKED",
    "blocked_until": [
        "the teacher's own terms are resolved: the publisher tags the conversion MIT while "
        "openai/whisper-large-v3 is tagged Apache-2.0 and neither states which governs the "
        "converted weights",
        "source-data rights are resolved for every audio sample whose transcript could become a target",
    ],
    "refusal": TEACHER_TARGET_REFUSAL,
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def normalise(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split())


def words(text: str) -> list[str]:
    return normalise(text).split()


def near_duplicate(first: str, second: str) -> str:
    """Why these two sentences are the same example, or "" when a trainer may keep both.

    Only called once the pair is known not to be identical, so the reasons below describe
    partial overlap rather than duplicates.
    """
    a, b = set(words(first)), set(words(second))
    if not a or not b:
        return ""
    if a <= b or b <= a:
        return "one sentence's word set contains the other's"
    jaccard = len(a & b) / len(a | b)
    if jaccard >= NEAR_DUPLICATE_JACCARD:
        return f"word sets share {jaccard:.2f} of their vocabulary, bar is {NEAR_DUPLICATE_JACCARD}"
    shared = 0
    for left, right in zip(words(first), words(second)):
        if left != right:
            break
        shared += 1
    if shared >= SHARED_PREFIX_TOKENS:
        return f"{shared} leading words are identical, bar is {SHARED_PREFIX_TOKENS}"
    return ""


IDENTICAL_REASON = "the same sentence once punctuation, case and spacing are normalised"


def collisions(training: Sequence[str], evaluation: Sequence[str]) -> list[tuple[str, str, str]]:
    """Every training sentence that may not sit beside an evaluation sentence.

    Empty means the two pools are disjoint under `DISJOINTNESS_RULE`. The rule is stated as
    data rather than as a bare boolean because a rejected pool has to be explainable: the
    reason decides whether the fix is to move the sentence to the evaluation side or to
    rewrite it, and "it collided" answers neither.
    """
    breaches: list[tuple[str, str, str]] = []
    for candidate in training:
        for scored in evaluation:
            if normalise(candidate) == normalise(scored):
                breaches.append((candidate, scored, IDENTICAL_REASON))
                continue
            reason = near_duplicate(candidate, scored)
            if reason:
                breaches.append((candidate, scored, reason))
    return breaches


class CorpusBoundaryError(RuntimeError):
    """A quarantine record could not be read, so the boundary cannot be certified."""


class CorpusNotEligible(CorpusBoundaryError):
    """A corpus may not move model weights, and the message lists every reason why.

    The dedicated name exists so a future trainer can catch exactly this refusal and nothing
    else. It is not enforcement: see `corpus_training_eligible()`.
    """


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strings(value: object) -> bool:
    # An empty list fails: both callers read it as a scope declaration, and "no uses listed"
    # is a missing declaration rather than a well-formed empty one.
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item for item in value)


def _digest_shape(value: object) -> bool:
    return isinstance(value, str) and len(value) == HEX_DIGEST_LENGTH and all(c in "0123456789abcdef" for c in value.lower())


def pool_set_sha256(training: Sequence[str], evaluation: Sequence[str]) -> str:
    """One digest over both sentence pools as they are defined in code.

    Order-sensitive and normalised: the point is to bind a manifest to the exact text a
    reviewer read, so a shuffled pool is a different pool while a re-cased or re-punctuated one
    is not. The register in `artifacts/corpus/legacy-batches.json` carries the same digest,
    which is what turns "this manifest says it used the approved pool" into a claim checked
    against a tracked file rather than against itself.

    **What a digest here proves is byte identity, and nothing else.** Two matching digests say
    two pool definitions hold the same sentences. They say nothing about licensing, consent,
    authorship or redistribution rights, and this is reviewable provenance rather than
    cryptographic attestation: anyone who can edit the register can register a pool set, which
    is precisely why the register is a tracked file whose every change is a reviewable diff.
    """
    blob = POOL_SET_SEPARATOR.join(
        [" ".join(normalise(sentence) for sentence in training), " ".join(normalise(sentence) for sentence in evaluation)]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _register_pool_sets(document: dict, path: Path) -> list[dict]:
    entries = document.get("pool_sets")
    if not isinstance(entries, list):
        raise CorpusBoundaryError(f"the boundary register at {path} records no pool_sets list")
    for entry in entries:
        if not isinstance(entry, dict) or not _digest_shape(entry.get("pool_set_sha256")):
            raise CorpusBoundaryError(f"the boundary register at {path} has a pool set with no valid digest")
        if not _integer(entry.get("boundary_contract_version")) or entry["boundary_contract_version"] <= 0:
            raise CorpusBoundaryError(f"the boundary register at {path} has a pool set with no contract version")
        rule = entry.get("disjointness_rule")
        if not isinstance(rule, str) or not rule.strip():
            raise CorpusBoundaryError(f"the boundary register at {path} has a pool set with no recorded disjointness rule")
        for key in ("training_sentences", "evaluation_sentences"):
            if not _integer(entry.get(key)) or entry[key] <= 0:
                raise CorpusBoundaryError(f"the boundary register at {path} has a pool set with no {key}")
    return entries


def _register_batches(document: dict, path: Path) -> list[dict]:
    entries = document.get("pre_contract_batches")
    if not isinstance(entries, list):
        raise CorpusBoundaryError(f"the boundary register at {path} records no pre_contract_batches list")
    for entry in entries:
        if not isinstance(entry, dict):
            raise CorpusBoundaryError(f"the boundary register at {path} has a batch record that is not an object")
        if not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise CorpusBoundaryError(f"the boundary register at {path} has a batch record with no name")
        if not _digest_shape(entry.get("manifest_sha256")):
            raise CorpusBoundaryError(f"the boundary register at {path} names {entry['name']} without a valid manifest digest")
        status = entry.get("status")
        if not isinstance(status, str) or not status.strip():
            raise CorpusBoundaryError(f"the boundary register at {path} names {entry['name']} with no status ruling")
    return entries


def boundary_register(path: Path = BOUNDARY_REGISTER) -> dict:
    """Read the tracked boundary register: which pool sets are approved, and which batches predate them.

    Re-read on every call and never cached, like `quarantined_corpus_texts()`: a register
    updated after a build has to change the next answer. Damage raises rather than reading as
    an empty register - a gate that treats "I could not parse the approval file" as "nothing is
    approved" is the same failure as no gate, and here it would be a silent one.

    Absent counts as damage. Without the register there is no anchor outside the manifest, so
    no batch can be anchored, and the honest answer is a refusal rather than an empty list.
    """
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CorpusBoundaryError(f"cannot read the boundary register at {path}: {error}") from error
    if not isinstance(document, dict):
        raise CorpusBoundaryError(f"the boundary register at {path} is not a JSON object")
    if document.get("schema_version") != REGISTER_SCHEMA_VERSION:
        raise CorpusBoundaryError(
            f"the boundary register at {path} declares schema_version {document.get('schema_version')!r}, "
            f"not {REGISTER_SCHEMA_VERSION}"
        )
    return {"pool_sets": _register_pool_sets(document, path), "pre_contract_batches": _register_batches(document, path)}


def quarantined_corpus_texts(root: Path = CORPUS_ROOT, *, exclude: str = "") -> tuple[str, ...]:
    """Every sentence recorded by a corpus batch that is not cleared for training.

    Read on each call, never cached: a batch quarantined after a build has to block the next
    build rather than be remembered as clean. An unreadable record raises instead of being
    skipped - a gate that ignores a damaged quarantine file is the same failure as no gate,
    and here it would be a silent one. `exclude` names a batch whose own text is not reuse of
    something else, so a quarantined corpus is not reported for repeating itself.
    """
    texts: list[str] = []
    if not root.is_dir():
        return ()
    for path in sorted(root.glob("*/manifest.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CorpusBoundaryError(f"cannot read the corpus record at {path}: {error}") from error
        if not isinstance(document, dict):
            raise CorpusBoundaryError(f"the corpus record at {path} is not a JSON object")
        quarantine = document.get(QUARANTINE_FIELD)
        if not isinstance(quarantine, dict) or quarantine.get("status") == QUARANTINE_CLEARED:
            continue
        if document.get("name") == exclude:
            continue
        for field in ("samples", "dropped"):
            rows = document.get(field)
            if not isinstance(rows, list):
                raise CorpusBoundaryError(f"the quarantined record at {path} has no readable {field} list")
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("text"), str) and row["text"]:
                    texts.append(row["text"])
    return tuple(texts)


def corpus_evaluation_uses(manifest: dict) -> list[str]:
    """Which evaluation uses this descriptor permits, independent of any training decision.

    The quarantine stops gradient updates from reusing leaked text; it does not retire what
    the batch already measured, and register row A41's phase-1 probe relied on that. Keeping
    the two questions in separate functions is what makes the distinction enforceable rather
    than merely stated in a comment.

    Only a use from `ALLOWED_USES` is reported: an invented scope string is not an evaluation
    permission, and this is the function an evaluation reader is required to consult first.
    """
    uses = manifest.get("usable_for")
    if not _strings(uses):
        return []
    return [use for use in uses if use in ALLOWED_USES and use not in (TRAINING_USE, ADAPTATION_USE)]


def boundary_provenance_blockers(
    manifest: dict,
    *,
    boundary: object,
    register: dict | None,
    register_error: str = "",
    manifest_sha256: object = None,
    texts: Sequence[str] = (),
) -> tuple[str, ...]:
    """Whether this batch's boundary evidence is anchored outside the batch itself.

    Two ways to be anchored, and only two:

    * **A supported contract.** `training_boundary.boundary_contract` names a version and a
      pool-set digest, and both are checked against *this code* and against the tracked
      register: the version must equal `SUPPORTED_BOUNDARY_VERSION`, the digest must equal
      `pool_set_sha256(TRAIN_SENTENCES, EVAL_SENTENCES)` as it computes right now, that same
      digest must be listed in the register bound to this version, to these pool sizes and to
      this very `DISJOINTNESS_RULE`, and every kept sample must be a sentence of that registered
      pool. A registered pool set that
      this code no longer defines fails closed - the register records a retired pool, so the
      sentences a reviewer approved are not the sentences being shipped.
    * **A registered pre-contract ruling.** A batch written before the contract existed is
      eligible only if the caller supplies the digest of the manifest as read from disk, the
      register lists that name *and* that digest, and its recorded status is exactly `CLEARED`.
      A hand edit to the manifest changes its digest and invalidates its own entry, which is
      why the digest is the caller's to supply rather than something derived from the document
      under test.

    Everything else is a refusal, including an absent quarantine field, an unreadable register,
    and a batch that carries a valid-looking contract while the register still quarantines its
    name. `boundary_contract.issued_by` is deliberately *not* checked: it names who claims to
    have issued the contract, and a claim about an author is exactly the kind of self-declared
    field this gate exists to stop relying on.

    None of this attests anything. A digest proves byte identity - not licensing, consent,
    authorship or redistribution rights - and the register is a tracked file a reviewer edits,
    which makes it reviewable provenance rather than cryptographic attestation.
    """
    if register_error:
        return (f"the boundary register cannot be read, so no batch can be anchored to it: {register_error}",)
    if register is None:
        return ("no boundary register was supplied, so no batch can be anchored to a reviewed ruling",)
    blockers: list[str] = []
    name = manifest.get("name")
    if not isinstance(name, str) or not name.strip():
        blockers.append("the batch records no name, so no ruling in the register can be matched to it")
        name = ""
    ruling = next((entry for entry in register["pre_contract_batches"] if entry.get("name") == name), None)
    live_pool = pool_set_sha256(TRAIN_SENTENCES, EVAL_SENTENCES)
    contract = boundary.get("boundary_contract") if isinstance(boundary, dict) else None

    if contract is not None:
        if not isinstance(contract, dict):
            return ("training_boundary.boundary_contract is present but is not an object",)
        version = contract.get("version")
        if version != SUPPORTED_BOUNDARY_VERSION:
            blockers.append(
                f"training_boundary.boundary_contract.version is {version!r}, not the supported contract "
                f"version {SUPPORTED_BOUNDARY_VERSION}"
            )
        claimed = contract.get("pool_set_sha256")
        if not _digest_shape(claimed):
            blockers.append(
                "training_boundary.boundary_contract.pool_set_sha256 is missing or is not a 64-character "
                "hexadecimal digest"
            )
        elif claimed != live_pool:
            blockers.append(
                "training_boundary.boundary_contract names a pool set this code no longer defines: the "
                "registered sentences a reviewer approved are not the sentences in TRAIN_SENTENCES and "
                "EVAL_SENTENCES now"
            )
        else:
            entry = next((item for item in register["pool_sets"] if item["pool_set_sha256"] == claimed), None)
            if entry is None:
                blockers.append(
                    "the pool set this batch was forged under is not recorded in "
                    "artifacts/corpus/legacy-batches.json: a sentence-pool change needs a reviewed, tracked "
                    "register entry before any batch can claim it"
                )
            elif entry["boundary_contract_version"] != SUPPORTED_BOUNDARY_VERSION:
                blockers.append(
                    f"the register binds this pool set to contract version {entry['boundary_contract_version']}, "
                    f"not the supported version {SUPPORTED_BOUNDARY_VERSION}"
                )
            elif (entry["training_sentences"], entry["evaluation_sentences"]) != (
                len(TRAIN_SENTENCES),
                len(EVAL_SENTENCES),
            ):
                blockers.append(
                    "the register's pool sizes do not match the pools in code: "
                    f"{entry['training_sentences']}/{entry['evaluation_sentences']} against "
                    f"{len(TRAIN_SENTENCES)}/{len(EVAL_SENTENCES)}"
                )
            elif entry["disjointness_rule"] != DISJOINTNESS_RULE:
                # The rule travels with the pool: re-tightening or loosening
                # `DISJOINTNESS_RULE` silently changes what an already-registered contract
                # means, so the register names the rule a reviewer read and a mismatch is
                # refused until the tracked file says otherwise.
                blockers.append(
                    "the register records this pool set under a different disjointness rule than this code "
                    f"enforces: recorded {entry['disjointness_rule']!r}, enforced {DISJOINTNESS_RULE!r}"
                )
            else:
                approved = {normalise(sentence) for sentence in TRAIN_SENTENCES}
                for text in texts:
                    if normalise(text) not in approved:
                        blockers.append(f"a sample is not a sentence of the registered training pool: {text!r}")
        if ruling is not None and ruling.get("status") != QUARANTINE_CLEARED:
            blockers.append(
                f"the register records {name} as {ruling.get('status')!r}: a boundary contract written into a "
                "batch that is still quarantined on record does not release it"
            )
        return tuple(blockers)

    if ruling is None:
        blockers.append(
            f"{name or 'this batch'} has no supported boundary_contract and is not a registered pre-contract "
            "batch: the provenance of its boundary evidence is unknown"
        )
    elif not _digest_shape(manifest_sha256):
        blockers.append(
            "no caller-supplied digest of the manifest as read from disk, so its registration cannot be "
            "confirmed: a pre-contract batch is not eligible"
        )
    elif ruling["manifest_sha256"] != manifest_sha256:
        blockers.append(
            f"{name} is registered under manifest digest {ruling['manifest_sha256']!r} but the caller read "
            f"{manifest_sha256!r}: the document no longer matches the record that was reviewed, so its "
            "boundary evidence is unanchored"
        )
    elif ruling.get("status") != QUARANTINE_CLEARED:
        blockers.append(
            f"the register records {name} as {ruling.get('status')!r}, not {QUARANTINE_CLEARED!r}: clearing a "
            "pre-contract batch takes an explicit ruling in the tracked register"
        )
    return tuple(blockers)


def corpus_training_blockers(
    manifest: dict,
    *,
    clip_digests: Mapping[str, str] | None = None,
    manifest_sha256: str | None = None,
    register_path: Path = BOUNDARY_REGISTER,
) -> tuple[str, ...]:
    """Why this corpus may not update model weights. Empty means a trainer may proceed.

    Nothing here trusts the manifest's own conclusion. `usable_for`, `collisions_found` and
    the quarantine status are each re-checked against something outside themselves: the live
    evaluation pool, the quarantine records on disk, the tracked boundary register, and the
    arithmetic of the counts against the rows. The audio itself cannot be proven by a
    descriptor, so the caller that hashed it passes those digests in; without them the batch
    stays blocked, and no field written by the generator can change that.

    **The quarantine flags are records, not evidence** (P4). `training_quarantine.status`
    reading `CLEARED` and `digests_verified` reading `true` are each required, and neither can
    ever be the reason a batch is accepted: an absent quarantine field does not make a batch
    permissive either, because acceptance then has to come from a supported boundary contract
    or from a registered pre-contract ruling - see `boundary_provenance_blockers()`. Only the
    digests the caller read from the audio bytes count as evidence about audio, and a matching
    digest proves byte identity alone: not licensing, not consent, not authorship, not a right
    to redistribute.

    `manifest_sha256` is the caller's digest of the manifest file as read from disk. It is
    optional in the signature so existing callers keep compiling, and its absence is a blocker
    for any batch that relies on the register - a legacy batch cannot pass without it.
    """
    blockers: list[str] = []
    if not isinstance(manifest, dict):
        return ("a corpus manifest must be a JSON object",)

    uses = manifest.get("usable_for")
    if not _strings(uses):
        blockers.append("usable_for is missing, empty or not a list of non-empty strings")
        uses = []
    for required_use in (TRAINING_USE, ADAPTATION_USE):
        if required_use not in uses:
            blockers.append(f"usable_for does not claim {required_use!r}")
    unknown = [use for use in uses if use not in ALLOWED_USES]
    if unknown:
        blockers.append("usable_for names an unsupported use: " + ", ".join(sorted(unknown)))

    quarantine = manifest.get(QUARANTINE_FIELD)
    if quarantine is not None:
        if not isinstance(quarantine, dict):
            blockers.append("training_quarantine is present but is not an object")
        else:
            if quarantine.get("status") != QUARANTINE_CLEARED:
                blockers.append(f"training_quarantine status is {quarantine.get('status')!r}, not {QUARANTINE_CLEARED!r}")
            if quarantine.get("digests_verified") is not True:
                blockers.append("training_quarantine records digests_verified as anything other than true")

    if manifest.get("split") != TRAINING_USE:
        blockers.append("no split designation of 'training' is recorded")

    boundary = manifest.get("training_boundary")
    if not isinstance(boundary, dict):
        blockers.append("training_boundary is absent or malformed: this batch predates the disjointness contract")
    else:
        rule = boundary.get("disjointness_rule")
        if rule != DISJOINTNESS_RULE:
            blockers.append(
                "training_boundary.disjointness_rule is not the rule this code enforces: recorded "
                f"{rule!r}, enforced {DISJOINTNESS_RULE!r}"
            )
        pools = boundary.get("sentence_pools")
        if not isinstance(pools, dict):
            blockers.append("training_boundary.sentence_pools is missing or malformed")
        else:
            for key, live_size in (("training", len(TRAIN_SENTENCES)), ("evaluation", len(EVAL_SENTENCES))):
                if not _integer(pools.get(key)) or pools[key] <= 0:
                    blockers.append(f"training_boundary.sentence_pools.{key} is not a positive integer")
                elif pools[key] != live_size:
                    blockers.append(
                        f"training_boundary.sentence_pools.{key} declares {pools[key]} sentences, this code "
                        f"defines {live_size}"
                    )
            declared = pools.get("collisions_found")
            if not _integer(declared):
                blockers.append("training_boundary.sentence_pools.collisions_found is not an integer")
            elif declared != 0:
                blockers.append(f"training_boundary declares {declared} collisions, not zero")
        excluded = boundary.get("excluded_inputs")
        if not _strings(excluded):
            blockers.append("training_boundary.excluded_inputs is missing or malformed")
        else:
            missing = [item for item in LEGACY_EVAL_ONLY_INPUTS if item not in excluded]
            if missing:
                blockers.append("training_boundary.excluded_inputs omits " + ", ".join(missing))
            extras = [item for item in excluded if item not in LEGACY_EVAL_ONLY_INPUTS]
            if extras:
                blockers.append(
                    "training_boundary.excluded_inputs names inputs no review has recorded as excluded: "
                    + ", ".join(sorted(extras))
                )
        targets = boundary.get("teacher_targets")
        if not isinstance(targets, dict):
            blockers.append("training_boundary.teacher_targets is missing or malformed")
        elif targets.get("enabled") is not False:
            blockers.append(
                "training_boundary.teacher_targets.enabled is not false while teacher-use and source-data rights are unresolved"
            )

    samples = manifest.get("samples")
    texts: list[str] = []
    if not isinstance(samples, list) or not samples:
        blockers.append("samples is absent, malformed or empty")
        samples = []
    seen: set[str] = set()
    for position, row in enumerate(samples):
        if not isinstance(row, dict):
            blockers.append(f"samples[{position}] is not an object")
            continue
        clip = row.get("clip")
        if not isinstance(clip, str) or not clip:
            blockers.append(f"samples[{position}] records no clip name")
            continue
        if clip in seen:
            blockers.append(f"{clip} is listed more than once")
        seen.add(clip)
        if not _digest_shape(row.get("sha256")):
            blockers.append(f"{clip} carries no 64-character hexadecimal digest")
        if not isinstance(row.get("seconds"), (int, float)) or isinstance(row.get("seconds"), bool) or row["seconds"] <= 0:
            blockers.append(f"{clip} records no positive duration")
        if row.get("kept") is not True:
            blockers.append(f"{clip} is listed as a training sample but is not marked kept")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            blockers.append(f"{clip} records no transcript text")
        else:
            texts.append(text)

    # The anchor is checked after the rows are parsed because a contract batch is verified
    # against its own sample texts: naming an approved pool is not the same as drawing its
    # sentences from it.
    try:
        register: dict | None = boundary_register(register_path)
        register_error = ""
    except CorpusBoundaryError as error:
        register, register_error = None, str(error)
    blockers.extend(
        boundary_provenance_blockers(
            manifest,
            boundary=boundary,
            register=register,
            register_error=register_error,
            manifest_sha256=manifest_sha256,
            texts=texts,
        )
    )

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        blockers.append("counts is absent or malformed")
    else:
        generated, kept, dropped = (counts.get(key) for key in ("generated", "kept", "dropped"))
        if not all(_integer(value) for value in (generated, kept, dropped)):
            blockers.append("counts.generated, counts.kept and counts.dropped must all be integers")
        else:
            if generated != kept + dropped:
                blockers.append(f"counts.generated {generated} is not kept {kept} plus dropped {dropped}")
            if kept != len(samples):
                blockers.append(f"counts.kept {kept} does not match the {len(samples)} listed samples")
            records = manifest.get("dropped")
            if not isinstance(records, list):
                blockers.append("dropped is absent or malformed: the failures the batch hides were never recorded")
            elif len(records) != dropped:
                blockers.append(f"counts.dropped {dropped} does not match the {len(records)} dropped records")

    for field in ("licence_basis", "consent_basis"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            blockers.append(f"{field} is missing or empty: the batch's provenance is undeclared")
    if manifest.get("synthetic") is not True:
        blockers.append("synthetic is not recorded as true")
    for section in ("generator", "verifier"):
        entry = manifest.get(section)
        if not isinstance(entry, dict) or not isinstance(entry.get("model_id"), str) or not entry["model_id"].strip():
            blockers.append(f"{section}.model_id is missing: the asset that produced or checked this batch is unnamed")

    for candidate, _scored, reason in collisions(texts, list(EVAL_SENTENCES)):
        blockers.append(f"a sample collides with the evaluation pool ({reason}): {candidate!r}")
    try:
        # A batch is not reuse of itself: excluding its own name keeps the honest blockers -
        # the collision with the evaluation pool - and reports reuse only across batches.
        quarantined = quarantined_corpus_texts(exclude=str(manifest.get("name") or ""))
    except CorpusBoundaryError as error:
        blockers.append(f"the quarantine records cannot be read, so reuse cannot be ruled out: {error}")
    else:
        for candidate, _scored, reason in collisions(texts, list(quarantined)):
            blockers.append(f"a sample reuses quarantined corpus text ({reason}): {candidate!r}")

    if clip_digests is None:
        blockers.append("no clip digests were supplied by the caller: a manifest cannot prove its own audio exists")
    elif not isinstance(clip_digests, Mapping):
        blockers.append("clip_digests must map a clip name to the digest the caller read from its bytes")
    else:
        for row in samples:
            if not isinstance(row, dict) or not isinstance(row.get("clip"), str):
                continue
            clip = row["clip"]
            if clip not in clip_digests:
                blockers.append(f"{clip} was not read by the caller: its bytes are unverified")
            elif clip_digests[clip] != row.get("sha256"):
                blockers.append(f"{clip} digest mismatch: the caller read {clip_digests[clip]!r}, the manifest records {row.get('sha256')!r}")

    return tuple(blockers)


def read_clip_digests(directory: Path, names: Sequence[str]) -> dict[str, str]:
    """Hash the audio a manifest describes, so a caller can answer the gate with facts.

    A missing file raises: the gate asks for a digest of every clip a batch lists, and a silent
    gap would report an absent corpus as merely an unverified one.
    """
    directory = Path(directory)
    digests = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"no audio for {name} at {path}")
        digests[name] = digest(path)
    return digests


def corpus_training_eligible(
    manifest_path: Path | str,
    audio_dir: Path | str,
    *,
    register_path: Path | str = BOUNDARY_REGISTER,
) -> None:
    """Ask whether a corpus batch may update model weights, reading the evidence yourself.

    THE FUTURE TRAINING ENTRY POINT - NOT PRODUCTION ENFORCEMENT. Nothing in `src/` or `tools/`
    calls this, because no training path exists yet. Adding a trainer means calling this and
    handling `CorpusNotEligible`; `tests/test_corpus_reader_fence.py` fails if a module that
    touches weights appears without doing either, and that fence is a review trigger, not a
    runtime guard.

    It exists as one named door rather than a bare blocker list because the evidence has to be
    gathered in the right order for the gate to mean anything: the manifest is read once and its
    bytes hashed for the register comparison, every clip a batch lists is re-hashed from disk,
    and unreadable input raises instead of reading as an unverified-but-usable batch. A caller
    that skipped either step would be answering the gate with the manifest's own claims.

    Returns None when the batch is eligible and raises `CorpusNotEligible` listing every
    blocker otherwise. Eligibility here is a provenance statement: it says the boundary evidence
    matches a reviewed register and live pools, and it says nothing about licensing, consent,
    authorship or redistribution rights, which the corpus licence and consent fields only assert.
    """
    path = Path(manifest_path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CorpusNotEligible(f"the corpus manifest at {path} cannot be read: {error}") from error
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise CorpusNotEligible(f"the corpus manifest at {path} is not readable JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise CorpusNotEligible(f"the corpus manifest at {path} is not a JSON object")

    rows = manifest.get("samples")
    clips = [
        row["clip"]
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and isinstance(row.get("clip"), str) and row["clip"]
    ]
    try:
        clip_digests = read_clip_digests(Path(audio_dir), clips)
    except OSError as error:
        raise CorpusNotEligible(
            f"the audio behind {path} cannot be read, so no clip digest can be verified: {error}"
        ) from error

    blockers = corpus_training_blockers(
        manifest,
        clip_digests=clip_digests,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        register_path=Path(register_path),
    )
    if blockers:
        name = manifest.get("name") or path.parent.name
        raise CorpusNotEligible(
            f"{name} is not eligible for training: " + "; ".join(blockers) + ". A digest match would prove byte "
            "identity only, never a right to use the audio"
        )


def build_corpus_manifest(
    *,
    name: str,
    rows: list[dict],
    wall_seconds: float,
    slots: Sequence[int],
    tts_manifest,
    stt_manifest,
    max_wer: float,
) -> dict:
    """The corpus descriptor, as one object so the training boundary is testable.

    The boundary fields are the reason this is a function rather than a literal inside the
    run: `excluded_inputs`, `teacher_targets` and `sentence_pools` are the contract a later
    training tool reads, and a contract nothing constructs cannot be checked. The
    `boundary_contract` header is written from the pools as this module defines them, so a
    batch always carries the pool set it was actually forged from - and the gate still verifies
    that header against the live pools and the tracked register rather than believing it.
    """
    kept = [row for row in rows if row.get("kept")]
    return {
        "name": name,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "language": "en",
        "kind": "synthetic",
        "split": "training",
        "usable_for": ["training", "adaptation", "asr-domain-evaluation"],
        "not_usable_for": ["human speech quality claims", "intelligibility scoring of real speech"],
        "synthetic": True,
        "licence_basis": (
            "Audio derived from an Apache-2.0 pretrained checkpoint (Kokoro-82M) selected for local "
            "inference; transcripts authored for this project. No scraped audio, no donor recordings, "
            "no third-party dataset. Redistribution of derived models must respect the checkpoint's "
            "Apache-2.0 attribution and the bundled espeak-ng GPL-3.0 data terms."
        ),
        "consent_basis": "not applicable: no human speaker contributed a recording to this corpus",
        "training_boundary": {
            "disjointness_rule": DISJOINTNESS_RULE,
            # Not a claim the batch gets to make about itself: `version` and `pool_set_sha256`
            # are checked against this code and against artifacts/corpus/legacy-batches.json,
            # so writing them into a manifest authorises nothing. `issued_by` records who the
            # generator says issued the contract, and the gate deliberately does not read it.
            "boundary_contract": {
                "version": SUPPORTED_BOUNDARY_VERSION,
                "issued_by": CONTRACT_ISSUED_BY,
                "pool_set_sha256": pool_set_sha256(TRAIN_SENTENCES, EVAL_SENTENCES),
            },
            "sentence_pools": {
                "training": len(TRAIN_SENTENCES),
                "evaluation": len(EVAL_SENTENCES),
                "collisions_found": 0,
            },
            # Named in the artefact, not only in the code: a future tool that assembles a
            # training set reads this list and never has to guess whether these are inputs.
            "excluded_inputs": list(LEGACY_EVAL_ONLY_INPUTS),
            "excluded_inputs_basis": (
                "legacy third-party voice-preview evaluation records: they carry speaker names, "
                "source digests and model transcripts of audio with no licence, consent basis or "
                "redistribution statement recorded, and they are evidence about coverage only"
            ),
            # Copied, not shared: the constant is a module-level record, so two built
            # manifests that aliased it would let a caller's edit to one silently change the
            # other - and a training gate reads `enabled` as a refusal.
            "teacher_targets": copy.deepcopy(TEACHER_TARGET_RECORD),
        },
        "generator": {
            "engine": tts_manifest.engine,
            "model_id": tts_manifest.model_id,
            "licence": tts_manifest.provenance.licence_spdx,
            "slots": list(slots),
        },
        "verifier": {
            "engine": stt_manifest.engine,
            "model_id": stt_manifest.model_id,
            "licence": stt_manifest.provenance.licence_spdx,
            "policy": f"clip kept only when round-trip WER <= {max_wer}",
        },
        "counts": {"generated": len(rows), "kept": len(kept), "dropped": len(rows) - len(kept)},
        # dropped clips are listed, not merely counted: a corpus that only records its
        # successes hides the rate at which the generator and the verifier disagree
        "dropped": [
            {
                "clip": f"en_slot{row['slot']}_{row['index']:03d}.wav",
                "wer": row.get("wer"),
                "text": row.get("text"),
                "heard": row.get("heard"),
            }
            for row in rows
            if not row.get("kept")
        ],
        "wall_seconds": round(wall_seconds, 2),
        "samples": kept,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slots", type=int, nargs="+", default=[0], help="speaker slots to render each sentence with")
    # A new default batch, deliberately: the pool changed shape, and forging into
    # en-synthetic-01 would overwrite the committed manifest that records what that corpus
    # actually contained. Nothing here rewrites existing evidence.
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "corpus" / "en-synthetic-02"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-wer", type=float, default=0.0, help="keep only clips whose own transcription matches exactly")
    parser.add_argument(
        "--teacher-targets",
        action="store_true",
        help="ask for teacher transcripts as targets; refused until teacher and source-data rights are resolved",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Both checks run before the registry is read and before an engine exists, so a refused
    # run is a refused run: it touches no model, writes nothing, and costs nothing.
    if args.teacher_targets:
        print(TEACHER_TARGET_REFUSAL, file=sys.stderr)
        return 2
    breaches = collisions(TRAIN_SENTENCES, EVAL_SENTENCES)
    if breaches:
        print(f"training and evaluation pools are not disjoint under the rule: {DISJOINTNESS_RULE}", file=sys.stderr)
        for candidate, scored, reason in breaches:
            print(f"  {reason}\n    training:   {candidate}\n    evaluation: {scored}", file=sys.stderr)
        return 2
    try:
        quarantined = list(quarantined_corpus_texts())
    except CorpusBoundaryError as error:
        print(f"the quarantine records are unreadable, so reuse cannot be ruled out: {error}", file=sys.stderr)
        return 2
    reused = collisions(TRAIN_SENTENCES, quarantined)
    if reused:
        print("the training pool reuses text from a batch that is not cleared for training", file=sys.stderr)
        for candidate, scored, reason in reused:
            print(f"  {reason}\n    training: {candidate}\n    quarantined: {scored}", file=sys.stderr)
        return 2
    # The run stamps the pool it is about to render into the manifest's contract header, so the
    # pool has to be one a reviewer already recorded. Refusing here is what makes a pool rewrite
    # a tracked-register change rather than an edit to one file: forging first and registering
    # afterwards would let the manifest authorise itself.
    try:
        register = boundary_register()
    except CorpusBoundaryError as error:
        print(f"the boundary register cannot be read, so no pool set is authorised: {error}", file=sys.stderr)
        return 2
    live_pool = pool_set_sha256(TRAIN_SENTENCES, EVAL_SENTENCES)
    entry = next((item for item in register["pool_sets"] if item["pool_set_sha256"] == live_pool), None)
    rule_matches = entry is not None and entry["disjointness_rule"] == DISJOINTNESS_RULE
    if entry is None or entry["boundary_contract_version"] != SUPPORTED_BOUNDARY_VERSION or not rule_matches:
        print(
            "the current sentence pools and the enforced disjointness rule are not recorded in "
            f"artifacts/corpus/legacy-batches.json under contract version {SUPPORTED_BOUNDARY_VERSION}: a batch "
            "forged now would be born ineligible, so record the pool set - and the ruling behind it - first",
            file=sys.stderr,
        )
        return 2
    if (entry["training_sentences"], entry["evaluation_sentences"]) != (len(TRAIN_SENTENCES), len(EVAL_SENTENCES)):
        print(
            "the boundary register records pool sizes that do not match the pools in code: "
            f"{entry['training_sentences']}/{entry['evaluation_sentences']} against "
            f"{len(TRAIN_SENTENCES)}/{len(EVAL_SENTENCES)}",
            file=sys.stderr,
        )
        return 2

    registry = ModelRegistry(models_dir())
    manifests = {item.model_id: item for item in registry.all()}
    tts_id = next((key for key in manifests if "kokoro" in key), None)
    stt_id = next((key for key in manifests if "whisper-base" in key), None)
    if not tts_id or not stt_id:
        print("no approved TTS/STT manifest found; run tools/provision.py first")
        return 1
    for model_id in (tts_id, stt_id):
        status = registry.status(model_id)
        if not status.usable:
            print(f"{model_id} is not usable: {status.explain()}")
            return 1

    from nativevoice.stt.adapters.faster_whisper_engine import FasterWhisperEngine
    from nativevoice.tts.adapters.sherpa_kokoro import SherpaKokoroEngine

    out = Path(args.out)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    synthesizer = SherpaKokoroEngine(manifest=manifests[tts_id], num_threads=args.threads)
    transcriber = FasterWhisperEngine(manifest=manifests[stt_id], cpu_threads=args.threads)
    synthesizer.warm("en")
    transcriber.warm()

    tts_manifest = manifests[tts_id]
    stt_manifest = manifests[stt_id]
    rows = []
    started = time.perf_counter()
    for slot in args.slots:
        reference = VoiceReference(
            voice_id=f"corpus.slot{slot}", speaker_slot=slot, source=f"{tts_id} speaker {slot}"
        )
        for index, sentence in enumerate(TRAIN_SENTENCES):
            outcome = synthesizer.synthesize(
                SynthesisRequest(text=sentence, language=LanguageTag("en"), voice=reference)
            )
            if outcome.audio.frame_count == 0:
                rows.append({"index": index, "slot": slot, "status": "empty", "text": sentence})
                continue
            name = f"en_slot{slot}_{index:03d}.wav"
            path = write_wav(str(clips / name), resample(outcome.audio, 16000))
            feed = Path(path)
            heard = transcriber.transcribe(
                TranscriptionRequest(
                    audio=read_wav(str(feed)),
                    language=LanguageTag("en"),
                    allow_auto_detection=False,
                    max_duration_s=30.0,
                )
            )
            from nativevoice.eval.transcription import score as score_pair

            result = score_pair(reference=sentence, hypothesis=heard.text, language=LanguageTag("en"))
            keep = float(result.wer) <= args.max_wer
            rows.append(
                {
                    "index": index,
                    "slot": slot,
                    "text": sentence,
                    "heard": heard.text,
                    "normalised_reference": normalise(sentence),
                    "normalised_heard": normalise(heard.text),
                    "wer": round(float(result.wer), 5),
                    "cer": round(float(result.cer), 5),
                    "exact_match": normalise(sentence) == normalise(heard.text),
                    "kept": keep,
                    "clip": name if keep else "",
                    "sha256": digest(feed) if keep else "",
                    "seconds": round(outcome.audio.duration_s, 4),
                    "sample_rate": 16000,
                    "status": "ok",
                }
            )
            print(f"  slot {slot} clip {index:02d} wer={result.wer:.3f} {'kept' if keep else 'dropped'}", flush=True)

    synthesizer.close()
    transcriber.close()
    kept = [row for row in rows if row.get("kept")]
    manifest = build_corpus_manifest(
        name=out.name,
        rows=rows,
        wall_seconds=time.perf_counter() - started,
        slots=args.slots,
        tts_manifest=tts_manifest,
        stt_manifest=stt_manifest,
        max_wer=args.max_wer,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"corpus {out.name}: kept {len(kept)} of {len(rows)} clips, {sum(r['seconds'] for r in kept):.1f}s of audio")
    print(f"manifest -> {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
