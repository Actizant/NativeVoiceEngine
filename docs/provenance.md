# Dependency and asset provenance inventory

Per §3 rules 7–10, code dependencies, pretrained weights, training datasets, tokenizer assets, speaker
embeddings and voice recordings are inventoried as **separate** licensing questions. This file is the record;
`models/api.py` is the enforcement.

Status vocabulary: `RESIDENT` already on this machine · `ABSENT` not installed · `APPROVED` reviewed and
cleared by a named approver · `UNREVIEWED` present but not cleared · `PROHIBITED` must not enter the product.

**A licence string below marked "per upstream" was recorded from the upstream project's own declaration and
has not been re-verified against the released artefact on this machine.** Rule 9 forbids claiming an asset is
legally safe without sufficient evidence, so those rows are `UNREVIEWED` regardless of how permissive the
licence looks, until someone reads the actual `LICENSE` of the exact version.

## 1. Python libraries the deliverable imports

| Package | Version | Where imported | Licence (per upstream) | Verification | Status |
|---|---|---|---|---|---|
| *standard library only* | 3.14.7 | `core`, `audio`, `models`, `hardware`, `config`, `language`, `stt`, `tts`, `profiles`, `session`, `eval`, `diagnostics`, `integration` | PSF / Python-2.0 | n/a | **no third-party runtime dependency** |
| `sounddevice` | 0.5.6 | lazily, only `audio/io.py` live capture+playback | MIT | not re-read | RESIDENT, OPTIONAL |
| `ctypes`/`subprocess` → `nvidia-smi` | — | `hardware/detect.py` | NVIDIA driver utility, not linked | — | RESIDENT, OPTIONAL, read-only |

Nothing in the control path imports numpy, torch, onnxruntime, psutil or fastapi. Where an optional
accelerator is used the absence must degrade gracefully, and the test suite runs with it absent.

## 2. Libraries present on this machine that the deliverable does NOT import

Recorded because their presence invites use, and each would be a provenance decision.

| Package | Version | Licence (per upstream) | Why it is not imported |
|---|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause | control path stays dependency-free; PCM maths is stdlib |
| onnxruntime | 1.30.0 (CPU) | MIT | no ONNX model approved; also would imply an execution-provider story we have not measured |
| faster-whisper | 1.2.1 | MIT | candidate STT *instrument* only — see S1; would become a runtime dependency if wired to the product path |
| sherpa_onnx | 1.13.8 | Apache-2.0 | no sherpa model resident; **see the copyleft note in §5** |
| psutil | 7.2.2 | BSD-3-Clause | hardware detection falls back to `ctypes`/`os`, so it is optional |
| fastapi / uvicorn / httpx | 0.141.1 / — / — | MIT | this is a library, not a service; no HTTP surface is offered |
| torch | **ABSENT** | BSD-3-Clause | required by any synthesis model trained in this project; not installed (S2) |
| soundfile | **ABSENT** | BSD-3-Clause | WAVE container is parsed directly; `audioop` is gone from 3.13+ |
| ruff, mypy | **ABSENT** | MIT / MIT | dev tooling; static checks are a hand-pass until S2 is answered |

## 3. Pretrained model assets

| Asset | Where | Bytes | Licence (per upstream) | Provenance risk | Status |
|---|---|---|---|---|---|
| Systran/faster-whisper-base | `~/.cache/huggingface` | 141.0 MB | Apache-2.0 (model card) / MIT (whisper.cpp & OpenAI code, to confirm) | converted from OpenAI weights by a third party; no digest recorded locally | **UNREVIEWED** |
| faster-whisper-base.en | same | 140.9 MB | as above | as above | UNREVIEWED |
| faster-whisper-small / small.en | same | 463.7 / 463.6 MB | as above | as above | UNREVIEWED |
| faster-whisper-medium | same | 1,459.7 MB | as above | as above | UNREVIEWED |
| faster-whisper-large-v3 | same | 2,947.7 MB | as above | cannot fit 4 GB VRAM alongside live apps; 128-lang claim is upstream's, not measured | UNREVIEWED |
| any sherpa-onnx ASR (Parakeet/Nemotron/Cohere) | **not present** | — | model licences vary; OpenWhispr records only one of them | would require download | ABSENT, not requested |
| Kyutai pocket-tts weights | **not present** | — | **not stated in the repository**; cloning set gated behind accepting terms; ungated mirror exists | gated/ungated fork changes capability silently | ABSENT, **do not fetch** |
| Pocket-TTS preset voices (26) | not present | — | per-voice consent not disclosed in repo; derived from VCTK / EXPRESSO / EARS / Mozilla Common Voice / VoiceZero / donations / one dev recording | consent basis unknown per voice | ABSENT, **PROHIBITED pending review** |
| HiFiTTS-2 (reference training set) | not present | — | NVIDIA research licence | redistribution rights unclear | ABSENT, **PROHIBITED in product** |
| Silero VAD / pyannote / 3D-Speaker / MiniLM (OpenWhispr's assets) | not present | — | **no licence recorded in their registry**; pyannote and Silero have changed upstream terms | untraceable from the archive | ABSENT, **do not adopt** |
| **Weights owned by this project** | to be created | — | ours | requires the training pipeline in §4 | **NONE EXIST YET** |

The honest summary: **there is not a single approved speech weight in this project.** Recognition and
synthesis quality numbers therefore cannot be produced without your S1 ruling, and the capability matrix
rightly reports `NO_MODEL` rather than an estimate.

## 4. Tokenizers, embeddings, and the training data they imply

- No tokenizer asset is shipped. TTS text processing uses our own normalisation tables and a byte/word
  budget counter; a real subword tokenizer is a training-time artefact that must be recorded separately
  (`AssetKind.TOKENIZER`) when a model is trained in this project.
- No speaker embedding or KV-cache voice state is shipped. `profiles/` stores user recordings plus consent
  records plus digests; the acoustic conditioning that would turn those into a voice is gated on an approved
  synthesis model and currently raises with the missing component named.
- Evaluation corpora required by §14, none present: paired audio+transcript for English (open, licence
  clearable), **Bengali and Hindi (must be approved or user-supplied)**, and cross-language identity pairs
  for cloning evaluation. `eval/samples.py` records per-sample licence and consent and refuses to let a
  synthetic fixture contribute to a quality score.
- **Training inputs are a separate question from evaluation evidence, and §7 is the boundary.** Nothing in
  §4 that is labelled evaluation may be read as a training corpus, including the five records named there.

## 5. Copyleft and supply-chain notes carried forward from the reference study

- sherpa-onnx **TTS builds statically link espeak-ng (GPL-3.0)**; OpenWhispr deliberately switched to
  `-no-tts` archives for this reason. Any future TTS runtime choice must repeat that check rather than
  assume an Apache-2.0 wrapper makes the linked tree permissive.
- ONNX Runtime in the reference stack came from a **third-party GitHub mirror**, not Microsoft. Any
  binary we ever ship must name its actual origin, and a mirror is not a provenance.
- `nircmd.exe` in OpenWhispr's Windows package is **"free for non-commercial use"** while their README
  claims commercial freedom. That is the concrete shape of the risk rule 9 exists to prevent, and it is why
  the gate here is default-deny in code.
- Their Whisper GGML downloads were **size-checked, not hash-verified**; ours must record digests, and an
  unrecorded digest reports as *unverifiable*, never as verified.

## 6. Reference projects themselves

| Project | Version | Licence | Our use |
|---|---|---|---|
| OpenWhispr (Gizmo Labs) | 1.10.3 | MIT (`LICENSE:1`) | **research reference only.** No file imported, vendored, ported or renamed. Working copy is a local research copy held outside this repository; its licence terms were not independently re-reviewed here, so this row records attribution, not clearance. |
| Pocket TTS (Kyutai) | 3.3.0 | MIT text present but **no copyright-holder line and no `license` field in `pyproject.toml`** | research reference only, same terms. The missing attribution is itself recorded as an ambiguity. |

MIT permits use of their *code* with attribution; §3 rules 1–4 forbid that here regardless, because the
deliverable must be original. Their architectures informed decisions documented with reasoning in
`docs/research-report.md` §4 and `docs/architecture.md` §3 — concepts, not transplants.

## 7. Legacy evaluation-only evidence: never a training input (recorded 2026-10-15)

Five committed records describe the 13 third-party voice-product previews behind register row **A50**. They
are evidence about language coverage and instrument behaviour. **They are not training data, and no future
training manifest may cite them as inputs:**

| Path | What it is | Why it is evaluation-only |
|---|---|---|
| `artifacts/evaluation/multilingual-inventory.json` | 13 sample paths (redacted), `sha256_before`, byte counts, 16 kHz working-copy decode records, `total_audio_seconds` 209.74 | An inventory of other parties' audio. It records digests and locations, not rights. |
| `artifacts/evaluation/multilingual-base-default.json` | One recognition pass over those 13 clips | Carries model-generated transcripts of unlicensed audio. |
| `artifacts/evaluation/multilingual-base-int8.json` | The same 13 clips, quantised compute | as above |
| `artifacts/evaluation/multilingual-large-v3-int8.json` | The same 13 clips, the candidate teacher | as above, and it is the record that would most easily be mistaken for pseudo-labels |
| `artifacts/evaluation/multilingual-comparison.json` | Rows comparing the three passes | Its `reference_transcript` field reads `"none"`: there is no cleared label to train on, only machine output |

**Measured on 2026-10-15, from the committed bytes:** 13 working copies, 209.736 s decoded audio (the
inventory rounds this to 209.74), 5,867,002 recorded source bytes. Eleven are vendor-titled previews totalling
≈160.05 s; two carry an `ULTRON` timestamp in the filename and total 49.685 s.

**Status of those five files: retained, unchanged, and not inputs.** Not deleted, not untracked, not
regenerated by this change. A50's disposition stands as written — read in place, hashed before and after,
never copied into the repository. One deviation from the row's wording is worth recording: it describes 16 kHz
working copies living outside the repository at `../_multilingual_eval`, and on this machine that
directory and the source attachments were **both measured absent** on 2026-10-15, so today only the committed
text records survive. Absence of the audio is not a licence finding, and it does not make the transcripts
cleaner.

**The two ULTRON-dated previews are deliberately unclassified.** Their filenames are the only evidence in the
repo that ties them to this project, their script is not in the tree, and their timestamps sit inside the same
download window as the vendor files. They are **neither cleared nor prohibited here**; that ruling needs the
provenance of the recording itself. Likewise nothing in this section is a clearance claim: no licence, consent
basis or redistribution right has been established for any of the 13 clips, and the transcripts are machine
output, so they cannot be treated as human-verified targets.

**Legacy text leaks into a corpus by way of the sentences, not only the audio.** `artifacts/corpus/en-synthetic-01/manifest.json`
records Kokoro-82M audio, our own transcripts and a WER check, but no independent licence, consent basis or
redistribution clearance has been established for it: its clip audio is absent from this machine and its
recorded clip digests cannot be recomputed from files. The batch is quarantined because 4 of its 8 kept clips
(`en_slot0_001/002/003/006.wav`, of 16 generated / 8 kept / 24.4271 s) repeat an English evaluation sentence
from `tools/evaluate.py`: three by adding a clause, `en_slot0_006.wav` character-for-character. A model trained
on those clips and then scored on that evaluation is being shown the answer, so the batch is not a training
input.

**Corrected the same day, dated 2026-10-15.** An earlier version of this section recorded the manifest as
"kept exactly as committed". It is no longer, and the sentence was retired rather than edited away. By the
owner's ruling of 2026-10-15 the batch is **quarantined from training and adaptation**: its `usable_for` is now
`["asr-domain-evaluation"]` and it carries a `training_quarantine` record naming the four colliding clips, the
**11.2355 s** of colliding audio measured from their recorded durations, the disjointness rule applied, and
`digests_verified: false`. Every sample, sample digest, count, dropped record, generator and verifier field and
timestamp is byte-preserved, and no `training_boundary` block was added to it — that batch predates the
contract, and back-filling one would be a claim about checks that were never run on it.

**What the manifest cannot show.** The clip audio was never tracked (`.gitignore` excludes `artifacts/**/*.wav`)
and is absent from this machine, so the 8 SHA-256 values in it **cannot be recomputed from files**; the
descriptor is the only surviving record of them, and `digests_verified: false` states that rather than hiding
it. **Retention of its evaluation use is deliberate and unchanged**: `asr-domain-evaluation` remains permitted,
including the read-only probe register row **A41** phase 1 describes over those 8 kept clips. Quarantining
training does not retire a measurement that was already taken, and the code keeps the two questions separate
(`corpus_evaluation_uses()` reads uses independently of `corpus_training_blockers()`).

**Agreement, re-checked on every run.** `tools/corpus_forge.py` keeps the two pools as separate named constants and
refuses to generate when they collide under a written rule (identical after case/punctuation normalisation,
one word set containing the other, word-set Jaccard ≥ 0.8, or ≥ 5 shared leading words). It refuses a second
way too: the training pool is compared against every sentence recorded by any batch whose quarantine is not
`CLEARED`, and an unreadable quarantine record stops the run rather than being skipped. The replacement pool
was rewritten **twice** — first to drop the eight evaluation colliders, then to drop eight replacements that
turned out to recycle text the quarantined batch had already used (its dropped records included them); the
third version shares no sentence with either record.

Every manifest the tool emits carries the rule, the pool sizes, this file's five excluded paths and the
teacher-target block under `training_boundary`, plus a `boundary_contract` header: the contract version and
`pool_set_sha256(TRAIN_SENTENCES, EVAL_SENTENCES)`, the two pools normalised and joined by a unit separator so no
pool boundary can be shifted to manufacture a match. The header anchors nothing by itself. Generation also refuses
when that digest, the contract version, the rule verbatim, or either declared pool size is not recorded in the
tracked register `artifacts/corpus/legacy-batches.json`. **Changing a sentence or the rule is therefore a reviewed
edit to a versioned file, not a quiet change to a constant.**

**What the register is, and what it is not.** The register records the pool sets this project has cleared for
training, and for each batch that predates the contract the SHA-256 of its manifest as it sat on disk when the owner
ruled on it — `en-synthetic-01` appears there as `QUARANTINED`, `cleared_for_training: false`. A registered-but-retired
pool set stays refused, because the gate compares the header against the live pools as well as against the register,
so rolling a constant back fails closed. None of this is attestation. The register is a tracked file, reviewable in
a diff, and a digest match proves **byte identity only**: not licensing, not consent, not authorship, not a right to
redistribute, and not that the audio is any good.

For a batch that already exists, `corpus_training_blockers(manifest, clip_digests=…, manifest_sha256=…)` decides
training eligibility by re-deriving it: `usable_for` must claim training *and* adaptation and may hold no use outside
the closed vocabulary `ALLOWED_USES`, a well-formed `training_boundary` must be present with `collisions_found == 0`,
the counts must add up against the rows listed, `disjointness_rule` must be the rule this code enforces verbatim, the
declared pool sizes must equal the live pools, `excluded_inputs` must be exactly the five paths in the table above,
licence and consent bases must be non-empty, and the samples' sentences are re-checked against the live evaluation
pool, against the registered training pool, and against the quarantine records on disk.

**The quarantine flags are records, not evidence.** `training_quarantine.status` and `digests_verified` describe a
review that happened in the past; they authorise nothing, and an absent quarantine field does not make a batch
permissive either. What authorises a batch is an anchor outside the manifest, and there are exactly two. Either the
`boundary_contract` names a pool set that this code still defines *and* the register records under the supported
version, rule and pool sizes — in which case every kept sample text must also be a sentence of that registered pool —
or the batch predates the contract and is matched by the SHA-256 of its manifest **as the caller read it from disk**
against its registered digest *together with* an explicit `CLEARED` ruling. A missing caller digest, an unreadable
register, a damaged register, a malformed descriptor, an edited file and a live collision all refuse; none of them is
skipped on the way to a permission.

Nothing self-declared is taken as the answer, so writing `"status": "CLEARED"` into `en-synthetic-01` still refuses
it — it has no split designation, no boundary block, its clips still collide, and the register names it `QUARANTINED`
whatever the manifest now claims. The converse case is the recomputed checks, not an absent record: a newly forged
batch carries no quarantine block, and what stops it leaking is that its sentences are compared against the evaluation
pool and against every quarantined sentence the records still hold. Finally, a descriptor cannot prove its own audio
exists: the caller that hashed the clips passes those digests in, which is what `read_clip_digests()` produces —
recomputed from the bytes, raising on a missing file — and without them the batch stays blocked.
`tests/test_corpus_forge_boundary.py` pins all of it, refusals and the accepted case alike, without touching an
engine.

**The door, and the honest state of it.** `corpus_training_eligible(manifest_path, audio_dir)` is the single named
entry point a future trainer is required to call: it reads the descriptor and the WAV bytes, hashes the manifest as it
lies on disk, runs every check above, and raises `CorpusNotEligible` naming each remaining reason. **No training path
calls it yet** — this repository has no trainer — so what exists is a contract under test, not runtime enforcement.
The read side is fenced the same way and with the same limit: `tools/corpus_forge.py` is the only module allowed to
open a corpus descriptor, `SampleManifest.load()` accepts only the evaluation schema and so refuses a corpus batch
rather than quietly downgrading it, `tools/evaluate.py` hand-writes its own `CORPUS` and reads no manifest at all, and
`tests/test_corpus_reader_fence.py` fails if a new reader appears without the accessors or a weight-updating caller
appears without the gate. Those fences are review triggers, not enforcement: they catch a change in this tree and
cannot prevent one.

One consequence of that needs stating plainly, because §7's table could otherwise be read as a capability. The
`asr-domain-evaluation` use `en-synthetic-01` retains is **a permission on record, not a runnable path**: no tool in
this repository scores a corpus batch today, and the round-trip quality path still refuses synthetic audio outright.
Retaining the use keeps the historical record honest; it does not mean a domain-evaluation run exists to be had.

**Teacher targets stay off until two rights questions close.** `tools/corpus_forge.py --teacher-targets`
returns 2 with the refusal on stderr, and the manifest records `enabled: false, status: BLOCKED`. The block is
the one §3 and the 2026-10-13 correction already describe for `Systran/faster-whisper-large-v3`:
`licence_spdx=NOASSERTION`, `redistributable=False`, the publisher's MIT tag conflicting with OpenAI's
Apache-2.0 tag with no statement of which governs a conversion — and no source-data rights for any audio whose
transcript could become a pseudo-label. Resolve both, per asset, in this file; until then no target transcript
is written, and this refusal is not itself a licence finding.

---

# Dated correction record - redistribution rights on installed assets (2026-10-13)

**What was wrong:** `tools/provision.py` wrote `redistributable: true` into every manifest it created, as a
hard-coded default rather than a decision. Three installed assets therefore carried a redistribution claim
that no review had established - including the Kitten trial asset, whose model card discloses no training
data and no voice-actor consent at all. That is the failure rule 9 of the mandate exists to prevent, and it
was ours, not a reference project's.

**What was done:** `tools/asset_status_audit.py` audits each manifest individually, records what evidence
exists locally and upstream, and rewrites the claim to what that evidence supports.

| Asset | Previous claim | Corrected status | Evidence inspected locally | Loads for local use after |
|---|---|---|---|---|
| `faster-whisper-base` | `true` | **unverified** (`False`) | HF tag `license:mit`; no licence text archived locally; third-party conversion of OpenAI weights; training-data provenance undisclosed | yes |
| `faster-whisper-small` | `true` | **unverified** (`False`) | as above | yes |
| `kokoro-int8-multi-lang-v1_1` | `true` | **unverified** (`False`) | Apache-2.0 text shipped inside the package, read directly; bundled espeak-ng data is GPL-3.0; per-voice consent page referenced upstream, never read or archived | yes |
| `kitten-nano-en-v0_8-int8` (trial) | never over-claimed | **unverified** (`False`) | card front matter `apache-2.0`; tarball carries no licence of its own; no training-data or voice-actor disclosure | yes (trial root) |

**Missing evidence is not the same as an established prohibition, and the record keeps them apart.** The
boolean exists only to gate loading: `None` means unknown and is deliberately refused, because a model whose
rights were never checked must not become loadable by silence. **The manifest schema has no rights-status
field**, so `unverified` cannot be stored on an asset as a value distinct from `false`: it is written as the
conservative `False`, a `provenance.permissions` sentence - "not established either way; local use approved,
redistribution claim withdrawn until reviewed" - and a dated line in `provenance.notes`, while the word
itself is carried by the machine-readable `artifacts/audit/asset-status-corrections.json`. A genuine
prohibition would be recorded there as `rights_status: prohibited` with the offending term quoted; none was
found for these four, and none was assumed.

**History preserved:** each manifest was copied verbatim to
`artifacts/audit/manifest-snapshots/*.pre-audit` before being rewritten. The manifests live in the local
data root and are not under version control, so without that snapshot the value the earlier tool wrote
would simply have disappeared.

**Still outstanding for any distributable bundle**, now stated per asset rather than in general: the
espeak-ng GPL-3.0 obligation on every option that needs its data; the Whisper conversion's upstream terms;
and a voice-consent review for any speaker asset. None of this blocks local use, which is what was approved.
