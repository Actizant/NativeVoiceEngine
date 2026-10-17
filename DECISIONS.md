# Decisions

## Status as of 2026-10-13

| # | Decision | Status |
|---|---|---|
| S1 | Third-party weights as a measurement instrument | **Answered** - pretrained models approved for local use with licence verification first; Kokoro (Apache-2.0) and faster-whisper (MIT) provisioned and digest-verified |
| S2 | Package installs | **Answered** - nothing installed; ASR fine-tuning **held** |
| S3 | Bengali/Hindi promises | **Answered** - infrastructure plus an honest per-language ledger; hi/bn synthesis and acoustic LID remain documented gaps |
| S4 | Project location and name | **Partly answered** - namespace renamed to `nativevoice`; the working directory is still the Qoder session folder; see N6 and N9 |
| S5 | Benchmark headroom and evaluation audio | **Partly answered** - benchmarks now run with machine context recorded beside every number; still no licensed human-speech corpus |
| S6 | Voice cloning assets | **Answered** - out of the feature set; no permissively licensed local zero-shot asset was found |
| S7 | Definition of complete | **Answered** - evidence-based status per capability, never an overall "complete" while criteria are unmet |

### Newly open

**N6 - data directory name.** The local data root is still `%LOCALAPPDATA%\ultron-voice`. Renaming it would
strand the verified assets, the model manifests and the installed voice identities, so it was left alone.
Say the word and I move it and re-verify every digest; nothing else in the code depends on the old name.

**N7 - commit author line.** The initial commit carries a `Co-Authored-By:` trailer naming an AI model. It is
local and unpushed, so removing it is one `git commit --amend` whenever you want it gone. I did not rewrite
history on my own initiative.

**N8 - TTS replacement proposal.** Real-time synthesis is not reachable with the current asset: best measured
RTF is 1.59 and first audio 5.06 s after the chunk-budget fix. The options are a different approved asset
(identity, provenance, terms, expected performance and resources presented for sign-off before any download),
or GPU execution, which needs an onnxruntime build with a CUDA provider. Nothing is downloaded until that
proposal is approved.

**N9 - repository location.** Written on 2026-10-13, when the working copy was still the hashed Qoder session
path and the history was two commits deep. The tree has since moved to a named folder and the history is
28 commits deep, so both figures here are the 2026-10-13 facts: moving the working copy is now costlier than
this paragraph describes, and rewriting the history to make it cheap again is not on the table.

**N10 - a stray interpreter in the working tree.** `Python/pythoncore-3.15-64` (45 MB) materialised inside
the project directory during the session - not created by any command here, and deliberately **not** deleted.
It is git-ignored; removing it is yours to decide.

---

# Original decision text (2026-10-12)

Everything in `Phase B` that I can build without you is being built now. These seven items are the ones
where I either need money/compute/time from you, or where the specification contradicts itself and I will
not silently pick a reading. Each has my recommended default; if you do not answer, that default is what I
will have assumed.

---

## S1 — May any third-party speech *weights* be used for measurement?

**The contradiction I will not resolve silently.** §5 demands "Actual speech recognition must be
demonstrated with measurable results" and §17 demands "a successful API call does not establish natural
or intelligible speech". §3 rule 10 demands "Do not add third-party code or pretrained model assets to the
product without explicit approval". Independently *training* an STT and a TTS model is not possible on this
machine (§13 anticipates exactly this). So the spec can be satisfied either as a genuine measurement or as
owned-only infrastructure — not both at once, without your ruling.

Options:

- **(a) Recommended.** Use assets that are *already on this machine* — the six faster-whisper CT2 models
  in `~/.cache/huggingface` (5,617 MB: base, base.en, small, small.en, medium, large-v3) — strictly as a
  **measurement instrument** for benchmarks, capability matrices and WER/RTF evaluation. Nothing
  redistributed, nothing downloaded, each recorded in the provenance inventory as
  `review=approved_for_local_benchmarking_only`, and the registry refuses to serve them to the product path.
  You then get real numbers this month, and ULTRON-owned models remain a separate, honest gap.
- **(b)** No third-party weights at all. I complete infrastructure, interfaces, data pipelines, training
  pipelines and evaluation harnesses, and every recognition/synthesis quality criterion is reported
  **NOT RUN** until a ULTRON-trained model exists. This satisfies §3 maximally and §5's measurement clause
  not at all.
- **(c)** Download additional approved open weights (e.g. a sherpa-onnx Parakeet INT8, a Bengali-capable
  Indic ASR model) for benchmarking. Adds network provisioning and a provenance review per asset; I will
  not download anything without a per-asset yes.

**Cost of getting this wrong:** under (b) we can honestly claim nothing about accuracy; under (a) we must
never let a borrowed model's number become ULTRON's advertised capability. The code enforces the difference
either way — `models/api.py` keeps `ReviewStatus` separate from `CapabilityStatus`, and a `DECLARED` vendor
claim can never satisfy a strict routing decision.

---

## S2 — May I install Python packages?

`torch` is **not installed**; neither are `soundfile`, `ruff`, `mypy`. Present and usable: numpy 2.5.3,
onnxruntime 1.30.0 (CPU), faster-whisper 1.2.1, sherpa_onnx 1.13.8, sounddevice 0.5.6, psutil 7.2.2,
fastapi, pytest 9.1.1.

- **(a) Recommended.** Yes for **dev/test tooling only** (`ruff`, `mypy`) — no runtime effect, and it makes
  the §14 quality gates executable. No runtime packages added this phase; the control path stays
  standard-library-only by design.
- **(b)** No installs at all. Then static typing is checked by hand and I report that as a coverage gap.
- **(c)** Also install CPU `torch` (~200 MB) so the training pipelines are *executable* rather than
  specified-and-untested. Note this machine has 12 cores / 15.71 GB RAM / 4 GB VRAM: torch CPU can *run*
  the pipeline; it cannot train a 100M-parameter TTS in any useful time, and evaluation as designed by the
  reference (LibriSpeech WER + WavLM similarity + UTMOS) requires CUDA.

I am not installing anything while you decide.

---

## S3 — Bengali and Hindi: what do we promise?

Facts from the archives, restated because they bound everything else:

- OpenWhispr's 61-language picker has **no Bengali entry at all**; the only Bengali in the repository is
  inside Deepgram Nova-3's internal cloud fallback set. Hindi is present but supported **only** by the
  Whisper family.
- Pocket TTS officially supports **7 languages**, none Indic. Bengali does not exist there; Hindi is a
  community checkpoint that cannot use the shipped voices.

So today: Bengali recognition is plausible via Whisper-class models (needs a measurement, S1); Bengali
**synthesis has no approved model and cannot be produced by any asset we have discussed**; Hindi synthesis
likewise except via an unvetted community checkpoint.

- **(a) Recommended.** We promise the *infrastructure* for ~60 languages and report per-language status
  honestly (`measured` / `declared` / `partial` / `unsupported` / `no_model`), with Bengali and Hindi
  carrying real capability records and a named, itemised path to a trained model. The system must, when
  asked for Bengali speech with no model: refuse and explain, exactly as §7.6 requires.
- **(b)** I could wire in the community Hindi checkpoint to make a Hindi demo exist. I recommend against:
  unvetted provenance, no licence statement, and §17 forbids letting a working demo imply a capability.

**Nothing here makes Bengali the default language** — §7 says it is a required language to *evaluate*, and
that is how it is implemented.

---

## S4 — Where does this project live, and under what name?

The working directory `[USER_HOME]\Documents\Qoder\2026-10-12\76969c5c` is the hashed Qoder session
folder, which is a poor home for a standalone repository you will keep.

- **(a) Recommended.** Keep building here now (it is the authorised workspace), then move it — or `git init`
  and push — to a path you choose, e.g. `Documents\ultron-native-voice`. Say the word and the path and I
  will do the move as its own step.
- **(b)** Build directly at a path you name today.
- Nothing here touches the main ULTRON repository, per §2; no `git init` has been run yet and I will not run
  one until you say so, since that creates history you may want authored differently.

---

## S5 — Benchmarks need memory headroom, and a real recording corpus

Measured at inspection: **15.71 GB total, 0.56 GB available, 96.5 % in use.** A latency, RTF or peak-memory
number taken in that state is noise, and §10 explicitly forbids deriving profile thresholds without
measurement.

- **(a) Recommended.** I build the benchmark harness and capability matrix now with `measured: false`
  everywhere; you close the memory-hungry apps and say "run the benchmark" when the machine is otherwise
  idle, and I fill the numbers in with the command, host summary and timestamp recorded beside them.
- **(b)** Also needed for real quality numbers, not just latency: **evaluation audio with usable licence
  and transcript pairs** (§14: "legally usable audio samples"). English is easy from open corpora; for
  Bengali and Hindi I need either an approved open dataset or recordings you are entitled to use. I have
  built the sample-manifest format and the synthetic/real separation so a synthetic tone can *never*
  contribute to a quality score — but I cannot invent the corpus.

---

## S6 — Voice cloning assets

§8 asks for the profile/conditioning architecture now, with recordings later. Implemented now: local
profile store, consent record required before a profile can exist, quality assessment (duration, RMS band,
clipping, silence fraction, SNR estimate), stdlib `zipfile` export/import with checksum verification and
format versioning, and deletion that proves what it removed. What cannot work yet is the conditioning
itself, because it needs an approved synthesis model (§S1/S3) — so `conditioning.py` refuses with the
missing component named. Confirm you accept that split: governance and storage complete, acoustic
conditioning gated on an approved model.

---

## S7 — Scope of "complete" for this first pass

§15 Phase B says do not stop at scaffolding. Read together with §3 rule 10 and §13, the achievable
completion for this machine is: **all original infrastructure complete and tested, every capability either
measured or explicitly marked awaiting-model, and the training/evaluation pipelines executable on approved
assets.** Interfaces-with-no-implementation are reported as such and never counted as features.

- **(a) Recommended.** You accept that definition, and the final acceptance report states it in those terms
  with PASS / FAIL / NOT RUN per criterion — no "complete" claim while §5's measured-recognition or §6's
  synthesis criteria remain unmet.
- **(b)** If you want a specific quality criterion (say, Bengali ASR WER under a stated budget) to be
  mandatory for "complete", tell me and I will treat it as a blocker rather than a NOT RUN.

---

### What is not waiting on you

Phase B implementation is running now, unblocked: core contracts, audio transport, VAD, model registry with
the licence gate, hardware detection and provisional profiles, ~60-language registry with per-task
capability ledger, text-domain language identification and routing, STT/TTS interfaces with strict
capability gating, synthesis text pipeline, voice-profile governance, offline-compliance proof, and the
benchmark/evaluation harness with PASS/FAIL/NOT RUN reporting.
