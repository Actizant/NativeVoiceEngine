# Bodhva Voice Engine

### Independent voice technology. Built for developers.

Bodhva Voice Engine is an independent speech-technology project building toward a developer-first platform
for integrating voice capabilities into applications through a unified API.

Our long-term vision is to make powerful speech technology accessible to developers and businesses through
reliable interfaces, measurable performance, expanding language support, and flexible deployment options.

**Status today: engineering preview, not a production service.** The engine, its asset gates and its
measurement harness are implemented and tested; the hosted API, usage billing and any commercially cleared
distribution described below are **planned, not built**. Every claim in this file is labelled with the tier
it actually belongs to, and the repository's own audit register lists what is still open.

**Naming.** **Bodhva** is the intended product brand. **NativeVoiceEngine** is the name of this repository and
of the implementation inside it, and the installed Python package is `nativevoice` (version `0.1.0`,
`pyproject.toml`). The on-disk data root still uses the directory name the engine inherited (`ultron-voice`).
These are four names for one thing at four different layers, and they are being kept that way on purpose: the
brand is a product decision, the repository and package names are what code and verified assets are pinned to,
and no identifier, path or environment variable has been renamed here. Where this file says "the engine", it
means the code in this repository.

## 1. What the project is

Bodhva Voice Engine is a speech engine, not an assistant. It recognises speech, synthesises speech, and
reports per-language capability as a measured status rather than a boolean. It is designed so that a host
application contributes one function — utterance in, reply text out — and keeps its own state, memory,
permissions and interface; nothing in the recognition or synthesis path knows which application is calling.
Voice identity is treated as an installable, versioned package rather than a name baked into the synthesis
path. See `docs/platform.md` and `docs/architecture.md`.

Two properties make it usable as a component rather than only as an app:

- **One narrow integration seam.** `VoiceHost` is an in-process protocol (`integration/contract.py:29`)
  consumed by `VoiceInterface` (`integration/adapter.py:37`). No engine module imports a host application.
- **No hidden state.** The engine writes no conversation store; the integration contract states that it holds
  no persistent memory of its own — not transcripts, not answers, not embeddings — and the host boundary
  reports `history_persisted: false` (`integration/contract.py:5-8`, `integration/adapter.py:209`).

## 2. Vision and key capabilities

The direction is a platform on which developers integrate recognition, synthesis and language coverage
through a stable interface, with deployment flexibility and usage-based commercial terms.

| Capability | Tier | What that tier means here |
|---|---|---|
| Speech recognition on a developer machine | `MEASURED` (English only) | faster-whisper / CTranslate2 adapters run in this tree and were scored on audio this engine synthesised (`tools/evaluate.py`) |
| Speech synthesis on a developer machine | `MEASURED` (English only) | Kokoro-82M through sherpa-onnx, int8, CPU |
| Multilingual coverage | `partial` in the tier sense, and mostly absent | 61 languages are registered; only 4 carry any recorded status, and no language reports `CapabilityStatus.PARTIAL`. See §3 |
| Voice identity as an installable package | `BUILT` | registry, consent fields and install states are implemented and tested |
| Reproducible measurement and refusal paths | `BUILT` | harness, capability statuses, offline proof, asset gates |
| A hosted, documented developer **API** | `PLANNED — does not exist` | there is **no** HTTP service, server, SDK or REST surface anywhere in this repository |
| **Usage-based billing** and a pricing model | `PLANNED — does not exist` | no metering, accounts, keys or payment code is present |
| Hosted, private and on-premises **deployment options** | `PLANNED` | today the engine runs only where the code is installed |
| **Independently trained** speech models | `PLANNED — not started` | the first development priority: a Conformer-TDT-class recognition student distilled against a Whisper large-v3 teacher, then a flow-matching synthesis student. Every model in use today is a third-party pretrained asset loaded through an adapter; nothing has been trained or fine-tuned in this project |
| Voice cloning | `REFUSED, not a feature` | every zero-shot asset surveyed is non-commercial or unlicensed (`docs/provenance.md`) |

Distinguishing these tiers is the point of the table. The engine's own vocabulary is stricter than a typical
feature list: capability is a status, never a checkbox.

## 3. Current implementation status, from repository evidence

Language coverage is the honest boundary. `core/language.py:CapabilityStatus` defines six values, and they
mean what they say: `MEASURED` — evaluated here with a reported metric, and the only status acceptance may
cite; `MEASURED_FAILED` — attempted, measured, and the number did not clear the bar; `DECLARED` — an upstream
model card claims it and nobody here verified it; `PARTIAL` — works with named gaps; `UNSUPPORTED` — the
installed engine cannot do the task; `NO_MODEL` — the infrastructure exists but no approved asset is
installed. `MEASURED_FAILED` is kept distinct from `UNSUPPORTED` on purpose: collapsing them destroys the
number that tells an engineer whether the gap is a *missing* asset or a *bad* one.

The registry carries **exactly 61 languages** (`language/registry.py:31-96`). Recorded statuses exist for
four of them (`docs/capability-matrix.md`, generated 2026-10-12 from
`artifacts/evaluation/roundtrip-2026-10-12.json`):

| Language | Recognition | Synthesis |
|---|---|---|
| English | `MEASURED` — round-trip WER 0.0494, CER 0.0242 | `MEASURED` |
| Chinese | `DECLARED` only (vendor model card) | `MEASURED_FAILED` — CER 0.463 against a 0.10 bar |
| Hindi | `DECLARED` only | `UNSUPPORTED` — no front-end in the approved asset |
| Bengali | `DECLARED` only | `UNSUPPORTED` — no front-end in the approved asset |
| the other 57 languages | `NO_MODEL` | `NO_MODEL` |

Concretely, in this tree:

- **Recognition** runs through `faster-whisper` (MIT, CTranslate2) adapters
  (`stt/adapters/faster_whisper.py:50`, `stt/adapters/faster_whisper_engine.py:136`), loaded with
  `local_files_only=True` — the engine never downloads a model at runtime. Variants wired into
  provenance are `faster-whisper-base`, `-small` and `-large-v3` (`src/nativevoice/provenance.py:14-35`);
  the measured product path uses `faster-whisper-base` (`tools/evaluate.py:57`). **These are third-party
  pretrained weights, not models produced by this project.** Live capture (`audio/io.py:91`) needs the
  optional `mic` extra and refuses cleanly without it (`audio/io.py:114`).
- **Synthesis** runs Kokoro-82M through sherpa-onnx `OfflineTts`, int8-quantized on CPU
  (`tts/adapters/sherpa_kokoro.py:53,265-268`) — again a third-party Apache-2.0 asset
  (`hexgrad/Kokoro-82M`, packaged by k2-fsa; `src/nativevoice/provenance.py:95-96`). The approved asset ships
  front-ends for `en-us`, `en-gb` and `zh` only, and the adapter **refuses** rather than emitting romanised
  audio and calling it speech (`sherpa_kokoro.py:60-65`). The same adapter raises `"Kokoro is a fixed
  multi-speaker model, not a zero-shot voice cloner"` (`sherpa_kokoro.py:343-344`).
- **Voice identity** is functional code, not a stub: provenance with `consent_basis` and `redistributable`
  fields, install states, slot binding and per-host registries (`voices/identity.py:64-122`,
  `voices/registry.py:137-265`), covered by 22 tests in `tests/test_voice_identity.py`. Consent applies to
  recorded human voices, not to anonymous model slots.
- **Hardware-adaptive profiles** (Lite / Balanced / Performance) are derived from measurement, and every
  profile currently carries `measured=False` (`hardware/profiles.py:6-7`) — unmeasured thresholds are reported
  as unmeasured instead of being asserted.
- **There is no API surface.** No web framework, HTTP server, client, SDK or API key exists in `src/`. Within
  the installed package the offline guard module is the only socket importer — a tested invariant
  (`tests/test_diagnostics_offline.py:210`) — and it monkeypatches `socket.socket`, `create_connection` and
  `getaddrinfo` to prove nothing leaves the machine (`diagnostics/offline.py:143-194`). There is no
  `[project.scripts]` entry point, so there is no
  `nativevoice` console command, and no `.github/` directory, so no CI is configured.
- **Packaging is deliberately minimal.** `requires-python >= 3.11`, `dependencies = []`, and optional extras
  `mic`, `onnx`, `dev` (`pyproject.toml:9-15`). The control path imports the standard library only: `audioop`
  was removed from the standard library in Python 3.13, so RMS, peak, DC blocking, fades, crossfade and the
  WAVE container are implemented directly in `audio/pcm.py`. Installing this package downloads no model.

Performance is reported as measured, on one machine, and synthesis is slower than real time there at every
profile. `docs/baseline.md` records the environment and `artifacts/benchmarks/profiles.json` the runs, which
are single-repeat observations taken under memory pressure; that same baseline states that latency and
real-time-factor figures taken while the machine is in that state are invalid, so this file quotes no headline
number for speed. Two facts shape expectations anywhere: ctranslate2 behind faster-whisper can use CUDA, while
onnxruntime CPU-only builds cannot accelerate an ONNX model on the same machine — accelerator support is a
per-engine fact in the hardware layer, not a boolean called `has_gpu`.

## 4. Architecture and supported technologies

Confirmed by the code in this repository:

| Layer | Package | What is there |
|---|---|---|
| Capability and status model | `core/`, `language/` | `CapabilityStatus`, the 61-language registry, routing that refuses to cite an unmeasured status |
| Audio | `audio/` | PCM, fades, RMS/peak, WAVE container, device capture behind the `mic` extra |
| Recognition | `stt/adapters/` | faster-whisper / CTranslate2 adapters, self-test echo engine, provenance gate at load |
| Synthesis | `tts/adapters/` | sherpa-onnx Kokoro adapter with a hard front-end language gate, scripted/plumbing engines |
| Voice identities | `voices/` | installable versioned identities, consent and redistribution fields, per-host registry |
| Models and roots | `models/`, `config/` | default-deny asset gate, isolated measurement root, data-root confinement |
| Hardware | `hardware/` | profiles and per-engine accelerator facts, with `measured` flags |
| Integration | `integration/` | the `VoiceHost` protocol and adapter — the seam a developer writes against today |
| Diagnostics | `diagnostics/`, `cli/` | capability reports, offline scan, strict acceptance exits |
| Evaluation | `eval/`, `tools/` | round-trip scoring, benchmarks, corpus tooling, asset-status auditor |

Runtime technologies actually used: **faster-whisper** with **CTranslate2** for recognition, **sherpa-onnx**
with **ONNX Runtime** for synthesis, **espeak-ng** front-end data shipped inside the approved Kokoro asset,
and `sounddevice` for live capture when the `mic` extra is installed. Heavy runtimes appear only inside
`stt/adapters/` and `tts/adapters/`, are imported lazily, and are declared as optional extras rather than
install-time dependencies; they are provisioned out of band by `tools/provision.py`, which is the seam
the project deliberately keeps separate — "initial asset acquisition may use the network, ordinary
runtime may not" (`tools/provision.py:3-5`), and no module under `nativevoice/` imports it.

On-disk layout: engine data root `%LOCALAPPDATA%\ultron-voice` on Windows and `$XDG_DATA_HOME/ultron-voice`
elsewhere, overridable with `NATIVEVOICE_HOME` (`config/paths.py:25-42`); inside it `models/`, `voices/`,
`cache/`, `logs/`. Measurement-only assets live in a deliberately separate sibling tree,
`<root>-measurement`, not behind a flag inside one tree. That layer confines itself: `subpath()` rejects an
unknown subdirectory name and `_ensure_within_root()` raises before any `mkdir` outside the root
(`config/paths.py:45-74`). The guard covers directory creation under the data root, not every write in the
tree — a caller that names an explicit destination (settings save, evidence export, `--out`) gets that path.

## 5. Existing foundations and target architecture

This section describes where the platform is heading, and each row carries the tier it deserves. Code cited
inside a row describes what exists today, not the finished component the row names. `PARTLY BUILT` means
code and tests exist in this tree.
`CANDIDATE` means a surveyed approach not usable here yet. No row carries it: the two that did are now
`PROPOSED — chosen design, nothing built`, the owner having ruled which architecture to build. A
`PROPOSED` component exists nowhere in this build, and any code it cites is adjacent machinery.

| Component | Tier | What this repository actually contains |
|---|---|---|
| Provider-independent orchestration | `PARTLY BUILT` | engine registry and per-task orchestrators (`stt/orchestrator.py`, `session/orchestrator.py`); adapters lazy, behind optional extras; `dependencies = []` |
| Direct speech-to-speech (STS) | `PROPOSED` | `hear()` and `speak()` sit on one object (`integration/adapter.py:84,104`); a turn loop exists and is tested on scripted engines (`:131-159`, `:161-186`; `tests/test_integration.py:93-103,159,175`), but every turn passes through text, so **direct** audio-to-audio STS does not exist |
| Conformer-TDT-class recognition student | `PROPOSED` — chosen design, nothing built | this is the project's first model-development priority, with Whisper large-v3 as the planned teacher. No Conformer-class model is installed, wired or trained here; the nearest survey records the sherpa-onnx TDT-class families as **ABSENT, not requested** (`docs/provenance.md:51`, `docs/research-report.md:39`) |
| Flow-matching TTS student | `PROPOSED` — chosen design, nothing built | the planned synthesis architecture, with Qwen3-TTS recorded as its intended teacher; neither the student nor a training path for it exists here. The published architecture reading behind the choice came from a third-party project and is recorded with claim tiers (`docs/research-report.md:146-153`). **EmbeddingGemma 2 (740M)** is recorded as the confirmed training-side encoder for this work; no computation-graph connection to it has been designed or implemented, and nothing of it is installed |
| Safeguarded voice cloning | `PROPOSED`, rights-gated | consent and refusal machinery is built (`voices/identity.py:64-85`, `profiles/conditioning.py:55-68`, `tts/adapters/sherpa_kokoro.py:343-344`); no approved asset exists |
| Hosted API, tenancy, quotas, metering | `PROPOSED` | nothing — no server, client library, account model or usage counter anywhere in `src/` |
| This project's own trained weights | `PROPOSED` | **NONE EXIST YET** (`docs/provenance.md:56`); "there is not a single approved speech weight in this project" (`:58`) |

**Orchestration is the honest row.** The host-facing contract already forbids a host from importing "an
engine, a model id, a licence decision or a file path" (`integration/contract.py:10-11`), and the seam is
deliberately text-in/text-out (`integration/contract.py:13-14`). What does **not** exist is anything above a
single process: cross-machine routing, tenancy, quotas or metering. Orchestration here means one engine
registry per task on one machine, which is what the code does today.

Two constraints shape every row above, and both were documented before this section was written rather than
discovered while writing it.

**Rights.** Recognition and synthesis currently run on third-party pretrained assets, and the project's own
provenance table records refusals rather than assumptions: Kyutai Pocket TTS weights are marked **do not
fetch**, its 26 preset voices are **PROHIBITED pending review** because per-voice consent is not disclosed,
HiFiTTS-2 is **PROHIBITED in product**, and Silero / pyannote and the rest of the surveyed research assets are
**do not adopt** with no licence recorded (`docs/provenance.md:52-55`). Every zero-shot cloning candidate
examined so far is non-commercial — XTTS `license: other`, F5-TTS and ChatTTS `cc-by-nc-4.0`, fish-speech
`cc-by-nc-sa-4.0` (`docs/audit-register.md` A21), which is the same row that records cloning as "out of the
feature set until an asset is separately approved". So the cloning row is a *gated* path, not a roadmap
promise: the consent, provenance and install-state machinery exists, the model that would need it does not.

**Compute.** The flow-matching reading is useful here as a boundary, not as a plan. That project's own
training path requires Linux plus an NVIDIA GPU at roughly a 56 GB batch default (about 16 GB with gradient
accumulation) and its evaluation requires CUDA (`docs/research-report.md:203-205`), while this project's
recorded baseline shows `onnxruntime` exposing only `AzureExecutionProvider` and `CPUExecutionProvider`, so
synthesis cannot use a GPU at all on the measured machine (`docs/tts-options.md:20`, `docs/baseline.md:84`).
A move to a larger or GPU-bound architecture has to be cleared by measurement on the target hardware, not by a
vendor claim — which is exactly the rule `CapabilityStatus` already enforces for languages today.

**Turn-taking, if STS is ever built, starts from real ground.** The session layer already has its own tests
behind it: turn model, partial/final/retract reducer, barge-in, stale-answer discard,
listening modes, device recovery and disabled-voice routing live in `session/state.py` and
`session/orchestrator.py`, recorded as 25 tests in `tests/test_session.py` (`PROGRESS.md:15`, 2026-10-12).
That is turn-taking plumbing, not a speech-to-speech model, and this file does not claim otherwise.

The sequence in which any of this gets built is §6. Until a component moves out of this section it is not a
feature, and no interface described here should be treated as available.

## 6. Roadmap

**Current work — in this repository, with gates and tests already in place:**

1. **Recognition quality from real speech.** The evaluation protocol exists (`docs/real-speech-protocol.md`);
   human-quality claims do not, because synthetic fixtures are refused in a quality path by design.
2. **Train the native recognition model**: a Conformer-TDT-class student against a Whisper large-v3 teacher, on
   data that clears the boundary register. Blocked on an approved training corpus, on a training stack this
   project has not declared yet (`torch` is in no dependency list and no trainer module exists in this tree),
   and on a GPU validation run that has been proposed but not executed.
3. **Indic synthesis.** Hindi and Bengali need a front-end the approved asset does not have: new assets and a
   rights decision before code.
4. **Closing the open licence and provenance findings** listed in `docs/audit-register.md`, which gate
   anything shipped.

**Future plans — explicitly not built, and not claimed by this repository:**

5. **A documented developer API.** No service exists. Designing one is a separate piece of work, not a flag in
   this engine.
6. **Usage-based pricing, accounts, keys and metering.** Nothing of it is implemented.
7. **Deployment options** — hosted, private and on-premises — beyond running the engine where it is installed.
8. **Independently trained synthesis** — the §5 flow-matching student, its intended Qwen3-TTS teacher and the
   recorded EmbeddingGemma 2 training-side encoder are decisions on paper: no weight, trainer or checkpoint.
9. **Language expansion** — moving registered languages out of `NO_MODEL` by earning `MEASURED` status, one
   measured evaluation at a time.
10. **A separate public Client SDK.** Not started and not part of this repository; shipping any derived asset
    is gated on the licence blockers in §8.

## 7. Testing and evaluation

The repository's default position is that a number either comes from measurement or says it does not.

**What is in the tree (static counts, `tests/`; a census re-taken on 2026-10-27 for the commits that track the
student files, and not a property of any commit):**
57 `test_*.py` files — 46 tracked at `ef49a1b`, plus the 11 student files that were still untracked at the A-1
snapshot and are tracked by the commits that add them — and 1,395 `def test_` definitions; both are textual censuses,
not pytest totals: a parametrised definition expands into many collected nodes, and `tests/` also holds four
non-`test_*` modules (`conftest.py`, `fixtures.py`, `tdt_lattice_oracle.py`, `tdt_lattice_recursion.py`) that define
none. Pytest declares
four markers — `device`, `slow`, `network`, `requires_approved_asset` (`pyproject.toml:23-28`).

**Recorded result.** `python -m pytest -q` needs no model, no network and no approved asset. A small number of
modules probe the local machine — `nvidia-smi`, the audio-device list, a loopback socket, `git ls-files`, a
diagnostic subprocess — so the recorded release figure was taken with those excluded:

```bash
python -m pytest -q \
  --ignore=tests/test_hardware.py \
  --ignore=tests/test_diagnostics_offline.py \
  --ignore=tests/test_no_personal_paths.py \
  --ignore=tests/test_cli_diagnose.py \
  --deselect tests/test_audio.py::test_device_enumeration_never_raises
```

Recorded as **1,071 passed, 7 skipped, 1 deselected, 86 excluded** (2026-10-17, at commit `6056f0e` with the
release snapshot staged; reconciled in `docs/audit-register.md`). That figure is a record against one commit,
not a live claim; this tree returned, on 2026-10-26 only, **1,783 passed, 75 skipped, 1 deselected, 86 excluded** on
that command — a dated historical measurement taken with the student files present in the working tree but untracked,
and NOT re-measured for the commits that add them, so no total in this paragraph may be quoted as a property of those
commits; re-measurement is a separate, authorised act (original measurement, 2026-10-26, with `python -B -m pytest -o addopts="" -q` — `tests/conftest.py` puts `src` on `sys.path`, so no `PYTHONPATH` was needed — at verbosity −1; the command's own `-q` on top of `addopts = "-q"` is verbosity −2 and prints no summary line). That 2026-10-26 measurement was reconciled against collected nodes rather than inferred, and its figures are
historical: 1,945 overall, 86 inside the four ignored modules, 1,859 collected by the command, 1 deselected,
75 skipped. The latest authorised collection census is a separate, later measurement — 2,182 collected nodes summed
from the per-file `--collect-only` report on 2026-10-27 — and a collected-node count is not a test result: nothing ran
to produce it, and no historical figure has been overwritten with it. The 75 skip is measured by reason, not assumed: 10 are asset- and platform-dependent — four Kokoro and two faster-whisper inference nodes behind `--run-approved-assets`, three symlink-refusal cases in `tests/test_audit_tool.py`, one unreviewed-CT2 case in `tests/test_stt.py` — and the remainder are gated behind `torch`, which is absent from this environment. Measured here by gate **call site**,
not by node: 78 `pytest.importorskip("torch")` sites across eight files — `tests/test_modeling_encoder_masks.py` 34,
`tests/test_modeling_normalization.py` 12, `tests/test_modeling_encoder_shapes.py` 10, `tests/test_tdt_lattice_reference.py` 9,
`tests/test_scorer_contract_guards.py` 6, `tests/test_tdt_length_seam.py` 4, `tests/test_tdt_zero_length_contract.py` 2,
`tests/test_tdt_batch_length_contract.py` 1.
A call site is neither a collected node nor a skipped node: `tests/test_tdt_batch_length_contract.py` has one site
that expands into 43 collected nodes. Measured per file in this environment on 2026-10-27, torch-gated skips are 35 in
`tests/test_tdt_lattice_reference.py`, 25 in `tests/test_tdt_batch_length_contract.py`, 10 in
`tests/test_tdt_length_seam.py`, 4 in `tests/test_scorer_contract_guards.py` and 7 in
`tests/test_tdt_zero_length_contract.py` — 81 in all, from a ten-file run that returned 567 passed and 81 skipped.
The skipped-node counts for `tests/test_modeling_encoder_masks.py` (68 collected nodes),
`tests/test_modeling_encoder_shapes.py` (63) and `tests/test_modeling_normalization.py` (33) are NOT MEASURED in
*this* environment, because those three files were not run individually here; they do execute in the isolated CPU-torch
environment, where the whole tree returned 2,162 passed and 33 skipped on 2026-10-27. No whole-tree pass or skip total
is claimed for this interpreter. The earlier split of "34, 24 and 7" totalling 65 correctly read its own
2026-10-26 tree and is now stale: two further student files have joined the torch-gated set since. None of those sites executes here: they are **skipped, not passed**. The figures 169 passed with nothing skipped in `tests/test_tdt_lattice_reference.py`, 63 in
`tests/test_modeling_encoder_shapes.py` and 19 in `tests/test_tdt_zero_length_contract.py` are historical results of
the separate CPU-torch verification environment installed outside this repository, recorded against their own earlier
tree state. They are not re-verified for these commits, and none of those three files was run individually in this
phase. An earlier version of this paragraph counted 12 asset- and platform-dependent skips; that figure had absorbed the two torch-dependent nodes in `tests/test_tdt_lattice_reference.py` that predate the encoder work, and the corrected split is 10 asset- or platform-dependent skips versus a torch-gated remainder; the torch-gated
per-file node figures above are measured, whereas the older totalling-65 split is retained only as a dated reading of
the 2026-10-26 tree and is not a statement about this one. The passed figures at each step were measured, not derived: 1,063 at `f403b47`, 1,195 at `2b167a9` (+132), 1,205 at `a353272` (+10), 1,297 with the audit-tool module (+92 passed and three new NOT RUN over the 95 nodes of `tests/test_audit_tool.py`) and 1,308 with O1 (+11: ten new tests in `tests/test_offline_proof.py` and one added case in `tests/test_evidence_writers.py`) and 1,547 with the promoted TDT lattice suite (+239 nodes from 25 test definitions across `tests/test_tdt_lattice.py`, `tests/tdt_lattice_oracle.py` and `tests/tdt_lattice_recursion.py`; pure-stdlib exact-rational arithmetic, no dependency, no model, no asset, no new frontend). This dated series stops at 1,547; the 1,783 at the head of this
sentence is the next measurement in the same chain, taken on 2026-10-26 once the encoder and seam suites had also
landed, so the two are consecutive readings rather than a discrepancy — and no figure in the series has been replaced
by a later one. Those three NOT RUNs are honest: this platform refuses `os.symlink` for an unprivileged process (WinError 1314), so the three symlink-refusal cases cannot run here; what is proven is each refusal function's own `is_symlink`/`lexists` branch and the directory, escape and sub-path controls, not a write attempted through a real link on this machine, and not `O_EXCL`'s behaviour through a planted link.

**Asset-gated tests** are skipped unless a named human has approved the asset *and*
`--run-approved-assets` is passed (`tests/conftest.py:23-35`); `docs/audit-register.md` records seven
gated nodes across `test_stt.py`, `test_stt_real.py` and `test_tts_sherpa.py`. Approval is a per-asset
decision, not a build flag.

**Measurement harness.** `tools/evaluate.py` scores forced and automatic WER/CER, TTS real-time factor, TTS
first-audio latency and STT real-time factor against ceilings of WER 0.15 and CER 0.10 (`:106-107,194-218`).
Its audio is **project-synthesized round-trip speech, not human recordings** — the module says so in its own
docstring (`:59-62`) — which is why English clears the bar and Chinese is recorded as `MEASURED_FAILED`.
`tools/multilingual_eval.py` does run 13 real human clips, and deliberately reports **no WER/CER** for them
(`docs/multilingual-eval-2026-10-13.md`); it changes no capability status.

**Reproducibility instruments**, all operator-run and none of them a runtime dependency:
`python -m nativevoice.cli.diagnose` reports a host summary, real per-language status and offline scan
findings, with `--max-wer` / `--max-first-audio-ms` turning measured figures into an exit code and `--strict`
treating a criterion that was never run as a failure rather than a pass; `tools/audit.py` runs mechanical checks
and reports NOT RUN separately from FAIL, and writes nothing at all: an ordinary run only prints — which a test now proves by comparing every file name in the repository and the digest of everything in the evidence areas before and after the shipped twelve-check run — `--out PATH` writes to a destination the operator named outside the protected evidence areas, replacing an existing file there and re-checking nothing before the write because it publishes no evidence, `--publish PATH --yes` overwrites a committed findings record only while its bytes still equal `HEAD`, which is re-read immediately before writing, and `--publish-new PATH --yes` creates one in `artifacts/audit/` as `mechanical-findings-YYYY-MM-DD.json`, refusing any other name and claiming the name exclusively with `O_EXCL` so a file that appeared mid-run is not followed: two operations with two different mechanisms, neither falling back into the other; and `--check-artefact PATH` reads a record and reports drift without writing anything; `tools/offline_proof.py` runs the real
synthesis-then-recognition pair under an offline guard (`artifacts/audit/offline-proof.json`, verdict PASS,
zero connection attempts, single run) — evidence of network silence only, not of ownership or
licensing rights. Separately, `tests/test_no_personal_paths.py` blocks personal filesystem paths from
re-entering the tree.

**Model assets are gated at load time, default-deny:** no SHA-256 digest, no run; no named human approver in
the record, no run; an unknown or missing licence field, no run. No environment variable bypasses that gate.
Approving an asset for *use* does not establish a right to *redistribute* it, and the registry keeps those two
questions separate.

**The training boundary.** Any corpus that may move weights is checked against a register before use, so
a batch cannot authorise itself: `artifacts/corpus/legacy-batches.json` pins reviewed sentence pools and
legacy batches by SHA-256; `artifacts/corpus/en-synthetic-01/manifest.json` is **QUARANTINED from training**
because 4 of its 8 kept clips repeat an English evaluation sentence, so training on it and scoring against
that evaluation would measure memorisation; a replacement batch clears the gate only when the sentence
pools behind it match a `pool_set_sha256` recorded in that register, which
`tests/test_corpus_forge_boundary.py` pins. This is a documented contract, not
production enforcement: no trainer in this tree consults it, because no trainer exists here yet. The register
is reviewable provenance, not attestation — a digest proves byte identity and says nothing about licence,
consent or authorship.

## 8. Licensing and third-party assets — read before commercial use

**No licence is granted by this repository.** There is no `LICENSE`, `NOTICE` or third-party-notices file, no
SPDX field in `pyproject.toml`, and no licence is implied by reading the code. Ownership is addressed at §9,
which separates the project's intended ownership from third-party materials under their own terms.

This is a commercial direction, so the open items are stated rather than summarised away. Recorded in
`docs/audit-register.md` and `docs/provenance.md`:

- **espeak-ng GPL-3.0 data** is bundled by the synthesis front-end. This is an **open pre-distribution
  blocker** for anything shipped to a third party.
- **Kokoro-82M**: the Apache-2.0 text is verified from the licence shipped inside the asset, yet the asset is
  recorded as **not redistributable**.
- **faster-whisper base and small** checkpoints carry a `license:mit` hub tag with **no licence text archived
  locally**, so their terms are recorded as unreviewed rather than assumed.
- **faster-whisper large-v3**: recorded licence status `NOASSERTION`; redistribution not established.
- **HiFiTTS-2** is recorded as **prohibited in product**. **Silero** and **pyannote** have **no licence
  recorded** and are marked do-not-adopt. **Pocket TTS** has MIT text present with **no
  copyright-holder line**.
- **en-synthetic-01** remains `QUARANTINED` from training, an open finding, not a closed one.
- Third-party model weights, checkpoints, voice embeddings, espeak-ng data files, datasets, Python
  dependencies and generated measurement artefacts remain under their own terms, and several are recorded
  here as *not* redistributable even though local use is approved. **Teacher weights are not in this
  repository** — only digest and size evidence is.
Nothing here has had a production hardening pass, and no claim of licence clearance, regulatory compliance,
fitness for distribution or production readiness is made anywhere in this project.

## 9. Ownership, contribution and contact

**Actizant is the company developing Bodhva Voice Engine, and the intended owner of the product and of
this repository.** "Intended" is load-bearing: it names who builds this and who the business expects to
hold it. It is not a verified record of legal ownership of every contribution here, and it claims nothing
about the copyright position of any individual file.

**No `Copyright ©` notice is asserted over the tree**, because nothing in it ties each contribution to that
entity: no `LICENSE`, `NOTICE`, `AUTHORS` or `CONTRIBUTORS` file, no assignment or contributor agreement, no
`authors` field in `pyproject.toml`, and no ownership ruling in `DECISIONS.md`. The commit history names
individual accounts, and the unsupported AI co-author trailer on its first commit is removed in this chain. An
entity name evidences control of the account that hosts this repository, not authorship of its contents
and not a legal ownership position over every contribution in it, and no such inference is drawn here. The
full evidence behind this section — file-level anchors, adapter behaviour and the asset register — is held
in `docs/audit-register.md` and `docs/provenance.md`, not summarised on this page.

**Third-party materials are not Actizant's, and two groups must not be conflated.** What the engine's code
paths load — only through the default-deny asset gate, and only where an approved asset sits locally — is
the Kokoro package behind `tts/adapters/sherpa_kokoro.py`, including the espeak-ng data files bundled
inside it, which is why §8's GPL-3.0 item is a live blocker rather than a hypothetical, and the
faster-whisper checkpoints behind `stt/adapters/faster_whisper.py`. Everything else §8 names is **examined
but not adopted**: Kyutai Pocket TTS, HiFiTTS-2, Silero and pyannote, and the warnings §8 records about them
are licence findings, not dependencies. This tree holds no Pocket TTS code: the reference adapter that
documented its architecture is retired, and the reading survives in `docs/research-report.md`.

**Nothing here grants a licence, transfers any third-party material to Actizant, or states that commercial
redistribution has been cleared.**

**Contribution.** There is no `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue template or CLA in
this repository, and no formal contribution policy is claimed. Nothing in this file commits this project to a
response time, a review process or a support channel.
In practice, the work that has to happen first is the audit-register findings in §8 — a contribution that
touches a capability claim should come with the measurement that justifies it, or it should say that it has
none.

**Contact.** No support address, maintainer handle, discussion forum or other channel is defined anywhere in
this repository, and none is implied by any name that appears in the code. Commercial and API enquiries are
listed as `PLANNED` in §2 precisely because no route for them exists yet; when one does it will be stated
here rather than inferred from a username in the source.

**Reading order:** `docs/platform.md` (the host and identity model) → `docs/architecture.md` →
`docs/capability-matrix.md` → `docs/baseline.md` (the measured environment) → `docs/provenance.md` and
`docs/audit-register.md` (what may be shipped, and what is still open) → `docs/research-report.md`.
`DECISIONS.md` holds the rulings that gate measured speech quality; `PROGRESS.md` is the dated working record.

## 10. What this is not

Not a cloud service today, and not a wrapper around one — the hosted API in §6 is a plan, not a running thing.
Not a chatbot, and not the voice layer of any other application. Not an integration of another voice product
or assistant: the research archives read here were reference-only, are held outside this repository, no file
from them is imported, vendored, ported or renamed — a tested fence over the whole package says so
(`tests/test_integration.py:224-232`) — and their licence terms were not reviewed for redistribution. Design
reasoning is in `docs/research-report.md`.

Three invariants worth knowing before reading the code:

1. **Capability is a status, not a boolean** — the six `CapabilityStatus` values in §3. A vendor's claim
   cannot satisfy a strict routing decision, and a missing model is a normal state rather than a gap to
   paper over.
2. **Approval is enforced at load time, default-deny** — no digest, no run; no named approver, no run; unknown
   licence field, no run.
3. **Numbers come from measurement or say they don't** — thresholds and resource requirements carry a
   `measured: bool`, and anything unmeasured is reported as unmeasured instead of asserted.
