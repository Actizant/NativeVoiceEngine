# NativeVoice

**Local-first speech infrastructure by Actizant — recognition and synthesis that run on the machine the
user already owns.**

NativeVoice is an engine, not an assistant. It listens, transcribes and speaks on the host machine, and
it treats *voice identity* as an installable, versioned package instead of a name baked into the
synthesis path. One engine can serve several independent applications: each host contributes a single
function — utterance in, reply text out — and keeps its own state, memory, permissions and UI. Nothing
in the recognition or synthesis path knows which application is calling. See `docs/platform.md`.

**Status: engineering preview.** The control plane, the asset gates and the measurement harness are
built and tested, English speech works end to end on this code, and most other languages are an explicit
refusal that names what is missing. Nothing here has had a production hardening pass, and no
fine-tuned or proprietary model has been trained yet. "What is planned, not built" below says what is
actually next.

## What works today

| Capability | State | What that is based on |
|---|---|---|
| English recognition | `MEASURED` | faster-whisper (MIT) transcribes audio this engine synthesised; the measured path is `tools/evaluate.py`, round-trip WER/CER |
| English synthesis | `MEASURED` | Kokoro-82M through sherpa-onnx, int8 quantized, CPU execution |
| Voice identity as a package | built | speaker slots resolve to installable identities; consent applies to recorded human voices, not to anonymous model slots |
| Hardware-adaptive profiles | built, thresholds provisional | Lite / Balanced / Performance are derived from measurement; unmeasured thresholds carry `measured: false` rather than an asserted number |
| Offline operation | built | an offline-compliance proof runs against the real engines, not against the test suite |

Statuses come from `core/language.py:CapabilityStatus` and mean what they say: `MEASURED` — evaluated
here, with a reported metric, and the only status acceptance may cite; `MEASURED_FAILED` — attempted,
measured, and the number did not clear the bar; `DECLARED` — an upstream model card claims it and nobody
here verified it; `PARTIAL` — works with named gaps; `UNSUPPORTED` — the installed engine cannot do the
task; `NO_MODEL` — the infrastructure exists but no approved asset is installed.
`MEASURED_FAILED` is kept distinct from `UNSUPPORTED` on purpose: collapsing them destroys the number
that tells an engineer whether the gap is a *missing* asset or a *bad* one.

## What does not work, stated plainly

- **Chinese synthesis** — `MEASURED_FAILED`: measured at round-trip CER 0.463 and it did not clear the
  bar, partly a simplified-vs-traditional mismatch with no approved converter available. The front-end
  is present; the number is the problem.
- **Hindi and Bengali synthesis** — `UNSUPPORTED`. The approved asset ships front-ends for `en-us`,
  `en-gb` and `zh` only, and the adapter refuses rather than emitting romanised audio and calling it
  speech.
- **Hindi and Bengali recognition** — `DECLARED` only. Whisper's model card claims them; nothing here
  measured them, and an unmeasured claim cannot satisfy a routing decision.
- **Every other language** — the registry carries roughly sixty of them with a per-language capability
  status, and most (task, language) pairs are `NO_MODEL`: infrastructure present, approved asset absent.
- **Voice cloning** — not a feature. Every zero-shot asset surveyed is non-commercial or unlicensed
  (`docs/provenance.md`).
- **ASR adaptation / training** — not started. The training-boundary gates exist and are tested; no
  trainer in this tree consults them, because no trainer exists in this tree yet.

## Setup

```bash
python -V                              # pyproject requires >= 3.11; the last recorded run here used 3.13.14
python -m venv .venv
.venv\Scripts\activate                 # Windows  (source .venv/bin/activate elsewhere)
pip install -e ".[dev]"
python -m pytest -q
```

The control path imports the **standard library only**. `audioop` was removed from the standard library
in Python 3.13, so RMS, peak, DC blocking, fades, crossfade and the WAVE container are implemented
directly in `audio/pcm.py`. Heavy runtimes appear only inside `stt/adapters/` and `tts/adapters/`, are
imported lazily, and are declared as optional extras (`mic`, `onnx`) rather than as install-time
dependencies: `pyproject.toml` declares `dependencies = []`. Installing the package downloads no model.

`python -m pytest -q` needs no model, no network and no approved asset. A small number of modules probe
the local machine — `nvidia-smi`, the audio-device list, a loopback socket, `git ls-files` and a
diagnostic subprocess. To run the suite without touching any of them:

```bash
python -m pytest -q \
  --ignore=tests/test_hardware.py \
  --ignore=tests/test_diagnostics_offline.py \
  --ignore=tests/test_no_personal_paths.py \
  --ignore=tests/test_cli_diagnose.py \
  --deselect tests/test_audio.py::test_device_enumeration_never_raises
```

Recorded result of that command in this tree: **1,071 passed, 7 skipped, 1 deselected, 86 excluded**
(2026-10-17, at `6056f0e` with this release snapshot staged). Asset-gated tests are skipped
unless a named human has approved the asset *and* `--run-approved-assets` is passed; approval is a
per-asset decision, not a build flag.

## Where things live on disk

| What | Where |
|---|---|
| Engine data root | `%LOCALAPPDATA%\ultron-voice` on Windows, `$XDG_DATA_HOME/ultron-voice` elsewhere; override with `NATIVEVOICE_HOME` |
| Inside the root | `models/`, `voices/`, `cache/`, `logs/` |
| Measurement-only assets | a sibling tree, `<root>-measurement` — deliberately separate, not a flag inside one tree |
| Hugging Face hub cache | `HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub` |

The engine writes no conversation store: the integration contract states that it holds no persistent
memory of its own — not transcripts, not answers, not embeddings — and the host boundary reports
`history_persisted: false` (`integration/contract.py:5-8`, `integration/adapter.py:202`). Everything
the data-root layer creates lives under one root so a user can find it, back it up or delete it as a
unit, and that layer confines itself: `subpath()` rejects an unknown subdirectory name and
`_ensure_within_root()` raises before any `mkdir` of a path outside the root (`config/paths.py:45-74`).
That guard covers directory creation under the data root, not every write in the tree — a caller that
names an explicit destination (settings save, evidence export, profile export, a `--out` flag) gets
that path. The directory is still named `ultron-voice` (`config/paths.py:26`): the data path the
`nativevoice` namespace inherited, kept because renaming it would strand verified assets, model
manifests and installed voice identities (`DECISIONS.md` N6). No engine module imports or requires any
host application, though the inherited `ultron-voice` directory name and older labels and docstrings
remain in the tree. The data root resolves from `NATIVEVOICE_HOME` when that variable is set, and
otherwise from the OS-standard location: `LOCALAPPDATA` on Windows, `XDG_DATA_HOME` elsewhere
(`config/paths.py:25,30-42`).

## Model assets are gated, and approval is not clearance

No asset loads without a manifest that passes the registry, and the gate is default-deny:

- no SHA-256 digest, no run;
- no named human approver in the record, no run;
- an unknown or missing licence field, no run;
- there is no environment variable that bypasses it, and enforcement happens at load time.

Approving an asset for *local use* does not establish a right to *redistribute* it, and the registry
keeps those two questions separate. Two consequences a distributor has to solve before shipping:
Kokoro-82M's Apache-2.0 text is verified from the licence shipped inside the asset, yet the asset is
recorded as not redistributable; and espeak-ng GPL-3.0 data is bundled by the front-end, which is an
open pre-distribution blocker (`docs/audit-register.md`, `docs/provenance.md`). Faster-whisper's base and
small checkpoints carry a `license:mit` tag with no licence text shipped, so their terms are recorded as
unreviewed rather than assumed.

## The training boundary

Any corpus that may move weights is checked against a boundary register before it is used, so a batch
cannot authorise itself:

- `artifacts/corpus/legacy-batches.json` — the tracked register of reviewed sentence pools and legacy
  batches, with each batch's manifest pinned by SHA-256;
- `artifacts/corpus/en-synthetic-01/manifest.json` — a batch **QUARANTINED from training** on
  2026-10-15, because 4 of its 8 kept clips repeat an English evaluation sentence, so training on it and
  scoring against that evaluation would measure memorisation. It remains permitted for
  `asr-domain-evaluation`;
- `tools/corpus_forge.py` refuses to produce a training-eligible batch that collides with a registered
  pool, and `tests/test_corpus_forge_boundary.py` pins the rule.

The register is reviewable provenance, not attestation: a digest proves byte identity and says nothing
about licence, consent or authorship. The control is that the register is a tracked file, so every change
to it is a diff someone sees.

## Performance as measured, on one machine

Synthesis is slower than real time on the one machine measured here, at every profile. See
`docs/baseline.md` for the environment and `artifacts/benchmarks/profiles.json` for the recorded runs,
which are single-repeat observations taken under memory pressure.

Two hardware facts that shape expectations on any laptop: ctranslate2 behind faster-whisper can use CUDA,
while onnxruntime 1.30 CPU-only builds cannot accelerate an ONNX model on the same machine. So
accelerator support is a per-engine fact in the hardware layer, not a boolean called `has_gpu`, and a
GPU is never assumed to be faster. Profile thresholds seeded against a memory-starved baseline stay
`measured: false` until the harness is re-run on an idle machine.

## Run the diagnostics

```bash
python -m nativevoice.cli.diagnose --capabilities capabilities.json
python -m nativevoice.cli.diagnose --scan-path <tree> --strict --json
```

`diagnose` reports a host summary, real per-language capability status and offline scan findings. Use
`--max-wer` and `--max-first-audio-ms` to turn measured figures into an exit code, and `--strict` to
treat a criterion that was never run as a failure instead of a pass. There is no `nativevoice` console
command: the package ships no `[project.scripts]` entry point.

The `tools/` directory holds the measurement instruments — `provision`, `evaluate`,
`benchmark_profiles`, `offline_proof`, `platform_demo`, `corpus_forge`, `seed_voices`, the TTS labs and
the asset-status auditor. They are operator tools, not runtime dependencies, and several of them run
real inference against approved assets.

## What is planned, not built

1. **ASR first.** Adapt a compact recognition model against a teacher, on data that clears the boundary
   register, and evaluate on real speech rather than synthetic round-trips. Blocked on an approved
   training corpus, on a training stack that this project has not declared yet (`torch` is in no
   dependency list and no trainer module exists in this tree), and on a GPU validation run that has
   been proposed but not executed.
2. **Indic synthesis.** Hindi and Bengali need a front-end the approved asset does not have, which means
   new assets and a rights decision before code.
3. **Real-speech evaluation.** The protocol exists (`docs/real-speech-protocol.md`); human-quality claims
   do not, because synthetic fixtures are refused in a quality path by design.
4. **A separate public Client SDK.** Not started and not part of this repository. Shipping any derived
   asset is gated on the licence blockers above.

## Reading order

`docs/platform.md` (the host/identity model) → `docs/architecture.md` → `docs/capability-matrix.md` →
`docs/baseline.md` (the measured environment) → `docs/provenance.md` and `docs/audit-register.md`
(what may be shipped, and what is still open) → `docs/research-report.md`. `DECISIONS.md` holds the
rulings that gate measured speech quality; `PROGRESS.md` is the dated working record.

## Three invariants worth knowing before reading the code

1. **Capability is a status, not a boolean** — the six `CapabilityStatus` values above. A vendor's claim
   cannot satisfy a strict routing decision, and a missing model is a normal state rather than a gap to
   paper over.
2. **Approval is enforced at load time, default-deny** — no digest, no run; no named approver, no run;
   unknown licence field, no run.
3. **Numbers come from measurement or say they don't** — thresholds and resource requirements carry a
   `measured: bool`, and anything unmeasured is reported as unmeasured instead of asserted.

## What this is not

Not an integration of OpenWhispr or Pocket TTS. Those archives were read as research references and are
held outside this repository; no file from either is imported, vendored, ported or renamed, and their
licence terms were not reviewed for redistribution. Design reasoning is in `docs/research-report.md`.

Not a cloud service, and not a wrapper around one. Not a chatbot, and not the voice layer of any other
application: hosts plug in through one narrow seam, and nothing in the engine imports a host.

## Ownership

Copyright © 2026 Actizant

This notice covers the original NativeVoice source, tests, tooling and documentation in this
repository. It claims nothing over anything the project did not author: third-party model weights and
checkpoints, voice embeddings, espeak-ng data files, datasets, Python dependencies and any generated
measurement artefact remain under their own terms, and several are recorded here as *not* redistributable
even though local use is approved.

No licence is granted by this repository. There is no `LICENSE` file at the root and none is implied by
reading the code; the asset licence blockers that gate any distribution are recorded in
`docs/audit-register.md` and `docs/provenance.md`.
