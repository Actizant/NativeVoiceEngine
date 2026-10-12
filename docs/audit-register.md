# Comprehensive system audit — consolidated defect register (Phase C)

One audit, one register, as the workflow requires. Executed 2026-10-12 across all twelve
packages and their interactions. Instrument: `tools/audit.py` (11 mechanical checks, re-runnable,
artefact `artifacts/audit/mechanical-findings.json`), plus targeted reading, a real-hardware pass
(`tools/selftest_audio.py`), an offline proof (`tools/offline_proof.py`) and a profile benchmark
(`tools/benchmark_profiles.py`).

Severity: **S1** breaks a promise the system makes about itself (a false capability claim, a
privacy or licence breach, an unusable core path). **S2** breaks or misleads a feature. **S3** is a
correctness or quality defect with a workaround. **S4** is hygiene.

Status: FIXED (code changed + regression test), OPEN (recorded, not fixed), FALSE (investigated and
not a defect).

| # | Severity | Component | Defect | Evidence | Root cause | Status |
|---|---|---|---|---|---|---|
| A1 | **S1** | `language/lid.py` | Every Latin-script utterance was classified **English at confidence 1.000**; Spanish, French and German were never compared | measured: es/fr/de → `en`, conf 1.000; Hinglish → `en` 1.000 | a special-case branch scored English and returned, bypassing the comparative path that already existed | **FIXED** — comparative scoring for all Latin lexicons; romanisation caps confidence at 0.6; uncalibrated scores capped <1.0; unique-script ≤0.9. 28 tests in `tests/test_multilingual_verification.py` |
| A2 | **S1** | `audio/io.py` | `CapturingOutput.stop()` set a flag that permanently disabled the sink: **one barge-in ended all future playback** for the session | found only by the end-to-end conversation test | stop was modelled as a latch instead of a generation boundary | **FIXED** — generation-based halt; `tests/test_audio.py`, `tests/test_integration.py` |
| A3 | **S1** | `audio/io.py` | `SounddeviceOutput` never called `start()` on its raw stream (first write threw), reused a wrong-rate stream after a sample-rate change, and leaked `sounddevice.PortAudioError` past our error boundary; a stop during an in-flight `write()` raised instead of ending the span | real-hardware run: `PortAudioError: Stream is stopped [−9983]` on the first frame | unit tests only exercised the software sink, so the device path was never executed | **FIXED** — start on open, reopen on format change, `PortAudioError` → `DeviceError`/`DeviceDisconnected`, generation check makes a requested stop quiet. Verified on device: 12,000 of 72,000 frames emitted then halt, recovery 6,000 |
| A4 | **S1** | `models/registry.py` | The registry discovered **zero** manifests: provisioning wrote `models/manifests/*.json`, discovery scanned `models/*.json`. Every engine would have reported "no model installed" | `ModelRegistry(models_dir()).ids()` returned `()` against three real manifests | two components agreed on a convention only in prose | **FIXED** — scans both locations; audit check asserts discovery is non-empty |
| A5 | **S1** | `models/integrity.py` | Integrity resolved every asset against one flat root, so a Hugging Face-cached ASR model reported "asset missing" while its files were present and correct; `manifest_io.to_dict` claimed a "full, lossless" form while dropping `asset_root` | `faster-whisper-base` → `installed=False verified=False` while `kokoro` verified | single-root assumption; an explicit field list that was never reconciled with the dataclass | **FIXED** — `asset_root` on the manifest + `base_directory()`, serialised in all three places, both adapters honour it. All three assets now `installed=True verified=True usable=True` on 377/4/4 digests |
| A6 | **S2** | `hardware/profiles.py` | `mark_measured()` set `measured=True, provisional=False` **whatever number of thresholds were supplied**, so benchmarking resident memory alone certified thread counts, buffer sizes and concurrency, and `unmeasured_fields()` returned `[]` | existing test marked 2 of 5 thresholds and asserted `is_validated is True` | a bool standing in for a set | **FIXED** — per-field `measured_fields`; partial evidence keeps `provisional=True` and `require_validated()` raises; two tests now pin both outcomes |
| A7 | **S2** | `eval/` + round-trip | Chinese was scored with **word-level WER = 1.000** for every utterance, which is meaningless for an unsegmented script; CER was 0.14–0.78 and the real errors were visible only there | `artifacts/evaluation/roundtrip-2026-10-12.json`: zh `auto_wer=1.0` with `auto_cer=0.14` on "今天天气很好" | one metric applied to all languages regardless of `space_delimited` | **OPEN** — verdict logic for non-space-delimited languages must take CER as primary. Also unresolved: Kokoro renders simplified, Whisper transcribes traditional, and a converter (OpenCC) is an unapproved dependency, so simplified/traditional mismatch inflates every zh number. Recorded as a measurement-validity limit, not a language capability |
| A8 | **S2** | capability model | `CapabilityStatus` has no "measured and failed" state, so a measured FAIL had to be recorded as `UNSUPPORTED` to keep routing safe, erasing the difference between "refuses" and "tried and was bad" | zh TTS measured, verdict FAIL, stored as UNSUPPORTED | enum conflated usability with evidence | **OPEN** — needs a `MEASURED_FAILED` status plus a routing rule that treats it exactly like UNSUPPORTED while reports keep the number |
| A9 | **S2** | `core/errors.py` | `ResourceLimitExceeded.__str__` hid the actionable sentence behind sorted details | audit reading | `explanation` was not a detail field | **FIXED** — leads with the explanation, details retained; test asserts the rendered string |
| A10 | S3 | `core/events.py` | `SynthesisEvent` had no metadata, forcing per-chunk diagnostics into `text`/`reason` | agent report | contract written before the need | **FIXED** |
| A11 | S3 | `stt/api.py` | `Transcriber.stream()` had no cancellation token, making "honour cancellation at every yield boundary" unenforceable outside the happy path | agent worked around it with an optional kwarg | protocol gap | **FIXED** — token is now in the protocol |
| A12 | S3 | `core/language.py` | `LanguageTag.parse` dropped the region when a variant followed it (`de-DE-1901`, `bn-Beng-IN`) | test failure `bn-Beng` ≠ `bn-Beng-IN` | position-based subtag parsing | **FIXED** — shape-based classification |
| A13 | S3 | 8 packages, 26 sites | Argument validation raised bare `ValueError`, so a host could not catch the category | `tools/audit.py` taxonomy check | taxonomy defined but not enforced | **FIXED** — now `ConfigurationError`, which also subclasses `ValueError` so no existing caller breaks; audit check holds at 0 |
| A14 | S3 | `session/state.py` | `WakeWordBackendUnavailable(Exception)` put a capability refusal outside the taxonomy | audit check | declared before the taxonomy existed | **FIXED** — derives from `CapabilityUnavailable` |
| A15 | S3 | `tools/provision.py` | Not idempotent: a second run re-downloaded 140 MB and produced a duplicated tree, which would also have broken digest resolution | observed on re-run | "download then extract" without a presence check | **FIXED** — skip-if-present with `--force`, always flatten after extraction; duplicate removed and re-verified (205 MB, 377 files) |
| A16 | S4 | `tests/test_audio.py` | A duplicated `TestInputs`/`TestOutputs` block: the second definition silently replaced the first, so tests had been lost while the suite stayed green | duplicate-definition scan | editing by line index | **FIXED** — removed, and `tests/test_suite_hygiene.py` now fails the suite on duplicate defs, empty test files or unconditional skips |
| A17 | S4 | `tools/audit.py` | My own first checks were naive: matched `def open(self)` as a file write, `str.replace()` as `Path.replace()`, and a relative `from .faster_whisper import` as a third-party runtime import | three false FAILs | regex over source instead of syntax | **FIXED** — AST-based, and the persistence claim is now proved **at runtime** (10 turns in an empty temp dir must leave it empty) |
| A18 | S3 | `eval/synthesis.py` | Raised bare `RuntimeError` while its own docstring promised the condition would be "reported as NOT_RUN rather than crashing" | audit reading | docstring and behaviour diverged | **FIXED** — raises `EngineUnavailable` |
| A19 | S2 | product capability | **Speech synthesis is slower than real time on this machine at every profile** | `tts_rtf` 1.55–2.22 (evaluation run) and 1.79–2.38 (profile benchmark); first-audio 3.9–4.8 s, and 6.3–8.3 s under the profile harness | an int8 ONNX flow/LM stack on CPU with `onnxruntime` having **no CUDA provider** | **OPEN — the headline limitation.** Not a code defect: a measured capability ceiling. STT is comfortable (RTF 0.44–0.85 at 4–8 threads) |
| A20 | S3 | product capability | Bengali and Hindi **synthesis is unavailable**: the approved asset ships lexicons for `en-us`, `en-gb`, `zh` only, and the adapter refuses rather than emitting romanised audio | `LanguageUnsupported ... renderable=('en','zh')` for hi/bn, with the refusal recorded verbatim in the artefact | the asset's own file list | **OPEN by design** — the refusal is correct behaviour; closing it needs a different or trained asset |
| A21 | S2 | distribution | The Kokoro package bundles an **espeak-ng data tree under GPL-3.0**; the piper voices on Hugging Face carry **no licence tag**, and every zero-shot cloning candidate examined is non-commercial (XTTS `other`, F5-TTS/ChatTTS `cc-by-nc-4.0`, fish-speech `cc-by-nc-sa-4.0`) | licence texts read from the package and from upstream `COPYING`; HF tags queried directly | n/a | **OPEN — pre-distribution blocker.** Local use is fine. A distributable bundle requires a GPL-3.0 compliance pass, and cloning stays out of the feature set until an asset is separately approved |

## Checks that came back clean

Reported because "audited" should name what was looked at, not only what broke.

- **Offline isolation.** No network module is imported by any package on the speech path; a
  runtime guard around a full TTS→STT round-trip recorded **zero connection attempts**, and the
  round-trip transcribed its own input exactly (`artifacts/audit/offline-proof.json`, verdict PASS).
  The ASR adapter passes `local_files_only=True`, and an audit check now fails any model-load call
  without an offline guarantee.
- **Host independence.** No import of ULTRON, FieldMind, or either reference project anywhere.
- **Memory non-duplication.** Ten full conversations in an empty working directory create no file;
  the turn window is a bounded in-memory deque; `diagnose()` reports `history_persisted: false`.
- **Approval gate.** Four manifest shapes tried; all four refused. No environment-variable bypass
  exists anywhere in the tree.
- **Declared capability.** A `DECLARED`-only language routes to `unavailable`; after a `MEASURED`
  record it routes. Proved on the live ledger.
- **Resource cleanup.** 25 full session open/close cycles leak no threads; every class that opens a
  device or stores a handle exposes `close()`.
- **Import weight.** No heavy runtime is imported at module level; the control path pulls in zero
  third-party modules (verified by diffing `sys.modules`).

## Measurement validity — read this before quoting any number

1. **Round-trip WER measures the TTS+STT pair, not human speech.** There is no licensed human-speech
   corpus on this machine and `eval/samples.py` correctly refuses synthetic fixtures in a quality
   path. English round-trip **WER 0.0494 / CER 0.0242** is therefore the strongest intelligibility
   evidence available, and it is a *floor* on quality, not a measure of it.
2. **The two latency runs disagree**, and both are reported. Evaluation harness: TTS RTF 1.55–2.22,
   first-audio 3.9–4.8 s. Profile benchmark: TTS RTF 1.79–2.38, first-chunk 6.3–8.3 s. The profile
   run had a busy machine and adds per-request pipeline overhead. Neither is discarded; the honest
   statement is that first-audio is 4–8 s and synthesis RTF is >1 at every thread count tried.
3. **No profile is validated.** Peak RSS was measured (1,001 / 1,015 / 1,046 MiB at 2 / 4 / 8
   threads), which is one of five threshold fields. Buffer size, concurrency and quantization policy
   remain unbenchmarked, so `provisional` stays `True` on all three profiles by design (A6).
4. **Chinese numbers are inflated** by the simplified/traditional mismatch in A7 and should be read
   from CER, not WER.
5. **`hi` and `bn` recognition is declared, not measured.** The Whisper family claims them; without
   synthesable or licensed audio for them, no number exists. `en` STT is measured.

## Fixes applied in this pass

A1–A6, A9–A18 are code changes with regression tests. A7, A8, A19, A20, A21 are recorded as open:
A7 and A8 are metric-model changes I am not making silently inside an audit pass, A19 and A20 are
measured limits of the approved asset rather than bugs, and A21 is a decision that needs you.

## Post-fix verification

`python -m pytest` → **639 passed, 7 skipped** (the 7 are real-inference tests that skip unless
`--run-approved-assets` is passed; they pass when run). `python tools/audit.py` → 11/11 PASS.
`python tools/offline_proof.py` → PASS with zero attempts. See `PROGRESS.md` entry 3.

---

## Addendum — findings from the platform repositioning (same audit pass, later run)

The product direction was clarified mid-audit: one reusable platform for several hosts, with voice
identity as an installable package rather than a property of the engine. Implementing that exposed four
more defects, all in the same category — a boundary that existed in prose but not in code.

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A22 | **S1** | `voices/` ↔ `tts/adapters/sherpa_kokoro.py` | The engine resolved a speaker by **parsing the voice display name** (`int("voice.platform.default")`). Any host that chose its own naming broke synthesis entirely | `CapabilityUnavailable: voice id does not name a Kokoro speaker index` on the first platform run | **FIXED** — `VoiceReference.speaker_slot`; identity supplies the slot, name parsing survives only as a labelled compatibility path. `tests/test_voice_identity.py` |
| A23 | **S1** | `tts/pipeline.py` | The consent gate fired for **every** voice reference, so an installed identity from an Apache-2.0 checkpoint could not speak. Consent is the right instrument for a person's recording and the wrong one for a model embedding | `ProfileInvalid: a voice reference needs consent on record` for `voice.platform.default` | **FIXED** — consent required only when the reference carries recorded audio; both directions pinned by tests |
| A24 | S2 | `voices/registry.py` | An explicitly requested but uninstalled voice silently fell through to the platform default, so a host asking for its own narrator would hear a different one | test written against the intended behaviour failed first | **FIXED** — an unknown explicit request raises `StorageError` listing what is installed |
| A25 | S3 | `tests/test_suite_hygiene.py` | The duplicate-definition check flagged two *different* local helper classes named `Stt` in separate test functions — a false positive that punishes the convention it is meant to protect | hygiene failure with no lost coverage | **FIXED** — the check is scope-aware, keying on the full enclosing path |

### What was verified, not just changed

- `tools/platform_demo.py`: two hosts (ULTRON, FieldMind) through the **same** compiled engines, each
  with its own answer function, output sink, voice identity and data root, inside an offline guard.
  **Verdict PASS** — both spoke (84,644 and 121,744 frames), different speaker slots, isolated roots,
  **zero network attempts**. Artefacts: `artifacts/platform/platform-run.json` plus both reply WAVs.
- A second voice is offered only after measurement: slot 7 was synthesised, transcribed back, and
  recorded `MEASURED` at WER 0.000 against the 0.15 bar; both identities remain **unnamed** because the
  asset supplies no speaker names, and inventing one would become metadata someone trusts.
- Honest observation from the run: the recogniser heard *"what the weth will do"* for *"what the weather
  will do"* on our own synthesised input — the round-trip path is not error-free, and the transcript is
  recorded verbatim rather than tidied.

### Post-addendum verification

`python -m pytest` → **667 passed, 7 skipped**. `python tools/audit.py` → 11 checks, all PASS, with the
`voices` package now inside the audited speech path. `python tools/platform_demo.py` → PASS.

### Addendum 2 — finding from the repository relocation

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A26 | S2 | `diagnostics/offline.py` | `static_scan` derived reported paths by splitting the full path on the literal package name at its **first** occurrence. In a checkout whose directory is itself named `nativevoice`, the first match is the repository, so every finding was returned as `src/nativevoice/diagnostics/offline.py` and stopped matching the allowlist - the offline-compliance scan would have reported the guard module as an unexplained network importer | the relocated clone failed `test_the_guard_module_is_the_only_socket_importer_in_the_package`; the original passed | **FIXED** - paths are now relative to the directory actually scanned, with a regression test that recreates the nested-directory case in a temporary tree |

The defect was invisible in the original workspace purely because of where the repository happened to
live. It is the strongest argument in this audit for relocating early: a test suite can be green for
reasons that are properties of a directory name rather than of the code.
