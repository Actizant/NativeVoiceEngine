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
| A19 | S2 | product capability | **Speech synthesis is slower than real time on this machine at every profile** | 14-utterance evaluation (8 en, 6 zh): `tts_real_time_factor` 2.0585–2.5271, `tts_first_audio_ms` 4,026.8–6,471.2 (`artifacts/evaluation/roundtrip-2026-10-12.json`, 3.846 GB free at start). Profile benchmark, median of 3 sentences at 1 repeat per profile: `tts_rtf_median` 1.7913–2.3766, `tts_first_chunk_ms_median` 6,275.5–8,326.0 (`artifacts/benchmarks/profiles.json`, 4.17 GB free at start, 269 background processes) | an int8 ONNX flow/LM stack on CPU with `onnxruntime` having **no CUDA provider** | **OPEN — the headline limitation.** Not a code defect: a measured capability ceiling. STT is comfortable (RTF 0.44–0.85 at 4–8 threads) |
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
2. **The two latency runs disagree**, and both are reported. The 14-utterance evaluation (8 en, 6 zh)
   records TTS RTF 2.0585–2.5271 and first-audio 4.03–6.47 s, with 3.846 GB free at the start. The
   profile benchmark records `tts_rtf_median` 1.7913–2.3766 and first-chunk 6.28–8.33 s as the median
   of 3 sentences at 1 repeat per profile, with 4.17 GB free and 269 background processes running. The
   earlier text quoted a TTS RTF floor of 1.55 and a first-audio band of 3.9–4.8 s; neither traces to any
   numeric field in any artefact under `artifacts/`, and both are corrected here. Neither run is
   discarded; the honest statement is that synthesis RTF is >1 at every thread count tried and first
   audio is 4.0–8.3 s.
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


### Addendum 3 — findings from the KittenTTS trial

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A27 | **S1** | `tools/provision.py` | `redistributable` was hard-coded **`True`** for every asset it wrote, asserting a redistribution right nobody had reviewed - exactly the claim rule 9 forbids | the generated Kitten manifest carried `redistributable: true` while its card discloses no training data or voice consent | **FIXED** in the tool: rights are a per-asset catalogue value defaulting to **False**, and the Kitten record now reads `False` with the reason attached. **OPEN on the incumbent:** the existing `faster-whisper`/`Kokoro` manifests still assert `true` from the earlier run. Correcting an approved asset's licence record is the approver's call, so it is reported rather than silently rewritten |
| A28 | S2 | `tools/provision.py` | `asset_root` was written as a path **relative to the invoking working directory**, so a registry opened from anywhere else could not resolve the asset and reported it missing | first Kitten provision: `usable=False`, asset missing, while all 360 files existed | **FIXED** - absolute resolved path; the trial asset now reports `installed/verified/usable` with 360 digests |
| A29 | S3 | `tools/tts_head_to_head.py` | A replace anchor that matched two near-identical code blocks applied to **both**, injecting a Kokoro-only field into the Kitten row (`NameError` at run time) | first benchmark run crashed in the candidate stage | **FIXED**, and the incident is why the two rows are now built by separate explicit functions rather than shared text surgery |

Method note worth keeping: the comparison is only meaningful because both engines were timed on the
**same basis** - single-sentence, whole-utterance generation - since Kitten's runtime has no chunked path.
The first draft of the harness measured the incumbent through our sentence chunker and the candidate
without it, which would have understated the gap. The chunked figure is measured separately and labelled.


### Addendum 4 — findings from microphone step 2 and the incumbent rights audit

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A30 | **S1** | `faster-whisper-base`, `faster-whisper-small`, `kokoro-int8-multi-lang-v1_1` manifests | `redistributable: true` was still in force on all three incumbent assets, written by the provisioning default flagged as A27 and never established by any review - the licence claim rule 9 forbids | `tools/asset_status_audit.py` re-read each package on disk: neither Whisper snapshot carries licence text inside the installed directory (`licence_text_shipped_in_package: false`) and the weights are a third-party conversion of OpenAI models with undisclosed training data; Kokoro does ship its Apache-2.0 text but bundles an espeak-ng data tree under GPL-3.0 with hundreds of named voice files whose consent terms were never read | **FIXED** - boolean set to `false`, `permissions` records "not established either way; local use approved, redistribution claim withdrawn until reviewed", a dated correction line appended to `notes`, verbatim pre-audit manifests kept under `artifacts/audit/manifest-snapshots/`, machine-readable record in `artifacts/audit/asset-status-corrections.json`. Closes the "OPEN on the incumbent" half of A27. Local use unchanged: all three still `usable=True` |
| A31 | S3 | `tools/real_speech_check.py` | The predeclared criterion "VAD onset error on speech, 150 ms or less" cannot be evaluated by a manual read-aloud run: nothing in the harness knows when speech actually began, so the row has no value at any outcome | step-2 artefact `artifacts/device/real-speech-read-aloud.json`: 74.98 s captured, 0 segments, `vad_onset_error_ms = NOT RUN` | **OPEN by design** - reported NOT RUN instead of being dropped or guessed. Closing it needs a keyed-stimulus run (file playback with a logged start mark), which is not authorised |
| A32 | S4 | `tools/asset_status_audit.py` and its corrections record | The `schema_note` said the missing-evidence versus prohibited distinction "lives in `rights_status`, `permissions`", naming a field the model manifest schema does not have - a reader would look for it on the asset and find nothing | live manifest `provenance` keys dumped: `source`, `upstream_project`, `licence_spdx`, `licence_notice`, `consent_basis`, `permissions`, `redistributable`, `notes`. No status field anywhere in `src/nativevoice` | **FIXED** - wording corrected in the tool and in place inside the record, stating that `unverified` is expressible on a manifest only as the conservative `false` plus prose. The tool was deliberately **not** re-run: a second pass would overwrite the `.pre-audit` snapshots with already-corrected manifests and destroy the very values the record exists to preserve |


### Addendum 5 — findings from the first read-aloud run with a speaker present

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A33 | S2 | `tools/ab_samples.py`, and my own report | The report handed out `--play all` as a listening command; the tool implements only a single pair number, so the advertised command raises before playing anything | reading `play()` against the claim: `int(sys.argv[i+1])` on the string `all` | **FIXED** - a `play_argument()` parser now refuses a non-numeric or absent pair with a message naming the valid range, and a source-scan test asserts no `--play all` is advertised anywhere in the tool. The wrong claim stays in progress-ledger entry 8 as written and is retired here |
| A34 | **S1** | `tools/ab_samples.py` | Playback order came from `sorted(glob("NN_*.wav"))`, and `kitten` sorts before `kokoro`, so every pair played the **candidate first** while the manifest and my report both promised incumbent-first - a first-heard anchor in a listening test is a bias, not a cosmetic detail | the report claimed "plays A then B"; the glob order was verified against the twelve real files before and after the change | **FIXED** - `PLAYBACK_ORDER = ("kokoro", "kitten")` and `LABELS` declare the sequence and name each role; `pair_clips()` walks that declaration. Verified on the real sample directory: all six pairs now resolve Kokoro-then-Kitten, and pair 05 was played audibly through the corrected path |
| A35 | **S1** | `tools/real_speech_check.py` | Recognised segments were paired to target sentences **by position**, so a single utterance was scored against sentence 1 regardless of what it said | the first speaker-present run transcribed sentence 3 word for word and reported WER **1.3333**; measured against its own sentence the same audio scores **0.0000** WER / **0.0000** CER | **FIXED** - content alignment by minimum word error, greedy with a 0.60 refusal cap and deterministic tie-breaks; unmatched targets are `MISSING`, unmatched segments are `extra` and duplicates are named as duplicates rather than invented into a pairing. 11 tests in `tests/test_real_speech_harness.py` |
| A36 | **S1** | `tools/real_speech_check.py` | The two-second post-countdown "quiet baseline" pushed its audio into a separate list that was measured and then **never scored**, and any opening in it was reported as a *false* turn-open - so speech begun immediately after the countdown was discarded as room noise | `segments_detected: 1` alongside 2 openings swallowed in the window, for a run in which three sentences were read | **FIXED** - one stream: the window only samples levels while the segmenter keeps collecting, every opening is logged with its audio-time onset, and a false opening is now one that fell inside the window **and** claimed no sentence. Tested with bursts starting at 0.05 s, 0.0 s and inside a 0.4 s window |
| A37 | S3 | `tools/real_speech_check.py` | `quiet_window_seconds` was assigned `processed.duration_s`, the length of one 20 ms chunk, instead of the window actually measured | artefact reported `quiet_window_seconds: 0.02` for a 2.0 s window | **FIXED** - the measured window is accumulated in audio seconds and reported with its own median and peak level; a test asserts a 0.4 s window reads back as 0.4 s |
| A38 | S3 | `tools/real_speech_check.py` | `segment_onsets[].seconds` carried the segment **duration** under a key that reads as onset, so the log could not be trusted about when speech began | the run's only segment logged `seconds: 3.8` while its `first_sample` 341440 divides to a 21.34 s onset; my verbal report repeated the wrong value | **FIXED** - `openings` now report `onset_seconds` and `duration_seconds` as separate fields, with the real numbers pinned by a test |
| A39 | S3 | `sherpa-onnx` phonemise path (upstream, observed not introduced) | The runtime logs `Skip unknown phonemes. Unicode codepoint: ə` while synthesising, meaning at least one IPA symbol our front-end produces is dropped by the engine's Kokoro/Kitten phoneme table | repeated stderr during `tools/platform_demo.py`, exit 0 and verdict PASS - so it degrades silently rather than failing | **OPEN** - a pre-existing upstream observation, deliberately not touched in this harness pass. It needs its own authorisation: measure whether unstressed vowels survive on the affected sentences before deciding whether to remap the phoneme set or accept the loss |


### Addendum 6 — findings from the supervised rerun at 15:31

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A40 | S2 | `tools/real_speech_check.py` | The report printed transcriptions straight to a cp1252 console, so a non-Latin hallucination raised `UnicodeEncodeError` on the way to the screen and aborted the run with a non-zero exit | the 15:18 run crashed printing `アイルーブ ライカー`, and the 15:31 run crashed the same way - both **after** the artefact was written, so the evidence survived and only the console report was lost | **FIXED** - `configure_console()` reconfigures stdout/stderr to UTF-8 with `backslashreplace`, `emit()` reprints any line the stream still cannot encode as ASCII escapes and swallows `OSError`/`ValueError`, and `write_artefact()` runs before anything is printed. 5 tests, including a fake ASCII-only console proving the file is byte-complete after the console has been attacked |
| A42 | S2 | `tools/real_speech_check.py` | The input stream was opened **before** the countdown, and `SounddeviceInput.frames()` drains an unbounded callback queue - so every second of countdown silence was pushed into the analysis at CPU speed once capture began | `--countdown 60 --limit 100` reported `wall_seconds 100.136`, which only adds up if ~60 s of backlog was replayed instantly. The quiet baseline therefore measured the room during the countdown, the 0 false openings was judged against the wrong interval, and the speaker's usable window after GO was ~40 s of the 100 s promised | **FIXED** - `open_and_capture()` gives the start cue and only then opens the stream, so no pre-GO audio exists to contaminate either the capture or the quiet baseline; `wall_seconds` now measures the capture alone and the artefact records `countdown_seconds` and `stream_opened_at`. 3 tests with a device double that refuses to yield frames while closed, asserting the order GO -> open -> frames -> close |

A41 and A39 stay **OPEN and untouched** by instruction. A41 gained a data point rather than a verdict: the
15:31 run detected `en` correctly on both matched utterances (0.9782 and 0.9557) where the 15:18 run had
answered `nn` at 0.9467 for clear English, so the vendor's auto-detection is **unstable**, not consistently
wrong - which is a different and worse property to design against. The Japanese hallucination in a
silence-adjacent fragment appeared in both runs.


### Addendum 7 — observations from the 17:19 clean run (recorded, deliberately not fixed)

| # | Severity | Component | Observation | Evidence | Status |
|---|---|---|---|---|---|
| A43 | S3 | `tools/real_speech_check.py` | Unmatched segments are reported with their text and closest distance but **not** with the language the engine asserted, so a whole-sentence language substitution cannot be attributed after the fact | the 17:19 run returned "Setelah menonton, sebelum anda mempunyai terlihat." for the lights sentence; `heard[]` held a language per segment but only matched sentences were carried into `results[]`, so this run cannot say whether the engine claimed Indonesian or still said `en` | **OPEN** - a report-completeness gap in the harness, named rather than fixed, because the instruction for this pass was no unrelated changes |
| A44 | S3 | `docs/real-speech-protocol.md` criterion | The predeclared false-opening bar is scoped to the quiet room before speech, so a turn opening on non-speech anywhere **after** that window is not counted as false | the 17:19 run opened a turn at 32.36 s lasting 0.78 s that transcribed to an empty string, and `false_turn_openings` still read 0 - correct against the written criterion, and misleading as a description of the run | **OPEN** - the criterion is what was agreed before any audio existed, so changing its meaning is the approver's decision, not a mid-task edit. Steps 4 and 6 need a definition of a false opening by intent rather than by clock position |

A40 and A42 are now confirmed on live audio rather than only by tests: the report printed a non-Latin
hallucination and exited 0, and 90.02 s of audio took 90.541 s of wall time with the first opening at 0.84 s
after GO.


### Addendum 8 — outcome as ruled by the approver (2026-10-13, after the 17:19 run)

Seven statuses were ruled on the evidence rather than accepted as a pass. They are recorded here in the
approver's own terms and are the authoritative reading of this validation state:

| # | Ruling | Evidence it rests on |
|---|---|---|
| 1 | **A40 verified by the live run** | the 17:19 run printed Indonesian and empty transcripts and exited 0; `write_artefact()` had already written 5,561 bytes that parsed back to the same report |
| 2 | **A42 verified by measured capture and wall duration** | `countdown_seconds: 15`, `stream_opened_at: GO`, `audio_seconds 90.02` against `wall_seconds 90.541`, first opening at 0.84 s - no backlog, no offset |
| 3 | **Basic English live recognition demonstrated, not universally validated** | word-exact pangram (WER 0.0000) plus 0.2500 in the same run, and a whole-sentence Indonesian substitution in the same run; four runs, six matched sentences, no coverage claim |
| 4 | **Endpointing on deliberate 1.5-second pauses NOT RUN** | the 17:19 gaps were ~0.5 s, not the predeclared 1.5 s; three segments for three sentences is encouraging evidence, not the agreed test |
| 5 | **Clean room-noise baseline NOT obtained in this run** | speech began at 0.84 s, inside the 2.0 s window, so `quiet_window_median_level 0.002742` / `peak 0.0672` describe an utterance, not the room; the room reference stays the earlier quiet captures (~0.0004 median, 0.0021-0.0029 peak) |
| 6 | **A41 and A43 remain OPEN** | language substitution observed in the same run that produced two correct `en` results; and the asserted language for an unmatched segment was not recorded, so it is now unrecoverable |
| 7 | **A44 remains OPEN** | the 32.36 s opening with an empty transcript is not counted as false under the window-scoped criterion; the criterion predates any audio, so redefining it is the approver's decision |

Nothing in this addendum changes a status by inference: A40 and A42 move from "fixed and tested" to "fixed,
tested, and verified on live audio", and items 3-7 stay exactly as unresolved as the evidence makes them.


### Addendum 9 — A43 and A44 closed, and the first A41 measurement

| # | Severity | Component | Change | Evidence | Status |
|---|---|---|---|---|---|
| A43-closure | S3 | `tools/real_speech_check.py` | Unmatched segments recorded text only, so a language substitution could not be attributed | `annotate_alignment()` now copies the engine's asserted language and confidence onto every extra segment, and onto each missing target's closest candidate; an empty or absent code stays empty and is never filled in | **FIXED** - 4 tests, including a whole-sentence Indonesian substitution carrying `id`/0.81 and an empty transcript carrying `""`/`None` |
| A44-closure | S3 | `tools/real_speech_check.py` + protocol criterion | A turn opening on non-speech after the quiet window went uncounted | `empty_transcript_openings_outside_quiet_window` plus its onset list is now its own clearly named, explicitly informational measure with no pass bar; `false_turn_openings` is byte-for-byte the same calculation as before | **FIXED as ruled** - 5 tests, including one asserting the pre-existing quiet-window numbers are identical with and without transcripts supplied, and one asserting the new measure is `null` - not zero - when no transcripts exist |

| A41 | S2 | `stt/adapters/faster_whisper_engine.py:305` + `stt/orchestrator.py:102` | The engine returns whatever language Whisper asserts and the orchestrator then routes on it, with no cross-check against the project's own language identification | four supervised runs: `nn` at 0.9467 for clear English (15:18), correct `en` twice (15:31), correct `en` twice plus a whole-sentence Indonesian substitution in the same run (17:19); the synthetic corpus reproduces none of it | **OPEN** - given a register row here, late, because it was first raised in prose instead of as a numbered defect. Unrated as S2 pending one measurement: if routing accepts an unmeasured code without refusal it is S1, because the objective forbids a language error producing unrelated speech |

### A41 phase 1, measured read-only against the existing synthetic corpus

The 8 kept clips of `artifacts/corpus/en-synthetic-01` were transcribed twice with the installed
`faster-whisper-base`: once with `language=None, allow_auto_detection=True` and once pinned to `en`.

| Measure | Auto-detect | Pinned `en` |
|---|---|---|
| Language returned | `en` on **8/8**, confidence 0.9756-0.9919 | `en` by force |
| Mean WER | **0.0000** | **0.0000** |
| Identical transcript | 8/8 | 8/8 |

Two conclusions, and they point in opposite directions:

1. **The instability is specific to human, microphone-path audio.** Nothing in this probe reproduced the
   `nn`-at-0.9467 or the Indonesian substitution from the live runs, so the failure is not in the
   transcription call path itself.
2. **The synthetic corpus is structurally blind to this class of defect** - it was generated by the TTS
   engine and verified by the same ASR engine, so agreement is built in. The manifest already said so
   (`not_usable_for: ["human speech quality claims", ...]`); this is the first measurement that shows *why*.

No microphone was used, nothing was persisted, and the corpus was untouched: the probe ran inside the
offline guard with zero attempts, and all 8 kept clips were re-hashed afterwards and match the digests in
their own manifest. **A41 stays OPEN.** The only remaining way to advance it is a supervised human run
with the language pinned, which needs a `--language` flag on the harness - a change not yet authorised -
or `--retention keep`, which needs an explicit consent decision.


### Addendum 10 — Test A: the clean room-noise baseline is now obtained (2026-10-13 18:03:52)

Ruling item 5 in Addendum 8 said a clean baseline was **not obtained**, because the 17:19 reading began at
0.84 s inside the measurement window. It has since been measured on its own terms: the reader stayed
silent after GO for a whole capture, with no sentences read.

Command `python tools/real_speech_check.py --countdown 15 --limit 12 --out artifacts/device/mic-baseline-post-go.json` -
a separate path, so no read-aloud evidence was overwritten; the live read-aloud artefact still hashes to
`add7a3969bc545edf09d4e789cb821668dfaae8ac096b8318a27bda7dc4c8170`, unchanged, and the synthetic corpus
manifest is likewise untouched.

| Measure | Value | Against |
|---|---|---|
| `quiet_window_seconds` | **2.0** | as measured, not a chunk length |
| `quiet_window_median_level` | **0.00148** | earlier quiet captures: 0.000358 and 0.000412 |
| `quiet_window_peak_level` | **0.0086** | earlier quiet captures: 0.0021 and 0.0086 |
| whole-window median / peak | 0.000484 / 0.0086 | 12.02 s of audio in 12.533 s of wall |
| `openings_in_quiet_window` | **0** | the criterion's denominator |
| `false_turn_openings` | **0** | a PASS that now means something |
| `segments_detected` | **0** | silence produced no turn at all |
| `empty_transcript_openings_outside_quiet_window` | **0**, onset list `[]` | A44's new measure, idle because there were no openings |
| retention / persistence | `none`, `recordings_written: []`, 381 files before and after, `created_during_run: []` | |
| network | `attempts: []` | offline guard held |

Two readings worth recording, and one caution.

1. **The detector has real headroom, now measured.** At a room peak of 0.0086 the energy detector opened
   **nothing**, while the three supervised read-aloud runs peaked at 0.0788-0.0866 - about nine times
   higher. That is the first quantitative support for the claim that a turn opens on speech and not on
   this room.
2. **The room was noisier than during the earlier baselines** - the window median is roughly four times
   the 0.000358-0.000412 recorded earlier, at a different time of day. Room tone is not a constant, so the
   baseline has to be re-measured near the run it qualifies rather than cited from history.
3. **Caution on the window median itself:** 0.00148 sits well above the whole-capture median of 0.000484,
   which suggests an elevated level at stream start - a device ramp-in or first-frame transient - rather
   than flat room tone. Not investigated and not claimed as a defect: naming it is enough here, and a
   `--stimulus`-style keyed test would separate the two explanations.

Ruling item 5 can therefore move from "NOT obtained" to **obtained**, and the false-opening PASS from the
17:19 run is now corroborated by a silent room. Items 3, 4, 6 and 7 of the ruling stand exactly as ruled:
recognition demonstrated but not universally validated, deliberate 1.5-second endpointing still NOT RUN,
and A41, A43, A44 and A39 untouched. A43's new console rendering was not exercised by this run either -
with no segments there was nothing to attribute - so it remains unit-tested only until a run produces an
unmatched segment.


### Addendum 11 — Test B produced two captures, and neither met the 1.5-second condition

**The anomaly first, because it changes how the evidence should be read.** Two artefacts carry Test B's
exact parameters: one generated **18:09:08**, which this agent did not launch, and one generated
**18:13:12**, which it did. The 18:09:08 capture falls between the previous turn's commit and this agent's
launch, so the likeliest explanation is that the reader ran Test B himself. That is a hypothesis, not a
finding: the runs are reported separately and neither is attributed beyond its own metadata. Nothing was
lost - both were preserved automatically, and the 17:19 run still hashes to
`add7a3969bc545edf09d4e789cb821668dfaae8ac096b8318a27bda7dc4c8170`.

| | Run X, 18:09:08 | Run Y, 18:13:12 |
|---|---|---|
| Ended by | reached the 90 s limit | **reached 6 turn openings**, truncated at 65.9 s |
| Openings | 4 | 6, of which **3 had empty transcripts** |
| Matched | pangram 0.1111, lights **0.0000 exact** | coffee tail only, 0.5833 (7 deletions) |
| Unmatched asserted as | `en` 0.4173 ("Oh"), **`ms` 0.9871** | **`bn` 0.7196**, **`bn` 0.8210**, empty at 0.4892 / 0.5200 / 0.2760 |
| Inter-sentence silence | ~0.28 s, **derived** (0.15 s padding at each boundary means a reported gap of -0.02 s is ~0.28 s true silence) | not interpretable |
| Quiet window | median 0.000554, peak 0.004, 0 openings - a clean room | median 0.00093, peak 0.02, 1 opening which was speech |
| Retention / network / files | none / `[]` / 381 to 381 | none / `[]` / 381 to 381 |

**The criterion was not met.** Step 3 asks for deliberate 1.5 s pauses; Run X's pauses were about 0.28 s and
Run Y never reached the question. `endpointing_on_long_pause` stays **NOT RUN** after four attempts.

**What Test B did establish:** A43 verified end-to-end for the first time - unmatched segments carried their
asserted language and confidence into both artefact and console - and A44 measured on live audio
(`empty_transcript_openings_outside_quiet_window: 3`, onsets `[15.28, 17.28, 65.14]`) while the window-scoped
bar stayed at 0, visibly independent as ruled. A41 widened: five captures have now produced `nn`, `ja`, `id`,
`ms` and `bn` assertions against English, several above 0.7 confidence.

| # | Severity | Component | Observation | Evidence | Status |
|---|---|---|---|---|---|
| A45 | S3 | `tools/real_speech_check.py` | `language_detection_top1` counts matched sentences only, so a run can report perfect detection while asserting foreign languages for the sentences that failed | Run X: top-1 1.000 alongside an unmatched segment asserted `ms` at 0.9871 | **OPEN** - the metric is what was predeclared; changing its meaning is a ruling, not an edit |
| A46 | S3 | `tools/real_speech_check.py` | The stop rule counts raw turn openings, so empty or noisy openings can truncate a supervised run before the reader finishes | Run Y stopped at 6 openings with 65.9 s of a 90 s window used; 3 of the 6 produced no transcript | **OPEN** - named, not changed |


### Addendum 12 - retraction: the 18:09:08 capture WAS launched by this agent

Addendum 11 said the likeliest explanation for the unexplained 18:09:08 capture was that the reader ran
Test B himself. **That was wrong, and it is withdrawn.** The approver's instruction not to assume he had
launched it was correct; I assumed it anyway, and the assumption was false.

The origin is now established from this session's own transcript log
(`~/.qoder/projects/C--Users-ACCOUNT-Documents-Qoder-2026-10-12-76969c5c/76969c5c-....jsonl`), which records
every tool call with a timestamp. Local times, +05:30:

| Time | Log line | Event |
|---|---|---|
| 18:03:51 | 4235 | agent launches **Test A** (`--countdown 15 --limit 12 --out mic-baseline-post-go.json`) |
| 18:07:05 | 4260 | a user turn carrying the **Test A approval text again** |
| 18:07:24 | 4263 | `[Request interrupted by user]` |
| 18:08:55 | 4267 | a user turn carrying the **Test B approval** |
| **18:09:07** | **4272** | **agent launches Test B** - `python tools/real_speech_check.py --countdown 15 --limit 90` - this is the 18:09:08 artefact |
| 18:11:05 | 4274 | its result returns |
| 18:11:27 | 4281 | `[Request interrupted by user]` |
| 18:11:31 | 4285 | the **same Test B approval arrives a second time** |
| 18:11:43 | 4290 | agent runs a hash check instead of a capture |
| 18:12:39 | 4294 | `[Request interrupted by user for tool use]` |
| 18:13:02 | 4298 | the **same Test B approval arrives a third time** |
| **18:13:11** | **4302/4303** | **agent launches Test B again** - the 18:13:12 artefact |

So both Test B captures are agent-launched runs of the same approved command, triggered by repeated
deliveries of one approval while my context had been compacted and I could no longer see the first launch.
Both were supervised - peak input levels 0.1197 (18:09) and 0.0312 (18:13) confirm a voice at the
microphone in each - which is also why the earlier unattended run at 14:51, at peak 0.0866 with the whole
window silent apart from one utterance, looked so different.

What is **not** established: why the approval text arrived three times. The log shows user-role entries with
identical content and three interruptions, but it cannot distinguish the approver sending again from a client
re-delivery, and I will not guess. That is recorded as unknown.

Consequences for the evidence: nothing is invalidated - both artefacts were produced by the approved command
under supervision and both are reported on their own terms. But Test B is now **one approved test executed
twice by the same agent**, not two independent sources, and the endpointing conclusion is unchanged: neither
run contained the deliberate 1.5-second pauses between matched sentences.

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A47 | S2 | `tools/real_speech_check.py` | A capture run has **no run identifier and no duplicate-launch guard**: nothing in the artefact says which process started it, and re-delivery of an approval silently produced a second microphone capture of the same test | the log shows Test B approved once and launched twice (18:09:07 and 18:13:11) from three deliveries of the same message; the artefacts carry a timestamp and a countdown but no run id, pid or approval reference | **OPEN** - proposed mitigation is recorded in the ledger entry beside this row; not implemented, and no capture was run to demonstrate it |


### Addendum 12 — the multilingual sample evaluation (2026-10-13): four findings

Full results in `docs/multilingual-eval-2026-10-13.md`. These four are the ones that change what we do next.

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A48 | S2 | `tools/multilingual_eval.py`, `tools/multilingual_compare.py` | Three defects in the instrument I wrote for this evaluation. (1) The cue extractor read the **speaker prefix** of the filename instead of the quoted sentence, so **every** cue score came out 0.0. (2) The function named "longest common subsequence" was a greedy first-match scan, which is not an LCS. (3) Neither tool guarded the console, so CJK and Bengali output crashed the printers on a cp1252 terminal | the Chinese transcript that is in fact verbatim scored **0.04** under the greedy scan and **0.90** under a real LCS; the cue fix moved the set mean from 0.000 to 0.848; both tools raised `UnicodeEncodeError` after writing their JSON | **FIXED** - cue read from after the timestamp, LCS computed by dynamic programming, `_guard_console`/`emit` in both tools. 18 tests in `tests/test_multilingual_eval.py`, one of which asserts the exact greedy-failure case. Had this gone unexamined, the evaluation would have reported that both models failed on every unspaced script |
| A49 | S3 | `tools/multilingual_eval.py` | The repeated-clause detector compares comma-separated clauses, so repetition **inside** one clause is invisible; `repeated_clause: false` therefore does not mean no repetition happened | the baseline's Chinese output contains 都會**屬於屬於屬於** and the flag reads false | **OPEN** - named and labelled in the evaluation document rather than silently fixed, so nobody reads the field as a clean bill of health |
| A50 | S2 | evaluation dataset provenance | The 13 attachments are third-party **synthetic speech** with no licence, consent basis, redistribution statement or provenance metadata. They are voice-product previews, and a voice preview is a product asset, not a corpus | the filenames carry only speaker labels; the recognised text of the English clip is itself vendor copy - "After months of intensive development, **Fish Audio 1.5** has just launched with major improvements to expressiveness, stability, and cross-language capabilities"; 11 of 13 timestamps fall inside one 24-minute window (18:25-18:49) | **OPEN, pre-distribution** - read in place, hashed before and after, never copied into the repository; only isolated 16 kHz working copies exist outside it at `../_multilingual_eval`, and only text metrics plus short excerpts are committed. They must not become a shipped evaluation set or a training corpus without a decision. Local use is unblocked |
| A51 | S2 | `core/language.py` capability model | Capability status is keyed on **language alone**, so "Hindi recognised" can be satisfied by output in the **wrong writing system**. The baseline asserted `hi` and produced Perso-Arabic script, and every existing gate - status, routing, metrics - passed it | Hindi cue recall **0.00** under both baseline quantisations versus **0.90** for `large-v3` in Devanagari; Bengali behaved identically, asserted `bn` and written in Arabic script across 13 segments. Both are invisible to a language-only check | **OPEN** - this is a design gap in how support is represented, not a defect to patch mid-evaluation. Whether script becomes part of a capability claim is a ruling, and it changes the registry schema and the routing gate |

### Addendum 13 - the published repository carried personal paths (2026-10-14): two findings

The repository was made public with `origin/master` at `511a660`. An unauthenticated fetch of a committed
evaluation artefact returned HTTP 200 with its JSON intact, and that JSON contained the operator's Windows
account name inside absolute home-directory paths. The two findings below are deliberately kept apart: one is
already in public history and cannot be undone without a rewrite that has been declined twice; the other is a
live property of the code and will republish the same datum on the next run.

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A52 | **S3 proposed, awaiting his ruling** | committed artefacts + docs (25 files) | Machine-generated evidence recorded absolute home paths containing the operator's Windows account name, and those files were then published. The path layout of the working machine - cache roots, data root, scratch workspace, session directory - became public alongside it | **87** occurrences in **25** tracked files at `511a660`, all of them path segments and none of them a measurement. A scan of all **270** historical blobs for ten credential shapes (private-key blocks, GitHub/Slack/AWS/Google/OpenAI token forms, JWTs, URLs with embedded credentials, bearer headers, credential-shaped assignments) returned **0 hits**, so the account name and the directory layout are the whole of it. Author name and personal email, and a `+05:30` offset, are present in all **24** commit objects | **MITIGATED AT THE TIP ONLY - NOT REMEDIATED IN HISTORY, NOT CLOSED.** Committed as `c93a75f` (2026-10-14): all 87 occurrences in the tracked tree now read `[USER_HOME]`, a repository-relative path, or the literal `ACCOUNT` for one folder slug; `src/` and `tools/` were untouched and every altered line was a path line. That is the entire extent of what this does. The 24 earlier commits, `511a660` among them, still carry the original paths and remain publicly fetchable, as do the author name and personal email, the `+05:30` offsets, the initial commit's co-author trailer and the A50 third-party transcripts. Rewriting history has been declined. **This row is explicitly not a closure of the exposure.** |
| A53 | **S2 — ACCEPTED WITH CONCERNS (2026-10-15)** | artefact writers in `tools/` (nine call sites) | The writers emit real absolute paths at serialisation time, so **any** future capture, evaluation, asset audit or platform demonstration reintroduces the account name into newly committed evidence. A52 without A53 is a one-time cleanup of a leak that is still open | `tools/multilingual_eval.py:175,205,314,388,389`; `tools/real_speech_check.py:356,599` via `preserve_previous():400`; `tools/platform_demo.py:248,251,252`; `tools/asset_status_audit.py:59,148`; and the voice-identity `evidence`/`source` fields seeded through `src/nativevoice/voices/identity.py:58`. Found at implementation, the same defect class sat in the identity serializer, which writes both runtime state **and** tracked evidence from one class: `src/nativevoice/voices/registry.py:45-56` + `save():176`, the fields at `identity.py:58,252`, and the destination decision in the two tools that publish it (`tools/platform_demo.py:153-161`, `tools/seed_voices.py:50-62`) | **ACCEPTED WITH CONCERNS (2026-10-15) — implementation delivered, acceptance scoped; see "Addendum 14" below.** Landed before the next capture or evaluation run, which the guard forces anyway: `tests/test_no_personal_paths.py` fails on any newly written personal path by design. The constraint recorded at proposal time held — `asset_root` is load-bearing (`models/integrity.py:34`, `stt/adapters/faster_whisper_engine.py:68`), so normalisation is confined to the serialisation boundary of **committed copies** (`src/nativevoice/diagnostics/evidence_paths.py:28,48-104`) and live manifests and live voice packages keep their real paths (`registry.py:56`). Accepted on: a second independent read-only review at **PASS WITH CONCERNS**, focused suite **81 passed**, full suite **900 passed / 7 skipped** with the asset-authorisation gates left closed, four temporary mutation probes in which the write-through tests fail when redaction is bypassed (**13** failures), when the destination predicate is forced false (**4**), forced true (**3**), and when the live branch is sanitised too (**3**), and direct verification of the live-versus-committed serialization split and of destination classification including case, traversal, lookalike-prefix, absent-drive and reparse-point routes. **This acceptance establishes none of**: teacher runtime compatibility, GPU compatibility, model accuracy, commercial licensing clearance, training readiness, production readiness. It is a working-tree record only — the boundary module and its tests are still untracked, nothing was staged or committed by this decision, and the `tools/offline_proof.py` attribute mismatch is recorded **separately**, not under A53 |

#### Coverage limit of the guard's local-only inventory check (2026-10-14)

`tests/test_no_personal_paths.py` has two scopes and they must not be read as one. The tracked-tree scan
(`test_no_home_path_names_a_real_account`, `test_operator_account_name_is_not_committed`) is a repository
invariant and runs everywhere; `.trials` is excluded from it by design. The separate local-only inventory
check inspects the ignored residue itself, and **its detection half runs only when the declared root exists**
- `.trials/` is skipped when the directory is absent, because an absent directory holds no personal path. On
any machine that has never provisioned a candidate - including the relocated workspace - that check therefore
verifies **nothing about trial residue**: a pass there is the absence of a directory, not evidence of clean
residue. What the machine cannot exercise, the ledger must not imply. The ignore-rule assertion for every
declared local-only root now runs regardless of existence, so a root that ceases to be git-ignored fails on
every machine; before this correction it sat behind the existence check and so was silently skipped on
precisely the machines with no trial directory.

The declared exception - `.trials/kitten/models/manifests/kitten-nano-en-v0_8-int8.json` - stays in the list
and stays correct in substance: `tools/provision.py:252` writes `asset_root` as a resolved absolute path, so a
trial manifest necessarily carries the account name, and A53's constraint forbids normalising live manifests.
Fail-closed behaviour is preserved for both real regressions: a present trial root whose declared exception has
been cleaned reports `gone=`, and unexpected residue reports `added=`. **This note records a coverage limit; it
closes nothing. A52 remains tip-only; A53 was implemented and accepted with concerns on 2026-10-15 (Addendum 14),
and that acceptance does not change what this note says — a declared local-only exception still carries the real
account name, and the guard still fails closed on it.**

#### Falsification procedure for A52 - an audit step, deliberately not a test

The committed suite is self-contained: no ordinary test reads Git objects, so the suite runs identically in any
checkout. Proving the guard would actually have caught the published exposure is therefore a separate,
reproducible audit step. It loads `personal_path_findings()` straight out of `tests/test_no_personal_paths.py`
and runs it over the **pre-scrub blobs at `HEAD`** alongside the working-tree files:

```bash
python - <<'PY'
import importlib.util, shutil, subprocess
spec = importlib.util.spec_from_file_location("guard", "tests/test_no_personal_paths.py")
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
git = shutil.which("git")
FILES = ("artifacts/evaluation/multilingual-inventory.json",
         "artifacts/audit/manifest-snapshots/faster-whisper-base.json.pre-audit",
         "artifacts/platform/platform-run.json",
         "artifacts/device/mic-baseline-post-go.json",
         "docs/relocation.md", "docs/audit-register.md", "PROGRESS.md")
PRE = "d9270c052cdf0df819507edb4781281ee5836f95"   # pre-cleanup revision; NOT HEAD, which is clean since c93a75f. Re-anchored 2026-10-20: this chain was rewritten locally to drop an unsupported co-author trailer, so the pre-rewrite name of this revision is no longer on it. That old name is kept locally at refs/heads/pre-rewrite-main, and the remote still serves those original commits at their direct URLs although no remote ref names that chain now. The push executed on 2026-10-22 replaced the remote branch with the rewritten chain, so the name this constant now carries resolves on the remote branch history as well as here. Addendum 28 maps every old commit to the new one holding its identical tree.
tb = ta = 0
for name in FILES:
    before = subprocess.run([git, "show", f"{PRE}:{name}"],
                            capture_output=True).stdout.decode("utf-8", "replace")
    after = open(name, encoding="utf-8").read()
    hb = sum(len(guard.personal_path_findings(l)) for l in before.splitlines())
    ha = sum(len(guard.personal_path_findings(l)) for l in after.splitlines())
    tb += hb; ta += ha
    print(f"{name}: {PRE}={hb} tip={ha}")
print(f"TOTAL {PRE}={tb} tip={ta}")
PY
```

**Historical measurement, taken at revision `511a660` before the cleanup was committed.** These are results from
that revision; the same command run against the current tip yields 0 and 0 and demonstrates nothing:

```
artifacts/evaluation/multilingual-inventory.json: 511a660=41 tip=0
artifacts/audit/manifest-snapshots/faster-whisper-base.json.pre-audit: 511a660=1 tip=0
artifacts/platform/platform-run.json: 511a660=6 tip=0
artifacts/device/mic-baseline-post-go.json: 511a660=1 tip=0
docs/relocation.md: 511a660=4 tip=0
docs/audit-register.md: 511a660=1 tip=0
PROGRESS.md: 511a660=3 tip=0
TOTAL 511a660=57 tip=0
```

Interpretation, stated narrowly: at `511a660` the detector flagged **57** real published path segments across
these seven files - including the `.pre-audit` manifest copy, which is in scope - while the same seven files at
the current tip report **0**. Detection was confirmed separately for the Windows-JSON, Git-Bash `/c/Users/…` and
folder-slug forms. That 57 differs from the 87 total is expected: this list is seven files, not all 25. A
non-zero count at the pre-cleanup revision against zero at the tip is the point - a guard that only passes on its
own fixtures proves nothing.

This step depends on the **pre-cleanup** revision still containing the unremediated blobs, which is why it pins
`511a660` explicitly rather than reading `HEAD`: since `c93a75f` the tracked tree is clean, so a `HEAD`-based
version of this command now returns zero on both sides and proves nothing. It stays here as the record of how
A52 was verified, not as a permanent regression check - the permanent checks are the fixture-driven negative and
positive controls inside the committed test file.

### Addendum 14 — A53 accepted with concerns (2026-10-15): the ruling, its evidence, and what it does not establish

The approver ruled **A53 — ACCEPTED WITH CONCERNS** on 2026-10-15, on the strength of the second independent
read-only review that followed the identity-writer remediation. This section is that record. It changes no code
and no artefact: the register entry above and the notes below are the whole of it.

**What was accepted.** Redaction of account-bearing paths at the serialisation boundary of committed evidence
(`src/nativevoice/diagnostics/evidence_paths.py:28` for the token, `:31-45` for the three home spellings,
`:48-58` `redact`, `:61-73` `scrub`, `:76-84` `dumps`, `:87-92` `write_json`, `:95-104` `redact_copy`); the
live-versus-committed split in the identity serializer (`src/nativevoice/voices/registry.py:45-56`, with the live
branch byte-for-byte the body it had at `HEAD`, and the single call-site change at `:176`); the destination
decision in the two tools that publish identities (`tools/platform_demo.py:153-161`, `tools/seed_voices.py:50-62`);
the routing of the six tool writers through the boundary; and the write-through tests that pin all of it
(`tests/test_evidence_writers.py`, `tests/test_evidence_paths.py`).

**Evidence the acceptance rests on.** The review verdict was **PASS WITH CONCERNS**, on the actual working-tree
diff and tests rather than on an implementation report. Focused suite **81 passed**; full suite **900 passed,
7 skipped**, both under `PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m pytest -o addopts="" -p no:cacheprovider -q`
(that is the repository virtual-environment interpreter, not a system `python`), with
`--run-approved-assets` and `ULTRON_RUN_APPROVED_ASSET` left untouched so the seven skips stayed skipped. The
classification matrix was exercised through the real tool functions over the tracked output route, the live data
root, the repository root itself, `..` traversals in and out, a sibling lookalike prefix, lowercase drive and
account spellings, an absent drive, scratch roots, and relative `--out` text resolved from inside and outside the
repository; NTFS junctions were used to confirm the predicate classifies where the bytes physically land, and that
the repository guard traverses a junction, so the one direction the classifier can miss is caught fail-closed.
Four temporary mutation probes, loaded from outside the repository and removed afterwards, established that the
tests are not vacuous:

| probe | mutation | failures |
|---|---|---|
| A | boundary serialiser replaced by raw `json.dumps` | 13 (10 writer tests + 3 in `tests/test_evidence_paths.py`) |
| B | destination predicate forced `False` | 4 |
| C | destination predicate forced `True` | 3 |
| D | live branch sanitised regardless of destination | 3 |

Probe D additionally showed that the pre-existing voice suite (`tests/test_voice_identity.py`, 22 tests) does not
examine live-package bytes at all: the live-branch contract is guarded by exactly one new test,
`tests/test_evidence_writers.py:238`. That thinness is recorded below rather than argued away.

**Concerns retained as documented follow-ups. None of them blocks the acceptance above, and none was fixed while
recording it.**

| follow-up | location | impact | why it is not a blocker |
|---|---|---|---|
| case-variant home prefixes are not recognised | `evidence_paths.py:48-58` (case-sensitive `str.replace`, longest form first) | a path whose drive letter or account component differs in case from the canonical `Path.home()` spelling survives redaction | the repository guard agrees - its detectors (`tests/test_no_personal_paths.py:51-58`) and its derived account-name scan (`:149-158`) are case-sensitive too - and the writer values produced on this machine reach the boundary from `Path.home()` or `resolve()`, which are canonical spellings. The documented environment overrides (`ML_EVAL_WORK`, `NATIVEVOICE_HOME`; `evidence_paths.py:33-38`) are the acknowledged exception: they arrive as caller-supplied text in the Git-Bash shape, which `home_prefixes()` covers as one of its three forms but cannot case-fold. A case-insensitive sweep would rewrite transcript prose |
| two published forms of the same path | tracked voice packages hold a repository-relative evidence path (`artifacts/platform/platform-run.json`) while `tools/platform_demo.py:191` seeds an absolute one, which the boundary publishes placeholder-prefixed | a fresh demonstration run changes the field's **form** (not its safety); historical artefacts and new ones do not read alike | both forms pass the guard's positive controls; making the boundary relativise paths under the repository root is a second behaviour nobody approved |
| the "expands back" wording | `evidence_paths.py:10-13` | the docstring asserts a machine round-trip; no code performs expansion and no consumer requires one (nothing reads `install.source`, `identity.py:252`, to locate a file) | a wording defect, not a leak and not a broken path. The claim should be narrowed to reader-recoverability |
| dictionary keys are not scrubbed | `evidence_paths.py:61-73` (`{key: scrub(value) …}`) | a path carried as a JSON key would survive | latent, and the two key shapes are not the same claim: across the 34 JSON artefacts and pre-audit copies under `artifacts/` (7108 keys, counted by parsing each file) **no key carries an absolute or home-rooted path**, while **732 keys are relative-path-shaped** - the `install.digests` file-name entries of the two tracked voice packages, such as `dict/README.md` and `espeak-ng-data/af_dict`. Those name files inside an asset directory and carry no account, which is why the absence of the first shape is not the absence of the second |
| the writer fence stops at `tools/` | `tests/test_evidence_writers.py:366-373` scans flat `tools/*.py` for two textual patterns | three `src/` JSON publishers are outside it (`eval/report.py:277`, `eval/runner.py:229`, `profiles/store.py:349`), and an indirected `text = json.dumps(...); path.write_text(text)` would slip the patterns | none of the three publishes into `artifacts/`: `AcceptanceReport.write_json` has no caller, `save_baseline` carries timings/host/measured-at only, and the profile store writes runtime state to the data root where real paths are required |
| the named evidence list is a subset | `tests/test_no_personal_paths.py:59-71` names 10 paths; 18 tracked artefacts **under `artifacts/`** carry `[USER_HOME]`, of which 6 are named | a future narrowing of the tree scan would quietly drop the 12 unnamed files from the guarantee | the whole-tree scan covers all 18 today. Of the named entries that lack the token, `artifacts/platform/platform-run.json` publishes repository-relative paths instead and `artifacts/audit/mechanical-findings.json` has no path field at all; the two names outside `artifacts/` are `docs/relocation.md` (which carries the token) and `tests/test_config.py` (a test module whose only home-shaped string is the approved `tester` placeholder at `:139`) - neither of the last two is a repository-relative artefact |
| trailing-newline conventions differ | the three committed `.pre-audit` copies end in LF while `evidence_paths.py:87-92` emits no final newline; `eval/report.py:278` and `eval/runner.py:230` append one | the next audit regeneration changes bytes with no content change | cosmetic, and it must **not** be closed by regenerating protected artefacts |
| manual identity export is unresolved | `identity.py:58` keeps a real path by design; the registry route redacts only what it writes itself | copying a live identity package into `artifacts/` fails the guard instead of sanitising it | fail-closed, so nothing is published by accident. Awaiting his ruling between a publish/export step and the present refusal |
| the `HOME not in text` assertions are weak | `tests/test_evidence_writers.py:68` (helper), `:254`; `tests/test_evidence_paths.py:140,146,191`; `tests/test_evidence_writers.py:160` | JSON escapes each backslash, so the literal home string never appears in an artefact's text and these assertions pass on a leaking file. `:160` is the one check on the kept backup | the same tests also assert placeholder presence, account absence and exact parsed values, which is what probe A killed; the leakage guarantee does not rest on the weak line except at `:160` |

**What this acceptance explicitly does not establish.** It is not evidence of teacher runtime compatibility, GPU
compatibility, model accuracy, training readiness, commercial licensing clearance, or production readiness - the
same limits the approver attached to the inherited architecture work. It establishes a property of the serialisation
boundary and of the tests that hold it.

**A separate finding, deliberately kept out of A53.** `tools/offline_proof.py:97` reads `finding.path` and
`finding.lineno`, while `Finding` defines `file` and `line` (`src/nativevoice/diagnostics/offline.py:243-244`). The
static-findings line would raise the moment a finding existed, which is why the published record shows an empty
list. It is untouched here and is not part of the acceptance above.

**State of the tree at the ruling, and what remains someone's decision.** HEAD `6056f0e` on `master`, ahead of
`origin/master` by one unpushed commit; nothing staged, no commit, push, reset, clean or history operation performed
to record this decision; every protected audit, evaluation, provenance, device and voice-package artefact
**unchanged in the working tree** by this pass (mtimes unchanged, hashes re-read after the review) - which is not
the same claim as byte identity against `HEAD`: of the 36 tracked files under `artifacts/`, 34 match their `HEAD`
bytes, `artifacts/audit/mechanical-findings.json` differs only by the line endings the `eol=lf` attribute will
rewrite at the next `git add`, `artifacts/audit/offline-proof.json` carries timings from a run that predates this
pass, and `artifacts/provenance/faster-whisper-large-v3.json` is untracked, so it has no committed baseline to be
identical to; no audit, provisioning, model load, inference, GPU,
training or install executed. The boundary module and both new test files are still **untracked**, so what is
accepted lives only in the working tree until a commit is separately authorised - and that commit inherits the
`.gitattributes` `eol=lf` rewrite of the CRLF artefacts and the unresolved commit identity. **A3c remains a separate
decision; this addendum records no acceptance of it.**

### Addendum 15 — the exposure decision as ruled, and re-measured against the repository (2026-10-15)

**The ruling.** The repository stays **public** and its history stays **as committed**: no visibility change, no
history rewrite, no replacement repository, no force-push. This is written down because A52's disposition records
that a rewrite was declined but says nothing about the visibility posture, and because a decision to keep the
repository that carries the exposure public should be on the record before anyone reasons from a clean working tree
that the exposure was remediated. It was not.

**Re-measured, not recalled.** Every figure below was produced against this repository on 2026-10-15 by the method
named with it, and each one agrees with what A52 recorded on 2026-10-14 - measured independently, with different
commands, by a different pass. That agreement is the point of restating it. One convention governs the whole table:
**this addendum names the exposed strings by their role and never transcribes them**, because a record that
re-publishes an account name or an email in the tracked tree has widened the exposure it was written to describe.
The literals are in the commit graph, in `git log`, and in this document's own history; they are deliberately not
in its text.

| quantity | value | how it was measured |
|---|---|---|
| Commit trees still carrying the previous machine's account name | **24 of the 27 commits already on `origin/master`**, 6-25 files each | `git rev-list` crossed with `git grep -l -I <account-name> <commit>` |
| Occurrences at the commit that published them | **87 occurrences across 25 tracked files** at `511a660` | the same, counted per file |
| Occurrences at `origin/master`, at local `HEAD`, and in the working tree | **0** in all three scopes | the same three scopes. Measured *before* this addendum was written and re-measured after: the table's own method string appears here only as a name, so the write did not change the result |
| Emails in commit metadata | the author-side Gmail address used on the previous machine across the pushed side; a **different** personal Gmail address on the single unpushed commit `6056f0e` | `git log --format='%ae'` scoped to `origin/master` and to `HEAD`; each address is spelled in that output, not here |
| Commit objects carrying a `+05:30` offset | **28 of 28** | `git log --format='%cd' --date=iso` |
| `Co-Authored-By` trailers | **1**, on the initial commit | full-message scan of all 28 commits |
| Credential-shaped strings in the tracked tree | **0** | private-key markers, GitHub/AWS/Slack token forms, bearer headers and long base64 assignments, scanned over every tracked file. A52's separate scan of all **270** historical blobs against ten shapes also returned **0** |

**What keeping it public changes about the record, and what it does not.** The published datum is now permanent and
unreachable by any edit, so no further work in this repository can reduce it. What still carries weight is the
**forward** guarantee - that no *new* personal path leaves a tool on this machine - and that is precisely the
property A53 was accepted with concerns over and that Remediation Batch 1 hardened. The probes re-run on 2026-10-15
(raw serialiser: 16 failures; `redact` bypass: 24; a dropped inventory name: 1, and only in the new derived-coverage
test; a dropped final newline: 2; the stale `Finding` field names: 2) are evidence about that forward guarantee. They
are not evidence about the published past, and must not be quoted as if they were. Two consequences follow for
anything committed from here: no commit message may describe the working-tree redaction as a remediation of A52, and
the `[USER_HOME]` boundary plus the guard tests that hold it are the only thing standing between a future artefact
writer and a second exposure of the same shape.

**One gap this addendum found by writing itself.** The guard's account-name test (`tests/test_no_personal_paths.py`)
derives the string it forbids from the machine running the test - `getpass.getuser()` and `Path.home().name` - and
compares it case-sensitively against every scanned file. That is why it is a *forward* guarantee: it cannot see a
hand-written instance of a **previous** machine's account name, and neither can it see a case-variant of the current
one. Drafting this table briefly put the previous account name and two email addresses into the tracked tree, and
the suite stayed green while doing so. Two consequences are recorded rather than argued away: the test's name,
`test_operator_account_name_is_not_committed` (`tests/test_no_personal_paths.py:172`), reads more broadly than its
body - it guards *the account of the machine running it*, which is the right rule for machine-generated artefacts
and is not a claim about prose a person types - and any future writing in this repository that needs to name the
exposed strings should name their role, as this table now does. Nothing here closes the case-variant follow-up row
above; it is the same limitation, now demonstrated on the documentation side rather than the artefact side.

**Statuses are not changed by this addendum.** A52 remains `S3 proposed, awaiting his ruling` and remains **not
closed**; the ruling above concerns the repository, not an acceptance of the finding. A53 remains `S2 — ACCEPTED
WITH CONCERNS (2026-10-15)`. A3c remains a separate, non-accepted decision. No row's severity, verdict or
disposition was edited to write this addendum, and no artefact was regenerated to check it.

### Addendum 16 — A54: the synthetic corpus repeated evaluation sentences (2026-10-15)

The Legacy Dataset Isolation Audit compared the two hand-written English sentence lists in this repository and
found they were not disjoint. `tools/corpus_forge.py` forged `artifacts/corpus/en-synthetic-01` from 16
sentences; **8** of them are the same example as one of the **8** sentences `tools/evaluate.py` scores an
English round-trip on. Four of those eight became kept clips — **11.2355 s** of the batch's 24.4271 s — and one
of the four, `en_slot0_006.wav`, is character-for-character the evaluation sentence *"Could you tell me the time
and the nearest train station?"* The remaining three differ from their evaluation sentence only by an added
clause. The other four collided inside the generator's own discard list, so they are recorded as dropped
records rather than audio.

This is a measurement, not an inference: the durations, clip names and sentences were read from the committed
manifest and the collision re-derived under the rule now written in `DISJOINTNESS_RULE`. **Nothing in the
register, the evaluation records or `artifacts/audit/` was regenerated to check it.**

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A54 | **QUARANTINE RULED 2026-10-15 — severity proposed, awaiting owner ruling** | `artifacts/corpus/en-synthetic-01`, `tools/corpus_forge.py` | Training audio and the English evaluation set share sentences, so a model fine-tuned on the batch and then scored on the round-trip evaluation is measured on text it was handed. The two lists lived in different files and nothing compared them, which is why the leak survived the batch being pinned, hashed and described as licence-clean | **4 of 8** kept clips collide (`en_slot0_001/002/003/006.wav`, 11.2355 s of 24.4271 s); **4 of 8** dropped records collide (`en_slot0_000/004/005/007.wav`). One match is identical after normalisation, three are word-set containment. `13.1916 s` of the batch (clips 008/009/014/015) is clean at worst Jaccard **0.222** | **QUARANTINED FROM TRAINING, OPEN — not closed.** Owner ruling of 2026-10-15: `usable_for` is now `["asr-domain-evaluation"]`; `training` and `adaptation` are withdrawn; `asr-domain-evaluation` is retained, including A41's phase-1 read-only probe. The batch's samples, digests, counts, dropped records, generator/verifier fields and timestamps are byte-preserved and no `training_boundary` was back-filled, because that batch predates the checks such a block would claim. Training stays blocked until a new batch carries valid boundary evidence **and** its clip digests have been recomputed from audio that exists |

**The verification limitation is part of the finding, not a footnote to it.** The clip audio was never tracked —
`.gitignore` excludes `artifacts/**/*.wav` — and no WAV of this batch is present on this machine, so the
**8** SHA-256 digests in the manifest **cannot be recomputed from files**. The manifest itself is the only
surviving record of them. `training_quarantine.digests_verified` is therefore `false`, and the gate that reads
it will not accept a manifest's own assertion about its audio: a caller has to pass digests it hashed from
bytes, which on this machine no caller can do for this batch. The quarantine is consequently a **text-boundary
decision evidenced by the descriptor**, not a re-verified audio claim, and it is not nothing-alternative: no
number in this row depends on the absent files.

**Two defects in my own gate code, found by the tests written for this row and fixed in the same pass.**
`build_corpus_manifest` emitted the module-level `TEACHER_TARGET_RECORD` **by reference**, so any caller that
flipped one manifest's `enabled` flag flipped it for every manifest built afterwards — including the refusal a
trainer reads; the builder now deep-copies it and a test asserts two builds are distinct objects. And
`_strings()` accepted an empty list, so a manifest with `"usable_for": []` would have been reported as
"does not claim 'training'" rather than for the undeclared scope it is. Both were caught because the focused
tests assert an *accepted* case next to each refused one; a suite of refusals alone would have passed with
neither fixed.

**Statuses are not changed by this addendum.** A50, A52 and A53 keep the verdicts they carried before it, and no
other row's severity, wording or disposition was touched. A54's severity is proposed, not ruled. This row
records a quarantine and a documentation correction; it authorises no generation, no training run, no commit and
no publication.
### Addendum 17 — two findings from the documentation pass (2026-10-17): recorded, not fixed

Both surfaced while correcting an API example in `docs/architecture.md` (D-17), and both are handed to this
register rather than folded into that edit. Neither is a privacy or rights finding; both are
release-readiness items. No severity below is ruled, and nothing here is closed.

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A55 | **proposed S3 — awaiting owner ruling** | `src/nativevoice/integration/adapter.py:84-85` -> `src/nativevoice/session/orchestrator.py:130` | The facade's `hear()` forwards `language=` to a session method that does not accept it, so the documented one-shot transcription entry point raises on every call instead of transcribing | `adapter.py:85` is `return self.session.hear(audio, language=language)`; `orchestrator.py:130` is `def hear(self, audio, token=None)`. A read-only stub probe (no engine, no device, no network) raised `TypeError: VoiceSession.hear() got an unexpected keyword argument 'language'` both for `voice.hear(audio)` and for the exact forwarding line, while the **positive control** through the same session - `exchange()`, which reaches that method without the keyword - returned a `HostReply`; the probe therefore isolates the keyword path, not a broken fixture. Coverage: **4** `.hear(` call sites exist in `tests/*.py` and all four are on a `VoiceSession` built at `tests/test_session.py:88` (`:101,107,115,118`); the two test files that mention `VoiceInterface` contain **none**, and `tests/test_integration.py` exercises only `stream_hearing` (`:132`) | **OPEN - RECORDED, NOT FIXED.** Owner instruction of 2026-10-17: record it as a separate defect and leave the implementation alone in this pass. The corrected public example carries no reference to this row. Fixing it touches `src/` and needs a facade-level test alongside, so it requires its own approval; until then `VoiceInterface.hear()` is unusable and hosts go through `exchange()`/`converse()` |
| A56 | **proposed S2 release-readiness blocker - awaiting owner ruling** | `tests/test_no_personal_paths.py:59-87`, consumed at `:273` | One declared evidence path is untracked, and the consumer that checks it reads the filesystem, so a clean clone fails its own privacy guard | `EVIDENCE_MUST_BE_SCANNED` holds **23** paths: **22** tracked, **1** not - `artifacts/provenance/faster-whisper-large-v3.json`. It is untracked but **not** ignored (`git check-ignore` exits 1), so it is a force-add hazard rather than a protected one. Set arithmetic against a fresh clone's file set (exactly the tracked paths) yields `missing = ['artifacts/provenance/faster-whisper-large-v3.json']`, which fails `test_scan_coverage_is_reported` at `:274`; the **positive control** on this working tree yields `missing = []`, so the check does match. The second consumer (`:311`) derives its set from `git ls-files` and is immune: **18** tracked artefacts under `artifacts/` carry `[USER_HOME]` and all **18** are declared. `README.md:60` instructs a clean-clone reader to run `python -m pytest -q` with no ignores, and there is no CI (`.github/workflows` does not exist), so this is what a new user meets first. **INFERENCE, not an executed result: the module was never run here** - it is excluded from the authorised subset - and these outcomes come from reading `:250` and `:106-118` plus the arithmetic above | **OPEN - RECORDED AS A BLOCKER (Option 5, owner ruling 2026-10-17).** No edit to `tests/test_no_personal_paths.py`, no run of the excluded module, no creation or staging of the provenance JSON. Its untracked state is recorded as a fact at `docs/audit-register.md:559`, and no ruling was found that keeps it untracked deliberately, so the disposition is still the owner's. Options 1-4 were presented and declined for this pass: track the file, delete the declaration, soften `:273`, or correct the pointer at `src/nativevoice/provenance.py:52` that sends a reader to a path a fresh clone will not contain |

**Statuses are not changed by this addendum.** A50, A52, A53 and A54 keep the verdicts they carried
before it. This addendum records two rows and authorises nothing: no implementation change, no test edit,
no test run, no staging, no commit and no publication of either row's subject.

### Addendum 18 — the offline-proof timing discrepancy (2026-10-17): recorded, preserved, not reconciled

Surfaced while a staging snapshot was being verified, not while anything was being measured. The protected
working-tree file and the staged/committed file are two different successful runs of the same offline proof,
and nothing in the tree says so. It is a record-consistency item, not a privacy or rights finding, and no
severity below is ruled.

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A57 | **proposed S3 record-consistency - awaiting owner ruling** | `artifacts/audit/offline-proof.json` | The same protected evidence path carries two different successful timing runs depending on which layer is read, so "the offline-proof timings" stays ambiguous until the layer is named: the index and `HEAD` hold the earlier run, the working tree holds a later one | Read-only byte and JSON comparison; no measurement was re-executed. `git rev-parse :artifacts/audit/offline-proof.json` and `git rev-parse HEAD:artifacts/audit/offline-proof.json` return the *same git object*, `dfb62dac109048d61164fe83ab83e8f7e0f196cf` (577 bytes, content sha256 `cebcc99f456589be…`), so the staged version is the committed version, not a third state. The working tree is 597 bytes, sha256 `f8ae6c9e753a04e8…`, and it is that later digest which the protected set pins for this path. The gap is **not** line endings: the file holds 18 CRLF pairs, and LF-normalising it still leaves 579 bytes against 577. Both parse to the same 17 keys; 11 are identical, including `tts_asset`, `stt_asset`, `heard`, `heard_language`, `guard_clean: true`, empty `attempts` and `static_findings`, `verdict: "PASS"` and `tts_audio_seconds: 2.595` - the same synthesised utterance, so the changed numbers are compute, not a different sample. The 6 differing fields are all timing: `tts_load_seconds` 2.144 -> 2.118, `tts_wall_seconds` 3.511 -> 2.723, `tts_real_time_factor` 1.353 -> 1.0493, `stt_load_seconds` 0.89 -> 0.642, `stt_wall_seconds` 2.329 -> 2.145, `total_seconds` 9.14 -> 7.93. `git log` shows 7 commits touching the path, the last `511a660` "Refresh verification artefacts from the evaluation pass" dated 2026-10-13 19:39:38 +0530, while the working-tree mtime is 2026-10-13 23:26:22 - later than the commit that produced the staged bytes, which is what makes the working tree the newer run | **OPEN - RECORDED, PRESERVED, NOT RECONCILED.** Owner ruling of 2026-10-17: keep the working-tree bytes intact, do not stage or modify this JSON in this pass, do not regenerate the evidence and do not reconcile the two measurements. The consequence is stated rather than left implicit: the snapshot prepared in this pass publishes the index/HEAD run (2.144 / 3.511 / 1.353 / 9.14), so the newer run stays as uncommitted local state and a fresh clone will not contain the numbers the pinned protected digest describes. Which run is authoritative is a separate owner decision |

**Statuses are not changed by this addendum.** A55 and A56 keep the dispositions recorded in Addendum 17, and
no earlier row's severity, wording or disposition was touched. This addendum records one discrepancy and
authorises nothing: no edit to `artifacts/audit/offline-proof.json`, no staging of it, no regeneration of
evidence, no re-measurement, no commit and no publication.

### Addendum 19 — two identity findings from the pre-commit screen (2026-10-17): both corrections applied

Both surfaced while screening the staged diff for a commit, and neither is a licence or rights finding. A
hostname was about to enter tracked prose for the first time in this repository, and a personal name
belonging to a third party was about to be copied into a file where it had never appeared. The owner approved
two minimal corrections on 2026-10-17 - one prose deletion and one fixture substitution - and both are
recorded here as applied at the tip rather than as open defects. Neither row's severity is ruled, and neither
correction changes a committed evaluation artefact, a protected file or any historical evidence.

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A58 | **proposed S3 privacy - a device identifier entered tracked prose; severity still awaiting owner ruling** | `docs/provenance.md:122` | The Windows computer name of the machine this work is done on was written into documentation prose, so committing the reviewed set would have published a live device identifier with the repository | Found by screening the **9,149 added lines** of the staged diff across all 55 paths, with twelve patterns each proven to fire on its own fixture before any zero was read. **Newly introduced, not inherited:** the pre-edit `git show HEAD:docs/provenance.md` held **0** occurrences of the token and the index blob held **1**. A token census over all tracked paths after the correction reads **0** worktree files, **1** index blob (the pre-edit staged version, retired by the re-stage below) and **0** files at `HEAD`. Scanning every one of the **360** blobs in the object store returns **2** matches and both are transient rather than published: the index blob just named, and one **24,011**-byte intermediate that no ref and no index entry names. Restricted to the **300** blobs reachable from a ref the count is **0** - VERIFIED - so the name has never been committed. INFERENCE, not measured: a ref-scoped push transfers what is reachable from the pushed ref, so neither transient blob is in the set a push would send. `COMPUTERNAME` on this machine equals the token, which is what makes it a live identifier rather than a placeholder. It is a hostname, not a credential, and not the account name of A52 - the same screen returned **0** home-shaped absolute paths, **0** token shapes, **0** secret assignments, **0** bearer headers and **0** case-insensitive account-name hits | **APPLIED AT THE TIP (2026-10-17), owner approval of the same date.** Option A taken: the parenthetical is deleted, so the clause reads `and on this machine that` - one line, `-9` bytes, file 24,038 -> 24,029, line-endings and line count unchanged (274 LF, 0 CR), result 97 characters. The technical statement is preserved intact: the directory and the source attachments were measured absent on the machine used for this work, which is what the paragraph claims and what a hostname was never needed to say. The wording also matches the repository's own convention: measured over tracked paths after the correction, `this machine` occurs **25** times across **11** files under `docs/`, **0** times in `README.md`, and **0** tracked files name the computer. Option B (`on the machine this pass ran on`, 113 characters) was not used because it exceeds the wrap of its own paragraph. Because the token sits in **0** ref-reachable blobs, this edit closes the datum for the whole repository: unlike A52, no history rewrite is implicated. Nothing else on the page changed - the replacement is a fixed-string substitution whose surrounding byte segments were compared for identity before writing |
| A59 | **proposed S3 privacy - a third-party personal name was copied to a new location; severity still awaiting owner ruling** | `tests/test_evidence_paths.py:156,158` | A real speaker's Chinese personal name, used as a fixture filename in the `[USER_HOME]` boundary tests, had been staged into a file that did not previously carry it - two further published copies of an identifier belonging to someone outside this project | The name is the basename of an A50 Chinese self-description recording, registered at `tools/multilingual_eval.py:70`, and it is **already published**: **14** occurrences across **8** tracked paths at `HEAD` (five `artifacts/evaluation/multilingual-*.json`, `docs/multilingual-eval-2026-10-13.md:34,78`, `tests/test_multilingual_eval.py`, `tools/multilingual_eval.py`). The staged diff adds **2** of those occurrences in `tests/test_evidence_paths.py`, a file **new to Git**, and removes **1** elsewhere (`tests/test_multilingual_eval.py`, retired by an earlier approved edit inside this same staged set), so before this correction the snapshot was net +1; the line-level screen attributed exactly those three changes and nothing else. The fixture's load-bearing properties are that the value is a `Downloads` path, ends `.mp3`, sits beside an integer and contains non-ASCII text; the identity of a person is not one of them, and the module's own docstring at `:10-11` states that fixtures are assembled from fragments precisely so a test cannot become the leak it guards against. Behaviour was proved unchanged before editing: throwaway copies run outside the repository with the repository interpreter gave **23 passed** unmodified and **23 passed** with the proposed synthetic tail | **APPLIED AT THE TIP (2026-10-17), owner approval of the same date.** The real name is replaced by a three-character synthetic tail meaning `example name`: the module stays **12,925** bytes with a **zero** byte delta, both lines keep their widths (79 and 70 characters), 262 LF, 0 CR, and the diff is two lines with no layout drift. Verification after the edit, in place: `tests/test_evidence_paths.py` -> **23 passed**, identical to the pre-edit baseline and run with the same flags as the throwaway copies; **no other module was run**, and no excluded module was touched. A mutation control recorded during review stays recorded rather than argued away: an **ASCII** fixture filename also passes **23**, because `assert "\\u" not in text` at `:160` cannot fail on ASCII-only input - so the non-ASCII property this test exists to protect is carried by author intent, not by an assertion. Pinning it would be test expansion and needs its own approval; it was not done here. **This pass touched no other carrier.** The five evaluation artefacts are protected and byte-identical at `HEAD`, in the index and in the working copy; `docs/multilingual-eval-2026-10-13.md` and `tools/multilingual_eval.py` still carry the name in the staged bytes this snapshot already had before this pass, produced by earlier approved edits; and `tests/test_multilingual_eval.py` reaches this snapshot with its occurrence already removed by one of those edits, not by this one |

**Statuses are not changed by this addendum.** A55, A56 and A57 keep the dispositions recorded in Addenda 17
and 18, and no earlier row's severity, wording or status was touched. **A57 remains open and unreconciled:**
`artifacts/audit/offline-proof.json` was not opened for writing, not staged and not regenerated here, and
nothing in this addendum should be read as closing the timing discrepancy it records. This pass wrote exactly
three files - the prose correction, the fixture correction and this register - staged exactly those three
paths one at a time, ran exactly one already-authorized test module, and performed no commit, no push, no
audit run and no change to credentials, remotes, branches, visibility or configuration.

### Addendum 20 — A60: a fixture shaped like a third-party sample name (2026-10-17): correction applied

This finding is the residue of the screen that produced Addendum 19, reviewed as its own item after
that pass closed. It is not a proven transcription and it is not claimed to be one. The defect is that a
fixture in a test module new to Git had exactly the shape of the real third-party sample filenames this
repository evaluates, in a module whose own docstring states that every fixture is synthetic and that no
third-party filename is reproduced - so a reader cannot tell, from the published text alone, whether the
string was invented or transcribed. The owner approved a minimal synthetic replacement and one factual
correction on 2026-10-17; both are recorded here as applied at the tip. No evaluation artefact, protected
file or historical evidence was opened for writing, and no earlier row was touched.

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A60 | **proposed S3 privacy - a test fixture was indistinguishable from a real third-party sample filename; severity still awaiting owner ruling** | `tests/test_multilingual_scoring.py:49-50` | The unspaced-cue fixture was assembled from a speaker-role prefix, a timestamp of the same form this project's own recordings carry, and a quoted Chinese sentence, which is precisely the shape of the real third-party sample filenames enumerated at `tools/multilingual_eval.py:67-81`; the module's docstring at `:18-21` states that **every fixture is synthetic** and that the third-party names those tests replaced **are not reproduced here**, so the published file asserted a property it was shaped in a way that stopped a reader from checking it | Exposure measured two ways before editing, so neither zero is vacuous. `git grep -l` over `HEAD` returns **0** blobs for the full thirteen-character run and **0** blobs for the role prefix - VERIFIED - and the same screen over the working tree attributes both occurrences to this one file, which is staged as an addition and whose worktree bytes equalled its index blob `1efc651e…` byte for byte. The positive control is the six-character prefix of that run, `今天天气很好`: **10** occurrences across **7** `HEAD` paths (`tools/evaluate.py:79`, `tests/test_stt_real.py:287`, `tests/test_multilingual_verification.py:87`, `tests/test_tts_sherpa.py:351`, `artifacts/evaluation/roundtrip-2026-10-12.json` and `.md`, `docs/audit-register.md:24`), which is why this row does not allege that a private sentence was published - that phrase is the project's own long-standing synthetic reference text - and why the screen needed both a run length and a frame to separate the fixture from real sample names. **UNVERIFIED and not resolvable from repository evidence:** whether the string was invented or written down from a real file. The fixture's tested property is that a cue can arrive as one unbroken token after the timestamp, which is a property of its *shape*, not of its provenance - proven by the ASCII siblings at `:46-47` and `:51`, which exercise the same functions and carry no identity at all. Behaviour was measured, not assumed, in a throwaway copy outside the repository built from `src`, three `tools` modules, `tests/conftest.py`, `tests/fixtures.py` and this module (102 files, no `.git`, no `artifacts`), with resolution proven from the copy's own paths: **39 passed** unmodified. Verbatim reproduction of the replaced string is not carried forward into this register, precisely so that this row cannot become the copy the finding was about | **APPLIED AT THE TIP (2026-10-17), owner approval of the same date.** Variant B taken, as approved: the quoted sentence is replaced by the thirteen-character hiragana recitation `あいうえおかきくけこさしす` - the Japanese analogue of an `abc` placeholder, a recitation table rather than anything a person could have been recorded saying, and not the real Japanese sample name registered at `tools/multilingual_eval.py:73` - and the timestamp frame is neutralised to `-2026-01-02-03-04-`. The role prefix `話者一` is **deliberately retained**: it is a common noun meaning `speaker one`, the non-spaced analogue of `SPACED_NAME`'s `Speaker-One`, and it is what makes the fixture still test that a prefix before the stamp never reaches the cue. The edit is three lines (`3` insertions, `3` deletions) with **no byte movement anywhere else**: the file stays **13,701** bytes, line 49 stays **56** characters, line 50 stays **31**, **264** LF and **0** CR, because both the old Han run and the new kana run are thirteen three-byte characters and the stamp swap is ASCII-for-ASCII of equal length. Every property is re-measured after the write: `quote_from_name` still returns exactly **one** token, `quote_tail_from_name` still equals it, `len == 13` holds, no character of the new run is in the splitter set at `tools/multilingual_eval.py:220`, the new run is fully non-ASCII, and the new fixture string occurs in **0** blobs at `HEAD`, exactly as the replaced one did, so neither is inherited from committed content. The module was run again in the throwaway copy: **39 passed** with the correction, identical to the baseline, plus **6 passed** for the six `TestQuoteFromName` nodes that own the cue boundary. Two mutation controls were then applied **to the copy only** and reverted, to prove the new fixture still binds the assertions instead of satisfying them vacuously: inserting a hyphen inside the cue returns **2 failed**, and removing the stamp's trailing dash so no cue boundary opens also returns **2 failed**; restoring the approved bytes returns **39 passed** - VERIFIED. Isolation of this pass: the repository module was never collected by pytest, no `__pycache__`, `.pytest_cache` or `artifacts` path was created or changed in it, and no excluded module, asset gate or full suite was run |
| A60b | **proposed S4 documentation accuracy inside a test - a character count in an assertion message was wrong; severity still awaiting owner ruling** | `tests/test_multilingual_scoring.py:82` | The failure message described the unspaced cue as `one 14-character 'word'` while the run it refers to is thirteen characters long, both before and after the A60 replacement, so the message that exists to explain a cue-boundary regression was itself inaccurate about the fixture | VERIFIED by measurement, not by reading: the fixture tail is **13** characters (`len == 13` asserted before writing), and `14-character` occurs exactly **once** in the file, at this line - so no other prose carries the same error and no second correction was needed. The count is invariant under the approved replacement because the old Han run and the new kana run are both thirteen characters, which is why the fix is one token rather than a rewrite | **APPLIED AT THE TIP (2026-10-17) as the owner's approved factual correction.** `14-character` becomes `13-character`; the line stays **111** characters, the assertion expression `[UNSPACED_TAIL]` and the compared values are untouched, and the surrounding test contract is unchanged - only the message string moved. This is an accuracy correction to a literal, not a wording preference, which is why it was made in this pass rather than deferred |
| A60c | **proposed S4 documentation accuracy inside a test - wording looseness created by the correction; recorded, explicitly not authorised in this pass** | `tests/test_multilingual_scoring.py:48,80` | The comment at `:48` calls the cue `the whole quoted sentence` and the test name at `:80` reads `test_an_unspaced_sentence_arrives_as_exactly_one_token`, but after the approved replacement the cue is a recitation table rather than a sentence, so two phrases now describe an older fixture than the one in the file | VERIFIED: both phrases still read accurately about `SPACED_NAME`, which is a sentence fragment, and inaccurately about `UNSPACED_NAME`; no assertion depends on either string, and both lines are outside the three-line diff that was approved and applied | **RECORDED, NOT FIXED (2026-10-17).** The owner ruled, in the same instruction that approved A60b, that the A60c cosmetic changes are not to be made in this pass, so they stay open with this ID rather than being folded into an approved edit. Renaming a test is a change to a published identifier and is left as a separate decision |

**Statuses are not changed by this addendum.** A55, A56 and A57 keep the dispositions recorded in Addenda
17, 18 and 19; in particular **A57 remains open and unreconciled** and nothing here bears on it. Two adjacent
items are recorded without being acted on: the dated ASCII fixture names at `:73-78` were left exactly as
they are, because they are synthetic and carry no identity datum, and the same six-character phrase still
occurs in the seven `HEAD` paths listed above, which this correction neither hides nor claims to remediate.
This pass wrote exactly two files - the fixture and this register - staged exactly those two paths one at a
time, ran one authorised module inside a throwaway copy outside the repository, performed no commit, no push,
no audit run and no change to credentials, remotes, branches, visibility or configuration.

### Addendum 21 — release completion (2026-10-17): one fix, three corrections, one determination

This pass continues release preparation under the owner's continuation directive of the same date, whose section 3
authorises the fix recorded in the first row and the two corrections in rows A61 and A62.
Nothing here restates an earlier row: A55's and A56's rows in Addendum 17 keep their wording, and the new dispositions
below are additions to the register rather than edits of it. A57 stays open and untouched. No path was staged by this
pass, so the audited index is exactly the one Addendum 20 left behind it.

| # | Severity | Component | Defect | Evidence | Status |
|---|---|---|---|---|---|
| A55 | **update to the row recorded at Addendum 17; that row is unchanged** | `src/nativevoice/integration/adapter.py:84-92` -> `src/nativevoice/session/orchestrator.py:130-132` | The facade's `hear()` forwarded `language=` to a session method that does not accept it, so the documented one-shot transcription entry point raised on every call | Reproduced before editing with a read-only stub probe (no engine, no device, no network): `TypeError: VoiceSession.hear() got an unexpected keyword argument 'language'`, and the positive control through the same session - `exchange()`, which reaches that method without the keyword - returned a reply, so the probe isolates the keyword path. The fix moves the argument where the session can honour it: `begin_turn(language)` then `hear(audio)`, which is what `exchange()` at `:134-135` and `stream_hearing()` at `:91` already do, because the session takes a language when a turn opens, not when it hears. Consequence recorded rather than glossed: `begin_turn()` supersedes an in-flight turn (`orchestrator.py:103-105`), so `hear()` now always opens a turn - identical to the two working entry points, and the only way a forced language can reach `TranscriptionRequest.language` at `:136`. Two regression nodes added to `tests/test_integration.py`: `test_hear_transcribes_one_finished_utterance` (one transcription invocation) and `test_hear_applies_the_language_it_is_asked_to_force` (the forced tag arrives, and `allow_auto_detection` is `False`). Focused run: `tests/test_integration.py tests/test_session.py` -> **50 passed**. The module now collects **21** nodes against **19** in both the index and `HEAD`, which is the whole of the suite delta below | **FIXED IN THE WORKING TREE - NOT STAGED, NOT COMMITTED.** The owner's earlier instruction in Addendum 17 to leave the implementation alone was superseded by section 3 of the 2026-10-17 continuation directive, which asked for reproduction and, if contractually correct, a fix with a regression test and a focused run. Staging is a snapshot decision and was not taken here |
| A56 | **update to the row recorded at Addendum 17; that row is unchanged** | `tests/test_no_personal_paths.py:59-94`, consumed at `:273-274` | One declared evidence path is untracked while the consumer that checks it reads the filesystem, so a clean clone fails its own privacy guard | The Addendum 17 row labelled this outcome **INFERENCE, not an executed result**. That label is now retired by execution, in throwaway copies outside the repository, each built with `git clone --no-hardlinks` and then given the staged bytes of all 55 paths, so each copy holds what this snapshot would publish: **candidate** (the provenance artefact absent, as a clone leaves it) -> **19 tests, 1 failure**, and the failure is `test_scan_coverage_is_reported` with the message `evidence files outside the scan: ['artifacts/provenance/faster-whisper-large-v3.json']` - exactly the one path, not a cascade; **control** (the same tree with that one file present) -> **19 tests, 0 failures**. Classification: the artefact is a release artefact and not a stray - it is not ignored (`git check-ignore` exits 1), it passes the identity screen, it is referenced from `src/nativevoice/provenance.py:52`, and its content is the byte-pin record that the Large-v3 row depends on | **EXECUTED PROOF REPLACES THE INFERENCE - STILL AWAITING AN OWNER RULING, NOT STAGED.** The two routes are asymmetric: adding the path makes a new evidence file public, so it changes publication scope and needs a ruling; leaving it out publishes a snapshot whose own guard fails in a fresh clone. No test policy was weakened to obtain either number, and the excluded module was run only inside the two throwaway copies |
| A61 | **proposed S5 diff hygiene - severity not ruled** | `tests/test_corpus_forge_boundary.py` EOF | The staged blob ends with a blank line after the final newline, so the published diff carries `new blank line at EOF` | Measured at both levels before and after: the staged blob is **91,134** bytes ending `...in message\n\n` (sha256 prefix `3e9db8397d3ea798`), the corrected working tree is **91,133** bytes ending `...in message\n` (prefix `807319eeedaefaa0`), and the first **20,000** bytes are byte-identical, so the change cannot reach a single assertion. `git diff --check` returns **0** after the correction; `git diff --cached --check` still returns **2** because the correction is unstaged | **APPLIED IN THE WORKING TREE - NOT STAGED.** One byte removed, no other edit to the file. It closes only the whitespace defect; it is not a reformat of the module and does not touch the corpus-boundary assertions |
| A62 | **proposed S4 documentation accuracy - severity not ruled** | `PROGRESS.md:211` | The measured-summary row attributed model-load times of **2.56 s** TTS and **1.18 s** STT to `artifacts/audit/offline-proof.json`, a figure that artefact has never recorded | Every revision of the cited artefact was enumerated rather than sampled - **7** commits touch the path - and the `tts_load_seconds` / `stt_load_seconds` pairs are `2.144/0.89`, `2.369/1.085`, `2.152/1.094`, `2.387/1.099`, `3.759/1.507`, `3.57/1.626`, `2.376/1.184`. **No revision holds 2.56 or 1.18 in any numeric field**, verified by exact value comparison after parsing rather than by substring search: a first pass that searched for the strings reported a hit on the earliest blob and that hit is `1.184` matching the prefix `1.18`, which is why the screen was redone. `2.56` occurs twice in the whole tracked tree: in this row, and in `artifacts/evaluation/roundtrip-2026-10-12.json` as `tts_audio_duration_s` **2.5657** - a length of synthesised audio, not a load time. INFERENCE, stated as such: the pair looks transcribed from a run other than the one the citation names | **CORRECTED IN THE WORKING TREE - NOT STAGED.** The row now reads `TTS 2.144 s, STT 0.89 s`, which is what the artefact holds in the index and in `HEAD` and therefore what a fresh clone will contain. It deliberately does **not** cite the newer working-tree run (`2.118` / `0.642`), because that run is uncommitted and its disposition is A57's, which this row does not prejudge. Line width and table shape are unchanged; no other row was touched |
| A63 | **proposed S4 documentation accuracy - severity not ruled; recorded, explicitly not fixed in this pass** | `docs/tts-options.md:14` | The same unsupported lower bound appears in a second document: the `Load` row of its measured-baseline table gives `2.56–3.7 s per language instance` and classes that figure `measured` | Every `load`-named key under `artifacts/` was read: the TTS latency laboratory holds **9** `load_seconds` values spanning **2.74–3.728**, the profile benchmark holds `tts_load_seconds` **2.066 / 3.151 / 2.94**, and the offline proof holds **2.144** staged. A screen for a load key equal to `2.5x` returns **0** files, with the positive control that `3.7x` returns exactly **1** (`3.728`), so the upper bound is supported and the lower bound is not. **INFERENCE:** "per language instance" also overstates what those rows measure, since the laboratory rows are thread and chunk sweeps of a rebuilt engine rather than per-language loads | **RECORDED, NOT FIXED.** This is adjacent to A62, not part of it: A62's authority is one line in `PROGRESS.md`, and editing a second document under that authority would be scope creep. Proposed correction for a separate ruling is to cite the measured span **2.74–3.73 s per engine rebuild** from the laboratory artefact, or to delete the row if the owner prefers not to re-scope the claim |
| A64 | **closure of the FIX-1 question - determination, not a defect** | `tests/test_evidence_paths.py:97-98` | Whether the non-ASCII filename-shaped fixture in the `[USER_HOME]` boundary test is synthetic, established without inferring anything from the script the string is written in | Four independent measurements, each with a control. **Uniqueness:** the exact fixture string occurs in **1** tracked file (**2** lines - the value and its expected redaction). **Form discrimination:** the spaced sequence used by the fixture occurs in **0** tracked paths other than that file, at worktree, index and `HEAD` alike, while the unspaced vendor sequence occurs in **7** `HEAD` paths, every one of them an evaluation artefact, a document or the sample table, and all of them a different name shape (`.mp3`, `-18-37-` time stamp). **Newness:** `git log -S` over the path returns empty and `HEAD`'s blob holds **0** matches, so the fixture exists only in the staged index. **Referent:** the unspaced phrase is registered at `tools/multilingual_eval.py:73` as a vendor voice-category label meaning "energetic woman", not a person's name. Two existence checks complete it, printed as booleans only so that no directory listing or path is exposed: no file anywhere in the working tree has a name containing the phrase (**219** files scanned, **0** matches) and the exact path the fixture describes is absent from the machine's downloads location, with a positive control returning true for a file this repository owns and a negative control returning false for a fabricated name | **DETERMINED SYNTHETIC - NO EDIT MADE, NO EDIT SOUGHT.** A constructed variation of a vendor label, not a transcription of an artefact path, so this is not an A59 or A60 recurrence. Recorded honestly as a limit rather than argued away: absence today does not prove the string never described a real file, and the determination therefore rests on the form discrimination above - every recorded real name datum in this repository is unspaced and `.mp3` - not on that absence. If the owner wants the residue removed anyway, the A59 substitution style (`示例名` plus an extension) is the pattern, and it needs its own approval |
| A65 | **proposed S5 record hygiene - severity not ruled** | `artifacts/audit/mechanical-findings.json` | A protected artefact reads as modified in the working tree while its published bytes are unchanged, which invites a re-stage that would look like an evidence edit | The three levels were compared as bytes: working tree **1,500** bytes with **56** CRLF pairs (sha256 prefix `d7d97731ea0274b4`), index and `HEAD` both **1,444** bytes, pure LF, identical to each other (prefix `1babf3fab05b3dbd`), and the LF-normalised working tree equals the index byte for byte, so `git diff` for the path is empty while `git status` shows it dirty. `.gitattributes:1` is `* text=auto eol=lf`. The file's mtime is 2026-10-13 23:26 and a scan of `artifacts/` for anything modified on 2026-10-17 returns **0** files, so no test run in this pass wrote it | **RECORDED, NO ACTION.** The published content is unaffected; opening this path for writing or staging it would change line endings in a protected artefact for no evidential gain. Distinguished from A57, where the working tree genuinely holds a different run |

**Verification of the whole tree, run after every edit above and before this row was written.** The documented release
command at `README.md:73-80` - the four `--ignore` paths plus the one `--deselect` node - returns **1,071 passed,
7 skipped, 1 deselected, 86 excluded, 0 failed, 0 errors** in **6.587 s**, taken from a JUnit report written outside
the repository because this environment's pytest suppresses the terminal summary line and a count read off a progress
line would be a guess. The arithmetic reconciles against collection rather than against memory: **1,165** nodes
collected over the whole tree, **1,078** selected here, **86** inside the four ignored modules
(`test_hardware` 31, `test_no_personal_paths` 19, `test_cli_diagnose` 18, `test_diagnostics_offline` 18) and **1**
deselected, and **1,078 − 7 = 1,071**. The baseline recorded at `README.md:82` before these corrections was **1,069 passed** for HEAD `6056f0e`,
so the delta is **+2**, which is A55's two regression nodes and nothing else - proved by the module collecting **21**
where its index and `HEAD` blobs define **19**. All seven skips are the asset-gated real-inference nodes in
`tests/test_stt.py` (1), `tests/test_stt_real.py` (2) and `tests/test_tts_sherpa.py` (4); none of them ran a model,
and no microphone, network, device-enumeration, audit-tool or asset-gated test was executed. The
excluded-module policy was not modified to obtain this result.

**Isolation.** This pass wrote five files and staged none of them: `src/nativevoice/integration/adapter.py`,
`tests/test_integration.py`, `tests/test_corpus_forge_boundary.py`, `PROGRESS.md` and this register. It created no
Git object, ran no `git add`, no commit, no push and no remote write; it changed no remote, branch, visibility,
configuration or credential; and it regenerated no evidence - the two protected artefacts under `artifacts/audit/`
keep their 2026-10-13 mtimes and their published bytes, and the newer offline-proof run stays exactly as A57 records
it. The A56 proof ran in two throwaway clones outside the repository, and the identity report stays a local untracked
file outside it, so no identity datum from that report is transcribed here: two distinct author and committer
identities are involved, one covering 27 of the 28 commits and one covering HEAD, and HEAD is the commit on no remote.

### Addendum 22 — the release candidate is staged (2026-10-17): one ruling applied, one stale figure corrected, one census refined

**The ruling.** The owner directed that Snapshot C be prepared as the release candidate: the five corrections recorded
in Addendum 21, plus the provenance artefact A56 had left open, plus the suite figure recorded here as A66. The two
protected artefacts under `artifacts/audit/` stay exactly as they are, `docs/vision.md` stays out of this release,
`docs/tts-options.md` stays unedited, and no commit and no push is made in this pass. Staging was one explicit path per
`git add` call - seven calls, no wildcard, no `-A`, no `.` - and the index was re-read after each one.

| ID | severity and status | where | what was wrong | how it was established | disposition |
|---|---|---|---|---|---|
| A56 | **S2 release-readiness blocker - ruled and closed by this staging** | `artifacts/provenance/faster-whisper-large-v3.json`, declared at `tests/test_no_personal_paths.py:85` and consumed at `:273` | A declared evidence path was untracked, so a clean clone failed its own privacy guard and the tracked citation at `src/nativevoice/provenance.py:47-52` pointed at a file the published repository would not contain | The two rows above (Addendum 17 and Addendum 21) stand unchanged; what was added here is execution against the staged content. A `git clone --no-hardlinks` copy was given the staged bytes of all 58 paths and they were staged **inside that copy**, so `git ls-files` there described a published tree rather than a directory of loose files, and the import was proved to resolve to the copy's own source by printing the adapter module path before running | **CLOSED - TRACKED, STAGED, NOT REGENERATED.** Candidate: **254 tests, 0 failures**. Control, the same tree with that one file removed: **254 tests, 1 failure**, naming only `artifacts/provenance/faster-whisper-large-v3.json`. Restored: **254 tests, 0 failures**. The staged bytes are the 7,217-byte record measured on 2026-10-14 - 110 LF lines, 0 CR, no account-name datum, no `Users/` form, no drive-letter path, three `[USER_HOME]` placeholders - so what became public is the byte-pin record and no new claim |
| A66 | **proposed S4 documentation accuracy - corrected in this pass** | `README.md:82` | The line read "Recorded result of that command in this tree: **1,069 passed, 7 skipped, 1 deselected, 86 excluded** (2026-10-16, HEAD `6056f0e`)" while the tree it describes had gained A55's two regression nodes, so the same command now returns 1,071 | The wording test was applied before editing rather than assumed: the sentence is present-tense about *this tree*, not a dated historical report of a superseded state, so it was a live claim and not a record. Its four figures were re-measured in this pass, not derived - **1,078** selected, **7** skipped, **1** deselected, **86** inside the four ignored modules, 0 failures and 0 errors, so **1,078 - 7 = 1,071**; collection over the whole `tests/` tree is **1,165** nodes. The stale `1,069` string itself was counted before being edited and occurs in exactly two tracked places - this sentence and the reconciliation sentence in Addendum 21 | **CORRECTED TO 1,071, dated 2026-10-17, attributed to `6056f0e` with this release snapshot staged.** The release commit was created after this row was written, so the hash that was then unavailable is now recorded: `b2df40394f8ff5e448387070a78e394ec2faf549`, parent `6056f0e`, tree `3f4a66162fa0677ef054141e5f0cc379d2987fcb`. That commit published exactly the index these four figures were measured against, so no figure changes here; the worktree the run used differs from that published tree only in the two deliberately unstaged protected artefacts - six measurement lines in `offline-proof.json` and line endings alone in `mechanical-findings.json`. The correction replaces three prose lines with four - that single net line is the whole of the **+1** insertion the staging statistics carry - and the four widths are now 100, 91, 98 and 37 columns (the file's widest pre-existing line is 201), LF-only, 0 trailing whitespace, `git diff --check` clean. The attribution was then reworded once more, from "this snapshot's five corrections" to "this release snapshot", because the staged set is not five files and a count in a present-tense sentence is as much a live claim as a figure is; that rewording changed no line count. The file's contribution across the release commit is **+210 / -59** (`6056f0e` to `b2df403`), which was not this pass's work: an earlier pass had replaced the README's opening, so `6056f0e` held 77 lines where the release tree holds 228. The Addendum 21 sentence that also carried `1,069` was made past-tense, so it remains a true record of the pass that wrote it |
| A67 | **proposed S5 record accuracy - the exposure census refined** | `docs/audit-register.md` Addendum 15, and the earlier report in this pass's conversation | Two figures needed re-measuring before they were relied on a third time, and one negative result was void | Addendum 15 records **6-25** files per affected commit; re-running the census over all **28** commit trees returns **5-25**, the low value being the initial commit `3c28575`, which carries the datum in 5 files because it tracked fewer files - the count of affected commits is unchanged at **24**, and `511a660` reproduces exactly **87 occurrences across 25 files**. Separately, a screen reported as searching history had used `git grep --all`, which is not an option this build's `git grep` offers: it was accepted silently and searched the working tree, where the datum is absent, so its "none" was vacuous. Replaced by `git rev-list` crossed with `git grep -l -I -F` per commit, with a positive control at `511a660` returning the 25 files | **RECORDED, WITH THE FIGURES ABOVE AS THE CORRECTION.** Nothing in the ruling changes: the same single account-name datum, in the same 24 commit trees, all of them already on the private remote, 0 occurrences at worktree, index, HEAD and the `origin/master` tree. The credential screen was re-run the same way over **355** blob objects (300 reachable from `master`, 207 index entries, 5,527,875 bytes) against ten shapes and returns **0**, with all ten positive controls passing - one control had to be lengthened before it matched, because a synthetic payload shorter than the pattern's own minimum would have made that zero meaningless too |

**Verification after staging, in place.** `git diff --cached --check` now returns **0**: the `new blank line at EOF` that
the staged blob carried is gone, which is A61 closing in the index rather than only in the working tree. The index holds
**208** entries over **208** distinct paths and **207** distinct blob objects - the two device artefacts that share bytes
account for the difference, so no path is duplicated and none is double-staged. The staged diff is **58** files over
**14** additions, **44** modifications and **0** deletions, and its insertion count grows with this Addendum: the first
staging measured **+9,372 / -473** with the register still carrying only the Addendum 21 text, the second measured
**+9,409 / -473** once these rows were in the index, and the **37**-line difference is this Addendum itself. An insertion
total or an index fingerprint quoted inside this file is therefore one stage behind the file that quotes it, because
staging the sentence changes the blob the sentence describes; the authoritative figures are the ones the staging report
carries, not the ones written here. What that report measured for the fingerprint was
`51dbab1dde9f89cfbb577fc458ca6bbed3009f2580c9fe6e71a0fc08e712f225` over **207** entries before this pass and a
**208**-entry value after it, and the move has an exact cause rather than being drift: of the seven paths this pass
staged, **one** was new to the index (`artifacts/provenance/faster-whisper-large-v3.json`) and **six** were already
tracked and were re-hashed. `HEAD` is still `6056f0e`, `master` is still one commit ahead of and zero behind the private
remote, and the remote list is unchanged.

**Isolation.** Seven paths were staged and nothing else was written to the index: the two protected artefacts remain
unstaged with their 2026-10-13 mtimes, `offline-proof.json` still differs from its own staged bytes by the six lines A57
records, and `mechanical-findings.json` is still line-endings-only dirty with staged bytes equal to `HEAD`. No commit, no
push, no remote write, no branch, visibility, configuration or credential change was made, and no evidence was
regenerated. Two method errors made earlier in this pass are retired here rather than quietly dropped: a JUnit report
written **inside** a throwaway clone was scanned by that clone's own privacy guard and produced two spurious exposure
failures - reports now live outside every copy under test - and a shadow probe that ran without the clone's source on
its path reported the audited repository's module, so the clone run it described did not prove what it was said to prove.
Both were re-run: the numbers in the A56 row are from the corrected simulation, in which the module under test is the
clone's own copy. The A55 fix, the A61 correction, the A62 figure and the Addendum 21 register text are all in the index
now, so the dispositions in those rows that read "not staged" describe the pass that wrote them and are superseded for
the snapshot by this row.

### Addendum 23 — the copyright/ownership attribution is unverified (2026-10-19): recorded, README wording neutralised in the working tree

**Why this row exists.** `README.md` asserted a copyright holder for the repository's own material and no record in
the tree corroborates it. The owner ruled that the claim be removed **without naming a substitute**, in the wording
variant that does not advertise the open question publicly, so the README now says nothing about ownership at all.
A landing page is not where that question gets settled and a register row is where it stays visible, so the finding
is recorded here rather than left in a conversation. The register itself was off-limits in the pass that made the
README edit, which is why the row is appended in this one.

| ID | severity and status | where | what was wrong | how it was established | disposition |
|---|---|---|---|---|---|
| A68 | **proposed S4 ownership attribution - severity not ruled** | `README.md` §8 pointer (pre-edit `:303-304`) and §9 "Ownership, contribution and contact" (pre-edit `:329-330`); the published copy at `main` = `94de4f9`, `README.md:3,218` | An **unverified copyright/ownership attribution**. The file stated `**Copyright © 2026 Actizant.**` and, at the head of the page, "Local-first speech infrastructure by Actizant", while nothing in the repository records the legal identity of Actizant, its authorship of the source, tests, tooling or documentation, or any assignment of rights in them. No ruling on ownership exists in `DECISIONS.md`, in this register or in `docs/provenance.md`. The assertion was published on 2026-10-18, so it is a live public claim rather than a drafting artefact, and editing a later working copy does not retire it | `git grep -n "Actizant" HEAD -- .` returns exactly **2** hits, both in `README.md` (`:3` and `:218`) and **0** in every other tracked file; the same grep over the working tree after the edit returns **0**. The root directory holds **no** `LICENSE`, `NOTICE`, `COPYING` or third-party-notices file. `pyproject.toml` declares **0** `authors`, `maintainers`, `license` or `licence` fields. A case-insensitive scan for `copyright` outside `README.md` returns **3** hits (`docs/provenance.md:95`, `docs/research-report.md:145,237`), all of them about **Pocket TTS's** missing holder line, i.e. about a third-party asset rather than about this repository, and `©` outside `README.md` returns **0**. History scope: `git log -S "Actizant" -- README.md` names **one** commit, `b2df403` (2026-10-17), and the per-tree census of all **30** commits on `master` - `git rev-list` crossed with `git grep -l -I -F`, the method adopted at A67 after `git grep --all` was found to be silently unsupported - finds the string in **2** trees (`b2df403`, `94de4f9`), **1** file each, and in **0** of the other **28**. Published scope, read anonymously with no credential offered: `git -c credential.helper= ls-remote https://github.com/Actizant/NativeVoiceEngine.git` returns rc **0** with `HEAD` and `refs/heads/main` both `94de4f932c74bb3938ec0268c38f9fe998cfb0ea`. Recorded as a fact and read as no more than a fact: the GitHub organisation hosting that destination is itself named `Actizant`, which evidences control of the repository account, **not** authorship and **not** an assignment of rights; this row draws neither inference | **RECORDED, NOT RULED.** The owner directed that the attribution be removed without naming any substitute holder, and chose the variant that does not state the open question publicly. Applied in the working tree on 2026-10-19 (`README.md` mtime 00:47), unstaged and uncommitted, two hunks only: §9 now opens `**No copyright holder is asserted in this file.**`, and the §8 pointer that used to send the reader to the notice reads "Ownership is addressed at §9, where none is asserted" - **5** lines introduced, longest **111** columns, **0** trailing whitespace, `git diff --check` clean, with the `**No licence is granted by this repository.**` sentence and every §8 blocker bullet untouched. That wording is precisely why this row is needed: it leaves the repository **silent** on ownership, not resolved. **This row establishes none of**: that Actizant does *not* own the material - absence of a record is not a finding of fact; that some other party (the product brand named in the README heading, an individual author, or any organisation) does own it; that the published claim is retracted or removed; that it is absent from any blob already fetched from `b2df403` or `94de4f9`; or that a copyright line is required at all. What it puts to the approver is the question the README cannot answer: who, if anyone, is asserted as holder, on what evidence, and whether a licence file is to exist. Two consequences the ruling has to cover. **(1)** The string stays fetchable from `b2df403` and `94de4f9` indefinitely; rewriting history to remove it is neither performed nor proposed here, and any such route would strand the citations this register already makes to those commits. **(2)** A ruling that a holder *should* be named is a different action with its own ID, requiring the corroborating record - assignment, entity, or an explicit owner ruling - before any name enters a published file. Severity deliberately not ruled; `S4` borrows the label of the documentation-accuracy rows above and the owner sets it |

**Scope of this pass.** The register is the only file written, and only by appending: the pre-image bytes are
unchanged as a prefix of the post-image, the disposition is `+n / -0`, and `git diff --check` on the path is clean.
`README.md` received no further edit after the two OPEN-C1 hunks recorded in the row above; nothing was staged; no
commit, push, tag, branch, configuration or credential change was made; and the two protected artefacts under
`artifacts/audit/` were neither opened for writing nor staged. A68 is the finding's only identifier: the label under
which this item was quoted earlier in the pass has no referent anywhere in the repository and is not repeated here.
As recorded at Addendum 22, any staging total, line count or fingerprint quoted inside this file is one stage behind
the file that quotes it, so this row deliberately quotes none of its own.
### Addendum 24 — a register row cited a file timestamp, and the timestamp moved (2026-10-19): one citation corrected by append, A68 untouched

**Why this row exists.** Addendum 23 records the README edit with `README.md` mtime `00:47`. A modification time is
not version-controlled: no commit, tree or blob in this repository carries it, and git does not preserve one across
a checkout, so a citation to an mtime was written to decay the moment it was written and cannot be re-verified by
anyone reading the published copy. The README has since been edited again by an authorized documentation pass, which
moved the value. The correction is appended rather than written into A68 because the register is append-only, and
A68's sentence stands as the record of what the pass that wrote it observed at the moment it observed it.

| ID | severity and status | where | what was wrong | how it was established | disposition |
|---|---|---|---|---|---|
| A69 | **proposed S4 record hygiene - severity not ruled** | `docs/audit-register.md`, the A68 disposition cell in Addendum 23: the `README.md` mtime citation | A dated filesystem timestamp was used as evidence for a working-copy documentation edit. The value was accurate when written and is now superseded; and in principle it was never verifiable from the repository at all, because mtimes are not git objects and a fresh clone assigns its own | Measured now: `README.md` carries mtime `2026-10-19 02:17:30 +0530`, which is **90** minutes after the `00:47` edit A68 describes and **82** minutes after this register was last written (`docs/audit-register.md` mtime `2026-10-19 00:55:34 +0530`). The later write is working-copy only: `git log -1 --format=%h -- README.md` returns `b2df403` - the same commit A68 already names as the one that introduced the attribution - `94de4f9` touched only this register, and `git diff -- README.md` is **2** hunks, `+348 / -206`, `228 -> 370` lines. Census of the pattern rather than an assertion: `mtime` appears on **6** lines of this file, **5** of them describe the two protected artefacts and still re-verify - `artifacts/audit/offline-proof.json` and `artifacts/audit/mechanical-findings.json` both read `2026-10-13 23:26` - so A68's README citation is the only dated mtime attached to a working-copy documentation edit | **CORRECTED BY APPEND. A68 IS NOT RE-DATED, NOT EDITED, AND ITS FINDING DOES NOT CHANGE.** Every content assertion A68 makes re-verifies against the current working copy: `grep -c Actizant README.md` returns **0**, `git show HEAD:README.md` still names it at `README.md:3` and `README.md:218`, §9 still opens `**No copyright holder is asserted in this file.**` at `README.md:332`, and the §8 pointer still reads "Ownership is addressed at §9, where none is asserted" across `README.md:306-307`; the row's status stays `RECORDED, NOT RULED`. **This row establishes none of**: that Actizant owns or does not own the material; that the `02:17` pass altered any ownership wording - on the re-verification above it did not; that A68's remaining evidence is unsound; that a timestamp convention exists or is hereby imposed for this register; or that any other row may be re-dated. What it does put to the approver is the durable half of the finding: a register meant to be checkable against a commit should cite git-recoverable anchors - commit, path, line - and this row records the one place where it cited something the repository cannot carry |

**Scope of this pass.** The register is the only file written by this addendum, and only by appending: the pre-image
bytes are unchanged as a prefix of the post-image, the disposition is `+n / -0`, and `git diff --check` on the path is
clean; no byte of Addendum 23 was edited. Three documentation-accuracy edits made in the same conversation are
deliberately **not** registered here, because no register row was authorized for them: the unsupported "original"
claim was removed from `pyproject.toml:8`, `src/nativevoice/__init__.py:1` and `docs/architecture.md:3`, each a
same-line-count replacement, and `docs/provenance.md:98` was left untouched because its "the deliverable must be
original" sentence is the §3 constraint imposed on this project, not a claim made by it. Nothing was staged; no
commit, push, tag, branch, configuration or credential change was made; the two protected artefacts under
`artifacts/audit/` were neither opened for writing nor regenerated - `tools/audit.py` was not run; and
`artifacts/evaluation/roundtrip-2026-10-12.json` and `.md` keep their branded titles, so the generator/evidence title
difference the approved `tools/evaluate.py` edit introduced stays documented rather than hidden. As recorded at
Addendum 22, any staging total, line count or fingerprint quoted inside this file is one stage behind the file that
quotes it, so this row deliberately quotes none of its own.

### Addendum 25 — ownership wording names the developing company as intended owner; the blanket notice stays withheld (2026-10-19): README §9 rewritten in the working tree, three files prepared, nothing published

**Why this row exists.** A68 left `README.md` **silent** on ownership, by ruling, and said so: it "leaves the repository silent on ownership, not resolved". The owner then ruled the other way on the first half of that question — the company developing the product is to be named — while withholding the second half: no `Copyright ©` notice over the tree, and no implication that contributor rights were assigned. Naming a holder is exactly the case A68 anticipated ("a ruling that a holder *should* be named is a different action with its own ID"), so it is appended here rather than edited into the earlier row. The same pass introduced a second, unrelated claim — that some third-party projects are dependencies of the engine and others are not — and a draft sentence overstated the behaviour of the refusing adapter; both are recorded below. A fourth item reached only a draft and never the published page: the first shortened §9 sent the reader to this row by its ID, which would have made a public page depend on a record still under review, and how that coupling was resolved is recorded in the disposition.

| ID | severity and status | where | what was wrong | how it was established | disposition |
|---|---|---|---|---|---|
| A70 | **proposed S4 ownership attribution - severity not ruled** | `README.md` §8 pointer `:305-307` and §9 "Ownership, contribution and contact" `:330-356` in the **working copy**, on top of `a7301d5`; the copy being corrected is the published one at `refs/heads/main` = `a7301d5`, whose §9 opens `**No copyright holder is asserted in this file.**` | Three statements were wrong in three different ways. **(1)** §9's blanket denial — no name, person, company or project is claimed as owner of the source, tests, tooling or documentation — misdescribes the project once the owner has ruled that a specific company develops it and is expected to hold it, and reads as a denial in the same file whose hosting organisation carries that company's name. **(2)** Nothing in the published page separated materials the engine's code paths load from materials this repository merely **examined**, so §8's licence warnings against Kyutai Pocket TTS, HiFiTTS-2, Silero and pyannote were readable as dependency disclosures for a product that declares none of them — the more so because Pocket TTS names an adapter file. **(3)** The draft that answered (2) then overstated the refusing adapter: it said the adapter "raises at every entry point", which `close()` falsifies, and it anchored that behaviour to the class docstring rather than to the code | Counted, not argued. `git show HEAD:README.md \| grep -c -i actizant` → **0**; the working copy → **3** lines carrying the name (`:332`, `:346`, `:355`), all of them inside §9, so a commit would introduce the entity name into the README for the first time, and would do so in the owner's own ruling rather than by inference from the hosting account. Anchors were preserved because this register cites them: §9 still **opens at `:332`** and the §8 pointer still spans `:306-307`, and the census that produced A71 found the register's highest README line citation to be `:332` — **nothing** at or beyond 367 — which is why §9's body could contract from 35 lines to **25** without stranding a locator. That measurement is of the register as committed at `a7301d5`: re-measured against the file this change produces, the `README.md:NN` occurrences go from **9 to 28** and the deepest citation becomes `:344`, which is inside §9's body. The conclusion stands, because 344 is still below 367, but from this commit forward an edit above `:344` can strand a locator belonging to this row, and that is the hazard A71's convention note exists to catch. The incorporation/research split re-verifies at the code and at the asset register, and those anchors live **here** rather than in the public section: `src/nativevoice/tts/adapters/sherpa_kokoro.py` and `src/nativevoice/stt/adapters/faster_whisper.py` are the two adapters behind the loaded assets; `docs/provenance.md:52-55` records `ABSENT, **do not fetch**`, `ABSENT, **PROHIBITED pending review**`, `ABSENT, **PROHIBITED in product**` and `ABSENT, **do not adopt**` for the Pocket weights, the 26 preset voices, HiFiTTS-2 and Silero/pyannote, and `docs/provenance.md:56` records `**NONE EXIST YET**` against weights owned by this project; `pyproject.toml:10` is `dependencies = []`, the only declared packages sitting in the `mic`/`onnx`/`dev` extras at `:13-15`. The overstatement in (3) was corrected before any commit, line by line against the source, and the surviving claims were then re-checked where they were overstated: `pocket_reference.py` imports nothing from any upstream synthesis package (`:4-5`), `capabilities()` returns `CapabilityStatus.NO_MODEL` for every reference and target language (`:151-164`), `voices()` returns `()` (`:166-167`), `condition`/`synthesize`/`stream` raise `CapabilityUnavailable` only *past* the gate (`:171,181,185`), `_gate()` itself raises `ModelNotApproved` for want of a manifest or of an approved review (`:189-206`, sites `:192,199`), `warm()` runs the gate and nothing else (`:173-174`), and `close()` returns `None` (`:176-177`) — so "every entry point" is false, and the working copy says only what survives that: the adapter "declares no usable capability and synthesises nothing". Routability was measured rather than assumed: `_ROUTABLE = (MEASURED, PARTIAL)` at `tts/engine_registry.py:28`, `accepted` adds only `DECLARED` when `strict=False` at `:128`, and `NO_MODEL` falls to the non-scoring branch at `:139`, so no route selects it in either mode. Evidence still **absent**, re-listed because it is why no notice was written: no `LICENSE`, `NOTICE`, `AUTHORS` or `CONTRIBUTORS` file at the root (measured **0**), **0** `authors`/`maintainers`/`license` fields in `pyproject.toml`, no ownership ruling in `DECISIONS.md` — S4 is the heading "Where does this project live, and under what name?" and rules on the working **directory** (at `:144` after the N11 entry this change inserts, `:128` before it), and N7 at `:21-23`, above the insertion point and unmoved, records the initial commit's `Co-Authored-By:` AI trailer — and no contributor agreement or assignment anywhere in the tree | **WRITTEN TO THE WORKING TREE, NOT PUBLISHED. NOTHING IN THIS ROW HAS TAKEN EFFECT ON THE PUBLISHED PAGE.** Three files carry this change and none is staged, committed or pushed: `README.md` (2 hunks, `+27 / −5`, 370 lines published → 392 in the tree), `docs/audit-register.md` (this addendum and Addendum 26, appended only) and `DECISIONS.md` (N11 inserted, additions only). `refs/heads/main` = `a7301d5` still publishes the §9 this row replaces, so every statement below about what the page "says" describes the **working copy** until a commit and a push that have not happened and are not authorized by this row. Two hunks: `README.md:305-307` and `:330-356`. In the working copy the §8 pointer's second line reads "which separates the project's intended ownership from third-party materials under their own terms", which names the claim as intended without asserting it legally. §9 `:332-356` opens by naming Actizant as the developing company and **intended** owner and states that "intended" is not a verified record of legal ownership of every contribution; the next paragraph lists what the repository does **not** hold — no `LICENSE`, `NOTICE`, `AUTHORS` or `CONTRIBUTORS` file, no assignment or contributor agreement, no `authors` field, no ownership ruling in `DECISIONS.md` — keeps the commit-history facts (individual accounts, the initial `Co-Authored-By` AI trailer), states the anti-inference warning outright — an entity name on the account that hosts the repository "evidences control of the account, not authorship of its contents and not a legal ownership position over every contribution in it" — and then names `docs/audit-register.md` and `docs/provenance.md` as where the evidence sits, **without naming a row of this register**. The third paragraph separates loaded assets (Kokoro, including the espeak-ng data bundled inside it, and the faster-whisper checkpoints, both named only as adapters and both conditional on the default-deny gate and a locally present approved asset) from **examined but not adopted** projects (Kyutai Pocket TTS, HiFiTTS-2, Silero, pyannote), and the section closes `**Nothing here grants a licence, transfers any third-party material to Actizant, or states that commercial redistribution has been cleared.**` **The forward-reference coupling, resolved.** The owner's ruling offered two routes: re-word the pointer to "the project's internal audit records", or publish this row in the same change so the ID resolves. Neither literal wording was used, for two measured reasons: `docs/audit-register.md` is a tracked file served from the published repository, so calling it an *internal* record on a public page misdescribes it; and naming `A70` would make the public page depend on a record whose addendum number is positional — A71 keeps its ID but would move to Addendum 25 if this row were declined. The route taken deletes the three words ` at A70` from `README.md:344` and keeps both file names, so the sentence is accurate whether or not this row survives review, and no other line of the README moves; `grep -c A70 README.md` returns **0**. If the approver prefers the precise pointer, it is one re-insertion away and then `README.md` and this register must land in the **same** commit or the page cites a row that does not exist. Implementation line references were deliberately removed from the public section and retained in the evidence cell above: a scan of `:332-356` returns **0** `(:NN)` locators. `git diff --check` is clean on all three paths; introduced README lines are at most **108** columns with no trailing whitespace and no CR, and **0** personal account names, home paths or email addresses appear in them; §8's blocker bullets (`:308-329`, same numbering as `HEAD`), the `**Contribution.**` block from `:358` onward (against `HEAD`'s `:336` onward), `docs/provenance.md`, `src/` and `pyproject.toml` are byte-identical to `HEAD`; region compares confirm the only differing README lines in the whole file are `:306-307` and `:332-356`, and `git diff -U0` reports exactly two hunks. **Recorded verification for this wording pass** — it is a regression check on documentation, not evidence about rights: `tests/test_provision.py` **44 passed**, `tests/test_no_personal_paths.py` **19 passed**, `tests/test_tts_adapters.py` + `tests/test_tts_pipeline.py` + `tests/test_stt.py` **82 passed, 1 skipped** (the standing `test_stt.py:354` refusal to run real CT2 inference on an unreviewed asset, unrelated to this pass), the whole suite **1,158 passed, 7 skipped**, and the exact command `README.md:253-259` documents returned **1,071 passed, 7 skipped, 1 deselected**, identical to the figure recorded at `README.md:261`; that number is therefore verified current and is not corrected here. **Consequence for the two rows above, stated so neither is silently broken:** a commit would remove the README sentences A68 quotes as the corrected wording and A69 re-verifies as still-present, including A69's quotation of the §8 pointer, whose two lines are exactly the `README.md:306-307` span this row rewrites. The locators themselves keep resolving — `:332` is still §9's first body line and `:306-307` still spans the §8 pointer — so a reader following either arrives at the right place and finds this row in place of the sentence the earlier row quoted. A68's and A69's findings are unchanged and neither row is re-dated or edited; this row follows the append-only convention rather than overriding it, and A71 records the locator drift as its own finding. **(3) is recorded as draft hygiene, not published error**: the overstated sentence never entered Git, so no fetchable blob carries it, and no reader of `main` was ever told that `close()` raises. **This row establishes none of**: that Actizant is the legal owner of any contribution in the tree; that contributor rights were assigned; that a copyright notice is required, or is hereby declined forever; that the engine uses Pocket TTS, HiFiTTS-2, Silero or pyannote — the opposite is recorded above; that a passing suite corroborates any of it; that this wording constitutes legal clearance; or that anything in §8 is resolved. A21 (espeak-ng GPL-3.0 data), the Kokoro non-redistributable record, the faster-whisper MIT-text and large-v3 `NOASSERTION` items, HiFiTTS-2 prohibited in product, Silero and pyannote with no licence recorded, `en-synthetic-01` `QUARANTINED` (A54), A57 and A65, and A68's exposure of the `b2df403`/`94de4f9` blobs all **remain open**, and a future push would not be legal clearance. What is put back to the owner is the closure evidence, not the wording: an entity record, an assignment or authorship record for anything authored outside it, and an explicit ruling on whether a `LICENSE` file is to exist at all. Severity deliberately not ruled; like A69 this is documentation and record hygiene, and the owner sets the label |

**Scope of this pass.** `README.md`, `docs/audit-register.md` and `DECISIONS.md` were written in the working tree; the register and `DECISIONS.md` were written **only by inserting whole lines** — for each file the `a7301d5` pre-image remains an unchanged prefix or subsequence of the post-image and `git diff --numstat` reports 0 deleted lines, which is the append-only proof this register already uses at Addendum 23. No earlier row was edited, re-dated or re-worded, including A68 and A69. Nothing was staged, committed, pushed, tagged or branched; no Git configuration or credential was touched and no history was rewritten. The two protected artefacts under `artifacts/audit/` were neither opened for writing nor regenerated and keep the working-tree bytes this pass found them with (`d7d97731ea0274b4…`, `f8ae6c9e753a04e8…`) before and after the test runs; they are **not** part of this change and must not be staged. Those bytes are not the published ones — `HEAD` holds `1babf3fab05b3dbd…` and `cebcc99f456589be…`; measured by CR-stripped comparison the mechanical-findings gap is line endings alone (56 CRLF pairs against 1,444 published bytes) and the offline-proof gap is the six timing lines A57 records and deliberately leaves unreconciled, with A65 recording the read-as-modified condition. This row adds nothing to either record and changes neither file. `tools/audit.py` was not run. As recorded at Addendum 22, any staging total, line count or fingerprint quoted inside this file is one stage behind the file that quotes it, so this row deliberately quotes none of its own. A71, drafted separately as Addendum 26 in the same change, is the citation-drift record; this row does not carry it.

### Addendum 26 — the register's README.md line citations resolved to a file that has since been rewritten (2026-10-19): the drift measured, migrated and reported

**Why this row exists.** A69 established the principle that this register should cite git-recoverable anchors. Four of its own rows cite `README.md` by line number with no commit qualifier, and the README was rewritten wholesale in `a7301d5` (+348 / −206), so those numbers now land on different prose. The drift was found while writing A70, which had to know which README line numbers it was allowed to break — and per the owner's standing rule an adjacent defect gets its own ID rather than a cell inside the row that happened to find it.

| ID | severity and status | where | what was wrong | how it was established | disposition |
|---|---|---|---|---|---|
| A71 | **proposed S4 record hygiene - severity not ruled** | `docs/audit-register.md` `:670` (Addendum 17), `:759` and `:765` (Addendum 21), `:792` (Addendum 22, the A66 row), and the qualifier-less half of `:856` (Addendum 24, the A69 row) — each of them cites `README.md:NN` | A locator, not a finding. Each of those rows cites a line of `README.md` **without naming the commit it resolved against**, so a reader who follows the citation lands on unrelated prose: `:670` points at `README.md:60` for the sentence that "instructs a clean-clone reader to run" the suite, `:759` at `README.md:73-80` for the release pytest command, `:765` and `:792` at `README.md:82` for the recorded result line, and `:856` re-verifies A68 by asserting that `git show HEAD:README.md` names the entity at `README.md:3` and `:218` — an assertion whose subject, `HEAD`, has since moved | All **9** `README.md:NN` citations in the register **as committed at `a7301d5`** were enumerated by regex and each was resolved against three versions of the file, so the classification is a census and not a sample. Appending Addendum 25 and 26 raises that to **28** occurrences and moves the deepest cited line to `:344`; all 19 additions sit in those two rows, and one of them re-quotes the `:306-314` span that `docs/research-report.md:190` uses in a third-party sense rather than citing this README, so the enumeration below is a census of the published register, not of the file after this change. **Correct-as-written, now drifted (4):** at `94de4f9`, `:60` is `python -m pytest -q`, `:73` opens the ```bash fence and `:82` reads "Recorded result of that command in this tree: **1,071 passed, 7 skipped, 1 deselected, 86 excluded**"; at `a7301d5` those same lines read "feature list: capability is a status, never a checkbox.", "four of them (`docs/capability-matrix.md`, generated 2026-10-12 from" and "\| the other 57 languages \| `NO_MODEL` \| `NO_MODEL` \|". **Migration, measured in the current working copy:** the quick-start command now sits at `README.md:248`, the release block at `:252-259` and the recorded result at `:261`. **Correct-as-written, commit-qualified, NOT drifted (1):** `:835` (A68) writes "the published copy at `main` = `94de4f9`, `README.md:3,218`" — and `git show 94de4f9:README.md` does hold the byline at `:3` and `Copyright © 2026 Actizant` at `:218`, so the qualifier does its job. **Ambiguous (1, at `:856`):** A69's same `:3`/`:218` pair is pinned only to the word `HEAD`, which was `94de4f9` when the row was written and is `a7301d5` now; at today's `HEAD` those two lines are "### Independent voice technology. Built for developers." and a capability-matrix bullet, and `grep -c -i actizant` on the published README returns **0**. **Drift this change creates, disclosed rather than corrected (1 pair):** `:856`'s `README.md:332` still resolves — §9's first body line, unchanged in position — but `:306-307` stops resolving to the sentence A69 quotes there, because Addendum 25 rewrites both lines of the §8 pointer. The locator survives, the quotation does not; that is the same failure mode as the four drifted rows, arrived at from the other direction, and it is recorded here rather than corrected in A69 because the register is append-only. Scope statement: the same regex was run over `docs/research-report.md` (**8** citations), `PROGRESS.md`, `DECISIONS.md` and `docs/provenance.md` (**0** each). The research-report hits are deliberately **excluded** here because they point at the upstream Kokoro and faster-whisper READMEs read from the research archives, not at this repository's file — and that ambiguity is itself a hazard worth naming, since `docs/research-report.md:190` cites `README.md:306-314` in a third-party sense one line above this register's own `README.md:306-307` pointer into §8. **Checked before writing, because this change inserts lines into `DECISIONS.md`:** a census of `DECISIONS.md:NN` citations across `README.md`, `docs/audit-register.md`, `docs/provenance.md`, `docs/research-report.md`, `docs/platform.md`, `docs/architecture.md`, `docs/capability-matrix.md`, `PROGRESS.md` and `DECISIONS.md` itself returns **0**, so the N11 insertion at `:40-54` strands no existing locator; the only numbers now quoted for that file are post-insertion (`:144` for S4, `:128` before) or above the insertion point (`:21-23` for N7). Method limit, stated rather than hidden: the regex counted the compound form `README.md:3,218` as one citation, so the true referent count is 10, not 9 | **RECORDED BY APPENDING THIS ADDENDUM TO THE REGISTER IN THE SAME UNPUBLISHED WORKING-TREE CHANGE AS ADDENDUM 25 — NOT STAGED, NOT COMMITTED, NOT PUSHED. NO HISTORICAL ROW WAS EDITED, AND NOTHING IS PUBLISHED YET.** The four drifted citations in Addendum 17, 21 and 22 stay exactly as written, because rewriting a row's evidence in place is what A69 was opened to stop; this row is the correction, and a reader who arrives at `README.md:60`, `:73-80` or `:82` from those rows now has the migration table above. **The substance of those rows survives the drift intact**: the recorded release figure they cite was re-run at this working copy with the command `README.md:253-259` documents and returned **1,071 passed, 7 skipped, 1 deselected**, identical to `README.md:261`, so a moved line number has not made a moved number. **That re-run produced a finding of its own, and it supports the argument below:** the same command with its fifth line omitted — the `--deselect tests/test_audio.py::test_device_enumeration_never_raises` at `:258` — returns **1,072 passed, 7 skipped** and **no** deselection. A prose summary of that block, or a citation of only its opening line, reproduces a different number from the one the page records; the command has to be taken whole, which is a further reason to cite re-findable content rather than a line span. Two consequences to put to the approver rather than settle quietly. **(1)** This register contains no convention requiring a commit qualifier on a line locator, so the drift is not a rule-break by the rows above; if one is wanted, it is a ruling of its own, and the cheapest form is to write `README.md:60 @ 94de4f9` in future rows. **(2)** A line locator is only ever as durable as the file it points into, and this one has been committed three times in seven days — `3c28575` (2026-10-13), `b2df403` (2026-10-17), `a7301d5` (2026-10-19) — twice with whole sections restructured, so the rows that matter most should cite content the register can re-find: a heading, a sentence, or a commit. **This row establishes none of**: that any cited figure is wrong; that Addendum 17, 21, 22 or 24 misreported anything when they were written; that A68's or A69's findings are affected; that any earlier row may now be edited; or that `docs/research-report.md` needs a companion row — its citations resolve against third-party documents this repository does not own and has not rewritten. Severity deliberately not ruled; like A69 it is record hygiene, and the owner sets the label |

**Scope of this pass.** This addendum and Addendum 25 are the only content this change writes into the register, and both are inserted **after** the last line of Addendum 24's scope paragraph: every `docs/audit-register.md:NN` locator cited anywhere in this file, including the `:670`, `:759`, `:765`, `:792`, `:835` and `:856` lines this row enumerates, sits above that insertion point and still resolves to the same content, because appending at the end cannot shift a line above it. Per Addendum 22 no staging total, line count or fingerprint of the register itself is quoted in this row; the append-only proof is that the committed pre-image survives as an unchanged prefix of the post-image with 0 deleted lines in `git diff --numstat`. Nothing was staged, committed, pushed, tagged or branched; no Git configuration or credential was touched, no history was rewritten, `tools/audit.py` was not run, and the two protected artefacts under `artifacts/audit/` were neither opened nor regenerated. The census was re-measured after §9 took its approved form — the working copy at **392** lines with §9's body at `:332-356`, against the published 370 — and after the final edit inside that body, which changed `:344` alone by removing the three words ` at A70`. Neither event disturbs the table above: the register **as published at `a7301d5`** cites README lines {3, 60, 73, 80, 82, 218, 306, 307, 332}, and the quick-start command (`:248`), the release block (`:252-259`) and the recorded result (`:261`) all sit **above** §9. The set this change produces is {3, 60, 73, 80, 82, 218, 248, 253, 259, 261, 305, 306, 307, 332, 344}, and `:344` does point into the interior of §9 — it is Addendum 25's record of the single line edited there. The safety claim is therefore restated where it belongs: nothing in the register reaches beyond `:356`, §9's last body line, which is why the body could contract to 25 lines, and from this commit an edit above `:344` would strand Addendum 25 — the first time this register cites the interior of a section it also rewrites. That is the practical value of the census — and the one locator this change does break, `:306-307`, is disclosed in the row above instead of being quietly repaired in the row that owns it.

### Addendum 27 — the Pocket TTS reference adapter is removed; A70's source anchors strand with it, and the mechanical audit cannot be re-run without touching a protected artefact (2026-10-20)

**Why this row exists.** The owner ruled that the project's model direction is its own: a Conformer-TDT-class recognition student against a Whisper large-v3 teacher, then a flow-matching synthesis student, with EmbeddingGemma 2 (740M) as the recorded training-side encoder. Under that ruling the reference adapter that documented a third-party flow-matching architecture lost its reason to exist, and the instruction was explicit that a file is not deleted for containing the word "Pocket" but only after its dependencies and purpose are traced. This row records the trace, the removal, the two consequences the register owns (A70's now-unresolvable `file:NN` locators and the README sentences A70 verified line by line), the re-measured counts, and one blocker the owner has not yet been told about.

| ID | severity and status | where | what was wrong | how it was established | disposition |
|---|---|---|---|---|---|
| A72 | **proposed S4 record and inventory hygiene - severity not ruled** | `src/nativevoice/tts/adapters/pocket_reference.py` (deleted, 227 lines), `src/nativevoice/tts/adapters/__init__.py:3-4,10-11,18,20,23`, `tests/test_tts_adapters.py` (`:1` docstring, `:5` `import sys`, `:9-21`, `:36-46`, `:90-157`), `README.md` `:56`, `:134`, `:163-165`, `:171-172`, `:219-222`, `:234-235`, `:245`, `:262-263`, `:350-353`, `docs/architecture.md` (new §7), `docs/tts-options.md` `:3-4`, `:46`, §4, §5, `docs/tts-kitten-trial.md` recommendation paragraph | An adapter that documented Kyutai Pocket TTS's architecture stood in the shipped package as `ENGINE_ID = "pocket-reference"`, exported from `tts/adapters/__init__.py`, carried 11 tests, and was named as a component in four documentation surfaces — while the project's direction had moved to its own student models, and while the README could only ever describe it as declaring no usable capability and synthesising nothing. Its presence also made §9's "examined but not adopted" sentence weaker than the truth: the tree held code shaped like the examined project | Traced before deleting, not after. A whole-tree scan for the six public symbols (`pocket_reference`, `PocketReferenceAdapter`, `POCKET_REFERENCE_ENGINE_ID`, `development_plan`, `MISSING_COMPONENTS`, `REFERENCE_LANGUAGES`/`TARGET_LANGUAGES`) returns **7 paths**: the module, its package `__init__`, `tests/test_tts_adapters.py`, this register, `.pytest_cache/v/cache/nodeids`, and the untracked `src/nativevoice.egg-info/SOURCES.txt` — i.e. **no** engine registry, orchestrator, session, integration, CLI, diagnostics, manifest, asset catalogue or committed artefact consumes it. Routability was already measured at A70 (`_ROUTABLE = (MEASURED, PARTIAL)` at `tts/engine_registry.py:28`), so nothing could select it. The architecture facts it carried are not lost: `docs/research-report.md:144-181` holds the Mimi VAE, 12.5 Hz / 24 kHz framing, ≤50-token chunk budget, ~4000-symbol SentencePiece front end and the ~100M/200 ms/6× claims with their tier labels, and `docs/provenance.md:52-53,95` keeps the rights findings. **Shared machinery was checked separately and kept**: `tts/phonemiser.py`, `normalise.py`, `chunking.py`, the engine registry, the orchestrators, `voices/` identity and consent, `models/` approval, integrity and provenance and `eval/` are all retained — none was removed with the adapter, and the `pocket` term in `tools/audit.py:142`'s host-independence deny-list stays in place, now guarding a removal instead of describing a component | **APPLIED TO THE WORKING TREE, PUBLISHED WITH THIS CHANGE.** The deletion is one coherent unit: module + its five package lines + the 11 adapter-specific tests and the `manifest()` helper and the four imports only it used. No test was weakened to make it pass: `TestPronunciationGap` (13) and `TestScriptedEngine` (5) are untouched, and the one assertion in the deleted class that was not about the deleted object — `"pocket_tts" not in sys.modules` — was carried forward as `TestNoUpstreamSynthesisRuntime` with two more guards (`importlib` now raises `ModuleNotFoundError` for the retired module; `tts.adapters.__all__` equals the four live symbols), so the file loses 11 nodes and gains 3. **Counts, measured not inferred:** `def test_` goes 1,004 → **996** across **43** unchanged files; the command `README.md:253-259` documents returns **1,063 passed, 7 skipped, 1 deselected** with the **86 excluded** re-counted at 86; the full suite (no ignores) returns **1,150 passed, 7 skipped** where A70 recorded 1,158 + 7 — the 8-node delta is exactly this change, so both figures reconcile. `README.md:245` now says 996 and `:262-263` states the live figure while keeping the dated `1,071` at `:261` intact. **Consequences for this register, disclosed rather than repaired upstream:** A70's evidence cell cites `pocket_reference.py:4-5, 151-164, 166-167, 171,181,185, 173-174, 176-177, 189-206` with sites `:192,199`, and **all of them are now unresolvable** — the blob is recoverable at `1998c0d`, and the readings behind them (a `NO_MODEL`-only capability set, `voices() == ()`, `close()` returning `None`) stay true of that historical file. A70 is not edited and not re-dated. Its README quotes do change: §9's clause "what this tree holds of Pocket TTS is a reference adapter that documents the target architecture, declares no usable capability and synthesises nothing" becomes "This tree holds no Pocket TTS code: the reference adapter that documented its architecture is retired, and the reading survives in `docs/research-report.md`", and §4's `refusal-only reference adapter` becomes the Kokoro front-end gate. Every README edit is **line-count neutral above `:358`** and the file is still **392 lines**, re-verified by reading each cited anchor (`:3, :60, :73-80, :82, :218, :248, :253-259, :261, :305-307, :308-329, :332, :344, :358`) after the edits, so no locator above 358 moved and the two prose lines added in `docs/tts-options.md` sit below its only cited anchors (`:14` here, `:20` in `README.md:200`). **The blocker, reported not bypassed:** `tools/audit.py` writes `ROOT / "artifacts/audit/mechanical-findings.json"` at `:568` with **no `--out` flag**, so re-running the mechanical audit after a source deletion necessarily modifies a protected artefact. Per the standing instruction neither protected file was regenerated, staged or committed; both sha256 values were recomputed before the first edit, after every test run and after the final edit and read `d7d97731…` and `f8ae6c9e…` — unchanged. The 12 checks were instead run **in-process with `write_findings` not called**: **11 PASS, 1 FAIL**. The FAIL is `error taxonomy` on `diagnostics/evidence_paths.py:58` (`raise TypeError` outside `core.errors`), introduced at `b2df403` and unrelated to this change; the artefact also holds **11** findings where the tree now defines **12** checks, so the committed evidence predates a check and its `error taxonomy` row reads `0 raise site(s)`. Recorded as an observation about a protected artefact this task may not rewrite — the fix is a separately authorised decision to regenerate it, and no count in this change is affected by it. **Documentation alignment in the same unit:** §5's two surveyed rows move from `CANDIDATE` to `PROPOSED — chosen design, nothing built`, the tier paragraph says no row still carries `CANDIDATE`, §2's trained-models row names the students and teacher, §6 item 2 is retitled "Train the native recognition model" with its blockers unchanged (`torch` in no dependency list, no trainer module, GPU validation proposed but not executed), item 8 names the synthesis student, and `docs/architecture.md` §7 records the stack — the **first appearance of Qwen3-TTS and EmbeddingGemma 2 in any tracked file**, recorded as decisions with no manifest, no provenance row, no approval entry and no computation-graph connection, which leaves them outside the default-deny gate exactly as §3's rules require. Third-party research stays attributed: `docs/research-report.md` and `docs/provenance.md` are unedited, and `docs/tts-options.md` and `docs/tts-kitten-trial.md` now carry supersession markers on their recommendation sections instead of having their measurements removed |

**Scope of this pass.** Eight paths are touched: one source module deleted, its package `__init__`, one test file, `README.md`, `docs/architecture.md`, `docs/tts-options.md`, `docs/tts-kitten-trial.md` and this register, and the register and the two TTS documents were changed only by in-place rewrites plus whole-line insertions **below** every cited anchor. The two protected artefacts, `artifacts/corpus/en-synthetic-01/manifest.json`, `docs/research-report.md`, `docs/provenance.md`, `DECISIONS.md`, `PROGRESS.md`, the scratch research trees under `%LOCALAPPDATA%\Temp\nvtts` and `nvsrc`, model caches and every path outside this repository are byte-identical to where this pass found them. Nothing was downloaded, installed, benchmarked or provisioned, no approval state changed, and no Conformer-TDT training code was written — the architecture is recorded, not implemented. **What this row does not establish**: that the retired adapter's absence means Pocket TTS licence findings are closed (A21 and `docs/provenance.md:52-53,95` remain open), that the planned stack is rights-cleared or compute-feasible on any machine, that EmbeddingGemma 2 or Qwen3-TTS has been examined for licence or availability at all, or that the stale `mechanical-findings.json` is evidence about the tree after this change.

### Addendum 28 — the root commit's unsupported AI co-author trailer is removed by a message-only rewrite of the whole chain, applied locally on 2026-10-20 and published to the remote branch on 2026-10-22; this row is the old→new map, the citations it strands here and on the published branch, and the limits of what the correction proves (2026-10-20)

**Why this row exists.** The repository's root commit carried a `Co-Authored-By:` trailer naming an AI model, and
GitHub renders that trailer as a second name in the contributor line on the repository home page. The owner states
that Claude was not used and asked for the attribution to be investigated and corrected if it is unsupported.
Phase 1 established what the metadata can and cannot show: the trailer is message text, not an identity field; it
appears in exactly **1** of the **34** commit objects; GitHub's REST API exposes no co-author field at all, so the
"committed last week" pairing is a presentation product of parsing that trailer; and no author, committer or email
field anywhere in the history names an AI model or an Anthropic address. Phase 1 also refused the inference in both
directions — the trailer does not prove Claude wrote code, and the owner's recollection does not prove the trailer
was invented. What can actually be corrected is this repository's own record, which is what this row records;
whether the hosting service's rendered contributor line changes is a separate question, answerable only by reading
the rendered page after a push, and this row does not answer it and does not assume its answer.

**What changed and what did not.** The commit graph was reconstructed with `git commit-tree` one commit at a time,
oldest first, each commit fed **its own tree**, its remapped parents, and its original author and committer name,
email, timestamp and UTC offset through `GIT_AUTHOR_*` / `GIT_COMMITTER_*`, with the message replaced only where the
exact-string rule fired: the final non-empty line byte-equal to the target trailer, preceded by a blank line. It
fired on **1** of **34** messages. That message went **1599 → 1544 bytes**; the removed suffix is exactly a blank
line plus the trailer line, 55 bytes. Every tree, every blob, every file content, every identity, every timestamp,
every parent relationship and the other **33** messages are unchanged. Nothing was deleted, no source file was
touched, and no unrelated history was rewritten for appearance. The rewritten historical chain is linear at **34**
commits with **0** merges and **0** tags. Adding this documentation commit on top of it is what makes the proposed
final `main` history **35** commits, and the two figures describe different things.

**The map, and why it could be written before the rewrite was applied.** A commit hashes only its own tree, parents
and message. Editing the **tip** commit — which is what inserting this table did — therefore cannot move the SHAs of
the commits beneath it. The 34 rows below were produced by a dry run that moved no reference, inserted, and then
re-produced by a second dry run and diffed row-for-row; the two are byte-identical. Neither this row nor
`PROGRESS.md` records the SHA of the commit they become part of — a commit cannot cite itself, and entry 20 there
declines to do so deliberately — so the corrected tip is named by content: it is the only commit whose message is this
documentation change, and by reference it is whatever `main` resolves to in the clone where the rewrite ran.

| original (still the remote's `main`) | rewritten (local `main`, not yet pushed) | tree — identical in both |
|---|---|---|
| `3c2857597f323227b4946ef10b073d760775a32a` | `25661e2af9b0200f8ac0fca5cd57be7af64ea1fc` | `bf289bcae1796633c37b2b7d57064bc757e242d2` |
| `ee1f78b05fb913ff212fd3a91e0f283260b21fda` | `f6cd2decab2335172881f1c13db5a10e371bd154` | `cc3c3e6591efbfb414b9dcde9a3db78bef62817a` |
| `e9b3da48717d63ae1155dbecab571c545ce0175e` | `7557dd012f57eea1d4a033df27541d42c6fdd72f` | `144421ccfeb49bdfc4996c6581f5cb5f4360a450` |
| `f8a49e122718ee5ef5bcb3a7ca32ceb2a393cf76` | `138f25e379a2c8f0d455c10c68b4490a55c02504` | `4553944ab23e5dd51949b0496179e5c1d81a868e` |
| `39dffeba719d3cfc69e11a885093cf0991aacde3` | `62ef33aa41630047ba17ca79dc5c6862b0aae3fc` | `ad936f827c54de64ec60c1bafb032e50774716ec` |
| `55d1cde7fe13aa0f0081b9c3639a3e2344ebc533` | `0591059d5febce677bc7d9c258b6fc5531059556` | `538b93be2af2630dc87aee8a1784574d8272a5e0` |
| `8dd36a020eb7696d746c7244e6372697ee28ff17` | `2103b807aa0571eb7d1b3182c65d525f320fe4da` | `e152aa4ced9854b865af08782cce8db873a1598d` |
| `2b100602374b51e64605d4040709d201e9931eef` | `fea0cc62343c23d52523ff77185ce712043b84b2` | `e05efbeb62faf5e436d5fb2dabb0052d3e0892aa` |
| `f7f4689806034d66d2d82b640f44dad7eff1f7d1` | `63a24b4ac33a379ea8d0e7a94495b11a741f8356` | `2405142d42671e31e99c7df330f75ce8d99ca287` |
| `e16e1e3c418389024b18109e74f2fe8fcd906fd1` | `032cba49b6092cd49b3f0414b79d07bdde4f577d` | `0da498368d3bf62a1bbae38113d5e4d38195b68c` |
| `150b20aec8a2e062e6e37bcfe9a78ffb22413ea2` | `a05621304f1c86ad2355ca98bcd6778505df7801` | `b24b84204644bf6fa883e984a573ecc4e99f7db0` |
| `3d1258c07772d8eca1ed8247493a99bcf2367444` | `23ef0f48e85392a3bd72d9b2cf2979c671c48f0e` | `d6e41b10b76f769ac2958989f6888da675d0e5fa` |
| `7c52db983c8c364bf5af7c400ac80395c60bcf1f` | `6f2c25b921842ed1e0f5f8908b66a70f169e6bba` | `c85f3c004cff799a9ab90f7c55e18f2bcdde0df9` |
| `bc21b2388c3a95f7b2a74ad47c709fa0d4552df3` | `bd5e5462f7751a51d007c75eb93c8275f36e9c44` | `6140db6069ad40b6a86faf53c14ee86ab6f47dd4` |
| `feecf75a3f7ccea2e023ce40ce78b0041cb0e5d7` | `e3ed598f1384d1f2ace0a47022fa81f5abc703cc` | `61e461e3693d4e8e7f001372b83b2e2b630dbe3c` |
| `2c433e4fcd33cc467717385d7e5d16bb110fee65` | `cca634b45925e2856f3b0eadea821054e97810f7` | `01ee9a70225555b690e29aebeeecda110b95c6db` |
| `9e9ef3d6c8fad49480bf0a1acf6fa6bab840472b` | `a8d7f06649eae2dd97242eb7ec2cc7faa12a38b6` | `baea1067721331548a60d5219d97e561be19d0f8` |
| `81b7fc115aa08aa1cef5265706d8b83d761725ef` | `5a02b953bb9b1ff09f4a67d79d5b12a77f0a532d` | `d53de68279590b07893912d4cdf2c1225bacbbf2` |
| `9594364b68025426919729e91d59a5b166ce205a` | `e1d040a12a6fe698c2d45174559b0ad743ef30ab` | `010b70a9014ff6f3d2f46c44f590d322d61b7972` |
| `67eb8a9944c2b2d51de3db6dd9f5c6235e278001` | `1249abbc13e5c0318402b4788505f2a6dab2fe07` | `a8519fd2473b6a941892823e7d123546e7dab0e0` |
| `676d181d2a4a307db595fc163bb25dd7540770f5` | `83381b3d79909bacec21f25429c056e9d40c866d` | `f12a8a1adfd795501cb003e9448eb7500e0688d1` |
| `426f545111038dd0444c2c2aa2e825d79efe7e86` | `8496844b5f25826a542d0d8141faccd309a59f9d` | `9cd5bed5aaaa3700b71c23b9cd59ec808731d805` |
| `0912e17a988b4949c43e33fe7354e31aa85fee3d` | `b140ad576d5a5f91213400129783629e12e80a7a` | `e24ea240b35bb87104a085a67ec734d6934e5761` |
| `511a6607df0fc49c86f5afef62933db29599773d` | `d9270c052cdf0df819507edb4781281ee5836f95` | `2ae42e29690cbf37967e025a20e9f438094ce493` |
| `c93a75f5df6e05d4a84ca2b6c6f038425ccf4b61` | `17a884648e81226d0468aa9dd52ac0575c1e027f` | `9cb8bc4987177bb2f91691c85ef35ef0c2cd9d73` |
| `8745bf3992b07da2f02244d9cb673c823e11ae4e` | `33a181701da2466fbebc4eb392d55a2ba634ef0b` | `bf9e5c4192218493ff83da4adf3dc5e8ea580585` |
| `24ec1d3ea419db91514132c43212748442f670af` | `fac13d3a1aa7c490bf1bcd2d29a6c587a1e5a02c` | `662bb5bf8f1c159b00ad6e575e10e730fd9b984b` |
| `6056f0e936ed12ff2617b7ed74e23f89c422931b` | `9d841793c4b4a3fa5c729a1cf6625b79381ccbb3` | `a18b05d1fca4cfc8c327bb5502d7d465e296d229` |
| `b2df40394f8ff5e448387070a78e394ec2faf549` | `849de9ab4d8cdeb5656ec7c058c4c11f8c08fa26` | `3f4a66162fa0677ef054141e5f0cc379d2987fcb` |
| `94de4f932c74bb3938ec0268c38f9fe998cfb0ea` | `69e764c7e748f1bad276dc18db9f04e056d76f4e` | `4992db5d6fc5ff1e7f7f36886f1183c3f1c4f541` |
| `5544624ccc9a751f32368f2ae2dd7de0b49f6253` | `e7c5e104eeb8766c21782d3d407b73a82b3059e1` | `0aab27716ccf1ecacd8206544537f50e9b053149` |
| `a7301d530028a37bf73b128c89dfd0a43d7a65f2` | `08dd32305aa08f09a1248ba211c48a3fe5431321` | `d1389e567fe8a178ef1fa2704dbdea11dcf92e5b` |
| `1998c0dd8c35189c6d9e1ae9e0fa6e4e5fec94fe` | `495636553a4f0159fa5aca9d6e78e9a8afa86e34` | `0ae70676e1db2c96a7f652d297523f6a86b4543d` |
| `6ab1e58631f584df4b06d8046a0738a3655bb52d` | `41ca01fa0430555955c1b4f61ae2a3172b12ade0` | `d231fc7a1453b224d897efc2d22cb48afb1c837b` |

**The convention governing every other commit name in this register.** The rows above are the only place where an
old SHA is replaced. Everything else is left exactly as written. The census this change was measured against is **83** short-form
commit references across **44** lines in **4** tracked files — `docs/audit-register.md` (65), `PROGRESS.md` (16),
`README.md` (1), `docs/relocation.md` (1) — counting each of the **14** cited 7-character prefixes as a whole token
with non-hex boundaries, so a full 40-character digest in the same file is not miscounted as a commit reference.
That is the pre-rewrite tree's figure; the census of the tree this change commits is re-measured in A73 (2) below.
**1** of those 83 was replaced, the operational constant at `:447`; the other **82** still sit on the line they were
written on. Rows A52 through A72, `README.md:261`, `docs/relocation.md:23` and the dated journal entries are records
of the objects that existed when those rows were written, and re-writing them would re-date the evidence. Row
**A52** at `:403` is the one row whose *conclusion* this change supersedes rather than merely outlives, and A73 (7)
below records that in full instead of editing A52. They now
name commits that are not reachable from `main`. Each one is answered by the map above, which names a commit with an
**identical tree** and an identical message except where stated. A reader who needs the old bytes can still have
them: the pre-rewrite tip is held as `refs/heads/pre-rewrite-main` in the working clone and as a verified,
self-contained bundle. Whether the hosting service continues to serve the old objects by full SHA after a push is
the host's behaviour, is evidenced by nothing in this row, and is not assumed by it.

| ID | severity and status | where | what was wrong | how it was established | disposition |
|---|---|---|---|---|---|
| A73 | **record hygiene created by this change - severity not ruled** | The census this change was measured against: **83** short-form commit references across **44** lines in **4** tracked files — `docs/audit-register.md` (65), `PROGRESS.md` (16), `README.md` (1), `docs/relocation.md` (1) — taken by whole-token match on the **14** cited 7-character prefixes with non-hex boundaries. The operational ones are `docs/audit-register.md:447` and the prose pin at `PROGRESS.md:1225-1227`. The residues are this file's `:589` trailer census, this file's own row **A52** at `:403`, `b2df403`'s own commit message, and the `README.md:340` sentence A70 paraphrases and partly quotes | A history rewrite of a **root** commit moves all **34** SHAs, so once such a chain is pushed any citation naming a commit by its old name stops resolving on the branch history; that push was executed on 2026-10-22, and the original chain is reachable locally at `refs/heads/pre-rewrite-main` and at the direct commit URLs the remote continues to serve, and the register's own append-only convention forbids editing those citations in place. Separately, one citation is not a historical record but a **live instruction**: the `PRE = "…"` constant inside the fenced audit procedure at `:447`, which a future auditor copies verbatim — left stale, the documented check fails with `unknown revision` and the audit step silently stops being runnable | Read-only, before the rewrite. `git rev-list --max-parents=0` gives the root; `git rev-list --parents` classifies **33** single-parent commits plus **1** root and **0** merges; the exact-string strip rule was applied to all **34** real messages and fired on **1**; each raw object was walked header by header to confirm no `gpgsig`, `encoding` or `comment` header had to be carried through the reconstruction; the trailer-shaped final line was searched across all **34** messages to confirm the rule cannot over-match; the citation census was taken by regex over every tracked file at `HEAD`, not sampled. Counts that describe **contents** rather than identifiers — `:584` "24 of the 27", `:588` "28 of 28", `:590`'s 270-blob scan — were re-checked against the rewritten chain rather than assumed: a message-only rewrite reuses every tree and preserves committer dates, so they still hold. `artifacts/corpus/legacy-batches.json` pins two 64-char content digests and **zero** commit references, so the corpus gate is not exposed to this rewrite | **APPLIED IN THE SAME CHANGE.** **(1)** `:447` is the one exception made to append-only, line-count neutral, re-worded once more on the same date so the comment says which chain each form of the name resolves on, with the reason inline on the line itself and the map above — a silently broken audit step is worse than one annotated constant. **(2)** The other **82** references are **untouched**, at their original line numbers — re-checked position by position against the pre-rewrite tree rather than asserted: for each of the four ref-bearing paths the (file, line) pairs that held a reference still hold the same one, the sole exception being the operational constant at `:447`. What this row's first draft then said about the committed tree was wrong and is measured here instead. Pre-rewrite: **83** references over **44** lines in **4** files (`docs/audit-register.md` 65, `PROGRESS.md` 16, `README.md` 1, `docs/relocation.md` 1). Committed by this change, re-scanned with the same whole-token rule over all **207** tracked paths: **85** over **45** lines in the same **4** files — `docs/audit-register.md` **67**, `PROGRESS.md` **16**, `README.md` **1**, `docs/relocation.md` **1** — and no other tracked path carries one. The delta is fully accounted: **+2** in this file, being the **3** occurrences of `b2df403` in this row (its first draft said the row named it twice) less the **1** reference at `:447` that is now a 40-character digest the rule does not count (that line still carries one older counted reference, which this change leaves alone). 83 + 2 = **85**. **(3)** `:589`'s "**1**, on the initial commit" is left as the dated measurement it was; the re-scan of the corrected chain is stated here instead: the same rule, run over all **34** rewritten messages, returns **0**, with its positive control being the pre-rewrite chain at `refs/heads/pre-rewrite-main`, which returns **1** on the same command, and its negative control being the P-0 acceptance suite, where a commit carrying a *different* trailer keeps it verbatim. **(4)** `PROGRESS.md:1225-1227` is left as written and re-anchored by a new dated entry, not edited. **(5)** Drift this row creates, disclosed rather than repaired upstream, and the attribution this row's first draft made of it was wrong: the row whose §9 description `README.md:340` strands is **A70** at `:878`, not A69. Measured — A69 cites `README.md:3`, `:218`, `:332` and `:306-307` and never mentions the trailer, so no quotation of it is stranded; A70 both asserts that §9 keeps the commit-history facts "(individual accounts, the initial `Co-Authored-By` AI trailer)" — which `:340` no longer says — and quotes the anti-inference sentence as "evidences control of the account, not authorship of its contents and not a legal ownership position over every contribution in it", a contiguous string the rewrite of `:340-341` broke. A70 is not re-dated or edited. In the same file, `DECISIONS.md`'s N7 has grown and moved to `:21-33`, the S4 heading is still at `:160`, and A70's `:21-23`, `:144` and `:128` DECISIONS locators are stale as locators while remaining true measurements of the date they were written; this row is the correction, in the same form A71 used. **(6)** The accepted residue: `b2df403`'s message asserts "History is preserved as it stands: 28 commits on master, **no rewrite**" and that sentence is carried **verbatim** into the corrected chain, because fixing a non-target commit message is exactly the unrelated rewrite this task forbids. A reader of the corrected history will find a commit that denies the rewrite that produced it; this row exists so that they do not have to work it out alone. **(7)** The residue this file creates in itself: row **A52** at `:403` states that the earlier commits "still carry the original paths and remain publicly fetchable, as do the author name and personal email, the `+05:30` offsets, the initial commit's co-author trailer and the A50 third-party transcripts", and closes "Rewriting history has been declined." Both were true as written on 2026-10-14, and most of A52's enumeration is still true of the corrected chain: because this rewrite changes no tree, every path, author name, personal email, `+05:30` offset and third-party transcript it lists is carried by the corresponding commit here too. Two clauses are superseded — the trailer is no longer in the root commit of this chain, and rewriting was later ruled on and executed. Neither is edited out of A52: the append-only convention holds, so A52 stands verbatim, is re-dated only by this item, and its personal-path exposure finding stays open and is unclosed by anything in this addendum. After the 2026-10-22 push no remote ref points at the original chain, but the remote still serves those commits at their direct URLs, the original root among them with its trailer, so the A52 claim about public readability holds of those objects by URL while it no longer holds of the branch. **(8)** What this change establishes none of: that Claude did or did not write any line of this code; that any earlier attribution statement was accurate; that the contributor line on the hosting service has changed — that is verified against the rendered page after the push and is not claimed here; that any of §8's licence blockers moved; or that the counts in rows A52-A72 were wrong when written. Severity is the owner's to set |

**Scope of this pass.** Four tracked paths are written by this change and nothing else: `README.md` (one line pair
rewritten in §9, file still **392** lines, every register-cited README anchor re-read and still resolving),
`DECISIONS.md` (N7 resolved and its dating qualified, N9 narrowed, N11's trailer clause corrected, file still
**224** lines; the literal token `DECISIONS.md:NN` occurs **0** times in the tree, and the bare
DECISIONS locators that do exist — A70's `:21-23`, `:144` and `:128` — are answered in A73 (5) above),
`PROGRESS.md` (a new dated entry appended below every cited line) and this register (`:447` re-anchored in place,
so no pre-existing line moved: the `:24` and `:559` self-citations still resolve to the same text, and this addendum
is appended below them, taking the file from **900** lines to **1009**. No source file, no test, no artifact
and no configuration was modified. Both protected artefacts under `artifacts/audit/` were re-read before the first
edit and after the last and still hold their recorded digests, and neither was opened for writing, staged or
regenerated. The rewrite ran in a throwaway clone of the remote's `main`, outside this working copy; the permanent
working copy's own references were not touched. Rollback evidence: the pre-rewrite chain is kept as an in-clone ref
and as a bundle that `git bundle verify` reports as recording a complete history, and that bundle was restored into
a separate empty repository and re-checked against the original tip's tree to prove the rollback is real rather than
asserted. Nothing was pushed. The remote was read anonymously — with the credential helper explicitly disabled, so
the check could not consume or damage the stored credential — before the clone, before the rewrite, and again
immediately before the push decision.
