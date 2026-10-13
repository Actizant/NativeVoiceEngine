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
