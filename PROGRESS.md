# Progress ledger

Concise running record required by §15 Phase B. Updated as work lands; the acceptance report at the end
supersedes nothing here — dated entries stay, corrections are appended.

Timestamps are 2026-10-12. Test counts are the output of `python -m pytest` on the named files, executed on
this machine.

## Complete and tested

| Area | Files | Tests | Evidence |
|---|---|---|---|
| Core contracts: audio types, error taxonomy, cancellation, streaming events, timings | `core/{audio,errors,cancel,events,language,time}.py` | 34 in `tests/test_core.py` | executed green |
| Audio transport: PCM maths, band-limited + linear resampling, ring buffer + look-back, hysteresis VAD, segmentation, device I/O, WAVE container | `audio/{pcm,resample,buffer,vad,io,preprocess,wav}.py` | 45 in `tests/test_audio.py` | executed green |
| Voice session: turn model, partial/final/retract reducer, barge-in, stale-answer discard, listening modes, device recovery, disabled-voice routing | `session/{state,orchestrator}.py` | 25 in `tests/test_session.py` | executed green |

**104 tests, all passing.** Standard library only; no model, no microphone and no network is required to
run them, which is the point.

Defects found and fixed during implementation, all caught by tests rather than review:

- `bytes_per_frame` and array-element stride were conflated, making every frame-indexed operation 2x off for
  16-bit PCM. Split into `bytes_per_frame` / `samples_per_frame`; a test now pins both.
- `AudioBuffer.zeros` and `fade_edges` tail indexing were wrong; polyphase filter rows were allocated in
  bytes where doubles were needed, so every band-limited resample raised.
- `AudioBuffer.from_bytes` leaked a raw `ValueError`; now a `ConfigurationError` naming the byte count.
- The VAD segmenter counted 20 ms analysis *frames* where output offsets had to be *samples*, producing
  negative `first_sample` on any utterance with look-back padding.
- `resample(..., mode=<bad>)` was not validated when source and target rates matched; validation now precedes
  the shortcut.
- The session's disabled-voice note reported "no synthesizer attached" whenever voice was switched off,
  because an inverted ternary picked the wrong cause. Two branches, two causes, asserted separately.
- `begin_turn` interrupted the previous turn without recording `superseded_by`, so stale-answer detection had
  nothing to read.

## In build (four concurrent work streams)

`models/` + `hardware/` + `config/` · `language/` + `stt/` · `tts/` + `profiles/` · `eval/` +
`diagnostics/` + `cli/`. Their status and test counts are recorded when they land; nothing in this table is
claimed on their behalf until their tests have been run by me, not reported by them.

## Unresolved implementation dependencies (what genuinely blocks completion)

1. **No approved speech model exists.** Recognition and synthesis quality are therefore unmeasurable. This
   is `DECISIONS.md` S1, and until it is answered the capability matrix reports `NO_MODEL` for every
   (task, language) pair. It is not a code gap — the interfaces, gates and harness are built — it is an
   asset-authorisation gap.
2. **No Bengali or Hindi synthesis path exists anywhere.** OpenWhispr's registry has no Bengali entry;
   Pocket TTS ships seven languages with no Indic one. A ULTRON-trained model is required, with the data and
   compute that implies (§13 anticipated this).
3. **`torch` is absent**, so the training pipelines can be specified and partially exercised but not run at
   scale. S2.
4. **No licence-usable evaluation corpus** for any language. Latency can be measured with synthetic audio;
   accuracy cannot, and `eval/samples.py` refuses synthetic fixtures in a quality path by design. S5.
5. **Free RAM was 0.56 GiB at inspection.** Any profile threshold or RTF number measured in that state is
   invalid; thresholds are marked `measured=False` until a run happens on an otherwise-idle machine. S5.
6. **`ruff`/`mypy` absent**, so static typing is a hand-pass. S2.

## Known gaps deliberately left as gaps

- Wake-word detection: interface plus a refusal, no backend. §9 forbids counting it.
- Continuous listening: mode exists and requires explicit consent, but has not been exercised against a real
  microphone.
- Neural (acoustic-model) VAD: the energy detector is real and tested; the modelled slot is `NO_MODEL` and is
  not silently filled by the fallback.
- Grapheme-to-phoneme / pronunciation: no approved lexicon or model, so the provider reports `NO_MODEL` for
  every language including Bengali and Hindi, rather than inventing romanisation rules.
- Voice conditioning (cloning): storage, consent, quality gating and portable export are built; the acoustic
  conditioning step needs an approved synthesis model and names it when refusing.
- System-audio loopback / AEC for meetings: needs a native binary; shipping one is an unapproved provenance
  decision, so it is absent and recorded, not faked.
- Compressed audio import (mp3/m4a/ogg): needs ffmpeg; the WAVE reader raises with the reason instead of
  pretending.

## Test status

`python -m pytest -q` → **104 passed** at this entry. No test has been deleted, weakened or marked skip to
reach that number, and none is expected to be.

---

# Progress ledger — entry 2 (2026-10-12, later same day)

Authorisation relied on: *"Complete TTS and voice profiles using suitable pretrained models.
Do not train large models from scratch; prioritize local inference and verify the licensing of
every selected model and dependency."* This retires S1/S2/S3 of `DECISIONS.md` as answered:
pretrained assets are approved for use, licence verification is mandatory first, and no large
model is trained from scratch.

## Full build landed

All twelve packages exist. **`python -m pytest` → 584 passed, 1 skipped, 15.0 s**, no microphone,
no model and no network required for the 584. The single skip is the real-inference test waiting
on the asset work below, and it names that reason.

New since entry 1: `models/` `hardware/` `config/` (71 tests) · `language/` `stt/` (62) ·
`tts/` `profiles/` (162) · `eval/` `diagnostics/` `cli/` (~200) · `session/` (29) ·
`integration/` (24) · `tests/test_suite_hygiene.py`.

## Asset provisioning — licences verified before use, not after

`tools/provision.py` (new, standalone, never imported by the runtime) fetched and manifested:

| Asset | Task | Licence evidence | Size | Status |
|---|---|---|---|---|
| `kokoro-int8-multi-lang-v1_1` | TTS | **Apache-2.0 text shipped inside the package** (`LICENSE`, read directly) + HF tag `license:apache-2.0` on `hexgrad/Kokoro-82M` | 205 MB, 377 files | installed, manifested |
| `faster-whisper-base` | STT | HF tag `license:mit` on `Systran/faster-whisper-base`; already resident, nothing downloaded | 141 MB | manifested |
| `faster-whisper-small` | STT | same | 464 MB | manifested |

Runtime for both is **already installed and licence-clean**: sherpa-onnx 1.13.8 (Apache-2.0) and
onnxruntime 1.30.0 (MIT) for TTS; faster-whisper 1.2.1 + ctranslate2 4.8.2 (MIT) for STT.
**No package was installed and no other model was downloaded.**

Rejected on licence grounds, with the evidence recorded rather than assumed: Coqui XTTS-v2
(`license: other` — non-commercial), F5-TTS and ChatTTS (`cc-by-nc-4.0`), fish-speech
(`cc-by-nc-sa-4.0`), and the sherpa-hosted piper Indic voices carry **no licence tag at all**, so
their provenance is incomplete and the approval gate refuses them by design. Voice cloning
therefore has **no permissively-licensed local asset**, and remains an explicit gap rather than a
feature.

Two constraints recorded against the approved asset, because they are true and not cosmetic:
the Kokoro package bundles an **espeak-ng data directory under GPL-3.0**, which is fine for local
use and carries redistribution obligations if the asset is ever shipped; and
`onnxruntime` reports only `CPUExecutionProvider` here, so **TTS cannot use the GPU at all on
this machine** even though CT2 sees one for Whisper.

Language coverage was verified from the files, not the model card: dedicated lexicons exist for
`en-us`, `en-gb` and `zh` only. Hindi and Bengali text produced audio via the espeak-ng path,
which proves *output*, not *intelligibility* — that is what the round-trip measurement is for.

## End-to-end workflow and independence

`integration/` is the host seam: `VoiceInterface.exchange(audio, host)` runs
capture-shaped bytes → transcription → host answer → synthesis → playback, and `converse()` drives
the same path from a continuous stream with barge-in. A host implements exactly one method
(`respond`). Proven by test, not asserted:

- text-only is a routing decision: with `voice_output_enabled=False` the capability checks still
  run and zero frames reach the speaker;
- a refused language ends one turn, not the conversation;
- **no host application module is imported anywhere** in the package tree, and **no network module
  is imported by any package on the speech path** (`diagnostics/offline.py` excepted on purpose —
  it is the guard that refuses sockets);
- the engine **persists nothing**: three exchanges run under a temporary working directory leave
  the directory empty, `history` is a bounded in-memory deque, and `diagnose()` reports
  `history_persisted: false`. ULTRON owns persistent memory; FieldMind owns its own; this layer
  duplicates neither.

## Defects found and fixed in this stretch

1. **`CapturingOutput.stop()` permanently disabled the sink**, so playback could never resume
   after a single barge-in — a real product bug that only surfaced once the whole path was wired
   end to end, and invisible to every earlier unit test. Replaced with generation-based halting:
   a stop abandons the in-flight span, later spans play, and a cancelled token still drops its own
   work. Three tests that had encoded the wrong invariant were retargeted to the right one.
2. `LanguageTag.parse` lost the region when a variant subtag followed it (`de-DE-1901`,
   `bn-Beng-IN`); subtags are now classified by shape rather than position.
3. `SynthesisEvent` had no place for per-chunk diagnostics, forcing overloads of `text`/`reason`;
   it now carries `metadata`.
4. `ResourceLimitExceeded.__str__` hid the one actionable sentence behind sorted details; it now
   leads with the explanation and keeps the numbers.
5. `Transcriber.stream()` omitted a cancellation token, which made "honour cancellation at every
   yield boundary" unenforceable for anything but the happy path. Added to the protocol.
6. `ResourceRequirement.as_dict()` dropped `measured_on`, turning a measured figure into an
   unsourced one on serialisation. Pinned with a round-trip test.
7. An acceptance criterion in `test_eval_end_to_end.py` judged a 0.9 s resample against an absolute
   5 ms ceiling; restated as a **real-time factor** (0.044 measured, 0.25 allowed), which is the
   number that actually says whether a profile can sustain a live stream. Report rows are now
   looked up by name, not by position.
8. **My own provisioning tool was not idempotent** — a second run re-downloaded 140 MB and left a
   duplicate tree, which would also have broken digest resolution because the registry expects
   assets under `root/model_id/`. Fixed: skip-if-present with `--force` to refetch, always flatten
   after extraction, and the duplicate was removed (205 MB, 377 files, verified).
9. One reported defect was **not** a defect: the claim that `pcm.measure()` mis-divides RMS was
   checked against hand calculation on 7-frame and stereo buffers and is correct, because
   `AudioBuffer` guarantees whole frames. Left unchanged, stated here rather than silently kept.
10. A duplicated block in `tests/test_audio.py` — two identical `TestInputs`/`TestOutputs` classes,
    where the second definition had silently replaced the first — was found and removed. This is
    why `tests/test_suite_hygiene.py` now exists: it fails the suite if any scope defines the same
    test twice, if a test file contributes nothing, or if a test is unconditionally skipped.

## Still unmeasured, stated plainly

- No WER, RTF, time-to-first-audio or peak-RAM figure exists yet for any engine. The assets are
  installed and English synthesis works when driven directly, but **no capability is marked
  MEASURED** until the round-trip evaluation completes. In progress now.
- Every hardware profile threshold remains `measured=False`. Free RAM at the last check was
  3.51 GB (was 0.56 GB at baseline); benchmarks still need a deliberately idle machine to be
  worth anything.
- Bengali and Hindi synthesis intelligibility is unverified; Bengali recognition is unverified.
- Real microphone and speaker I/O has not been exercised on this machine, only the device
  enumeration and the software paths.

---

# Progress ledger — entry 3 (2026-10-12, audit and measurement)

## Real inference is live and measured

Both engines run through the model registry, the licence gate and integrity verification. Nothing was
installed and nothing downloaded after the earlier provisioning.

| Measurement | Value | Where |
|---|---|---|
| English round-trip WER / CER | **0.0494 / 0.0242** | `artifacts/evaluation/roundtrip-2026-10-12.json` |
| TTS real-time factor | **1.55–2.38** — slower than real time at every thread count tried | evaluation + `artifacts/benchmarks/profiles.json` |
| TTS first-audio | **3.9–4.8 s** (eval harness) and **6.3–8.3 s** (profile harness) | both retained, neither discarded |
| STT real-time factor | **0.85** at 2 threads, **0.50** at 4, **0.44** at 8 | profile benchmark |
| Peak RSS during evaluation | **708.8 MiB** | evaluation artefact |
| Peak RSS per profile | 1,001 / 1,015 / 1,046 MiB at 2 / 4 / 8 threads | profile benchmark |
| Model load | TTS 2.56 s, STT 1.18 s | `artifacts/audit/offline-proof.json` |
| Offline proof | full TTS→STT round-trip, **0 connection attempts**, static scan clean, verdict PASS | `artifacts/audit/offline-proof.json` |
| Assets verified by digest | 377 kokoro files + 4 + 4 whisper files, all installed/verified/usable | registry `status_all()` |

Chinese was measured and **failed** (CER 0.463 against a 0.10 bar): the asset renders simplified, Whisper
transcribes traditional, and no approved converter exists. Hindi and Bengali synthesis are refused because
the asset has no front-end for them; recognition for both stays vendor-declared and unmeasured.

## Hardware validated on real devices

Microphone capture sustained real time — 3.98 s of audio in 4.099 s wall, 199 chunks, zero frames dropped —
and the VAD produced **no false turn-opens** on 4 s of a real room. Speaker playback emitted 24,000 frames,
recovered after a stop with 6,000 more, and a mid-span barge-in emitted 12,000 of 72,000 frames then
stopped cleanly. Artefacts under `artifacts/device/`. Real *speech* through the microphone has not been
captured yet: that needs a person speaking, so VAD-on-voice and ASR-on-human-audio remain unverified.

## Audit (Phase C) outcome

`tools/audit.py`: 11 mechanical checks, **11 PASS / 0 FAIL / 0 NOT RUN**. Register:
`docs/audit-register.md` — 21 findings (5 × S1, 6 × S2, 7 × S3, 1 × S4, plus recorded clean results).
Sixteen fixed with regression tests; five open, of which four are measured limits or decisions rather than
bugs.

The five S1 findings were all invisible to unit tests and surfaced only through end-to-end, hardware or
registry-driven execution: Latin-script LID always answering English at confidence 1.000; `stop()`
permanently disabling the speaker; the raw output stream never being started; the registry discovering zero
manifests; and integrity resolving every asset against one wrong root.

Two findings were about honesty rather than correctness. `mark_measured()` let a single benchmarked
threshold certify an entire profile — now tracked per field, so partial evidence keeps a profile
provisional. And a measured failure had nowhere to live except `UNSUPPORTED` — now `MEASURED_FAILED`,
unroutable like a refusal but evidence-bearing unlike one, with a completeness test that fails if a future
status is added without being wired into the matrix.

## Suite status

`python -m pytest` → **645 passed, 7 skipped** in 13.6 s.
`python -m pytest tests/test_tts_sherpa.py tests/test_stt_real.py --run-approved-assets` → **32 passed**
(the model-backed tier; those are the 7 skips when run by default). No test was deleted, weakened or marked
skip to reach these numbers, and `tests/test_suite_hygiene.py` now guards against duplicate definitions,
empty test files and unconditional skips.

## Genuine remaining blockers

1. **Synthesis is not real-time on this machine** (RTF > 1, first audio 4–8 s), at every profile. Options: a
   smaller or faster asset, GPU execution (needs an onnxruntime build with a CUDA provider — absent here),
   or accepting press-to-hear instead of conversation. A measured ceiling, not a code defect.
2. **Bengali and Hindi synthesis needs a different asset or a trained model**, and no licensed human-speech
   corpus exists for either, so their recognition accuracy is unmeasurable on this machine today.
3. **Distribution is blocked pending a licence pass**: the approved TTS asset bundles GPL-3.0 espeak-ng
   data, and every zero-shot cloning candidate surveyed is non-commercial or carries no licence at all.
4. **Profiles are not validated** — one of five threshold fields has real evidence behind it.
5. **A human is needed at the microphone** for the two claims only real speech can support.

---

# Progress ledger — entry 4 (2026-10-12, platform repositioning)

Authorisation relied on: *"an original, standalone, local-first voice platform that can power multiple
independent AI applications, including ULTRON, FieldMind and future projects. It must not be designed
exclusively around ULTRON."* Nothing earlier was discarded; the audit register, artefacts, manifests and
regression tests from entries 1–3 stand.

## What changed

**`voices/` is a new package: voice identity as an installable, versioned package, separate from the
engine.** An identity carries an id, display name, per-language status with metrics, accents and styles,
required engine/model/slot/quantization/sample-rate, provenance (creator, training method, licence,
permissions, redistributability, consent basis), measured quality/latency/resource figures, and
install/update/removal records. `VoiceIdentity.as_reference()` produces the engine-facing form; the
engine receives a **slot number**, never a name to parse.

**Hosts are first-class and isolated.** `HostProfile` gives each application its own root, recordings,
profiles and logs directories, plus its own default voice. Resolution order is explicit → host default →
platform default, and an explicitly requested voice that is not installed is an error listing what *is*
installed — never a quiet substitution.

**`tools/platform_demo.py` proves the product claim on real engines.** ULTRON and FieldMind ran through
the same compiled Kokoro + Whisper models inside an offline guard, each with its own answer, sink,
identity and data root. **Verdict PASS**: both spoke (84,644 and 121,744 frames), from different speaker
slots, with **zero network attempts**. Artefacts: `artifacts/platform/platform-run.json` + two reply WAVs.

**Voice status is earned, not assumed.** Slot 7 was synthesised, transcribed back, and recorded `MEASURED`
at WER 0.000 against the 0.15 bar before it could be selected. Both identities remain **unnamed** because
the asset ships no speaker names. The recogniser heard *"what the weth will do"* for *"what the weather
will do"* on our own synthesised input — recorded verbatim rather than tidied.

**Stage 1 of model ownership is built and executed: `tools/corpus_forge.py`.** 16 English sentences ×
slot 0, synthesised, transcribed, and kept only at exact round-trip match: **8 clips kept (24.4 s), 8
dropped and listed with their WER and heard text**, every kept clip SHA-256 pinned, licence basis and
consent basis recorded. The manifest marks `usable_for: [training, adaptation, asr-domain-evaluation]`
and `not_usable_for: [human speech quality claims]` — the same distinction `eval/samples.py` enforces.
English only, because that is all the approved asset can render; the schema accepts hi/bn the day a
front-end exists. Stages 2–4 (fine-tune, distil, train) need your approval for a new checkpoint and, for
training, `torch` — none downloaded, nothing installed.

## Four further defects found and fixed (register addendum A22–A25)

1. **S1** — the Kokoro adapter resolved a speaker by parsing the voice *display name*, so any host with
   its own naming broke synthesis. Fixed by carrying `speaker_slot` on the reference.
2. **S1** — the consent gate fired for every voice reference, so an installed Apache-2.0 identity could
   not speak at all. Consent now governs references carrying **recorded audio**; provenance governs model
   slots. Both directions tested.
3. **S2** — an unknown explicitly-requested voice fell through to the platform default.
4. **S3** — my own hygiene check falsely flagged two distinct local helper classes named `Stt`; it is now
   scope-aware.

The `voices` package was added to the audit's speech-path scope rather than left outside it.

## Suite and verification status after this entry

`python -m pytest` → **667 passed, 7 skipped** (the 7 skip by default and pass under
`--run-approved-assets`). `python tools/audit.py` → **11/11 PASS**. `python tools/offline_proof.py` →
PASS, 0 attempts. `python tools/platform_demo.py` → PASS. `python tools/corpus_forge.py` → 8 clips.

## Standing blockers, unchanged in kind

TTS is slower than real time at every profile; Bengali/Hindi synthesis and acoustic LID remain gaps;
distribution needs the GPL-3.0 espeak-ng pass and cloning stays out of scope; profiles stay provisional
(1 of 5 threshold fields measured); and real human speech at the microphone is still untested.

---

# Progress ledger — entry 5 (2026-10-13, namespace, version control, TTS latency)

Authorisation relied on: approve the neutral namespace rename; **hold** ASR fine-tuning (no torch, no new
checkpoint, no training); investigate TTS performance against the current implementation with no new asset
without approval; establish version control safely.

## Namespace migration: `ultron_voice` → `nativevoice`

Directory moved, 60 text files rewritten (imports, packaging, tests, tools, docs), `ULTRON_VOICE_HOME` →
`NATIVEVOICE_HOME`, distribution name `ultron-native-voice` → `nativevoice`. **No behaviour change and no
unrelated refactoring**: 668 tests pass exactly as before plus one new test, the audit passes 11/11, and
importing `nativevoice` still pulls in zero third-party modules. Historical artefacts under `artifacts/`
were deliberately **not** rewritten — they record what a past run actually did, and editing them would
make the evidence agree with the code instead of with history.

Two things left alone on purpose, both recorded rather than tidied:
- the local data directory is still `%LOCALAPPDATA%\ultron-voice`, because renaming it would strand the
  verified assets, manifests and installed voice identities. Migration is a one-command move plus digest
  re-verification, held as decision **N6**.
- a Windows Python runtime store (`Python/pythoncore-3.15-64`, 45 MB) materialised inside the working
  directory during the session. It is not ours, was not created by any command here, and was **not
  deleted** — it is git-ignored and reported instead.

## Version control baseline

No parent repository exists at any level up to `/c/Users/shres`, so this is a new root, not a nested one.
`.gitignore` excludes build noise, generated audio (the corpus keeps its digests, so the manifest is the
evidence and the WAVs are reproducible), credential-shaped files, and the stray interpreter directory.
`.gitattributes` pins `text=auto eol=lf`, so the CRLF churn git warned about cannot become a diff.

Reviewed before committing: 157 files, 29,867 insertions, **no binaries staged**, secret-pattern scan
(api key / password / private key / AWS / GitHub token forms) clean, exclusions verified by querying the
index rather than trusting the ignore file. Identity was already configured as Shreshtha, so the author is
his, not an invented one. `git init` deliberately ran **without** `-b main` because this git's
`init.defaultBranch` is unset — the flag would have failed the command outright; the branch is `master`.

## TTS latency: measured, not assumed

`tools/tts_latency_lab.py` sweeps the levers that exist **without a new asset**, on a two-sentence reply,
with 3.62 GB free and 272 processes running (recorded beside the numbers).

| Lever | Result | Measured |
|---|---|---|
| Threads 1 → 8 | RTF 2.01 → **1.59**, first-audio 14.1 s → 11.2 s | yes |
| Threads 12 | **worse**: RTF 1.97, first-audio 13.9 s — oversubscription | yes |
| Chunk budget 50 → 24 tokens | first-audio **11,545 ms → 5,061 ms**; RTF essentially unchanged | yes |
| Chunk budget 12 tokens | no further gain (5,270 ms) | yes |
| Language switch en → zh | **+3.32 s**, because the adapter reloads the engine per language | yes |
| Runtime streaming callback | `OfflineTts.generate` **does** accept a per-chunk callback — chunk-level latency is our choice, not a hard limit | inspected |
| Model variants in the asset | only `model.int8.onnx` exists, so quantization comparison is impossible without a new asset | verified |

**Action taken:** the default chunk budget is now 24 tokens, chosen from the measurement, with the reason
and the trade-off (more crossfade seams, not less throughput) in the code, and pinned by a test so it
cannot drift back silently. **Not taken:** changing threads past 8, or pretending the remaining gap to
real time is closed. RTF is still ~1.6 at best and first audio ~5 s, so conversational output is **not**
met on this machine with this asset; press-to-hear stays the working mode, and the replacement case now
rests on measured numbers.

## ASR fine-tuning: held, as directed

Nothing installed, nothing downloaded, no training started. The record states plainly why the corpus is not
evidence of ASR quality: **24.4 seconds of audio, 8 clips kept by exact round-trip match**, which measures
agreement between two models we already have on sentences both could handle. It is training material with
documented provenance; it is not a quality result. A baseline, a held-out evaluation set and a concrete
checkpoint/training proposal come before any training decision.

## Verification at this entry

`python -m pytest` → **668 passed, 7 skipped**. `python tools/audit.py` → **11/11 PASS**.
`python tools/offline_proof.py` → PASS, zero network attempts. `python tools/platform_demo.py` → PASS
(two hosts, two slots, isolated roots). `git log` → one commit, clean working tree.

---

# Progress ledger — entry 6 (2026-10-13, relocation, speech protocol, TTS proposal)

Directives taken: relocate safely (N9); keep the data directory (N6) and the commit history (N7) as they
are; keep the stray interpreter ignored and untouched (N10); prepare a real-speech procedure and **wait**
before recording; produce a TTS comparison as a proposal only; ASR fine-tuning stays on hold.

## Relocation (N9) - verified in the new copy, original untouched

`C:\Users\shres\Documents\Qoder\2026-10-12\nativevoice` now holds a `git clone` of the repository. A
clone rather than a copy, deliberately: it can only contain tracked history, so anything missing is
something that was never versioned, which is the question a relocation should answer.

Verified from the new location: 6 commits, identical `HEAD` and identical tree hash, 157 = 157 tracked
files, ignored material absent (`Python/`, generated WAVs, corpus clips, caches), `.gitignore` and
`.gitattributes` tracked, working tree clean, **669 passed / 7 skipped**, audit **11/11**, assets resolve
to the same absolute data root with all three models `installed/verified/usable`, the installed identity
reads `en measured 0.049383 / zh measured_failed 0.462963 / hi,bn unsupported`, and a real offline
round-trip transcribed correctly with **zero network attempts**. The original workspace was not deleted.

Full record: `docs/relocation.md`, including the remaining steps that are his - choosing the permanent
path, deleting the original, and repointing the clone's `origin` before the original goes away.

**Finding A26** came out of this and is fixed: the offline static scan built reported paths by splitting
on the package name, which only worked while the repository directory was not named `nativevoice`. A green
suite had been green partly because of a directory name.

## Real-speech procedure (not executed)

`docs/real-speech-protocol.md` specifies the seven steps, the acceptance bars agreed **before** recording
(so no bar is chosen after seeing a number), and what synthetic results may never be presented as.
Retention and deletion are implemented rather than promised: default `--retention none` writes **no audio
at all**, `keep` writes detected segments only under the local data root with timestamps in the filenames,
and `--delete-recordings DIR` lists every path it removes and refuses if the directory is absent.
**Awaiting his confirmation before any microphone is opened.**

## TTS comparison (N8) - proposal only, nothing downloaded

`docs/tts-options.md`: measured baseline, the execution headroom that needs no asset (two changes
remaining: per-chunk callback playback, which the runtime does support, and a per-language engine cache to
remove the measured 3.32 s switch cost), and seven candidates with identity, terms, runtime compatibility,
expected effect and quality evidence. KittenTTS nano int8 is the recommended first trial: Apache-2.0
verified, 25.6-61 MB, and already supported by the installed sherpa-onnx, so no package would be
installed. Supertonic is OpenRail (not OSI-approved) and blocked pending his reading of the terms;
matcha-icefall and the piper/Coqui repositories carry **no licence tag at all** and therefore fail our own
provenance gate rather than being excluded by taste.

Stated plainly in the document: **no candidate has a measured number**, and none of the permissively
licensed ones provides Bengali or Hindi synthesis - coverage and latency are separate decisions.

## Standing gaps, unchanged and still marked

Bengali/Hindi synthesis and acoustic language identification; human-speech recognition accuracy;
the GPL-3.0 espeak-ng redistribution pass; voice cloning (no suitable licensed asset); profile
validation (one of five threshold fields measured); VAD and barge-in on real speech. Nothing was
integrated into ULTRON or FieldMind, no asset was downloaded, nothing was trained, and no package was
installed.


---

# Progress ledger — entry 7 (2026-10-13, microphone baseline and KittenTTS trial)

Two approvals taken with their boundaries: step 1 only of the microphone protocol, retention `none`; and a
controlled isolated KittenTTS trial, verified before acquisition, with Kokoro and every default unchanged
and no promotion without a further decision.

## Microphone baseline — quiet room, 8 s, nothing persisted

Device confirmed before capture: default input **index 1, "Microphone Array (Intel Smart Sound Technology
for Digital Microphones)"**, 4 channels at 44.1 kHz native, MME host API, low/high latency 0.09 / 0.18 s.
The capture stream requests 16 kHz mono int16 directly, so PortAudio resamples for live capture - our own
band-limited resampler is used on files and synthesis output, not on this path.

Two identical runs, `--retention none`:

| Measurement | Run 1 | Run 2 |
|---|---|---|
| Audio captured | 7.98 s in 8.101 s wall (RTF 1.0152) | 7.98 s in ~8.08 s (RTF 1.0149) |
| Chunks / dropped frames | 399 / **0** | 399 / **0** |
| Median input level | 0.000412 | 0.000358 |
| Peak input level | 0.0086 | 0.0021 |
| Silence fraction | 0.9967 | 1.0 |
| **Turns opened** | **0** | **0** |
| Recordings written | `[]`, `retained: false` | `[]`, `retained: false` |

Persistence verified independently of the tool's own claim: 381 files under the data root before, **381
after, nothing created and nothing deleted**, and the host recordings directory was never created at all.

One predictive observation rather than a pass/fail: run 1's peak (0.0086) reached ~72% of the rise
threshold (0.012) while the median sat ~29x below it. The gate is comfortable on a quiet room, and a fan,
traffic or a neighbouring conversation is expected to open turns - which is the case step 4 of the
protocol exists to measure, not a defect in the quiet-room result. **No human speech was recorded and
nothing was transcribed from the microphone**, as instructed; results in `artifacts/device/mic-baseline-quiet*.json`.

## KittenTTS nano — trial measured, incumbent still the default

Verified before acquiring (Apache-2.0 on the model card, `kitten-nano-en-v0_8-int8.tar.bz2` 29.8 MB, only
the four needed components, no package installed), then provisioned into `.trials/kitten/` with 360
recorded digests and `redistributable: False` because training data and voice-actor consent are undisclosed.

Two runs, 16 trials each per engine, single-sentence whole-utterance basis on both:

| | Kokoro int8 | Kitten nano int8 |
|---|---|---|
| RTF | 2.3206 / 2.3615 | **0.5701 / 0.6161** |
| First audio (median) | 5.89 s / 5.99 s | **1.99 s / 2.16 s** |
| Engine init | 3.64 s / 4.12 s | **1.49 s / 1.63 s** |
| Peak RSS | 1,044 / 1,049 MiB | **764 / 769 MiB** |
| Round-trip WER | 0.052 / 0.052 | 0.0364 / 0.0239 |
| Languages | en, zh | **en only** |
| Speaker slots | 103 unnamed | **8 named** |
| Offline | 0 attempts | 0 attempts |

Playback startup measured separately as a device cost: stream open median **62.4 ms**, no audio emitted.

**First configuration on this machine measured generating faster than real time.** Not established, and
recorded as such in `docs/tts-kitten-trial.md`: naturalness (nobody has listened), anything about human
speech, Chinese coverage (Kitten loses it), and distribution clarity (it is worse, not better - no
disclosed training or consent provenance). First audio is still ~2 s, so press-to-hear remains the honest
description. Recommendation offered, not executed: add Kitten as a selectable English voice and leave
Kokoro the default until he decides.

## Findings from this entry

A27 (**S1**, my own provisioning tool asserted `redistributable=True` for every asset - the tool now
defaults to False and per-asset, but the incumbent's existing manifest still says `true`, which is his
correction to make, not mine to sneak in); A28 relative `asset_root` making a provisioned asset
unresolvable; A29 a shared-text edit that injected a field into the wrong engine row and crashed the
first benchmark. Method note kept in the register: the first harness draft would have compared the
incumbent's chunked path against the candidate's unchunked one, understating the gap.

## Verification after this entry

Full suite, audit, offline proof and platform demo re-run below. Repository relocation stays reversible:
both workspaces intact, nothing deleted, ASR training and host integration still on hold.
