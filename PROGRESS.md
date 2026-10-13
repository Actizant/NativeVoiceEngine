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


---

# Progress ledger — entry 8 (2026-10-13, step-2 attempt, A/B listening set, rights correction)

Three approvals worked, each with its own boundary: microphone step 2 with retention `none` and a stop
after the step; matched Kokoro/Kitten listening samples with no promotion or default change; and an
individual rights audit of the incumbent manifests that must not keep a claim just because a script once
wrote it.

## Microphone step 2 — attempted, and it failed for a reason worth stating exactly

`python tools/real_speech_check.py --countdown 12` captured 74.98 s (1,199,680 frames) from the confirmed
default input and wrote **nothing**: 381 files in the data root before, 381 after, `created_during_run`
empty, `recordings_written` empty, and zero network attempts under the runtime offline guard. The three
target sentences were printed before capture so the intended reading is verifiable rather than
reconstructed.

VAD found **0 segments**; all three targets are recorded as `MISSING_SEGMENT`. Peak input level over the
whole window was 0.0044 against a median of 0.000349 — the quiet-room baseline from entry 7, i.e. nothing
that could be called speech reached the microphone. The likely cause is the run itself: it was launched
unattended to stay inside the authorised step, so no one was there to read. The capture chain is not
implicated — the same harness produced clean quiet-room baselines twice, and VAD open/close behaviour on
synthetic tones is measured.

Criteria, each on its own line, since two of them cannot be reached by this method at all: **PASS** no
audio persisted; **PASS** offline, zero attempts; **PASS** zero false turn-opens; **FAIL** all sentences
detected; **NOT RUN** read-aloud WER/CER (`sentences_scored: 0`, means `null`); **NOT RUN** language
detection top-1 (`null`); **NOT RUN** VAD onset error — and structurally unreachable in a manual
read-aloud run, which has no ground-truth onset to measure against; it needs a keyed stimulus; **NOT RUN**
endpointing (step 3), noise and barge-in (steps 4 and 6).

So the two claims only real speech can prove — recognition quality on human audio and language detection
on human audio — remain **unproven**, and step 2 stays open rather than being reported as closed. The
harness needs no changes; it needs a voice. These numbers are kept separate from the synthetic set: the
round-trip English WER 0.0494 / CER 0.0242 remains a synthetic-speech measurement and is not evidence
about human speech. Full table in `docs/real-speech-protocol.md`, artefact
`artifacts/device/real-speech-read-aloud.json`.

## A/B listening set — six matched pairs, same text and same conditions

`python tools/ab_samples.py` renders six sentence types (punctuation list, question, numbers and
abbreviation, long utterance, short imperative, names and places) through **both** engines at identical
text, speed 1.0, 24 kHz output, 4 threads, and the same normalisation-and-writer chain.
`python tools/ab_samples.py --play NN` plays A then B for one pair; `--play all` plays every pair. Twelve
WAV files under `artifacts/ab/` (git-ignored as generated audio) plus `manifest.json`, whose status line
reads "trial evidence only; default engine and shared catalogue unchanged".

The measurement side of that manifest is worth reading before listening:

| Pair | Kokoro audio / RTF | Kitten audio / RTF | length ratio |
|---|---|---|---|
| 1 punctuation list | 2.625 s / 2.315 | 2.714 s / 0.569 | 1.034 |
| 2 question | 2.610 s / 2.037 | 3.109 s / 0.553 | 1.191 |
| 3 numbers + abbreviation | 4.838 s / 1.875 | 6.073 s / 0.481 | 1.255 |
| 4 long utterance | 8.655 s / 1.964 | 11.917 s / 0.482 | 1.377 |
| 5 short imperative | 1.068 s / 2.948 | 1.689 s / 0.668 | 1.581 |
| 6 names and places | 3.430 s / 2.083 | 5.267 s / 0.514 | 1.536 |

**Kitten speaks the same text in 1.03–1.58× the duration**, consistently longer and most visibly on short
utterances. That is a prosodic difference, not a truncation: no pair clips (`clipped_fraction` 0.0
everywhere) and Kitten's RMS runs roughly twice the incumbent's (0.096–0.107 vs 0.042–0.046), so the
candidate is louder and slower per utterance while still generating 3–4× faster than real time. Whether
that reads as warmer or as padded is exactly the judgment no metric here can make — which is why the set
exists. Nothing was promoted, no default moved, Kitten stays under `.trials/kitten/`, and its language
claim is unchanged: English only, 8 named voices, versus the incumbent's en/zh anonymous slots.

## Incumbent asset rights — audited individually, three claims withdrawn

`python tools/asset_status_audit.py` re-reads each manifest's own evidence rather than trusting the field
an earlier tool wrote. All three registry assets carried `redistributable: true` from my own provisioning
default — the over-claim logged as A27 — and all three are now `redistributable: false`, with permissions
reading "not established either way; local use approved, redistribution claim withdrawn until reviewed"
and a dated correction line in `notes`. **The manifest schema has no rights-status field**, so the
`unverified` verdict is expressible on an asset only as that conservative boolean plus prose; the word
itself lives in the corrections record (logged as A32). Inspected per asset, with the directory actually
read recorded alongside:

- **faster-whisper-base / small**: the Hugging Face tag says MIT, but **no licence text ships inside the
  installed package** (`licence_text_shipped_in_package: false`); the weights are a third-party conversion
  of OpenAI Whisper and training-data provenance is undisclosed. Digests recorded 4/4 each.
- **kokoro-int8-multi-lang-v1_1**: the Apache-2.0 text genuinely does ship in the package and was read
  (`true`, head "Apache License"), digests 377/377 — but the bundle carries an **espeak-ng data tree under
  GPL-3.0** including hundreds of named files under `espeak-ng-data/voices/`, whose consent terms are
  referenced on an upstream page nobody has read or archived.

**Missing evidence and established prohibition are kept apart, as directed.** `None` is refused by the
loader on purpose — an asset whose rights were never checked must not become loadable by silence — so the
boolean stays conservative at `false` while the distinction lives in `permissions` and in the
machine-readable `artifacts/audit/asset-status-corrections.json`. A real prohibition would be recorded in
that file as `rights_status: prohibited` with the offending term quoted; none was found for these three
and none was assumed. Each manifest was copied to `artifacts/audit/manifest-snapshots/*.pre-audit` **before** rewrite:
they live in the local data root outside version control, so without the snapshot the value the earlier
tool wrote would simply have vanished. All three still load for local use (`usable=True`, approval
blockers empty) and no inference behaviour changed. `src/nativevoice/provenance.py` now records candidate
evidence per asset — upstream, read date, evidence, gap, prohibition — so the next review starts from a
file rather than from memory.

## Findings from this entry

A30 (**S1**, the rights over-claim inherited from A27: three incumbent manifests asserted redistribution
that no review had established — now corrected, snapshotted and recorded). A31: the step-2 harness cannot
measure VAD onset error by construction, because a countdown has no ground-truth onset to compare against;
that criterion needs a keyed-stimulus test, and is reported NOT RUN rather than quietly dropped.

## Verification after this entry

`python -m pytest` → **669 passed, 7 skipped**; `python tools/audit.py` → **11 passed, 0 failed**;
`python tools/offline_proof.py` → PASS with `"attempts": []`; model-backed tier with
`--run-approved-assets` → 33 passed; `python tools/platform_demo.py` → PASS, 84,765 and 121,432 frames
played. All re-run **after** the manifest corrections, so the corrected rights fields are what the suites
actually exercised. ASR training stays on hold, no asset beyond the approved trial was downloaded, nothing
was integrated into ULTRON or FieldMind, and both workspaces remain intact and reversible.


---

# Progress ledger — entry 9 (2026-10-13, first human speech captured, harness corrected)

Two things happened in one sitting: the six A/B pairs were played for his listening, and microphone step 2
was run with a speaker present for the first time. The run produced real speech, and reading its output
against its own source produced four defects in the harness - which is why the run is described here as
evidence about the system *and* about my tooling, not as a clean pass.

## Correction to entry 8, dated 2026-10-13

Entry 8 told him `python tools/ab_samples.py --play all` exists and that `--play NN` plays "A then B".
Neither was true: the switch was never implemented, and the playback order came from sorting filenames,
which put `kitten` before `kokoro` - the candidate first, in a listening test, against a manifest promising
the incumbent first. Entry 8 is left exactly as written; this entry retires those two claims. Findings
A33 and A34.

## The first run with a voice in the room

`python tools/real_speech_check.py --countdown 20`, captured 2026-10-13 14:51:57, preserved byte-for-byte at
`artifacts/device/real-speech-read-aloud.20261013-145157.human-run.original.json`. Peak input level **0.0866**
against the documented quiet-room 0.0044. One turn opened, at audio-time **21.34 s**, holding 3.8 s, and it
was transcribed **word for word** - WER 0.0000 and CER 0.0000 against the sentence actually spoken, language
`en` at confidence 1.000. Nothing persisted (381 files before and after), no network attempt.

So three claims the master prompt said only real speech could prove are now demonstrated on human audio:
VAD opening on a voice, recognition of a spoken sentence, and language identification of it. The fourth -
endpointing on real pauses - is **not** settled by this run, because the harness threw away two of the three
utterances before scoring them.

## Four harness defects, and what replaced each

| # | Defect | Replacement |
|---|---|---|
| A35 | segments paired to sentences by **position** | content alignment: minimum WER, greedy, 0.60 refusal cap, deterministic ties; unmatched targets `MISSING`, unmatched segments `extra`, repeats named as duplicates |
| A36 | the 2 s post-countdown baseline collected audio into a list that was **measured then discarded**, and counted those openings as *false* | one stream: the window samples levels while the segmenter keeps collecting; a false opening is one inside the window that claimed **no** sentence |
| A37 | `quiet_window_seconds` reported a single 20 ms chunk | the measured window in audio seconds, with its own median and peak |
| A38 | `segment_onsets[].seconds` was really the duration | every opening logged with `onset_seconds` **and** `duration_seconds` |

A36 is the one that mattered most for interpretation: a report of "1 of 3 sentences detected" was not a
finding about endpointing, it was my harness eating the first two utterances and then calling them noise.
A39 is logged as pre-existing and untouched: sherpa-onnx writes `Skip unknown phonemes: ə` during
synthesis while still returning PASS, so a vowel our front-end produces is dropped silently by the engine's
phoneme table. That needs its own authorisation, not an opportunistic fix inside a harness pass.

## Verification of this pass

`tests/test_real_speech_harness.py` - **28 passed in 0.31 s**, no microphone, no loudspeaker, no model: the
capture policy takes an iterable of chunks and the alignment and opening log are pure functions, so the
speech-inside-the-window and speech-immediately-after-the-countdown cases are actually executed rather
than described. Full suite **697 passed, 7 skipped**; with `--run-approved-assets` **704 passed in 74.09 s**;
`tools/audit.py` **11 passed, 0 failed**; `tools/offline_proof.py` **PASS**; `tools/platform_demo.py` exit 0,
**PASS**. Playback order verified against the twelve real files (all six pairs resolve Kokoro then Kitten)
and pair 05 played audibly through the corrected code path. Samples were **not** regenerated.

History handling: the human-run artefact is copied under its own timestamp, and the harness now copies any
existing artefact to `<stamp>.previous.json` before overwriting, so a corrected rerun can never erase the
run it corrects. Boundaries kept as instructed - retention `none` is still the default, Kokoro is still the
platform default, Kitten is still isolated under `.trials/kitten/`, no voice identity changed, no ASR
training, no integration into ULTRON or FieldMind.

**Step 2 remains open.** It needs one rerun with the corrected harness while he is present, and until that
happens endpointing on human pauses and the false-opening criterion are `NOT RUN`. Nothing was captured
after the corrections; the next capture is his to trigger.


---

# Progress ledger — entry 10 (2026-10-13, supervised rerun, then A40 and A42 fixed)

The speaker was present for both runs recorded here. The first produced the first honest human-speech
numbers; the second was the run that exposed a timing defect in my own harness before it could be trusted
for anything else. Neither run is interpreted as a verdict on endpointing, and no capture was started
after the fixes without the reader.

## The 15:31 run, in one paragraph

Retention `none`, nothing persisted (381 files before and after), offline clean. Five turn openings; two
sentences matched - the lights sentence at **WER 0.2000** and the coffee sentence at **WER 0.4167**, both
detected as `en` at 0.978 and 0.956; mean WER **0.3084**, CER 0.2408, no exact matches. The pangram was
split into two fragments 0.2 s apart ("The quick brown walks." / "Johns water leasing."), each beyond the
0.60 alignment cap, so it was reported as `MISSING` with its closest distance rather than forced into a
match. One fragment hallucinated Japanese. Full table in `docs/real-speech-protocol.md`.

## A42 — the harness was measuring the wrong minute

`--countdown 60 --limit 100` reported `wall_seconds 100.136`. With an unbounded callback queue that starts
filling at `open()`, that adds up only if the countdown's audio was replayed instantly into the analysis.
It was. So the speaker got roughly **40 s of usable window instead of 100 s**, the quiet baseline measured
the room *during* the countdown rather than after the cue, "0 false openings" was judged against the wrong
interval, and onset times carried a 60 s offset. Fixed by moving `source.open()` behind the start cue in
`open_and_capture()`; `wall_seconds` now covers the capture alone and the artefact records
`countdown_seconds` and `stream_opened_at`.

## A40 — the console lost a report twice, and now cannot

Both runs printed a Japanese hallucination into a cp1252 console and raised `UnicodeEncodeError` on the
final report, after the artefact was written. `configure_console()` now reconfigures stdout and stderr to
UTF-8 with `backslashreplace`, `emit()` reprints anything the stream still cannot encode as ASCII escapes
and swallows a dead stream, `emit_block()` prints line by line so one bad glyph cannot swallow the rest,
and `write_artefact()` is called before any printing. The exit code of a run no longer depends on the
terminal.

## Verification of this pass

`python -m pytest tests/test_real_speech_harness.py` → **40 passed in ~1 s**, no device and no model: a
fake console that refuses non-ASCII proves A40, and a device double that refuses to yield frames while
closed proves the GO → open → frames → close order for A42. Full suite → **709 passed, 7 skipped in
15.02 s**; with `--run-approved-assets` → **716 passed in 79.18 s**; `tools/audit.py` → **11 passed, 0
failed, 0 not run**; `tools/offline_proof.py` → **PASS**, `attempts: []`; `tools/platform_demo.py` → exit
0, **PASS**, 84,706 and 121,458 frames played.

## Findings and boundaries

A40 and A42 fixed and tested; the register stands at **41 findings, 34 fixed, 7 open**. A39 (sherpa-onnx
skipping `ə`) and A41 (vendor auto-detection unstable on human speech) remain **open and untouched** by
instruction - A41 is now characterised as instability rather than consistent error, which is harder to
design around. No capture has been run since the fixes. Retention stays `none`, the offline guard stays
enabled, all three artefact generations are on disk, Kokoro is still the platform default, Kitten is still
isolated under `.trials/kitten/`, no identity or engine changed, nothing was downloaded, nothing was
trained, nothing was integrated into ULTRON or FieldMind, and no production behaviour was modified: every
line of code changed in this entry lives under `tools/` or `tests/`.


---

# Progress ledger — entry 11 (2026-10-13, the clean supervised run at 17:19)

One capture, exactly as authorised, and nothing after it:
`python tools/real_speech_check.py --countdown 15 --limit 90`, retention `none`, offline guard on.

## The run proved both fixes on human audio, not only in tests

A40: a Japanese-to-Indonesian set of hallucinations printed without raising, and the process exited **0**.

A42: `countdown_seconds: 15`, `stream_opened_at: GO`, `wall_seconds 90.541` against `audio_seconds 90.02` -
ninety seconds of audio took ninety seconds of wall time, so no backlog is being replayed, and the first
turn opened at **0.84 s after GO** rather than carrying a countdown offset.

And the protocol's hardest case arrived uninvited: the reader **started speaking at 0.84 s, inside the
two-second quiet baseline**. Under A36 that audio would have been measured, filed as noise, discarded, and
then counted against the run as a false opening. Here it was captured, transcribed **word for word**,
matched to the pangram at WER 0.0000 / CER 0.0000 with `en` at 0.9202, and reported as
`speech_inside_quiet_window: 1` with `false_turn_openings: 0`.

## What the audio said

Four openings: 0.84 s (4.00 s, matched the pangram exactly), 5.34 s (4.02 s, returned
"Setelah menonton, sebelum anda mempunyai terlihat."), 9.90 s (4.54 s, matched the coffee sentence at WER
**0.2500** / CER 0.1842, `en` 0.9781), and 32.36 s (0.78 s, empty transcript).

Mean over matched sentences **WER 0.125 / CER 0.0921**, one exact match, language top-1 **1.000**. The
lights sentence is the failure: four seconds of his voice came back as an Indonesian sentence. Acceptance:
nothing persisted **PASS** (381 files before and after, `created_during_run` and `recordings_written` both
empty), offline **PASS** with `network_attempts: []`, false openings **PASS** by the predeclared definition,
all-sentences-matched **FAIL** at 2 of 3, WER and language **MEASURED**, onset error / long-pause
endpointing / noise / barge-in **NOT RUN**.

Three consecutive sentences at roughly half-second gaps produced exactly three segments - no merge, no
split - which is the first clean segmentation evidence we have, and the opposite of the 15:31 split
pangram. It is still **not** the predeclared endpointing test, which asks for deliberate 1.5 s pauses, so
that criterion stays NOT RUN rather than being upgraded because the news was good.

## Two observations recorded and deliberately not fixed

A43 - unmatched segments carry their text but not the language the engine asserted, so this run cannot say
what language was claimed for the Indonesian substitution. A44 - the false-opening bar is scoped to the
quiet window, so the 32.36 s opening on non-speech that transcribed to nothing is not counted; that is
correct against the criterion agreed before any audio existed, and misleading as a description of the run.
Both left OPEN: naming them is allowed by this pass, editing them is not.

Register stands at **43 findings, 34 fixed, 9 open**. A39 and A41 untouched by instruction; A41 now has a
fourth data point (2 of 2 correct this run, 0 of 2 in the 15:18 run, and a whole-sentence language
substitution in between), which keeps it firmly "unstable" rather than "wrong".

## Verification and boundaries

No code changed in this entry - it is one artefact and three documents - so the suites above stand as the
record for the current tree: **709 passed, 7 skipped**, **716 passed** with `--run-approved-assets`, audit
**11 passed / 0 failed**, offline proof **PASS** with zero attempts, platform demo **PASS**. Nothing was
downloaded, promoted, integrated or retrained; Kokoro remains the default, Kitten remains under
`.trials/kitten/`, `src/` is untouched, and all five artefact generations are on disk. **No further capture
has been started.**


---

# Progress ledger — entry 12 (2026-10-13, ruling on the 17:19 evidence; documentation only)

Seven statuses were ruled by the approver on the evidence from the 17:19 run and are recorded verbatim in
`docs/audit-register.md` Addendum 8. No code, artefact or configuration was changed by this entry.

What the ruling settles: **A40 and A42 are verified on live audio**, not merely tested - a report that
prints non-Latin text and exits 0, and 90.02 s of audio that took 90.541 s of wall time with the first
opening at 0.84 s after GO.

What it deliberately does not settle, in the approver's words: basic English live recognition is
**demonstrated, not universally validated**; endpointing on deliberate 1.5-second pauses is **NOT RUN**; a
**clean room-noise baseline was not obtained**, because the reading began inside the measurement window;
and **A41, A43 and A44 stay open**.

That last distinction matters for how the system is described anywhere else. One word-exact sentence in the
same run as one whole-sentence Indonesian substitution is not a capability claim - it is two data points
about unstable behaviour. Any status field, capability row or summary that reads "English ASR: verified"
from this evidence would be over-claiming, and the capability registry is built so that a claim like that
needs a measured status per language and task, not one good run.

Register: **43 findings, 34 fixed, 9 open**, plus the seven ruled statuses above. Workspace stays clean at
the committed tree, all artefact generations intact, no capture started since 17:19, `src/` untouched,
Kitten still confined to `.trials/kitten/`, Kokoro still the default, A39 untouched, nothing downloaded,
trained or integrated.

The closure plan for each open validation criterion is delivered as a report-only summary alongside this
entry; nothing in it has been started.


---

# Progress ledger — entry 13 (2026-10-13, A43 and A44 closed; first A41 measurement)

A preparation pass before the next microphone session, with no microphone used. Three things were asked
for and all three were done, plus one bookkeeping correction found on the way.

## A43 closed

Every segment the run mentions now carries the language the engine asserted and its confidence: extra
segments get `detected_language` and `confidence`, and each missing target gets its closest candidate's
text, language and confidence. Values are copied from what came back - an empty code stays empty and a
missing confidence stays `None`, never filled in. Without this, the 17:19 run's Indonesian sentence was
unattributable after the fact; with it, the next run can say what the engine believed.

## A44 closed the way it was ruled

`false_turn_openings` is the same calculation as before, unchanged, still scoped to the quiet window
because that bar was agreed before any audio existed. A second, separately named measure now reports
`empty_transcript_openings_outside_quiet_window` with its onsets, informational and with no pass bar set;
when no transcripts exist it is `null`, not zero. A test asserts the quiet-window numbers are identical
with and without the new data, so the old criterion provably did not move.

## A41 phase 1, and the answer is not the one expected

The 8 kept synthetic clips were transcribed twice: auto-detect and pinned English. **8 of 8 returned `en`**
(confidence 0.9756-0.9919), mean WER **0.0000 both ways**, identical transcripts on all 8. So the
instability seen on live audio does **not** exist in this call path on synthetic audio - it is specific to
human voice through the microphone chain. The corollary matters more: the synthetic corpus is structurally
blind to this whole class of defect, because its audio was made by our TTS engine and verified by the same
ASR engine, so agreement is built in. The manifest already disclaimed human-speech claims; this is the
first measurement showing why that disclaimer was necessary. A41 stays open, and advancing it now needs a
`--language` pin on the harness plus one supervised run, or a retention decision that is the approver's.

The probe was read-only and provable: it ran inside the offline guard with zero connection attempts, and
afterwards all 8 kept clips were re-hashed and **match the digests recorded in their own manifest**.
Nothing new was persisted.

## Register correction

**A41 had no register row.** It was raised in prose during Addendum 6 and referenced ever since, which is
exactly how a defect becomes invisible to the one list that is supposed to hold them all. It now has its
own row, marked as added late and honestly, with a severity caveat: it is rated S2 until someone measures
whether routing refuses an unmeasured language code - if it does not, it is S1, because the objective says
a language error must never produce unrelated speech.

Register: **44 findings, 36 fixed, 8 open** (A7, A8, A19, A20, A21, A31, A39, A41). The original dated rows
for A43 and A44 stay as ruled; their closure is recorded as dated closure rows beside them.

## Verification of this pass

`python -m pytest tests/test_real_speech_harness.py` → **49 passed in 0.72 s** (9 new, all harness-only, no
device and no model); full suite → **718 passed, 7 skipped in 15.18 s**; `--run-approved-assets` → **725
passed in 56.76 s**; `tools/audit.py` → **11 passed, 0 failed, 0 not run**; `tools/offline_proof.py` →
**PASS** with `"attempts": []`; `tools/platform_demo.py` → exit **0**, verdict **PASS**, 84,768 and 121,438
frames played.

## History note, and boundaries

Two commit subjects were duplicated by my own copy-and-paste while committing, and the second happened
*while correcting the first*. Shared history is not rewritten for this, so the map is recorded here for
anyone reading `git log`:

| Commit | Subject it shares | What it actually contains |
|---|---|---|
| `bc21b23` | shared with `feecf75` | the 17:19 run artefact, ledger entry 11, register A43/A44 rows |
| `feecf75` | shared with `bc21b23` | ledger entry 12 and register Addendum 8 - the approver's seven ruled statuses |
| `150b20a` | shared with `9e9ef3d` | the A35-A38 harness corrections and the first 28 regression tests |
| `9e9ef3d` | shared with `150b20a` | ledger entry 13, register Addendum 9, and the late A41 register row - documentation only |

Recording the collision is the fix that was authorised. Amending would rewrite commits the second workspace
has already fetched, which was ruled out, so the subjects stay and the disambiguation lives here.

No microphone capture was started. No model was downloaded, trained, promoted or integrated; no default
changed; no voice retention; `src/` untouched; A39 and every unrelated finding left exactly as found.


---

# Progress ledger — entry 14 (2026-10-13, Test A: a room-noise baseline with nobody speaking)

One supervised capture, approved as Test A on its own and stopped there. Test B was not launched.

The result is the quietest kind of evidence this project has produced: **zero turn openings from 12.02
seconds of a silent room**, with the quiet window measured at 2.0 s, median 0.00148 and peak 0.0086, and
`false_turn_openings: 0`. Against the three read-aloud runs - which peaked at 0.0788 to 0.0866 - the
detector showed roughly nine times of headroom between this room and a voice. That converts a criterion
which had only ever passed by accident into one that passes with a measured margin.

The baseline also said two less flattering things. The room is noisier now than during the earlier quiet
captures - about four times the median - so a baseline has to be taken near the run it qualifies, not cited
from history. And the window median (0.00148) sits well above the whole-capture median (0.000484), which
looks like a device ramp-in transient at stream start rather than flat room tone; named, not investigated,
not claimed.

A44's new informational measure appeared in the report as designed and read `0` with an empty onset list.
A43's language attribution had nothing to attribute in a silent run, so its console rendering is still
unit-tested only. Full numbers, both readings and the caution are in `docs/audit-register.md` Addendum 10.

Nothing else moved: the read-aloud artefact still hashes to `add7a3969bc545edf09d4e789cb821668dfaae8ac096b8318a27bda7dc4c8170`
unchanged, the synthetic corpus manifest is untouched, retention was `none` with no recordings written and
381 files before and after, and the offline guard recorded zero attempts. A separate `--out` path was used
precisely so that no existing evidence would be overwritten.


---

# Progress ledger — entry 15 (2026-10-13, Test B: two captures, criterion still unmet)

Two artefacts carry Test B's parameters: one generated **18:09:08** that this agent did not launch, and one
generated **18:13:12** that it did. The likeliest explanation is that the reader ran it himself first - stated
as a hypothesis, and both runs are reported separately rather than merged. Nothing was overwritten: each was
preserved automatically, and the 17:19 evidence still hashes to
`add7a3969bc545edf09d4e789cb821668dfaae8ac096b8318a27bda7dc4c8170`.

**The predeclared condition was not met by either run.** Run X's two matched sentences are separated by about
0.28 s of silence - derived, not read off the artefact, because the segmenter's 0.15 s padding at each
boundary turns a reported gap of -0.02 s into roughly 0.28 s of real silence - and Run Y never got that far.
So endpointing on deliberate 1.5-second pauses stays **NOT RUN** after four supervised attempts, and a run
that did not contain the pauses is not evidence about them.

Run X is the better measurement on its own terms: lights **WER 0.0000 exact**, pangram 0.1111, mean 0.0556,
a clean quiet window (median 0.000554, peak 0.004, zero openings), no audio persisted, zero network
attempts. Its cost was a Malay hallucination asserted at 0.9871 confidence.

Run Y was truncated by the harness's own stop rule at six openings with 65.9 of 90 seconds used, three of
them producing no transcript and two producing Bengali hallucinations. It did, however, close the loop the
preparation pass was for: **A43 verified end-to-end** (unmatched segments carried `bn` 0.7196, `bn` 0.8210,
`ms` 0.9871 and empties at 0.4892, 0.5200, 0.2760 into both artefact and console) and **A44 measured on live
audio** (3 empty openings outside the quiet window, onsets `[15.28, 17.28, 65.14]`) while the window-scoped
false-opening bar stayed at 0 - the two measures visibly independent, as ruled.

Two new observations are logged and deliberately unfixed: **A45**, `language_detection_top1` counts only
matched sentences, so Run X reported perfect detection while asserting Malay for a failed one; **A46**, the
stop rule counts raw openings, so noise and breaths can end a supervised run early. Both would change what
a criterion means, which makes them rulings for the approver, not edits for me.

A41 gets harder rather than clearer: five captures, and the vendor has now asserted `nn`, `ja`, `id`, `ms`
and `bn` for English speech. Open and untouched, as instructed.

No capture was launched after Test B. Retention `none` throughout, no recordings written, 381 files before
and after in both runs, zero network attempts, and `src/`, the synthetic corpus, the models, the defaults and
A39 untouched.
