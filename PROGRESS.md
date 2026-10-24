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

Runtime for both was **already installed as recorded on 2026-10-12**: sherpa-onnx 1.13.8 (Apache-2.0) and
onnxruntime 1.30.0 (MIT) for TTS; faster-whisper 1.2.1 + ctranslate2 4.8.2 (MIT) for STT.
**No package was installed and no other model was downloaded.** Those identifiers are the software packages' own licences; they do not establish clearance for the model weights, the data bundled with them, or the speech the packages generate, and no asset-level clearance is recorded here.

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
| TTS real-time factor | **1.79–2.53** — slower than real time at every thread count tried | 14-utterance evaluation: `tts_real_time_factor` 2.0585–2.5271; profile benchmark (median of 3 sentences, 1 repeat): `tts_rtf_median` 1.7913–2.3766 |
| TTS first-audio | **4.03–6.47 s** (14-utterance evaluation) and **6.28–8.33 s** (profile harness, 1 repeat per profile) | both retained, neither discarded |
| STT real-time factor | **0.85** at 2 threads, **0.50** at 4, **0.44** at 8 | profile benchmark |
| Peak RSS during evaluation | **708.8 MiB** | evaluation artefact |
| Peak RSS per profile | 1,001 / 1,015 / 1,046 MiB at 2 / 4 / 8 threads | profile benchmark |
| Model load | TTS 2.144 s, STT 0.89 s | `artifacts/audit/offline-proof.json` |
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

No parent repository exists at any level up to `[USER_HOME]`, so this is a new root, not a nested one.
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

`[USER_HOME]\Documents\Qoder\2026-10-12\nativevoice` now holds a `git clone` of the repository. A
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


---

# Progress ledger — entry 16 (2026-10-13, the second Test B capture was mine; proposals only)

## Retraction first

I reported that the 18:09:08 capture was not launched by me and that the likeliest explanation was the
reader running Test B himself. **That was wrong.** The session transcript log settles it: at **18:09:07**
this agent issued `python tools/real_speech_check.py --countdown 15 --limit 90`, and again at **18:13:11**
after the same approval text arrived a third time. Both Test B artefacts are mine. The approver's
instruction not to assume he had launched it was the right instruction; I assumed anyway, and the
assumption cost the report its accuracy on the one thing that mattered - provenance.

What the log does not say is *why* the approval arrived three times (18:07:05 Test A text, 18:08:55,
18:11:31, 18:13:02 Test B text) alongside three interruptions at 18:07:24, 18:11:27 and 18:12:39. Sender
resent or client redelivered cannot be distinguished from the available metadata, and I am recording that
as unknown rather than guessing at it.

Both captures remain valid measurements - peaks of 0.1197 and 0.0312 show a voice present in each - but
Test B is now correctly described as **one approved test executed twice by the same agent**. Nothing about
the endpointing conclusion changes: neither run contained deliberate 1.5-second pauses between matched
sentences, so that criterion stays NOT RUN.

## A47: what the incident actually exposed, and the proposal

The harness records a timestamp, a countdown and a level, but **no run identifier, no process id, and no
reference to which approval a capture answers**. Nothing in the artefact could have told me which launch
produced it, and nothing stopped the second capture from starting while I had lost sight of the first.

Proposed, not implemented:

1. `run_id` (uuid4), `pid`, `started_at` and `argv` written into every artefact, so a capture can always be
   traced to the invocation that made it.
2. A single-flight guard: an active-run lock under the data root holding `run_id` + `pid` + start time; a
   second `real_speech_check` refuses to open the microphone while a live lock exists, and reports the
   running `run_id` instead. It must clear the lock on exit including on crash-by-exception, so it cannot
   wedge the tool.
3. A process control that needs no code and that I adopt immediately: **one capture per fresh confirmation.**
   If an approval for a capture arrives more than once, I stop and ask rather than treating the repeat as
   authorisation to run again.

None of these touch `src/`, any engine, or the models.

## A45: a whole-run language-attribution summary, alongside the existing metric

The predeclared `language_detection_top1` stays exactly as written - matched sentences only - because it is
what was agreed before any audio existed. Proposed **in addition**, as its own block:

- `language_attribution.per_opening`: for every opening, asserted code, confidence, whether a transcript
  exists, whether it matched a sentence, and whether it fell in the quiet window.
- `language_attribution.run_summary`: a histogram of asserted codes with min/max confidence,
  `non_expected_language_openings`, `no_language_asserted_openings`, and a clearly named
  `whole_run_language_top1` computed over **speech openings** rather than matched sentences only.

Why it is needed, from the data: Run X reported language detection **1.000** while an opening asserted `ms`
at 0.9871 for your voice; Run Y reported **1.000** on its single matched sentence while two openings asserted
`bn` at 0.7196 and 0.8210 and three asserted `en` on nothing. Under the current metric, a run can score
perfect language detection while the engine names four different languages for one English session. A41
stays open regardless; this only makes its size visible.

## A46: finishing the three-sentence test without hiding any opening

The stop rule is currently `2 x sentence count` = six **raw** openings. Run Y hit it at 65.9 s of a 90 s
window, and three of the six openings had produced no transcript - so breaths consumed the budget and the
harness's own arithmetic ended the test before the reading did.

Proposed, harness-only, in preference order:

1. **Bound by time, not by opening count.** Capture ends at `--limit` seconds of audio, with a generous
   runaway ceiling (e.g. 12 openings) purely as a safety valve and reported in `stopped_because`. Every
   opening is kept, transcribed and listed, empties included - and A44's measure keeps counting them, so
   nothing is hidden: the quiet-window false-opening bar is untouched and the empty-transcript count
   still covers the whole window.
2. **Keep the budget but report it**: add `planned_openings_budget`, `openings_observed` and
   `budget_exhausted`, so a truncated run is visibly truncated instead of looking like a short session.

Rejected on purpose: "stop as soon as all three sentences match." It would end the capture on the first
clean pass and **hide** exactly the duplicates, late openings and false openings the protocol exists to
find. A stop rule should bound cost, not curate evidence.

One alternative needs no change at all: run the test as it stands and treat the six-opening ceiling as part
of the difficulty - stay very still between sentences so the budget survives. It makes the measurement
depend on the reader's discipline rather than the harness's, which is a worse instrument, but it is honest
and zero-risk.


---

# Progress ledger — entry 17 (2026-10-13, multilingual sample evaluation: 13 clips, three ASR passes)

Priority moved from the microphone protocol to the uploaded samples, and A45/A46/A47 were left exactly as
found - unimplemented. Nothing was captured, downloaded, registered, promoted or integrated.

## Accessibility came first, and it passed

All 13 attachments were found in `[USER_HOME]\Downloads`, opened read-only, hashed before and after, and
**verified unchanged**. Isolated 16 kHz mono copies went to `../_multilingual_eval/`, outside the repository -
the same rule already applied to the reference archives. No audio, and no third-party bytes of any kind,
entered the repo. Total audio 209.7 s; every file decoded cleanly, so nothing below is a decoding artefact.

`large-v3` turned out to be **already cached locally** (3.09 GB, complete, zero `.incomplete` blobs, dated
2026-10-04 - before this project existed), so the alternative-model leg ran without a download. My stored
notes said only base and small were resident; that was wrong and has been corrected. `medium` is cached too.
None of the three unregistered models has a manifest, a provenance review or an approval.

## The result that matters is about my instrument, not the models

Three defects in the tooling I wrote for this evaluation, all found before any number was reported:

1. the cue extractor read the **speaker prefix** instead of the quoted sentence, so **every** cue score was 0.0;
2. the function named "longest common subsequence" was a greedy first-match scan - it scored a **verbatim**
   Chinese transcript at **0.04** where a real LCS gives **0.90**;
3. neither tool guarded the console, so CJK and Bengali output crashed both printers on a cp1252 terminal (the
   same defect class as A40, reproduced twice by me within one session), in both cases *after* the JSON had
   been written, so no evidence was lost.

All three are fixed and pinned by `tests/test_multilingual_eval.py` - 18 tests, no device, no model, no
network. Had I not checked my own numbers, the evaluation would have concluded that both models fail on every
unspaced script, which is an artefact of my code and not a property of anyone's audio. That is A48. A fourth
gap (A49: intra-clause repetition is invisible to the current detector) is recorded rather than quietly
patched.

## The finding that outranks the model comparison

**The baseline asserts the right language and writes the wrong script, and every existing gate passes it.** On
the Hindi clip it produced Perso-Arabic text while reporting `hi` at 0.8164 (cue recall **0.00** against the
Devanagari quote in the same filename); `large-v3` produced Devanagari at 0.90. The Bengali clip behaved the
same way - `bn` asserted, Arabic script written, 13 segments; `large-v3` returned coherent Bengali at 0.405.
A capability model keyed on **language alone** cannot express the difference between those two outputs, so
"Hindi supported" can be satisfied by text a Hindi reader cannot read. That is **A51**, and it is a question
about what a capability claim contains, not a bug to patch mid-evaluation.

## Side-by-side, in one line each

| Pass | Mean cue recall (12 cued) | Mean RTF | Load | Peak RSS | Script mismatches |
|---|---|---|---|---|---|
| `base-default` (as measured) | 0.8480 | 0.231 | 0.61 s | 607.9 MiB | 2 |
| `base-int8` | 0.8688 | **0.151** | 0.50 s | 306.0 MiB | 2 |
| `large-v3-int8` | **0.9562** | 2.363 | 10.16 s | 3246.4 MiB | **0** |

`large-v3` is better on every cued sample it beats (zh, ko, ar, ru, hi) and correct in both Indic scripts, at
roughly **15x latency** and **5x memory**, loaded while the machine had 1.9-2.4 GiB free - so its numbers are
*this machine under memory pressure*, and it ran partly from the pagefile. Quantisation alone changed 8 of 13
transcripts and moved the Chinese segment count from 4 to 10 on the **same weights**, which is why segment count
is not a stable observable and why the passes were run one at a time, never overlapping.

## What the samples turned out to be

The clip names carry only speaker labels; it is the **transcripts** that give the game away - the English one
reads "After months of intensive development, **Fish Audio 1.5** has just launched with major improvements to
expressiveness, stability, and cross-language capabilities". So this is **machine speech from a synthesis
vendor's preview set**, not a human corpus and not a licensed one: no licence terms, no consent basis, no
provenance (A50, pre-distribution). Two consequences:

- nothing here can be read as evidence about human speech, and this project's own microphone runs already score
  far worse than these clips;
- the unidentified extra file is **Bengali**, and it is the same ULTRON sentence as the Hindi clip. It matches
  `voice_preview_ultron (2).mp3` exactly in duration and frame count, with a 0.1 LSB mean delta - a re-export,
  not a distinct recording - and `large-v3` rendered it as "আমি প্রায় আকাশ থেকে উলকাপতের কথা ভাবি…",
  which is the Devanagari filename sentence in Bengali. Labelled as an inference: script, asserted code,
  matching content, same-stem sibling. Bengali is therefore present as **one** clip, not absent.

## Claims withheld deliberately

No WER and no CER anywhere, because no sample has a verified reference transcript - a filename quote is
marketing text cut mid-sentence. Hindi and Bengali are **not** declared supported. The **capability registry is
unchanged**, so routing still refuses to act on them; only `docs/capability-matrix.md` gained a dated pointer
saying exactly that. The remaining ~48 target languages have no sample at all here, and the evaluation says so.

## Verification

`python -m pytest tests/test_multilingual_eval.py` → **18 passed in 0.08 s**; `python -m pytest` → **736
passed, 7 skipped in 14.16 s**; `--run-approved-assets` → **743 passed in 50.94 s**; `tools/audit.py` → **11
passed, 0 failed, 0 not run**; `tools/offline_proof.py` → **PASS**, `"attempts": []`, `"static_findings": []`
(the new tools introduce no network path); `tools/platform_demo.py` → exit **0**, verdict **PASS**, zero
network attempts. `src/` is untouched: everything added is a new evaluation tool, a new test file, and
documentation.

## The repository was made public carrying the operator's account name

`origin` was added and `master` pushed at `511a660`, and the repository is **public**. The published evidence
files turned out to contain absolute home-directory paths naming the operator's Windows account: **87**
occurrences across **25** tracked files. Measured rather than inferred - an unauthenticated fetch of
`raw.githubusercontent.com` returned the evaluation inventory with HTTP 200 and its JSON intact.

The account name and the directory layout are the whole of the personal datum. A scan of all **270** historical
blobs against ten credential shapes returned **0** hits, and the operator's email address appears in no
committed file. What does remain in public history, and cannot be removed without a rewrite that has been
declined twice: the pre-scrub blobs of these same files, the author name and personal email on all 24 commits,
a `+05:30` offset on every timestamp, the AI co-author trailer on the initial commit, the A50 third-party
transcripts and speaker-labelled filenames, and the SHA-256 digests of the 13 source audio files. **This is not
a closed finding - it is a tip-only remediation of an exposure that is still open.**

Every occurrence now reads `[USER_HOME]`, a repository-relative path, or the neutral literal `ACCOUNT` for the
one case where the path is embedded in a folder name. Structure was dropped where it carried no evidential
meaning and kept where the prose was *about* the location, so `docs/relocation.md` and `docs/baseline.md` still
argue what they argued about the hashed session directory. Measurements, verdicts, timestamps, filenames,
digests and provenance statements are byte-identical: the path replacements are **87 deletions answered
line-for-line by 87 rewrites**, every altered pre-existing line is a path line, and `src/` and `tools/` are
untouched. Everything else in the diff is new ledger prose in this file and `docs/audit-register.md`.

`tests/test_no_personal_paths.py` is the part that keeps working: 17 tests over five path formats, with negative
controls assembled from synthetic account names, positive controls for every approved placeholder, `.pre-audit`
manifest copies included in the parse check, a coverage floor derived from `git ls-files` rather than a pinned
count, and the one local-only exception declared explicitly instead of silently excluded. It was checked against
the real pre-cleanup blobs at `511a660` - **57** findings across seven files, **0** in those same files at the
current tip. That is a measurement of `511a660`, not a live result: with the tip now clean, the procedure pins
`511a660` explicitly, and the check is recorded as a reproducible audit step in `docs/audit-register.md` rather
than committed as a test, so the ordinary suite never depends on Git objects. A guard that only passes on its own
fixtures proves nothing.

Two findings are recorded in `docs/audit-register.md`: **A52** (historical exposure, S3 proposed) and **A53**
(the nine writer call sites that will republish it on the next run, S2 proposed). The severity call is his, not
mine, and neither row is closed: A52 sanitises the working-tree version of those files, not the published
history. Corrected on 2026-10-14: this pass was **committed as `c93a75f`**, and nothing has been pushed. A52
stays open regardless of the commit, because the 24 earlier commits still carry the paths, the author identity
and the A50 transcripts.

## Verification of this pass

`python -m pytest` → **753 passed, 7 skipped** (736 pre-existing plus the 17 guard tests; the 7 skips are the
standing `NOT RUN` asset gates, unchanged). `python -m pytest tests/test_no_personal_paths.py` → **17 passed in
2.03 s**. The privacy cleanup is **`c93a75f`** - 26 paths, being **87 path replacements** with each deletion
answered line-for-line by its rewrite, plus new ledger prose - and the two follow-up corrections are
**`8745bf3`**. Insertion and deletion totals are not quoted here: every later documentation edit changes them,
while the commits and the 87 replacements do not. Parity against
`HEAD`: for each of the 23 path-only files, applying the placeholder substitution to the committed blob
reproduces the working tree byte-for-byte; the two prose files differ only by additions. Zero files under `src/`
or `tools/` were modified, and `git grep` for the account name over tracked files returns **0 files**. Lint and
typecheck: **NOT RUN** - `ruff` and `mypy` are not installed and neither is configured in `pyproject.toml`.

# Progress ledger — entry 18 (2026-10-15, the synthetic corpus was quarantined from training)

## Correction to entry 4, dated 2026-10-15

Entry 4 recorded that `artifacts/corpus/en-synthetic-01/manifest.json` "marks `usable_for: [training,
adaptation, asr-domain-evaluation]`", and at the time it was written that was true of the file. It is no longer
true, and the claim is retired here rather than edited in place: **4 of the batch's 8 kept clips repeat a
sentence the English round-trip evaluation scores** — `en_slot0_001/002/003/006.wav`, 11.2355 s of its 24.4271 s,
three collided by carrying an extra clause and `en_slot0_006.wav` character-for-character. Train on that and
then score it on `tools/evaluate.py` and the number measures memorisation. Register row **A54** (Addendum 16)
carries the measurement; the two hand-written sentence lists lived in different files and nothing compared them.

## Quarantine as applied

Quarantine of the whole batch from training and adaptation, with its history intact. Its `usable_for` is now
`["asr-domain-evaluation"]` — that use is **retained**, including the read-only probe A41 phase 1 ran over these
same 8 clips — and it carries a `training_quarantine` record: the four colliding clip names, the 11.2355 s
measured from their recorded durations, the dropped records that collided too, the written rule, the ruling, and
`digests_verified: false`. Every sample, digest, count, dropped record, generator and verifier field and
timestamp is byte-preserved, and **no `training_boundary` block was added to it**: the batch predates the
disjointness checks, so back-filling one would assert evidence that was never produced. No audio was deleted or
regenerated — none exists to delete, which is the next paragraph.

**The audio is absent and the digests cannot be checked.** `.gitignore` excludes `artifacts/**/*.wav`, so the
clips were never tracked, and no WAV of this batch is on this machine: the manifest's 8 SHA-256 values are the
only surviving record of them and are unverifiable here. Training therefore stays blocked on a condition this
batch can never meet — a caller must hand the gate digests recomputed from bytes it actually read — and the
quarantine is a text-boundary decision evidenced by the descriptor, not a re-verified audio claim.

**Enforcement, because a JSON field that nothing reads is not a decision.** `usable_for` had no consumer before
this pass. `tools/corpus_forge.py` now has `corpus_training_blockers(manifest, clip_digests=…)`, which refuses a
batch for reasons it re-derives rather than reads: `usable_for` must claim both training and adaptation, a
recorded quarantine must read exactly `CLEARED` with digests verified, `split` must say training, a
`training_boundary` must be well-formed with `collisions_found == 0` and the excluded inputs named, the counts
must add up against the rows they describe, provenance fields must be non-empty, and each sample sentence is
re-checked against the live evaluation pool **and** against every sentence in any batch not cleared for
training. Writing `"status": "CLEARED"` into `en-synthetic-01` still returns blockers — no split, no boundary,
four live collisions, no caller digests. Evaluation eligibility is answered by a separate function so the
training decision cannot retire the measurement. `main()` refuses before a registry is opened, and refuses when
the quarantine records are unreadable rather than assuming no leak.

**The training pool was rewritten twice, and the second rewrite was the finding.** Replacing the eight
evaluation colliders produced eight sentences that turned out to recycle text the quarantined batch had already
used (its dropped records named them). The third version — eight sentences now in `TRAIN_SENTENCES` — shares no
sentence with the evaluation pool or with any of the 16 recorded by the quarantine, measured at worst Jaccard
0.176 against both. No audio was generated and no new corpus was created in this pass.

## Two gate defects found by the tests written for the gate

`build_corpus_manifest` emitted the module-level `TEACHER_TARGET_RECORD` by reference, so flipping one
manifest's `enabled` flag flipped it for every manifest built afterwards — including the refusal a trainer reads;
the builder copies it now and a test asserts two builds are distinct objects. `_strings()` accepted an empty
list, so `"usable_for": []` would have been reported as a missing *claim* instead of an undeclared scope. Both
were caught because the focused tests pin an accepted case alongside each refusal; a suite of refusals alone
would have gone green over both.

## Verification of this pass

`tests/test_corpus_forge_boundary.py` → **106 passed** (34 before this entry; 72 added, engine-free, and the run
reaches no registry — a fixture makes `ModelRegistry` raise). Full suite → **1053 passed, 7 skipped** (the 7
skips are the standing `NOT RUN` asset gates). Manifest JSON re-parsed after the edit; 16 top-level keys, up
from 15, with the quarantine record the only addition and `usable_for` the only changed value. `artifacts/audit/
offline-proof.json` and `mechanical-findings.json` were not opened for writing, no audit tool was run, nothing
was staged or committed, and no package, dataset, model, benchmark or GPU probe was touched — the suites were
bracketed by a byte-level snapshot of the repository tree and the two snapshots are identical (`13821c6d69f991b5`,
206 files, no path added or removed). Lint and typecheck: **NOT RUN**, `ruff` and `mypy` remain uninstalled.

**The suite is not decorative — six temporary mutation probes, each reverted.** `return ()` from the gate:
**45 failed**, 61 passed. Drop the quarantine reuse check: **2**. Make a damaged quarantine record skip instead
of raising: **1**. Remove the caller-digest demand: **3**. Re-introduce the shared `TEACHER_TARGET_RECORD`:
**1**. Let `_strings()` accept an empty list again: **1** — so each of the two defects above has a test that
fails the moment its fix is undone, and no refusal test passes because the gate answers nothing. `tools/
corpus_forge.py` was restored from a byte-exact copy after each probe and is byte-identical to it.

**Blocked, unchanged by this entry:** a usable training corpus needs a batch forged from the new pool, whose
manifest carries valid boundary evidence, whose digests are recomputed from audio that exists on the machine
verifying them, and whose rights position (§3, §4, §7 of `docs/provenance.md`) is settled. None of that is
performed here, and A54's severity is proposed and not yet decided.

# Progress ledger — entry 19 (2026-10-15, the corpus boundary was anchored outside the manifest)

**What entry 18 left open, stated plainly.** Its gate re-derives eligibility from the batch's own fields, and
every one of those fields lives in the document being judged. A batch could therefore authorise itself: write a
`training_boundary` block naming the live pools and this file's five excluded paths, and the recorded leak stops
mattering, because nothing outside the manifest was consulted. The authority was moved out of the
descriptor and into a tracked file a reviewer sees in a diff, and the record is required to say what a hash does
**not** prove.

## What was built

- **P1, a versioned contract.** Every manifest `build_corpus_manifest` emits carries
  `training_boundary.boundary_contract`: the contract version and `pool_set_sha256(TRAIN_SENTENCES, EVAL_SENTENCES)`
  — both pools normalised, joined by a unit separator so a pool boundary cannot be shifted to manufacture a
  match. The header is checked against the pools as the code defines them now, not taken on trust.
- **P2, the tracked register.** `artifacts/corpus/legacy-batches.json` (new, untracked) records the pool set cleared
  for training, under the version, the verbatim disjointness rule and both pool sizes, plus `en-synthetic-01`
  as `QUARANTINED` against the SHA-256 of its manifest **as read from disk** (`7a39791a…`). Editing a manifest
  invalidates its own registration. `boundary_register()` re-reads on every call and raises on damage — an
  unreadable, absent, non-object, wrong-schema or under-fielded register is a refusal, never an empty one.
- **P3, a closed use vocabulary.** `ALLOWED_USES` is the only set a `usable_for` entry may draw from; a use
  outside it is a blocker, and `corpus_evaluation_uses()` reports only allowed uses.
- **P4, the quarantine flags demoted to records.** `training_quarantine.status` and `digests_verified` describe a
  past review and authorise nothing; an absent quarantine field does not make a batch permissive either.
  `boundary_provenance_blockers()` anchors a batch one of two ways only: a contract the register records *and*
  this code still implements — with every kept sample text re-checked against that registered pool — or a
  registered pre-contract batch matched by the caller's digest of the file on disk *and* an explicit `CLEARED`
  decision. `training_boundary.issued_by` is deliberately never read: it is a claim about an author.
- **M1, the reader and caller fences.** `tests/test_corpus_reader_fence.py` (new, 16 tests) pins that
  `tools/corpus_forge.py` is the only module opening a corpus descriptor, that `tools/evaluate.py` hand-writes its
  own `CORPUS` and reads none, that `SampleManifest.load()` refuses a corpus batch rather than downgrading it, and
  that any module touching weights names the gate. Both fences are tested against synthetic non-vacuous fixtures:
  an unlisted reader, a listed reader that skips the accessors, a trainer without the gate, and a compliant trainer.
- **The named door.** `corpus_training_eligible(manifest_path, audio_dir)` reads the bytes itself, hashes the
  manifest, re-hashes every listed clip, and raises `CorpusNotEligible` — a member of `CorpusBoundaryError` so a
  future trainer can catch exactly this — listing every remaining reason. **Nothing calls it: there is no trainer
  in this repository**, and entry 19 does not add one.

Judgment calls applied as approved: exact set equality for `excluded_inputs`; keyword-only `manifest_sha256=None`,
with a missing digest never releasing a legacy batch; `CorpusNotEligible` inside the existing family; acceptance
fixtures updated to the new contract rather than weakened.

## The consequence a reviewer must see without asking

Two adversarial controls (H and J in this pass's probe script) show a batch forged from the live pool, with audio
on disk and the shipped register intact, answering **0 blockers** — and a *pre-contract* batch whose `training_boundary`
is intact, registered by digest, and explicitly ruled `CLEARED` likewise answering 0. That is what a real release
would look like, and it is reachable today only by editing the tracked register: no such entry ships, and
`en-synthetic-01` stays `QUARANTINED` and ineligible. Provenance, not permission: nothing here establishes
licensing, consent, authorship or redistribution, and the rights fields stay unanchored because P5 was declined.

## Acceptance expectations changed, and why

1. The committed quarantined batch's blocker count is pinned at **12, up from 11**: a pre-contract batch now also
   has to show the caller read the file it is registered under.
2. `disjointness_rule` cases moved from *"is missing or empty"* to *"is not the rule this code enforces"*, and a
   weakened rule is now a case of its own.
3. Two new declaration cases: a boundary claiming `training_sentences: 800` / `evaluation_sentences: 800` is
   refused as declaring pools this code does not define; `excluded_inputs` naming a sixth path is refused as naming
   an input no review recorded.
4. `{"status": "CLEARED", "digests_verified": true}` forged over a cleared-looking batch now yields **only** the
   clip-digest blocker — the P4 decision made the flags inert, which is the behaviour change, not a relaxation.
5. `TestGenerationRefuses` gained four register cases: unregistered pool set, unreadable register, a different
   recorded rule, wrong recorded sizes.

## Two mutation probes survived, and both were test gaps

Round one removed each critical check in turn (23 probes, full suite each time). Twenty-one failed tests; two did
not. `boundary_register()`'s requirements that a pool set carry a **disjointness rule** and that a batch record
carry a **manifest digest** were untested: the damaged-register parametrization broke several fields per case and
asserted only the generic *"the boundary register cannot be read"*, so any other validator satisfied it — a green
test passing for the wrong reason. The cases were rebuilt as fifteen single-field documents, each asserting its own
refusal phrase in both the blocker list and the raised error. Round two: **23 of 23 probes fail at least one
test** (1 to 48 failures; removing the whole anchoring step fails 48), and `tools/corpus_forge.py` was restored
byte-identically after each, ending at `de82810d3a5b53cc…`.

## Adversarial pass

Ten bypass attempts against the shipped register, all read-only under a temporary directory: the committed batch
(12 blockers); its quarantine flags forged to `CLEARED`/`digests_verified: true`/`usable_for: training+adaptation`
(8); a hand-written valid-looking contract stamped onto it (18 — the register's `QUARANTINED` entry for its *name*
refuses it, and 8 of its samples are named as outside the registered pool); that header re-forged under a fresh
name (25); an invented `training-and-redistribution` use (11); the register edited to clear `en-synthetic-01`
without a caller digest (12) and with one (11 — `training_boundary` absent is still missing evidence); a training
sentence swapped for an evaluation sentence (3, all three mechanisms naming it); a sixth `excluded_inputs` path (1);
the register withdrawing the pool set (1); the register loosening the rule while the code stays strict (1). Through
the named door: absent audio and misnamed audio both raise, an absent register raises, and only the clean control
batch returns None.

## Verification of this pass

`tools/corpus_forge.py` **+438 / −12** against the copy held at the start of this pass; `tests/test_corpus_forge_boundary.py`
**106 → 191**; `tests/test_corpus_reader_fence.py` new at **16**; `docs/provenance.md` §7 — the *Enforcement, not
intention* and *For a batch that already exists* passages replaced by seven paragraphs describing the contract, the
register, the byte-identity limit, the door, and the fact that `asr-domain-evaluation` is a permission on record
rather than a runnable path. Focused: **209 passed** on the 2026-10-16 re-run of this pair, exit code 0 (**207 passed** when this entry was first written; `--run-approved-assets` was not passed and the full suite was not re-run alongside it). Full suite: **1154 passed, 7 skipped**
(entry 18's 1053 plus the 101 new tests, 85 boundary and 16 fence; the same seven standing `NOT RUN` asset gates,
none force-enabled).
Lint/typecheck: **NOT RUN**, `ruff` and `mypy` remain uninstalled.

No-write proof: the baseline tree snapshot (206 files, aggregate `b383274f86f8c77f`) versus the snapshot taken
before this entry was written (212 files, `6918585e4c423428`) differ in exactly three changed paths —
`tools/corpus_forge.py`, `tests/test_corpus_forge_boundary.py`, `docs/provenance.md` — two intended new paths,
`artifacts/corpus/legacy-batches.json` and `tests/test_corpus_reader_fence.py`, and the four foreign files recorded
in the next section. All 15 protected digests are
unchanged: the two audit evidence files, the three `.pre-audit` manifest snapshots, `asset-status-corrections.json`,
the eight evaluation artefacts and `en-synthetic-01/manifest.json` still at `7a39791a…`. No audit tool was run,
nothing was staged, committed, pushed or deleted, no model, dataset, audio, GPU or dependency was touched. Writing
this entry adds one further changed path, `PROGRESS.md`.

## An unresolved filesystem anomaly observed during this pass

Four Windows shell-cache files (`cversions.2.db` and three GUID-named `.db` files, 16 KB–662 KB) appeared **inside
the working tree** at 2026-10-15 21:32 under a literal `%SystemDrive%/ProgramData/Microsoft/Windows/Caches/`
directory — the unexpanded form of an environment variable, used as a relative path. **The cause is unidentified.**
What is established is only that the write burst falls inside this session's validation window, between the
protected-hash baseline and the mutation-probe results file, so the timing is a correlation with the work this pass
did and not evidence about which process wrote them. No writer process was identified. Three of the four are
byte-identical copies of the live
`C:\ProgramData\Microsoft\Windows\Caches` files under a different version suffix, and the fourth has the same size
with different content, so the hazard is machine cache data, not project data. A read-only screen for identity
strings found none: no account name in either ASCII or UTF-16LE, no user-home path, no repository path — and the
screen was shown to be capable of matching, so the zeros mean something. That screen is a bounded set of needles,
not a proof that nothing sensitive is stored in them.

They were untracked and **not** ignored, which made them a publication hazard of the same family as A53. The
root-anchored line `/%SystemDrive%/` is now in `.gitignore` (`.gitignore:25`), so an ordinary `git add -A` does not
sweep them up; a deliberate force-add still would. They are left in place — removing an untracked file is a
separate decision, not part of this entry.

## Remaining limitations

The register is editable by anyone with write access to the tree, so it is reviewable provenance and not
attestation. No trainer consults the gate, so P1–P4 bind a build and a test, not a run. The rights fields
(`licence_basis`, `consent_basis`, `synthetic`, `generator.model_id`, `verifier.model_id`) are still checked for
shape only — P5 declined. `main()` still returns 0 when every asset is refused. The corpus audio remains absent,
so `digests_verified` stays false and unverifiable by construction. And A54 remains **proposed S2**, with A50,
A52 and A53 untouched.

## Architecture-first cleanup — the Pocket reference adapter retired, direction recorded (2026-10-20)

The owner ruled the model direction: a Conformer-TDT-class recognition student against a Whisper large-v3
teacher as the first priority, then a flow-matching synthesis student with Qwen3-TTS as its intended teacher,
and EmbeddingGemma 2 (740M) recorded as the training-side encoder. `pocket_reference.py` was traced first and
removed second: a whole-tree scan for its six public symbols found only its own package `__init__`, its test
file, this register and two untracked caches, so it had no runtime consumer and — with every capability
`NO_MODEL` — no route could select it. Its exports and the 11 adapter-specific tests went with it; the
`sys.modules` guard it also carried was rewritten forward rather than dropped, and the `pocket` term in
`tools/audit.py:142`'s deny-list stays to guard the removal. Shared machinery — phonemiser, normalisation,
chunking, voice identity and consent, approval, integrity, evaluation, orchestration — is untouched.

**Verification.** `def test_` 1,004 → 996 over 43 unchanged files. The command `README.md:253-259` documents
returns **1,063 passed, 7 skipped, 1 deselected, 86 excluded**; the whole suite returns **1,150 passed,
7 skipped**. Both protected artefacts read `d7d97731…` and `f8ae6c9e…` before the first edit, after every run
and after the last, and neither was regenerated, staged or committed. Every README edit is line-count neutral
above `:358`, so the register's README locators still resolve; `docs/architecture.md` gained §7.

**One blocker, reported rather than bypassed.** `tools/audit.py` hard-codes its write to
`artifacts/audit/mechanical-findings.json` and has no `--out`, so re-running it after a source deletion must
modify a protected artefact. The 12 checks ran in-process with the writer not called: 11 PASS, 1 FAIL, and the
FAIL is `error taxonomy` on `evidence_paths.py:58` — introduced at `b2df403`, unrelated to this pass. The
artefact still records 11 findings and `0 raise site(s)`, which is why it reads as stale rather than as
contradicted. Fixing that is a separate authorised decision. **Nothing in the recorded direction is built**:
no trainer, no dataloader, no tokenizer, no checkpoint, `torch` in no dependency list, and Qwen3-TTS and
EmbeddingGemma 2 have no manifest, provenance row or approval entry — they are now named in tracked files for
the first time, and named as decisions only.

# Progress ledger — entry 20 (2026-10-20, the unsupported AI co-author trailer is removed; chain published 2026-10-22)

**What this entry reverses, and what it does not.** Entry 17 recorded that history was preserved as it stands and
listed the AI co-author trailer on the initial commit among the facts that stayed. That part has now been ruled on
differently — locally: the trailer is gone from the rewritten chain, and it is still on the remote's chain, which
this entry does not touch. Everything else that entry undertook to preserve is preserved — every tree, every blob,
every file content, every author and committer identity, every timestamp, UTC offset, and the other 33 messages byte
for byte. The correction is narrow on purpose: it removes one line of message text, adds and deletes no code, and does
not touch any commit whose message did not end with that exact line. `docs/audit-register.md` Addendum 28 carries the
full 34-row old→new map, the local/remote split, and the reasoning.

**The authorship question stays open, deliberately.** The owner states Claude was not used. Git metadata cannot
corroborate or refute that: the trailer is message text, no identity field names an AI model, and the hosting
service's contributor line is a rendering of that one line. Removing an unsupported trailer fixes the record; it is
not a finding about who typed anything, and nothing here should be read as one.

**Method, so the result is reproducible rather than merely asserted.** Plumbing, not rebasing: `git commit-tree`
per commit, oldest first, feeding each its original tree, its remapped parents and its original author and committer
name, email, timestamp and offset via `GIT_AUTHOR_*` / `GIT_COMMITTER_*`. A rebase would have re-stamped the
committer of every replayed commit with this machine's identity and changed 29 commits' attribution — a worse
attribution defect than the one being fixed. The tool ran with an explicit expected-changed commit and expected
chain length, verified each recreated commit against its original field by field, and moved the branch only with a
compare-and-set that fails if anything else moved the reference first. Two dry runs that moved no reference produced
the map and proved it stable across the tip amendments that followed it.

**The audit step that had to be re-anchored.** `docs/audit-register.md` documents a personal-path check whose
procedure pins the pre-cleanup revision by SHA inside a fenced constant. The local rewrite moved that
revision's name off this chain, so the constant now carries its rewritten form,
`d9270c052cdf0df819507edb4781281ee5836f95`, which since the push authorized on 2026-10-22 resolves on the
remote branch history as well as here. The pre-rewrite name is kept at `refs/heads/pre-rewrite-main`; no
remote ref names that chain now, though its commits are still served at their direct URLs. The historical
prose at `:1225-1227` is left exactly as it was written, because it records a measurement taken against the object
that existed at the time. The map in Addendum 28 is the bridge between the two names.

**Not recorded here on purpose:** the SHA of the commit this entry becomes part of. A commit cannot cite itself, so
the corrected tip is identified by content — it is the only commit whose message is this documentation change — and
by whatever `main` resolves to once the branch is moved.

**Local state and remote state, stated separately, because the two are easy to conflate.** Locally, in the throwaway
clone: `main` names the rewritten chain of 35 commits — the 34 recreated historical commits plus this documentation
commit — and `pre-rewrite-main` names the original 34, also held as a verified bundle. On the hosting service,
`refs/heads/main` still names the original tip, the original root commit still carries the trailer, and the rendered
contributor line is therefore unchanged. Nothing in these entries says otherwise: the correction is complete locally
and entirely pending remotely.

**Not done in this entry, and not implied by it:** nothing has been pushed. The rewrite is local, the old chain is
kept as a reference and a verified bundle, and the push is a separate decision with the old and new HEADs presented
for approval first.


## ASR Phase 1 and 1B — log-mel frontend, adversarial review, teacher frontend parity (2026-10-23)

**Scope.** A read-only audit of the existing ASR path was followed by two implementation phases, both left
**uncommitted** for owner review: a stdlib-only log-mel frontend (Phase 1) and numerical parity with the installed
Whisper feature extractor (Phase 1B). No model, trainer, dataloader, tokenizer, adapter, checkpoint or dependency
was added. `pyproject.toml` is untouched; `evidence/`, `artifacts/`, `docs/audit-register.md` and this file's
earlier entries are untouched.

**Implemented and tested.** `src/nativevoice/audio/features.py`: `FBankConfig` (all geometry, mel, log and
post-processing choices as fields), `mel_filters`, `power_spectrum`, `window_function`, `log_fbank`,
`FeatureGrid`. `tests/test_audio_features.py` (98 tests) covers the student contract;
`tests/test_audio_features_parity.py` (34 tests) covers the reference comparison. Exports added to
`nativevoice.audio`; `models/`, `stt/`, `tools/` and every gate are unchanged.

**Reference parity verified, within limits.** `FBankConfig.whisper_parity()` reproduces
`faster_whisper==1.2.1` `FeatureExtractor(feature_size=80)` on identical input: worst absolute difference
**1.0e-05** in the reference's own post-normalised units against an 8-unit usable span (about 1.3e-06 % of it),
with frame counts and shapes matching on all 16 compared cases from one sample to one second of audio. The mel
matrix agrees to 9.2e-10. The same reference algorithm restated in float64 agrees to 4.8e-09, which is what
separates "different precision" from "different formula" — the residual is the reference's float32 arithmetic,
not a divergent implementation. **No bitwise equivalence is claimed.** This is environment-dependent: with
`faster_whisper` made unimportable the 21 reference tests report NOT RUN and 111 pass, and a module that exists
but fails to import is reported as a failure rather than skipped. The reference is pinned in
`docs/asr-architecture.md` §9 by version and file digest, and in no dependency list.

**Measured divergence, kept on the record.** The project's own default geometry is **not** teacher-compatible:
mapped into the reference's units it differs by mean 1.23–1.41 and max 1.67–2.59, i.e. 15–18 % of the reference's
usable dynamic range, at 98 frames per second against the reference's 101. That number was written down before the
parity config existed, and it is not hidden by the parity result.

**Found and fixed during review.** A reversed right-hand reflect mirror (invisible in the padding itself, visible
only as wrong late-frame spectra); a **fabricated** refusal of buffers shorter than the reflect pad, where numpy
keeps bouncing at any pad length; two `dom_len`-style comparisons that were vacuous because a test generator had
truncated every tone to zero samples; a silent short row from a malformed `FeatureGrid`; a `TypeError` where a
`ConfigurationError` was intended; and two claims in `docs/` that a later phase made stale.

**Verification.** `python -m pytest` → **1275 passed, 7 skipped** in 29.3 s, against 1150 passed / 7 skipped
before this work: +132 tests, the same 7 pre-existing asset-gated skips, no regressions. Focused frontend and
audio/STT runs, `tools/audit.py` and `tools/offline_proof.py` results are recorded in the checkpoint report.

**Not claimed.** No recognition accuracy, no measured capability, no change to any `CapabilityStatus`, no
competitive performance, and no training. `docs/capability-matrix.md` is unchanged; English remains the only
`MEASURED` recognition language.

**Awaiting the owner's approval, nothing assumed.** D1, which frontend a student trains on
(`docs/asr-architecture.md` §7.1 carries a recommendation with its reasoning and did not apply it); whether
`faster-whisper` should become an opt-in `parity` extra (§9) or stay an undeclared local convenience; and the
pre-existing D2–D10 (licence semantics for a self-trained asset, checkpoint formats, teacher-target rights,
corpus rights, the EmbeddingGemma 2 scope conflict, and the closed AI4Bharat question of zero occurrences).

**Commit state.** Nothing has been committed or pushed by this work; `main` still resolves to the last published
commit and the changes sit in the working tree for review.

## ASR Phase 1B — owner rulings and the controlled cleanup (2026-10-24)

**Ruled, not assumed.** D1 was ruled as recommended: `FBankConfig()` stays the student frontend
contract and `FBankConfig.whisper_parity()` stays a separate reference instrument. **No default was
changed** — the ruling was verified by reading both objects field by field afterwards, and the
student geometry still answers `fft_size=1024, window=povey, center=none, mel_scale=htk,
dynamic_floor=None, post_scale=None`. The optional `parity` extra was **not added**; the exact
reference environment (`faster-whisper 1.2.1`, its `feature_extractor.py` at 230 lines and sha256
`e403966d…7be7a397c`, numpy 2.5.3) stays documented in `docs/asr-architecture.md` §9, and the
dependency question returns when parity enters CI. Nothing under `evidence/` or `artifacts/` was
regenerated, staged or committed; the support request was not submitted.

**README's live figure was stale and is corrected.** `README.md:263` asserted in the present tense
that the tree returns 1,063 passed on the documented command; that was true at `f403b47` and false
after `2b167a9`. All three points were measured rather than reasoned about: the documented exclusions
and deselection, run against a `git archive` extract of `f403b47`, return **1,063 passed, 7 skipped,
1 deselected**; against `2b167a9`, **1,195**; against this working tree, **1,205 passed, 7 skipped,
1 deselected**. Whole-tree collection is **1,299**, the four ignored modules hold **86** of it and the
command collects **1,213**, so 1,213 − 1 deselected = 1,212 run, of which 7 skip — the arithmetic
closes without inferring anything, and the two steps from 1,063 are this work's own **+132** (frontend
and parity at `2b167a9`) and **+10** (below). The static counts moved with it: **45** test files,
**1,095** `def test_` functions, counted two ways (`grep -o` over `tests/` and anchored per-file) to the
same number — the ten new nodes come from five new test functions, one of them parametrised six ways,
so `def test_` gains 5 (1,090 at `2b167a9`) while the node count gains 10. Both edits are **line-count neutral** — `README.md` is still 392 lines
and 2 insertions / 2 deletions — so every `README.md:NN` anchor the audit register cites still resolves:
`:245`, `:253-259`, `:261-263`, `:305-307`, `:332`, `:358` were each re-read after editing.

**One unguarded optional-dependency test was found and fixed.**
`tests/test_stt.py::test_whisper_names_missing_pieces_when_authorized` asserted `expected_path`,
which is the *weights-missing* branch, so it passed here only because `faster-whisper` happens to be
installed: with the package unimportable it failed for the wrong reason, and the adapter had described
both states as "not installed". That conflation is the bug — a corrupted wheel told a host to install
what is already present hides the corruption. The two states are now named once, in
`src/nativevoice/stt/adapters/optional_backend.py` (`package-absent` only when `ModuleNotFoundError`
names the requested module itself), and both `_import_backend` call sites raise `ModelNotInstalled`
with an added `import_state` field and a message that says which. The category is unchanged, so no host
that catches `ModelNotInstalled` or `CapabilityUnavailable` branches differently, and no existing
assertion was weakened: the test keeps every assertion it had, and now skips with a labelled NOT RUN on
absence while **failing by name** on a broken install. Ten tests cover it (six classifier cases,
absence, broken, the `sys.modules` measurement, and one that fails if the two messages collapse).
Falsified rather than trusted: with the classifier replaced by a constant — the pre-fix behaviour —
**7 of the 10 fail**; with `faster_whisper` genuinely unimportable the whole suite is **1270 passed,
29 skipped, 0 failed** (it was 1 failed / 1260 / 28 before); with a shadowing package that raises on
import it is **22 failed, 1270 passed, 7 skipped**, and the stt failure names the corrupt install.
The two classifier cases the review added are the ones the suite did not yet pin: a
`ModuleNotFoundError` naming a **submodule** of an installed package
(`faster_whisper.feature_extractor`) and one naming **nothing at all** — both answer `package-broken`,
because guessing absence from an unattributable failure would turn a corrupt environment into a skip.

**Measured along the way:** `sys.modules["faster_whisper"] = None`, which `tests/test_measurement.py`
uses to block the backend, raises `ModuleNotFoundError` naming the module on **both** interpreters
present here (CPython 3.12.10 and 3.14.7), so that control simulates *absence*, not corruption. It is
now pinned by a test rather than by a reading, and the classifier does not consult the module table.

The three real cases were also driven through the actual import machinery, outside the suite, because
an injected exception cannot prove what CPython itself raises: an interpreter on which `faster_whisper`
is genuinely not installed (3.12.10) gives `package-absent` and "is not installed"; a shadow package
whose own import of a missing dependency fails gives `package-broken`; a shadow package that raises
`ImportError` while loading gives `package-broken`; and a shadow package that imports but has no
`feature_extractor` submodule reports the submodule failure as broken, not missing. The harness is kept
at `…/scratch/p1b_verify/` (`import_case.py` plus `case2`, `case3`, `case4`) and stays outside the
repository; the in-suite tests exercise the shipped `import_state` and the shipped `_import_backend`,
not a second copy of the rule.

**Suite state after the cleanup.** Full suite **1292 passed, 7 skipped in 22.04s**; with
`ULTRON_RUN_APPROVED_ASSET=1` **1299 passed in 56.90s**; targeted set (`test_stt`, `test_measurement`,
both frontend files, both hygiene guards) **229 passed, 1 skipped**. `tools/audit.py` in a throwaway
copy of this tree reports **11 passed, 1 failed, 0 not run** with the single offender still
`evidence_paths.py:58:TypeError` — identical to the HEAD-only baseline, so this change contributes
nothing to it. Leaving that FAIL in place is deliberate: the raise is asserted by
`tests/test_evidence_paths.py:151`, and widening the audit's allowlist would rewrite the meaning of a
mechanical check whose output is itself evidence. Both that and the stale
`artifacts/audit/mechanical-findings.json` (11 findings where the code emits 12 checks) remain the
owner's calls, as does the working-copy fingerprint drift of that artefact reported to him in this
pass: the restored working copy now equals `HEAD` (`1babf3fab05b3dbd…`, 1,444 B) where rows A65/A72
recorded `d7d97731ea0274b4…`, 1,500 B with 56 CRLF pairs. Nothing in `docs/audit-register.md` was
appended or edited.

**Still not claimed, and unchanged by this pass:** no trainer, no dataloader, no tokenizer, no
checkpoint, no Conformer-TDT adapter, no `torch`; no model ran, so no accuracy, no WER and no
competitiveness. Phase 2 stays closed and nothing was pushed — `main` is `2b167a9` locally, one ahead
of `origin/main` at `f403b47`, and the cleanup above is an uncommitted working-tree change for review.
