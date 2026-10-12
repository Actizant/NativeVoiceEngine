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
