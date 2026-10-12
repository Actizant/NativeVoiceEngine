# Architecture — ULTRON Native Voice

Original, provider-independent, local-first. This document records the shape and the reasoning; the
code is the authority on details.

## 1. Package map and dependency direction

```
                      nativevoice/
  core/ ───────────────────────────────  typed contracts, errors, cancellation, events, time
    ▲        ▲        ▲        ▲
  audio/   models/  language/   │        transport + VAD   asset registry    identity + routing
    ▲        ▲          ▲       │
    └────┬───┴──────┬───┴───────┤
       hardware/  config/    diagnostics/                    capability + budgets
    ▲        ▲          ▲
  stt/      tts/     profiles/                                 engines behind interfaces
    ▲        ▲          ▲
    └───┬────┴──────────┘
      session/ ──────────  turn state machine, barge-in, listening modes
        ▲
      eval/  integration/  cli/                             measurement + the ULTRON seam
```

The rule is mechanical: **a package imports `core` and the packages above it, never a peer below it,
never a sibling's internals.** `stt` does not import `tts`; `session` talks to both through the
`Transcriber` and `SpeechSynthesizer` protocols. `eval` may consume any engine but no engine may consume
`eval`. There is one writer per package and no global mutable state anywhere — registries are instances
owned by a session or an application, never module-level singletons, so two conversations cannot contend.

## 2. Why the control path is standard-library-only

The specification requires local-first operation, offline inference, and ownership of the orchestration
layer. Those are all satisfiable with heavy dependencies, but the *failure* mode of a local-first system is
that its control path acquires an optional dependency and then behaves differently when that dependency is
missing. So the split is enforced structurally:

- `core`, `audio`, `models`, `hardware`, `config`, `language`, `session`, `profiles`, `eval`, `diagnostics`,
  `integration` — **stdlib only at import time**. `audioop` was removed in Python 3.13, so the PCM maths
  (RMS, peak, DC blocker, fades, crossfade, band-limited resampling, WAVE container parsing) is written
  directly rather than inherited.
- `stt/adapters/*`, `tts/adapters/*` — the only places a heavy runtime may appear, imported lazily inside a
  method, gated by the model registry's approval state, and raising a named error when absent.

This buys three things: the test suite runs on a machine with no audio hardware and no model; the offline
proof is about the program rather than about the environment; and "which third-party code ran during that
utterance" has a short, enumerable answer.

## 3. The five load-bearing decisions

### 3.1 Capability is a status, not a boolean

`core/language.py` defines `Capability(task, status, language, engine, metric, evidence)` with
`MEASURED / DECLARED / PARTIAL / UNSUPPORTED / NO_MODEL`, and routing may only act on `MEASURED` or
`PARTIAL` unless a caller explicitly opts into `DECLARED`.

This exists because of what the reference archives actually did. OpenWhispr ships a 61-entry language
registry whose entries are per-engine booleans, and **Bengali is not in it at all** — the only Bengali in
the repository sits in a cloud provider's internal fallback set, so a picker could show 60 languages while
silently having no path for one of them. Pocket TTS documents 7 languages behind a name that implies a
general tool. A boolean registry reproduces that failure exactly. A status enum makes "the vendor says so"
unusable as evidence for a promise we make.

Consequence: `NO_MODEL` is a normal, non-error state. Bengali synthesis in this system today is `NO_MODEL`
with a named path to `MEASURED`, and the runtime refuses rather than degrading to unrelated audio.

### 3.2 Licence approval is enforced at load time, default-deny

`models/api.py` separates three independent axes that the reference projects conflated:

- **review** — `UNREVIEWED` (the default) / `APPROVED` (requires a named approver, a date, and complete
  provenance) / `REJECTED`;
- **integrity** — every declared asset has a SHA-256 digest and a size; an unrecorded digest is reported as
  *unverifiable*, never as verified;
- **requirements** — `ResourceRequirement` carries `measured: bool`, so a declared RAM figure may inform a
  suggestion but never an automatic profile decision.

`ModelManager.load()` cannot reach an engine object without passing all three. There is no environment
variable that flips approval, because the failure mode we are guarding against is a benchmark convenience
quietly becoming a shipping dependency. `Provenance.incomplete()` returns the missing fields, so
"we do not know the licence" is data rather than a forgotten comment.

### 3.3 Streaming is revision-shaped and cancellation is a token with cleanup

`TranscriptionEvent` carries `PARTIAL / FINAL / RETRACT`. The session holds a pure reducer that inserts by
timestamp and retracts by exact match, learned from the reference meeting pipeline: a decoder that revises
with more context will contradict what it already showed, and a design without a retraction path
rediscovers that the first time a user watches the live text.

`CancellationToken` is not a boolean. It runs registered cleanups exactly once in reverse order, cascades to
children, and is checked at every yield boundary by `guard()`. That is what makes §9's
"if a spoken response is cancelled, its queued audio must not continue playing" a testable property rather
than a hope: `CapturingOutput.play(buffer, token)` returns zero frames written after cancellation, and the
test asserts it.

### 3.4 Detection thresholds and profile ceilings come from measurement or say they don't

`hardware/profiles.py` seeds three profiles with deliberately conservative numbers and marks every one of
them `measured=False`, with `unmeasured_fields()` enumerable. A profile cannot be reported as validated
without a benchmark artefact behind it. The reference stack published cloud latency receipts with
self-limiting disclaimers while documenting **no local-model RTF or latency number anywhere** — that is the
gap this layer must close with runs on this machine, and it is why `eval/runner.py` records min/median/p95
per stage with the command and host summary beside it.

The energy VAD is written the same way: hysteresis with rise above fall, an adaptive floor tracking room
noise, and a `score` documented as monotone-but-not-a-probability. The neural detector slot is `NO_MODEL`
rather than quietly filled by the energy fallback, because a system that silently swaps its acoustic model
for an energy gate will report confidence it does not have.

### 3.5 Continuity failure is a device failure, not silence

`SounddeviceInput` raises `DeviceDisconnected` when no capture callback has arrived for `capture_stall_s`.
The reference project documented a Teams meeting where loopback delivered digital silence for the entire
call regardless of configuration, and had to add a silent-capture watchdog. Silence that should not be
silence is the most expensive failure in a hearing system, because every downstream component reports
success.

## 4. Real-time shape of one turn

```
device callback ─► RingBuffer(+PreRoll) ─► resample 16k mono ─► DC block ─► level gate
     └─► UtteranceSegmenter (hysteresis VAD, per-context config)
             └─► segment ─► LanguageRouter ──► TranscriptionService ─► partial/final/retract ─► session
                                    │                                          ▲
                                    └── ambiguous / unavailable ──► ask or report limitation
  ULTRON answer text ─► LanguageRouter(TTS) ─► SpeechPipeline ─► normalise ─► chunk ─► engine.stream()
        └─► SynthesisEvent(seq, audio) ─► AudioOutput.play(token)   [cancel ⇒ stop, no queued frames]
```

Two structural notes. First, the text and voice paths are independent: with voice output disabled the
synthesis result is routed to `CapturingOutput` (or dropped) and every check still runs, so "text only" is
a routing decision rather than a code path that skips validation. Second, barge-in is a *cancellation of the
playback token plus a discard of stale partials*, not a mute: the reducer knows a newer turn has begun, so a
late final from the superseded request cannot be committed.

## 5. Interfaces ULTRON will consume

`integration/` exposes one façade — request in, typed events out — so the main repository never imports a
subsystem:

```python
voice = UltronVoice(settings)                     # hardware profile + registry + session
voice.hear(request)         -> TranscriptionResult  # or LanguageUnsupported / EngineUnavailable
voice.stream(request)       -> Iterator[TranscriptionEvent]
voice.speak(answer)         -> Iterator[SynthesisEvent]
voice.cancel(reason)                             # stops playback and discards stale work
voice.capabilities(task)    -> {lang: Capability}   # honest per-language truth
voice.diagnose()            -> report                                                      
```

The seam is deliberately narrow: ULTRON's reasoning never sees an engine, a model id, or a licence decision.
Adopting this system later means satisfying its `ModelManifest` requirements, not editing its core.
Wake-word detection and continuous listening have interfaces and no implementation; §9 says explicitly not
to report them as implemented because an interface exists, and neither is counted in the capability matrix.

## 6. Testing strategy

Layered, and mapped to §14 rather than invented:

- **contract tests** on `core` (formats reject partial frames, cancellation cleans up once, `DECLARED` is
  not evidence);
- **property/edge tests** on audio maths (a 12 kHz tone must be gone after a 48→16 kHz band-limited cut,
  linear mode must leak it, crossfade length arithmetic, ring-buffer overflow keeps the newest tail);
- **engine-agnostic orchestration tests** using a `DeterministicEchoEngine` / `ScriptedSynthesisEngine`
  that declare every language `UNSUPPORTED`, so they can prove plumbing while being *incapable of
  satisfying a production routing decision* — the honest way to test a stream without a model;
- **capability-gate tests** asserting that an unsupported language raises before an engine is touched, with
  a probe engine recording whether it was invoked;
- **offline proof** in two independent forms: a runtime `OfflineGuard` that refuses non-loopback sockets and
  records attempts, and a static scan for network-module imports with a declared allowlist — because a
  runtime guard only catches code that actually ran;
- **benchmark suite** with warmup/repeats/p95 and a `no_baseline` first-run outcome, never a fake regression;
- **real speech quality** gated on approved assets and a licence-usable corpus, with synthetic fixtures
  refused by name in any quality path.
