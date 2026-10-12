# Native Voice — a local-first voice platform

A standalone, original voice platform that can power **multiple independent applications**: ULTRON,
FieldMind and anything else that needs to hear and speak. It is not built around one host application's
identity or architecture — hosts contribute a single function and keep their own state, memory,
permissions and UI. See `docs/platform.md`.

Hearing and speaking are provided as one reusable engine with pluggable **voice identity**: each host
selects its own installed voice package, and no voice name is baked into the synthesis path.

Not an integration of OpenWhispr or Pocket TTS. Those archives were read as research references and live
outside this project at `../_refs/`; no file from either is imported, vendored, ported or renamed. Design
reasoning is in `docs/research-report.md` and `docs/architecture.md`.

## What is here

```
src/nativevoice/
  core/         typed contracts, error taxonomy, cancellation tokens, streaming events, timings
  audio/        PCM maths, band-limited resampling, ring buffers + look-back, VAD, devices, WAVE I/O
  models/       manifest registry, SHA-256 integrity, licence approval gate (default deny), lifecycle
  hardware/     detection, resource budgets, Lite / Balanced / Performance profiles
  config/       local settings + data paths + retention policy
  language/     ~60-language registry, capability ledger, script+lexicon LID, routing policy
  stt/          recognition interfaces, engine registry, transcriber orchestrator, gated adapters
  tts/          synthesis interfaces, normalisation, chunking, pronunciation gap, gated adapters
  profiles/     voice-profile store, consent records, quality gating, export/import, conditioning gap
  session/      turn state machine, barge-in, stale-response cancellation, listening modes
  eval/         WER/CER, RTF, first-audio latency, resource sampling, benchmark runner, capability matrix
  diagnostics/  structured event sink, offline-compliance proof, per-stage latency timeline
  integration/  the stable ULTRON seam
  cli/          diagnostics command
tests/          contract, property, gate, offline and benchmark tests
docs/           baseline, research report, architecture, provenance inventory
DECISIONS.md    the seven rulings that gate measured speech quality
PROGRESS.md     completed / unresolved / test status / blockers
```

The control path imports the **standard library only**. `audioop` is gone from Python 3.13+, so RMS, peak,
DC blocking, fades, crossfade and the WAVE container are implemented directly. Heavy runtimes appear only
inside `stt/adapters/` and `tts/adapters/`, imported lazily, and refused unless the model registry says a
named human approved that asset.

## Run it

```
python -m pytest -q                              # no hardware, no models, no network required
python -m nativevoice.cli.diagnose --capabilities capabilities.json
```

English speech works locally today, through the model registry and the licence gate: Kokoro-82M
(Apache-2.0, verified from the licence text shipped inside the asset) synthesises, and faster-whisper (MIT)
transcribes what was just synthesised. Every other language is an honest refusal naming what is missing,
never silence:

- `hi`, `bn` synthesis: **unsupported** — the approved asset ships front-ends for `en-us`, `en-gb` and `zh`
  only, and the adapter refuses rather than emitting romanised audio;
- `zh` synthesis: **measured and failed the bar** — round-trip CER 0.463, partly a simplified-vs-traditional
  mismatch with no approved converter available;
- `hi`, `bn` recognition: **declared only** — Whisper's model card claims them, nothing here measured them,
  and an unmeasured claim cannot satisfy a routing decision;
- voice cloning: **not a feature**, because every zero-shot asset surveyed is non-commercial or unlicensed
  (`docs/provenance.md`).

Synthesis is slower than real time on this machine at every profile (RTF 1.8–2.4, first audio 4–8 s). That
is a measured ceiling rather than a bug — see `docs/audit-register.md` A19 and
`artifacts/benchmarks/profiles.json`.

## Three invariants worth knowing before reading the code

1. **Capability is a status, not a boolean** — `MEASURED` / `DECLARED` / `PARTIAL` / `UNSUPPORTED` /
   `NO_MODEL`. A vendor's claim cannot satisfy a strict routing decision, and `NO_MODEL` is a normal state
   rather than an error to be papered over.
2. **Approval is enforced at load time, default-deny** — no digest, no run; no named approver, no run;
   unknown licence field, no run. There is no environment variable that bypasses it.
3. **Numbers come from measurement or say they don't** — profile thresholds and resource requirements carry
   a `measured: bool`, and anything unmeasured is reported as unmeasured instead of asserted.
